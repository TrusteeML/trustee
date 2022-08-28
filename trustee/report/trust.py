"""
Trust Report
====================================
The module that implements Trust Reports
"""
import os
import copy
import pickle
import graphviz
import numpy as np
import pandas as pd

from sklearn import tree
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, r2_score

from prettytable import PrettyTable

from trustee import ClassificationTrustee, RegressionTrustee
from trustee.utils.tree import get_dt_info

from .plot import (
    plot_top_nodes,
    plot_top_branches,
    plot_top_features,
    plot_all_branches,
    plot_distribution,
    plot_samples_by_level,
    plot_stability,
    plot_stability_heatmap,
    plot_dts_fidelity_by_size,
    plot_accuracy_by_feature_removed,
)


class TrustReport:
    """Class to generate trust report"""

    def __init__(
        self,
        blackbox,
        X=None,
        y=None,
        X_train=None,
        X_test=None,
        y_train=None,
        y_test=None,
        max_iter=10,
        num_pruning_iter=10,
        train_size=0.7,
        predict_method_name="predict",
        trustee_num_iter=50,
        trustee_num_stability_iter=10,
        trustee_sample_size=0.5,
        trustee_max_leaf_nodes=None,
        trustee_max_depth=None,
        trustee_ccp_alpha=0.0,
        analyze_branches=False,
        analyze_stability=False,
        skip_retrain=False,
        top_k=10,
        logger=None,
        verbose=False,
        class_names=None,
        feature_names=None,
        is_classify=True,
    ):
        """
        Builds trust report for given black-box model using the Trustee method to extract white-box explanations as Decision Trees.
        """
        self.blackbox = blackbox
        self.X = X
        self.y = y
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.train_size = train_size
        self.max_iter = max_iter
        self.num_pruning_iter = num_pruning_iter
        self.predict_method_name = predict_method_name
        self.trustee_num_iter = trustee_num_iter
        self.trustee_num_stability_iter = trustee_num_stability_iter
        self.trustee_sample_size = trustee_sample_size
        self.trustee_max_leaf_nodes = trustee_max_leaf_nodes
        self.trustee_max_depth = trustee_max_depth
        self.trustee_ccp_alpha = trustee_ccp_alpha
        self.analyze_branches = analyze_branches
        self.analyze_stability = analyze_stability
        self.skip_retrain = skip_retrain
        self.top_k = top_k
        self.logger = logger
        self.verbose = verbose
        self.class_names = class_names
        self.feature_names = feature_names
        self.is_classify = is_classify

        self.step = 0
        """
            total_steps = 
                _prepare_data (1) + 
                _collect_blackbox (1) +
                _collect_trustee (1) + 
                _collect_top_k_prunning (1) + 
                _collect_ccp_prunning (num_pruning_iter) +
                _collect_max_depth_prunning (num_pruning_iter) +
                _collect_max_leaves_prunning (num_pruning_iter) +
                _collect_features_iter_removal (max_iter)
        """
        self.total_steps = 4 + (num_pruning_iter * 3) + (max_iter if not skip_retrain else 0)
        """
            if analyze_branches:
                total_steps += _collect_branch_analysis (num_leaves = trustee_max_leaf_nodes or guess 100)

        """
        if analyze_branches:
            self.total_steps += trustee_max_leaf_nodes if trustee_max_leaf_nodes else 100

        """
            if analyze_stability:
                total_steps += _collect_stability_analysis (max_iter) 
        """
        if analyze_stability:
            self.total_steps += max_iter

        self.bb_n_input_features = 0
        self.bb_n_output_classes = 0

        self.trustee = None
        self.y_pred = None
        self.max_dt = None
        self.max_dt_y_pred = None

        self.max_dt_samples_by_level = []
        self.max_dt_leaves_by_level = []
        self.max_dt_top_features = []
        self.max_dt_top_nodes = []
        self.max_dt_top_branches = []
        self.max_dt_all_branches = []

        self.min_dt = None
        self.min_y_pred = None
        self.min_dt_y_pred = None

        self.branch_iter = []
        self.stability_iter = []
        self.ccp_iter = []
        self.max_depth_iter = []
        self.max_leaves_iter = []
        self.top_k_prune_iter = []
        self.whitebox_iter = []

        log = self.logger.log if self.logger else print
        log("Running Trust Report...")
        self._prepare_data()
        self._collect()

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["logger"]
        del state["blackbox"]
        del state["trustee"].expert
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.logger = None
        self.blackbox = None
        self.trustee.expert = None

    def __str__(self):
        """Formats collected data into a reporto using PrettyTable"""
        report = PrettyTable(title="Classification Trust Report", header=False)

        summary = PrettyTable(title="Summary")
        blackbox_report = PrettyTable(border=False, header=False)
        blackbox_report.align = "l"
        blackbox_report.add_row(["Model:", type(self.blackbox).__name__])
        blackbox_report.add_row(["Dataset size:", self.dataset_size])
        blackbox_report.add_row(
            [
                "Train/Test Split:",
                f"{self.train_size * 100:.2f}% / {(1 - self.train_size) * 100:.2f}%",
            ]
        )
        blackbox_report.add_row(["", ""])
        blackbox_report.add_row(["", ""])
        blackbox_report.add_row(["", ""])
        blackbox_report.add_row(["", ""])
        blackbox_report.add_row(["", ""])
        blackbox_report.add_row(["", ""])
        blackbox_report.add_row(
            [
                "# Input features:",
                self.bb_n_input_features,
            ]
        )
        blackbox_report.add_row(["# Output classes:", self.bb_n_output_classes])
        blackbox_report.add_row(["", ""])

        performance_report = PrettyTable(title="Performance", header=False)
        performance_report.add_column(
            "Performance",
            [self._score_report(self.y_test, self.y_pred)],
        )

        summary.add_column("Blackbox", [blackbox_report, performance_report])

        whitebox_report = PrettyTable(border=False, header=False)
        whitebox_report.align = "l"
        whitebox_report.add_row(["Explanation method:", "Trustee"])
        whitebox_report.add_row(["Model:", type(self.max_dt).__name__])
        whitebox_report.add_row(["Iterations:", self.trustee_num_iter])
        whitebox_report.add_row(["Sample size:", f"{self.trustee_sample_size * 100:.2f}%"])
        whitebox_report.add_row(["", ""])
        whitebox_report.add_row(["Decision Tree Info", ""])
        whitebox_report.add_row(["Size:", self.max_dt.tree_.node_count])
        whitebox_report.add_row(["Depth:", self.max_dt.get_depth()])
        whitebox_report.add_row(["Leaves:", self.max_dt.get_n_leaves()])
        whitebox_report.add_row(
            [
                "# Input features:",
                f"{self.trustee.get_n_features()} ({self.trustee.get_n_features() / self.bb_n_input_features * 100:.2f}%)",
            ]
        )
        whitebox_report.add_row(
            [
                "# Output classes:",
                f"{self.trustee.get_n_classes()} ({self.trustee.get_n_classes() / self.bb_n_output_classes * 100:.2f}%)",
            ]
        )
        whitebox_report.add_row(["", ""])

        fidelity_report = PrettyTable(title="Fidelity", header=False)
        fidelity_report.add_column(
            "Fidelity",
            [self._score_report(self.y_pred, self.max_dt_y_pred)],
        )
        summary.add_column("Whitebox", [whitebox_report, fidelity_report])

        if self.min_dt:
            min_whitebox_report = PrettyTable(border=False, header=False)
            min_whitebox_report.align = "l"
            min_whitebox_report.add_row(["Explanation method:", "Trustee"])
            min_whitebox_report.add_row(["Model:", type(self.min_dt).__name__])
            min_whitebox_report.add_row(["Iterations:", self.trustee_num_iter])
            min_whitebox_report.add_row(["Sample size:", f"{self.trustee_sample_size * 100:.2f}%"])
            min_whitebox_report.add_row(["", ""])
            min_whitebox_report.add_row(["Decision Tree Info", ""])
            min_whitebox_report.add_row(["Size:", self.min_dt.tree_.node_count])
            min_whitebox_report.add_row(["Depth:", self.min_dt.get_depth()])
            min_whitebox_report.add_row(["Leaves:", self.min_dt.get_n_leaves()])
            min_whitebox_report.add_row(["Top-k:", self.top_k])
            min_whitebox_report.add_row(["# Input features:", "-"])
            min_whitebox_report.add_row(
                [
                    "# Output classes:",
                    f"{self.min_dt.tree_.n_classes[0]} ({self.min_dt.tree_.n_classes[0] / self.bb_n_output_classes * 100:.2f}%)",
                ]
            )
            min_whitebox_report.add_row(["", ""])

            min_fidelity_report = PrettyTable(title="Fidelity", header=False)
            min_fidelity_report.add_column(
                "Fidelity",
                [self._score_report(self.y_pred, self.min_dt_y_pred)],
            )
            summary.add_column("Top-k Whitebox", [min_whitebox_report, min_fidelity_report])

        single_analysis = PrettyTable(title="Single-run Analysis", header=False)
        single_analysis_first_row = PrettyTable(header=False, border=False)

        top_features = PrettyTable(
            title=f"Top {len(self.max_dt_top_features)} Features",
            field_names=["Feature", "# of Nodes (%)", "Data Split % - ↓"],
        )

        sum_nodes = 0
        sum_nodes_perc = 0
        sum_data_split = 0
        for (feat, values) in self.max_dt_top_features:
            node, node_perc, data_split = (
                values["count"],
                (values["count"] / (self.max_dt.tree_.node_count - self.max_dt.tree_.n_leaves)) * 100,
                (values["samples"] / self.trustee.get_samples_sum()) * 100,
            )
            sum_nodes += node
            sum_nodes_perc += node_perc
            sum_data_split += data_split

            top_features.add_row(
                [
                    self.feature_names[feat] if self.feature_names else feat,
                    f"{node} ({node_perc:.2f}%)",
                    f"{values['samples']} ({data_split:.2f}%)",
                ]
            )
            top_features.add_row(["", "", ""])

        top_features.add_row(["-" * 10, "-" * 10, "-" * 10])
        top_features.add_row(
            [
                f"Top {len(self.max_dt_top_features)} Summary",
                f"{sum_nodes} ({sum_nodes_perc:.2f}%)",
                f"{sum_data_split:.2f}%",
            ]
        )

        top_nodes = PrettyTable(
            title=f"Top {len(self.max_dt_top_nodes)} Nodes",
            field_names=[
                "Decision",
                "Gini  Split - ↓",
                "Data Split % - ↓",
                "Data Split % by Class (L/R)",
            ],
        )
        top_nodes.align = "l"
        top_nodes.valign = "m"

        for node in self.max_dt_top_nodes:
            samples_by_class = [
                (
                    self.class_names[idx] if self.class_names is not None and idx < len(self.class_names) else idx,
                    (count_left / self.max_dt.tree_.value[0][0][idx]) * 100,
                    (count_right / self.max_dt.tree_.value[0][0][idx]) * 100,
                )
                for idx, (count_left, count_right) in enumerate(node["data_split_by_class"])
            ]
            samples_left = (node["data_split"][0] / self.max_dt.tree_.n_node_samples[0]) * 100
            samples_right = (node["data_split"][1] / self.max_dt.tree_.n_node_samples[0]) * 100
            top_nodes.add_row(
                [
                    f"{self.feature_names[node['feature']] if self.feature_names else node['feature']} <= {node['threshold']}",
                    f"Left: {node['gini_split'][0]:.2f} \nRight: {node['gini_split'][1]:.2f}",
                    f"Left: {samples_left:.2f}% \nRight: {samples_right:.2f}%",
                    "\n".join(f"{row[0]}: {row[1]:.2f}% / {row[2]:.2f}%" for row in samples_by_class),
                ]
            )
            top_nodes.add_row(["", "", "", ""])

        top_branches = PrettyTable(
            title=f"Top {len(self.max_dt_top_branches)} Branches",
            field_names=["Rule", "Decision (P(x))", "Samples (%) - ↓", "Class Samples (%)"],
        )
        top_branches.align = "l"
        top_branches.valign = "m"

        sum_samples = 0
        sum_samples_perc = 0
        sum_class_samples_perc = {}
        for branch in self.max_dt_top_branches:
            samples, samples_perc, class_samples_perc = (
                branch["samples"],
                (branch["samples"] / self.max_dt.tree_.n_node_samples[0]) * 100,
                (branch["samples"] / self.max_dt.tree_.value[0][0][branch["class"]]) * 100 if self.is_classify else 0,
            )
            sum_samples += samples
            sum_samples_perc += samples_perc

            if branch["class"] not in sum_class_samples_perc:
                sum_class_samples_perc[branch["class"]] = 0
            sum_class_samples_perc[branch["class"]] += class_samples_perc

            branch_class = (
                self.class_names[branch["class"]]
                if self.class_names is not None and branch["class"] < len(self.class_names)
                else branch["class"],
            )
            top_branches.add_row(
                [
                    "\n and ".join(
                        [
                            f"{self.feature_names[feat] if self.feature_names else feat} {op} {threshold}"
                            for (_, feat, op, threshold) in branch["path"]
                        ]
                    ),
                    f"{branch_class}\n({branch['prob']:.2f}%)",
                    f"{samples}\n({samples_perc:.2f}%)",
                    f"{class_samples_perc:.2f}%",
                ]
            )
            top_branches.add_row(["", "", "", ""])

        top_branches.add_row(["-" * 10, "-" * 10, "-" * 10, "-" * 10])
        top_branches.add_row(
            [
                f"Top {len(self.max_dt_top_branches)} Summary",
                "-",
                f"{sum_samples} ({sum_samples_perc:.2f}%)",
                "\n".join(
                    [
                        f"{self.class_names[class_idx] if self.class_names is not None and class_idx < len(self.class_names) else class_idx}:{class_perc:.2f}%"
                        for (class_idx, class_perc) in sum_class_samples_perc.items()
                    ]
                ),
            ]
        )

        single_analysis_first_row.add_column("Top Nodes", [top_nodes])
        single_analysis_first_row.add_column("Top Branches", [top_branches])

        single_analysis.add_column("Single Analysis", [top_features, single_analysis_first_row])

        if self.num_pruning_iter > 0:
            prunning_analysis = PrettyTable(title="Prunning Analysis", header=False)
            top_k_prune_performance = PrettyTable(
                title="Trustee Top-k Iteration",
                field_names=[
                    "k",
                    "DT Size",
                    "DT Depth",
                    "DT Num Leaves",
                    "Performance",
                    "Fidelity",
                ],
            )
            top_k_prune_performance.align = "l"
            top_k_prune_performance.valign = "m"

            for i in self.top_k_prune_iter:
                top_k_prune_performance.add_row(
                    [
                        i["top_k"],
                        i["dt"].tree_.node_count,
                        i["dt"].get_depth(),
                        i["dt"].get_n_leaves(),
                        i["score_report"],
                        i["fidelity_report"],
                    ]
                )
                top_k_prune_performance.add_row(["", "", "", "", "", ""])

            alpha_performance = PrettyTable(
                title="CCP Alpha Iteration",
                field_names=[
                    "Alpha",
                    "Gini",
                    "DT Size",
                    "DT Depth",
                    "DT Num Leaves",
                    "Performance",
                    "Fidelity",
                ],
            )
            alpha_performance.align = "l"
            alpha_performance.valign = "m"

            for i in self.ccp_iter:
                alpha_performance.add_row(
                    [
                        i["ccp_alpha"],
                        f"{i['gini']:.3f}",
                        i["dt"].tree_.node_count,
                        i["dt"].get_depth(),
                        i["dt"].get_n_leaves(),
                        i["score_report"],
                        i["fidelity_report"],
                    ]
                )
                alpha_performance.add_row(["", "", "", "", "", "", ""])

            max_depth_performance = PrettyTable(
                title="Max Depth Iteration",
                field_names=[
                    "Max Depth",
                    "DT Size",
                    "DT Depth",
                    "DT Num Leaves",
                    "Performance",
                    "Fidelity",
                ],
            )
            max_depth_performance.align = "l"
            max_depth_performance.valign = "m"

            for i in self.max_depth_iter:
                max_depth_performance.add_row(
                    [
                        i["max_depth"],
                        i["dt"].tree_.node_count,
                        i["dt"].get_depth(),
                        i["dt"].get_n_leaves(),
                        i["score_report"],
                        i["fidelity_report"],
                    ]
                )
                max_depth_performance.add_row(["", "", "", "", "", ""])

            max_leaves_performance = PrettyTable(
                title="Max Leaves Iteration",
                field_names=[
                    "Max Leaves",
                    "DT Size",
                    "DT Depth",
                    "DT Num Leaves",
                    "Performance",
                    "Fidelity",
                ],
            )
            max_leaves_performance.align = "l"
            max_leaves_performance.valign = "m"

            for i in self.max_leaves_iter:
                max_leaves_performance.add_row(
                    [
                        i["max_leaves"],
                        i["dt"].tree_.node_count,
                        i["dt"].get_depth(),
                        i["dt"].get_n_leaves(),
                        i["score_report"],
                        i["fidelity_report"],
                    ]
                )
                max_leaves_performance.add_row(["", "", "", "", "", ""])

            prunning_analysis.add_column(
                "Prunning Iteration",
                [top_k_prune_performance, alpha_performance, max_depth_performance, max_leaves_performance],
            )

        if not self.skip_retrain:
            repeated_analysis = PrettyTable(title="Repeated-run Analysis", header=False)
            iter_performance = PrettyTable(
                title="Iterative Feature Removal",
                field_names=[
                    "Iteration",
                    "Feature Removed",
                    "# Features Removed",
                    "Performance",
                    "Decision Tree Size",
                    "Fidelity",
                ],
            )
            iter_performance.align = "l"
            iter_performance.valign = "m"

            for i in self.whitebox_iter:
                iter_performance.add_row(
                    [
                        i["it"],
                        self.feature_names[i["feature_removed"]] if self.feature_names else i["feature_removed"],
                        i["n_features_removed"],
                        i["score_report"],
                        i["dt"].tree_.node_count,
                        i["fidelity_report"],
                    ]
                )
                iter_performance.add_row(["", "", "", "", "", ""])

            repeated_analysis.add_column("Iterative Feature Removal", [iter_performance])

        report.add_column(
            "Report",
            [summary, single_analysis]
            + [prunning_analysis if self.num_pruning_iter > 0 else []]
            + ([repeated_analysis] if not self.skip_retrain else []),
        )

        return f"\n{report}"

    def _score(self, y, y_pred):
        if self.is_classify:
            return f1_score(y, y_pred, average="macro", zero_division=0)

        return r2_score(y, y_pred)

    def _score_report(self, y, y_pred):
        if self.is_classify:
            return f"\n{classification_report(y, y_pred, digits=3, zero_division=0)}"

        return f"R2 Score: {r2_score(y, y_pred)}"

    def _progress(self, finish=False, length=100, fill="█", end="\r"):
        """
        Call in a loop to create terminal progress bar
        @params:
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
            end         - Optional  : end character (e.g. "\r", "\r\n") (Str)
        """
        self.step = self.step + 1
        if self.step > self.total_steps or finish:
            self.step = self.total_steps

        percent = f"{100 * (self.step / float(self.total_steps)):.1f}"
        filled_length = int(length * self.step // self.total_steps)
        progress_bar = fill * filled_length + "-" * (length - filled_length)
        print(f"\rProgress |{progress_bar}| {percent}% Complete", end=end)
        if self.step == self.total_steps or self.verbose:
            # if it's running verbose, log messages will get in the way, so we better print the bar multiple times
            print()

    def _prepare_data(self):
        """Data preparation for trust report"""
        log = self.logger.log if self.logger else print

        if self.verbose:
            log("Preparing data...")

        if (self.X is None and (self.X_train is None or self.X_test is None)) or (
            self.y is None and (self.y_train is None or self.y_test is None)
        ):
            raise ValueError("Missing either X and y arguments or X_train, X_test, y_train and y_test arguments.")

        if self.X_train is not None:
            # convert to pandas DataFrame to avoid dealing with other formats
            self.X_train = pd.DataFrame(self.X_train)
            self.X_test = pd.DataFrame(self.X_test)
            self.y_train = pd.Series(self.y_train)
            self.y_test = pd.Series(self.y_test)
        else:
            # if data split is not given as a param, split the dataset randomly
            if self.verbose:
                log("Splitting dataset for training and testing...")

            # convert to pandas DataFrame to avoid dealing with other formats
            self.X = pd.DataFrame(self.X)
            self.y = pd.Series(self.y)

            X_indexes = np.arange(0, self.X.shape[0])
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X_indexes, self.y, train_size=self.train_size
            )
            self.X_train = self.X.iloc[self.X_train]
            self.X_test = self.X.iloc[self.X_test]

            if self.verbose:
                log(f"X size: {len(self.X)}; y size: {len(self.y)}")
                log("Done!")

        self.dataset_size = len(self.X_train) + len(self.X_test)
        self.train_size = len(self.X_train) / self.dataset_size

        if self.feature_names is not None:
            self.feature_names = list(self.feature_names)

        if self.verbose:
            log("Done!")

        self._progress()

    def _fit_and_explain(
        self,
        X_train=None,
        X_test=None,
        trustee_num_stability_iter=None,
        trustee_ccp_alpha=0.0,
        trustee_max_leaf_nodes=None,
        trustee_max_depth=None,
        trustee_use_features=None,
    ):
        """
        Fits blacbox with the given X and y data, and uses Trustee to extract DT explanation
        """
        log = self.logger.log if self.logger else print

        if self.verbose:
            log("Fitting blackbox model...")

        X_train = X_train if X_train is not None else self.X_train
        X_test = X_test if X_test is not None else self.X_test

        if self.skip_retrain:
            blackbox_copy = self.blackbox
        else:
            # clone blackbox params but resets training weights to allow retraining with new dataset
            try:
                # scikit-learn models
                blackbox_copy = clone(self.blackbox)
            except Exception as warn1:
                if self.verbose:
                    log("WARNING", warn1)
                try:
                    # pytorch models
                    blackbox_copy = copy.copy(self.blackbox)
                    if hasattr(blackbox_copy, "children"):
                        for layer in blackbox_copy.children():
                            if hasattr(layer, "reset_parameters"):
                                layer.reset_parameters()
                    else:
                        raise Exception("Not a PyTorch model")
                except Exception as warn2:
                    if self.verbose:
                        log("WARNING", warn2)
                    # AutoGluon and any other models
                    blackbox_copy = (
                        self.blackbox.__class__(self.blackbox._learner.label)
                        if hasattr(self.blackbox, "_learner")
                        else self.blackbox.__class__()
                    )

            # retrain model
            if hasattr(self.blackbox, "_learner"):
                # AutoGluon model
                args = {}
                args[self.blackbox._learner.label] = self.y_train
                training_data = X_train.assign(**args)
                blackbox_copy.fit(training_data)
            else:
                blackbox_copy.fit(X_train, self.y_train)

            del self.blackbox

        if self.verbose:
            log("Done!")

        y_pred = getattr(blackbox_copy, self.predict_method_name)(X_test)

        if self.verbose:
            log("Blackbox model score report with training data:")
            log(self._score_report(self.y_test, y_pred))
            log("Using Classification Trustee algorithm to extract DT...")

        trustee = (
            ClassificationTrustee(expert=blackbox_copy) if self.is_classify else RegressionTrustee(expert=blackbox_copy)
        )

        stability_iter = trustee_num_stability_iter if trustee_num_stability_iter else self.trustee_num_stability_iter
        trustee.fit(
            X_train,
            self.y_train,
            top_k=self.top_k,
            num_iter=self.trustee_num_iter,
            num_stability_iter=stability_iter,
            samples_size=self.trustee_sample_size,
            predict_method_name=self.predict_method_name,
            max_leaf_nodes=trustee_max_leaf_nodes if trustee_max_leaf_nodes else self.trustee_max_leaf_nodes,
            max_depth=trustee_max_depth if trustee_max_depth else self.trustee_max_depth,
            ccp_alpha=trustee_ccp_alpha if trustee_ccp_alpha else self.trustee_ccp_alpha,
            use_features=trustee_use_features,
            verbose=self.verbose,
        )

        if self.verbose:
            log("Done!")

        dt, min_dt, agreement, reward = trustee.explain()
        if self.verbose:
            log(f"Model explanation training (agreement, fidelity): ({agreement}, {reward})")
            log(f"Top-k Prunned explanation size: {min_dt.tree_.node_count}")

        if trustee_use_features:
            X_test = X_test.iloc[:, trustee_use_features]

        dt_y_pred = dt.predict(X_test.values)
        min_dt_y_pred = min_dt.predict(X_test.values)

        if self.verbose:
            log("Model explanation global fidelity report:")
            log(self._score_report(y_pred, dt_y_pred))
            log("Top-k Model explanation global fidelity report:")
            log(self._score_report(y_pred, min_dt_y_pred))

            log("Model explanation score report:")
            log(self._score_report(self.y_test, dt_y_pred))
            log("Top-k Model explanation score report:")
            log(self._score_report(self.y_test, min_dt_y_pred))

        self.blackbox = blackbox_copy

        return trustee, y_pred, dt, dt_y_pred, min_dt, min_dt_y_pred

    def _collect(self):
        """Collects data to build the make report"""
        self._collect_blackbox()
        self._collect_trustee()

        if self.analyze_stability:
            self._collect_stability_analysis()

        if self.analyze_branches:
            self._collect_branch_analysis()

        if self.num_pruning_iter > 0:
            self._collect_top_k_prunning()
            self._collect_ccp_prunning()
            self._collect_max_depth_prunning()
            self._collect_max_leaves_prunning()

        if not self.skip_retrain:
            self._collect_features_iter_removal()

        self._progress(finish=100)

    def _collect_blackbox(self):
        """Collects information on analyzed blackbox"""
        log = self.logger.log if self.logger else print
        if self.verbose:
            log("Collecting blackbox information...")

        self.bb_n_input_features = len(self.X_train.columns)
        self.bb_n_output_classes = len(np.unique(self.y_train))
        if self.verbose:
            log("Done!")

        self._progress()

    def _collect_trustee(self):
        """Uses provided dataset to train a Decision Tree and fetch first decision tree info"""
        log = self.logger.log if self.logger else print
        if self.verbose:
            log("Collecting trustee information...")

        (
            self.trustee,
            self.y_pred,
            self.max_dt,
            self.max_dt_y_pred,
            self.min_dt,
            self.min_dt_y_pred,
        ) = self._fit_and_explain()

        self.max_dt_leaves_by_level = self.trustee.get_leaves_by_level()
        self.max_dt_samples_by_level = self.trustee.get_samples_by_level()

        self.max_dt_top_nodes = self.trustee.get_top_nodes(top_k=self.top_k)
        self.max_dt_top_features = self.trustee.get_top_features(top_k=self.top_k)
        self.max_dt_top_branches = self.trustee.get_top_branches(top_k=self.top_k)
        self.max_dt_all_branches = self.trustee.get_top_branches(top_k=self.max_dt.get_n_leaves())

        if self.analyze_branches:
            # removes initial guess value and updates total steps with the correct number of branches
            initial_guess = self.trustee_max_leaf_nodes if self.trustee_max_leaf_nodes else 100
            self.total_steps += len(self.max_dt_all_branches) - initial_guess

        if self.verbose:
            log("Done!")

        self._progress()

    def _collect_branch_analysis(self):
        """Uses trained trustee explainer to show how different branches affect fidelity"""
        log = self.logger.log if self.logger else print
        if self.verbose:
            log("Collecting branch analysis information...")

        self.branch_iter = []
        for top_k in np.arange(1, self.max_dt.get_n_leaves()):
            if self.verbose:
                log(f"Iteration {top_k}/{self.max_dt.get_n_leaves()}")

            pruned_dt = self.trustee.prune(top_k=top_k)
            pruned_dt_y_pred = pruned_dt.predict(self.X_test.values)

            self.branch_iter.append(
                {
                    "top_k": top_k,
                    "dt": pruned_dt,
                    "y_pred": self.y_pred,
                    "dt_y_pred": pruned_dt_y_pred,
                    "score": self._score(self.y_test, pruned_dt_y_pred),
                    "score_report": self._score_report(self.y_test, pruned_dt_y_pred),
                    "fidelity": self._score(self.y_pred, pruned_dt_y_pred),
                    "fidelity_report": self._score_report(self.y_pred, pruned_dt_y_pred),
                }
            )
            self._progress()

        if self.verbose:
            log("Done!")

    def _collect_stability_analysis(self):
        """Uses trained trustee explainer to analyze the stability of top-k branches over multiple iterations"""
        log = self.logger.log if self.logger else print
        if self.verbose:
            log("Collecting stability analysis information...")

        self.stability_iter = []
        for i in range(self.max_iter):
            if self.verbose:
                log(f"Iteration {i}/{self.max_iter}")

            (trustee, y_pred, max_dt, max_dt_y_pred, min_dt, min_dt_y_pred) = self._fit_and_explain(
                trustee_num_stability_iter=1  # prevents trustee`s outer loop from running so we can see how unstable explanations are
            )
            top_branches = trustee.get_top_branches(top_k=max_dt.get_n_leaves())
            self.stability_iter.append(
                {
                    "max_dt": max_dt,
                    "min_dt": min_dt,
                    "max_dt_fidelity": self._score(y_pred, max_dt_y_pred),
                    "min_dt_fidelity": self._score(y_pred, min_dt_y_pred),
                    "iteration": i,
                    "top_branches": top_branches,
                }
            )
            self._progress()

        if self.verbose:
            log("Done!")

    def _collect_top_k_prunning(self):
        """Uses trained trustee explainer to prune the decision tree with different top_k branches"""
        log = self.logger.log if self.logger else print
        if self.verbose:
            log("Collecting top-k prunning information...")

        self.top_k_prune_iter = []
        for top_k in np.arange(1, self.num_pruning_iter + 1):
            if self.verbose:
                log(f"Iteration {top_k}/{self.num_pruning_iter}")

            pruned_dt = self.trustee.prune(top_k=top_k)
            pruned_dt_y_pred = pruned_dt.predict(self.X_test.values)

            self.top_k_prune_iter.append(
                {
                    "top_k": top_k,
                    "dt": pruned_dt,
                    "y_pred": self.y_pred,
                    "dt_y_pred": pruned_dt_y_pred,
                    "score": self._score(self.y_test, pruned_dt_y_pred),
                    "score_report": self._score_report(self.y_test, pruned_dt_y_pred),
                    "fidelity": self._score(self.y_pred, pruned_dt_y_pred),
                    "fidelity_report": self._score_report(self.y_pred, pruned_dt_y_pred),
                }
            )
            self._progress()

        if self.verbose:
            log("Done!")

    def _collect_ccp_prunning(self):
        """Uses provided dataset to train multiple Decision Trees and fetch CCP prunning info"""
        log = self.logger.log if self.logger else print
        if self.verbose:
            log("Collecting CCP prunning information...")

        ccp_clf = tree.DecisionTreeClassifier(random_state=0)
        ccp_path = ccp_clf.cost_complexity_pruning_path(self.X_train, self.y_train)
        ccp_alphas, impurities = ccp_path.ccp_alphas, ccp_path.impurities

        self.ccp_iter = []
        for idx, ccp_alpha in enumerate(ccp_alphas[-self.num_pruning_iter :]):
            if ccp_alpha >= 0:
                if self.verbose:
                    log(f"Iteration {idx}/{self.num_pruning_iter}")

                gini = impurities[idx]
                _, _, ccp_dt, ccp_dt_y_pred, _, _ = self._fit_and_explain(trustee_ccp_alpha=ccp_alpha)

                self.ccp_iter.append(
                    {
                        "ccp_alpha": ccp_alpha,
                        "gini": gini,
                        "dt": ccp_dt,
                        "y_pred": self.y_pred,
                        "dt_y_pred": ccp_dt_y_pred,
                        "score": self._score(self.y_test, ccp_dt_y_pred),
                        "score_report": self._score_report(self.y_test, ccp_dt_y_pred),
                        "fidelity": self._score(self.y_pred, ccp_dt_y_pred),
                        "fidelity_report": self._score_report(self.y_pred, ccp_dt_y_pred),
                    }
                )
                self._progress()

        if self.verbose:
            log("Done!")

    def _collect_max_depth_prunning(self):
        """Uses provided dataset to train multiple Decision Trees and fetch max depth prunning info"""
        log = self.logger.log if self.logger else print
        if self.verbose:
            log("Collecting max depth prunning information...")

        self.max_depth_iter = []
        for max_depth in np.arange(1, self.num_pruning_iter + 1):
            if max_depth > 0:
                if self.verbose:
                    log(f"Iteration {max_depth}/{self.num_pruning_iter}")

                _, _, max_depth_dt, max_depth_dt_y_pred, _, _ = self._fit_and_explain(trustee_max_depth=max_depth)

                self.max_depth_iter.append(
                    {
                        "max_depth": max_depth,
                        "dt": max_depth_dt,
                        "y_pred": self.y_pred,
                        "dt_y_pred": max_depth_dt_y_pred,
                        "score": self._score(self.y_test, max_depth_dt_y_pred),
                        "score_report": self._score_report(self.y_test, max_depth_dt_y_pred),
                        "fidelity": self._score(self.y_pred, max_depth_dt_y_pred),
                        "fidelity_report": self._score_report(self.y_pred, max_depth_dt_y_pred),
                    }
                )
                self._progress()

        if self.verbose:
            log("Done!")

    def _collect_max_leaves_prunning(self):
        """Uses provided dataset to train multiple Decision Trees and fetch max leaves prunning info"""
        log = self.logger.log if self.logger else print
        if self.verbose:
            log("Collecting max leaves prunning information...")

        self.max_leaves_iter = []
        for max_leaves in np.arange(2, self.num_pruning_iter + 2):
            if max_leaves > 1:
                if self.verbose:
                    log(f"Iteration {max_leaves}/{self.num_pruning_iter}")

                _, _, max_leaves_dt, max_leaves_dt_y_pred, _, _ = self._fit_and_explain(
                    trustee_max_leaf_nodes=max_leaves
                )

                self.max_leaves_iter.append(
                    {
                        "max_leaves": max_leaves,
                        "dt": max_leaves_dt,
                        "y_pred": self.y_pred,
                        "dt_y_pred": max_leaves_dt_y_pred,
                        "score": self._score(self.y_test, max_leaves_dt_y_pred),
                        "score_report": self._score_report(self.y_test, max_leaves_dt_y_pred),
                        "fidelity": self._score(self.y_pred, max_leaves_dt_y_pred),
                        "fidelity_report": self._score_report(self.y_pred, max_leaves_dt_y_pred),
                    }
                )
                self._progress()

        if self.verbose:
            log("Done!")

    def _collect_features_iter_removal(self):
        """Uses provided dataset to train multiple Decision Trees by iteratively removing the most important features from the dataset"""
        log = self.logger.log if self.logger else print
        if self.verbose:
            log("Collecting features prunning information...")

        self.whitebox_iter = []

        i = 0
        n_features_removed = 0
        top_feature_to_remove = self.max_dt_top_features[0][0]
        X_train_iter = self.X_train.copy()
        X_test_iter = self.X_test.copy()

        while i < self.max_iter and n_features_removed < self.bb_n_input_features - 1:
            if self.verbose:
                log(f"Iteration {i + 1}/{min(self.max_iter, self.bb_n_input_features)}")

            # remove most significant feature
            X_train_iter.iloc[:, top_feature_to_remove] = 0
            X_test_iter.iloc[:, top_feature_to_remove] = 0

            n_features_removed += 1
            _, y_pred, dt, dt_y_pred, _, _ = self._fit_and_explain(X_train=X_train_iter, X_test=X_test_iter)

            self.whitebox_iter.append(
                {
                    "it": i,
                    "dt": dt,
                    "y_pred": y_pred,
                    "dt_y_pred": dt_y_pred,
                    "feature_removed": top_feature_to_remove,
                    "n_features_removed": n_features_removed,
                    "score": self._score(self.y_test, y_pred),
                    "score_report": self._score_report(self.y_test, y_pred),
                    "fidelity": self._score(y_pred, dt_y_pred),
                    "fidelity_report": self._score_report(y_pred, dt_y_pred),
                }
            )

            iter_dt_features, _, _ = get_dt_info(dt)
            iter_dt_top_features = sorted(iter_dt_features.items(), key=lambda p: p[1]["samples"], reverse=True)[
                : self.top_k
            ]
            top_feature_to_remove = iter_dt_top_features[0][0]
            i += 1
            self._progress()

        if self.verbose:
            log("Done!")

    def _save_dts(self, output_dir, save_all=False):
        """
        Save the decision trees.

        Parameters
        ----------
        output_dir : str
            The output directory to save the decision trees.
        """
        log = self.logger.log if self.logger else print
        if self.verbose:
            log("Saving decision trees...")

        dot_data = tree.export_graphviz(
            self.max_dt,
            class_names=self.class_names,
            feature_names=self.feature_names,
            filled=True,
            rounded=True,
            special_characters=True,
        )
        graph = graphviz.Source(dot_data)
        graph.render(f"{output_dir}/trust_report_dt")

        if self.min_dt:
            dot_data = tree.export_graphviz(
                self.min_dt,
                class_names=self.class_names,
                feature_names=self.feature_names,
                filled=True,
                rounded=True,
                special_characters=True,
            )
            graph = graphviz.Source(dot_data)
            graph.render(f"{output_dir}/trust_report_pruned_dt")

        if save_all:
            stability_output_dir = f"{output_dir}/stability_max"
            if not os.path.exists(stability_output_dir):
                os.makedirs(stability_output_dir)

            log("Saving stability decision trees...")
            for idx, i in enumerate(self.stability_iter):
                dot_data = tree.export_graphviz(
                    i["max_dt"],
                    class_names=self.class_names,
                    feature_names=self.feature_names,
                    filled=True,
                    rounded=True,
                    special_characters=True,
                )
                graph = graphviz.Source(dot_data)
                graph.render(f"{stability_output_dir}/stability_dt_{idx}")

            stability_output_dir = f"{output_dir}/stability_min"
            if not os.path.exists(stability_output_dir):
                os.makedirs(stability_output_dir)

            log("Saving stability decision trees...")
            for idx, i in enumerate(self.stability_iter):
                dot_data = tree.export_graphviz(
                    i["min_dt"],
                    class_names=self.class_names,
                    feature_names=self.feature_names,
                    filled=True,
                    rounded=True,
                    special_characters=True,
                )
                graph = graphviz.Source(dot_data)
                graph.render(f"{stability_output_dir}/stability_dt_{idx}")

            prunning_output_dir = f"{output_dir}/prunning"
            if not os.path.exists(prunning_output_dir):
                os.makedirs(prunning_output_dir)

            for idx, i in enumerate(self.ccp_iter):
                dot_data = tree.export_graphviz(
                    i["dt"],
                    class_names=self.class_names,
                    feature_names=self.feature_names,
                    filled=True,
                    rounded=True,
                    special_characters=True,
                )
                graph = graphviz.Source(dot_data)
                graph.render(f"{prunning_output_dir}/ccp_dt_{idx}_{i['dt'].tree_.node_count}")

            for idx, i in enumerate(self.max_depth_iter):
                dot_data = tree.export_graphviz(
                    i["dt"],
                    class_names=self.class_names,
                    feature_names=self.feature_names,
                    filled=True,
                    rounded=True,
                    special_characters=True,
                )
                graph = graphviz.Source(dot_data)
                graph.render(f"{prunning_output_dir}/max_depth_dt_{idx}_{i['dt'].tree_.node_count}")

            for idx, i in enumerate(self.max_leaves_iter):
                dot_data = tree.export_graphviz(
                    i["dt"],
                    class_names=self.class_names,
                    feature_names=self.feature_names,
                    filled=True,
                    rounded=True,
                    special_characters=True,
                )
                graph = graphviz.Source(dot_data)
                graph.render(f"{prunning_output_dir}/max_leaves_dt_{idx}_{i['dt'].tree_.node_count}")

        if self.verbose:
            log("Done!")

    def plot(self, output_dir, aggregate=False):
        """
        Plot the analysis results.

        Parameters
        ----------
        output_dir : str
            The output directory to save the plots.
        """
        log = self.logger.log if self.logger else print
        if self.verbose:
            log("Plotting...")

        plots_output_dir = f"{output_dir}/plots"
        if not os.path.exists(plots_output_dir):
            os.makedirs(plots_output_dir)

        plot_top_features(
            self.max_dt_top_features,
            self.trustee.get_samples_sum(),
            (self.max_dt.tree_.node_count - self.max_dt.tree_.n_leaves),
            plots_output_dir,
            feature_names=self.feature_names,
        )
        plot_top_nodes(
            self.max_dt_top_nodes,
            self.max_dt.tree_.value[0][0],
            self.max_dt.tree_.n_node_samples[0],
            plots_output_dir,
            feature_names=self.feature_names,
            class_names=self.class_names,
        )
        plot_top_branches(
            self.max_dt_top_branches,
            self.max_dt.tree_.value[0][0],
            self.max_dt.tree_.n_node_samples[0],
            plots_output_dir,
            class_names=self.class_names,
            is_classify=self.is_classify,
        )
        plot_all_branches(
            self.max_dt_all_branches,
            self.max_dt.tree_.value[0][0],
            self.max_dt.tree_.n_node_samples[0],
            plots_output_dir,
            class_names=self.class_names,
            is_classify=self.is_classify,
        )
        plot_samples_by_level(
            self.max_dt_samples_by_level,
            self.max_dt_leaves_by_level,
            self.max_dt.tree_.n_node_samples[0],
            plots_output_dir,
        )
        plot_dts_fidelity_by_size(
            [
                {"type": "CCP", "iter": self.ccp_iter},
                {"type": "Max Depth", "iter": self.max_depth_iter},
                {"type": "Max Leaves", "iter": self.max_leaves_iter},
                {"type": "Trustee Top-k", "iter": self.top_k_prune_iter},
            ],
            plots_output_dir,
        )
        plot_dts_fidelity_by_size(
            [{"type": "Top-k Branches", "iter": self.branch_iter}],
            plots_output_dir,
            filename="branches",
        )

        plot_stability(
            self.stability_iter,
            self.X_test,
            self.y_test,
            self.max_dt,
            "max_dt",
            self.max_dt_top_branches,
            plots_output_dir,
            class_names=self.class_names,
            is_classify=self.is_classify,
        )

        plot_stability(
            self.stability_iter,
            self.X_test,
            self.y_test,
            self.min_dt,
            "min_dt",
            self.max_dt_top_branches,
            plots_output_dir,
            class_names=self.class_names,
            is_classify=self.is_classify,
        )

        plot_stability_heatmap(
            self.stability_iter,
            self.X_test,
            self.y_test,
            "max_dt",
            self.max_dt_top_branches,
            plots_output_dir,
            class_names=self.class_names,
            is_classify=self.is_classify,
        )

        plot_stability_heatmap(
            self.stability_iter,
            self.X_test,
            self.y_test,
            "min_dt",
            self.max_dt_top_branches,
            plots_output_dir,
            class_names=self.class_names,
            is_classify=self.is_classify,
        )

        if self.is_classify:
            plot_distribution(
                self.X if self.X is not None else self.X_train,
                self.y if self.y is not None else self.y_train,
                self.max_dt_top_branches,
                plots_output_dir,
                feature_names=self.feature_names,
                class_names=self.class_names,
                aggregate=aggregate,
            )

        if not self.skip_retrain:
            plot_accuracy_by_feature_removed(
                self.whitebox_iter,
                plots_output_dir,
                feature_names=self.feature_names,
            )

        if self.verbose:
            log("Done!")

    @classmethod
    def load(cls, path):
        """
        Load the trust report from a file.

        Parameters
        ----------
        path : str
            The path to the file.
        """
        report = None
        with open(path, "rb") as file:
            report = pickle.load(file)

        return report

    def save(self, output_dir, aggregate=False, save_all_dts=False):
        """Saves report and plots to output dir"""
        if output_dir:
            log = self.logger.log if self.logger else print
            if self.verbose:
                log("Saving...")

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            report_output_dir = f"{output_dir}/report"
            if not os.path.exists(report_output_dir):
                os.makedirs(report_output_dir)

            with open(f"{report_output_dir}/trust_report.txt", "w", encoding="utf-8") as file:
                file.write(f"\n{str(self)}")

            with open(f"{report_output_dir}/trust_report.obj", "wb") as file:
                pickle.dump(self, file)

            self._save_dts(report_output_dir, save_all=save_all_dts)
            self.plot(report_output_dir, aggregate=aggregate)

            if self.verbose:
                log("Done!")
