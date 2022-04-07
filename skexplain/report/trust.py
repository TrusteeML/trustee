import os
import graphviz
import copy
import torch
import pickle
import numpy as np
import pandas as pd

from sklearn import tree
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.base import clone

from autogluon.tabular import TabularPredictor
from prettytable import PrettyTable

from skexplain.imitation import ClassificationDagger

from .helpers import (
    get_dt_info,
    dt_similarity,
    plot_top_nodes,
    plot_top_branches,
    plot_top_features,
    plot_distribution,
    plot_samples_by_level,
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
        num_quantiles=10,
        train_size=0.7,
        predict_method_name="predict",
        dagger_num_iter=100,
        dagger_sample_size=0.5,
        dagger_max_leaf_nodes=None,
        dagger_max_depth=None,
        dagger_ccp_alpha=0.0,
        skip_retrain=False,
        top_n=10,
        logger=None,
        verbose=False,
        class_names=None,
        feature_names=None,
    ):
        """
        Builds trust report for given black-box model using the Dagger method to extract white-box explanations as Decision Trees.
        """
        self.blackbox = blackbox
        self.X = X
        self.y = y
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.max_iter = max_iter
        self.num_quantiles = num_quantiles
        self.train_size = train_size
        self.predict_method_name = predict_method_name
        self.dagger_num_iter = dagger_num_iter
        self.dagger_sample_size = dagger_sample_size
        self.dagger_max_leaf_nodes = dagger_max_leaf_nodes
        self.dagger_max_depth = dagger_max_depth
        self.dagger_ccp_alpha = dagger_ccp_alpha
        self.skip_retrain = skip_retrain
        self.top_n = top_n
        self.logger = logger
        self.verbose = verbose
        self.class_names = class_names
        self.feature_names = feature_names

        self.first_dt = None
        self.first_y_pred = None
        self.first_dt_y_pred = None

        self.bb_class = None
        self.bb_n_input_features = 0
        self.bb_n_output_classes = 0

        self.first_dt_features = None
        self.first_dt_nodes = None
        self.first_dt_branches = None
        self.first_dt_sum_samples = None

        self.first_dt_class = None
        self.first_dt_size = 0
        self.first_dt_n_leaves = 0
        self.first_dt_samples = 0
        self.first_dt_samples_by_class = 0
        self.first_dt_n_features = 0
        self.first_dt_n_classes = 0

        self.first_dt_samples_by_level = []
        self.first_dt_nodes_by_level = []
        self.first_dt_top_features = []
        self.first_dt_top_nodes = []
        self.first_dt_top_branches = []

        self.ccp_iter = []
        self.max_depth_iter = []
        self.max_leaves_iter = []
        self.whitebox_iter = []

        self._prepare_data()
        self._collect()

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["logger"]
        del state["blackbox"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.logger = None
        self.blackbox = None

    def __str__(self):
        """Formats collected data into a reporto using PrettyTable"""
        ################################################
        #                    REPORT                    #
        ################################################
        report = PrettyTable(title="Classification Trust Report", header=False)

        summary = PrettyTable(title="Summary")
        blackbox_report = PrettyTable(border=False, header=False)
        blackbox_report.align = "l"
        blackbox_report.add_row(["Model:", self.bb_class])
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
        blackbox_report.add_row(["# Input features:", self.bb_n_input_features])
        blackbox_report.add_row(["# Output classes:", self.bb_n_output_classes])
        blackbox_report.add_row(["", ""])

        performance_report = PrettyTable(title="Performance", header=False)
        performance_report.add_column(
            "Performance",
            [
                classification_report(
                    self.y_test,
                    self.first_y_pred,
                    digits=3,
                    # labels=range(len(self.class_names)) if self.class_names else None,
                    # target_names=self.class_names,
                )
            ],
        )

        summary.add_column("Blackbox", [blackbox_report, performance_report])

        whitebox_report = PrettyTable(border=False, header=False)
        whitebox_report.align = "l"
        whitebox_report.add_row(["Explanation method:", "Dagger"])
        whitebox_report.add_row(["Model:", self.first_dt_class])
        whitebox_report.add_row(["Iterations:", self.dagger_num_iter])
        whitebox_report.add_row(["Sample size:", f"{self.dagger_sample_size * 100:.2f}%"])
        whitebox_report.add_row(["", ""])
        whitebox_report.add_row(["Decision Tree Info", ""])
        whitebox_report.add_row(["Size:", self.first_dt_size])
        whitebox_report.add_row(["CCP Alpha:", self.dagger_ccp_alpha])
        whitebox_report.add_row(
            [
                "# Input features:",
                f"{self.first_dt_n_features} ({self.first_dt_n_features / self.bb_n_input_features * 100:.2f}%)",
            ]
        )
        whitebox_report.add_row(
            [
                "# Output classes:",
                f"{self.first_dt_n_classes} ({self.first_dt_n_classes / self.bb_n_output_classes * 100:.2f}%)",
            ]
        )
        whitebox_report.add_row(["", ""])

        fidelity_report = PrettyTable(title="Fidelity", header=False)
        fidelity_report.add_column(
            "Fidelity",
            [
                classification_report(
                    self.first_y_pred,
                    self.first_dt_y_pred,
                    digits=3,
                    # labels=range(len(self.class_names)) if self.class_names else None,
                    # target_names=self.class_names,
                )
            ],
        )
        summary.add_column("Whitebox", [whitebox_report, fidelity_report])

        single_analysis = PrettyTable(title="Single-run Analysis", header=False)
        single_analysis_first_row = PrettyTable(header=False, border=False)

        top_features = PrettyTable(
            title=f"Top {len(self.first_dt_top_features)} Features",
            field_names=["Feature", "# of Nodes (%)", "Data Split % - ↓"],
        )

        sum_nodes = 0
        sum_nodes_perc = 0
        sum_data_split = 0
        for (feat, values) in self.first_dt_top_features:
            node, node_perc, data_split = (
                values["count"],
                (values["count"] / (self.first_dt_size - self.first_dt_n_leaves)) * 100,
                (values["samples"] / self.first_dt_sum_samples) * 100,
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
                f"Top {len(self.first_dt_top_features)} Summary",
                f"{sum_nodes} ({sum_nodes_perc:.2f}%)",
                f"{sum_data_split:.2f}%",
            ]
        )

        top_nodes = PrettyTable(
            title=f"Top {len(self.first_dt_top_nodes)} Nodes",
            field_names=[
                "Decision",
                "Gini  Split - ↓",
                "Data Split % - ↓",
                "Data Split % by Class (L/R)",
            ],
        )
        top_nodes.align = "l"
        top_nodes.valign = "m"

        for node in self.first_dt_top_nodes:
            top_nodes.add_row(
                [
                    # node["idx"],
                    "{} <= {}".format(
                        self.feature_names[node["feature"]] if self.feature_names else node["feature"],
                        node["threshold"],
                    ),
                    f"Left: {node['gini_split'][0]:.2f} \nRight: {node['gini_split'][1]:.2f}",
                    f"Left: {((node['data_split'][0] / self.first_dt_samples) * 100):.2f}% \nRight: {((node['data_split'][1] / self.first_dt_samples) * 100):.2f}%",
                    "\n".join(
                        [
                            "{}: {:.2f}% / {:.2f}%".format(
                                self.class_names[idx] if self.class_names and idx < len(self.class_names) else idx,
                                (count_left / self.first_dt_samples_by_class[idx]) * 100,
                                (count_right / self.first_dt_samples_by_class[idx]) * 100,
                            )
                            for idx, (count_left, count_right) in enumerate(node["data_split_by_class"])
                        ]
                    ),
                ]
            )
            top_nodes.add_row(["", "", "", ""])

        top_branches = PrettyTable(
            title=f"Top {len(self.first_dt_top_branches)} Branches",
            field_names=["Rule", "Decision (P(x))", "Samples (%) - ↓", "Class Samples (%)"],
        )
        top_branches.align = "l"
        top_branches.valign = "m"

        sum_samples = 0
        sum_samples_perc = 0
        sum_class_samples_perc = {}
        for branch in self.first_dt_top_branches:
            samples, samples_perc, class_samples_perc = (
                branch["samples"],
                (branch["samples"] / self.first_dt_samples) * 100,
                (branch["samples"] / self.first_dt_samples_by_class[branch["class"]]) * 100,
            )
            sum_samples += samples
            sum_samples_perc += samples_perc

            if branch["class"] not in sum_class_samples_perc:
                sum_class_samples_perc[branch["class"]] = 0
            sum_class_samples_perc[branch["class"]] += class_samples_perc

            top_branches.add_row(
                [
                    "\n and ".join(
                        [
                            "{} {} {}".format(
                                self.feature_names[feat] if self.feature_names else feat,
                                op,
                                threshold,
                            )
                            for (feat, op, threshold) in branch["path"]
                        ]
                    ),
                    "{}\n({:.2f}%)".format(
                        self.class_names[branch["class"]]
                        if self.class_names and branch["class"] < len(self.class_names)
                        else branch["class"],
                        branch["prob"],
                    ),
                    f"{samples}\n({samples_perc:.2f}%)",
                    f"{class_samples_perc:.2f}%",
                ]
            )
            top_branches.add_row(["", "", "", ""])

        top_branches.add_row(["-" * 10, "-" * 10, "-" * 10, "-" * 10])
        top_branches.add_row(
            [
                f"Top {len(self.first_dt_top_branches)} Summary",
                "-",
                f"{sum_samples} ({sum_samples_perc:.2f}%)",
                "\n".join(
                    [
                        "{}: {:.2f}%".format(
                            self.class_names[class_idx]
                            if self.class_names and class_idx < len(self.class_names)
                            else class_idx,
                            class_perc,
                        )
                        for (class_idx, class_perc) in sum_class_samples_perc.items()
                    ]
                ),
            ]
        )

        single_analysis_first_row.add_column("Top Nodes", [top_nodes])
        single_analysis_first_row.add_column("Top Branches", [top_branches])

        single_analysis.add_column("Single Analysis", [top_features, single_analysis_first_row])

        prunning_analysis = PrettyTable(title="Prunning Analysis", header=False)
        alpha_performance = PrettyTable(
            title="CCP Alpha Iteration",
            field_names=[
                "Alpha",
                "Gini",
                "DT Size",
                "DT Depth",
                "DT Num Leaves",
                "Similarity",
                "Performance",
                "Fidelity",
            ],
        )
        alpha_performance.align = "l"
        alpha_performance.valign = "m"

        for i in self.ccp_iter:
            joined_similarity = "\n".join([f"  {sim:.3f}" for sim in i["similarity_vec"]])
            alpha_performance.add_row(
                [
                    i["ccp_alpha"],
                    f"{i['gini']:.3f}",
                    i["dt"].tree_.node_count,
                    i["dt"].get_depth(),
                    i["dt"].get_n_leaves(),
                    f"{i['similarity']:.3f}\n\nVector:[\n{joined_similarity}\n]",
                    i["classification_report"],
                    i["fidelity_report"],
                ]
            )
            alpha_performance.add_row(["", "", "", "", "", "", "", ""])

        max_depth_performance = PrettyTable(
            title="Max Depth Iteration",
            field_names=[
                "Max Depth",
                "DT Size",
                "DT Depth",
                "DT Num Leaves",
                "Similarity",
                "Performance",
                "Fidelity",
            ],
        )
        max_depth_performance.align = "l"
        max_depth_performance.valign = "m"

        for i in self.max_depth_iter:
            joined_similarity = "\n".join([f"  {sim:.3f}" for sim in i["similarity_vec"]])
            max_depth_performance.add_row(
                [
                    i["max_depth"],
                    i["dt"].tree_.node_count,
                    i["dt"].get_depth(),
                    i["dt"].get_n_leaves(),
                    f"{i['similarity']:.3f}\n\nVector:[\n{joined_similarity}\n]",
                    i["classification_report"],
                    i["fidelity_report"],
                ]
            )
            max_depth_performance.add_row(["", "", "", "", "", "", ""])

        max_leaves_performance = PrettyTable(
            title="Max Leaves Iteration",
            field_names=[
                "Max Leaves",
                "DT Size",
                "DT Depth",
                "DT Num Leaves",
                "Similarity",
                "Performance",
                "Fidelity",
            ],
        )
        max_leaves_performance.align = "l"
        max_leaves_performance.valign = "m"

        for i in self.max_leaves_iter:
            joined_similarity = "\n".join([f"  {sim:.3f}" for sim in i["similarity_vec"]])
            max_leaves_performance.add_row(
                [
                    i["max_leaves"],
                    i["dt"].tree_.node_count,
                    i["dt"].get_depth(),
                    i["dt"].get_n_leaves(),
                    f"{i['similarity']:.3f}\n\nVector:[\n{joined_similarity}\n]",
                    i["classification_report"],
                    i["fidelity_report"],
                ]
            )
            max_leaves_performance.add_row(["", "", "", "", "", "", ""])

        prunning_analysis.add_column(
            "Prunning Iteration", [alpha_performance, max_depth_performance, max_leaves_performance]
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
                    "CCP Alpha",
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
                        i["classification_report"],
                        i["dt"].tree_.node_count,
                        self.dagger_ccp_alpha,
                        i["fidelity_report"],
                    ]
                )
                iter_performance.add_row(["", "", "", "", "", "", ""])

            repeated_analysis.add_column("Iterative Feature Removal", [iter_performance])

        report.add_column(
            "Report",
            [summary, single_analysis, prunning_analysis] + ([repeated_analysis] if not self.skip_retrain else []),
        )

        return f"\n{report}"

    def _prepare_data(self):
        """Data preparation for trust report"""
        log = self.logger.log if self.logger else print

        if (self.X is None and (self.X_train is None or self.X_test is None)) or (
            self.y is None and (self.y_train is None or self.y_test is None)
        ):
            raise ValueError("Missing either X and y arguments or X_train, X_test, y_train and y_test arguments.")

        if self.X_train is None:
            # if data split is not given as a param, split the dataset randomly
            log("Splitting dataset for training and testing...")
            if isinstance(self.X, pd.DataFrame):
                X_indexes = np.arange(0, self.X.shape[0])
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                    X_indexes, self.y, train_size=self.train_size
                )
                self.X_train = self.X.iloc[self.X_train]
                self.X_test = self.X.iloc[self.X_test]
            else:
                X_indexes = np.arange(0, len(self.X))
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                    X_indexes, self.y, train_size=self.train_size
                )
                self.X_train = np.array(self.X)[self.X_train]
                self.X_test = np.array(self.X)[self.X_test]

            log(f"X size: {len(self.X)}; y size: {len(self.y)}")
            log("Done!")

        self.dataset_size = len(self.X_train) + len(self.X_test)
        self.train_size = len(self.X_train) / self.dataset_size

        if self.feature_names is not None:
            self.feature_names = list(self.feature_names)
            if isinstance(self.X_train, pd.DataFrame):
                self.feature_names = list(self.X_train.columns)

    def _fit_and_explain(
        self, X_train=None, X_test=None, dagger_ccp_alpha=0.0, dagger_max_leaf_nodes=None, dagger_max_depth=None
    ):
        """
        Fits blacbox with the given X and y data, and uses Dagger to extract DT explanation
        """
        log = self.logger.log if self.logger else print

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
                log("warning", warn1)
                try:
                    # pytorch models
                    blackbox_copy = copy.copy(self.blackbox)
                    if hasattr(blackbox_copy, "children"):
                        for layer in blackbox_copy.children():
                            if hasattr(layer, "reset_parameters"):
                                layer.reset_parameters()
                    else:
                        raise Exception("Not a pytorch model")
                except Exception as warn2:
                    log("warning", warn2)
                    # AutoGluon and any other models
                    blackbox_copy = (
                        self.blackbox.__class__(self.blackbox._learner.label)
                        if hasattr(self.blackbox, "_learner")
                        else self.blackbox.__class__()
                    )

            if isinstance(self.blackbox, TabularPredictor):
                args = {}
                args[self.blackbox._learner.label] = self.y_train.values
                training_data = X_train.assign(**args)
                blackbox_copy.fit(training_data)
            else:
                blackbox_copy.fit(X_train, self.y_train)

            del self.blackbox

        y_pred = getattr(blackbox_copy, self.predict_method_name)(X_test)

        log("Blackbox model classification report with training data:")
        log(f"\n{classification_report(self.y_test, y_pred, digits=3)}")

        # Decision tree extraction
        log("Using Classification Dagger algorithm to extract DT...")
        dagger = ClassificationDagger(expert=blackbox_copy)

        dagger.fit(
            X_train,
            self.y_train,
            num_iter=self.dagger_num_iter,
            samples_size=self.dagger_sample_size,
            predict_method_name=self.predict_method_name,
            max_leaf_nodes=dagger_max_leaf_nodes if dagger_max_leaf_nodes else self.dagger_max_leaf_nodes,
            max_depth=dagger_max_depth if dagger_max_depth else self.dagger_max_depth,
            ccp_alpha=dagger_ccp_alpha if dagger_ccp_alpha else self.dagger_ccp_alpha,
            verbose=self.verbose,
        )

        log("#" * 10, "Explanation validation", "#" * 10)
        (dt, reward, idx) = dagger.explain()
        log(f"Model explanation {idx} training fidelity: {reward}")
        dt_y_pred = dt.predict(X_test)

        log("Model explanation global fidelity report:")
        log(
            "\n{}".format(
                classification_report(
                    y_pred,
                    dt_y_pred,
                    digits=3,
                    # labels=range(len(self.class_names)) if self.class_names else None,
                    # target_names=self.class_names,
                )
            )
        )

        log("Model explanation classification report:")
        log(
            "\n{}".format(
                classification_report(
                    self.y_test,
                    dt_y_pred,
                    digits=3,
                    # labels=range(len(self.class_names)) if self.class_names else None,
                    # target_names=self.class_names,
                )
            )
        )
        log("#" * 10, "Done", "#" * 10)

        self.blackbox = blackbox_copy

        return dt, y_pred, dt_y_pred

    def _collect(self):
        """Collects data to build the make report"""
        self._collect_first_dt()
        self._collect_ccp_prunning()
        self._collect_max_depth_prunning()
        self._collect_max_leaves_prunning()

        if not self.skip_retrain:
            self._collect_features_iter_removal()

    def _collect_first_dt(self):
        """Uses provided dataset to train a Decision Tree and fetch first decision tree info"""
        self.first_dt, self.first_y_pred, self.first_dt_y_pred = self._fit_and_explain()

        self.bb_class = type(self.blackbox).__name__
        self.bb_n_input_features = (
            len(self.X_train.columns) if isinstance(self.X_train, pd.DataFrame) else len(self.X_train[0])
        )
        self.bb_n_output_classes = len(np.unique(self.y_train))

        self.first_dt_features, self.first_dt_nodes, self.first_dt_branches, self.first_dt_sum_samples = get_dt_info(
            self.first_dt
        )

        self.first_dt_class = type(self.first_dt).__name__
        self.first_dt_size = self.first_dt.tree_.node_count
        self.first_dt_n_leaves = self.first_dt.tree_.n_leaves
        self.first_dt_samples = self.first_dt.tree_.n_node_samples[0]
        self.first_dt_samples_by_class = self.first_dt.tree_.value[0][0]

        self.first_dt_samples_by_level = list(np.zeros(self.first_dt.get_depth() + 1))
        self.first_dt_nodes_by_level = list(np.zeros(self.first_dt.get_depth() + 1).astype(int))
        for node in self.first_dt_nodes:
            self.first_dt_samples_by_level[node["level"]] += node["samples"]
            # self.first_dt_nodes_by_level[node["level"]] += 1

        for node in self.first_dt_branches:
            self.first_dt_samples_by_level[node["level"]] += node["samples"]
            self.first_dt_nodes_by_level[node["level"]] += 1

        self.first_dt_top_features = sorted(
            self.first_dt_features.items(), key=lambda p: p[1]["samples"], reverse=True
        )[: self.top_n]
        self.first_dt_top_nodes = sorted(
            self.first_dt_nodes,
            key=lambda p: p["samples"] * abs(p["gini_split"][0] - p["gini_split"][1]),
            reverse=True,
        )[: self.top_n]
        self.first_dt_top_branches = sorted(self.first_dt_branches, key=lambda p: p["samples"], reverse=True)[
            : self.top_n
        ]
        self.first_dt_n_features = len(self.first_dt_features.keys())
        self.first_dt_n_classes = self.first_dt.tree_.n_classes[0]

    def _collect_ccp_prunning(self):
        """Uses provided dataset to train multiple Decision Trees and fetch CCP prunning info"""
        ccp_clf = tree.DecisionTreeClassifier(random_state=0)
        ccp_path = ccp_clf.cost_complexity_pruning_path(self.X_train, self.y_train)
        ccp_alphas, impurities = ccp_path.ccp_alphas, ccp_path.impurities

        self.ccp_iter = []
        for quantil in np.linspace(0, 1, self.num_quantiles, endpoint=False):
            ccp_alpha = np.quantile(ccp_alphas, quantil)
            if ccp_alpha >= 0:
                gini = np.quantile(impurities, quantil)

                ccp_dt, ccp_y_pred, ccp_dt_y_pred = self._fit_and_explain(dagger_ccp_alpha=ccp_alpha)
                ccp_dt_sim, ccp_dt_sim_vec = dt_similarity(ccp_dt, self.first_dt)

                self.ccp_iter.append(
                    {
                        "ccp_alpha": ccp_alpha,
                        "gini": gini,
                        "dt": ccp_dt,
                        "similarity": ccp_dt_sim,
                        "similarity_vec": ccp_dt_sim_vec,
                        "y_pred": ccp_y_pred,
                        "dt_y_pred": ccp_dt_y_pred,
                        "f1": f1_score(self.y_test, ccp_dt_y_pred, average="macro"),
                        "classification_report": classification_report(
                            self.y_test,
                            ccp_dt_y_pred,
                            digits=3,
                            # labels=range(len(self.class_names)) if self.class_names else None,
                            # target_names=self.class_names,
                        ),
                        "fidelity": f1_score(ccp_y_pred, ccp_dt_y_pred, average="macro"),
                        "fidelity_report": classification_report(
                            ccp_y_pred,
                            ccp_dt_y_pred,
                            digits=3,
                            # labels=range(len(self.class_names)) if self.class_names else None,
                            # target_names=self.class_names,
                        ),
                    }
                )

    def _collect_max_depth_prunning(self):
        """Uses provided dataset to train multiple Decision Trees and fetch max depth prunning info"""
        self.max_depth_iter = []
        for quantil in np.linspace(0, 1, self.num_quantiles, endpoint=False):
            max_depth = int(np.quantile(np.arange(1, self.first_dt.get_depth()), quantil))
            if max_depth > 0:
                max_depth_dt, max_depth_y_pred, max_depth_dt_y_pred = self._fit_and_explain(dagger_max_depth=max_depth)

                max_depth_dt_sim, max_depth_dt_sim_vec = dt_similarity(max_depth_dt, self.first_dt)
                self.max_depth_iter.append(
                    {
                        "max_depth": max_depth,
                        "dt": max_depth_dt,
                        "similarity": max_depth_dt_sim,
                        "similarity_vec": max_depth_dt_sim_vec,
                        "y_pred": max_depth_y_pred,
                        "dt_y_pred": max_depth_dt_y_pred,
                        "f1": f1_score(self.y_test, max_depth_dt_y_pred, average="macro"),
                        "classification_report": classification_report(
                            self.y_test,
                            max_depth_dt_y_pred,
                            digits=3,
                            # labels=range(len(self.class_names)) if self.class_names else None,
                            # target_names=self.class_names,
                        ),
                        "fidelity": f1_score(max_depth_y_pred, max_depth_dt_y_pred, average="macro"),
                        "fidelity_report": classification_report(
                            max_depth_y_pred,
                            max_depth_dt_y_pred,
                            digits=3,
                            # labels=range(len(self.class_names)) if self.class_names else None,
                            # target_names=self.class_names,
                        ),
                    }
                )

    def _collect_max_leaves_prunning(self):
        """Uses provided dataset to train multiple Decision Trees and fetch max leaves prunning info"""
        self.max_leaves_iter = []
        for quantil in np.linspace(0, 1, self.num_quantiles, endpoint=False):
            max_leaves = int(np.quantile(np.arange(2, self.first_dt.get_n_leaves()), quantil))
            if max_leaves > 1:
                max_leaves_dt, max_leaves_y_pred, max_leaves_dt_y_pred = self._fit_and_explain(
                    dagger_max_leaf_nodes=max_leaves
                )

                max_leaves_dt_sim, max_leaves_dt_sim_vec = dt_similarity(max_leaves_dt, self.first_dt)
                self.max_leaves_iter.append(
                    {
                        "max_leaves": max_leaves,
                        "dt": max_leaves_dt,
                        "similarity": max_leaves_dt_sim,
                        "similarity_vec": max_leaves_dt_sim_vec,
                        "y_pred": max_leaves_y_pred,
                        "dt_y_pred": max_leaves_dt_y_pred,
                        "f1": f1_score(self.y_test, max_leaves_dt_y_pred, average="macro"),
                        "classification_report": classification_report(
                            self.y_test,
                            max_leaves_dt_y_pred,
                            digits=3,
                            # labels=range(len(self.class_names)) if self.class_names else None,
                            # target_names=self.class_names,
                        ),
                        "fidelity": f1_score(max_leaves_y_pred, max_leaves_dt_y_pred, average="macro"),
                        "fidelity_report": classification_report(
                            max_leaves_y_pred,
                            max_leaves_dt_y_pred,
                            digits=3,
                            # labels=range(len(self.class_names)) if self.class_names else None,
                            # target_names=self.class_names,
                        ),
                    }
                )

    def _collect_features_iter_removal(self):
        """Uses provided dataset to train multiple Decision Trees by iteratively removing the most important features from the dataset"""

        self.whitebox_iter = []

        i = 0
        n_features_removed = 0
        top_feature_to_remove = self.first_dt_top_features[0][0]
        if isinstance(self.X_train, pd.DataFrame):
            X_train_iter = self.X_train.copy()
            X_test_iter = self.X_test.copy()
        elif isinstance(self.X_train, torch.Tensor):
            X_train_iter = self.X_train.clone()
            X_test_iter = self.X_test.clone()
        else:
            X_train_iter = np.copy(self.X_train)
            X_test_iter = np.copy(self.X_test)

        while i < self.max_iter and n_features_removed < self.bb_n_input_features:
            # remove most significant feature
            if isinstance(self.X_train, pd.DataFrame):
                X_train_iter.iloc[:, top_feature_to_remove] = 0
                X_test_iter.iloc[:, top_feature_to_remove] = 0
            elif isinstance(self.X_train, torch.Tensor):
                X_train_iter[:, top_feature_to_remove] = torch.zeros(len(self.X_train))
                X_test_iter[:, top_feature_to_remove] = torch.zeros(len(self.X_test))
            else:
                X_train_iter[:, top_feature_to_remove] = np.zeros(len(self.X_train))
                X_test_iter[:, top_feature_to_remove] = np.zeros(len(self.X_test))

            n_features_removed += 1

            dt, y_pred, dt_y_pred = self._fit_and_explain(X_train=X_train_iter, X_test=X_test_iter)

            self.whitebox_iter.append(
                {
                    "it": i,
                    "dt": dt,
                    "y_pred": y_pred,
                    "dt_y_pred": dt_y_pred,
                    "feature_removed": top_feature_to_remove,
                    "n_features_removed": n_features_removed,
                    "f1": f1_score(self.y_test, y_pred, average="macro"),
                    "classification_report": classification_report(
                        self.y_test,
                        y_pred,
                        digits=3,
                        # labels=range(len(self.class_names)) if self.class_names else None,
                        # target_names=self.class_names,
                    ),
                    "fidelity": f1_score(y_pred, dt_y_pred, average="macro"),
                    "fidelity_report": classification_report(
                        y_pred,
                        dt_y_pred,
                        digits=3,
                        # labels=range(len(self.class_names)) if self.class_names else None,
                        # target_names=self.class_names,
                    ),
                }
            )

            iter_dt_features, _, _, _ = get_dt_info(dt)
            iter_dt_top_features = sorted(iter_dt_features.items(), key=lambda p: p[1]["samples"], reverse=True)[
                : self.top_n
            ]
            top_feature_to_remove = iter_dt_top_features[0][0]
            i += 1

    def _save_dts(self, output_dir):
        """
        Save the decision trees.

        Parameters
        ----------
        output_dir : str
            The output directory to save the decision trees.
        """
        dot_data = tree.export_graphviz(
            self.first_dt,
            class_names=self.class_names,
            feature_names=self.feature_names,
            filled=True,
            rounded=True,
            special_characters=True,
        )
        graph = graphviz.Source(dot_data)
        graph.render(f"{output_dir}/trust_report_dt")

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

    def plot(self, output_dir):
        """
        Plot the analysis results.

        Parameters
        ----------
        output_dir : str
            The output directory to save the plots.
        """
        plots_output_dir = f"{output_dir}/plots"
        if not os.path.exists(plots_output_dir):
            os.makedirs(plots_output_dir)

        plot_top_features(
            self.first_dt_top_features,
            self.first_dt_sum_samples,
            (self.first_dt_size - self.first_dt_n_leaves),
            plots_output_dir,
            feature_names=self.feature_names,
        )
        plot_top_nodes(
            self.first_dt_top_nodes,
            self.first_dt_samples_by_class,
            self.first_dt_samples,
            plots_output_dir,
            feature_names=self.feature_names,
            class_names=self.class_names,
        )
        plot_top_branches(
            self.first_dt_top_branches,
            self.first_dt_samples_by_class,
            self.first_dt_samples,
            plots_output_dir,
            class_names=self.class_names,
        )
        plot_samples_by_level(
            self.first_dt_samples_by_level,
            self.first_dt_nodes_by_level,
            self.first_dt_samples,
            plots_output_dir,
        )
        plot_dts_fidelity_by_size(
            self.ccp_iter,
            self.max_depth_iter,
            self.max_leaves_iter,
            plots_output_dir,
        )

        plot_distribution(
            self.X if self.X is not None else self.X_train,
            self.y if self.y is not None else self.y_train,
            self.first_dt_top_branches,
            plots_output_dir,
            feature_names=self.feature_names,
            class_names=self.class_names,
        )

        if not self.skip_retrain:
            plot_accuracy_by_feature_removed(
                self.whitebox_iter,
                plots_output_dir,
                feature_names=self.feature_names,
            )

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

    def save(self, output_dir):
        """Saves report and plots to output dir"""
        ################################################
        #                    OUTPUT                    #
        ################################################
        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            report_output_dir = f"{output_dir}/report"
            if not os.path.exists(report_output_dir):
                os.makedirs(report_output_dir)

            with open(f"{report_output_dir}/trust_report.txt", "w", encoding="utf-8") as file:
                file.write(f"\n{str(self)}")

            with open(f"{report_output_dir}/trust_report.obj", "wb") as file:
                pickle.dump(self, file)

            self._save_dts(report_output_dir)
            self.plot(report_output_dir)
