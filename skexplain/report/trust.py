import os
import graphviz
import copy
import torch
import numpy as np
import pandas as pd
from scipy import spatial


from sklearn import tree
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.base import clone


from autogluon.tabular import TabularPredictor


from prettytable import PrettyTable


from skexplain.imitation import ClassificationDagger
from skexplain.utils import plot


def plot_top_features(top_features, dt_sum_samples, dt_nodes, output_dir, feature_names=[]):
    """Uses top features informations and plots CDF with it"""
    features = [feature_names[feat] if feature_names else feat for (feat, _) in top_features]
    count = [(values["count"] / dt_nodes) * 100 for (_, values) in top_features]
    count_sum = np.cumsum(count)
    data = [(values["samples"] / dt_sum_samples) * 100 for (_, values) in top_features]
    data_sum = np.cumsum(data)

    plot.plot_lines(
        features,
        [count_sum, data_sum],
        y_lim=(0, 100),
        labels=["Nodes", "Samples"],
        path=f"{output_dir}/top_features_lines.pdf",
    )
    plot.plot_bars(
        features,
        [count_sum, data_sum],
        y_lim=(0, 100),
        labels=["Nodes", "Samples"],
        path=f"{output_dir}/top_features_bars.pdf",
    )

    plot.plot_lines_and_bars(
        features,
        [count_sum, data_sum],
        [count, data],
        y_lim=(0, 100),
        labels=["Nodes", "Samples"],
        path=f"{output_dir}/top_features_lines_bars.pdf",
    )


def plot_top_nodes(top_nodes, dt_samples_by_class, dt_samples, output_dir, feature_names=[], class_names=[]):
    """Uses top features informations and plots CDF with it"""
    plot.plot_stacked_bars_split(
        [
            "{} <= {:.2f}".format(
                feature_names[node["feature"]] if feature_names else node["feature"],
                node["threshold"],
            )
            for node in top_nodes
        ],
        [[(node["data_split"][0] / dt_samples) * 100 for node in top_nodes]],
        [[(node["data_split"][1] / dt_samples) * 100 for node in top_nodes]],
        y_placeholder=[100],
        y_lim=(0, 100),
        path=f"{output_dir}/top_nodes.pdf",
    )

    plot.plot_stacked_bars_split(
        [
            "{} <= {:.2f}".format(
                feature_names[node["feature"]] if feature_names else node["feature"],
                node["threshold"],
            )
            for node in top_nodes
        ],
        [
            [(node["data_split_by_class"][idx][0] / dt_samples) * 100 for node in top_nodes]
            for idx in range(len(dt_samples_by_class))
        ],
        [
            [(node["data_split_by_class"][idx][1] / dt_samples) * 100 for node in top_nodes]
            for idx in range(len(dt_samples_by_class))
        ],
        y_placeholder=[(samples / dt_samples) * 100 for samples in dt_samples_by_class],
        y_lim=(0, 100),
        labels=class_names,
        path=f"{output_dir}/top_nodes_by_class.pdf",
    )


def plot_top_branches(top_branches, dt_samples_by_class, dt_samples, output_dir, class_names=[]):
    """Uses top features informations and plots CDF with it"""
    colors = [
        "#d75d5b",
        "#524a47",
        "#8a4444",
        "#edeef0",
        "#c8c5c3",
        "#f5f0ed",
        "#a7c3cd",
    ]

    samples = []
    colors_by_class = {}
    colors_by_samples = []
    for branch in top_branches:
        class_label = class_names[branch["class"]] if class_names else branch["class"]
        if class_label not in colors_by_class:
            colors_by_class[class_label] = colors.pop()
        samples.append((branch["samples"] / dt_samples) * 100)
        colors_by_samples.append(colors_by_class[class_label])

    plot.plot_stacked_bars(
        [f"Top {idx + 1}" for idx in range(len(top_branches))],
        [np.cumsum(samples)],
        y_placeholder=[100],
        y_lim=(0, 100),
        path=f"{output_dir}/top_branches.pdf",
    )

    plot.plot_stacked_bars(
        [f"Top {idx + 1}" for idx in range(len(top_branches))],
        [
            np.cumsum(
                [((branch["samples"] / dt_samples) * 100) if idx == branch["class"] else 0 for branch in top_branches]
            )
            for idx, _ in enumerate(dt_samples_by_class)
        ],
        y_placeholder=[(samples / dt_samples) * 100 for samples in dt_samples_by_class],
        y_lim=(0, 100),
        labels=class_names,
        path=f"{output_dir}/cum_top_branches_by_class.pdf",
    )

    plot.plot_lines_and_bars(
        [f"Top {idx + 1}" for idx in range(len(top_branches))],
        [np.cumsum(samples)],
        [samples],
        y_lim=(0, 100),
        legend={"CDF": "#d75d5b", **colors_by_class},
        colors_by_x=colors_by_samples,
        path=f"{output_dir}/top_branches_by_class.pdf",
    )


def plot_samples_by_level(dt_samples_by_level, dt_samples, output_dir):
    """Uses top features informations and plots CDF with it"""
    samples = []
    for idx, level_samples in enumerate(dt_samples_by_level):
        if idx < len(dt_samples_by_level) - 1:
            samples.append(((level_samples - dt_samples_by_level[idx + 1]) / dt_samples) * 100)
        else:
            samples.append((level_samples / dt_samples) * 100)

    plot.plot_lines_and_bars(
        [f"Level {level}" for level, _ in enumerate(dt_samples_by_level)],
        [np.cumsum(samples)],
        [samples],
        y_lim=(0, 100),
        labels=["Samples"],
        path=f"{output_dir}/samples_by_level.pdf",
    )

    # plot.plot_stacked_bars(
    #     ["Top {}".format(idx + 1) for idx in range(len(top_branches))],
    #     [
    #         np.cumsum(
    #             [((branch["samples"] / dt_samples) * 100) if idx == branch["class"] else 0 for branch in top_branches]
    #         )
    #         for idx, _ in enumerate(dt_samples_by_class)
    #     ],
    #     y_placeholder=[(samples / dt_samples) * 100 for samples in dt_samples_by_class],
    #     y_lim=(0, 100),
    #     labels=class_names,
    #     path="{}/top_branches_by_class.pdf".format(output_dir),
    # )


def dt_similarity(dt_one, dt_two):
    """Compare decision tree elements and compute the similarity between them"""
    dt_one_matrix = np.array(
        [
            dt_one.tree_.children_left,
            dt_one.tree_.children_right,
            dt_one.tree_.feature,
            dt_one.tree_.threshold,
            [np.argmax(node[0]) for node in dt_one.tree_.value],
        ]
    )

    dt_two_matrix = np.array(
        [
            dt_two.tree_.children_left,
            dt_two.tree_.children_right,
            dt_two.tree_.feature,
            dt_two.tree_.threshold,
            [np.argmax(node[0]) for node in dt_two.tree_.value],
        ]
    )

    diff_tree_size = abs(len(dt_one.tree_.feature) - len(dt_two.tree_.feature))
    if len(dt_one.tree_.feature) > len(dt_two.tree_.feature):
        dt_two_matrix = np.pad(dt_two_matrix, [(0, 0), (0, diff_tree_size)], mode="constant")
    else:
        dt_one_matrix = np.pad(dt_one_matrix, [(0, 0), (0, diff_tree_size)], mode="constant")

    similarity_vector = [1 - spatial.distance.cosine(x, y) for x, y in zip(dt_one_matrix, dt_two_matrix)]
    similarity = np.mean(similarity_vector)

    return similarity, similarity_vector


def get_dt_info(dt):
    """Iterates through the given Decision Tree to collect relevant information for the trust report."""
    children_left = dt.tree_.children_left
    children_right = dt.tree_.children_right
    features = dt.tree_.feature
    thresholds = dt.tree_.threshold
    values = dt.tree_.value
    samples = dt.tree_.n_node_samples
    impurity = dt.tree_.impurity

    splits = []
    features_used = {}
    # sum of all samples in all non-leaf nodes
    samples_sum = np.sum(
        [node_sample if children_left[node] != children_right[node] else 0 for node, node_sample in enumerate(samples)]
    )

    def walk_tree(node, level, path):
        """Recursively iterates through all nodes in given decision tree and returns them as a list."""
        if children_left[node] == children_right[node]:  # if leaf node
            node_class = np.argmax(values[node][0])
            return [
                {
                    "level": level,
                    "path": path,
                    "class": node_class,
                    "prob": (values[node][0][node_class] / np.sum(values[node][0])) * 100,
                    "samples": samples[node],
                }
            ]

        feature = features[node]
        threshold = thresholds[node]
        left = children_left[node]
        right = children_right[node]

        if feature not in features_used:
            features_used[feature] = {"count": 0, "samples": 0}

        features_used[feature]["count"] += 1
        features_used[feature]["samples"] += samples[node]

        splits.append(
            {
                "idx": node,
                "level": level,
                "feature": feature,
                "threshold": threshold,
                "samples": samples[node],
                "values": values[node],
                "gini_split": (impurity[left], impurity[right]),
                "data_split": (np.sum(values[left]), np.sum(values[right])),
                "data_split_by_class": [
                    (c_left, c_right) for (c_left, c_right) in zip(values[left][0], values[right][0])
                ],
            }
        )

        return walk_tree(left, level + 1, path + [(feature, "<=", threshold)]) + walk_tree(
            right, level + 1, path + [(feature, ">", threshold)]
        )

    branches = walk_tree(0, 0, [])
    return (features_used, splits, branches, samples_sum)


def fit_and_explain(
    blackbox,
    X_train,
    y_train,
    X_test,
    y_test,
    predict_method_name="predict",
    dagger_num_iter=100,
    dagger_sample_size=0.5,
    dagger_max_leaf_nodes=None,
    dagger_max_depth=None,
    dagger_ccp_alpha=0.0,
    skip_retrain=False,
    class_names=None,
    logger=None,
    verbose=False,
):
    """
    Fits blacbox with the given X and y data, and uses Dagger to extract DT explanation
    """
    log = logger.log if logger else print

    if skip_retrain:
        blackbox_copy = blackbox
    else:
        # clone blackbox params but resets training weights to allow retraining with new dataset
        try:
            # scikit-learn models
            blackbox_copy = clone(blackbox)
        except Exception:
            # pytorch models
            blackbox_copy = copy.copy(blackbox)
            try:
                for layer in blackbox_copy.children():
                    if hasattr(layer, "reset_parameters"):
                        layer.reset_parameters()
            except Exception as warn:
                log("warning", warn)
                # AutoGluon and any other models
                blackbox_copy = (
                    blackbox.__class__(blackbox._learner.label)
                    if hasattr(blackbox, "_learner")
                    else blackbox.__class__()
                )

        if isinstance(blackbox, TabularPredictor):
            args = {}
            args[blackbox._learner.label] = y_train.values
            training_data = X_train.assign(**args)
            blackbox_copy.fit(training_data)
        else:
            blackbox_copy.fit(X_train, y_train)

    y_pred = getattr(blackbox_copy, predict_method_name)(X_test)

    log("Blackbox model classification report with training data:")
    log(f"\n{classification_report(y_test, y_pred, digits=3)}")

    # Decision tree extraction
    log("Using Classification Dagger algorithm to extract DT...")
    dagger = ClassificationDagger(expert=blackbox_copy)

    dagger.fit(
        X_train,
        y_train,
        num_iter=dagger_num_iter,
        samples_size=dagger_sample_size,
        max_leaf_nodes=dagger_max_leaf_nodes,
        predict_method_name=predict_method_name,
        max_depth=dagger_max_depth,
        ccp_alpha=dagger_ccp_alpha,
        verbose=verbose,
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
                labels=range(len(class_names)) if class_names else None,
                target_names=class_names,
            )
        )
    )

    log("Model explanation classification report:")
    log(
        "\n{}".format(
            classification_report(
                y_test,
                dt_y_pred,
                digits=3,
                labels=range(len(class_names)) if class_names else None,
                target_names=class_names,
            )
        )
    )
    log("#" * 10, "Done", "#" * 10)

    return (dt, y_pred, dt_y_pred)


def prepare_data(
    X=None,
    y=None,
    X_train=None,
    X_test=None,
    y_train=None,
    y_test=None,
    train_size=0.7,
    feature_names=None,
    logger=None,
):
    """Data preparation for trust report"""
    log = logger.log if logger else print

    if (X is None and (X_train is None or X_test is None)) or (y is None and (y_train is None or y_test is None)):
        raise ValueError("Missing either X and y arguments or X_train, X_test, y_train and y_test arguments.")

    if X_train is None:
        # if data split is not given as a param, split the dataset randomly
        log("Splitting dataset for training and testing...")
        if isinstance(X, pd.DataFrame):
            X_indexes = np.arange(0, X.shape[0])
            X_train, X_test, y_train, y_test = train_test_split(X_indexes, y, train_size=train_size)
            X_train = X.iloc[X_train]
            X_test = X.iloc[X_test]
        else:
            X_indexes = np.arange(0, len(X))
            X_train, X_test, y_train, y_test = train_test_split(X_indexes, y, train_size=train_size)
            X_train = np.array(X)[X_train]
            X_test = np.array(X)[X_test]
        log(f"X size: {len(X)}; y size: {len(y)}")
        log("Done!")

    dataset_size = len(X_train) + len(X_test)
    train_size = len(X_train) / dataset_size

    if feature_names is not None:
        feature_names = list(feature_names)
        feature_names = list(X_train.columns) if isinstance(X_train, pd.DataFrame) else feature_names

    return X_train, X_test, y_train, y_test, dataset_size, train_size, feature_names


def make_report(
    bb_class,
    dataset_size,
    bb_n_input_features,
    bb_n_output_classes,
    train_size,
    first_dt_class,
    dagger_num_iter,
    dagger_sample_size,
    dagger_ccp_alpha,
    y_test,
    first_y_pred,
    first_dt_n_features,
    first_dt_y_pred,
    first_dt_size,
    first_dt_n_leaves,
    first_dt_sum_samples,
    first_dt_n_classes,
    first_dt_top_features,
    first_dt_top_nodes,
    first_dt_top_branches,
    first_dt_samples,
    first_dt_samples_by_class,
    ccp_iter,
    whitebox_iter,
    skip_retrain,
    class_names,
    feature_names,
):
    """Formats collected data into a reporto using PrettyTable"""
    ################################################
    #                    REPORT                    #
    ################################################

    report = PrettyTable(title="Classification Trust Report", header=False)

    summary = PrettyTable(title="Summary")
    blackbox_report = PrettyTable(border=False, header=False)
    blackbox_report.align = "l"
    blackbox_report.add_row(["Model:", bb_class])
    blackbox_report.add_row(["Dataset size:", dataset_size])
    blackbox_report.add_row(
        [
            "Train/Test Split:",
            f"{train_size * 100:.2f}% / {(1 - train_size) * 100:.2f}%",
        ]
    )
    blackbox_report.add_row(["", ""])
    blackbox_report.add_row(["", ""])
    blackbox_report.add_row(["", ""])
    blackbox_report.add_row(["", ""])
    blackbox_report.add_row(["# Input features:", bb_n_input_features])
    blackbox_report.add_row(["# Output classes:", bb_n_output_classes])
    blackbox_report.add_row(["", ""])

    performance_report = PrettyTable(title="Performance", header=False)
    performance_report.add_column(
        "Performance",
        [
            classification_report(
                y_test,
                first_y_pred,
                digits=3,
                labels=range(len(class_names)) if class_names else None,
                target_names=class_names,
            )
        ],
    )

    summary.add_column("Blackbox", [blackbox_report, performance_report])

    whitebox_report = PrettyTable(border=False, header=False)
    whitebox_report.align = "l"
    whitebox_report.add_row(["Explanation method:", "Dagger"])
    whitebox_report.add_row(["Model:", first_dt_class])
    whitebox_report.add_row(["Iterations:", dagger_num_iter])
    whitebox_report.add_row(["Sample size:", f"{dagger_sample_size * 100:.2f}%"])
    whitebox_report.add_row(["", ""])
    whitebox_report.add_row(["Decision Tree Info", ""])
    whitebox_report.add_row(["Size:", first_dt_size])
    whitebox_report.add_row(["CCP Alpha:", dagger_ccp_alpha])
    whitebox_report.add_row(
        [
            "# Input features:",
            f"{first_dt_n_features} ({first_dt_n_features / bb_n_input_features * 100:.2f}%)",
        ]
    )
    whitebox_report.add_row(
        [
            "# Output classes:",
            f"{first_dt_n_classes} ({first_dt_n_classes / bb_n_output_classes * 100:.2f}%)",
        ]
    )
    whitebox_report.add_row(["", ""])

    fidelity_report = PrettyTable(title="Fidelity", header=False)
    fidelity_report.add_column(
        "Fidelity",
        [
            classification_report(
                first_y_pred,
                first_dt_y_pred,
                digits=3,
                labels=range(len(class_names)) if class_names else None,
                target_names=class_names,
            )
        ],
    )
    summary.add_column("Whitebox", [whitebox_report, fidelity_report])

    single_analysis = PrettyTable(title="Single-run Analysis", header=False)
    single_analysis_first_row = PrettyTable(header=False, border=False)

    top_features = PrettyTable(
        title=f"Top {len(first_dt_top_features)} Features",
        field_names=["Feature", "# of Nodes (%)", "Data Split % - ↓"],
    )

    sum_nodes = 0
    sum_nodes_perc = 0
    sum_data_split = 0
    for (feat, values) in first_dt_top_features:
        node, node_perc, data_split = (
            values["count"],
            (values["count"] / (first_dt_size - first_dt_n_leaves)) * 100,
            (values["samples"] / first_dt_sum_samples) * 100,
        )
        sum_nodes += node
        sum_nodes_perc += node_perc
        sum_data_split += data_split

        top_features.add_row(
            [
                feature_names[feat] if feature_names else feat,
                f"{node} ({node_perc:.2f}%)",
                f"{values['samples']} ({data_split:.2f}%)",
            ]
        )
        top_features.add_row(["", "", ""])

    top_features.add_row(["-" * 10, "-" * 10, "-" * 10])
    top_features.add_row(
        [
            f"Top {len(first_dt_top_features)} Summary",
            f"{sum_nodes} ({sum_nodes_perc:.2f}%)",
            f"{sum_data_split:.2f}%",
        ]
    )

    top_nodes = PrettyTable(
        title=f"Top {len(first_dt_top_nodes)} Nodes",
        field_names=[
            "Decision",
            "Gini  Split - ↓",
            "Data Split % - ↓",
            "Data Split % by Class (L/R)",
        ],
    )
    top_nodes.align = "l"
    top_nodes.valign = "m"

    for node in first_dt_top_nodes:
        top_nodes.add_row(
            [
                # node["idx"],
                "{} <= {}".format(
                    feature_names[node["feature"]] if feature_names else node["feature"],
                    node["threshold"],
                ),
                f"Left: {node['gini_split'][0]:.2f} \nRight: {node['gini_split'][1]:.2f}",
                f"Left: {((node['data_split'][0] / first_dt_samples) * 100):.2f}% \nRight: {((node['data_split'][1] / first_dt_samples) * 100):.2f}%",
                "\n".join(
                    [
                        "{}: {:.2f}% / {:.2f}%".format(
                            class_names[idx] if class_names and idx < len(class_names) else idx,
                            (count_left / first_dt_samples_by_class[idx]) * 100,
                            (count_right / first_dt_samples_by_class[idx]) * 100,
                        )
                        for idx, (count_left, count_right) in enumerate(node["data_split_by_class"])
                    ]
                ),
            ]
        )
        top_nodes.add_row(["", "", "", ""])

    top_branches = PrettyTable(
        title=f"Top {len(first_dt_top_branches)} Branches",
        field_names=["Rule", "Decision (P(x))", "Samples (%) - ↓", "Class Samples (%)"],
    )
    top_branches.align = "l"
    top_branches.valign = "m"

    sum_samples = 0
    sum_samples_perc = 0
    sum_class_samples_perc = {}
    for branch in first_dt_top_branches:
        samples, samples_perc, class_samples_perc = (
            branch["samples"],
            (branch["samples"] / first_dt_samples) * 100,
            (branch["samples"] / first_dt_samples_by_class[branch["class"]]) * 100,
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
                            feature_names[feat] if feature_names else feat,
                            op,
                            threshold,
                        )
                        for (feat, op, threshold) in branch["path"]
                    ]
                ),
                "{}\n({:.2f}%)".format(
                    class_names[branch["class"]]
                    if class_names and branch["class"] < len(class_names)
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
            f"Top {len(first_dt_top_branches)} Summary",
            "-",
            f"{sum_samples} ({sum_samples_perc:.2f}%)",
            "\n".join(
                [
                    "{}: {:.2f}%".format(
                        class_names[class_idx] if class_names and class_idx < len(class_names) else class_idx,
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

    ccp_analysis = PrettyTable(title="CCP Analysis", header=False)
    alpha_performance = PrettyTable(
        title="CCP Alpha Iteration",
        field_names=[
            "Alpha",
            "Gini",
            "Decision Tree Size",
            "Decision Tree Depth",
            "Similarity",
            "Performance",
            "Fidelity",
        ],
    )
    alpha_performance.align = "l"
    alpha_performance.valign = "m"

    for it in ccp_iter:
        joined_similarity = "\n".join([f"  {sim:.3f}" for sim in it["similarity_vec"]])
        alpha_performance.add_row(
            [
                it["ccp_alpha"],
                f"{it['gini']:.3f}",
                it["dt"].tree_.node_count,
                it["dt"].get_depth(),
                f"{it['similarity']:.3f}\n\nVector:[\n{joined_similarity}\n]",
                it["classification_report"],
                it["fidelity_report"],
            ]
        )
        alpha_performance.add_row(["", "", "", "", "", "", ""])

    ccp_analysis.add_column("CCP Alpha Iteration", [alpha_performance])

    if not skip_retrain:
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

        for it in whitebox_iter:
            iter_performance.add_row(
                [
                    it["it"],
                    feature_names[it["feature_removed"]] if feature_names else it["feature_removed"],
                    it["n_features_removed"],
                    it["classification_report"],
                    it["dt"].tree_.node_count,
                    dagger_ccp_alpha,
                    it["fidelity_report"],
                ]
            )
            iter_performance.add_row(["", "", "", "", "", "", ""])

        repeated_analysis.add_column("Iterative Feature Removal", [iter_performance])

    report.add_column(
        "Report",
        [summary, single_analysis, ccp_analysis] + ([repeated_analysis] if not skip_retrain else []),
    )

    return report


def trust_report(
    blackbox,
    X=None,
    y=None,
    X_train=None,
    X_test=None,
    y_train=None,
    y_test=None,
    max_iter=10,
    train_size=0.7,
    predict_method_name="predict",
    dagger_num_iter=100,
    dagger_sample_size=0.5,
    dagger_max_leaf_nodes=None,
    dagger_max_depth=None,
    dagger_ccp_alpha=0.0,
    skip_retrain=False,
    top_n=10,
    output_dir=None,
    logger=None,
    verbose=False,
    class_names=None,
    feature_names=None,
):
    """
    Builds trust report for given black-box model using the Dagger method to extract white-box explanations as Decision Trees.
    """
    log = logger.log if logger else print

    X_train, X_test, y_train, y_test, dataset_size, train_size, feature_names = prepare_data(
        X=X,
        y=y,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        train_size=train_size,
        feature_names=feature_names,
    )

    ###############################################
    #               DATA COLLECTION               #
    ###############################################

    (first_dt, first_y_pred, first_dt_y_pred) = fit_and_explain(
        blackbox,
        X_train,
        y_train,
        X_test,
        y_test,
        dagger_num_iter=dagger_num_iter,
        dagger_sample_size=dagger_sample_size,
        dagger_max_leaf_nodes=dagger_max_leaf_nodes,
        dagger_max_depth=dagger_max_depth,
        dagger_ccp_alpha=dagger_ccp_alpha,
        predict_method_name=predict_method_name,
        class_names=class_names,
        skip_retrain=skip_retrain,
        verbose=verbose,
        logger=logger,
    )

    bb_class = type(blackbox).__name__
    bb_n_input_features = len(X_train.columns) if isinstance(X_train, pd.DataFrame) else len(X_train[0])
    bb_n_output_classes = len(np.unique(y_train))

    (
        first_dt_features,
        first_dt_nodes,
        first_dt_branches,
        first_dt_sum_samples,
    ) = get_dt_info(first_dt)

    first_dt_class = type(first_dt).__name__
    first_dt_size = first_dt.tree_.node_count
    first_dt_n_leaves = first_dt.tree_.n_leaves
    first_dt_samples = first_dt.tree_.n_node_samples[0]
    first_dt_samples_by_class = first_dt.tree_.value[0][0]

    first_dt_samples_by_level = np.zeros(first_dt.get_depth())
    for node in first_dt_nodes:
        first_dt_samples_by_level[node["level"]] += node["samples"]

    first_dt_top_features = sorted(first_dt_features.items(), key=lambda p: p[1]["samples"], reverse=True)[:top_n]
    first_dt_top_nodes = sorted(
        first_dt_nodes,
        key=lambda p: p["samples"] * abs(p["gini_split"][0] - p["gini_split"][1]),
        reverse=True,
    )[:top_n]
    first_dt_top_branches = sorted(first_dt_branches, key=lambda p: p["samples"], reverse=True)[:top_n]
    first_dt_n_features = len(first_dt_features.keys())
    first_dt_n_classes = first_dt.tree_.n_classes[0]

    ccp_clf = tree.DecisionTreeClassifier(random_state=0)
    ccp_path = ccp_clf.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = ccp_path.ccp_alphas, ccp_path.impurities

    ccp_iter = []
    for idx, quantil in enumerate(np.linspace(0, 1, 10, endpoint=False)):
        ccp_alpha = np.quantile(ccp_alphas, quantil)
        gini = np.quantile(impurities, quantil)

        (ccp_dt, ccp_y_pred, ccp_dt_y_pred) = fit_and_explain(
            blackbox,
            X_train,
            y_train,
            X_test,
            y_test,
            dagger_num_iter=dagger_num_iter,
            dagger_sample_size=dagger_sample_size,
            dagger_max_leaf_nodes=dagger_max_leaf_nodes,
            dagger_max_depth=dagger_max_depth,
            dagger_ccp_alpha=ccp_alpha,
            predict_method_name=predict_method_name,
            class_names=class_names,
            skip_retrain=skip_retrain,
            verbose=verbose,
            logger=logger,
        )

        ccp_dt_sim, ccp_dt_sim_vec = dt_similarity(ccp_dt, ccp_iter[idx - 1]["dt"] if idx > 0 else first_dt)
        ccp_iter.append(
            {
                "ccp_alpha": ccp_alpha,
                "gini": gini,
                "dt": ccp_dt,
                "similarity": ccp_dt_sim,
                "similarity_vec": ccp_dt_sim_vec,
                "y_pred": ccp_y_pred,
                "dt_y_pred": ccp_dt_y_pred,
                "f1": f1_score(y_test, ccp_y_pred, average="macro"),
                "classification_report": classification_report(
                    y_test,
                    ccp_y_pred,
                    digits=3,
                    labels=range(len(class_names)) if class_names else None,
                    target_names=class_names,
                ),
                "fidelity": f1_score(ccp_y_pred, ccp_dt_y_pred, average="macro"),
                "fidelity_report": classification_report(
                    ccp_y_pred,
                    ccp_dt_y_pred,
                    digits=3,
                    labels=range(len(class_names)) if class_names else None,
                    target_names=class_names,
                ),
            }
        )

    whitebox_iter = []
    if not skip_retrain:
        it = 0
        n_features_removed = 0
        top_feature_to_remove = first_dt_top_features[0][0]
        while it < max_iter and n_features_removed < bb_n_input_features:
            # remove most significant feature
            if isinstance(X_train, pd.DataFrame):
                X_train.iloc[:, top_feature_to_remove] = 0
                X_test.iloc[:, top_feature_to_remove] = 0
            elif isinstance(X_train, torch.Tensor):
                X_train[:, top_feature_to_remove] = torch.zeros(len(X_train))
                X_test[:, top_feature_to_remove] = torch.zeros(len(X_test))
            else:
                X_train[:, top_feature_to_remove] = np.zeros(len(X_train))
                X_test[:, top_feature_to_remove] = np.zeros(len(X_test))

            n_features_removed += 1

            (dt, y_pred, dt_y_pred) = fit_and_explain(
                blackbox,
                X_train,
                y_train,
                X_test,
                y_test,
                dagger_num_iter=dagger_num_iter,
                dagger_sample_size=dagger_sample_size,
                dagger_max_depth=dagger_max_depth,
                dagger_ccp_alpha=dagger_ccp_alpha,
                predict_method_name=predict_method_name,
                class_names=class_names,
                verbose=verbose,
                logger=logger,
            )

            whitebox_iter.append(
                {
                    "it": it,
                    "dt": dt,
                    "y_pred": y_pred,
                    "dt_y_pred": dt_y_pred,
                    "feature_removed": top_feature_to_remove,
                    "n_features_removed": n_features_removed,
                    "f1": f1_score(y_test, y_pred, average="macro"),
                    "classification_report": classification_report(
                        y_test,
                        y_pred,
                        digits=3,
                        labels=range(len(class_names)) if class_names else None,
                        target_names=class_names,
                    ),
                    "fidelity": f1_score(y_pred, dt_y_pred, average="macro"),
                    "fidelity_report": classification_report(
                        y_pred,
                        dt_y_pred,
                        digits=3,
                        labels=range(len(class_names)) if class_names else None,
                        target_names=class_names,
                    ),
                }
            )

            (iter_dt_features, _, _, _) = get_dt_info(dt)
            iter_dt_top_features = sorted(iter_dt_features.items(), key=lambda p: p[1]["samples"], reverse=True)[:top_n]
            top_feature_to_remove = iter_dt_top_features[0][0]
            it += 1

    ################################################
    #                    OUTPUT                    #
    ################################################

    report = make_report(
        bb_class,
        dataset_size,
        bb_n_input_features,
        bb_n_output_classes,
        train_size,
        first_dt_class,
        dagger_num_iter,
        dagger_sample_size,
        dagger_ccp_alpha,
        y_test,
        first_y_pred,
        first_dt_n_features,
        first_dt_y_pred,
        first_dt_size,
        first_dt_n_leaves,
        first_dt_sum_samples,
        first_dt_n_classes,
        first_dt_top_features,
        first_dt_top_nodes,
        first_dt_top_branches,
        first_dt_samples,
        first_dt_samples_by_class,
        ccp_iter,
        whitebox_iter,
        skip_retrain,
        class_names,
        feature_names,
    )

    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        report_output_dir = f"{output_dir}/report"
        if not os.path.exists(report_output_dir):
            os.makedirs(report_output_dir)

        # print report to file
        with open(f"{report_output_dir}/trust_report.txt", "w") as f:
            f.write(f"\n{report}")

        # save first decision tree
        dot_data = tree.export_graphviz(
            first_dt,
            class_names=class_names,
            feature_names=feature_names,
            filled=True,
            rounded=True,
            special_characters=True,
        )
        graph = graphviz.Source(dot_data)
        graph.render(f"{report_output_dir}/trust_report_dt")

        prunning_output_dir = f"{report_output_dir}/prunning"
        if not os.path.exists(prunning_output_dir):
            os.makedirs(prunning_output_dir)

        for idx, it in enumerate(ccp_iter):
            dot_data = tree.export_graphviz(
                it["dt"],
                class_names=class_names,
                feature_names=feature_names,
                filled=True,
                rounded=True,
                special_characters=True,
            )
            graph = graphviz.Source(dot_data)
            graph.render(f"{prunning_output_dir}/ccp_dt_{idx}_{it['dt'].tree_.node_count}")

        plots_output_dir = f"{report_output_dir}/plots"
        if not os.path.exists(plots_output_dir):
            os.makedirs(plots_output_dir)

        plot.plot_confusion_matrix(
            confusion_matrix(y_test, first_y_pred, normalize="all"),
            labels=class_names,
            path=f"{plots_output_dir}/confusion_matrix_accuracy.pdf",
        )
        plot.plot_confusion_matrix(
            confusion_matrix(first_dt_y_pred, first_y_pred, normalize="all"),
            labels=class_names,
            path=f"{plots_output_dir}/confusion_matrix_fidelity.pdf",
        )

        plot_top_features(
            first_dt_top_features,
            first_dt_sum_samples,
            (first_dt_size - first_dt_n_leaves),
            plots_output_dir,
            feature_names=feature_names,
        )
        plot_top_nodes(
            first_dt_top_nodes,
            first_dt_samples_by_class,
            first_dt_samples,
            plots_output_dir,
            feature_names=feature_names,
            class_names=class_names,
        )
        plot_top_branches(
            first_dt_top_branches,
            first_dt_samples_by_class,
            first_dt_samples,
            plots_output_dir,
            class_names=class_names,
        )
        plot_samples_by_level(
            first_dt_samples_by_level,
            first_dt_samples,
            plots_output_dir,
        )

        plot.plot_lines(
            range(len(ccp_alphas)),
            [ccp_alphas],
            path=f"{plots_output_dir}/ccp_alphas.pdf",
        )
        plot.plot_lines(
            ccp_alphas,
            [impurities],
            path=f"{plots_output_dir}/ccp_alpha_x_gini.pdf",
        )

    return f"\n{report}"
