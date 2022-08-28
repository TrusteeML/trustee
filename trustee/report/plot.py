import os
import re
import math
import numbers
import numpy as np
import pandas as pd

from copy import deepcopy
from pandas.api.types import is_numeric_dtype

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import f1_score, r2_score

from trustee.utils import plot
from trustee.utils.tree import get_dt_info


def plot_top_features(top_features, dt_sum_samples, dt_nodes, output_dir, feature_names=[]):
    """Uses top features information and plots CDF with it"""
    if not np.array(top_features).size or not np.array(dt_sum_samples).size or not np.array(dt_nodes).size:
        return

    features = [feature_names[feat] if feature_names else str(feat) for (feat, _) in top_features]
    count = [(values["count"] / dt_nodes) * 100 for (_, values) in top_features]
    count_sum = np.cumsum(count)
    data = [(values["samples"] / dt_sum_samples) * 100 for (_, values) in top_features]
    data_sum = np.cumsum(data)

    plot.plot_lines(
        features,
        [count_sum, data_sum],
        ylim=(0, 100),
        xlabel="Feature",
        ylabel="% of total",
        labels=["Nodes", "Samples"],
        path=f"{output_dir}/top_features_lines.pdf",
    )
    plot.plot_bars(
        features,
        [count_sum, data_sum],
        ylim=(0, 100),
        xlabel="Feature",
        ylabel="% of total",
        labels=["Nodes", "Samples"],
        path=f"{output_dir}/top_features_bars.pdf",
    )

    plot.plot_lines_and_bars(
        features,
        [count_sum, data_sum],
        [count, data],
        ylim=(0, 100),
        xlabel="Feature",
        ylabel="% of total",
        labels=["Nodes", "Samples"],
        path=f"{output_dir}/top_features_lines_bars.pdf",
    )


def plot_top_nodes(top_nodes, dt_samples_by_class, dt_samples, output_dir, feature_names=[], class_names=[]):
    """Uses top features information and plots CDF with it"""
    if not np.array(top_nodes).size or not np.array(dt_samples_by_class).size or not np.array(dt_samples).size:
        return

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
        ylim=(0, 100),
        xlabel="Node",
        ylabel="% of total samples",
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
        ylim=(0, 100),
        xlabel="Node",
        ylabel="% of total samples",
        labels=class_names if class_names is not None else [],
        path=f"{output_dir}/top_nodes_by_class.pdf",
    )


def plot_top_branches(
    top_branches,
    dt_samples_by_class,
    dt_samples,
    output_dir,
    filename="top_branches",
    class_names=[],
    is_classify=True,
):
    """Uses top features information and plots CDF with it"""
    if not np.array(top_branches).size or not np.array(dt_samples_by_class).size or not np.array(dt_samples).size:
        return

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
        class_label = class_names[branch["class"]] if class_names is not None else branch["class"]
        if class_label not in colors_by_class:
            colors_by_class[class_label] = (
                colors.pop() if colors else "#%02x%02x%02x" % tuple(np.random.randint(256, size=3))
            )
        samples.append((branch["samples"] / dt_samples) * 100)
        colors_by_samples.append(colors_by_class[class_label])

    plot.plot_lines(
        range(len(top_branches[:600])),
        [np.cumsum(samples[:600])],
        ylim=(0, 100),
        xlabel="Top-k Branches",
        ylabel="% of Samples",
        path=f"{output_dir}/{filename}.pdf",
    )

    plot.plot_lines_and_bars(
        [f"Top {idx + 1}" for idx in range(len(top_branches))] if len(top_branches) < 20 else range(len(top_branches)),
        [np.cumsum(samples)],
        [samples],
        ylim=(0, 100),
        xlabel="Branches",
        ylabel="% of total samples",
        legend={"CDF": "#d75d5b", **colors_by_class},
        colors_by_x=colors_by_samples,
        path=f"{output_dir}/{filename}_by_class.pdf",
    )

    # TODO: This only works for classification problems, fix for refression in the future.
    if is_classify:
        plot.plot_stacked_bars(
            [f"Top {idx + 1}" for idx in range(len(top_branches))]
            if len(top_branches) < 20
            else range(len(top_branches)),
            [np.cumsum(samples)],
            y_placeholder=[100],
            ylim=(0, 100),
            xlabel="Branches",
            ylabel="% of total samples",
            path=f"{output_dir}/{filename}_bars.pdf",
        )

        plot.plot_stacked_bars(
            [f"Top {idx + 1}" for idx in range(len(top_branches))]
            if len(top_branches) < 20
            else range(len(top_branches)),
            [
                np.cumsum(
                    [
                        ((branch["samples"] / dt_samples) * 100) if idx == branch["class"] else 0
                        for branch in top_branches
                    ]
                )
                for idx, _ in enumerate(dt_samples_by_class)
            ],
            y_placeholder=[(samples / dt_samples) * 100 for samples in dt_samples_by_class],
            ylim=(0, 100),
            xlabel="Branches",
            ylabel="% of total samples",
            labels=class_names,
            path=f"{output_dir}/cum_{filename}_by_class.pdf",
        )


def plot_all_branches(top_branches, dt_samples_by_class, dt_samples, output_dir, class_names=[], is_classify=True):
    """Uses all features information and plots CDF with it"""
    plot_top_branches(
        top_branches,
        dt_samples_by_class,
        dt_samples,
        output_dir,
        filename="all_branches",
        class_names=class_names,
        is_classify=is_classify,
    )


def plot_samples_by_level(dt_samples_by_level, dt_nodes_by_level, dt_samples, output_dir):
    """Uses dt information to plot number of samples per level"""
    if not np.array(dt_nodes_by_level).size or not np.array(dt_nodes_by_level).size or not np.array(dt_samples).size:
        return

    samples = []
    for idx, level_samples in enumerate(dt_samples_by_level):
        if idx < len(dt_samples_by_level) - 1:
            samples.append(((level_samples - dt_samples_by_level[idx + 1]) / dt_samples) * 100)
        else:
            samples.append((level_samples / dt_samples) * 100)

    plot.plot_lines_and_bars(
        [level for level, _ in enumerate(dt_samples_by_level)],
        [np.cumsum(samples)],
        [samples],
        ylim=(0, 100),
        xlabel="Level",
        ylabel="% of total samples",
        second_x_axis=dt_nodes_by_level,
        second_x_axis_label="Leaves at Level",
        labels=["Samples"],
        legend={"CDF": "#d75d5b", "Samples": "#c8c5c3"},
        path=f"{output_dir}/samples_by_level.pdf",
    )


def plot_dts_fidelity_by_size(pruning_list, output_dir, filename="dts"):
    """Uses pruning information to plot fidelity vs size of decision trees"""
    if not np.array(pruning_list).size:
        return

    num_leaves = {}
    depth = {}
    fidelity = {}

    for pr in pruning_list:
        for i in pr["iter"]:
            if pr["type"] not in num_leaves:
                num_leaves[pr["type"]] = []
                depth[pr["type"]] = []
                fidelity[pr["type"]] = []

            num_leaves[pr["type"]].append(i["dt"].get_n_leaves())
            depth[pr["type"]].append(i["dt"].get_depth())
            fidelity[pr["type"]].append(i["fidelity"])

    plot.plot_lines(
        list(num_leaves.values()),
        list(fidelity.values()),
        ylim=(0, 1),
        xlim=(0, 50),
        xlabel="Number of Branches",
        ylabel="Fidelity",
        labels=list(num_leaves.keys()),
        path=f"{output_dir}/{filename}_fidelity_x_leaves.pdf",
    )

    plot.plot_lines(
        list(depth.values()),
        list(fidelity.values()),
        ylim=(0, 1),
        xlabel="Depth",
        ylabel="Fidelity",
        labels=list(depth.keys()),
        path=f"{output_dir}/{filename}_fidelity_x_depth.pdf",
    )


def plot_stability(
    stability_iter,
    X_test,
    y_test,
    base_tree,
    base_tree_key,
    top_branches,
    output_dir,
    class_names=[],
    is_classify=True,
):
    """Uses stability information to plot the edit-distance between decision trees"""
    if not np.array(stability_iter).size:
        return

    agreement = []
    agreement_by_class = {}
    nodes = {}
    features = {}
    features_by_it = {}
    total_nodes = 0
    number_of_splits = []
    fidelity = []
    base_y_pred = base_tree.predict(X_test.values)
    base_df = pd.DataFrame(deepcopy(X_test))
    base_df["label"] = y_test
    grouped_df = base_df.groupby("label") if is_classify else []

    for idx, it in enumerate(stability_iter):
        iter_tree = it[f"{base_tree_key}"]
        _, splits, _ = get_dt_info(iter_tree)
        total_nodes += len(splits)
        features_by_it[idx] = 0
        number_of_splits.append(len(splits))

        for split in splits:
            split_str = f"{split['feature']}-{split['threshold']}"
            if split_str not in nodes:
                nodes[split_str] = 0
            nodes[split_str] += 1

            if split["feature"] not in features:
                features[split["feature"]] = {}

            if idx not in features[split["feature"]]:
                features[split["feature"]][idx] = 0
                features_by_it[idx] += 1
            features[split["feature"]][idx] += 1

        y_pred = iter_tree.predict(X_test.values)
        fidelity.append(it[f"{base_tree_key}_fidelity"])
        agreement.append(
            f1_score(y_pred, base_y_pred, average="weighted") if is_classify else r2_score(y_pred, base_y_pred)
        )

        for group, data in grouped_df:
            y_pred_class = iter_tree.predict(data.drop("label", axis=1).values)
            base_y_pred_class = base_tree.predict(data.drop("label", axis=1).values)
            if group not in agreement_by_class:
                agreement_by_class[group] = []

            agreement_by_class[group].append(f1_score(y_pred_class, base_y_pred_class, average="weighted"))

    plot.plot_lines(
        range(len(number_of_splits)),
        [number_of_splits],
        xlabel="Iteration",
        ylabel="Number of Splits",
        path=f"{output_dir}/{base_tree_key}_num_nodes_stability.pdf",
    )

    plot.plot_lines(
        range(len(features_by_it.keys())),
        [features_by_it.values()],
        xlabel="Iteration",
        ylabel="Stability",
        labels=["Features"],
        path=f"{output_dir}/{base_tree_key}_feature_stability.pdf",
    )

    plot.plot_lines(
        range(len(stability_iter)),
        [agreement, fidelity],
        ylim=(0, 1),
        xlabel="Iteration",
        ylabel="Score",
        labels=["Agreement", "Fidelity"],
        path=f"{output_dir}/{base_tree_key}_stability.pdf",
    )

    if is_classify:
        top_branch_agreement = {}
        for branch in top_branches[:5]:
            class_name = class_names[branch["class"]] if class_names is not None else branch["class"]
            class_id = class_name if class_name in agreement_by_class else branch["class"]
            top_branch_agreement[class_id] = agreement_by_class[class_id]

        plot.plot_lines(
            range(len(stability_iter)),
            [agreement for _, agreement in top_branch_agreement.items()],
            ylim=(0, 1),
            xlabel="Iteration",
            ylabel="Agreement (Score)",
            labels=[
                class_names[group] if class_names is not None and not isinstance(group, str) else group
                for group, _ in top_branch_agreement.items()
            ],
            path=f"{output_dir}/{base_tree_key}_stability_by_class.pdf",
            size=(6, 4),
        )


def plot_stability_heatmap(
    stability_iter,
    X_test,
    y_test,
    tree_key,
    top_branches,
    output_dir,
    class_names=[],
    is_classify=True,
):
    """Uses stability information to plot the edit-distance between decision trees"""
    if not np.array(stability_iter).size:
        return

    heatmap_size = 30
    agreement = []
    fidelity = []
    mean_agreement = []
    agreement_by_class = {}
    base_df = pd.DataFrame(deepcopy(X_test))
    base_df["label"] = y_test
    grouped_df = base_df.groupby("label") if is_classify else []

    for i, _ in enumerate(stability_iter):
        base_tree = stability_iter[i][f"{tree_key}"]
        fidelity.append(stability_iter[i][f"{tree_key}_fidelity"])
        agreement.append([])

        for j, _ in enumerate(stability_iter):
            iter_tree = stability_iter[j][f"{tree_key}"]

            iter_y_pred = iter_tree.predict(X_test.values)
            base_y_pred = base_tree.predict(X_test.values)

            agreement[i].append(
                f1_score(iter_y_pred, base_y_pred, average="weighted")
                if is_classify
                else r2_score(iter_y_pred, base_y_pred)
            )

            for group, data in grouped_df:
                y_pred_class = iter_tree.predict(data.drop("label", axis=1).values)
                base_y_pred_class = base_tree.predict(data.drop("label", axis=1).values)
                if group not in agreement_by_class:
                    agreement_by_class[group] = []

                if i >= len(agreement_by_class[group]):
                    agreement_by_class[group].append([])

                agreement_by_class[group][i].append(
                    f1_score(y_pred_class, base_y_pred_class, average="weighted")
                    if is_classify
                    else r2_score(y_pred_class, base_y_pred_class)
                )
        mean_agreement.append(np.mean(agreement[i]))

    plot.plot_lines(
        range(len(stability_iter)),
        [mean_agreement, fidelity],
        ylim=(0, 1),
        xlim=(0, 50),
        xlabel="Iteration",
        ylabel="Score",
        labels=["Mean Agreement", "Fidelity"],
        path=f"{output_dir}/{tree_key}_mean_stability.pdf",
    )

    plot.plot_heatmap(
        np.array([arr[:heatmap_size] for arr in agreement[:heatmap_size]]),
        labels=range(min(len(stability_iter), heatmap_size)),
        path=f"{output_dir}/{tree_key}_stability_heatmap.pdf",
    )

    if is_classify:
        top_branch_agreement = {}
        for branch in top_branches[:5]:
            class_name = class_names[branch["class"]] if class_names is not None else branch["class"]
            class_id = class_name if class_name in agreement_by_class else branch["class"]
            top_branch_agreement[class_id] = agreement_by_class[class_id]

        for group, group_agreement in top_branch_agreement.items():
            plot.plot_heatmap(
                np.array(group_agreement[:heatmap_size]),
                labels=range(min(len(stability_iter), heatmap_size)),
                path=f"{output_dir}/{tree_key}_{class_names[group] if class_names is not None and not isinstance(group, str) else group}_stability_heatmap.pdf",
            )


def plot_accuracy_by_feature_removed(whitebox_iter, output_dir, feature_names=[]):
    """Uses iterative analysis information to plot f1-score from the trained blackbox vs number of features removed"""
    if not np.array(whitebox_iter).size:
        return

    blackbox_scores = [i["score"] * 100 for i in whitebox_iter]
    fidelity = [i["fidelity"] * 100 for i in whitebox_iter]
    features = [feature_names[i["feature_removed"]] if feature_names else i["feature_removed"] for i in whitebox_iter]
    plot.plot_lines(
        features,
        [blackbox_scores, fidelity],
        ylim=(0, 100),
        xlabel="Features removed",
        ylabel="Metric (%)",
        labels=["Blackbox Score", "DT Fidelity"],
        path=f"{output_dir}/accuracy_by_feature_removed.pdf",
    )


def plot_distribution(X, y, top_branches, output_dir, aggregate=False, feature_names=[], class_names=[]):
    """Plots the distribution of the data based on the top branches"""
    if not np.array(X).size or not np.array(y).size or not np.array(top_branches).size:
        return

    plots_output_dir = f"{output_dir}/dist" if not aggregate else f"{output_dir}/aggr_dist"
    if not os.path.exists(plots_output_dir):
        os.makedirs(plots_output_dir)

    colors = [
        "#d75d5b",
        "#524a47",
        "#8a4444",
        "#edeef0",
        "#c8c5c3",
        "#f5f0ed",
        "#a7c3cd",
    ]

    df = pd.DataFrame(X, columns=feature_names if feature_names else None)
    if isinstance(df.columns[0], numbers.Number):
        df.columns = [str(i) for i in range(len(df.columns))]

    if aggregate:
        col_regex = "([\w_]+)_([0-9]+)"
        opt_prefixes = set({})
        non_opt_prefixes = set({})
        field_size = {}
        non_aggr_cols = df.columns  # for plotting
        for col in df.columns:
            match_groups = re.findall(col_regex, col)[0]
            prefix = match_groups[0]
            bit = int(match_groups[1])
            if "opt" in col:
                opt_prefixes.add(prefix)
            else:
                non_opt_prefixes.add(prefix)

            if prefix not in field_size:
                field_size[prefix] = bit

            if field_size[prefix] < bit:
                field_size[prefix] = bit

        # we need to treat option differently
        opt_df = df[[col for col in df.columns if "opt" in col]]
        non_opt_df = df[[col for col in df.columns if "opt" not in col]]

        def bin_to_int(num):
            try:
                return int(num, 2)
            except:
                return -1

        grouper = [next(p for p in non_opt_prefixes if p in c) for c in non_opt_df.columns]
        non_opt_df = non_opt_df.groupby(grouper, axis=1).apply(
            lambda x: x.astype(str).apply("".join, axis=1).apply(bin_to_int)
        )
        df = pd.concat([non_opt_df, opt_df], axis=1)

    df["label"] = y
    if class_names is not None and is_numeric_dtype(df["label"]):
        df["label"] = df["label"].map(lambda x: class_names[int(x)])

    num_classes = len(np.unique(y))
    split_dfs = [x for _, x in df.groupby("label")]

    for idx, branch in enumerate(top_branches):
        branch_class = class_names[branch["class"]] if class_names is not None else str(branch["class"])
        branch_output_dir = f"{plots_output_dir}/{idx}_branch_{branch_class}"

        if not os.path.exists(branch_output_dir):
            os.makedirs(branch_output_dir)

        filtered_dfs = [x.copy(deep=True) for _, x in df.groupby("label")]
        for rule_idx, (_, feat, op, thresh) in enumerate(branch["path"]):
            if aggregate:
                column = non_aggr_cols[int(feat)] if (isinstance(feat, numbers.Number) or feat.isdigit()) else feat
                if "opt" not in column:
                    match_groups = re.findall(col_regex, column)[0]
                    column = match_groups[0]
                    bit = match_groups[1]
                    shift = field_size[column]
                    thresh = (1 << (shift - int(bit))) - (1 - int(thresh))
            else:
                column = df.columns[int(feat)] if (isinstance(feat, numbers.Number) or feat.isdigit()) else feat

            plots_per_row = 5
            if num_classes > plots_per_row:
                n_rows = math.gcd(num_classes, plots_per_row)
                n_cols = num_classes if num_classes <= plots_per_row else int(num_classes / n_rows)
            else:
                n_rows = num_classes
                n_cols = 1

            fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols)
            axes = axes.flatten()
            for df_idx, split_df in enumerate(split_dfs):
                df_class = split_df["label"].unique()[0]

                ax = axes[df_idx]
                ax.hist(
                    split_df[column].values,
                    histtype="bar",
                    label="All" if df_idx == 0 else None,
                    color=colors[0],
                )
                ax.yaxis.set_major_formatter(PercentFormatter(xmax=split_df.shape[0]))
                ax.tick_params(axis="both", labelsize=6)
                ax.set_title(df_class, fontsize=8)

            tlt = fig.suptitle(f"{column} {op} {thresh:.3f}")
            lgd = fig.legend(loc="lower center", bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=5)
            plt.tight_layout()
            plt.savefig(
                f"{branch_output_dir}/{rule_idx}_{column.replace('/', '_')}_hist_all.pdf",
                bbox_extra_artists=(lgd, tlt),
                bbox_inches="tight",
            )
            plt.close()

            fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols)
            axes = axes.flatten()
            for df_idx, split_df in enumerate(split_dfs):
                branch_filter = f"filtered_dfs[{df_idx}]['{column}'] {op} {thresh}"
                filtered_dfs[df_idx] = filtered_dfs[df_idx][eval(branch_filter)]
                df_class = split_df["label"].unique()[0]

                ax = axes[df_idx]
                ax.hist(
                    filtered_dfs[df_idx][column].values,
                    histtype="bar",
                    label=f"Branch ({branch_class.strip()})" if df_idx == 0 else None,
                    color=colors[-1],
                )
                ax.yaxis.set_major_formatter(PercentFormatter(xmax=split_df.shape[0]))
                ax.tick_params(axis="both", labelsize=6)
                ax.set_title(df_class, fontsize=8)

            tlt = fig.suptitle(f"{column} {op} {thresh:.3f}")
            lgd = fig.legend(loc="lower center", bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=5)
            plt.tight_layout()
            plt.savefig(
                f"{branch_output_dir}/{rule_idx}_{column.replace('/', '_')}_hist_branch.pdf",
                bbox_extra_artists=(lgd, tlt),
                bbox_inches="tight",
            )
            plt.close()

            fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols)
            axes = axes.flatten()
            for df_idx, split_df in enumerate(split_dfs):
                df_class = split_df["label"].unique()[0]
                branch_filter = f"filtered_dfs[{df_idx}]['{column}'] {op} {thresh}"
                filtered_dfs[df_idx] = filtered_dfs[df_idx][eval(branch_filter)]

                ax = axes[df_idx]
                ax.hist(
                    split_df[column].values,
                    histtype="bar",
                    label="All" if df_idx == 0 else None,
                    color=colors[0],
                )
                ax.hist(
                    filtered_dfs[df_idx][column].values,
                    histtype="bar",
                    label=f"Branch ({branch_class.strip()})" if df_idx == 0 else None,
                    color=colors[-1],
                )
                ax.yaxis.set_major_formatter(PercentFormatter(xmax=split_df.shape[0]))
                ax.tick_params(axis="both", labelsize=6)
                ax.set_title(df_class, fontsize=8)

            tlt = fig.suptitle(f"{column} {op} {thresh:.3f}")
            lgd = fig.legend(loc="lower center", bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=5)
            plt.tight_layout()
            plt.savefig(
                f"{branch_output_dir}/{rule_idx}_{column.replace('/', '_')}_hist.pdf",
                bbox_extra_artists=(lgd, tlt),
                bbox_inches="tight",
            )
            plt.close()
