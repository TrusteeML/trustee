import os
import re
import math
import numbers
import numpy as np
import pandas as pd
import pingouin as pg
from scipy.stats import kurtosis, skew
from numpy.core.defchararray import isdigit
from pandas.api.types import is_numeric_dtype

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


from scipy import spatial

from skexplain.utils import plot


def plot_top_features(top_features, dt_sum_samples, dt_nodes, output_dir, feature_names=[]):
    """Uses top features information and plots CDF with it"""
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
        labels=class_names,
        path=f"{output_dir}/top_nodes_by_class.pdf",
    )


def plot_top_branches(
    top_branches, dt_samples_by_class, dt_samples, output_dir, filename="top_branches", class_names=[]
):
    """Uses top features information and plots CDF with it"""
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
            colors_by_class[class_label] = (
                colors.pop() if colors else "#%02x%02x%02x" % tuple(np.random.randint(256, size=3))
            )
        samples.append((branch["samples"] / dt_samples) * 100)
        colors_by_samples.append(colors_by_class[class_label])

    plot.plot_stacked_bars(
        [f"Top {idx + 1}" for idx in range(len(top_branches))] if len(top_branches) < 20 else range(len(top_branches)),
        [np.cumsum(samples)],
        y_placeholder=[100],
        ylim=(0, 100),
        xlabel="Branches",
        ylabel="% of total samples",
        path=f"{output_dir}/{filename}.pdf",
    )

    plot.plot_stacked_bars(
        [f"Top {idx + 1}" for idx in range(len(top_branches))] if len(top_branches) < 20 else range(len(top_branches)),
        [
            np.cumsum(
                [((branch["samples"] / dt_samples) * 100) if idx == branch["class"] else 0 for branch in top_branches]
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


def plot_all_branches(top_branches, dt_samples_by_class, dt_samples, output_dir, class_names=[]):
    """Uses all features information and plots CDF with it"""
    plot_top_branches(
        top_branches, dt_samples_by_class, dt_samples, output_dir, filename="all_branches", class_names=class_names
    )


def plot_samples_by_level(dt_samples_by_level, dt_nodes_by_level, dt_samples, output_dir):
    """Uses dt information to plot number of samples per level"""
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
            fidelity[pr["type"]].append(i["fidelity"] * 100)

    plot.plot_lines(
        list(num_leaves.values()),
        list(fidelity.values()),
        ylim=(0, 100),
        xlabel="Number of leaves",
        ylabel="Fidelity (%)",
        labels=list(num_leaves.keys()),
        path=f"{output_dir}/{filename}_fidelity_x_leaves.pdf",
    )

    plot.plot_lines(
        list(depth.values()),
        list(fidelity.values()),
        ylim=(0, 100),
        xlabel="Depth",
        ylabel="Fidelity (%)",
        labels=list(depth.keys()),
        path=f"{output_dir}/{filename}_fidelity_x_depth.pdf",
    )


def plot_accuracy_by_feature_removed(whitebox_iter, output_dir, feature_names=[]):
    """Uses iterative analysis information to plot f1-score from the trained blackbox vs number of features removed"""
    blackbox_f1_scores = [i["f1"] * 100 for i in whitebox_iter]
    fidelity = [i["fidelity"] * 100 for i in whitebox_iter]
    features = [feature_names[i["feature_removed"]] if feature_names else i["feature_removed"] for i in whitebox_iter]
    plot.plot_lines(
        features,
        [blackbox_f1_scores, fidelity],
        ylim=(0, 100),
        xlabel="Features removed",
        ylabel="Metric (%)",
        labels=["Blackbox F1-Score", "DT Fidelity"],
        path=f"{output_dir}/accuracy_by_feature_removed.pdf",
    )


def plot_distribution(X, y, top_branches, output_dir, aggregate=False, feature_names=[], class_names=[]):
    """Plots the distribution of the data based on the top branches"""
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
        # print(non_opt_df)

        # grouper = [next(p for p in opt_prefixes if p in c) for c in opt_df.columns]
        # opt_df = opt_df.groupby(grouper, axis=1).apply(lambda x: x.astype(str).apply("".join, axis=1))
        # for col in opt_df.columns:
        #     for idx, start in enumerate(range(0, 320, 32)):
        #         print(idx, start)
        #         opt_df[f"{col}_{idx}"] = opt_df[col].str.slice(start, start + 32)

        #     print(opt_df[col].str.extract("(.{32,32})" * 10, expand=True))

        # print(opt_df)
        df = pd.concat([non_opt_df, opt_df], axis=1)

    df["label"] = y
    if class_names and is_numeric_dtype(df["label"]):
        df["label"] = df["label"].map(lambda x: class_names[int(x)])

    num_classes = len(np.unique(y))
    split_dfs = [x for _, x in df.groupby("label")]

    for idx, branch in enumerate(top_branches):
        branch_class = class_names[branch["class"]] if class_names else branch["class"]
        branch_output_dir = f"{plots_output_dir}/{idx}_branch_{branch_class.strip()}"

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
                    # bins=50,
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
                    # bins=50,
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
                    # bins=50,
                    histtype="bar",
                    label="All" if df_idx == 0 else None,
                    color=colors[0],
                )
                ax.hist(
                    filtered_dfs[df_idx][column].values,
                    # bins=50,
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


def plot_skewness_heatmaps(X, top_features, output_dir, feature_names=[]):
    """Combines all top features into pairs and calculate kurtosis metric with each pair to form a heatmap."""
    df = pd.DataFrame(X, columns=feature_names if feature_names else None)
    if isinstance(df.columns[0], numbers.Number):
        df.columns = [str(i) for i in range(len(df.columns))]

    skew_matrix = np.zeros((len(top_features), len(top_features)))
    kurtosis_matrix = np.zeros((len(top_features), len(top_features)))
    # multivariate_normal_matrix = np.zeros((len(top_features), len(top_features)))
    features = []
    print(len(top_features))
    for i in range(len(top_features)):
        for j in range(len(top_features)):
            feat_i = top_features[i][0]
            feat_j = top_features[j][0]
            feat_names = feature_names[top_features[i][0]] if feature_names else top_features[i][0]
            if feat_names not in features:
                features.append(feat_names)

            print(df.iloc[:, [feat_i, feat_j]])
            print(f"Skew ({feat_i},{feat_j})", skew(df.iloc[:, [feat_i, feat_j]].values))
            print(f"Kurtosis ({feat_i},{feat_j})", kurtosis(df.iloc[:, [feat_i, feat_j]].values))
            # print(
            #     f"Multivariate normal ({feat_i},{feat_j})",
            #     pg.multivariate_normality(df.head(10000).iloc[:, [feat_i, feat_j]], alpha=0.05),
            # )
            skew_matrix[i, j] = np.mean(skew(df.iloc[:, [feat_i, feat_j]].values))
            kurtosis_matrix[i, j] = np.mean(kurtosis(df.iloc[:, [feat_i, feat_j]].values))
            # multivariate_normal_matrix[i, j] = pg.multivariate_normality(df.iloc[:, [feat_i, feat_j]], alpha=0.05)

    plot.plot_heatmap(
        skew_matrix,
        labels=features,
        path=f"{output_dir}/skew.pdf",
    )

    plot.plot_heatmap(
        kurtosis_matrix,
        labels=features,
        path=f"{output_dir}/kurtosis.pdf",
    )
