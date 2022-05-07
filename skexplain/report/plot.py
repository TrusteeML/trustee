import os
import re
import math
import numbers
import numpy as np
import pandas as pd

from pprint import pprint
from scipy.stats import kurtosis, skew
from pandas.api.types import is_numeric_dtype

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


from skexplain.utils import plot


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
        labels=class_names,
        path=f"{output_dir}/top_nodes_by_class.pdf",
    )


def plot_top_branches(
    top_branches, dt_samples_by_class, dt_samples, output_dir, filename="top_branches", class_names=[]
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
        class_label = class_names[branch["class"]] if class_names else branch["class"]
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

    plot.plot_stacked_bars(
        [f"Top {idx + 1}" for idx in range(len(top_branches))] if len(top_branches) < 20 else range(len(top_branches)),
        [np.cumsum(samples)],
        y_placeholder=[100],
        ylim=(0, 100),
        xlabel="Branches",
        ylabel="% of total samples",
        path=f"{output_dir}/{filename}_bars.pdf",
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
        xlabel="Top-k Branches",
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


def plot_stability_by_top_k(stability_iter, top_k, output_dir):
    """Uses stabitlity information to plot most stable branches over multiple iterations"""
    if not np.array(stability_iter).size:
        return

    top_k_branches = {}
    top_k_branches_wo_order = {}

    heartbleed_branches = []
    similar_branches = {}
    for it in stability_iter:
        num_branches = len(it["top_branches"])
        for k in range(min(top_k, num_branches)):
            top_k_branch = it["top_branches"][k]
            feature_list = [f"{feat} {op} " for (_, feat, op, _) in top_k_branch["path"]]
            sorted_feature_list = [
                f"{feat} {op}" for (_, feat, op, _) in sorted(top_k_branch["path"], key=lambda x: x[1])
            ]
            sorted_feature_list_with_thresh = [
                f"{feat} {op}" for (_, feat, op, _) in sorted(top_k_branch["path"], key=lambda x: x[1])
            ]

            if top_k_branch["class"] == 8:
                heartbleed_branches.append(sorted_feature_list_with_thresh)

            branch = f"{','.join(feature_list)} = {top_k_branch['class']}"
            top_k_branches.setdefault(branch, 0)
            top_k_branches[branch] += 1

            sorted_branch = f"{','.join(sorted_feature_list)} = {top_k_branch['class']}"
            top_k_branches_wo_order.setdefault(sorted_branch, 0)
            top_k_branches_wo_order[sorted_branch] += 1

            similar_branches.setdefault(sorted_branch, [])
            similar_branches[sorted_branch].append(sorted_feature_list_with_thresh)

    top_10_branches = {}
    top_20_branches = {}
    top_30_branches = {}
    for it in stability_iter:
        num_branches = len(it["top_branches"])
        for k in range(min(30, num_branches)):
            top_k_branch = it["top_branches"][k]
            sorted_feature_list = [
                f"{feat} {op}" for (_, feat, op, _) in sorted(top_k_branch["path"][:-1], key=lambda x: x[1])
            ]
            branch = f"{','.join(sorted_feature_list)} = {top_k_branch['class']}"
            top_30_branches.setdefault(branch, 0)
            top_30_branches[branch] += 1

            if k < 20:
                top_20_branches.setdefault(branch, 0)
                top_20_branches[branch] += 1

            if k < 10:
                top_10_branches.setdefault(branch, 0)
                top_10_branches[branch] += 1

    top_branches = sorted(top_k_branches.items(), key=lambda item: item[1], reverse=True)
    branch_stability = [(x[1] / len(stability_iter)) * 100 for x in top_branches][:top_k]
    top_branches_wo_order = sorted(top_k_branches_wo_order.items(), key=lambda item: item[1], reverse=True)
    branch_stability_wo_order = [(x[1] / len(stability_iter)) * 100 for x in top_branches_wo_order][:top_k]

    plot.plot_lines(
        range(1, min(top_k + 1, len(branch_stability) + 1)),
        [branch_stability, branch_stability_wo_order],
        ylim=(0, 100),
        xlabel="Top Branches",
        ylabel="Stability (%)",
        labels=["W/ Feature Order", "W/O Feature Order"],
        path=f"{output_dir}/branch_stability.pdf",
    )

    branch_stability = [(x[1] / len(stability_iter)) * 100 for x in top_branches]
    branch_stability_wo_order = [(x[1] / len(stability_iter)) * 100 for x in top_branches_wo_order]
    plot.plot_lines(
        [range(1, len(branch_stability) + 1), range(1, len(branch_stability_wo_order) + 1)],
        [branch_stability, branch_stability_wo_order],
        ylim=(0, 100),
        xlabel="Top Branches",
        ylabel="Stability (%)",
        labels=["W/ Feature Order", "W/O Feature Order"],
        path=f"{output_dir}/branch_stability_uncapped.pdf",
    )

    num_branches = 50
    top_10 = sorted(top_10_branches.items(), key=lambda item: item[1], reverse=True)
    top_20 = sorted(top_20_branches.items(), key=lambda item: item[1], reverse=True)
    top_30 = sorted(top_30_branches.items(), key=lambda item: item[1], reverse=True)
    branch_stability_top_10 = [(x[1] / len(stability_iter)) * 100 for x in top_10][:num_branches]
    branch_stability_top_20 = [(x[1] / len(stability_iter)) * 100 for x in top_20][:num_branches]
    branch_stability_top_30 = [(x[1] / len(stability_iter)) * 100 for x in top_30][:num_branches]
    plot.plot_lines(
        [
            range(1, len(branch_stability_top_10) + 1),
            range(1, len(branch_stability_top_20) + 1),
            range(1, len(branch_stability_top_30) + 1),
        ],
        [branch_stability_top_10, branch_stability_top_20, branch_stability_top_30],
        ylim=(0, 100),
        xlabel="Top Branches",
        ylabel="Stability (%)",
        labels=["Top-10", "Top-20", "Top-30"],
        path=f"{output_dir}/branch_stability_multitop.pdf",
    )


def plot_accuracy_by_feature_removed(whitebox_iter, output_dir, feature_names=[]):
    """Uses iterative analysis information to plot f1-score from the trained blackbox vs number of features removed"""
    if not np.array(whitebox_iter).size:
        return

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


def plot_heartbleed_distribution(X, y, output_dir, feature_names=[], class_names=[]):
    if not np.array(X).size or not np.array(y).size:
        return

    plots_output_dir = f"{output_dir}/dist"
    if not os.path.exists(plots_output_dir):
        os.makedirs(plots_output_dir)

    plt.rcParams["figure.figsize"] = (5, 2)
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

    df["label"] = y
    if class_names and is_numeric_dtype(df["label"]):
        df["label"] = df["label"].map(lambda x: class_names[int(x)])

    grouped_df = df.groupby("label")
    heartbleed_df = grouped_df.get_group("Heartbleed")
    others_df = df.drop(heartbleed_df.index)

    fig, axes = plt.subplots(nrows=1, ncols=2)
    axes = axes.flatten()

    column = "Bwd Packet Length Max"
    filtered_other_df = others_df[others_df[column] > 12332]
    filtered_heartbleed_df = heartbleed_df[heartbleed_df[column] > 12332]

    ax = axes[0]
    ax.hist(
        others_df[column].values,
        # bins=50,
        histtype="bar",
        label="Others",
        color=colors[0],
    )
    ax.hist(
        filtered_other_df[column].values,
        # bins=50,
        histtype="bar",
        label=f"Branch (Heartbleed)",
        color=colors[-1],
    )
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=others_df.shape[0]))
    ax.tick_params(axis="both", labelsize=10)
    ax.set_title("Others", fontsize=12, fontweight=plot.FONT_WEIGHT, fontname=plot.FONT_NAME)

    ax = axes[1]
    ax.hist(
        heartbleed_df[column].values,
        # bins=50,
        histtype="bar",
        # label="Others",
        color=colors[0],
    )
    ax.hist(
        filtered_heartbleed_df[column].values,
        # bins=50,
        histtype="bar",
        # label=f"Branch (Heartbleed)",
        color=colors[-1],
    )
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=heartbleed_df.shape[0]))
    ax.tick_params(axis="both", labelsize=10)
    ax.set_title("Heartbleed", fontsize=12, fontweight=plot.FONT_WEIGHT, fontname=plot.FONT_NAME)

    tlt = fig.suptitle(f"{column} > 12332", fontweight=plot.FONT_WEIGHT, fontname=plot.FONT_NAME)
    lgd = fig.legend(loc="lower center", bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=5)
    plt.tight_layout()
    plt.savefig(
        f"{plots_output_dir}/heartbleed_{column.replace('/', '_')}_hist.pdf",
        bbox_extra_artists=(lgd, tlt),
        bbox_inches="tight",
    )
    plt.close()

    fig, axes = plt.subplots(nrows=1, ncols=2)
    axes = axes.flatten()

    column = "Bwd IAT Total"
    filtered_other_df = others_df[others_df[column] > 110000000]
    filtered_heartbleed_df = heartbleed_df[heartbleed_df[column] > 110000000]

    ax = axes[0]
    ax.hist(
        others_df[column].values,
        # bins=50,
        histtype="bar",
        label="Others",
        color=colors[0],
    )
    ax.hist(
        filtered_other_df[column].values,
        # bins=50,
        histtype="bar",
        label=f"Branch (Heartbleed)",
        color=colors[-1],
    )
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=others_df.shape[0]))
    ax.tick_params(axis="both", labelsize=10)
    ax.set_title("Others", fontsize=12, fontweight=plot.FONT_WEIGHT, fontname=plot.FONT_NAME)

    ax = axes[1]
    ax.hist(
        heartbleed_df[column].values,
        # bins=50,
        histtype="bar",
        # label="Others",
        color=colors[0],
    )
    ax.hist(
        filtered_heartbleed_df[column].values,
        # bins=50,
        histtype="bar",
        range=[0, 120000000],
        # label=f"Branch (Heartbleed)",
        color=colors[-1],
    )
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=heartbleed_df.shape[0]))
    ax.tick_params(axis="both", labelsize=10)
    ax.set_title("Heartbleed", fontsize=12, fontweight=plot.FONT_WEIGHT, fontname=plot.FONT_NAME)

    tlt = fig.suptitle(f"{column} > 1.1e8", fontweight=plot.FONT_WEIGHT, fontname=plot.FONT_NAME)
    lgd = fig.legend(loc="lower center", bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=5)
    plt.tight_layout()
    plt.savefig(
        f"{plots_output_dir}/heartbleed_{column.replace('/', '_')}_hist.pdf",
        bbox_extra_artists=(lgd, tlt),
        bbox_inches="tight",
    )
    plt.close()


def plot_skewness_heatmaps(X, top_features, output_dir, feature_names=[]):
    """Combines all top features into pairs and calculate kurtosis metric with each pair to form a heatmap."""
    if not np.array(X).size or not np.array(top_features).size:
        return

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
