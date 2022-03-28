import numpy as np
from scipy import spatial

from skexplain.utils import plot


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
    return features_used, splits, branches, samples_sum


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


def plot_samples_by_level(dt_samples_by_level, dt_nodes_by_level, dt_samples, output_dir):
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
        second_x_axis=dt_nodes_by_level,
        labels=["Samples"],
        path=f"{output_dir}/samples_by_level.pdf",
    )
