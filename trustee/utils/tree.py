import numpy as np

from copy import deepcopy

from sklearn.tree._tree import TREE_LEAF, TREE_UNDEFINED


def prune_index(dt, index, prune_level):
    """Prunes the given decision tree at the given index and returns the number of pruned nodes"""
    if index < 0:
        return 0

    left_idx = dt.tree_.children_left[index]
    right_idx = dt.tree_.children_right[index]

    # turn node into a leaf by "unlinking" its children
    dt.tree_.children_left[index] = TREE_LEAF if prune_level == 0 else TREE_UNDEFINED
    dt.tree_.children_right[index] = TREE_LEAF if prune_level == 0 else TREE_UNDEFINED

    # if there are shildren, visit them as well
    if left_idx != TREE_LEAF and right_idx != TREE_LEAF:
        prune_index(dt, left_idx, prune_level + 1)
        prune_index(dt, right_idx, prune_level + 1)


def get_dt_dict(dt):
    """Iterates through the given Decision Tree to collect updated tree node structure"""
    children_left = dt.tree_.children_left
    children_right = dt.tree_.children_right
    features = dt.tree_.feature
    thresholds = dt.tree_.threshold
    samples = dt.tree_.n_node_samples
    weighted_samples = dt.tree_.weighted_n_node_samples
    impurity = dt.tree_.impurity
    values = dt.tree_.value

    idx_inc = 0
    nodes = []
    # values = []

    def walk_tree(node, level, idx):
        """Recursively iterates through all nodes in given decision tree and returns them as a list."""
        left = children_left[node]
        right = children_right[node]

        nonlocal idx_inc
        if left != right:  # if not  leaf node
            idx_inc += 1
            left = walk_tree(left, level + 1, idx_inc)
            idx_inc += 1
            right = walk_tree(right, level + 1, idx_inc)

        nodes.append(
            {
                "idx": idx,
                "node": node,
                "left": left,
                "right": right,
                "level": level,
                "feature": features[node],
                "threshold": thresholds[node],
                "impurity": impurity[node],
                "samples": samples[node],
                "values": values[node],
                "weighted_samples": weighted_samples[node],
            }
        )

        return idx

    walk_tree(0, 0, idx_inc)

    node_dtype = [
        ("left_child", "<i8"),
        ("right_child", "<i8"),
        ("feature", "<i8"),
        ("threshold", "<f8"),
        ("impurity", "<f8"),
        ("n_node_samples", "<i8"),
        ("weighted_n_node_samples", "<f8"),
    ]
    node_ndarray = np.array([], dtype=node_dtype)
    node_values = []
    max_depth = 0
    for node in sorted(nodes, key=lambda x: x["idx"]):
        if node["level"] > max_depth:
            max_depth = node["level"]

        node_ndarray = np.append(
            node_ndarray,
            np.array(
                [
                    (
                        node["left"],
                        node["right"],
                        node["feature"],
                        node["threshold"],
                        node["impurity"],
                        node["samples"],
                        node["weighted_samples"],
                    )
                ],
                dtype=node_dtype,
            ),
        )
        node_values.append(node["values"])
    value_ndarray = np.array(node_values, dtype=np.float64)

    dt_dict = {
        "max_depth": max_depth,
        "node_count": len(node_ndarray),
        "nodes": node_ndarray,
        "values": value_ndarray,
    }

    return dt_dict


def get_dt_info(dt):
    """Iterates through the given Decision Tree to collect relevant information."""
    children_left = dt.tree_.children_left
    children_right = dt.tree_.children_right
    features = dt.tree_.feature
    thresholds = dt.tree_.threshold
    values = dt.tree_.value
    samples = dt.tree_.n_node_samples
    impurity = dt.tree_.impurity

    splits = []
    features_used = {}

    def walk_tree(node, level, path):
        """Recursively iterates through all nodes in given decision tree and returns them as a list."""
        if children_left[node] == children_right[node]:  # if leaf node
            node_class = np.argmax(values[node][0]) if len(np.array(values[node][0])) > 1 else values[node][0][0]
            node_prob = (
                (values[node][0][node_class] / np.sum(values[node][0])) * 100
                if np.array(values[node][0]).ndim > 1
                else 0
            )
            return [
                {
                    "level": level,
                    "path": path,
                    "class": node_class,
                    "prob": node_prob,
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

        return walk_tree(left, level + 1, path + [(node, feature, "<=", threshold)]) + walk_tree(
            right, level + 1, path + [(node, feature, ">", threshold)]
        )

    branches = walk_tree(0, 0, [])
    return features_used, splits, branches


def top_k_prune(dt, top_k, max_impurity=0.1):
    """Prunes a given decision tree down to its top-k branches, sorted by number of samples covered"""
    _, nodes, branches = get_dt_info(dt)
    top_branches = sorted(branches, key=lambda p: p["samples"], reverse=True)[:top_k]
    prunned_dt = deepcopy(dt)

    nodes_to_keep = set({})
    for branch in top_branches:
        for (node, _, _, _) in branch["path"]:
            if dt.tree_.impurity[node] > max_impurity:
                nodes_to_keep.add(node)

    for node in nodes:
        if node["idx"] not in nodes_to_keep:
            prune_index(prunned_dt, node["idx"], 0)

    # update classifier with prunned model
    prunned_dt.tree_.__setstate__(get_dt_dict(prunned_dt))

    return prunned_dt
