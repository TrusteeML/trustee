"""Tests for the decision-tree utilities.

Several of these guard the scikit-learn 1.3 change that turned classifier
``tree_.value`` from raw sample counts into per-node class fractions. That
change breaks silently -- the numbers stay finite and plausible-looking -- so
these assertions check magnitudes, not just shapes.
"""

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

from trustee.utils.tree import get_dt_info, get_node_counts, top_k_prune


class TestGetNodeCounts:
    def test_classifier_values_are_counts_not_fractions(self, fitted_classifier):
        counts = get_node_counts(fitted_classifier)
        root_total = counts[0].sum()
        assert root_total == pytest.approx(fitted_classifier.tree_.n_node_samples[0])
        # The bug this guards would make every node sum to exactly 1.0.
        assert root_total > 1.0

    def test_every_node_sums_to_its_sample_count(self, fitted_classifier):
        counts = get_node_counts(fitted_classifier)
        weighted = fitted_classifier.tree_.weighted_n_node_samples
        assert counts.sum(axis=2).ravel() == pytest.approx(weighted)

    def test_class_totals_match_the_training_data(self, iris, fitted_classifier):
        _, y = iris
        counts = get_node_counts(fitted_classifier)
        expected = np.bincount(y)
        assert counts[0][0] == pytest.approx(expected)

    def test_regressor_values_pass_through_untouched(self, fitted_regressor):
        # Regressor values hold the mean target, not counts, so they must not be scaled.
        assert get_node_counts(fitted_regressor) == pytest.approx(fitted_regressor.tree_.value)

    def test_shape_is_preserved(self, fitted_classifier):
        assert get_node_counts(fitted_classifier).shape == fitted_classifier.tree_.value.shape


class TestGetDtInfo:
    def test_returns_features_splits_and_branches(self, fitted_classifier):
        features, splits, branches = get_dt_info(fitted_classifier)
        assert features and splits and branches
        assert len(branches) == fitted_classifier.get_n_leaves()

    def test_data_split_halves_sum_to_the_parent(self, fitted_classifier):
        _, splits, _ = get_dt_info(fitted_classifier)
        for split in splits:
            left, right = split["data_split"]
            assert left + right == pytest.approx(split["samples"])

    def test_root_data_split_covers_the_whole_dataset(self, iris, fitted_classifier):
        X, _ = iris
        _, splits, _ = get_dt_info(fitted_classifier)
        root = next(s for s in splits if s["idx"] == 0)
        assert sum(root["data_split"]) == pytest.approx(len(X))

    def test_data_split_by_class_holds_counts(self, fitted_classifier):
        _, splits, _ = get_dt_info(fitted_classifier)
        root = next(s for s in splits if s["idx"] == 0)
        total = sum(left + right for left, right in root["data_split_by_class"])
        assert total == pytest.approx(fitted_classifier.tree_.n_node_samples[0])

    def test_leaf_probability_is_a_real_percentage(self, fitted_classifier):
        """`prob` used to be hard-zero because of an `.ndim > 1` test on a 1-D array."""
        _, _, branches = get_dt_info(fitted_classifier)
        probs = [b["prob"] for b in branches]
        assert all(0 <= p <= 100 for p in probs)
        assert any(p > 0 for p in probs)

    def test_pure_leaves_report_full_confidence(self, fitted_classifier):
        # An unconstrained tree on iris grows to purity, so every leaf is 100%.
        _, _, branches = get_dt_info(fitted_classifier)
        assert all(b["prob"] == pytest.approx(100.0) for b in branches)

    def test_branch_samples_sum_to_the_dataset(self, iris, fitted_classifier):
        X, _ = iris
        _, _, branches = get_dt_info(fitted_classifier)
        assert sum(b["samples"] for b in branches) == len(X)

    def test_regressor_splits_are_sample_based(self, diabetes, fitted_regressor):
        X, _ = diabetes
        _, splits, _ = get_dt_info(fitted_regressor)
        root = next(s for s in splits if s["idx"] == 0)
        assert sum(root["data_split"]) == pytest.approx(len(X))


class TestTopKPrune:
    def test_pruning_does_not_grow_the_tree(self, fitted_classifier):
        pruned = top_k_prune(fitted_classifier, top_k=2)
        assert pruned.tree_.node_count <= fitted_classifier.tree_.node_count

    def test_pruned_tree_still_predicts(self, iris, fitted_classifier):
        X, _ = iris
        pruned = top_k_prune(fitted_classifier, top_k=3)
        assert pruned.predict(X).shape == (len(X),)

    def test_original_tree_is_not_mutated(self, fitted_classifier):
        before = fitted_classifier.tree_.node_count
        top_k_prune(fitted_classifier, top_k=1)
        assert fitted_classifier.tree_.node_count == before

    def test_smaller_k_gives_a_smaller_tree(self, iris):
        X, y = iris
        dt = DecisionTreeClassifier(random_state=0).fit(X, y)
        assert top_k_prune(dt, top_k=1).tree_.node_count <= top_k_prune(dt, top_k=5).tree_.node_count

    def test_pruned_tree_is_still_walkable(self, fitted_classifier):
        """Guards the tree __setstate__ round-trip against node-dtype drift."""
        pruned = top_k_prune(fitted_classifier, top_k=3)
        _, _, branches = get_dt_info(pruned)
        assert len(branches) == pruned.get_n_leaves()
