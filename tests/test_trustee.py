"""Tests for the Trustee explainers."""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from trustee import ClassificationTrustee, RegressionTrustee

# Keep the loops small; these tests check contracts, not explanation quality.
FIT_KWARGS = dict(num_iter=3, num_stability_iter=2, samples_size=0.3)


@pytest.fixture(scope="module")
def trained(iris_split, blackbox):
    X_train, _, y_train, _ = iris_split
    trustee = ClassificationTrustee(expert=blackbox)
    trustee.fit(X_train, y_train, **FIT_KWARGS)
    return trustee


class TestFitValidation:
    def test_mismatched_lengths_are_rejected(self, blackbox):
        trustee = ClassificationTrustee(expert=blackbox)
        with pytest.raises(ValueError, match="same length"):
            trustee.fit(np.zeros((10, 4)), np.zeros(9), **FIT_KWARGS)

    def test_accessors_require_a_fitted_explainer(self, blackbox):
        trustee = ClassificationTrustee(expert=blackbox)
        with pytest.raises(ValueError, match="No student models"):
            trustee.explain()


class TestClassificationTrustee:
    def test_explain_returns_the_documented_tuple(self, trained):
        dt, pruned_dt, agreement, reward = trained.explain()
        assert isinstance(dt, DecisionTreeClassifier)
        assert isinstance(pruned_dt, DecisionTreeClassifier)
        assert 0 <= agreement <= 1
        assert 0 <= reward <= 1

    def test_pruning_does_not_grow_the_explanation(self, trained):
        dt, pruned_dt, _, _ = trained.explain()
        assert pruned_dt.tree_.node_count <= dt.tree_.node_count

    def test_explanation_predicts_the_expected_shape(self, trained, iris_split):
        _, X_test, _, _ = iris_split
        dt, _, _, _ = trained.explain()
        assert dt.predict(X_test).shape == (len(X_test),)

    def test_explanation_broadly_agrees_with_the_blackbox(self, trained, iris_split, blackbox):
        _, X_test, _, _ = iris_split
        dt, _, _, _ = trained.explain()
        fidelity = (dt.predict(X_test) == blackbox.predict(X_test)).mean()
        assert fidelity > 0.7

    def test_reports_the_number_of_classes(self, trained):
        assert trained.get_n_classes() == 3

    def test_reports_the_features_it_used(self, trained):
        assert 0 < trained.get_n_features() <= 4

    def test_top_branches_are_ordered_by_coverage(self, trained):
        samples = [b["samples"] for b in trained.get_top_branches(top_k=5)]
        assert samples == sorted(samples, reverse=True)

    def test_top_features_are_ordered_by_coverage(self, trained):
        samples = [values["samples"] for _, values in trained.get_top_features(top_k=5)]
        assert samples == sorted(samples, reverse=True)

    def test_top_k_bounds_the_result(self, trained):
        assert len(trained.get_top_branches(top_k=2)) <= 2
        assert len(trained.get_top_nodes(top_k=2)) <= 2

    def test_samples_by_level_covers_every_level(self, trained):
        dt, _, _, _ = trained.explain()
        assert len(trained.get_samples_by_level()) == dt.get_depth() + 1

    def test_leaves_by_level_totals_the_leaf_count(self, trained):
        dt, _, _, _ = trained.explain()
        assert sum(trained.get_leaves_by_level()) == dt.get_n_leaves()

    def test_all_students_matches_the_loop_shape(self, trained):
        students = trained.get_all_students()
        assert len(students) == FIT_KWARGS["num_stability_iter"]
        assert all(len(row) == FIT_KWARGS["num_iter"] for row in students)

    def test_one_top_student_per_outer_iteration(self, trained):
        assert len(trained.get_top_students()) == FIT_KWARGS["num_stability_iter"]

    def test_get_stable_filters_on_the_threshold(self, trained):
        assert list(trained.get_stable(threshold=1.1)) == []
        assert len(list(trained.get_stable(threshold=0))) == FIT_KWARGS["num_stability_iter"]

    def test_prune_returns_a_usable_tree(self, trained, iris_split):
        _, X_test, _, _ = iris_split
        pruned = trained.prune(top_k=3)
        assert pruned.predict(X_test).shape == (len(X_test),)

    def test_use_features_restricts_the_student(self, iris_split, blackbox):
        X_train, _, y_train, _ = iris_split
        trustee = ClassificationTrustee(expert=blackbox)
        trustee.fit(X_train, y_train, use_features=[0, 1], **FIT_KWARGS)
        dt, _, _, _ = trustee.explain()
        assert dt.n_features_in_ == 2


@pytest.fixture(scope="module")
def trained_regressor(diabetes):
    X, y = diabetes
    expert = RandomForestRegressor(n_estimators=10, random_state=0).fit(X, y)
    trustee = RegressionTrustee(expert=expert)
    trustee.fit(X, y, **FIT_KWARGS)
    return trustee


class TestRegressionTrustee:
    def test_explain_returns_regression_trees(self, trained_regressor):
        dt, pruned_dt, _, reward = trained_regressor.explain()
        assert isinstance(dt, DecisionTreeRegressor)
        assert isinstance(pruned_dt, DecisionTreeRegressor)
        assert reward <= 1

    def test_explanation_predicts_continuous_values(self, trained_regressor, diabetes):
        X, _ = diabetes
        dt, _, _, _ = trained_regressor.explain()
        assert np.issubdtype(dt.predict(X).dtype, np.floating)


class TestDataFrameInput:
    def test_accepts_pandas_input(self, iris_frame):
        X, y = iris_frame
        expert = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
        trustee = ClassificationTrustee(expert=expert)
        trustee.fit(X, y, **FIT_KWARGS)
        dt, _, _, _ = trustee.explain()
        assert dt.tree_.node_count > 0
