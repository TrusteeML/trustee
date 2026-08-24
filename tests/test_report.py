"""Tests for TrustReport.

The percentage assertions here guard the scikit-learn `tree_.value`
normalization change, which corrupted the report's numbers without raising:
class shares rendered as 200% and 1400%, and "Data Split %" was constant.
"""

import re
import shutil

import pytest

from trustee.report.trust import TrustReport

REPORT_KWARGS = dict(
    max_iter=2,
    num_pruning_iter=2,
    train_size=0.7,
    trustee_num_iter=3,
    trustee_num_stability_iter=2,
    trustee_sample_size=0.3,
    top_k=5,
)

has_graphviz = pytest.mark.skipif(shutil.which("dot") is None, reason="graphviz `dot` binary not installed")


@pytest.fixture(scope="module")
def report(iris_frame, iris_bunch):
    X, y = iris_frame
    from sklearn.ensemble import RandomForestClassifier

    return TrustReport(
        RandomForestClassifier(n_estimators=10, random_state=0),
        X=X,
        y=y,
        class_names=iris_bunch.target_names,
        feature_names=iris_bunch.feature_names,
        is_classify=True,
        **REPORT_KWARGS,
    )


@pytest.fixture(scope="module")
def rendered(report):
    return str(report)


class TestConstruction:
    def test_accepts_x_and_y(self, report):
        """`X`/`y` is the documented entry point; it used to raise AttributeError."""
        assert report.max_dt is not None

    def test_accepts_a_prefitted_split(self, iris_split, iris_bunch):
        from sklearn.ensemble import RandomForestClassifier

        X_train, X_test, y_train, y_test = iris_split
        built = TrustReport(
            RandomForestClassifier(n_estimators=10, random_state=0),
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            class_names=iris_bunch.target_names,
            is_classify=True,
            **REPORT_KWARGS,
        )
        assert built.max_dt is not None

    def test_defaults_use_features_from_x(self, report):
        assert len(report.use_features) == 4


class TestRendering:
    def test_renders_a_non_empty_report(self, rendered):
        assert len(rendered) > 500

    def test_class_names_render_plainly(self, rendered):
        """A stray trailing comma used to render these as `(np.str_('setosa'),)`."""
        assert "np.str_" not in rendered
        assert not re.search(r"\(np\.\w+\(", rendered)
        assert any(name in rendered for name in ("setosa", "versicolor", "virginica"))

    def test_no_percentage_exceeds_one_hundred(self, rendered):
        """Class shares rendered as 200%/1400% before the value fix."""
        percentages = [float(m) for m in re.findall(r"(\d+\.\d\d)%", rendered)]
        assert percentages, "expected percentages in the report"
        assert max(percentages) <= 100.0

    def test_probabilities_are_not_all_zero(self, rendered):
        """Leaf P(x) was hard-zero because of an `.ndim > 1` test on a 1-D array."""
        probabilities = [float(m) for m in re.findall(r"\((\d+\.\d\d)%\)", rendered)]
        assert probabilities
        assert any(p > 0 for p in probabilities)

    def test_reports_the_dataset_shape(self, rendered):
        assert "# Input features:" in rendered
        assert "# Output classes:" in rendered


class TestPersistence:
    @has_graphviz
    def test_save_writes_the_expected_tree(self, report, tmp_path):
        report.save(str(tmp_path))
        assert (tmp_path / "report" / "trust_report.txt").is_file()
        assert (tmp_path / "report" / "trust_report.obj").is_file()
        assert list((tmp_path / "report" / "plots").glob("*.pdf"))

    @has_graphviz
    def test_round_trips_through_disk(self, report, tmp_path):
        """The blackbox is deliberately dropped, but every computed result must survive."""
        report.save(str(tmp_path))
        loaded = TrustReport.load(str(tmp_path / "report" / "trust_report.obj"))

        assert loaded.max_dt.tree_.node_count == report.max_dt.tree_.node_count
        assert loaded.max_dt.get_n_leaves() == report.max_dt.get_n_leaves()
        assert loaded.max_dt_y_pred.tolist() == report.max_dt_y_pred.tolist()
        assert [b["samples"] for b in loaded.max_dt_top_branches] == [b["samples"] for b in report.max_dt_top_branches]
        assert list(loaded.use_features) == list(report.use_features)
        assert str(loaded)  # still renders

    @has_graphviz
    def test_save_is_repeatable(self, report, tmp_path):
        """__getstate__ used to detach `trustee.expert` from the live object."""
        report.save(str(tmp_path / "first"))
        report.save(str(tmp_path / "second"))
        assert (tmp_path / "second" / "report" / "trust_report.obj").is_file()


class TestRegressionReport:
    def test_builds_for_a_regression_blackbox(self, diabetes):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.datasets import load_diabetes

        data = load_diabetes(as_frame=True)
        built = TrustReport(
            RandomForestRegressor(n_estimators=10, random_state=0),
            X=data.data,
            y=data.target,
            feature_names=data.feature_names,
            is_classify=False,
            **REPORT_KWARGS,
        )
        assert "# Input features:" in str(built)
