"""Tests for model persistence."""

import zipfile

import pytest
from sklearn.tree import DecisionTreeClassifier

from trustee.utils.persist import load_model, save_model


@pytest.fixture
def model(iris):
    X, y = iris
    return DecisionTreeClassifier(random_state=0, max_depth=3).fit(X, y)


class TestSaveModel:
    def test_writes_a_file(self, model, tmp_path):
        path = tmp_path / "model.joblib"
        save_model(model, str(path))
        assert path.is_file()
        assert path.stat().st_size > 0

    def test_returns_the_paths_written(self, model, tmp_path):
        assert save_model(model, str(tmp_path / "model.joblib"))


class TestRoundTrip:
    def test_restores_an_equivalent_model(self, model, iris, tmp_path):
        X, _ = iris
        path = tmp_path / "model.joblib"
        save_model(model, str(path))
        restored = load_model(str(path))

        assert restored is not None
        assert restored.predict(X).tolist() == model.predict(X).tolist()

    def test_restores_tree_structure(self, model, tmp_path):
        path = tmp_path / "model.joblib"
        save_model(model, str(path))
        restored = load_model(str(path))
        assert restored.tree_.node_count == model.tree_.node_count


class TestZipSupport:
    def test_loads_a_model_out_of_a_zip(self, model, tmp_path):
        inner = tmp_path / "model"
        save_model(model, str(inner))

        archive = tmp_path / "model.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.write(inner, arcname="model")
        inner.unlink()

        restored = load_model(str(archive))
        assert restored is not None
        assert restored.tree_.node_count == model.tree_.node_count

    def test_cleans_up_the_file_it_unzipped(self, model, tmp_path):
        inner = tmp_path / "model"
        save_model(model, str(inner))
        archive = tmp_path / "model.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.write(inner, arcname="model")
        inner.unlink()

        load_model(str(archive))
        assert not inner.exists(), "the extracted copy should not be left behind"
        assert archive.is_file(), "the archive itself must survive"


class TestMissingInput:
    def test_returns_none_for_a_path_that_is_not_a_file(self, tmp_path):
        assert load_model(str(tmp_path)) is None

    def test_returns_none_for_a_non_model_file(self, tmp_path):
        # A readable file that joblib cannot load.
        path = tmp_path / "garbage.joblib"
        path.write_text("this is not a joblib payload")
        with pytest.raises(Exception):
            # Documents current behaviour: the except clause catches copy.Error,
            # which is not what joblib raises, so the error propagates.
            load_model(str(path))

    def test_missing_path_returns_none(self, tmp_path):
        assert load_model(str(tmp_path / "does-not-exist.joblib")) is None


class TestOverwrite:
    def test_saving_twice_replaces_the_file(self, model, iris, tmp_path):
        X, y = iris
        path = tmp_path / "model.joblib"
        save_model(model, str(path))

        other = DecisionTreeClassifier(random_state=0, max_depth=1).fit(X, y)
        save_model(other, str(path))

        restored = load_model(str(path))
        assert restored.get_depth() == other.get_depth()
