"""Tests for the dataset helpers."""

import numpy as np
import pandas as pd
import pytest

from trustee.enums.feature_type import FeatureType
from trustee.utils.dataset import convert_to_df, convert_to_series, read

METADATA = {
    "has_header": True,
    "fields": [
        ("id", FeatureType.IDENTIFIER, None, False),
        ("age", FeatureType.NUMERICAL, None, False),
        ("city", FeatureType.CATEGORICAL, None, False),
        ("score", FeatureType.NUMERICAL, None, False),
        ("label", FeatureType.NUMERICAL, None, True),
    ],
}

CSV = """id,age,city,score,label
1,25,NY,3.5,0
2,31,SF,4.5,1
3,,NY,,0
4,45,LA,2.5,1
"""


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text(CSV)
    return str(path)


class TestConvertToDf:
    def test_dataframe_passes_through(self):
        df = pd.DataFrame({"a": [1, 2]})
        assert convert_to_df(df) is df

    def test_ndarray_becomes_a_dataframe(self):
        assert convert_to_df(np.arange(6).reshape(3, 2)).shape == (3, 2)

    def test_series_becomes_a_dataframe(self):
        assert isinstance(convert_to_df(pd.Series([1, 2, 3])), pd.DataFrame)

    def test_unsupported_type_is_rejected(self):
        with pytest.raises(ValueError):
            convert_to_df([1, 2, 3])


class TestConvertToSeries:
    def test_series_passes_through(self):
        s = pd.Series([1, 2])
        assert convert_to_series(s) is s

    def test_ndarray_is_flattened(self):
        assert len(convert_to_series(np.arange(6).reshape(3, 2))) == 6

    def test_dataframe_yields_its_first_column(self):
        out = convert_to_series(pd.DataFrame({"a": [1, 2], "b": [3, 4]}))
        assert out.tolist() == [1, 2]

    def test_unsupported_type_is_rejected(self):
        with pytest.raises(ValueError):
            convert_to_series("not data")


class TestRead:
    def test_identifier_column_is_dropped(self, csv_path):
        _, _, columns, _, _ = read(csv_path, metadata=METADATA, as_df=True)
        assert "id" not in list(columns)

    def test_categorical_column_is_one_hot_encoded(self, csv_path):
        _, _, columns, _, _ = read(csv_path, metadata=METADATA, as_df=True)
        assert {"city_NY", "city_SF", "city_LA"}.issubset(set(columns))

    def test_target_column_is_separated(self, csv_path):
        _, y, columns, _, _ = read(csv_path, metadata=METADATA, as_df=True)
        assert "label" not in list(columns)
        assert np.asarray(y).ravel().tolist() == [0, 1, 0, 1]

    def test_numpy_output_is_numeric(self, csv_path):
        """pandas >= 2 returns bool dummies, which would force `object` dtype here."""
        X, _, _, _, _ = read(csv_path, metadata=METADATA)
        assert np.issubdtype(X.dtype, np.number)

    def test_missing_values_are_filled(self, csv_path):
        X, _, _, _, _ = read(csv_path, metadata=METADATA)
        assert not np.isnan(X).any()
        assert (X == -1).any()

    def test_reports_numerical_and_categorical_indices(self, csv_path):
        _, _, _, numerical, categorical = read(csv_path, metadata=METADATA, as_df=True)
        assert numerical == [0, 2]
        assert categorical == [[2, 3, 4]]

    def test_output_feeds_a_sklearn_estimator(self, csv_path):
        from sklearn.tree import DecisionTreeClassifier

        X, y, *_ = read(csv_path, metadata=METADATA)
        DecisionTreeClassifier().fit(X, np.asarray(y).ravel())

    def test_last_column_is_the_target_by_default(self, csv_path):
        metadata = {
            "has_header": True,
            "fields": [(name, FeatureType.NUMERICAL, None, False) for name in ("id", "age", "score", "label")],
        }
        path = csv_path.replace("data.csv", "plain.csv")
        with open(path, "w") as handle:
            handle.write("id,age,score,label\n1,25,3.5,0\n2,31,4.5,1\n")
        _, y, columns, _, _ = read(path, metadata=metadata, as_df=True)
        assert "label" not in list(columns)
        assert np.asarray(y).ravel().tolist() == [0, 1]


class TestReadVerbose:
    def test_verbose_prints_a_summary(self, csv_path, capsys):
        read(csv_path, metadata=METADATA, verbose=True, as_df=True)
        out = capsys.readouterr().out
        assert "Metadata start." in out
        assert "Pandas read_csv complete." in out
        assert "Total memory usage" in out

    def test_verbose_routes_through_a_logger(self, csv_path, tmp_path):
        from trustee.utils.log import Logger

        log_file = tmp_path / "read.log"
        read(csv_path, metadata=METADATA, verbose=True, logger=Logger(path=str(log_file)), as_df=True)
        assert "Metadata start." in log_file.read_text()


class TestReadCategories:
    def test_applies_an_ordered_categorical_dtype(self, tmp_path):
        path = tmp_path / "sized.csv"
        path.write_text("size,score\nsmall,1\nlarge,3\nmedium,2\n")
        metadata = {
            "has_header": True,
            "fields": [
                ("size", FeatureType.NUMERICAL, None, False),
                ("score", FeatureType.NUMERICAL, None, True),
            ],
            "categories": {"size": ["small", "medium", "large"]},
        }
        X, _, _, _, _ = read(str(path), metadata=metadata, as_df=True)
        assert str(X["size"].dtype) == "category"
        assert X["size"].cat.ordered
        assert list(X["size"].cat.categories) == ["small", "medium", "large"]


class TestReadDelimiter:
    def test_honours_a_custom_delimiter(self, tmp_path):
        path = tmp_path / "semi.csv"
        path.write_text("a;b\n1;2\n3;4\n")
        metadata = {
            "has_header": True,
            "delimiter": ";",
            "fields": [
                ("a", FeatureType.NUMERICAL, None, False),
                ("b", FeatureType.NUMERICAL, None, True),
            ],
        }
        X, y, _, _, _ = read(str(path), metadata=metadata, as_df=True)
        assert list(X.columns) == ["a"]
        assert np.asarray(y).ravel().tolist() == [2, 4]


class TestReadDirectory:
    def test_concatenates_every_csv_in_a_directory(self, tmp_path):
        data_dir = tmp_path / "parts"
        data_dir.mkdir()
        (data_dir / "part1.csv").write_text("a,b\n1,2\n3,4\n")
        (data_dir / "part2.csv").write_text("a,b\n5,6\n")
        metadata = {
            "has_header": True,
            "is_dir": True,
            "fields": [
                ("a", FeatureType.NUMERICAL, None, False),
                ("b", FeatureType.NUMERICAL, None, True),
            ],
        }
        X, y, _, _, _ = read(str(data_dir), metadata=metadata, as_df=True)
        assert len(X) == 3
        assert sorted(np.asarray(y).ravel().tolist()) == [2, 4, 6]


class TestReadConverters:
    def test_applies_a_converter_per_column(self, tmp_path):
        path = tmp_path / "conv.csv"
        path.write_text("label,score\nBENIGN,1\nBot,2\n")
        metadata = {
            "has_header": True,
            "fields": [
                ("label", FeatureType.NUMERICAL, None, False),
                ("score", FeatureType.NUMERICAL, None, True),
            ],
            "converters": {"label": lambda v: {"BENIGN": 0, "Bot": 1}.get(v.strip(), -1)},
        }
        X, _, _, _, _ = read(str(path), metadata=metadata, as_df=True)
        assert X["label"].tolist() == [0, 1]
