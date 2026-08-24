"""Structural validation of the bundled dataset metadata.

These dicts are consumed by `trustee.utils.dataset.read()`, which unpacks every
field as a 4-tuple and indexes by position. A malformed entry would surface as an
opaque unpacking error at read time, so validate the shape here instead.
"""

import pytest

from trustee.enums.feature_type import FeatureType
from trustee.utils import const

METADATA = {name: getattr(const, name) for name in dir(const) if name.endswith("_DATASET_META")}


def test_metadata_dicts_are_exposed():
    assert METADATA, "expected at least one *_DATASET_META constant"


@pytest.mark.parametrize("name", sorted(METADATA))
class TestDatasetMetadata:
    def test_has_the_keys_read_depends_on(self, name):
        meta = METADATA[name]
        assert "fields" in meta
        assert isinstance(meta["fields"], list) and meta["fields"]

    def test_every_field_is_a_four_tuple(self, name):
        for field in METADATA[name]["fields"]:
            assert len(field) == 4, f"{name}: {field!r} is not a 4-tuple"

    def test_field_components_have_the_expected_types(self, name):
        for field_name, feature_type, _dtype, is_result in METADATA[name]["fields"]:
            assert isinstance(field_name, str) and field_name
            assert isinstance(feature_type, FeatureType)
            assert isinstance(is_result, bool)

    def test_declares_exactly_one_target_column(self, name):
        results = [f for f in METADATA[name]["fields"] if f[3]]
        assert len(results) == 1, f"{name} declares {len(results)} target columns"

    def test_field_names_are_unique(self, name):
        names = [f[0] for f in METADATA[name]["fields"]]
        assert len(names) == len(set(names)), f"{name} has duplicate field names"

    def test_target_column_is_not_an_identifier(self, name):
        for field_name, feature_type, _dtype, is_result in METADATA[name]["fields"]:
            if is_result:
                assert feature_type is not FeatureType.IDENTIFIER

    def test_declared_type_is_a_known_string(self, name):
        meta = METADATA[name]
        if "type" in meta:
            assert meta["type"] in {"classification", "regression"}


class TestFeatureType:
    def test_members_are_distinct(self):
        values = [member.value for member in FeatureType]
        assert len(values) == len(set(values))

    def test_exposes_the_three_kinds_read_branches_on(self):
        assert {"CATEGORICAL", "NUMERICAL", "IDENTIFIER"} <= {m.name for m in FeatureType}


class TestCicIds2017LabelConverter:
    convert = staticmethod(const.cic_ids_2017_label_converter)

    def test_maps_benign_to_zero(self):
        assert self.convert("BENIGN") == 0

    @pytest.mark.parametrize(
        "label,expected",
        [("Bot", 1), ("DDoS", 2), ("Heartbleed", 8), ("PortScan", 10), ("Web Attack XSS", 14)],
    )
    def test_maps_known_attack_labels(self, label, expected):
        assert self.convert(label) == expected

    def test_strips_surrounding_whitespace(self):
        assert self.convert("  DDoS  ") == self.convert("DDoS")

    def test_returns_uint8(self):
        import numpy as np

        assert isinstance(self.convert("BENIGN"), np.uint8)

    def test_unknown_label_falls_back(self, capsys):
        # -1 cast to uint8 wraps to 255, which is the sentinel callers see.
        assert self.convert("Not A Real Label") == 255
        assert "Exception" in capsys.readouterr().out

    def test_non_string_input_falls_back(self, capsys):
        assert self.convert(None) == 255
        assert "Exception" in capsys.readouterr().out

    def test_every_mapped_value_fits_in_uint8(self):
        for label in ["BENIGN", "Bot", "DDoS", "Web Attack Sql Injection"]:
            assert 0 <= self.convert(label) <= 255
