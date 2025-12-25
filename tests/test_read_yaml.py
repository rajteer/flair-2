"""Tests for YAML utilities in src/utils/read_yaml.py."""

import tempfile
from pathlib import Path

import pytest

from src.utils.read_yaml import read_yaml, yaml_loader


class TestReadYaml:
    """Tests for read_yaml function."""

    def test_reads_valid_yaml_file(self) -> None:
        """Should correctly parse a valid YAML file."""
        yaml_content = """
model:
  type: Unet
  encoder: resnet18
training:
  epochs: 100
  batch_size: 16
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = Path(f.name)

        try:
            config = read_yaml(temp_path)

            assert config["model"]["type"] == "Unet"
            assert config["model"]["encoder"] == "resnet18"
            assert config["training"]["epochs"] == 100
            assert config["training"]["batch_size"] == 16
        finally:
            temp_path.unlink()

    def test_parses_scientific_notation_as_float(self) -> None:
        """Scientific notation should be parsed as float, not string."""
        yaml_content = """
learning_rate: 1e-4
weight_decay: 5e-5
small_value: 1.5e-10
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = Path(f.name)

        try:
            config = read_yaml(temp_path)

            assert isinstance(config["learning_rate"], float)
            assert config["learning_rate"] == pytest.approx(1e-4)
            assert isinstance(config["weight_decay"], float)
            assert config["weight_decay"] == pytest.approx(5e-5)
            assert isinstance(config["small_value"], float)
            assert config["small_value"] == pytest.approx(1.5e-10)
        finally:
            temp_path.unlink()

    def test_handles_nested_structures(self) -> None:
        """Should handle deeply nested YAML structures."""
        yaml_content = """
level1:
  level2:
    level3:
      value: 42
      list:
        - item1
        - item2
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = Path(f.name)

        try:
            config = read_yaml(temp_path)

            assert config["level1"]["level2"]["level3"]["value"] == 42
            assert config["level1"]["level2"]["level3"]["list"] == ["item1", "item2"]
        finally:
            temp_path.unlink()


class TestYamlLoader:
    """Tests for yaml_loader function."""

    def test_parses_scientific_notation_in_all_positions(self) -> None:
        """Scientific notation should work in various positions."""
        yaml_string = """
positive: 1e5
negative: -2.5e-3
in_list:
  - 1e-4
  - 2e-5
in_nested:
  deep:
    value: 3.14e2
"""
        config = yaml_loader(yaml_string)

        assert config["positive"] == pytest.approx(1e5)
        assert config["negative"] == pytest.approx(-2.5e-3)
        assert config["in_list"][0] == pytest.approx(1e-4)
        assert config["in_list"][1] == pytest.approx(2e-5)
        assert config["in_nested"]["deep"]["value"] == pytest.approx(3.14e2)

    def test_handles_string_that_looks_like_float(self) -> None:
        """Strings that don't match float pattern should remain strings."""
        yaml_string = """
string_value: "1e-4"
actual_float: 1e-4
"""
        config = yaml_loader(yaml_string)

        assert isinstance(config["string_value"], str)
        assert isinstance(config["actual_float"], float)

    def test_handles_special_float_values(self) -> None:
        """Should handle special float values like inf and nan."""
        yaml_string = """
infinity: .inf
neg_infinity: -.inf
not_a_number: .nan
"""
        config = yaml_loader(yaml_string)

        import math

        assert math.isinf(config["infinity"])
        assert config["infinity"] > 0
        assert math.isinf(config["neg_infinity"])
        assert config["neg_infinity"] < 0
        assert math.isnan(config["not_a_number"])


class TestJoinConstructor:
    """Tests for custom !join YAML constructor."""

    def test_joins_path_parts(self) -> None:
        """Should join path parts with /."""
        yaml_string = """
path: !join [data, train, images]
"""
        config = yaml_loader(yaml_string)

        assert config["path"] == "data/train/images"

    def test_joins_single_part(self) -> None:
        """Should handle single path part."""
        yaml_string = """
path: !join [single]
"""
        config = yaml_loader(yaml_string)

        assert config["path"] == "single"

    def test_joins_empty_parts(self) -> None:
        """Should handle empty list."""
        yaml_string = """
path: !join []
"""
        config = yaml_loader(yaml_string)

        assert config["path"] == ""

    def test_joins_with_variables(self) -> None:
        """Should work with YAML anchors and aliases."""
        yaml_string = """
base: &base data
full_path: !join [*base, train, images]
"""
        config = yaml_loader(yaml_string)

        assert config["full_path"] == "data/train/images"
