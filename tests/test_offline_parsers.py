"""Tests for generic offline JSON and XML parsers."""

import json
from textwrap import dedent

import pandas as pd
import pytest

from backend.ingestion.errors import IngestionError
from backend.ingestion.json_parser import parse_json
from backend.ingestion.xml_parser import parse_xml


def test_parse_json_valid_list_of_dicts(tmp_path):
    """List of dictionaries should parse directly to DataFrame."""
    path = tmp_path / "data.json"
    data = [{"id": 1, "value": "A"}, {"id": 2, "value": "B"}]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    result = parse_json(path)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert list(result.columns) == ["id", "value"]


def test_parse_json_valid_dict(tmp_path):
    """Generic dictionary should fall back to returning a dict."""
    path = tmp_path / "data.json"
    data = {"metadata": {"source": "test"}, "records": [1, 2, 3]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    result = parse_json(path)
    assert isinstance(result, dict)
    assert result["metadata"]["source"] == "test"


def test_parse_json_file_not_found(tmp_path):
    """Missing file should raise IngestionError."""
    with pytest.raises(IngestionError, match="JSON file not found"):
        parse_json(tmp_path / "missing.json")


def test_parse_json_invalid(tmp_path):
    """Invalid JSON should raise IngestionError."""
    path = tmp_path / "data.json"
    path.write_text("{invalid json}")

    with pytest.raises(IngestionError, match="Failed to parse JSON file"):
        parse_json(path)


def test_parse_xml_valid(tmp_path):
    """Valid XML should parse to DataFrame."""
    path = tmp_path / "data.xml"
    xml_content = dedent("""\
        <?xml version="1.0" encoding="UTF-8"?>
        <data>
            <record>
                <id>1</id>
                <value>A</value>
            </record>
            <record>
                <id>2</id>
                <value>B</value>
            </record>
        </data>
    """)
    path.write_text(xml_content)

    result = parse_xml(path)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert "id" in result.columns
    assert "value" in result.columns


def test_parse_xml_file_not_found(tmp_path):
    """Missing file should raise IngestionError."""
    with pytest.raises(IngestionError, match="XML file not found"):
        parse_xml(tmp_path / "missing.xml")


def test_parse_xml_invalid(tmp_path):
    """Invalid XML should raise IngestionError."""
    path = tmp_path / "data.xml"
    path.write_text("<data><record>unclosed tag</data>")

    with pytest.raises(IngestionError, match="Failed to parse XML file"):
        parse_xml(path)
