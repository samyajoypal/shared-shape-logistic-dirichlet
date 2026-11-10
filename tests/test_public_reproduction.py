"""The quick table path preserves scope and detects modified public inputs."""
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
import reproduce_saved_tables as tables


@pytest.fixture
def isolated_inputs(tmp_path, monkeypatch):
    root = tmp_path / "inputs"
    root.mkdir()
    shutil.copytree(tables.ROOT / "saved_summaries", root / "saved_summaries")
    (root / "configs").mkdir()
    shutil.copyfile(tables.ROOT / "configs/phase3_applications.json",
                    root / "configs/phase3_applications.json")
    shutil.copyfile(tables.ROOT / "export_manifest.json", root / "export_manifest.json")
    monkeypatch.setattr(tables, "ROOT", root)
    return root


def test_all_eight_main_tables_keep_declared_scope(isolated_inputs, tmp_path):
    output = tmp_path / "tables"
    receipt = tables.build(output)
    counts = [6, 30, 12, 8, 18, 9, 3, 15]
    outputs = sorted(output.glob("table_*.csv"))
    assert len(outputs) == 8
    assert [len(pd.read_csv(path)) for path in outputs] == counts
    assert receipt["new_fits"] is False
    assert receipt["data_downloads"] is False
    assert "configs/phase3_applications.json" in receipt["input_sha256"]
    for filename, expected in receipt["output_sha256"].items():
        assert tables.sha(output / filename) == expected


@pytest.mark.parametrize("relative", [
    "configs/phase3_applications.json",
    "saved_summaries/additional_mean/prediction_summary.csv",
])
def test_changed_scientific_input_is_rejected(isolated_inputs, tmp_path, relative):
    path = isolated_inputs / relative
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Changed summary"):
        tables.build(tmp_path / "tables")
    assert not list((tmp_path / "tables").glob("table_*.csv"))
