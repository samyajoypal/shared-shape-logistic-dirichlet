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
    for name in ["phase3_applications.json", "phase2_applications.json",
                 "phase2_simulations.json", "phase2_gaussian_control.json"]:
        shutil.copyfile(tables.ROOT / "configs" / name, root / "configs" / name)
    shutil.copyfile(tables.ROOT / "export_manifest.json", root / "export_manifest.json")
    monkeypatch.setattr(tables, "ROOT", root)
    return root


def test_all_eight_extended_tables_keep_declared_scope(isolated_inputs, tmp_path):
    output = tmp_path / "tables"
    receipt = tables.build(output)
    counts = [6, 30, 12, 8, 18, 9, 3, 15]
    outputs = sorted(output.glob("table_*.csv"))
    assert len(outputs) == 8
    assert [len(pd.read_csv(path)) for path in outputs] == counts
    assert receipt["new_fits"] is False
    assert receipt["data_downloads"] is False
    assert "configs/phase3_applications.json" in receipt["input_sha256"]
    assert "configs/phase2_applications.json" in receipt["input_sha256"]
    assert len(receipt["output_sha256"]) == 23
    for filename, expected in receipt["output_sha256"].items():
        assert tables.sha(output / filename) == expected


def test_focused_applications_keep_all_four_predictors(isolated_inputs, tmp_path):
    output = tmp_path / "tables"
    receipt = tables.build(output)
    focused = pd.read_csv(output / "focused_table_02_applications.csv")
    assert focused.dataset_id.tolist() == ["olive6_oleic", "abalone_log4"]
    assert focused.label.tolist() == ["Six-acid olive oil", "Abalone"]
    assert focused.parameters_shared.tolist() == [38, 31]
    assert focused.parameters_separate.tolist() == [50, 41]
    original = pd.read_csv(isolated_inputs / "saved_summaries/original_mean/prediction_summary.csv")
    additional = pd.read_csv(isolated_inputs / "saved_summaries/additional_mean/prediction_summary.csv")
    for method in tables.METHODS:
        oil = original[(original.dataset_id == "olive6_oleic") & (original.method == method)
                       & (original.version == "refined") & (original.comparison == "primary")
                       & (original.kind == "cv") & (original.weighting == "observation")]
        abalone = additional[(additional.dataset_id == "abalone_log4") & (additional.method == method)
                             & (additional.version == "phase3") & (additional.comparison == "primary")
                             & (additional.kind == "cv") & (additional.weighting == "observation")]
        assert len(oil) == len(abalone) == 1
        assert focused[method + "_mean_prediction_loss"].tolist() == [oil.iloc[0].loss, abalone.iloc[0].loss]
    assert focused.variance_ratio.tolist() == pytest.approx([0.440392, 0.208646], abs=5e-7)
    assert receipt["focused_display_files"] == ["table_01_core_simulations.csv", "focused_table_02_applications.csv"]
    assert len(receipt["extended_comparison_files"]) == 7


@pytest.mark.parametrize("relative", [
    "configs/phase2_applications.json",
    "configs/phase3_applications.json",
    "saved_summaries/additional_mean/prediction_summary.csv",
])
def test_changed_scientific_input_is_rejected(isolated_inputs, tmp_path, relative):
    path = isolated_inputs / relative
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Changed summary"):
        tables.build(tmp_path / "tables")
    assert not list((tmp_path / "tables").glob("table_*.csv"))


@pytest.mark.parametrize("variant", ["olive6_oleic", "abalone_log4"])
def test_focused_table_rejects_incomplete_bootstrap_pairs(isolated_inputs, tmp_path, variant):
    relative = ("saved_summaries/original_stability/shared_variance_ratios.csv" if variant == "olive6_oleic"
                else "saved_summaries/additional_stability/paired_stability.csv")
    path = isolated_inputs / relative
    frame = pd.read_csv(path)
    key = "variant" if variant == "olive6_oleic" else "dataset_id"
    selected = (frame[key] == variant) & (frame.left_method == "shared_ld") & (frame.right_method == "full_ld")
    frame.loc[selected, "both_available"] = 198
    frame.to_csv(path, index=False)
    # A complete, freshly hashed input may still contain unavailable outcomes.
    manifest_path = isolated_inputs / "export_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest["entries"]:
        if entry["file"] == relative:
            entry["export_sha256"] = tables.sha(path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Incomplete focused .* bootstrap pairs"):
        tables.build(tmp_path / "tables")
