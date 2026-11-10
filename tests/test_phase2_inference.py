"""Interval formulas, observable labels, isolated failures and restart integrity."""
import copy
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import digamma, expit

import phase2_inference as phase


def small_config():
    config = json.loads(phase.DEFAULT_CONFIG.read_text())
    config["replications_per_cell"] = 1
    config["sample_sizes"] = [12]
    config["scenarios"] = config["scenarios"][:1]
    config["intervals"].update(bootstrap_draws=4, minimum_valid_draws=2, minimum_valid_fraction=0.5)
    return config


def example_theta():
    alpha = np.array([0.8, 2.4, 1.4])
    scales = np.array([[0.65, 0.55], [0.9, 0.7]])
    means = np.array([[-2.0, -1.2], [2.0, 1.2]])
    locations = means - scales * (digamma(alpha[:-1]) - digamma(alpha[-1]))
    return np.r_[np.log(0.55 / 0.45), alpha, (1 / scales).ravel(), (-locations / scales).ravel()]


def diagnostic(z=None, se=0.2, regular=True, eligible=True):
    z = np.array([np.log(0.8), np.log(0.55 / 0.45), 4.0]) if z is None else np.asarray(z)
    return {"fit_reasons": [] if eligible else ["nonstationary"],
            "information_reasons": [] if regular else ["bound_hit"],
            "bootstrap_estimate_eligible": eligible, "regular_information_eligible": regular,
            "component_means": [[-2.0, -1.2], [2.0, 1.2]], "order": [0, 1], "ordering_gap": 4.0,
            "targets": {name: {"estimate": float(phase.inverse_target(j, z[j])),
                                "working_estimate": float(z[j]), "working_se": se,
                                "variance_valid": True} for j, name in enumerate(phase.TARGETS)}}


def test_task_mapping_and_random_streams_are_stable_and_separate():
    config = json.loads(phase.DEFAULT_CONFIG.read_text())
    assert phase.task_specification(config, 0)["scenario"] == "separated"
    assert phase.task_specification(config, 500)["n"] == 250
    assert phase.task_specification(config, 1500)["scenario"] == "overlap"
    assert phase.task_specification(config, 2999)["replicate"] == 499
    assert phase.task_specification(config, 1500)["separation_multiplier"] == 0.45
    assert phase.task_specification(config, 3000)["scenario"] == "severe_overlap"
    assert phase.task_specification(config, 4499)["replicate"] == 499
    with pytest.raises(ValueError):
        phase.task_specification(config, 4500)
    first = np.random.default_rng(phase.seed_sequence(config, 16, 2, 7)).normal(size=20)
    same = np.random.default_rng(phase.seed_sequence(config, 16, 2, 7)).normal(size=20)
    other = np.random.default_rng(phase.seed_sequence(config, 16, 2, 8)).normal(size=20)
    np.testing.assert_array_equal(first, same)
    assert not np.array_equal(first, other)


def test_ordered_targets_are_label_invariant_and_gradients_match_differences():
    theta = example_theta()
    swapped = theta.copy()
    swapped[0] = -theta[0]
    swapped[4:8] = theta[4:8].reshape(2, 2)[::-1].ravel()
    swapped[8:] = theta[8:].reshape(2, 2)[::-1].ravel()
    original = phase.target_values(theta)
    permuted = phase.target_values(swapped)
    np.testing.assert_allclose(original["values"], [0.8, 0.55, 4.0], atol=1e-13)
    np.testing.assert_allclose(original["values"], permuted["values"], atol=1e-13)
    assert original["order"] == [0, 1] and permuted["order"] == [1, 0]
    numeric = []
    for j in range(len(theta)):
        h = 1e-6 * max(1.0, abs(theta[j]))
        left, right = theta.copy(), theta.copy()
        left[j] -= h
        right[j] += h
        numeric.append((phase.target_values(right)["working_values"]
                        - phase.target_values(left)["working_values"]) / (2 * h))
    np.testing.assert_allclose(original["working_gradients"], np.asarray(numeric).T, rtol=1e-6, atol=2e-8)


def test_percentile_basic_and_studentized_formulas_on_working_scale():
    config = small_config()
    outer = diagnostic()
    center = np.array([outer["targets"][name]["working_estimate"] for name in phase.TARGETS])
    offsets = np.array([-0.5, -0.1, 0.2, 0.8])
    draws = [{"diagnostic": diagnostic(center + offset)} for offset in offsets]
    truths = {name: outer["targets"][name]["estimate"] for name in phase.TARGETS}
    rows = phase.build_intervals(outer, draws, truths, config)
    indexed = {(r["method"], r["target"]): r for r in rows}
    low, high = np.quantile(offsets, [0.025, 0.975])
    for j, name in enumerate(phase.TARGETS):
        percentile = indexed[("bootstrap_percentile", name)]
        basic = indexed[("bootstrap_basic", name)]
        studentized = indexed[("bootstrap_t", name)]
        np.testing.assert_allclose([percentile["lower"], percentile["upper"]],
                                   phase.inverse_target(j, center[j] + np.array([low, high])))
        np.testing.assert_allclose([basic["lower"], basic["upper"]],
                                   phase.inverse_target(j, center[j] - np.array([high, low])))
        np.testing.assert_allclose([studentized["lower"], studentized["upper"]],
                                   [basic["lower"], basic["upper"]])
        assert percentile["full_draw_availability"] and studentized["available"]


def test_boundaries_failures_and_availability_denominators_are_not_hidden():
    config = small_config()
    config["intervals"]["minimum_valid_fraction"] = 0.75
    outer = diagnostic()
    truths = {name: outer["targets"][name]["estimate"] for name in phase.TARGETS}
    draws = [{"diagnostic": diagnostic(regular=False)}, {"diagnostic": diagnostic()},
             {"diagnostic": diagnostic()}, {"diagnostic": diagnostic(eligible=False)}]
    rows = phase.build_intervals(outer, draws, truths, config)
    for row in rows:
        if row["method"] in ("bootstrap_percentile", "bootstrap_basic"):
            assert row["available"] and row["bootstrap_valid"] == 3
            assert not row["full_draw_availability"]
        if row["method"] == "bootstrap_t":
            assert not row["available"] and row["bootstrap_valid"] == 2
            assert row["exclusion_reasons"] == ["too_few_valid_bootstrap_draws"]
    outer_boundary = diagnostic(regular=False)
    rows = phase.build_intervals(outer_boundary, draws, truths, config)
    assert all(not row["available"] for row in rows if row["method"] in ("wald", "bootstrap_t"))
    assert all(row["available"] for row in rows if row["method"] == "bootstrap_percentile")


def test_checkpoint_recovers_only_partial_final_line(tmp_path):
    path = tmp_path / "checkpoint.jsonl"
    phase.append_checkpoint(path, {"bootstrap_index": 0, "value": 1.0})
    good = path.read_bytes()
    with path.open("ab") as handle:
        handle.write(b'{"bootstrap_index":1,"value":')
    assert len(phase.read_checkpoint(path)) == 1
    assert path.read_bytes() == good
    with path.open("ab") as handle:
        handle.write(b'broken complete line\n')
    with pytest.raises(json.JSONDecodeError):
        phase.read_checkpoint(path)


def test_failed_continuation_retains_earlier_finite_fit(monkeypatch):
    config = small_config()
    theta = example_theta()
    calls = []
    monkeypatch.setattr(phase.ld, "make_starts", lambda *args: [theta.copy() for _ in range(3)])
    def fitter(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise ArithmeticError("injected continuation failure")
        records = [{"score_converged": False, "mean_train_log_density": -3.0,
                    "failed": False, "selected": False, "min_weight": 0.4,
                    "boundary_hits": [], "projected_gradient": 0.01} for _ in range(3)]
        return SimpleNamespace(theta=theta, records=records, selected_start=0)
    monkeypatch.setattr(phase.ld, "fit", fitter)
    result = phase.fit_model(np.zeros((12, 2)), 17, config)
    assert result["finite_fit"] and len(result["starts"]) == 4
    assert result["starts"][-1]["failed"]
    assert "injected continuation failure" in result["starts"][-1]["message"]
    np.testing.assert_array_equal(result["theta"], theta)


def test_task_resume_compression_and_source_integrity(tmp_path, monkeypatch):
    config = small_config()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    theta = example_theta()
    calls = []
    def fitter(y, seed, config, warm_start=None):
        calls.append(seed)
        return {"finite_fit": True, "theta": theta, "selected_start": 0,
                "record": {"score_converged": True, "boundary_hits": [], "min_weight": 0.45},
                "starts": [], "elapsed_seconds": 0.01, "exception": ""}
    monkeypatch.setattr(phase, "fit_model", fitter)
    monkeypatch.setattr(phase, "diagnose_fit", lambda *args: diagnostic())
    output = tmp_path / "study"
    stopped = phase.run_task(config, config_path, 0, output, stop_after_bootstrap=2)
    assert stopped["status"] == "checkpointed" and len(calls) == 3
    result = phase.run_task(config, config_path, 0, output)
    assert result["status"] == "complete" and result["bootstrap_attempted"] == 4
    assert len(calls) == 5  # Outer plus exactly four draws, including after resume.
    directory = output / "task_000000"
    assert (directory / "bootstrap_records.jsonl.gz").exists()
    assert not (directory / "bootstrap_checkpoint.jsonl").exists()
    phase.run_task(config, config_path, 0, output)
    assert len(calls) == 5
    summary = phase.summarize_results(output, config)
    assert summary["completed_tasks"] == 1 and not summary["missing_task_indices"]
    assert all(row["attempted_completed"] == 1 for row in summary["coverage"])
    snapshot = directory / "sources" / "shared_shape_ld.py"
    snapshot.write_bytes(snapshot.read_bytes() + b"\n# changed\n")
    with pytest.raises(RuntimeError, match="snapshot changed"):
        phase.run_task(config, config_path, 0, output)
    with pytest.raises(RuntimeError, match="snapshot changed"):
        phase.summarize_results(output, config)


def test_outer_failure_keeps_all_methods_and_skips_bootstrap(tmp_path, monkeypatch):
    config = small_config()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    failure = {"finite_fit": False, "theta": None, "selected_start": None,
               "record": {}, "starts": [], "elapsed_seconds": 0.02, "exception": "injected failure"}
    monkeypatch.setattr(phase, "fit_model", lambda *args, **kwargs: failure)
    result = phase.run_task(config, config_path, 0, tmp_path / "failed")
    assert result["bootstrap_attempted"] == 0
    assert len(result["intervals"]) == 12
    assert all(not row["available"] and row["exclusion_reasons"] == ["fit_exception"] for row in result["intervals"])
    summary = phase.summarize_results(tmp_path / "failed", config)
    assert all(row["available"] == 0 and row["covered_and_available_fraction"] == 0 for row in summary["coverage"])


def test_changed_configuration_cannot_resume_same_task(tmp_path, monkeypatch):
    config = small_config()
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    failure = {"finite_fit": False, "theta": None, "selected_start": None,
               "record": {}, "starts": [], "elapsed_seconds": 0.02, "exception": "injected failure"}
    monkeypatch.setattr(phase, "fit_model", lambda *args, **kwargs: failure)
    phase.run_task(config, path, 0, tmp_path / "study")
    changed = copy.deepcopy(config)
    changed["master_seed"] += 1
    path.write_text(json.dumps(changed))
    with pytest.raises(RuntimeError, match="manifest mismatch"):
        phase.run_task(changed, path, 0, tmp_path / "study")


def test_half_open_task_range_batches_without_reimporting(tmp_path, monkeypatch, capsys):
    config = small_config()
    config["sample_sizes"] = [12, 15]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    calls = []
    def run(config, config_path, index, output, stop):
        calls.append(index)
        return {"status": "complete", "bootstrap_attempted": 4}
    monkeypatch.setattr(phase, "run_task", run)
    monkeypatch.setattr(sys, "argv", ["phase2_inference", "--config", str(path),
                                     "--output", str(tmp_path / "output"), "--task-range", "0:2"])
    phase.main()
    assert calls == [0, 1]
    assert len(capsys.readouterr().out.splitlines()) == 2


@pytest.mark.parametrize("corruption", ["duplicate_backup", "wrong_task_metadata", "out_of_range"])
def test_aggregation_rejects_copied_or_misidentified_tasks(tmp_path, monkeypatch, corruption):
    config = small_config()
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    failure = {"finite_fit": False, "theta": None, "selected_start": None,
               "record": {}, "starts": [], "elapsed_seconds": 0.02, "exception": "injected failure"}
    monkeypatch.setattr(phase, "fit_model", lambda *args, **kwargs: failure)
    output = tmp_path / "study"
    phase.run_task(config, path, 0, output)
    directory = output / "task_000000"
    if corruption == "duplicate_backup":
        shutil.copytree(directory, output / "task_000000_backup")
    else:
        result_path = directory / "result.json"
        result = json.loads(result_path.read_text())
        if corruption == "wrong_task_metadata":
            result["task"]["n"] += 1
        else:
            result["task"]["task_index"] = 99
        result_path.write_text(json.dumps(result))
    with pytest.raises(RuntimeError):
        phase.summarize_results(output, config)
