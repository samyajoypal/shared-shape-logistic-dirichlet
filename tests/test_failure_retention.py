"""Fault injection for failed starts and complete unsuccessful studies."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import run_inference_experiments as inference
import shared_shape_ld as shared


def optimizer_result(theta):
    return SimpleNamespace(x=np.asarray(theta).copy(), success=True, nit=1,
                           message="injected finite optimizer result")


@pytest.mark.parametrize("failed_start", [0, 1])
def test_failed_start_does_not_discard_other_candidates(monkeypatch, failed_start):
    y = np.array([[-1.0], [.2], [.8], [2.0]])
    starts = [np.array([1.2, 2.0, 1.1, b]) for b in [-.2, -.8, .4]]
    calls = []

    def controlled_minimize(objective, initial, **kwargs):
        index = len(calls)
        calls.append(index)
        if index == failed_start:
            raise ArithmeticError("injected start failure")
        return optimizer_result(initial)

    monkeypatch.setattr(shared, "minimize", controlled_minimize)
    fitted = shared.fit(y, 1, starts=starts)
    surviving = [index for index in range(len(starts)) if index != failed_start]
    expected = max(surviving, key=lambda index:
                   shared.evaluation(starts[index], y, 1)[0].mean())

    assert calls == [0, 1, 2]
    assert len(fitted.records) == 3
    assert fitted.records[failed_start]["failed"]
    assert "injected start failure" in fitted.records[failed_start]["message"]
    assert fitted.selected_start == expected
    assert fitted.record is fitted.records[expected]
    assert sum(record["selected"] for record in fitted.records) == 1
    np.testing.assert_array_equal(fitted.theta, starts[expected])


@pytest.mark.parametrize("nonfinite", ["parameters", "objective", "score"])
def test_nonfinite_candidate_cannot_win(monkeypatch, nonfinite):
    y = np.array([[-1.0], [.2], [.8], [2.0]])
    bad = np.array([1.2, 2.0, 1.1, 37.0])
    good = np.array([1.2, 2.0, 1.1, -.2])
    original_evaluation = shared.evaluation

    def controlled_minimize(objective, initial, **kwargs):
        result = optimizer_result(initial)
        if nonfinite == "parameters" and initial[-1] == 37.0:
            result.x[0] = np.nan
        return result

    def controlled_evaluation(theta, y, k, common_scale=False, gradient=False):
        answer = original_evaluation(theta, y, k, common_scale, gradient)
        if theta[-1] == 37.0 and gradient:
            ell, tau, score = answer
            if nonfinite == "objective":
                ell = np.full_like(ell, np.nan)
            if nonfinite == "score":
                score = np.full_like(score, np.nan)
            return ell, tau, score
        return answer

    monkeypatch.setattr(shared, "minimize", controlled_minimize)
    monkeypatch.setattr(shared, "evaluation", controlled_evaluation)
    fitted = shared.fit(y, 1, starts=[bad, good])

    assert fitted.records[0]["failed"]
    assert fitted.records[0]["exception_type"] == "FloatingPointError"
    assert not fitted.records[0]["selected"]
    assert fitted.selected_start == 1
    assert not fitted.record["failed"]
    np.testing.assert_array_equal(fitted.theta, good)


@pytest.mark.parametrize("failure_stage", ["optimizer", "before_starts"])
def test_all_failed_study_writes_complete_outputs(monkeypatch, tmp_path, failure_stage):
    theta = np.r_[0.0, [1.2, 2.0, 1.3], np.ones(4), np.zeros(4)]

    def controlled_starts(y, k, n_starts, seed, common_scale=False):
        return [theta.copy() for _ in range(n_starts)]

    def always_fails(*args, **kwargs):
        raise ArithmeticError("injected study failure")

    monkeypatch.setattr(shared, "make_starts", controlled_starts)
    monkeypatch.setattr(shared, "minimize", always_fails)
    if failure_stage == "before_starts":
        monkeypatch.setattr(inference, "fit", always_fails)
    args = SimpleNamespace(separation=1.0, sizes=[8, 12], reps=2, starts=3,
                           seed=9321, maxiter=2, gtol=1e-6)

    inference.run_pilot(args, tmp_path)

    fits = pd.read_csv(tmp_path / "selected_fits.csv")
    starts = pd.read_csv(tmp_path / "all_starts.csv")
    intervals = pd.read_csv(tmp_path / "intervals.csv")
    coverage = pd.read_csv(tmp_path / "coverage_summary.csv")
    diagnostics = pd.read_csv(tmp_path / "fit_diagnostics.csv")
    assert len(fits) == 4
    assert fits.fit_exception.str.contains("Failure|failure").all()
    assert not fits.finite_fit.any()
    assert not fits.intervals_available.any()
    assert len(intervals) == 12
    assert not intervals.interval_available.any()
    assert len(coverage) == 6
    assert (coverage.attempted == 2).all()
    assert (coverage.available == 0).all()
    assert (coverage.unavailable == 2).all()
    assert coverage.conditional_coverage.isna().all()
    assert (coverage.covered_and_available_fraction == 0).all()
    assert len(diagnostics) == 2
    assert (diagnostics.fit_exception == 2).all()
    assert (diagnostics.all_target_intervals_available == 0).all()
    if failure_stage == "optimizer":
        assert len(starts) == 12
        assert starts.failed.all()
        assert not starts.selected.any()
        assert starts.message.str.contains("injected study failure").all()
        assert (fits.starts_attempted == 3).all()
        assert (fits.starts_failed == 3).all()
        assert (fits.fit_seconds_all_starts >= 0).all()
    else:
        assert starts.empty
        assert (fits.starts_attempted == 0).all()
    with np.load(tmp_path / "fitted_parameters.npz") as parameters:
        assert not parameters.files
    assert (tmp_path / "specification.json").exists()
