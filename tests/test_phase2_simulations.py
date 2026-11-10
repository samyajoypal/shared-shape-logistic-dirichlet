"""Scientific invariants, coordinate measures, and failed-shard accounting."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.special import digamma
from threadpoolctl import threadpool_limits

import phase2_simulations as study
import run_confirmatory_experiments as fitters


@pytest.fixture
def config():
    return json.loads(study.CONFIG.read_text())


def tiny_config(config, methods=None):
    config = copy.deepcopy(config)
    cell = dict(config["cells"][0], id="fault_injection", n_train=30, repetitions=2)
    config.update(cells=[cell], cell_count=1, dataset_count=2, n_test=40)
    config["primary_methods"] = methods or ["shared_ld", "full_ld"]
    config["secondary"] = dict.fromkeys(config["secondary"], 0)
    return config


def test_complete_design_and_deterministic_subsets(config):
    study.validate_config(config)
    plan = list(study.tasks(config))
    assert len(plan) == 10500
    assert [task["task_index"] for task in plan] == list(range(10500))
    assert sum(len(study.schedule(config, task["replicate"])) for task in plan) == 91700
    assert len({tuple(study.seed_words(config, t["cell_id"], t["replicate"], "train")) for t in plan}) == 10500
    local_cells = [cell for cell in config["cells"] if cell["block"] == "local_alternative"]
    assert len(local_cells) == 6
    for cell in local_cells:
        assert cell["heterogeneity"]*np.sqrt(cell["n_train"]) == pytest.approx(cell["local_h_coefficient"])
    assert ("primary", "bounded_skew_normal") in study.schedule(config, 99)
    assert ("primary", "bounded_skew_normal") not in study.schedule(config, 100)
    assert any(v == "ten_starts" for v, _ in study.schedule(config, 19))
    assert not any(v == "ten_starts" for v, _ in study.schedule(config, 20))


@pytest.mark.parametrize("weights", ["balanced", "imbalanced"])
@pytest.mark.parametrize("heterogeneity", [0., .2, .5, 1.])
def test_shape_heterogeneity_is_exact(config, weights, heterogeneity):
    cell = dict(config["cells"][0], dimension=6, components=3,
                weight_profile=weights, heterogeneity=heterogeneity)
    spec = study.generating_parameters(cell, config)
    differences = np.log(spec["alpha"])-np.log(spec["base_alpha"])
    np.testing.assert_allclose(spec["weights"] @ differences, 0., atol=1e-15)
    measured = np.sqrt(np.sum(spec["weights"][:, None]*differences**2)/7)
    assert measured == pytest.approx(heterogeneity, abs=1e-14)


def test_changing_shapes_preserves_marginal_variances_and_separation(config):
    base = dict(config["cells"][0], dimension=10, components=3)
    specs = [study.generating_parameters(dict(base, heterogeneity=h), config) for h in [0., .2, .5, 1.]]
    for spec in specs:
        covariance = spec["covariances"]
        np.testing.assert_allclose(np.diagonal(covariance, axis1=1, axis2=2),
                                   np.diagonal(specs[0]["covariances"], axis1=1, axis2=2), atol=1e-14)
        analytic_mean = spec["locations"]+spec["scales"]*(digamma(spec["alpha"][:, :-1])-digamma(spec["alpha"][:, -1:]))
        np.testing.assert_allclose(analytic_mean, spec["means"], atol=1e-14)
        inverse = np.linalg.inv(np.einsum("j,jab->ab", spec["weights"], covariance))
        distances = [np.sqrt((spec["means"][j]-spec["means"][l]) @ inverse @
                             (spec["means"][j]-spec["means"][l])) for j in range(3) for l in range(j)]
        assert min(distances) == pytest.approx(3.)


@pytest.mark.parametrize("family", sorted(study.FAMILIES))
def test_generated_component_moments_and_oracle(config, family):
    cell = dict(config["cells"][0], family=family, heterogeneity=.5)
    spec = study.generating_parameters(cell, config)
    y, labels = study.generate_sample(80000, spec, np.random.default_rng(81723))
    for j in range(2):
        part = y[labels == j]
        whitened = np.linalg.solve(np.linalg.cholesky(spec["covariances"][j]),
                                   (part-spec["means"][j]).T).T
        np.testing.assert_allclose(whitened.mean(axis=0), 0., atol=.025)
        np.testing.assert_allclose(np.cov(whitened, rowvar=False, ddof=0), np.eye(2), atol=.06)
    log_density, responsibilities = study.oracle_density(y[:100], spec)
    assert np.isfinite(log_density).all()
    np.testing.assert_allclose(responsibilities.sum(axis=1), 1., atol=1e-13)


def test_training_only_transform_and_raw_coordinate_density(config):
    cell = config["cells"][0]
    spec = study.generating_parameters(cell, config)
    train, _ = study.generate_sample(100, spec, np.random.default_rng(91))
    test, _ = study.generate_sample(70, spec, np.random.default_rng(92))
    z, ztest, center, scale = study.standardize_training(train, test)
    _, _, second_center, second_scale = study.standardize_training(train, test+10000)
    np.testing.assert_array_equal(center, second_center)
    np.testing.assert_array_equal(scale, second_scale)
    inverse_scale = scale[None, :]/spec["scales"]
    scaled_location = (center[None, :]-spec["locations"])/spec["scales"]
    theta = np.r_[np.log(spec["weights"][:-1]/spec["weights"][-1]), spec["alpha"][0],
                  inverse_scale.ravel(), scaled_location.ravel()]
    fitted_density = fitters.ld_evaluate(theta, ztest, 2, "shared_ld")[0]-np.log(scale).sum()
    true_density = study.oracle_density(test, spec)[0]
    np.testing.assert_allclose(fitted_density, true_density, atol=1e-12)


def test_native_dirichlet_is_evaluated_on_raw_alr(config):
    cell = dict(config["cells"][0], family="native_dirichlet", heterogeneity=.5)
    spec = study.generating_parameters(cell, config)
    y, _ = study.generate_sample(100, spec, np.random.default_rng(112))
    params = {"method": "fmvmm_dmm", "k": 2, "d": 2,
              "weights": spec["weights"], "alpha": spec["alpha"]}
    np.testing.assert_allclose(fitters.evaluate_params(params, y)[0],
                               study.oracle_density(y, spec)[0], atol=1e-12)


def test_ten_partition_prefix_matches_primary():
    y = np.random.default_rng(91).normal(size=(70, 3))
    with threadpool_limits(1):
        three = study.partitions_for(y, 3, 129, 3)
        ten = study.partitions_for(y, 3, 129, 10)
    for left, right in zip(three, ten):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("stage", ["fit", "generation"])
def test_all_failed_tasks_preserve_denominators_and_attempts(config, tmp_path, monkeypatch, stage):
    config = tiny_config(config)
    def fails(*args, **kwargs):
        raise fitters.FitFailure("Injected total failure", [
            {"start": j, "failed": True, "selected": False, "message": "injected"} for j in range(3)])
    if stage == "generation":
        monkeypatch.setattr(study, "generate_sample", fails)
    result = study.run_task(config, 0, tmp_path, fitter=fails)
    assert result["recorded"] == result["failed"] == 2
    location = tmp_path/"tasks/task_000000/attempt_001"
    fit = json.loads((location/"primary__shared_ld.json").read_text())
    assert fit["parameters"] is None
    assert len(fit["starts"]) == (3 if stage == "fit" else 0)
    study.aggregate(config, tmp_path)
    summary = pd.read_csv(tmp_path/"summary.csv")
    assert (summary.planned == 2).all()
    assert (summary.recorded == 1).all()
    assert (summary.missing == 1).all()
    assert (summary.failed_recorded == 1).all()
    assert (summary.evaluation_available == 0).all()
    assert summary.test_log_density_mean.isna().all()
    with pytest.raises(FileExistsError):
        study.run_task(config, 0, tmp_path, fitter=fails)
    study.run_task(config, 0, tmp_path, attempt=2, fitter=fails)
    assert (location/"completion.json").exists()
    study.aggregate(config, tmp_path)
    assert pd.read_csv(tmp_path/"fits.csv").query("recorded")["attempt"].eq(2).all()


def test_prediction_failure_retains_fitted_parameters(config, tmp_path):
    config = tiny_config(config, ["shared_ld"])
    def supplied_fit(method, y, k, *args):
        params = {"sentinel": [1., 2., 3.]}
        starts = [{"start": 0, "selected": True, "failed": False, "parameters": params}]
        def failing_prediction(y):
            raise ArithmeticError("Injected prediction failure")
        return SimpleNamespace(params=params, starts=starts, selected_start=0,
                               record={"mean_train_log_density": -1.}, evaluate=failing_prediction)
    study.run_task(config, 0, tmp_path, fitter=supplied_fit)
    saved = json.loads((tmp_path/"tasks/task_000000/attempt_001/primary__shared_ld.json").read_text())
    assert saved["row"]["fit_available"]
    assert not saved["row"]["evaluation_available"]
    assert saved["row"]["failed"]
    assert saved["parameters"] == {"sentinel": [1., 2., 3.]}
    assert saved["starts"][0]["selected"]


def test_native_methods_receive_raw_training_coordinates(config, tmp_path):
    config = tiny_config(config, ["shared_ld", "fmvmm_dmm_native"])
    seen = {}
    def supplied_fit(method, y, *args):
        seen[method] = y.copy()
        raise ValueError("Intentional stop after recording fitting coordinates")
    study.run_task(config, 0, tmp_path, fitter=supplied_fit)
    data = np.load(tmp_path/"tasks/task_000000/attempt_001/data.npz")
    np.testing.assert_array_equal(seen["fmvmm_dmm_native"], data["train"])
    np.testing.assert_allclose(seen["shared_ld"].mean(axis=0), 0., atol=1e-15)
    np.testing.assert_allclose(seen["shared_ld"].std(axis=0), 1., atol=1e-15)


def test_frozen_run_rejects_changed_configuration(config, tmp_path):
    output = tmp_path/"frozen"
    study.freeze_run(tiny_config(config), output)
    study.verify_run(output)
    manifest = json.loads((output/"manifest.json").read_text())
    manifest["configuration"]["master_seed"] += 1
    (output/"manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="configuration hash"):
        study.verify_run(output)


def test_bounded_skew_timeout_attempts_survive(config, monkeypatch):
    def timeout(*args, **kwargs):
        raise TimeoutError("Injected optimizer deadline")
    monkeypatch.setattr(study.skew, "minimize", timeout)
    y = np.random.default_rng(313).normal(size=(40, 2))
    partitions = [np.arange(40) % 2]*3
    with pytest.raises(study.skew.SkewNormalFitFailure) as caught:
        study.fit_one("bounded_skew_normal", y, 2, 9, config["fitting"], partitions,
                      skew_settings=config["bounded_skew"])
    assert len(caught.value.starts) == 3
    assert all(record["failed"] for record in caught.value.starts)
    assert all("Injected optimizer deadline" in record["message"] for record in caught.value.starts)
    assert study.skew.minimize is timeout


def test_bounded_skew_final_start_parameters_are_recorded(config):
    y = np.random.default_rng(313).normal(size=(40, 2))
    with threadpool_limits(1):
        model = study.fit_one("bounded_skew_normal", y, 2, 9, config["fitting"], [np.arange(40) % 2],
                             skew_settings={**config["bounded_skew"], "maxiter": 2, "polish_maxiter": 0})
    for record in model.starts:
        if not record["failed"]:
            assert record["parameters"] is not None
            density, _ = study.skew.evaluate(y, record["parameters"])
            assert density.mean() == pytest.approx(record["mean_train_log_density"], abs=1e-12)
