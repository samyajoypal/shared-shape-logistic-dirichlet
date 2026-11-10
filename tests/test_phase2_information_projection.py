"""Independent score, local path, projection and provenance checks."""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.special import digamma

import phase2_information_projection as bridge
import phase2_simulations as simulation


@pytest.fixture
def config():
    return json.loads(bridge.DEFAULT_CONFIG.read_text())


def central_derivative(function, vector, step=1e-6):
    columns = []
    for j in range(len(vector)):
        displacement = np.zeros(len(vector))
        displacement[j] = step
        columns.append((function(vector+displacement)-function(vector-displacement))/(2*step))
    return np.stack(columns, axis=-1)


@pytest.mark.parametrize("k,d", [(1, 1), (2, 2), (3, 4)])
def test_full_raw_score_matches_density_finite_differences(k, d):
    rng = np.random.default_rng(6543+k+d)
    weights = rng.dirichlet(np.ones(k))
    alpha = rng.uniform(.5, 3., (k, d+1))
    locations = rng.normal(size=(k, d))
    scales = rng.uniform(.5, 1.5, (k, d))
    theta = bridge.pack_raw(weights, alpha, locations, scales)
    y = rng.normal(size=(7, d))*2
    density, score, responsibilities = bridge.full_raw_score(theta, y, k)
    numeric = central_derivative(lambda t: bridge.full_raw_score(t, y, k)[0], theta)
    np.testing.assert_allclose(score, numeric, atol=2e-8, rtol=3e-6)
    np.testing.assert_allclose(responsibilities.sum(axis=1), 1., atol=2e-15)
    spec = dict(family="ld", weights=weights, alpha=alpha, locations=locations, scales=scales)
    np.testing.assert_allclose(density, simulation.oracle_density(y, spec)[0], atol=1e-13)


@pytest.mark.parametrize("d,k", [(2, 2), (6, 3), (10, 3)])
def test_embedding_and_existing_shared_score_after_coordinate_map(config, d, k):
    cell = dict(config["cells"][0], dimension=d, components=k, heterogeneity=0.)
    spec = simulation.generating_parameters(cell, config)
    theta = bridge.raw_from_specification(spec)
    y, _ = simulation.generate_sample(15, spec, np.random.default_rng(d*10+k))
    J = bridge.shared_embedding(k, d)
    raw_shared = np.r_[spec["weights"][:-1], spec["alpha"][0],
                       spec["locations"].ravel(), spec["scales"].ravel()]
    np.testing.assert_allclose(J @ raw_shared, theta, atol=1e-15)
    assert np.linalg.matrix_rank(J) == J.shape[1]
    check = bridge.score_mapping_check(theta, y, k)
    assert check["score_mapping_max_abs"] < 1e-11
    assert check["density_mapping_max_abs"] < 1e-11


@pytest.mark.parametrize("d,k", [(2, 2), (6, 3)])
def test_component_mean_gradient_has_independent_shapes(config, d, k):
    cell = dict(config["cells"][0], dimension=d, components=k, heterogeneity=.4)
    spec = simulation.generating_parameters(cell, config)
    theta = bridge.raw_from_specification(spec)
    for j, m in [(0, 0), (k-1, d-1)]:
        value, gradient = bridge.component_mean_gradient(theta, k, d, j, m)
        numerical = central_derivative(lambda t: np.array(bridge.component_mean_gradient(t, k, d, j, m)[0]), theta)
        np.testing.assert_allclose(gradient, numerical, atol=2e-8, rtol=2e-6)
        assert value == pytest.approx(spec["means"][j, m], abs=1e-14)
        shape_gradient = gradient[bridge.layout(k, d)["alpha"]].reshape(k, d+1)
        assert np.count_nonzero(shape_gradient) == 2
        assert np.count_nonzero(np.delete(shape_gradient, j, axis=0)) == 0


@pytest.mark.parametrize("d,k,weight_profile", [(2, 2, "balanced"), (6, 3, "balanced"),
                                                (6, 3, "imbalanced")])
def test_local_generator_derivative_includes_nuisance_parameters(config, d, k, weight_profile):
    cell = dict(config["cells"][0], dimension=d, components=k, weight_profile=weight_profile,
                heterogeneity=0.)
    spec, direction, details = bridge.generator_path_derivative(cell, config)
    step = 1e-6
    moved = simulation.generating_parameters(dict(cell, heterogeneity=step), config)
    numerical = (bridge.raw_from_specification(moved)-bridge.raw_from_specification(spec))/step
    np.testing.assert_allclose(direction, numerical, rtol=3e-5, atol=2e-5)
    np.testing.assert_allclose(np.diagonal(details["covariance_derivative"], axis1=1, axis2=2),
                               0., atol=3e-15)
    theta = bridge.raw_from_specification(spec)
    targets = bridge.target_definitions(theta, k, d)
    derivatives = np.array([target["gradient"] @ direction for target in targets[:k*d]])
    np.testing.assert_allclose(derivatives.reshape(k, d), details["mean_derivative"], atol=1e-13)
    assert np.linalg.norm(details["scale_derivative"]) > .05
    assert np.linalg.norm(details["location_derivative"]) > .05


def test_local_config_coefficients_are_used_and_correspond_to_same_path(config):
    local = [cell for cell in config["cells"] if cell.get("block") == "local_alternative"]
    assert {cell["local_h_coefficient"] for cell in local} == {1., 3.}
    for cell in local:
        assert cell["heterogeneity"] == pytest.approx(cell["local_h_coefficient"]/np.sqrt(cell["n_train"]))
        _, _, details = bridge.generator_path_derivative(cell, config)
        assert len(details["active_minimum_pairs"]) == 1


def test_information_projection_equals_independent_whitened_projection():
    rng = np.random.default_rng(4321)
    J = bridge.shared_embedding(2, 2)
    A = rng.normal(size=(J.shape[0], J.shape[0]))
    I = A.T @ A+np.eye(len(A))
    full, restricted, P, diagnostic = bridge.information_projection(I, J)
    values, vectors = np.linalg.eigh(I)
    square_root = (vectors*np.sqrt(values)) @ vectors.T
    inverse_root = (vectors/np.sqrt(values)) @ vectors.T
    Q, _ = np.linalg.qr(square_root @ J, mode="reduced")
    expected = inverse_root @ Q @ Q.T @ inverse_root
    np.testing.assert_allclose(restricted, expected, atol=2e-14)
    np.testing.assert_allclose(full, np.linalg.inv(I), atol=1e-14)
    np.testing.assert_allclose(P @ J, J, atol=2e-14)
    assert diagnostic["minimum_variance_difference_eigenvalue"] > -1e-13
    assert diagnostic["projection_idempotence_max_abs"] < 1e-13
    for broken in [np.zeros_like(I), I*np.nan]:
        with pytest.raises((ValueError, np.linalg.LinAlgError)):
            bridge.information_projection(broken, J)


def test_risk_reduction_bias_and_aggregate_loss_are_correct():
    J = bridge.shared_embedding(2, 2)
    full, restricted, P, _ = bridge.information_projection(np.eye(J.shape[0]), J)
    theta = bridge.pack_raw([.6, .4], [[1., 2., 1.5]]*2,
                            [[0., 0.], [2., 1.]], [[1., .7], [.8, 1.2]])
    targets = bridge.target_definitions(theta, 2, 2)
    # An exactly shared direction cannot cause first-order restriction bias.
    shared_direction = J @ np.arange(J.shape[1])*.01
    rows = bridge.risk_rows(full, restricted, P, targets, shared_direction, [0., 1., 3.], 2, 2)
    assert max(row["bias_squared"] for row in rows) < 1e-27
    direction = np.zeros(len(theta))
    direction[1] = 1.
    direction[4] = -1.
    rows = bridge.risk_rows(full, restricted, P, targets, direction, [0., 1., 3.], 2, 2)
    first = [row for row in rows if row["target"] == "component_0_mean_0"]
    assert first[2]["bias_squared"] == pytest.approx(9*first[1]["bias_squared"])
    assert first[0]["shared_to_full_variance_ratio"] <= 1.
    for coefficient in [0., 1., 3.]:
        component = [r for r in rows if r["target"].startswith("component_") and r["local_coefficient"] == coefficient]
        aggregate = next(r for r in rows if r["target_type"] == "average_squared_vector_error" and r["local_coefficient"] == coefficient)
        assert aggregate["shared_limiting_normal_risk"] == pytest.approx(np.mean([r["shared_limiting_normal_risk"] for r in component]))


def test_information_accumulation_is_reproducible_and_uncentered(config):
    cell = bridge.scenarios(config)[0]
    spec, _, _ = bridge.generator_path_derivative(cell, config)
    theta = bridge.raw_from_specification(spec)
    first = bridge.estimate_information(theta, spec, 300, np.random.default_rng(222), chunk_size=300)
    second = bridge.estimate_information(theta, spec, 300, np.random.default_rng(222), chunk_size=300)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    y, _ = simulation.generate_sample(300, spec, np.random.default_rng(222))
    score = bridge.full_raw_score(theta, y, 2)[1]
    np.testing.assert_allclose(first[0], score.T @ score/300, atol=0)
    np.testing.assert_allclose(first[1], score.mean(axis=0), atol=0)


def test_small_run_records_sources_matrices_uncertainty_and_no_overwrite(config, tmp_path):
    output = tmp_path/"run"
    rows = bridge.run(config, output, batches=2, draws_per_batch=300,
                      chunk_size=150, selected=["d2_k2_separated"])
    manifest = json.loads((output/"manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["total_generated_observations"] == 600
    assert manifest["local_coefficients"] == [0., 1., 3.]
    assert len(rows) == (4+1+1)*3
    assert all(row["shared_to_full_variance_ratio_batch_mcse"] >= 0 for row in rows)
    matrices = np.load(output/"d2_k2_separated/pooled.npz")
    assert matrices["information"].shape == (15, 15)
    assert (output/"source_snapshot/phase2_information_projection.py").exists()
    with pytest.raises(FileExistsError):
        bridge.run(config, output, batches=2, draws_per_batch=300,
                   selected=["d2_k2_separated"])


def test_failed_monte_carlo_batch_is_retained(config, tmp_path, monkeypatch):
    def fails(*args, **kwargs):
        raise ArithmeticError("Injected score integration failure")
    monkeypatch.setattr(bridge, "estimate_information", fails)
    output = tmp_path/"failed"
    with pytest.raises(ArithmeticError, match="Injected"):
        bridge.run(config, output, batches=2, draws_per_batch=100, selected=["d2_k2_overlap"])
    assert json.loads((output/"manifest.json").read_text())["status"] == "failed"
    assert json.loads((output/"failure.json").read_text())["completed_batches"] == 0
    assert (output/"d2_k2_overlap/truth_and_path.json").exists()
