"""Common-correlation Gaussian comparator: derivatives, nesting and failures."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import logsumexp, polygamma
from scipy.stats import multivariate_normal

import gaussian_shared_correlation as model


def example(k=2, d=3, seed=723):
    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(np.full(k, 3.))
    means = rng.normal(size=(k, d))
    std = rng.uniform(.5, 1.4, (k, d))
    corr = np.array([.37]) if d == 2 else (rng.uniform(.2, .7, d) if d >= 3 else np.empty(0))
    return model.pack(weights, means, std, corr)


def numerical_gradient(function, theta, step=1e-6):
    result = np.zeros(len(theta))
    for j in range(len(theta)):
        difference = np.zeros(len(theta))
        difference[j] = step
        result[j] = (function(theta+difference)-function(theta-difference))/(2*step)
    return result


@pytest.mark.parametrize("k,d", [(1, 1), (2, 2), (2, 3), (3, 6)])
def test_analytic_gradient_against_finite_differences(k, d):
    theta = example(k, d)
    y = np.random.default_rng(13).normal(size=(19, d))
    value, gradient = model.objective(theta, y, k)
    numerical = numerical_gradient(lambda t: model.objective(t, y, k)[0], theta)
    np.testing.assert_allclose(gradient, numerical, atol=5e-8, rtol=3e-6)
    assert np.isfinite(value)


@pytest.mark.parametrize("k,d", [(1, 1), (2, 2), (3, 6)])
def test_density_matches_independent_gaussian_evaluation(k, d):
    theta = example(k, d)
    params = model.parameters(theta, k, d)
    y = np.random.default_rng(41).normal(size=(20, d))
    terms = np.column_stack([np.log(params["weights"][j])+multivariate_normal.logpdf(
        y, mean=params["means"][j], cov=params["covariances"][j]) for j in range(k)])
    expected = logsumexp(terms, axis=1)
    actual, tau = model.evaluate(y, params)
    np.testing.assert_allclose(actual, expected, atol=2e-13)
    np.testing.assert_allclose(tau, np.exp(terms-expected[:, None]), atol=2e-14)
    np.testing.assert_allclose(tau.sum(axis=1), 1., atol=2e-15)


def test_scales_are_marginal_sds_and_correlation_is_shared():
    params = model.parameters(example(3, 6), 3, 6)
    np.testing.assert_allclose(np.diagonal(params["covariances"], axis1=1, axis2=2), params["std"]**2)
    recovered = params["covariances"]/params["std"][:, :, None]/params["std"][:, None, :]
    np.testing.assert_allclose(recovered, np.broadcast_to(params["correlation"], (3, 6, 6)), atol=2e-16)


def test_same_correlation_family_as_shared_logistic_dirichlet():
    alpha = np.array([.8, 1.4, 2.2, .9, 1.8, 2.8, 1.2])
    reference = polygamma(1, alpha[-1])
    marginal_variance = polygamma(1, alpha[:-1])+reference
    r = np.sqrt(reference/marginal_variance)
    kernel_covariance = np.diag(polygamma(1, alpha[:-1]))+reference
    expected = kernel_covariance/np.sqrt(marginal_variance[:, None]*marginal_variance[None, :])
    np.testing.assert_allclose(model.correlation_matrix(r, 6), expected, atol=2e-16)
    np.testing.assert_allclose(model.correlation_matrix([r[0]*r[1]], 2), expected[:2, :2], atol=2e-16)


@pytest.mark.parametrize("d", [1, 2, 3, 10])
def test_parameter_counts_and_positive_definite_boundary(d):
    k = 3
    count = 0 if d == 1 else (1 if d == 2 else d)
    assert model.correlation_parameter_count(d) == count
    assert model.parameter_count(k, d) == (k-1)+2*k*d+count
    assert len(example(k, d)) == model.parameter_count(k, d)
    upper = model.RHO_UPPER if d == 2 else model.LOADING_UPPER
    R = model.correlation_matrix(np.full(count, upper), d)
    assert np.linalg.eigvalsh(R)[0] >= model.CORRELATION_EPS-2e-14
    np.testing.assert_array_equal(np.diag(R), np.ones(d))
    np.testing.assert_array_equal(model.correlation_matrix(np.zeros(count), d), np.eye(d))
    if d == 2:
        with pytest.raises(ValueError):
            model.correlation_matrix([.3, .3], d)
    shared_ld_count = (k-1)+(d+1)+2*k*d
    if d >= 3:
        assert shared_ld_count-model.parameter_count(k, d) == 1
    elif d == 2:
        assert shared_ld_count-model.parameter_count(k, d) == 2


def test_one_factor_zero_boundary_is_not_globally_identified():
    np.testing.assert_array_equal(model.correlation_matrix([.6, 0., 0.], 3), np.eye(3))
    np.testing.assert_array_equal(model.correlation_matrix([.3, .8, 0.], 3),
                                   model.correlation_matrix([.4, .6, 0.], 3))
    theta = example(2, 3)
    theta[-3:] = 0
    y = np.random.default_rng(4).normal(size=(17, 3))
    _, gradient = model.objective(theta, y, 2)
    np.testing.assert_array_equal(gradient[-3:], np.zeros(3))
    assert any(hit.startswith('loading') and hit.endswith(':lower') for hit in model.boundary_hits(theta, 2, 3))


def test_density_affine_jacobian_and_means_are_consistent():
    theta = example(2, 3)
    logpi, means, std, correlation, _ = model.unpack(theta, 2, 3)
    y = np.random.default_rng(57).normal(size=(20, 3))
    scales, shift = np.array([.5, 2., 1.4]), np.array([1., -2., 3.])
    changed = model.pack(np.exp(logpi), means*scales+shift, std*scales, correlation)
    np.testing.assert_allclose(model.evaluation(changed, y*scales+shift, 2)[0],
                               model.evaluation(theta, y, 2)[0]-np.log(scales).sum(), atol=1e-13)
    np.testing.assert_array_equal(model.component_means(model.parameters(theta, 2, 3)), means)


@pytest.mark.parametrize("d", [2, 6])
def test_diagonal_candidate_is_exact_without_clipping(d):
    k = 2
    rng = np.random.default_rng(899)
    candidate = {"weights": [.6, .4], "means": rng.normal(size=(k, d)),
                 "covariances": rng.uniform(.2, 2., (k, d))}
    theta = model.diagonal_initial(candidate, k, d)
    params = model.parameters(theta, k, d)
    y = rng.normal(size=(10, d))
    expected = logsumexp(np.column_stack([np.log(candidate["weights"][j])+multivariate_normal.logpdf(
        y, mean=candidate["means"][j], cov=np.diag(candidate["covariances"][j])) for j in range(k)]), axis=1)
    np.testing.assert_allclose(model.evaluate(y, params)[0], expected, atol=1e-13)
    np.testing.assert_array_equal(params["correlation"], np.eye(d))
    wrong = {**candidate, "covariances": np.tile(np.eye(d)+.1*np.ones((d, d)), (k, 1, 1))}
    with pytest.raises(ValueError, match="not exactly diagonal"):
        model.diagonal_initial(wrong, k, d)


def sample_and_partitions(d=3):
    rng = np.random.default_rng(617)
    y = rng.normal(size=(60, d))
    return y, [np.arange(60) % 2, (np.arange(60)//3) % 2]


def fake_optimizer(fun, initial, *args, **kwargs):
    return SimpleNamespace(x=initial.copy(), success=True, nit=0, nfev=1, message="Injected finite endpoint")


@pytest.mark.parametrize("failure", ["exception", "nan"])
def test_failed_first_start_cannot_discard_later_candidates(monkeypatch, failure):
    y, partitions = sample_and_partitions()
    calls = []
    def faulty(fun, initial, *args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            if failure == "exception":
                raise ArithmeticError("Injected first-start failure")
            return SimpleNamespace(x=np.full_like(initial, np.nan), success=True, nit=1, nfev=2, message="NaN")
        return fake_optimizer(fun, initial, *args, **kwargs)
    monkeypatch.setattr(model, "minimize", faulty)
    fitted = model.fit(y, 2, partitions=partitions, polish_maxiter=0)
    assert fitted.starts[0]["failed"]
    assert not fitted.starts[1]["failed"]
    assert not fitted.starts[fitted.selected_start]["failed"]
    assert np.isfinite(fitted.evaluate(y)[0]).all()
    assert sum(record["selected"] for record in fitted.starts) == 1


def test_all_optimizer_failures_retain_explicit_partition_diagonal(monkeypatch):
    def fails(*args, **kwargs):
        raise TimeoutError("Injected optimizer timeout")
    monkeypatch.setattr(model, "minimize", fails)
    y, partitions = sample_and_partitions()
    fitted = model.fit(y, 2, partitions=partitions, polish_maxiter=0)
    assert all(record["failed"] for record in fitted.starts[:2])
    assert fitted.record["initialization"] == "partition_diagonal_candidate"
    assert "not a fitted diagonal MLE" in fitted.record["message"]
    assert not fitted.record["correlation_loading_identified"]
    np.testing.assert_array_equal(fitted.params["correlation"], np.eye(3))


def test_all_initializations_failed_raise_with_every_record():
    y, _ = sample_and_partitions()
    invalid = [np.zeros(len(y), dtype=int), np.full(len(y), 9)]
    with pytest.raises(model.GaussianSharedCorrelationFitFailure) as caught:
        model.fit(y, 2, partitions=invalid, polish_maxiter=0)
    assert len(caught.value.starts) == 2
    assert all(record["failed"] for record in caught.value.starts)
    assert caught.value.elapsed_seconds >= 0


def test_infeasible_external_candidate_is_retained_rejected_and_not_clipped(monkeypatch):
    monkeypatch.setattr(model, "minimize", fake_optimizer)
    y, partitions = sample_and_partitions()
    candidate = {"weights": [.5, .5], "means": np.full((2, 3), 30.),
                 "covariances": np.ones((2, 3))}
    fitted = model.fit(y, 2, partitions=partitions, diagonal_candidate=candidate, polish_maxiter=0)
    external = next(row for row in fitted.starts if row["initialization"] == "exact_diagonal_candidate")
    assert external["failed"] and "not clipped" in external["message"]
    np.testing.assert_array_equal(external["initial_parameters"][1:7], np.full(6, 30.))
    assert any(row["initialization"] == "partition_diagonal_candidate" for row in fitted.starts)
    assert not fitted.record["external_diagonal_candidate_usable"]


def test_continuation_failure_preserves_immutable_exact_candidate(monkeypatch):
    y, partitions = sample_and_partitions(d=2)
    candidate = {"weights": [.5, .5], "means": np.array([[0., 0.], [.1, .1]]),
                 "covariances": np.ones((2, 2))}
    calls = []
    def faulty(fun, initial, *args, **kwargs):
        calls.append(1)
        if len(calls) > 1:
            raise ArithmeticError("Injected continuation failure")
        return fake_optimizer(fun, initial, *args, **kwargs)
    monkeypatch.setattr(model, "minimize", faulty)
    fitted = model.fit(y, 2, partitions=partitions[:1], diagonal_candidate=candidate,
                       polish_maxiter=1, gtol=1e-12)
    exact = next(row for row in fitted.starts if row["initialization"] == "exact_diagonal_candidate")
    np.testing.assert_array_equal(exact["parameters"]["theta"], model.diagonal_initial(candidate, 2, 2))
    assert any(row["failed"] and row["initialization"] == "selected_candidate_continuation" for row in fitted.starts)
    assert fitted.record["mean_train_log_density"] >= exact["mean_train_log_density"]


def test_real_small_fit_retains_all_parameters_and_diagonal_nesting():
    y, partitions = sample_and_partitions(d=2)
    candidate = {"weights": [.5, .5], "means": np.array([[0., 0.], [.1, .1]]),
                 "covariances": np.ones((2, 2))}
    fitted = model.fit(y, 2, seed=11, partitions=partitions, diagonal_candidate=candidate,
                       maxiter=20, polish_maxiter=0)
    exact = next(row for row in fitted.starts if row["initialization"] == "exact_diagonal_candidate")
    assert fitted.record["mean_train_log_density"] >= exact["mean_train_log_density"]-1e-13
    assert fitted.record["parameter_count"] == 10
    for row in fitted.starts:
        if not row["failed"]:
            assert row["parameters"] is not None
            assert model.evaluate(y, row["parameters"])[0].mean() == pytest.approx(row["mean_train_log_density"], abs=1e-13)
    for start in range(5):
        initial, details = model.initial_parameters(y, partitions[0], 2, start)
        assert model.unpack(initial, 2, 2)[3][0] > 0
        assert details["initial_equicorrelation"] > 0
