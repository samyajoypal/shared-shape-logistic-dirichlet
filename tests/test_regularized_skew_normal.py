"""Independent density/derivative checks for the bounded skew-normal sensitivity."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.linalg import sqrtm
from scipy.special import log_ndtr, logsumexp
from scipy.stats import multivariate_normal

import regularized_skew_normal as skew


def numeric_gradient(function, x, step=2e-6):
    answer = []
    for j in range(len(x)):
        h = step * max(1.0, abs(x[j]))
        left, right = x.copy(), x.copy()
        left[j] -= h
        right[j] += h
        answer.append((function(right) - function(left)) / (2 * h))
    return np.asarray(answer)


def arbitrary_parameters(k, d, seed=741):
    rng = np.random.default_rng(seed + 10 * k + d)
    weights = rng.dirichlet(np.ones(k))
    locations = rng.normal(size=(k, d))
    cholesky = np.tril(rng.normal(scale=0.2, size=(k, d, d)))
    cholesky[:, np.arange(d), np.arange(d)] = rng.uniform(0.6, 1.6, (k, d))
    beta = rng.uniform(-2.0, 2.0, (k, d))
    return skew.pack(weights, locations, cholesky, beta)


@pytest.mark.parametrize("k,d", [(1, 1), (2, 1), (2, 2), (3, 3), (2, 4)])
def test_analytic_gradient_against_central_differences(k, d):
    rng = np.random.default_rng(112 + k + d)
    y = rng.normal(size=(31, d))
    theta = arbitrary_parameters(k, d)
    value, gradient = skew.objective(theta, y, k)
    numerical = numeric_gradient(lambda x: skew.objective(x, y, k)[0], theta)
    np.testing.assert_allclose(gradient, numerical, rtol=2e-5, atol=2e-7)
    ell, tau, score = skew.evaluation(theta, y, k, gradient=True)
    np.testing.assert_allclose(tau.sum(axis=1), 1.0, atol=2e-14)
    np.testing.assert_allclose(value, -ell.mean(), atol=1e-14)
    np.testing.assert_allclose(gradient, -score / len(y), atol=1e-14)


@pytest.mark.parametrize("k,d", [(1, 1), (2, 2), (3, 4)])
def test_density_against_independent_scipy_expression(k, d):
    theta = arbitrary_parameters(k, d)
    y = np.random.default_rng(92).normal(size=(21, d))
    logpi, locations, cholesky, beta = skew.unpack(theta, k, d)
    terms = np.column_stack([
        logpi[j] + np.log(2.0)
        + multivariate_normal.logpdf(y, mean=locations[j], cov=cholesky[j] @ cholesky[j].T)
        + log_ndtr((y - locations[j]) @ beta[j]) for j in range(k)])
    expected = logsumexp(terms, axis=1)
    ell, tau = skew.evaluation(theta, y, k)
    np.testing.assert_allclose(ell, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(tau, np.exp(terms - expected[:, None]), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("d", [1, 2, 4])
def test_same_density_as_installed_fmvmm(d):
    # fmvmm's density adds epsilon only to its symmetric square-root map.
    # Set its lambda to sqrt(Omega+epsilon I) beta to compare exact densities.
    from fmvmm.mixtures.skewnormmix_smsn import dmvSN
    theta = arbitrary_parameters(1, d)
    _, locations, cholesky, beta = skew.unpack(theta, 1, d)
    omega = cholesky[0] @ cholesky[0].T
    shape = sqrtm(omega + 1e-12 * np.eye(d)) @ beta[0]
    y = np.random.default_rng(58).normal(scale=0.3, size=(20, d)) + locations[0]
    reference = np.log(dmvSN(y, locations[0], omega, shape))
    actual = skew.evaluation(theta, y, 1)[0]
    np.testing.assert_allclose(actual, reference, rtol=3e-11, atol=3e-11)


def test_zero_beta_exactly_embeds_full_gaussian_mixture():
    k, d = 3, 2
    rng = np.random.default_rng(583)
    _, means, cholesky, _ = skew.unpack(arbitrary_parameters(k, d), k, d)
    gaussian = {"weights": np.array([0.2, 0.3, 0.5]), "means": means,
                "covariances": cholesky @ cholesky.transpose(0, 2, 1)}
    theta = skew.gaussian_initial(gaussian, k, d)
    y = rng.normal(size=(45, d))
    expected = logsumexp(np.column_stack([
        np.log(gaussian["weights"][j]) + multivariate_normal.logpdf(
            y, mean=means[j], cov=gaussian["covariances"][j]) for j in range(k)]), axis=1)
    np.testing.assert_allclose(skew.evaluation(theta, y, k)[0], expected, atol=1e-12)
    np.testing.assert_allclose(skew.component_means(skew.parameters(theta, k, d)), means, atol=1e-14)


@pytest.mark.parametrize("beta", [-5.0, 0.0, 3.0])
def test_univariate_normalization_and_component_mean(beta):
    theta = skew.pack([1.0], [[0.7]], [[[1.4]]], [[beta]])
    params = skew.parameters(theta, 1, 1)
    def density(x):
        return float(np.exp(skew.evaluation(theta, np.array([[x]]), 1)[0][0]))
    mass = quad(density, -np.inf, np.inf, epsabs=1e-10)[0]
    mean = quad(lambda x: x * density(x), -np.inf, np.inf, epsabs=1e-10)[0]
    assert abs(mass - 1.0) < 1e-9
    assert abs(mean - skew.component_means(params)[0, 0]) < 1e-9


def test_extreme_negative_skewing_arguments_remain_finite():
    theta = skew.pack([1.0], [[0.0]], [[[1.0]]], [[10.0]])
    y = np.array([[-100.0], [-10.0], [0.0], [100.0]])
    ell, tau, score = skew.evaluation(theta, y, 1, gradient=True)
    assert np.isfinite(ell).all() and np.isfinite(score).all()
    np.testing.assert_array_equal(tau, np.ones((4, 1)))


def fake_result(theta):
    return SimpleNamespace(x=theta.copy(), success=True, nit=1, nfev=1,
                           message="injected finite optimizer result")


def test_failed_start_retained_and_exact_gaussian_candidate_survives(monkeypatch):
    y = np.array([[-1.0], [-0.5], [0.2], [0.9], [1.4]])
    labels = np.zeros(len(y), dtype=int)
    gaussian = {"weights": [1.0], "means": [[y.mean()]],
                "covariances": [[[y.var()]]]}
    calls = []
    def optimizer(function, initial, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise ArithmeticError("injected first-start failure")
        return fake_result(initial)
    monkeypatch.setattr(skew, "minimize", optimizer)
    result = skew.fit(y, 1, partitions=[labels, labels, labels], gaussian_candidate=gaussian,
                      polish_maxiter=0)
    assert len(result.starts) == 5
    assert result.starts[0]["failed"]
    assert "injected first-start failure" in result.starts[0]["message"]
    baseline = skew.evaluation(skew.gaussian_initial(gaussian, 1, 1), y, 1)[0].mean()
    assert result.record["mean_train_log_density"] >= baseline - 1e-13
    assert sum(r["selected"] for r in result.starts) == 1
    assert np.isfinite(result.evaluate(y)[0]).all()


def test_out_of_bounds_gaussian_not_silently_clipped(monkeypatch):
    y = np.array([[-1.0], [0.0], [1.0]])
    labels = np.zeros(len(y), dtype=int)
    gaussian = {"weights": [1.0], "means": [[0.0]], "covariances": [[[1e-10]]]}
    monkeypatch.setattr(skew, "minimize", lambda function, initial, **kwargs: fake_result(initial))
    result = skew.fit(y, 1, partitions=[labels], gaussian_candidate=gaussian, polish_maxiter=0)
    assert len(result.starts) == 2
    assert result.starts[1]["failed"]
    assert "outside the fixed bounds" in result.starts[1]["message"]
    np.testing.assert_array_equal(result.starts[1]["initial_parameters"], skew.gaussian_initial(gaussian, 1, 1))


def test_nonfinite_optimizer_endpoint_is_recorded_as_failure(monkeypatch):
    y = np.array([[-1.0], [0.0], [1.0]])
    labels = np.zeros(len(y), dtype=int)
    def optimizer(function, initial, **kwargs):
        result = fake_result(initial)
        result.x[0] = np.nan
        return result
    monkeypatch.setattr(skew, "minimize", optimizer)
    with pytest.raises(skew.SkewNormalFitFailure) as error:
        skew.fit(y, 1, partitions=[labels, labels], polish_maxiter=0)
    assert len(error.value.starts) == 2
    assert all(r["failed"] and r["exception_type"] == "FloatingPointError" for r in error.value.starts)


def test_small_actual_fit_keeps_gaussian_nesting_and_box_constraints():
    rng = np.random.default_rng(7731)
    y = np.r_[rng.normal(-1.0, 0.5, 25), rng.normal(1.3, 0.7, 25)][:, None]
    labels = (y[:, 0] > 0).astype(int)
    gaussian = {"weights": np.array([(labels == j).mean() for j in range(2)]),
                "means": np.array([y[labels == j].mean(0) for j in range(2)]),
                "covariances": np.array([[[y[labels == j].var()]] for j in range(2)])}
    result = skew.fit(y, 2, partitions=[labels] * 3, gaussian_candidate=gaussian,
                      maxiter=70, polish_maxiter=30)
    baseline = skew.evaluation(skew.gaussian_initial(gaussian, 2, 1), y, 2)[0].mean()
    assert result.record["mean_train_log_density"] >= baseline - 1e-12
    lower, upper = skew._box_arrays(skew.bounds(2, 1))
    assert np.all(result.params["theta"] >= lower) and np.all(result.params["theta"] <= upper)
    assert np.isfinite(result.component_means()).all()
    assert len(result.starts) in (5, 6)
