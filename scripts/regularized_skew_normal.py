"""Bounded direct fitting of the same scalar-skewing normal family as fmvmm.

This diagnostic-driven sensitivity uses
  2 * Normal_d(y; location, Omega) * Phi(beta.T @ (y-location)),
where Omega = L @ L.T and L is lower triangular with positive diagonal.
The skew vector beta is free, so covariance derivatives hold beta fixed.
Bounds are fixed on TRAINING-standardized coordinates; this module neither
standardizes data nor examines held-out data. "Regularized" here means box
constraints, not a penalized likelihood. There is no global-optimum claim.

API: fit(y, k, seed, partitions=..., gaussian_candidate=...) returns an
object with params, record, starts, selected_start, evaluate(y), and
component_means(). evaluate returns (per-row log density, responsibilities).
All numerical starts, failures, and the optional exact Gaussian candidate
are retained. The selected finite candidate gets at most one continuation.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
import warnings

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
from scipy.special import log_ndtr, logsumexp
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

METHOD = "bounded_skew_normal"
LOCATION_BOUNDS = (-20.0, 20.0)
CHOLESKY_DIAGONAL_BOUNDS = (0.005, 10.0)
CHOLESKY_OFFDIAGONAL_BOUNDS = (-10.0, 10.0)
BETA_BOUNDS = (-10.0, 10.0)
SHAPE_START_LEVELS = (0.5, 1.0, 2.0)
INITIAL_COVARIANCE_RIDGE = 1e-6
LOG_TWO_PI = float(np.log(2.0 * np.pi))


def parameter_count(k, d):
    return k - 1 + k * (2 * d + d * (d + 1) // 2)


def bounds(k, d):
    rows, columns = np.tril_indices(d)
    chol_bounds = [tuple(np.log(CHOLESKY_DIAGONAL_BOUNDS)) if r == c
                   else CHOLESKY_OFFDIAGONAL_BOUNDS for r, c in zip(rows, columns)]
    return ([(None, None)] * (k - 1) + [LOCATION_BOUNDS] * (k * d)
            + chol_bounds * k + [BETA_BOUNDS] * (k * d))


def unpack(theta, k, d):
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (parameter_count(k, d),):
        raise ValueError("Incorrect parameter vector length")
    r, q = k - 1, d * (d + 1) // 2
    logits = np.r_[theta[:r], 0.0]
    logpi = logits - logsumexp(logits)
    locations = theta[r:r + k * d].reshape(k, d)
    packed = theta[r + k * d:r + k * d + k * q].reshape(k, q)
    rows, columns = np.tril_indices(d)
    diagonal = rows == columns
    cholesky = np.zeros((k, d, d))
    cholesky[:, rows, columns] = packed
    cholesky[:, rows[diagonal], columns[diagonal]] = np.exp(packed[:, diagonal])
    beta = theta[r + k * d + k * q:].reshape(k, d)
    return logpi, locations, cholesky, beta


def pack(weights, locations, cholesky, beta):
    weights, locations = np.asarray(weights, dtype=float), np.asarray(locations, dtype=float)
    cholesky, beta = np.asarray(cholesky, dtype=float), np.asarray(beta, dtype=float)
    k, d = locations.shape
    if weights.shape != (k,) or cholesky.shape != (k, d, d) or beta.shape != (k, d):
        raise ValueError("Incompatible component array dimensions")
    if not all(np.isfinite(x).all() for x in (weights, locations, cholesky, beta)):
        raise ValueError("Parameters must be finite")
    if np.any(weights <= 0) or not np.isclose(weights.sum(), 1.0, rtol=1e-10, atol=1e-12):
        raise ValueError("Weights must be positive and sum to one")
    if np.any(np.diagonal(cholesky, axis1=1, axis2=2) <= 0):
        raise ValueError("Cholesky diagonals must be positive")
    if np.any(np.triu(cholesky, k=1) != 0):
        raise ValueError("Cholesky matrices must be lower triangular")
    rows, columns = np.tril_indices(d)
    diagonal = rows == columns
    packed = cholesky[:, rows, columns].copy()
    packed[:, diagonal] = np.log(packed[:, diagonal])
    return np.r_[np.log(weights[:-1]) - np.log(weights[-1]), locations.ravel(),
                 packed.ravel(), beta.ravel()]


def evaluation(theta, y, k, gradient=False):
    """Mixture log densities, responsibilities, and optional summed score.

    Location score is Omega^{-1}r - Mills(beta'r)*beta. The score in beta
    is Mills(beta'r)*r. The weighted L score is
    [sum_i tau_i v_i v_i' - sum_i tau_i Omega^{-1}]L, v_i=Omega^{-1}r_i;
    diagonal entries are multiplied by L_mm for log-diagonal coordinates.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 2 or not np.isfinite(y).all():
        raise ValueError("Observations must be a finite two-dimensional array")
    n, d = y.shape
    logpi, locations, cholesky, beta = unpack(theta, k, d)
    terms = np.empty((n, k))
    cached = []
    for j in range(k):
        residual = y - locations[j]
        inverse_l = solve_triangular(cholesky[j], np.eye(d), lower=True,
                                     check_finite=False)
        whitened = residual @ inverse_l.T
        argument = residual @ beta[j]
        logcdf = log_ndtr(argument)
        terms[:, j] = (logpi[j] + np.log(2.0) - 0.5 * d * LOG_TWO_PI
                       - np.log(np.diag(cholesky[j])).sum()
                       - 0.5 * np.einsum("ij,ij->i", whitened, whitened) + logcdf)
        if gradient:
            inverse_omega = inverse_l.T @ inverse_l
            v = whitened @ inverse_l
            mills = np.exp(-0.5 * argument * argument - 0.5 * LOG_TWO_PI - logcdf)
            cached.append((residual, inverse_omega, v, mills))
    ell = logsumexp(terms, axis=1)
    tau = np.exp(terms - ell[:, None])
    if not gradient:
        return ell, tau
    rows, columns = np.tril_indices(d)
    diagonal = rows == columns
    location_score = np.empty((k, d))
    cholesky_score = np.empty((k, len(rows)))
    beta_score = np.empty((k, d))
    for j, (residual, inverse_omega, v, mills) in enumerate(cached):
        mass = tau[:, j].sum()
        weighted_mills = tau[:, j] * mills
        location_score[j] = tau[:, j] @ v - weighted_mills.sum() * beta[j]
        beta_score[j] = weighted_mills @ residual
        matrix_score = ((v * tau[:, j, None]).T @ v - mass * inverse_omega) @ cholesky[j]
        cholesky_score[j] = matrix_score[rows, columns]
        cholesky_score[j, diagonal] *= np.diag(cholesky[j])
    weight_score = tau[:, :k - 1].sum(axis=0) - n * np.exp(logpi[:k - 1])
    score = np.r_[weight_score, location_score.ravel(), cholesky_score.ravel(), beta_score.ravel()]
    return ell, tau, score


def objective(theta, y, k):
    """Mean negative log likelihood and its analytic gradient."""
    ell, _, score = evaluation(theta, y, k, gradient=True)
    return -float(ell.mean()), -score / len(y)


def parameters(theta, k, d):
    logpi, locations, cholesky, beta = unpack(theta, k, d)
    return {"method": METHOD, "k": k, "d": d, "theta": np.array(theta, copy=True),
            "weights": np.exp(logpi), "locations": locations.copy(),
            "cholesky": cholesky.copy(), "covariances": cholesky @ cholesky.transpose(0, 2, 1),
            "beta": beta.copy(), "parameterization": "location, lower Cholesky, free beta"}


def evaluate(y, params):
    return evaluation(np.asarray(params["theta"]), np.asarray(y, dtype=float), int(params["k"]))


def component_means(params):
    _, locations, cholesky, beta = unpack(params["theta"], int(params["k"]), int(params["d"]))
    omega = cholesky @ cholesky.transpose(0, 2, 1)
    omega_beta = np.einsum("kij,kj->ki", omega, beta)
    denominator = np.sqrt(1.0 + np.einsum("ki,ki->k", beta, omega_beta))
    return locations + np.sqrt(2.0 / np.pi) * omega_beta / denominator[:, None]


def gaussian_initial(gaussian_candidate, k, d):
    """Exact zero-beta embedding, with no clipping or covariance alteration."""
    candidate = (gaussian_candidate.params if hasattr(gaussian_candidate, "params")
                 else gaussian_candidate)
    means, covariance = np.asarray(candidate["means"]), np.asarray(candidate["covariances"])
    if means.shape != (k, d) or covariance.shape != (k, d, d):
        raise ValueError("Gaussian candidate must have full covariance and matching k,d")
    return pack(candidate["weights"], means, np.linalg.cholesky(covariance), np.zeros((k, d)))


def initial_parameters(y, labels, k, level):
    """Use a matched training partition and nonzero empirical-skew direction."""
    d = y.shape[1]
    labels = np.asarray(labels)
    if labels.shape != (len(y),):
        raise ValueError("Partition length differs from training size")
    weights, locations, cholesky, betas = [], [], [], []
    for j in range(k):
        group = y[labels == j]
        if not len(group):
            raise ValueError("Initialization partition has an empty component")
        mean = group.mean(axis=0)
        centered = group - mean
        covariance = centered.T @ centered / len(group) + INITIAL_COVARIANCE_RIDGE * np.eye(d)
        l = np.linalg.cholesky(covariance)
        rows, columns = np.tril_indices(d, -1)
        l[rows, columns] = np.clip(l[rows, columns], *CHOLESKY_OFFDIAGONAL_BOUNDS)
        l[np.diag_indices(d)] = np.clip(np.diag(l), *CHOLESKY_DIAGONAL_BOUNDS)
        omega = l @ l.T
        signs = np.sign(np.mean(centered ** 3, axis=0))
        signs[signs == 0] = 1.0
        beta = level * signs
        omega_beta = omega @ beta
        location = mean - np.sqrt(2.0 / np.pi) * omega_beta / np.sqrt(1.0 + beta @ omega_beta)
        weights.append(len(group) / len(y))
        locations.append(np.clip(location, *LOCATION_BOUNDS))
        cholesky.append(l)
        betas.append(beta)
    return pack(weights, locations, cholesky, betas)


def _box_arrays(limits):
    return (np.array([-np.inf if lo is None else lo for lo, _ in limits]),
            np.array([np.inf if hi is None else hi for _, hi in limits]))


def projected_gradient(theta, gradient, limits):
    lower, upper = _box_arrays(limits)
    return float(np.max(np.abs(theta - np.clip(theta - gradient, lower, upper))))


def boundary_hits(theta, limits, k, d):
    rows, columns = np.tril_indices(d)
    names = ([f"eta[{j}]" for j in range(k - 1)]
             + [f"location[{j},{m}]" for j in range(k) for m in range(d)]
             + [f"{'log_diag_L' if r == c else 'L'}[{j},{r},{c}]"
                for j in range(k) for r, c in zip(rows, columns)]
             + [f"beta[{j},{m}]" for j in range(k) for m in range(d)])
    hits = []
    for name, value, (lower, upper) in zip(names, theta, limits):
        if lower is None or upper is None:
            continue
        tolerance = 1e-4 * (upper - lower)
        if value - lower <= tolerance:
            hits.append(name + ":lower")
        elif upper - value <= tolerance:
            hits.append(name + ":upper")
    return hits


@dataclass
class SkewNormalFit:
    params: dict
    record: dict
    starts: list
    selected_start: int
    elapsed_seconds: float

    @property
    def method(self):
        return METHOD

    @property
    def k(self):
        return int(self.params["k"])

    @property
    def d(self):
        return int(self.params["d"])

    def evaluate(self, y):
        return evaluate(y, self.params)

    def component_means(self):
        return component_means(self.params)


class SkewNormalFitFailure(RuntimeError):
    def __init__(self, message, starts, elapsed_seconds):
        super().__init__(message)
        self.starts = starts
        self.records = starts
        self.elapsed_seconds = elapsed_seconds


def fit(y, k, seed=20261107, partitions=None, gaussian_candidate=None,
        n_starts=3, maxiter=3000, polish_maxiter=2000, gtol=1e-6):
    """Fit training data using fixed bounds and matched partition starts.

    The optional feasible Gaussian embedding is both an immutable candidate
    and an additional optimizer start. A single continuation is attempted
    only when the best surviving candidate has projected score above gtol.
    Selection always uses training likelihood, including after continuation.
    """
    beginning = time.perf_counter()
    y = np.asarray(y, dtype=float)
    if y.ndim != 2 or not len(y) or not y.shape[1] or not np.isfinite(y).all():
        raise ValueError("Training observations must be a nonempty finite matrix")
    if not isinstance(k, (int, np.integer)) or not 1 <= k <= len(y):
        raise ValueError("Component count must be an integer in [1,N]")
    if n_starts < 1 or maxiter < 1 or polish_maxiter < 0 or gtol <= 0:
        raise ValueError("Invalid optimizer settings")
    d = y.shape[1]
    limits = bounds(k, d)
    lower, upper = _box_arrays(limits)
    records, candidates = [], []

    def attempt(initial, reason, iterations, optimize=True, parent_start=None):
        index, t0 = len(records), time.perf_counter()
        rec = {"start": index, "initialization": reason, "parent_start": parent_start,
               "failed": True, "selected": False, "optimizer_success": False,
               "score_converged": False, "projected_gradient": np.nan,
               "mean_train_log_density": np.nan, "iterations": 0, "boundary_hits": [],
               "boundary_hit": False, "parameter_count": parameter_count(k, d),
               "min_weight": np.nan, "message": ""}
        try:
            theta0 = initial() if callable(initial) else np.asarray(initial, dtype=float).copy()
            rec["initial_parameters"] = theta0.copy()
            if theta0.shape != lower.shape or not np.isfinite(theta0).all():
                raise ValueError("Invalid initial parameter vector")
            if np.any(theta0 < lower) or np.any(theta0 > upper):
                raise ValueError("Initial candidate is outside the fixed bounds; exact embedding not clipped")
            initial_value, initial_gradient = objective(theta0, y, k)
            if not np.isfinite(initial_value) or not np.isfinite(initial_gradient).all():
                raise FloatingPointError("Nonfinite initial density or score")
            rec["initial_mean_train_log_density"] = -initial_value
            theta = theta0
            if optimize:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    result = minimize(objective, theta0, args=(y, k), jac=True, method="L-BFGS-B",
                                      bounds=limits, options={"maxiter": iterations, "maxls": 50,
                                                            "ftol": 1e-15, "gtol": gtol})
                rec.update({"optimizer_success": bool(result.success), "iterations": int(result.nit),
                            "function_evaluations": int(result.nfev), "message": str(result.message),
                            "warnings": [str(w.message) for w in caught]})
                theta = result.x
            else:
                rec["message"] = "Exact feasible zero-beta Gaussian candidate; not optimized in this record"
            if not np.isfinite(theta).all():
                raise FloatingPointError("Nonfinite fitted parameters")
            value, gradient = objective(theta, y, k)
            if not np.isfinite(value) or not np.isfinite(gradient).all():
                raise FloatingPointError("Nonfinite fitted objective or gradient")
            if value > initial_value:
                rec["discarded_worse_optimizer_endpoint"] = True
                rec["optimizer_endpoint_mean_train_log_density"] = -value
                theta, value, gradient = theta0, initial_value, initial_gradient
                rec["optimizer_success"] = False
                rec["message"] += "; retained superior initial point"
            pg = projected_gradient(theta, gradient, limits)
            hits = boundary_hits(theta, limits, k, d)
            logpi, _, _, beta = unpack(theta, k, d)
            rec.update({"failed": False, "projected_gradient": pg, "score_converged": bool(pg <= gtol),
                        "mean_train_log_density": -value, "boundary_hits": hits, "boundary_hit": bool(hits),
                        "min_weight": float(np.exp(logpi).min()), "max_beta_norm": float(np.linalg.norm(beta, axis=1).max())})
            candidates.append((index, theta.copy()))
        except Exception as exc:
            rec.update({"failed": True, "optimizer_success": False, "message": repr(exc),
                        "exception_type": type(exc).__name__})
        rec["elapsed_seconds"] = time.perf_counter() - t0
        records.append(rec)

    with threadpool_limits(limits=1):
        if partitions is None:
            partitions = [KMeans(k, n_init=1, random_state=int(seed + j)).fit_predict(y)
                          for j in range(n_starts)]
        for index, labels in enumerate(partitions):
            level = SHAPE_START_LEVELS[index % len(SHAPE_START_LEVELS)]
            attempt(lambda labels=labels, level=level: initial_parameters(y, labels, k, level),
                    f"matched_KMeans_beta_{level:g}", maxiter)
        if gaussian_candidate is not None:
            gaussian_index = len(records)
            attempt(lambda: gaussian_initial(gaussian_candidate, k, d), "exact_Gaussian_candidate", 0, optimize=False)
            if not records[gaussian_index]["failed"]:
                initial = records[gaussian_index]["initial_parameters"]
                attempt(initial, "exact_Gaussian_start", maxiter, parent_start=gaussian_index)
        if not candidates:
            raise SkewNormalFitFailure("Every bounded skew-normal candidate failed", records, time.perf_counter() - beginning)
        best_index, best_theta = max(candidates, key=lambda x: records[x[0]]["mean_train_log_density"])
        if polish_maxiter and not records[best_index]["score_converged"]:
            attempt(best_theta, "selected_candidate_continuation", polish_maxiter, parent_start=best_index)
        selected, theta = max(candidates, key=lambda x: records[x[0]]["mean_train_log_density"])
    records[selected]["selected"] = True
    elapsed = time.perf_counter() - beginning
    record = dict(records[selected])
    record["total_elapsed_seconds"] = elapsed
    record["fixed_bounds"] = {"location": LOCATION_BOUNDS, "cholesky_diagonal": CHOLESKY_DIAGONAL_BOUNDS,
                              "cholesky_offdiagonal": CHOLESKY_OFFDIAGONAL_BOUNDS, "beta": BETA_BOUNDS}
    record["optimizer_settings"] = {"maxiter": maxiter, "polish_maxiter": polish_maxiter,
                                    "gtol": gtol, "ftol": 1e-15, "maxls": 50, "cpu_threads": 1}
    return SkewNormalFit(parameters(theta, k, d), record, records, selected, elapsed)
