"""Bounded Gaussian mixtures with a shared positive one-factor correlation.

This is an existing restricted Gaussian model used as a scientific comparator,
not a proposed new distribution. Component means and marginal standard
deviations vary; their correlation matrix is common. For d>=3,
R=diag(1-r**2)+r r.T. For d=2 one identifiable rho parameter replaces the
nonidentified pair of loadings. Zero correlation is allowed exactly.
For d>=3, loading identification is generic in the positive interior;
configurations with fewer than three positive loadings are nonidentified.

All bounds apply to training-standardized observations. This module does not
standardize data or inspect validation/test data. No global optimum is claimed.
Supplied diagonal Gaussian candidates are never clipped. If none is usable,
a separately recorded diagonal candidate is built from the first feasible
partition initialization; it is explicitly not a fitted diagonal MLE.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import signal
import time
import warnings

import numpy as np
from scipy.linalg import cho_solve
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

METHOD = "gaussian_shared_correlation"
MEAN_BOUNDS = (-20., 20.)
STD_BOUNDS = (.005, 10.)
MINIMUM_CORRELATION_EIGENVALUE_BOUND = 1e-6
# Public aliases used by the separate companion runner's freeze validation.
SD_BOUNDS = STD_BOUNDS
CORRELATION_EPS = MINIMUM_CORRELATION_EIGENVALUE_BOUND
RHO_UPPER = 1.-MINIMUM_CORRELATION_EIGENVALUE_BOUND
LOADING_UPPER = float(np.sqrt(1.-MINIMUM_CORRELATION_EIGENVALUE_BOUND))
CORRELATION_START_MULTIPLIERS = (.5, 1., 1.5, .25, 2.)
BOUNDARY_TOLERANCE_FRACTION = 1e-4
LOG_TWO_PI = float(np.log(2*np.pi))


def correlation_parameter_count(d):
    if d < 1:
        raise ValueError("Dimension must be positive")
    return 0 if d == 1 else (1 if d == 2 else d)


def parameter_count(k, d):
    if k < 1:
        raise ValueError("Component count must be positive")
    return k-1+2*k*d+correlation_parameter_count(d)


def correlation_matrix(values, d):
    values = np.asarray(values, float)
    if values.shape != (correlation_parameter_count(d),) or not np.isfinite(values).all():
        raise ValueError("Invalid correlation parameter vector")
    if np.any(values < 0) or np.any(values >= 1):
        raise ValueError("Correlation parameters must lie in [0,1)")
    if d == 1:
        return np.ones((1, 1))
    if d == 2:
        return np.array([[1., values[0]], [values[0], 1.]])
    result = np.outer(values, values)
    np.fill_diagonal(result, 1.)
    return result


def bounds(k, d):
    correlation_upper = RHO_UPPER if d == 2 else LOADING_UPPER
    return ([(None, None)]*(k-1)+[MEAN_BOUNDS]*(k*d)
            +[tuple(np.log(STD_BOUNDS))]*(k*d)
            +[(0., correlation_upper)]*correlation_parameter_count(d))


def unpack(theta, k, d):
    theta = np.asarray(theta, float)
    if theta.shape != (parameter_count(k, d),) or not np.isfinite(theta).all():
        raise ValueError("Invalid Gaussian parameter vector")
    offset = k-1
    logits = np.r_[theta[:offset], 0.]
    logpi = logits-logsumexp(logits)
    means = theta[offset:offset+k*d].reshape(k, d)
    std = np.exp(theta[offset+k*d:offset+2*k*d].reshape(k, d))
    correlation_parameters = theta[offset+2*k*d:]
    if not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Standard deviations must be finite and positive")
    R = correlation_matrix(correlation_parameters, d)
    return logpi, means, std, correlation_parameters, R


def pack(weights, means, std, correlation_parameters):
    weights, means, std = map(lambda x: np.asarray(x, float), (weights, means, std))
    if means.ndim != 2:
        raise ValueError("Means must form a k by d matrix")
    k, d = means.shape
    if weights.shape != (k,) or std.shape != means.shape:
        raise ValueError("Incompatible weights, means or standard deviations")
    if not all(np.isfinite(x).all() for x in (weights, means, std)):
        raise ValueError("Parameters must be finite")
    if np.any(weights <= 0) or not np.isclose(weights.sum(), 1., rtol=0, atol=1e-10):
        raise ValueError("Weights must be positive and sum to one")
    if np.any(std <= 0):
        raise ValueError("Standard deviations must be positive")
    correlation_matrix(correlation_parameters, d)
    return np.r_[np.log(weights[:-1])-np.log(weights[-1]), means.ravel(),
                 np.log(std).ravel(), np.asarray(correlation_parameters, float)]


def evaluation(theta, y, k, gradient=False):
    """Log density, responsibilities, and optional summed analytic score.

    With z=(y-mu)/std and v=R^{-1}z, mean score is v/std and
    log-SD score is z*v-1. The common R score is H=1/2 sum(tau vv')-(N/2)R^-1.
    Its rho score is 2H_12; its loading score is 2(H-diag(diag(H)))r.
    """
    y = np.asarray(y, float)
    if y.ndim != 2 or not len(y) or not np.isfinite(y).all():
        raise ValueError("Observations must be a nonempty finite matrix")
    n, d = y.shape
    logpi, means, std, loading, R = unpack(theta, k, d)
    cholesky = np.linalg.cholesky(R)
    inverse = cho_solve((cholesky, True), np.eye(d), check_finite=False)
    logdet = 2*np.log(np.diag(cholesky)).sum()
    terms, cached = np.empty((n, k)), []
    for j in range(k):
        z = (y-means[j])/std[j]
        v = z @ inverse
        terms[:, j] = (logpi[j]-.5*(d*LOG_TWO_PI+logdet)
                       -np.log(std[j]).sum()-.5*np.einsum("ij,ij->i", z, v))
        if gradient:
            cached.append((z, v))
    ell = logsumexp(terms, axis=1)
    tau = np.exp(terms-ell[:, None])
    if not gradient:
        return ell, tau
    mean_score, logstd_score = np.empty((k, d)), np.empty((k, d))
    H = -.5*n*inverse
    for j, (z, v) in enumerate(cached):
        weighted_v = tau[:, j, None]*v
        mean_score[j] = weighted_v.sum(axis=0)/std[j]
        logstd_score[j] = np.sum(tau[:, j, None]*(z*v-1.), axis=0)
        H += .5*weighted_v.T @ v
    if d == 1:
        correlation_score = np.empty(0)
    elif d == 2:
        correlation_score = np.array([H[0, 1]+H[1, 0]])
    else:
        off_diagonal = H.copy()
        np.fill_diagonal(off_diagonal, 0.)
        correlation_score = (off_diagonal+off_diagonal.T) @ loading
    weight_score = tau[:, :k-1].sum(axis=0)-n*np.exp(logpi[:k-1])
    return ell, tau, np.r_[weight_score, mean_score.ravel(), logstd_score.ravel(), correlation_score]


def objective(theta, y, k):
    ell, _, score = evaluation(theta, y, k, gradient=True)
    return -float(ell.mean()), -score/len(y)


def parameters(theta, k, d):
    logpi, means, std, loading, R = unpack(theta, k, d)
    return {"method": METHOD, "k": k, "d": d, "theta": np.asarray(theta).copy(),
            "weights": np.exp(logpi), "means": means.copy(), "std": std.copy(),
            "correlation_parameters": loading.copy(), "correlation": R,
            "covariances": std[:, :, None]*R[None, :, :]*std[:, None, :],
            "correlation_parameterization": "rho" if d == 2 else ("positive_one_factor_loadings" if d >= 3 else "none")}


def evaluate(y, params):
    return evaluation(np.asarray(params["theta"], float), np.asarray(y, float), int(params["k"]))


def evaluate_params(params, y):
    return evaluate(y, params)


def component_means(params):
    return unpack(np.asarray(params["theta"]), int(params["k"]), int(params["d"]))[1].copy()


def diagonal_initial(candidate, k, d):
    """Exact Gaussian-diagonal embedding; bounds are checked without clipping."""
    candidate = candidate.params if hasattr(candidate, "params") else candidate
    means = np.asarray(candidate["means"], float)
    if means.shape != (k, d):
        raise ValueError("Diagonal candidate means have incompatible dimensions")
    if "covariances" in candidate:
        covariance = np.asarray(candidate["covariances"], float)
        if covariance.shape == (k, d):
            variances = covariance
        elif covariance.shape == (k, d, d):
            variances = np.diagonal(covariance, axis1=1, axis2=2)
            diagonal = np.zeros_like(covariance)
            diagonal[:, np.arange(d), np.arange(d)] = variances
            if not np.array_equal(covariance, diagonal):
                raise ValueError("Supplied full covariance is not exactly diagonal")
        else:
            raise ValueError("Diagonal candidate covariance has incompatible dimensions")
        if not np.isfinite(variances).all() or np.any(variances <= 0):
            raise ValueError("Diagonal candidate variances must be positive")
        std = np.sqrt(variances)
    else:
        std = np.asarray(candidate["std"], float)
    return pack(candidate["weights"], means, std, np.zeros(correlation_parameter_count(d)))


def initial_parameters(y, labels, k, start=0):
    """Initialize from within-partition residuals; no true labels are used."""
    d = y.shape[1]
    labels = np.asarray(labels)
    if labels.shape != (len(y),) or np.any(~np.isin(labels, np.arange(k))):
        raise ValueError("Partition must assign every observation to one of k components")
    means, std, weights = [], [], []
    residuals = np.empty_like(y, dtype=float)
    for j in range(k):
        selected = labels == j
        group = y[selected]
        if not len(group):
            raise ValueError("Initialization partition has an empty component")
        empirical_mean = group.mean(axis=0)
        empirical_std = group.std(axis=0, ddof=0)
        sd = np.clip(empirical_std, *STD_BOUNDS)
        means.append(np.clip(empirical_mean, *MEAN_BOUNDS))
        std.append(sd)
        weights.append(len(group)/len(y))
        residuals[selected] = (group-empirical_mean)/sd
    pooled = residuals.T @ residuals/len(y)
    denominator = np.sqrt(np.maximum(np.diag(pooled), 1e-14))
    pooled_correlation = pooled/denominator[:, None]/denominator[None, :]
    observed = float(np.mean(pooled_correlation[np.tril_indices(d, -1)])) if d > 1 else 0.
    # Every optimized start is strictly positive even when empirical residual
    # correlations are negative. The diagonal point is a separate candidate.
    level = CORRELATION_START_MULTIPLIERS[start % len(CORRELATION_START_MULTIPLIERS)]
    initial_rho = float(np.clip(level*np.clip(observed, .02, .9), .01, .95))
    loading = (np.array([initial_rho]) if d == 2 else
               (np.full(d, np.sqrt(initial_rho)) if d >= 3 else np.empty(0)))
    theta = pack(weights, means, std, loading)
    details = {"pooled_within_partition_mean_correlation": observed,
               "initial_equicorrelation": initial_rho if d > 1 else 0.,
               "correlation_start_multiplier": level,
               "partition_sha256": hashlib.sha256(np.asarray(labels, dtype=np.int64).tobytes()).hexdigest()}
    return theta, details


def _box_arrays(limits):
    return (np.array([-np.inf if lo is None else lo for lo, _ in limits]),
            np.array([np.inf if hi is None else hi for _, hi in limits]))


def projected_gradient(theta, gradient, limits):
    lower, upper = _box_arrays(limits)
    return float(np.max(np.abs(theta-np.clip(theta-gradient, lower, upper))))


def boundary_hits(theta, k, d):
    names = ([f"weight_logit[{j}]" for j in range(k-1)]
             +[f"mean[{j},{m}]" for j in range(k) for m in range(d)]
             +[f"log_std[{j},{m}]" for j in range(k) for m in range(d)]
             +(["rho"] if d == 2 else [f"loading[{m}]" for m in range(correlation_parameter_count(d))]))
    result = []
    for name, value, (lo, hi) in zip(names, theta, bounds(k, d)):
        if lo is None:
            continue
        tolerance = BOUNDARY_TOLERANCE_FRACTION*(hi-lo)
        if value-lo <= tolerance:
            result.append(name+":lower")
        elif hi-value <= tolerance:
            result.append(name+":upper")
    return result


@contextmanager
def _deadline(seconds):
    if seconds is None:
        yield
        return
    if seconds <= 0:
        raise ValueError("Start timeout must be positive")
    def expired(signum, frame):
        raise TimeoutError(f"Prespecified per-start timeout ({seconds}s)")
    old_handler = signal.signal(signal.SIGALRM, expired)
    previous = signal.setitimer(signal.ITIMER_REAL, seconds)
    started = time.monotonic()
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        if previous[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, max(previous[0]-(time.monotonic()-started), 1e-9), previous[1])


@dataclass
class GaussianSharedCorrelationFit:
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


class GaussianSharedCorrelationFitFailure(RuntimeError):
    def __init__(self, message, starts, elapsed_seconds):
        super().__init__(message)
        self.starts = self.records = starts
        self.elapsed_seconds = elapsed_seconds


def fit(y, k, seed=20261107, partitions=None, diagonal_candidate=None,
        n_starts=3, maxiter=3000, polish_maxiter=2000, gtol=1e-6,
        ftol=1e-15, start_timeout_seconds=None, maxls=50):
    """Fit fixed training data and retain every start and exact candidate.

    Supplied partitions determine the actual number of partition starts;
    n_starts controls their count only when they are generated here.
    A continuation is a separate record and cannot overwrite its parent.
    Candidate and continuation selection uses training likelihood only.
    """
    beginning = time.perf_counter()
    y = np.asarray(y, float)
    if y.ndim != 2 or not len(y) or not y.shape[1] or not np.isfinite(y).all():
        raise ValueError("Training observations must be a nonempty finite matrix")
    if not isinstance(k, (int, np.integer)) or not 1 <= k <= len(y):
        raise ValueError("Component count must be an integer in [1,N]")
    if n_starts < 1 or maxiter < 1 or polish_maxiter < 0 or min(gtol, ftol) <= 0 or maxls < 1:
        raise ValueError("Invalid optimizer settings")
    if start_timeout_seconds is not None and start_timeout_seconds <= 0:
        raise ValueError("Start timeout must be positive")
    d = y.shape[1]
    limits = bounds(k, d)
    lower, upper = _box_arrays(limits)
    records, candidates = [], []
    fallback_diagonal = []
    if partitions is not None:
        partitions = list(partitions)
        if not partitions:
            raise ValueError("At least one partition is required")
    actual_starts = n_starts if partitions is None else len(partitions)

    def attempt(initializer, reason, iterations=0, optimize=True, parent_start=None):
        index, started = len(records), time.perf_counter()
        rec = {"start": index, "initialization": reason, "parent_start": parent_start,
               "failed": True, "selected": False, "parameters": None,
               "optimizer_success": False, "score_converged": False,
               "projected_gradient": None, "mean_train_log_density": None,
               "boundary_hits": None, "boundary_hit": None, "min_weight": None,
               "parameter_count": parameter_count(k, d), "iterations": 0,
               "function_evaluations": 0, "warnings": [], "message": ""}
        try:
            with _deadline(start_timeout_seconds):
                initial = initializer() if callable(initializer) else np.asarray(initializer).copy()
                if isinstance(initial, tuple):
                    initial, details = initial
                    rec.update(details)
                initial = np.asarray(initial, float)
                rec["initial_parameters"] = initial.copy()
                if initial.shape != lower.shape or not np.isfinite(initial).all():
                    raise ValueError("Invalid initial parameter vector")
                if np.any(initial < lower) or np.any(initial > upper):
                    raise ValueError("Candidate outside fixed bounds; exact candidate was not clipped")
                value0, gradient0 = objective(initial, y, k)
                if not np.isfinite(value0) or not np.isfinite(gradient0).all():
                    raise FloatingPointError("Nonfinite initial objective or gradient")
                rec["initial_mean_train_log_density"] = -value0
                if reason.startswith("partition_start") and not fallback_diagonal:
                    diagonal = initial.copy()
                    if correlation_parameter_count(d):
                        diagonal[-correlation_parameter_count(d):] = 0.
                    fallback_diagonal.append((diagonal, index))
                theta, value, gradient = initial, value0, gradient0
                if optimize:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        result = minimize(objective, initial, args=(y, k), jac=True,
                                          method="L-BFGS-B", bounds=limits,
                                          options={"maxiter": iterations, "maxls": maxls,
                                                   "ftol": ftol, "gtol": gtol})
                    rec.update(optimizer_success=bool(result.success), iterations=int(result.nit),
                               function_evaluations=int(result.nfev), message=str(result.message),
                               warnings=[str(w.message) for w in caught])
                    theta = np.asarray(result.x, float)
                    rec["optimizer_endpoint_theta"] = theta.copy()
                    if theta.shape != initial.shape or not np.isfinite(theta).all():
                        raise FloatingPointError("Nonfinite or malformed optimizer endpoint")
                    if np.any(theta < lower) or np.any(theta > upper):
                        raise ValueError("Optimizer endpoint lies outside the fixed bounds")
                    value, gradient = objective(theta, y, k)
                    if not np.isfinite(value) or not np.isfinite(gradient).all():
                        raise FloatingPointError("Nonfinite fitted objective or gradient")
                    rec["optimizer_endpoint_parameters"] = parameters(theta, k, d)
                    rec["optimizer_endpoint_mean_train_log_density"] = -value
                    if value > value0:
                        theta, value, gradient = initial, value0, gradient0
                        rec["discarded_worse_optimizer_endpoint"] = True
                        rec["optimizer_success"] = False
                        rec["message"] += "; retained superior initial point"
                else:
                    rec["message"] = "Exact zero-correlation candidate; not optimized in this record"
                    if reason == "partition_diagonal_candidate":
                        rec["message"] += "; partition initialization, not a fitted diagonal MLE"
                pg = projected_gradient(theta, gradient, limits)
                hits = boundary_hits(theta, k, d)
                params = parameters(theta, k, d)
                rec.update(failed=False, parameters=params, projected_gradient=pg,
                           score_converged=bool(pg <= gtol), mean_train_log_density=-value,
                           boundary_hits=hits, boundary_hit=bool(hits),
                           min_weight=float(np.min(params["weights"])),
                           small_weight=bool(np.min(params["weights"]) < 1e-8),
                           correlation_loading_identified=bool(d <= 2 or np.count_nonzero(params["correlation_parameters"] > 0) >= 3),
                           minimum_correlation_eigenvalue=float(np.linalg.eigvalsh(params["correlation"])[0]))
                candidates.append((index, theta.copy()))
        except Exception as exc:
            rec.update(failed=True, message=repr(exc), exception_type=type(exc).__name__, optimizer_success=False)
        rec["elapsed_seconds"] = time.perf_counter()-started
        records.append(rec)
        return index

    with threadpool_limits(1):
        for start in range(actual_starts):
            def initializer(start=start):
                labels = (KMeans(k, n_init=1, random_state=int(seed+start)).fit_predict(y)
                          if partitions is None else partitions[start])
                return initial_parameters(y, labels, k, start)
            attempt(initializer, f"partition_start_{start}", maxiter)
        external_usable = False
        if diagonal_candidate is not None:
            index = attempt(lambda: diagonal_initial(diagonal_candidate, k, d),
                            "exact_diagonal_candidate", optimize=False)
            external_usable = not records[index]["failed"]
        if not external_usable and fallback_diagonal:
            theta, parent = fallback_diagonal[0]
            attempt(theta, "partition_diagonal_candidate", optimize=False, parent_start=parent)
        if not candidates:
            raise GaussianSharedCorrelationFitFailure(
                "Every shared-correlation Gaussian start and candidate failed", records,
                time.perf_counter()-beginning)
        best, theta = max(candidates, key=lambda item: records[item[0]]["mean_train_log_density"])
        if polish_maxiter and not records[best]["score_converged"]:
            attempt(theta, "selected_candidate_continuation", polish_maxiter, parent_start=best)
        selected, theta = max(candidates, key=lambda item: records[item[0]]["mean_train_log_density"])
    records[selected]["selected"] = True
    elapsed = time.perf_counter()-beginning
    record = dict(records[selected])
    record.update(total_elapsed_seconds=elapsed, fit_seconds=elapsed,
                  actual_partition_starts=actual_starts,
                  attempted_records=len(records), failed_records=sum(r["failed"] for r in records),
                  external_diagonal_candidate_supplied=diagonal_candidate is not None,
                  external_diagonal_candidate_usable=external_usable,
                  fixed_bounds={"means": MEAN_BOUNDS, "standard_deviations": STD_BOUNDS,
                                "correlation_parameters": (0., RHO_UPPER if d == 2 else LOADING_UPPER),
                                "correlation_eigenvalue_lower_bound": MINIMUM_CORRELATION_EIGENVALUE_BOUND,
                                "near_boundary_tolerance_fraction": BOUNDARY_TOLERANCE_FRACTION},
                  optimizer_settings={"n_starts_requested": n_starts, "maxiter": maxiter,
                                      "polish_maxiter": polish_maxiter, "gtol": gtol,
                                      "ftol": ftol, "maxls": maxls,
                                      "start_timeout_seconds": start_timeout_seconds, "threads": 1})
    return GaussianSharedCorrelationFit(parameters(theta, k, d), record, records, selected, elapsed)
