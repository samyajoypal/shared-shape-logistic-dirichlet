"""Research implementation: LD mixtures with a common unknown shape vector.

Two models are supported: component-specific diagonal scales, and common
diagonal scales. Locations always vary by component. This implements the declared bounded numerical models.

The direct optimizer and ECM use exactly the same bounded alpha, inverse
scale a, and scaled location b coordinates. Weights use free logits.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import digamma, gammaln, logsumexp, polygamma
from sklearn.cluster import KMeans

ALPHA_BOUNDS = (0.15, 60.0)
A_BOUNDS = (0.15, 8.0)
B_BOUNDS = (-40.0, 40.0)


def log_simplex(z):
    augmented = np.concatenate((z, np.zeros(z.shape[:-1] + (1,))), axis=-1)
    return augmented - logsumexp(augmented, axis=-1, keepdims=True)


def dimensions(k, d, common_scale):
    return k - 1, d + 1, (1 if common_scale else k) * d, k * d


def unpack(theta, k, d, common_scale):
    r, p, na, _ = dimensions(k, d, common_scale)
    logits = np.r_[theta[:r], 0.0]
    logpi = logits - logsumexp(logits)
    alpha = theta[r:r+p]
    a = theta[r+p:r+p+na].reshape(-1, d)
    b = theta[r+p+na:].reshape(k, d)
    return logpi, alpha, a, b


def bounds(k, d, common_scale):
    r, p, na, nb = dimensions(k, d, common_scale)
    return [(None, None)] * r + [ALPHA_BOUNDS] * p + [A_BOUNDS] * na + [B_BOUNDS] * nb


def evaluation(theta, y, k, common_scale=False, gradient=False):
    n, d = y.shape
    logpi, alpha, a, b = unpack(theta, k, d, common_scale)
    z = y[:, None, :] * a[None, :, :] + b[None, :, :]
    lq = log_simplex(z)
    logf = (gammaln(alpha.sum()) - gammaln(alpha).sum()
            + np.log(a).sum(axis=1)[None, :] + lq @ alpha)
    terms = logf + logpi
    ell = logsumexp(terms, axis=1)
    tau = np.exp(terms - ell[:, None])
    if not gradient:
        return ell, tau
    v = alpha[:d] - alpha.sum() * np.exp(lq[:, :, :d])
    ga = (tau[:, :, None] * (1.0 / a + y[:, None, :] * v)).sum(axis=0)
    if common_scale:
        ga = ga.sum(axis=0, keepdims=True)
    gb = (tau[:, :, None] * v).sum(axis=0)
    gshape = (tau[:, :, None] * (digamma(alpha.sum()) - digamma(alpha) + lq)).sum(axis=(0, 1))
    gw = tau[:, :k-1].sum(axis=0) - n * np.exp(logpi[:k-1])
    return ell, tau, np.r_[gw, gshape, ga.ravel(), gb.ravel()]


def projected_gradient(theta, grad, limits):
    """Infinity norm of theta - projection(theta-grad), for minimizing."""
    lower = np.array([-np.inf if lo is None else lo for lo, _ in limits])
    upper = np.array([np.inf if hi is None else hi for _, hi in limits])
    return float(np.max(np.abs(theta - np.clip(theta - grad, lower, upper))))


def bound_hits(theta, limits):
    hits = []
    for index, (value, (lower, upper)) in enumerate(zip(theta, limits)):
        if lower is None or upper is None:
            continue
        tol = 1e-4 * (upper - lower)
        if value - lower <= tol:
            hits.append(f"{index}:lower")
        elif upper - value <= tol:
            hits.append(f"{index}:upper")
    return hits


def initial_parameters(y, labels, k, common_scale=False, shape_level=2.0):
    d = y.shape[1]
    alpha = np.full(d+1, shape_level)
    kernel_mean = digamma(alpha[:d]) - digamma(alpha[-1])
    kernel_variance = polygamma(1, alpha[:d]) + polygamma(1, alpha[-1])
    means = np.array([y[labels == j].mean(axis=0) for j in range(k)])
    variances = np.array([y[labels == j].var(axis=0) for j in range(k)])
    weights = np.array([(labels == j).mean() for j in range(k)])
    if common_scale:
        variances = (weights[:, None] * variances).sum(axis=0, keepdims=True)
    a = np.clip(np.sqrt(kernel_variance / np.maximum(variances, 1e-5)),
                A_BOUNDS[0] * 1.05, A_BOUNDS[1] / 1.05)
    b = np.clip(kernel_mean - means * a, B_BOUNDS[0] + 1, B_BOUNDS[1] - 1)
    return np.r_[np.log(weights[:-1] / weights[-1]), alpha, a.ravel(), b.ravel()]


def make_starts(y, k, n_starts, seed, common_scale=False):
    result = []
    for start in range(n_starts):
        labels = KMeans(k, n_init=1, random_state=seed+start).fit_predict(y)
        result.append(initial_parameters(y, labels, k, common_scale,
                                         shape_level=(1.0, 2.5, 5.0)[start % 3]))
    return result


def affine_q(parameters, y, tau, alpha, common_scale):
    """Negative conditional objective and score, scaled by N."""
    n, d = y.shape
    k = tau.shape[1]
    na = (1 if common_scale else k) * d
    a = parameters[:na].reshape(-1, d)
    b = parameters[na:].reshape(k, d)
    lq = log_simplex(y[:, None, :] * a[None, :, :] + b[None, :, :])
    value = (tau * (np.log(a).sum(axis=1)[None, :] + lq @ alpha)).sum() / n
    v = alpha[:d] - alpha.sum() * np.exp(lq[:, :, :d])
    ga = (tau[:, :, None] * (1/a + y[:, None, :] * v)).sum(axis=0) / n
    if common_scale:
        ga = ga.sum(axis=0, keepdims=True)
    gb = (tau[:, :, None] * v).sum(axis=0) / n
    return -value, -np.r_[ga.ravel(), gb.ravel()]


def run_ecm(y, k, initial, common_scale, maxiter, gtol):
    n, d = y.shape
    r, p, na, nb = dimensions(k, d, common_scale)
    limits = bounds(k, d, common_scale)
    theta = initial.copy()
    history = [float(evaluation(theta, y, k, common_scale)[0].mean())]
    block_failures = 0
    for iteration in range(maxiter):
        _, tau = evaluation(theta, y, k, common_scale)
        logpi, alpha, a, b = unpack(theta, k, d, common_scale)
        proportions = np.maximum(tau.mean(axis=0), 1e-100)
        theta[:r] = np.log(proportions[:-1] / proportions[-1])
        lq = log_simplex(y[:, None, :] * a[None, :, :] + b[None, :, :])
        mean_lq = (tau[:, :, None] * lq).sum(axis=(0, 1)) / n

        def shape_objective(shape):
            return (-(gammaln(shape.sum()) - gammaln(shape).sum() + mean_lq @ shape),
                    -(digamma(shape.sum()) - digamma(shape) + mean_lq))

        shape_result = minimize(shape_objective, alpha, jac=True, method="L-BFGS-B",
                                bounds=[ALPHA_BOUNDS] * p,
                                options={"maxiter": 200, "ftol": 1e-14, "gtol": gtol/10})
        if shape_result.fun <= shape_objective(alpha)[0] + 1e-12:
            theta[r:r+p] = shape_result.x
        else:
            block_failures += 1
        affine_start = theta[r+p:].copy()
        affine_result = minimize(affine_q, affine_start,
                                 args=(y, tau, theta[r:r+p], common_scale),
                                 jac=True, method="L-BFGS-B", bounds=limits[r+p:],
                                 options={"maxiter": 200, "ftol": 1e-14, "gtol": gtol/10})
        if affine_result.fun <= affine_q(affine_start, y, tau, theta[r:r+p], common_scale)[0] + 1e-12:
            theta[r+p:] = affine_result.x
        else:
            block_failures += 1
        ell, _, score = evaluation(theta, y, k, common_scale, True)
        history.append(float(ell.mean()))
        if history[-1] < history[-2] - 1e-9:
            raise RuntimeError("ECM likelihood decreased")
        pg = projected_gradient(theta, -score/n, limits)
        if pg <= gtol:
            break
    return theta, {
        "optimizer_success": pg <= gtol, "iterations": iteration+1,
        "message": "projected score tolerance" if pg <= gtol else "iteration cap",
        "history": history, "block_failures": block_failures,
    }


@dataclass
class SharedShapeFit:
    theta: np.ndarray
    k: int
    common_scale: bool
    algorithm: str
    records: list
    elapsed_seconds: float
    selected_start: int

    def evaluate(self, y):
        return evaluation(self.theta, y, self.k, self.common_scale)

    @property
    def record(self):
        return self.records[self.selected_start]


class SharedShapeFitFailure(RuntimeError):
    """No finite start survived; retain every attempted start and its cost."""

    def __init__(self, message, records, elapsed_seconds):
        super().__init__(message)
        self.records = records
        self.elapsed_seconds = elapsed_seconds


def fit(y, k, common_scale=False, algorithm="direct", n_starts=3,
        seed=20261106, maxiter=800, gtol=1e-6, starts=None):
    beginning = time.perf_counter()
    if algorithm not in ("direct", "ecm"):
        raise ValueError(algorithm)
    limits = bounds(k, y.shape[1], common_scale)
    starts = make_starts(y, k, n_starts, seed, common_scale) if starts is None else starts
    candidates, records = [], []

    def objective(theta):
        ell, _, score = evaluation(theta, y, k, common_scale, True)
        return -ell.mean(), -score/len(y)

    for index, initial in enumerate(starts):
        t0 = time.perf_counter()
        record = {"start": index, "failed": True, "selected": False,
                  "mean_train_log_density": np.nan, "projected_gradient": np.nan,
                  "score_converged": False, "boundary_hits": [],
                  "min_weight": np.nan, "optimizer_success": False,
                  "iterations": 0, "message": ""}
        try:
            if algorithm == "direct":
                result = minimize(objective, initial, jac=True, method="L-BFGS-B", bounds=limits,
                                  options={"maxiter": maxiter, "maxls": 40,
                                           "ftol": 1e-14, "gtol": gtol})
                theta = result.x
                detail = {"optimizer_success": bool(result.success), "iterations": int(result.nit),
                          "message": str(result.message)}
            else:
                theta, detail = run_ecm(y, k, initial, common_scale, maxiter, gtol)
            record.update(detail)
            if not np.isfinite(theta).all():
                raise FloatingPointError("Nonfinite fitted parameters")
            value, gradient = objective(theta)
            record["mean_train_log_density"] = -float(value)
            if not np.isfinite(value) or not np.isfinite(gradient).all():
                raise FloatingPointError("Nonfinite fitted objective or score")
            pg = projected_gradient(theta, gradient, limits)
            if not np.isfinite(pg):
                raise FloatingPointError("Nonfinite projected score")
            logpi = unpack(theta, k, y.shape[1], common_scale)[0]
            record.update({"failed": False, "projected_gradient": pg,
                           "score_converged": pg <= gtol,
                           "boundary_hits": bound_hits(theta, limits),
                           "min_weight": float(np.exp(logpi).min())})
            candidates.append((index, theta))
        except Exception as exc:
            record.update({"failed": True, "optimizer_success": False,
                           "message": repr(exc), "exception_type": type(exc).__name__})
        record["elapsed_seconds"] = time.perf_counter()-t0
        records.append(record)
    if not candidates:
        raise SharedShapeFitFailure("No finite optimization start", records,
                                    time.perf_counter()-beginning)
    best, theta = max(candidates, key=lambda item: records[item[0]]["mean_train_log_density"])
    records[best]["selected"] = True
    return SharedShapeFit(theta, k, common_scale, algorithm, records,
                          time.perf_counter()-beginning, best)


def simulate(n, alpha, locations, scales, proportions, rng):
    labels = rng.choice(len(proportions), n, p=proportions)
    gamma = rng.gamma(np.asarray(alpha), 1.0, (n, len(alpha)))
    w = np.log(gamma[:, :-1]) - np.log(gamma[:, -1:])
    y = np.asarray(locations)[labels] + np.asarray(scales)[labels] * w
    return y, labels
