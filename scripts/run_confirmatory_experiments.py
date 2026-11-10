"""Reproducible expanded benchmark, not a statistically confirmatory study.

Reusable API:
  fit_method(method, y, k, seed, settings=None, partitions=None, warm_start=None)
returns FitResult with evaluate(y), component_means(), record, starts, params,
and selected_start. Defaults are in configs/experiments.json['fitting'].
Methods: shared_ld, full_ld, base_ld, gaussian_full, gaussian_diag,
fmvmm_t, fmvmm_skew_normal, fmvmm_dmm. Bounds can be overridden in settings.

The full-model warm_start may be a shared FitResult: it supplies a predeclared
nesting safeguard, used only if the ordinary full fit has worse training fit.
A DMM FitResult may supply an additional base_ld start, for compositional data.
No installed package files or earlier research scripts are modified.
"""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import time
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ld-mpl")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.linalg import sqrtm
from scipy.optimize import linear_sum_assignment, minimize
from scipy.special import digamma, gammaln, log_ndtr, logsumexp, polygamma
from scipy.stats import multivariate_normal, multivariate_t, norm
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits

import shared_shape_ld as shared
from fmvmm.mixtures.DMM_Soft import DMM_Soft, estimate_alphas as dmm_mstep
from fmvmm.mixtures.tmix_smsn import TMix
from fmvmm.mixtures.skewnormmix_smsn import SkewNormalMix, estimate_alphas_skewnormal, dmvSN
from fmvmm.mixsmsn.dens import dmvt_ls, d_mixedmvST

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/experiments.json"
DEFAULT_SETTINGS = json.loads(CONFIG_PATH.read_text())["fitting"]


def serializable(value):
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return serializable(value.tolist())
    if isinstance(value, np.generic):
        return serializable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serializable(value), indent=2, allow_nan=False) + "\n")


@contextlib.contextmanager
def timed_start(seconds):
    def expired(signum, frame):
        raise TimeoutError(f"Prespecified per-start timeout ({seconds}s)")
    old = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def make_partitions(y, k, seed, n_starts=3):
    return [KMeans(k, n_init=1, random_state=int(seed+s)).fit_predict(y) for s in range(n_starts)]


def ld_bounds(method, k, d, settings):
    shapes = d+1 if method == "shared_ld" else k*(d+1)
    bounds = [(None, None)]*(k-1) + [tuple(settings["alpha_bounds"])]*shapes
    if method != "base_ld":
        bounds += [tuple(settings["inverse_scale_bounds"])]*(k*d)
        bounds += [tuple(settings["scaled_location_bounds"])]*(k*d)
    return bounds


def unpack_ld(theta, method, k, d):
    r, p = k-1, d+1
    logits = np.r_[theta[:r], 0.0]
    logpi = logits-logsumexp(logits)
    na = p if method == "shared_ld" else k*p
    alpha = theta[r:r+na].reshape(-1, p)
    if method == "base_ld":
        a, b = np.ones((k, d)), np.zeros((k, d))
    else:
        a = theta[r+na:r+na+k*d].reshape(k, d)
        b = theta[r+na+k*d:].reshape(k, d)
    return logpi, alpha, a, b


def ld_evaluate(theta, y, k, method, gradient=False):
    """Raw shapes and affine inverse coordinates; same coordinates for all LD fits."""
    if method == "shared_ld":
        return shared.evaluation(theta, y, k, common_scale=False, gradient=gradient)
    logpi, alpha, a, b = unpack_ld(theta, method, k, y.shape[1])
    total = alpha.sum(axis=1)
    z = y[:, None, :]*a[None, :, :]+b[None, :, :]
    lq = shared.log_simplex(z)
    terms = (gammaln(total)-gammaln(alpha).sum(axis=1)+np.log(a).sum(axis=1)
             +(lq*alpha[None, :, :]).sum(axis=2)+logpi)
    ell = logsumexp(terms, axis=1)
    tau = np.exp(terms-ell[:, None])
    if not gradient:
        return ell, tau
    gweight = tau[:, :k-1].sum(axis=0)-len(y)*np.exp(logpi[:k-1])
    gshape = (tau[:, :, None]*(digamma(total)[None, :, None]-digamma(alpha)[None, :, :]+lq)).sum(axis=0)
    if method == "base_ld":
        return ell, tau, np.r_[gweight, gshape.ravel()]
    v = alpha[None, :, :-1]-total[None, :, None]*np.exp(lq[:, :, :-1])
    ga = (tau[:, :, None]*(1/a[None, :, :]+y[:, None, :]*v)).sum(axis=0)
    gb = (tau[:, :, None]*v).sum(axis=0)
    return ell, tau, np.r_[gweight, gshape.ravel(), ga.ravel(), gb.ravel()]


def initial_ld(y, labels, k, method, settings, start):
    d, p = y.shape[1], y.shape[1]+1
    weights = np.array([(labels == j).mean() for j in range(k)])
    if np.any(weights == 0):
        raise ValueError("Empty starting cluster")
    logits = np.log(weights[:-1]/weights[-1])
    lo, hi = settings["alpha_bounds"]
    if method == "base_ld":
        x = np.exp(shared.log_simplex(y))
        values = []
        for j in range(k):
            group = x[labels == j]
            m, v = group.mean(0), group.var(0)
            concentration = np.median(m*(1-m)/np.maximum(v, 1e-10)-1)
            values.append(np.clip(m*max(concentration, 0.5), lo, hi))
        return np.r_[logits, np.asarray(values).ravel()]
    level = settings["shape_levels"][start % len(settings["shape_levels"])]
    alpha = np.full(p, np.clip(level, lo, hi))
    means = np.array([y[labels == j].mean(0) for j in range(k)])
    variances = np.array([y[labels == j].var(0) for j in range(k)])
    kernel_var = polygamma(1, alpha[:-1])+polygamma(1, alpha[-1])
    amin, amax = settings["inverse_scale_bounds"]
    a = np.clip(np.sqrt(kernel_var/np.maximum(variances, 1e-5)), amin*1.05, amax/1.05)
    b = digamma(alpha[:-1])-digamma(alpha[-1])-means*a
    blo, bhi = settings["scaled_location_bounds"]
    b = np.clip(b, blo+1, bhi-1)
    shapes = alpha if method == "shared_ld" else np.tile(alpha, k)
    return np.r_[logits, shapes, a.ravel(), b.ravel()]


def embed_shared(theta, k, d):
    r, p = k-1, d+1
    return np.r_[theta[:r], np.tile(theta[r:r+p], k), theta[r+p:]]


def optimize_ld(initial, y, k, method, settings):
    bounds = ld_bounds(method, k, y.shape[1], settings)
    def objective(theta):
        ell, _, gradient = ld_evaluate(theta, y, k, method, True)
        return -ell.mean(), -gradient/len(y)
    theta, stages = initial.copy(), []
    for stage in range(settings["refinements"]+1):
        result = minimize(objective, theta, jac=True, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": settings["maxiter"] if stage == 0 else settings["refinement_maxiter"],
                                   "ftol": settings["ftol"], "gtol": settings["gtol"], "maxls": settings["maxls"]})
        if not np.isfinite(result.fun) or not np.all(np.isfinite(result.x)):
            raise FloatingPointError("Nonfinite LD optimizer result")
        # Never lose a feasible warm-start objective to an incomplete solve.
        if result.fun <= objective(theta)[0]+1e-13:
            theta = result.x
        value, gradient = objective(theta)
        pg = shared.projected_gradient(theta, gradient, bounds)
        stages.append({"optimizer_success": bool(result.success), "iterations": int(result.nit),
                       "message": str(result.message), "projected_gradient": pg, "objective": float(value)})
        if pg <= settings["gtol"]:
            break
    logpi, _, _, _ = unpack_ld(theta, method, k, y.shape[1])
    record = {"mean_train_log_density": -float(value), "optimizer_success": stages[-1]["optimizer_success"],
              "score_converged": pg <= settings["gtol"], "projected_gradient": pg,
              "iterations": sum(s["iterations"] for s in stages), "stages": stages,
              "message": stages[-1]["message"], "boundary_hits": shared.bound_hits(theta, bounds),
              "min_weight": float(np.exp(logpi).min()), "parameter_count": len(theta)}
    record["boundary_hit"] = bool(record["boundary_hits"])
    return {"theta": theta, "method": method, "k": k, "d": y.shape[1]}, record


def normal_cluster_parameters(y, labels, k, reg):
    weights, means, covariances = [], [], []
    for j in range(k):
        group = y[labels == j]
        weights.append(len(group)/len(y))
        means.append(group.mean(0))
        covariances.append(np.atleast_2d(np.cov(group, rowvar=False, bias=True))+reg*np.eye(y.shape[1]))
    return np.array(weights), np.array(means), np.array(covariances)


def skew_logpdf(y, mu, covariance, shape):
    # Same epsilon convention as the installed fmvmm skew-normal density.
    root = sqrtm(covariance+1e-12*np.eye(y.shape[1]))
    if np.iscomplexobj(root):
        raise FloatingPointError("Complex skew-normal scale root")
    beta = shape @ np.linalg.inv(root)
    return np.log(2)+multivariate_normal.logpdf(y, mean=mu, cov=covariance)+log_ndtr((y-mu)@beta)


def fast_skew_mstep(x, tau, previous):
    """Vectorized algebra of fmvmm estimate_alphas_skewnormal; checked in preflight."""
    d, answer = x.shape[1], []
    for j, (mu, covariance, shape) in enumerate(previous):
        delta = sqrtm(covariance+1e-12*np.eye(d)) @ (shape/np.sqrt(1+shape@shape))
        gamma = covariance-np.outer(delta, delta)
        inv_gamma = np.linalg.inv(gamma+1e-12*np.eye(d))
        mt2 = 1/(1+delta@inv_gamma@delta)
        mut = mt2*((x-mu)@inv_gamma@delta)
        z = mut/np.sqrt(mt2)
        ratio = np.exp(norm.logpdf(z)-log_ndtr(z))
        e = mut+np.sqrt(mt2)*ratio
        e2 = mut*mut+mt2+np.sqrt(mt2)*mut*ratio
        w = tau[:, j]
        mass, s2, s3 = w.sum(), w@e, w@e2
        new_mu = (w@x-s2*delta)/(mass+1e-300) if mass >= 1e-15 else mu
        centered = x-new_mu
        weighted_cross = (w*e)@centered
        new_delta = weighted_cross/(s3+1e-300)
        new_gamma = ((centered*w[:, None]).T@centered
                     -np.outer(new_delta, weighted_cross)-np.outer(weighted_cross, new_delta)
                     +s3*np.outer(new_delta, new_delta))/(mass+1e-300)
        new_covariance = new_gamma+np.outer(new_delta, new_delta)
        new_covariance = (new_covariance+new_covariance.T)/2
        numerator = np.linalg.solve(sqrtm(new_covariance+1e-12*np.eye(d)), new_delta)
        denom = np.sqrt(max(1e-300, 1-new_delta@np.linalg.solve(new_covariance+1e-15*np.eye(d), new_delta)))
        answer.append((new_mu, new_covariance, numerator/denom))
    return answer


class VectorizedSkewNormal(SkewNormalMix):
    def _log_pdf_skewnormal(self, x, alphas):
        return np.column_stack([skew_logpdf(x, *a) for a in alphas])

    def _estimate_alphas_wrapper(self, x, tau, previous, **kwargs):
        return fast_skew_mstep(x, tau, previous)


class VectorizedDMM(DMM_Soft):
    def _log_pdf_dirichlet(self, x, alphas):
        alpha = np.asarray(alphas)
        return np.log(x)@(alpha-1).T+gammaln(alpha.sum(1))-gammaln(alpha).sum(1)


def evaluate_params(params, y):
    method, k = params["method"], params["k"]
    if method in ["shared_ld", "full_ld", "base_ld"]:
        return ld_evaluate(np.asarray(params["theta"]), y, k, method)
    pi = np.asarray(params["weights"])
    if method == "fmvmm_dmm":
        alpha = np.asarray(params["alpha"])
        terms = shared.log_simplex(y)@alpha.T+gammaln(alpha.sum(1))-gammaln(alpha).sum(1)+np.log(pi)
    elif method == "fmvmm_t":
        terms = np.column_stack([multivariate_t.logpdf(y, loc=a[0], shape=a[1], df=a[3]) for a in params["components"]])+np.log(pi)
    elif method == "fmvmm_skew_normal":
        terms = np.column_stack([skew_logpdf(y, *[np.asarray(v) for v in a]) for a in params["components"]])+np.log(pi)
    else:
        means, covariances = np.asarray(params["means"]), np.asarray(params["covariances"])
        terms = np.column_stack([multivariate_normal.logpdf(y, mean=means[j], cov=np.diag(covariances[j]) if method == "gaussian_diag" else covariances[j]) for j in range(k)])+np.log(pi)
    ell = logsumexp(terms, axis=1)
    return ell, np.exp(terms-ell[:, None])


def fit_package(method, y, k, labels, settings):
    d = y.shape[1]
    pi, means, covariances = normal_cluster_parameters(y, labels, k, settings["covariance_regularization"])
    if method == "fmvmm_dmm":
        initial = initial_ld(y, labels, k, "base_ld", settings, 0)
        pi = np.exp(unpack_ld(initial, "base_ld", k, d)[0])
        alphas = initial[k-1:].reshape(k, d+1).tolist()
        x = np.exp(shared.log_simplex(y))
        if np.any(x <= 0):
            raise FloatingPointError("Inverse ALR underflow; no pseudocount used")
        model = VectorizedDMM(k, method="meanprecision", tol=settings["package_tol"], max_iter=settings["package_maxiter"], verbose=False)
        model.data, model.n, model.p = x, len(x), d+1
        model.pi_temp, model.alpha_temp = pi, alphas
        weights, fitted, loglik, _ = model._fit(x, pi, alphas, dmm_mstep, method="meanprecision", true_m=None, true_s=None)
        params = {"method": method, "k": k, "d": d, "weights": weights, "alpha": fitted}
        target_loglik = loglik+shared.log_simplex(y).sum()
        count, hits = k-1+k*(d+1), None
    else:
        if method == "fmvmm_t":
            model = TMix(k, tol=settings["package_tol"], max_iter=settings["package_maxiter"], verbose=False)
            alphas = [(means[j], covariances[j], np.zeros(d), 4.0) for j in range(k)]
            model.dmvt_ls_func, model.d_mixedmvST_func = dmvt_ls, d_mixedmvST
            count = k-1+k*(d+d*(d+1)//2)+1
        else:
            model = VectorizedSkewNormal(k, tol=settings["package_tol"], max_iter=settings["package_maxiter"], verbose=False)
            alphas = [(means[j], covariances[j], np.sign(((y[labels == j]-means[j])**3).sum(0))) for j in range(k)]
            count = k-1+k*(2*d+d*(d+1)//2)
        model.data, model.n, model.p = y, len(y), d
        model.pi_temp, model.alpha_temp = pi, alphas
        weights, fitted, loglik, _ = model._fit(y, pi, alphas, model._estimate_alphas_wrapper)
        params = {"method": method, "k": k, "d": d, "weights": weights, "components": fitted}
        target_loglik = loglik
        hits = []
        if method == "fmvmm_t" and min(fitted[0][3]-1e-4, 100-fitted[0][3]) <= .01:
            hits.append("common_df_bound")
    history = np.asarray(model.log_likelihoods)
    relative = abs(history[-1]-history[-2])/max(abs(history[-2]), np.finfo(float).tiny)
    ell, _ = evaluate_params(params, y)
    error = float(abs(ell.sum()-target_loglik))
    if error > 1e-6:
        raise AssertionError(f"Package independent density mismatch: {error}")
    pg = None
    if method == "fmvmm_dmm":
        theta = np.r_[np.log(np.asarray(weights)[:-1]/weights[-1]), np.asarray(fitted).ravel()]
        pg = float(np.max(abs(ld_evaluate(theta, y, k, "base_ld", True)[2]/len(y))))
    return params, {"mean_train_log_density": float(ell.mean()), "optimizer_success": bool(relative <= settings["package_tol"]),
                    "score_converged": None if pg is None else pg <= settings["gtol"], "projected_gradient": pg,
                    "iterations": len(history)-1, "message": "relative likelihood tolerance" if relative <= settings["package_tol"] else "package iteration cap",
                    "boundary_hits": hits, "boundary_hit": None if hits is None else bool(hits), "parameter_count": count,
                    "min_weight": float(np.min(weights)), "package_density_check": error,
                    "minimum_likelihood_increment": float(np.diff(history).min())}


@dataclass
class FitResult:
    method: str
    k: int
    d: int
    params: dict
    record: dict
    starts: list
    selected_start: int

    def evaluate(self, y):
        return evaluate_params(self.params, np.asarray(y, dtype=float))

    def component_means(self):
        params = self.params
        if self.method in ["shared_ld", "full_ld", "base_ld"]:
            _, alpha, a, b = unpack_ld(np.asarray(params["theta"]), self.method, self.k, self.d)
            return (-b+digamma(alpha[:, :-1])-digamma(alpha[:, -1:]))/a
        if self.method == "fmvmm_dmm":
            alpha = np.asarray(params["alpha"])
            return digamma(alpha[:, :-1])-digamma(alpha[:, -1:])
        if self.method == "fmvmm_t":
            return np.asarray([a[0] if a[3] > 1 else np.full(self.d, np.nan) for a in params["components"]])
        if self.method == "fmvmm_skew_normal":
            means = []
            for mu, covariance, shape in params["components"]:
                mu, covariance, shape = np.asarray(mu), np.asarray(covariance), np.asarray(shape)
                beta = shape@np.linalg.inv(sqrtm(covariance+1e-12*np.eye(self.d)))
                means.append(mu+np.sqrt(2/np.pi)*(covariance@beta)/np.sqrt(1+beta@covariance@beta))
            return np.asarray(means)
        return np.asarray(params["means"])


class FitFailure(RuntimeError):
    def __init__(self, message, starts):
        super().__init__(message)
        self.starts = starts


def fit_method(method, y, k, seed, settings=None, partitions=None, warm_start=None):
    """Fit using training data only; retain all starts, including failures."""
    settings = {**DEFAULT_SETTINGS, **(settings or {})}
    y = np.asarray(y, dtype=float)
    n, d = y.shape
    if not np.isfinite(y).all():
        raise ValueError("Training data must be finite")
    partitions = make_partitions(y, k, seed, settings["n_starts"]) if partitions is None else partitions
    beginning, records, candidates = time.perf_counter(), [], []

    def do_start(index, labels, initial=None, reason="matched_KMeans"):
        t0 = time.perf_counter()
        rec = {"start": index, "initialization": reason, "selected": False}
        try:
            with warnings.catch_warnings(record=True) as caught, timed_start(settings["start_timeout_seconds"]):
                warnings.simplefilter("always")
                if method in ["shared_ld", "full_ld", "base_ld"]:
                    init = initial_ld(y, labels, k, method, settings, index) if initial is None else initial
                    params, detail = optimize_ld(init, y, k, method, settings)
                    rec["initial_parameters"] = init
                elif method.startswith("gaussian"):
                    pi, means, cov = normal_cluster_parameters(y, labels, k, settings["covariance_regularization"])
                    covariance_type = "diag" if method.endswith("diag") else "full"
                    precision = 1/np.diagonal(cov, axis1=1, axis2=2) if covariance_type == "diag" else np.linalg.inv(cov)
                    model = GaussianMixture(k, covariance_type=covariance_type, max_iter=settings["maxiter"],
                                            tol=settings["gaussian_tol"], reg_covar=settings["covariance_regularization"],
                                            n_init=1, random_state=seed+index, weights_init=pi, means_init=means, precisions_init=precision)
                    model.fit(y)
                    minimum = float(model.covariances_.min() if covariance_type == "diag" else np.linalg.eigvalsh(model.covariances_).min())
                    params = {"method": method, "k": k, "d": d, "weights": model.weights_, "means": model.means_, "covariances": model.covariances_}
                    detail = {"mean_train_log_density": float(model.score(y)), "optimizer_success": bool(model.converged_),
                              "score_converged": None, "projected_gradient": None, "iterations": int(model.n_iter_),
                              "message": "Gaussian likelihood tolerance" if model.converged_ else "iteration cap",
                              "boundary_hit": minimum <= 1.01*settings["covariance_regularization"],
                              "boundary_hits": ["covariance_floor"] if minimum <= 1.01*settings["covariance_regularization"] else [],
                              "parameter_count": model._n_parameters(), "min_weight": float(model.weights_.min())}
                else:
                    params, detail = fit_package(method, y, k, labels, settings)
            ell, tau = evaluate_params(params, y)
            if not np.isfinite(ell).all() or not np.isfinite(tau).all():
                raise FloatingPointError("Nonfinite fit evaluation")
            rec.update(detail)
            rec.update({"failed": False, "parameters": params, "warnings": [str(w.message) for w in caught]})
            candidates.append((params, rec))
        except Exception as exc:
            rec.update({"failed": True, "message": repr(exc), "parameters": None})
        rec["fit_seconds"] = time.perf_counter()-t0
        records.append(rec)

    for index, labels in enumerate(partitions):
        do_start(index, labels)
    nested_violation = False
    dependency_fit_seconds = 0.0
    if warm_start is not None:
        if method == "full_ld" and warm_start.method == "shared_ld":
            best = max((r["mean_train_log_density"] for _, r in candidates), default=-np.inf)
            nested_violation = best < warm_start.record["mean_train_log_density"]-settings["nested_training_tolerance"]
            if nested_violation:
                dependency_fit_seconds = warm_start.record["fit_seconds"]
                initial = embed_shared(np.asarray(warm_start.params["theta"]), k, d)
                do_start(len(records), partitions[0], initial, "nested_shared_fit_safeguard")
                # This feasible point remains available even if optimization fails.
                if max((r["mean_train_log_density"] for _, r in candidates), default=-np.inf) < warm_start.record["mean_train_log_density"]-settings["nested_training_tolerance"]:
                    params = {"theta": initial, "method": method, "k": k, "d": d}
                    score = ld_evaluate(initial, y, k, method, True)[2]
                    pg = shared.projected_gradient(initial, -score/n, ld_bounds(method,k,d,settings))
                    rec = {"start": len(records), "initialization": "retained_feasible_shared_point", "selected": False,
                           "failed": False, "parameters": params, "mean_train_log_density": warm_start.record["mean_train_log_density"],
                           "optimizer_success": False, "score_converged": pg <= settings["gtol"], "projected_gradient": pg,
                           "boundary_hits": shared.bound_hits(initial, ld_bounds(method,k,d,settings)), "parameter_count": len(initial),
                           "fit_seconds": 0.0, "iterations": 0, "message": "feasible nesting fallback"}
                    rec["boundary_hit"] = bool(rec["boundary_hits"])
                    records.append(rec);candidates.append((params,rec))
        elif method == "base_ld" and warm_start.method == "fmvmm_dmm":
            dependency_fit_seconds = warm_start.record["fit_seconds"]
            pi = np.asarray(warm_start.params["weights"])
            alpha = np.clip(np.asarray(warm_start.params["alpha"]), *settings["alpha_bounds"])
            initial = np.r_[np.log(pi[:-1]/pi[-1]), alpha.ravel()]
            do_start(len(records), partitions[0], initial, "package_DMM_warm_start")
    if not candidates:
        raise FitFailure(f"No finite fit for {method}", records)
    params, selected = max(candidates, key=lambda item:item[1]["mean_train_log_density"])
    selected["selected"] = True
    record = {key:value for key,value in selected.items() if key not in ["parameters","initial_parameters"]}
    record.update({"fit_seconds": time.perf_counter()-beginning, "starts_attempted": len(records),
                   "starts_failed": sum(r["failed"] for r in records), "nested_violation_before_safeguard": nested_violation,
                   "dependency_fit_seconds": dependency_fit_seconds})
    record["standalone_fit_seconds"] = record["fit_seconds"]+dependency_fit_seconds
    return FitResult(method,k,d,params,record,records,selected["start"])


def generating_specs(config, mechanism=None):
    g = {**config["generating_parameters"], **((mechanism or {}).get("generating_parameters", {}))}
    alpha, scales, means = np.asarray(g["shared_alpha"]), np.asarray(g["component_scales"]), np.asarray(g["component_means"])
    d = means.shape[1]
    kernel_cov = np.diag(polygamma(1,alpha[:-1]))+polygamma(1,alpha[-1])*np.ones((d,d))
    covariances = np.asarray([np.diag(s)@kernel_cov@np.diag(s) for s in scales])
    negative = covariances.copy() if d == 2 else None
    if negative is not None:
        for covariance in negative:
            covariance[0,1] = covariance[1,0] = g["negative_correlation"]*np.sqrt(covariance[0,0]*covariance[1,1])
    varying_alpha = np.asarray(g["varying_alpha"]) if d == 2 else np.tile(alpha,(len(means),1))
    return {"weights":np.asarray(g["weights"]),"alpha":alpha,"varying_alpha":varying_alpha,
            "scales":scales,"means":means,"matched_covariances":covariances,"negative_covariances":negative}


def generate(n,family,spec,rng):
    k, d = spec["means"].shape
    label = rng.choice(k,n,p=spec["weights"])
    y = np.empty((n,d))
    for j in range(k):
        mask = label==j
        if family.startswith("gaussian"):
            covariance=spec["negative_covariances"] if family.endswith("negative") else spec["matched_covariances"]
            y[mask]=rng.multivariate_normal(spec["means"][j],covariance[j],size=mask.sum())
        else:
            alpha=spec["varying_alpha"][j] if family=="full_ld" else spec["alpha"]
            gamma=rng.gamma(alpha,size=(mask.sum(),d+1))
            w=np.log(gamma[:,:d])-np.log(gamma[:,d:])
            mu=spec["means"][j]-spec["scales"][j]*(digamma(alpha[:d])-digamma(alpha[-1]))
            y[mask]=mu+spec["scales"][j]*w
    return y,label


def mean_rmse(estimated,true):
    if not np.isfinite(estimated).all():
        return None,None
    cost=((estimated[:,None,:]-true[None,:,:])**2).sum(2)
    rows,columns=linear_sum_assignment(cost)
    return float(np.sqrt(((estimated[rows]-true[columns])**2).mean())),columns.tolist()


def preflight():
    rng=np.random.default_rng(202611077)
    score_errors=[]
    for k,d in [(2,2),(3,3)]:
        y=rng.normal(size=(12,d));labels=np.arange(len(y))%k
        for method in ["shared_ld","full_ld","base_ld"]:
            theta=initial_ld(y,labels,k,method,DEFAULT_SETTINGS,1)
            ell,tau,gradient=ld_evaluate(theta,y,k,method,True)
            numeric=[]
            for index in range(len(theta)):
                change=np.zeros(len(theta));change[index]=1e-5*max(1,abs(theta[index]))
                numeric.append((ld_evaluate(theta+change,y,k,method)[0].sum()-ld_evaluate(theta-change,y,k,method)[0].sum())/(2*change[index]))
            score_errors.append(float(np.max(abs(gradient-numeric)/(1+abs(gradient)))))
        common=initial_ld(y,labels,k,"shared_ld",DEFAULT_SETTINGS,1)
        full=embed_shared(common,k,d)
        assert np.max(abs(ld_evaluate(common,y,k,"shared_ld")[0]-ld_evaluate(full,y,k,"full_ld")[0]))<1e-12
    x=rng.normal(size=(20,2));tau=rng.dirichlet([2,3],size=len(x))
    previous=[(np.array([.2,-.3]),np.array([[1.2,.2],[.2,.8]]),np.array([.5,-.7])),
              (np.array([-.4,.5]),np.array([[.7,-.1],[-.1,1.3]]),np.array([-.2,.8]))]
    reference=estimate_alphas_skewnormal(x,tau,previous,np.array([.4,.6]),None,None)
    fast=fast_skew_mstep(x,tau,previous)
    discrepancy=max(float(np.max(abs(a-b))) for r,f in zip(reference,fast) for a,b in zip(r,f))
    density_error=max(float(np.max(abs(np.log(dmvSN(x,*p))-skew_logpdf(x,*p)))) for p in previous)
    assert max(score_errors)<1e-6 and discrepancy<1e-8 and density_error<1e-8
    return {"maximum_LD_score_relative_error":max(score_errors),"shared_full_initial_density_identity":True,
            "vectorized_skew_Mstep_max_absolute_error":discrepancy,"skew_logdensity_max_error":density_error,"passed":True}


def summarize(out,rows):
    frame=pd.DataFrame(rows)
    summary=[]
    paired=[]
    for (scenario,method),group in frame.groupby(["scenario","method"],sort=False):
        finite=group[~group.failed]
        summary.append({"scenario":scenario,"method":method,"attempted":len(group),"finite":len(finite),"failed":int(group.failed.sum()),
                        "test_log_density_mean":finite.test_log_density.mean(),"test_log_density_sd":finite.test_log_density.std(ddof=1),
                        "mean_RMSE_mean":finite.mean_rmse.mean(),"mean_RMSE_available":int(finite.mean_rmse.notna().sum()),
                        "ARI_mean":finite.ARI.mean(),"fit_seconds_mean":group.fit_seconds.mean(),
                        "standalone_fit_seconds_mean":group.standalone_fit_seconds.mean(),
                        "score_checked":int(finite.score_converged.notna().sum()),"score_passed":int(finite.score_converged.fillna(False).sum()),
                        "optimizer_successes":int(finite.optimizer_success.fillna(False).sum()),
                        "boundary_checked":int(finite.boundary_hit.notna().sum()),"boundary_hits":int(finite.boundary_hit.fillna(False).sum()),
                        "parameter_count":finite.parameter_count.iloc[0] if len(finite) else None})
        if method != "shared_ld":
            primary=frame[(frame.scenario==scenario)&(frame.method=="shared_ld")]
            p=primary.merge(group,on="rep",suffixes=("_primary","_other"));v=p[~p.failed_primary&~p.failed_other]
            delta=v.test_log_density_primary-v.test_log_density_other
            mse_delta=v.mean_rmse_primary-v.mean_rmse_other
            paired.append({"scenario":scenario,"comparator":method,"paired_attempted":len(p),"paired_finite":len(v),
                           "mean_log_density_gain":delta.mean(),"gain_sd":delta.std(ddof=1),"gain_MCSE":delta.std(ddof=1)/np.sqrt(len(delta)) if len(delta) else np.nan,
                           "mean_RMSE_difference":mse_delta.mean(),"mean_RMSE_difference_MCSE":mse_delta.std(ddof=1)/np.sqrt(mse_delta.notna().sum()) if mse_delta.notna().any() else np.nan})
    frame.to_csv(out/"fits.csv",index=False)
    pd.DataFrame(summary).to_csv(out/"summary.csv",index=False)
    pd.DataFrame(paired).to_csv(out/"paired_comparisons.csv",index=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=CONFIG_PATH)
    parser.add_argument("--outdir",type=Path,default=ROOT/"results/confirmatory")
    parser.add_argument("--repetitions",type=int)
    parser.add_argument("--preflight-only",action="store_true")
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    if args.repetitions is not None:
        config["repetitions"]=args.repetitions
        for mechanism in config["mechanisms"]:mechanism["repetitions"]=args.repetitions
    out=args.outdir;out.mkdir(parents=True,exist_ok=True)
    if (out/"fits.csv").exists():raise RuntimeError("Output already has results; use a new directory to retain previous results")
    execution={"specification":config,"specification_sha256":hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),
               "python":platform.python_version(),"versions":{p:importlib.metadata.version(p) for p in ["fmvmm","numpy","scipy","pandas","scikit-learn","threadpoolctl"]},
               "source_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/"scripts/shared_shape_ld.py"]},
               "threads":1,"label":"Expanded benchmark; scenarios informed by earlier pilot, not statistically confirmatory"}
    write_json(out/"executed_specification.json",execution)
    with threadpool_limits(1):
        write_json(out/"preflight.json",preflight())
        if args.preflight_only:return
        specifications={m["name"]:generating_specs(config,m) for m in config["mechanisms"]}
        write_json(out/"generating_specifications.json",specifications)
        rows=[];started=time.perf_counter()
        with (out/"starts.jsonl").open("w") as start_file:
            for mechanism_index,mechanism in enumerate(config["mechanisms"]):
                spec=specifications[mechanism["name"]]
                k,d=spec["means"].shape
                for rep in range(mechanism.get("repetitions",config["repetitions"])):
                    seed=config["master_seed"]+mechanism_index*100000+rep*100
                    rng=np.random.default_rng(seed)
                    train,train_labels=generate(mechanism["n_train"],mechanism["family"],spec,rng)
                    test,test_labels=generate(config["n_test"],mechanism["family"],spec,rng)
                    tag=f"{mechanism['name']}_{rep+1:03d}"
                    (out/"data").mkdir(exist_ok=True);(out/"predictions").mkdir(exist_ok=True)
                    np.savez_compressed(out/"data"/(tag+".npz"),train=train,test=test,train_labels=train_labels,test_labels=test_labels)
                    partitions=make_partitions(train,k,seed+1,config["fitting"]["n_starts"])
                    fitted={};predictions={"test_labels":test_labels}
                    for method in config["methods"]:
                        head={"scenario":mechanism["name"],"rep":rep+1,"seed":seed,"method":method,"n_train":len(train),"n_test":len(test),"d":d,"k":k}
                        t0=time.perf_counter()
                        row={**head,"failed":True,"test_log_density":np.nan,"mean_rmse":np.nan,"ARI":np.nan,
                             "optimizer_success":None,"score_converged":None,"boundary_hit":None,"projected_gradient":None,"parameter_count":None,
                             "standalone_fit_seconds":np.nan,"dependency_fit_seconds":np.nan}
                        try:
                            model=fit_method(method,train,k,seed+1,settings=config["fitting"],partitions=partitions,
                                             warm_start=fitted.get("shared_ld") if method=="full_ld" else None)
                            fitted[method]=model
                            for record in model.starts:start_file.write(json.dumps(serializable({**head,**record}))+"\n")
                            ell,tau=model.evaluate(test)
                            if not np.isfinite(ell).all():raise FloatingPointError("Nonfinite test log density")
                            means=model.component_means();rmse,matching=mean_rmse(means,spec["means"])
                            row.update({key:model.record.get(key) for key in ["optimizer_success","score_converged","boundary_hit","projected_gradient","parameter_count","iterations","starts_attempted","starts_failed","nested_violation_before_safeguard","message"]})
                            row.update({"failed":False,"test_log_density":float(ell.mean()),"mean_rmse":rmse,"ARI":float(adjusted_rand_score(test_labels,tau.argmax(1))),
                                        "train_log_density":model.record["mean_train_log_density"],"selected_start":model.selected_start,"fit_seconds":model.record["fit_seconds"],
                                        "dependency_fit_seconds":model.record["dependency_fit_seconds"],"standalone_fit_seconds":model.record["standalone_fit_seconds"]})
                            predictions[method+"_logpdf"]=ell;predictions[method+"_cluster"]=tau.argmax(1)
                            predictions[method+"_component_means"]=means
                        except Exception as exc:
                            row.update({"message":repr(exc),"fit_seconds":time.perf_counter()-t0})
                            if isinstance(exc,FitFailure):
                                for record in exc.starts:start_file.write(json.dumps(serializable({**head,**record}))+"\n")
                        rows.append(row);start_file.flush()
                        print(mechanism["name"],rep+1,method,"testLL",round(row["test_log_density"],5),"score",row["score_converged"],"bound",row["boundary_hit"],"fail",row["failed"],flush=True)
                    np.savez_compressed(out/"predictions"/(tag+".npz"),**predictions)
                    summarize(out,rows)
        execution["completed"]={"fits":len(rows),"failed_fits":sum(r["failed"] for r in rows),"elapsed_seconds":time.perf_counter()-started}
        write_json(out/"executed_specification.json",execution)
        print("Completed",json.dumps(execution["completed"]),flush=True)


if __name__=="__main__":main()
