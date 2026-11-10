#!/usr/bin/env python3
"""Reproducible coverage experiment with self-contained analytic information.

Run separate commands for well-separated and overlapping component settings.
All fits remain in the record, including unavailable intervals. The executable
specification is saved before simulation. This extends the earlier pilot.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import platform
import time
import traceback
import warnings
import hashlib
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import linear_sum_assignment
from scipy.special import digamma, expit
from scipy.stats import binomtest, norm
from threadpoolctl import threadpool_limits
from shared_shape_ld import (A_BOUNDS, ALPHA_BOUNDS, B_BOUNDS, SharedShapeFitFailure,
                             evaluation, fit, simulate, unpack)
from ld_information import information, component_mean_and_gradient
ROOT = Path(__file__).resolve().parents[1]

def targets(theta, k, d, true_means, common_scale=False):
    """Truth is used here for evaluation alignment ONLY."""
    logpi, alpha, a, b = unpack(theta, k, d, common_scale)
    pi = np.exp(logpi)
    estimated_means = (digamma(alpha[:d])-digamma(alpha[-1])-b)/a
    fitted_indices, true_indices = linear_sum_assignment(
        ((estimated_means[:, None]-true_means[None])**2).sum(axis=2))
    alignment = np.empty(k, dtype=int)
    alignment[true_indices] = fitted_indices
    r = k-1
    alpha_gradient = np.zeros(len(theta))
    alpha_gradient[r] = 1
    j0, j1 = alignment[0], alignment[1]
    weight_gradient = np.zeros(len(theta))
    for s in range(r):
        weight_gradient[s] = pi[j0]*(float(j0 == s)-pi[s])
    mean0, grad0 = component_mean_and_gradient(theta, k, d, j0, 0, common_scale)
    mean1, grad1 = component_mean_and_gradient(theta, k, d, j1, 0, common_scale)
    return {
        "alpha_1": (float(alpha[0]), alpha_gradient),
        "weight_true_component_0": (float(pi[j0]), weight_gradient),
        "mean_difference_coordinate_0": (mean1-mean0, grad1-grad0),
    }, alignment.tolist(), estimated_means.tolist()


def confidence_interval(name, estimate, variance, probability=.95):
    z = norm.ppf((1+probability)/2)
    se = float(np.sqrt(variance))
    if name == "alpha_1":
        interval = np.exp(np.log(estimate)+np.array([-1, 1])*z*se/estimate)
    elif name == "weight_true_component_0":
        logit = np.log(estimate/(1-estimate))
        interval = expit(logit+np.array([-1, 1])*z*se/(estimate*(1-estimate)))
    else:
        interval = estimate+np.array([-1, 1])*z*se
    return se, float(interval[0]), float(interval[1])


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def run_pilot(args, outdir):
    alpha = np.array([.8, 2.4, 1.4])
    true_means = args.separation*np.array([[-2., -1.2], [2., 1.2]])
    scales = np.array([[.65, .55], [.9, .7]])
    weights = np.array([.55, .45])
    locations = true_means-scales*(digamma(alpha[:-1])-digamma(alpha[-1]))
    true_targets = {"alpha_1": .8, "weight_true_component_0": .55,
                    "mean_difference_coordinate_0": 4.*args.separation}
    config = {"status": "expanded inference experiment; no outcome-dependent stopping",
              "separation_multiplier": args.separation,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "sizes": args.sizes, "replications_per_size": args.reps,
              "seed_base": args.seed, "seed_rule": "seed_base + 100000*size_index + replicate",
              "fit_seed_rule": "data_seed + 2000000", "fit_starts": args.starts,
              "fit_maxiter": args.maxiter, "mean_objective_gradient_tolerance": args.gtol,
              "common_scale": False, "alpha": alpha.tolist(), "means": true_means.tolist(),
              "locations": locations.tolist(), "scales": scales.tolist(), "weights": weights.tolist(),
              "alpha_bounds": ALPHA_BOUNDS, "inverse_scale_bounds": A_BOUNDS,
              "scaled_shift_bounds": B_BOUNDS, "weight_logit_bounds": None,
              "information_condition_cutoff": 1e12, "minimum_weight_for_intervals": 1e-5,
              "interval_rule": "finite fit; no bound flag; projected mean score <= gtol; min weight > 1e-5; positive information with condition < 1e12; finite positive target variances",
              "coverage_denominator": "available intervals, reported together with ALL attempted replications",
              "labels": "true means align evaluation only; neither simulation labels nor truth initialize/select fits",
              "threads": 1, "python": platform.python_version(), "numpy": np.__version__,
              "scipy": scipy.__version__}
    save_json(outdir/"specification.json", config)
    all_fits, all_starts, interval_rows = [], [], []
    fit_parameters = {}
    for size_index, n in enumerate(args.sizes):
        for rep in range(args.reps):
            seed = args.seed+100000*size_index+rep
            rng = np.random.default_rng(seed)
            y, unused_labels = simulate(n, alpha, locations, scales, weights, rng)
            row = {"n": n, "replicate": rep, "data_seed": seed, "fit_seed": seed+2000000,
                   "fit_exception": "", "finite_fit": False, "information_exception": "",
                   "intervals_available": False, "exclusion_reasons": "",
                   "boundary_hits": "", "score_converged": False,
                   "optimizer_success": False, "projected_gradient": np.nan,
                   "mean_train_log_density": np.nan, "min_weight": np.nan,
                   "fit_seconds_all_starts": np.nan, "information_seconds": np.nan,
                   "information_condition_number": np.nan,
                   "information_min_eigenvalue": np.nan,
                   "information_max_eigenvalue": np.nan, "info_symmetry_error": np.nan,
                   "mapped_direct_score_error": np.nan,
                   "starts_attempted": 0, "starts_failed": 0}
            beginning = time.perf_counter()
            try:
                with warnings.catch_warnings(record=True) as emitted:
                    fitted = fit(y, 2, common_scale=False, algorithm="direct", n_starts=args.starts,
                                 seed=seed+2000000, maxiter=args.maxiter, gtol=args.gtol)
                row.update(fitted.record)
                row["warnings"] = " | ".join(str(item.message) for item in emitted)
                row["boundary_hits"] = ";".join(row["boundary_hits"])
                row["fit_seconds_all_starts"] = fitted.elapsed_seconds
                row["starts_attempted"] = len(fitted.records)
                row["starts_failed"] = sum(record.get("failed", False) for record in fitted.records)
                row["finite_fit"] = bool(np.isfinite(fitted.theta).all()
                                         and np.isfinite(row["mean_train_log_density"]))
                fit_parameters[f"n{n}_rep{rep}"] = fitted.theta
                all_starts.extend({"n": n, "replicate": rep, "data_seed": seed,
                                   **record, "boundary_hits": ";".join(record["boundary_hits"])}
                                  for record in fitted.records)
                selected_targets, alignment, estimated_means = targets(fitted.theta, 2, 2, true_means)
                row["evaluation_alignment"] = json.dumps(alignment)
                row["estimated_means"] = json.dumps(estimated_means)
                reasons = []
                if not row["finite_fit"]:
                    reasons.append("nonfinite_fit")
                if row["boundary_hits"]:
                    reasons.append("bound_hit")
                if not row["score_converged"]:
                    reasons.append("nonstationary")
                if row["min_weight"] <= 1e-5:
                    reasons.append("tiny_weight")
                covariance = None
                try:
                    info_start = time.perf_counter()
                    ell, score, info = information(fitted.theta, y, 2, False)
                    row["information_seconds"] = time.perf_counter()-info_start
                    row["info_symmetry_error"] = float(np.abs(info-info.T).max())
                    symmetric_info = (info+info.T)/2
                    eigenvalues = np.linalg.eigvalsh(symmetric_info)
                    row["information_min_eigenvalue"] = float(eigenvalues[0])
                    row["information_max_eigenvalue"] = float(eigenvalues[-1])
                    condition = float(eigenvalues[-1]/eigenvalues[0]) if eigenvalues[0] > 0 else np.inf
                    row["information_condition_number"] = condition
                    if not np.isfinite(eigenvalues).all() or eigenvalues[0] <= 0:
                        reasons.append("information_not_positive_definite")
                    elif condition >= 1e12:
                        reasons.append("information_numerically_ill_conditioned")
                    else:
                        covariance = np.linalg.inv(symmetric_info)
                    direct_score = evaluation(fitted.theta, y, 2, False, True)[2]
                    row["mapped_direct_score_error"] = float(np.max(np.abs(score-direct_score)/(1+np.abs(direct_score))))
                except Exception as exc:
                    row["information_exception"] = repr(exc)
                    reasons.append("information_exception")
                for name, (estimate, gradient) in selected_targets.items():
                    variance = float(gradient @ covariance @ gradient) if covariance is not None else np.nan
                    available = not reasons and np.isfinite(variance) and variance > 0
                    interval = {"n": n, "replicate": rep, "data_seed": seed, "target": name,
                                "truth": true_targets[name], "estimate": estimate,
                                "variance": variance, "interval_available": available,
                                "exclusion_reasons": ";".join(reasons)}
                    if available:
                        se, lower, upper = confidence_interval(name, estimate, variance)
                        if not np.isfinite([se, lower, upper]).all():
                            available = False
                            interval["interval_available"] = False
                            interval["exclusion_reasons"] += ";nonfinite_interval"
                        else:
                            interval.update({"se": se, "lower": lower, "upper": upper,
                                             "length": upper-lower,
                                             "covered": lower <= true_targets[name] <= upper})
                    elif not reasons:
                        interval["exclusion_reasons"] = "invalid_target_variance"
                    interval_rows.append(interval)
                row["intervals_available"] = all(item["interval_available"]
                                                  for item in interval_rows[-len(selected_targets):])
                row["exclusion_reasons"] = ";".join(reasons)
            except Exception as exc:
                row["fit_exception"] = repr(exc)
                row["exclusion_reasons"] = "fit_exception"
                row["traceback"] = traceback.format_exc()
                if isinstance(exc, SharedShapeFitFailure):
                    row["fit_seconds_all_starts"] = exc.elapsed_seconds
                    row["starts_attempted"] = len(exc.records)
                    row["starts_failed"] = sum(record["failed"] for record in exc.records)
                    all_starts.extend({"n": n, "replicate": rep, "data_seed": seed,
                                       **record, "boundary_hits": ";".join(record["boundary_hits"])}
                                      for record in exc.records)
                for name, truth in true_targets.items():
                    interval_rows.append({"n": n, "replicate": rep, "data_seed": seed,
                                          "target": name, "truth": truth,
                                          "interval_available": False,
                                          "exclusion_reasons": "fit_exception"})
            row["total_seconds"] = time.perf_counter()-beginning
            all_fits.append(row)
            pd.DataFrame(all_fits).to_csv(outdir/"selected_fits.csv", index=False)
            start_frame = pd.DataFrame(all_starts)
            if start_frame.empty:
                start_frame = pd.DataFrame(columns=["n", "replicate", "data_seed", "start",
                    "failed", "selected", "mean_train_log_density", "projected_gradient",
                    "score_converged", "boundary_hits", "min_weight", "optimizer_success",
                    "iterations", "message", "elapsed_seconds"])
            start_frame.to_csv(outdir/"all_starts.csv", index=False)
            pd.DataFrame(interval_rows).to_csv(outdir/"intervals.csv", index=False)
            np.savez_compressed(outdir/"fitted_parameters.npz", **fit_parameters)
            print(json.dumps({"n": n, "replicate": rep,
                              "available": row["intervals_available"],
                              "reasons": row["exclusion_reasons"],
                              "seconds": round(row["total_seconds"], 2)}), flush=True)
    fit_df, interval_df = pd.DataFrame(all_fits), pd.DataFrame(interval_rows)
    summaries = []
    for (n, name), part in interval_df.groupby(["n", "target"]):
        valid = part[part.interval_available]
        count = len(valid)
        covered = int(valid.covered.fillna(False).astype(bool).sum()) if count else 0
        coverage = covered/count if count else np.nan
        mc_interval = binomtest(covered, count).proportion_ci(method="exact") if count else None
        estimates = part.estimate.dropna() if "estimate" in part else pd.Series(dtype=float)
        summaries.append({"n": n, "target": name, "attempted": len(part),
                          "available": count, "unavailable": len(part)-count,
                          "availability_fraction": count/len(part), "covered_count": covered,
                          "conditional_coverage": coverage,
                          "covered_and_available_fraction": covered/len(part),
                          "conditional_coverage_mcse": np.sqrt(coverage*(1-coverage)/count) if count else np.nan,
                          "conditional_coverage_mc95_lower": mc_interval.low if count else np.nan,
                          "conditional_coverage_mc95_upper": mc_interval.high if count else np.nan,
                          "nominal_95_coverage_mcse": np.sqrt(.95*.05/count) if count else np.nan,
                          "mean_interval_length_conditional": valid.length.mean() if count else np.nan,
                          "mean_estimated_se_conditional": valid.se.mean() if count else np.nan,
                          "estimate_bias_all_finite": estimates.mean()-part.truth.iloc[0],
                          "estimate_sd_all_finite": estimates.std(ddof=1),
                          "estimate_bias_conditional": valid.estimate.mean()-part.truth.iloc[0] if count else np.nan,
                          "estimate_sd_conditional": valid.estimate.std(ddof=1) if count else np.nan})
    summary = pd.DataFrame(summaries)
    summary.to_csv(outdir/"coverage_summary.csv", index=False)
    diagnostics = []
    for n, part in fit_df.groupby("n"):
        diagnostics.append({"n": int(n), "attempted": len(part),
                            "fit_exception": int(part.fit_exception.fillna("").ne("").sum()),
                            "nonfinite_fit": int((~part.finite_fit).sum()),
                            "bound_hit": int(part.boundary_hits.fillna("").ne("").sum()),
                            "nonstationary": int((~part.score_converged.fillna(False).astype(bool)).sum()),
                            "information_failure": int(part.exclusion_reasons.str.contains("information", na=False).sum()),
                            "all_target_intervals_available": int(part.intervals_available.sum()),
                            "median_information_condition": float(part.information_condition_number.median()),
                            "median_fit_seconds_all_starts": float(part.fit_seconds_all_starts.median()),
                            "median_information_seconds": float(part.information_seconds.median())})
    pd.DataFrame(diagnostics).to_csv(outdir/"fit_diagnostics.csv", index=False)
    print(summary.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--reps', type=int, default=100)
    parser.add_argument('--sizes', type=int, nargs='+', default=[250,1000])
    parser.add_argument('--separation', type=float, default=1.)
    parser.add_argument('--starts', type=int, default=3)
    parser.add_argument('--maxiter', type=int, default=2000)
    parser.add_argument('--gtol', type=float, default=1e-6)
    parser.add_argument('--seed', type=int, default=202611070)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    if (args.outdir/'specification.json').exists():
        raise SystemExit('Output directory already contains a study; choose a new directory.')
    with threadpool_limits(limits=1):
        run_pilot(args, args.outdir)
if __name__ == '__main__':
    main()
