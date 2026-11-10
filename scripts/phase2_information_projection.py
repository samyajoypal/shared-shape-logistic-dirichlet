#!/usr/bin/env python3
"""Monte Carlo bridge from local LD information theory to phase-2 generators.

No fitting occurs. Coordinates are raw free weights, separate raw shapes,
locations, then scales. Information is the uncentered score second moment.
Independent batches quantify Monte Carlo uncertainty; no eigenvalue floors or
pseudoinverses are used. Risk outputs are second moments of limiting normal
laws, not a claim that actual finite-sample MSEs equal those expressions.

Example: python scripts/phase2_information_projection.py --output RUN
Small pilot: add --batches 2 --draws-per-batch 2000.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import shutil
import time
import traceback

import numpy as np
import scipy
from scipy.linalg import cho_factor, cho_solve
from scipy.special import digamma, gammaln, logsumexp, polygamma
from threadpoolctl import threadpool_limits

import phase2_simulations as simulation
import shared_shape_ld as shared

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "phase2_simulations.json"


def layout(k, d):
    if k < 1 or d < 1:
        raise ValueError("Positive component count and dimension required")
    r, p = k - 1, d + 1
    return {"weight": slice(0, r), "alpha": slice(r, r + k*p),
            "mu": slice(r + k*p, r + k*p + k*d),
            "sigma": slice(r + k*p + k*d, r + k*p + 2*k*d),
            "q": r + k*p + 2*k*d, "shared_q": r + p + 2*k*d}


def pack_raw(weights, alpha, locations, scales):
    weights, alpha = np.asarray(weights, float), np.asarray(alpha, float)
    locations, scales = np.asarray(locations, float), np.asarray(scales, float)
    if locations.ndim != 2:
        raise ValueError("Locations must be a k by d matrix")
    k, d = locations.shape
    if (weights.shape != (k,) or alpha.shape != (k, d+1)
            or scales.shape != (k, d) or not np.isclose(weights.sum(), 1., rtol=0, atol=1e-12)):
        raise ValueError("Incompatible parameters or weights that do not sum to one")
    theta = np.r_[weights[:-1], alpha.ravel(), locations.ravel(), scales.ravel()]
    unpack_raw(theta, k, d)
    return theta


def unpack_raw(theta, k, d):
    theta = np.asarray(theta, float)
    ix = layout(k, d)
    if theta.shape != (ix["q"],) or not np.isfinite(theta).all():
        raise ValueError("Invalid full raw parameter vector")
    weights = np.r_[theta[ix["weight"]], 1-theta[ix["weight"]].sum()]
    alpha = theta[ix["alpha"]].reshape(k, d+1)
    mu = theta[ix["mu"]].reshape(k, d)
    sigma = theta[ix["sigma"]].reshape(k, d)
    if np.any(weights <= 0) or np.any(alpha <= 0) or np.any(sigma <= 0):
        raise ValueError("Weights, shapes and scales must be positive")
    return weights, alpha, mu, sigma


def raw_from_specification(spec):
    return pack_raw(spec["weights"], spec["alpha"], spec["locations"], spec["scales"])


def full_raw_score(theta, y, k):
    """Return each log density, observed score row, and responsibilities."""
    y = np.asarray(y, float)
    if y.ndim != 2 or not np.isfinite(y).all():
        raise ValueError("Observations must be a finite n by d matrix")
    n, d = y.shape
    weights, alpha, mu, sigma = unpack_raw(theta, k, d)
    z = (y[:, None, :]-mu[None, :, :])/sigma[None, :, :]
    augmented = np.concatenate((z, np.zeros((n, k, 1))), axis=2)
    lq = augmented-logsumexp(augmented, axis=2, keepdims=True)
    total = alpha.sum(axis=1)
    component = (gammaln(total)-gammaln(alpha).sum(axis=1)
                 - np.log(sigma).sum(axis=1)
                 + np.einsum("nkp,kp->nk", lq, alpha))
    joint = component+np.log(weights)
    ell = logsumexp(joint, axis=1)
    tau = np.exp(joint-ell[:, None])
    shape = tau[:, :, None]*(digamma(total)[None, :, None]-digamma(alpha)[None, :, :]+lq)
    v = alpha[None, :, :d]-total[None, :, None]*np.exp(lq[:, :, :d])
    location = -tau[:, :, None]*v/sigma[None, :, :]
    scale = -tau[:, :, None]*(1+z*v)/sigma[None, :, :]
    free_weight = tau[:, :k-1]/weights[None, :k-1]-tau[:, -1:]/weights[-1]
    score = np.column_stack((free_weight, shape.reshape(n, -1),
                             location.reshape(n, -1), scale.reshape(n, -1)))
    if not np.isfinite(ell).all() or not np.isfinite(score).all():
        raise FloatingPointError("Nonfinite density or raw score")
    return ell, score, tau


def shared_embedding(k, d):
    """J repeats the common shape and copies all other raw coordinates."""
    ix = layout(k, d)
    r, p = k-1, d+1
    J = np.zeros((ix["q"], ix["shared_q"]))
    J[:r, :r] = np.eye(r)
    for j in range(k):
        J[r+j*p:r+(j+1)*p, r:r+p] = np.eye(p)
    J[ix["mu"], r+p:r+p+k*d] = np.eye(k*d)
    J[ix["sigma"], r+p+k*d:] = np.eye(k*d)
    return J


def fitting_coordinate_map(theta, k, d):
    """Shared fit coordinates and derivative B: raw_shared = map(fit).

    Fit coordinates are free weight logits, common alpha, a=1/sigma,
    b=-mu/sigma. Hence full raw scores times J times B must equal the
    existing fitter's per-observation scores.
    """
    weights, alpha, mu, sigma = unpack_raw(theta, k, d)
    if not np.allclose(alpha, alpha[0], rtol=0, atol=1e-13):
        raise ValueError("Mapping to shared coordinates requires a common shape")
    r, p = k-1, d+1
    a, b = 1/sigma, -mu/sigma
    fitted = np.r_[np.log(weights[:-1]/weights[-1]), alpha[0], a.ravel(), b.ravel()]
    B = np.zeros((len(fitted), len(fitted)))
    B[:r, :r] = np.diag(weights[:-1])-np.outer(weights[:-1], weights[:-1])
    B[r:r+p, r:r+p] = np.eye(p)
    for index in range(k*d):
        raw_mu, raw_sigma = r+p+index, r+p+k*d+index
        fit_a, fit_b = r+p+index, r+p+k*d+index
        B[raw_mu, fit_a] = b.ravel()[index]/a.ravel()[index]**2
        B[raw_mu, fit_b] = -1/a.ravel()[index]
        B[raw_sigma, fit_a] = -1/a.ravel()[index]**2
    return fitted, B


def score_mapping_check(theta, y, k):
    d = y.shape[1]
    fitted, B = fitting_coordinate_map(theta, k, d)
    ell, score, _ = full_raw_score(theta, y, k)
    mapped = score @ shared_embedding(k, d) @ B
    expected = np.stack([shared.evaluation(fitted, row[None, :], k, gradient=True)[2]
                         for row in y])
    old_ell = shared.evaluation(fitted, y, k)[0]
    return {"score_mapping_max_abs": float(np.max(np.abs(mapped-expected))),
            "density_mapping_max_abs": float(np.max(np.abs(ell-old_ell)))}


def component_mean_gradient(theta, k, d, component, coordinate):
    _, alpha, mu, sigma = unpack_raw(theta, k, d)
    if not 0 <= component < k or not 0 <= coordinate < d:
        raise ValueError("Invalid component or coordinate")
    j, m = component, coordinate
    difference = digamma(alpha[j, m])-digamma(alpha[j, -1])
    value = mu[j, m]+sigma[j, m]*difference
    ix = layout(k, d)
    gradient = np.zeros(ix["q"])
    start = ix["alpha"].start+j*(d+1)
    gradient[start+m] = sigma[j, m]*polygamma(1, alpha[j, m])
    gradient[start+d] = -sigma[j, m]*polygamma(1, alpha[j, -1])
    gradient[ix["mu"].start+j*d+m] = 1
    gradient[ix["sigma"].start+j*d+m] = difference
    return float(value), gradient


def target_definitions(theta, k, d):
    targets = []
    for j in range(k):
        for m in range(d):
            value, gradient = component_mean_gradient(theta, k, d, j, m)
            targets.append({"target": f"component_{j}_mean_{m}", "value": value,
                            "gradient": gradient, "component": j, "coordinate": m})
    means0 = [targets[j*d]["value"] for j in range(k)]
    if k > 1:
        low, high = int(np.argmin(means0)), int(np.argmax(means0))
        targets.append({"target": "highest_minus_lowest_component_mean_0",
                        "value": means0[high]-means0[low],
                        "gradient": targets[high*d]["gradient"]-targets[low*d]["gradient"],
                        "fixed_local_components": [low, high]})
    return targets


def generator_path_derivative(cell, config):
    """Full raw right derivative at h=0 of the actual phase-2 generator.

    This includes scale changes preserving marginal variances and location
    changes preserving the specified minimum Mahalanobis separation. At a
    tie, the minimum of active pair derivatives defines the h>=0 direction.
    Actual phase-2 local-alternative cells have k=2 and a unique pair.
    """
    base = dict(cell, family="ld", heterogeneity=0.)
    spec = simulation.generating_parameters(base, config)
    weights, alpha = spec["weights"], spec["alpha"]
    sigma, means = spec["scales"], spec["means"]
    k, d = sigma.shape
    alpha_dot = alpha*spec["shape_contrast"]
    kernel = np.array([np.diag(polygamma(1, a[:-1]))+polygamma(1, a[-1]) for a in alpha])
    kernel_dot = np.array([np.diag(polygamma(2, a[:-1])*ad[:-1])
                           +polygamma(2, a[-1])*ad[-1] for a, ad in zip(alpha, alpha_dot)])
    sigma_dot = -0.5*sigma*np.diagonal(kernel_dot, axis1=1, axis2=2)/np.diagonal(kernel, axis1=1, axis2=2)
    covariance_dot = (sigma_dot[:, :, None]*kernel*sigma[:, None, :]
                      +sigma[:, :, None]*kernel_dot*sigma[:, None, :]
                      +sigma[:, :, None]*kernel*sigma_dot[:, None, :])
    pooled = np.einsum("j,jab->ab", weights, spec["covariances"])
    pooled_dot = np.einsum("j,jab->ab", weights, covariance_dot)
    pairs = []
    for j in range(k):
        for l in range(j):
            difference = means[j]-means[l]
            solved = np.linalg.solve(pooled, difference)
            distance = np.sqrt(difference @ solved)
            derivative = -0.5*(solved @ pooled_dot @ solved)/distance
            pairs.append({"components": [j, l], "distance": float(distance),
                          "fixed_mean_distance_derivative": float(derivative)})
    if not pairs:
        raise ValueError("The phase-2 separation path requires at least two components")
    minimum = min(pair["distance"] for pair in pairs)
    active = [pair for pair in pairs if abs(pair["distance"]-minimum) <= 1e-10*max(1., minimum)]
    minimum_dot = min(pair["fixed_mean_distance_derivative"] for pair in active)
    mean_dot = -means*minimum_dot/minimum
    difference = digamma(alpha[:, :-1])-digamma(alpha[:, -1:])
    difference_dot = (polygamma(1, alpha[:, :-1])*alpha_dot[:, :-1]
                      -polygamma(1, alpha[:, -1:])*alpha_dot[:, -1:])
    location_dot = mean_dot-sigma_dot*difference-sigma*difference_dot
    direction = np.r_[np.zeros(k-1), alpha_dot.ravel(), location_dot.ravel(), sigma_dot.ravel()]
    return spec, direction, {"alpha_derivative": alpha_dot,
                            "scale_derivative": sigma_dot, "mean_derivative": mean_dot,
                            "location_derivative": location_dot,
                            "covariance_derivative": covariance_dot,
                            "pair_distances": pairs, "active_minimum_pairs": active,
                            "direction": "right derivative in h at zero; local coefficient c is nonnegative"}


def information_projection(information, J):
    I, J = np.asarray(information, float), np.asarray(J, float)
    if (I.ndim != 2 or I.shape[0] != I.shape[1] or J.shape[0] != I.shape[0]
            or not np.isfinite(I).all() or not np.isfinite(J).all()
            or not np.allclose(I, I.T, rtol=1e-12, atol=1e-12)):
        raise ValueError("Finite symmetric information and compatible embedding required")
    I = (I+I.T)/2
    eigenvalues = np.linalg.eigvalsh(I)
    if eigenvalues[0] <= 0:
        raise np.linalg.LinAlgError("Information is not positive definite; no regularization applied")
    full = cho_solve(cho_factor(I, lower=True), np.eye(len(I)))
    restricted_information = J.T @ I @ J
    restricted = J @ cho_solve(cho_factor(restricted_information, lower=True), J.T)
    full, restricted = (full+full.T)/2, (restricted+restricted.T)/2
    projection = restricted @ I
    difference = full-restricted
    diagnostics = {"minimum_information_eigenvalue": float(eigenvalues[0]),
                   "maximum_information_eigenvalue": float(eigenvalues[-1]),
                   "information_condition_number": float(eigenvalues[-1]/eigenvalues[0]),
                   "minimum_variance_difference_eigenvalue": float(np.linalg.eigvalsh(difference)[0]),
                   "projection_idempotence_max_abs": float(np.max(np.abs(projection @ projection-projection))),
                   "projection_shared_identity_max_abs": float(np.max(np.abs(projection @ J-J))),
                   "restricted_covariance_identity_max_abs": float(np.max(np.abs(restricted @ I @ restricted-restricted))),
                   "full_inverse_identity_max_abs": float(np.max(np.abs(I @ full-np.eye(len(I)))))}
    return full, restricted, projection, diagnostics


def estimate_information(theta, spec, draws, rng, chunk_size=4096):
    if draws < 2 or chunk_size < 1:
        raise ValueError("At least two draws and a positive chunk size required")
    k, d = np.asarray(spec["locations"]).shape
    q = layout(k, d)["q"]
    second, first = np.zeros((q, q)), np.zeros(q)
    mapping = None
    for start in range(0, draws, chunk_size):
        y, _ = simulation.generate_sample(min(chunk_size, draws-start), spec, rng)
        _, score, _ = full_raw_score(theta, y, k)
        second += score.T @ score
        first += score.sum(axis=0)
        if mapping is None:
            mapping = score_mapping_check(theta, y[:min(12, len(y))], k)
    I, mean = second/draws, first/draws
    z = mean/np.sqrt(np.diag(I)/draws)
    return I, mean, {**mapping, "draws": int(draws),
                     "score_mean_max_abs": float(np.max(np.abs(mean))),
                     "score_mean_max_standardized": float(np.max(np.abs(z)))}


def risk_rows(full, restricted, projection, targets, direction, coefficients, k, d):
    rows = []
    bias_vector = (projection-np.eye(len(full))) @ direction
    for target in targets:
        a = target["gradient"]
        vf, vr = float(a @ full @ a), float(a @ restricted @ a)
        bias = float(a @ bias_vector)
        if vf <= 0 or vr < 0 or not np.isfinite([vf, vr, bias]).all():
            raise FloatingPointError("Invalid target projection variance or bias")
        for coefficient in coefficients:
            rows.append({"target": target["target"], "target_type": "scalar_mean",
                         "local_coefficient": float(coefficient),
                         "full_asymptotic_variance": vf, "shared_asymptotic_variance": vr,
                         "shared_to_full_variance_ratio": vr/vf,
                         "limiting_bias": coefficient*bias,
                         "bias_squared": (coefficient*bias)**2,
                         "full_limiting_normal_risk": vf,
                         "shared_limiting_normal_risk": vr+(coefficient*bias)**2,
                         "shared_to_full_risk_ratio": (vr+(coefficient*bias)**2)/vf})
    # Squared Frobenius error per mean entry matches the squared RMSE loss.
    # Its limiting expectation is not the expectation of the RMSE itself.
    gradients = np.stack([t["gradient"] for t in targets[:k*d]])
    vf = float(np.trace(gradients @ full @ gradients.T)/(k*d))
    vr = float(np.trace(gradients @ restricted @ gradients.T)/(k*d))
    bias2 = float(np.mean((gradients @ bias_vector)**2))
    for coefficient in coefficients:
        rows.append({"target": "all_component_means_average_squared_error",
                     "target_type": "average_squared_vector_error",
                     "local_coefficient": float(coefficient),
                     "full_asymptotic_variance": vf, "shared_asymptotic_variance": vr,
                     "shared_to_full_variance_ratio": vr/vf, "limiting_bias": None,
                     "bias_squared": coefficient**2*bias2,
                     "full_limiting_normal_risk": vf,
                     "shared_limiting_normal_risk": vr+coefficient**2*bias2,
                     "shared_to_full_risk_ratio": (vr+coefficient**2*bias2)/vf})
    return rows


def scenarios(config, include_d10=False):
    result = []
    for d, k in [(2, 2), (6, 3)]+([(10, 3)] if include_d10 else []):
        candidates = [cell for cell in config["cells"] if cell["family"] == "ld"
                      and cell["dimension"] == d and cell["components"] == k
                      and cell["heterogeneity"] == 0 and cell["concentration"] == 1
                      and cell["weight_profile"] == "balanced"]
        if not candidates:
            raise ValueError(f"No phase-2 base cell for d={d}, k={k}")
        for name, separation in [("overlap", 1.5), ("separated", 3.)]:
            result.append(dict(candidates[0], id=f"d{d}_k{k}_{name}", separation=separation))
    return result


def json_value(value):
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(json_value(value), indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(config, output, batches=5, draws_per_batch=20000, chunk_size=4096,
        seed=2026110705, selected=None, include_d10=False):
    if batches < 2 or draws_per_batch < 2:
        raise ValueError("At least two independent batches and two draws per batch required")
    cells = scenarios(config, include_d10)
    if selected:
        missing = set(selected)-{cell["id"] for cell in cells}
        if missing:
            raise ValueError(f"Unknown scenarios: {sorted(missing)}")
        cells = [cell for cell in cells if cell["id"] in selected]
    coefficients = sorted({0.} | {float(cell["local_h_coefficient"]) for cell in config["cells"]
                                 if cell.get("block") == "local_alternative"})
    if any(value < 0 for value in coefficients):
        raise ValueError("This bridge uses the nonnegative local h direction")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output/"source_snapshot"
    snapshot.mkdir()
    dependencies = [Path(__file__), Path(simulation.__file__), Path(shared.__file__),
                    ROOT/"scripts/run_confirmatory_experiments.py",
                    ROOT/"scripts/regularized_skew_normal.py"]
    hashes = {}
    for source in dependencies:
        shutil.copy2(source, snapshot/source.name)
        hashes[source.name] = hashlib.sha256((snapshot/source.name).read_bytes()).hexdigest()
    manifest = {"status": "running", "method": "known-truth Monte Carlo score outer products; no fitting",
                "batches": batches, "draws_per_batch": draws_per_batch,
                "chunk_size": chunk_size, "seed": seed, "threads": 1,
                "scenarios": cells, "local_coefficients": coefficients,
                "configuration": config, "source_sha256": hashes,
                "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
                "mc_uncertainty": "Independent batch plug-in estimates: sample SD divided by sqrt(batch count). This approximates uncertainty of pooled nonlinear estimates; finite Monte Carlo inverse bias is not removed.",
                "interpretation": "Local likelihood theory at a known shared truth. Limiting-normal risk is not an actual finite-sample MSE guarantee. Squared mean-RMSE loss is averaged entrywise; mean RMSE itself has a different expectation.",
                "information_estimator": "uncentered score second moment; no ridge, eigenvalue flooring, or pseudoinverse"}
    write_json(output/"manifest.json", manifest)
    batch_rows, diagnostics, pooled_rows = [], [], []
    started = time.perf_counter()
    try:
        with threadpool_limits(1):
            for cell in cells:
                k, d = cell["components"], cell["dimension"]
                spec, direction, path_details = generator_path_derivative(cell, config)
                theta = raw_from_specification(spec)
                J = shared_embedding(k, d)
                targets = target_definitions(theta, k, d)
                folder = output/cell["id"]
                folder.mkdir()
                write_json(folder/"truth_and_path.json", {"cell": cell, "specification": spec,
                           "raw_theta": theta, "raw_path_direction": direction,
                           "path_details": path_details, "targets": targets})
                pooled_I = np.zeros((len(theta), len(theta)))
                pooled_mean = np.zeros(len(theta))
                cell_batches = []
                key = int.from_bytes(hashlib.sha256(cell["id"].encode()).digest()[:4], "little")
                for batch in range(batches):
                    sequence = np.random.SeedSequence([seed, key, batch])
                    batch_start = time.perf_counter()
                    I, mean, score_diagnostics = estimate_information(
                        theta, spec, draws_per_batch, np.random.default_rng(sequence), chunk_size)
                    np.savez_compressed(folder/f"batch_{batch:02d}.npz", information=I, score_mean=mean)
                    full, restricted, P, algebra = information_projection(I, J)
                    pooled_I += I/batches
                    pooled_mean += mean/batches
                    entry = {"scenario": cell["id"], "batch": batch,
                             "seed_words": sequence.generate_state(4).tolist(),
                             "seconds": time.perf_counter()-batch_start, **score_diagnostics, **algebra}
                    diagnostics.append(entry)
                    rows = [{"scenario": cell["id"], "batch": batch, **row} for row in
                            risk_rows(full, restricted, P, targets, direction, coefficients, k, d)]
                    cell_batches.append(rows)
                    batch_rows.extend(rows)
                    write_json(folder/f"batch_{batch:02d}.json", {"diagnostics": entry, "targets": rows})
                full, restricted, P, algebra = information_projection(pooled_I, J)
                np.savez_compressed(folder/"pooled.npz", information=pooled_I, score_mean=pooled_mean,
                                    J=J, full_covariance=full, shared_covariance=restricted,
                                    projection=P, raw_path_direction=direction)
                pooled = risk_rows(full, restricted, P, targets, direction, coefficients, k, d)
                for row_index, row in enumerate(pooled):
                    result = {"scenario": cell["id"], "dimension": d, "components": k,
                              "total_draws": batches*draws_per_batch, **row}
                    for metric in ["full_asymptotic_variance", "shared_asymptotic_variance",
                                   "shared_to_full_variance_ratio", "bias_squared",
                                   "shared_limiting_normal_risk", "shared_to_full_risk_ratio"]:
                        values = np.array([items[row_index][metric] for items in cell_batches])
                        result[metric+"_batch_mean"] = float(values.mean())
                        result[metric+"_batch_mcse"] = float(values.std(ddof=1)/np.sqrt(batches))
                    pooled_rows.append(result)
                write_json(folder/"pooled_diagnostics.json", {**algebra,
                           "score_mean_max_abs": float(np.max(np.abs(pooled_mean))),
                           "score_mean_max_standardized": float(np.max(np.abs(
                               pooled_mean/np.sqrt(np.diag(pooled_I)/(batches*draws_per_batch))))),
                           "total_draws": batches*draws_per_batch})
                print(json.dumps({"scenario": cell["id"], "completed_batches": batches,
                                  "seconds_so_far": time.perf_counter()-started}), flush=True)
        write_csv(output/"batch_targets.csv", batch_rows)
        write_csv(output/"batch_diagnostics.csv", [{**row, "seed_words": json.dumps(row["seed_words"])} for row in diagnostics])
        write_csv(output/"projection_summary.csv", pooled_rows)
        manifest.update(status="complete", elapsed_seconds=time.perf_counter()-started,
                        total_generated_observations=len(cells)*batches*draws_per_batch)
        write_json(output/"manifest.json", manifest)
    except Exception as exc:
        write_json(output/"failure.json", {"exception": type(exc).__name__, "message": str(exc),
                   "traceback": traceback.format_exc(), "completed_batches": len(diagnostics)})
        manifest.update(status="failed", elapsed_seconds=time.perf_counter()-started)
        write_json(output/"manifest.json", manifest)
        raise
    return pooled_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=5)
    parser.add_argument("--draws-per-batch", type=int, default=20000)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=2026110705)
    parser.add_argument("--scenario", action="append")
    parser.add_argument("--include-d10", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    run(config, args.output, args.batches, args.draws_per_batch, args.chunk_size,
        args.seed, args.scenario, args.include_d10)


if __name__ == "__main__":
    main()
