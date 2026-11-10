"""Independent phase-2 simulation tasks; no large run occurs on import.

Freeze first: python scripts/phase2_simulations.py --freeze --output RUN
One shard:   python scripts/phase2_simulations.py --task-index 0 --output RUN
Batch:       python scripts/phase2_simulations.py --task-range 0:20 --output RUN
Aggregate:   python scripts/phase2_simulations.py --aggregate --output RUN
Task indices and replicate numbers are zero based. Existing attempts are never
overwritten. --attempt 2 records an infrastructure retry under the same seeds.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time
import traceback

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.special import digamma, gammaln, logsumexp, polygamma
from scipy.stats import multivariate_normal, multivariate_t
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

import run_confirmatory_experiments as fitters
import regularized_skew_normal as skew

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/phase2_simulations.json"
FAMILIES = {"ld", "native_dirichlet", "gaussian", "student_t", "skew_normal", "negative_correlation"}
VERSIONS = ["numpy", "scipy", "pandas", "scikit-learn", "fmvmm", "threadpoolctl"]


def canonical_hash(value):
    return hashlib.sha256(json.dumps(fitters.serializable(value), sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str(value.dtype).encode()+str(value.shape).encode()+value.tobytes()).hexdigest()


def tasks(config):
    index = 0
    for cell in config["cells"]:
        for replicate in range(cell["repetitions"]):
            yield {"task_index": index, "cell_id": cell["id"], "replicate": replicate}
            index += 1


def seed_words(config, cell_id, replicate, purpose):
    text = f"{config['master_seed']}|{cell_id}|{replicate}|{purpose}"
    digest = hashlib.sha256(text.encode()).digest()
    return [int.from_bytes(digest[i:i+4], "little") for i in range(0, 16, 4)]


def generating_parameters(cell, config):
    """Moment-controlled generators; truth is never passed to fitting routines.

    Weighted coordinatewise mean log shape is log(c*base_alpha). Heterogeneity
    is exactly the weighted RMS log-shape deviation, with h=0 shared shape.
    All affine LD component marginal SDs are fixed while shapes vary.
    """
    d, k = cell["dimension"], cell["components"]
    g = config["generation"]
    weights = (np.full(k, 1/k) if cell["weight_profile"] == "balanced"
               else np.array(g[f"imbalanced_weights_k{k}"]))
    alpha0 = np.r_[np.resize(g["base_shape_cycle"], d), g["reference_shape"]]*cell["concentration"]
    contrast = np.cos(2*np.pi*np.arange(k)[:, None]/k+2*np.pi*np.arange(d+1)[None, :]/(d+1))
    contrast -= weights @ contrast
    contrast /= np.sqrt(np.sum(weights[:, None]*contrast**2)/(d+1))
    alpha = alpha0[None, :]*np.exp(cell["heterogeneity"]*contrast)
    kernel_cov = np.array([np.diag(polygamma(1, a[:-1]))+polygamma(1, a[-1]) for a in alpha])
    target_sd = np.exp(g["log_standard_deviation_amplitude"]*
                       np.cos(2*np.pi*np.arange(k)[:, None]/k+2*np.pi*np.arange(d)[None, :]/d))
    scales = target_sd/np.sqrt(np.diagonal(kernel_cov, axis1=1, axis2=2))
    covariances = kernel_cov*scales[:, :, None]*scales[:, None, :]
    if cell["family"] == "negative_correlation":
        correlation = g["negative_ar1_rho"]**abs(np.arange(d)[:, None]-np.arange(d)[None, :])
        covariances = correlation[None, :, :]*target_sd[:, :, None]*target_sd[:, None, :]
    # Dense orthonormal directions, with the same equilateral geometry for k=3.
    directions = np.column_stack([np.ones(d)/np.sqrt(d),
                                  np.sqrt(2/d)*np.cos(np.pi*(np.arange(d)+.5)/d)])
    vertices = np.column_stack([np.cos(2*np.pi*np.arange(k)/k), np.sin(2*np.pi*np.arange(k)/k)])
    means = vertices @ directions.T
    means -= weights @ means
    average_covariance = np.einsum("j,jab->ab", weights, covariances)
    inverse = np.linalg.inv(average_covariance)
    distances = [np.sqrt((means[j]-means[l]) @ inverse @ (means[j]-means[l]))
                 for j in range(k) for l in range(j)]
    means *= cell["separation"]/min(distances)
    locations = means-scales*(digamma(alpha[:, :-1])-digamma(alpha[:, -1:]))
    if cell["family"] == "native_dirichlet":
        scales = np.ones((k, d)); locations = np.zeros((k, d))
        means = digamma(alpha[:, :-1])-digamma(alpha[:, -1:])
        covariances = kernel_cov
    spec = {"family": cell["family"], "weights": weights, "alpha": alpha,
            "base_alpha": alpha0, "shape_contrast": contrast, "means": means,
            "scales": scales, "locations": locations, "covariances": covariances,
            "student_t_df": g["student_t_df"]}
    if cell["family"] == "skew_normal":
        delta = np.array([np.linalg.cholesky(covariances[j]) @ directions[:, j % 2]*g["skew_delta_norm"]
                          for j in range(k)])
        omega = covariances+(2/np.pi)*delta[:, :, None]*delta[:, None, :]
        beta = np.array([np.linalg.solve(omega[j], delta[j]) /
                         np.sqrt(1-delta[j] @ np.linalg.solve(omega[j], delta[j])) for j in range(k)])
        spec.update({"skew_delta": delta, "skew_omega": omega, "skew_beta": beta,
                     "skew_locations": means-np.sqrt(2/np.pi)*delta})
    return spec


def generate_sample(n, spec, rng):
    weights, means = np.asarray(spec["weights"]), np.asarray(spec["means"])
    k, d = means.shape
    labels = rng.choice(k, size=n, p=weights)
    y = np.empty((n, d))
    for j in range(k):
        selected = labels == j; count = int(selected.sum())
        if not count:
            continue
        family = spec["family"]
        if family in ["ld", "native_dirichlet"]:
            gamma = rng.gamma(spec["alpha"][j], size=(count, d+1))
            if np.any(gamma <= 0):
                raise FloatingPointError("Gamma underflow in generator; no clipping or resampling")
            logs = np.log(gamma)
            y[selected] = spec["locations"][j]+spec["scales"][j]*(logs[:, :-1]-logs[:, -1:])
        elif family == "student_t":
            df = spec["student_t_df"]
            normal = rng.multivariate_normal(np.zeros(d), spec["covariances"][j]*(df-2)/df, count)
            y[selected] = means[j]+normal/np.sqrt(rng.chisquare(df, count)/df)[:, None]
        elif family == "skew_normal":
            delta = spec["skew_delta"][j]
            residual = spec["skew_omega"][j]-np.outer(delta, delta)
            y[selected] = (spec["skew_locations"][j]+rng.multivariate_normal(np.zeros(d), residual, count)
                           +abs(rng.normal(size=count))[:, None]*delta)
        else:
            y[selected] = rng.multivariate_normal(means[j], spec["covariances"][j], count)
    if not np.isfinite(y).all():
        raise FloatingPointError("Nonfinite generated observations")
    return y, labels


def oracle_density(y, spec):
    terms = []
    for j, weight in enumerate(spec["weights"]):
        family = spec["family"]
        if family in ["ld", "native_dirichlet"]:
            a = spec["alpha"][j]
            z = (y-spec["locations"][j])/spec["scales"][j]
            density = (gammaln(a.sum())-gammaln(a).sum()-np.log(spec["scales"][j]).sum()
                       +fitters.shared.log_simplex(z) @ a)
        elif family == "student_t":
            df = spec["student_t_df"]
            density = multivariate_t.logpdf(y, loc=spec["means"][j],
                                           shape=spec["covariances"][j]*(df-2)/df, df=df)
        elif family == "skew_normal":
            from scipy.special import log_ndtr
            residual = y-spec["skew_locations"][j]
            density = (np.log(2)+multivariate_normal.logpdf(y, mean=spec["skew_locations"][j],
                       cov=spec["skew_omega"][j])+log_ndtr(residual @ spec["skew_beta"][j]))
        else:
            density = multivariate_normal.logpdf(y, mean=spec["means"][j], cov=spec["covariances"][j])
        terms.append(np.log(weight)+density)
    terms = np.column_stack(terms); density = logsumexp(terms, axis=1)
    return density, np.exp(terms-density[:, None])


def standardize_training(train, test):
    center, scale = train.mean(axis=0), train.std(axis=0, ddof=0)
    if np.any(scale <= 0) or not np.isfinite(scale).all():
        raise ValueError("Training SD is zero or nonfinite; no silent replacement")
    return (train-center)/scale, (test-center)/scale, center, scale


def partitions_for(y, k, seed, count):
    return [KMeans(k, init="k-means++" if j < 3 else "random", n_init=1,
                   random_state=int((seed+j) % (2**32))).fit_predict(y) for j in range(count)]


def schedule(config, replicate):
    answer = [("primary", name) for name in config["primary_methods"]]
    secondary = config["secondary"]
    if replicate < secondary["bounded_skew_repetitions_per_cell"]:
        answer.append(("primary", "bounded_skew_normal"))
    if replicate < secondary["ten_start_repetitions_per_cell"]:
        answer += [("ten_starts", name) for name in config["primary_methods"]]
    if replicate < secondary["wide_bound_repetitions_per_cell"]:
        answer += [("wide_bounds", name) for name in ["shared_ld", "full_ld"]]
    return answer


@contextmanager
def bounded_deadlines(seconds):
    """Deadline and endpoint instrumentation; no existing source is edited."""
    original = skew.minimize
    calls = defaultdict(deque)
    def wrapped(fun, initial, *args, **kwargs):
        item = {"endpoint": None}
        calls[array_hash(np.asarray(initial))].append(item)
        with fitters.timed_start(seconds):
            result = original(fun, initial, *args, **kwargs)
        item["endpoint"] = np.asarray(result.x).copy()
        return result
    skew.minimize = wrapped
    try:
        yield calls
    finally:
        skew.minimize = original


def attach_skew_endpoints(records, calls, k, d):
    for record in records:
        initial = record.get("initial_parameters")
        if initial is None:
            continue
        initial = np.asarray(initial)
        if record["initialization"] == "exact_Gaussian_candidate":
            endpoint = initial
        else:
            queue = calls[array_hash(initial)]
            endpoint = queue.popleft()["endpoint"] if queue else None
            if record.get("discarded_worse_optimizer_endpoint"):
                endpoint = initial
        record["parameters"] = (skew.parameters(endpoint, k, d)
                                if endpoint is not None and not record["failed"] else None)


def fit_one(method, train, k, seed, settings, partitions, dependency=None, skew_settings=None):
    if method != "bounded_skew_normal":
        internal = {"fmvmm_dmm_native": "fmvmm_dmm", "base_ld_native": "base_ld"}.get(method, method)
        return fitters.fit_method(internal, train, k, seed, settings=settings,
                                  partitions=partitions, warm_start=dependency)
    with bounded_deadlines(settings["start_timeout_seconds"]) as calls:
        try:
            model = skew.fit(train, k, seed, partitions=partitions, gaussian_candidate=dependency,
                             n_starts=len(partitions), **{key: skew_settings[key]
                             for key in ["maxiter", "polish_maxiter", "gtol"]})
        except Exception as exc:
            attach_skew_endpoints(getattr(exc, "starts", []), calls, k, train.shape[1])
            raise
    attach_skew_endpoints(model.starts, calls, k, train.shape[1])
    return model


def mean_matching(means, truth):
    if not np.isfinite(means).all():
        return None, np.nan
    left, right = linear_sum_assignment(((means[:, None]-truth[None, :])**2).sum(axis=2))
    order = left[np.argsort(right)]
    return order, float(np.sqrt(np.mean((means[order]-truth)**2)))


def recovery(model, method, means, spec, cell):
    order, mean_rmse = mean_matching(means, spec["means"])
    result = {"mean_rmse": mean_rmse, "weight_rmse": np.nan,
              "log_shape_rmse": np.nan, "shape_recovery_applicable": False}
    if order is None:
        return result
    params = model.params
    if method in ["shared_ld", "full_ld", "base_ld_native"]:
        internal = "base_ld" if method == "base_ld_native" else method
        logpi, alpha, _, _ = fitters.unpack_ld(np.asarray(params["theta"]), internal, len(order), means.shape[1])
        weights = np.exp(logpi)
        if len(alpha) == 1:
            alpha = np.repeat(alpha, len(order), axis=0)
    else:
        weights = np.asarray(params["weights"])
        alpha = np.asarray(params["alpha"]) if method == "fmvmm_dmm_native" else None
    result["weight_rmse"] = float(np.sqrt(np.mean((weights[order]-spec["weights"])**2)))
    applicable = (spec["family"] in ["ld", "native_dirichlet"] and
                  (method == "full_ld" or method == "shared_ld" and cell["heterogeneity"] == 0 or
                   method in ["base_ld_native", "fmvmm_dmm_native"] and spec["family"] == "native_dirichlet"))
    if applicable:
        result.update({"shape_recovery_applicable": True,
                       "log_shape_rmse": float(np.sqrt(np.mean((np.log(alpha[order])-np.log(spec["alpha"]))**2)))})
    return result


def validate_config(config):
    cells = config["cells"]
    if len({c["id"] for c in cells}) != len(cells):
        raise ValueError("Duplicate cell identifiers")
    if len(cells) != config["cell_count"] or sum(c["repetitions"] for c in cells) != config["dataset_count"]:
        raise ValueError("Declared design counts are inconsistent")
    for cell in cells:
        if cell["family"] not in FAMILIES or cell["components"] not in [2, 3] or cell["dimension"] < 2:
            raise ValueError("Unsupported generating cell")
        spec = generating_parameters(cell, config)
        if not np.isfinite(spec["alpha"]).all() or np.min(spec["alpha"]) <= 0:
            raise ValueError("Invalid generating shape")
    declared = config["bounded_skew"]
    for name, current in [("location_bounds", skew.LOCATION_BOUNDS),
                          ("cholesky_diagonal_bounds", skew.CHOLESKY_DIAGONAL_BOUNDS),
                          ("cholesky_offdiagonal_bounds", skew.CHOLESKY_OFFDIAGONAL_BOUNDS),
                          ("beta_bounds", skew.BETA_BOUNDS)]:
        if tuple(declared[name]) != tuple(current):
            raise ValueError("Bounded-skew module and declared constraints differ")


def source_paths():
    return [Path(__file__), ROOT/"scripts/run_confirmatory_experiments.py",
            ROOT/"scripts/shared_shape_ld.py", ROOT/"scripts/regularized_skew_normal.py",
            ROOT/"configs/experiments.json", ROOT/"requirements-lock.txt"]


def freeze_run(config, output):
    validate_config(config)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"configuration": config, "configuration_sha256": canonical_hash(config),
                "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in source_paths()},
                "versions": {package: importlib.metadata.version(package) for package in VERSIONS},
                "python": platform.python_version(), "threads": 1,
                "task_count": config["dataset_count"], "indexing": "zero-based task and replicate indices"}
    fitters.write_json(output/"manifest.json", manifest)
    for source in source_paths():
        destination = output/"source_snapshot"/source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    fitters.write_json(output/"source_snapshot/configs/phase2_simulations.json", config)
    with (output/"tasks.jsonl").open("w") as handle:
        for task in tasks(config):
            handle.write(json.dumps(task)+"\n")
    return manifest


def verify_run(output):
    manifest = json.loads((output/"manifest.json").read_text())
    if canonical_hash(manifest["configuration"]) != manifest["configuration_sha256"]:
        raise ValueError("Frozen configuration hash does not match")
    for relative, digest in manifest["source_sha256"].items():
        if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Source changed after freezing: {relative}")
    for package, version in manifest["versions"].items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f"Environment differs for {package}; freeze a separate run")
    if platform.python_version_tuple()[:2] != tuple(manifest["python"].split(".")[:2]):
        raise ValueError("Python major/minor version differs from the frozen run")
    return manifest["configuration"]


def run_task(config, task_index, output, attempt=1, fitter=fit_one):
    task_list = list(tasks(config))
    if not 0 <= task_index < len(task_list) or attempt < 1:
        raise ValueError("Invalid task or attempt index")
    task = task_list[task_index]
    cell = next(cell for cell in config["cells"] if cell["id"] == task["cell_id"])
    directory = output/"tasks"/f"task_{task_index:06d}"/f"attempt_{attempt:03d}"
    directory.mkdir(parents=True, exist_ok=False)
    planned = schedule(config, task["replicate"])
    seeds = {purpose: seed_words(config, cell["id"], task["replicate"], purpose)
             for purpose in ["train", "test", "initialization"]}
    fitters.write_json(directory/"task.json", {**task, "attempt": attempt, "cell": cell,
                       "configuration_sha256": canonical_hash(config), "seeds": seeds,
                       "environment": {"python": platform.python_version(), "host": "not_recorded_in_public_rerun",
                                       "platform": platform.platform(), "process_id": os.getpid(),
                                       "started_unix_seconds": time.time(),
                                       "versions": {p: importlib.metadata.version(p) for p in VERSIONS}},
                       "planned_fits": planned, "status": "started"})
    beginning = time.perf_counter(); rows = []; fitted = {}; setup_exception = None
    try:
        spec = generating_parameters(cell, config)
        train, train_labels = generate_sample(cell["n_train"], spec, np.random.default_rng(seeds["train"]))
        test, labels = generate_sample(config["n_test"], spec, np.random.default_rng(seeds["test"]))
        standardized, standardized_test, center, scale = standardize_training(train, test)
        jacobian = float(np.log(scale).sum())
        seed = seeds["initialization"][0]
        partition_count = 10 if any(variant == "ten_starts" for variant, _ in planned) else 3
        with threadpool_limits(1):
            partitions = partitions_for(standardized, cell["components"], seed, partition_count)
        oracle, oracle_tau = oracle_density(test, spec)
        np.savez_compressed(directory/"data.npz", train=train, test=test, train_labels=train_labels,
                            test_labels=labels, center=center, scale=scale, partitions=np.array(partitions),
                            oracle_log_density=oracle, oracle_responsibilities=oracle_tau)
        fitters.write_json(directory/"generation.json", {"parameters": spec,
                           "array_sha256": {"train": array_hash(train), "test": array_hash(test)},
                           "oracle_mean_log_density": float(oracle.mean()),
                           "oracle_expected_classification_error": float(np.mean(1-oracle_tau.max(axis=1))),
                           "oracle_empirical_classification_error": float(np.mean(oracle_tau.argmax(axis=1) != labels))})
    except Exception as exc:
        setup_exception = {"exception": repr(exc), "traceback": traceback.format_exc()}
        fitters.write_json(directory/"setup_failure.json", setup_exception)
    with (directory/"fits.jsonl").open("w") as table, threadpool_limits(1):
        for variant, method in planned:
            key = f"{variant}__{method}"
            row = {**task, "attempt": attempt, "variant": variant, "method": method,
                   "n_train": cell["n_train"], "n_test": config["n_test"], "d": cell["dimension"],
                   "k": cell["components"], "failed": True, "fit_available": False,
                   "evaluation_available": False, "test_log_density": None, "train_log_density": None,
                   "mean_rmse": None, "weight_rmse": None, "log_shape_rmse": None,
                   "shape_recovery_applicable": False, "ARI": None, "fit_seconds": 0.,
                   "dependency_fit_seconds": 0., "policy_fit_seconds": 0., "exception": None,
                   "optimizer_success": None, "score_converged": None, "boundary_hit": None,
                   "projected_gradient": None, "parameter_count": None, "starts_attempted": 0, "starts_failed": 0}
            model = None; starts = []; dependency = None; dependency_cost = 0.; t0 = time.perf_counter()
            try:
                if setup_exception:
                    raise RuntimeError("Task setup failed: "+setup_exception["exception"])
                settings = dict(config["fitting"])
                if variant == "wide_bounds":
                    settings.update(config["wide_bounds"])
                settings["n_starts"] = 10 if variant == "ten_starts" else 3
                native = method.endswith("_native")
                if native:
                    settings["alpha_bounds"] = config["native_alpha_bounds"]
                if method == "full_ld":
                    dependency_key = (variant, "shared_ld")
                elif method == "base_ld_native":
                    dependency_key = (variant, "fmvmm_dmm_native")
                elif method == "bounded_skew_normal":
                    dependency_key = ("primary", "gaussian_full")
                else:
                    dependency_key = None
                if dependency_key in fitted:
                    dependency, dependency_cost = fitted[dependency_key]
                fitting_data = train if native else standardized
                fit_start = time.perf_counter()
                model = fitter(method, fitting_data, cell["components"], seed, settings,
                               partitions[:settings["n_starts"]], dependency, config["bounded_skew"])
                own_cost = time.perf_counter()-fit_start
                starts = model.starts
                fitted[(variant, method)] = (model, own_cost)
                row.update({"fit_available": True, "fit_seconds": own_cost,
                            "dependency_fit_seconds": dependency_cost,
                            "policy_fit_seconds": own_cost+dependency_cost,
                            "starts_attempted": len(starts),
                            "starts_failed": sum(record["failed"] for record in starts)})
                for name in ["optimizer_success", "score_converged", "boundary_hit", "projected_gradient", "parameter_count"]:
                    row[name] = model.record.get(name)
                row["train_log_density"] = model.record["mean_train_log_density"]-(0 if native else jacobian)
                density, tau = model.evaluate(test if native else standardized_test)
                density = density-(0 if native else jacobian)
                if not np.isfinite(density).all() or not np.isfinite(tau).all():
                    raise FloatingPointError("Nonfinite held-out evaluation; fitted point is retained")
                means = model.component_means()
                if not native:
                    means = center+scale*means
                row.update(recovery(model, method, means, spec, cell))
                difference = density-oracle
                row.update({"failed": False, "evaluation_available": True,
                            "test_log_density": float(density.mean()),
                            "test_log_density_minus_oracle": float(difference.mean()),
                            "test_integration_mcse": float(difference.std(ddof=1)/np.sqrt(len(test))),
                            "ARI": float(adjusted_rand_score(labels, tau.argmax(axis=1))),
                            "standardization_log_jacobian": 0. if native else jacobian,
                            "fitting_coordinates": "raw_ALR" if native else "training_standardized"})
                np.savez_compressed(directory/f"{key}.npz", log_density=density,
                                    cluster=tau.argmax(axis=1), component_means=means)
            except Exception as exc:
                row["exception"] = repr(exc)
                row["traceback"] = traceback.format_exc()
                if model is None:
                    starts = getattr(exc, "starts", [])
                    row.update({"fit_seconds": time.perf_counter()-t0,
                                "dependency_fit_seconds": dependency_cost,
                                "policy_fit_seconds": time.perf_counter()-t0+dependency_cost,
                                "starts_attempted": len(starts), "starts_failed": sum(s["failed"] for s in starts)})
            fitters.write_json(directory/f"{key}.json", {"row": row, "parameters": model.params if model is not None else None,
                              "selected_start": model.selected_start if model is not None else None,
                              "record": model.record if model is not None else None, "starts": starts})
            table.write(json.dumps(fitters.serializable(row), allow_nan=False)+"\n"); table.flush()
            rows.append(row)
    completion = {**task, "attempt": attempt, "planned": len(planned), "recorded": len(rows),
                  "failed": sum(row["failed"] for row in rows), "elapsed_seconds": time.perf_counter()-beginning}
    fitters.write_json(directory/"completion.json", completion)
    return completion


def aggregate(config, output):
    rows = []; complete = 0; missing_tasks = 0
    for task in tasks(config):
        folder = output/"tasks"/f"task_{task['task_index']:06d}"
        attempts = sorted(folder.glob("attempt_*")) if folder.exists() else []
        recorded = {}
        if attempts:
            latest = attempts[-1]
            if (latest/"completion.json").exists():
                complete += 1
            if (latest/"fits.jsonl").exists():
                with (latest/"fits.jsonl").open() as handle:
                    for line in handle:
                        try:
                            value = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        recorded[(value["variant"], value["method"])] = value
        else:
            missing_tasks += 1
        for variant, method in schedule(config, task["replicate"]):
            value = recorded.get((variant, method))
            rows.append({**task, "variant": variant, "method": method,
                         "recorded": value is not None,
                         **(value or {"failed": True, "fit_available": False, "evaluation_available": False})})
    frame = pd.DataFrame(rows)
    for name in ["test_log_density", "mean_rmse", "weight_rmse", "log_shape_rmse", "fit_seconds", "policy_fit_seconds", "score_converged", "boundary_hit"]:
        if name not in frame:
            frame[name] = np.nan
    frame.to_csv(output/"fits.csv", index=False)
    summaries = []; paired = []
    for (cell, variant, method), group in frame.groupby(["cell_id", "variant", "method"], sort=False):
        finite = group[group.evaluation_available]
        summary = {"cell_id": cell, "variant": variant, "method": method, "planned": len(group),
                   "recorded": int(group.recorded.sum()), "missing": int((~group.recorded).sum()),
                   "evaluation_available": len(finite), "failed_recorded": int((group.recorded & group.failed).sum()),
                   "score_checked": int(group.score_converged.notna().sum()),
                   "score_passed": int((group.score_converged == True).sum()),
                   "boundary_checked": int(group.boundary_hit.notna().sum()),
                   "boundary_hits": int((group.boundary_hit == True).sum()),
                   "mean_fit_seconds": group.fit_seconds.mean(), "mean_policy_fit_seconds": group.policy_fit_seconds.mean()}
        for measure in ["test_log_density", "mean_rmse", "weight_rmse", "log_shape_rmse"]:
            values = pd.to_numeric(finite[measure], errors="coerce").dropna()
            summary[measure+"_available"] = len(values)
            summary[measure+"_mean"] = values.mean()
            summary[measure+"_MCSE"] = values.std(ddof=1)/np.sqrt(len(values)) if len(values) else np.nan
        summaries.append(summary)
        if method != "shared_ld":
            primary = frame[(frame.cell_id == cell) & (frame.variant == variant) & (frame.method == "shared_ld")]
            pairs = primary.merge(group, on="task_index", suffixes=("_shared", "_other"))
            good = pairs[pairs.evaluation_available_shared & pairs.evaluation_available_other]
            result = {"cell_id": cell, "variant": variant, "comparator": method,
                      "planned_pairs": len(pairs), "available_pairs": len(good)}
            for measure in ["test_log_density", "mean_rmse", "weight_rmse"]:
                delta = (pd.to_numeric(good[measure+"_shared"], errors="coerce")-
                         pd.to_numeric(good[measure+"_other"], errors="coerce")).dropna()
                result[measure+"_shared_minus_other"] = delta.mean()
                result[measure+"_difference_MCSE"] = delta.std(ddof=1)/np.sqrt(len(delta)) if len(delta) else np.nan
            paired.append(result)
    pd.DataFrame(summaries).to_csv(output/"summary.csv", index=False)
    pd.DataFrame(paired).to_csv(output/"paired_comparisons.csv", index=False)
    fitters.write_json(output/"aggregation.json", {"planned_tasks": config["dataset_count"],
                       "complete_tasks": complete, "absent_tasks": missing_tasks,
                       "planned_fit_rows": len(frame), "recorded_fit_rows": int(frame.recorded.sum()),
                       "selection": "Latest attempt number, irrespective of performance or completion",
                       "interpretation": "Means and MCSE condition on available finite evaluations; missing and failed denominators remain explicit"})
    return {"complete_tasks": complete, "absent_tasks": missing_tasks, "recorded_fits": int(frame.recorded.sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--task-index", type=int)
    group.add_argument("--task-range", help="Half-open START:STOP indices; one import for several independent tasks")
    group.add_argument("--aggregate", action="store_true")
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    if args.freeze:
        manifest = freeze_run(json.loads(args.config.read_text()), args.output)
        print(json.dumps({"task_count": manifest["task_count"], "configuration_sha256": manifest["configuration_sha256"]}))
    else:
        config = verify_run(args.output)
        if args.aggregate:
            print(json.dumps(aggregate(config, args.output)))
        elif args.task_range:
            start, stop = map(int, args.task_range.split(":"))
            if not 0 <= start < stop <= config["dataset_count"]:
                raise ValueError("Invalid half-open task range")
            with threadpool_limits(1):
                for index in range(start, stop):
                    print(json.dumps(run_task(config, index, args.output, args.attempt)), flush=True)
        else:
            with threadpool_limits(1):
                print(json.dumps(run_task(config, args.task_index, args.output, args.attempt)))


if __name__ == "__main__":
    main()
