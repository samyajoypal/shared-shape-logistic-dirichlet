#!/usr/bin/env python3
"""Independent, restartable outer tasks comparing LD confidence intervals.

One --task-index selects one scenario/size/outer-replication combination.
The only random data used to fit an interval are the observed sample and its
parametric-bootstrap samples. Labels are defined by ordered component means.
Final files are replaced atomically. An fsynced append-only JSONL checkpoint
retains each bootstrap fit; only an incomplete final line is repairable.
Finished tasks keep a compressed checkpoint and do not create per-draw files.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import tempfile
import time
import traceback
import warnings

import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment
from scipy.special import digamma, expit
from scipy.stats import binomtest, norm
import sklearn
from threadpoolctl import threadpool_limits

import shared_shape_ld as ld
import ld_information as info_module
from ld_information import information, component_mean_and_gradient

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "phase2_inference.json"
TARGETS = ("alpha_1", "lower_mean_weight", "mean_contrast_0")
METHODS = ("wald", "bootstrap_percentile", "bootstrap_basic", "bootstrap_t")


def serializable(value):
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return serializable(value.tolist())
    if isinstance(value, np.generic):
        return serializable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def json_bytes(value):
    return (json.dumps(serializable(value), sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path, value):
    atomic_bytes(path, json_bytes(value))


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def validate_config(config):
    if config["schema_version"] != 1:
        raise ValueError("Unsupported configuration schema")
    model = config["generating_model"]
    if (model["components"], model["dimension"]) != (2, 2):
        raise ValueError("This interval study is specified for k=2,d=2")
    if config["replications_per_cell"] < 1 or any(n < 3 for n in config["sample_sizes"]):
        raise ValueError("Invalid replication count or sample size")
    fitting, intervals = config["fit"], config["intervals"]
    if (fitting["alpha_bounds"] != list(ld.ALPHA_BOUNDS)
            or fitting["inverse_scale_bounds"] != list(ld.A_BOUNDS)
            or fitting["scaled_location_bounds"] != list(ld.B_BOUNDS)):
        raise ValueError("Configuration bounds differ from the imported shared-shape fitter")
    if fitting["ftol"] != 1e-14 or fitting["maxls"] != 40:
        raise ValueError("Configuration must state the actual imported fitter tolerances")
    if fitting["n_starts"] < 1 or fitting["maxiter"] < 1 or fitting["continuation_maxiter"] < 0:
        raise ValueError("Invalid fitting budget")
    if fitting["gtol"] <= 0 or config["threads"] != 1:
        raise ValueError("Positive gradient tolerance and one CPU thread are required")
    if tuple(intervals["methods"]) != METHODS or tuple(intervals["targets"]) != TARGETS:
        raise ValueError("The four interval methods and three targets are fixed")
    if tuple(intervals["working_coordinates"]) != ("log", "logit", "identity"):
        raise ValueError("Unsupported target transformation")
    if intervals["quantile_method"] != "linear":
        raise ValueError("This specification uses linear empirical quantiles")
    if not 0 < intervals["confidence"] < 1 or not 0 < intervals["minimum_valid_fraction"] <= 1:
        raise ValueError("Invalid confidence or valid-draw fraction")
    if not 2 <= intervals["minimum_valid_draws"] <= intervals["bootstrap_draws"]:
        raise ValueError("Minimum valid bootstrap count must lie in [2,B]")
    if not intervals["bootstrap_percentile_basic_include_boundary_fits"]:
        raise ValueError("Percentile/basic boundary policy differs from this implementation")
    if not intervals["wald_and_bootstrap_t_require_interior_information"]:
        raise ValueError("Wald/bootstrap-t interior policy differs from this implementation")


def task_specification(config, task_index):
    count = config["replications_per_cell"]
    number_sizes = len(config["sample_sizes"])
    total = len(config["scenarios"]) * number_sizes * count
    if not 0 <= task_index < total:
        raise ValueError(f"Task index must lie in [0,{total - 1}]")
    cell, replicate = divmod(int(task_index), count)
    scenario_index, size_index = divmod(cell, number_sizes)
    scenario = config["scenarios"][scenario_index]
    return {"task_index": int(task_index), "cell_index": cell, "replicate": replicate,
            "scenario": scenario["name"], "separation_multiplier": scenario["separation_multiplier"],
            "n": config["sample_sizes"][size_index], "total_tasks": total}


def seed_sequence(config, task_index, stage, replicate=0):
    return np.random.SeedSequence(config["master_seed"],
                                  spawn_key=(int(task_index), int(stage), int(replicate)))


def seed_record(sequence):
    return {"entropy": sequence.entropy, "spawn_key": list(sequence.spawn_key),
            "state_words": sequence.generate_state(4).tolist()}


def generating_parameters(config, task):
    source = config["generating_model"]
    alpha = np.asarray(source["alpha"], dtype=float)
    means = np.asarray(source["base_component_means"]) * task["separation_multiplier"]
    scales = np.asarray(source["component_scales"])
    weights = np.asarray(source["weights"])
    locations = means - scales * (digamma(alpha[:-1]) - digamma(alpha[-1]))
    order = np.lexsort((means[:, 1], means[:, 0]))
    truths = {"alpha_1": float(alpha[0]), "lower_mean_weight": float(weights[order[0]]),
              "mean_contrast_0": float(means[order[1], 0] - means[order[0], 0])}
    return alpha, locations, scales, weights, means, truths


def target_values(theta, k=2, d=2):
    """Observable mean ordering; no true parameter or latent label is accepted."""
    logpi, alpha, a, b = ld.unpack(np.asarray(theta), k, d, False)
    weights = np.exp(logpi)
    means = (digamma(alpha[:-1]) - digamma(alpha[-1]) - b) / a
    order = np.lexsort((means[:, 1], means[:, 0]))
    first, second = map(int, order)
    gradient_alpha = np.zeros(len(theta))
    gradient_alpha[k - 1] = 1.0
    gradient_weight = np.zeros(len(theta))
    for j in range(k - 1):
        gradient_weight[j] = weights[first] * (float(first == j) - weights[j])
    mean_first, grad_first = component_mean_and_gradient(theta, k, d, first, 0, False)
    mean_second, grad_second = component_mean_and_gradient(theta, k, d, second, 0, False)
    values = np.array([alpha[0], weights[first], mean_second - mean_first])
    gradients = np.stack([gradient_alpha, gradient_weight, grad_second - grad_first])
    if not np.isfinite(values).all() or values[0] <= 0 or not 0 < values[1] < 1:
        raise FloatingPointError("Target values are outside the transformation domain")
    working = np.array([np.log(values[0]), np.log(values[1]) - np.log1p(-values[1]), values[2]])
    derivatives = np.array([1 / values[0], 1 / (values[1] * (1 - values[1])), 1.0])
    return {"values": values, "working_values": working,
            "working_gradients": gradients * derivatives[:, None],
            "order": order.tolist(), "component_means": means,
            "ordering_gap": float(values[2])}


def inverse_target(index, value):
    value = np.asarray(value)
    if index == 0:
        return np.exp(value)
    if index == 1:
        return expit(value)
    return value


def fit_model(y, seed, config, warm_start=None):
    """The existing bounded direct fitter, with one documented continuation."""
    beginning = time.perf_counter()
    records, candidates = [], []
    fitting = config["fit"]
    try:
        starts = ld.make_starts(y, 2, fitting["n_starts"], seed, False)
        if warm_start is not None and fitting["bootstrap_parent_warm_start"]:
            starts.append(np.asarray(warm_start).copy())
        rounds = [(starts, fitting["maxiter"], "multistart")]
        for round_index in range(2):
            if round_index >= len(rounds):
                break
            current_starts, maxiter, reason = rounds[round_index]
            offset = len(records)
            fitted = None
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    fitted = ld.fit(y, 2, common_scale=False, algorithm="direct", starts=current_starts,
                                    seed=seed, maxiter=maxiter, gtol=fitting["gtol"])
                new_records = fitted.records
                candidates.append((offset + fitted.selected_start, fitted.theta.copy()))
                warning_messages = [str(w.message) for w in caught]
            except ld.SharedShapeFitFailure as exc:
                new_records, warning_messages = exc.records, []
            except Exception as exc:
                new_records = [{"failed": True, "selected": False, "optimizer_success": False,
                                "score_converged": False, "boundary_hits": [], "iterations": 0,
                                "mean_train_log_density": None, "projected_gradient": None,
                                "min_weight": None, "message": repr(exc),
                                "exception_type": type(exc).__name__}
                               for _ in current_starts]
                warning_messages = []
            for local, record in enumerate(new_records):
                record = dict(record)
                record.update({"start": offset + local, "round": reason, "selected": False,
                               "initial_parameters": current_starts[local], "warnings": warning_messages})
                if reason == "multistart":
                    record["initialization"] = ("parent_fit_warm_start" if local >= fitting["n_starts"]
                                                 else "KMeans")
                else:
                    record["initialization"] = "selected_candidate_continuation"
                records.append(record)
            if round_index == 0 and candidates:
                best, theta = max(candidates, key=lambda item: records[item[0]]["mean_train_log_density"])
                if not records[best]["score_converged"] and fitting["continuation_maxiter"]:
                    rounds.append(([theta], fitting["continuation_maxiter"], "continuation"))
        if not candidates:
            raise RuntimeError("No finite candidate survived the fitting protocol")
        selected, theta = max(candidates, key=lambda item: records[item[0]]["mean_train_log_density"])
        records[selected]["selected"] = True
        return {"finite_fit": True, "theta": theta, "selected_start": selected,
                "record": records[selected], "starts": records,
                "elapsed_seconds": time.perf_counter() - beginning, "exception": ""}
    except Exception as exc:
        return {"finite_fit": False, "theta": None, "selected_start": None, "record": {},
                "starts": records, "elapsed_seconds": time.perf_counter() - beginning,
                "exception": repr(exc), "traceback": traceback.format_exc()}


def diagnose_fit(fitted, y, config):
    """Separate estimator availability from regular-information availability."""
    diagnostic = {"fit_reasons": [], "information_reasons": [], "bootstrap_estimate_eligible": False,
                  "regular_information_eligible": False, "targets": {}, "information": {}}
    if not fitted["finite_fit"]:
        diagnostic["fit_reasons"].append("fit_exception")
        return diagnostic
    record, theta = fitted["record"], np.asarray(fitted["theta"])
    if not record["score_converged"]:
        diagnostic["fit_reasons"].append("nonstationary")
    if record["min_weight"] <= config["intervals"]["minimum_weight"]:
        diagnostic["fit_reasons"].append("tiny_weight")
    if record["boundary_hits"]:
        diagnostic["information_reasons"].append("bound_hit")
    try:
        target = target_values(theta)
        diagnostic.update({"order": target["order"], "component_means": target["component_means"],
                           "ordering_gap": target["ordering_gap"]})
        if target["ordering_gap"] == 0:
            diagnostic["information_reasons"].append("component_mean_ordering_tie")
        for j, name in enumerate(TARGETS):
            diagnostic["targets"][name] = {"estimate": target["values"][j],
                                            "working_estimate": target["working_values"][j],
                                            "working_se": None, "variance_valid": False}
        diagnostic["bootstrap_estimate_eligible"] = not diagnostic["fit_reasons"]
    except Exception as exc:
        diagnostic["fit_reasons"].append("invalid_target")
        diagnostic["target_exception"] = repr(exc)
        return diagnostic
    try:
        beginning = time.perf_counter()
        loglik, score, info = information(theta, y, 2, False)
        symmetric = (info + info.T) / 2
        eigenvalues = np.linalg.eigvalsh(symmetric)
        condition = eigenvalues[-1] / eigenvalues[0] if eigenvalues[0] > 0 else np.inf
        direct_score = ld.evaluation(theta, y, 2, False, True)[2]
        diagnostic["information"] = {"seconds": time.perf_counter() - beginning,
            "minimum_eigenvalue": eigenvalues[0], "maximum_eigenvalue": eigenvalues[-1],
            "condition": condition, "symmetry_error": np.abs(info - info.T).max(),
            "direct_score_relative_error": np.max(np.abs(score - direct_score) / (1 + np.abs(direct_score))),
            "log_likelihood": loglik}
        if not np.isfinite(eigenvalues).all() or eigenvalues[0] <= 0:
            diagnostic["information_reasons"].append("information_not_positive_definite")
        elif condition >= config["intervals"]["maximum_information_condition"]:
            diagnostic["information_reasons"].append("information_ill_conditioned")
        else:
            variances = np.einsum("ij,ji->i", target["working_gradients"],
                                  np.linalg.solve(symmetric, target["working_gradients"].T))
            for j, name in enumerate(TARGETS):
                if np.isfinite(variances[j]) and variances[j] > 0:
                    diagnostic["targets"][name].update({"working_se": np.sqrt(variances[j]),
                                                        "variance_valid": True})
        diagnostic["regular_information_eligible"] = not (diagnostic["fit_reasons"]
                                                           or diagnostic["information_reasons"])
    except Exception as exc:
        diagnostic["information_reasons"].append("information_exception")
        diagnostic["information_exception"] = repr(exc)
    return diagnostic


def build_intervals(outer, bootstrap, truths, config):
    intervals = config["intervals"]
    total = intervals["bootstrap_draws"]
    required = max(intervals["minimum_valid_draws"], int(np.ceil(total * intervals["minimum_valid_fraction"])))
    probability = (1 - intervals["confidence"]) / 2
    rows = []
    for target_index, name in enumerate(TARGETS):
        point = outer.get("targets", {}).get(name)
        ordinary, studentized = [], []
        for draw in bootstrap:
            diagnostic = draw["diagnostic"]
            value = diagnostic.get("targets", {}).get(name)
            if not value or not diagnostic["bootstrap_estimate_eligible"]:
                continue
            ordinary.append(value["working_estimate"])
            if (point and diagnostic["regular_information_eligible"] and value["variance_valid"]
                    and value["working_se"] is not None and value["working_se"] > 0):
                studentized.append((value["working_estimate"] - point["working_estimate"]) / value["working_se"])
        for method in METHODS:
            row = {"method": method, "target": name, "truth": truths[name],
                   "estimate": point["estimate"] if point else None, "available": False,
                   "exclusion_reasons": [], "bootstrap_requested": total,
                   "bootstrap_attempted": len(bootstrap), "bootstrap_valid": None,
                   "bootstrap_required": required, "full_draw_availability": False,
                   "lower": None, "upper": None, "length": None, "covered": None}
            if point is None or not outer["bootstrap_estimate_eligible"]:
                row["exclusion_reasons"] = outer["fit_reasons"] or ["outer_target_unavailable"]
            elif method == "wald":
                if not outer["regular_information_eligible"] or not point["variance_valid"]:
                    row["exclusion_reasons"] = outer["information_reasons"] or ["invalid_target_variance"]
                else:
                    z = norm.ppf(1 - probability)
                    endpoints = point["working_estimate"] + np.array([-1, 1]) * z * point["working_se"]
            else:
                values = studentized if method == "bootstrap_t" else ordinary
                row["bootstrap_valid"] = len(values)
                row["full_draw_availability"] = len(values) == total
                if method == "bootstrap_t" and (not outer["regular_information_eligible"] or not point["variance_valid"]):
                    row["exclusion_reasons"] = outer["information_reasons"] or ["invalid_target_variance"]
                elif len(bootstrap) != total:
                    row["exclusion_reasons"] = ["bootstrap_incomplete"]
                elif len(values) < required:
                    row["exclusion_reasons"] = ["too_few_valid_bootstrap_draws"]
                else:
                    quantiles = np.quantile(values, [probability, 1 - probability], method="linear")
                    row["bootstrap_working_quantiles"] = quantiles
                    if method == "bootstrap_percentile":
                        endpoints = quantiles
                    elif method == "bootstrap_basic":
                        endpoints = 2 * point["working_estimate"] - quantiles[::-1]
                    else:
                        endpoints = point["working_estimate"] - quantiles[::-1] * point["working_se"]
            if not row["exclusion_reasons"]:
                with np.errstate(over="ignore", invalid="ignore"):
                    lower, upper = inverse_target(target_index, endpoints)
                if np.isfinite([lower, upper]).all() and lower <= upper:
                    row.update({"available": True, "lower": float(lower), "upper": float(upper),
                                "length": float(upper - lower), "covered": bool(lower <= truths[name] <= upper)})
                else:
                    row["exclusion_reasons"] = ["nonfinite_or_reversed_interval"]
            rows.append(row)
    return rows


def runtime_versions():
    return {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "scikit_learn": sklearn.__version__}


def initialize_manifest(directory, config, config_path, task):
    files = [Path(__file__).resolve(), Path(ld.__file__).resolve(), Path(info_module.__file__).resolve()]
    sources = {path.name: path.read_bytes() for path in files}
    config_bytes = Path(config_path).read_bytes()
    manifest = {"schema_version": 1, "task": task, "config": config,
                "config_sha256": sha256(json_bytes(config)), "config_file_sha256": sha256(config_bytes),
                "source_sha256": {name: sha256(data) for name, data in sources.items()},
                "versions": runtime_versions(), "seed_protocol": "SeedSequence(master_seed, spawn_key=(task,stage,bootstrap_index)); stages 0=data,1=outer_fit,2=bootstrap_data,3=bootstrap_fit",
                "platform": platform.platform(), "cpu_threads": 1,
                "scheduler": {}}
    destination = directory / "manifest.json"
    if destination.exists():
        previous = json.loads(destination.read_text())
        for key in ("task", "config_sha256", "config_file_sha256", "source_sha256", "versions"):
            if previous[key] != serializable(manifest[key]):
                raise RuntimeError(f"Cannot resume: manifest mismatch in {key}")
        for name, expected in previous["source_sha256"].items():
            if sha256((directory / "sources" / name).read_bytes()) != expected:
                raise RuntimeError(f"Saved source snapshot changed: {name}")
        if sha256((directory / "sources" / "configuration.json").read_bytes()) != previous["config_file_sha256"]:
            raise RuntimeError("Saved configuration snapshot changed")
        return previous
    for name, data in sources.items():
        atomic_bytes(directory / "sources" / name, data)
    atomic_bytes(directory / "sources" / "configuration.json", config_bytes)
    atomic_json(destination, manifest)
    return manifest


def read_checkpoint(path):
    """Recover complete records; truncate only a partial final JSONL line."""
    path = Path(path)
    if not path.exists():
        return []
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    records, valid_bytes = [], 0
    for index, line in enumerate(lines):
        if not line.endswith(b"\n"):
            if index != len(lines) - 1:
                raise RuntimeError("Incomplete nonfinal checkpoint line")
            with path.open("r+b") as handle:
                handle.truncate(valid_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            break
        record = json.loads(line)
        if record["bootstrap_index"] != len(records):
            raise RuntimeError("Checkpoint bootstrap indices are not consecutive")
        records.append(record)
        valid_bytes += len(line)
    return records


def append_checkpoint(path, record):
    with Path(path).open("ab") as handle:
        handle.write(json_bytes(record))
        handle.flush()
        os.fsync(handle.fileno())


@contextlib.contextmanager
def task_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".task.lock").open("a+b") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("This task already has an active writer") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def run_task(config, config_path, task_index, output, stop_after_bootstrap=None):
    validate_config(config)
    task = task_specification(config, task_index)
    directory = Path(output) / f"task_{task_index:06d}"
    with task_lock(directory), threadpool_limits(limits=1):
        manifest = initialize_manifest(directory, config, config_path, task)
        result_path = directory / "result.json"
        if result_path.exists():
            return json.loads(result_path.read_text())
        source = generating_parameters(config, task)
        alpha, locations, scales, weights, true_means, truths = source
        data_seed = seed_sequence(config, task_index, 0)
        y, _ = ld.simulate(task["n"], alpha, locations, scales, weights, np.random.default_rng(data_seed))
        data_buffer = io.BytesIO()
        np.savez_compressed(data_buffer, y=y)
        if not (directory / "data.npz").exists():
            atomic_bytes(directory / "data.npz", data_buffer.getvalue())
        else:
            with np.load(directory / "data.npz") as saved:
                if not np.array_equal(saved["y"], y):
                    raise RuntimeError("Regenerated data differ from the saved training sample")
        outer_path = directory / "outer_fit.json"
        if outer_path.exists():
            outer = json.loads(outer_path.read_text())
        else:
            fit_seed = seed_sequence(config, task_index, 1)
            fitted = fit_model(y, int(fit_seed.generate_state(1)[0]), config)
            diagnostic = diagnose_fit(fitted, y, config)
            outer = {"data_seed": seed_record(data_seed), "fit_seed": seed_record(fit_seed),
                     "fit": fitted, "diagnostic": diagnostic, "truths": truths}
            if diagnostic.get("component_means") is not None:
                estimated = np.asarray(diagnostic["component_means"])
                estimated_index, true_index = linear_sum_assignment(
                    ((estimated[:, None] - true_means[None]) ** 2).sum(axis=2))
                alignment = np.empty(2, dtype=int)
                alignment[true_index] = estimated_index
                outer["truth_matching_diagnostic_only"] = alignment.tolist()
                outer["canonical_order_matches_truth_matching"] = alignment.tolist() == diagnostic["order"]
            atomic_json(outer_path, outer)
            outer = serializable(outer)
        checkpoint = directory / "bootstrap_checkpoint.jsonl"
        bootstrap = read_checkpoint(checkpoint)
        total = config["intervals"]["bootstrap_draws"]
        if len(bootstrap) > total:
            raise RuntimeError("Checkpoint contains more draws than the specification")
        if outer["diagnostic"]["bootstrap_estimate_eligible"]:
            theta = np.asarray(outer["fit"]["theta"])
            logpi, fitted_alpha, a, b = ld.unpack(theta, 2, 2, False)
            for index in range(len(bootstrap), total):
                if stop_after_bootstrap is not None and index >= stop_after_bootstrap:
                    status = {"status": "checkpointed", "task": task, "bootstrap_completed": len(bootstrap),
                              "bootstrap_requested": total}
                    atomic_json(directory / "status.json", status)
                    return status
                data_stream = seed_sequence(config, task_index, 2, index)
                fit_stream = seed_sequence(config, task_index, 3, index)
                sample, _ = ld.simulate(task["n"], fitted_alpha, -b / a, 1 / a, np.exp(logpi),
                                       np.random.default_rng(data_stream))
                fitted = fit_model(sample, int(fit_stream.generate_state(1)[0]), config, warm_start=theta)
                diagnostic = diagnose_fit(fitted, sample, config)
                record = serializable({"bootstrap_index": index, "data_seed": seed_record(data_stream),
                                       "fit_seed": seed_record(fit_stream), "fit": fitted, "diagnostic": diagnostic})
                append_checkpoint(checkpoint, record)
                bootstrap.append(record)
                if (index + 1) % 10 == 0 or index + 1 == total:
                    print(json.dumps({"task_index": task_index, "bootstrap_completed": index + 1,
                                      "bootstrap_requested": total}), flush=True)
        rows = build_intervals(outer["diagnostic"], bootstrap, truths, config)
        compressed = gzip.compress(b"".join(json_bytes(record) for record in bootstrap), mtime=0)
        atomic_bytes(directory / "bootstrap_records.jsonl.gz", compressed)
        result = {"status": "complete", "task": task, "config_sha256": manifest["config_sha256"],
                  "source_sha256": manifest["source_sha256"], "truths": truths,
                  "outer_fit": outer["fit"]["record"], "outer_exception": outer["fit"]["exception"],
                  "outer_diagnostic": outer["diagnostic"], "intervals": rows,
                  "bootstrap_requested": total, "bootstrap_attempted": len(bootstrap),
                  "bootstrap_skipped_reason": "" if len(bootstrap) == total else "outer_estimator_ineligible",
                  "bootstrap_fit_failures": sum(not item["fit"]["finite_fit"] for item in bootstrap),
                  "bootstrap_nonstationary": sum("nonstationary" in item["diagnostic"]["fit_reasons"] for item in bootstrap),
                  "bootstrap_bound_hits": sum(bool(item["fit"]["record"].get("boundary_hits")) for item in bootstrap),
                  "bootstrap_regular_information": sum(item["diagnostic"]["regular_information_eligible"] for item in bootstrap),
                  "outer_fit_seconds": outer["fit"]["elapsed_seconds"],
                  "bootstrap_fit_seconds": sum(item["fit"]["elapsed_seconds"] for item in bootstrap),
                  "timing_note": "Recorded completed-fit cost; an interrupted incomplete fit is rerun and its lost cost is not reconstructed"}
        atomic_json(result_path, result)
        atomic_json(directory / "status.json", {"status": "complete", "task": task,
                                                "bootstrap_completed": len(bootstrap)})
        if checkpoint.exists():
            checkpoint.unlink()
        return serializable(result)


def summarize_results(output, config):
    """Summarize all completed tasks, with explicit planned/completed counts."""
    expected_hash = sha256(json_bytes(config))
    results = []
    common_sources = None
    common_versions = None
    seen_indices = set()
    for path in sorted(Path(output).glob("task_*/result.json")):
        result = json.loads(path.read_text())
        recorded_task = result.get("task", {})
        index = recorded_task.get("task_index")
        if type(index) is not int:
            raise RuntimeError(f"Invalid task index in {path}")
        try:
            expected_task = task_specification(config, index)
        except ValueError as exc:
            raise RuntimeError(f"Out-of-range task index in {path}") from exc
        if path.parent.name != f"task_{index:06d}":
            raise RuntimeError(f"Noncanonical task directory: {path.parent}")
        if index in seen_indices:
            raise RuntimeError(f"Duplicate completed task index: {index}")
        if recorded_task != expected_task or result.get("status") != "complete":
            raise RuntimeError(f"Completed-task metadata mismatch in {path}")
        seen_indices.add(index)
        if result["config_sha256"] != expected_hash:
            raise RuntimeError(f"Configuration mismatch in {path}")
        manifest = json.loads((path.parent / "manifest.json").read_text())
        if manifest["task"] != expected_task:
            raise RuntimeError(f"Manifest task metadata mismatch in {path}")
        if (manifest["config_sha256"] != expected_hash
                or sha256(json_bytes(manifest["config"])) != expected_hash):
            raise RuntimeError(f"Manifest configuration mismatch in {path}")
        if result["source_sha256"] != manifest["source_sha256"]:
            raise RuntimeError(f"Result/manifest source mismatch in {path}")
        for name, expected in manifest["source_sha256"].items():
            if sha256((path.parent / "sources" / name).read_bytes()) != expected:
                raise RuntimeError(f"Saved source snapshot changed in {path.parent}: {name}")
        saved_config = (path.parent / "sources" / "configuration.json").read_bytes()
        if (sha256(saved_config) != manifest["config_file_sha256"]
                or sha256(json_bytes(json.loads(saved_config))) != expected_hash):
            raise RuntimeError(f"Saved configuration snapshot changed in {path.parent}")
        if common_sources is None:
            common_sources, common_versions = result["source_sha256"], manifest["versions"]
        elif result["source_sha256"] != common_sources or manifest["versions"] != common_versions:
            raise RuntimeError("Cannot pool tasks from different source or runtime versions")
        results.append(result)
    grouped, pairs = {}, {}
    for result in results:
        task = result["task"]
        index = {(row["method"], row["target"]): row for row in result["intervals"]}
        for row in result["intervals"]:
            key = (task["scenario"], task["n"], row["target"], row["method"])
            grouped.setdefault(key, []).append(row)
        for target in TARGETS:
            baseline = index[("wald", target)]
            for method in METHODS[1:]:
                key = (task["scenario"], task["n"], target, method)
                pairs.setdefault(key, []).append((baseline, index[(method, target)]))
    summary = []
    for key, rows in sorted(grouped.items()):
        scenario, n, target, method = key
        available = [row for row in rows if row["available"]]
        full_draw_rows = [row for row in available if row["full_draw_availability"]]
        count, covered = len(available), sum(row["covered"] for row in available)
        interval = binomtest(covered, count).proportion_ci(method="exact") if count else None
        coverage = covered / count if count else None
        summary.append({"scenario": scenario, "n": n, "target": target, "method": method,
                        "planned": config["replications_per_cell"], "attempted_completed": len(rows),
                        "available": count, "unavailable": len(rows) - count,
                        "availability_fraction": count / len(rows), "covered_count": covered,
                        "conditional_coverage": coverage, "covered_and_available_fraction": covered / len(rows),
                        "conditional_coverage_mcse": np.sqrt(coverage * (1 - coverage) / count) if count else None,
                        "conditional_coverage_mc95": [interval.low, interval.high] if interval else None,
                        "mean_length_conditional": np.mean([row["length"] for row in available]) if count else None,
                        "full_draw_available": len(full_draw_rows) if method != "wald" else None,
                        "full_draw_conditional_coverage": np.mean([row["covered"] for row in full_draw_rows])
                        if full_draw_rows and method != "wald" else None,
                        "full_draw_covered_and_available_fraction": sum(row["covered"] for row in full_draw_rows) / len(rows)
                        if method != "wald" else None})
    paired = []
    for key, values in sorted(pairs.items()):
        common = [(left, right) for left, right in values if left["available"] and right["available"]]
        difference = np.array([int(right["covered"]) - int(left["covered"]) for left, right in common])
        paired.append({"scenario": key[0], "n": key[1], "target": key[2], "method": key[3],
                       "attempted_completed": len(values), "both_available": len(common),
                       "coverage_difference_on_common_available": difference.mean() if len(common) else None,
                       "paired_mcse": difference.std(ddof=1) / np.sqrt(len(common)) if len(common) > 1 else None,
                       "mean_length_difference_on_common_available": np.mean([right["length"] - left["length"]
                                                                               for left, right in common]) if common else None})
    diagnostic_groups = {}
    for result in results:
        key = (result["task"]["scenario"], result["task"]["n"])
        diagnostic_groups.setdefault(key, []).append(result)
    diagnostics = []
    for (scenario, n), values in sorted(diagnostic_groups.items()):
        diagnostics.append({"scenario": scenario, "n": n, "completed_outer_tasks": len(values),
                            "outer_fit_exceptions": sum(bool(item["outer_exception"]) for item in values),
                            "outer_nonstationary": sum("nonstationary" in item["outer_diagnostic"]["fit_reasons"] for item in values),
                            "outer_bound_hits": sum(bool(item["outer_fit"].get("boundary_hits")) for item in values),
                            "bootstrap_attempted": sum(item["bootstrap_attempted"] for item in values),
                            "bootstrap_fit_failures": sum(item["bootstrap_fit_failures"] for item in values),
                            "bootstrap_nonstationary": sum(item["bootstrap_nonstationary"] for item in values),
                            "bootstrap_bound_hits": sum(item["bootstrap_bound_hits"] for item in values),
                            "bootstrap_regular_information": sum(item["bootstrap_regular_information"] for item in values),
                            "recorded_outer_fit_seconds": sum(item["outer_fit_seconds"] for item in values),
                            "recorded_bootstrap_fit_seconds": sum(item["bootstrap_fit_seconds"] for item in values)})
    total = len(config["scenarios"]) * len(config["sample_sizes"]) * config["replications_per_cell"]
    result = {"config_sha256": expected_hash, "source_sha256": common_sources,
              "versions": common_versions, "expected_tasks": total, "completed_tasks": len(results),
              "missing_task_indices": sorted(set(range(total)) - {result["task"]["task_index"] for result in results}),
              "coverage": summary, "paired": paired, "fit_diagnostics": diagnostics}
    atomic_json(Path(output) / "aggregate.json", result)
    for name, rows in (("coverage_summary", summary), ("paired_interval_comparison", paired),
                       ("fit_diagnostics", diagnostics)):
        if rows:
            stream = io.StringIO()
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(serializable(rows))
            atomic_bytes(Path(output) / (name + ".csv"), stream.getvalue().encode())
    return serializable(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--task-range", help="Half-open START:STOP range, processed sequentially in one worker")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--stop-after-bootstrap", type=int,
                        help="Checkpoint after this many draws; does not change the configured study")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate_config(config)
    if args.aggregate:
        result = summarize_results(args.output, config)
        print(json.dumps({"completed_tasks": result["completed_tasks"], "expected_tasks": result["expected_tasks"]}))
    else:
        if (args.task_index is None) == (args.task_range is None):
            parser.error("Supply exactly one of --task-index or --task-range")
        if args.task_range is None:
            indices = [args.task_index]
        else:
            try:
                start, stop = map(int, args.task_range.split(":"))
                if start < 0 or stop <= start:
                    raise ValueError
                task_specification(config, stop - 1)
            except ValueError:
                parser.error("--task-range must be START:STOP within the configured half-open task range")
            indices = range(start, stop)
        for index in indices:
            result = run_task(config, args.config, index, args.output, args.stop_after_bootstrap)
            print(json.dumps({"task_index": index, "status": result["status"],
                              "bootstrap_attempted": result.get("bootstrap_attempted", result.get("bootstrap_completed"))}),
                  flush=True)


if __name__ == "__main__":
    main()
