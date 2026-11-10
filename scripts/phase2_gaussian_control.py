"""Prespecified Gaussian shared-correlation control on frozen primary tasks.

Freeze: --freeze --source-kind simulations --primary-run MAIN --output CONTROL
Workers: --task-index I or --task-range START:STOP --output CONTROL
Report: --aggregate --output CONTROL

Application companions select all primary-variant CV/full/bootstrap tasks.
No primary source or result is modified. Upstream tasks must be complete;
otherwise the worker reports dependency_pending and can be reissued unchanged.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import tempfile
import time
import traceback

import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

import run_confirmatory_experiments as baseline

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/"configs/phase2_gaussian_control.json"
METHOD = "gaussian_shared_correlation"
PACKAGES = ["numpy", "scipy", "pandas", "scikit-learn", "fmvmm", "threadpoolctl"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def semantic_hash(value):
    return hashlib.sha256(json.dumps(baseline.serializable(value), sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="."+path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(baseline.serializable(value), handle, indent=2, allow_nan=False)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


@contextmanager
def task_lock(path):
    path.mkdir(parents=True, exist_ok=True)
    with (path/"task.lock").open("a+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def versions():
    return {"python": platform.python_version(),
            **{name: importlib.metadata.version(name) for name in PACKAGES}}


def source_files():
    return [Path(__file__), ROOT/"scripts/gaussian_shared_correlation.py",
            ROOT/"scripts/run_confirmatory_experiments.py", ROOT/"scripts/shared_shape_ld.py",
            ROOT/"configs/experiments.json", ROOT/"requirements-lock.txt"]


def primary_metadata(primary, kind):
    filename = "manifest.json" if kind == "simulations" else "specification.json"
    path = Path(primary)/filename
    return path, json.loads(path.read_text())


def task_map(primary_spec, kind):
    answer = []
    if kind == "simulations":
        config = primary_spec["configuration"]
        index = 0
        for cell in config["cells"]:
            for replicate in range(cell["repetitions"]):
                answer.append({"task_index": index, "primary_task_index": index,
                               "cell_id": cell["id"], "replicate": replicate})
                index += 1
        if len(answer) != config["dataset_count"]:
            raise ValueError("Primary task count does not match its configuration")
    else:
        primary_variants = {v["id"] for v in primary_spec["config"]["variants"] if v.get("primary")}
        for index, task in enumerate(primary_spec["tasks"]):
            if task["task_index"] != index:
                raise ValueError("Noncanonical primary application task map")
            if (task["variant"] in primary_variants and task["bounds"] == "primary"
                    and task["kind"] in ["cv", "full", "bootstrap"]):
                answer.append({"task_index": len(answer), "primary_task_index": index,
                               "variant": task["variant"], "kind": task["kind"],
                               "repeat": task["repeat"], "fold": task["fold"]})
    return answer


def validate_control_config(config):
    module = importlib.import_module("gaussian_shared_correlation")
    if config["schema_version"] != 1 or config["method"] != METHOD or config["threads"] != 1:
        raise ValueError("Unsupported control specification")
    if tuple(config["bounds"]["means"]) != tuple(module.MEAN_BOUNDS):
        raise ValueError("Declared mean bounds differ from fitter")
    if tuple(config["bounds"]["standard_deviations"]) != tuple(module.SD_BOUNDS):
        raise ValueError("Declared SD bounds differ from fitter")
    if config["bounds"]["correlation_residual_epsilon"] != module.CORRELATION_EPS:
        raise ValueError("Declared correlation bounds differ from fitter")


def freeze(output, primary, kind, config_path=CONFIG):
    config = json.loads(Path(config_path).read_text())
    validate_control_config(config)
    path, specification = primary_metadata(primary, kind)
    primary_versions = ({"python": specification["python"], **specification["versions"]}
                        if kind == "simulations" else specification["runtime_versions"])
    if primary_versions != versions():
        raise RuntimeError("Freeze the companion in the same runtime as its primary run")
    mapping = task_map(specification, kind)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    manifest = {"config": config, "config_sha256": semantic_hash(config), "source_kind": kind,
                "primary_run": str(Path(primary).resolve()), "primary_metadata_name": path.name,
                "primary_metadata_sha256": digest(path), "tasks": mapping,
                "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in source_files()},
                "versions": versions(), "threads": 1,
                "interpretation": "Prespecified separate phase-2 structural control, with no outcome-based task selection"}
    atomic_json(output/"manifest.json", manifest)
    (output/"primary_metadata.json").write_bytes(path.read_bytes())
    for source in source_files():
        target = output/"source_snapshot"/source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(source.read_bytes())
    atomic_json(output/"source_snapshot/configs/phase2_gaussian_control.json", config)
    return manifest


def verify_run(output, primary_override=None):
    output = Path(output); manifest = json.loads((output/"manifest.json").read_text())
    if semantic_hash(manifest["config"]) != manifest["config_sha256"]:
        raise RuntimeError("Control configuration hash mismatch")
    archived_config = json.loads((output/"source_snapshot/configs/phase2_gaussian_control.json").read_text())
    if semantic_hash(archived_config) != manifest["config_sha256"]:
        raise RuntimeError("Archived control configuration hash mismatch")
    if versions() != manifest["versions"]:
        raise RuntimeError("Runtime differs from frozen control run")
    for relative, expected in manifest["source_sha256"].items():
        if digest(ROOT/relative) != expected or digest(output/"source_snapshot"/relative) != expected:
            raise RuntimeError("Frozen control source changed: "+relative)
    primary = Path(primary_override) if primary_override else Path(manifest["primary_run"])
    path, spec = primary_metadata(primary, manifest["source_kind"])
    if (digest(path) != manifest["primary_metadata_sha256"]
            or digest(output/"primary_metadata.json") != manifest["primary_metadata_sha256"]
            or task_map(spec, manifest["source_kind"]) != manifest["tasks"]):
        raise RuntimeError("Primary specification or control task map changed")
    return manifest, primary, spec


class DependencyPending(RuntimeError):
    pass


class IntegrityError(RuntimeError):
    pass


def gaussian_density(params, y, keep=None):
    """Stable Gaussian mixture density, using the shared correlation directly."""
    y = np.asarray(y)
    indices = np.arange(y.shape[1]) if keep is None else np.asarray(keep, dtype=int)
    if not len(indices):
        weights = np.asarray(params["weights"])
        return np.zeros(len(y)), np.broadcast_to(weights, (len(y), len(weights))).copy()
    means = np.asarray(params["means"])[:, indices]
    std = np.asarray(params["std"])[:, indices]
    correlation = np.asarray(params["correlation"])[np.ix_(indices, indices)]
    lower = np.linalg.cholesky(correlation)
    logdet = np.log(np.diag(lower)).sum()
    terms = []
    for weight, mean, scale in zip(params["weights"], means, std):
        residual = (y[:, indices]-mean)/scale
        whitened = solve_triangular(lower, residual.T, lower=True, check_finite=False).T
        terms.append(np.log(weight)-np.log(scale).sum()-logdet-.5*len(indices)*np.log(2*np.pi)
                     -.5*np.einsum("ij,ij->i", whitened, whitened))
    terms = np.column_stack(terms); density = logsumexp(terms, axis=1)
    return density, np.exp(terms-density[:, None])


def application_data(primary, spec, variant):
    filename = "oliveoil.csv" if variant["dataset"] == "oliveoil" else "crabs.csv"
    relative = "data/"+filename
    path = primary/"executed_sources"/relative
    if digest(path) != spec["data_hashes"][relative]:
        raise RuntimeError("Frozen application data changed")
    frame = pd.read_csv(path)
    raw = frame[variant["columns"]].to_numpy(float)+variant.get("add_to_all", 0.)
    if np.any(raw <= 0):
        raise ValueError("Nonpositive scientific measurement")
    if variant["dataset"] == "oliveoil":
        reference = variant["columns"].index(variant["reference"])
        keep = [j for j in range(raw.shape[1]) if j != reference]
        y = np.log(raw[:, keep]/raw[:, reference, None])
        names = [variant["columns"][j]+"/"+variant["reference"] for j in keep]
        return y, frame["macro.area"].to_numpy(), np.zeros_like(y), names, path
    is_log = variant["transform"] == "log"
    labels = (frame["sp"].astype(str)+"_"+frame["sex"].astype(str)).to_numpy()
    return (np.log(raw) if is_log else raw), labels, (np.log(raw) if is_log else np.zeros_like(raw)), variant["columns"], path


def load_case(primary, spec, task, kind):
    """Read a completed frozen upstream task; fitting receives no truth fields."""
    index = task["primary_task_index"]
    if kind == "simulations":
        candidates = sorted((primary/"tasks"/f"task_{index:06d}").glob("attempt_*"))
        if not candidates or not (candidates[-1]/"completion.json").exists():
            raise DependencyPending("Latest primary simulation attempt is not complete")
        directory = candidates[-1]
        completed = json.loads((directory/"completion.json").read_text())
        identity = json.loads((directory/"task.json").read_text())
        if (completed["task_index"] != index or completed["cell_id"] != task["cell_id"]
                or completed["replicate"] != task["replicate"]
                or identity["configuration_sha256"] != spec["configuration_sha256"]):
            raise RuntimeError("Primary simulation identity mismatch")
        used = [directory/"completion.json", directory/"task.json"]
        if not (directory/"data.npz").exists():
            failure = directory/"setup_failure.json"
            if failure.exists():
                used.append(failure)
            return {"upstream_data_failure": True, "used_paths": used, "directory": directory}
        data = np.load(directory/"data.npz")
        generation_path = directory/"generation.json"
        generation = json.loads(generation_path.read_text())["parameters"]
        used += [directory/"data.npz", generation_path]
        baseline_records = {}; diagonal = None
        for path in sorted(directory.glob("primary__*.json")):
            saved = json.loads(path.read_text()); row = saved["row"]
            if (row["task_index"] != index or row["variant"] != "primary"
                    or row["cell_id"] != task["cell_id"] or row["replicate"] != task["replicate"]):
                raise RuntimeError("Primary fit record identity mismatch")
            baseline_records[row["method"]] = row
            if row["method"] == "gaussian_diag":
                diagonal = saved
            used.append(path)
        return {"train": data["train"], "test": data["test"], "labels": data["test_labels"],
                "center": data["center"], "scale": data["scale"],
                "partitions": data["partitions"][:spec["configuration"]["fitting"]["n_starts"]],
                "seed": identity["seeds"]["initialization"][0], "k": identity["cell"]["components"],
                "settings": spec["configuration"]["fitting"], "true_means": np.asarray(generation["means"]),
                "true_weights": np.asarray(generation["weights"]), "oracle": data["oracle_log_density"],
                "raw_jacobian": np.zeros_like(data["test"]),
                "train_raw_jacobian": np.zeros_like(data["train"]), "diagonal": diagonal,
                "baseline_records": baseline_records, "used_paths": used, "directory": directory,
                "evaluation_role": "independent_test", "conditional_target": None}
    upstream = spec["tasks"][index]
    directory = primary/"tasks"/f"{index:05d}"
    if not (directory/"completion.json").exists():
        raise DependencyPending("Primary application task is not complete")
    completed = json.loads((directory/"completion.json").read_text())
    split = json.loads((directory/"split.json").read_text())
    if (completed["task"] != upstream or any(split.get(key) != value for key, value in upstream.items())
            or completed["specification_sha256"] != digest(primary/"specification.json")):
        raise RuntimeError("Primary application identity mismatch")
    variant = next(v for v in spec["config"]["variants"] if v["id"] == task["variant"])
    y, labels, jacobian, names, data_path = application_data(primary, spec, variant)
    train, test = np.asarray(upstream["train"]), np.asarray(upstream["test"])
    diagonal_path = directory/"gaussian_diag.json"
    diagonal = json.loads(diagonal_path.read_text()) if diagonal_path.exists() else None
    used = [directory/"completion.json", directory/"split.json", data_path]
    if diagonal_path.exists():
        used.append(diagonal_path)
    records = {row["method"]: row for row in completed["records"]}
    return {"train": y[train], "test": y[test], "labels": labels[test],
            "center": np.asarray(split["center"]), "scale": np.asarray(split["scale"]),
            "partitions": [np.asarray(p) for p in split["partitions"]], "seed": upstream["seed"],
            "k": variant["k"], "settings": {**baseline.DEFAULT_SETTINGS, **spec["config"]["fitting"]},
            "true_means": None, "true_weights": None, "oracle": None,
            "raw_jacobian": jacobian[test], "train_raw_jacobian": jacobian[train],
            "diagonal": diagonal, "baseline_records": records,
            "used_paths": used, "directory": directory, "variant_spec": variant,
            "names": names, "target_index": names.index(variant["conditional_target"]),
            "conditional_target": variant["conditional_target"],
            "evaluation_role": "held_out_CV" if task["kind"] == "cv" else "descriptive_original_sample",
            "original_data_sd": y.std(axis=0), "test_indices": test}


def input_binding(case, primary, task):
    return {"primary_task_index": task["primary_task_index"],
            "primary_directory": str(case["directory"].relative_to(primary)),
            "input_sha256": {str(p.relative_to(primary)): digest(p) for p in case["used_paths"]}}


def validate_stage(value, identity, name):
    if any(value.get(key) != expected for key, expected in identity.items()):
        raise IntegrityError(name+" checkpoint has a different task, manifest or input binding")


def upstream_cost(external):
    row = external.get("row", external.get("record", {}))
    value = row.get("fit_seconds")
    if value is not None and np.isfinite(value) and value >= 0:
        return float(value), "upstream_fit_seconds"
    starts = external.get("starts", [])
    durations = [s.get("fit_seconds", s.get("elapsed_seconds")) for s in starts]
    if durations and all(value is not None and np.isfinite(value) and value >= 0 for value in durations):
        return float(sum(durations)), "sum_upstream_start_seconds_excludes_wrapper"
    value = row.get("total_seconds")
    if value is not None and np.isfinite(value) and value >= 0:
        return float(value), "upstream_total_seconds_including_evaluation"
    return None, "upstream_cost_unavailable"


def diagonal_dependency(case, directory, identity):
    path = directory/"diagonal_dependency.json"
    if path.exists():
        saved = json.loads(path.read_text())
        validate_stage(saved, identity, "Diagonal dependency")
        return saved
    external = case["diagonal"]
    if external and external.get("parameters") is not None:
        row = external.get("row", external.get("record", {}))
        cost, cost_source = upstream_cost(external)
        result = {"source": "primary_fit", "parameters": external["parameters"],
                  "fit_seconds": cost, "cost_source": cost_source, "starts": external.get("starts", []),
                  "record": row, "failed": False, "upstream_record_failed": row.get("failed"),
                  "upstream_evaluation_available": row.get("evaluation_available")}
    else:
        model = None; started = time.perf_counter()
        try:
            z = (case["train"]-case["center"])/case["scale"]
            model = baseline.fit_method("gaussian_diag", z, case["k"], case["seed"],
                                        settings=case["settings"], partitions=case["partitions"])
            result = {"source": "companion_fit", "parameters": model.params,
                      "fit_seconds": model.record["fit_seconds"], "starts": model.starts,
                      "record": model.record, "failed": False, "cost_source": "companion_fit_seconds"}
        except Exception as exc:
            result = {"source": "companion_fit", "parameters": None,
                      "fit_seconds": time.perf_counter()-started, "starts": getattr(exc, "starts", []),
                      "failed": True, "exception": repr(exc), "traceback": traceback.format_exc(),
                      "cost_source": "companion_failed_fit_elapsed_seconds"}
    result.update(identity)
    atomic_json(path, result)
    return baseline.serializable(result)


def evaluate_fit(params, case):
    z = (case["test"]-case["center"])/case["scale"]
    ell, tau = gaussian_density(params, z)
    ell = ell-np.log(case["scale"]).sum()-case["raw_jacobian"].sum(axis=1)
    if not np.isfinite(ell).all() or not np.isfinite(tau).all():
        raise FloatingPointError("Nonfinite control test predictions")
    means = case["center"]+case["scale"]*np.asarray(params["means"])
    metrics = {"test_log_density": float(ell.mean()), "ARI": float(adjusted_rand_score(case["labels"], tau.argmax(axis=1))),
               "mean_rmse": None, "weight_rmse": None, "conditional_log_density": None}
    conditional = None
    if case["true_means"] is not None:
        cost = ((means[:, None]-case["true_means"][None, :])**2).sum(axis=2)
        left, right = linear_sum_assignment(cost); order = left[np.argsort(right)]
        metrics["mean_rmse"] = float(np.sqrt(np.mean((means[order]-case["true_means"])**2)))
        metrics["weight_rmse"] = float(np.sqrt(np.mean((np.asarray(params["weights"])[order]-case["true_weights"])**2)))
        metrics["test_log_density_minus_oracle"] = float(np.mean(ell-case["oracle"]))
    if case["conditional_target"] is not None:
        target = case["target_index"]
        keep = [i for i in range(z.shape[1]) if i != target]
        marginal, _ = gaussian_density(params, z, keep)
        conditional = (gaussian_density(params, z)[0]-marginal-np.log(case["scale"][target])
                       -case["raw_jacobian"][:, target])
        if not np.isfinite(conditional).all():
            raise FloatingPointError("Nonfinite control conditional predictions")
        metrics["conditional_log_density"] = float(conditional.mean())
    return metrics, ell, tau, means, conditional


def run_task(output, index, primary_override=None, fit_function=None):
    output = Path(output); manifest, primary, spec = verify_run(output, primary_override)
    if not 0 <= index < len(manifest["tasks"]):
        raise ValueError("Control task index outside frozen map")
    task = manifest["tasks"][index]
    directory = output/"tasks"/f"task_{index:06d}"
    with task_lock(directory), threadpool_limits(1):
        try:
            case = load_case(primary, spec, task, manifest["source_kind"])
        except DependencyPending as exc:
            status = {**task, "status": "dependency_pending", "reason": str(exc)}
            atomic_json(directory/"status.json", status); return status
        binding = input_binding(case, primary, task)
        binding_path = directory/"input_binding.json"
        if binding_path.exists() and json.loads(binding_path.read_text()) != binding:
            raise RuntimeError("Primary task inputs changed; do not silently reuse a companion fit")
        atomic_json(binding_path, binding)
        stage_identity = {"task_index": index, "primary_task_index": task["primary_task_index"],
                          "manifest_sha256": digest(output/"manifest.json"),
                          "input_binding_sha256": semantic_hash(binding)}
        result_path = directory/"result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if (result["task"] != task or result["manifest_sha256"] != digest(output/"manifest.json")
                    or result["input_binding_sha256"] != semantic_hash(binding) or result["status"] != "complete"):
                raise RuntimeError("Completed control task identity mismatch")
            return result
        row = {**task, "method": METHOD, "source_kind": manifest["source_kind"], "failed": True,
               "fit_available": False, "evaluation_available": False, "test_log_density": None,
               "mean_rmse": None, "weight_rmse": None, "ARI": None, "conditional_log_density": None,
               "fit_seconds": 0., "dependency_fit_seconds": 0., "policy_fit_seconds": 0.,
               "parameter_count": None, "optimizer_success": None, "score_converged": None,
               "projected_gradient": None, "boundary_hit": None, "starts_attempted": 0,
               "starts_failed": 0, "exception": None}
        selected = None; parameters = None; profiles = None; dependency = None; start = time.perf_counter()
        try:
            if case.get("upstream_data_failure"):
                raise RuntimeError("Primary data generation failed; no replacement dataset is generated")
            row.update({"n_train": len(case["train"]), "n_test": len(case["test"]),
                        "d": case["train"].shape[1], "k": case["k"], "evaluation_role": case["evaluation_role"]})
            np.testing.assert_allclose(case["center"], case["train"].mean(axis=0), rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(case["scale"], case["train"].std(axis=0), rtol=1e-12, atol=1e-12)
            if np.any(case["scale"] <= 0):
                raise ValueError("Invalid saved training scale")
            dependency = diagonal_dependency(case, directory, stage_identity)
            row["dependency_fit_seconds"] = dependency["fit_seconds"]
            row["dependency_cost_source"] = dependency["cost_source"]
            row["diagonal_dependency_source"] = dependency["source"]
            fit_path = directory/"control_fit.json"
            if fit_path.exists():
                selected = json.loads(fit_path.read_text())
                validate_stage(selected, stage_identity, "Control fit")
            else:
                model = None; fit_start = time.perf_counter()
                try:
                    function = fit_function or importlib.import_module("gaussian_shared_correlation").fit
                    settings = case["settings"]
                    ztrain = (case["train"]-case["center"])/case["scale"]
                    model = function(ztrain, case["k"], case["seed"], partitions=case["partitions"],
                        diagonal_candidate=dependency["parameters"], n_starts=settings["n_starts"],
                        maxiter=settings["maxiter"], polish_maxiter=settings["refinement_maxiter"],
                        gtol=settings["gtol"], ftol=settings["ftol"],
                        start_timeout_seconds=settings["start_timeout_seconds"])
                    selected = {"failed": False, "parameters": model.params, "record": model.record,
                                "starts": model.starts, "selected_start": model.selected_start,
                                "fit_seconds": time.perf_counter()-fit_start}
                except Exception as exc:
                    selected = {"failed": True, "parameters": None, "record": {},
                                "starts": getattr(exc, "starts", []), "fit_seconds": time.perf_counter()-fit_start,
                                "exception": repr(exc), "traceback": traceback.format_exc()}
                selected.update(stage_identity)
                atomic_json(fit_path, selected)
                selected = baseline.serializable(selected)
            row.update({"fit_seconds": selected["fit_seconds"], "starts_attempted": len(selected["starts"]),
                        "starts_failed": sum(bool(s["failed"]) for s in selected["starts"])})
            row["policy_fit_seconds"] = (row["fit_seconds"]+row["dependency_fit_seconds"]
                                         if row["dependency_fit_seconds"] is not None else None)
            if selected["failed"]:
                raise RuntimeError(selected["exception"])
            parameters = selected["parameters"]; record = selected["record"]
            row["fit_available"] = True
            for field in ["parameter_count", "optimizer_success", "score_converged", "projected_gradient", "boundary_hit",
                          "correlation_loading_identified", "minimum_correlation_eigenvalue", "min_weight", "small_weight"]:
                row[field] = record.get(field)
            train_density = gaussian_density(parameters, (case["train"]-case["center"])/case["scale"])[0]
            row["density_identity_error"] = abs(float(train_density.mean())-record["mean_train_log_density"])
            if row["density_identity_error"] > 1e-8*(1+abs(record["mean_train_log_density"])):
                raise ArithmeticError("Independent training density disagrees with the control fitter")
            row["train_log_density"] = float(train_density.mean()-np.log(case["scale"]).sum()
                                               -case["train_raw_jacobian"].sum(axis=1).mean())
            metrics, density, tau, means, conditional = evaluate_fit(parameters, case)
            row.update(metrics); row.update({"failed": False, "evaluation_available": True})
            arrays = {"log_density": density, "responsibilities": tau, "component_means": means}
            if conditional is not None:
                arrays["conditional_log_density"] = conditional
            np.savez_compressed(directory/"predictions.npz", **arrays)
            profiles = {"means_in_analysis_coordinates": means,
                        "correlation": parameters["correlation"],
                        "marginal_sd_in_analysis_coordinates": np.asarray(parameters["std"])*case["scale"]}
            if manifest["source_kind"] == "applications":
                profiles.update({"coordinate_names": case["names"], "original_data_sd": case["original_data_sd"],
                    "analysis_transform": ("log_ratios" if case["variant_spec"]["dataset"] == "oliveoil" else
                                           "log_measurements" if case["variant_spec"].get("transform") == "log" else "raw_measurements"),
                    "geometric_means_or_ratios": np.exp(means) if case["variant_spec"].get("transform") != "raw" else None})
        except IntegrityError:
            raise
        except Exception as exc:
            row["failed"] = True
            row["evaluation_available"] = False
            row["exception"] = repr(exc); row["traceback"] = traceback.format_exc()
        row["total_seconds_this_invocation"] = time.perf_counter()-start
        result = {"status": "complete", "task": task, "row": row, "profiles": profiles,
                  "parameters": parameters, "baseline_records": case.get("baseline_records", {}),
                  "manifest_sha256": digest(output/"manifest.json"), "input_binding_sha256": semantic_hash(binding),
                  "environment": {"versions": versions(), "host": "not_recorded_in_public_rerun"}}
        atomic_json(result_path, result); atomic_json(directory/"status.json", {"status": "complete", **task})
        return baseline.serializable(result)


def aggregate(output, primary_override=None):
    output = Path(output); manifest, primary, spec = verify_run(output, primary_override)
    rows = []; results = {}; baselines = []
    for task in manifest["tasks"]:
        directory = output/"tasks"/f"task_{task['task_index']:06d}"
        path = directory/"result.json"
        if not path.exists():
            rows.append({**task, "recorded": False, "failed": True, "evaluation_available": False})
            continue
        result = json.loads(path.read_text())
        if (result["status"] != "complete" or result["task"] != task
                or result["manifest_sha256"] != digest(output/"manifest.json")):
            raise RuntimeError("Completed companion identity mismatch")
        binding = json.loads((directory/"input_binding.json").read_text())
        if semantic_hash(binding) != result["input_binding_sha256"]:
            raise RuntimeError("Recorded input binding changed")
        for relative, expected in binding["input_sha256"].items():
            if digest(primary/relative) != expected:
                raise RuntimeError("Paired primary input changed: "+relative)
        results[task["task_index"]] = result
        rows.append({**result["row"], "recorded": True})
        for name, record in result["baseline_records"].items():
            baselines.append({**task, "baseline_method": name, **record})
    frame = pd.DataFrame(rows)
    for field in ["test_log_density", "conditional_log_density", "mean_rmse", "weight_rmse", "fit_seconds", "policy_fit_seconds", "score_converged", "boundary_hit"]:
        if field not in frame:
            frame[field] = np.nan
    frame.to_csv(output/"fits.csv", index=False)
    base = pd.DataFrame(baselines)
    summary = []; paired = []
    if manifest["source_kind"] == "simulations":
        for name, group in frame.groupby("cell_id", sort=False):
            finite = group[group.evaluation_available]
            row = {"cell_id": name, "planned": len(group), "recorded": int(group.recorded.sum()),
                   "available": len(finite), "failed_recorded": int((group.recorded & group.failed).sum()),
                   "missing": int((~group.recorded).sum()),
                   "score_checked": int(group.score_converged.notna().sum()),
                   "score_passed": int((group.score_converged == True).sum()),
                   "boundary_checked": int(group.boundary_hit.notna().sum()),
                   "boundary_hits": int((group.boundary_hit == True).sum()),
                   "mean_fit_seconds": group.fit_seconds.mean(), "mean_policy_fit_seconds": group.policy_fit_seconds.mean()}
            for metric in ["test_log_density", "mean_rmse", "weight_rmse"]:
                values = pd.to_numeric(finite[metric], errors="coerce").dropna()
                row[metric+"_mean"] = values.mean()
                row[metric+"_MCSE"] = values.std(ddof=1)/np.sqrt(len(values)) if len(values) else np.nan
            summary.append(row)
            if len(base):
                for method, other in base[base.cell_id == name].groupby("baseline_method"):
                    common = finite.merge(other[~other.failed], on="primary_task_index", suffixes=("_control", "_baseline"))
                    record = {"cell_id": name, "baseline_method": method, "planned": len(group), "available_pairs": len(common)}
                    for metric in ["test_log_density", "mean_rmse", "weight_rmse"]:
                        delta = (pd.to_numeric(common[metric+"_baseline"], errors="coerce")-
                                 pd.to_numeric(common[metric+"_control"], errors="coerce")).dropna()
                        record[metric+"_baseline_minus_control"] = delta.mean()
                        record[metric+"_difference_MCSE"] = delta.std(ddof=1)/np.sqrt(len(delta)) if len(delta) else np.nan
                    paired.append(record)
    else:
        # Whole-repeat CV estimates, not artificial independence of overlapping folds.
        repeat_rows = []
        for (variant, kind), group in frame.groupby(["variant", "kind"], sort=False):
            summary_row = {"variant": variant, "kind": kind, "planned": len(group),
                            "recorded": int(group.recorded.sum()), "available": int(group.evaluation_available.sum()),
                            "failed_recorded": int((group.recorded & group.failed).sum()),
                            "missing": int((~group.recorded).sum()),
                            "score_checked": int(group.score_converged.notna().sum()),
                            "score_passed": int((group.score_converged == True).sum()),
                            "boundary_checked": int(group.boundary_hit.notna().sum()),
                            "boundary_hits": int((group.boundary_hit == True).sum())}
            if kind != "cv":
                summary.append(summary_row)
                continue
            complete_values = []
            for repeat, fold_rows in group.groupby("repeat"):
                if len(fold_rows) != spec["config"]["folds"] or not fold_rows.evaluation_available.all():
                    continue
                value = {"variant": variant, "repeat": repeat,
                    "control_joint": float(np.average(fold_rows.test_log_density, weights=fold_rows.n_test)),
                    "control_conditional": float(np.average(fold_rows.conditional_log_density, weights=fold_rows.n_test))}
                repeat_rows.append(value); complete_values.append(value)
            summary_row["complete_cv_repetitions"] = len(complete_values)
            for measure in ["joint", "conditional"]:
                values = np.array([value["control_"+measure] for value in complete_values])
                summary_row[measure+"_mean"] = values.mean() if len(values) else None
                summary_row[measure+"_repeat_min"] = values.min() if len(values) else None
                summary_row[measure+"_repeat_max"] = values.max() if len(values) else None
            summary.append(summary_row)
        repeats = pd.DataFrame(repeat_rows)
        repeats.to_csv(output/"complete_cv_repetitions.csv", index=False)
        if len(repeats) and len(base):
            base_cv = base[base.kind == "cv"]
            for (variant, method), group in base_cv.groupby(["variant", "baseline_method"]):
                values = []
                for repeat, folds in group.groupby("repeat"):
                    control = repeats[(repeats.variant == variant) & (repeats["repeat"] == repeat)]
                    if len(folds) != spec["config"]["folds"] or folds.failed.any() or len(control) != 1:
                        continue
                    values.append({"joint": float(np.average(folds.mean_log_density, weights=folds.n_test))-control.iloc[0].control_joint,
                                   "conditional": float(np.average(folds.mean_conditional_log_density, weights=folds.n_test))-control.iloc[0].control_conditional})
                for measure in ["joint", "conditional"]:
                    difference = np.array([v[measure] for v in values])
                    paired.append({"variant": variant, "baseline_method": method, "measure": measure,
                                   "complete_paired_repetitions": len(values),
                                   "mean_baseline_minus_control": float(difference.mean()) if len(values) else None,
                                   "difference_min": float(difference.min()) if len(values) else None,
                                   "difference_max": float(difference.max()) if len(values) else None,
                                   "uncertainty": "Descriptive range; overlapping repetitions are not independent"})
        bootstrap_stability(manifest, results, output)
    pd.DataFrame(summary).to_csv(output/"summary.csv", index=False)
    pd.DataFrame(paired).to_csv(output/"paired_comparisons.csv", index=False)
    status = {"planned": len(frame), "recorded": int(frame.recorded.sum()),
              "available": int(frame.evaluation_available.sum()),
              "failed_recorded": int((frame.recorded & frame.failed).sum()),
              "missing_task_indices": frame.loc[~frame.recorded, "task_index"].tolist(),
              "interpretation": "Prediction summaries condition on available finite fits; every missing/failed count remains explicit"}
    atomic_json(output/"aggregation.json", status)
    return status


def bootstrap_stability(manifest, results, output):
    rows = []; summaries = []; target_grid = {}
    variants = sorted({task["variant"] for task in manifest["tasks"]})
    for variant in variants:
        selected = [task for task in manifest["tasks"] if task["variant"] == variant]
        full = next(task for task in selected if task["kind"] == "full")
        reference_result = results.get(full["task_index"])
        if not reference_result or reference_result["row"]["failed"]:
            continue
        profile = reference_result["profiles"]
        reference = np.asarray(profile["means_in_analysis_coordinates"])
        reference = reference[np.argsort(reference[:, 0], kind="stable")]
        standard = np.asarray(profile["original_data_sd"])
        boot_tasks = [task for task in selected if task["kind"] == "bootstrap"]
        transform = profile["analysis_transform"]
        scale_label = {"log_ratios": "mean of natural log ratio",
                       "log_measurements": "mean of natural log length in millimetres",
                       "raw_measurements": "mean length in original millimetres"}[transform]
        for component in range(len(reference)):
            for coordinate in profile["coordinate_names"]:
                target_grid[(variant, component, coordinate)] = {"planned": len(boot_tasks),
                    "target": ("mean_log_" if transform.startswith("log_") else "mean_")+coordinate,
                    "target_scale": scale_label}
        for task in boot_tasks:
            result = results.get(task["task_index"])
            if not result or result["row"]["failed"]:
                continue
            means = np.asarray(result["profiles"]["means_in_analysis_coordinates"])
            costs = (((reference[:, None]-means[None, :])/standard)**2).sum(axis=2)
            left, right = linear_sum_assignment(costs); matched = means[right[np.argsort(left)]]
            for component, values in enumerate(matched):
                for coordinate, value in enumerate(values):
                    rows.append({"variant": variant, "replicate": task["repeat"], "component": component,
                                 "coordinate": profile["coordinate_names"][coordinate], "estimate": value,
                                 "matching_RMSE_standardized": float(np.sqrt(costs[left, right].mean()/means.shape[1])),
                                 "boundary_hit": result["row"]["boundary_hit"], "score_converged": result["row"]["score_converged"],
                                 "correlation_loading_identified": result["row"].get("correlation_loading_identified"),
                                 "planned_bootstrap": len(boot_tasks)})
    frame = pd.DataFrame(rows)
    frame.to_csv(output/"bootstrap_matched_means.csv", index=False)
    for (variant, component, coordinate), target in target_grid.items():
        group = (frame[(frame.variant == variant) & (frame.component == component) & (frame.coordinate == coordinate)]
                 if len(frame) else frame)
        quantiles = np.quantile(group.estimate, [.025, .5, .975]) if len(group) else [None]*3
        summaries.append({"variant": variant, "component": component, "coordinate": coordinate, **target,
                          "available": len(group), "unavailable": target["planned"]-len(group),
                          "percentile025": quantiles[0], "median": quantiles[1], "percentile975": quantiles[2],
                          "boundary_checked": int(group.boundary_hit.notna().sum()) if len(group) else 0,
                          "boundary_hits": int((group.boundary_hit == True).sum()) if len(group) else 0,
                          "score_checked": int(group.score_converged.notna().sum()) if len(group) else 0,
                          "score_passed": int((group.score_converged == True).sum()) if len(group) else 0,
                          "nonstationary_where_checked": int((group.score_converged == False).sum()) if len(group) else 0,
                          "loading_identification_checked": int(group.correlation_loading_identified.notna().sum()) if len(group) else 0,
                          "loading_nonidentified": int((group.correlation_loading_identified == False).sum()) if len(group) else 0,
                          "matching_RMSE_standardized_median": group.matching_RMSE_standardized.median() if len(group) else None,
                          "matching_RMSE_standardized_max": group.matching_RMSE_standardized.max() if len(group) else None,
                          "interpretation": "Conditional resampling stability; nominal coverage not established"})
    pd.DataFrame(summaries).to_csv(output/"bootstrap_stability.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--primary-run", type=Path)
    parser.add_argument("--source-kind", choices=["simulations", "applications"])
    parser.add_argument("--config", type=Path, default=CONFIG)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--task-index", type=int)
    group.add_argument("--task-range")
    group.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        if args.primary_run is None or args.source_kind is None:
            parser.error("Freezing requires --primary-run and --source-kind")
        result = freeze(args.output, args.primary_run, args.source_kind, args.config)
        print(json.dumps({"task_count": len(result["tasks"]), "config_sha256": result["config_sha256"]}))
    elif args.aggregate:
        print(json.dumps(aggregate(args.output, args.primary_run)))
    else:
        manifest, _, _ = verify_run(args.output, args.primary_run)
        if args.task_range:
            start, stop = map(int, args.task_range.split(":"))
            if not 0 <= start < stop <= len(manifest["tasks"]):
                raise ValueError("Invalid half-open task range")
            indices = range(start, stop)
        else:
            indices = [args.task_index]
        for index in indices:
            result = run_task(args.output, index, args.primary_run)
            print(json.dumps({"task_index": index, "status": result["status"],
                              "failed": result.get("row", {}).get("failed")}), flush=True)


if __name__ == "__main__":
    main()
