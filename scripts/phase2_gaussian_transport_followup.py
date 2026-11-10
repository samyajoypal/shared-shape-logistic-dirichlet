"""Post-result Gaussian comparator for every original olive area-out task.

This adapter changes task selection and reporting only. It uses the frozen
Gaussian fitter, training transforms, partitions, bounds and optimizer settings.
The original phase-2 studies and source files are never edited.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "_phase2_transport_control_core", ROOT / "scripts/phase2_gaussian_control.py")
control = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(control)
CONFIG = ROOT / "configs/phase2_gaussian_transport_followup.json"
_ORIGINAL_FILES = control.source_files
_ORIGINAL_LOAD_CASE = control.load_case


def task_map(spec, kind):
    if kind != "applications":
        raise ValueError("This follow-up is only for application transport")
    variants = {v["id"] for v in spec["config"]["variants"]
                if v.get("primary") and v["dataset"] == "oliveoil"}
    answer = []
    for index, task in enumerate(spec["tasks"]):
        if task["task_index"] != index:
            raise ValueError("Noncanonical primary task map")
        if (task["variant"] in variants and task["kind"] == "leave_area_out"
                and task["bounds"] == "primary"):
            answer.append({"task_index": len(answer), "primary_task_index": index,
                           **{key: task[key] for key in ["variant", "kind", "repeat", "fold", "area"]}})
    if variants != {"olive3_oleic", "olive6_oleic"} or len(answer) != 18:
        raise ValueError("Expected both original primary oil representations and all 18 area folds")
    if any(sum(t["variant"] == v for t in answer) != 9 for v in variants):
        raise ValueError("Expected nine collection areas for each representation")
    return answer


def load_case(*args, **kwargs):
    case = _ORIGINAL_LOAD_CASE(*args, **kwargs)
    case["evaluation_role"] = "held_out_collection_area"
    return case


def source_files():
    return _ORIGINAL_FILES() + [Path(__file__), CONFIG]


# Explicit adapter hooks; the imported implementation is unchanged on disk.
control.task_map = task_map
control.load_case = load_case
control.source_files = source_files


def freeze(output, primary, config_path=CONFIG):
    manifest = control.freeze(output, primary, "applications", config_path)
    manifest["created_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["interpretation"] = (
        "Post-result comparator follow-up for all18 original olive collection-area folds; "
        "motivated after inspection of primary aggregate outcomes. No model, bounds, "
        "optimizer or area selection was tuned. Not part of the original frozen design.")
    control.atomic_json(Path(output) / "manifest.json", manifest)
    return manifest


def aggregate(output, primary_override=None):
    output = Path(output)
    manifest, primary, _ = control.verify_run(output, primary_override)
    manifest_hash = control.digest(output / "manifest.json")
    rows = []
    for task in manifest["tasks"]:
        directory = output / "tasks" / f"task_{task['task_index']:06d}"
        path = directory / "result.json"
        if not path.exists():
            rows.append({**task, "method": control.METHOD, "recorded": False,
                         "failed": True, "evaluation_available": False})
            continue
        value = json.loads(path.read_text())
        binding = json.loads((directory / "input_binding.json").read_text())
        if (value["status"] != "complete" or value["task"] != task
                or value["manifest_sha256"] != manifest_hash
                or value["input_binding_sha256"] != control.semantic_hash(binding)):
            raise ValueError("Follow-up result identity mismatch")
        for relative, expected in binding["input_sha256"].items():
            if control.digest(primary / relative) != expected:
                raise ValueError("Follow-up input changed: " + relative)
        row = value["row"]
        rows.append({**task, **row, "recorded": True,
                     "joint": row["test_log_density"], "conditional": row["conditional_log_density"]})
        for method, saved in value["baseline_records"].items():
            rows.append({**task, **saved, "task_index": task["task_index"],
                         "method": method, "recorded": True,
                         "evaluation_available": not saved["failed"],
                         "joint": saved.get("mean_log_density"),
                         "conditional": saved.get("mean_conditional_log_density")})
    frame = pd.DataFrame(rows)
    for field in ["joint", "conditional", "n_test"]:
        if field not in frame:
            frame[field] = np.nan
    available = frame[frame.evaluation_available]
    if (not np.isfinite(available[["joint", "conditional", "n_test"]].to_numpy(float)).all()
            or (available.n_test <= 0).any() or available.failed.any()):
        raise ValueError("Invalid score, row count or failure flag in an available result")
    frame.to_csv(output / "area_records.csv", index=False)
    summaries = []
    comparisons = []
    for variant in sorted({t["variant"] for t in manifest["tasks"]}):
        subset = frame[frame.variant == variant]
        shared = subset[(subset.method == "shared_ld") & subset.evaluation_available]
        for method, group in subset.groupby("method"):
            valid = group[group.evaluation_available]
            entry = {"variant": variant, "method": method, "planned_areas": 9,
                     "recorded": int(group.recorded.sum()), "available_areas": len(valid),
                     "complete": len(valid) == 9,
                     "boundary_checked": int(group.boundary_hit.notna().sum()) if "boundary_hit" in group else 0,
                     "boundary_hits": int(group.boundary_hit.eq(True).sum()) if "boundary_hit" in group else 0,
                     "score_checked": int(group.score_converged.notna().sum()) if "score_converged" in group else 0,
                     "score_passed": int(group.score_converged.eq(True).sum()) if "score_converged" in group else 0}
            for measure in ["joint", "conditional"]:
                entry[measure + "_weighted"] = (float(np.average(valid[measure], weights=valid.n_test))
                                                 if len(valid) == 9 else None)
            summaries.append(entry)
            if method == "shared_ld":
                continue
            common = shared.merge(valid, on="primary_task_index", suffixes=("_shared", "_other"))
            if not np.array_equal(common.n_test_shared, common.n_test_other):
                raise ValueError("Paired methods evaluated different row counts")
            for measure in ["joint", "conditional"]:
                delta = common[measure + "_shared"] - common[measure + "_other"]
                comparisons.append({"variant": variant, "comparator": method, "measure": measure,
                                    "planned_areas": 9, "available_pairs": len(common),
                                    "shared_minus_other_weighted": float(np.average(delta, weights=common.n_test_shared)) if len(common) == 9 else None,
                                    "shared_minus_other_equal_area_mean": float(delta.mean()) if len(common) == 9 else None,
                                    "areas_shared_better": int((delta > 0).sum()),
                                    "minimum_area_difference": float(delta.min()) if len(common) else None,
                                    "maximum_area_difference": float(delta.max()) if len(common) else None,
                                    "uncertainty": "Descriptive geographic transport contrast; selected areas are not independent random replications"})
    pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(output / "paired_comparisons.csv", index=False)
    controls = frame[frame.method == control.METHOD]
    status = {"planned": 18, "recorded": int(controls.recorded.sum()),
              "available": int(controls.evaluation_available.sum()),
              "failed_recorded": int((controls.recorded & controls.failed).sum()),
              "followup_status": "Added after observing the original results; no model or fitting-policy tuning"}
    control.atomic_json(output / "aggregation.json", status)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--primary-run", type=Path)
    parser.add_argument("--config", type=Path, default=CONFIG)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--task-index", type=int)
    group.add_argument("--task-range")
    group.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        if args.primary_run is None:
            parser.error("--freeze requires --primary-run")
        value = freeze(args.output, args.primary_run, args.config)
        print(json.dumps({"tasks": len(value["tasks"]), "config_sha256": value["config_sha256"]}))
    elif args.aggregate:
        print(json.dumps(aggregate(args.output, args.primary_run)))
    else:
        manifest, _, _ = control.verify_run(args.output, args.primary_run)
        if args.task_range:
            first, last = map(int, args.task_range.split(":"))
            if not 0 <= first < last <= len(manifest["tasks"]):
                raise ValueError("Invalid task range")
            indices = range(first, last)
        else:
            indices = [args.task_index]
        for index in indices:
            value = control.run_task(args.output, index, args.primary_run)
            print(json.dumps({"task_index": index, "status": value["status"],
                              "failed": value.get("row", {}).get("failed")}), flush=True)


if __name__ == "__main__":
    main()
