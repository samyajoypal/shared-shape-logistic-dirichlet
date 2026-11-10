"""Interval availability, coverage, length and paired Monte Carlo summaries.

Reads saved numerical interval records and a reference aggregate. No model
fitting, data download or document generation is performed.
"""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import math
import numpy as np
import pandas as pd
from scipy.stats import binomtest

KEY = ["scenario", "n", "target", "method"]

METHODS = ["wald", "bootstrap_percentile", "bootstrap_basic", "bootstrap_t"]

TARGETS = ["alpha_1", "lower_mean_weight", "mean_contrast_0"]

REASONS = ["fit_exception", "nonstationary", "tiny_weight", "invalid_target", "outer_target_unavailable",
           "bound_hit", "component_mean_ordering_tie", "information_not_positive_definite",
           "information_ill_conditioned", "information_exception", "invalid_target_variance",
           "bootstrap_incomplete", "too_few_valid_bootstrap_draws", "nonfinite_or_reversed_interval"]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def semantic_hash(value):
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode()
    return hashlib.sha256(raw).hexdigest()

def strict_bool(series, missing=False):
    allowed = {"True": True, "False": False}
    if missing:
        allowed[""] = None
    if not set(series.unique()) <= set(allowed):
        raise ValueError("Invalid Boolean or missing-value encoding")
    return series.map(allowed)

def tokens(value):
    result = set(value.split("|")) if value else set()
    if value and len(result) != len(value.split("|")):
        raise ValueError("Duplicate encoded exclusion reason")
    if not result <= set(REASONS):
        raise ValueError("Unknown recorded exclusion reason: " + str(result-set(REASONS)))
    return result

def proportion(successes, count):
    if count == 0:
        return {"estimate": None, "mcse": None, "mc95_low": None, "mc95_high": None}
    p = successes / count
    ci = binomtest(successes, count).proportion_ci(confidence_level=.95, method="exact")
    return {"estimate": p, "mcse": math.sqrt(p*(1-p)/count), "mc95_low": ci.low, "mc95_high": ci.high}

def mean_mcse(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None, None
    return float(values.mean()), float(values.std(ddof=1)/np.sqrt(len(values))) if len(values)>1 else None

def add_proportion(row, prefix, successes, count):
    for key, value in proportion(successes, count).items():
        row[prefix + ("" if key == "estimate" else "_" + key)] = value

def summarize_group(frame, planned):
    available = frame[frame.available]
    a = len(available)
    covered = int(available.covered.sum())
    row = {"planned": planned, "recorded": len(frame), "missing": planned-len(frame),
           "available": a, "unavailable": len(frame)-a, "covered": covered}
    add_proportion(row, "availability", a, planned)
    add_proportion(row, "conditional_coverage", covered, a)
    add_proportion(row, "covered_and_available", covered, planned)
    row["mean_length"], row["mean_length_mcse"] = mean_mcse(available.length)
    for key, q in [("median_length", .5), ("length_p90", .9), ("length_p99", .99)]:
        row[key] = float(np.quantile(available.length, q)) if a else None
    row["maximum_length"] = float(available.length.max()) if a else None
    row["largest_length_share"] = float(available.length.max()/available.length.sum()) if a and available.length.sum()>0 else None
    row["interval_below_truth"] = int((available.upper < available.truth).sum())
    row["interval_above_truth"] = int((available.lower > available.truth).sum())
    full = available[available.full_draw_availability]
    row["full_draw_available"] = len(full)
    add_proportion(row, "full_draw_availability", len(full), planned)
    add_proportion(row, "full_draw_conditional_coverage", int(full.covered.sum()), len(full))
    add_proportion(row, "full_draw_covered_and_available", int(full.covered.sum()), planned)
    if frame.method.iloc[0] == "wald":
        for key in list(row):
            if key.startswith("full_draw_"):
                row[key] = None
    return row

def paired_group(left, right, planned):
    paired = left.merge(right, on="task_index", suffixes=("_left", "_right"), validate="one_to_one")
    common = paired[paired.available_left & paired.available_right]
    n = len(common)
    row = {"planned": planned, "recorded_pairs": len(paired), "both_available": n,
           "left_only_available": int((paired.available_left & ~paired.available_right).sum()),
           "right_only_available": int((~paired.available_left & paired.available_right).sum()),
           "neither_available": int((~paired.available_left & ~paired.available_right).sum())}
    add_proportion(row, "common_availability", n, planned)
    add_proportion(row, "left_coverage_on_common", int(common.covered_left.sum()), n)
    add_proportion(row, "right_coverage_on_common", int(common.covered_right.sum()), n)
    difference = common.covered_right.astype(int) - common.covered_left.astype(int)
    row["coverage_difference"], row["coverage_difference_mcse"] = mean_mcse(difference)
    row["left_only_covers"] = int(((common.covered_left == True) & (common.covered_right == False)).sum())
    row["right_only_covers"] = int(((common.covered_left == False) & (common.covered_right == True)).sum())
    row["left_mean_length_on_common"], row["left_mean_length_mcse_on_common"] = mean_mcse(common.length_left)
    row["right_mean_length_on_common"], row["right_mean_length_mcse_on_common"] = mean_mcse(common.length_right)
    row["length_difference"], row["length_difference_mcse"] = mean_mcse(common.length_right-common.length_left)
    return row

def read_inputs(raw, frozen):
    paths = {"intervals": raw/"inference_task_intervals.csv.gz", "outer": raw/"inference_outer_diagnostics.csv.gz",
             "extraction": raw/"extraction_manifest.json", "aggregate": frozen/"aggregate.json",
             "config": frozen/"production_configuration.json"}
    hashes = {key: sha(path) for key, path in paths.items()}
    extraction = json.loads(paths["extraction"].read_text())
    aggregate = json.loads(paths["aggregate"].read_text())
    config = json.loads(paths["config"].read_text())
    if hashes["aggregate"] != extraction["aggregate_sha256"]:
        raise ValueError("Frozen aggregate differs from the extraction receipt")
    if not semantic_hash(config) == extraction["config_sha256"] == aggregate["config_sha256"]:
        raise ValueError("Configuration identity differs")
    if extraction["frozen_source_sha256"] != aggregate["source_sha256"]:
        raise ValueError("Extraction and aggregate source identities differ")
    if config["intervals"]["methods"] != METHODS or config["intervals"]["targets"] != TARGETS:
        raise ValueError("The complete four-method three-target design is required")
    frame = pd.read_csv(paths["intervals"], dtype=str, keep_default_na=False)
    outer = pd.read_csv(paths["outer"], dtype=str, keep_default_na=False)
    for table in [frame, outer]:
        for name in ["task_index", "cell_index", "replicate", "n", "total_tasks", "bootstrap_attempted"]:
            if not table[name].str.fullmatch(r"\d+").all():
                raise ValueError("Invalid integer field " + name)
            table[name] = table[name].astype(int)
        table["outer_bound_hit"] = strict_bool(table.outer_bound_hit)
        table["separation_multiplier"] = pd.to_numeric(table.separation_multiplier, errors="raise")
    for name in ["available", "full_draw_availability", "miss_interval_below_truth", "miss_interval_above_truth"]:
        frame[name] = strict_bool(frame[name])
    frame["covered"] = strict_bool(frame.covered, missing=True)
    for name in ["length", "lower", "upper", "truth", "estimate", "bootstrap_valid", "bootstrap_requested", "bootstrap_required"]:
        frame[name] = pd.to_numeric(frame[name].replace("", np.nan), errors="raise")
    for name in ["ordering_gap", "min_weight", "information_condition", "bootstrap_bound_hits"]:
        outer[name] = pd.to_numeric(outer[name].replace("", np.nan), errors="raise")
    frame["reasons"] = frame.exclusion_reasons.map(tokens)
    outer["fit_flags"] = outer.fit_reasons.map(tokens)
    outer["information_flags"] = outer.information_reasons.map(tokens)
    validate_records(frame, outer, config, aggregate, extraction)
    return frame, outer, config, aggregate, paths, hashes

def validate_records(frame, outer, config, aggregate, extraction):
    r = config["replications_per_cell"]
    cells = list(itertools.product(config["scenarios"], config["sample_sizes"]))
    total = len(cells)*r
    if len(cells) != 9 or r != 500 or aggregate["completed_tasks"] != total or aggregate["missing_task_indices"]:
        raise ValueError("This publication requires all nine settings and all4500 completed tasks")
    if set(map(int, extraction["result_sha256"])) != set(range(total)):
        raise ValueError("Extraction receipt does not cover every canonical task")
    if outer.task_index.duplicated().any() or set(outer.task_index) != set(range(total)):
        raise ValueError("Duplicate or missing outer task")
    if frame.duplicated(["task_index", "target", "method"]).any() or len(frame) != total*12:
        raise ValueError("Duplicate or missing interval slot")
    expected_slots = set(itertools.product(TARGETS, METHODS))
    for index, part in frame.groupby("task_index", sort=False):
        if set(zip(part.target, part.method)) != expected_slots:
            raise ValueError("Changed method/target scope")
    for cell, (scenario, size) in enumerate(cells):
        part = outer[outer.cell_index == cell]
        if (len(part) != r or set(part.task_index) != set(range(cell*r,(cell+1)*r))
                or set(part.replicate) != set(range(r)) or not part.scenario.eq(scenario["name"]).all()
                or not part.n.eq(size).all() or not part.total_tasks.eq(total).all()
                or not part.separation_multiplier.eq(scenario["separation_multiplier"]).all()):
            raise ValueError("Task metadata differs from the frozen design")
    joined = frame.merge(outer, on="task_index", suffixes=("", "_outer"), validate="many_to_one")
    for key in ["cell_index", "replicate", "scenario", "n", "total_tasks", "separation_multiplier", "outer_bound_hit", "bootstrap_attempted"]:
        if not joined[key].eq(joined[key+"_outer"]).all():
            raise ValueError("Outer and interval metadata differ: " + key)
    a = frame[frame.available]; u = frame[~frame.available]
    if (not np.isfinite(a[["lower", "upper", "length", "truth", "estimate"]].to_numpy(float)).all()
            or not a.lower.le(a.upper).all() or not np.allclose(a.length,a.upper-a.lower,rtol=1e-13,atol=1e-13)
            or not a.covered.eq(a.lower.le(a.truth) & a.upper.ge(a.truth)).all()
            or not a.exclusion_reasons.eq("").all() or u.exclusion_reasons.eq("").any()
            or not u[["lower", "upper", "length"]].isna().all().all() or u.covered.notna().any()):
        raise ValueError("Malformed available/unavailable interval")
    if not frame.bootstrap_requested.eq(config["intervals"]["bootstrap_draws"]).all():
        raise ValueError("Changed bootstrap count")
    required = max(config["intervals"]["minimum_valid_draws"],math.ceil(config["intervals"]["bootstrap_draws"]*config["intervals"]["minimum_valid_fraction"]))
    if not frame.bootstrap_required.eq(required).all():
        raise ValueError("Changed bootstrap eligibility threshold")
    wald = frame[frame.method.eq("wald")]
    boot = frame[~frame.method.eq("wald")]
    if (wald.bootstrap_valid.notna().any() or wald.full_draw_availability.any()
            or not boot.full_draw_availability.eq(boot.bootstrap_valid.eq(config["intervals"]["bootstrap_draws"])).all()):
        raise ValueError("Full-draw availability disagrees with eligible bootstrap count")
    for target in TARGETS:
        part = frame[frame.target == target]
        truth = (.8 if target == "alpha_1" else .55 if target == "lower_mean_weight" else 4*part.separation_multiplier)
        if not np.allclose(part.truth, truth, rtol=0, atol=1e-14):
            raise ValueError("Changed generating target")

def derive(frame, outer, config, aggregate):
    r = config["replications_per_cell"]
    summary, paired, exclusions, diagnostics, bootstrap = [], [], [], [], []
    frozen_diag = {(v["scenario"],v["n"]):v for v in aggregate["fit_diagnostics"]}
    for scenario, n in itertools.product(config["scenarios"], config["sample_sizes"]):
        name = scenario["name"]
        selected = frame[frame.scenario.eq(name) & frame.n.eq(n)]
        outer_part = outer[outer.scenario.eq(name) & outer.n.eq(n)]
        d = {"scenario":name,"n":n,"planned":r,"recorded":len(outer_part)}
        combined = outer_part.apply(lambda v: v.fit_flags | v.information_flags,axis=1)
        for reason in REASONS:
            d[reason] = int(combined.map(lambda flags:reason in flags).sum())
        d["information_problem"] = int(combined.map(lambda flags:bool(flags & {"information_not_positive_definite","information_ill_conditioned","information_exception","invalid_target_variance"})).sum())
        d["any_recorded_reason"] = int(combined.map(bool).sum())
        base = frozen_diag[(name,n)]
        assert d["bound_hit"] == int(outer_part.outer_bound_hit.sum()) == base["outer_bound_hits"]
        assert d["nonstationary"] == base["outer_nonstationary"]
        assert d["fit_exception"] == base["outer_fit_exceptions"]
        diagnostics.append(d)
        b = {k:v for k,v in base.items() if k.startswith("bootstrap_")}
        b.update({"scenario":name,"n":n,"outer_samples":r,"maximum_planned_draws":r*config["intervals"]["bootstrap_draws"],
                  "outer_batches_skipped":int(outer_part.bootstrap_attempted.eq(0).sum()),
                  "draws_missing_regular_information":base["bootstrap_attempted"]-base["bootstrap_regular_information"]})
        assert b["bootstrap_attempted"] == int(outer_part.bootstrap_attempted.sum())
        assert b["bootstrap_bound_hits"] == int(outer_part.bootstrap_bound_hits.sum())
        bootstrap.append(b)
        for target in TARGETS:
            target_part = selected[selected.target.eq(target)]
            left = target_part[target_part.method.eq("wald")]
            for method in METHODS:
                part = target_part[target_part.method.eq(method)]
                identity = dict(scenario=name,n=n,target=target,method=method)
                summary.append({**identity,**summarize_group(part,r)})
                e = {**identity,"planned":r,"unavailable":int((~part.available).sum())}
                for reason in REASONS:
                    e[reason] = int(part.reasons.map(lambda flags:reason in flags).sum())
                exclusions.append(e)
                if method != "wald":
                    paired.append({**identity,"baseline":"wald",**paired_group(left,part,r)})
    return tuple(pd.DataFrame(x) for x in [summary,paired,diagnostics,bootstrap,exclusions])

def reconcile(summary, paired, aggregate):
    maximum_error = 0.
    def same(a,b):
        nonlocal maximum_error
        if a is None or pd.isna(a) or b is None or pd.isna(b):
            assert (a is None or pd.isna(a)) and (b is None or pd.isna(b))
        else:
            maximum_error=max(maximum_error,abs(float(a)-float(b)))
            if not np.isclose(a,b,rtol=2e-13,atol=2e-13):
                raise ValueError(("Frozen-summary mismatch",a,b))
    lookup = summary.set_index(KEY)
    mapping = {"available":"available","unavailable":"unavailable","covered":"covered_count",
               "conditional_coverage":"conditional_coverage","covered_and_available":"covered_and_available_fraction",
               "conditional_coverage_mcse":"conditional_coverage_mcse","mean_length":"mean_length_conditional"}
    for original in aggregate["coverage"]:
        row = lookup.loc[tuple(original[k] for k in KEY)]
        for new,old in mapping.items():
            same(row[new],original[old])
    lookup = paired.set_index(KEY)
    for original in aggregate["paired"]:
        row=lookup.loc[tuple(original[k] for k in KEY)]
        for new,old in [("both_available","both_available"),("coverage_difference","coverage_difference_on_common_available"),
                        ("coverage_difference_mcse","paired_mcse"),("length_difference","mean_length_difference_on_common_available")]:
            same(row[new],original[old])
    if len(summary)!=108 or len(paired)!=81:
        raise ValueError("Incomplete scientific summary")
    return {"coverage_rows_reconciled":108,"paired_rows_reconciled":81,"maximum_absolute_roundoff_difference":maximum_error}


def run(raw, output, frozen=None):
    raw, output = Path(raw), Path(output)
    frozen = Path(frozen) if frozen is not None else raw
    frame, outer, config, aggregate, paths, hashes = read_inputs(raw, frozen)
    tables = derive(frame, outer, config, aggregate)
    checks = reconcile(tables[0], tables[1], aggregate)
    output.mkdir(parents=True, exist_ok=True)
    filenames = ["interval_summary.csv", "paired_comparisons.csv", "outer_diagnostics.csv",
                 "bootstrap_diagnostics.csv", "interval_exclusions.csv"]
    output_hashes = {}
    for name, table in zip(filenames, tables):
        path = output / name
        table.to_csv(path, index=False)
        output_hashes[name] = sha(path)
    if any(sha(paths[k]) != value for k, value in hashes.items()):
        raise ValueError("An input changed during reporting")
    receipt = {"status": "pass", "new_fits": False,
               "outer_samples": len(outer), "interval_records": len(frame),
               "input_sha256": hashes, "input_files": {k: p.name for k, p in paths.items()},
               "output_sha256": output_hashes, "source_sha256": sha(__file__),
               "reconciliation": checks}
    (output / "report_manifest.json").write_text(json.dumps(receipt, indent=2)+"\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True,
                        help="Directory containing interval and outer CSVs and the extraction receipt")
    parser.add_argument("--frozen", type=Path,
                        help="Directory containing aggregate.json and production_configuration.json; defaults to --raw")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.raw, args.output, args.frozen), indent=2))
