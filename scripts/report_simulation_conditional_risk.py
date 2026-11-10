"""Independent, read-only tables from the complete saved conditional-risk report.

No fit or prediction is evaluated. All fixed cells and both eligibility subsets
are retained, and paired ratios are never reconstructed from unpaired means.
"""
from pathlib import Path
import argparse
import hashlib
import itertools
import json

import numpy as np
import pandas as pd

METHODS=("shared_ld","full_ld","gaussian_shared_correlation")
SUBSETS=("all_finite","all_methods_score_eligible")


class IntegrityError(RuntimeError):pass
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def flag(value):
    if pd.isna(value):return None
    if isinstance(value,(bool,np.bool_)):return bool(value)
    raise IntegrityError("Numerical flags must be true, false or unknown")


def flag_counts(values,prefix):
    parsed=[flag(v) for v in values]
    return {prefix+"_true":sum(v is True for v in parsed),prefix+"_false":sum(v is False for v in parsed),
            prefix+"_unknown":sum(v is None for v in parsed)}


def risk_statistics(values):
    x=np.asarray(values,float)
    return {"available":len(x),"mean_risk":float(np.mean(x)) if len(x) else np.nan,
            "risk_mcse":float(np.sqrt(np.var(x,ddof=1)/len(x))) if len(x)>1 else np.nan}


def paired_statistics(left,right):
    x,y=np.asarray(left,float),np.asarray(right,float);n=len(x)
    if x.shape!=y.shape:raise IntegrityError("Unmatched replicate risk vectors")
    out={"paired_available":n,"mean_left":np.nan,"mean_right":np.nan,"difference":np.nan,
         "difference_mcse":np.nan,"risk_ratio":np.nan,"ratio_mcse":np.nan}
    if not n:return out
    left_mean,right_mean=x.mean(),y.mean();out.update(mean_left=left_mean,mean_right=right_mean,difference=left_mean-right_mean)
    if right_mean>0:out["risk_ratio"]=left_mean/right_mean
    if n>1:
        centered=np.column_stack([x-left_mean,y-right_mean])
        contrast=np.array([1.,-1.]);out["difference_mcse"]=np.linalg.norm(centered@contrast)/np.sqrt(n*(n-1))
        if right_mean>0:
            gradient=np.array([1/right_mean,-left_mean/right_mean**2])
            out["ratio_mcse"]=np.linalg.norm(centered@gradient)/np.sqrt(n*(n-1))
    return out


def verify_number(actual,expected,name):
    if pd.isna(actual) and pd.isna(expected):return
    if pd.isna(actual) or pd.isna(expected) or not np.isclose(float(actual),float(expected),rtol=2e-10,atol=2e-14):
        raise IntegrityError("Frozen aggregate differs from independent calculation: "+name)


def build_tables(manifest,records,summary,paired):
    config=manifest["config"];tasks=manifest["tasks"];cells={c["id"]:c for c in config["cells"]}
    if config["methods"]!=list(METHODS) or config["subsets"]!=list(SUBSETS):raise IntegrityError("Changed reporting procedure scope")
    if len(cells)!=len(config["cells"]) or {t["cell_id"] for t in tasks}!=set(cells):
        raise IntegrityError("Planned tasks and unique configured cells differ")
    expected={(t["task_index"],method) for t in tasks for method in METHODS}
    if (records.duplicated(["task_index","method"]).any() or set(zip(records.task_index,records.method))!=expected):
        raise IntegrityError("Method records omit or duplicate a planned dataset/procedure")
    taskmap={t["task_index"]:t for t in tasks}
    if len(taskmap)!=len(tasks) or [t["task_index"] for t in tasks]!=list(range(len(tasks))):
        raise IntegrityError("Noncanonical planned reporting task indices")
    for cell,spec in cells.items():
        if sorted(t["replicate"] for t in tasks if t["cell_id"]==cell)!=list(range(spec["repetitions"])):
            raise IntegrityError("Planned cell replicate sequence differs")
    for row in records.to_dict("records"):
        task=taskmap[row["task_index"]]
        if row["cell_id"]!=task["cell_id"] or row["replicate"]!=task["replicate"] or row["primary_task_index"]!=task["primary_task_index"]:
            raise IntegrityError("Method row belongs to a different planned dataset")
        available=flag(row["available"])
        recorded=flag(row["recorded"])
        if recorded is None or available is True and recorded is not True:
            raise IntegrityError("Prediction availability differs from recorded task status")
        if recorded is False and (available is not False or not pd.isna(row["test_integration_mcse"])
            or any(flag(row[key]) is not None for key in ("source_failed","score_converged","boundary_hit"))):
            raise IntegrityError("Unrecorded task has source diagnostics or an integration error")
        if available is None or (available and (not np.isfinite(row["risk"]) or row["risk"]<0)):
            raise IntegrityError("Invalid risk availability or numerical value")
        if not available and not pd.isna(row["risk"]):raise IntegrityError("Unavailable method has a numeric risk")
        for field in ("recorded","source_failed","score_converged","boundary_hit"):flag(row[field])
    skeys=["cell_id","method","subset"];pkeys=["cell_id","left_method","right_method","subset"]
    sexpected={(c,m,s) for c in cells for m in METHODS for s in SUBSETS}
    pexpected={(c,l,r,s) for c in cells for l,r in itertools.combinations(METHODS,2) for s in SUBSETS}
    if summary.duplicated(skeys).any() or set(map(tuple,summary[skeys].to_numpy()))!=sexpected:
        raise IntegrityError("Frozen method summary omits or duplicates a fixed cell")
    if paired.duplicated(pkeys).any() or set(map(tuple,paired[pkeys].to_numpy()))!=pexpected:
        raise IntegrityError("Frozen paired summary omits or duplicates a fixed comparison")
    methods=[];comparisons=[]
    for cell,spec in cells.items():
        block=records[records.cell_id==cell];planned=spec["repetitions"]
        if any(len(block[block.method==m])!=planned for m in METHODS):raise IntegrityError("Cell repetition count differs")
        common=set(block.replicate)
        for method in METHODS:
            good=block[(block.method==method)&block.available.eq(True)&block.score_converged.eq(True)]
            common&=set(good.replicate)
        meta={"cell_id":cell,"block":spec["block"],"d":spec["dimension"],"k":spec["components"],
              "n_train":spec["n_train"],"heterogeneity":spec["heterogeneity"],"local_h_coefficient":spec.get("local_h_coefficient"),
              "target_index":config["target_index"],"test_grid_rows":config["n_test"],"planned":planned}
        for subset in SUBSETS:
            selection=block if subset==SUBSETS[0] else block[block.replicate.isin(common)]
            for method in METHODS:
                original=block[block.method==method];good=selection[(selection.method==method)&selection.available.eq(True)]
                stats=risk_statistics(good.risk)
                expected=summary[(summary.cell_id==cell)&(summary.method==method)&(summary.subset==subset)].iloc[0]
                for key,value in {**stats,"planned":planned,"recorded":int(original.recorded.eq(True).sum())}.items():verify_number(expected[key],value,key)
                additional={"unavailable_or_ineligible":planned-len(good),"source_failures":int(original.source_failed.eq(True).sum()),
                    "score_passed":int(original.score_converged.eq(True).sum()),"boundary_hits":int(original.boundary_hit.eq(True).sum()),
                    "mean_test_integration_mcse":good.test_integration_mcse.mean() if len(good) else np.nan}
                for key,value in additional.items():
                    if key in expected:verify_number(expected[key],value,key)
                methods.append({**meta,"method":method,"subset":subset,**stats,"unavailable_or_ineligible":planned-len(good),
                    "recorded":int(original.recorded.eq(True).sum()),"missing_task_records":int(original.recorded.eq(False).sum()),
                    **flag_counts(original.source_failed,"source_failure"),**flag_counts(original.score_converged,"score"),
                    **flag_counts(original.boundary_hit,"boundary"),
                    "mean_test_integration_mcse":good.test_integration_mcse.mean() if len(good) else np.nan})
            for left,right in itertools.combinations(METHODS,2):
                l=selection[(selection.method==left)&selection.available.eq(True)]
                r=selection[(selection.method==right)&selection.available.eq(True)]
                common_rows=l.merge(r,on="replicate",suffixes=("_left","_right"),validate="one_to_one")
                stats=paired_statistics(common_rows.risk_left,common_rows.risk_right)
                expected=paired[(paired.cell_id==cell)&(paired.left_method==left)&(paired.right_method==right)&(paired.subset==subset)].iloc[0]
                for key,value in {**stats,"planned":planned}.items():verify_number(expected[key],value,key)
                if "paired_unavailable_or_ineligible" in expected:
                    verify_number(expected["paired_unavailable_or_ineligible"],planned-len(common_rows),"paired_unavailable_or_ineligible")
                comparisons.append({**meta,"left_method":left,"right_method":right,"subset":subset,**stats,
                    "paired_unavailable_or_ineligible":planned-len(common_rows),
                    "uncertainty":"Monte Carlo SE across independent simulation replicates; not real-data population uncertainty"})
    return pd.DataFrame(methods),pd.DataFrame(comparisons)


def run(frozen_report,output):
    frozen_report=Path(frozen_report).resolve();output=Path(output).resolve()
    if output==frozen_report or frozen_report in output.parents:raise ValueError("Presentation output must be separate from the frozen report")
    manifest=read(frozen_report/"manifest.json");aggregation=read(frozen_report/"aggregation.json")
    if aggregation["manifest_sha256"]!=digest(frozen_report/"manifest.json"):raise IntegrityError("Aggregation belongs to another report")
    required={"method_records.csv","risk_summary.csv","paired_risk_summary.csv"}
    if not required<=set(aggregation["outputs_sha256"]):raise IntegrityError("Aggregation omits a required artifact hash")
    hashes={str(frozen_report/"manifest.json"):digest(frozen_report/"manifest.json"),
            str(frozen_report/"aggregation.json"):digest(frozen_report/"aggregation.json")}
    for name,h in aggregation["outputs_sha256"].items():
        path=frozen_report/name
        if digest(path)!=h:raise IntegrityError("Frozen aggregate artifact changed: "+name)
        hashes[str(path)]=h
    records=pd.read_csv(frozen_report/"method_records.csv");summary=pd.read_csv(frozen_report/"risk_summary.csv");paired=pd.read_csv(frozen_report/"paired_risk_summary.csv")
    methods,comparisons=build_tables(manifest,records,summary,paired)
    if (aggregation["planned_predictions"]!=len(records) or aggregation["available_predictions"]!=records.available.eq(True).sum()
        or aggregation["recorded_predictions"]!=records.recorded.eq(True).sum()):raise IntegrityError("Aggregation availability totals differ")
    output.mkdir(parents=True,exist_ok=False)
    methods.to_csv(output/"all_cell_method_risks.csv",index=False)
    comparisons.to_csv(output/"all_cell_paired_risks.csv",index=False)
    comparisons[comparisons.left_method=="shared_ld"].to_csv(output/"shared_shape_comparisons.csv",index=False)
    primary=comparisons[(comparisons.subset==SUBSETS[0])&(comparisons.left_method=="shared_ld")]
    for block in ("core","local_alternative"):primary[primary.block==block].to_csv(output/(block+"_primary_pairs.csv"),index=False)
    (output/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    notes=["Simulation conditional-mean risk tables","======================================","",
        "All fixed cells and eligibility subsets are retained. No fit or prediction was recomputed.",
        "Risk is squared fitted-minus-exact-true conditional mean, averaged over the saved independent test covariates.",
        "It excludes irreducible response variance. Smaller risk is better; ratios divide paired mean risks.",
        "Method means may have different available sets; comparison tables include their own paired means and denominators.",
        "MCSEs quantify finite simulation error across independent training/test replicates, not population uncertainty for real applications.",
        "No estimator variance/bias decomposition is performed. Common-score subsets retain active bounds and are secondary.",
        "The local-alternative block changes its generating law with n; it is not a fixed-law replication of one finite-sample problem.",""]
    for row in primary.to_dict("records"):
        text=(f"{row['cell_id']} | shared / {row['right_method']} | pairs {row['paired_available']}/{row['planned']} | "
              f"difference {row['difference']:.6g} (MCSE {row['difference_mcse']:.3g}) | ratio {row['risk_ratio']:.6g} (MCSE {row['ratio_mcse']:.3g})")
        notes.append(text)
    (output/"table_notes.rst").write_text("\n".join(notes)+"\n")
    for path,h in hashes.items():
        if digest(path)!=h:raise IntegrityError("Aggregate input changed during table generation")
    receipt={"input_sha256":hashes,"source_sha256":digest(__file__),"refitting":False,"risk_predictions_recomputed":False,
        "all_cells":len(manifest["config"]["cells"]),"planned_predictions":len(records),
        "outputs_sha256":{p.name:digest(p) for p in output.iterdir() if p.is_file()}}
    (output/"presentation_manifest.json").write_text(json.dumps(receipt,indent=2)+"\n")
    return receipt


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--frozen-report",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();print(json.dumps(run(a.frozen_report,a.output)))
