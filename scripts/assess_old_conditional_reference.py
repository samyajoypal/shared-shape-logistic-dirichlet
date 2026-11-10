"""Post-result conditional Gaussian reference on every original primary split.

No original fit, split, score, or stability artifact is modified. This script
fits only the separately declared deterministic conditional reference.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import scipy
from threadpoolctl import threadpool_limits

import conditional_log_linear as reference
from assess_conditional_prediction_stability import clean, semantic_hash
from phase3_stability import summarize_weighted_method, summarize_weighted_pair

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/"docs/public_analysis_rules.json"
PROTOCOL_SHA256='1307bd7d13968a986c6bd6995a5baa6bd7e546858cb4f3d7bb87fddd04806440'
VARIANTS=("olive3_oleic","olive6_oleic","crabs_log5")


class IntegrityError(RuntimeError):
    pass


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(clean(value),sort_keys=True,indent=2,allow_nan=False)+"\n")
    os.replace(temporary,path)


def load_grid(primary,spec,variant):
    relative="data/"+("oliveoil.csv" if variant["dataset"]=="oliveoil" else "crabs.csv")
    path=primary/"executed_sources"/relative
    if digest(path)!=spec["data_hashes"][relative]:raise ValueError("Original scientific data hash changed")
    frame=pd.read_csv(io.BytesIO(path.read_bytes()))
    raw=frame[variant["columns"]].to_numpy(float)+variant.get("add_to_all",0.)
    if not np.isfinite(raw).all() or np.any(raw<=0):raise ValueError("Original primary measurements are not positive")
    if variant["dataset"]=="oliveoil":
        r=variant["columns"].index(variant["reference"]);keep=[j for j in range(raw.shape[1]) if j!=r]
        y=np.log(raw[:,keep]/raw[:,r,None]);names=[variant["columns"][j]+"/"+variant["reference"] for j in keep]
        log_jacobian=np.zeros(len(y))
    elif variant["transform"]=="log":
        y=np.log(raw);names=variant["columns"]
        log_jacobian=-y[:,names.index(variant["conditional_target"])]
    else:raise ValueError("Unsupported original primary representation")
    return y,names.index(variant["conditional_target"]),names,log_jacobian


def run(primary,output,protocol=PROTOCOL,expected_protocol_sha256=PROTOCOL_SHA256,expected_tasks=768):
    primary,output,protocol=Path(primary).resolve(),Path(output).resolve(),Path(protocol).resolve()
    if output==primary or primary in output.parents:raise ValueError("Reference output must be separate")
    if digest(protocol)!=expected_protocol_sha256:raise ValueError("Public reproduction protocol hash mismatch")
    spec_path=primary/"specification.json";spec=json.loads(spec_path.read_text());spec_hash=digest(spec_path)
    for relative,sha in spec["source_hashes"].items():
        if digest(primary/"executed_sources"/relative)!=sha:raise ValueError("Original archived source changed")
    variants={v["id"]:v for v in spec["config"]["variants"] if v.get("primary")}
    if set(variants)!=set(VARIANTS):raise ValueError("Unexpected original primary representations")
    tasks=[t for t in spec["tasks"] if t["variant"] in variants and t["bounds"]=="primary"
           and t["kind"] in ("cv","full","bootstrap","leave_area_out")]
    if len(tasks)!=expected_tasks:raise ValueError("Original reference scope differs from the declared task count")
    if any(t["task_index"]!=j for j,t in enumerate(spec["tasks"])):raise ValueError("Noncanonical primary tasks")
    source_names=["scripts/assess_old_conditional_reference.py","scripts/conditional_log_linear.py",
                  "scripts/phase3_stability.py","scripts/assess_conditional_prediction_stability.py"]
    identity={"primary_specification_sha256":spec_hash,"protocol_sha256":digest(protocol),
        "source_sha256":{s:digest(ROOT/s) for s in source_names},"residual_sd_floor":1e-4,
        "score_tolerance":1e-6,"expected_tasks":expected_tasks,"tasks":tasks,
        "versions":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"pandas":pd.__version__}}
    manifest_path=output/"manifest.json"
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if any(manifest[key]!=value for key,value in identity.items()):raise ValueError("Reference source/configuration/input changed on resume")
        for relative,sha in identity["source_sha256"].items():
            if digest(output/"source_snapshot"/relative)!=sha:raise ValueError("Archived reference source changed")
    else:
        output.mkdir(parents=True,exist_ok=False)
        manifest={**identity,"created_utc":datetime.now(timezone.utc).isoformat(),"primary_root":str(primary),
                  "status":"Post-result conditional reference; not an original preregistered comparator"}
        for relative in source_names:
            destination=output/"source_snapshot"/relative;destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes((ROOT/relative).read_bytes())
        (output/protocol.name).write_bytes(protocol.read_bytes());atomic_json(manifest_path,manifest)
    manifest_hash=digest(manifest_path)
    grids={key:load_grid(primary,spec,v) for key,v in variants.items()}
    records=[]
    for task in tasks:
        old=primary/"tasks"/f"{task['task_index']:05d}";split_path=old/"split.json";complete_path=old/"completion.json"
        split=json.loads(split_path.read_text());complete=json.loads(complete_path.read_text())
        if complete["task"]!=task or complete["specification_sha256"]!=spec_hash or any(split.get(k)!=v for k,v in task.items()):
            raise ValueError("Original completed task or split identity mismatch")
        y,target,names,jac=grids[task["variant"]];train=np.asarray(task["train"],int);test=np.asarray(task["test"],int)
        center,scale=np.asarray(split["center"]),np.asarray(split["scale"])
        if (not np.isfinite(scale).all() or np.any(scale<=0) or not np.allclose(center,y[train].mean(0),rtol=1e-12,atol=1e-12)
                or not np.allclose(scale,y[train].std(0),rtol=1e-12,atol=1e-12)):
            raise ValueError("Original training transformation changed")
        bound={"task":task,"manifest_sha256":manifest_hash,"split_sha256":digest(split_path),"completion_sha256":digest(complete_path)}
        folder=output/"tasks"/f"task_{task['task_index']:06d}";folder.mkdir(parents=True,exist_ok=True)
        result_path=folder/"result.json";fit_path=folder/"fit.json"
        if result_path.exists():
            result=json.loads(result_path.read_text())
            if any(result.get(k)!=v for k,v in bound.items()):raise ValueError("Completed reference task binding changed")
            if result["row"]["evaluation_available"] and digest(folder/"predictions.npz")!=result["predictions_sha256"]:
                raise ValueError("Saved reference predictions changed")
            records.append(result["row"]);continue
        row={**task,"method":reference.METHOD,"conditional_only":True,"failed":True,"fit_available":False,
             "evaluation_available":False,"n_train":len(train),"n_test":len(test),"d":y.shape[1],
             "target":names[target],"mean_conditional_log_density":None,"score_converged":None,"boundary_hit":None}
        params=None;fit_record=None;prediction_hash=None
        try:
            if fit_path.exists():
                checkpoint=json.loads(fit_path.read_text())
                if any(checkpoint.get(k)!=v for k,v in bound.items()):raise IntegrityError("Reference fit checkpoint identity changed")
                params,fit_record=checkpoint["parameters"],checkpoint["record"]
            else:
                with threadpool_limits(1):fitted=reference.fit((y[train]-center)/scale,target)
                params,fit_record=fitted.params,fitted.record
                atomic_json(fit_path,{**bound,"parameters":params,"record":fit_record,
                    "training_transform":{"center":center,"scale":scale}})
            row.update(fit_available=True,**{k:fit_record[k] for k in ("score_converged","projected_gradient","boundary_hit",
                "residual_floor_active","rank_deficient","design_rank","design_condition","parameter_count","nominal_parameter_count","fit_seconds")})
            mean,density=reference.original_scale_predictions(params,y[test],center,scale,jac[test])
            if not np.isfinite(mean).all() or not np.isfinite(density).all():raise FloatingPointError("Nonfinite reference prediction")
            np.savez_compressed(folder/"predictions.npz",indices=test,conditional_log_mean=mean,conditional_log_density=density)
            prediction_hash=digest(folder/"predictions.npz")
            row.update(failed=False,evaluation_available=True,mean_conditional_log_density=float(density.mean()))
        except IntegrityError:
            raise
        except Exception as exc:
            row.update(failed=True,evaluation_available=False,exception=repr(exc))
        result={**bound,"status":"complete","row":row,"parameters":params,"record":fit_record,
                "training_transform":{"center":center,"scale":scale},"predictions_sha256":prediction_hash}
        atomic_json(result_path,result);records.append(clean(row))
    frame=pd.DataFrame(records);frame.to_csv(output/"fit_records.csv",index=False)
    repetitions=[]
    for (variant,repeat),group in frame[frame.kind.eq("cv")].groupby(["variant","repeat"]):
        available=group[group.evaluation_available]
        repetitions.append({"variant":variant,"repeat":repeat,"folds_planned":len(group),"folds_available":len(available),
            "complete":len(group)==spec["config"]["folds"] and len(available)==len(group),
            "conditional_log_density":float(np.average(available.mean_conditional_log_density,weights=available.n_test))
               if len(group)==spec["config"]["folds"] and len(available)==len(group) else None})
    pd.DataFrame(repetitions).to_csv(output/"cv_repetitions.csv",index=False)
    stability=[]
    B=spec["config"]["bootstrap_repetitions"]
    for variant,(y,target,names,_) in grids.items():
        curves=np.full((B,len(y)),np.nan);full=np.full(len(y),np.nan)
        group=frame[frame.variant.eq(variant)&frame.kind.isin(["full","bootstrap"])]
        for record in group.to_dict("records"):
            if not record["evaluation_available"]:continue
            a=np.load(output/"tasks"/f"task_{record['task_index']:06d}"/"predictions.npz")
            if not np.array_equal(a["indices"],np.arange(len(y))):raise ValueError("Reference stability grid is not all original rows")
            if record["kind"]=="full":full=a["conditional_log_mean"]
            else:curves[int(record["repeat"])]=a["conditional_log_mean"]
        np.savez_compressed(output/(variant+"_predictions.npz"),bootstrap=curves,full=full,analysis_grid=y,target_index=target)
        stability.append({"variant":variant,"method":reference.METHOD,**summarize_weighted_method(curves,full,np.full(len(y),1/len(y)),B)})
    pd.DataFrame(stability).to_csv(output/"stability_summary.csv",index=False)
    summary={"expected_tasks":expected_tasks,"recorded_tasks":len(frame),"evaluation_available":int(frame.evaluation_available.sum()),
        "failed":int(frame.failed.sum()),"score_passed":int(frame.score_converged.eq(True).sum()),
        "rank_deficient":int(frame.get("rank_deficient",pd.Series(dtype=bool)).eq(True).sum()),
        "residual_floor_active":int(frame.get("residual_floor_active",pd.Series(dtype=bool)).eq(True).sum()),
        "recorded_fit_seconds":float(frame.get("fit_seconds",pd.Series(dtype=float)).sum()),
        "manifest_sha256":manifest_hash,"completed_utc":datetime.now(timezone.utc).isoformat()}
    atomic_json(output/"aggregate.json",summary)
    return summary


def compare_stability(reference_run,mixture_stability,output):
    reference_run,mixture_stability,output=map(Path,(reference_run,mixture_stability,output))
    output.mkdir(parents=True,exist_ok=False);rows=[];hashes={}
    for variant in VARIANTS:
        lp=mixture_stability/(variant+"_predictions.npz");rp=reference_run/(variant+"_predictions.npz")
        hashes[str(lp)]=digest(lp);hashes[str(rp)]=digest(rp)
        left,right=np.load(lp),np.load(rp)
        if not np.array_equal(left["analysis_grid"],right["analysis_grid"]):raise ValueError("Mixture and reference stability grids differ")
        weights=np.full(len(left["analysis_grid"]),1/len(left["analysis_grid"]))
        for method in ("shared_ld","full_ld","gaussian_shared_correlation"):
            rows.append({"variant":variant,"left_method":method,"right_method":reference.METHOD,
                **summarize_weighted_pair(left[method+"_bootstrap"],right["bootstrap"],left[method+"_full"],right["full"],weights,len(right["bootstrap"]))})
    pd.DataFrame(rows).to_csv(output/"paired_stability.csv",index=False)
    atomic_json(output/"manifest.json",{"input_sha256":hashes,"source_sha256":digest(__file__),"refitting":False})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary",type=Path);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--protocol",type=Path,default=PROTOCOL)
    parser.add_argument("--reference-run",type=Path);parser.add_argument("--mixture-stability",type=Path)
    args=parser.parse_args()
    if args.reference_run:
        if not args.mixture_stability:parser.error("Reference comparison needs --mixture-stability")
        compare_stability(args.reference_run,args.mixture_stability,args.output)
    else:
        if not args.primary:parser.error("Reference fitting needs --primary")
        print(json.dumps(run(args.primary,args.output,args.protocol)))


if __name__=="__main__":main()
