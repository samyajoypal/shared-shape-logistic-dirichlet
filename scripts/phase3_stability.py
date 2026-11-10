"""Read-only conditional log-mean stability for the frozen phase-3 panel.

Grid weighting changes aggregation only, never a fitted prediction or its
bootstrap sample. No refitting, component matching, or population-MSE claim.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import scipy

from assess_conditional_prediction_stability import (
    jackknife_mcse, ld_conditional_mean, gaussian_conditional_mean,
    clean, semantic_hash,
)
import conditional_log_linear as conditional_reference

ROOT=Path(__file__).resolve().parents[1]
METHODS=("shared_ld","full_ld","gaussian_shared_correlation",conditional_reference.METHOD)
GROUP_SENSITIVITY=("hydrochem_cations","gemas_major6")


def grid_weights(groups, equal_groups=False):
    """Return fixed row weights, summing to one, on the original data grid."""
    groups = np.asarray(groups)
    if groups.ndim != 1 or len(groups) == 0:
        raise ValueError("Need a nonempty one-dimensional group vector")
    if any(value is None or str(value).strip() == "" for value in groups.tolist()):
        raise ValueError("Group identifiers must be nonempty")
    # Group IDs are strings in the frozen CSV. Avoid dependence on their order.
    _, inverse, counts = np.unique(groups.astype(str), return_inverse=True, return_counts=True)
    if equal_groups:
        return 1.0/(len(counts)*counts[inverse])
    return np.full(len(groups), 1.0/len(groups))


def checked_weights(weights, n):
    weights = np.asarray(weights, float)
    if weights.shape != (n,) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Every original grid row must have a positive finite weight")
    if not np.isclose(weights.sum(), 1., rtol=1e-12, atol=1e-14):
        raise ValueError("Grid weights must sum to one")
    return weights


def weighted_variance_and_delete_one(predictions, weights):
    """Weighted pointwise sample variance and delete-one-bootstrap values."""
    predictions = np.asarray(predictions, float)
    if predictions.ndim != 2 or len(predictions) < 2 or not np.isfinite(predictions).all():
        raise ValueError("Need at least two finite curves on the fixed grid")
    weights = checked_weights(weights, predictions.shape[1])
    B = len(predictions)
    centered = predictions-predictions.mean(axis=0)
    squared_norm = centered**2 @ weights
    value = float(squared_norm.sum()/(B-1))
    leave = ((B-1)*value-B/(B-1)*squared_norm)/(B-2) if B > 2 else np.full(B, np.nan)
    if np.any(leave < -1e-12*(1+value)):
        raise ArithmeticError("Negative delete-one variance beyond rounding error")
    return value, np.maximum(leave, 0.)


def summarize_weighted_method(predictions, full_prediction, weights, requested):
    predictions, full_prediction = np.asarray(predictions, float), np.asarray(full_prediction, float)
    if predictions.ndim != 2 or len(predictions) != requested or full_prediction.shape != predictions.shape[1:]:
        raise ValueError("Predictions must retain every declared bootstrap index and grid row")
    weights = checked_weights(weights, predictions.shape[1])
    valid = np.isfinite(predictions).all(axis=1)
    draws = predictions[valid]
    result = {"bootstrap_requested":requested,"bootstrap_available":int(valid.sum()),
              "bootstrap_unavailable":int(requested-valid.sum()),
              "all_draws_available":bool(valid.sum() == requested),"grid_rows":len(weights),
              "variance":None,"variance_mcse":None,"mean_squared_bootstrap_shift":None,
              "mean_squared_deviation_from_full":None,
              "full_prediction_available":bool(np.isfinite(full_prediction).all())}
    if len(draws) >= 2:
        value, leave = weighted_variance_and_delete_one(draws, weights)
        result.update(variance=value,variance_mcse=jackknife_mcse(leave))
        if result["full_prediction_available"]:
            shift = float((draws.mean(axis=0)-full_prediction)**2 @ weights)
            deviation = float(np.mean((draws-full_prediction[None, :])**2 @ weights))
            np.testing.assert_allclose(deviation,(len(draws)-1)/len(draws)*value+shift,rtol=1e-11,atol=1e-13)
            result.update(mean_squared_bootstrap_shift=shift,mean_squared_deviation_from_full=deviation)
    return result


def summarize_weighted_pair(left, right, left_full, right_full, weights, requested):
    left,right = np.asarray(left, float),np.asarray(right, float)
    if left.shape != right.shape or left.ndim != 2 or len(left) != requested:
        raise ValueError("Paired methods must retain identical bootstrap indices and grid rows")
    weights = checked_weights(weights,left.shape[1])
    valid = np.isfinite(left).all(axis=1)&np.isfinite(right).all(axis=1)
    L,R = left[valid],right[valid]
    row = {"bootstrap_requested":requested,"both_available":int(valid.sum()),
           "both_unavailable_or_unmatched":int(requested-valid.sum()),
           "all_draws_available":bool(valid.sum()==requested),
           "variance_left":None,"variance_right":None,"variance_difference":None,
           "variance_difference_mcse":None,"variance_ratio":None,"log_variance_ratio_mcse":None,
           "ratio_mc95_low":None,"ratio_mc95_high":None,"variance_reduction_percent":None,
           "full_prediction_mean_difference":None,"full_prediction_rms_difference":None}
    if len(L)>=2:
        vl,jl = weighted_variance_and_delete_one(L,weights)
        vr,jr = weighted_variance_and_delete_one(R,weights)
        row.update(variance_left=vl,variance_right=vr,variance_difference=vl-vr,
                   variance_difference_mcse=jackknife_mcse(jl-jr))
        if vl>0 and vr>0:
            ratio=vl/vr
            se=jackknife_mcse(np.log(jl)-np.log(jr)) if np.all(jl>0) and np.all(jr>0) else np.nan
            row.update(variance_ratio=ratio,log_variance_ratio_mcse=se,
                ratio_mc95_low=np.exp(np.log(ratio)-1.96*se),ratio_mc95_high=np.exp(np.log(ratio)+1.96*se),
                variance_reduction_percent=100*(1-ratio))
    lf,rf=np.asarray(left_full,float),np.asarray(right_full,float)
    if lf.shape != weights.shape or rf.shape != weights.shape:
        raise ValueError("Full-data predictions must use the same original grid")
    if np.isfinite(lf).all() and np.isfinite(rf).all():
        delta=lf-rf
        row.update(full_prediction_mean_difference=float(delta@weights),
                   full_prediction_rms_difference=float(np.sqrt(delta**2@weights)))
    return row


def verify_group_bootstrap(groups, drawn_groups, train_indices):
    """Verify whole-group sampling with exact multiplicity and original order."""
    groups=np.asarray(groups).astype(str)
    levels=np.unique(groups)
    drawn=np.asarray(drawn_groups).astype(str)
    if drawn.ndim != 1 or len(drawn)!=len(levels) or not np.isin(drawn,levels).all():
        raise ValueError("Each bootstrap must draw G known original groups with replacement")
    expected=np.concatenate([np.flatnonzero(groups==group) for group in drawn])
    actual=np.asarray(train_indices)
    if actual.dtype.kind not in "iu" or not np.array_equal(actual,expected):
        raise ValueError("Bootstrap row indices do not preserve every sampled group and multiplicity")
    return {"original_groups":len(levels),"groups_drawn":len(drawn),
            "distinct_groups_drawn":len(np.unique(drawn)),"bootstrap_rows":len(expected)}


class IntegrityError(RuntimeError):
    pass


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    Path(path).write_text(json.dumps(clean(value),indent=2,allow_nan=False)+"\n")


def conditional_mean(params,method,y,target,center,scale):
    """Return the same original analysis-coordinate conditional mean."""
    y=np.asarray(y,float);center=np.asarray(center,float);scale=np.asarray(scale,float)
    if (y.ndim!=2 or center.shape!=y.shape[1:] or scale.shape!=center.shape
        or not np.isfinite(center).all() or not np.isfinite(scale).all() or np.any(scale<=0)):
        raise ValueError("Invalid saved training transformation")
    z=(y-center)/scale
    if method in ("shared_ld","full_ld"):
        value=ld_conditional_mean(params["theta"],z,int(params["k"]),int(params["d"]),target,method=="shared_ld")
    elif method=="gaussian_shared_correlation":
        value=gaussian_conditional_mean(params["theta"],z,int(params["k"]),int(params["d"]),target)
    elif method==conditional_reference.METHOD:
        if int(params["target"])!=target:raise IntegrityError("Conditional-reference target changed")
        value=conditional_reference.conditional_mean(params,z)
    else:raise ValueError("Undeclared stability method")
    return center[target]+scale[target]*value


class SavedPanel:
    """Independent read-only verification of the phase-3 saved-data contract."""
    def __init__(self,run):
        self.run=Path(run).resolve();self.hashes={};self.cache={};self.grids={}
        self.manifest=self.read(self.run/"manifest.json");self.manifest_sha256=self.hashes[str(self.run/"manifest.json")]
        m=self.manifest
        if semantic_hash(m["config"])!=m["config_sha256"] or semantic_hash(m["tasks"])!=m["task_plan_sha256"]:
            raise IntegrityError("Frozen configuration or task plan hash differs")
        if semantic_hash(self.read(self.run/"source_snapshot/configs/phase3_applications.json"))!=m["config_sha256"]:
            raise IntegrityError("Archived phase-3 configuration differs")
        for group in ("sources_sha256","data_sha256"):
            for relative,expected in m[group].items():self.bytes(self.run/"source_snapshot"/relative,expected)
        for i,task in enumerate(m["tasks"]):
            if task["task_index"]!=i or task["split_file"]!=f"splits/task_{i:06d}.json":
                raise IntegrityError("Noncanonical phase-3 task map")
        for name,metadata in m["datasets"].items():
            if self.read(self.run/"datasets"/(name+".json"))!=metadata:
                raise IntegrityError("Saved dataset metadata differs from manifest")

    def bytes(self,path,expected=None):
        path=Path(path).resolve();raw=path.read_bytes();sha=hashlib.sha256(raw).hexdigest()
        if expected is not None and sha!=expected:raise IntegrityError("Input hash mismatch: "+str(path))
        if str(path) in self.hashes and self.hashes[str(path)]!=sha:raise IntegrityError("Input changed while reading: "+str(path))
        self.hashes[str(path)]=sha
        return raw

    def read(self,path):
        path=Path(path).resolve()
        if str(path) not in self.cache:self.cache[str(path)]=json.loads(self.bytes(path))
        return self.cache[str(path)]

    def grid(self,name):
        if name in self.grids:return self.grids[name]
        m=self.manifest["datasets"][name]
        frame=pd.read_csv(io.BytesIO(self.bytes(self.run/"source_snapshot"/m["positive_csv"],m["raw_data_sha256"])),
            keep_default_na=False,dtype={"row_id":str,"group":str,"label":str},
            converters={name:float for name in m["feature_columns"]})
        if list(frame.columns)!=["row_id","group","label"]+m["feature_columns"]:
            raise IntegrityError("Positive measurement CSV column order differs")
        raw=frame[m["feature_columns"]].to_numpy(float)
        if not np.isfinite(raw).all() or np.any(raw<=0):raise IntegrityError("Prepared measurements are not positive and finite")
        if frame.row_id.duplicated().any() or frame.row_id.eq("").any() or frame.group.eq("").any():
            raise IntegrityError("Missing/duplicated row identifier or missing group")
        if m["transform"]=="alr":
            ref=m["feature_columns"].index(m["reference_column"]);keep=[j for j in range(raw.shape[1]) if j!=ref]
            y=np.log(raw[:,keep])-np.log(raw[:,ref,None]);names=[m["feature_columns"][j]+"/"+m["reference_column"] for j in keep]
        elif m["transform"]=="log":y=np.log(raw);names=m["feature_columns"]
        else:raise IntegrityError("Undeclared measurement transformation")
        if (names!=m["analysis_coordinate_names"] or names.index(m["target_name"])!=m["target_index"]
            or len(frame)!=m["n_rows"] or frame.group.nunique()!=m["n_groups"]
            or frame.row_id.tolist()!=m["row_ids"] or frame.group.tolist()!=m["group_ids"]):
            raise IntegrityError("Original analysis-grid metadata differs")
        self.grids[name]=(y,frame.group.to_numpy(str),frame.row_id.to_numpy(str),m)
        return self.grids[name]

    def tasks(self,name):
        tasks=[t for t in self.manifest["tasks"] if t["dataset_id"]==name and t["primary"]
               and t["comparison"]=="primary" and t["kind"] in ("full","bootstrap")]
        full=[t for t in tasks if t["kind"]=="full"]
        boots=sorted((t for t in tasks if t["kind"]=="bootstrap"),key=lambda t:t["repeat"])
        B=self.manifest["config"]["bootstrap_repetitions"]
        if len(full)!=1 or [t["repeat"] for t in boots]!=list(range(B)):
            raise IntegrityError("Missing or duplicate declared primary full/bootstrap task")
        for task in tasks:
            if set(task["conditional_methods"])!={conditional_reference.METHOD} or not set(METHODS[:3])<=set(task["methods"]):
                raise IntegrityError("Primary stability methods differ from frozen plan")
        return full[0],boots

    def parameters(self,task,method,allow_cv=False):
        """Verify one selected fit; CV access must be explicitly requested.

        The default retains the stability contract: full-data/bootstrap fits
        evaluated on all original rows. The read-only mean-loss assessor may
        opt into declared CV tasks, including their fixed sensitivity cells.
        """
        index=task.get("task_index")
        if type(index) is not int or not 0<=index<len(self.manifest["tasks"]) or task!=self.manifest["tasks"][index]:
            raise IntegrityError("Task differs from canonical frozen plan")
        if task["kind"] not in ("full","bootstrap") and not (allow_cv and task["kind"]=="cv"):
            raise ValueError("CV parameters require explicit allow_cv=True")
        if method not in METHODS or method not in task["methods"]+task["conditional_methods"]:
            raise IntegrityError("Method is not declared for this task")
        y,groups,_,metadata=self.grid(task["dataset_id"])
        path=self.run/task["split_file"];self.bytes(path,task["split_sha256"]);split=self.read(path)
        if split["task_index"]!=task["task_index"]:raise IntegrityError("Split identity differs")
        train=np.asarray(split["train"]);test=np.asarray(split["test"])
        if any(a.ndim!=1 or not len(a) or a.dtype.kind not in "iu" or np.any(a<0) or np.any(a>=len(y)) for a in (train,test)):
            raise IntegrityError("Invalid saved training indices")
        if task["kind"]=="bootstrap":
            try:verify_group_bootstrap(groups,split["bootstrap_draw_groups"],train)
            except ValueError as exc:raise IntegrityError(str(exc)) from exc
            expected=np.random.default_rng(split["bootstrap_draw_seed"]).choice(np.unique(groups),size=len(np.unique(groups)),replace=True)
            if not np.array_equal(expected,split["bootstrap_draw_groups"]):raise IntegrityError("Bootstrap group stream differs from frozen seed")
        elif task["kind"]=="full":
            if not np.array_equal(train,np.arange(len(y))):raise IntegrityError("Full fit does not use every original row")
        else:
            if (len(np.unique(train))!=len(train) or len(np.unique(test))!=len(test)
                or set(train)&set(test) or set(train)|set(test)!=set(range(len(y)))
                or set(groups[train])&set(groups[test])):
                raise IntegrityError("Frozen CV split is not a complete disjoint group partition")
            # Independently replay the frozen seeded greedy group-fold rule.
            levels,counts=np.unique(groups,return_counts=True)
            folds=int(self.manifest["config"]["folds"])
            rng=np.random.default_rng(split["group_split_seed"]);loads=np.zeros(folds,int);assignment={}
            for j in rng.permutation(len(levels)):
                fold=int(rng.choice(np.flatnonzero(loads==loads.min())))
                assignment[levels[j]]=fold;loads[fold]+=counts[j]
            expected=np.array([assignment[g] for g in groups])
            if (not np.array_equal(train,np.flatnonzero(expected!=task["fold"]))
                or not np.array_equal(test,np.flatnonzero(expected==task["fold"]))):
                raise IntegrityError("CV rows differ from frozen group-split seed")
        if task["kind"]!="cv" and not np.array_equal(test,np.arange(len(y))):
            raise IntegrityError("Stability task grid differs from original rows")
        center,scale=y[train].mean(0),y[train].std(0)
        folder=self.run/"tasks"/f"task_{task['task_index']:06d}"
        completion=self.read(folder/"completion.json");binding=self.read(folder/"input_binding.json")
        expected_binding={"task":task,"split_sha256":task["split_sha256"],"positive_csv_sha256":metadata["raw_data_sha256"],
            "training_transform":{"center":center.tolist(),"scale":scale.tolist()},
            "training_checkpoint_sha256":binding["training_checkpoint_sha256"]}
        if binding!=expected_binding:raise IntegrityError("Bound original training inputs differ")
        training_path=folder/"training.json";self.bytes(training_path,binding["training_checkpoint_sha256"])
        training=self.read(training_path);base_binding={k:v for k,v in binding.items() if k!="training_checkpoint_sha256"}
        if (training["task"]!=task or training["manifest_sha256"]!=self.manifest_sha256
            or training["input_binding_sha256"]!=semantic_hash(base_binding)
            or training["training_transform"]!=binding["training_transform"]):
            raise IntegrityError("Training checkpoint identity or transformation differs")
        identity={"task":task,"manifest_sha256":self.manifest_sha256,"input_binding_sha256":semantic_hash(binding)}
        if completion.get("status")!="complete" or any(completion.get(k)!=v for k,v in identity.items()):
            raise IntegrityError("Task completion identity differs")
        value=self.read(folder/(method+".json"));stage={**identity,"method":method}
        if any(value.get(k)!=v for k,v in stage.items()):raise IntegrityError("Selected method identity differs")
        expected_rows=completion["conditional_rows"] if method==conditional_reference.METHOD else completion["rows"]
        listed=[r for r in expected_rows if r["method"]==method]
        if len(listed)!=1 or listed[0]!=value["row"]:raise IntegrityError("Result row differs from completed task")
        metadata_file="datasets/"+task["dataset_id"]+".json"
        compact_metadata={**{k:v for k,v in metadata.items() if k not in ("row_ids","group_ids")},
            "dataset_metadata_file":metadata_file,"dataset_metadata_sha256":self.hashes[str((self.run/metadata_file).resolve())]}
        if value["training_transform"]!=binding["training_transform"] or value["dataset_metadata"]!=compact_metadata:
            raise IntegrityError("Selected result grid or training transformation differs")
        selected_path=folder/(method+"_fit.json");self.bytes(selected_path,value["selected_checkpoint_sha256"])
        selected=self.read(selected_path)
        if any(selected.get(k)!=v for k,v in stage.items()):raise IntegrityError("Selected fit checkpoint identity differs")
        if selected.get("parameters")!=value.get("parameters") or selected.get("record")!=value.get("record"):
            raise IntegrityError("Selected parameter point differs from evaluated result")
        if selected.get("initial_fit_sha256"):
            initial_path=folder/(method+"_initial_fit.json");self.bytes(initial_path,selected["initial_fit_sha256"])
            initial=self.read(initial_path)
            if any(initial.get(k)!=v for k,v in stage.items()):raise IntegrityError("Initial fit checkpoint identity differs")
        params=value.get("parameters")
        if params is None:raise ValueError("No finite saved selected parameter point")
        if (params["method"]!=method or int(params["d"])!=y.shape[1]
            or method!=conditional_reference.METHOD and int(params["k"])!=task["k"]):
            raise IntegrityError("Saved parameter family/dimension/component count differs")
        return params,center,scale,value["row"]


def run_analysis(run,output):
    panel=SavedPanel(run);output=Path(output).resolve()
    if output==panel.run or panel.run in output.parents:raise ValueError("Stability outputs must be separate from frozen fitting inputs")
    output.mkdir(parents=True,exist_ok=False)
    source_names=["scripts/phase3_stability.py","scripts/assess_conditional_prediction_stability.py","scripts/conditional_log_linear.py"]
    metadata={"created_utc":datetime.now(timezone.utc).isoformat(),"fitting_run":str(panel.run),
        "fitting_manifest_sha256":panel.manifest_sha256,"refitting":False,"methods":list(METHODS),
        "grid_weighting":{"primary":"1/N on every original row","sensitivity":"1/(G*n_g) for Hydrochem and GEMAS; aggregation only"},
        "uncertainty":"Paired delete-one-bootstrap Monte Carlo error; not population sampling uncertainty or efficiency",
        "source_sha256":{p:digest(ROOT/p) for p in source_names},
        "versions":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"pandas":pd.__version__}}
    write_json(output/"analysis_plan.json",metadata)
    for relative in source_names:
        dest=output/"source_snapshot"/relative;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/relative).read_bytes())
    diagnostics=[];summaries=[];pairs=[];grid_records=[]
    for name in panel.manifest["datasets"]:
        y,groups,row_ids,data=panel.grid(name);full_task,bootstrap=panel.tasks(name);B=len(bootstrap)
        curves={m:np.full((B,len(y)),np.nan) for m in METHODS};full={m:np.full(len(y),np.nan) for m in METHODS}
        flags={m:{"score":np.zeros(B,bool),"boundary":np.zeros(B,bool)} for m in METHODS}
        for task in [full_task]+bootstrap:
            for method in METHODS:
                diagnostic={"dataset_id":name,"task_index":task["task_index"],"kind":task["kind"],"repeat":task["repeat"],
                    "method":method,"prediction_available":False,"source_failed":None,"score_converged":None,"boundary_hit":None,"reason":None}
                try:
                    params,center,scale,record=panel.parameters(task,method)
                    values=conditional_mean(params,method,y,data["target_index"],center,scale)
                    if not np.isfinite(values).all():raise FloatingPointError("Nonfinite original-grid conditional mean")
                    if task["kind"]=="full":full[method]=values
                    else:
                        b=task["repeat"];curves[method][b]=values
                        flags[method]["score"][b]=record.get("score_converged") is True
                        flags[method]["boundary"][b]=record.get("boundary_hit") is True
                    diagnostic.update(prediction_available=True,source_failed=record.get("failed"),
                        score_converged=record.get("score_converged"),boundary_hit=record.get("boundary_hit"))
                except IntegrityError:raise
                except (FileNotFoundError,ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:
                    diagnostic["reason"]=type(exc).__name__+": "+str(exc)
                diagnostics.append(diagnostic)
        finite=np.logical_and.reduce([np.isfinite(curves[m]).all(axis=1) for m in METHODS])
        common_score=finite&np.logical_and.reduce([flags[m]["score"] for m in METHODS])
        weightings={"equal_observations":grid_weights(groups)}
        if name in GROUP_SENSITIVITY:weightings["equal_groups"]=grid_weights(groups,True)
        for weighting,weights in weightings.items():
            for subset,mask in {"all_finite":np.ones(B,bool),"common_score_eligible":common_score}.items():
                for method in METHODS:
                    selected=curves[method].copy();selected[~mask]=np.nan
                    summaries.append({"dataset_id":name,"method":method,"weighting":weighting,"subset":subset,
                        "target":data["target_name"],"target_scale":"original analysis-coordinate log/ALR mean",
                        "score_passed":int(flags[method]["score"].sum()),"boundary_hits":int(flags[method]["boundary"].sum()),
                        **summarize_weighted_method(selected,full[method],weights,B)})
                for other in METHODS[1:]:
                    L,R=curves["shared_ld"].copy(),curves[other].copy();L[~mask]=np.nan;R[~mask]=np.nan
                    pairs.append({"dataset_id":name,"left_method":"shared_ld","right_method":other,"weighting":weighting,"subset":subset,
                        **summarize_weighted_pair(L,R,full["shared_ld"],full[other],weights,B)})
        arrays={"analysis_grid":y,"row_ids":row_ids,"groups":groups,"target_index":data["target_index"],
            "bootstrap_task_indices":np.array([t["task_index"] for t in bootstrap])}
        for method in METHODS:arrays[method+"_bootstrap"]=curves[method];arrays[method+"_full"]=full[method]
        for weighting,weights in weightings.items():arrays[weighting+"_weights"]=weights
        np.savez_compressed(output/(name+"_predictions.npz"),**arrays)
        grid_records.append({"dataset_id":name,"grid_rows":len(y),"original_groups":len(np.unique(groups)),
            "bootstrap_requested":B,"all_methods_finite":int(finite.sum()),"common_score_eligible":int(common_score.sum()),
            "weightings":list(weightings)})
    pd.DataFrame(diagnostics).to_csv(output/"prediction_diagnostics.csv",index=False)
    pd.DataFrame(summaries).to_csv(output/"stability_summary.csv",index=False)
    pd.DataFrame(pairs).to_csv(output/"paired_stability.csv",index=False)
    write_json(output/"grid_and_eligibility.json",grid_records)
    metadata.update(completed_utc=datetime.now(timezone.utc).isoformat(),input_sha256=panel.hashes,
        expected_predictions=len(panel.manifest["datasets"])*len(METHODS)*(panel.manifest["config"]["bootstrap_repetitions"]+1),
        recorded_predictions=len(diagnostics),available_predictions=sum(r["prediction_available"] for r in diagnostics),
        outputs_sha256={p.name:digest(p) for p in output.iterdir() if p.is_file() and p.name!="analysis_plan.json"})
    write_json(output/"analysis_manifest.json",metadata)
    return metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    result=run_analysis(args.run,args.output)
    print(json.dumps({k:result[k] for k in ["expected_predictions","recorded_predictions","available_predictions"]}))


if __name__=="__main__":main()
