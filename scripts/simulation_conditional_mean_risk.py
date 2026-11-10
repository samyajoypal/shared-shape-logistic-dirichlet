"""Read-only, independently resumable conditional-mean risk on saved simulations.

No fitting module is imported. Freeze once; workers use --task-index or
--task-range START:STOP; --aggregate preserves every planned denominator.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import platform
import tempfile
import time

import numpy as np
import pandas as pd
import scipy
from scipy.special import digamma,gammaln,logsumexp
from threadpoolctl import threadpool_limits

from assess_conditional_prediction_stability import clean,semantic_hash,ld_conditional_mean,gaussian_conditional_mean

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/"configs/simulation_conditional_mean_risk.json"
PROTOCOL=ROOT/"docs/public_analysis_rules.json"
RECEIPT=ROOT/"docs/public_risk_receipt.json"
SOURCES=("scripts/simulation_conditional_mean_risk.py","scripts/assess_conditional_prediction_stability.py")
METHODS=("shared_ld","full_ld","gaussian_shared_correlation")


class IntegrityError(RuntimeError):pass


def sha(raw):return hashlib.sha256(raw).hexdigest()
def digest(path):return sha(Path(path).read_bytes())
def read(path):return json.loads(Path(path).read_text())
def versions():return {"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"pandas":pd.__version__}
def array_hash(x):
    x=np.ascontiguousarray(x)
    return sha(str(x.dtype).encode()+str(x.shape).encode()+x.tobytes())


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix="."+path.name,dir=path.parent)
    try:
        with os.fdopen(fd,"w") as f:
            json.dump(clean(value),f,indent=2,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def atomic_npz(path,arrays):
    path=Path(path);fd,tmp=tempfile.mkstemp(prefix="."+path.name,dir=path.parent)
    try:
        with os.fdopen(fd,"wb") as f:
            np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def primary_tasks(config):
    tasks=[]
    for cell in config["cells"]:
        for replicate in range(cell["repetitions"]):
            tasks.append({"task_index":len(tasks),"cell_id":cell["id"],"replicate":replicate})
    return tasks


def validate_scope(config,primary,gaussian):
    source=primary["configuration"];tasks=primary_tasks(source)
    cells=[c for c in source["cells"] if c["block"] in config["blocks"]]
    keep={c["id"] for c in cells}
    expected=[{"task_index":i,"primary_task_index":t["task_index"],"cell_id":t["cell_id"],"replicate":t["replicate"]}
              for i,t in enumerate(t for t in tasks if t["cell_id"] in keep)]
    if (config["cells"]!=cells or config["tasks"]!=expected or len(cells)!=config["cell_count"]
        or len(expected)!=config["dataset_count"] or any(c["family"]!="ld" for c in cells)
        or config["target_index"]!=0 or config["methods"]!=list(METHODS) or config["n_test"]!=source["n_test"]
        or config["primary_configuration_sha256"]!=primary["configuration_sha256"]
        or semantic_hash(source)!=primary["configuration_sha256"] or config["refitting"] is not False):
        raise IntegrityError("Risk scope differs from its frozen upstream design")
    if gaussian["source_kind"]!="simulations":raise IntegrityError("Gaussian run is not a simulation companion")
    mapping={t["primary_task_index"]:t for t in gaussian["tasks"]}
    if len(mapping)!=len(gaussian["tasks"]) or any(t["task_index"]!=i for i,t in enumerate(gaussian["tasks"])):
        raise IntegrityError("Noncanonical Gaussian task mapping")
    for t in expected:
        g=mapping.get(t["primary_task_index"])
        if g is None or g["cell_id"]!=t["cell_id"] or g["replicate"]!=t["replicate"]:
            raise IntegrityError("Gaussian task scope differs")
    return expected,mapping


def freeze(primary,gaussian,output,config_path=CONFIG,protocol=PROTOCOL,receipt=RECEIPT):
    primary,gaussian,output=map(lambda p:Path(p).resolve(),(primary,gaussian,output))
    if any(output==p or p in output.parents for p in (primary,gaussian)):
        raise ValueError("Reporting output must be separate from frozen fits")
    config=read(config_path);record=read(receipt);p=read(primary/"manifest.json");g=read(gaussian/"manifest.json")
    if (digest(config_path)!=record["config_sha256"] or digest(protocol)!=config["protocol_sha256"]
        or record["protocol_sha256"]!=config["protocol_sha256"]
        or record["new_conditional_mean_risk_outcomes_inspected"] is not False or record["new_model_fits"] is not False
        or digest(primary/"manifest.json")!=config["primary_manifest_sha256"]
        or digest(gaussian/"manifest.json")!=config["gaussian_manifest_sha256"]
        or g["primary_metadata_sha256"]!=config["primary_manifest_sha256"]):
        raise IntegrityError("Protocol, receipt or upstream manifest hash differs")
    tasks,_=validate_scope(config,p,g)
    verified={}
    for root,m in ((primary,p),(gaussian,g)):
        for relative,expected in m["source_sha256"].items():
            path=root/"source_snapshot"/relative
            if digest(path)!=expected:raise IntegrityError("Upstream archived source changed: "+str(path))
            verified[str(path)]=expected
    output.mkdir(parents=True,exist_ok=False)
    for name in SOURCES:
        dest=output/"source_snapshot"/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/name).read_bytes())
    for src,name in ((Path(config_path),"config.json"),(Path(protocol),"protocol.rst"),(Path(receipt),"design_receipt.json"),
                     (primary/"manifest.json","primary_manifest.json"),(gaussian/"manifest.json","gaussian_manifest.json")):
        (output/name).write_bytes(src.read_bytes())
    manifest={"created_utc":datetime.now(timezone.utc).isoformat(),"config":config,"config_sha256":semantic_hash(config),
        "primary":str(primary),"gaussian":str(gaussian),"tasks":tasks,"tasks_sha256":semantic_hash(tasks),
        "source_sha256":{name:digest(ROOT/name) for name in SOURCES},"upstream_source_sha256":verified,
        "protocol_sha256":digest(protocol),"receipt_sha256":digest(receipt),"runtime":versions(),"refitting":False}
    atomic_json(output/"manifest.json",manifest)
    return manifest


def oracle_conditional_mean(spec,y,target=0):
    """Independent gamma/LD marginal derivation, in generating coordinates."""
    y=np.asarray(y,float);alpha=np.asarray(spec["alpha"],float);mu=np.asarray(spec["locations"],float)
    scale=np.asarray(spec["scales"],float);weights=np.asarray(spec["weights"],float)
    k,p=alpha.shape;d=p-1;keep=[m for m in range(d) if m!=target];shape_keep=keep+[d]
    if (spec["family"]!="ld" or y.ndim!=2 or y.shape[1]!=d or mu.shape!=(k,d) or scale.shape!=(k,d)
        or weights.shape!=(k,) or np.any(alpha<=0) or np.any(scale<=0) or np.any(weights<=0)
        or not np.isclose(weights.sum(),1) or not all(np.isfinite(x).all() for x in (alpha,mu,scale,weights,y[:,keep]))):
        raise ValueError("Invalid LD generating parameters or conditioning coordinates")
    z=(y[:,None,keep]-mu[None,:,keep])/scale[None,:,keep]
    c=logsumexp(np.concatenate([np.zeros((len(y),k,1)),z],axis=2),axis=2)
    other=alpha[:,shape_keep].sum(1)
    marginal=(gammaln(other)-gammaln(alpha[:,shape_keep]).sum(1)-np.log(scale[:,keep]).sum(1))[None,:]
    marginal=marginal+(alpha[None,:,keep]*z).sum(2)-other[None,:]*c
    logw=np.log(weights)[None,:]+marginal;posterior=np.exp(logw-logsumexp(logw,axis=1)[:,None])
    component=mu[None,:,target]+scale[None,:,target]*(c+digamma(alpha[:,target])[None,:]-digamma(other)[None,:])
    return (posterior*component).sum(1)


def fitted_mean(params,method,y,target,center,scale):
    y=np.asarray(y,float);center=np.asarray(center,float);scale=np.asarray(scale,float)
    if params["method"]!=method or int(params["d"])!=y.shape[1]:raise IntegrityError("Fitted method/dimension differs")
    z=(y-center)/scale;z[:,target]=np.nan
    if method in METHODS[:2]:
        mean=ld_conditional_mean(params["theta"],z,int(params["k"]),int(params["d"]),target,method=="shared_ld")
    else:mean=gaussian_conditional_mean(params["theta"],z,int(params["k"]),int(params["d"]),target)
    return center[target]+scale[target]*mean


class Reporter:
    def __init__(self,output,primary=None,gaussian=None):
        self.output=Path(output).resolve();self.manifest=read(self.output/"manifest.json");m=self.manifest
        self.manifest_sha=digest(self.output/"manifest.json");self.config=m["config"]
        self.primary=Path(primary or m["primary"]).resolve();self.gaussian=Path(gaussian or m["gaussian"]).resolve()
        if (versions()!=m["runtime"] or semantic_hash(self.config)!=m["config_sha256"]
            or read(self.output/"config.json")!=self.config or semantic_hash(m["tasks"])!=m["tasks_sha256"]
            or digest(self.output/"protocol.rst")!=m["protocol_sha256"] or digest(self.output/"design_receipt.json")!=m["receipt_sha256"]):
            raise IntegrityError("Frozen reporter runtime or design differs")
        for name,expected in m["source_sha256"].items():
            if digest(ROOT/name)!=expected or digest(self.output/"source_snapshot"/name)!=expected:
                raise IntegrityError("Frozen reporting source differs: "+name)
        for root,name,key in ((self.primary,"primary_manifest.json","primary_manifest_sha256"),
                              (self.gaussian,"gaussian_manifest.json","gaussian_manifest_sha256")):
            if digest(root/"manifest.json")!=self.config[key] or digest(self.output/name)!=self.config[key]:
                raise IntegrityError("Frozen upstream manifest differs")
        self.pmanifest=read(self.output/"primary_manifest.json");self.gmanifest=read(self.output/"gaussian_manifest.json")
        for root,upstream_manifest in ((self.primary,self.pmanifest),(self.gaussian,self.gmanifest)):
            for relative,expected in upstream_manifest["source_sha256"].items():
                if digest(root/"source_snapshot"/relative)!=expected:raise IntegrityError("Archived upstream source changed")
        tasks,self.gmap=validate_scope(self.config,self.pmanifest,self.gmanifest)
        if tasks!=self.manifest["tasks"]:raise IntegrityError("Frozen reporting task mapping differs")
        self.cells={c["id"]:c for c in self.config["cells"]}

    def input_path(self,path):
        """Resolve an optional relocation of the same byte-identical frozen run."""
        path=Path(path)
        for new in (self.primary,self.gaussian):
            if path==new or new in path.parents:return path
        for name,new in (("primary",self.primary),("gaussian",self.gaussian)):
            old=Path(self.manifest[name])
            if path==old or old in path.parents:return new/path.relative_to(old)
        raise IntegrityError("Recorded consumed input lies outside both frozen runs")

    def load_case(self,task):
        hashes={};self.last_input_hashes=hashes;cache={}
        def raw(path,expected=None):
            path=Path(path).resolve()
            if str(path) in cache:
                if expected is not None and hashes[str(path)]!=expected:raise IntegrityError("Inconsistent bound input hash")
                return cache[str(path)]
            before=path.stat();b=path.read_bytes();after=path.stat();h=sha(b)
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise IntegrityError("Input changed during read")
            if expected is not None and h!=expected:raise IntegrityError("Bound upstream input changed: "+str(path))
            hashes[str(path)]=h;cache[str(path)]=b;return b
        def js(path,expected=None):return json.loads(raw(path,expected))
        gtask=self.gmap[task["primary_task_index"]];gdir=self.gaussian/"tasks"/f"task_{gtask['task_index']:06d}"
        gresult=None;binding=None;fallback=False
        if (gdir/"input_binding.json").exists():
            binding=js(gdir/"input_binding.json")
            if binding["primary_task_index"]!=task["primary_task_index"]:raise IntegrityError("Gaussian input task differs")
            directory=(self.primary/binding["primary_directory"]).resolve()
            expected_parent=self.primary/"tasks"/f"task_{task['primary_task_index']:06d}"
            if directory.parent!=expected_parent or not directory.name.startswith("attempt_"):
                raise IntegrityError("Gaussian binding names another primary attempt")
            for relative,expected in binding["input_sha256"].items():raw(self.primary/relative,expected)
            if (gdir/"result.json").exists():
                gresult=js(gdir/"result.json")
                if (gresult["status"]!="complete" or gresult["task"]!=gtask
                    or gresult["manifest_sha256"]!=self.config["gaussian_manifest_sha256"]
                    or gresult["input_binding_sha256"]!=semantic_hash(binding)):
                    raise IntegrityError("Gaussian saved result identity differs")
                selected_path=gdir/"control_fit.json"
                if selected_path.exists():
                    selected=js(selected_path)
                    expected={"task_index":gtask["task_index"],"primary_task_index":task["primary_task_index"],
                        "manifest_sha256":self.config["gaussian_manifest_sha256"],"input_binding_sha256":semantic_hash(binding)}
                    if any(selected.get(k)!=v for k,v in expected.items()):raise IntegrityError("Gaussian selected checkpoint identity differs")
                    if gresult.get("parameters") is not None:
                        if selected.get("failed") or selected.get("parameters")!=gresult["parameters"]:
                            raise IntegrityError("Gaussian result differs from selected parameter point")
                        for key in ("parameter_count","score_converged","projected_gradient","boundary_hit"):
                            if selected["record"].get(key)!=gresult["row"].get(key):raise IntegrityError("Gaussian selected numerical flag differs")
        else:
            fallback=True
            attempts=sorted((self.primary/"tasks"/f"task_{task['primary_task_index']:06d}").glob("attempt_*"))
            completed=[p for p in attempts if (p/"completion.json").exists()]
            if not completed:raise FileNotFoundError("No complete frozen primary attempt")
            directory=completed[-1]
        identity=js(directory/"task.json");completion=js(directory/"completion.json");cell=self.cells[task["cell_id"]]
        for item in (identity,completion):
            if any(item[k]!=task[k] for k in ("cell_id","replicate")) or item["task_index"]!=task["primary_task_index"]:
                raise IntegrityError("Primary task/attempt identity differs")
        if (identity["configuration_sha256"]!=self.config["primary_configuration_sha256"] or identity["cell"]!=cell
            or identity["attempt"]!=completion["attempt"] or directory.name!=f"attempt_{identity['attempt']:03d}"):
            raise IntegrityError("Primary generator or attempt differs")
        data=np.load(io.BytesIO(raw(directory/"data.npz")))
        train,test,center,scale=(np.asarray(data[k]) for k in ("train","test","center","scale"))
        generation=js(directory/"generation.json")
        if (test.shape!=(self.config["n_test"],cell["dimension"]) or train.shape!=(cell["n_train"],cell["dimension"])
            or any(array_hash(x)!=generation["array_sha256"][name] for x,name in ((train,"train"),(test,"test")))
            or not np.isfinite(train).all() or not np.isfinite(test).all()
            or not np.array_equal(center,train.mean(0)) or not np.array_equal(scale,train.std(0)) or np.any(scale<=0)):
            raise IntegrityError("Saved arrays or training-only transformation differ")
        required=[directory/name for name in ("task.json","completion.json","data.npz","generation.json","primary__shared_ld.json","primary__full_ld.json")]
        if binding is not None and any(str(p.relative_to(self.primary)) not in binding["input_sha256"] for p in required):
            raise IntegrityError("Gaussian binding omits a required primary input")
        fits={}
        for method in METHODS[:2]:
            path=directory/("primary__"+method+".json")
            if not path.exists():fits[method]=(None,{"source_missing":True});continue
            value=js(path);row=value["row"]
            if (row["task_index"]!=task["primary_task_index"] or row["cell_id"]!=task["cell_id"]
                or row["replicate"]!=task["replicate"] or row["attempt"]!=identity["attempt"]
                or row["method"]!=method or row["variant"]!="primary"):
                raise IntegrityError("Saved LD fit identity differs")
            if gresult is not None and gresult["baseline_records"].get(method)!=row:
                raise IntegrityError("Gaussian-bound LD record differs")
            fits[method]=(value.get("parameters"),row)
        if gresult is None:fits[METHODS[2]]=(None,{"source_missing":True})
        else:
            row=gresult["row"]
            if row["task_index"]!=gtask["task_index"] or row["method"]!=METHODS[2]:raise IntegrityError("Gaussian record differs")
            fits[METHODS[2]]=(gresult.get("parameters"),row)
        return {"test":test,"center":center,"scale":scale,"truth":generation["parameters"],"fits":fits,"input_sha256":hashes,
                "primary_directory":str(directory),"primary_attempt":identity["attempt"],"attempt_fallback":fallback}

    def validate_result(self,path,task):
        value=read(path)
        if value["task"]!=task or value["manifest_sha256"]!=self.manifest_sha or value["status"]!="complete":
            raise IntegrityError("Completed reporting result identity differs")
        if [r["method"] for r in value["rows"]]!=list(METHODS):raise IntegrityError("Completed method scope differs")
        for row in value["rows"]:
            if any(row.get(k)!=v for k,v in task.items()) or row["n_test"]!=self.config["n_test"]:
                raise IntegrityError("Completed method row task differs")
            if row["available"]:
                if row["risk"] is None or not np.isfinite(row["risk"]) or row["risk"]<0:
                    raise IntegrityError("Available prediction has an invalid risk")
            elif row["risk"] is not None:raise IntegrityError("Unavailable prediction has a risk")
        for name,h in value["input_sha256"].items():
            if digest(self.input_path(name))!=h:raise IntegrityError("Consumed input changed on resume")
        if value.get("prediction_sha256") is not None:
            raw=(path.parent/"predictions.npz").read_bytes()
            if sha(raw)!=value["prediction_sha256"]:raise IntegrityError("Saved risk predictions changed")
            arrays=np.load(io.BytesIO(raw));oracle=arrays["true_conditional_mean"]
            if oracle.shape!=(self.config["n_test"],) or not np.isfinite(oracle).all():raise IntegrityError("Saved oracle grid changed")
            for row in value["rows"]:
                if row["available"]:
                    predicted=arrays[row["method"]];loss=(predicted-oracle)**2
                    if (predicted.shape!=oracle.shape or not np.isfinite(loss).all()
                        or not np.isclose(loss.mean(),row["risk"],rtol=1e-12,atol=1e-15)
                        or not np.isclose(loss.std(ddof=1)/np.sqrt(len(loss)),row["test_integration_mcse"],rtol=1e-12,atol=1e-15)):
                        raise IntegrityError("Recorded risk differs from saved prediction arrays")
        elif any(r["available"] for r in value["rows"]):raise IntegrityError("Available risk lacks saved predictions")
        return value

    def task(self,index):
        if type(index) is not int or not 0<=index<len(self.manifest["tasks"]):raise ValueError("Invalid reporting task index")
        task=self.manifest["tasks"][index];folder=self.output/"tasks"/f"task_{index:06d}";folder.mkdir(parents=True,exist_ok=True)
        with (folder/".lock").open("a") as lock,threadpool_limits(1):
            fcntl.flock(lock,fcntl.LOCK_EX)
            path=folder/"result.json"
            if path.exists():return self.validate_result(path,task)
            started=time.perf_counter();cell=self.cells[task["cell_id"]];case=None;setup_reason=None;arrays={};rows=[];self.last_input_hashes={}
            try:
                case=self.load_case(task);masked=case["test"].copy();masked[:,self.config["target_index"]]=np.nan
                oracle=oracle_conditional_mean(case["truth"],masked,self.config["target_index"])
                if not np.isfinite(oracle).all():raise FloatingPointError("Nonfinite exact generating conditional mean")
                arrays["true_conditional_mean"]=oracle
            except IntegrityError:raise
            except (FileNotFoundError,ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:
                setup_reason=type(exc).__name__+": "+str(exc)
            for method in METHODS:
                row={**task,"method":method,"n_test":self.config["n_test"],"available":False,"risk":None,
                    "test_integration_mcse":None,"source_failed":None,"score_converged":None,"boundary_hit":None,
                    "projected_gradient":None,"reason":setup_reason}
                try:
                    if setup_reason:raise ValueError(setup_reason)
                    params,record=case["fits"][method]
                    row.update(source_failed=record.get("failed"),score_converged=record.get("score_converged"),
                        boundary_hit=record.get("boundary_hit"),projected_gradient=record.get("projected_gradient"))
                    if params is None:raise ValueError("No saved finite parameter point")
                    if int(params["k"])!=cell["components"]:raise IntegrityError("Saved component count differs")
                    prediction=fitted_mean(params,method,masked,self.config["target_index"],case["center"],case["scale"])
                    with np.errstate(over="ignore",invalid="ignore"):loss=(prediction-oracle)**2
                    if prediction.shape!=oracle.shape or not np.isfinite(loss).all():raise FloatingPointError("Nonfinite conditional mean or risk")
                    row.update(available=True,risk=float(loss.mean()),test_integration_mcse=float(loss.std(ddof=1)/np.sqrt(len(loss))),reason=None)
                    arrays[method]=prediction
                except IntegrityError:raise
                except (ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:row["reason"]=type(exc).__name__+": "+str(exc)
                rows.append(row)
            prediction_hash=None
            if arrays:
                atomic_npz(folder/"predictions.npz",arrays);prediction_hash=digest(folder/"predictions.npz")
            result={"task":task,"manifest_sha256":self.manifest_sha,"status":"complete","rows":rows,
                "input_sha256":case["input_sha256"] if case else self.last_input_hashes,"prediction_sha256":prediction_hash,
                "primary_directory":case["primary_directory"] if case else None,
                "primary_attempt":case["primary_attempt"] if case else None,"attempt_fallback":case["attempt_fallback"] if case else None,
                "setup_reason":setup_reason,"elapsed_seconds":time.perf_counter()-started,"runtime":versions()}
            atomic_json(path,result);return result


def paired_summary(left,right):
    left,right=np.asarray(left,float),np.asarray(right,float);n=len(left)
    if left.shape!=right.shape or left.ndim!=1 or not np.isfinite(left).all() or not np.isfinite(right).all():raise ValueError("Invalid paired risks")
    out={"paired_available":n,"mean_left":None,"mean_right":None,"difference":None,"difference_mcse":None,"risk_ratio":None,"ratio_mcse":None}
    if n:
        l,r=left.mean(),right.mean();out.update(mean_left=l,mean_right=r,difference=l-r,
            difference_mcse=(left-right).std(ddof=1)/np.sqrt(n) if n>1 else None)
        if r>0:
            ratio=l/r;out.update(risk_ratio=ratio,ratio_mcse=((left-ratio*right)/r).std(ddof=1)/np.sqrt(n) if n>1 else None)
    return out


def aggregate(reporter):
    output=reporter.output;rows=[]
    for path in (output/"tasks").glob("*/result.json"):
        item=read(path);index=item.get("task",{}).get("task_index")
        if type(index) is not int or not 0<=index<len(reporter.manifest["tasks"]) or path.parent.name!=f"task_{index:06d}":
            raise IntegrityError("Noncanonical reporting result directory")
    for task in reporter.manifest["tasks"]:
        path=output/"tasks"/f"task_{task['task_index']:06d}"/"result.json"
        if path.exists():rows.extend([{**r,"recorded":True} for r in reporter.validate_result(path,task)["rows"]])
        else:rows.extend([{**task,"method":method,"recorded":False,"available":False,"risk":None,
            "source_failed":None,"score_converged":None,"boundary_hit":None,"test_integration_mcse":None} for method in METHODS])
        if (task["task_index"]+1)%100==0:print(json.dumps({"aggregation_tasks_verified":task["task_index"]+1}),flush=True)
    frame=pd.DataFrame(rows);summaries=[];pairs=[]
    for cell,group in frame.groupby("cell_id",sort=False):
        spec=reporter.cells[cell];planned=spec["repetitions"]
        common=set(group.loc[group.available,"replicate"])
        for method in METHODS:
            good=group[(group.method==method)&group.available&group.score_converged.eq(True)]
            common&=set(good.replicate)
        for subset in reporter.config["subsets"]:
            selected=group if subset=="all_finite" else group[group.replicate.isin(common)]
            for method in METHODS:
                source=group[group.method==method];good=selected[(selected.method==method)&selected.available]
                summaries.append({"cell_id":cell,"block":spec["block"],"method":method,"subset":subset,"planned":planned,
                    "recorded":int(source.recorded.sum()),"available":len(good),"unavailable_or_ineligible":planned-len(good),
                    "source_failures":int(source.source_failed.eq(True).sum()),"score_passed":int(source.score_converged.eq(True).sum()),
                    "boundary_hits":int(source.boundary_hit.eq(True).sum()),"mean_risk":good.risk.mean() if len(good) else None,
                    "risk_mcse":good.risk.std(ddof=1)/np.sqrt(len(good)) if len(good)>1 else None,
                    "mean_test_integration_mcse":good.test_integration_mcse.mean() if len(good) else None})
            for left,right in itertools.combinations(METHODS,2):
                l=selected[(selected.method==left)&selected.available];r=selected[(selected.method==right)&selected.available]
                merged=l.merge(r,on="replicate",suffixes=("_left","_right"),validate="one_to_one")
                pairs.append({"cell_id":cell,"block":spec["block"],"left_method":left,"right_method":right,"subset":subset,
                    "planned":planned,"paired_unavailable_or_ineligible":planned-len(merged),
                    **paired_summary(merged.risk_left.to_numpy(float),merged.risk_right.to_numpy(float))})
    for name,table in (("method_records",frame),("risk_summary",pd.DataFrame(summaries)),("paired_risk_summary",pd.DataFrame(pairs))):
        table.to_csv(output/(name+".csv"),index=False)
    status={"planned_tasks":len(reporter.manifest["tasks"]),"planned_predictions":len(frame),"recorded_predictions":int(frame.recorded.sum()),
        "available_predictions":int(frame.available.sum()),"manifest_sha256":reporter.manifest_sha,"refitting":False,
        "outputs_sha256":{name:digest(output/name) for name in ("method_records.csv","risk_summary.csv","paired_risk_summary.csv")}}
    atomic_json(output/"aggregation.json",status);return status


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--primary",type=Path);p.add_argument("--gaussian",type=Path);p.add_argument("--config",type=Path,default=CONFIG)
    p.add_argument("--freeze",action="store_true");p.add_argument("--aggregate",action="store_true")
    p.add_argument("--task-index",type=int);p.add_argument("--task-range");a=p.parse_args()
    if sum((a.freeze,a.aggregate,a.task_index is not None,a.task_range is not None))!=1:p.error("Choose exactly one operation")
    if a.freeze:
        if not a.primary or not a.gaussian:p.error("Freeze requires both upstream roots")
        m=freeze(a.primary,a.gaussian,a.output,a.config);print(json.dumps({"tasks":len(m["tasks"]),"manifest_sha256":digest(a.output/"manifest.json")}));return
    r=Reporter(a.output,a.primary,a.gaussian)
    if a.aggregate:print(json.dumps(aggregate(r)));return
    indices=[a.task_index] if a.task_index is not None else range(*map(int,a.task_range.split(":")))
    for i in indices:
        result=r.task(i);print(json.dumps({"task_index":i,"available":sum(x["available"] for x in result["rows"])}),flush=True)


if __name__=="__main__":main()
