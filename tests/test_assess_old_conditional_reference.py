import sys,json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import assess_old_conditional_reference as study


@pytest.fixture
def old_primary(tmp_path,monkeypatch):
    monkeypatch.setattr(study,"VARIANTS",("crabs_log5",))
    primary=tmp_path/"primary";primary.mkdir();data=primary/"executed_sources/data/crabs.csv";data.parent.mkdir(parents=True)
    rng=np.random.default_rng(31);y=rng.normal(size=(12,2));raw=np.exp(y+[2,3]);y=np.log(raw)
    pd.DataFrame(raw,columns=["FL","BD"]).to_csv(data,index=False)
    variant={"id":"crabs_log5","primary":True,"dataset":"crabs","transform":"log","columns":["FL","BD"],"conditional_target":"BD","k":2}
    tasks=[]
    definitions=[("cv",0,0,list(range(6)),list(range(6,12))),("cv",0,1,list(range(6,12)),list(range(6))),
                 ("full",0,0,list(range(12)),list(range(12))),
                 ("bootstrap",0,0,[0,1,2,3,4,5,6,7,1,3,6,8],list(range(12))),
                 ("bootstrap",1,0,[0,1,2,3,4,5,6,7,2,5,9,10],list(range(12))),
                 ("bootstrap",2,0,[0,1,2,3,4,5,6,7,0,4,10,11],list(range(12))),
                 ("leave_area_out",0,0,list(range(8)),list(range(8,12)))]
    for i,(kind,repeat,fold,train,test) in enumerate(definitions):
        tasks.append({"task_index":i,"variant":"crabs_log5","bounds":"primary","kind":kind,"repeat":repeat,"fold":fold,"train":train,"test":test})
    spec={"config":{"variants":[variant],"folds":2,"bootstrap_repetitions":3},"source_hashes":{},
          "data_hashes":{"data/crabs.csv":study.digest(data)},"tasks":tasks}
    study.atomic_json(primary/"specification.json",spec);sha=study.digest(primary/"specification.json")
    for task in tasks:
        folder=primary/"tasks"/f"{task['task_index']:05d}";folder.mkdir(parents=True)
        train=y[task["train"]]
        study.atomic_json(folder/"split.json",{**task,"center":train.mean(0),"scale":train.std(0)})
        study.atomic_json(folder/"completion.json",{"task":task,"specification_sha256":sha,"records":[]})
    protocol=tmp_path/"protocol.rst";protocol.write_text("Synthetic before-fit protocol\n")
    return primary,protocol,study.digest(protocol)


def test_all_task_kinds_run_and_resume_without_refitting(old_primary,tmp_path,monkeypatch):
    primary,protocol,sha=old_primary;output=tmp_path/"reference"
    first=study.run(primary,output,protocol,sha,7)
    assert first["expected_tasks"]==first["evaluation_available"]==7 and first["failed"]==0
    rows=pd.read_csv(output/"fit_records.csv")
    assert rows.conditional_only.all() and "mean_log_density" not in rows and "ARI" not in rows
    assert pd.read_csv(output/"cv_repetitions.csv").complete.all()
    stability=pd.read_csv(output/"stability_summary.csv").iloc[0]
    assert stability.bootstrap_available==3 and stability.all_draws_available
    def reject(*args,**kwargs):raise AssertionError("Unexpected refit")
    monkeypatch.setattr(study.reference,"fit",reject)
    second=study.run(primary,output,protocol,sha,7)
    assert second["evaluation_available"]==7


def test_all_fit_failures_keep_every_denominator(old_primary,tmp_path,monkeypatch):
    primary,protocol,sha=old_primary
    def reject(*args,**kwargs):raise RuntimeError("Injected fit failure")
    monkeypatch.setattr(study.reference,"fit",reject)
    out=tmp_path/"failed";result=study.run(primary,out,protocol,sha,7)
    assert result["failed"]==7 and result["evaluation_available"]==0
    assert pd.read_csv(out/"fit_records.csv").shape[0]==7
    summary=pd.read_csv(out/"stability_summary.csv").iloc[0]
    assert summary.bootstrap_available==0 and summary.bootstrap_requested==3


def test_postfit_evaluation_failure_retains_parameters_and_failed_flag(old_primary,tmp_path,monkeypatch):
    primary,protocol,sha=old_primary
    def reject(*args,**kwargs):raise RuntimeError("Injected storage/evaluation failure")
    monkeypatch.setattr(study.reference,"original_scale_predictions",reject)
    out=tmp_path/"evaluation_failed";result=study.run(primary,out,protocol,sha,7)
    saved=json.loads((out/"tasks/task_000000/result.json").read_text())
    assert saved["parameters"] is not None and saved["row"]["fit_available"]
    assert saved["row"]["failed"] and not saved["row"]["evaluation_available"]
    assert result["failed"]==7


def test_prediction_corruption_and_changed_protocol_are_rejected(old_primary,tmp_path):
    primary,protocol,sha=old_primary;out=tmp_path/"reference";study.run(primary,out,protocol,sha,7)
    path=out/"tasks/task_000000/predictions.npz";path.write_bytes(path.read_bytes()+b"changed")
    with pytest.raises(ValueError,match="predictions changed"):study.run(primary,out,protocol,sha,7)
    protocol.write_text("Changed later protocol")
    with pytest.raises(ValueError,match="protocol hash"):study.run(primary,tmp_path/"other",protocol,sha,7)


def test_original_positive_unit_conditional_score_is_independently_reconstructed(old_primary,tmp_path):
    primary,protocol,sha=old_primary;out=tmp_path/"reference";study.run(primary,out,protocol,sha,7)
    spec=json.loads((primary/"specification.json").read_text());variant=spec["config"]["variants"][0]
    y,t,_,jac=study.load_grid(primary,spec,variant);task=spec["tasks"][0]
    saved=json.loads((out/"tasks/task_000000/result.json").read_text());p=saved["parameters"]
    center=np.array(saved["training_transform"]["center"]);scale=np.array(saved["training_transform"]["scale"])
    z=(y[task["test"]]-center)/scale
    expected=study.reference.conditional_log_density(p,z)-np.log(scale[t])-y[task["test"],t]
    actual=np.load(out/"tasks/task_000000/predictions.npz")
    np.testing.assert_allclose(actual["conditional_log_density"],expected)
