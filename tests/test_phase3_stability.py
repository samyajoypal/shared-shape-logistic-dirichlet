import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import phase3_stability as study
import assess_conditional_prediction_stability as previous
from test_phase3_applications import frozen, run, read, write, runner


def test_grid_weights_are_row_or_group_averages():
    groups=np.array(["a","a","a","b"])
    np.testing.assert_array_equal(study.grid_weights(groups),np.full(4,.25))
    np.testing.assert_allclose(study.grid_weights(groups,True),[1/6,1/6,1/6,1/2])


def test_weighted_delete_one_matches_direct_calculation():
    rng=np.random.default_rng(80);x=rng.normal(size=(11,7));w=np.arange(1,8,dtype=float);w/=w.sum()
    v,leave=study.weighted_variance_and_delete_one(x,w)
    assert v==pytest.approx(np.var(x,axis=0,ddof=1)@w)
    expected=[np.var(np.delete(x,j,axis=0),axis=0,ddof=1)@w for j in range(len(x))]
    np.testing.assert_allclose(leave,expected,rtol=3e-15)


def test_equal_row_weights_reproduce_frozen_original_implementation():
    rng=np.random.default_rng(18);x=rng.normal(size=(20,8));y=rng.normal(size=(20,8));f=rng.normal(size=8)
    w=np.full(8,1/8)
    left=study.summarize_weighted_pair(x,y,f,f+.2,w,20)
    right=previous.summarize_pair(x,y,f,f+.2,20)
    for key in right:
        if right[key] is not None:assert left[key]==pytest.approx(right[key],rel=1e-12,abs=1e-14)
    left=study.summarize_weighted_method(x,f,w,20);right=previous.summarize_method(x,f,20)
    for key in right:
        if right[key] is not None:assert left[key]==pytest.approx(right[key],rel=1e-12,abs=1e-14)


def test_equal_group_weighting_is_invariant_to_replicating_one_groups_grid_rows():
    rng=np.random.default_rng(9);x=rng.normal(size=(20,4));groups=np.array(["a","a","b","b"])
    ids=np.array([0,1,0,1,0,1,2,3]);expanded=x[:,ids]
    v,j=study.weighted_variance_and_delete_one(x,study.grid_weights(groups,True))
    vr,jr=study.weighted_variance_and_delete_one(expanded,study.grid_weights(groups[ids],True))
    assert vr==pytest.approx(v);np.testing.assert_allclose(jr,j)
    # The primary observation-average estimand intentionally gives the enlarged group more weight.
    assert not np.isclose(vr,study.weighted_variance_and_delete_one(expanded,study.grid_weights(groups[ids]))[0])


def test_weighted_paired_mcse_and_shift_identity_keep_common_draws():
    rng=np.random.default_rng(22);right=rng.normal(size=(30,5));left=.5*right+np.arange(5)
    w=np.array([.1,.2,.3,.15,.25]);lf=np.arange(5,dtype=float);rf=np.zeros(5)
    result=study.summarize_weighted_pair(left,right,lf,rf,w,30)
    assert result["variance_ratio"]==pytest.approx(.25)
    assert result["log_variance_ratio_mcse"]<1e-13
    assert result["full_prediction_mean_difference"]==pytest.approx(lf@w)
    m=study.summarize_weighted_method(left,lf,w,30)
    assert m["mean_squared_deviation_from_full"]==pytest.approx(29/30*m["variance"]+m["mean_squared_bootstrap_shift"])
    left[1]=np.nan;right[2]=np.nan
    result=study.summarize_weighted_pair(left,right,lf,rf,w,30)
    assert result["both_available"]==28 and result["both_unavailable_or_unmatched"]==2
    assert not result["all_draws_available"]


def test_group_bootstrap_preserves_duplicated_groups_in_original_row_order():
    groups=["b","a","b","c","a"]
    result=study.verify_group_bootstrap(groups,["a","b","a"],[1,4,0,2,1,4])
    assert result=={"original_groups":3,"groups_drawn":3,"distinct_groups_drawn":2,"bootstrap_rows":6}
    for drawn,indices in [(["a","a"],[1,4,1,4]),(["a","b","a"],[1,4,0,2]),(["a","b","a"],[4,1,0,2,1,4]),(["a","b","z"],[1,4,0,2])]:
        with pytest.raises(ValueError):study.verify_group_bootstrap(groups,drawn,indices)


@pytest.mark.parametrize("weights",[[0,1],[.2,.2],[-.1,1.1],[np.nan,1]])
def test_nonpositive_or_unnormalized_weights_are_rejected(weights):
    with pytest.raises(ValueError):study.weighted_variance_and_delete_one(np.ones((4,2)),weights)


def test_inadequate_or_constant_bootstrap_has_no_invented_ratio_uncertainty():
    x=np.ones((2,3));w=np.full(3,1/3)
    result=study.summarize_weighted_pair(x,x,x[0],x[0],w,2)
    assert result["variance_left"]==0 and result["variance_ratio"] is None
    assert np.isnan(result["variance_difference_mcse"])


def complete_stability_tasks(output,manifest):
    for task in manifest["tasks"]:
        if task["primary"] and task["kind"] in ("full","bootstrap"):
            run(output,task["task_index"])


def test_saved_panel_reconstructs_original_grids_four_methods_and_group_draws(frozen,tmp_path):
    output,manifest=frozen
    complete_stability_tasks(output,manifest)
    panel=study.SavedPanel(output)
    for name in manifest["datasets"]:
        y,groups,row_ids,metadata=panel.grid(name)
        full,boots=panel.tasks(name)
        assert len(boots)==2 and len(y)==24 and len(set(groups))==6
        for task in [full]+boots:
            for method in study.METHODS:
                params,center,scale,record=panel.parameters(task,method)
                predictions=study.conditional_mean(params,method,y,metadata["target_index"],center,scale)
                changed=y.copy();changed[:,metadata["target_index"]]=np.nan
                np.testing.assert_array_equal(predictions,
                    study.conditional_mean(params,method,changed,metadata["target_index"],center,scale))
                assert np.isfinite(predictions).all()
    destination=tmp_path/"stability"
    result=study.run_analysis(output,destination)
    assert result["expected_predictions"]==result["recorded_predictions"]==result["available_predictions"]==24
    table=__import__("pandas").read_csv(destination/"stability_summary.csv")
    assert len(table)==16 and set(table.bootstrap_requested)=={2}
    assert (table.loc[table.subset=="all_finite","bootstrap_available"]==2).all()
    for name in manifest["datasets"]:
        arrays=np.load(destination/(name+"_predictions.npz"))
        np.testing.assert_array_equal(arrays["row_ids"],manifest["datasets"][name]["row_ids"])
        assert arrays["conditional_log_linear_bootstrap"].shape==(2,24)


def test_missing_fits_retain_every_bootstrap_and_method_denominator(frozen,tmp_path):
    output,manifest=frozen
    result=study.run_analysis(output,tmp_path/"missing")
    assert result["available_predictions"]==0 and result["recorded_predictions"]==24
    table=__import__("pandas").read_csv(tmp_path/"missing/stability_summary.csv")
    assert (table.bootstrap_requested==2).all() and (table.bootstrap_unavailable==2).all()
    assert table.variance.isna().all()


def test_changed_selected_point_and_compact_metadata_rejected(frozen):
    output,manifest=frozen
    task=next(t for t in manifest["tasks"] if t["primary"] and t["kind"]=="full")
    run(output,task["task_index"])
    path=output/"tasks"/f"task_{task['task_index']:06d}"/"shared_ld.json"
    value=read(path);original=__import__("copy").deepcopy(value)
    value["parameters"]["theta"][0]+=.1;write(path,value)
    with pytest.raises(study.IntegrityError,match="parameter point"):
        study.SavedPanel(output).parameters(task,"shared_ld")
    value=original;value["dataset_metadata"]["dataset_metadata_sha256"]="0"*64;write(path,value)
    with pytest.raises(study.IntegrityError,match="grid or training"):
        study.SavedPanel(output).parameters(task,"shared_ld")


def test_archived_source_and_data_changes_rejected(frozen):
    output,manifest=frozen
    relative=next(iter(manifest["sources_sha256"]))
    path=output/"source_snapshot"/relative
    path.write_bytes(path.read_bytes()+b"\n# changed\n")
    with pytest.raises(study.IntegrityError,match="Input hash mismatch"):
        study.SavedPanel(output)


def test_stability_outputs_cannot_overwrite_frozen_run(frozen):
    output,_=frozen
    with pytest.raises(ValueError,match="separate"):
        study.run_analysis(output,output/"analysis")


@pytest.mark.parametrize("comparison",["primary","wide","kminus1","kplus1"])
def test_explicit_cv_loader_preserves_training_transform_and_sensitivity_scope(frozen,comparison):
    output,manifest=frozen
    task=next(t for t in manifest["tasks"] if t["kind"]=="cv" and t["comparison"]==comparison)
    run(output,task["task_index"])
    panel=study.SavedPanel(output)
    with pytest.raises(ValueError,match="allow_cv"):
        panel.parameters(task,"shared_ld")
    params,center,scale,record=panel.parameters(task,"shared_ld",allow_cv=True)
    case=runner.load_case(output,manifest,task)
    np.testing.assert_array_equal(panel.grid(task["dataset_id"])[0],case["y"])
    np.testing.assert_array_equal(center,case["center"])
    np.testing.assert_array_equal(scale,case["scale"])
    assert params["k"]==task["k"]
    if comparison=="primary":
        panel.parameters(task,"conditional_log_linear",allow_cv=True)
    else:
        with pytest.raises(study.IntegrityError,match="not declared"):
            panel.parameters(task,"conditional_log_linear",allow_cv=True)


def test_saved_panel_float_parsing_matches_frozen_preparer(tmp_path):
    # Python's explicit float converter and pandas' default parser can differ
    # by a bit; the frozen runner uses the explicit converter.
    raw=b"row_id,group,label,A,B\n1,g,,0.12345678901234567,1.0000000000000002\n"
    path=tmp_path/"source_snapshot/data/test.csv";path.parent.mkdir(parents=True);path.write_bytes(raw)
    metadata={"positive_csv":"data/test.csv","raw_data_sha256":study.digest(path),
        "feature_columns":["A","B"],"transform":"log","analysis_coordinate_names":["A","B"],
        "target_name":"B","target_index":1,"n_rows":1,"n_groups":1,"row_ids":["1"],"group_ids":["g"]}
    panel=object.__new__(study.SavedPanel);panel.run=tmp_path;panel.hashes={};panel.cache={};panel.grids={}
    panel.manifest={"datasets":{"test":metadata}}
    expected=np.log([[float("0.12345678901234567"),float("1.0000000000000002")]])
    np.testing.assert_array_equal(panel.grid("test")[0],expected)
