"""Independent moments, derivatives, fixed-grid targets and paired MC error."""
import numpy as np
import json
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.optimize._numdiff import approx_derivative
from scipy.special import digamma, logsumexp
from scipy.stats import multivariate_normal

import assess_conditional_prediction_stability as study
import run_confirmatory_experiments as baseline


def ld_point(k, d, shared):
    rng = np.random.default_rng(801+k+d)
    shapes = rng.uniform(.7, 3., d+1 if shared else k*(d+1))
    return np.r_[rng.normal(size=k-1)*.3, shapes,
                 rng.uniform(.6, 1.8, k*d), rng.normal(size=k*d)*.4]


def gaussian_point(k, d):
    rng = np.random.default_rng(531+k+d)
    c = 0 if d == 1 else 1 if d == 2 else d
    return np.r_[rng.normal(size=k-1)*.3, rng.normal(size=k*d),
                 rng.normal(size=k*d)*.2, rng.uniform(.2, .7, c)]


@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("d,target", [(1, 0), (2, 1), (5, 2)])
def test_ld_prediction_gradient_and_unused_target(shared, d, target):
    k = 2; theta = ld_point(k, d, shared)
    y = np.random.default_rng(61).normal(size=(7, d))
    value, gradient = study.ld_conditional_mean(theta, y, k, d, target, shared, True)
    numeric = approx_derivative(lambda t: study.ld_conditional_mean(t, y, k, d, target, shared), theta, method="3-point")
    np.testing.assert_allclose(gradient, numeric, rtol=3e-6, atol=3e-8)
    changed = y.copy(); changed[:, target] = np.nan
    np.testing.assert_array_equal(study.ld_conditional_mean(theta, changed, k, d, target, shared), value)


@pytest.mark.parametrize("d,target", [(1, 0), (2, 1), (5, 2)])
def test_gaussian_prediction_gradient_and_unused_target(d, target):
    k = 2; theta = gaussian_point(k, d)
    y = np.random.default_rng(72).normal(size=(7, d))
    value, gradient = study.gaussian_conditional_mean(theta, y, k, d, target, True)
    numeric = approx_derivative(lambda t: study.gaussian_conditional_mean(t, y, k, d, target), theta, method="3-point")
    np.testing.assert_allclose(gradient, numeric, rtol=3e-6, atol=3e-8)
    changed = y.copy(); changed[:, target] = np.nan
    np.testing.assert_array_equal(study.gaussian_conditional_mean(theta, changed, k, d, target), value)


@pytest.mark.parametrize("shared", [True, False])
def test_ld_mixture_mean_matches_numerical_integration_of_original_density(shared):
    k, d, target = 2, 2, 1
    theta = ld_point(k, d, shared)
    method = "shared_ld" if shared else "full_ld"
    y = np.array([[.3, 0.]])
    def density(t):
        point = y.copy(); point[0, target] = t
        return float(np.exp(baseline.ld_evaluate(theta, point, k, method)[0][0]))
    denominator = quad(density, -np.inf, np.inf, epsabs=1e-10)[0]
    numerator = quad(lambda t:t*density(t), -np.inf, np.inf, epsabs=1e-10)[0]
    actual = study.ld_conditional_mean(theta, y, k, d, target, shared)[0]
    assert actual == pytest.approx(numerator/denominator, abs=2e-8)


def test_one_dimensional_ld_reduces_to_unconditional_mixture_mean():
    theta = np.array([.2, .8, 1.4, 1.2, .7, -.3, .6])
    logpi, alpha, a, b = baseline.unpack_ld(theta, "shared_ld", 2, 1)
    expected = np.sum(np.exp(logpi)*(-b[:, 0]/a[:, 0]
        +(digamma(alpha[0, 0])-digamma(alpha[0, 1]))/a[:, 0]))
    actual = study.ld_conditional_mean(theta, np.full((4, 1), np.nan), 2, 1, 0, True)
    np.testing.assert_allclose(actual, expected)


def test_gaussian_mixture_mean_matches_independent_joint_integral():
    theta = gaussian_point(2, 2); y = np.array([[.4, 0.]])
    logpi = np.r_[theta[0], 0.]; logpi -= logsumexp(logpi)
    means = theta[1:5].reshape(2, 2); std = np.exp(theta[5:9].reshape(2, 2))
    R = np.array([[1., theta[-1]], [theta[-1], 1.]])
    covs = std[:, :, None]*R[None, :, :]*std[:, None, :]
    def density(t):
        return np.sum([np.exp(logpi[j]+multivariate_normal.logpdf([.4, t], mean=means[j], cov=covs[j])) for j in range(2)])
    denominator = quad(density, -np.inf, np.inf, epsabs=1e-10)[0]
    numerator = quad(lambda t:t*density(t), -np.inf, np.inf, epsabs=1e-10)[0]
    assert study.gaussian_conditional_mean(theta, y, 2, 2, 1)[0] == pytest.approx(numerator/denominator, abs=2e-9)


@pytest.mark.parametrize("method", ["shared_ld", "full_ld", "gaussian_shared_correlation"])
def test_saved_training_transform_returns_log_target_units(method):
    k, d, target = 2, 2, 1
    theta = gaussian_point(k, d) if method.startswith("gaussian") else ld_point(k, d, method == "shared_ld")
    params = {"theta":theta, "k":k, "d":d}
    z = np.random.default_rng(62).normal(size=(5, d)); center = np.array([2., -1.]); scale = np.array([.3, 2.])
    plain, g = study.predict(params, method, z, target, np.zeros(d), np.ones(d), True)
    transformed, gt = study.predict(params, method, center+scale*z, target, center, scale, True)
    np.testing.assert_allclose(transformed, center[target]+scale[target]*plain)
    np.testing.assert_allclose(gt, scale[target]*g)


def test_delete_one_variance_matches_direct_refits_without_row_independence():
    rng = np.random.default_rng(781)
    predictions = rng.normal(size=(19, 4)) @ rng.normal(size=(4, 17))
    value, leave = study.variance_and_delete_one(predictions)
    assert value == pytest.approx(np.var(predictions, axis=0, ddof=1).mean())
    expected = [np.var(np.delete(predictions, j, axis=0), axis=0, ddof=1).mean() for j in range(19)]
    np.testing.assert_allclose(leave, expected, atol=5e-14)
    # Repeating grid rows changes neither estimate nor Monte Carlo uncertainty.
    doubled, leave2 = study.variance_and_delete_one(np.tile(predictions, (1, 3)))
    assert doubled == pytest.approx(value)
    assert study.jackknife_mcse(leave2) == pytest.approx(study.jackknife_mcse(leave))


def test_paired_variance_ratio_mcse_and_shift_decomposition():
    rng = np.random.default_rng(53); right = rng.normal(size=(31, 8))
    left = .5*right+np.arange(8)[None, :]
    result = study.summarize_pair(left, right, np.arange(8), np.zeros(8), 31)
    assert result["variance_ratio"] == pytest.approx(.25)
    assert result["log_variance_ratio_mcse"] < 1e-13
    assert result["variance_reduction_percent"] == pytest.approx(75.)
    single = study.summarize_method(left, np.arange(8), 31)
    assert single["all_draws_available"]
    assert single["mean_squared_deviation_from_full"] == pytest.approx(
        30/31*single["variance"]+single["mean_squared_bootstrap_shift"])


def test_unavailable_draws_keep_requested_and_paired_denominators():
    left = np.arange(30.).reshape(6, 5); right = left*.8
    left[0] = np.nan; right[1] = np.nan
    single = study.summarize_method(left, np.zeros(5), 6)
    paired = study.summarize_pair(left, right, np.zeros(5), np.zeros(5), 6)
    assert single["bootstrap_available"] == 5 and not single["all_draws_available"]
    assert paired["both_available"] == 4 and not paired["all_draws_available"]
    assert single["bootstrap_requested"] == paired["bootstrap_requested"] == 6


def test_zero_variance_ratio_is_explicitly_unavailable():
    values = np.ones((5, 3))
    row = study.summarize_pair(values, values, values[0], values[0], 5)
    assert row["variance_left"] == row["variance_right"] == 0
    assert row["variance_ratio"] is None


@pytest.mark.parametrize("method", ["shared_ld", "full_ld", "gaussian_shared_correlation"])
def test_conditional_mean_is_invariant_to_component_permutation(method):
    k, d = 2, 3
    y = np.random.default_rng(22).normal(size=(5, d))
    if method.startswith("gaussian"):
        theta = gaussian_point(k, d)
        changed = np.r_[-theta[0],theta[1:1+k*d].reshape(k,d)[::-1].ravel(),
                        theta[1+k*d:1+2*k*d].reshape(k,d)[::-1].ravel(),theta[-d:]]
        function = lambda t:study.gaussian_conditional_mean(t,y,k,d,1)
    else:
        shared = method == "shared_ld"
        theta = ld_point(k,d,shared)
        logpi,alpha,a,b = baseline.unpack_ld(theta,method,k,d)
        changed = np.r_[-theta[0],alpha.ravel() if shared else alpha[::-1].ravel(),a[::-1].ravel(),b[::-1].ravel()]
        function = lambda t:study.ld_conditional_mean(t,y,k,d,1,shared)
    np.testing.assert_allclose(function(theta),function(changed),atol=2e-15)


@pytest.fixture
def saved_application(tmp_path, monkeypatch):
    monkeypatch.setitem(study.PLAN,"targets",{"crabs_log5":"BD"})
    monkeypatch.setitem(study.PLAN,"bootstrap_repetitions",3)
    primary=tmp_path/"primary"; gaussian=tmp_path/"gaussian"
    primary.mkdir(); gaussian.mkdir()
    data=primary/"executed_sources/data/crabs.csv";data.parent.mkdir(parents=True)
    raw=np.exp(np.random.default_rng(91).normal(size=(12,2))*.2+[2.,3.])
    pd.DataFrame(raw,columns=["FL","BD"]).to_csv(data,index=False)
    variant={"id":"crabs_log5","dataset":"crabs","primary":True,"transform":"log", "columns":["FL","BD"],"conditional_target":"BD","k":1}
    tasks=[]
    for j in range(4):
        train=list(range(12)) if j==0 else list(range(8))+list(range(j-1,j+3))
        tasks.append({"task_index":j,"variant":"crabs_log5","bounds":"primary","kind":"full" if j==0 else "bootstrap",
                      "repeat":max(0,j-1),"fold":0,"train":train,"test":list(range(12)),"seed":700+j})
    spec={"config":{"variants":[variant],"bootstrap_repetitions":3},"tasks":tasks,"data_hashes":{"data/crabs.csv":study.digest(data)}}
    study.write_json(primary/"specification.json",spec); spec_hash=study.digest(primary/"specification.json")
    gmapping=[{"task_index":j,"primary_task_index":task["task_index"],"variant":"crabs_log5","kind":task["kind"],"repeat":task["repeat"],"fold":0} for j,task in enumerate(tasks[::-1])]
    gspec={"source_kind":"applications","primary_metadata_sha256":spec_hash,"tasks":gmapping}
    study.write_json(gaussian/"manifest.json",gspec); ghash=study.digest(gaussian/"manifest.json")
    y=np.log(raw)
    for task in tasks:
        j=task["task_index"];folder=primary/"tasks"/f"{j:05d}";folder.mkdir(parents=True)
        train=y[task["train"]];center=train.mean(0);scale=train.std(0)
        study.write_json(folder/"split.json",{**task,"center":center,"scale":scale})
        records=[]
        for method in ["shared_ld","full_ld"]:
            row={**{key:task[key] for key in ["task_index","variant","kind","repeat","fold"]},"method":method,"failed":False,
                 "score_converged":True,"boundary_hit":method=="full_ld" and j==1}
            theta=np.array([.8,1.4,2.,1.1,.9,.1,.2+j*.03])
            params={"method":method,"k":1,"d":2,"theta":theta}
            study.write_json(folder/(method+".json"),{"parameters":params,"record":row,"specification_sha256":spec_hash})
            records.append(row)
        study.write_json(folder/"completion.json",{"task":task,"specification_sha256":spec_hash,"records":records})
        mapped=next(m for m in gmapping if m["primary_task_index"]==j)
        gf=gaussian/"tasks"/f"task_{mapped['task_index']:06d}";gf.mkdir(parents=True)
        split_name=f"tasks/{j:05d}/split.json"
        binding={"primary_task_index":j,"input_sha256":{split_name:study.digest(folder/"split.json")}}
        study.write_json(gf/"input_binding.json",binding)
        row={**mapped,"method":"gaussian_shared_correlation","failed":False,"score_converged":j!=2,"boundary_hit":False}
        params={"method":"gaussian_shared_correlation","k":1,"d":2,"theta":np.array([.2,.1+j*.04,0.,.1,.3])}
        study.write_json(gf/"result.json",{"status":"complete","task":mapped,"manifest_sha256":ghash,
            "input_binding_sha256":study.semantic_hash(binding),"parameters":params,"row":row})
    protocol=tmp_path/"protocol.rst";protocol.write_text("Synthetic frozen test protocol\n")
    return primary,gaussian,protocol


def test_saved_inputs_complete_analysis_and_declared_sensitivities(saved_application,tmp_path):
    primary,gaussian,protocol=saved_application
    out=tmp_path/"analysis"
    manifest=study.run_analysis(primary,gaussian,out,protocol)
    assert manifest["recorded_predictions"]==manifest["available_predictions"]==12
    summary=pd.read_csv(out/"stability_summary.csv")
    main=summary[summary.subset=="all_finite"]
    assert main.bootstrap_available.eq(3).all() and main.all_draws_available.all()
    eligible=json.loads((out/"grid_and_eligibility.json").read_text())[0]
    assert eligible["all_methods_score_eligible"]==2
    assert eligible["all_methods_score_eligible_and_interior"]==1
    assert summary[summary.subset=="all_methods_score_eligible_and_interior"].variance.isna().all()
    paired=pd.read_csv(out/"paired_stability.csv")
    same=paired[(paired.subset=="all_finite")&(paired.right_method=="full_ld")].iloc[0]
    assert same.variance_ratio==pytest.approx(1.)
    assert not manifest["refitting"] and len(manifest["input_sha256"])>10


@pytest.mark.parametrize("change",["specification","split","gaussian_binding","data"])
def test_input_identity_and_scientific_data_changes_are_rejected(saved_application,tmp_path,change):
    primary,gaussian,protocol=saved_application
    if change=="specification":
        path=primary/"specification.json";path.write_text(path.read_text()+"\n")
    elif change=="split":
        path=primary/"tasks/00000/split.json";v=json.loads(path.read_text());v["seed"]+=1;study.write_json(path,v)
    elif change=="gaussian_binding":
        path=gaussian/"tasks/task_000003/input_binding.json";v=json.loads(path.read_text());v["primary_task_index"]=1;study.write_json(path,v)
    else:
        path=primary/"executed_sources/data/crabs.csv";path.write_text(path.read_text()+"\n")
    with pytest.raises(study.IntegrityError):
        study.run_analysis(primary,gaussian,tmp_path/"rejected",protocol)


def test_missing_saved_point_is_reported_without_replacing_draw(saved_application,tmp_path):
    primary,gaussian,protocol=saved_application
    (primary/"tasks/00001/shared_ld.json").unlink()
    out=tmp_path/"incomplete"
    manifest=study.run_analysis(primary,gaussian,out,protocol)
    assert manifest["available_predictions"]==11 and manifest["expected_predictions"]==12
    summary=pd.read_csv(out/"stability_summary.csv")
    row=summary[(summary.method=="shared_ld")&(summary.subset=="all_finite")].iloc[0]
    assert row.bootstrap_requested==3 and row.bootstrap_available==2 and not row.all_draws_available
    diagnostic=pd.read_csv(out/"prediction_diagnostics.csv")
    assert diagnostic.prediction_available.eq(False).sum()==1


@pytest.fixture
def refined_application(saved_application,tmp_path):
    primary,gaussian,protocol=saved_application
    original=study.SavedInputs(primary,gaussian)
    refined=tmp_path/"refined";refined.mkdir()
    tasks=[{"task_index":j,"primary_task_index":t["task_index"],"variant":t["variant"],
            "kind":t["kind"],"bounds":"primary","repeat":t["repeat"],"fold":0,"methods":list(study.METHODS)}
           for j,t in enumerate(original.spec["tasks"][::-1])]
    manifest={"tasks":tasks,"metadata_sha256":{"primary":original.primary_hash,"gaussian":original.gaussian_hash},"sources_sha256":{}}
    study.write_json(refined/"manifest.json",manifest);mh=study.digest(refined/"manifest.json")
    variant=original.variants()[0];y,_,_=original.grid(variant)
    for task in tasks:
        upstream=original.spec["tasks"][task["primary_task_index"]]
        folder=refined/"tasks"/f"task_{task['task_index']:06d}";folder.mkdir(parents=True)
        split=f"tasks/{upstream['task_index']:05d}/split.json"
        binding={"task":task,"files":[{"source":"primary","relative_path":split,"sha256":study.digest(primary/split)}]}
        study.write_json(folder/"input_binding.json",binding)
        identity={"task":task,"manifest_sha256":mh,"input_binding_sha256":study.semantic_hash(binding)}
        rows=[]
        for method in study.METHODS:
            params,center,scale,row=original.parameters(upstream,method,y)
            row={**row,"task_index":task["task_index"]};rows.append(row)
            study.write_json(folder/(method+".json"),{**identity,"method":method,"parameters":params,"row":row,
                "training_transform":{"center":center,"scale":scale}})
        study.write_json(folder/"completion.json",{**identity,"status":"complete","rows":rows})
    return primary,gaussian,protocol,refined


def test_refined_loader_matches_original_parameter_predictions(refined_application,tmp_path):
    primary,gaussian,protocol,refined=refined_application
    out=tmp_path/"refined_analysis"
    m=study.run_analysis(primary,gaussian,out,protocol,refined)
    assert m["procedure"]=="refined" and m["available_predictions"]==12
    original=study.SavedInputs(primary,gaussian);changed=study.SavedInputs(primary,gaussian,refined)
    variant=original.variants()[0];y,target,_=original.grid(variant)
    for task in original.spec["tasks"]:
        for method in study.METHODS:
            left=original.parameters(task,method,y);right=changed.parameters(task,method,y)
            np.testing.assert_array_equal(study.predict(left[0],method,y,target,left[1],left[2]),
                                          study.predict(right[0],method,y,target,right[1],right[2]))


@pytest.mark.parametrize("change",["training_transform","row","binding_task"])
def test_refined_loader_rejects_changed_task_and_training_identity(refined_application,tmp_path,change):
    primary,gaussian,protocol,refined=refined_application
    folder=refined/"tasks/task_000003"
    path=folder/("input_binding.json" if change=="binding_task" else "shared_ld.json")
    v=json.loads(path.read_text())
    if change=="training_transform":v["training_transform"]["center"][0]+=.01
    elif change=="row":v["row"]["score_converged"]=False
    else:v["task"]["primary_task_index"]+=1
    study.write_json(path,v)
    with pytest.raises(study.IntegrityError):
        study.run_analysis(primary,gaussian,tmp_path/"rejected_refined",protocol,refined)
