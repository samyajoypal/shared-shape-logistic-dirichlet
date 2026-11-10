"""Oracle mathematics, frozen saved-fit identities and complete denominators."""
import copy
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.special import gammaln,logsumexp

import simulation_conditional_mean_risk as risk


def truth():
    return {"family":"ld","weights":[.4,.6],"alpha":[[.8,1.7,1.3],[.8,1.7,1.3]],
        "locations":[[-1.,.2],[1.4,-.8]],"scales":[[.7,.9],[1.2,.6]]}


def ld_point(spec,center,scale,shared):
    alpha=np.array(spec["alpha"]);w=np.array(spec["weights"]);s=np.array(spec["scales"]);mu=np.array(spec["locations"])
    return {"method":"shared_ld" if shared else "full_ld","k":len(w),"d":len(center),
        "theta":np.r_[np.log(w[:-1]/w[-1]),alpha[0] if shared else alpha.ravel(),(scale/s).ravel(),((center-mu)/s).ravel()].tolist()}


def test_oracle_agrees_with_independent_numeric_conditional_density_integration():
    spec=truth();spec["alpha"][1]=[2.2,.9,1.8]
    w=np.array(spec["weights"]);alpha=np.array(spec["alpha"]);mu=np.array(spec["locations"]);s=np.array(spec["scales"])
    def joint(t,x):
        z=(np.array([t,x])-mu)/s
        logc=logsumexp(np.column_stack([np.zeros(2),z]),axis=1)
        lp=np.log(w)+gammaln(alpha.sum(1))-gammaln(alpha).sum(1)-np.log(s).sum(1)+(alpha[:,:2]*z).sum(1)-alpha.sum(1)*logc
        return np.exp(logsumexp(lp))
    for x in (-2.,0.,1.5):
        denom=quad(lambda t:joint(t,x),-np.inf,np.inf,epsabs=1e-11)[0]
        expected=quad(lambda t:t*joint(t,x),-np.inf,np.inf,epsabs=1e-11)[0]/denom
        got=risk.oracle_conditional_mean(spec,np.array([[np.nan,x]]))[0]
        assert got==pytest.approx(expected,rel=2e-9,abs=2e-9)


def test_true_parameter_point_has_zero_risk_after_training_standardization():
    spec=truth();rng=np.random.default_rng(18);y=rng.normal(size=(25,2));center=np.array([2.,-1.]);scale=np.array([.3,4.])
    oracle=risk.oracle_conditional_mean(spec,y)
    for shared in (True,False):
        params=ld_point(spec,center,scale,shared)
        prediction=risk.fitted_mean(params,params["method"],y,0,center,scale)
        np.testing.assert_allclose(prediction,oracle,atol=2e-15,rtol=2e-15)
        assert np.mean((prediction-oracle)**2)<1e-28
    changed=y.copy();changed[:,0]=np.nan
    np.testing.assert_array_equal(risk.oracle_conditional_mean(spec,changed),oracle)
    changed[:,0]=1e9
    np.testing.assert_array_equal(risk.oracle_conditional_mean(spec,changed),oracle)


def test_oracle_weights_use_conditioning_marginal_and_are_label_invariant():
    spec=truth();y=np.array([[np.nan,-8.],[np.nan,0.],[np.nan,8.]])
    original=risk.oracle_conditional_mean(spec,y)
    perm={key:np.asarray(value)[::-1].tolist() if key in ('weights','alpha','locations','scales') else value for key,value in spec.items()}
    np.testing.assert_allclose(original,risk.oracle_conditional_mean(perm,y),atol=1e-14)
    # It is not the mixture of component conditional means at fixed prior weights.
    component=[]
    for j in range(2):
        one={key:[value[j]] if key in ('alpha','locations','scales') else [1.] if key=='weights' else value for key,value in spec.items()}
        component.append(risk.oracle_conditional_mean(one,y))
    assert not np.allclose(original,np.array(spec['weights'])@np.array(component))


def test_paired_ratio_mcse_uses_covariance_and_ratio_of_means():
    right=np.array([1.,2.,4.,8.]);left=2*right
    result=risk.paired_summary(left,right)
    assert result['risk_ratio']==2 and result['ratio_mcse']==0
    left=np.array([1.,3.,5.,8.]);right=np.array([2.,2.,6.,10.]);out=risk.paired_summary(left,right)
    ratio=left.mean()/right.mean();influence=(left-ratio*right)/right.mean()
    assert out['ratio_mcse']==pytest.approx(influence.std(ddof=1)/2)
    assert out['risk_ratio']!=pytest.approx(np.mean(left/right))
    assert risk.paired_summary([],[])['risk_ratio'] is None


@pytest.fixture
def saved(tmp_path):
    primary=tmp_path/'primary';gaussian=tmp_path/'gaussian';primary.mkdir();gaussian.mkdir()
    cell={'id':'core_test','block':'core','family':'ld','repetitions':3,'dimension':2,'components':2,'n_train':35,'heterogeneity':0.}
    excluded={**cell,'id':'outside_test','block':'outside','family':'gaussian','repetitions':1}
    source={'cells':[cell,excluded],'n_test':60};p={'configuration':source,'configuration_sha256':risk.semantic_hash(source),'source_sha256':{}}
    risk.atomic_json(primary/'manifest.json',p)
    gt=[{'task_index':i,'primary_task_index':i,'cell_id':cell['id'],'replicate':i} for i in range(3)]
    extra={'task_index':3,'primary_task_index':3,'cell_id':'outside_test','replicate':0}
    g={'source_kind':'simulations','primary_metadata_sha256':risk.digest(primary/'manifest.json'),'tasks':gt+[extra],'source_sha256':{}}
    risk.atomic_json(gaussian/'manifest.json',g)
    spec=truth();rng=np.random.default_rng(82)
    for i in range(3):
        folder=primary/'tasks'/f'task_{i:06d}'/'attempt_001';folder.mkdir(parents=True)
        task={'task_index':i,'cell_id':cell['id'],'replicate':i,'attempt':1}
        risk.atomic_json(folder/'task.json',{**task,'cell':cell,'configuration_sha256':p['configuration_sha256']})
        risk.atomic_json(folder/'completion.json',task)
        train=rng.normal(size=(35,2));test=rng.normal(size=(60,2));center=train.mean(0);scale=train.std(0)
        np.savez_compressed(folder/'data.npz',train=train,test=test,center=center,scale=scale,
                            test_labels=rng.integers(0,2,len(test)))
        risk.atomic_json(folder/'generation.json',{'parameters':spec,'array_sha256':{'train':risk.array_hash(train),'test':risk.array_hash(test)}})
        baseline={}
        for method in risk.METHODS[:2]:
            row={**task,'method':method,'variant':'primary','failed':False,'score_converged':True,'boundary_hit':False,'projected_gradient':0.}
            baseline[method]=row;params=ld_point(spec,center,scale,method=='shared_ld')
            risk.atomic_json(folder/('primary__'+method+'.json'),{'row':row,'parameters':params})
        gdir=gaussian/'tasks'/f'task_{i:06d}';gdir.mkdir(parents=True)
        binding={'primary_task_index':i,'primary_directory':str(folder.relative_to(primary)),
            'input_sha256':{str(path.relative_to(primary)):risk.digest(path) for path in folder.iterdir()}}
        risk.atomic_json(gdir/'input_binding.json',binding)
        gp={'method':risk.METHODS[2],'k':2,'d':2,'theta':[0.,-1.,0.,1.,0.,0.,0.,0.,0.,.2]}
        gr={**gt[i],'method':risk.METHODS[2],'failed':False,'score_converged':i!=1,'boundary_hit':i==1,'projected_gradient':0. if i!=1 else .001}
        risk.atomic_json(gdir/'result.json',{'task':gt[i],'status':'complete','parameters':gp,'row':gr,'baseline_records':baseline,
            'manifest_sha256':risk.digest(gaussian/'manifest.json'),'input_binding_sha256':risk.semantic_hash(binding)})
        risk.atomic_json(gdir/'control_fit.json',{'task_index':i,'primary_task_index':i,'failed':False,'parameters':gp,
            'record':{key:gr.get(key) for key in ('parameter_count','score_converged','projected_gradient','boundary_hit')},
            'manifest_sha256':risk.digest(gaussian/'manifest.json'),'input_binding_sha256':risk.semantic_hash(binding)})
    protocol=tmp_path/'protocol.rst';protocol.write_text('Synthetic validation protocol; no application fits.\n')
    config={'cells':[cell],'blocks':['core'],'tasks':gt,'cell_count':1,'dataset_count':3,'n_test':60,'target_index':0,
        'methods':list(risk.METHODS),'refitting':False,'subsets':['all_finite','all_methods_score_eligible'],
        'primary_manifest_sha256':risk.digest(primary/'manifest.json'),'gaussian_manifest_sha256':risk.digest(gaussian/'manifest.json'),
        'primary_configuration_sha256':p['configuration_sha256'],'protocol_sha256':risk.digest(protocol)}
    cfg=tmp_path/'config.json';risk.atomic_json(cfg,config)
    receipt=tmp_path/'receipt.json';risk.atomic_json(receipt,{'config_sha256':risk.digest(cfg),'protocol_sha256':risk.digest(protocol),
        'new_conditional_mean_risk_outcomes_inspected':False,'new_model_fits':False})
    out=tmp_path/'report';risk.freeze(primary,gaussian,out,cfg,protocol,receipt)
    return primary,gaussian,out


def test_saved_reporting_resume_and_complete_risk_aggregation(saved,monkeypatch):
    primary,gaussian,out=saved;r=risk.Reporter(out)
    assert len(r.gmanifest['tasks'])==4 and len(r.manifest['tasks'])==3
    for i in range(3):
        value=r.task(i);assert all(row['available'] for row in value['rows'])
        assert all(row['risk']<1e-28 for row in value['rows'][:2])
    def forbidden(*args,**kwargs):raise AssertionError('Completed task recomputed')
    monkeypatch.setattr(r,'load_case',forbidden)
    assert r.task(0)['rows'][0]['available']
    status=risk.aggregate(r);assert status['available_predictions']==status['planned_predictions']==9
    summary=pd.read_csv(out/'risk_summary.csv')
    assert summary[summary.subset=='all_finite'].available.eq(3).all()
    assert summary[summary.subset=='all_methods_score_eligible'].available.eq(2).all()
    assert summary[summary.method==risk.METHODS[2]].boundary_hits.eq(1).all()


def test_missing_tasks_keep_all_declared_denominators(saved):
    _,_,out=saved;r=risk.Reporter(out);r.task(0);status=risk.aggregate(r)
    assert status['planned_predictions']==9 and status['recorded_predictions']==3
    table=pd.read_csv(out/'risk_summary.csv')
    assert table.planned.eq(3).all() and table.available.eq(1).all()


def test_absent_gaussian_result_retains_both_ld_predictions(saved):
    _,gaussian,out=saved;(gaussian/'tasks/task_000000/result.json').unlink()
    value=risk.Reporter(out).task(0)
    assert [r['available'] for r in value['rows']]==[True,True,False]
    assert value['attempt_fallback'] is False


def test_missing_gaussian_binding_uses_recorded_infrastructure_fallback(saved):
    _,gaussian,out=saved;(gaussian/'tasks/task_000000/result.json').unlink();(gaussian/'tasks/task_000000/input_binding.json').unlink()
    value=risk.Reporter(out).task(0)
    assert value['attempt_fallback'] and value['primary_attempt']==1
    assert [r['available'] for r in value['rows']]==[True,True,False]


@pytest.mark.parametrize('mutation',['upstream_parameter','stored_prediction','duplicate_result','source'])
def test_immutable_inputs_predictions_and_task_identity_are_checked(saved,mutation):
    primary,gaussian,out=saved;r=risk.Reporter(out);r.task(0)
    if mutation=='upstream_parameter':
        p=primary/'tasks/task_000000/attempt_001/primary__shared_ld.json';v=risk.read(p);v['parameters']['theta'][0]+=.2;risk.atomic_json(p,v)
        with pytest.raises(risk.IntegrityError):r.task(0)
    elif mutation=='stored_prediction':
        p=out/'tasks/task_000000/predictions.npz';p.write_bytes(p.read_bytes()+b'changed')
        with pytest.raises(risk.IntegrityError):r.task(0)
    elif mutation=='duplicate_result':
        p=out/'tasks/task_000000_backup';p.mkdir();(p/'result.json').write_bytes((out/'tasks/task_000000/result.json').read_bytes())
        with pytest.raises(risk.IntegrityError):risk.aggregate(r)
    else:
        p=out/'source_snapshot/scripts/simulation_conditional_mean_risk.py';p.write_text(p.read_text()+'\n#changed')
        with pytest.raises(risk.IntegrityError):risk.Reporter(out)


def test_saved_target_values_and_true_labels_do_not_enter_predictions(saved):
    _,_,out=saved;r=risk.Reporter(out);case=r.load_case(r.manifest['tasks'][0])
    original=risk.oracle_conditional_mean(case['truth'],case['test'])
    changed=case['test'].copy();changed[:,0]=12345
    np.testing.assert_array_equal(original,risk.oracle_conditional_mean(case['truth'],changed))
    for method,(params,_) in case['fits'].items():
        np.testing.assert_array_equal(risk.fitted_mean(params,method,changed,0,case['center'],case['scale']),
                                      risk.fitted_mean(params,method,case['test'],0,case['center'],case['scale']))


@pytest.mark.parametrize('mutation',['row_task','recorded_risk'])
def test_aggregation_rejects_changed_row_identity_or_array_inconsistent_risk(saved,mutation):
    _,_,out=saved;r=risk.Reporter(out);r.task(0);path=out/'tasks/task_000000/result.json';v=risk.read(path)
    if mutation=='row_task':v['rows'][0]['replicate']=2
    else:v['rows'][0]['risk']=.8
    risk.atomic_json(path,v)
    with pytest.raises(risk.IntegrityError):risk.aggregate(r)


def test_incomplete_setup_retains_hashes_and_complete_missing_denominators(saved):
    primary,gaussian,out=saved
    (gaussian/'tasks/task_000000/result.json').unlink();(gaussian/'tasks/task_000000/input_binding.json').unlink()
    (primary/'tasks/task_000000/attempt_001/data.npz').unlink()
    r=risk.Reporter(out);value=r.task(0)
    assert not any(row['available'] for row in value['rows'])
    assert len(value['input_sha256'])==2 and value['setup_reason'].startswith('FileNotFoundError')
    status=risk.aggregate(r)
    assert status['available_predictions']==0 and status['planned_predictions']==9 and status['recorded_predictions']==3


def test_gaussian_selected_point_must_match_its_reported_parameters(saved):
    _,gaussian,out=saved;p=gaussian/'tasks/task_000000/control_fit.json';v=risk.read(p);v['parameters']['theta'][0]+=.3;risk.atomic_json(p,v)
    with pytest.raises(risk.IntegrityError,match='selected parameter point'):risk.Reporter(out).task(0)


def test_byte_identical_relocated_sources_support_resume(saved,tmp_path):
    primary,gaussian,out=saved;risk.Reporter(out).task(0)
    moved_primary=tmp_path/'moved_primary';moved_gaussian=tmp_path/'moved_gaussian'
    shutil.copytree(primary,moved_primary);shutil.copytree(gaussian,moved_gaussian)
    r=risk.Reporter(out,moved_primary,moved_gaussian)
    assert r.task(0)['rows'][0]['available']
    assert all(row['available'] for row in r.task(1)['rows'])
    assert all(row['available'] for row in r.task(1)['rows'])
