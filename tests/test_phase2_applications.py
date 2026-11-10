"""Independent checks of marginal integration and application coordinates."""
import json
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import gammaln,logsumexp
from scipy.stats import multivariate_normal
import phase2_applications as app
from run_confirmatory_experiments import evaluate_params
import regularized_skew_normal as skew
from types import SimpleNamespace


def test_ld_marginal_matches_independent_numerical_integration():
    # One component, unequal shapes, nonunit diagonal scales and nonzero shifts.
    theta=np.r_[.9,2.3,1.4, 1.2,.8, .4,-.3]
    p={'method':'shared_ld','k':1,'d':2,'theta':theta}
    for x in [-1.,0.7]:
        exact=np.exp(app.marginal_log_density(p,np.array([[x,0.]]),[0])[0])
        numerical=quad(lambda v:np.exp(evaluate_params(p,np.array([[x,v]]))[0][0]),-60,60,epsabs=1e-10)[0]
        assert np.isclose(exact,numerical,rtol=1e-8)


def test_ld_conditional_integrates_to_one():
    p={'method':'shared_ld','k':1,'d':2,'theta':np.r_[.9,2.3,1.4, 1.2,.8, .4,-.3]}
    x=.6
    marginal=app.marginal_log_density(p,np.array([[x,0.]]),[0])[0]
    integral=quad(lambda v:np.exp(evaluate_params(p,np.array([[x,v]]))[0][0]-marginal),-80,80)[0]
    assert abs(integral-1)<1e-8


def test_shared_shapes_broadcast_to_every_component():
    theta=np.r_[.3,.9,2.3,1.4, 1.2,.8,.7,1.1, .4,-.3,-.8,.6]
    p={'method':'shared_ld','k':2,'d':2,'theta':theta}
    assert app.ld_components(p)[1].shape==(2,3)
    y=np.array([[-.3,.2],[.7,-1.]])
    assert np.allclose(app.marginal_log_density(p,y,[0,1]),evaluate_params(p,y)[0])
    exact=np.exp(app.marginal_log_density(p,y[:1],[0])[0])
    numerical=quad(lambda v:np.exp(evaluate_params(p,np.array([[y[0,0],v]]))[0][0]),-80,80)[0]
    assert np.isclose(exact,numerical,rtol=1e-8)


def test_skew_marginal_matches_numerical_integration():
    theta=skew.pack([1.],[[.2,-.3]],[[[1.2,0.],[.5,.9]]],[[1.3,-.8]])
    p={'method':skew.METHOD,'k':1,'d':2,'theta':theta}
    for x in [-1.,.7]:
        exact=np.exp(app.marginal_log_density(p,np.array([[x,0.]]),[0])[0])
        integrated=quad(lambda v:np.exp(skew.evaluate(np.array([[x,v]]),p)[0][0]),-20,20,epsabs=1e-10)[0]
        assert np.isclose(exact,integrated,rtol=1e-8)


def test_full_subset_equals_joint_for_each_family():
    p={'method':'gaussian_full','k':1,'d':2,'weights':[1.],
       'means':[[.2,-.3]],'covariances':[[[1.2,.3],[.3,.9]]]}
    y=np.array([[.1,.5],[-.8,.7]])
    assert np.allclose(app.marginal_log_density(p,y,[0,1]),evaluate_params(p,y)[0])
    t={'method':'fmvmm_t','k':1,'d':2,'weights':[1.],
       'components':[[p['means'][0],p['covariances'][0],[0.,0.],5.]]}
    assert np.allclose(app.marginal_log_density(t,y,[0,1]),evaluate_params(t,y)[0])
    assert np.array_equal(app.marginal_log_density(p,y,[]),[0.,0.])


@pytest.mark.skipif(not all((app.ROOT/name).exists() for name in app.DATA_NAMES), reason="Run scripts/fetch_data.py --datasets original for integration data")
def test_data_transform_and_jacobian_are_explicit():
    config=json.loads(app.CONFIG.read_text())
    variant=next(v for v in config['variants'] if v['id']=='crabs_log5')
    y,labels,blocks,jac,names=app.load_data(variant)
    assert y.shape==(200,5)
    assert np.array_equal(y,jac)
    raw=next(v for v in config['variants'] if v['id']=='crabs_raw5')
    values,_,_,zero,_=app.load_data(raw)
    assert np.allclose(np.log(values),y)
    assert np.array_equal(zero,np.zeros_like(zero))
    olive=next(v for v in config['variants'] if v['id']=='olive6_oleic')
    y,*rest=app.load_data(olive)
    assert y.shape==(572,5) and np.isfinite(y).all()


@pytest.mark.skipif(not all((app.ROOT/name).exists() for name in app.DATA_NAMES), reason="Run scripts/fetch_data.py --datasets original for integration data")
def test_frozen_tasks_share_splits_and_preserve_bootstrap_strata():
    config=json.loads(app.CONFIG.read_text());tasklist=app.tasks(config)
    a=next(t for t in tasklist if t['variant']=='olive3_oleic' and t['kind']=='cv')
    b=next(t for t in tasklist if t['variant']=='olive6_oleic' and t['kind']=='cv')
    assert a['train']==b['train'] and a['test']==b['test']
    variant=next(v for v in config['variants'] if v['id']=='crabs_log5')
    _,_,blocks,_,_=app.load_data(variant)
    boot=next(t for t in tasklist if t['variant']=='crabs_log5' and t['kind']=='bootstrap')
    assert all(np.sum(blocks[np.asarray(boot['train'])]==v)==50 for v in set(blocks))


def test_crab_log_score_jacobian_for_conditional_target():
    # For X=exp(Y), f_X(x_t|x_o)=f_Y(log x_t|log x_o)/x_t.
    p={'method':'gaussian_full','k':1,'d':2,'weights':[1.],
       'means':[[0.,0.]],'covariances':[[[1.,0.],[0.,1.]]]}
    x=np.array([[2.,3.]])
    y=np.log(x)
    cond=evaluate_params(p,y)[0]-app.marginal_log_density(p,y,[0])-y[:,1]
    from scipy.stats import lognorm
    assert np.allclose(cond,lognorm.logpdf(x[:,1],s=1.))


@pytest.mark.skipif(not all((app.ROOT/name).exists() for name in app.DATA_NAMES), reason="Run scripts/fetch_data.py --datasets original for integration data")
def test_profile_failure_is_failed_and_not_a_warm_start(tmp_path,monkeypatch):
    config=json.loads(app.CONFIG.read_text())
    config['variants']=[{**config['variants'][0],'repeats':0}]
    config['bootstrap_repetitions']=0;config['methods']=['shared_ld','full_ld']
    config_path=tmp_path/'config.json';config_path.write_text(json.dumps(config))
    output=tmp_path/'study';app.freeze(output,config_path)
    warms=[]
    def fake_fit(method,y,k,seed,settings,partitions,warm):
        warms.append(warm)
        return SimpleNamespace(method=method,k=k,d=y.shape[1],params={'method':method},
             starts=[],record={'fit_seconds':0.,'parameter_count':1},
             evaluate=lambda data:(np.zeros(len(data)),np.full((len(data),k),1/k)))
    monkeypatch.setattr(app,'fit_method',fake_fit)
    monkeypatch.setattr(app,'marginal_log_density',lambda params,data,keep:np.zeros(len(data)))
    def fail_profile(*args,**kwargs):raise RuntimeError('injected profile failure')
    monkeypatch.setattr(app,'component_profiles',fail_profile)
    completion=app.run_task(output,0)
    assert all(r['failed'] for r in completion['records'])
    assert all('injected profile failure' in r['failure'] for r in completion['records'])
    assert warms==[None,None]
    # A restart keeps the two failure records and does not refit them.
    again=app.run_task(output,0)
    assert all(r['failed'] for r in again['records']) and len(warms)==2
