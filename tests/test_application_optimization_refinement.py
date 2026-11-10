"""Feasible-set identity, chain-rule scores, and training-only safeguards."""
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative

import application_optimization_refinement as refine
import gaussian_shared_correlation as gaussian
import run_confirmatory_experiments as base
import shared_shape_ld as shared


def model_point(method, d=2):
    rng = np.random.default_rng(617+d)
    y = rng.normal(size=(35, d)); k = 2
    settings = dict(base.DEFAULT_SETTINGS)
    if method == gaussian.METHOD:
        theta = gaussian.pack([.4,.6], np.array([np.full(d,-.2),np.full(d,.3)]),
                              np.full((k,d),.9), np.full(gaussian.correlation_parameter_count(d),.4))
        limits = gaussian.bounds(k,d)
    else:
        theta = base.initial_ld(y,np.arange(len(y))%2,k,method,settings,0)
        limits = base.ld_bounds(method,k,d,settings)
    return y,k,settings,theta,limits


@pytest.mark.parametrize('method',['shared_ld','full_ld',gaussian.METHOD])
@pytest.mark.parametrize('d',[2,5])
def test_transformed_gradient_and_identical_box(method,d):
    y,k,settings,theta,limits=model_point(method,d)
    coordinates=refine.Coordinates(method,k,d,limits)
    q=coordinates.to_transformed(theta)
    np.testing.assert_allclose(coordinates.to_raw(q),theta,rtol=2e-15,atol=2e-15)
    value,gradient=refine.objective(theta,y,k,method)
    expected=gradient*coordinates.derivative(theta)
    numeric=approx_derivative(lambda v:np.array([refine.objective(coordinates.to_raw(v),y,k,method)[0]]),q,method='3-point').ravel()
    np.testing.assert_allclose(expected,numeric,rtol=2e-5,atol=2e-8)
    for direction in ['lower','upper']:
        endpoint=theta.copy()
        for j,(lo,hi) in enumerate(limits):
            bound=lo if direction=='lower' else hi
            if bound is not None: endpoint[j]=bound
        np.testing.assert_allclose(coordinates.to_raw(coordinates.to_transformed(endpoint)),endpoint,rtol=2e-15,atol=2e-15)
        assert np.all(coordinates.to_raw(coordinates.to_transformed(endpoint))>=coordinates.lower)
        assert np.all(coordinates.to_raw(coordinates.to_transformed(endpoint))<=coordinates.upper)


def test_outside_saved_point_is_rejected_not_clipped():
    y,k,settings,theta,limits=model_point('shared_ld')
    theta[k-1]=limits[k-1][1]+.01
    with pytest.raises(ValueError,match='never clipped'):
        refine.refine_point(theta,y,k,'shared_ld',settings)


def test_safeguarded_newton_solves_ill_conditioned_box_problem():
    diagonal=np.array([1e-3,1.,1e3]);target=np.array([-2.,.3,-.2]);seen=[]
    coordinates=SimpleNamespace(q_lower=np.full(3,-1.),q_upper=np.ones(3),transformed_bounds=[(-1.,1.)]*3)
    def fun(q):
        assert np.all(q>=-1.) and np.all(q<=1.)
        seen.append(q.copy());delta=q-target
        return .5*np.sum(diagonal*delta**2),diagonal*delta
    def stop(q):
        return shared.projected_gradient(q,fun(q)[1],coordinates.transformed_bounds)<1e-8
    point,history=refine.newton_polish(np.array([.8,-.5,.5]),fun,coordinates,stop,refine.DEFAULT_POLICY)
    np.testing.assert_allclose(point,[-1.,.3,-.2],atol=2e-7)
    assert all(item['accepted'] for item in history)


@pytest.mark.parametrize('method',['shared_ld','full_ld',gaussian.METHOD])
def test_refinement_retains_or_improves_training_likelihood(method):
    y,k,settings,theta,limits=model_point(method)
    result=refine.refine_point(theta,y,k,method,settings,
        {'transformed_maxiter':20,'newton_maxiter':3,'raw_maxiter':20,'start_timeout_seconds':10})
    assert result['training_mean_log_density_gain']>=-1e-12
    assert result['original_candidate_retained']
    assert np.isfinite(result['theta']).all()
    original_pg=shared.projected_gradient(result['theta'],refine.objective(result['theta'],y,k,method)[1],limits)
    assert result['projected_gradient']==pytest.approx(original_pg)
    assert result['score_converged']==(original_pg<=1e-6)
    np.testing.assert_array_equal(result['original_parameters'],theta)


def test_optimizer_failure_keeps_original_point(monkeypatch):
    y,k,settings,theta,limits=model_point('full_ld')
    def timeout(*args,**kwargs): raise TimeoutError('Injected optimizer timeout')
    monkeypatch.setattr(refine,'minimize',timeout)
    result=refine.refine_point(theta,y,k,'full_ld',settings)
    np.testing.assert_array_equal(result['theta'],theta)
    assert 'Injected optimizer timeout' in result['exceptions'][0]['exception']
    assert not result['failed']
    assert result['training_mean_log_density_gain']==0.


def test_all_finite_failed_endpoints_are_retained_and_duplicates_removed():
    theta=np.arange(5,dtype=float)
    saved={'parameters':{'theta':theta},'starts':[
        {'failed':True,'optimizer_endpoint_theta':theta+1},
        {'failed':False,'parameters':{'theta':theta}},
        {'failed':True,'optimizer_endpoint_theta':[np.nan]*5}]}
    points,omitted=refine.saved_endpoints(saved)
    assert len(points)==2 and len(omitted)==1
    assert points[1][0]=='saved_start_0_optimizer_endpoint'


def test_full_model_preserves_new_shared_nesting_candidate():
    y,k,settings,theta,limits=model_point('shared_ld')
    policy={'transformed_maxiter':10,'newton_maxiter':2,'raw_maxiter':10,'start_timeout_seconds':10}
    common=refine.refine_saved({'parameters':{'theta':theta},'starts':[]},y,k,'shared_ld',settings,policy)
    full=refine.refine_saved({'parameters':None,'starts':[]},y,k,'full_ld',settings,policy,refined_shared=common)
    assert full.record['mean_train_log_density']>=common.record['mean_train_log_density']-1e-10
    assert full.starts[0]['origin']=='embedded_refined_shared'
    assert full.record['dependency_fit_seconds']==common.record['fit_seconds']
    assert full.record['policy_fit_seconds']>=full.record['fit_seconds']


def test_rounding_better_probe_cannot_discard_stationary_endpoint(monkeypatch):
    y,k,settings,theta,limits=model_point('shared_ld')
    start=theta[0];stationary=start+.2;probe=start+.200001
    def objective(point,*args):
        gradient=np.zeros_like(point)
        if abs(point[0]-stationary)<1e-12:return -1.,gradient
        if abs(point[0]-probe)<1e-12:
            gradient[0]=.01;return -1.-5e-13,gradient
        gradient[0]=.1;return 0.,gradient
    def minimize(fun,initial,**kwargs):
        optimum=initial.copy();optimum[0]=stationary;fun(optimum)
        perturbed=initial.copy();perturbed[0]=probe;fun(perturbed)
        return SimpleNamespace(x=perturbed,nit=1,message='Synthetic finite-difference probe')
    monkeypatch.setattr(refine,'objective',objective);monkeypatch.setattr(refine,'minimize',minimize)
    result=refine.refine_point(theta,y,k,'shared_ld',settings)
    assert result['theta'][0]==stationary
    assert result['score_converged']
    assert 0<result['selected_training_gap_to_best']<1e-12


def test_retained_optimizer_timeout_has_separate_model_counters(monkeypatch):
    y,k,settings,theta,limits=model_point('shared_ld')
    def timeout(*args,**kwargs):raise TimeoutError('Injected retained timeout')
    monkeypatch.setattr(refine,'minimize',timeout)
    model=refine.refine_saved({'parameters':{'theta':theta},'starts':[]},y,k,'shared_ld',settings)
    assert model.record['starts_failed']==0
    assert model.record['retained_starts_with_exceptions']==1
    assert model.record['optimizer_exception_count']==model.record['timeout_exception_count']==1
    assert model.params is not None
