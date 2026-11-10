import sys
from pathlib import Path
import numpy as np
import pytest
from scipy.stats import norm

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import conditional_log_linear as model


def test_svd_fit_matches_orthogonal_projection_and_mle_not_unbiased_sd():
    rng=np.random.default_rng(64);x=rng.normal(size=(80,3));target=1+2*x[:,0]-.4*x[:,1]+rng.normal(size=80)*.6
    y=np.column_stack([x[:,0],target,x[:,1:]])
    fit=model.fit(y,1);design=np.column_stack([np.ones(80),x])
    expected=np.linalg.solve(design.T@design,design.T@target)
    np.testing.assert_allclose(fit.params["coefficients"],expected,atol=2e-15)
    residual=target-design@expected
    assert fit.params["residual_sd"]==pytest.approx(np.sqrt(residual@residual/80))
    assert fit.record["parameter_count"]==5 and fit.record["design_rank"]==4
    assert fit.record["score_converged"] and not fit.record["boundary_hit"]
    np.testing.assert_allclose(fit.conditional_log_density(y),norm.logpdf(target,design@expected,fit.params["residual_sd"]))


def test_prediction_ignores_target_values_including_nan():
    rng=np.random.default_rng(32);y=rng.normal(size=(30,4));fit=model.fit(y,2)
    expected=fit.conditional_mean(y);changed=y.copy();changed[:,2]=np.nan
    np.testing.assert_array_equal(fit.conditional_mean(changed),expected)
    with pytest.raises(ValueError):fit.conditional_log_density(changed)


def test_rank_deficiency_uses_minimum_norm_without_ridge():
    x=np.arange(20.)-10;y=np.column_stack([x,2*x,.5+3*x+np.sin(x)])
    fit=model.fit(y,2);beta=fit.params["coefficients"]
    assert fit.record["rank_deficient"] and fit.record["design_rank"]==2
    assert fit.record["parameter_count"]==3 and fit.record["nominal_parameter_count"]==4
    assert np.isinf(fit.record["design_condition"])
    assert beta[2]==pytest.approx(2*beta[1])
    assert fit.record["score_converged"]


def test_exact_response_relation_uses_declared_sd_floor_and_records_boundary():
    x=np.arange(20.);y=np.column_stack([x,2+x])
    fit=model.fit(y,1)
    assert fit.params["residual_sd"]==1e-4
    assert fit.record["residual_floor_active"] and fit.record["boundary_hit"]
    assert np.isfinite(fit.conditional_log_density(y)).all()
    with pytest.raises(ValueError):model.fit(y,1,residual_sd_floor=0)


def test_original_units_conditional_jacobian_matches_direct_lognormal_formula():
    rng=np.random.default_rng(42);raw=np.exp(rng.normal(size=(70,3)));analysis=np.log(raw)
    center=analysis.mean(0);scale=analysis.std(0);standardized=(analysis-center)/scale
    fit=model.fit(standardized,2)
    mean,density=model.original_scale_predictions(fit.params,analysis,center,scale,-analysis[:,2])
    expected_mean=center[2]+scale[2]*fit.conditional_mean(standardized)
    np.testing.assert_allclose(mean,expected_mean)
    np.testing.assert_allclose(density,norm.logpdf(analysis[:,2],mean,scale[2]*fit.params["residual_sd"])-analysis[:,2])
    _,alr_density=model.original_scale_predictions(fit.params,analysis,center,scale)
    np.testing.assert_allclose(alr_density,density+analysis[:,2])


def test_intercept_only_conditional_model_is_supported():
    y=np.array([[-1.],[.2],[.5],[2.]])
    fit=model.fit(y,0)
    np.testing.assert_allclose(fit.conditional_mean(np.full((3,1),np.nan)),y.mean())
    assert fit.params["residual_sd"]==pytest.approx(y.std(ddof=0))
    assert fit.record["parameter_count"]==2


def test_target_permutation_preserves_conditional_predictions():
    rng=np.random.default_rng(9);y=rng.normal(size=(50,4));fit=model.fit(y,1)
    perm=[3,2,1,0];other=model.fit(y[:,perm],2)
    np.testing.assert_allclose(fit.conditional_mean(y),other.conditional_mean(y[:,perm]),atol=2e-15)
    np.testing.assert_allclose(fit.conditional_log_density(y),other.conditional_log_density(y[:,perm]),atol=2e-15)
