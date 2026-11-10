"""Mathematical checks that can detect errors independent of an optimizer."""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import gammaln, logsumexp
from ld_information import information, component_mean_and_gradient
from shared_shape_ld import dimensions, evaluation, log_simplex


def jacobian(fun, x, step=1e-5):
    columns = []
    for index in range(len(x)):
        delta = step*max(1., abs(x[index]))
        left, right = x.copy(), x.copy()
        left[index] -= delta
        right[index] += delta
        columns.append((np.asarray(fun(right))-np.asarray(fun(left)))/(2*delta))
    return np.stack(columns, axis=-1)


@pytest.mark.parametrize('common', [False, True])
@pytest.mark.parametrize('k,d', [(2,1), (2,2), (3,4)])
def test_information_and_mean_gradients(common, k, d):
    rng = np.random.default_rng(291+k+d)
    r,p,na,nb = dimensions(k,d,common)
    theta = np.r_[rng.normal(size=r), rng.uniform(.3,4.,p),
                  rng.uniform(.4,2.,na), rng.normal(size=nb)]
    y = rng.normal(size=(13,d))
    ell, score, info = information(theta,y,k,common)
    values, _, direct_score = evaluation(theta,y,k,common,True)
    np.testing.assert_allclose(ell, values.sum(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(score,direct_score,rtol=1e-11,atol=1e-11)
    numerical = -jacobian(lambda v:evaluation(v,y,k,common,True)[2],theta)
    np.testing.assert_allclose(info,numerical,rtol=2e-5,atol=2e-6)
    np.testing.assert_allclose(info,info.T,rtol=1e-12,atol=1e-12)
    mean, grad = component_mean_and_gradient(theta,k,d,0,0,common)
    numeric = jacobian(lambda v:component_mean_and_gradient(v,k,d,0,0,common)[0],theta)
    np.testing.assert_allclose(grad,numeric,rtol=2e-5,atol=1e-7)


@pytest.mark.parametrize('shapes', [(.3,2.), (2.,.4), (3.,3.)])
def test_univariate_normalization(shapes):
    # k=1: alpha, a, b. Improper integral over the complete real line.
    theta = np.r_[shapes, 1.3, -.7]
    value, error = quad(lambda x:np.exp(evaluation(theta,np.array([[x]]),1)[0][0]),
                        -np.inf,np.inf,epsabs=1e-10)
    assert abs(value-1.) < 2e-8


def test_change_of_variables_and_responsibilities():
    rng = np.random.default_rng(78)
    y = rng.normal(size=(50,3))
    alpha = np.array([[.4,1.2,2.,3.],[3.,.7,1.,2.]])
    weights = np.array([.4,.6])
    logx = log_simplex(y)
    dir_terms = (gammaln(alpha.sum(axis=1))-gammaln(alpha).sum(axis=1)
                 +logx@(alpha-1).T+np.log(weights))
    ld_terms = (gammaln(alpha.sum(axis=1))-gammaln(alpha).sum(axis=1)
                +logx@alpha.T+np.log(weights))
    np.testing.assert_allclose(logsumexp(ld_terms,axis=1),
                               logsumexp(dir_terms,axis=1)+logx.sum(axis=1),atol=1e-12)
    np.testing.assert_allclose(np.exp(ld_terms-logsumexp(ld_terms,axis=1)[:,None]),
                               np.exp(dir_terms-logsumexp(dir_terms,axis=1)[:,None]),atol=1e-12)


def test_extreme_log_ratios_remain_finite():
    theta=np.r_[[.5,2.,.3], [1.,1.], [0.,0.]]
    ell, _=evaluation(theta,np.array([[-1000.,0.],[1000.,-1000.]]),1)
    assert np.isfinite(ell).all()
