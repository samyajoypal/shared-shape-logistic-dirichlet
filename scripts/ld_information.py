"""Exact observed information in the common-shape fitting coordinates.

Coordinates: weight logits, common raw shapes, inverse scales a, intercepts b.
This implementation is self-contained and does not load the archived pilot.
It uses the mixture Hessian identity (conditional mean Hessian plus the
conditional covariance of complete-data scores). No numerical derivatives
are used by this module.
"""
from __future__ import annotations

import numpy as np
from scipy.special import digamma, polygamma

from shared_shape_ld import dimensions, evaluation, log_simplex, unpack


def information(theta, y, k, common_scale=False):
    """Return total log likelihood, total score and negative total Hessian."""
    y = np.asarray(y, dtype=float)
    n, d = y.shape
    r, p, na, _ = dimensions(k, d, common_scale)
    qpar = len(theta)
    logpi, alpha, a, b = unpack(theta, k, d, common_scale)
    pi = np.exp(logpi)
    ell, tau = evaluation(theta, y, k, common_scale)
    lq = log_simplex(y[:, None, :] * a[None, :, :] + b[None, :, :])
    scores = np.zeros((n, k, qpar))
    expected_hessian = np.zeros((qpar, qpar))
    hweight = -(np.diag(pi[:r]) - np.outer(pi[:r], pi[:r]))
    hshape = polygamma(1, alpha.sum()) * np.ones((p, p)) - np.diag(polygamma(1, alpha))
    expected_hessian[:r, :r] = n*hweight
    expected_hessian[r:r+p, r:r+p] = n*hshape
    shape_indices = np.arange(r, r+p)
    for j in range(k):
        aj = a[0 if common_scale else j]
        a_indices = np.arange(r+p+(0 if common_scale else j*d), r+p+(0 if common_scale else j*d)+d)
        b_indices = np.arange(r+p+na+j*d, r+p+na+(j+1)*d)
        probs = np.exp(lq[:, j, :d])
        v = alpha[:d] - alpha.sum()*probs
        scores[:, j, :r] = -pi[:r]
        if j < r:
            scores[:, j, j] += 1
        scores[:, j, r:r+p] = digamma(alpha.sum())-digamma(alpha)+lq[:, j]
        scores[:, j, a_indices] = 1/aj + y*v
        scores[:, j, b_indices] = v
        hzz = alpha.sum()*np.einsum('ni,nj->nij', probs, probs)
        diagonal = np.arange(d)
        hzz[:, diagonal, diagonal] -= alpha.sum()*probs
        haz = np.broadcast_to(np.vstack((np.eye(d), np.zeros((1, d)))), (n, p, d)).copy()
        haz -= probs[:, None, :]
        weighted = tau[:, j]

        def block(rows, cols, values, transpose=False):
            value = np.einsum('n,nij->ij', weighted, values)
            expected_hessian[np.ix_(rows, cols)] += value
            if transpose:
                expected_hessian[np.ix_(cols, rows)] += value.T

        block(shape_indices, a_indices, haz*y[:, None, :], True)
        block(shape_indices, b_indices, haz, True)
        haa = hzz*y[:, :, None]*y[:, None, :]
        haa[:, diagonal, diagonal] -= 1/aj**2
        block(a_indices, a_indices, haa)
        block(a_indices, b_indices, hzz*y[:, :, None], True)
        block(b_indices, b_indices, hzz)
    observed_scores = np.einsum('nk,nkp->np', tau, scores)
    score_second_moment = np.einsum('nk,nkp,nkq->pq', tau, scores, scores)
    hessian = expected_hessian + score_second_moment-observed_scores.T@observed_scores
    return float(ell.sum()), observed_scores.sum(axis=0), -hessian


def component_mean_and_gradient(theta, k, d, component, coordinate, common_scale=False):
    """Mean and analytic gradient; locations are not confused with means."""
    r, p, na, _ = dimensions(k, d, common_scale)
    _, alpha, a, b = unpack(theta, k, d, common_scale)
    row = 0 if common_scale else component
    m = coordinate
    aj = a[row, m]
    mean = (digamma(alpha[m])-digamma(alpha[-1])-b[component, m])/aj
    gradient = np.zeros(len(theta))
    gradient[r+m] = polygamma(1, alpha[m])/aj
    gradient[r+p-1] = -polygamma(1, alpha[-1])/aj
    gradient[r+p+row*d+m] = -mean/aj
    gradient[r+p+na+component*d+m] = -1/aj
    return float(mean), gradient
