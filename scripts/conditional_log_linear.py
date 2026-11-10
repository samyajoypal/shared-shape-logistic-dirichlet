"""Conditional Gaussian reference fitted by SVD least squares.

The inputs are training-standardized log/ALR coordinates. This is a
conditional model only: it has no joint density or component assignments.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

METHOD = "conditional_log_linear"
RESIDUAL_SD_FLOOR = 1e-4


def design_matrix(y, target):
    y = np.asarray(y, float)
    if y.ndim != 2 or y.shape[1] < 1 or not 0 <= target < y.shape[1]:
        raise ValueError("Need a data matrix and a valid target coordinate")
    keep = [m for m in range(y.shape[1]) if m != target]
    if not np.isfinite(y[:, keep]).all():
        raise ValueError("Conditioning coordinates must be finite")
    return np.column_stack([np.ones(len(y)), y[:, keep]])


def validate_params(params):
    d, target = int(params["d"]), int(params["target"])
    beta = np.asarray(params["coefficients"], float)
    sigma = float(params["residual_sd"])
    if (params["method"] != METHOD or not 0 <= target < d or beta.shape != (d,)
            or not np.isfinite(beta).all() or not np.isfinite(sigma) or sigma <= 0):
        raise ValueError("Invalid conditional Gaussian parameters")
    return d, target, beta, sigma


def conditional_mean(params, y):
    d, target, beta, _ = validate_params(params)
    y = np.asarray(y, float)
    if y.ndim != 2 or y.shape[1] != d:
        raise ValueError("Prediction dimension differs from the fitted model")
    return design_matrix(y, target) @ beta


def conditional_log_density(params, y):
    _, target, _, sigma = validate_params(params)
    y = np.asarray(y, float)
    mean = conditional_mean(params, y)
    if not np.isfinite(y[:, target]).all():
        raise ValueError("Observed targets must be finite for density evaluation")
    residual = (y[:, target]-mean)/sigma
    return -.5*np.log(2*np.pi)-np.log(sigma)-.5*residual**2


def original_scale_predictions(params, analysis_y, center, scale, target_log_jacobian=None):
    """Return original analysis-coordinate mean and target-measure log density.

    target_log_jacobian is log|d analysis_target/d original_target|.
    It is zero for ALR scores and -log(x_target) for positive-unit log scores.
    The mean always remains a log/ALR mean, not an arithmetic original-unit mean.
    """
    d, target, _, _ = validate_params(params)
    y = np.asarray(analysis_y, float)
    center, scale = np.asarray(center, float), np.asarray(scale, float)
    if (center.shape != (d,) or scale.shape != (d,) or not np.isfinite(center).all()
            or not np.isfinite(scale).all() or np.any(scale <= 0)):
        raise ValueError("Invalid saved training transformation")
    standardized = (y-center)/scale
    mean = center[target]+scale[target]*conditional_mean(params, standardized)
    density = conditional_log_density(params, standardized)-np.log(scale[target])
    if target_log_jacobian is not None:
        jacobian = np.asarray(target_log_jacobian, float)
        if jacobian.shape != (len(y),) or not np.isfinite(jacobian).all():
            raise ValueError("Target Jacobian must be finite with one value per row")
        density = density+jacobian
    return mean, density


@dataclass
class ConditionalLogLinearFit:
    params: dict
    record: dict

    @property
    def method(self):
        return METHOD

    def conditional_mean(self, y):
        return conditional_mean(self.params, y)

    def conditional_log_density(self, y):
        return conditional_log_density(self.params, y)


def fit(y, target, residual_sd_floor=RESIDUAL_SD_FLOOR, gtol=1e-6):
    """SVD least squares, then Gaussian MLE residual SD with a fixed floor.

    NumPy lstsq(rcond=None) determines numerical rank using its documented
    machine-precision cutoff. No ridge, tuning, labels or test sample is used.
    Rank deficiency is retained and flagged, with the minimum-norm solution.
    """
    started = time.perf_counter()
    y = np.asarray(y, float)
    x = design_matrix(y, target)
    if len(y) == 0 or not np.isfinite(y[:, target]).all():
        raise ValueError("Training observations must be nonempty and finite")
    if not np.isfinite(residual_sd_floor) or residual_sd_floor <= 0 or not np.isfinite(gtol) or gtol <= 0:
        raise ValueError("Residual SD floor and score tolerance must be positive")
    response = y[:, target]
    beta, _, rank, singular = np.linalg.lstsq(x, response, rcond=None)
    residual = response-x@beta
    rss = float(residual@residual)
    unconstrained_sd = float(np.sqrt(rss/len(y)))
    sigma = max(unconstrained_sd, float(residual_sd_floor))
    if not np.isfinite(beta).all() or not np.isfinite(sigma):
        raise FloatingPointError("Nonfinite SVD fit")
    params = {"method":METHOD,"d":y.shape[1],"target":int(target),
              "coefficients":beta,"residual_sd":sigma,"residual_sd_floor":float(residual_sd_floor)}
    floor_active = unconstrained_sd <= residual_sd_floor
    # Score of the mean NLL in beta and log(sigma), projected at the SD floor.
    gradient_beta = -(x.T@residual)/(len(y)*sigma**2)
    gradient_log_sigma = 1-rss/(len(y)*sigma**2)
    projected_sigma = min(gradient_log_sigma, 0.) if floor_active else gradient_log_sigma
    pg = float(max(np.max(np.abs(gradient_beta)),abs(projected_sigma)))
    full_rank = int(rank) == x.shape[1]
    condition = float(singular[0]/singular[-1]) if full_rank and singular[-1] > 0 else float("inf")
    density = conditional_log_density(params, y)
    if not np.isfinite(density).all():
        raise FloatingPointError("Nonfinite fitted conditional density")
    record = {"method":METHOD,"conditional_only":True,"failed":False,"optimizer_success":True,
        "solver":"numpy.linalg.lstsq SVD; rcond=None; Gaussian MLE residual SD",
        "n_train":len(y),"d":y.shape[1],"target":int(target),"design_rank":int(rank),
        "design_columns":x.shape[1],"rank_deficient":not full_rank,"design_condition":condition,
        "singular_values":singular,"parameter_count":int(rank)+1,
        "nominal_parameter_count":y.shape[1]+1,"residual_sum_squares":rss,
        "unconstrained_residual_sd":unconstrained_sd,"residual_sd":sigma,
        "residual_sd_floor":float(residual_sd_floor),"residual_floor_active":floor_active,
        "boundary_hit":floor_active,"boundary_hits":["residual_sd_floor"] if floor_active else [],
        "score_converged":pg <= gtol,"projected_gradient":pg,"score_tolerance":float(gtol),
        "mean_train_conditional_log_density":float(density.mean()),
        "fit_seconds":time.perf_counter()-started,"starts_attempted":1,"starts_failed":0}
    return ConditionalLogLinearFit(params,record)
