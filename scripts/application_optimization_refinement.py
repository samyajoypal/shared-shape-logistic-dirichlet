"""Training-only continuation with identical bounds and original score checks.

Positive LD coordinates are optimized on a log scale; Gaussian loadings use
u=-log(1-r). These are bijections of the existing boxes, not different models.
The original finite parameter point is always retained. No evaluation sample
is accepted by this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
import traceback
import warnings

import numpy as np
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

import gaussian_shared_correlation as gaussian
import run_confirmatory_experiments as base
import shared_shape_ld as shared


DEFAULT_POLICY = {'original_gtol': 1e-6, 'transformed_gtol': 1e-12,
                  'transformed_maxiter': 3000, 'newton_maxiter': 60,
                  'raw_maxiter': 2000, 'ftol': 1e-15, 'maxls': 80,
                  'maxcor': 30, 'start_timeout_seconds': 120.,
                  'hessian_relative_step': float(np.finfo(float).eps**(1/3)),
                  'hessian_floor_fraction': 1e-7, 'maximum_transformed_step': 2.,
                  'line_search_steps': 35, 'training_likelihood_tie_tolerance': 1e-12}


class Converged(Exception):
    pass


@dataclass
class Coordinates:
    method: str
    k: int
    d: int
    raw_bounds: list

    def __post_init__(self):
        n = len(self.raw_bounds)
        self.positive = np.zeros(n, dtype=bool)
        self.correlation = np.zeros(n, dtype=bool)
        if self.method in ['shared_ld', 'full_ld']:
            shapes = self.d+1 if self.method == 'shared_ld' else self.k*(self.d+1)
            self.positive[self.k-1:self.k-1+shapes+self.k*self.d] = True
        elif self.method == gaussian.METHOD:
            c = gaussian.correlation_parameter_count(self.d)
            if c:
                self.correlation[-c:] = True
        else:
            raise ValueError('Unsupported refinement family')
        self.lower = np.array([-np.inf if lo is None else lo for lo, _ in self.raw_bounds])
        self.upper = np.array([np.inf if hi is None else hi for _, hi in self.raw_bounds])
        self.q_lower = self.to_transformed(self.lower, finite_only=False)
        self.q_upper = self.to_transformed(self.upper, finite_only=False)
        self.transformed_bounds = [(None if np.isneginf(lo) else float(lo),
                                    None if np.isposinf(hi) else float(hi))
                                   for lo, hi in zip(self.q_lower, self.q_upper)]

    def to_transformed(self, theta, finite_only=True):
        theta = np.asarray(theta, float)
        if theta.shape != self.positive.shape or (finite_only and not np.isfinite(theta).all()):
            raise ValueError('Invalid raw parameter point')
        if finite_only and (np.any(theta < self.lower) or np.any(theta > self.upper)):
            raise ValueError('Saved point lies outside the unchanged box; never clipped')
        result = theta.copy()
        result[self.positive] = np.log(result[self.positive])
        result[self.correlation] = -np.log1p(-result[self.correlation])
        return result

    def to_raw(self, q):
        q = np.asarray(q, float)
        if q.shape != self.positive.shape or not np.isfinite(q).all():
            raise ValueError('Invalid transformed parameter point')
        tolerance = 32*np.finfo(float).eps*(1+np.maximum(np.abs(q), 1))
        if np.any(q < self.q_lower-tolerance) or np.any(q > self.q_upper+tolerance):
            raise ValueError('Transformed optimizer point is outside its identical box')
        result = q.copy()
        result[self.positive] = np.exp(result[self.positive])
        result[self.correlation] = -np.expm1(-result[self.correlation])
        # Remove only floating-point roundoff at mathematically identical endpoints.
        return np.minimum(np.maximum(result, self.lower), self.upper)

    def derivative(self, theta):
        result = np.ones(len(theta))
        result[self.positive] = theta[self.positive]
        result[self.correlation] = 1-theta[self.correlation]
        return result


def objective(theta, y, k, method):
    if method == gaussian.METHOD:
        return gaussian.objective(theta, y, k)
    ell, _, score = base.ld_evaluate(theta, y, k, method, gradient=True)
    return -float(ell.mean()), -score/len(y)


def diagnostics(theta, y, k, method, limits, gtol):
    value, gradient = objective(theta, y, k, method)
    if not np.isfinite(value) or not np.isfinite(gradient).all():
        raise FloatingPointError('Nonfinite training objective or score')
    pg = shared.projected_gradient(theta, gradient, limits)
    hits = (gaussian.boundary_hits(theta, k, y.shape[1]) if method == gaussian.METHOD
            else shared.bound_hits(theta, limits))
    return {'mean_train_log_density': -float(value), 'projected_gradient': float(pg),
            'score_converged': bool(pg <= gtol), 'boundary_hits': hits,
            'boundary_hit': bool(hits), 'objective_gradient': gradient}, value, gradient


def finite_difference_hessian(q, fun, free, limits, relative_step):
    _, gradient = fun(q)
    hessian = np.zeros((len(free), len(free)))
    for column, index in enumerate(free):
        step = relative_step*max(1., abs(q[index]))
        lo, hi = limits[index]
        left = step if lo is None else min(step, q[index]-lo)
        right = step if hi is None else min(step, hi-q[index])
        # Keep the derivative inside the same box, including an active endpoint.
        if left >= step*.5 and right >= step*.5:
            h = min(left, right)
            plus, minus = q.copy(), q.copy(); plus[index] += h; minus[index] -= h
            column_gradient = (fun(plus)[1]-fun(minus)[1])/(2*h)
        elif right > 1e-12:
            plus = q.copy(); plus[index] += right
            column_gradient = (fun(plus)[1]-gradient)/right
        elif left > 1e-12:
            minus = q.copy(); minus[index] -= left
            column_gradient = (gradient-fun(minus)[1])/left
        else:
            raise ArithmeticError('No feasible numerical-Hessian step')
        hessian[:, column] = column_gradient[free]
    return (hessian+hessian.T)/2


def newton_polish(initial, fun, coordinates, stop, policy):
    q = initial.copy(); lower, upper = coordinates.q_lower, coordinates.q_upper
    history = []
    for iteration in range(policy['newton_maxiter']):
        value, gradient = fun(q)
        if stop(q):
            break
        threshold = 1e-10*(1+np.abs(q))
        active = ((q-lower <= threshold) & (gradient > 0)) | ((upper-q <= threshold) & (gradient < 0))
        free = np.flatnonzero(~active)
        if not len(free):
            break
        hessian = finite_difference_hessian(q, fun, free, coordinates.transformed_bounds,
                                            policy['hessian_relative_step'])
        eigenvalues, eigenvectors = np.linalg.eigh(hessian)
        floor = max(1e-9, policy['hessian_floor_fraction']*np.max(np.abs(eigenvalues)))
        direction = np.zeros(len(q))
        direction[free] = -eigenvectors@((eigenvectors.T@gradient[free])/np.maximum(eigenvalues, floor))
        largest = np.max(np.abs(direction))
        if largest > policy['maximum_transformed_step']:
            direction *= policy['maximum_transformed_step']/largest
        accepted = False; used_fallback = False
        for fallback in [False, True]:
            if fallback:
                used_fallback = True
                direction[:] = 0
                direction[free] = -gradient[free]/np.maximum(np.abs(np.diag(hessian)), floor)
                largest = np.max(np.abs(direction))
                if largest > policy['maximum_transformed_step']:
                    direction *= policy['maximum_transformed_step']/largest
            for power in range(policy['line_search_steps']):
                candidate = np.clip(q+(0.5**power)*direction, lower, upper)
                step = candidate-q; slope = float(gradient@step)
                if slope >= 0 or not np.any(step):
                    continue
                candidate_value, _ = fun(candidate)
                if candidate_value <= value+1e-4*slope+1e-14:
                    q = candidate; accepted = True; break
            if accepted:
                break
        history.append({'iteration': iteration, 'objective': float(value), 'accepted': accepted,
                        'minimum_hessian_eigenvalue': float(eigenvalues.min()),
                        'hessian_floor': floor, 'fallback_direction': used_fallback})
        if not accepted:
            break
    return q, history


def refine_point(theta, y, k, method, settings, policy=None):
    policy = {**DEFAULT_POLICY, **(policy or {})}
    y = np.asarray(y, float); theta = np.asarray(theta, float)
    limits = gaussian.bounds(k, y.shape[1]) if method == gaussian.METHOD else base.ld_bounds(method, k, y.shape[1], settings)
    coordinates = Coordinates(method, k, y.shape[1], limits)
    q0 = coordinates.to_transformed(theta)
    initial, initial_value, initial_gradient = diagnostics(theta, y, k, method, limits, policy['original_gtol'])
    best_theta, best_value = theta.copy(), initial_value
    best_pg = initial['projected_gradient']
    lowest_value, lowest_theta, lowest_pg = initial_value, theta.copy(), best_pg
    record = {'original_parameters': theta.copy(), 'original_diagnostics': initial,
              'stages': [], 'exceptions': [], 'failed': False}
    started = time.perf_counter()

    def raw_fun(point):
        nonlocal best_theta, best_value, best_pg, lowest_value, lowest_theta, lowest_pg
        value, gradient = objective(point, y, k, method)
        if not np.isfinite(value) or not np.isfinite(gradient).all():
            raise FloatingPointError('Nonfinite objective during refinement')
        pg = shared.projected_gradient(point, gradient, limits)
        if value < lowest_value:
            lowest_theta, lowest_value, lowest_pg = point.copy(), float(value), pg
        if best_value > lowest_value+policy['training_likelihood_tie_tolerance']:
            best_theta, best_value, best_pg = lowest_theta.copy(), lowest_value, lowest_pg
        if value <= lowest_value+policy['training_likelihood_tie_tolerance']:
            if pg < best_pg or (pg == best_pg and value < best_value):
                best_theta, best_value, best_pg = point.copy(), float(value), pg
        return value, gradient

    def transformed_fun(q):
        raw = coordinates.to_raw(q)
        value, gradient = raw_fun(raw)
        return value, gradient*coordinates.derivative(raw)

    def converged(q):
        raw = coordinates.to_raw(q)
        _, gradient = raw_fun(raw)
        return shared.projected_gradient(raw, gradient, limits) <= policy['original_gtol']

    def callback(q):
        if converged(q):
            raise Converged()

    with threadpool_limits(1):
        if not initial['score_converged']:
            try:
                with base.timed_start(policy['start_timeout_seconds']):
                    stage_start = time.perf_counter()
                    try:
                        result = minimize(transformed_fun, q0, jac=True, method='L-BFGS-B',
                            bounds=coordinates.transformed_bounds, callback=callback,
                            options={'maxiter': policy['transformed_maxiter'], 'maxfun': 200000,
                                     'gtol': policy['transformed_gtol'], 'ftol': policy['ftol'],
                                     'maxls': policy['maxls'], 'maxcor': policy['maxcor']})
                        endpoint = coordinates.to_raw(result.x)
                        record['stages'].append({'stage': 'transformed_LBFGSB', 'message': str(result.message),
                            'iterations': int(result.nit), 'endpoint_theta': endpoint,
                            'elapsed_seconds': time.perf_counter()-stage_start})
                    except Converged:
                        record['stages'].append({'stage': 'transformed_LBFGSB', 'message': 'Original score criterion reached',
                            'elapsed_seconds': time.perf_counter()-stage_start})
                    q = coordinates.to_transformed(best_theta)
                    if not converged(q):
                        stage_start = time.perf_counter()
                        endpoint, history = newton_polish(q, transformed_fun, coordinates, converged, policy)
                        raw_fun(coordinates.to_raw(endpoint))
                        record['stages'].append({'stage': 'safeguarded_Newton', 'history': history,
                            'endpoint_theta': coordinates.to_raw(endpoint), 'elapsed_seconds': time.perf_counter()-stage_start})
                    if not converged(coordinates.to_transformed(best_theta)):
                        stage_start = time.perf_counter()
                        result = minimize(raw_fun, best_theta, jac=True, method='L-BFGS-B', bounds=limits,
                            options={'maxiter': policy['raw_maxiter'], 'maxfun': 200000,
                                     'gtol': policy['original_gtol'], 'ftol': policy['ftol'],
                                     'maxls': policy['maxls'], 'maxcor': policy['maxcor']})
                        raw_fun(result.x)
                        record['stages'].append({'stage': 'original_coordinate_LBFGSB',
                            'message': str(result.message), 'iterations': int(result.nit),
                            'endpoint_theta': result.x.copy(), 'elapsed_seconds': time.perf_counter()-stage_start})
            except Exception as exc:
                record['exceptions'].append({'exception': repr(exc), 'exception_type': type(exc).__name__,
                                             'traceback': traceback.format_exc()})
    final, value, gradient = diagnostics(best_theta, y, k, method, limits, policy['original_gtol'])
    record.update(final)
    record.update(theta=best_theta, elapsed_seconds=time.perf_counter()-started,
                  training_mean_log_density_gain=initial_value-value,
                  highest_mean_train_log_density_evaluated=-lowest_value,
                  selected_training_gap_to_best=value-lowest_value,
                  original_candidate_retained=True, optimizer_success=final['score_converged'])
    if value > initial_value+1e-12:
        raise ArithmeticError('Refinement lost the saved training likelihood')
    return record


def saved_endpoints(saved):
    """All distinct finite saved endpoints, including failed optimizer endpoints."""
    points = []; seen = set(); omitted = []
    def add(value, origin):
        if value is None:
            return
        theta = np.asarray(value, float)
        if theta.ndim != 1 or not np.isfinite(theta).all():
            omitted.append({'origin': origin, 'reason': 'Nonfinite or malformed saved endpoint',
                            'saved_theta': theta})
            return
        key = (theta.shape, theta.tobytes())
        if key not in seen:
            seen.add(key); points.append((origin, theta.copy()))
    params = saved.get('parameters')
    if params is not None:
        add(params.get('theta'), 'original_selected_point')
    for index, start in enumerate(saved.get('starts', [])):
        params = start.get('parameters')
        if params is not None:
            add(params.get('theta'), f'saved_start_{index}_retained_point')
        add(start.get('optimizer_endpoint_theta'), f'saved_start_{index}_optimizer_endpoint')
        endpoint = start.get('optimizer_endpoint_parameters')
        if endpoint is not None:
            add(endpoint.get('theta'), f'saved_start_{index}_optimizer_endpoint_parameters')
    return points, omitted


def refine_saved(saved, y, k, method, settings, policy=None, refined_shared=None):
    """Refine every saved endpoint; training values alone choose the result."""
    policy = {**DEFAULT_POLICY, **(policy or {})}; d = y.shape[1]
    points, omitted = saved_endpoints(saved)
    dependency_seconds = 0.
    if method == 'full_ld' and refined_shared is not None:
        points.append(('embedded_refined_shared', base.embed_shared(
            np.asarray(refined_shared.params['theta']), k, d)))
        dependency_seconds = refined_shared.record['fit_seconds']
    records = []; candidates = []; beginning = time.perf_counter()
    for origin, initial in points:
        started = time.perf_counter()
        try:
            result = refine_point(initial, y, k, method, settings, policy)
            params = (gaussian.parameters(result['theta'], k, d) if method == gaussian.METHOD
                      else {'method': method, 'k': k, 'd': d, 'theta': result['theta']})
            result.update(origin=origin, start=len(records), selected=False, parameters=params)
            candidates.append(len(records))
        except Exception as exc:
            result = {'origin': origin, 'start': len(records), 'selected': False, 'failed': True,
                      'original_parameters': initial, 'parameters': None, 'exception': repr(exc),
                      'exception_type': type(exc).__name__,
                      'traceback': traceback.format_exc(), 'elapsed_seconds': time.perf_counter()-started}
        records.append(result)
    if not candidates:
        raise base.FitFailure('No finite saved parameter point available for refinement', records)
    highest = max(records[i]['mean_train_log_density'] for i in candidates)
    eligible = [i for i in candidates if records[i]['mean_train_log_density'] >=
                highest-policy['training_likelihood_tie_tolerance']]
    selected = min(eligible, key=lambda i: (records[i]['projected_gradient'],
                                          -records[i]['mean_train_log_density'], i))
    records[selected]['selected'] = True
    chosen = records[selected]
    if refined_shared is not None and method == 'full_ld':
        if chosen['mean_train_log_density'] < refined_shared.record['mean_train_log_density']-1e-10:
            raise ArithmeticError('Full refinement lost its retained nested common-shape point')
    elapsed = time.perf_counter()-beginning
    record = {key: chosen[key] for key in ['mean_train_log_density','projected_gradient','score_converged',
                                         'optimizer_success','boundary_hits','boundary_hit']}
    record.update(fit_seconds=elapsed, dependency_fit_seconds=dependency_seconds,
                  policy_fit_seconds=elapsed+dependency_seconds, starts_attempted=len(records),
                  starts_failed=sum(r['failed'] for r in records),
                  retained_starts_with_exceptions=sum(not r['failed'] and bool(r.get('exceptions')) for r in records),
                  optimizer_exception_count=sum(len(r.get('exceptions', [])) for r in records),
                  timeout_exception_count=sum(sum(e.get('exception_type')=='TimeoutError' for e in r.get('exceptions', []))
                                              +int(r.get('exception_type')=='TimeoutError') for r in records),
                  omitted_saved_endpoints=len(omitted),
                  original_points_omitted=omitted, parameter_count=len(chosen['theta']),
                  highest_training_mean_log_density=highest,
                  selection_tie_tolerance=policy['training_likelihood_tie_tolerance'],
                  original_start_records_retained=True,
                  min_weight=float(np.min(gaussian.parameters(chosen['theta'],k,d)['weights']))
                      if method == gaussian.METHOD else
                      float(np.exp(base.unpack_ld(chosen['theta'],method,k,d)[0]).min()))
    if method == gaussian.METHOD:
        return gaussian.GaussianSharedCorrelationFit(chosen['parameters'],record,records,selected,elapsed)
    return base.FitResult(method,k,d,chosen['parameters'],record,records,selected)
