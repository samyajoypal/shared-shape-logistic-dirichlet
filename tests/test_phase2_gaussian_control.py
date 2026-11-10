"""Frozen pairing, original measures, and failed-checkpoint accounting."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from threadpoolctl import threadpool_limits

import phase2_gaussian_control as control
import phase2_simulations as primary
import run_confirmatory_experiments as baseline


def read(path):
    return json.loads(Path(path).read_text())


@pytest.fixture
def paired(tmp_path):
    config = read(primary.CONFIG)
    cell = dict(config['cells'][0], id='control_regression', n_train=30, repetitions=2)
    config.update(cells=[cell], cell_count=1, dataset_count=2, n_test=25)
    config['primary_methods'] = ['gaussian_diag']
    config['secondary'] = dict.fromkeys(config['secondary'], 0)
    config['fitting'].update(maxiter=25, refinement_maxiter=10)
    main = tmp_path/'primary'
    primary.freeze_run(config, main)
    primary.run_task(config, 0, main)
    companion = tmp_path/'control'
    manifest = control.freeze(companion, main, 'simulations')
    return SimpleNamespace(primary=main, output=companion, config=config, manifest=manifest,
                           directory=companion/'tasks/task_000000')


def exact_diagonal_fit(y, k, seed, *, diagonal_candidate, **kwargs):
    assert set(kwargs) == {'partitions', 'n_starts', 'maxiter', 'polish_maxiter',
                           'gtol', 'ftol', 'start_timeout_seconds'}
    candidate = diagonal_candidate
    params = {'method': control.METHOD, 'k': k, 'd': y.shape[1],
              'weights': candidate['weights'], 'means': candidate['means'],
              'std': np.sqrt(candidate['covariances']), 'correlation': np.eye(y.shape[1])}
    ell = control.gaussian_density(params, y)[0]
    record = {'mean_train_log_density': float(ell.mean()), 'parameter_count': 11,
              'optimizer_success': True, 'score_converged': True, 'boundary_hit': True,
              'projected_gradient': 0., 'correlation_loading_identified': True,
              'minimum_correlation_eigenvalue': 1.}
    return SimpleNamespace(params=params, record=record, starts=[{'failed': False, 'parameters': params}],
                           selected_start=0)


def test_simulation_map_complete_and_application_scope():
    spec = {'configuration': read(primary.CONFIG)}
    mapping = control.task_map(spec, 'simulations')
    assert len(mapping) == 10500
    assert all(x['task_index'] == x['primary_task_index'] for x in mapping)
    kinds = ['cv', 'full', 'bootstrap', 'leave_area_out']
    tasks = []
    for variant in ['primary', 'secondary']:
        for kind in kinds:
            for bounds in ['primary', 'wide']:
                tasks.append(dict(task_index=len(tasks), variant=variant, kind=kind,
                                  bounds=bounds, repeat=0, fold=0))
    spec = {'config': {'variants': [{'id': 'primary', 'primary': True},
                                    {'id': 'secondary', 'primary': False}]}, 'tasks': tasks}
    mapping = control.task_map(spec, 'applications')
    assert [x['primary_task_index'] for x in mapping] == [0, 2, 4]
    assert [x['task_index'] for x in mapping] == [0, 1, 2]


@pytest.mark.parametrize('d', [1, 2, 5])
def test_joint_and_marginal_match_independent_scipy(d):
    rng = np.random.default_rng(235+d)
    loadings = rng.uniform(.2, .7, d)
    correlation = np.diag(1-loadings**2)+np.outer(loadings, loadings)
    params = {'weights': [.3, .7], 'means': rng.normal(size=(2, d)),
              'std': rng.uniform(.3, 2, (2, d)), 'correlation': correlation}
    y = rng.normal(size=(30, d))
    for keep in [list(range(d)), list(range(1, d)), []]:
        got, tau = control.gaussian_density(params, y, keep)
        if keep:
            terms = []
            for j in range(2):
                std = params['std'][j, keep]
                cov = std[:, None]*correlation[np.ix_(keep, keep)]*std
                terms.append(np.log(params['weights'][j])+multivariate_normal.logpdf(
                    y[:, keep], mean=params['means'][j, keep], cov=cov))
            expected = logsumexp(np.column_stack(terms), axis=1)
        else:
            expected = np.zeros(len(y))
            np.testing.assert_allclose(tau, np.tile(params['weights'], (len(y), 1)))
        np.testing.assert_allclose(got, expected, atol=2e-13)
        np.testing.assert_allclose(tau.sum(axis=1), 1., atol=2e-14)


@pytest.mark.parametrize('log_transform', [False, True])
def test_raw_measure_joint_conditional_and_truth_only_evaluation(log_transform):
    params = {'weights': [1.], 'means': [[-.4, .7]], 'std': [[.5, 1.4]],
              'correlation': [[1., .3], [.3, 1.]]}
    y = np.array([[1.1, 2.3], [1.4, 2.8], [2.1, 3.2]])
    center, scale = np.array([1., 2.]), np.array([.2, .4])
    jacobian = y if log_transform else np.zeros_like(y)
    mean = center+scale*np.asarray(params['means'])[0]
    std = scale*np.asarray(params['std'])[0]
    cov = std[:, None]*np.asarray(params['correlation'])*std
    case = {'test': y, 'center': center, 'scale': scale, 'raw_jacobian': jacobian,
            'labels': np.zeros(len(y)), 'true_means': None, 'conditional_target': 'second', 'target_index': 1}
    metrics, ell, tau, means, conditional = control.evaluate_fit(params, case)
    expected = multivariate_normal.logpdf(y, mean=mean, cov=cov)-jacobian.sum(axis=1)
    marginal = multivariate_normal.logpdf(y[:, 0], mean=mean[0], cov=cov[0, 0])-jacobian[:, 0]
    np.testing.assert_allclose(ell, expected, atol=2e-13)
    np.testing.assert_allclose(conditional, expected-marginal, atol=2e-13)
    changed = dict(case, true_means=np.array([[900., 800.]]), true_weights=np.ones(1), oracle=np.zeros(len(y)))
    second = control.evaluate_fit(params, changed)
    np.testing.assert_array_equal(second[1], ell)
    np.testing.assert_array_equal(second[2], tau)
    assert metrics['mean_rmse'] is None and second[0]['mean_rmse'] > 700


def test_pending_success_resume_and_same_diagonal_prediction(paired):
    pending = control.run_task(paired.output, 1, fit_function=exact_diagonal_fit)
    assert pending['status'] == 'dependency_pending'
    assert not (paired.output/'tasks/task_000001/result.json').exists()
    result = control.run_task(paired.output, 0, fit_function=exact_diagonal_fit)
    assert not result['row']['failed'] and result['row']['evaluation_available']
    assert result['row']['diagonal_dependency_source'] == 'primary_fit'
    assert result['row']['test_log_density'] == pytest.approx(
        result['baseline_records']['gaussian_diag']['test_log_density'], abs=1e-12)
    def forbidden(*args, **kwargs):
        raise AssertionError('Completed checkpoint should not refit')
    assert control.run_task(paired.output, 0, fit_function=forbidden) == result
    status = control.aggregate(paired.output)
    assert status['available'] == 1 and status['missing_task_indices'] == [1]
    pair = pd.read_csv(paired.output/'paired_comparisons.csv').iloc[0]
    assert pair.test_log_density_baseline_minus_control == pytest.approx(0., abs=1e-12)


@pytest.mark.parametrize('stage', ['control_fit.json', 'diagonal_dependency.json'])
def test_foreign_checkpoint_rejected_without_completed_result(paired, stage):
    control.run_task(paired.output, 0, fit_function=exact_diagonal_fit)
    (paired.directory/'result.json').unlink()
    path = paired.directory/stage
    saved = read(path); saved['primary_task_index'] += 1
    control.atomic_json(path, saved)
    with pytest.raises(control.IntegrityError, match='different task'):
        control.run_task(paired.output, 0, fit_function=exact_diagonal_fit)
    assert not (paired.directory/'result.json').exists()


def test_selected_fit_checkpoint_resumes_without_refit(paired):
    first = control.run_task(paired.output, 0, fit_function=exact_diagonal_fit)
    (paired.directory/'result.json').unlink()
    def forbidden(*args, **kwargs):
        raise AssertionError('Finite saved fit should not be recomputed')
    second = control.run_task(paired.output, 0, fit_function=forbidden)
    assert second['parameters'] == first['parameters']
    assert second['row']['test_log_density'] == first['row']['test_log_density']


def test_storage_failure_is_failed_with_finite_fit_retained(paired, monkeypatch):
    def fails(*args, **kwargs):
        raise OSError('Injected prediction storage failure')
    monkeypatch.setattr(control.np, 'savez_compressed', fails)
    result = control.run_task(paired.output, 0, fit_function=exact_diagonal_fit)
    assert result['row']['failed'] and not result['row']['evaluation_available']
    assert result['row']['fit_available'] and result['parameters'] is not None
    assert len(read(paired.directory/'control_fit.json')['starts']) == 1
    status = control.aggregate(paired.output)
    assert status['available'] == 0 and status['failed_recorded'] == 1
    summary = pd.read_csv(paired.output/'summary.csv').iloc[0]
    assert summary.planned == 2 and summary.missing == 1
    assert np.isnan(summary.test_log_density_mean)


def test_total_fit_failure_preserves_attempts_and_all_denominators(paired):
    def fails(*args, **kwargs):
        raise baseline.FitFailure('Injected failures', [{'failed': True, 'start': j} for j in range(3)])
    result = control.run_task(paired.output, 0, fit_function=fails)
    assert result['row']['failed'] and not result['row']['fit_available']
    assert result['row']['starts_attempted'] == result['row']['starts_failed'] == 3
    assert len(read(paired.directory/'control_fit.json')['starts']) == 3
    status = control.aggregate(paired.output)
    assert status['planned'] == 2 and status['failed_recorded'] == 1 and status['available'] == 0


@pytest.mark.parametrize('target', ['config', 'source', 'input'])
def test_changed_archived_or_paired_input_rejected(paired, target):
    control.run_task(paired.output, 0, fit_function=exact_diagonal_fit)
    if target == 'config':
        path = paired.output/'source_snapshot/configs/phase2_gaussian_control.json'
        value = read(path); value['threads'] = 2; control.atomic_json(path, value)
    elif target == 'source':
        path = paired.output/'source_snapshot/scripts/gaussian_shared_correlation.py'
        path.write_text(path.read_text()+'\n# changed\n')
    else:
        path = paired.primary/'tasks/task_000000/attempt_001/generation.json'
        value = read(path); value['unrelated_but_changed'] = True; control.atomic_json(path, value)
    with pytest.raises(RuntimeError, match='hash mismatch|source changed|inputs changed'):
        control.run_task(paired.output, 0, fit_function=exact_diagonal_fit)
    with pytest.raises(RuntimeError):
        control.aggregate(paired.output)


def test_upstream_cost_fallback_and_unknown_cost():
    assert control.upstream_cost({'row': {'fit_seconds': 3.}}) == (3., 'upstream_fit_seconds')
    assert control.upstream_cost({'row': {'failed': True}, 'starts': [{'elapsed_seconds': 1.}, {'fit_seconds': 2.}]}) == (
        3., 'sum_upstream_start_seconds_excludes_wrapper')
    assert control.upstream_cost({'row': {'total_seconds': 4.}}) == (4., 'upstream_total_seconds_including_evaluation')
    assert control.upstream_cost({'row': {'failed': True}, 'starts': [{}]}) == (None, 'upstream_cost_unavailable')


def test_unreported_dependency_cost_is_not_zero(paired):
    path = paired.primary/'tasks/task_000000/attempt_001/primary__gaussian_diag.json'
    value = read(path); value['row'].pop('fit_seconds'); value['starts'] = []
    control.atomic_json(path, value)
    result = control.run_task(paired.output, 0, fit_function=exact_diagonal_fit)
    assert result['row']['evaluation_available']
    assert result['row']['dependency_fit_seconds'] is None
    assert result['row']['policy_fit_seconds'] is None


def test_all_failed_bootstrap_targets_keep_denominator_and_log_labels(tmp_path):
    tasks = [{'task_index': 0, 'kind': 'full', 'variant': 'crabs'}]+[
        {'task_index': j+1, 'kind': 'bootstrap', 'variant': 'crabs', 'repeat': j} for j in range(2)]
    reference = {'row': {'failed': False}, 'profiles': {'means_in_analysis_coordinates': [[1., 2.], [2., 4.]],
        'original_data_sd': [1., 1.], 'coordinate_names': ['FL', 'BD'], 'analysis_transform': 'log_measurements'}}
    control.bootstrap_stability({'tasks': tasks}, {0: reference}, tmp_path)
    summary = pd.read_csv(tmp_path/'bootstrap_stability.csv')
    assert len(summary) == 4 and summary.planned.eq(2).all() and summary.available.eq(0).all()
    assert summary.unavailable.eq(2).all() and summary.target.str.startswith('mean_log_').all()
    assert summary.target_scale.eq('mean of natural log length in millimetres').all()
    assert summary.boundary_checked.eq(0).all() and summary.score_checked.eq(0).all()


def test_actual_control_small_fit_runs_and_keeps_diagonal_point(paired):
    with threadpool_limits(1):
        result = control.run_task(paired.output, 0)
    assert not result['row']['failed']
    assert result['row']['train_log_density'] >= result['baseline_records']['gaussian_diag']['train_log_density']-1e-12
    assert result['row']['density_identity_error'] < 1e-10
    fit = read(paired.directory/'control_fit.json')
    assert fit['starts'] and all('failed' in start for start in fit['starts'])


@pytest.fixture
def application_pair(tmp_path):
    main = tmp_path/'application_primary'
    raw_path = main/'executed_sources/data/crabs.csv'
    raw_path.parent.mkdir(parents=True)
    rng = np.random.default_rng(441)
    raw = np.exp(rng.normal(size=(24, 2))*.2+np.array([2., 2.8]))
    pd.DataFrame({'FL': raw[:, 0], 'BD': raw[:, 1], 'sp': ['B']*12+['O']*12,
                  'sex': ['M', 'F']*12}).to_csv(raw_path, index=False)
    variant = {'id': 'crabs_test', 'primary': True, 'dataset': 'crabs', 'columns': ['FL', 'BD'],
               'transform': 'log', 'k': 2, 'conditional_target': 'BD'}
    tasks = [dict(task_index=0, variant='crabs_test', kind='cv', repeat=0, fold=0, bounds='primary',
                  train=list(range(18)), test=list(range(18, 24)), seed=123),
             dict(task_index=1, variant='crabs_test', kind='full', repeat=0, fold=0, bounds='primary',
                  train=list(range(24)), test=list(range(24)), seed=124),
             dict(task_index=2, variant='crabs_test', kind='bootstrap', repeat=0, fold=0, bounds='primary',
                  train=list(range(20))+[0, 1, 2, 3], test=list(range(24)), seed=125)]
    spec = {'runtime_versions': control.versions(), 'config': {'variants': [variant], 'folds': 1,
            'fitting': {**baseline.DEFAULT_SETTINGS, 'n_starts': 1, 'maxiter': 25, 'refinement_maxiter': 10}},
            'tasks': tasks, 'data_hashes': {'data/crabs.csv': control.digest(raw_path)}}
    control.atomic_json(main/'specification.json', spec)
    for task in tasks:
        directory = main/'tasks'/f"{task['task_index']:05d}"
        y = np.log(raw); train = y[task['train']]
        center, scale = train.mean(axis=0), train.std(axis=0)
        partitions = [np.arange(len(train)) % 2]
        control.atomic_json(directory/'split.json', {**task, 'center': center, 'scale': scale, 'partitions': partitions})
        if task['kind'] != 'bootstrap':
            params = {'weights': [.4, .6], 'means': [[-.2, -.1], [.2, .1]], 'covariances': [[1., .9], [.8, 1.]]}
            correlated = {'weights': params['weights'], 'means': params['means'],
                          'std': np.sqrt(params['covariances']), 'correlation': np.eye(2)}
            case = {'test': y[task['test']], 'center': center, 'scale': scale,
                    'raw_jacobian': y[task['test']], 'labels': np.zeros(len(task['test'])),
                    'true_means': None, 'conditional_target': 'BD', 'target_index': 1}
            metrics = control.evaluate_fit(correlated, case)[0]
            row = {**task, 'method': 'gaussian_diag', 'failed': False, 'fit_seconds': .02,
                   'n_test': len(task['test']), 'mean_log_density': metrics['test_log_density'],
                   'mean_conditional_log_density': metrics['conditional_log_density']}
            control.atomic_json(directory/'gaussian_diag.json', {'parameters': params, 'row': row, 'starts': []})
            records = [row]
        else:
            records = []
        control.atomic_json(directory/'completion.json', {'task': task, 'records': records,
                            'specification_sha256': control.digest(main/'specification.json')})
    out = tmp_path/'application_control'
    control.freeze(out, main, 'applications')
    return SimpleNamespace(primary=main, output=out)


def test_application_cv_full_bootstrap_pairing_and_original_measure(application_pair):
    for task_index in range(3):
        result = control.run_task(application_pair.output, task_index, fit_function=exact_diagonal_fit)
        assert not result['row']['failed']
        if task_index < 2:
            assert result['row']['diagonal_dependency_source'] == 'primary_fit'
            assert result['row']['test_log_density'] == pytest.approx(
                result['baseline_records']['gaussian_diag']['mean_log_density'], abs=1e-12)
        else:
            assert result['row']['diagonal_dependency_source'] == 'companion_fit'
            assert result['row']['policy_fit_seconds'] > result['row']['fit_seconds']
        assert result['profiles']['analysis_transform'] == 'log_measurements'
    status = control.aggregate(application_pair.output)
    assert status['available'] == status['planned'] == 3
    paired = pd.read_csv(application_pair.output/'paired_comparisons.csv')
    np.testing.assert_allclose(paired.mean_baseline_minus_control, 0., atol=1e-12)
    stability = pd.read_csv(application_pair.output/'bootstrap_stability.csv')
    assert len(stability) == 4 and stability.available.eq(1).all()
    assert stability.boundary_hits.eq(1).all() and stability.score_passed.eq(1).all()
    assert stability.target.str.startswith('mean_log_').all()


def test_application_split_must_match_frozen_task(application_pair):
    path = application_pair.primary/'tasks/00000/split.json'
    value = read(path); value['seed'] += 1; control.atomic_json(path, value)
    with pytest.raises(RuntimeError, match='identity mismatch'):
        control.run_task(application_pair.output, 0, fit_function=exact_diagonal_fit)
