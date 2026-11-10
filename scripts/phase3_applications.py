"""Additional, prospectively frozen application panel after the phase2 study.

This adapter reuses validated density kernels without changing their sources.
All preprocessing/initialization is training-only. Group splits, whole-group
bootstrap samples, bounds and methods are frozen before fitting any panel data.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

import application_optimization_refinement as refine
import gaussian_shared_correlation as gaussian
import phase2_applications as apps
import phase2_gaussian_control as control
import regularized_skew_normal as skew
import run_confirmatory_experiments as base
import conditional_log_linear as conditional_model

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'configs/phase3_applications.json'
GC = gaussian.METHOD
CORE = ['shared_ld', 'full_ld', GC]
REAL_METHODS = ['gaussian_diag', 'gaussian_full', 'shared_ld', 'full_ld', GC,
                'fmvmm_t', skew.METHOD]
NATIVE_METHODS = ['fmvmm_dmm', 'base_ld']


def read(path):
    return json.loads(Path(path).read_text())


def source_files():
    return [ROOT/name for name in [
        'scripts/phase3_applications.py', 'scripts/application_optimization_refinement.py',
        'scripts/phase2_applications.py', 'scripts/phase2_gaussian_control.py',
        'scripts/gaussian_shared_correlation.py', 'scripts/run_confirmatory_experiments.py',
        'scripts/shared_shape_ld.py', 'scripts/regularized_skew_normal.py',
        'scripts/conditional_log_linear.py',
        'scripts/prepare_extended_application_data.py',
        'configs/experiments.json', 'requirements-lock.txt']]


def dataset_spec(spec):
    """Normalize only spelling aliases; never choose scientific settings."""
    return {**spec, 'id': spec.get('id', spec.get('dataset_id')),
            'csv': spec.get('csv', spec.get('positive_csv')),
            'columns': spec.get('columns', spec.get('feature_columns')),
            'reference': spec.get('reference', spec.get('reference_column')),
            'conditional_target': spec.get('conditional_target', spec.get('target_name'))}


def load_dataset(root, specification):
    spec = dataset_spec(specification)
    path = Path(root)/spec['csv']
    frame = pd.read_csv(path, keep_default_na=False, dtype={'row_id': str, 'group': str, 'label': str},
                        converters={name: float for name in spec['columns']})
    required = ['row_id', 'group', 'label']+spec['columns']
    if list(frame.columns) != required:
        raise ValueError('Prepared CSV feature/order contract differs: '+str(path))
    if frame.row_id.duplicated().any() or frame.row_id.eq('').any() or frame.group.eq('').any():
        raise ValueError('Missing/duplicate row ID or missing sampling group')
    raw = frame[spec['columns']].to_numpy(float)
    if not np.isfinite(raw).all() or np.any(raw <= 0):
        raise ValueError('All declared measurements must be finite and strictly positive')
    if spec['transform'] == 'alr':
        reference = spec['columns'].index(spec['reference'])
        keep = [j for j in range(raw.shape[1]) if j != reference]
        y = np.log(raw[:, keep])-np.log(raw[:, reference, None])
        names = [spec['columns'][j]+'/'+spec['reference'] for j in keep]
        raw_log_jacobian = np.zeros_like(y)
        measure = 'declared_ALR_coordinates'
    elif spec['transform'] == 'log':
        y = np.log(raw); names = list(spec['columns'])
        # Stored as the positive quantity to subtract for original-unit scores.
        raw_log_jacobian = y.copy(); measure = 'original_positive_measurement_units'
    else:
        raise ValueError('Only the prospectively declared log/ALR transforms are supported')
    if spec.get('analysis_coordinate_names', names) != names:
        raise ValueError('Analysis-coordinate names differ from the declared transformation')
    target = names.index(spec['conditional_target'])
    metadata = {'dataset_id': spec['id'], 'feature_columns': spec['columns'],
        'transform': spec['transform'], 'reference_column': spec['reference'],
        'analysis_coordinate_names': names, 'target_index': target, 'target_name': names[target],
        'positive_csv': spec['csv'], 'raw_data_sha256': control.digest(path),
        'group_column': 'group', 'row_id_column': 'row_id', 'label_column': 'label',
        'n_rows': len(frame), 'n_groups': frame.group.nunique(), 'k': int(spec['k']),
        'coordinate_measure': measure, 'row_ids': frame.row_id.tolist(),
        'group_ids': frame.group.tolist()}
    if spec.get('expected_rows', len(frame)) != len(frame) or spec.get('expected_groups', frame.group.nunique()) != frame.group.nunique():
        raise ValueError('Prepared row/group count differs from the frozen scientific specification')
    if type(spec['k']) is not int or not 1 <= spec['k'] < len(frame):
        raise ValueError('Invalid declared component count')
    return {'raw': raw, 'y': y, 'raw_log_jacobian': raw_log_jacobian,
            'labels': frame.label.to_numpy(), 'groups': frame.group.to_numpy(),
            'row_ids': frame.row_id.to_numpy(), 'metadata': metadata}


def group_folds(groups, folds, seed):
    """Random group order; greedily fill the smallest row-count fold.

    Ties between folds are broken by the same seeded generator. Groups remain
    indivisible, including when one group is larger than an ideal fold.
    """
    groups = np.asarray(groups, str)
    unique, counts = np.unique(groups, return_counts=True)
    if type(folds) is not int or not 2 <= folds <= len(unique):
        raise ValueError('Insufficient independent groups for declared folds')
    rng = np.random.default_rng(seed); order = rng.permutation(len(unique))
    loads = np.zeros(folds, int); mapping = {}
    for index in order:
        eligible = np.flatnonzero(loads == loads.min())
        fold = int(rng.choice(eligible)); mapping[unique[index]] = fold
        loads[fold] += counts[index]
    assigned = np.asarray([mapping[g] for g in groups], int)
    assert set(assigned) == set(range(folds))
    return assigned


def group_members(groups):
    unique, inverse, counts = np.unique(np.asarray(groups, str), return_inverse=True, return_counts=True)
    ordered = np.argsort(inverse, kind='stable')
    pieces = np.split(ordered, np.cumsum(counts)[:-1])
    return unique, dict(zip(unique, pieces))


def group_bootstrap(groups, seed):
    unique, members = group_members(groups)
    draws = np.random.default_rng(seed).choice(unique, size=len(unique), replace=True)
    train = np.concatenate([members[group] for group in draws])
    return train, draws.tolist()


def validate_split(split, dataset, task):
    n = len(dataset['y']); train = np.asarray(split['train'], int); test = np.asarray(split['test'], int)
    if not len(train) or not len(test) or np.any(train < 0) or np.any(train >= n) or np.any(test < 0) or np.any(test >= n):
        raise control.IntegrityError('Invalid frozen row indices')
    if len(np.unique(test)) != len(test):
        raise control.IntegrityError('Evaluation grid contains duplicate rows')
    groups = dataset['groups']
    if task['kind'] == 'cv':
        if len(np.unique(train)) != len(train) or set(groups[train]) & set(groups[test]):
            raise control.IntegrityError('Training/test group leakage')
        if set(train) | set(test) != set(range(n)):
            raise control.IntegrityError('CV split does not partition all rows')
    elif task['kind'] == 'bootstrap':
        draws = split['bootstrap_draw_groups']; unique, members = group_members(groups)
        if len(draws) != len(unique) or not set(draws) <= set(unique):
            raise control.IntegrityError('Malformed whole-group bootstrap draws')
        expected = np.concatenate([members[group] for group in draws])
        if not np.array_equal(train, expected) or not np.array_equal(test, np.arange(n)):
            raise control.IntegrityError('Bootstrap rows differ from sampled complete groups/original grid')
    elif task['kind'] == 'full':
        if not np.array_equal(train, np.arange(n)) or not np.array_equal(test, np.arange(n)):
            raise control.IntegrityError('Full-data task differs from the original grid')
    else:
        raise control.IntegrityError('Unknown task kind')


def build_plan(config, datasets):
    prediction = []; bootstrap = []
    seed = int(config['seed']); folds = int(config.get('folds', 5)); repeats = int(config.get('repeats', 10))
    for dataset_index, specification in enumerate(config['datasets']):
        spec = dataset_spec(specification); data = datasets[spec['id']]; n = len(data['y'])
        common = {'dataset_id': spec['id'], 'variant': spec['id']}
        assignments = [group_folds(data['groups'], folds, seed+100000*dataset_index+r) for r in range(repeats)]
        methods = REAL_METHODS+(NATIVE_METHODS if spec['transform'] == 'alr' else [])
        if set(spec.get('methods', methods)) != set(methods) or len(spec.get('methods', methods)) != len(methods):
            raise ValueError('Declared main method set differs from the supported fixed panel')
        for comparison, bounds, count, selected_methods, use_repeats in [
            ('primary', 'primary', spec['k'], methods, repeats),
            ('wide', 'wide', spec['k'], ['shared_ld', 'full_ld'], 1),
            ('kminus1', 'primary', spec['k']-1, CORE, 1),
            ('kplus1', 'primary', spec['k']+1, CORE, 1)]:
            if count < 1:
                raise ValueError('The predeclared k-minus-one sensitivity is invalid')
            for repeat in range(use_repeats):
                for fold in range(folds):
                    assigned = assignments[repeat]
                    split = {'train': np.flatnonzero(assigned != fold).tolist(),
                             'test': np.flatnonzero(assigned == fold).tolist(), 'bootstrap_draw_groups': [],
                             'group_split_seed': seed+100000*dataset_index+repeat, 'bootstrap_draw_seed': None}
                    task = {**common, 'kind': 'cv', 'comparison': comparison, 'primary': comparison == 'primary',
                            'bounds': bounds, 'k': count, 'repeat': repeat, 'fold': fold,
                            'seed': seed+100000*dataset_index+1000+repeat*folds+fold,
                            'phase': 'prediction', 'methods': list(selected_methods)}
                    prediction.append((task, split))
            if comparison == 'primary':
                prediction.append(({**common, 'kind': 'full', 'comparison': 'primary', 'primary': True,
                    'bounds': 'primary', 'k': spec['k'], 'repeat': None, 'fold': None,
                    'seed': seed+100000*dataset_index+2000, 'phase': 'prediction', 'methods': methods},
                    {'train': list(range(n)), 'test': list(range(n)), 'bootstrap_draw_groups': [],
                     'group_split_seed': None, 'bootstrap_draw_seed': None}))
        for repetition in range(int(config.get('bootstrap_repetitions', 199))):
            draw_seed = seed+100000*dataset_index+10000+repetition
            train, draws = group_bootstrap(data['groups'], draw_seed)
            task = {**common, 'kind': 'bootstrap', 'comparison': 'primary', 'primary': True,
                'bounds': 'primary', 'k': spec['k'], 'repeat': repetition, 'fold': None,
                'seed': seed+100000*dataset_index+50000+repetition, 'phase': 'bootstrap', 'methods': list(CORE)}
            bootstrap.append((task, {'train': train.tolist(), 'test': list(range(n)),
                                     'bootstrap_draw_groups': draws, 'bootstrap_draw_seed': draw_seed,
                                     'group_split_seed': None}))
    result = prediction+bootstrap
    for index, (task, split) in enumerate(result):
        task['task_index'] = index; split['task_index'] = index
        task['initialization_seed'] = task['seed']
        task['conditional_methods'] = ['conditional_log_linear'] if task['primary'] else []
        validate_split(split, datasets[task['dataset_id']], task)
    return result


def preparation_files(config, datasets):
    """Bind prepared data and scientific choices to the offline preparation."""
    relative = config['data_metadata']; path = ROOT/relative; preparation = read(path)
    files = {relative: control.digest(path)}; directory = path.parent
    for specification in config['datasets']:
        spec = dataset_spec(specification); meta = datasets[spec['id']]['metadata']
        entry = preparation['datasets'][spec['id']]
        expected = {'feature_order': spec['columns'], 'n_rows': meta['n_rows'], 'n_groups': meta['n_groups'],
                    'reference': spec['reference'], 'target': spec['conditional_target'],
                    'transform_plan': spec['transform'], 'analysis_coordinate_order': meta['analysis_coordinate_names'],
                    'csv_sha256': meta['raw_data_sha256']}
        if any(entry.get(key) != value for key,value in expected.items()) or directory/entry['csv'] != ROOT/spec['csv']:
            raise control.IntegrityError('Prepared dataset contract/hash differs: '+spec['id'])
        files[spec['csv']] = meta['raw_data_sha256']
        if entry.get('row_metadata_csv') is not None:
            row_file = directory/entry['row_metadata_csv']
            if control.digest(row_file) != entry['row_metadata_csv_sha256']:
                raise control.IntegrityError('Prepared row metadata changed')
            files[str(row_file.relative_to(ROOT))] = entry['row_metadata_csv_sha256']
    for entry in [*preparation.get('sources', {}).values(), *preparation.get('archive_members', {}).values()]:
        source = directory/entry['path']
        if control.digest(source) != entry['sha256']:
            raise control.IntegrityError('Raw source hash differs from preparation manifest')
        files[str(source.relative_to(ROOT))] = entry['sha256']
    preparer = preparation['preparer']
    if control.digest(ROOT/preparer['path']) != preparer['sha256']:
        raise control.IntegrityError('Data preparer changed since preparation')
    files[preparer['path']] = preparer['sha256']
    return files


def freeze(output, config_path=CONFIG):
    config_path = Path(config_path); config = read(config_path); output = Path(output)
    if config.get('threads', 1) != 1 or config['refinement_policy'] != refine.DEFAULT_POLICY:
        raise ValueError('Use the validated one-thread unchanged-box refinement policy')
    if config['bootstrap_methods'] != CORE or config['sensitivity_methods'] != CORE or config['sensitivities'] != {
            'wide_bounds_repeats': [0], 'wide_bounds_methods': ['shared_ld','full_ld'],
            'component_count_offsets': [-1,1], 'component_count_repeats': [0]}:
        raise ValueError('Unsupported or incomplete declared sensitivity/bootstrap method set')
    baseline = config['conditional_baseline']
    if baseline['method'] != conditional_model.METHOD or not baseline['enabled'] or not baseline['primary_tasks_only']:
        raise ValueError('Conditional baseline must cover every primary task only')
    specifications = [dataset_spec(s) for s in config['datasets']]
    if len(set(s['id'] for s in specifications)) != len(specifications):
        raise ValueError('Duplicate dataset identity')
    datasets = {s['id']: load_dataset(ROOT, s) for s in specifications}
    data_files = preparation_files(config, datasets)
    plan = build_plan(config, datasets)
    counts = {'prediction_tasks': sum(t['phase']=='prediction' for t,s in plan),
        'bootstrap_tasks': sum(t['phase']=='bootstrap' for t,s in plan), 'total_tasks': len(plan),
        'prediction_reported_mixture_fits': sum(len(t['methods']) for t,s in plan if t['phase']=='prediction'),
        'bootstrap_reported_mixture_fits': sum(len(t['methods']) for t,s in plan if t['phase']=='bootstrap'),
        'conditional_baseline_fits': sum(len(t['conditional_methods']) for t,s in plan),
        'additional_diagonal_dependencies': sum(GC in t['methods'] and 'gaussian_diag' not in t['methods'] for t,s in plan)}
    if config.get('expected_counts', counts) != counts:
        raise ValueError('Declared study counts differ from generated task plan')
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'config': config, 'config_sha256': control.semantic_hash(config),
        'runtime_versions': control.versions(), 'tasks': [],
        'counts': counts,
        'native_DMM_alpha_bounds': [.01, 10000.],
        'rng_seed_scheme': {'dataset_stride': 100000, 'group_cv_offset': 0, 'model_cv_offset': 1000,
                            'model_full_offset': 2000, 'bootstrap_draw_offset': 10000, 'model_bootstrap_offset': 50000,
                            'description': 'Distinct sampling/model offsets plus repeat; CV model seed also adds repeat*folds+fold'},
        'datasets': {name: data['metadata'] for name, data in datasets.items()},
        'sources_sha256': {str(p.relative_to(ROOT)): control.digest(p) for p in source_files()},
        'data_sha256': data_files}
    for source in source_files()+[ROOT/p for p in manifest['data_sha256']]:
        target = output/'source_snapshot'/source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(source.read_bytes())
    control.atomic_json(output/'source_snapshot/configs/phase3_applications.json', config)
    for name, data in datasets.items():
        control.atomic_json(output/'datasets'/(name+'.json'), data['metadata'])
    for task, split in plan:
        relative = f"splits/task_{task['task_index']:06d}.json"
        control.atomic_json(output/relative, split)
        manifest['tasks'].append({**task, 'split_file': relative, 'split_sha256': control.digest(output/relative)})
    manifest['task_plan_sha256'] = control.semantic_hash(manifest['tasks'])
    control.atomic_json(output/'manifest.json', manifest)
    return manifest


def verify_run(output):
    output = Path(output); manifest = read(output/'manifest.json')
    if manifest['runtime_versions'] != control.versions():
        raise control.IntegrityError('Runtime differs from frozen study')
    if control.semantic_hash(manifest['config']) != manifest['config_sha256'] or control.semantic_hash(
            read(output/'source_snapshot/configs/phase3_applications.json')) != manifest['config_sha256']:
        raise control.IntegrityError('Frozen configuration changed')
    for relative, expected in manifest['sources_sha256'].items():
        if control.digest(ROOT/relative) != expected or control.digest(output/'source_snapshot'/relative) != expected:
            raise control.IntegrityError('Frozen source changed: '+relative)
    for relative, expected in manifest['data_sha256'].items():
        if control.digest(output/'source_snapshot'/relative) != expected:
            raise control.IntegrityError('Frozen data changed: '+relative)
    for name, metadata in manifest['datasets'].items():
        if read(output/'datasets'/(name+'.json')) != metadata:
            raise control.IntegrityError('Dataset metadata changed')
    for index, task in enumerate(manifest['tasks']):
        if task['task_index'] != index or task['split_file'] != f'splits/task_{index:06d}.json':
            raise control.IntegrityError('Noncanonical frozen task map')
    if control.semantic_hash(manifest['tasks']) != manifest['task_plan_sha256']:
        raise control.IntegrityError('Frozen task plan changed')
    return manifest


def load_case(output, manifest, task):
    output = Path(output)
    spec = next(s for s in manifest['config']['datasets'] if dataset_spec(s)['id'] == task['dataset_id'])
    data = load_dataset(output/'source_snapshot', spec)
    if data['metadata'] != manifest['datasets'][task['dataset_id']]:
        raise control.IntegrityError('Prepared data/metadata differ')
    path = output/task['split_file']
    if control.digest(path) != task['split_sha256']:
        raise control.IntegrityError('Frozen split changed')
    split = read(path)
    if split['task_index'] != task['task_index']:
        raise control.IntegrityError('Split belongs to another task')
    validate_split(split, data, task)
    train = np.asarray(split['train'], int); test = np.asarray(split['test'], int)
    y = data['y']; center = y[train].mean(0); scale = y[train].std(0)
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('Degenerate training-only standardization')
    settings = {**base.DEFAULT_SETTINGS, **manifest['config']['fitting']}
    if task['bounds'] == 'wide':
        settings.update(manifest['config']['wide_bounds'])
    binding = {'task': task, 'split_sha256': task['split_sha256'],
               'positive_csv_sha256': data['metadata']['raw_data_sha256'],
               'training_transform': {'center': center.tolist(), 'scale': scale.tolist()}}
    return {**data, 'task': task, 'split': split, 'train_indices': train, 'test_indices': test,
            'train': (y[train]-center)/scale, 'test': (y[test]-center)/scale,
            'center': center, 'scale': scale, 'settings': settings, 'input_binding': binding,
            'compact_metadata': {**{key:value for key,value in data['metadata'].items() if key not in ['row_ids','group_ids']},
                'dataset_metadata_file': 'datasets/'+task['dataset_id']+'.json',
                'dataset_metadata_sha256': control.digest(output/'datasets'/(task['dataset_id']+'.json'))}}


def restore(value):
    params = value['parameters']; record = value['record']; starts = value.get('starts', [])
    selected = value.get('selected_start', -1)
    if params['method'] == GC:
        return gaussian.GaussianSharedCorrelationFit(params, record, starts, selected, record.get('fit_seconds', 0.))
    if params['method'] == skew.METHOD:
        return skew.SkewNormalFit(params, record, starts, selected, record.get('fit_seconds', 0.))
    return base.FitResult(params['method'], params['k'], params['d'], params, record, starts, selected)


def fit_initial(method, y, k, seed, settings, partitions, dependencies):
    """This function receives training observations only."""
    if method == GC:
        diagonal = dependencies.get('gaussian_diag')
        return gaussian.fit(y, k, seed, partitions,
            diagonal_candidate=diagonal.params if diagonal is not None else None,
            n_starts=settings['n_starts'], maxiter=settings['maxiter'],
            polish_maxiter=settings['refinement_maxiter'], gtol=settings['gtol'],
            ftol=settings['ftol'], start_timeout_seconds=settings['start_timeout_seconds'])
    if method == skew.METHOD:
        with apps.bounded_skew_optimizer(settings['start_timeout_seconds']):
            return skew.fit(y, k, seed, partitions, gaussian_candidate=dependencies.get('gaussian_full'),
                n_starts=settings['n_starts'], maxiter=settings['maxiter'],
                polish_maxiter=settings['refinement_maxiter'], gtol=settings['gtol'])
    warm = (dependencies.get('shared_ld') if method == 'full_ld' else
            dependencies.get('fmvmm_dmm') if method == 'base_ld' else None)
    return base.fit_method(method, y, k, seed, settings, partitions, warm)


def model_value(model):
    return {'failed': False, 'parameters': model.params, 'record': model.record,
            'starts': model.starts, 'selected_start': model.selected_start}


def failure_value(exc):
    return {'failed': True, 'parameters': None, 'record': {}, 'starts': getattr(exc, 'starts', []),
            'selected_start': None, 'exception': repr(exc), 'exception_type': type(exc).__name__,
            'traceback': traceback.format_exc()}


def evaluate(model, case):
    native = model.method in NATIVE_METHODS
    indices = case['test_indices']; values = case['y'][indices] if native else case['test']
    target = case['metadata']['target_index']; keep = [j for j in range(values.shape[1]) if j != target]
    ell, tau = model.evaluate(values)
    marginal = (control.gaussian_density(model.params, values, keep)[0] if model.method == GC
                else apps.marginal_log_density(model.params, values, keep))
    log_scale = np.zeros(values.shape[1]) if native else np.log(case['scale'])
    correction = case['raw_log_jacobian'][indices]
    conditional = ell-marginal-log_scale[target]-correction[:, target]
    joint = ell-log_scale.sum()-correction.sum(1)
    if not np.isfinite(joint).all() or not np.isfinite(conditional).all() or not np.isfinite(tau).all():
        raise FloatingPointError('Nonfinite original-measure prediction')
    return joint, conditional, tau


def group_mean(values, groups):
    values = np.asarray(values); groups = np.asarray(groups)
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    return float(np.mean(np.bincount(inverse, weights=values)/counts))


def fit_checkpoint(directory, method, case, manifest, identity, partitions, fitted,
                   initial_fitter=fit_initial, refiner=refine.refine_saved, dependency_costs=None):
    """Checkpoint both fitting stages before any held-out evaluation."""
    stage_identity = {**identity, 'method': method}
    initial_path = directory/(method+'_initial_fit.json'); final_path = directory/(method+'_fit.json')
    native = method in NATIVE_METHODS
    y = case['y'][case['train_indices']] if native else case['train']
    settings = dict(case['settings'])
    if native:
        settings['alpha_bounds'] = manifest['native_DMM_alpha_bounds']
    if initial_path.exists():
        initial = read(initial_path); control.validate_stage(initial, stage_identity, 'Initial model checkpoint')
    else:
        started = time.perf_counter()
        try:
            model = initial_fitter(method, y, case['task']['k'], case['task']['seed'], settings, partitions, fitted)
            initial = model_value(model)
        except Exception as exc:
            initial = failure_value(exc)
        initial.update(stage_identity); initial['elapsed_seconds'] = time.perf_counter()-started
        initial['settings'] = settings
        control.atomic_json(initial_path, initial); initial = base.serializable(initial)
    final_identity = {**stage_identity, 'initial_fit_sha256': control.digest(initial_path)}
    if final_path.exists():
        final = read(final_path); control.validate_stage(final, final_identity, 'Selected model checkpoint')
        return final
    started = time.perf_counter(); refinement_seconds = 0.
    try:
        if method in CORE:
            model = refiner(initial, y, case['task']['k'], method, settings, manifest['config']['refinement_policy'],
                            refined_shared=fitted.get('shared_ld') if method == 'full_ld' else None)
            refinement_seconds = time.perf_counter()-started
            final = model_value(model)
        elif initial['failed']:
            final = dict(initial)
        else:
            final = dict(initial)
        if not final['failed']:
            record = final['record']; own = initial['elapsed_seconds']+refinement_seconds
            dependency_name = {'full_ld': 'shared_ld', GC: 'gaussian_diag', skew.METHOD: 'gaussian_full',
                               'base_ld': 'fmvmm_dmm'}.get(method)
            dependency = fitted.get(dependency_name)
            dependency_seconds = dependency.record.get('policy_fit_seconds', 0.) if dependency is not None else 0.
            record.update(initial_fit_seconds=initial['elapsed_seconds'], refinement_fit_seconds=refinement_seconds,
                          fit_seconds=own, dependency_fit_seconds=dependency_seconds,
                          policy_fit_seconds=own+dependency_seconds, dependency_method=dependency_name,
                          initial_starts_attempted=len(initial['starts']),
                          initial_starts_failed=sum(bool(s.get('failed')) for s in initial['starts']))
            record['starts_attempted'] = len(final['starts'])
            record['starts_failed'] = sum(bool(s.get('failed')) for s in final['starts'])
            if method in CORE:
                initial_params = initial.get('parameters')
                if initial_params is not None:
                    old = -refine.objective(np.asarray(initial_params['theta']), y, case['task']['k'], method)[0]
                    gain = record['mean_train_log_density']-old
                    if gain < -1e-10:
                        raise ArithmeticError('Refinement lost a retained initial training point')
                    record['training_mean_log_density_gain'] = gain
            if method == 'full_ld' and fitted.get('shared_ld') is not None:
                gap = record['mean_train_log_density']-fitted['shared_ld'].record['mean_train_log_density']
                if gap < -1e-10:
                    raise ArithmeticError('Separate-shape fit lost its refined nested shared point')
                record['full_minus_shared_training_mean_log_density'] = gap
    except Exception as exc:
        # A final-stage failure remains explicit; the immutable initial stage and
        # any failure's finite endpoints are retained, never silently relabelled.
        final = failure_value(exc)
        refinement_seconds = time.perf_counter()-started if method in CORE else 0.
    final.setdefault('record', {}).update(initial_fit_seconds=initial['elapsed_seconds'],
        refinement_fit_seconds=refinement_seconds, initial_starts_attempted=len(initial['starts']),
        initial_starts_failed=sum(bool(s.get('failed')) for s in initial['starts']),
        initial_timeout_records=sum(s.get('exception_type')=='TimeoutError' or
            str(s.get('message','')).startswith('TimeoutError(') for s in initial['starts']),
        selected_stage='refinement' if method in CORE else 'initial',
        exception_counter_scope='Selected-stage exception records; initial failures/timeouts reported separately')
    final['record'].setdefault('fit_seconds', initial['elapsed_seconds']+refinement_seconds)
    dependency_name = {'full_ld':'shared_ld',GC:'gaussian_diag',skew.METHOD:'gaussian_full','base_ld':'fmvmm_dmm'}.get(method)
    dependency = fitted.get(dependency_name)
    dependency_seconds = ((dependency_costs or {}).get(dependency_name,
        dependency.record.get('policy_fit_seconds', 0.) if dependency is not None else 0.))
    final['record'].update(dependency_method=dependency_name, dependency_fit_seconds=dependency_seconds,
                           policy_fit_seconds=final['record']['fit_seconds']+dependency_seconds,
                           dependency_cost_includes_failed_attempts=True)
    final.update(final_identity)
    final['settings'] = settings
    final['initial_fit_file'] = initial_path.name
    final['training_transform'] = {'center': case['center'], 'scale': case['scale']}
    final['dataset_metadata'] = case['compact_metadata']
    control.atomic_json(final_path, final)
    return base.serializable(final)


def numerical_counters(selected):
    starts = selected.get('starts', [])
    return {'starts_attempted': len(starts), 'starts_failed': sum(bool(s.get('failed')) for s in starts),
        'retained_starts_with_exceptions': sum(not s.get('failed') and bool(s.get('exceptions')) for s in starts),
        'optimizer_exception_count': sum(len(s.get('exceptions', [])) for s in starts),
        'timeout_exception_count': sum(sum(e.get('exception_type') == 'TimeoutError' for e in s.get('exceptions', []))
                                      + int(s.get('exception_type') == 'TimeoutError') for s in starts)}


def validate_saved_result(directory, value, identity):
    control.validate_stage(value, identity, 'Model result')
    expected = value.get('selected_checkpoint_sha256')
    if expected is None:
        if value['row'].get('fit_available') or value['row'].get('evaluation_available'):
            raise control.IntegrityError('Available result has no selected-checkpoint identity')
        return
    fit_path = directory/(identity['method']+'_fit.json')
    if control.digest(fit_path) != expected:
        raise control.IntegrityError('Selected parameter checkpoint changed')
    selected = read(fit_path); control.validate_stage(selected, identity, 'Selected checkpoint')
    if value.get('parameters') is not None and (value['parameters'] != selected['parameters'] or value['record'] != selected['record']):
        raise control.IntegrityError('Evaluated result and selected fit differ')
    if selected.get('initial_fit_sha256') is not None:
        initial_path = directory/(identity['method']+'_initial_fit.json')
        if control.digest(initial_path) != selected['initial_fit_sha256']:
            raise control.IntegrityError('Initial fitting checkpoint changed')
        control.validate_stage(read(initial_path), identity, 'Original fitting stage')
    if value['row'].get('evaluation_available'):
        prediction = directory/(identity['method']+'_predictions.npz')
        if not prediction.exists() or control.digest(prediction) != value.get('prediction_sha256'):
            raise control.IntegrityError('Saved prediction artifact is missing or changed')


def validate_completion_scope(value, task):
    dependencies = ['gaussian_diag'] if GC in task['methods'] and 'gaussian_diag' not in task['methods'] else []
    if value.get('status') != 'complete' or [r['method'] for r in value['rows']] != task['methods'] or \
            [r['method'] for r in value['conditional_rows']] != task['conditional_methods'] or \
            [r['method'] for r in value['dependency_rows']] != dependencies:
        raise control.IntegrityError('Completed method rows differ from plan')


def run_task(output, index, initial_fitter=fit_initial, refiner=refine.refine_saved, conditional_fitter=None):
    output = Path(output); manifest = verify_run(output)
    if type(index) is not int or not 0 <= index < len(manifest['tasks']):
        raise ValueError('Invalid task index')
    task = manifest['tasks'][index]; directory = output/'tasks'/f'task_{index:06d}'
    with control.task_lock(directory), threadpool_limits(1):
        case = load_case(output, manifest, task)
        training_path = directory/'training.json'
        base_identity = {'task': task, 'manifest_sha256': control.digest(output/'manifest.json'),
                         'input_binding_sha256': control.semantic_hash(case['input_binding'])}
        if training_path.exists():
            training = read(training_path); control.validate_stage(training, base_identity, 'Training checkpoint')
            if training['training_transform'] != base.serializable({'center': case['center'], 'scale': case['scale']}):
                raise control.IntegrityError('Saved training transform differs')
            partitions = [np.asarray(p, int) for p in training['partitions']]
        else:
            partitions = base.make_partitions(case['train'], task['k'], task['seed'], case['settings']['n_starts'])
            training = {**base_identity, 'training_transform': {'center': case['center'], 'scale': case['scale']},
                        'partitions': partitions}
            control.atomic_json(training_path, training)
        if len(partitions) != case['settings']['n_starts'] or any(p.shape != (len(case['train']),) or
                    np.any(p < 0) or np.any(p >= task['k']) for p in partitions):
            raise control.IntegrityError('Invalid saved training partitions')
        binding = {**case['input_binding'], 'training_checkpoint_sha256': control.digest(training_path)}
        binding_path = directory/'input_binding.json'
        if binding_path.exists() and read(binding_path) != binding:
            raise control.IntegrityError('Bound training inputs changed')
        control.atomic_json(binding_path, binding)
        identity = {**base_identity, 'input_binding_sha256': control.semantic_hash(binding)}
        completion_path = directory/'completion.json'
        if completion_path.exists():
            completed = read(completion_path); control.validate_stage(completed, identity, 'Task completion')
            validate_completion_scope(completed, task)
            for row in completed['rows']+completed['conditional_rows']+completed['dependency_rows']:
                saved = read(directory/(row['method']+'.json'))
                validate_saved_result(directory, saved, {**identity, 'method': row['method']})
                if saved['row'] != row:
                    raise control.IntegrityError('Completion row differs from saved result')
            return completed
        fitted = {}; costs = {}; rows = []; dependency_rows = []
        execution_methods = list(task['methods'])
        if GC in execution_methods and 'gaussian_diag' not in execution_methods:
            execution_methods.insert(0, 'gaussian_diag')
        for method in execution_methods:
            dependency_only = method not in task['methods']; result_path = directory/(method+'.json')
            stage_identity = {**identity, 'method': method}
            if result_path.exists():
                value = read(result_path); validate_saved_result(directory, value, stage_identity)
                if value['parameters'] is not None:
                    fitted[method] = restore(value)
                costs[method] = value['record'].get('policy_fit_seconds', value['record'].get('fit_seconds', 0.))
                (dependency_rows if dependency_only else rows).append(value['row']); continue
            row = {**task, 'method': method, 'dependency_only': dependency_only, 'failed': True,
                'fit_available': False, 'evaluation_available': False, 'n_train': len(case['train']),
                'n_test': len(case['test']), 'n_test_groups': len(np.unique(case['groups'][case['test_indices']])),
                'd': case['train'].shape[1], 'mean_log_density': None, 'mean_conditional_log_density': None,
                'group_mean_log_density': None, 'group_mean_conditional_log_density': None, 'ARI': None,
                'score_converged': None, 'projected_gradient': None, 'boundary_hit': None,
                'evaluation_role': 'original_row_grid_descriptive' if task['kind'] in ['bootstrap', 'full'] else 'held_out_groups',
                'conditional_only': False, 'coordinate_measure': case['metadata']['coordinate_measure']}
            model = None; selected = None; profiles = None; started = time.perf_counter()
            try:
                selected = fit_checkpoint(directory, method, case, manifest, identity, partitions, fitted, initial_fitter, refiner, costs)
                row.update(numerical_counters(selected))
                row.update({key:selected['record'].get(key) for key in ['initial_starts_attempted', 'initial_starts_failed',
                    'initial_timeout_records','initial_fit_seconds','refinement_fit_seconds','fit_seconds',
                    'dependency_fit_seconds','policy_fit_seconds','selected_stage','exception_counter_scope']})
                costs[method] = selected['record']['policy_fit_seconds']
                if selected['failed']:
                    raise RuntimeError(selected.get('exception', 'Selected fit unavailable'))
                model = restore(selected); fitted[method] = model; row['fit_available'] = True
                for field in ['score_converged', 'projected_gradient', 'boundary_hit', 'boundary_hits',
                    'fit_seconds', 'initial_fit_seconds', 'refinement_fit_seconds', 'dependency_fit_seconds',
                    'policy_fit_seconds', 'initial_starts_attempted', 'initial_starts_failed',
                    'parameter_count', 'min_weight', 'training_mean_log_density_gain',
                    'full_minus_shared_training_mean_log_density', 'omitted_saved_endpoints']:
                    row[field] = model.record.get(field)
                train_values = case['y'][case['train_indices']] if method in NATIVE_METHODS else case['train']
                scale = np.ones(case['train'].shape[1]) if method in NATIVE_METHODS else case['scale']
                center = np.zeros_like(scale) if method in NATIVE_METHODS else case['center']
                row['train_mean_log_density'] = float(model.evaluate(train_values)[0].mean()-np.log(scale).sum()
                    -case['raw_log_jacobian'][case['train_indices']].sum(1).mean())
                profiles = {'means_in_analysis_coordinates': center+scale*model.component_means(),
                            'coordinate_names': case['metadata']['analysis_coordinate_names']}
                if not dependency_only:
                    joint, conditional, tau = evaluate(model, case); indices = case['test_indices']
                    groups = case['groups'][indices]; labels = case['labels'][indices]
                    np.savez_compressed(directory/(method+'_predictions.npz'), indices=indices,
                        row_ids=case['row_ids'][indices].astype(str), groups=groups.astype(str),
                        log_density=joint, conditional_log_density=conditional, responsibilities=tau)
                    row.update(evaluation_available=True, mean_log_density=float(joint.mean()),
                        mean_conditional_log_density=float(conditional.mean()), group_mean_log_density=group_mean(joint, groups),
                        group_mean_conditional_log_density=group_mean(conditional, groups),
                        ARI=float(adjusted_rand_score(labels, tau.argmax(1))) if np.all(labels != '') else None)
                row['failed'] = False
            except control.IntegrityError:
                raise
            except Exception as exc:
                row.update(failed=True, evaluation_available=False, exception=repr(exc), traceback=traceback.format_exc())
            row['total_seconds_this_invocation'] = time.perf_counter()-started
            value = {**stage_identity, 'row': row, 'parameters': model.params if model is not None else None,
                'record': model.record if model is not None else selected.get('record', {}) if selected else {},
                'starts': selected.get('starts', []) if selected else [],
                'selected_start': model.selected_start if model is not None else None, 'profiles': profiles,
                'training_transform': {'center': case['center'], 'scale': case['scale']},
                'dataset_metadata': case['compact_metadata'],
                'prediction_sha256': control.digest(directory/(method+'_predictions.npz')) if row['evaluation_available'] else None,
                'selected_checkpoint_sha256': control.digest(directory/(method+'_fit.json')) if selected else None}
            control.atomic_json(result_path, value)
            (dependency_rows if dependency_only else rows).append(base.serializable(row))
        conditional_rows = []
        for method in task['conditional_methods']:
            conditional_rows.append(run_conditional(directory, method, case, manifest, identity, conditional_fitter))
        completed = {**identity, 'status': 'complete', 'rows': rows, 'dependency_rows': dependency_rows,
                     'conditional_rows': conditional_rows, 'runtime_versions': control.versions(),
                     'completed_utc': datetime.now(timezone.utc).isoformat()}
        control.atomic_json(completion_path, completed)
        return base.serializable(completed)


def run_conditional(directory, method, case, manifest, identity, conditional_fitter=None):
    settings = manifest['config']['conditional_baseline']
    if method != conditional_model.METHOD or not settings['enabled'] or not settings['primary_tasks_only']:
        raise ValueError('Conditional baseline differs from its fixed scope')
    stage_identity = {**identity, 'method': method}
    fit_path = directory/(method+'_fit.json'); result_path = directory/(method+'.json')
    if result_path.exists():
        value = read(result_path); validate_saved_result(directory, value, stage_identity)
        return value['row']
    row = {**case['task'], 'method': method, 'conditional_only': True, 'failed': True,
           'fit_available': False, 'evaluation_available': False, 'conditional_available': False,
           'n_train': len(case['train']), 'n_test': len(case['test']),
           'n_test_groups': len(np.unique(case['groups'][case['test_indices']])),
           'mean_conditional_log_density': None, 'group_mean_conditional_log_density': None,
           'score_converged': None, 'boundary_hit': None, 'projected_gradient': None,
           'evaluation_role': 'original_row_grid_descriptive' if case['task']['kind'] in ['full','bootstrap'] else 'held_out_groups'}
    selected = None; target = case['metadata']['target_index']
    try:
        if fit_path.exists():
            selected = read(fit_path); control.validate_stage(selected, stage_identity, 'Conditional selected checkpoint')
        else:
            try:
                fit = conditional_fitter or conditional_model.fit
                model = fit(case['train'], target,
                    residual_sd_floor=settings['standardized_residual_sd_floor'], gtol=case['settings']['gtol'])
                selected = {'failed': False, 'parameters': model.params, 'record': model.record,
                            'starts': [], 'selected_start': None}
            except Exception as exc:
                selected = failure_value(exc)
            selected.update(stage_identity)
            selected['training_transform'] = {'center': case['center'], 'scale': case['scale']}
            selected['dataset_metadata'] = case['compact_metadata']
            control.atomic_json(fit_path, selected); selected = base.serializable(selected)
        if selected['failed']:
            raise RuntimeError(selected['exception'])
        row['fit_available'] = True
        for field in ['score_converged', 'projected_gradient', 'boundary_hit', 'parameter_count',
                      'fit_seconds', 'design_rank', 'design_condition', 'rank_deficient', 'residual_floor_active']:
            row[field] = selected['record'].get(field)
        indices = case['test_indices']
        means, density = conditional_model.original_scale_predictions(selected['parameters'], case['y'][indices],
            case['center'], case['scale'], -case['raw_log_jacobian'][indices, target])
        if not np.isfinite(means).all() or not np.isfinite(density).all():
            raise FloatingPointError('Nonfinite conditional baseline prediction')
        groups = case['groups'][indices]
        np.savez_compressed(directory/(method+'_predictions.npz'), indices=indices,
            row_ids=case['row_ids'][indices].astype(str), groups=groups.astype(str),
            conditional_log_density=density, conditional_mean_in_analysis_coordinates=means)
        row.update(failed=False, evaluation_available=True, conditional_available=True,
                   mean_conditional_log_density=float(density.mean()),
                   group_mean_conditional_log_density=group_mean(density, groups))
    except control.IntegrityError:
        raise
    except Exception as exc:
        row.update(failed=True, evaluation_available=False, conditional_available=False,
                   exception=repr(exc), traceback=traceback.format_exc())
    value = {**stage_identity, 'row': row, 'parameters': selected.get('parameters') if selected else None,
        'record': selected.get('record', {}) if selected else {}, 'starts': [], 'selected_start': None,
        'training_transform': {'center': case['center'], 'scale': case['scale']},
        'dataset_metadata': case['compact_metadata'],
        'prediction_sha256': control.digest(directory/(method+'_predictions.npz')) if row['evaluation_available'] else None,
        'selected_checkpoint_sha256': control.digest(fit_path) if selected else None}
    control.atomic_json(result_path, value)
    return base.serializable(row)


def aggregate(output, phase='all'):
    output = Path(output); manifest = verify_run(output)
    if phase not in ['all','prediction','bootstrap']:
        raise ValueError('Invalid aggregation phase')
    tasks = [t for t in manifest['tasks'] if phase=='all' or t['phase']==phase]
    rows = []; conditional_rows = []; dependencies = []
    # Unexpected backup/duplicate completion directories are rejected rather than
    # silently double-counted or treated as additional repetitions.
    for path in (output/'tasks').glob('*/completion.json'):
        value = read(path); index = value.get('task', {}).get('task_index')
        if type(index) is not int or not 0 <= index < len(manifest['tasks']) or path.parent.name != f'task_{index:06d}':
            raise control.IntegrityError('Noncanonical completed task directory')
    for task in tasks:
        directory = output/'tasks'/f"task_{task['task_index']:06d}"; path = directory/'completion.json'
        if not path.exists():
            for destination, methods, conditional_only in [(rows, task['methods'], False),
                                                         (conditional_rows, task['conditional_methods'], True)]:
                destination.extend([{**task, 'method': method, 'recorded': False, 'failed': True,
                    'fit_available': False, 'evaluation_available': False, 'conditional_only': conditional_only,
                    'score_converged': None, 'boundary_hit': None, 'mean_log_density': None,
                    'mean_conditional_log_density': None, 'group_mean_log_density': None,
                    'group_mean_conditional_log_density': None} for method in methods])
            continue
        case = load_case(output, manifest, task); training_path = directory/'training.json'
        training = read(training_path)
        original_identity = {'task': task, 'manifest_sha256': control.digest(output/'manifest.json'),
                             'input_binding_sha256': control.semantic_hash(case['input_binding'])}
        control.validate_stage(training, original_identity, 'Aggregate training')
        binding = {**case['input_binding'], 'training_checkpoint_sha256': control.digest(training_path)}
        if read(directory/'input_binding.json') != binding:
            raise control.IntegrityError('Aggregate input binding differs from frozen data')
        identity = {**original_identity, 'input_binding_sha256': control.semantic_hash(binding)}
        value = read(path); control.validate_stage(value, identity, 'Aggregate completion')
        validate_completion_scope(value, task)
        for destination, name in [(rows, 'rows'), (conditional_rows, 'conditional_rows'), (dependencies, 'dependency_rows')]:
            for row in value[name]:
                saved = read(directory/(row['method']+'.json'))
                validate_saved_result(directory, saved, {**identity, 'method': row['method']})
                if saved['row'] != row:
                    raise control.IntegrityError('Completed row differs from selected result')
                destination.append({**row, 'recorded': True})
    frame = pd.DataFrame(rows); conditional = pd.DataFrame(conditional_rows)
    frame.to_csv(output/f'fit_records_{phase}.csv', index=False)
    conditional.to_csv(output/f'conditional_records_{phase}.csv', index=False)
    pd.DataFrame(dependencies).to_csv(output/f'dependency_records_{phase}.csv', index=False)
    all_rows = pd.concat([frame, conditional], ignore_index=True)
    summaries = []; repetitions = []
    keys = ['dataset_id','kind','comparison','bounds','k','method','conditional_only']
    for values, group in all_rows.groupby(keys, dropna=False):
        valid = group[group.evaluation_available.eq(True)]
        summary = dict(zip(keys, values))
        summary.update(planned_tasks=len(group), recorded=int(group.recorded.sum()), available=len(valid),
            missing=int((~group.recorded).sum()), failed_recorded=int((group.recorded & group.failed).sum()),
            score_checked=int(group.score_converged.notna().sum()), score_passed=int(group.score_converged.eq(True).sum()),
            boundary_checked=int(group.boundary_hit.notna().sum()), boundary_hits=int(group.boundary_hit.eq(True).sum()))
        for field in ['initial_starts_attempted','initial_starts_failed','initial_timeout_records','starts_attempted','starts_failed',
                      'retained_starts_with_exceptions','optimizer_exception_count','timeout_exception_count']:
            summary[field] = int(group[field].fillna(0).sum()) if field in group else 0
        summaries.append(summary)
        if values[1] != 'cv':
            continue
        for repeat, folds in valid.groupby('repeat'):
            if len(folds) != manifest['config']['folds']:
                continue
            record = {**dict(zip(keys, values)), 'repeat': int(repeat), 'n_test': int(folds.n_test.sum()),
                      'n_test_groups': int(folds.n_test_groups.sum())}
            for metric, weights in [('mean_log_density','n_test'), ('mean_conditional_log_density','n_test'),
                                    ('group_mean_log_density','n_test_groups'), ('group_mean_conditional_log_density','n_test_groups')]:
                record[metric] = (float(np.average(folds[metric], weights=folds[weights]))
                                  if metric in folds and folds[metric].notna().all() else None)
            repetitions.append(record)
    pd.DataFrame(summaries).to_csv(output/f'summary_{phase}.csv', index=False)
    repeat_frame = pd.DataFrame(repetitions); repeat_frame.to_csv(output/f'cv_repetitions_{phase}.csv', index=False)
    paired = []
    if len(repeat_frame):
        for values, group in repeat_frame.groupby(['dataset_id','comparison','bounds','k']):
            for comparator in sorted(set(group.method)-{'shared_ld'}):
                shared_rows = group[group.method=='shared_ld']; other = group[group.method==comparator]
                joined = shared_rows.merge(other, on='repeat', suffixes=('_shared','_other'))
                for metric in ['mean_log_density','mean_conditional_log_density','group_mean_log_density','group_mean_conditional_log_density']:
                    finite = joined[[metric+'_shared', metric+'_other']].dropna()
                    difference = finite[metric+'_shared']-finite[metric+'_other']
                    paired.append({**dict(zip(['dataset_id','comparison','bounds','k'], values)),
                        'comparator': comparator, 'metric': metric, 'complete_paired_repetitions': len(difference),
                        'mean_shared_minus_comparator': difference.mean() if len(difference) else None,
                        'minimum_difference': difference.min() if len(difference) else None,
                        'maximum_difference': difference.max() if len(difference) else None,
                        'uncertainty': 'Descriptive mean/range across overlapping CV repetitions; no independent-fold MCSE'})
    pd.DataFrame(paired).to_csv(output/f'paired_cv_{phase}.csv', index=False)
    status = {'phase': phase, 'planned_tasks': len(tasks), 'planned_mixture_fits': len(frame),
        'recorded_mixture_fits': int(frame.recorded.sum()), 'mixture_evaluations_available': int(frame.evaluation_available.sum()),
        'planned_conditional_fits': len(conditional), 'recorded_conditional_fits': int(conditional.recorded.sum()) if len(conditional) else 0,
        'conditional_evaluations_available': int(conditional.evaluation_available.sum()) if len(conditional) else 0,
        'manifest_sha256': control.digest(output/'manifest.json'), 'created_utc': datetime.now(timezone.utc).isoformat()}
    control.atomic_json(output/f'aggregation_{phase}.json', status)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--freeze', action='store_true'); parser.add_argument('--task-index', type=int)
    parser.add_argument('--task-range'); parser.add_argument('--aggregate', action='store_true')
    parser.add_argument('--phase', choices=['all','prediction','bootstrap'], default='all')
    args = parser.parse_args()
    if sum([args.freeze, args.task_index is not None, args.task_range is not None, args.aggregate]) != 1:
        parser.error('Choose exactly one operation')
    if args.freeze:
        result = freeze(args.output, args.config); print(json.dumps(result['counts']), flush=True)
    elif args.aggregate:
        print(json.dumps(aggregate(args.output, args.phase)), flush=True)
    elif args.task_index is not None:
        result = run_task(args.output, args.task_index); print(json.dumps({'task_index': args.task_index, 'status': result['status']}), flush=True)
    else:
        start, stop = map(int, args.task_range.split(':'))
        for index in range(start, stop):
            result = run_task(args.output, index); print(json.dumps({'task_index': index, 'status': result['status']}), flush=True)


if __name__ == '__main__':
    main()
