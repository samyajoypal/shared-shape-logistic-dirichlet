"""Complete-panel reporting around the unchanged phase-3 stability evaluator.

This adapter never fits or selects a model. It retains the evaluator's finite
pair/common-score definitions, adds explicit planned diagnostic denominators,
and rehashes all consumed inputs at completion. Use after complete retrieval.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import scipy

import phase3_stability as kernel

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('scripts/assess_phase3_stability.py', 'scripts/phase3_stability.py',
           'scripts/assess_conditional_prediction_stability.py',
           'scripts/conditional_log_linear.py')
FLAGS = ('source_failed', 'score_converged', 'boundary_hit')


def require(condition, message):
    if not condition:
        raise kernel.IntegrityError(message)


def flag(value):
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value in ('True', 'False'):
        return value == 'True'
    raise kernel.IntegrityError('Invalid saved Boolean diagnostic')


def rehash_all(hashes):
    """Detect mutation/disappearance after any input was consumed."""
    for path, expected in hashes.items():
        require(Path(path).is_file() and kernel.digest(path) == expected,
                'Consumed input changed before report completion: ' + str(path))


def completed_rows(panel, task):
    """Read canonical saved flags even if that method has no finite parameters.

    Numerical prediction/parameter checks remain the unchanged kernel's job.
    This separately verifies the completion, bound data/split, training point
    and worker runtime before accepting any source failure or score flag.
    """
    folder = panel.run/'tasks'/f"task_{task['task_index']:06d}"
    path = folder/'completion.json'
    if not path.exists():
        return {}
    completion = panel.read(path)
    binding = panel.read(folder/'input_binding.json')
    identity = {'task': task, 'manifest_sha256': panel.manifest_sha256,
                'input_binding_sha256': kernel.semantic_hash(binding)}
    require(completion.get('status') == 'complete' and
            all(completion.get(k) == v for k, v in identity.items()),
            'Completed stability task identity differs')
    require(completion.get('runtime_versions') == panel.manifest['runtime_versions'],
            'Completed worker runtime differs')
    panel.bytes(panel.run/task['split_file'], task['split_sha256'])
    split = panel.read(panel.run/task['split_file'])
    y, groups, _, metadata = panel.grid(task['dataset_id'])
    train = np.asarray(split['train'])
    test = np.asarray(split['test'])
    require(split['task_index'] == task['task_index'] and train.ndim == 1 and
            train.dtype.kind in 'iu' and len(train) > 0 and np.all(train >= 0) and
            np.all(train < len(y)) and np.array_equal(test, np.arange(len(y))),
            'Completed stability split differs from original grid')
    if task['kind'] == 'bootstrap':
        kernel.verify_group_bootstrap(groups, split['bootstrap_draw_groups'], train)
        levels = np.unique(groups)
        draws = np.random.default_rng(split['bootstrap_draw_seed']).choice(levels, len(levels), replace=True)
        require(np.array_equal(draws, split['bootstrap_draw_groups']), 'Bootstrap stream differs')
    else:
        require(task['kind'] == 'full' and np.array_equal(train, np.arange(len(y))),
                'Full-data training rows differ')
    transform = {'center': y[train].mean(0).tolist(), 'scale': y[train].std(0).tolist()}
    require(binding == {'task': task, 'split_sha256': task['split_sha256'],
            'positive_csv_sha256': metadata['raw_data_sha256'], 'training_transform': transform,
            'training_checkpoint_sha256': binding['training_checkpoint_sha256']},
            'Completed stability inputs differ')
    panel.bytes(folder/'training.json', binding['training_checkpoint_sha256'])
    training = panel.read(folder/'training.json')
    base_binding = {k: v for k, v in binding.items() if k != 'training_checkpoint_sha256'}
    require(training['task'] == task and training['manifest_sha256'] == panel.manifest_sha256 and
            training['input_binding_sha256'] == kernel.semantic_hash(base_binding) and
            training['training_transform'] == transform, 'Training checkpoint identity differs')
    require([r['method'] for r in completion['rows']] == task['methods'] and
            [r['method'] for r in completion['conditional_rows']] == task['conditional_methods'],
            'Completed stability method scope differs')
    rows = {}
    for row in completion['rows'] + completion['conditional_rows']:
        if row['method'] not in kernel.METHODS:
            continue
        require(all(row.get(k) == task[k] for k in ('task_index', 'dataset_id', 'kind', 'comparison', 'k', 'bounds')),
                'Completed stability row identity differs')
        for key in ('failed', 'score_converged', 'boundary_hit'):
            flag(row.get(key))
        rows[row['method']] = row
    return rows


def enrich_diagnostics(panel, diagnostics):
    """Left-join every planned full/bootstrap method, retaining unknown flags."""
    require(not diagnostics.duplicated(['task_index', 'method']).any(), 'Duplicate stability diagnostic')
    supplied = {(int(r['task_index']), r['method']): r for r in diagnostics.to_dict('records')}
    output = []
    expected = set()
    for name in panel.manifest['datasets']:
        full, boots = panel.tasks(name)
        for task in [full] + boots:
            rows = completed_rows(panel, task)
            for method in kernel.METHODS:
                key = (task['task_index'], method)
                expected.add(key)
                require(key in supplied, 'Kernel omitted a planned diagnostic')
                original = supplied[key]
                require(original['dataset_id'] == name and original['kind'] == task['kind'],
                        'Kernel diagnostic task identity differs')
                require((pd.isna(original['repeat']) and task['repeat'] is None) or
                        original['repeat'] == task['repeat'], 'Kernel diagnostic repeat differs')
                record = rows.get(method)
                row = {**original, 'source_recorded': record is not None}
                for destination, source in [('source_failed', 'failed'),
                        ('score_converged', 'score_converged'), ('boundary_hit', 'boundary_hit')]:
                    saved = flag(record.get(source)) if record is not None else None
                    previous = flag(original.get(destination))
                    require(previous is None or previous == saved, 'Kernel/source diagnostic differs')
                    row[destination] = saved
                row['prediction_available'] = flag(original['prediction_available']) is True
                row['finite_from_failed_source'] = row['prediction_available'] and row['source_failed'] is True
                row['finite_score_ineligible'] = row['prediction_available'] and row['score_converged'] is not True
                output.append(row)
    require(set(supplied) == expected, 'Kernel contains an undeclared stability diagnostic')
    return pd.DataFrame(output)


def count_flags(frame, prefix=''):
    result = {prefix+'slots': len(frame), prefix+'source_recorded': int(frame.source_recorded.sum()),
              prefix+'source_missing': int((~frame.source_recorded).sum()),
              prefix+'prediction_available': int(frame.prediction_available.sum()),
              prefix+'prediction_missing': int((~frame.prediction_available).sum()),
              prefix+'finite_from_failed_source': int(frame.finite_from_failed_source.sum()),
              prefix+'finite_score_ineligible': int(frame.finite_score_ineligible.sum())}
    for name in FLAGS:
        values = [flag(x) for x in frame[name]]
        for state, value in [('true', True), ('false', False), ('unknown', None)]:
            result[prefix+name+'_'+state] = sum(x is value for x in values)
    return result


def diagnostic_tables(diagnostics, summary, paired):
    """Add counts without changing the evaluator's variance or MC definitions."""
    counts, summaries, pairs = [], [], []
    masks = {}
    for name, frame in diagnostics.groupby('dataset_id', sort=False):
        boot = frame[frame.kind == 'bootstrap']
        expected_repeats = sorted(boot.repeat.unique())
        require(expected_repeats == list(range(len(expected_repeats))), 'Bootstrap diagnostic indices differ')
        per_method = {m: boot[boot.method == m].sort_values('repeat') for m in kernel.METHODS}
        require(all(rows['repeat'].tolist() == expected_repeats for rows in per_method.values()),
                'Missing or duplicate bootstrap method index')
        common = np.logical_and.reduce([rows.prediction_available.to_numpy(bool) &
                    np.array([flag(v) is True for v in rows.score_converged]) for rows in per_method.values()])
        masks[name] = common
        for (kind, method), part in frame.groupby(['kind', 'method'], sort=False):
            counts.append({'dataset_id': name, 'kind': kind, 'method': method, **count_flags(part)})
    for saved in summary.to_dict('records'):
        name, method, subset = saved['dataset_id'], saved['method'], saved['subset']
        rows = diagnostics[(diagnostics.dataset_id == name)&(diagnostics.kind == 'bootstrap')&
                           (diagnostics.method == method)].sort_values('repeat')
        eligible = rows.prediction_available.to_numpy(bool, copy=True)
        if subset == 'common_score_eligible': eligible &= masks[name]
        else: require(subset == 'all_finite', 'Undeclared stability subset')
        require(int(eligible.sum()) == saved['bootstrap_available'], 'Summary availability differs from planned slots')
        summaries.append({**saved, **count_flags(rows, 'planned_'), **count_flags(rows[eligible], 'included_'),
                          'excluded_slots': int((~eligible).sum())})
    for saved in paired.to_dict('records'):
        name, subset = saved['dataset_id'], saved['subset']
        sides = [diagnostics[(diagnostics.dataset_id == name)&(diagnostics.kind == 'bootstrap')&
                            (diagnostics.method == saved[side+'_method'])].sort_values('repeat')
                 for side in ('left', 'right')]
        eligible = sides[0].prediction_available.to_numpy(bool)&sides[1].prediction_available.to_numpy(bool)
        if subset == 'common_score_eligible': eligible &= masks[name]
        else: require(subset == 'all_finite', 'Undeclared paired subset')
        require(int(eligible.sum()) == saved['both_available'], 'Pair availability differs from planned slots')
        pairs.append({**saved, 'paired_excluded_slots': int((~eligible).sum()),
                      **count_flags(sides[0][eligible], 'included_left_'),
                      **count_flags(sides[1][eligible], 'included_right_')})
    return pd.DataFrame(counts), pd.DataFrame(summaries), pd.DataFrame(pairs)


def write_assessment(run, output, retrieval_complete=False, expected_manifest_sha256=None):
    if not retrieval_complete:
        raise ValueError('Complete bootstrap retrieval must be confirmed before reading outputs')
    panel = kernel.SavedPanel(run)
    output = Path(output).resolve()
    require(output != panel.run and panel.run not in output.parents, 'Reporting output must be outside the frozen run')
    if output.exists():
        raise FileExistsError('Preserve earlier reports; use a new output directory')
    if expected_manifest_sha256 is not None:
        require(panel.manifest_sha256 == expected_manifest_sha256, 'Unexpected fitting manifest')
    source_hashes = {str(ROOT/p): kernel.digest(ROOT/p) for p in SOURCES}
    for name in panel.manifest['datasets']:
        panel.tasks(name)  # Verify declared all-dataset/all-replicate scope before calculation.
    output.mkdir(parents=True)
    plan = {'created_utc': datetime.now(timezone.utc).isoformat(), 'fitting_run': str(panel.run),
            'fitting_manifest_sha256': panel.manifest_sha256, 'refitting': False,
            'retrieval_complete_attested': True, 'methods': list(kernel.METHODS),
            'bootstrap_repetitions': panel.manifest['config']['bootstrap_repetitions'],
            'uncertainty': 'Paired bootstrap-draw Monte Carlo error only; not a population efficiency interval',
            'source_sha256': source_hashes,
            'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                         'pandas': pd.__version__, 'scipy': scipy.__version__}}
    kernel.write_json(output/'assessment_plan.json', plan)
    for relative in SOURCES:
        destination = output/'source_snapshot'/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT/relative).read_bytes())
    original = kernel.run_analysis(panel.run, output/'kernel_report')
    require(original['fitting_manifest_sha256'] == panel.manifest_sha256, 'Kernel used another manifest')
    for path, value in original['input_sha256'].items():
        require(path not in panel.hashes or panel.hashes[path] == value, 'Inputs changed between reporting passes')
        panel.hashes[path] = value
    folder = output/'kernel_report'
    diagnostics = enrich_diagnostics(panel, pd.read_csv(folder/'prediction_diagnostics.csv'))
    counts, summary, paired = diagnostic_tables(diagnostics, pd.read_csv(folder/'stability_summary.csv'),
                                                pd.read_csv(folder/'paired_stability.csv'))
    for name, table in [('prediction_diagnostics', diagnostics), ('diagnostic_counts', counts),
                        ('stability_summary', summary), ('paired_stability', paired)]:
        table.to_csv(output/(name+'.csv'), index=False)
    rehash_all(panel.hashes)
    rehash_all(source_hashes)
    for name, sha in original['outputs_sha256'].items():
        require(kernel.digest(folder/name) == sha, 'Kernel reporting output changed')
    manifest = {**plan, 'completed_utc': datetime.now(timezone.utc).isoformat(),
                'input_sha256': panel.hashes, 'final_input_rehash': True,
                'expected_predictions': original['expected_predictions'],
                'recorded_predictions': len(diagnostics),
                'available_predictions': int(diagnostics.prediction_available.sum()),
                'kernel_manifest_sha256': kernel.digest(folder/'analysis_manifest.json'),
                'outputs_sha256': {str(p.relative_to(output)): kernel.digest(p)
                    for p in output.rglob('*') if p.is_file()}}
    kernel.write_json(output/'assessment_manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--retrieval-complete', action='store_true')
    parser.add_argument('--expected-manifest-sha256')
    args = parser.parse_args()
    result = write_assessment(args.run, args.output, args.retrieval_complete, args.expected_manifest_sha256)
    print(json.dumps({k: result[k] for k in ('expected_predictions', 'recorded_predictions', 'available_predictions')}))


if __name__ == '__main__':
    main()
