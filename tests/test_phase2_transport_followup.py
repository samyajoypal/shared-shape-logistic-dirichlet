"""Transport adapter selection, reference measure and reporting integrity."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import phase2_gaussian_control as original
import phase2_gaussian_transport_followup as follow

ROOT = Path(__file__).resolve().parents[1]
PRIMARY = ROOT / 'results/reproduction/applications'


@pytest.fixture
def actual_spec():
    if not (PRIMARY / "specification.json").exists():
        pytest.skip("Optional integration test needs a newly reproduced application run")
    return json.loads((PRIMARY / 'specification.json').read_text())


def test_all_original_areas_selected_and_original_module_isolated(actual_spec):
    tasks = follow.task_map(actual_spec, 'applications')
    assert len(tasks) == 18
    assert [x['task_index'] for x in tasks] == list(range(18))
    assert [x['primary_task_index'] for x in tasks] == list(range(255, 264)) + list(range(519, 528))
    expected = ['Apulia.north', 'Apulia.south', 'Calabria', 'Liguria.east',
                'Liguria.west', 'Sardinia.coast', 'Sardinia.inland', 'Sicily', 'Umbria']
    for variant in ['olive3_oleic', 'olive6_oleic']:
        subset = [x for x in tasks if x['variant'] == variant]
        assert [x['area'] for x in subset] == expected
        assert [x['fold'] for x in subset] == list(range(9))
        assert all(x['kind'] == 'leave_area_out' for x in subset)
    assert follow.control is not original
    assert original.task_map is not follow.task_map
    assert len(original.task_map(actual_spec, 'applications')) == 750
    assert all(t['kind'] != 'leave_area_out' for t in original.task_map(actual_spec, 'applications'))


def test_followup_configuration_preserves_numerical_policy():
    old = json.loads((ROOT / 'configs/phase2_gaussian_control.json').read_text())
    new = json.loads(follow.CONFIG.read_text())
    for key in ['bounds', 'inherit_optimizer_settings', 'diagonal_policy',
                'method', 'threads', 'parameter_count', 'failure_policy']:
        assert new[key] == old[key]
    assert 'after seeing' in new['status']
    files = follow.source_files()
    assert ROOT / 'scripts/phase2_gaussian_control.py' in files
    assert ROOT / 'scripts/gaussian_shared_correlation.py' in files
    assert Path(follow.__file__) in files and follow.CONFIG in files


def test_freeze_is_explicitly_post_result_and_verifies_snapshot(tmp_path, monkeypatch, actual_spec):
    main = tmp_path / 'primary'
    main.mkdir()
    (main / 'specification.json').write_text(json.dumps(actual_spec))
    monkeypatch.setattr(follow.control, 'versions', lambda: actual_spec['runtime_versions'])
    out = tmp_path / 'followup'
    follow.freeze(out, main, follow.CONFIG)
    manifest, _, _ = follow.control.verify_run(out)
    assert len(manifest['tasks']) == 18
    assert 'post-result' in manifest['interpretation'].lower()
    assert 'prespecified separate' not in manifest['interpretation'].lower()
    path = out / 'source_snapshot/scripts/phase2_gaussian_transport_followup.py'
    path.write_text(path.read_text() + '\n# injected mismatch\n')
    with pytest.raises(RuntimeError, match='Frozen control source changed'):
        follow.control.verify_run(out)


def test_load_preserves_saved_split_transform_partitions_and_conditional_role(tmp_path, actual_spec):
    task = follow.task_map(actual_spec, 'applications')[0]
    upstream = actual_spec['tasks'][task['primary_task_index']]
    primary = tmp_path / 'primary'
    data = primary / 'executed_sources/data/oliveoil.csv'
    data.parent.mkdir(parents=True)
    data.write_bytes((ROOT / 'data/oliveoil.csv').read_bytes())
    follow.control.atomic_json(primary / 'specification.json', actual_spec)
    variant = next(v for v in actual_spec['config']['variants'] if v['id'] == task['variant'])
    y, _, _, _, _ = follow.control.application_data(primary, actual_spec, variant)
    train, test = np.asarray(upstream['train']), np.asarray(upstream['test'])
    partitions = [(np.arange(len(train)) + shift) % variant['k'] for shift in range(5)]
    directory = primary / 'tasks' / f"{task['primary_task_index']:05d}"
    follow.control.atomic_json(directory / 'split.json', {**upstream,
        'center': y[train].mean(0), 'scale': y[train].std(0), 'partitions': partitions})
    follow.control.atomic_json(directory / 'completion.json', {'task': upstream, 'records': [],
        'specification_sha256': follow.control.digest(primary / 'specification.json')})
    case = follow.load_case(primary, actual_spec, task, 'applications')
    assert case['evaluation_role'] == 'held_out_collection_area'
    assert case['conditional_target'] == 'linoleic/oleic'
    assert case['target_index'] == 1
    np.testing.assert_array_equal(case['test_indices'], test)
    np.testing.assert_array_equal(case['train'], y[train])
    np.testing.assert_array_equal(case['test'], y[test])
    np.testing.assert_array_equal(case['partitions'], partitions)
    np.testing.assert_array_equal(case['center'], y[train].mean(0))
    np.testing.assert_array_equal(case['scale'], y[train].std(0))
    assert case['settings']['n_starts'] == 5
    assert case['seed'] == upstream['seed']
    assert not np.any(case['raw_jacobian'])


@pytest.fixture
def aggregate_case(tmp_path, monkeypatch, actual_spec):
    output, primary = tmp_path / 'output', tmp_path / 'primary'
    output.mkdir(); primary.mkdir()
    source = primary / 'binding_source.json'; source.write_text('{"frozen":true}\n')
    manifest = {'tasks': follow.task_map(actual_spec, 'applications')}
    follow.control.atomic_json(output / 'manifest.json', manifest)
    monkeypatch.setattr(follow.control, 'verify_run', lambda *args: (manifest, primary, actual_spec))
    records = pd.read_csv(PRIMARY / 'fit_records.csv')
    for task in manifest['tasks']:
        rows = records[records.task_index == task['primary_task_index']]
        saved = json.loads(rows.to_json(orient='records'))
        baselines = {x['method']: x for x in saved}
        shared = baselines['shared_ld']
        # Deliberately area-dependent gap makes weighted/equal-area means differ.
        gap = .01 * (task['fold'] + 1)
        row = {**task, 'method': follow.control.METHOD, 'failed': False,
               'evaluation_available': True, 'n_test': shared['n_test'],
               'test_log_density': shared['mean_log_density'] - gap,
               'conditional_log_density': shared['mean_conditional_log_density'] - 2 * gap,
               'boundary_hit': False, 'score_converged': True}
        binding = {'input_sha256': {'binding_source.json': follow.control.digest(source)}}
        value = {'status': 'complete', 'task': task, 'row': row,
                 'manifest_sha256': follow.control.digest(output / 'manifest.json'),
                 'input_binding_sha256': follow.control.semantic_hash(binding),
                 'baseline_records': baselines}
        directory = output / 'tasks' / f"task_{task['task_index']:06d}"
        follow.control.atomic_json(directory / 'result.json', value)
        follow.control.atomic_json(directory / 'input_binding.json', binding)
    return output, primary, manifest


def test_weighted_baselines_reproduce_original_exactly_and_pair_areas(aggregate_case):
    output, _, _ = aggregate_case
    status = follow.aggregate(output)
    assert status['planned'] == status['recorded'] == status['available'] == 18
    result = pd.read_csv(output / 'summary.csv')
    old = pd.read_csv(PRIMARY / 'summary.csv')
    old = old[old.kind == 'leave_area_out']
    common = result.merge(old, on=['variant', 'method'])
    assert len(common) == 16
    np.testing.assert_allclose(common.joint_weighted, common.mean_log_density, atol=1e-10, rtol=0)
    np.testing.assert_allclose(common.conditional_weighted, common.mean_conditional_log_density, atol=1e-10, rtol=0)
    comparisons = pd.read_csv(output / 'paired_comparisons.csv')
    comparisons = comparisons[comparisons.comparator == follow.control.METHOD]
    weights = np.array([25, 206, 56, 50, 50, 33, 65, 36, 51])
    for _, row in comparisons.iterrows():
        multiplier = 2 if row.measure == 'conditional' else 1
        assert row.available_pairs == 9 and row.areas_shared_better == 9
        assert row.shared_minus_other_weighted == pytest.approx(np.average(.01 * np.arange(1, 10), weights=weights) * multiplier, abs=1e-10)
        assert row.shared_minus_other_equal_area_mean == pytest.approx(.05 * multiplier, abs=1e-10)


def test_missing_area_preserves_denominator_and_suppresses_partial_mean(aggregate_case):
    output, _, _ = aggregate_case
    (output / 'tasks/task_000000/result.json').unlink()
    status = follow.aggregate(output)
    assert status['recorded'] == status['available'] == 17
    summary = pd.read_csv(output / 'summary.csv')
    control = summary[(summary.variant == 'olive3_oleic') & (summary.method == follow.control.METHOD)].iloc[0]
    assert control.planned_areas == 9 and control.available_areas == 8
    assert not control.complete and pd.isna(control.joint_weighted)
    paired = pd.read_csv(output / 'paired_comparisons.csv')
    bad = paired[(paired.variant == 'olive3_oleic') & (paired.comparator == follow.control.METHOD)]
    assert bad.available_pairs.eq(8).all()
    assert bad.shared_minus_other_weighted.isna().all()


def test_all_missing_outputs_remain_complete_reports(aggregate_case):
    output, _, _ = aggregate_case
    for p in output.glob('tasks/*/result.json'):
        p.unlink()
    status = follow.aggregate(output)
    assert status['planned'] == 18 and status['recorded'] == status['available'] == 0
    summary = pd.read_csv(output / 'summary.csv')
    controls = summary[summary.method == follow.control.METHOD]
    assert len(controls) == 2 and controls.planned_areas.eq(9).all()
    assert controls.available_areas.eq(0).all() and controls.joint_weighted.isna().all()


@pytest.mark.parametrize('field,value', [('test_log_density', None), ('conditional_log_density', None), ('n_test', 0)])
def test_available_nonfinite_or_empty_evaluation_is_rejected(aggregate_case, field, value):
    output, _, _ = aggregate_case
    path = output / 'tasks/task_000000/result.json'
    result = json.loads(path.read_text()); result['row'][field] = value
    follow.control.atomic_json(path, result)
    with pytest.raises(ValueError):
        follow.aggregate(output)


def test_identity_and_upstream_content_changes_rejected(aggregate_case):
    output, primary, _ = aggregate_case
    path = output / 'tasks/task_000000/result.json'
    result = json.loads(path.read_text()); result['manifest_sha256'] = 'wrong'
    follow.control.atomic_json(path, result)
    with pytest.raises(ValueError, match='identity'):
        follow.aggregate(output)
    result['manifest_sha256'] = follow.control.digest(output / 'manifest.json')
    follow.control.atomic_json(path, result)
    (primary / 'binding_source.json').write_text('{"changed":true}\n')
    with pytest.raises(ValueError, match='input changed'):
        follow.aggregate(output)
