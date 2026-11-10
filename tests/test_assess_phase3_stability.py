"""No fits: synthetic saved checkpoints and direct paired-draw calculations."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import assess_phase3_stability as report
import phase3_stability as kernel
from test_phase3_applications import (
    frozen, run, read, write, runner, finite_initial, retained_refiner, base,
)


def complete_stability(output, manifest, initial_fitter=finite_initial):
    for task in manifest['tasks']:
        if task['primary'] and task['kind'] in ('full', 'bootstrap'):
            runner.run_task(output, task['task_index'], initial_fitter=initial_fitter,
                            refiner=retained_refiner)


def test_no_partial_retrieval_is_read(tmp_path, monkeypatch):
    def forbidden(*args): raise AssertionError('Input was read before retrieval acknowledgement')
    monkeypatch.setattr(kernel, 'SavedPanel', forbidden)
    with pytest.raises(ValueError, match='retrieval'):
        report.write_assessment(tmp_path/'missing', tmp_path/'out')


def test_complete_synthetic_report_preserves_kernel_numbers_and_all_slots(frozen, tmp_path, monkeypatch):
    source, manifest = frozen
    complete_stability(source, manifest)
    monkeypatch.setattr(kernel, 'GROUP_SENSITIVITY', ('chem',))
    output = tmp_path/'assessment'
    result = report.write_assessment(source, output, True, kernel.digest(source/'manifest.json'))
    assert result['final_input_rehash'] and result['refitting'] is False
    assert result['expected_predictions'] == result['recorded_predictions'] == result['available_predictions'] == 24
    diagnostics = pd.read_csv(output/'prediction_diagnostics.csv')
    assert len(diagnostics) == 24 and diagnostics.source_recorded.all()
    for filename in ('stability_summary.csv', 'paired_stability.csv'):
        enhanced = pd.read_csv(output/filename)
        original = pd.read_csv(output/'kernel_report'/filename)
        pd.testing.assert_frame_equal(enhanced[original.columns], original)
    summaries = pd.read_csv(output/'stability_summary.csv')
    assert set(summaries.weighting) == {'equal_observations', 'equal_groups'}
    assert set(summaries.planned_slots) == {2}
    assert (summaries.included_slots == summaries.bootstrap_available).all()
    assert (summaries.planned_score_converged_true + summaries.planned_score_converged_false +
            summaries.planned_score_converged_unknown == 2).all()
    for path, sha in result['outputs_sha256'].items():
        assert kernel.digest(output/path) == sha


def test_every_missing_draw_and_comparison_is_retained(frozen, tmp_path):
    source, manifest = frozen
    result = report.write_assessment(source, tmp_path/'missing', True)
    assert result['recorded_predictions'] == 24 and result['available_predictions'] == 0
    counts = pd.read_csv(tmp_path/'missing/diagnostic_counts.csv')
    assert counts.source_recorded.eq(0).all()
    assert (counts.source_failed_unknown == counts.slots).all()
    summary = pd.read_csv(tmp_path/'missing/stability_summary.csv')
    pairs = pd.read_csv(tmp_path/'missing/paired_stability.csv')
    assert summary.bootstrap_requested.eq(2).all() and summary.bootstrap_available.eq(0).all()
    assert pairs.both_available.eq(0).all() and pairs.variance_ratio.isna().all()
    assert len(pairs) == len(manifest['datasets'])*3*2


def test_failed_parameterless_method_keeps_original_flags(frozen, tmp_path):
    source, manifest = frozen
    def fail_shared(method, *args):
        if method == 'shared_ld':
            raise base.FitFailure('Synthetic failed start', [{'failed': True, 'message': 'failed'}])
        return finite_initial(method, *args)
    complete_stability(source, manifest, fail_shared)
    output = tmp_path/'failures'
    report.write_assessment(source, output, True)
    diagnostic = pd.read_csv(output/'prediction_diagnostics.csv')
    shared = diagnostic[diagnostic.method == 'shared_ld']
    assert shared.source_recorded.all() and shared.source_failed.all()
    assert not shared.prediction_available.any()
    original = pd.read_csv(output/'kernel_report/prediction_diagnostics.csv')
    assert original.loc[original.method == 'shared_ld', 'source_failed'].isna().all()
    counts = pd.read_csv(output/'diagnostic_counts.csv')
    shared_counts = counts[counts.method == 'shared_ld']
    assert (shared_counts.source_failed_true == shared_counts.slots).all()


def test_finite_points_from_failed_evaluation_keep_failure_labels(frozen, tmp_path, monkeypatch):
    source, manifest = frozen
    original = runner.evaluate
    def failed_evaluation(model, case):
        if model.method == 'shared_ld':
            raise FloatingPointError('Synthetic evaluation failure after saved finite fit')
        return original(model, case)
    monkeypatch.setattr(runner, 'evaluate', failed_evaluation)
    complete_stability(source, manifest)
    output = tmp_path/'finite_failed'
    report.write_assessment(source, output, True)
    diagnostics = pd.read_csv(output/'prediction_diagnostics.csv')
    shared = diagnostics[diagnostics.method == 'shared_ld']
    assert shared.prediction_available.all() and shared.source_failed.all()
    assert shared.finite_from_failed_source.all()
    counts = pd.read_csv(output/'diagnostic_counts.csv')
    shared_counts = counts[counts.method == 'shared_ld']
    assert (shared_counts.finite_from_failed_source == shared_counts.slots).all()


def synthetic_diagnostics():
    records = []
    for method in kernel.METHODS:
        for b in range(4):
            finite = not (method == 'full_ld' and b == 0)
            failed = method == 'shared_ld' and b == 1
            score = None if method == 'gaussian_shared_correlation' and b == 2 else True
            boundary = b == 3
            records.append({'dataset_id': 'chem', 'kind': 'bootstrap', 'method': method, 'repeat': b,
                'source_recorded': True, 'prediction_available': finite, 'source_failed': failed,
                'score_converged': score, 'boundary_hit': boundary,
                'finite_from_failed_source': finite and failed,
                'finite_score_ineligible': finite and score is not True})
    return pd.DataFrame(records)


def test_pair_counts_distinguish_unavailable_score_unknown_and_retained_bounds():
    diagnostics = synthetic_diagnostics()
    summaries = pd.DataFrame([
        {'dataset_id': 'chem', 'method': 'shared_ld', 'subset': 'all_finite', 'bootstrap_available': 4},
        {'dataset_id': 'chem', 'method': 'shared_ld', 'subset': 'common_score_eligible', 'bootstrap_available': 2}])
    pairs = pd.DataFrame([
        {'dataset_id': 'chem', 'left_method': 'shared_ld', 'right_method': 'full_ld',
         'subset': 'all_finite', 'both_available': 3},
        {'dataset_id': 'chem', 'left_method': 'shared_ld', 'right_method': 'full_ld',
         'subset': 'common_score_eligible', 'both_available': 2}])
    counts, methods, paired = report.diagnostic_tables(diagnostics, summaries, pairs)
    assert methods.iloc[0].included_source_failed_true == 1  # finite original failure is not erased
    assert methods.iloc[1].included_boundary_hit_true == 1   # boundary retained in strict subset
    assert paired.iloc[0].paired_excluded_slots == 1
    assert paired.iloc[1].paired_excluded_slots == 2
    gaussian = counts[counts.method == 'gaussian_shared_correlation'].iloc[0]
    assert gaussian.score_converged_unknown == 1 and gaussian.prediction_missing == 0


def test_duplicate_bootstrap_indices_are_rejected_even_when_count_matches():
    diagnostics = synthetic_diagnostics()
    diagnostics.loc[(diagnostics.method == 'full_ld') & (diagnostics.repeat == 3), 'repeat'] = 2
    with pytest.raises(kernel.IntegrityError, match='bootstrap method index'):
        report.diagnostic_tables(diagnostics, pd.DataFrame(), pd.DataFrame())


def test_diagnostic_repeat_must_match_canonical_task(frozen, tmp_path):
    source, _ = frozen
    kernel.run_analysis(source, tmp_path/'kernel')
    diagnostics = pd.read_csv(tmp_path/'kernel/prediction_diagnostics.csv')
    diagnostics.loc[diagnostics.kind == 'full', 'repeat'] = 0
    with pytest.raises(kernel.IntegrityError, match='diagnostic repeat'):
        report.enrich_diagnostics(kernel.SavedPanel(source), diagnostics)


def test_mc_ratio_interval_matches_direct_delete_one_and_is_paired():
    rng = np.random.default_rng(71)
    left = rng.normal(size=(20, 6))
    right = .3*left + rng.normal(size=left.shape)
    weights = kernel.grid_weights(['a', 'a', 'a', 'b', 'b', 'c'], True)
    actual = kernel.summarize_weighted_pair(left, right, left[0], right[0], weights, 20)
    ratio = (np.var(left, axis=0, ddof=1)@weights)/(np.var(right, axis=0, ddof=1)@weights)
    delete = np.array([np.log((np.var(np.delete(left, i, axis=0), axis=0, ddof=1)@weights)/
                             (np.var(np.delete(right, i, axis=0), axis=0, ddof=1)@weights)) for i in range(20)])
    se = np.sqrt(19/20*np.sum((delete-delete.mean())**2))
    assert actual['variance_ratio'] == pytest.approx(ratio)
    assert actual['log_variance_ratio_mcse'] == pytest.approx(se)
    np.testing.assert_allclose([actual['ratio_mc95_low'], actual['ratio_mc95_high']],
                              np.exp(np.log(ratio)+np.array([-1, 1])*1.96*se))
    shuffled = kernel.summarize_weighted_pair(left, right[::-1], left[0], right[0], weights, 20)
    assert shuffled['variance_ratio'] == pytest.approx(ratio)
    assert not np.isclose(shuffled['log_variance_ratio_mcse'], se)


@pytest.mark.parametrize('mutation', ['change', 'delete'])
def test_final_rehash_rejects_consumed_input_mutation(tmp_path, mutation):
    path = tmp_path/'input.json'; path.write_text('{}')
    hashes = {str(path): kernel.digest(path)}
    if mutation == 'change': path.write_text('{"changed":true}')
    else: path.unlink()
    with pytest.raises(kernel.IntegrityError, match='changed'):
        report.rehash_all(hashes)


def test_end_to_end_rejects_input_changed_during_reporting(frozen, tmp_path, monkeypatch):
    source, _ = frozen
    original = kernel.run_analysis
    def change_after_read(run_dir, output):
        result = original(run_dir, output)
        path = source/'manifest.json'; path.write_bytes(path.read_bytes()+b'\n')
        return result
    monkeypatch.setattr(kernel, 'run_analysis', change_after_read)
    with pytest.raises(kernel.IntegrityError, match='changed'):
        report.write_assessment(source, tmp_path/'changed', True)
    assert not (tmp_path/'changed/assessment_manifest.json').exists()


def test_runtime_and_completion_scope_are_checked(frozen):
    source, manifest = frozen
    task = next(t for t in manifest['tasks'] if t['kind'] == 'bootstrap')
    run(source, task['task_index'])
    folder = source/'tasks'/f"task_{task['task_index']:06d}"
    path = folder/'completion.json'; record = read(path)
    record['runtime_versions']['python'] = 'wrong'; write(path, record)
    with pytest.raises(kernel.IntegrityError, match='runtime'):
        report.completed_rows(kernel.SavedPanel(source), task)


def test_output_overwrite_and_wrong_manifest_are_rejected(frozen, tmp_path):
    source, _ = frozen
    with pytest.raises(kernel.IntegrityError, match='outside'):
        report.write_assessment(source, source/'report', True)
    with pytest.raises(kernel.IntegrityError, match='manifest'):
        report.write_assessment(source, tmp_path/'wrong', True, '0'*64)
    output = tmp_path/'existing'; output.mkdir()
    with pytest.raises(FileExistsError):
        report.write_assessment(source, output, True)
