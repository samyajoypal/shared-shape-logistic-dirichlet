"""All planned methods, exact repetition weights and matched sensitivities."""
import copy

import numpy as np
import pandas as pd
import pytest

import assess_phase3_predictions as report
from test_phase3_applications import frozen, finite_initial, retained_refiner
import phase3_applications as fitting_runner
import run_confirmatory_experiments as base


@pytest.fixture
def panel():
    tasks=[];rows=[];conditional=[]
    methods=['shared_ld','full_ld','gaussian_shared_correlation','gaussian_full']
    for comparison,k,bounds,repeats in [('primary',3,'primary',2),('wide',3,'wide',1),
                                       ('kminus1',2,'primary',1),('kplus1',4,'primary',1)]:
        selected=methods if comparison=='primary' else methods[:2] if comparison=='wide' else methods[:3]
        change={'primary':0.,'wide':-.4,'kminus1':-.2,'kplus1':.2}[comparison]
        for repeat in range(repeats):
            for fold in range(2):
                task={'task_index':len(tasks),'phase':'prediction','dataset_id':'sample','comparison':comparison,
                      'bounds':bounds,'k':k,'kind':'cv','repeat':repeat,'fold':fold,'methods':selected,
                      'conditional_methods':['conditional_log_linear'] if comparison=='primary' else []}
                tasks.append(task)
                for method in selected+task['conditional_methods']:
                    only=method=='conditional_log_linear';delta={'shared_ld':0.,'full_ld':.1,
                        'gaussian_shared_correlation':-.1,'gaussian_full':-.2,'conditional_log_linear':-.3}[method]
                    score=1+2*fold+repeat*.5+delta+change
                    row={**task,'method':method,'conditional_only':only,'recorded':True,'failed':False,
                         'fit_available':True,'evaluation_available':True,'n_test':2 if fold==0 else 6,
                         'n_test_groups':1 if fold==0 else 2,'score_converged':None if method=='gaussian_full' else True,
                         'boundary_hit':method=='full_ld','initial_starts_attempted':5,'initial_starts_failed':0,
                         'starts_attempted':6,'starts_failed':0,'initial_timeout_records':0,
                         'retained_starts_with_exceptions':0,'optimizer_exception_count':0,'timeout_exception_count':0,
                         'initial_fit_seconds':.4,'refinement_fit_seconds':.1,'fit_seconds':.5,
                         'dependency_fit_seconds':.2,'policy_fit_seconds':.7,
                         'mean_log_density':None if only else score,
                         'mean_conditional_log_density':2*score,
                         'group_mean_log_density':None if only else score,
                         'group_mean_conditional_log_density':2*score}
                    (conditional if only else rows).append(row)
    full={'task_index':len(tasks),'phase':'prediction','dataset_id':'sample','comparison':'primary',
          'bounds':'primary','k':3,'kind':'full','repeat':None,'fold':None,'methods':methods,
          'conditional_methods':['conditional_log_linear']}
    tasks.append(full)
    for method in methods+full['conditional_methods']:
        only=method=='conditional_log_linear';row={**full,'method':method,'conditional_only':only,'recorded':True,
            'failed':False,'fit_available':True,'evaluation_available':True,'score_converged':True,'boundary_hit':False,
            'mean_log_density':None if only else 99.,'mean_conditional_log_density':99.,
            'group_mean_log_density':None if only else 99.,'group_mean_conditional_log_density':99.}
        (conditional if only else rows).append(row)
    return {'config':{'folds':2},'tasks':tasks},pd.DataFrame(rows),pd.DataFrame(conditional)


def test_exact_observation_group_weighting_and_complete_repetition(panel):
    manifest,mixtures,conditional=panel;tables=report.tables(manifest,mixtures,conditional)
    reps=tables['complete_repetitions'];r=reps[(reps.comparison=='primary')&(reps.method=='shared_ld')&(reps.repeat==0)].iloc[0]
    assert r.mean_log_density==pytest.approx(2.5)
    assert r.group_mean_log_density==pytest.approx(7/3)
    assert r.mean_conditional_log_density==pytest.approx(5.)
    paired=tables['all_planned_paired_comparisons'];r=paired[(paired.comparison=='primary')&
        (paired.comparator=='gaussian_full')&(paired.metric=='mean_log_density')].iloc[0]
    assert r.mean_shared_minus_comparator==pytest.approx(.2)
    assert r.complete_paired_repetitions==2 and r.strict_score_eligible_pairs==0
    assert 'overlapping' in r.uncertainty


def test_wholly_unavailable_comparator_remains_an_explicit_na(panel):
    manifest,mixtures,conditional=panel
    selected=(mixtures.method=='gaussian_full')&(mixtures.kind=='cv')
    mixtures.loc[selected,['failed','fit_available','evaluation_available']]=[True,False,False]
    result=report.tables(manifest,mixtures,conditional)
    candidate=result['candidate_summary'];r=candidate[(candidate.method=='gaussian_full')&(candidate.kind=='cv')].iloc[0]
    assert r.planned_fits==4 and r.failed_recorded==4 and r.evaluation_available==0
    paired=result['all_planned_paired_comparisons'];rows=paired[paired.comparator=='gaussian_full']
    assert len(rows)==4 and rows.complete_paired_repetitions.eq(0).all()
    assert rows.mean_shared_minus_comparator.isna().all()
    assert rows.availability_note.eq('No complete paired repetition').all()


def test_all_missing_outputs_keep_every_planned_denominator(panel):
    manifest,mixtures,conditional=panel
    for frame in [mixtures,conditional]:
        frame.loc[:,['recorded','fit_available','evaluation_available']]=False
        frame.loc[:,'failed']=True
        frame['score_converged']=None;frame['boundary_hit']=None
        for metric in report.METRICS:frame.loc[:,metric]=np.nan
        for field in report.COSTS:frame.loc[:,field]=np.nan
        for field in report.COUNTERS:frame[field]=0
    result=report.tables(manifest,mixtures,conditional,pd.DataFrame(),pd.DataFrame())
    summary=result['candidate_summary']
    assert summary.planned_fits.sum()==len(result['all_planned_fit_rows'])
    assert summary.missing_fits.sum()==summary.planned_fits.sum()
    assert summary.fit_seconds_available.eq(0).all() and summary.fit_seconds_sum.isna().all()
    pairs=result['all_planned_paired_comparisons']
    assert len(pairs)>0 and pairs.complete_paired_repetitions.eq(0).all()
    assert pairs.mean_shared_minus_comparator.isna().all()


def test_missing_csv_rows_are_recovered_from_manifest_without_imputation(panel):
    manifest,mixtures,conditional=panel
    mixtures=mixtures[~((mixtures.method=='full_ld')&(mixtures.kind=='cv')&(mixtures.comparison=='primary')&(mixtures.repeat==1)&(mixtures.fold==0))]
    result=report.tables(manifest,mixtures,conditional)
    reps=result['complete_repetitions'];r=reps[(reps.method=='full_ld')&(reps.comparison=='primary')&(reps.repeat==1)].iloc[0]
    assert not r.complete and r.missing_fits==1 and r.folds_available==1
    assert pd.isna(r.mean_log_density)
    paired=result['all_planned_paired_comparisons'];r=paired[(paired.comparator=='full_ld')&(paired.comparison=='primary')&
        (paired.metric=='mean_log_density')].iloc[0]
    assert r.planned_repetition_pairs==2 and r.complete_paired_repetitions==1
    assert r.mean_shared_minus_comparator==pytest.approx(-.1)


def test_conditional_reference_has_no_joint_score_but_remains_listed(panel):
    result=report.tables(*panel);paired=result['all_planned_paired_comparisons']
    rows=paired[paired.comparator=='conditional_log_linear']
    joint=rows[rows.density=='joint'];conditional=rows[rows.density=='conditional']
    assert len(joint)==2 and not joint.applicable.any() and joint.mean_shared_minus_comparator.isna().all()
    assert conditional.complete_paired_repetitions.eq(2).all()
    np.testing.assert_allclose(conditional.mean_shared_minus_comparator,.6)


def test_boundaries_retained_strict_flags_separate_and_cost_denominators(panel):
    result=report.tables(*panel);paired=result['all_planned_paired_comparisons']
    row=paired[(paired.comparison=='primary')&(paired.comparator=='full_ld')&(paired.metric=='mean_log_density')].iloc[0]
    assert row.pairs_with_any_boundary==row.complete_paired_repetitions==row.strict_score_eligible_pairs==2
    summary=result['candidate_summary'];row=summary[(summary.comparison=='primary')&(summary.kind=='cv')&(summary.method=='full_ld')].iloc[0]
    assert row.boundary_hits==4 and row.strict_score_passes==4
    assert row.fit_seconds_available==4 and row.fit_seconds_sum==2.
    assert row.dependency_fit_seconds_sum==pytest.approx(.8)
    assert row.policy_fit_seconds_sum==pytest.approx(2.8)


def test_sensitivity_matches_repeat_zero_only(panel):
    result=report.tables(*panel);s=result['matched_sensitivity_changes']
    for comparison,expected in [('wide',-.4),('kminus1',-.2),('kplus1',.2)]:
        rows=s[(s.comparison==comparison)&(s.metric=='mean_log_density')]
        assert rows.complete_partition_pair.all() and rows.paired_folds_available.eq(2).all()
        np.testing.assert_allclose(rows.sensitivity_minus_primary,expected)
    assert set(s[s.comparison=='wide'].method)=={'shared_ld','full_ld'}


def test_incomplete_sensitivity_never_uses_available_fold_average(panel):
    manifest,mixtures,conditional=panel
    row=(mixtures.comparison=='wide')&(mixtures.method=='shared_ld')&(mixtures.fold==0)
    mixtures.loc[row,['failed','evaluation_available']]=[True,False]
    result=report.tables(manifest,mixtures,conditional);s=result['matched_sensitivity_changes']
    rows=s[(s.comparison=='wide')&(s.method=='shared_ld')]
    assert not rows.complete_partition_pair.any() and rows.paired_folds_available.eq(1).all()
    assert rows.sensitivity_minus_primary.isna().all()


def test_full_data_scores_are_descriptive_and_do_not_enter_cv(panel):
    result=report.tables(*panel);summary=result['candidate_summary']
    full=summary[(summary.kind=='full')&(summary.method=='shared_ld')].iloc[0]
    cv=summary[(summary.kind=='cv')&(summary.comparison=='primary')&(summary.method=='shared_ld')].iloc[0]
    assert full.mean_log_density_mean==99. and 'not held-out' in full.interpretation
    assert cv.mean_log_density_mean==pytest.approx(2.75)


def test_original_table_missing_zero_complete_row_is_preserved_and_equivalent(panel):
    manifest,mixtures,conditional=panel
    mask=(mixtures.method=='gaussian_full')&(mixtures.kind=='cv')
    mixtures.loc[mask,['failed','fit_available','evaluation_available']]=[True,False,False]
    result=report.tables(manifest,mixtures,conditional)
    old_reps=result['complete_repetitions'];old_reps=old_reps[old_reps.complete].copy()
    old_pairs=result['all_planned_paired_comparisons'];old_pairs=old_pairs[old_pairs.comparator!='gaussian_full'].copy()
    checked=report.tables(manifest,mixtures,conditional,old_reps,old_pairs)
    assert len(checked['all_planned_paired_comparisons'])==len(old_pairs)+4
    old_pairs.loc[old_pairs.index[0],'mean_shared_minus_comparator']+=1
    with pytest.raises(report.IntegrityError,match='paired summary differs'):
        report.tables(manifest,mixtures,conditional,old_reps,old_pairs)


@pytest.mark.parametrize('which',['duplicate','extra_method','scope','invalid_available'])
def test_malformed_source_rows_rejected(panel,which):
    manifest,mixtures,conditional=panel
    if which=='duplicate':mixtures=pd.concat([mixtures,mixtures.iloc[[0]]],ignore_index=True)
    elif which=='extra_method':mixtures.loc[0,'method']='undeclared_model'
    elif which=='scope':mixtures.loc[0,'k']=99
    else:mixtures.loc[0,'failed']=True
    with pytest.raises(report.IntegrityError):report.tables(manifest,mixtures,conditional)


def test_float_csv_roundtrip_keeps_boolean_availability_and_empty_fields(panel):
    manifest,mixtures,conditional=panel
    a=report.csv_table(mixtures.to_csv(index=False).encode());b=report.csv_table(conditional.to_csv(index=False).encode())
    result=report.tables(manifest,a,b)
    assert result['all_planned_fit_rows'].evaluation_available.all()
    assert result['complete_repetitions'].complete.all()


def test_no_partial_input_read_without_retrieval_confirmation(monkeypatch,tmp_path):
    def forbidden(*args,**kwargs):raise AssertionError('Partial run was read')
    monkeypatch.setattr(report,'SavedPanel',forbidden)
    with pytest.raises(ValueError,match='Complete retrieval'):
        report.write_assessment(tmp_path/'run',tmp_path/'out')


def test_end_to_end_synthetic_frozen_aggregate_keeps_omitted_candidate(frozen):
    output,manifest=frozen
    def failed_full_gaussian(method,y,*args):
        if method=='gaussian_full' and y.shape[1]==2:
            raise base.FitFailure('Synthetic all-start failure',[{'failed':True,'message':'injected'}])
        return finite_initial(method,y,*args)
    for task in manifest['tasks']:
        if task['phase']=='prediction':
            fitting_runner.run_task(output,task['task_index'],initial_fitter=failed_full_gaussian,refiner=retained_refiner)
    fitting_runner.aggregate(output,'prediction')
    assessment=output.parent/'assessment'
    metadata=report.write_assessment(output,assessment,retrieval_complete=True,
        expected_manifest_sha256=fitting_runner.control.digest(output/'manifest.json'))
    assert metadata['fitting_performed'] is False and metadata['paired_rows_absent_from_original_table']==4
    pairs=pd.read_csv(assessment/'all_planned_paired_comparisons.csv')
    missing=pairs[(pairs.dataset_id=='chem')&(pairs.comparator=='gaussian_full')]
    assert len(missing)==4 and missing.complete_paired_repetitions.eq(0).all()
    assert missing.mean_shared_minus_comparator.isna().all()
    assert metadata['planned_mixture_fits']==112 and metadata['planned_conditional_fits']==10
    with pytest.raises(FileExistsError):report.write_assessment(output,assessment,retrieval_complete=True)


def test_readonly_input_run_cannot_be_overwritten(frozen):
    output,manifest=frozen
    with pytest.raises(ValueError,match='outside the frozen'):
        report.write_assessment(output,output/'new_report',retrieval_complete=True)
