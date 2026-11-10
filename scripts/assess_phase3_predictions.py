"""Read-only all-candidate reporting for the frozen additional applications.

The frozen paired-CV table is based on available complete repetitions and can
omit a comparator with none. This report starts from the manifest and retains
such comparisons as explicit NA with zero available pairs. No fit, selection,
dataset or numerical criterion is changed. Use only after complete retrieval.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from phase3_stability import SavedPanel, IntegrityError, semantic_hash

METRICS = {'mean_log_density': ('joint','observation'),
           'mean_conditional_log_density': ('conditional','observation'),
           'group_mean_log_density': ('joint','equal_group'),
           'group_mean_conditional_log_density': ('conditional','equal_group')}
SCOPE = ['dataset_id','comparison','bounds','k']
REPEAT_KEY = SCOPE+['method','repeat']
COSTS = ['initial_fit_seconds','refinement_fit_seconds','fit_seconds',
         'dependency_fit_seconds','policy_fit_seconds']
COUNTERS = ['initial_starts_attempted','initial_starts_failed','initial_timeout_records',
            'starts_attempted','starts_failed','retained_starts_with_exceptions',
            'optimizer_exception_count','timeout_exception_count']


def need(condition,message):
    if not condition: raise IntegrityError(message)


def finite(value):
    return value is not None and pd.notna(value) and np.isfinite(value)


def truth(value):
    return (isinstance(value,(bool,np.bool_)) and bool(value)) or isinstance(value,str) and value=='True'


def optional_flag(value):
    if value is None or pd.isna(value):return None
    if isinstance(value,(bool,np.bool_)):return bool(value)
    if value in ['True','False']:return value=='True'
    raise IntegrityError('Invalid recorded Boolean flag')


def csv_table(raw):
    try: return pd.read_csv(io.BytesIO(raw))
    except pd.errors.EmptyDataError: return pd.DataFrame()


def planned_rows(manifest):
    rows=[]
    for task in manifest['tasks']:
        if task['phase']!='prediction': continue
        for conditional_only,methods in [(False,task['methods']),(True,task['conditional_methods'])]:
            for method in methods:
                rows.append({**task,'method':method,'conditional_only':conditional_only})
    return pd.DataFrame(rows)


def align_candidates(manifest,mixtures,conditional):
    """A missing table row becomes a planned unavailable fit, never disappears."""
    plan=planned_rows(manifest);reported=pd.concat([mixtures,conditional],ignore_index=True)
    need(not reported.duplicated(['task_index','method']).any(),'Duplicate reported task/method')
    keys=set(zip(plan.task_index,plan.method))
    need(set(zip(reported.task_index,reported.method)) <= keys,'Unexpected reported comparator or task')
    lookup={(int(r['task_index']),r['method']):r for r in reported.to_dict('records')}
    aligned=[]
    for expected in plan.to_dict('records'):
        saved=lookup.get((expected['task_index'],expected['method']))
        if saved is None:
            row={**expected,'recorded':False,'failed':True,'fit_available':False,'evaluation_available':False}
        else:
            for field in ['dataset_id','comparison','kind','bounds','k','conditional_only']:
                actual=truth(saved.get(field)) if field=='conditional_only' else saved.get(field)
                need(actual==expected[field],'Reported '+field+' differs from the manifest')
            for field in ['repeat','fold']:
                if pd.notna(expected[field]):
                    need(saved.get(field)==expected[field],'Reported '+field+' differs from the manifest')
                else:
                    need(saved.get(field) is None or pd.isna(saved.get(field)), 'Full-data task has an unexpected '+field)
            row={**saved,**expected}
            for flag in ['recorded','failed','fit_available','evaluation_available']:
                row[flag]=truth(saved.get(flag))
            need(not row['evaluation_available'] or row['recorded'] and row['fit_available'] and not row['failed'],
                 'Available prediction is not a recorded successful fit')
            if not row['recorded']:
                need(not row['fit_available'] and not row['evaluation_available'], 'Unrecorded fit is marked available')
                need(all(saved.get(field) is None or pd.isna(saved.get(field)) for field in
                         list(METRICS)+COSTS+['score_converged','boundary_hit']),
                     'Unrecorded fit has numerical results or costs')
                need(all(not finite(saved.get(field)) or saved.get(field)==0 for field in COUNTERS),
                     'Unrecorded fit has recorded optimization attempts')
        for metric in METRICS:
            row.setdefault(metric,np.nan)
        row.setdefault('score_converged',None);row.setdefault('boundary_hit',None)
        row['score_converged']=optional_flag(row['score_converged']);row['boundary_hit']=optional_flag(row['boundary_hit'])
        aligned.append(row)
    return pd.DataFrame(aligned)


def complete_repetitions(frame,folds):
    records=[]
    cv=frame[frame.kind=='cv']
    for keys,group in cv.groupby(REPEAT_KEY,dropna=False,sort=False):
        need(len(group)==folds and set(group.fold)==set(range(folds)),'Planned CV repetition has missing/duplicate folds')
        row=dict(zip(REPEAT_KEY,keys));conditional_only=bool(group.conditional_only.iloc[0])
        available=group.evaluation_available.eq(True)
        is_complete=bool(available.all())
        applicable=[m for m,(kind,weight) in METRICS.items() if not conditional_only or kind=='conditional']
        if is_complete:
            need(group.n_test.gt(0).all() and group.n_test_groups.gt(0).all(),'Invalid evaluation weights')
            need(all(np.isfinite(pd.to_numeric(group[m],errors='coerce')).all() for m in applicable),'Available repetition has nonfinite applicable score')
        row.update(conditional_only=conditional_only,folds_planned=folds,folds_available=int(available.sum()),
                   complete=is_complete,score_flags_available=int(group.score_converged.notna().sum()),
                   strict_score_eligible=is_complete and bool(group.score_converged.eq(True).all()),
                   any_boundary=bool(group.boundary_hit.eq(True).any()),
                   fit_failures=int((group.recorded & group.failed).sum()),missing_fits=int((~group.recorded).sum()))
        for metric,(kind,weighting) in METRICS.items():
            weights='n_test' if weighting=='observation' else 'n_test_groups'
            row[metric]=(float(np.average(group[metric],weights=group[weights]))
                         if is_complete and metric in applicable else np.nan)
        records.append(row)
    return pd.DataFrame(records)


def candidate_summary(frame,repetitions):
    records=[]
    for keys,group in frame.groupby(SCOPE+['kind','method','conditional_only'],dropna=False,sort=False):
        row=dict(zip(SCOPE+['kind','method','conditional_only'],keys))
        row.update(planned_fits=len(group),recorded_fits=int(group.recorded.sum()),
            missing_fits=int((~group.recorded).sum()),fit_available=int(group.fit_available.sum()),
            evaluation_available=int(group.evaluation_available.sum()),failed_recorded=int((group.recorded & group.failed).sum()),
            score_flags_available=int(group.score_converged.notna().sum()),
            strict_score_passes=int(group.score_converged.eq(True).sum()),
            strict_score_failures=int(group.score_converged.eq(False).sum()),
            boundary_flags_available=int(group.boundary_hit.notna().sum()),boundary_hits=int(group.boundary_hit.eq(True).sum()))
        for field in COUNTERS:
            row[field]=int(group[field].fillna(0).sum()) if field in group else 0
        for field in COSTS:
            values=pd.to_numeric(group[field],errors='coerce') if field in group else pd.Series(dtype=float)
            values=values[np.isfinite(values)]
            row[field+'_available']=len(values)
            for suffix,operation in [('sum',pd.Series.sum),('mean',pd.Series.mean),('median',pd.Series.median),('maximum',pd.Series.max)]:
                row[field+'_'+suffix]=float(operation(values)) if len(values) else np.nan
        if row['kind']=='cv':
            subset=repetitions.copy()
            for field in SCOPE+['method']:subset=subset[subset[field]==row[field]]
            complete=subset[subset.complete]
            row.update(planned_repetitions=len(subset),complete_repetitions=len(complete),
                       strict_score_complete_repetitions=int(subset.strict_score_eligible.sum()),
                       interpretation='Held-out complete CV repetitions; descriptive mean/range')
            for metric in METRICS:
                values=complete[metric].dropna()
                row[metric+'_mean']=values.mean() if len(values) else np.nan
                row[metric+'_minimum']=values.min() if len(values) else np.nan
                row[metric+'_maximum']=values.max() if len(values) else np.nan
        else:
            row['interpretation']='Descriptive full-data score; not held-out'
            for metric in METRICS:
                values=pd.to_numeric(group.loc[group.evaluation_available,metric],errors='coerce').dropna()
                row[metric+'_mean']=values.mean() if len(values) else np.nan
                row[metric+'_minimum']=row[metric+'_maximum']=np.nan
        records.append(row)
    return pd.DataFrame(records)


def paired_comparisons(repetitions):
    records=[]
    for keys,group in repetitions.groupby(SCOPE,dropna=False,sort=False):
        shared=group[group.method=='shared_ld']
        need(len(shared)>0,'Planned scope has no shared-shape comparator')
        for method in group.method.drop_duplicates():
            if method=='shared_ld':continue
            other=group[group.method==method]
            pairs=shared.merge(other,on='repeat',suffixes=('_shared','_comparator'),validate='one_to_one')
            need(len(pairs)==len(shared)==len(other),'Methods have different planned repetitions')
            for metric,(kind,weighting) in METRICS.items():
                applicable=kind!='joint' or not bool(other.conditional_only.iloc[0])
                valid=pairs.complete_shared & pairs.complete_comparator if applicable else np.zeros(len(pairs),bool)
                differences=pairs.loc[valid,metric+'_shared']-pairs.loc[valid,metric+'_comparator']
                strict=valid & pairs.strict_score_eligible_shared & pairs.strict_score_eligible_comparator
                strict_differences=pairs.loc[strict,metric+'_shared']-pairs.loc[strict,metric+'_comparator']
                row={**dict(zip(SCOPE,keys)),'comparator':method,'metric':metric,'density':kind,'weighting':weighting,
                    'applicable':applicable,'planned_repetition_pairs':len(pairs),'complete_paired_repetitions':int(valid.sum()),
                    'unavailable_repetition_pairs':len(pairs)-int(valid.sum()),
                    'pairs_with_any_boundary':int((valid & (pairs.any_boundary_shared | pairs.any_boundary_comparator)).sum()),
                    'strict_score_eligible_pairs':int(strict.sum()),
                    'mean_shared_minus_comparator':differences.mean() if len(differences) else np.nan,
                    'minimum_difference':differences.min() if len(differences) else np.nan,
                    'maximum_difference':differences.max() if len(differences) else np.nan,
                    'strict_subset_mean_difference':strict_differences.mean() if len(strict_differences) else np.nan,
                    'availability_note':('Conditional-only reference has no joint density' if not applicable else
                        'No complete paired repetition' if not len(differences) else 'All finite complete pairs; boundaries retained'),
                    'uncertainty':'Descriptive range across overlapping repetitions; no independent-fold MCSE'}
                records.append(row)
    return pd.DataFrame(records)


def sensitivity_changes(frame,repetitions):
    records=[];primary=repetitions[(repetitions.comparison=='primary') & (repetitions.repeat==0)]
    for _,sensitivity in repetitions[repetitions.comparison!='primary'].iterrows():
        need(sensitivity['repeat']==0,'Unexpected sensitivity repetition')
        old=primary[(primary.dataset_id==sensitivity.dataset_id)&(primary.method==sensitivity.method)]
        need(len(old)==1,'Missing primary repeat-zero method for sensitivity')
        old=old.iloc[0]
        current_rows=frame[(frame.dataset_id==sensitivity.dataset_id)&(frame.comparison==sensitivity.comparison)&
                           (frame.method==sensitivity.method)&(frame.kind=='cv')]
        primary_rows=frame[(frame.dataset_id==sensitivity.dataset_id)&(frame.comparison=='primary')&
                           (frame.method==sensitivity.method)&(frame.kind=='cv')&(frame.repeat==0)]
        folds=current_rows.merge(primary_rows,on='fold',suffixes=('_sensitivity','_primary'),validate='one_to_one')
        finite_folds=folds.evaluation_available_sensitivity & folds.evaluation_available_primary
        for metric,(kind,weighting) in METRICS.items():
            complete=bool(sensitivity.complete and old.complete)
            records.append({'dataset_id':sensitivity.dataset_id,'comparison':sensitivity.comparison,
                'method':sensitivity.method,'bounds':sensitivity.bounds,'primary_k':old.k,'sensitivity_k':sensitivity.k,
                'metric':metric,'density':kind,'weighting':weighting,'paired_folds_available':int(finite_folds.sum()),
                'planned_paired_folds':len(folds),'complete_partition_pair':complete,
                'primary_score':old[metric] if complete else np.nan,
                'sensitivity_score':sensitivity[metric] if complete else np.nan,
                'sensitivity_minus_primary':sensitivity[metric]-old[metric] if complete else np.nan,
                'primary_strict_score_eligible':bool(old.strict_score_eligible),
                'sensitivity_strict_score_eligible':bool(sensitivity.strict_score_eligible),
                'primary_any_boundary':bool(old.any_boundary),'sensitivity_any_boundary':bool(sensitivity.any_boundary),
                'interpretation':'One matched repeat-zero partition; descriptive sensitivity, not model selection'})
    return pd.DataFrame(records)


def check_original_aggregation(repetitions,paired,source_repetitions,source_paired):
    """Verify numerical equivalence wherever the frozen tables contain a row."""
    expected=repetitions[repetitions.complete]
    if not len(source_repetitions):
        need(not len(expected),'Frozen complete-repetition table omits available results')
    else:
        need(not source_repetitions.duplicated(REPEAT_KEY).any(),'Duplicate frozen repetition row')
        joined=expected.merge(source_repetitions,on=REPEAT_KEY,how='outer',indicator=True,suffixes=('_new','_old'))
        need(joined['_merge'].eq('both').all(),'Frozen repetition identities differ from raw fit records')
        for metric in METRICS:
            need(np.allclose(joined[metric+'_new'],joined[metric+'_old'],rtol=1e-10,atol=1e-10,equal_nan=True),
                 'Recomputed repetition scores differ: '+metric)
    if len(source_paired):
        key=SCOPE+['comparator','metric']
        need(not source_paired.duplicated(key).any(),'Duplicate frozen pair row')
        joined=paired.merge(source_paired,on=key,how='right',indicator=True,suffixes=('_new','_old'))
        need(joined['_merge'].eq('both').all(),'Frozen pair table contains an undeclared comparison')
        for field in ['complete_paired_repetitions','mean_shared_minus_comparator','minimum_difference','maximum_difference']:
            need(np.allclose(joined[field+'_new'],joined[field+'_old'],rtol=1e-10,atol=1e-10,equal_nan=True),
                 'Recomputed paired summary differs: '+field)
    positive=paired[paired.complete_paired_repetitions>0]
    keys=SCOPE+['comparator','metric']
    if len(positive):
        need(len(source_paired)>0,'Frozen pair table is absent despite available paired scores')
        need(set(map(tuple,positive[keys].to_numpy())) <= set(map(tuple,source_paired[keys].to_numpy())),
             'Frozen pair table omits an available paired comparison')


def tables(manifest,mixtures,conditional,source_repetitions=None,source_paired=None):
    frame=align_candidates(manifest,mixtures,conditional)
    repetitions=complete_repetitions(frame,manifest['config']['folds'])
    paired=paired_comparisons(repetitions)
    if source_repetitions is not None and source_paired is not None:
        check_original_aggregation(repetitions,paired,source_repetitions,source_paired)
    return {'all_planned_fit_rows':frame,'candidate_summary':candidate_summary(frame,repetitions),
            'complete_repetitions':repetitions,'all_planned_paired_comparisons':paired,
            'matched_sensitivity_changes':sensitivity_changes(frame,repetitions)}


def same_value(actual,expected):
    if expected is None:return actual is None or pd.isna(actual)
    if isinstance(expected,bool):return optional_flag(actual)==expected
    if isinstance(expected,(int,float)):
        return finite(actual) and np.isclose(actual,expected,rtol=1e-10,atol=1e-10)
    return actual==expected


def verify_completed_inputs(reader,frame,dependencies=None):
    """Bind CSV scores/counts to canonical completed JSON rows, without fitting."""
    manifest=reader.manifest;dependency_records=[]
    for task in manifest['tasks']:
        if task['phase']!='prediction':continue
        split_path=reader.run/task['split_file'];reader.bytes(split_path,task['split_sha256'])
        split=reader.read(split_path)
        need(split['task_index']==task['task_index'],'Prediction split task identity differs')
        directory=reader.run/'tasks'/f"task_{task['task_index']:06d}"
        completion_path=directory/'completion.json'
        group=frame[frame.task_index==task['task_index']]
        if not completion_path.exists():
            need(not group.recorded.any(),'Recorded table row has no completed task')
            continue
        need(group.recorded.all(),'Aggregation predates a now-completed task; regenerate the frozen aggregate')
        completed=reader.read(completion_path);binding=reader.read(directory/'input_binding.json')
        need(completed.get('status')=='complete' and completed['task']==task and
             completed['manifest_sha256']==reader.manifest_sha256 and
             completed['input_binding_sha256']==semantic_hash(binding),'Completed task identity differs')
        need(completed['runtime_versions']==manifest['runtime_versions'],'Completed worker runtime differs')
        need(binding['task']==task and binding['split_sha256']==task['split_sha256'] and
             binding['positive_csv_sha256']==manifest['datasets'][task['dataset_id']]['raw_data_sha256'],
             'Completed task is bound to different data/split')
        reader.bytes(directory/'training.json',binding['training_checkpoint_sha256'])
        need([r['method'] for r in completed['rows']]==task['methods'] and
             [r['method'] for r in completed['conditional_rows']]==task['conditional_methods'],
             'Completed comparison scope differs')
        expected_dependencies=['gaussian_diag'] if 'gaussian_shared_correlation' in task['methods'] and 'gaussian_diag' not in task['methods'] else []
        need([r['method'] for r in completed['dependency_rows']]==expected_dependencies,'Completed dependency scope differs')
        dependency_records.extend(completed['dependency_rows'])
        y,groups,ids,metadata=reader.grid(task['dataset_id'])
        expected_n=len(split['test']);expected_groups=len(np.unique(groups[split['test']]))
        for original in completed['rows']+completed['conditional_rows']:
            matched=group[group.method==original['method']]
            need(len(matched)==1,'Missing/duplicate planned comparison row')
            row=matched.iloc[0]
            for field in list(METRICS)+COSTS+COUNTERS+['failed','fit_available','evaluation_available',
                   'score_converged','projected_gradient','boundary_hit','n_train','n_test','n_test_groups','parameter_count']:
                need(same_value(row.get(field),original.get(field)),'CSV/completed value mismatch: '+field)
            need(original['n_test']==expected_n and original['n_test_groups']==expected_groups,
                 'Evaluation weights differ from frozen test rows/groups')
        if task['comparison']!='primary':
            primary=next(t for t in manifest['tasks'] if t['phase']=='prediction' and t['dataset_id']==task['dataset_id']
                and t['comparison']=='primary' and t['kind']=='cv' and t['repeat']==0 and t['fold']==task['fold'])
            reader.bytes(reader.run/primary['split_file'],primary['split_sha256'])
            reference=reader.read(reader.run/primary['split_file'])
            need(split['train']==reference['train'] and split['test']==reference['test'],
                 'Sensitivity does not use the same primary repetition-zero split')
    if dependencies is not None:
        need(len(dependencies)==len(dependency_records),'Dependency table has missing or extra records')
        if len(dependencies):
            need(not dependencies.duplicated(['task_index','method']).any(),'Duplicate dependency record')
            for original in dependency_records:
                matched=dependencies[(dependencies.task_index==original['task_index'])&(dependencies.method==original['method'])]
                need(len(matched)==1,'Dependency task identity differs')
                for field in COSTS+COUNTERS+['failed','fit_available','evaluation_available']:
                    need(same_value(matched.iloc[0].get(field),original.get(field)),'Dependency cost/availability differs: '+field)


def write_assessment(run,output,retrieval_complete=False,expected_manifest_sha256=None):
    if not retrieval_complete:raise ValueError('Complete retrieval must be confirmed before reading prediction outputs')
    reader=SavedPanel(run);output=Path(output).resolve()
    if output==reader.run or reader.run in output.parents:raise ValueError('Assessment output must be outside the frozen run')
    if output.exists():raise FileExistsError('Preserve earlier assessments; use a new output directory')
    if expected_manifest_sha256 is not None:
        need(reader.manifest_sha256==expected_manifest_sha256,'Assessment input differs from the submitted run')
    status=reader.read(reader.run/'aggregation_prediction.json')
    need(status['manifest_sha256']==reader.manifest_sha256 and status['phase']=='prediction','Aggregation belongs to another study/phase')
    filenames={'mixtures':'fit_records_prediction.csv','conditional':'conditional_records_prediction.csv',
               'repetitions':'cv_repetitions_prediction.csv','paired':'paired_cv_prediction.csv',
               'dependencies':'dependency_records_prediction.csv','summary':'summary_prediction.csv'}
    source={key:csv_table(reader.bytes(reader.run/name)) for key,name in filenames.items()}
    result=tables(reader.manifest,source['mixtures'],source['conditional'],source['repetitions'],source['paired'])
    aligned=result['all_planned_fit_rows'];verify_completed_inputs(reader,aligned,source['dependencies'])
    mixture=aligned[~aligned.conditional_only];conditional=aligned[aligned.conditional_only]
    for field,actual in [('planned_mixture_fits',len(mixture)),('recorded_mixture_fits',int(mixture.recorded.sum())),
        ('mixture_evaluations_available',int(mixture.evaluation_available.sum())),('planned_conditional_fits',len(conditional)),
        ('recorded_conditional_fits',int(conditional.recorded.sum())),('conditional_evaluations_available',int(conditional.evaluation_available.sum()))]:
        need(status[field]==actual,'Aggregation count differs from planned/aligned rows: '+field)
    output.mkdir(parents=True)
    for name,frame in result.items():frame.to_csv(output/(name+'.csv'),index=False)
    # Dependencies remain separate; summing model policy times double-counts them.
    source['dependencies'].to_csv(output/'dependency_fit_records.csv',index=False)
    original_missing=result['all_planned_paired_comparisons']
    key=SCOPE+['comparator','metric']
    if len(source['paired']):
        known=set(map(tuple,source['paired'][key].to_numpy()))
        original_missing=original_missing[[tuple(row) not in known for row in original_missing[key].to_numpy()]]
    original_missing.to_csv(output/'comparisons_absent_from_frozen_paired_table.csv',index=False)
    source_path=Path(__file__).resolve();snapshot=output/'executed_sources';snapshot.mkdir()
    analysis_sources={}
    for name in ['assess_phase3_predictions.py','phase3_stability.py',
                 'assess_conditional_prediction_stability.py','conditional_log_linear.py']:
        source=source_path.parent/name;raw=source.read_bytes()
        (snapshot/name).write_bytes(raw);analysis_sources[name]=hashlib.sha256(raw).hexdigest()
    metadata={'created_utc':datetime.now(timezone.utc).isoformat(),'input_run':str(reader.run),
        'manifest_sha256':reader.manifest_sha256,'source_sha256':hashlib.sha256(source_path.read_bytes()).hexdigest(),
        'analysis_sources_sha256':analysis_sources,
        'input_sha256':reader.hashes,'runtime':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__},
        'retrieval_complete_confirmed':True,'fitting_performed':False,'model_selection_performed':False,
        'planned_mixture_fits':len(mixture),'planned_conditional_fits':len(conditional),
        'planned_paired_rows':len(result['all_planned_paired_comparisons']),
        'paired_rows_absent_from_original_table':len(original_missing),
        'definitions':{'main_scores':'All finite complete repetition pairs, retaining active bounds and nonstationarity',
            'strict_subset':'Both procedures have true raw-score flags in every fold; untested package/EM flags are not made true',
            'range':'Descriptive range across overlapping repetitions, not an independent-repetition confidence interval',
            'sensitivity':'Sensitivity minus the same method at primary k/bounds on identical repeat-zero splits',
            'cost':'Per-procedure own-stage and dependency costs; summing policy costs across methods double-counts dependencies',
            'missing_row_caveat':'Frozen paired table may omit comparators with no complete repetition; this report begins with all planned comparisons'}}
    (output/'provenance.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
    return metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--retrieval-complete',action='store_true');parser.add_argument('--expected-manifest-sha256')
    args=parser.parse_args()
    metadata=write_assessment(args.run,args.output,args.retrieval_complete,args.expected_manifest_sha256)
    print(json.dumps({key:metadata[key] for key in ['manifest_sha256','planned_mixture_fits','planned_conditional_fits',
        'planned_paired_rows','paired_rows_absent_from_original_table','fitting_performed']}),flush=True)


if __name__=='__main__':main()
