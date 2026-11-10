"""Frozen, resumable training-only refinements of existing application fits.

Freeze with --primary-run, --gaussian-run and --transport-run. The task map puts
all186 prediction/wide tasks before all597 bootstrap tasks. No fit is selected
by its test score; the saved selected parameter checkpoint precedes evaluation.
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
import run_confirmatory_experiments as base

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/phase2_application_refinement.json'
GC=gaussian.METHOD


def source_files():
    names=['scripts/phase2_application_refinement.py','scripts/application_optimization_refinement.py',
           'scripts/phase2_applications.py','scripts/phase2_gaussian_control.py',
           'scripts/gaussian_shared_correlation.py','scripts/run_confirmatory_experiments.py',
           'scripts/shared_shape_ld.py','scripts/regularized_skew_normal.py',
           'configs/experiments.json','requirements-lock.txt']
    return [ROOT/name for name in names]


def task_map(spec,gc_spec,transport_spec,config):
    gc={task['primary_task_index']:('gaussian',task['task_index']) for task in gc_spec['tasks']}
    gc.update({task['primary_task_index']:('transport',task['task_index']) for task in transport_spec['tasks']})
    selected=[]
    for index,task in enumerate(spec['tasks']):
        if task['task_index']!=index:
            raise ValueError('Noncanonical primary application task map')
        if task['variant'] not in config['primary_variants']:
            continue
        if task['bounds']=='primary' and task['kind'] in ['cv','full','leave_area_out','bootstrap']:
            if index not in gc:
                raise ValueError('Missing declared Gaussian source mapping')
            source,source_index=gc[index]
            methods=config['methods']
        elif task['bounds']=='wide' and task['kind']=='cv':
            source,source_index=None,None;methods=['shared_ld','full_ld']
        else:
            continue
        selected.append({'primary_task_index':index,'variant':task['variant'],'kind':task['kind'],
            'repeat':task['repeat'],'fold':task['fold'],'bounds':task['bounds'],
            'phase':'bootstrap' if task['kind']=='bootstrap' else 'prediction',
            'gaussian_source':source,'gaussian_task_index':source_index,'methods':methods,
            **({'area':task['area']} if 'area' in task else {})})
    selected.sort(key=lambda t:(t['phase']=='bootstrap',t['primary_task_index']))
    for index,task in enumerate(selected): task['task_index']=index
    for phase in ['prediction','bootstrap']:
        group=[t for t in selected if t['phase']==phase]
        if len(group)!=config['expected_tasks'][phase] or sum(len(t['methods']) for t in group)!=config['expected_fits'][phase]:
            raise ValueError('Declared refinement scope count mismatch: '+phase)
    return selected


def freeze(output,primary,gaussian_run,transport,config_path=CONFIG):
    output=Path(output);config=json.loads(Path(config_path).read_text())
    if config['policy']!=refine.DEFAULT_POLICY or config['threads']!=1:
        raise ValueError('Declared policy differs from the tested refiner')
    roots={'primary':Path(primary).resolve(),'gaussian':Path(gaussian_run).resolve(),'transport':Path(transport).resolve()}
    metadata={name:json.loads((path/('specification.json' if name=='primary' else 'manifest.json')).read_text())
              for name,path in roots.items()}
    if metadata['primary']['runtime_versions']!=control.versions():
        raise RuntimeError('Freeze in the same runtime as the primary study')
    original_hash=control.digest(roots['primary']/'specification.json')
    for name in ['gaussian','transport']:
        if metadata[name]['primary_metadata_sha256']!=original_hash or metadata[name]['versions']!=control.versions():
            raise RuntimeError('Gaussian input is bound to another primary run or runtime')
    plan=task_map(metadata['primary'],metadata['gaussian'],metadata['transport'],config)
    output.mkdir(parents=True,exist_ok=False)
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'config':config,
        'config_sha256':control.semantic_hash(config),'tasks':plan,'runtime_versions':control.versions(),
        'sources_sha256':{str(p.relative_to(ROOT)):control.digest(p) for p in source_files()},
        'input_roots':{k:str(v) for k,v in roots.items()},
        'metadata_sha256':{name:control.digest(path/('specification.json' if name=='primary' else 'manifest.json'))
                           for name,path in roots.items()},'data_sha256':metadata['primary']['data_hashes']}
    control.atomic_json(output/'manifest.json',manifest)
    for name,path in roots.items():
        (output/(name+'_metadata.json')).write_bytes((path/('specification.json' if name=='primary' else 'manifest.json')).read_bytes())
    for source in source_files():
        target=output/'source_snapshot'/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(source.read_bytes())
    control.atomic_json(output/'source_snapshot/configs/phase2_application_refinement.json',config)
    for relative,expected in manifest['data_sha256'].items():
        source=roots['primary']/'executed_sources'/relative
        if control.digest(source)!=expected:raise RuntimeError('Frozen original data changed')
        target=output/'source_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source.read_bytes())
    return manifest


def verify_run(output,root_overrides=None):
    output=Path(output);manifest=json.loads((output/'manifest.json').read_text())
    if manifest['runtime_versions']!=control.versions():raise RuntimeError('Runtime changed after freezing')
    if (control.semantic_hash(manifest['config'])!=manifest['config_sha256'] or
        control.semantic_hash(json.loads((output/'source_snapshot/configs/phase2_application_refinement.json').read_text()))!=manifest['config_sha256']):
        raise RuntimeError('Frozen refinement configuration changed')
    for relative,expected in manifest['sources_sha256'].items():
        if control.digest(ROOT/relative)!=expected or control.digest(output/'source_snapshot'/relative)!=expected:
            raise RuntimeError('Frozen refinement source changed: '+relative)
    roots={name:Path(path) for name,path in manifest['input_roots'].items()}
    roots.update({name:Path(path) for name,path in (root_overrides or {}).items()})
    metadata={}
    for name,root in roots.items():
        path=root/('specification.json' if name=='primary' else 'manifest.json')
        if (control.digest(path)!=manifest['metadata_sha256'][name] or
            control.digest(output/(name+'_metadata.json'))!=manifest['metadata_sha256'][name]):
            raise RuntimeError('Upstream frozen metadata changed: '+name)
        metadata[name]=json.loads(path.read_text())
    if task_map(metadata['primary'],metadata['gaussian'],metadata['transport'],manifest['config'])!=manifest['tasks']:
        raise RuntimeError('Frozen refinement task map changed')
    for relative,expected in manifest['data_sha256'].items():
        if control.digest(output/'source_snapshot'/relative)!=expected:
            raise RuntimeError('Archived scientific data changed')
    return manifest,roots,metadata


def load_inputs(roots,metadata,task):
    spec=metadata['primary'];upstream=spec['tasks'][task['primary_task_index']]
    directory=roots['primary']/'tasks'/f"{task['primary_task_index']:05d}"
    complete=directory/'completion.json'
    if not complete.exists():raise control.DependencyPending('Primary task incomplete')
    completion=json.loads(complete.read_text());split_path=directory/'split.json';split=json.loads(split_path.read_text())
    if (completion['task']!=upstream or completion['specification_sha256']!=control.digest(roots['primary']/'specification.json')
            or any(split.get(k)!=v for k,v in upstream.items())):
        raise control.IntegrityError('Primary task/split identity mismatch')
    variant=next(v for v in spec['config']['variants'] if v['id']==task['variant'])
    y,labels,jacobian,names,data_path=control.application_data(roots['primary'],spec,variant)
    train=np.asarray(upstream['train']);test=np.asarray(upstream['test']);center=np.asarray(split['center']);scale=np.asarray(split['scale'])
    np.testing.assert_allclose(center,y[train].mean(0),rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(scale,y[train].std(0),rtol=1e-12,atol=1e-12)
    if np.any(scale<=0):raise ValueError('Invalid saved training scale')
    used=[('primary',complete),('primary',split_path),('primary',data_path)]
    saved={};original_rows={row['method']:row for row in completion['records']}
    for method in ['shared_ld','full_ld']:
        path=directory/(method+'.json');value=json.loads(path.read_text());row=value['record']
        if (value['specification_sha256']!=completion['specification_sha256'] or row['task_index']!=task['primary_task_index']
                or row['method']!=method or row['variant']!=task['variant'] or row['bounds']!=task['bounds']):
            raise control.IntegrityError('Original LD checkpoint identity mismatch')
        if row!=original_rows.get(method):
            raise control.IntegrityError('Original LD checkpoint record differs from completion record')
        saved[method]=value;used.append(('primary',path))
    if GC in task['methods']:
        source=task['gaussian_source'];index=task['gaussian_task_index']
        gd=roots[source]/'tasks'/f'task_{index:06d}'
        path=gd/'result.json'
        if not path.exists():raise control.DependencyPending('Gaussian source task incomplete')
        result=json.loads(path.read_text());binding_path=gd/'input_binding.json';binding=json.loads(binding_path.read_text())
        expected=metadata[source]['tasks'][index]
        if (result.get('status')!='complete' or result['task']!=expected or expected['primary_task_index']!=task['primary_task_index']
                or result['manifest_sha256']!=control.digest(roots[source]/'manifest.json')
                or result['input_binding_sha256']!=control.semantic_hash(binding)):
            raise control.IntegrityError('Original Gaussian task identity mismatch')
        fit_path=gd/'control_fit.json';value=json.loads(fit_path.read_text())
        identity={'task_index':index,'primary_task_index':task['primary_task_index'],
                  'manifest_sha256':result['manifest_sha256'],'input_binding_sha256':result['input_binding_sha256']}
        control.validate_stage(value,identity,'Original Gaussian fit')
        if (value.get('parameters') is not None and result.get('parameters') is not None
                and control.semantic_hash(value['parameters'])!=control.semantic_hash(result['parameters'])):
            raise control.IntegrityError('Gaussian fitted checkpoint and result parameters differ')
        for name,path0 in used:
            if name=='primary':
                relative=str(path0.relative_to(roots['primary']))
                if relative in binding['input_sha256'] and control.digest(path0)!=binding['input_sha256'][relative]:
                    raise control.IntegrityError('Gaussian and LD sources bind different training inputs')
        saved[GC]=value;original_rows[GC]={**result['row'],
            'mean_log_density':result['row'].get('test_log_density'),
            'mean_conditional_log_density':result['row'].get('conditional_log_density')}
        used.extend([(source,path),(source,binding_path),(source,fit_path)])
    input_binding={'task':task,'files':[{'source':name,'relative_path':str(path.relative_to(roots[name])),
                        'sha256':control.digest(path)} for name,path in used]}
    settings={**base.DEFAULT_SETTINGS,**spec['config']['fitting']}
    if task['bounds']=='wide':settings.update(spec['config']['wide_bounds'])
    return {'train':(y[train]-center)/scale,'test':(y[test]-center)/scale,'center':center,'scale':scale,
            'test_raw_jacobian':jacobian[test],'train_raw_jacobian':jacobian[train],
            'train_indices':train,'test_indices':test,'labels':labels[test],'names':names,
            'k':variant['k'],'variant':variant,'settings':settings,'saved':saved,
            'original_rows':original_rows,'input_binding':input_binding,'original_data_sd':y.std(0)}


def restore(value):
    params=value['parameters'];record=value['record'];starts=value['starts']
    if params['method']==GC:return gaussian.GaussianSharedCorrelationFit(params,record,starts,value['selected_start'],record['fit_seconds'])
    return base.FitResult(params['method'],params['k'],params['d'],params,record,starts,value['selected_start'])


def evaluate(model,case):
    ell,tau=model.evaluate(case['test']);target=case['names'].index(case['variant']['conditional_target'])
    keep=[j for j in range(case['test'].shape[1]) if j!=target]
    marginal=(control.gaussian_density(model.params,case['test'],keep)[0] if model.method==GC
              else apps.marginal_log_density(model.params,case['test'],keep))
    conditional=ell-marginal-np.log(case['scale'][target])-case['test_raw_jacobian'][:,target]
    ell=ell-np.log(case['scale']).sum()-case['test_raw_jacobian'].sum(1)
    if not np.isfinite(ell).all() or not np.isfinite(conditional).all() or not np.isfinite(tau).all():
        raise FloatingPointError('Nonfinite original-coordinate predictions')
    return ell,conditional,tau


def run_task(output,index,root_overrides=None,fitter=refine.refine_saved):
    output=Path(output);manifest,roots,metadata=verify_run(output,root_overrides)
    if type(index) is not int or not 0<=index<len(manifest['tasks']):raise ValueError('Invalid refinement task index')
    task=manifest['tasks'][index];directory=output/'tasks'/f'task_{index:06d}'
    with control.task_lock(directory),threadpool_limits(1):
        try:case=load_inputs(roots,metadata,task)
        except control.DependencyPending as exc:
            value={'status':'dependency_pending','task':task,'reason':str(exc)}
            control.atomic_json(directory/'status.json',value);return value
        binding=case['input_binding'];binding_path=directory/'input_binding.json'
        if binding_path.exists() and json.loads(binding_path.read_text())!=binding:
            raise control.IntegrityError('Refinement input bytes changed since previous attempt')
        control.atomic_json(binding_path,binding)
        identity={'manifest_sha256':control.digest(output/'manifest.json'),'task':task,
                  'input_binding_sha256':control.semantic_hash(binding)}
        completion=directory/'completion.json'
        if completion.exists():
            value=json.loads(completion.read_text());control.validate_stage(value,identity,'Refinement completion');return value
        fitted={};rows=[]
        for method in task['methods']:
            checkpoint=directory/(method+'_fit.json');result_path=directory/(method+'.json')
            stage_identity={**identity,'method':method}
            if result_path.exists():
                value=json.loads(result_path.read_text());control.validate_stage(value,stage_identity,'Refinement result')
                if value.get('parameters') is not None:fitted[method]=restore(value)
                rows.append(value['row']);continue
            row={**task,'method':method,'failed':True,'fit_available':False,'evaluation_available':False,
                 'mean_log_density':None,'mean_conditional_log_density':None,'ARI':None,
                 'n_train':len(case['train']),'n_test':len(case['test']),'k':case['k'],'d':case['train'].shape[1],
                 'score_converged':None,'projected_gradient':None,'boundary_hit':None,'fit_seconds':None,
                 'evaluation_role':'descriptive_original_sample' if task['kind'] in ['full','bootstrap'] else
                                   'held_out_collection_area' if task['kind']=='leave_area_out' else 'held_out_CV'}
            started=time.perf_counter();model=None;selected=None;profiles=None
            try:
                if checkpoint.exists():
                    selected=json.loads(checkpoint.read_text());control.validate_stage(selected,stage_identity,'Refinement fit')
                else:
                    try:
                        model=fitter(case['saved'][method],case['train'],case['k'],method,case['settings'],manifest['config']['policy'],
                            refined_shared=fitted.get('shared_ld') if method=='full_ld' else None)
                        selected={'failed':False,'parameters':model.params,'record':model.record,'starts':model.starts,
                                  'selected_start':model.selected_start}
                    except Exception as exc:
                        selected={'failed':True,'parameters':None,'record':{},'starts':getattr(exc,'starts',[]),
                                  'selected_start':None,'exception':repr(exc),'traceback':traceback.format_exc()}
                    selected.update(stage_identity);control.atomic_json(checkpoint,selected);selected=base.serializable(selected)
                row.update(starts_attempted=len(selected['starts']),
                    starts_failed=sum(bool(s.get('failed')) for s in selected['starts']),
                    retained_starts_with_exceptions=sum(not s.get('failed') and bool(s.get('exceptions')) for s in selected['starts']),
                    optimizer_exception_count=sum(len(s.get('exceptions',[])) for s in selected['starts']),
                    timeout_exception_count=sum(sum(e.get('exception_type')=='TimeoutError' for e in s.get('exceptions',[]))
                                                +int(s.get('exception_type')=='TimeoutError') for s in selected['starts']))
                if selected['failed']:raise RuntimeError(selected['exception'])
                model=restore(selected);fitted[method]=model;row['fit_available']=True
                for field in ['score_converged','projected_gradient','boundary_hit','boundary_hits','fit_seconds',
                              'dependency_fit_seconds','policy_fit_seconds','parameter_count','starts_attempted','starts_failed','min_weight']:
                    row[field]=model.record.get(field)
                for field in ['retained_starts_with_exceptions','optimizer_exception_count','timeout_exception_count','omitted_saved_endpoints']:
                    row[field]=model.record.get(field,0)
                old=case['original_rows'][method]
                row['original_projected_gradient']=old.get('projected_gradient')
                row['original_score_converged']=old.get('score_converged')
                row['original_boundary_hit']=old.get('boundary_hit')
                row['original_fit_seconds']=old.get('fit_seconds')
                row['original_policy_fit_seconds']=old.get('policy_fit_seconds',old.get('fit_seconds'))
                row['total_policy_fit_seconds_with_original']=(row['policy_fit_seconds']+row['original_policy_fit_seconds']
                    if row['policy_fit_seconds'] is not None and row['original_policy_fit_seconds'] is not None else None)
                original_params=case['saved'][method].get('parameters')
                original_ll=(-refine.objective(np.asarray(original_params['theta']),case['train'],case['k'],method)[0]
                             if original_params is not None else None)
                row['training_mean_log_density_gain']=(model.record['mean_train_log_density']-original_ll
                                                        if original_ll is not None else None)
                row['train_mean_log_density']=model.record['mean_train_log_density']-np.log(case['scale']).sum()-case['train_raw_jacobian'].sum(1).mean()
                if row['training_mean_log_density_gain'] is not None and row['training_mean_log_density_gain'] < -1e-10:
                    raise ArithmeticError('Refinement lost original training point')
                if method=='full_ld' and 'shared_ld' in fitted:
                    row['full_minus_shared_training_mean_log_density']=model.record['mean_train_log_density']-fitted['shared_ld'].record['mean_train_log_density']
                    if row['full_minus_shared_training_mean_log_density'] < -1e-10:raise ArithmeticError('Refined nested likelihood failed')
                ell,conditional,tau=evaluate(model,case)
                profiles=apps.component_profiles(model,case['center'],case['scale'],case['names'],case['variant'])
                profiles['original_data_sd']=case['original_data_sd']
                if method==GC:
                    profiles.update(correlation=model.params['correlation'],
                                    marginal_sd_in_analysis_coordinates=np.asarray(model.params['std'])*case['scale'])
                np.savez_compressed(directory/(method+'_predictions.npz'),indices=case['test_indices'],log_density=ell,
                    conditional_log_density=conditional,responsibilities=tau)
                row.update(failed=False,evaluation_available=True,mean_log_density=float(ell.mean()),
                    mean_conditional_log_density=float(conditional.mean()),ARI=float(adjusted_rand_score(case['labels'],tau.argmax(1))))
            except control.IntegrityError:raise
            except Exception as exc:
                row.update(failed=True,evaluation_available=False,exception=repr(exc),traceback=traceback.format_exc())
            row['total_seconds_this_invocation']=time.perf_counter()-started
            value={**stage_identity,'row':row,'parameters':model.params if model is not None else None,
                   'record':model.record if model is not None else {},'starts':selected.get('starts',[]) if selected else [],
                   'selected_start':model.selected_start if model is not None else None,'profiles':profiles,
                   'training_transform':{'center':case['center'],'scale':case['scale']},
                   'original_record':case['original_rows'].get(method)}
            control.atomic_json(result_path,value);rows.append(row)
        value={**identity,'status':'complete','rows':rows,'original_baseline_records':case['original_rows'],
               'completed_utc':datetime.now(timezone.utc).isoformat(),'runtime_versions':control.versions()}
        control.atomic_json(completion,value);return value


def aggregate(output,phase='all',root_overrides=None):
    output=Path(output);manifest,roots,metadata=verify_run(output,root_overrides)
    plan=[t for t in manifest['tasks'] if phase=='all' or t['phase']==phase]
    rows=[];comparison=[];original_refined=[];changes=[]
    for task in plan:
        directory=output/'tasks'/f"task_{task['task_index']:06d}";path=directory/'completion.json'
        if not path.exists():
            for method in task['methods']:
                missing={**task,'method':method,'recorded':False,'failed':True,'evaluation_available':False,
                         'score_converged':None,'boundary_hit':None,'mean_log_density':None,'mean_conditional_log_density':None}
                rows.append(missing);comparison.append({**missing,'version':'refined'})
            continue
        value=json.loads(path.read_text());binding=json.loads((directory/'input_binding.json').read_text())
        identity={'manifest_sha256':control.digest(output/'manifest.json'),'task':task,'input_binding_sha256':control.semantic_hash(binding)}
        control.validate_stage(value,identity,'Refinement aggregate completion')
        if value['status']!='complete' or [r['method'] for r in value['rows']]!=task['methods']:
            raise control.IntegrityError('Refinement method rows differ from plan')
        for entry in binding['files']:
            if control.digest(roots[entry['source']]/entry['relative_path'])!=entry['sha256']:
                raise control.IntegrityError('Refinement bound input changed')
        rows.extend([{**row,'recorded':True} for row in value['rows']])
        comparison.extend([{**task,**row,'version':'refined','recorded':True} for row in value['rows']])
        comparison.extend([{**row,**{k:task[k] for k in ['task_index','primary_task_index','variant','kind','bounds','repeat','fold']},
                            'version':'original_unchanged','recorded':True,'evaluation_available':not row['failed']}
                            for method,row in value['original_baseline_records'].items() if method not in task['methods']])
        for row in value['rows']:
            old=value['original_baseline_records'][row['method']]
            original_refined.append({**old,**{k:task[k] for k in ['task_index','primary_task_index','variant','kind','bounds','repeat','fold']},
                                     'version':'original_refined_method'})
            valid=bool(row['evaluation_available']) and not old['failed']
            changes.append({**task,'method':row['method'],'available_pair':valid,
                'original_failed':old['failed'],'refinement_failed':row['failed'],
                'original_projected_gradient':old.get('projected_gradient'),'refined_projected_gradient':row.get('projected_gradient'),
                'original_score_converged':old.get('score_converged'),'refined_score_converged':row.get('score_converged'),
                'original_boundary_hit':old.get('boundary_hit'),'refined_boundary_hit':row.get('boundary_hit'),
                'training_mean_log_density_gain':row.get('training_mean_log_density_gain'),
                'joint_prediction_change':row['mean_log_density']-old['mean_log_density'] if valid else None,
                'conditional_prediction_change':row['mean_conditional_log_density']-old['mean_conditional_log_density'] if valid else None})
    frame=pd.DataFrame(rows);frame.to_csv(output/('fit_records_'+phase+'.csv'),index=False)
    compare=pd.DataFrame(comparison);compare.to_csv(output/('comparison_records_'+phase+'.csv'),index=False)
    pd.DataFrame(original_refined).to_csv(output/('original_refined_method_records_'+phase+'.csv'),index=False)
    pd.DataFrame(changes).to_csv(output/('optimization_changes_'+phase+'.csv'),index=False)
    summaries=[];repeats=[]
    if len(compare):
        for keys,g in compare.groupby(['variant','kind','bounds','method']):
            valid=g[g.evaluation_available]
            target_tasks=[t for t in plan if (t['variant'],t['kind'],t['bounds'])==keys[:3]]
            row={'variant':keys[0],'kind':keys[1],'bounds':keys[2],'method':keys[3],
                 'planned_tasks':len(target_tasks),'recorded':int(g.recorded.sum()),'available':len(valid),
                 'missing':len(target_tasks)-int(g.recorded.sum()),
                 'failed_recorded':int((g.failed & g.recorded).sum()),'score_checked':int(g.score_converged.notna().sum()),
                 'score_passed':int(g.score_converged.eq(True).sum()),'boundary_checked':int(g.boundary_hit.notna().sum()),
                 'boundary_hits':int(g.boundary_hit.eq(True).sum())}
            for field in ['starts_attempted','starts_failed','retained_starts_with_exceptions','optimizer_exception_count','timeout_exception_count','omitted_saved_endpoints']:
                row[field]=int(g[field].fillna(0).sum()) if field in g else 0
            if keys[1]=='cv':
                for repeat,folds in valid.groupby('repeat'):
                    if len(folds)!=metadata['primary']['config']['folds']:continue
                    repeats.append({'variant':keys[0],'bounds':keys[2],'method':keys[3],'repeat':repeat,
                        'joint':float(np.average(folds.mean_log_density,weights=folds.n_test)),
                        'conditional':float(np.average(folds.mean_conditional_log_density,weights=folds.n_test))})
            if keys[1]=='leave_area_out' and len(valid)==len(target_tasks):
                row.update(joint_weighted=float(np.average(valid.mean_log_density,weights=valid.n_test)),
                           conditional_weighted=float(np.average(valid.mean_conditional_log_density,weights=valid.n_test)))
            summaries.append(row)
    pd.DataFrame(summaries).to_csv(output/('summary_'+phase+'.csv'),index=False)
    repeat_frame=pd.DataFrame(repeats);repeat_frame.to_csv(output/('cv_repetitions_'+phase+'.csv'),index=False)
    paired=[]
    if len(repeat_frame):
        for (variant,bounds),group in repeat_frame.groupby(['variant','bounds']):
            common=group[group.method=='shared_ld']
            for method,other in group[group.method!='shared_ld'].groupby('method'):
                pairs=common.merge(other,on='repeat',suffixes=('_shared','_other'))
                for measure in ['joint','conditional']:
                    delta=pairs[measure+'_shared']-pairs[measure+'_other']
                    paired.append({'variant':variant,'bounds':bounds,'comparator':method,'measure':measure,
                        'complete_paired_repetitions':len(pairs),'shared_minus_other':delta.mean(),
                        'repeat_min':delta.min(),'repeat_max':delta.max(),
                        'uncertainty':'Descriptive repeated-CV range; overlapping repetitions are not independent'})
    pd.DataFrame(paired).to_csv(output/('paired_cv_'+phase+'.csv'),index=False)
    status={'phase':phase,'planned_tasks':len(plan),'planned_fits':len(frame),'recorded_fits':int(frame.recorded.sum()),
            'evaluation_available':int(frame.evaluation_available.sum()),'failed_recorded':int((frame.recorded & frame.failed).sum()),
            'missing_tasks':sorted(set(frame.loc[~frame.recorded,'task_index']))}
    control.atomic_json(output/('aggregation_'+phase+'.json'),status);return status


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--config',type=Path,default=CONFIG)
    parser.add_argument('--primary-run',type=Path);parser.add_argument('--gaussian-run',type=Path);parser.add_argument('--transport-run',type=Path)
    parser.add_argument('--phase',choices=['prediction','bootstrap','all'],default='all')
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze',action='store_true');group.add_argument('--task-index',type=int)
    group.add_argument('--task-range');group.add_argument('--aggregate',action='store_true')
    args=parser.parse_args()
    roots={k:v for k,v in [('primary',args.primary_run),('gaussian',args.gaussian_run),('transport',args.transport_run)] if v is not None}
    if args.freeze:
        if len(roots)!=3:parser.error('Freeze requires all three upstream runs')
        value=freeze(args.output,roots['primary'],roots['gaussian'],roots['transport'],args.config)
        print(json.dumps({'tasks':len(value['tasks']),'config_sha256':value['config_sha256']}));return
    if args.aggregate:print(json.dumps(aggregate(args.output,args.phase,roots)));return
    manifest,_,_=verify_run(args.output,roots)
    if args.task_range:
        start,stop=map(int,args.task_range.split(':'))
        if not 0<=start<stop<=len(manifest['tasks']):raise ValueError('Invalid task range')
        indices=range(start,stop)
    else:indices=[args.task_index]
    for index in indices:
        value=run_task(args.output,index,roots)
        print(json.dumps({'task_index':index,'status':value['status']}),flush=True)


if __name__=='__main__':main()
