"""Read-only held-out squared loss for the fixed conditional log-mean target.

No model fitting, target selection, exponentiation or density Jacobians. The
old and new panels share the same four prediction functions and weighting.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import io
import itertools
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import scipy

import assess_conditional_prediction_stability as old
import phase3_stability as panel

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'docs/public_analysis_rules.json'
PROTOCOL_SHA='1307bd7d13968a986c6bd6995a5baa6bd7e546858cb4f3d7bb87fddd04806440'
RECEIPT=ROOT/'docs/public_analysis_receipt.json'
METHODS=panel.METHODS
BASELINE='conditional_log_linear'
GC='gaussian_shared_correlation'
IntegrityError=old.IntegrityError
SOURCE_NAMES=['scripts/assess_conditional_mean_prediction.py','scripts/assess_conditional_prediction_stability.py',
              'scripts/phase3_stability.py','scripts/conditional_log_linear.py']


def check_identity(value,expected,description):
    if any(value.get(k)!=v for k,v in expected.items()):
        raise IntegrityError(description+' identity differs')


def checked_indices(task,n):
    train=np.asarray(task['train']);test=np.asarray(task['test'])
    if (train.dtype.kind not in 'iu' or test.dtype.kind not in 'iu' or train.ndim!=1 or test.ndim!=1
        or not len(train) or not len(test) or np.any(train<0) or np.any(test<0)
        or np.any(train>=n) or np.any(test>=n) or len(np.unique(train))!=len(train)
        or len(np.unique(test))!=len(test) or np.intersect1d(train,test).size
        or set(train)|set(test)!=set(range(n))):
        raise IntegrityError('Held-out task must partition the complete original row grid')
    return train,test


class OldInputs(old.SavedInputs):
    """CV and area-out extension of the verified old saved-parameter reader."""
    def __init__(self,primary,gaussian,transport,reference,refined=None):
        super().__init__(primary,gaussian,refined)
        self.transport=Path(transport).resolve();self.reference=Path(reference).resolve()
        self.tspec=self.json(self.transport/'manifest.json');self.transport_hash=self.hashes[str(self.transport/'manifest.json')]
        if self.tspec['primary_metadata_sha256']!=self.primary_hash:
            raise IntegrityError('Area-out Gaussian companion belongs to another primary study')
        self.transport_tasks={t['primary_task_index']:t for t in self.tspec['tasks']}
        if len(self.transport_tasks)!=len(self.tspec['tasks']):raise IntegrityError('Duplicate area-out companion mapping')
        if any(t['task_index']!=i for i,t in enumerate(self.tspec['tasks'])):raise IntegrityError('Noncanonical area-out mapping')
        for relative,sha in self.tspec.get('source_sha256',{}).items():self.bytes(self.transport/'source_snapshot'/relative,sha)
        if self.refined and self.rspec['metadata_sha256']['transport']!=self.transport_hash:
            raise IntegrityError('Refinement area-out source differs')
        self.bspec=self.json(self.reference/'manifest.json');self.reference_hash=self.hashes[str(self.reference/'manifest.json')]
        if self.bspec['primary_specification_sha256']!=self.primary_hash:raise IntegrityError('Conditional reference primary study differs')
        self.baseline_tasks={t['task_index']:t for t in self.bspec['tasks']}
        if len(self.baseline_tasks)!=len(self.bspec['tasks']):raise IntegrityError('Duplicate conditional-reference mapping')
        for relative,sha in self.bspec.get('source_sha256',{}).items():self.bytes(self.reference/'source_snapshot'/relative,sha)

    def grid_groups(self,variant):
        y,target,names=self.grid(variant)
        relative='data/'+('oliveoil.csv' if variant['dataset']=='oliveoil' else 'crabs.csv')
        frame=pd.read_csv(io.BytesIO(self.bytes(self.primary/'executed_sources'/relative,self.spec['data_hashes'][relative])))
        groups=(frame.region.astype(str).to_numpy() if variant['dataset']=='oliveoil'
                else (frame.sp.astype(str)+'/'+frame.sex.astype(str)).to_numpy())
        return y,groups,np.arange(1,len(y)+1).astype(str),target,names

    def parameters(self,task,method,y):
        center,scale,completed=self.transform(task,y)
        folder=self.primary/'tasks'/f"{task['task_index']:05d}"
        if method==BASELINE:
            if self.baseline_tasks.get(task['task_index'])!=task:raise IntegrityError('Conditional-reference task changed')
            directory=self.reference/'tasks'/f"task_{task['task_index']:06d}"
            value=self.json(directory/'result.json');selected=self.json(directory/'fit.json')
            identity={'task':task,'manifest_sha256':self.reference_hash,
                'split_sha256':old.digest(folder/'split.json'),'completion_sha256':old.digest(folder/'completion.json')}
            check_identity(value,identity,'Conditional-reference result');check_identity(selected,identity,'Conditional-reference fit')
            if value.get('status')!='complete':raise ValueError('Conditional reference is incomplete')
            if (selected.get('parameters')!=value.get('parameters') or selected.get('record')!=value.get('record')):
                raise IntegrityError('Conditional-reference selected parameters differ')
            transform={'center':center.tolist(),'scale':scale.tolist()}
            if value['training_transform']!=transform or selected['training_transform']!=transform:
                raise IntegrityError('Conditional-reference training transform changed')
            if value['row'].get('evaluation_available'):
                self.bytes(directory/'predictions.npz',value['predictions_sha256'])
            params,record=value.get('parameters'),value['row']
        elif self.refined:
            mapped=self.refined_tasks[task['task_index']];directory=self.refined/'tasks'/f"task_{mapped['task_index']:06d}"
            value=self.json(directory/(method+'.json'));selected=self.json(directory/(method+'_fit.json'))
            completion=self.json(directory/'completion.json');binding=self.json(directory/'input_binding.json')
            identity={'manifest_sha256':self.refined_hash,'task':mapped,'input_binding_sha256':old.semantic_hash(binding)}
            for item in (value,selected,completion):check_identity(item,identity,'Refined checkpoint')
            if completion.get('status')!='complete' or binding.get('task')!=mapped or value.get('method')!=method:
                raise IntegrityError('Refined task/method identity differs')
            roots={'primary':self.primary,'gaussian':self.gaussian,'transport':self.transport}
            for item in binding['files']:
                if item['source'] not in roots:raise IntegrityError('Unexpected bound refinement input')
                self.bytes(roots[item['source']]/item['relative_path'],item['sha256'])
            if (selected.get('parameters')!=value.get('parameters') or selected.get('record')!=value.get('record')):
                raise IntegrityError('Refined selected parameter checkpoint differs')
            if value['training_transform']!={'center':center.tolist(),'scale':scale.tolist()}:
                raise IntegrityError('Refinement changed original training transformation')
            record,params=value['row'],value.get('parameters')
            if [r for r in completion['rows'] if r['method']==method]!=[record]:raise IntegrityError('Refined completion row differs')
            if record['task_index']!=mapped['task_index']:raise IntegrityError('Refined row task differs')
        elif method==GC and task['kind']=='leave_area_out':
            mapped=self.transport_tasks[task['task_index']];directory=self.transport/'tasks'/f"task_{mapped['task_index']:06d}"
            value=self.json(directory/'result.json');binding=self.json(directory/'input_binding.json')
            check_identity(value,{'task':mapped,'manifest_sha256':self.transport_hash,
                'input_binding_sha256':old.semantic_hash(binding),'status':'complete'},'Area-out Gaussian')
            if binding['primary_task_index']!=task['task_index']:raise IntegrityError('Area-out Gaussian primary task differs')
            if f"tasks/{task['task_index']:05d}/split.json" not in binding['input_sha256']:
                raise IntegrityError('Area-out Gaussian binding omits training split')
            for relative,sha in binding['input_sha256'].items():self.bytes(self.primary/relative,sha)
            record,params=value['row'],value.get('parameters')
            if record['task_index']!=mapped['task_index']:raise IntegrityError('Area-out Gaussian row task differs')
        else:
            return super().parameters(task,method,y)
        if record['method']!=method or record['variant']!=task['variant']:raise IntegrityError('Saved prediction family/variant differs')
        if params is None:raise ValueError('No finite saved parameter point')
        if params['method']!=method or int(params['d'])!=y.shape[1]:raise IntegrityError('Saved parameter family/dimension differs')
        return params,center,scale,record

    def cases(self):
        for variant in self.variants():
            y,groups,ids,target,names=self.grid_groups(variant)
            tasks=[t for t in self.spec['tasks'] if t['variant']==variant['id'] and t['bounds']=='primary'
                   and t['kind'] in ('cv','leave_area_out')]
            validate_task_partitions(tasks,len(y),int(self.spec['config']['folds']),int(variant['repeats']))
            area_count=sum(t['kind']=='leave_area_out' for t in tasks)
            for task in tasks:
                _,test=checked_indices(task,len(y))
                if task['kind']=='leave_area_out' and (len(np.unique(groups[test]))!=1 or str(groups[test][0])!=task['area']):
                    raise IntegrityError('Omitted-area test rows differ from the declared area')
                info={'dataset_id':variant['id'],'version':'refined' if self.refined else 'original','comparison':'primary',
                    'kind':task['kind'],'bounds':'primary','k':variant['k'],'repeat':task['repeat'],'fold':task['fold'],
                    'task_index':task['task_index'],'expected_folds':int(self.spec['config']['folds']) if task['kind']=='cv' else area_count,
                    'target':names[target],'group_role':'collection_area' if variant['dataset']=='oliveoil' else 'colour_form_by_sex',
                    'area':task.get('area')}
                yield info,task,y,groups,ids,target,test,list(METHODS)


def validate_task_partitions(tasks,n,folds,repeats):
    cv=[t for t in tasks if t['kind']=='cv']
    if sorted({t['repeat'] for t in cv})!=list(range(repeats)):raise IntegrityError('CV repetition scope differs')
    for repeat in range(repeats):
        subset=[t for t in cv if t['repeat']==repeat]
        if sorted(t['fold'] for t in subset)!=list(range(folds)):raise IntegrityError('CV fold scope differs')
        tests=[checked_indices(t,n)[1] for t in subset]
        if sorted(np.concatenate(tests).tolist())!=list(range(n)):raise IntegrityError('CV tests do not cover every row exactly once')
    area=[t for t in tasks if t['kind']=='leave_area_out']
    if area and sorted(np.concatenate([checked_indices(t,n)[1] for t in area]).tolist())!=list(range(n)):
        raise IntegrityError('Area-out tests do not cover every original row exactly once')


class NewInputs(panel.SavedPanel):
    def cases(self):
        for name in self.manifest['datasets']:
            y,groups,ids,meta=self.grid(name)
            tasks=[t for t in self.manifest['tasks'] if t['dataset_id']==name and t['kind']=='cv']
            if {t['comparison'] for t in tasks}!={'primary','wide','kminus1','kplus1'}:raise IntegrityError('Phase3 comparison scope differs')
            for comparison in ('primary','wide','kminus1','kplus1'):
                sub=[t for t in tasks if t['comparison']==comparison]
                expanded=[{**t,**{key:self.read(self.run/t['split_file'])[key] for key in ('train','test')}} for t in sub]
                validate_task_partitions(expanded,len(y),self.manifest['config']['folds'],
                    self.manifest['config']['repeats'] if comparison=='primary' else 1)
            for task in tasks:
                path=self.run/task['split_file'];self.bytes(path,task['split_sha256']);split=self.read(path)
                _,test=checked_indices(split,len(y))
                methods=[m for m in METHODS if m in task['methods']+task['conditional_methods']]
                expected=list(METHODS if task['comparison']=='primary' else METHODS[:2] if task['comparison']=='wide' else METHODS[:3])
                if methods!=expected:raise IntegrityError('Phase3 conditional-mean method scope differs')
                info={'dataset_id':name,'version':'phase3','comparison':task['comparison'],'kind':'cv','bounds':task['bounds'],
                    'k':task['k'],'repeat':task['repeat'],'fold':task['fold'],'task_index':task['task_index'],
                    'expected_folds':self.manifest['config']['folds'],'target':meta['target_name'],
                    'group_role':'frozen_sampling_group','area':None}
                yield info,task,y,groups,ids,meta['target_index'],test,methods

    def get_parameters(self,task,method,y):
        return self.parameters(task,method,allow_cv=True)


def loss_record(loss,weights,expected_rows):
    loss=np.asarray(loss,float);weights=np.asarray(weights,float)
    if loss.shape!=weights.shape or loss.ndim!=1 or len(loss)!=expected_rows or np.any(weights<=0) or not np.isfinite(weights).all():
        raise IntegrityError('Loss rows and fixed evaluation weights differ')
    valid=np.isfinite(loss)
    denominator=float(weights[valid].sum())
    numerator=float(loss[valid]@weights[valid])
    return {'planned_rows':expected_rows,'available_rows':int(valid.sum()),'complete':bool(valid.all()),
        'planned_weight':float(weights.sum()),'available_weight':denominator,'weighted_loss_sum':numerator,
        'available_row_loss':numerator/denominator if denominator>0 else None,
        'loss':numerator/denominator if valid.all() else None}


def summarize_folds(frame,paired=False):
    model_keys=['left_method','right_method'] if paired else ['method']
    keys=['dataset_id','version','comparison','kind','bounds','k',*model_keys,'weighting']
    repetitions=[]
    for values,g in frame.groupby(keys+['repeat'],dropna=False,sort=True):
        info=dict(zip(keys+['repeat'],values));expected=int(g.expected_folds.iloc[0])
        if g.expected_folds.nunique()!=1 or len(g)!=expected or g.fold.nunique()!=expected:
            raise IntegrityError('Repeated loss table omits or duplicates a planned fold')
        total_weight=float(g.planned_weight.sum())
        if not np.isclose(total_weight,1.,rtol=1e-12,atol=1e-13):raise IntegrityError('Complete partition weights do not sum to one')
        complete=bool(g.complete.all());denom=float(g.available_weight.sum());num=float(g.weighted_loss_sum.sum())
        repetitions.append({**info,'planned_folds':expected,'complete_folds':int(g.complete.sum()),'complete':complete,
            'planned_rows':int(g.planned_rows.sum()),'available_rows':int(g.available_rows.sum()),
            'available_weight':denom,'loss':num if complete else None,'available_row_loss':num/denom if denom else None,
            'negative_fold_differences':int((g.loss<0).sum()) if paired else None})
    reps=pd.DataFrame(repetitions);summaries=[]
    for values,g in reps.groupby(keys,dropna=False,sort=True):
        good=g[g.complete]
        summaries.append({**dict(zip(keys,values)),'planned_repetitions':len(g),'complete_repetitions':len(good),
            'loss':float(good.loss.mean()) if len(good) else None,
            'minimum_loss':float(good.loss.min()) if len(good) else None,'maximum_loss':float(good.loss.max()) if len(good) else None,
            'negative_repetition_differences':int((good.loss<0).sum()) if paired else None,
            'uncertainty':'Descriptive range across overlapping complete repetitions; no independent-fold standard error'})
    return reps,pd.DataFrame(summaries)


def verify_protocol(protocol=PROTOCOL,receipt=RECEIPT):
    protocol,receipt=Path(protocol),Path(receipt)
    if old.digest(protocol)!=PROTOCOL_SHA:raise IntegrityError('Public reproduction analysis rules hash differs')
    recorded=json.loads(receipt.read_text())
    if (recorded['sha256']!=PROTOCOL_SHA or recorded['artifact_role'] != 'public_reproduction_rules'
        or recorded['historical_preregistration_receipt'] is not False):
        raise IntegrityError('Public reproduction rules receipt differs')
    return recorded


def run_analysis(inputs,output,protocol=PROTOCOL,receipt=RECEIPT):
    verify_protocol(protocol,receipt);output=Path(output).resolve()
    sources=inputs if isinstance(inputs,list) else [inputs]
    roots=[]
    for source in sources:
        roots.extend(getattr(source,key) for key in ('primary','gaussian','transport','reference','refined','run') if getattr(source,key,None))
    if any(output==Path(root) or Path(root) in output.parents for root in roots):raise ValueError('Output must be separate from all frozen fitting inputs')
    output.mkdir(parents=True,exist_ok=False);(output/'predictions').mkdir()
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'refitting':False,'methods':list(METHODS),
        'loss':'Squared observed-minus-predicted conditional mean on the original log/ALR scale; lower is better',
        'paired_direction':'left loss minus right loss; negative favors left',
        'protocol_sha256':old.digest(protocol),'receipt_sha256':old.digest(receipt),
        'source_sha256':{p:old.digest(ROOT/p) for p in SOURCE_NAMES},
        'versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'pandas':pd.__version__},
        'input_roots':[str(x) for x in roots],'scope':'All declared held-out folds; original/refined separate; phase3 sensitivities separate',
        'group_weighting':'Each complete test partition assigns 1/(G*n_g) to original row in group g; no fitting reweighting',
        'missingness':'All planned rows retained; partial available-row scores labelled separately; primary summaries require complete repetitions'}
    old.write_json(output/'analysis_plan.json',manifest)
    for relative in SOURCE_NAMES:
        dest=output/'source_snapshot'/relative;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/relative).read_bytes())
    for path in (Path(protocol),Path(receipt)):(output/path.name).write_bytes(path.read_bytes())
    records=[];pairs=[];diagnostics=[];files={};case_count=0
    for source in sources:
        for info,task,y,groups,row_ids,target,test,methods in source.cases():
            case_count+=1;observed=y[test,target]
            weights={'observation':panel.grid_weights(groups)[test],'equal_original_group':panel.grid_weights(groups,True)[test]}
            losses={};arrays={'test_indices':test,'row_ids':np.asarray(row_ids)[test].astype(str),'groups':np.asarray(groups)[test].astype(str),
                            'observed_analysis_target':observed,'target_index':np.array(target)}
            for method in methods:
                predicted=np.full(len(test),np.nan)
                diagnostic={**info,'method':method,'planned_rows':len(test),'available_rows':0,'source_failed':None,
                    'score_converged':None,'boundary_hit':None,'reason':None}
                try:
                    loader=source.get_parameters if isinstance(source,NewInputs) else source.parameters
                    params,center,scale,record=loader(task,method,y)
                    if method!=BASELINE and int(params['k'])!=info['k']:raise IntegrityError('Saved component count differs from fixed task')
                    diagnostic.update(source_failed=record.get('failed'),score_converged=record.get('score_converged'),boundary_hit=record.get('boundary_hit'))
                    conditioning=y[test].copy();conditioning[:,target]=np.nan
                    predicted=panel.conditional_mean(params,method,conditioning,target,center,scale)
                    if predicted.shape!=observed.shape:raise IntegrityError('Prediction row shape differs from frozen held-out rows')
                except (old.IntegrityError,panel.IntegrityError):raise
                except (FileNotFoundError,ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:
                    diagnostic['reason']=type(exc).__name__+': '+str(exc)
                with np.errstate(over='ignore',invalid='ignore'):loss=(observed-predicted)**2
                loss[~np.isfinite(loss)]=np.nan
                diagnostic['available_rows']=int(np.isfinite(loss).sum())
                if diagnostic['available_rows']!=len(test) and diagnostic['reason'] is None:
                    diagnostic['reason']='Nonfinite conditional mean or squared loss on one or more planned rows'
                diagnostics.append(diagnostic)
                losses[method]=loss;arrays['prediction_'+method]=predicted;arrays['loss_'+method]=loss
                for weighting,w in weights.items():records.append({**info,'method':method,'weighting':weighting,**loss_record(loss,w,len(test))})
            for left,right in itertools.combinations(methods,2):
                difference=losses[left]-losses[right]
                for weighting,w in weights.items():pairs.append({**info,'left_method':left,'right_method':right,'weighting':weighting,
                    **loss_record(difference,w,len(test))})
            filename=f"{info['version']}_{info['dataset_id']}_{info['comparison']}_{info['task_index']:06d}.npz"
            path=output/'predictions'/filename;np.savez_compressed(path,**arrays);files[str(path.relative_to(output))]=old.digest(path)
    if not records:raise ValueError('No declared held-out cases')
    folds=pd.DataFrame(records);paired=pd.DataFrame(pairs)
    reps,summary=summarize_folds(folds);pair_reps,pair_summary=summarize_folds(paired,True)
    tables={'prediction_folds':folds,'paired_prediction_folds':paired,'prediction_repetitions':reps,'prediction_summary':summary,
        'paired_prediction_repetitions':pair_reps,'paired_prediction_summary':pair_summary,'prediction_diagnostics':pd.DataFrame(diagnostics)}
    for name,frame in tables.items():
        frame.to_csv(output/(name+'.csv'),index=False);files[name+'.csv']=old.digest(output/(name+'.csv'))
    old.write_json(output/'summary.json',{'prediction_summary':summary.to_dict('records'),'paired_prediction_summary':pair_summary.to_dict('records')})
    files['summary.json']=old.digest(output/'summary.json')
    input_hashes={}
    for source in sources:
        for path,sha in source.hashes.items():
            if path in input_hashes and input_hashes[path]!=sha:raise IntegrityError('Readers observed different input versions')
            if old.digest(path)!=sha:raise IntegrityError('Input changed during mean-loss analysis: '+path)
            input_hashes[path]=sha
    manifest.update(completed_utc=datetime.now(timezone.utc).isoformat(),held_out_cases=case_count,
        planned_task_method_predictions=len(diagnostics),complete_task_method_predictions=sum(r['available_rows']==r['planned_rows'] for r in diagnostics),
        input_sha256=input_hashes,output_sha256=files)
    if any(old.digest(ROOT/p)!=sha for p,sha in manifest['source_sha256'].items()):raise IntegrityError('Analysis source changed during execution')
    old.write_json(output/'analysis_manifest.json',manifest)
    lines=['Conditional log-mean prediction loss','====================================','',
        'Read-only assessment using saved likelihood-selected fits. Lower squared error is better.',
        'The target and prediction are on the original log/ALR scale. No exponentiation or density Jacobian is used.',
        'Primary scores require every planned held-out row in a complete repetition. Missingness and partial scores remain in the tables.',
        'Ranges across overlapping CV repetitions are descriptive, not independent-sample uncertainty.','',
        f"Held-out cases: {case_count}; complete task/method predictions: {manifest['complete_task_method_predictions']}/{len(diagnostics)}.",'']
    for row in summary.to_dict('records'):
        lines.append(f"{row['dataset_id']} | {row['version']} | {row['comparison']} | {row['kind']} | {row['weighting']} | {row['method']}: "
            f"{row['loss']:.8g} [{row['minimum_loss']:.8g}, {row['maximum_loss']:.8g}], complete {row['complete_repetitions']}/{row['planned_repetitions']}" if row['loss'] is not None and np.isfinite(row['loss']) else
            f"{row['dataset_id']} | {row['version']} | {row['method']}: no complete repetition")
    (output/'assessment.rst').write_text('\n'.join(lines)+'\n')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--phase3-run',type=Path)
    parser.add_argument('--primary',type=Path);parser.add_argument('--gaussian',type=Path);parser.add_argument('--transport',type=Path)
    parser.add_argument('--reference',type=Path);parser.add_argument('--refined',type=Path)
    parser.add_argument('--protocol',type=Path,default=PROTOCOL);parser.add_argument('--receipt',type=Path,default=RECEIPT)
    args=parser.parse_args();verify_protocol(args.protocol,args.receipt)
    if args.phase3_run:
        if any((args.primary,args.gaussian,args.transport,args.reference,args.refined)):parser.error('Choose old or phase3 inputs')
        inputs=NewInputs(args.phase3_run)
    else:
        if not all((args.primary,args.gaussian,args.transport,args.reference,args.refined)):parser.error('Old reporting requires all five frozen input roots')
        inputs=[OldInputs(args.primary,args.gaussian,args.transport,args.reference),OldInputs(args.primary,args.gaussian,args.transport,args.reference,args.refined)]
    result=run_analysis(inputs,args.output,args.protocol,args.receipt)
    print(json.dumps({key:result[key] for key in ('held_out_cases','planned_task_method_predictions','complete_task_method_predictions')}))


if __name__=='__main__':main()
