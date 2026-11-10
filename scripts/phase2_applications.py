"""Frozen, independently resumable application tasks and exact marginal scores.

The two scientific applications are olive-oil chemistry and crab morphology.
No external label enters fitting, initialization, or selection among starts.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import traceback
import fcntl
from contextlib import contextmanager
import numpy as np
import pandas as pd
from scipy.special import gammaln, digamma, polygamma, logsumexp, log_ndtr
from scipy.optimize import linear_sum_assignment
from scipy.stats import multivariate_normal, multivariate_t, beta
from sklearn.model_selection import KFold
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits
import regularized_skew_normal as skew
from run_confirmatory_experiments import (
    DEFAULT_SETTINGS, fit_method, make_partitions, serializable, unpack_ld,
    FitResult, evaluate_params, timed_start,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/phase2_applications.json'
SOURCE_NAMES = ['scripts/phase2_applications.py', 'scripts/run_confirmatory_experiments.py',
                'scripts/shared_shape_ld.py', 'scripts/regularized_skew_normal.py',
                'configs/experiments.json', 'requirements-lock.txt']
DATA_NAMES = ['data/oliveoil.csv', 'data/crabs.csv']


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(serializable(value), indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime_versions():
    return {'python':platform.python_version(),**{name:importlib.metadata.version(name)
            for name in ['numpy','scipy','pandas','scikit-learn','fmvmm','threadpoolctl']}}


@contextmanager
def bounded_skew_optimizer(seconds):
    """Give each optimizer invocation a deadline while preserving fitter records."""
    original=skew.minimize
    def limited(*args,**kwargs):
        with timed_start(seconds):
            return original(*args,**kwargs)
    skew.minimize=limited
    try:
        yield
    finally:
        skew.minimize=original


def restore_fit(value):
    params=value['parameters'];record=value['record'];starts=value['starts']
    if params['method']==skew.METHOD:
        return skew.SkewNormalFit(params,record,starts,-1,record.get('fit_seconds',0.0))
    return FitResult(params['method'],params['k'],params['d'],params,record,starts,-1)


def load_data(variant):
    """Return transformed observations, external labels, blocks and row Jacobians."""
    if variant['dataset'] == 'oliveoil':
        frame = pd.read_csv(ROOT/'data/oliveoil.csv')
        columns = variant['columns']
        raw = frame[columns].to_numpy(float) + variant.get('add_to_all', 0.0)
        if np.any(raw <= 0):
            raise ValueError('Nonpositive acid value; no undeclared replacement.')
        reference = columns.index(variant['reference'])
        indices = [i for i in range(len(columns)) if i != reference]
        y = np.log(raw[:, indices]/raw[:, reference, None])
        names = [columns[i]+'/'+columns[reference] for i in indices]
        # Density is defined in the selected ALR coordinates, not in eight-acid space.
        return y, frame['macro.area'].to_numpy(), frame['region'].to_numpy(), np.zeros(len(y)), names
    frame = pd.read_csv(ROOT/'data/crabs.csv')
    raw = frame[variant['columns']].to_numpy(float)
    if np.any(raw <= 0):
        raise ValueError('Crab lengths must be strictly positive.')
    labels = (frame['sp'].astype(str)+'_'+frame['sex'].astype(str)).to_numpy()
    is_log = variant['transform'] == 'log'
    # All crab scores are in original millimetre coordinates, including log fits.
    return (np.log(raw) if is_log else raw), labels, labels.copy(), (np.log(raw) if is_log else np.zeros_like(raw)), variant['columns']


def tasks(config):
    result = []
    for variant in config['variants']:
        y, labels, blocks, _, _ = load_data(variant)
        vi = variant['id']
        for repeat in range(variant['repeats']):
            # The identical row splits are reused across variants of one dataset.
            seed = config['seed'] + (0 if variant['dataset']=='oliveoil' else 100000) + repeat
            for fold, (train, test) in enumerate(KFold(config['folds'], shuffle=True, random_state=seed).split(y)):
                result.append({'variant':vi,'kind':'cv','repeat':repeat,'fold':fold,
                               'train':train.tolist(),'test':test.tolist(),'bounds':'primary'})
                if variant.get('primary') and repeat == 0:
                    result.append({'variant':vi,'kind':'cv','repeat':repeat,'fold':fold,
                                   'train':train.tolist(),'test':test.tolist(),'bounds':'wide'})
        if variant.get('primary'):
            result.append({'variant':vi,'kind':'full','repeat':0,'fold':0,
                           'train':list(range(len(y))),'test':list(range(len(y))),'bounds':'primary'})
            for rep in range(config['bootstrap_repetitions']):
                rng = np.random.default_rng(np.random.SeedSequence([config['seed'], 820, len(result), rep]))
                # Preserve the observed collection-area / colour-by-sex sampling design.
                train = np.concatenate([rng.choice(np.flatnonzero(blocks==b), size=np.sum(blocks==b), replace=True)
                                        for b in sorted(set(blocks))])
                result.append({'variant':vi,'kind':'bootstrap','repeat':rep,'fold':0,
                               'train':train.tolist(),'test':list(range(len(y))),'bounds':'primary'})
            if variant['dataset'] == 'oliveoil':
                for fold, area in enumerate(sorted(set(blocks))):
                    result.append({'variant':vi,'kind':'leave_area_out','repeat':0,'fold':fold,
                                   'area':str(area),'train':np.flatnonzero(blocks!=area).tolist(),
                                   'test':np.flatnonzero(blocks==area).tolist(),'bounds':'primary'})
    for i, task in enumerate(result):
        task['task_index'] = i
        task['seed'] = int(np.random.SeedSequence([config['seed'], 701, i]).generate_state(1)[0])
    return result


def freeze(output, config_path=CONFIG):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'specification.json').exists():
        raise FileExistsError('Study already frozen; use a fresh output directory.')
    config = json.loads(Path(config_path).read_text())
    spec = {'created_utc':datetime.now(timezone.utc).isoformat(), 'config':config,
            'runtime_versions':runtime_versions(),
            'source_hashes':{name:digest(ROOT/name) for name in SOURCE_NAMES},
            'data_hashes':{name:digest(ROOT/name) for name in DATA_NAMES},
            'tasks':tasks(config)}
    for name in SOURCE_NAMES + DATA_NAMES:
        target = output/'executed_sources'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    atomic_json(output/'specification.json',spec)
    print(json.dumps({'tasks':len(spec['tasks']),'specification_sha256':digest(output/'specification.json')}))
    return spec


def verify_sources(spec,output=None):
    for name, value in {**spec['source_hashes'], **spec['data_hashes']}.items():
        if digest(ROOT/name) != value:
            raise RuntimeError('Frozen source or data changed: '+name)
        if output is not None and digest(Path(output)/'executed_sources'/name)!=value:
            raise RuntimeError('Executed source/data snapshot changed: '+name)


def ld_components(params):
    method, k, d = params['method'], params['k'], params['d']
    if method == 'fmvmm_dmm':
        return np.log(np.asarray(params['weights'])), np.asarray(params['alpha']), np.ones((k,d)), np.zeros((k,d))
    logpi,alpha,a,b=unpack_ld(np.asarray(params['theta']), method, k, d)
    return logpi,np.broadcast_to(alpha,(k,d+1)),a,b


def marginal_log_density(params, y, keep):
    """Exact marginal density, integrating omitted coordinates; no refitting."""
    y = np.asarray(y, float)
    keep = np.asarray(keep, int)
    method, k, d = params['method'], params['k'], params['d']
    if len(keep)==0:
        return np.zeros(len(y))
    if len(set(keep.tolist())) != len(keep) or np.any((keep < 0) | (keep >= d)):
        raise ValueError('Invalid coordinate subset')
    values = y[:, keep]
    terms=[]
    if method in ['shared_ld','full_ld','base_ld','fmvmm_dmm']:
        logpi, alpha, a, b = ld_components(params)
        for j in range(k):
            shapes=np.r_[alpha[j,keep], alpha[j,-1]]
            w=values*a[j,keep]+b[j,keep]
            norm=logsumexp(np.column_stack([np.zeros(len(y)),w]),axis=1)
            ell=(gammaln(shapes.sum())-gammaln(shapes).sum()+np.log(a[j,keep]).sum()
                 +w@shapes[:-1]-shapes.sum()*norm)
            terms.append(logpi[j]+ell)
    elif method == skew.METHOD:
        logpi, locations, cholesky, betas = skew.unpack(np.asarray(params['theta']),k,d)
        omitted=np.asarray([i for i in range(d) if i not in keep],int)
        for j in range(k):
            omega=cholesky[j]@cholesky[j].T
            covariance=omega[np.ix_(keep,keep)]
            beta_keep=betas[j,keep].copy()
            variance=0.0
            if len(omitted):
                cross=omega[np.ix_(keep,omitted)]
                regression=np.linalg.solve(covariance,cross)
                conditional=omega[np.ix_(omitted,omitted)]-cross.T@regression
                beta_keep += regression@betas[j,omitted]
                variance=float(betas[j,omitted]@conditional@betas[j,omitted])
            residual=values-locations[j,keep]
            ell=np.log(2)+multivariate_normal.logpdf(values,mean=locations[j,keep],cov=covariance)
            ell+=log_ndtr(residual@beta_keep/np.sqrt(1+max(variance,0.0)))
            terms.append(logpi[j]+ell)
    else:
        logpi=np.log(np.asarray(params['weights']))
        for j in range(k):
            if method=='fmvmm_t':
                component=params['components'][j]
                ell=multivariate_t.logpdf(values,loc=np.asarray(component[0])[keep],
                                         shape=np.asarray(component[1])[np.ix_(keep,keep)],df=component[3])
            else:
                cov=np.asarray(params['covariances'])[j]
                cov=np.diag(cov) if method=='gaussian_diag' else cov
                ell=multivariate_normal.logpdf(values,mean=np.asarray(params['means'])[j,keep],
                                               cov=cov[np.ix_(keep,keep)])
            terms.append(logpi[j]+ell)
    return logsumexp(np.column_stack(terms),axis=1)


def component_profiles(model, center, scale, names, variant):
    means=center+scale*model.component_means()
    profiles={'means_in_analysis_coordinates':means,'coordinate_names':names,
              'geometric_means_or_ratios':np.exp(means) if variant.get('transform')!='raw' else None}
    if model.method not in ['shared_ld','full_ld','base_ld','fmvmm_dmm']:
        return profiles
    logpi, alpha, a, b=ld_components(model.params)
    location=center-scale*b/a
    sigma=scale/a
    covariance=[]
    for j in range(model.k):
        v=polygamma(1,alpha[j,:-1]);r=polygamma(1,alpha[j,-1])
        covariance.append(sigma[j,:,None]*sigma[j,None,:]*(np.diag(v)+r))
    covariance=np.asarray(covariance)
    sd=np.sqrt(np.diagonal(covariance,axis1=1,axis2=2))
    correlation=covariance/(sd[:,:,None]*sd[:,None,:])
    quantiles=[]
    for probability in [.1,.5,.9]:
        u=beta.ppf(probability,alpha[:,:-1],alpha[:,-1:])
        complement=beta.ppf(1-probability,alpha[:,-1:],alpha[:,:-1])
        # Invert the complementary beta distribution separately; 1-u can
        # round to zero when the log-ratio quantile is large but finite.
        quantiles.append(location+sigma*(np.log(u)-np.log(complement)))
    profiles.update({'weights':np.exp(logpi),'alpha':alpha,'location':location,'scale':sigma,
                     'covariance':covariance,'correlation':correlation,
                     'marginal_quantiles_10_50_90':np.asarray(quantiles),
                     'marginal_skewness':(polygamma(2,alpha[:,:-1])-polygamma(2,alpha[:,-1:]))/
                        (polygamma(1,alpha[:,:-1])+polygamma(1,alpha[:,-1:]))**1.5})
    if variant['dataset']=='crabs' and variant['transform']=='log':
        contrasts=[]
        for numerator,denominator in [('CW','FL'),('CW','BD'),('CL','RW')]:
            c=np.zeros(model.d);c[names.index(numerator)]=1;c[names.index(denominator)]=-1
            contrasts.append({'name':numerator+'/'+denominator,'geometric_ratio':np.exp(means@c),
                              'log_ratio_sd':np.sqrt(np.einsum('i,kij,j->k',c,covariance,c))})
        profiles['biological_contrasts']=contrasts
    return profiles


def _run_task(output, index):
    output=Path(output)
    spec=json.loads((output/'specification.json').read_text())
    verify_sources(spec,output)
    if runtime_versions()!=spec['runtime_versions']:
        raise RuntimeError('Worker runtime differs from the frozen study runtime')
    task=spec['tasks'][index]
    if task['task_index'] != index:
        raise AssertionError('Invalid task mapping')
    directory=output/'tasks'/f'{index:05d}'
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'completion.json').exists():
        saved=json.loads((directory/'completion.json').read_text())
        if saved['task']!=task or saved['specification_sha256']!=digest(output/'specification.json'):
            raise RuntimeError('Completed task identity differs from the frozen study')
        return saved
    config=spec['config'];variant=next(v for v in config['variants'] if v['id']==task['variant'])
    y,labels,blocks,raw_jacobian,names=load_data(variant)
    train=np.asarray(task['train']);test=np.asarray(task['test']);k=variant['k'];d=y.shape[1]
    center=y[train].mean(0);scale=y[train].std(0)
    if np.any(scale<=0):
        raise ValueError('Constant training coordinate')
    ztrain=(y[train]-center)/scale;ztest=(y[test]-center)/scale
    settings={**DEFAULT_SETTINGS,**config['fitting']}
    if task['bounds']=='wide':
        settings.update(config['wide_bounds'])
    methods=(config['bootstrap_methods'] if task['kind']=='bootstrap' else
             ['shared_ld','full_ld'] if task['bounds']=='wide' else config['methods'])
    partitions=make_partitions(ztrain,k,task['seed'],settings['n_starts'])
    atomic_json(directory/'split.json',{**task,'center':center,'scale':scale,'partitions':partitions})
    fitted={};records=[]
    target_name=variant['conditional_target']
    target=names.index(target_name)
    keep=[i for i in range(d) if i!=target]
    for method in methods:
        # Per-method checkpoints are immutable completed work, including failures.
        checkpoint=directory/(method+'.json')
        if checkpoint.exists():
            saved=json.loads(checkpoint.read_text())
            if (saved.get('specification_sha256')!=digest(output/'specification.json')
                    or saved['record']['task_index']!=index or saved['record']['method']!=method
                    or saved['record']['variant']!=variant['id']):
                raise RuntimeError('Method checkpoint has a different task identity')
            records.append(saved['record'])
            if not saved['record']['failed']:
                fitted[method]=restore_fit(saved)
            continue
        started=time.perf_counter();model=None
        row={'method':method,'failed':True,'task_index':index,'variant':variant['id'],
             'kind':task['kind'],'repeat':task['repeat'],'fold':task['fold'],
             'bounds':task['bounds'],'n_train':len(train),'n_test':len(test),'k':k,'d':d}
        try:
            native=variant['dataset']=='oliveoil' and method in ['base_ld','fmvmm_dmm']
            fit_y=y[train] if native else ztrain
            eval_y=y[test] if native else ztest
            method_settings=dict(settings)
            if native: method_settings['alpha_bounds']=[.01,10000.]
            if method==skew.METHOD:
                with bounded_skew_optimizer(settings['start_timeout_seconds']):
                    model=skew.fit(fit_y,k,task['seed'],partitions,gaussian_candidate=fitted.get('gaussian_full'),
                                   n_starts=settings['n_starts'],maxiter=settings['maxiter'],
                                   polish_maxiter=settings['refinement_maxiter'],gtol=settings['gtol'])
            else:
                warm=fitted.get('shared_ld') if method=='full_ld' else fitted.get('fmvmm_dmm') if native and method=='base_ld' else None
                model=fit_method(method,fit_y,k,task['seed'],method_settings,partitions,warm)
            ell,tau=model.evaluate(eval_y)
            marginal=marginal_log_density(model.params,eval_y,keep)
            conditional=ell-marginal
            jacobian=0. if native else np.log(scale).sum()
            conditional_jacobian=0. if native else np.log(scale[target])
            ell=ell-jacobian;conditional=conditional-conditional_jacobian
            if variant['dataset']=='crabs':
                ell-=raw_jacobian[test].sum(1)
                conditional-=raw_jacobian[test,target]
            if not np.isfinite(ell).all() or not np.isfinite(conditional).all():
                raise FloatingPointError('Nonfinite held-out density')
            row.update({key:model.record.get(key) for key in ['optimizer_success','score_converged',
                        'projected_gradient','boundary_hit','parameter_count','fit_seconds',
                        'starts_attempted','starts_failed','min_weight','message',
                        'dependency_fit_seconds','standalone_fit_seconds']})
            if method==skew.METHOD:
                dependency=fitted.get('gaussian_full')
                dependency_seconds=(dependency.record.get('fit_seconds',0.0) if dependency else 0.0)
                row.update(fit_seconds=model.elapsed_seconds,starts_attempted=len(model.starts),
                           starts_failed=sum(bool(s.get('failed')) for s in model.starts),
                           dependency_fit_seconds=dependency_seconds,
                           standalone_fit_seconds=model.elapsed_seconds+dependency_seconds)
            row['policy_fit_seconds']=row['fit_seconds']
            if method=='full_ld' and fitted.get('shared_ld') is not None:
                row['policy_fit_seconds']+=fitted['shared_ld'].record.get('fit_seconds',0.0)
            elif method==skew.METHOD:
                row['policy_fit_seconds']+=row['dependency_fit_seconds']
            elif native and method=='base_ld' and fitted.get('fmvmm_dmm') is not None:
                row['policy_fit_seconds']+=fitted['fmvmm_dmm'].record.get('fit_seconds',0.0)
            row.update({'failed':False,'mean_log_density':float(ell.mean()),
                        'mean_conditional_log_density':float(conditional.mean()),
                        'conditional_target':target_name,'ARI':float(adjusted_rand_score(labels[test],tau.argmax(1)))})
            profiles=component_profiles(model,np.zeros(d) if native else center,
                                         np.ones(d) if native else scale,names,variant)
            external={str(label):tau[labels[test]==label].sum(0) for label in sorted(set(labels[test]))}
            np.savez_compressed(directory/(method+'_predictions.npz'),indices=test,log_density=ell,
                                conditional_log_density=conditional,responsibilities=tau)
            value={'record':row,'parameters':model.params,'starts':model.starts,'profiles':profiles,
                   'external_label_responsibility_totals':external,'settings':method_settings,
                   'native_DMM':native,'evaluation_measure':'ALR' if variant['dataset']=='oliveoil' else 'millimetres'}
        except Exception as exc:
            row['failed']=True
            row['failure']=repr(exc)
            value={'record':row,'starts':model.starts if model is not None else getattr(exc,'starts',[]),
                   'parameters':model.params if model is not None else None,'traceback':traceback.format_exc()}
        row['total_seconds']=time.perf_counter()-started
        value['specification_sha256']=digest(output/'specification.json')
        atomic_json(checkpoint,value);records.append(row)
        if not row['failed']:
            fitted[method]=model
    completion={'task':task,'records':records,'completed_utc':datetime.now(timezone.utc).isoformat(),
                'specification_sha256':digest(output/'specification.json'),
                'runtime_versions':runtime_versions()}
    atomic_json(directory/'completion.json',completion)
    return completion


def run_task(output,index):
    directory=Path(output)/'tasks'/f'{index:05d}'
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'task.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            return _run_task(output,index)
        finally:
            fcntl.flock(handle,fcntl.LOCK_UN)


def aggregate(output):
    output=Path(output);spec=json.loads((output/'specification.json').read_text())
    verify_sources(spec,output)
    rows=[];completed=[]
    for task in spec['tasks']:
        path=output/'tasks'/f"{task['task_index']:05d}"/'completion.json'
        if path.exists():
            value=json.loads(path.read_text())
            if (value['specification_sha256'] != digest(output/'specification.json') or value['task']!=task
                    or value['runtime_versions']!=spec['runtime_versions']):
                raise ValueError('Task completed under a different specification')
            completed.append(task['task_index']);rows.extend(value['records'])
    frame=pd.DataFrame(rows)
    if not len(frame):
        return {'completed_tasks':0,'expected_tasks':len(spec['tasks'])}
    frame.to_csv(output/'fit_records.csv',index=False)
    summaries=[];repeat_rows=[]
    for (variant,kind,bounds,method),part in frame.groupby(['variant','kind','bounds','method']):
        valid=part[~part.failed]
        summary={'variant':variant,'kind':kind,'bounds':bounds,'method':method,
                 'attempted_fits':len(part),'finite_fits':len(valid),
                 'boundary_checked':int(valid.boundary_hit.notna().sum()) if 'boundary_hit' in valid else 0,
                 'boundary_hits':int(valid.boundary_hit.fillna(False).astype(bool).sum()) if 'boundary_hit' in valid else 0,
                 'stationarity_checked':int(valid.score_converged.notna().sum()) if 'score_converged' in valid else 0,
                 'stationarity_passes':int(valid.score_converged.fillna(False).astype(bool).sum()) if 'score_converged' in valid else 0,
                 'median_total_seconds':float(part.total_seconds.median())}
        if kind=='cv':
            complete=[]
            for repeat,g in valid.groupby('repeat'):
                if len(g)==spec['config']['folds']:
                    rec={'variant':variant,'bounds':bounds,'method':method,'repeat':repeat,
                         'mean_log_density':float(np.average(g.mean_log_density,weights=g.n_test)),
                         'mean_conditional_log_density':float(np.average(g.mean_conditional_log_density,weights=g.n_test))}
                    repeat_rows.append(rec);complete.append(rec)
            summary['complete_repetitions']=len(complete)
            for field in ['mean_log_density','mean_conditional_log_density']:
                values=[r[field] for r in complete]
                summary[field]=float(np.mean(values)) if values else None
                summary[field+'_repeat_min']=min(values) if values else None
                summary[field+'_repeat_max']=max(values) if values else None
        elif kind=='leave_area_out':
            for field in ['mean_log_density','mean_conditional_log_density']:
                summary[field]=float(np.average(valid[field],weights=valid.n_test)) if len(valid)==9 else None
        summaries.append(summary)
    pd.DataFrame(summaries).to_csv(output/'summary.csv',index=False)
    pd.DataFrame(repeat_rows).to_csv(output/'complete_cv_repetitions.csv',index=False)
    paired=[]
    repeats=pd.DataFrame(repeat_rows)
    if len(repeats):
        for (variant,bounds),part in repeats.groupby(['variant','bounds']):
            shared=part[part.method=='shared_ld']
            for method in sorted(set(part.method)-{'shared_ld'}):
                comparator=part[part.method==method]
                common=shared.merge(comparator,on=['variant','bounds','repeat'],suffixes=('_shared','_comparator'))
                for field in ['mean_log_density','mean_conditional_log_density']:
                    differences=common[field+'_shared']-common[field+'_comparator']
                    paired.append({'variant':variant,'bounds':bounds,'comparator':method,'measure':field,
                        'common_complete_repetitions':len(common),
                        'mean_shared_minus_comparator':float(differences.mean()) if len(common) else None,
                        'difference_repeat_min':float(differences.min()) if len(common) else None,
                        'difference_repeat_max':float(differences.max()) if len(common) else None,
                        'uncertainty':'descriptive repeat range; overlapping splits are not independent'})
    pd.DataFrame(paired).to_csv(output/'paired_cv_comparisons.csv',index=False)
    bootstrap_rows=[];stability=[]
    for variant in spec['config']['variants']:
        if not variant.get('primary'):continue
        y,_,_,_,names=load_data(variant);standard=y.std(0)
        matching_tasks=[t for t in spec['tasks'] if t['variant']==variant['id']]
        full=next(t for t in matching_tasks if t['kind']=='full')
        if full['task_index'] not in completed:continue
        for method in spec['config']['bootstrap_methods']:
            full_path=output/'tasks'/f"{full['task_index']:05d}"/(method+'.json')
            if not full_path.exists():continue
            ref=json.loads(full_path.read_text())
            if ref['record']['failed']:continue
            reference=np.asarray(ref['profiles']['means_in_analysis_coordinates'])
            order=np.argsort(reference[:,0],kind='stable');reference=reference[order]
            for task in matching_tasks:
                if task['kind']!='bootstrap':continue
                if task['task_index'] not in completed:continue
                path=output/'tasks'/f"{task['task_index']:05d}"/(method+'.json')
                if not path.exists():continue
                fit=json.loads(path.read_text())
                if fit['record']['failed']:continue
                means=np.asarray(fit['profiles']['means_in_analysis_coordinates'])
                cost=np.sum(((reference[:,None,:]-means[None,:,:])/standard)**2,axis=2)
                rr,cc=linear_sum_assignment(cost);matched=means[cc]
                values={('mean_log_'+name if variant.get('transform')!='raw' else 'mean_'+name):matched[:,i]
                        for i,name in enumerate(names)}
                if variant['dataset']=='crabs' and variant['transform']=='log':
                    for num,den in [('CW','FL'),('CW','BD'),('CL','RW')]:
                        values['mean_log_'+num+'/'+den]=matched[:,names.index(num)]-matched[:,names.index(den)]
                for target,value in values.items():
                    for component,estimate in enumerate(value):
                        bootstrap_rows.append({'variant':variant['id'],'method':method,'replicate':task['repeat'],
                           'component':component,'target':target,'estimate':estimate,
                           'matching_RMS_standardized':float(np.sqrt(cost[rr,cc].mean()/len(names))),
                           'boundary_hit':fit['record'].get('boundary_hit'),
                           'score_converged':fit['record'].get('score_converged')})
    boot=pd.DataFrame(bootstrap_rows)
    boot.to_csv(output/'bootstrap_matched_targets.csv',index=False)
    if len(boot):
        for keys,part in boot.groupby(['variant','method','component','target']):
            lo,median,hi=np.quantile(part.estimate,[.025,.5,.975])
            stability.append(dict(zip(['variant','method','component','target'],keys),
                n_finite=len(part),n_requested=spec['config']['bootstrap_repetitions'],
                percentile025=lo,median=median,percentile975=hi,
                boundary_checked=int(part.boundary_hit.notna().sum()),
                boundary_hits=int(part.boundary_hit.fillna(False).astype(bool).sum()),
                stationarity_checked=int(part.score_converged.notna().sum()),
                stationarity_passes=int(part.score_converged.fillna(False).astype(bool).sum()),
                matching_RMS_standardized_median=float(part.matching_RMS_standardized.median()),
                matching_RMS_standardized_max=float(part.matching_RMS_standardized.max()),
                target_scale='mean of natural log values/ratios' if keys[-1].startswith('mean_log_') else 'original millimetres',
                interpretation='conditional resampling stability; nominal coverage not established'))
    pd.DataFrame(stability).to_csv(output/'bootstrap_stability.csv',index=False)
    status={'completed_tasks':len(completed),'expected_tasks':len(spec['tasks']),
            'missing_task_indices':sorted(set(range(len(spec['tasks'])))-set(completed)),
            'attempted_fits':len(frame),'failed_fits':int(frame.failed.sum()),
            'complete':len(completed)==len(spec['tasks']),
            'generated_utc':datetime.now(timezone.utc).isoformat()}
    atomic_json(output/'aggregate_status.json',status)
    return status


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=CONFIG)
    parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--aggregate',action='store_true')
    parser.add_argument('--task-index',type=int)
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        if args.freeze: freeze(args.output,args.config)
        elif args.aggregate: print(json.dumps(aggregate(args.output)))
        elif args.task_index is not None:
            result=run_task(args.output,args.task_index)
            print(json.dumps({'task_index':args.task_index,'fits':len(result['records']),
                              'failed':sum(r['failed'] for r in result['records'])}))
        else:parser.error('Choose --freeze or --task-index')


if __name__=='__main__':
    main()
