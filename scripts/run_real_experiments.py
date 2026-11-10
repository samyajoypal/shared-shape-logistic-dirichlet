"""Repeated held-out studies; no data or model is selected by its outcome.

Primary: two repetitions of five-fold cross-validation on Faithful, Seeds,
and the declared olive-oil subcomposition. Secondary: widened LD bounds on
the first repetition. Overlapping folds are not independent replications.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits
from run_confirmatory_experiments import (
    DEFAULT_SETTINGS, fit_method, make_partitions, write_json, serializable,
)

ROOT = Path(__file__).resolve().parents[1]
METHODS = ['shared_ld','full_ld','gaussian_full','gaussian_diag',
           'fmvmm_t','fmvmm_skew_normal','fmvmm_dmm','base_ld']


def datasets():
    faithful = pd.read_csv(ROOT/'data/faithful.csv')
    seeds = pd.read_csv(ROOT/'data/seeds.csv')
    olive = pd.read_csv(ROOT/'data/oliveoil.csv')
    selected = ['area','perimeter','kernel_length','kernel_width',
                'asymmetry_coefficient','kernel_groove_length']
    acid = olive[['palmitic','linoleic','oleic']].to_numpy(float)
    yield 'faithful',faithful[['eruptions','waiting']].to_numpy(float),None,2,False
    yield 'seeds',seeds[selected].to_numpy(float),seeds['class'].to_numpy(),3,False
    yield 'oliveoil',np.log(acid[:,:2]/acid[:,2:]),olive['macro.area'].to_numpy(),3,True


def save_summary(outdir, rows):
    frame = pd.DataFrame(rows)
    frame.to_csv(outdir/'selected_fits.csv',index=False)
    summary = []
    for (dataset,variant,method),part in frame.groupby(['dataset','variant','method']):
        valid = part[part['failed']==False]
        repeats = []
        for repeat,g in valid.groupby('repeat'):
            # Incomplete folds do not silently become a complete repetition.
            complete = len(g)==5
            repeats.append({'repeat':int(repeat),'complete':complete,
                            'value':float(np.average(g.test_log_density,weights=g.n_test))})
        complete_values=[v['value'] for v in repeats if v['complete']]
        summary.append({'dataset':dataset,'variant':variant,'method':method,
                        'attempted_folds':len(part),'finite_folds':len(valid),
                        'complete_repetitions':len(complete_values),
                        'mean_log_density':np.mean(complete_values) if complete_values else np.nan,
                        'repeat_min':min(complete_values) if complete_values else np.nan,
                        'repeat_max':max(complete_values) if complete_values else np.nan,
                        'mean_ARI':valid.test_ARI.mean() if 'test_ARI' in valid else np.nan,
                        'boundary_checked_folds':int(valid.boundary_hit.notna().sum()),
                        'boundary_folds':valid.boundary_hit.fillna(False).astype(bool).sum(),
                        'nonconverged_folds':(~valid.optimizer_success.fillna(False).astype(bool)).sum(),
                        'score_passed_folds':valid.score_converged.fillna(False).astype(bool).sum(),
                        'score_checked_folds':int(valid.score_converged.notna().sum()),
                        'parameter_count':valid.parameter_count.iloc[0] if len(valid) else np.nan,
                        'median_fit_seconds':valid.fit_seconds.median() if len(valid) else np.nan,
                        'median_standalone_fit_seconds':valid.standalone_fit_seconds.median() if 'standalone_fit_seconds' in valid else np.nan,
                        'median_policy_fit_seconds':valid.policy_fit_seconds.median() if 'policy_fit_seconds' in valid else np.nan,
                        'median_total_seconds_including_failures':part.total_seconds.median()})
    pd.DataFrame(summary).to_csv(outdir/'summary.csv',index=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir',type=Path,default=ROOT/'results/real_expanded')
    parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--datasets',nargs='+',default=['faithful','seeds','oliveoil'])
    parser.add_argument('--seed',type=int,default=202611077)
    parser.add_argument('--skip-sensitivity',action='store_true')
    args=parser.parse_args()
    out=args.outdir
    out.mkdir(parents=True,exist_ok=True)
    (out/'predictions').mkdir(exist_ok=True)
    if (out/'specification.json').exists():
        raise SystemExit('Refusing to overwrite an existing study. Choose a fresh --outdir.')
    spec={'date':'2026-11-07','repetitions':args.repeats,'folds':5,'master_seed':args.seed,
          'datasets':args.datasets,'methods':METHODS,'fitting':DEFAULT_SETTINGS,
          'source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in [Path(__file__),ROOT/'scripts/run_confirmatory_experiments.py',
                                     ROOT/'scripts/shared_shape_ld.py']},
          'data_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in [ROOT/'data/faithful.csv',ROOT/'data/seeds.csv',ROOT/'data/oliveoil.csv']},
          'primary_measure':'Mean held-out log density in original measurement coordinates; olive uses original ALR coordinates.',
          'label_policy':'Labels not used to fit, initialize, stratify or select. Fixed k from external metadata; no k search.',
          'preprocessing':'Training-only mean and population SD. Common initial partitions are from standardized training data.',
          'olive_DMM':'base_ld and fmvmm_dmm fit original ALR (actual composition); alpha bounds [0.01,10000] for bounded base LD. Package DMM retains its unconstrained shape updates. Package fit supplies an extra recorded bounded base-LD start.',
          'other_DMM':'For noncompositional Faithful/Seeds, DMM is an induced baseline on inverse ALR of standardized measurements, not a natural composition model.',
          'sensitivity':None if args.skip_sensitivity else {'repeat':0,'methods':['shared_ld','full_ld'],
              'alpha_bounds':[.05,200.],'inverse_scale_bounds':[.05,20.],
              'scaled_location_bounds':[-80.,80.],'rule':'Secondary results only; never replace primary fits.'},
          'uncertainty':'Repeated folds overlap; no independent-fold standard error or p-value is reported.',
          'limitations':['Faithful observations are sequential/rounded; random splits are descriptive, not forecasting.',
              'Seeds excludes derived compactness based on its documented formula.',
              'Olive uses only the declared three-acid subcomposition; its reference is fixed before fitting.',
              'Real datasets are benchmarks, not evidence of causal or substantive scientific discoveries.']}
    write_json(out/'specification.json',spec)
    (out/'executed_sources').mkdir(exist_ok=True)
    for filename in spec['source_hashes']:
        source=ROOT/filename
        (out/'executed_sources'/source.name).write_bytes(source.read_bytes())
    rows=[]
    with threadpool_limits(limits=1):
        for di,(name,y,labels,k,compositional) in enumerate(datasets()):
            if name not in args.datasets:
                continue
            for repeat in range(args.repeats):
                folds=KFold(5,shuffle=True,random_state=args.seed+1000*di+repeat)
                for fold,(train,test) in enumerate(folds.split(y)):
                    seed=args.seed+100000*di+1000*repeat+10*fold
                    center=y[train].mean(0);scale=y[train].std(0)
                    if np.any(scale<=0):
                        raise ValueError('Constant training variable; no silent removal.')
                    ztrain=(y[train]-center)/scale;ztest=(y[test]-center)/scale
                    partitions=make_partitions(ztrain,k,seed,DEFAULT_SETTINGS['n_starts'])
                    split={'train_indices':train,'test_indices':test,'center':center,'scale':scale,
                           'seed':seed,'partitions':partitions}
                    write_json(out/'splits'/f'{name}_r{repeat}_f{fold}.json',split)
                    variants=['primary']+(['wide_bounds'] if repeat==0 and not args.skip_sensitivity else [])
                    for variant in variants:
                        methods=METHODS if variant=='primary' else ['shared_ld','full_ld']
                        fitted={}
                        for method in methods:
                            t0=time.perf_counter()
                            key=f'{name}_r{repeat}_f{fold}_{variant}_{method}'
                            settings=dict(DEFAULT_SETTINGS)
                            if variant=='wide_bounds':
                                settings.update(spec['sensitivity'])
                            native_dmm=compositional and method in ['fmvmm_dmm','base_ld']
                            fit_y=y[train] if native_dmm else ztrain
                            eval_y=y[test] if native_dmm else ztest
                            jacobian=0. if native_dmm else float(np.log(scale).sum())
                            if native_dmm:
                                settings['alpha_bounds']=[.01,10000.]
                            row={'dataset':name,'repeat':repeat,'fold':fold,'variant':variant,
                                 'method':method,'seed':seed,'n_train':len(train),'n_test':len(test),
                                 'failed':True,'test_log_density':np.nan,'test_ARI':np.nan,
                                 'optimizer_success':False,'score_converged':None,'boundary_hit':None,
                                 'parameter_count':None,'fit_seconds':np.nan,'failure':''}
                            model=None
                            try:
                                warm=fitted.get('shared_ld') if method=='full_ld' else (
                                    fitted.get('fmvmm_dmm') if method=='base_ld' and native_dmm else None)
                                model=fit_method(method,fit_y,k,seed,settings,partitions,warm)
                                fitted[method]=model
                                ell,tau=model.evaluate(eval_y)
                                ell=ell-jacobian
                                if not np.isfinite(ell).all():
                                    raise FloatingPointError('Nonfinite held-out density')
                                row.update({key:model.record.get(key) for key in ['optimizer_success',
                                    'score_converged','boundary_hit','projected_gradient','parameter_count',
                                    'fit_seconds','dependency_fit_seconds','standalone_fit_seconds',
                                    'starts_attempted','starts_failed','message']})
                                row.update({'failed':False,'test_log_density':float(ell.mean()),
                                    'train_log_density':model.record['mean_train_log_density']-jacobian,
                                    'test_ARI':adjusted_rand_score(labels[test],tau.argmax(1)) if labels is not None else np.nan})
                                means=model.component_means()
                                if not native_dmm:
                                    means=center+scale*means
                                write_json(out/'fits'/f'{key}.json',{'record':row,'parameters':model.params,
                                    'starts':model.starts,'settings':settings,'component_means_original_coordinates':means,
                                    'density_jacobian_subtracted':jacobian})
                                np.savez_compressed(out/'predictions'/f'{key}.npz',indices=test,
                                                    log_density=ell,responsibilities=tau)
                            except Exception as exc:
                                row['failed']=True
                                row['failure']=repr(exc)
                                write_json(out/'fits'/f'{key}.json',{'record':row,'settings':settings,
                                     'starts':model.starts if model is not None else getattr(exc,'starts',[]),
                                     'parameters':model.params if model is not None else None,
                                     'traceback':traceback.format_exc()})
                            row['total_seconds']=time.perf_counter()-t0
                            rows.append(row)
                            save_summary(out,rows)
                            print(json.dumps(serializable(row)),flush=True)


if __name__=='__main__':
    main()
