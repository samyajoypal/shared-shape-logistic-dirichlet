"""Group design, exact measures, failure retention and frozen resumability."""
import copy
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

import application_optimization_refinement as refine
import gaussian_shared_correlation as gaussian
import phase3_applications as runner
import regularized_skew_normal as skew
import run_confirmatory_experiments as base


def read(path): return json.loads(Path(path).read_text())
def write(path, value): runner.control.atomic_json(path, value)


def finite_initial(method, y, k, seed, settings, partitions, dependencies):
    """Construct finite synthetic candidates, without calling an optimizer."""
    d = y.shape[1]; weights = np.full(k, 1/k); means = np.zeros((k,d)); cov = np.tile(np.eye(d), (k,1,1))
    params = {'method': method, 'k': k, 'd': d, 'weights': weights, 'means': means, 'covariances': cov}
    if method in ['shared_ld','full_ld','base_ld']:
        if method=='full_ld' and 'shared_ld' in dependencies:
            theta = base.embed_shared(np.asarray(dependencies['shared_ld'].params['theta']), k, d)
        else:
            theta = base.initial_ld(y, np.arange(len(y))%k, k, method, settings, 0)
        params = {'method': method, 'k': k, 'd': d, 'theta': theta}
    elif method==runner.GC:
        theta = gaussian.pack(weights, means, np.ones((k,d)), np.full(gaussian.correlation_parameter_count(d), .2))
        params = gaussian.parameters(theta, k, d)
    elif method=='gaussian_diag':
        params['covariances'] = np.ones((k,d))
    elif method=='fmvmm_dmm':
        params['alpha'] = np.full((k,d+1), 2.)
    elif method=='fmvmm_t':
        params['components'] = [[means[j], cov[j], None, 7.] for j in range(k)]
    elif method==skew.METHOD:
        params = {'method': method, 'k': k, 'd': d,
                  'theta': skew.pack(weights, means, cov, np.zeros((k,d)))}
    rec = {'fit_seconds': .001, 'mean_train_log_density': 0., 'parameter_count': 1,
           'score_converged': False, 'projected_gradient': 1., 'boundary_hit': False,
           'boundary_hits': [], 'min_weight': float(weights.min()), 'optimizer_success': False}
    start = {'parameters': params, 'failed': False, 'selected': True}
    value = {'parameters': params, 'record': rec, 'starts': [start], 'selected_start': 0}
    model = runner.restore(value); rec['mean_train_log_density'] = float(model.evaluate(y)[0].mean())
    return model


def retained_refiner(saved, y, k, method, settings, policy, refined_shared=None):
    params = copy.deepcopy(saved['parameters']); d = y.shape[1]
    if method=='full_ld' and refined_shared is not None:
        params['theta'] = base.embed_shared(np.asarray(refined_shared.params['theta']), k, d)
    limits = gaussian.bounds(k,d) if method==runner.GC else base.ld_bounds(method,k,d,settings)
    record, objective, gradient = refine.diagnostics(np.asarray(params['theta']), y, k, method, limits, policy['original_gtol'])
    record.update(fit_seconds=.001, parameter_count=len(params['theta']), min_weight=1/k)
    value = {'parameters': params, 'record': record,
             'starts': [{'failed':False,'selected':True,'parameters':params,'exceptions':[]}], 'selected_start':0}
    return runner.restore(value)


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    project = tmp_path/'project'; original = runner.ROOT
    for source in runner.source_files():
        destination = project/source.relative_to(original); destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,destination)
    config = read(runner.CONFIG)
    config.update(folds=2,repeats=2,bootstrap_repetitions=2)
    config['data_metadata'] = 'data/preparation_manifest.json'
    config.pop('expected_counts')
    config['fitting']['n_starts'] = 1
    config['datasets'] = [
        {'id':'chem','csv':'data/chem.csv','columns':['A','B','C'],'transform':'alr',
         'reference':'C','conditional_target':'A/C','k':2},
        {'id':'positive','csv':'data/positive.csv','columns':['X','Y','T'],'transform':'log',
         'conditional_target':'T','k':2}]
    rng = np.random.default_rng(912)
    for spec in config['datasets']:
        raw = np.exp(rng.normal(size=(24,3))*.3+np.arange(3))
        frame = pd.DataFrame({'row_id':[str(i+1) for i in range(24)],
                              'group':np.repeat(['G0','G1','G2','G3','G4','G5'],4),
                              'label':['']*24})
        for j,name in enumerate(spec['columns']): frame[name] = raw[:,j]
        path = project/spec['csv']; path.parent.mkdir(parents=True,exist_ok=True); frame.to_csv(path,index=False)
    monkeypatch.setattr(runner,'ROOT',project)
    preparation={'datasets':{},'sources':{},'archive_members':{},'preparer':{
        'path':'scripts/prepare_extended_application_data.py',
        'sha256':runner.control.digest(project/'scripts/prepare_extended_application_data.py')}}
    for specification in config['datasets']:
        spec=runner.dataset_spec(specification);data=runner.load_dataset(project,spec);meta=data['metadata']
        preparation['datasets'][spec['id']]={'csv':Path(spec['csv']).name,'feature_order':spec['columns'],
            'n_rows':meta['n_rows'],'n_groups':meta['n_groups'],'reference':spec['reference'],
            'target':spec['conditional_target'],'transform_plan':spec['transform'],
            'analysis_coordinate_order':meta['analysis_coordinate_names'],'csv_sha256':meta['raw_data_sha256']}
    write(project/config['data_metadata'],preparation)
    config_path = project/'configs/phase3_applications.json'; write(config_path,config)
    output = tmp_path/'run'; manifest = runner.freeze(output,config_path)
    return output,manifest


def run(output,index):
    return runner.run_task(output,index,initial_fitter=finite_initial,refiner=retained_refiner)


def test_group_folds_deterministic_balancing_and_complete_groups():
    groups = np.array(['large']*9+['a']*2+['b']*3+['c']*5+['d']+['e']*4)
    folded = runner.group_folds(groups,3,75)
    np.testing.assert_array_equal(folded,runner.group_folds(groups,3,75))
    assert set(folded)=={0,1,2}
    assert all(len(set(folded[groups==g]))==1 for g in np.unique(groups))
    assert np.bincount(folded).max()-np.bincount(folded).min() <= 9
    with pytest.raises(ValueError): runner.group_folds(groups,7,75)


def test_bootstrap_samples_complete_groups_with_multiplicity():
    groups = np.array(['a']*2+['b']*3+['c']*4)
    train,draws = runner.group_bootstrap(groups,17)
    expected = np.concatenate([np.flatnonzero(groups==g) for g in draws])
    np.testing.assert_array_equal(train,expected)
    assert len(draws)==3 and len(set(draws))<3


def test_frozen_scope_shared_sensitivity_splits_and_seed_separation(frozen):
    output,manifest = frozen
    assert manifest['counts']=={'prediction_tasks':22,'bootstrap_tasks':4,'total_tasks':26,
        'prediction_reported_mixture_fits':112,'bootstrap_reported_mixture_fits':12,
        'conditional_baseline_fits':14,'additional_diagonal_dependencies':12}
    for task in manifest['tasks']:
        split = read(output/task['split_file'])
        if task['kind']=='bootstrap':
            assert task['seed']!=split['bootstrap_draw_seed']
        if task['comparison']!='primary':
            primary=next(t for t in manifest['tasks'] if t['dataset_id']==task['dataset_id'] and
                t['comparison']=='primary' and t['repeat']==0 and t['fold']==task['fold'] and t['kind']=='cv')
            old = read(output/primary['split_file'])
            assert split['train']==old['train'] and split['test']==old['test']
            assert task['conditional_methods']==[]


def test_training_only_transform_native_dmm_and_baseline_scope(frozen):
    output,manifest = frozen; seen={}
    def capture(method,y,*args):
        seen[method]=np.array(y);return finite_initial(method,y,*args)
    result = runner.run_task(output,0,initial_fitter=capture,refiner=retained_refiner)
    assert all(r['evaluation_available'] for r in result['rows'])
    case = runner.load_case(output,manifest,manifest['tasks'][0])
    np.testing.assert_allclose(seen['shared_ld'].mean(0),0,atol=1e-14)
    np.testing.assert_allclose(seen['shared_ld'].std(0),1,atol=1e-14)
    np.testing.assert_array_equal(seen['fmvmm_dmm'],case['y'][case['train_indices']])
    np.testing.assert_array_equal(seen['base_ld'],seen['fmvmm_dmm'])
    assert not np.allclose(case['test'].mean(0),0)
    baseline=result['conditional_rows'][0]
    assert baseline['conditional_only'] and baseline['conditional_available']
    assert 'ARI' not in baseline and 'mean_log_density' not in baseline
    assert all(r['ARI'] is None for r in result['rows'])


@pytest.mark.parametrize('dataset_id',['chem','positive'])
def test_joint_conditional_original_measure_identities(frozen,dataset_id):
    output,manifest=frozen;task=next(t for t in manifest['tasks'] if t['dataset_id']==dataset_id and t['kind']=='cv')
    case=runner.load_case(output,manifest,task);d=case['train'].shape[1];k=task['k']
    model=finite_initial('gaussian_full',case['train'],k,1,case['settings'],[],{})
    joint,conditional,tau=runner.evaluate(model,case)
    z=case['test'];target=case['metadata']['target_index'];keep=[j for j in range(d) if j!=target]
    raw_joint=multivariate_normal.logpdf(z,mean=np.zeros(d),cov=np.eye(d))
    raw_marginal=multivariate_normal.logpdf(z[:,keep],mean=np.zeros(d-1),cov=np.eye(d-1))
    jac=case['raw_log_jacobian'][case['test_indices']]
    np.testing.assert_allclose(joint,raw_joint-np.log(case['scale']).sum()-jac.sum(1),atol=2e-14)
    np.testing.assert_allclose(conditional,raw_joint-raw_marginal-np.log(case['scale'][target])-jac[:,target],atol=2e-14)
    assert np.allclose(tau.sum(1),1)


def test_complete_resume_and_selected_checkpoint_precedes_evaluation(frozen,monkeypatch):
    output,manifest=frozen;observed=[];original=runner.evaluate
    def check(model,case):
        path=output/'tasks/task_000000'/(model.method+'_fit.json')
        assert path.exists() and read(path)['parameters'] is not None
        observed.append(model.method);return original(model,case)
    monkeypatch.setattr(runner,'evaluate',check)
    first=run(output,0);assert len(observed)==9
    def forbidden(*args,**kwargs):raise AssertionError('Unexpected refit')
    assert runner.run_task(output,0,initial_fitter=forbidden,refiner=forbidden)==first
    directory=output/'tasks/task_000000';(directory/'completion.json').unlink();(directory/'shared_ld.json').unlink()
    resumed=runner.run_task(output,0,initial_fitter=forbidden,refiner=forbidden)
    assert all(r['evaluation_available'] for r in resumed['rows'])


def test_bootstrap_grid_dependency_and_complete_group_sample(frozen):
    output,manifest=frozen;task=next(t for t in manifest['tasks'] if t['kind']=='bootstrap')
    result=run(output,task['task_index'])
    assert len(result['rows'])==3 and len(result['dependency_rows'])==1
    assert result['dependency_rows'][0]['method']=='gaussian_diag'
    assert result['dependency_rows'][0]['fit_available'] and not result['dependency_rows'][0]['evaluation_available']
    case=runner.load_case(output,manifest,task)
    assert list(case['test_indices'])==list(range(24))
    assert not np.array_equal(case['train_indices'],case['test_indices'])
    assert len(result['conditional_rows'])==1


def test_storage_failure_retains_fits_and_all_denominators(frozen,monkeypatch):
    output,manifest=frozen
    def fail(*args,**kwargs):raise OSError('Injected prediction write failure')
    monkeypatch.setattr(runner.np,'savez_compressed',fail)
    result=run(output,0)
    assert all(r['fit_available'] and r['failed'] and not r['evaluation_available'] for r in result['rows'])
    assert result['conditional_rows'][0]['fit_available'] and result['conditional_rows'][0]['failed']
    status=runner.aggregate(output,'prediction')
    assert status['planned_mixture_fits']==112 and status['recorded_mixture_fits']==9
    assert status['mixture_evaluations_available']==0
    assert status['planned_conditional_fits']==10 and status['recorded_conditional_fits']==1


def test_total_fit_failures_preserve_initial_records(frozen):
    output,manifest=frozen
    def failed(*args,**kwargs):raise base.FitFailure('Injected all-start failure',[{'failed':True,'start':i,'message':'failed'} for i in range(2)])
    result=runner.run_task(output,0,initial_fitter=failed,refiner=failed)
    assert all(r['failed'] and not r['fit_available'] for r in result['rows'])
    assert all(r['initial_starts_attempted']==2 and r['initial_starts_failed']==2 for r in result['rows'])
    selected=read(output/'tasks/task_000000/shared_ld_initial_fit.json')
    assert len(selected['starts'])==2
    assert result['conditional_rows'][0]['conditional_available']


@pytest.mark.parametrize('which',['source','config','data','split','task_map','initial','selected','result','training','prediction','completion'])
def test_mutated_frozen_state_rejected_on_resume(frozen,which):
    output,manifest=frozen;run(output,0);directory=output/'tasks/task_000000'
    if which=='source':
        p=output/'source_snapshot/scripts/phase3_applications.py';p.write_text(p.read_text()+'\n# changed\n')
    elif which=='config':
        p=output/'source_snapshot/configs/phase3_applications.json';v=read(p);v['seed']+=1;write(p,v)
    elif which=='data':
        p=output/'source_snapshot/data/chem.csv';p.write_text(p.read_text()+'\n')
    elif which=='split':
        p=output/manifest['tasks'][0]['split_file'];v=read(p);v['train'][0]=v['test'][0];write(p,v)
    elif which=='task_map':
        p=output/'manifest.json';v=read(p);v['tasks'][0]['k']+=1;write(p,v)
    elif which=='initial':
        p=directory/'shared_ld_initial_fit.json';v=read(p);v['starts']=[];write(p,v)
    elif which=='selected':
        p=directory/'shared_ld_fit.json';v=read(p);v['method']='full_ld';write(p,v)
    elif which=='result':
        p=directory/'shared_ld.json';v=read(p);v['row']['mean_log_density']=9.;write(p,v)
    elif which=='prediction':
        (directory/'shared_ld_predictions.npz').unlink()
    elif which=='completion':
        p=directory/'completion.json';v=read(p);v['rows'].pop();write(p,v)
    else:
        p=directory/'training.json';v=read(p);v['partitions'][0][0]=1-v['partitions'][0][0];write(p,v)
    with pytest.raises((runner.control.IntegrityError,RuntimeError)):
        run(output,0)


def test_aggregation_complete_repetitions_and_scoped_conditional_pairing(frozen):
    output,manifest=frozen
    for task in manifest['tasks']:
        if task['kind']=='cv' and task['dataset_id']=='chem' and task['comparison']=='primary':run(output,task['task_index'])
    status=runner.aggregate(output,'prediction')
    assert status['recorded_mixture_fits']==36
    repeats=pd.read_csv(output/'cv_repetitions_prediction.csv')
    assert len(repeats)==20
    paired=pd.read_csv(output/'paired_cv_prediction.csv')
    conditional=paired[paired.comparator=='conditional_log_linear']
    assert conditional[conditional.metric=='mean_log_density'].complete_paired_repetitions.eq(0).all()
    assert conditional[conditional.metric=='mean_conditional_log_density'].complete_paired_repetitions.eq(2).all()
    assert paired[paired.comparator=='full_ld'].mean_shared_minus_comparator.abs().max()<1e-12


def test_aggregation_rejects_duplicate_completion_directory(frozen):
    output,manifest=frozen;run(output,0)
    shutil.copytree(output/'tasks/task_000000',output/'tasks/task_000000_backup')
    with pytest.raises(runner.control.IntegrityError,match='Noncanonical'):
        runner.aggregate(output,'prediction')


def test_equal_group_mean_is_not_observation_mean():
    values=np.array([0.,0.,0.,8.]);groups=np.array(['a','a','a','b'])
    assert runner.group_mean(values,groups)==4 and values.mean()==2


def test_split_validation_rejects_partial_resampled_group(frozen):
    output,manifest=frozen;task=next(t for t in manifest['tasks'] if t['kind']=='bootstrap')
    case=runner.load_case(output,manifest,task);split=copy.deepcopy(case['split']);split['train'].pop()
    with pytest.raises(runner.control.IntegrityError,match='sampled complete groups'):
        runner.validate_split(split,case,task)


def test_failed_dependency_cost_is_included_in_control_policy(frozen):
    output,manifest=frozen;task=next(t for t in manifest['tasks'] if t['kind']=='bootstrap')
    def partly_failed(method,*args):
        if method=='gaussian_diag':
            raise base.FitFailure('Failure after attempted diagonal work',[{'failed':True,'message':'TimeoutError(\'deadline\')'}])
        return finite_initial(method,*args)
    result=runner.run_task(output,task['task_index'],initial_fitter=partly_failed,refiner=retained_refiner)
    dependency=result['dependency_rows'][0]
    control=next(r for r in result['rows'] if r['method']==runner.GC)
    assert dependency['failed'] and dependency['initial_timeout_records']==1
    assert control['fit_available'] and control['dependency_fit_seconds']==dependency['policy_fit_seconds']>0
    assert control['policy_fit_seconds']==pytest.approx(control['fit_seconds']+dependency['policy_fit_seconds'])


def test_compact_metadata_points_to_immutable_original_grid(frozen):
    output,manifest=frozen;run(output,0)
    value=read(output/'tasks/task_000000/shared_ld_fit.json');meta=value['dataset_metadata']
    assert 'row_ids' not in meta and 'group_ids' not in meta
    target=output/meta['dataset_metadata_file'];full=read(target)
    assert runner.control.digest(target)==meta['dataset_metadata_sha256']
    assert len(full['row_ids'])==24 and len(full['group_ids'])==24
    assert meta['analysis_coordinate_names']==full['analysis_coordinate_names']


@pytest.mark.parametrize('mutation',['csv_hash','target','preparer','methods','counts'])
def test_freeze_rejects_inconsistent_scientific_or_preparation_contract(frozen,mutation):
    output,manifest=frozen;config_path=runner.ROOT/'configs/phase3_applications.json'
    config=read(config_path)
    if mutation in ['csv_hash','target','preparer']:
        path=runner.ROOT/config['data_metadata'];metadata=read(path)
        if mutation=='csv_hash':metadata['datasets']['chem']['csv_sha256']='0'*64
        elif mutation=='target':metadata['datasets']['chem']['target']='B/C'
        else:metadata['preparer']['sha256']='0'*64
        write(path,metadata)
    elif mutation=='methods':
        config['datasets'][0]['methods']=['shared_ld'];write(config_path,config)
    else:
        config['expected_counts']={'total_tasks':1};write(config_path,config)
    with pytest.raises((runner.control.IntegrityError,ValueError)):
        runner.freeze(output.parent/'invalid_freeze',config_path)


def test_group_members_preserves_naive_order_and_draws():
    groups=np.array(['b','a','b','d','a','c','b','d'])
    for seed in range(5):
        train,draws=runner.group_bootstrap(groups,seed)
        unique=np.unique(groups);expected_draws=np.random.default_rng(seed).choice(unique,size=len(unique),replace=True)
        assert draws==expected_draws.tolist()
        np.testing.assert_array_equal(train,np.concatenate([np.flatnonzero(groups==g) for g in expected_draws]))


def test_actual_small_training_pipeline_preserves_nesting(frozen):
    output,manifest=frozen;task=next(t for t in manifest['tasks'] if t['kind']=='bootstrap')
    def quick_initial(method,y,k,seed,settings,partitions,dependencies):
        settings={**settings,'maxiter':100,'refinement_maxiter':50}
        return runner.fit_initial(method,y,k,seed,settings,partitions,dependencies)
    result=runner.run_task(output,task['task_index'],initial_fitter=quick_initial)
    assert all(r['fit_available'] and r['evaluation_available'] for r in result['rows'])
    full=next(r for r in result['rows'] if r['method']=='full_ld')
    assert full['full_minus_shared_training_mean_log_density']>=-1e-10
    assert all(r['training_mean_log_density_gain']>=-1e-10 for r in result['rows'])
