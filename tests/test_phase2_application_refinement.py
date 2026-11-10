"""Frozen task scope, source identity, evaluation and failure retention."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import application_optimization_refinement as refine
import gaussian_shared_correlation as gaussian
import phase2_application_refinement as runner
import phase2_gaussian_control as control
import run_confirmatory_experiments as base


def write(path,value):control.atomic_json(path,value)
def read(path):return json.loads(path.read_text())


def retained(saved,y,k,method,settings,policy,refined_shared=None):
    params=saved['parameters'];theta=np.asarray(params['theta']);d=y.shape[1]
    limits=gaussian.bounds(k,d) if method==gaussian.METHOD else base.ld_bounds(method,k,d,settings)
    record,_,_=refine.diagnostics(theta,y,k,method,limits,1e-6)
    record.update(fit_seconds=.001,dependency_fit_seconds=.001 if refined_shared else 0.,
        policy_fit_seconds=.002 if refined_shared else .001,parameter_count=len(theta),
        starts_attempted=1,starts_failed=0,min_weight=.5)
    starts=[{'parameters':params,'failed':False,'selected':True}]
    if method==gaussian.METHOD:return gaussian.GaussianSharedCorrelationFit(params,record,starts,0,.001)
    return base.FitResult(method,k,d,params,record,starts,0)


@pytest.fixture
def frozen(tmp_path):
    roots={name:tmp_path/name for name in ['primary','gaussian','transport']}
    data_path=roots['primary']/'executed_sources/data/crabs.csv';data_path.parent.mkdir(parents=True)
    rng=np.random.default_rng(718);raw=np.exp(2+.2*rng.normal(size=(24,5)))
    names=['FL','RW','CL','CW','BD'];frame=pd.DataFrame(raw,columns=names)
    frame['sp']=['B']*12+['O']*12;frame['sex']=['M','F']*12;frame.to_csv(data_path,index=False)
    variant={'id':'crabs_log5','dataset':'crabs','columns':names,'transform':'log','k':2,
             'primary':True,'conditional_target':'BD'}
    settings={**base.DEFAULT_SETTINGS,'n_starts':1}
    tasks=[]
    for kind,bounds,train,test in [('cv','primary',list(range(18)),list(range(18,24))),
        ('cv','wide',list(range(18)),list(range(18,24))),('full','primary',list(range(24)),list(range(24))),
        ('bootstrap','primary',list(range(22))+[0,1],list(range(24)))]:
        tasks.append({'task_index':len(tasks),'variant':'crabs_log5','kind':kind,'bounds':bounds,
                      'repeat':0,'fold':0,'train':train,'test':test,'seed':992+len(tasks)})
    spec={'runtime_versions':control.versions(),'config':{'variants':[variant],'fitting':settings,
        'wide_bounds':{'alpha_bounds':[.01,1000.],'inverse_scale_bounds':[.01,50.],'scaled_location_bounds':[-120.,120.]},
        'folds':1},'tasks':tasks,'data_hashes':{'data/crabs.csv':control.digest(data_path)}}
    write(roots['primary']/'specification.json',spec);primary_hash=control.digest(roots['primary']/'specification.json')
    gc_tasks=[{'task_index':j,'primary_task_index':i,'variant':'crabs_log5','kind':tasks[i]['kind'],
               'repeat':0,'fold':0} for j,i in enumerate([0,2,3])]
    for name,plan in [('gaussian',gc_tasks),('transport',[])]:
        write(roots[name]/'manifest.json',{'primary_metadata_sha256':primary_hash,'versions':control.versions(),'tasks':plan})
    y=np.log(raw)
    for task in tasks:
        index=task['task_index'];directory=roots['primary']/'tasks'/f'{index:05d}'
        train=y[task['train']];center=train.mean(0);scale=train.std(0);z=(train-center)/scale
        split={**task,'center':center,'scale':scale,'partitions':[np.arange(len(train))%2]}
        write(directory/'split.json',split)
        common=base.initial_ld(z,np.arange(len(train))%2,2,'shared_ld',settings,0)
        records=[]
        case={'test':(y[task['test']]-center)/scale,'center':center,'scale':scale,
              'test_raw_jacobian':y[task['test']],'names':names,'variant':variant}
        for method,theta in [('shared_ld',common),('full_ld',base.embed_shared(common,2,5))]:
            params={'method':method,'k':2,'d':5,'theta':theta}
            model=base.FitResult(method,2,5,params,{},[],0)
            ell,conditional,_=runner.evaluate(model,case)
            record={**task,'method':method,'failed':False,'n_test':len(task['test']),
                    'mean_log_density':float(ell.mean()),'mean_conditional_log_density':float(conditional.mean()),
                    'projected_gradient':1.,'score_converged':False,'boundary_hit':False,
                    'fit_seconds':.02,'policy_fit_seconds':.03}
            write(directory/(method+'.json'),{'record':record,'parameters':params,
                'starts':[{'failed':False,'parameters':params}],'specification_sha256':primary_hash})
            records.append(record)
        write(directory/'completion.json',{'task':task,'records':records,'specification_sha256':primary_hash})
        if task['bounds']=='wide':continue
        gc_task=next(t for t in gc_tasks if t['primary_task_index']==index)
        gd=roots['gaussian']/'tasks'/f"task_{gc_task['task_index']:06d}"
        theta=gaussian.pack([.5,.5],np.array([np.full(5,-.3),np.full(5,.3)]),np.ones((2,5)),np.full(5,.3))
        params=gaussian.parameters(theta,2,5);model=gaussian.GaussianSharedCorrelationFit(params,{},[],0,.02)
        ell,conditional,_=runner.evaluate(model,case)
        row={**gc_task,'method':gaussian.METHOD,'failed':False,'fit_available':True,'evaluation_available':True,
             'test_log_density':float(ell.mean()),'conditional_log_density':float(conditional.mean()),
             'score_converged':False,'projected_gradient':1.,'boundary_hit':False,'fit_seconds':.02,'policy_fit_seconds':.03}
        used=[directory/'completion.json',directory/'split.json',directory/'shared_ld.json',directory/'full_ld.json',data_path]
        binding={'primary_task_index':index,'input_sha256':{str(p.relative_to(roots['primary'])):control.digest(p) for p in used}}
        identity={'task_index':gc_task['task_index'],'primary_task_index':index,
                  'manifest_sha256':control.digest(roots['gaussian']/'manifest.json'),'input_binding_sha256':control.semantic_hash(binding)}
        write(gd/'input_binding.json',binding)
        write(gd/'control_fit.json',{**identity,'failed':False,'parameters':params,'record':row,
                                    'starts':[{'failed':False,'parameters':params}],'selected_start':0})
        write(gd/'result.json',{**identity,'status':'complete','task':gc_task,'row':row,'parameters':params})
    config=read(runner.CONFIG);config.update(primary_variants=['crabs_log5'],expected_tasks={'prediction':3,'bootstrap':1},
                                         expected_fits={'prediction':8,'bootstrap':3})
    config_path=tmp_path/'config.json';write(config_path,config)
    output=tmp_path/'refinement';manifest=runner.freeze(output,roots['primary'],roots['gaussian'],roots['transport'],config_path)
    return output,roots,manifest


def test_scope_and_first_original_coordinate_predictions(frozen):
    output,roots,manifest=frozen
    assert [t['phase'] for t in manifest['tasks']]==['prediction']*3+['bootstrap']
    for index in range(4):
        result=runner.run_task(output,index,fitter=retained)
        assert all(r['evaluation_available'] for r in result['rows'])
        for row in result['rows']:
            old=result['original_baseline_records'][row['method']]
            assert row['mean_log_density']==pytest.approx(old['mean_log_density'],abs=1e-12)
            assert row['mean_conditional_log_density']==pytest.approx(old['mean_conditional_log_density'],abs=1e-12)
            saved=read(output/'tasks'/f'task_{index:06d}'/(row['method']+'.json'))
            assert saved['parameters'] is not None and saved['training_transform']['scale']
    status=runner.aggregate(output,'prediction')
    assert status['planned_fits']==status['recorded_fits']==status['evaluation_available']==8
    changes=pd.read_csv(output/'optimization_changes_prediction.csv')
    assert len(changes)==8
    np.testing.assert_allclose(changes.joint_prediction_change,0.,atol=1e-12)
    assert len(pd.read_csv(output/'original_refined_method_records_prediction.csv'))==8


def test_completed_and_fit_checkpoint_resume_do_not_refit(frozen):
    output,roots,manifest=frozen
    first=runner.run_task(output,0,fitter=retained)
    def forbidden(*args,**kwargs):raise AssertionError('Unexpected refit')
    assert runner.run_task(output,0,fitter=forbidden)==first
    directory=output/'tasks/task_000000'
    (directory/'completion.json').unlink();(directory/'shared_ld.json').unlink()
    second=runner.run_task(output,0,fitter=forbidden)
    assert all(r['evaluation_available'] for r in second['rows'])


def test_foreign_fit_checkpoint_rejected(frozen):
    output,roots,manifest=frozen;runner.run_task(output,0,fitter=retained)
    directory=output/'tasks/task_000000';(directory/'completion.json').unlink();(directory/'shared_ld.json').unlink()
    path=directory/'shared_ld_fit.json';value=read(path);value['method']='full_ld';write(path,value)
    with pytest.raises(control.IntegrityError,match='different task'):
        runner.run_task(output,0,fitter=retained)
    assert not (directory/'completion.json').exists()


def test_prediction_storage_failure_retains_finite_parameters(frozen,monkeypatch):
    output,roots,manifest=frozen
    def failed_storage(*args,**kwargs):raise OSError('Injected prediction storage error')
    monkeypatch.setattr(runner.np,'savez_compressed',failed_storage)
    result=runner.run_task(output,0,fitter=retained)
    assert all(r['failed'] and r['fit_available'] and not r['evaluation_available'] for r in result['rows'])
    assert read(output/'tasks/task_000000/shared_ld.json')['parameters'] is not None
    status=runner.aggregate(output,'prediction')
    assert status['planned_fits']==8 and status['recorded_fits']==3 and status['failed_recorded']==3
    summary=pd.read_csv(output/'summary_prediction.csv')
    assert summary.available.eq(0).all()
    assert summary.missing.sum()==5


def test_total_fit_failure_preserves_all_start_records(frozen):
    output,roots,manifest=frozen
    def failed(*args,**kwargs):raise base.FitFailure('Injected all-start failure',[{'failed':True,'start':i} for i in range(3)])
    result=runner.run_task(output,0,fitter=failed)
    assert all(r['failed'] and not r['fit_available'] for r in result['rows'])
    assert len(read(output/'tasks/task_000000/shared_ld.json')['starts'])==3
    assert runner.aggregate(output,'prediction')['evaluation_available']==0


@pytest.mark.parametrize('which',['snapshot','configuration','input'])
def test_source_configuration_and_input_mutations_rejected(frozen,which):
    output,roots,manifest=frozen;runner.run_task(output,0,fitter=retained)
    if which=='snapshot':
        path=output/'source_snapshot/scripts/application_optimization_refinement.py';path.write_text(path.read_text()+'\n# changed\n')
    elif which=='configuration':
        path=output/'source_snapshot/configs/phase2_application_refinement.json';value=read(path);value['threads']=2;write(path,value)
    else:
        path=roots['primary']/'tasks/00000/shared_ld.json';value=read(path);value['changed']=1;write(path,value)
    with pytest.raises((RuntimeError,control.IntegrityError)):
        runner.run_task(output,0,fitter=retained)


@pytest.mark.parametrize('which',['LD_record','Gaussian_status'])
def test_original_completion_consistency_required(frozen,which):
    output,roots,manifest=frozen
    if which=='LD_record':
        path=roots['primary']/'tasks/00000/shared_ld.json';value=read(path)
        value['record']['projected_gradient']=123.;write(path,value)
    else:
        path=roots['gaussian']/'tasks/task_000000/result.json';value=read(path)
        value['status']='unfinished';write(path,value)
    with pytest.raises(control.IntegrityError,match='record differs|identity mismatch'):
        runner.run_task(output,0,fitter=retained)
