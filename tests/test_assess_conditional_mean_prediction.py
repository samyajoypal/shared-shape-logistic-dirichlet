"""Common-target loss, full-partition weights, saved identities and missingness."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import assess_conditional_mean_prediction as study
from test_phase3_applications import frozen,run as run_synthetic


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);study.old.write_json(path,value)


def test_squared_error_is_on_original_log_scale_and_ignores_target():
    y=np.array([[.4,2.],[1.,-3.],[-.5,.2]])
    center=np.array([1.,2.]);scale=np.array([2.,3.])
    p={'method':study.BASELINE,'d':2,'target':1,'coefficients':np.array([.2,.5]),'residual_sd':2.}
    masked=y.copy();masked[:,1]=np.nan
    prediction=study.panel.conditional_mean(p,study.BASELINE,masked,1,center,scale)
    expected=2+3*(.2+.5*((y[:,0]-1)/2))
    np.testing.assert_allclose(prediction,expected)
    loss=(y[:,1]-prediction)**2
    assert not np.allclose(loss,(np.exp(y[:,1])-np.exp(prediction))**2)
    # A physical-unit change translates log targets and training centers together.
    shifted=y+np.array([.6,4.]);newcenter=center+np.array([.6,4.]);shifted[:,1]=np.nan
    translated=study.panel.conditional_mean(p,study.BASELINE,shifted,1,newcenter,scale)
    np.testing.assert_allclose((y[:,1]+4-translated)**2,loss)


def fold_table(loss,groups,indices):
    records=[]
    for fold,test in enumerate(indices):
        for weighting,w in [('observation',study.panel.grid_weights(groups)),('equal_original_group',study.panel.grid_weights(groups,True))]:
            records.append({'dataset_id':'test','version':'test','comparison':'primary','kind':'cv','bounds':'primary','k':1,
                'method':'shared_ld','weighting':weighting,'repeat':0,'fold':fold,'expected_folds':len(indices),
                **study.loss_record(np.array(loss)[test],w[test],len(test))})
    return pd.DataFrame(records)


def test_group_weighted_partition_is_not_average_of_fold_group_means():
    loss=np.array([0.,1.,4.,25.,36.]);groups=np.array(['a','a','a','b','b'])
    frame=fold_table(loss,groups,[np.array([0,3]),np.array([1,2,4])])
    reps,summary=study.summarize_folds(frame)
    assert reps.loc[reps.weighting=='observation','loss'].iloc[0]==pytest.approx(loss.mean())
    expected=(loss[:3].mean()+loss[3:].mean())/2
    assert reps.loc[reps.weighting=='equal_original_group','loss'].iloc[0]==pytest.approx(expected)
    wrong=(np.mean([loss[0],loss[3]])+np.mean([loss[[1,2]].mean(),loss[4]]))/2
    assert not np.isclose(expected,wrong)


def test_missing_rows_preserve_denominators_and_block_complete_repetition():
    frame=fold_table([0.,np.nan,4.,9.],np.array(['a','a','b','b']),[np.array([0,2]),np.array([1,3])])
    reps,summary=study.summarize_folds(frame)
    assert reps.planned_rows.eq(4).all() and reps.available_rows.eq(3).all()
    assert reps.complete.eq(False).all() and reps.loss.isna().all()
    assert summary.complete_repetitions.eq(0).all()
    assert reps.available_row_loss.eq(13/3).all()


def test_pair_losses_use_intersection_of_the_same_rows():
    left=np.array([1.,np.nan,5.,8.]);right=np.array([2.,3.,np.nan,4.]);w=np.full(4,.25)
    row=study.loss_record(left-right,w,4)
    assert row['available_rows']==2 and row['planned_rows']==4
    assert row['available_row_loss']==pytest.approx(1.5) and row['loss'] is None


@pytest.mark.parametrize('change',['duplicate_fold','wrong_weight','missing_fold'])
def test_partition_summary_rejects_invalid_scope(change):
    f=fold_table([1.,2.,3.,4.],np.array(['a','a','b','b']),[np.array([0,2]),np.array([1,3])])
    if change=='duplicate_fold':f.loc[f.fold==1,'fold']=0
    elif change=='wrong_weight':f['planned_weight']*=.9
    else:f=f[f.fold==0]
    with pytest.raises(study.IntegrityError):study.summarize_folds(f)


def test_protocol_rejects_changed_receipt_or_text(tmp_path):
    protocol=tmp_path/'protocol.rst';protocol.write_bytes(study.PROTOCOL.read_bytes())
    receipt=tmp_path/'receipt.json';receipt.write_bytes(study.RECEIPT.read_bytes())
    study.verify_protocol(protocol,receipt)
    v=json.loads(receipt.read_text());v['historical_preregistration_receipt']=True;write(receipt,v)
    with pytest.raises(study.IntegrityError):study.verify_protocol(protocol,receipt)
    receipt.write_bytes(study.RECEIPT.read_bytes());protocol.write_text('changed')
    with pytest.raises(study.IntegrityError):study.verify_protocol(protocol,receipt)


def test_phase3_missing_methods_do_not_change_fixed_scope(frozen,tmp_path):
    run,manifest=frozen
    result=study.run_analysis(study.NewInputs(run),tmp_path/'missing')
    assert result['held_out_cases']==20 and result['planned_task_method_predictions']==64
    assert result['complete_task_method_predictions']==0
    summary=pd.read_csv(tmp_path/'missing/prediction_summary.csv')
    assert summary.complete_repetitions.eq(0).all()
    assert set(summary[summary.method==study.BASELINE].comparison)=={'primary'}
    assert set(summary[summary.method==study.GC].comparison)=={'primary','kminus1','kplus1'}


def test_phase3_saved_cv_and_sensitivities_use_same_means_without_fitting(frozen,tmp_path):
    run,manifest=frozen
    for task in manifest['tasks']:
        if task['kind']=='cv':run_synthetic(run,task['task_index'])
    output=tmp_path/'loss';result=study.run_analysis(study.NewInputs(run),output)
    assert result['complete_task_method_predictions']==result['planned_task_method_predictions']==64
    folds=pd.read_csv(output/'prediction_folds.csv');pairs=pd.read_csv(output/'paired_prediction_folds.csv')
    assert folds.complete.all() and pairs.complete.all()
    # This synthetic fitter returns the identical full/shared parameter point.
    matched=pairs[(pairs.left_method=='shared_ld')&(pairs.right_method=='full_ld')]
    assert matched.loss.abs().max()<1e-13
    for path in (output/'predictions').glob('*.npz'):
        a=np.load(path)
        for key in a.files:
            if key.startswith('prediction_'):
                method=key[len('prediction_'):]
                np.testing.assert_array_equal(a['loss_'+method],(a['observed_analysis_target']-a[key])**2)
    assert result['protocol_sha256']==study.PROTOCOL_SHA
    assert (output/'source_snapshot/scripts/assess_conditional_mean_prediction.py').exists()


@pytest.fixture
def old_panel(tmp_path,monkeypatch):
    monkeypatch.setitem(study.old.PLAN,'targets',{'olive3_oleic':'linoleic/oleic'})
    roots={name:tmp_path/name for name in ('primary','gaussian','transport','reference','refined')}
    for p in roots.values():p.mkdir()
    primary=roots['primary'];rng=np.random.default_rng(912)
    raw=np.exp(rng.normal(size=(12,3))*.3+[1.,2.,3.])
    frame=pd.DataFrame(raw,columns=['palmitic','linoleic','oleic']);frame['region']=np.repeat(['A','B'],6)
    path=primary/'executed_sources/data/oliveoil.csv';path.parent.mkdir(parents=True);frame.to_csv(path,index=False)
    variant={'id':'olive3_oleic','dataset':'oliveoil','columns':['palmitic','linoleic','oleic'],
        'reference':'oleic','conditional_target':'linoleic/oleic','k':1,'repeats':2,'primary':True}
    tasks=[]
    for repeat in range(2):
        for fold in range(2):
            test=np.arange(12)[(np.arange(12)+repeat)%2==fold].tolist()
            tasks.append({'task_index':len(tasks),'variant':variant['id'],'kind':'cv','bounds':'primary','repeat':repeat,
                'fold':fold,'test':test,'train':[i for i in range(12) if i not in test],'seed':321+len(tasks)})
    for fold,area in enumerate(['A','B']):
        test=np.flatnonzero(frame.region==area).tolist()
        tasks.append({'task_index':len(tasks),'variant':variant['id'],'kind':'leave_area_out','bounds':'primary',
            'repeat':0,'fold':fold,'area':area,'test':test,'train':[i for i in range(12) if i not in test],'seed':321+len(tasks)})
    spec={'config':{'variants':[variant],'folds':2,'bootstrap_repetitions':199},'tasks':tasks,
        'data_hashes':{'data/oliveoil.csv':study.old.digest(path)}}
    write(primary/'specification.json',spec);ph=study.old.digest(primary/'specification.json')
    mappings={}
    for kind in ('gaussian','transport'):
        chosen=[t for t in tasks if (t['kind']=='cv')==(kind=='gaussian')][::-1]
        mapping=[{'task_index':i,'primary_task_index':t['task_index'],'variant':t['variant'],'kind':t['kind'],
            'repeat':t['repeat'],'fold':t['fold']} for i,t in enumerate(chosen)]
        mappings[kind]={t['primary_task_index']:t for t in mapping}
        write(roots[kind]/'manifest.json',{'source_kind':'applications','primary_metadata_sha256':ph,'tasks':mapping})
    write(roots['reference']/'manifest.json',{'primary_specification_sha256':ph,'tasks':tasks})
    refined_tasks=[{'task_index':i,'primary_task_index':t['task_index'],'variant':t['variant'],'kind':t['kind'],
        'repeat':t['repeat'],'fold':t['fold'],'bounds':'primary','methods':list(study.METHODS[:3])} for i,t in enumerate(tasks[::-1])]
    write(roots['refined']/'manifest.json',{'tasks':refined_tasks,'sources_sha256':{},
        'metadata_sha256':{k:study.old.digest(roots[k]/('specification.json' if k=='primary' else 'manifest.json'))
                           for k in ('primary','gaussian','transport')}})
    y=np.log(raw[:,:2]/raw[:,2,None])
    for task in tasks:
        i=task['task_index'];folder=primary/'tasks'/f'{i:05d}';folder.mkdir(parents=True)
        center=y[task['train']].mean(0);scale=y[task['train']].std(0);transform={'center':center,'scale':scale}
        write(folder/'split.json',{**task,**transform});records=[];parameters={}
        for method in study.METHODS[:2]:
            p={'method':method,'k':1,'d':2,'theta':[1.,2.,3.,1.1,.9,.1,-.2]};parameters[method]=p
            row={'task_index':i,'variant':variant['id'],'method':method,'failed':False,'score_converged':True,'boundary_hit':False}
            records.append(row);write(folder/(method+'.json'),{'parameters':p,'record':row,'specification_sha256':ph})
        write(folder/'completion.json',{'task':task,'specification_sha256':ph,'records':records})
        kind='gaussian' if task['kind']=='cv' else 'transport';mapped=mappings[kind][i];directory=roots[kind]/'tasks'/f"task_{mapped['task_index']:06d}"
        binding={'primary_task_index':i,'input_sha256':{f'tasks/{i:05d}/split.json':study.old.digest(folder/'split.json')}}
        write(directory/'input_binding.json',binding)
        p={'method':study.GC,'k':1,'d':2,'theta':[.2,.3,0.,.1,.4]};parameters[study.GC]=p
        write(directory/'result.json',{'status':'complete','task':mapped,'manifest_sha256':study.old.digest(roots[kind]/'manifest.json'),
            'input_binding_sha256':study.old.semantic_hash(binding),'parameters':p,
            'row':{**mapped,'method':study.GC,'failed':False,'score_converged':True,'boundary_hit':False}})
        directory=roots['reference']/'tasks'/f'task_{i:06d}';directory.mkdir(parents=True)
        p={'method':study.BASELINE,'d':2,'target':1,'coefficients':[.1,.3],'residual_sd':.8}
        identity={'task':task,'manifest_sha256':study.old.digest(roots['reference']/'manifest.json'),
            'split_sha256':study.old.digest(folder/'split.json'),'completion_sha256':study.old.digest(folder/'completion.json')}
        write(directory/'fit.json',{**identity,'parameters':p,'record':{'score_converged':True},'training_transform':transform})
        write(directory/'result.json',{**identity,'status':'complete','parameters':p,'record':{'score_converged':True},'training_transform':transform,
            'row':{**task,'method':study.BASELINE,'evaluation_available':False,'failed':False,'score_converged':True}})
        mapped=next(t for t in refined_tasks if t['primary_task_index']==i);directory=roots['refined']/'tasks'/f"task_{mapped['task_index']:06d}"
        binding={'task':mapped,'files':[{'source':'primary','relative_path':f'tasks/{i:05d}/split.json','sha256':study.old.digest(folder/'split.json')}]}
        write(directory/'input_binding.json',binding);identity={'task':mapped,'manifest_sha256':study.old.digest(roots['refined']/'manifest.json'),
            'input_binding_sha256':study.old.semantic_hash(binding)};rows=[]
        for method in study.METHODS[:3]:
            row={**mapped,'method':method,'failed':False,'score_converged':True,'boundary_hit':False};rows.append(row)
            selected={**identity,'method':method,'parameters':parameters[method],'record':{'score_converged':True}}
            write(directory/(method+'_fit.json'),selected)
            write(directory/(method+'.json'),{**selected,'row':row,'training_transform':transform})
        write(directory/'completion.json',{**identity,'status':'complete','rows':rows})
    return roots


def old_inputs(roots,refined=False):
    return study.OldInputs(roots['primary'],roots['gaussian'],roots['transport'],roots['reference'],roots['refined'] if refined else None)


def test_old_original_refined_cv_and_area_loss_use_saved_mapping_and_same_baseline(old_panel,tmp_path):
    result=study.run_analysis([old_inputs(old_panel),old_inputs(old_panel,True)],tmp_path/'old_loss')
    assert result['held_out_cases']==12 and result['complete_task_method_predictions']==48
    folds=pd.read_csv(tmp_path/'old_loss/prediction_folds.csv')
    original=folds[folds.version=='original'];refined=folds[folds.version=='refined']
    joined=original.merge(refined,on=['task_index','method','weighting'],suffixes=('_o','_r'),validate='one_to_one')
    np.testing.assert_array_equal(joined.loss_o,joined.loss_r)
    summaries=pd.read_csv(tmp_path/'old_loss/prediction_summary.csv')
    assert set(summaries.kind)=={'cv','leave_area_out'} and summaries.complete_repetitions.min()==1


@pytest.mark.parametrize('mutation',['refined_selected','reference_transform','transport_split'])
def test_old_parameter_identity_mutation_is_rejected(old_panel,tmp_path,mutation):
    if mutation=='refined_selected':
        p=old_panel['refined']/'tasks/task_000000/shared_ld_fit.json';v=json.loads(p.read_text());v['parameters']['theta'][0]+=.2;write(p,v)
    elif mutation=='reference_transform':
        p=old_panel['reference']/'tasks/task_000000/result.json';v=json.loads(p.read_text());v['training_transform']['center'][0]+=.1;write(p,v)
    else:
        p=old_panel['transport']/'tasks/task_000000/input_binding.json';v=json.loads(p.read_text());v['input_sha256']={};write(p,v)
    with pytest.raises(study.IntegrityError):study.run_analysis([old_inputs(old_panel),old_inputs(old_panel,True)],tmp_path/'reject')


def test_missing_old_parameter_point_keeps_fold_and_pair_denominators(old_panel,tmp_path):
    (old_panel['primary']/'tasks/00000/shared_ld.json').unlink()
    result=study.run_analysis(old_inputs(old_panel),tmp_path/'missing_old')
    assert result['planned_task_method_predictions']==24 and result['complete_task_method_predictions']==23
    summaries=pd.read_csv(tmp_path/'missing_old/prediction_summary.csv')
    row=summaries[(summaries.kind=='cv')&(summaries.method=='shared_ld')]
    assert row.planned_repetitions.eq(2).all() and row.complete_repetitions.eq(1).all()
