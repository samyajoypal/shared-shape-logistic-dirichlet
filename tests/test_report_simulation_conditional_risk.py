"""Independent risk tables retain all cells, matching and uncertainty scope."""
import copy
import itertools
import json

import numpy as np
import pandas as pd
import pytest

import report_simulation_conditional_risk as report
import simulation_conditional_mean_risk as frozen


def example():
    cells=[{'id':'core','block':'core','dimension':2,'components':2,'n_train':250,'heterogeneity':0.,'repetitions':4},
           {'id':'local','block':'local_alternative','dimension':2,'components':2,'n_train':800,'heterogeneity':.035355,'local_h_coefficient':1.,'repetitions':4}]
    config={'cells':cells,'methods':list(report.METHODS),'subsets':list(report.SUBSETS),'target_index':0,'n_test':5000}
    tasks=[];rows=[]
    for c in cells:
        for replicate in range(4):
            task={'task_index':len(tasks),'primary_task_index':100+len(tasks),'cell_id':c['id'],'replicate':replicate};tasks.append(task)
            for j,method in enumerate(report.METHODS):
                value=float((replicate+1)*(j+1)+(.2 if c['block']=='local_alternative' else 0.))
                rows.append({**task,'method':method,'recorded':True,'available':True,'risk':value,'test_integration_mcse':.01,
                    'source_failed':False,'score_converged':not(j==2 and replicate==0),'boundary_hit':j==1})
    # One missing shared risk, with every planned slot kept.
    rows[0].update(recorded=False,available=False,risk=np.nan,test_integration_mcse=np.nan,
                   source_failed=None,score_converged=None,boundary_hit=None)
    frame=pd.DataFrame(rows);summaries=[];pairs=[]
    for c in cells:
        b=frame[frame.cell_id==c['id']];common=set(b.replicate)
        for m in report.METHODS:common&=set(b.loc[(b.method==m)&b.available&b.score_converged.eq(True),'replicate'])
        for subset in report.SUBSETS:
            s=b if subset==report.SUBSETS[0] else b[b.replicate.isin(common)]
            for m in report.METHODS:
                x=s.loc[(s.method==m)&s.available,'risk'].to_numpy(float)
                summaries.append({'cell_id':c['id'],'method':m,'subset':subset,'planned':4,
                    'recorded':int(b.loc[b.method==m,'recorded'].sum()),'available':len(x),'mean_risk':x.mean(),
                    'risk_mcse':x.std(ddof=1)/np.sqrt(len(x))})
            for l,r in itertools.combinations(report.METHODS,2):
                a=s[(s.method==l)&s.available].merge(s[(s.method==r)&s.available],on='replicate',suffixes=('_left','_right'))
                pairs.append({'cell_id':c['id'],'left_method':l,'right_method':r,'subset':subset,'planned':4,
                              **frozen.paired_summary(a.risk_left,a.risk_right)})
    return {'config':config,'tasks':tasks},frame,pd.DataFrame(summaries),pd.DataFrame(pairs)


def test_all_cells_and_subsets_retain_denominators_and_paired_means():
    m,records,s,p=example();methods,pairs=report.build_tables(m,records,s,p)
    assert len(methods)==12 and len(pairs)==12 and set(methods.block)=={'core','local_alternative'}
    shared=methods[(methods.cell_id=='core')&(methods.subset=='all_finite')&(methods.method=='shared_ld')].iloc[0]
    assert shared.available==3 and shared.planned==4 and shared.missing_task_records==1 and shared.score_unknown==1
    pair=pairs[(pairs.cell_id=='core')&(pairs.subset=='all_finite')&(pairs.left_method=='shared_ld')&(pairs.right_method=='full_ld')].iloc[0]
    assert pair.mean_left==3 and pair.mean_right==6 and pair.risk_ratio==.5 and pair.paired_available==3
    unpaired=s[(s.cell_id=='core')&(s.subset=='all_finite')&(s.method=='full_ld')].mean_risk.iloc[0]
    assert unpaired==5 and pair.risk_ratio!=shared.mean_risk/unpaired
    # All-four score labels are not silently changed: this simulation has three methods.
    assert pairs[pairs.subset=='all_methods_score_eligible'].paired_available.eq(3).all()


def test_ratio_delta_se_and_difference_se_match_closed_form():
    x=np.array([1.,2.,4.,7.]);y=np.array([3.,4.,8.,12.]);out=report.paired_statistics(x,y)
    covariance=np.cov(np.stack([x,y]),ddof=1)/4
    gradient=np.array([1/y.mean(),-x.mean()/y.mean()**2])
    assert out['ratio_mcse']==pytest.approx(np.sqrt(gradient@covariance@gradient),rel=1e-13)
    assert out['difference_mcse']==pytest.approx(np.std(x-y,ddof=1)/2)
    assert report.paired_statistics(x,2*x)['ratio_mcse']<1e-15


def test_all_missing_results_remain_explicit_in_every_cell_and_pair():
    m,r,s,p=example();r['recorded']=False;r['available']=False;r['risk']=np.nan;r['test_integration_mcse']=np.nan
    for key in ('source_failed','score_converged','boundary_hit'):r[key]=None
    s['recorded']=0;s['available']=0;s['mean_risk']=np.nan;s['risk_mcse']=np.nan
    p['paired_available']=0
    for key in ('mean_left','mean_right','difference','difference_mcse','risk_ratio','ratio_mcse'):p[key]=np.nan
    methods,pairs=report.build_tables(m,r,s,p)
    assert methods.available.eq(0).all() and methods.missing_task_records.eq(4).all()
    assert methods.score_unknown.eq(4).all() and methods.score_false.eq(0).all()
    assert pairs.paired_available.eq(0).all() and pairs.paired_unavailable_or_ineligible.eq(4).all()
    assert pairs.risk_ratio.isna().all()


@pytest.mark.parametrize('change',['row_missing','row_duplicate','wrong_rep','summary_missing','pair_missing','wrong_ratio','wrong_mcse'])
def test_changed_scope_or_statistics_are_rejected(change):
    m,r,s,p=example()
    if change=='row_missing':r=r.iloc[1:]
    elif change=='row_duplicate':r=pd.concat([r,r.iloc[[0]]])
    elif change=='wrong_rep':r.loc[0,'replicate']=9
    elif change=='summary_missing':s=s.iloc[1:]
    elif change=='pair_missing':p=p.iloc[1:]
    elif change=='wrong_ratio':p.loc[0,'risk_ratio']+=.1
    else:s.loc[0,'risk_mcse']+=.1
    with pytest.raises(report.IntegrityError):report.build_tables(m,r,s,p)


@pytest.mark.parametrize('change',['unknown_recorded','available_unrecorded','unrecorded_score','unrecorded_integration','duplicate_replicate','noncanonical_task'])
def test_recorded_and_planned_identity_invariants_are_explicit(change):
    m,r,s,p=example()
    if change=='unknown_recorded':
        r['recorded']=r.recorded.astype(object);r.loc[1,'recorded']=None
    elif change=='available_unrecorded':r.loc[1,'recorded']=False
    elif change=='unrecorded_score':r.loc[0,'score_converged']=False
    elif change=='unrecorded_integration':r.loc[0,'test_integration_mcse']=.1
    elif change=='duplicate_replicate':m['tasks'][1]['replicate']=0
    else:m['tasks'][0]['task_index']=1000
    with pytest.raises(report.IntegrityError):report.build_tables(m,r,s,p)


@pytest.mark.parametrize('change',['duplicate_cell','unknown_cell_task'])
def test_every_planned_task_belongs_to_one_unique_reported_cell(change):
    m,r,s,p=example()
    if change=='duplicate_cell':m['config']['cells'].append(copy.deepcopy(m['config']['cells'][0]))
    else:
        task={'task_index':len(m['tasks']),'primary_task_index':999,'cell_id':'undeclared','replicate':0};m['tasks'].append(task)
        extra=[]
        for method in report.METHODS:
            row=copy.deepcopy(r[r.method==method].iloc[1].to_dict());row.update(task);extra.append(row)
        r=pd.concat([r,pd.DataFrame(extra)],ignore_index=True)
    with pytest.raises(report.IntegrityError,match='configured cells'):report.build_tables(m,r,s,p)


def test_frozen_summary_is_rehashed_and_output_cannot_enter_input_run(tmp_path):
    m,r,s,p=example();run=tmp_path/'frozen';run.mkdir()
    frozen.atomic_json(run/'manifest.json',m)
    for name,table in [('method_records',r),('risk_summary',s),('paired_risk_summary',p)]:table.to_csv(run/(name+'.csv'),index=False)
    aggregate={'manifest_sha256':report.digest(run/'manifest.json'),'planned_predictions':len(r),
        'recorded_predictions':int(r.recorded.sum()),'available_predictions':int(r.available.sum()),
        'outputs_sha256':{name:report.digest(run/name) for name in ['method_records.csv','risk_summary.csv','paired_risk_summary.csv']}}
    frozen.atomic_json(run/'aggregation.json',aggregate)
    with pytest.raises(ValueError):report.run(run,run/'presentation')
    receipt=report.run(run,tmp_path/'presentation')
    assert receipt['all_cells']==2 and not receipt['risk_predictions_recomputed']
    table=pd.read_csv(tmp_path/'presentation/shared_shape_comparisons.csv')
    assert len(table)==8 and set(table.subset)==set(report.SUBSETS)
    path=run/'risk_summary.csv';path.write_text(path.read_text()+'\n')
    with pytest.raises(report.IntegrityError):report.run(run,tmp_path/'changed')
