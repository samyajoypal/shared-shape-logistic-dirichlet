"""Recreate numerical tables from complete, hash-verified saved summaries.

No fitting, data download or document generation. The complete saved tables
remain available; these display views never replace failure denominators.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import complete_saved_reporting
ROOT=Path(__file__).resolve().parents[1]
CORE=['shared_ld','full_ld','gaussian_shared_correlation']
METHODS=CORE+['conditional_log_linear']
CONFIG_INPUTS={'configs/phase2_applications.json','configs/phase3_applications.json'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(name):return pd.read_csv(ROOT/'saved_summaries'/name)
def one(frame,**keys):
    for k,v in keys.items():frame=frame[frame[k]==v]
    if len(frame)!=1:raise ValueError(('Expected one fixed row',keys,len(frame)))
    return frame.iloc[0]
def final_procedures(f):
    return f[(f.method.isin(CORE)&f.version.eq('refined'))|(~f.method.isin(CORE)&f.version.eq('original'))].copy()

def build(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    entries=json.loads((ROOT/'export_manifest.json').read_text())['entries'];used={}
    for e in entries:
        if e['file'].startswith('saved_summaries/') or e['file'] in CONFIG_INPUTS:
            if sha(ROOT/e['file'])!=e['export_sha256']:raise ValueError('Changed summary: '+e['file'])
            used[e['file']]=e['export_sha256']
    t={};core=read('simulations/primary_common_shape_core.csv')
    g=read('simulations/gaussian_simulations__paired_comparisons.csv');g=g[g.baseline_method.eq('shared_ld')]
    t['table_01_core_simulations']=core.merge(g,on='cell_id',validate='one_to_one',suffixes=('','_gaussian'))
    density=final_procedures(read('original_density/primary_cv_summary.csv'))
    linear=[]
    for name,rows in read('original_reference/cv_repetitions.csv').groupby('variant'):
        if len(rows)!=10 or not rows.complete.all():raise ValueError('Incomplete conditional reference')
        linear.append({'variant':name,'bounds':'primary','method':'conditional_log_linear','version':'original',
            'planned_repetitions':10,'complete_repetitions':10,'joint':np.nan,'conditional':rows.conditional_log_density.mean()})
    t['table_02_original_density']=pd.concat([density,pd.DataFrame(linear)],ignore_index=True)
    means=read('original_mean/prediction_summary.csv')
    t['table_03_original_mean_prediction']=means[means.version.eq('refined')&means.comparison.eq('primary')&means.kind.eq('cv')&means.weighting.eq('observation')]
    areas=final_procedures(read('original_density/area_summary.csv'));areas=areas[areas.method.isin(CORE)]
    ref=read('original_reference/fit_records.csv');rows=[]
    for name,part in ref[ref.kind.eq('leave_area_out')].groupby('variant'):
        if len(part)!=9 or not part.evaluation_available.all():raise ValueError('Incomplete reference areas')
        rows.append({'variant':name,'bounds':'primary','method':'conditional_log_linear','version':'original',
            'planned_areas':9,'available_areas':9,'complete':True,'n_test_total':part.n_test.sum(),
            'joint_weighted':np.nan,'conditional_weighted':np.average(part.mean_conditional_log_density,weights=part.n_test),
            'joint_equal_area':np.nan,'conditional_equal_area':part.mean_conditional_log_density.mean()})
    areas=pd.concat([areas,pd.DataFrame(rows)],ignore_index=True)
    am=means[means.kind.eq('leave_area_out')&means.weighting.eq('observation')&means.version.eq('refined')]
    t['table_04_geographic_transfer']=areas.merge(am[['dataset_id','method','loss']],left_on=['variant','method'],right_on=['dataset_id','method'],how='left',validate='one_to_one')
    d=read('original_density/primary_diagnostics.csv');t['table_05_original_numerical_diagnostics']=d[d.method.isin(CORE)&d.kind.eq('cv')]
    t['table_06_original_stability']=read('original_stability/shared_variance_ratios.csv')
    m=read('additional_mean/prediction_summary.csv');st=read('additional_stability/paired_stability.csv');rows=[]
    for spec in json.loads((ROOT/'configs/phase3_applications.json').read_text())['datasets']:
        name=spec['id'];d=len(spec['columns'])-(spec['transform']=='alr');k=spec['k']
        r={'dataset_id':name,'parameters_shared':k-1+2*k*d+d+1,'parameters_separate':k-1+2*k*d+k*(d+1)}
        for method in METHODS:
            entry=one(m,dataset_id=name,comparison='primary',method=method,weighting='observation')
            if entry.complete_repetitions!=10:raise ValueError('Incomplete primary repetitions')
            r[method+'_mean_prediction_loss']=entry.loss
        pair=one(st,dataset_id=name,left_method='shared_ld',right_method='full_ld',weighting='equal_observations',subset='all_finite')
        r.update({key:pair[key] for key in ['variance_ratio','ratio_mc95_low','ratio_mc95_high','bootstrap_requested','both_available']});rows.append(r)
    t['table_07_additional_mean_and_stability']=pd.DataFrame(rows)
    d=read('additional_density/all_planned_paired_comparisons.csv')
    t['table_08_additional_density']=d[d.comparison.eq('primary')&d.weighting.eq('observation')&((d.density.eq('joint')&d.comparator.eq('full_ld'))|(d.density.eq('conditional')&d.comparator.isin(['full_ld','gaussian_shared_correlation','fmvmm_t','conditional_log_linear'])))]
    # Focused application display; the complete comparison views above remain unchanged.
    oil_spec=next(s for s in json.loads((ROOT/'configs/phase2_applications.json').read_text())['variants'] if s['id']=='olive6_oleic')
    dimension=len(oil_spec['columns'])-1;k=oil_spec['k']
    oil={'dataset_id':'olive6_oleic','label':'Six-acid olive oil',
         'parameters_shared':k-1+2*k*dimension+dimension+1,
         'parameters_separate':k-1+2*k*dimension+k*(dimension+1)}
    for method in METHODS:
        entry=one(t['table_03_original_mean_prediction'],dataset_id='olive6_oleic',method=method)
        if entry.planned_repetitions!=10 or entry.complete_repetitions!=10 or entry.k!=k:
            raise ValueError('Incomplete or inconsistent focused olive-oil repetitions')
        oil[method+'_mean_prediction_loss']=entry.loss
    pair=one(t['table_06_original_stability'],variant='olive6_oleic',left_method='shared_ld',right_method='full_ld',subset='all_finite')
    if pair.bootstrap_requested!=199 or pair.both_available!=199:
        raise ValueError('Incomplete focused olive-oil bootstrap pairs')
    oil['variance_ratio']=pair.variance_ratio
    abalone=one(t['table_07_additional_mean_and_stability'],dataset_id='abalone_log4')
    if abalone.bootstrap_requested!=199 or abalone.both_available!=199:
        raise ValueError('Incomplete focused Abalone bootstrap pairs')
    abalone={key:abalone[key] for key in oil if key!='label'}
    abalone['label']='Abalone'
    t['focused_table_02_applications']=pd.DataFrame([oil,abalone],columns=list(oil))
    complete=complete_saved_reporting.build(ROOT,output)
    used.update(complete['input_sha256'])
    outputs=dict(complete['output_sha256'])
    for name,frame in t.items():
        path=output/(name+'.csv');frame.to_csv(path,index=False);outputs[path.name]=sha(path)
    for name,rel in [('all_interval_results','intervals/inference_coverage_derived.csv'),('all_conditional_risks','conditional_risk/all_cell_paired_risks.csv'),('all_additional_sensitivities','additional_density/matched_sensitivity_changes.csv')]:
        path=output/(name+'.csv');path.write_bytes((ROOT/'saved_summaries'/rel).read_bytes());outputs[path.name]=sha(path)
    plan={'input_sha256':used,'output_sha256':outputs,'source_sha256':sha(__file__),
          'focused_display_files':['table_01_core_simulations.csv','focused_table_02_applications.csv'],
          'extended_comparison_files':[name+'.csv' for name in t if name not in {'table_01_core_simulations','focused_table_02_applications'}],
          'complete_report_files':list(complete['output_sha256']),
          'new_fits':False,'data_downloads':False,'summary_reproduction_only':True}
    (output/'reproduction_manifest.json').write_text(json.dumps(plan,indent=2)+'\n');return plan

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'artifacts/saved_tables');a=p.parse_args()
    r=build(a.output);print(json.dumps({'tables':len(r['output_sha256']),'new_fits':False}))
if __name__=='__main__':main()
