"""Prepare the fixed Hydrochem/GEMAS/Abalone panel from audited public bytes.

Offline by default. Existing raw bytes, then the audited source directories,
are preferred. --download only fetches a missing source and checks its frozen
SHA256 before use. No model fitting, logs, closure or standardization occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import rdata

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT=ROOT/'data/extended_applications'
CHEMICAL_AUDIT=DEFAULT_OUTPUT/'raw'
MEASUREMENT_AUDIT=DEFAULT_OUTPUT/'raw'
SOURCES={
    'Hydrochem.txt':{
        'url':'https://raw.githubusercontent.com/cran/compositions/master/data/Hydrochem.txt',
        'sha256':'8bd940b6ab9d7b7aac09b1189e865f6371260b41afe0342dfd0ecdbe17419b6b',
        'audit':'chemical'},
    'Hydrochem.Rd':{
        'url':'https://raw.githubusercontent.com/cran/compositions/master/man/Hydrochem.Rd',
        'sha256':'ef9e100f644cfeedc889940bb6b36049c0220cf191769af5b59778c77973b7a5',
        'audit':'chemical'},
    'gemas.rda':{
        'url':'https://raw.githubusercontent.com/cran/robCompositions/master/data/gemas.rda',
        'sha256':'241722063f95ce617fe403177ad00f3743519d9056ca4420c99fb288bf605454',
        'audit':'chemical'},
    'gemas.Rd':{
        'url':'https://raw.githubusercontent.com/cran/robCompositions/master/man/gemas.Rd',
        'sha256':'66dd7c3ff8c9f98ba0819befa2b6f86dc3ba3d14edb189f853cae51b21f19012',
        'audit':'chemical'},
    'abalone.zip':{
        'url':'https://archive.ics.uci.edu/static/public/1/abalone.zip',
        'sha256':'755a6a67c5b266961a3f149ea13be2cfb6e6c727e48cee3c83bc0b4526210ee4',
        'audit':'measurement'},
}
ZIP_MEMBERS={
    'abalone.data':'de37cdcdcaaa50c309d514f248f7c2302a5f1f88c168905eba23fe2fbc78449f',
    'abalone.names':'542eee5a2a0ee72db563de010dc3b5055b03c5dc624049ae1f2208fc66a14aac',
    'Index':'e0b1694a7e3eccef2304db5d59287c6ad3080a68ba1f60f9fa341e32b41d1fc9',
}
FEATURES={
    'hydrochem':['Na','K','Mg','Ca'],
    'gemas':['Ca','Fe','K','Mg','Si','Al'],
    'abalone':['Length','Diameter','Height','ShuckedWeight'],
}
DATASET_IDS={'hydrochem':'hydrochem_cations','gemas':'gemas_major6','abalone':'abalone_log4'}
ABALONE_COLUMNS=['Sex','Length','Diameter','Height','WholeWeight',
                 'ShuckedWeight','VisceraWeight','ShellWeight','Rings']
ABALONE_EXCLUSIONS=(1258,3997)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def checked_bytes(data,expected,name):
    if sha256(data)!=expected:
        raise ValueError('Audited source hash mismatch: '+name)
    return data


def acquire_sources(raw_dir,chemical_dir=CHEMICAL_AUDIT,measurement_dir=MEASUREMENT_AUDIT,download=False):
    """Validate every needed source before writing any new raw file."""
    raw_dir=Path(raw_dir)
    audits={'chemical':Path(chemical_dir),'measurement':Path(measurement_dir)}
    payload={}
    for name,source in SOURCES.items():
        available=next((p for p in [raw_dir/name,audits[source['audit']]/name] if p.exists()),None)
        if available is not None:
            data=available.read_bytes()
        elif download:
            with urllib.request.urlopen(source['url'],timeout=60) as response:
                data=response.read()
        else:
            raise FileNotFoundError('Missing audited raw source '+name+'; supply its offline directory or explicitly use --download')
        payload[name]=checked_bytes(data,source['sha256'],name)
    with zipfile.ZipFile(io.BytesIO(payload['abalone.zip'])) as archive:
        for name,expected in ZIP_MEMBERS.items():
            data=checked_bytes(archive.read(name),expected,'abalone.zip/'+name)
            if (raw_dir/name).exists():
                checked_bytes((raw_dir/name).read_bytes(),expected,name)
            # Preserve and cross-check the previously audited extracted data when supplied.
            audited=audits['measurement']/name
            if audited.exists(): checked_bytes(audited.read_bytes(),expected,str(audited))
            payload[name]=data
    raw_dir.mkdir(parents=True,exist_ok=True)
    for name,data in payload.items():
        target=raw_dir/name
        if not target.exists(): target.write_bytes(data)
    return payload


def validate_positive(frame,features,expected_n):
    if len(frame)!=expected_n or frame.row_id.duplicated().any():
        raise ValueError('Unexpected sample count or duplicate original row IDs')
    values=frame[features].to_numpy(dtype=float)
    if not np.isfinite(values).all() or not (values>0).all():
        raise ValueError('Selected features must remain finite and strictly positive')
    if frame.group.isna().any() or frame.label.isna().any():
        raise ValueError('Groups/labels must use explicit strings, including preserved blanks')


def raw_to_frames(raw_dir):
    raw_dir=Path(raw_dir)
    hydro=pd.read_csv(raw_dir/'Hydrochem.txt',sep=r'\s+',
                     converters={key:float for key in FEATURES['hydrochem']})
    if hydro.shape!=(485,19) or list(hydro.index)!=list(range(1,486)):
        raise ValueError('Hydrochem source row identity/schema changed')
    h=hydro.reset_index(drop=True)
    hydro_out=pd.DataFrame({'row_id':np.arange(1,486),'group':h.Site.astype(str),'label':h.River.astype(str)})
    for feature in FEATURES['hydrochem']: hydro_out[feature]=h[feature]
    hydro_rows=pd.concat([hydro_out[['row_id']],h[['Code','Site','Location','River','Date']]],axis=1)

    g=rdata.read_rda(raw_dir/'gemas.rda')['gemas'].reset_index(drop=True)
    if g.shape!=(2108,30): raise ValueError('GEMAS source schema changed')
    gemas_out=pd.DataFrame({'row_id':np.arange(1,2109),'group':g.COUNTRY.astype(str),
                            'label':g.soilclass.astype(str)})
    for feature in FEATURES['gemas']: gemas_out[feature]=g[feature].to_numpy()
    gemas_rows=pd.concat([gemas_out[['row_id']],g[['COUNTRY','longitude','latitude','Xcoord','Ycoord']]],axis=1)
    gemas_rows['soilclass']=gemas_out.label
    if int(gemas_out.label.eq('').sum())!=5:
        raise ValueError('GEMAS blank soilclass values were not preserved')

    a=pd.read_csv(raw_dir/'abalone.data',header=None,names=ABALONE_COLUMNS,
                  keep_default_na=False,converters={key:float for key in ABALONE_COLUMNS[1:-1]})
    a.insert(0,'row_id',np.arange(1,len(a)+1))
    if len(a)!=4177 or a.isna().any().any(): raise ValueError('Abalone source schema changed')
    nonpositive=a.loc[(a[FEATURES['abalone']]<=0).any(axis=1),'row_id'].tolist()
    if nonpositive!=list(ABALONE_EXCLUSIONS) or not a.loc[a.row_id.isin(ABALONE_EXCLUSIONS),'Height'].eq(0).all():
        raise ValueError('The only authorized Abalone exclusions must be the two original zero-height rows')
    excluded=a[a.row_id.isin(ABALONE_EXCLUSIONS)].copy()
    retained=a[~a.row_id.isin(ABALONE_EXCLUSIONS)].reset_index(drop=True)
    abalone_out=retained[['row_id']].copy()
    abalone_out['group']=retained.row_id.astype(str)
    abalone_out['label']=retained.Sex.astype(str)
    for feature in FEATURES['abalone']: abalone_out[feature]=retained[feature]
    abalone_rows=retained[['row_id','Sex','Rings']].copy()
    exclusions=[]
    source_lines=(raw_dir/'abalone.data').read_text().splitlines()
    for record in excluded.to_dict('records'):
        exclusions.append({'row_id':record['row_id'],'reason':'Height is exactly zero in the published source; log-height undefined',
            'original_values':record,'original_line':source_lines[record['row_id']-1]})
    frames={'hydrochem':hydro_out,'gemas':gemas_out,'abalone':abalone_out}
    row_metadata={'hydrochem':hydro_rows,'gemas':gemas_rows,'abalone':abalone_rows}
    for key,n in [('hydrochem',485),('gemas',2108),('abalone',4175)]:
        validate_positive(frames[key],FEATURES[key],n)
        if list(frames[key].columns)!=['row_id','group','label']+FEATURES[key]:
            raise ValueError('Processed feature order changed')
    return frames,row_metadata,exclusions


def count_strings(series):
    return {str(k):int(v) for k,v in series.value_counts(dropna=False).sort_index().items()}


def dataset_metadata(key,frame,exclusions):
    specification={
        'hydrochem':{
            'source_files':['raw/Hydrochem.txt','raw/Hydrochem.Rd'],
            'source_documentation':'https://cran.rstudio.com/web/packages/compositions/refman/compositions.html',
            'source_article':'https://doi.org/10.1016/j.watres.2005.01.012',
            'source_rows':485,'reference':'Ca','target':'Na/Ca','target_feature':'Na',
            'transform_plan':'alr','analysis_coordinate_order':['Na/Ca','K/Ca','Mg/Ca'],
            'group_source':'Site','label_source':'River',
            'group_meaning':'Sampling station; keep all dates of a Site together during grouped validation/resampling',
            'label_meaning':'Recorded river setting; not a fitted likelihood label',
            'units':{x:'As-recorded cation mass-concentration units; common absolute unit not independently established' for x in FEATURES['hydrochem']},
            'unit_notes':'Use as-recorded mass ratios. The authors describe these major-ion entries as mass proportions before optional molar conversion. No mg/L or mol/L conversion has been applied.',
            'data_limits':['30 recorded Sites despite package prose mentioning 31','Repeated dates within sites; source rows are not 485 independent stations',
                           'Distributed positive cation values carry no individual censoring flags'],
            'attribution':'Otero et al. (2005); R package compositions, Hydrochem; data help states GPL version 2 or newer'},
        'gemas':{
            'source_files':['raw/gemas.rda','raw/gemas.Rd'],
            'source_documentation':'https://search.r-project.org/CRAN/refmans/robCompositions/html/gemas.html',
            'source_project':'https://gemas.eurogeosurveys.org/',
            'source_rows':2108,'reference':'Al','target':'Ca/Al','target_feature':'Ca',
            'transform_plan':'alr','analysis_coordinate_order':['Ca/Al','Fe/Al','K/Al','Mg/Al','Si/Al'],
            'group_source':'COUNTRY','label_source':'soilclass',
            'group_meaning':'Source country code, retained literally; geographic grouping does not remove all spatial dependence',
            'label_meaning':'Source soil-class code, with all five blank strings retained; not a three-class truth',
            'units':{x:'Elemental concentration in mg/kg, as distributed' for x in FEATURES['gemas']},
            'unit_notes':'All six proposed elemental minima exceed their published oxide-equivalent laboratory detection limits. No conversion or censoring substitution is performed here.',
            'data_limits':['No per-observation censoring flags in the package object','Country group sizes are unequal','No land-use indicator separates agricultural and grazing soil in this object'],
            'attribution':'EuroGeoSurveys GEMAS/Eurometaux; Reimann et al. (2014); R integration by Filzmoser and Templ in robCompositions'},
        'abalone':{
            'source_files':['raw/abalone.zip','raw/abalone.data','raw/abalone.names','raw/Index'],
            'source_documentation':'https://archive.ics.uci.edu/dataset/1/abalone',
            'source_doi':'https://doi.org/10.24432/C55C7W',
            'source_rows':4177,'reference':None,'target':'ShuckedWeight','target_feature':'ShuckedWeight',
            'transform_plan':'log','analysis_coordinate_order':FEATURES['abalone'],
            'group_source':'original one-based source row_id','label_source':'Sex',
            'group_meaning':'Unique retained record; no source site/batch/specimen grouping beyond record identity is available',
            'label_meaning':'Published M/F/I category; I means infant, not a third adult sex',
            'units':{x:('Published scaled length unit' if x!='ShuckedWeight' else 'Published scaled weight unit') for x in FEATURES['abalone']},
            'unit_notes':'The original names file says continuous values were divided by 200. These stored values are retained exactly; no back-conversion is applied and they are not labelled unscaled millimetres or grams.',
            'source_scaling_statement':{'stated_divisor':200,'applied_back_conversion':False,
                'original_measurement_units_in_source':{'Length':'mm','Diameter':'mm','Height':'mm','ShuckedWeight':'grams'}},
            'data_limits':['Original UCI source already removed earlier missing-value examples','Only original rows 1258 and 3997 are excluded here because Height=0',
                           'All other positive records, including unusual dimensions, remain; no residual/outcome filtering','Rings retained only in row metadata, not as a fitted coordinate'],
            'attribution':'Nash, Sellers, Talbot, Cawthorn and Ford; UCI Abalone, DOI 10.24432/C55C7W; UCI lists CC BY 4.0'},
    }[key]
    return {'id':DATASET_IDS[key],'csv':DATASET_IDS[key]+'.csv','row_metadata_csv':DATASET_IDS[key]+'_rows.csv','feature_order':FEATURES[key],
            'columns':['row_id','group','label']+FEATURES[key],'n_rows':len(frame),
            'row_id_semantics':'One-based original source record in source order; never renumbered after exclusions',
            'n_groups':int(frame.group.nunique()),'group_counts':count_strings(frame.group),
            'label_counts':count_strings(frame.label),'blank_label_count':int(frame.label.eq('').sum()),
            'feature_minimum':{x:float(frame[x].min()) for x in FEATURES[key]},
            'feature_maximum':{x:float(frame[x].max()) for x in FEATURES[key]},
            'exclusions':exclusions if key=='abalone' else [],**specification}


def write_csv(path,frame):
    frame.to_csv(path,index=False,float_format='%.17g',lineterminator='\n')


def prepare(output=DEFAULT_OUTPUT,chemical_dir=CHEMICAL_AUDIT,measurement_dir=MEASUREMENT_AUDIT,download=False):
    output=Path(output).resolve()
    acquire_sources(output/'raw',chemical_dir,measurement_dir,download)
    frames,row_metadata,exclusions=raw_to_frames(output/'raw')
    datasets={}
    for key,frame in frames.items():
        dataset_id=DATASET_IDS[key]
        write_csv(output/(dataset_id+'.csv'),frame)
        write_csv(output/(dataset_id+'_rows.csv'),row_metadata[key])
        datasets[dataset_id]=dataset_metadata(key,frame,exclusions)
        for field in ['csv','row_metadata_csv']:
            datasets[dataset_id][field+'_sha256']=sha256((output/datasets[dataset_id][field]).read_bytes())
    metadata={
        'schema_version':1,'panel_fixed_date':'2026-11-09',
        'panel_status':'Hydrochem, GEMAS and Abalone selected before any model fitting for this expansion',
        'preparation':'Raw selected positive values only: no closure, logs, standardization, unit conversion, imputation or fitted preprocessing',
        'reader_contract':{'row_id':'integer','group':'string','label':'string; empty string is a real preserved metadata value',
            'pandas_example':"pd.read_csv(path, keep_default_na=False, dtype={'row_id':int,'group':str,'label':str}, converters={name:float for name in feature_order})"},
        'sources':{name:{**{k:v for k,v in source.items() if k!='audit'},'path':'raw/'+name} for name,source in SOURCES.items()},
        'archive_members':{name:{'sha256':value,'path':'raw/'+name,'container':'raw/abalone.zip'} for name,value in ZIP_MEMBERS.items()},
        'preparer':{'path':'scripts/prepare_extended_application_data.py','sha256':sha256(Path(__file__).read_bytes()),
                    'required_data_reader':'rdata==1.1.0 (existing project environment)'},
        'datasets':datasets,
    }
    (output/'preparation_manifest.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
    (output/'exclusions.json').write_text(json.dumps({'hydrochem_cations':[],'gemas_major6':[],'abalone_log4':exclusions},indent=2,allow_nan=False)+'\n')
    return metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--chemical-audit-dir',type=Path,default=CHEMICAL_AUDIT)
    parser.add_argument('--measurement-audit-dir',type=Path,default=MEASUREMENT_AUDIT)
    parser.add_argument('--download',action='store_true',help='Fetch only missing raw sources; check frozen hashes before use')
    args=parser.parse_args()
    metadata=prepare(args.output,args.chemical_audit_dir,args.measurement_audit_dir,args.download)
    print(json.dumps({'output':str(args.output.resolve()),'datasets':{key:{'n_rows':value['n_rows'],'n_groups':value['n_groups'],
        'csv_sha256':value['csv_sha256']} for key,value in metadata['datasets'].items()}}))


if __name__=='__main__':
    main()
