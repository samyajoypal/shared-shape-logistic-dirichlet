"""Raw values, exclusions, grouping and deterministic offline reproducibility."""
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rdata

import prepare_extended_application_data as prep

RAW=prep.DEFAULT_OUTPUT/'raw'
pytestmark = pytest.mark.skipif(not (RAW/'gemas.rda').exists(), reason='Run scripts/fetch_data.py before full source-data tests')


def read_processed(path,features):
    return pd.read_csv(path,keep_default_na=False,dtype={'row_id':int,'group':str,'label':str},
                       converters={name:float for name in features})


@pytest.fixture
def prepared(tmp_path):
    destination=tmp_path/'prepared'
    metadata=prep.prepare(destination,chemical_dir=RAW,measurement_dir=RAW)
    return destination,metadata


def test_every_declared_source_is_preserved_byte_for_byte(prepared):
    output,metadata=prepared
    for name,record in prep.SOURCES.items():
        assert prep.sha256((output/'raw'/name).read_bytes())==record['sha256']
        assert (output/'raw'/name).read_bytes()==(RAW/name).read_bytes()
    for name,value in prep.ZIP_MEMBERS.items():
        assert prep.sha256((output/'raw'/name).read_bytes())==value
    assert set(metadata['datasets'])=={'hydrochem_cations','gemas_major6','abalone_log4'}


def test_exact_hydro_values_original_row_order_and_station_groups(prepared):
    output,metadata=prepared
    raw=pd.read_csv(RAW/'Hydrochem.txt',sep=r'\s+',converters={x:float for x in prep.FEATURES['hydrochem']})
    value=read_processed(output/'hydrochem_cations.csv',prep.FEATURES['hydrochem'])
    assert list(value.columns)==['row_id','group','label','Na','K','Mg','Ca']
    assert value.row_id.tolist()==list(range(1,486))
    np.testing.assert_array_equal(value.iloc[:,3:].to_numpy(),raw[prep.FEATURES['hydrochem']].to_numpy())
    assert value.group.tolist()==raw.Site.astype(str).tolist()
    assert value.label.tolist()==raw.River.tolist()
    assert value.group.nunique()==30 and value.groupby('group').size().min()==4
    rows=pd.read_csv(output/'hydrochem_cations_rows.csv')
    assert rows.Site.nunique()==30 and rows.Date.nunique()==22


def test_gemas_features_country_groups_and_blank_labels_are_preserved(prepared):
    output,metadata=prepared
    original=rdata.read_rda(RAW/'gemas.rda')['gemas']
    value=read_processed(output/'gemas_major6.csv',prep.FEATURES['gemas'])
    assert list(value.columns)==['row_id','group','label','Ca','Fe','K','Mg','Si','Al']
    assert value.row_id.tolist()==list(range(1,2109))
    np.testing.assert_array_equal(value.iloc[:,3:].to_numpy(),original[prep.FEATURES['gemas']].to_numpy())
    assert value.group.tolist()==original.COUNTRY.astype(str).tolist()
    assert value.label.tolist()==original.soilclass.astype(str).tolist()
    assert value.label.eq('').sum()==5 and value.group.nunique()==33
    assert metadata['datasets']['gemas_major6']['blank_label_count']==5
    assert metadata['datasets']['gemas_major6']['exclusions']==[]


def test_abalone_only_the_two_authorized_zero_height_rows_are_removed(prepared):
    output,metadata=prepared
    original=pd.read_csv(RAW/'abalone.data',header=None,names=prep.ABALONE_COLUMNS,
                         converters={x:float for x in prep.ABALONE_COLUMNS[1:-1]})
    value=read_processed(output/'abalone_log4.csv',prep.FEATURES['abalone'])
    ids=[x for x in range(1,4178) if x not in [1258,3997]]
    assert value.row_id.tolist()==ids
    assert value.group.tolist()==[str(x) for x in ids] and value.group.nunique()==4175
    np.testing.assert_array_equal(value.iloc[:,3:].to_numpy(),original.iloc[np.array(ids)-1][prep.FEATURES['abalone']].to_numpy())
    assert value.label.tolist()==original.iloc[np.array(ids)-1].Sex.tolist()
    assert value.Height.max()==1.13  # Unusual positive dimensions are retained.
    excluded=metadata['datasets']['abalone_log4']['exclusions']
    assert [x['row_id'] for x in excluded]==[1258,3997]
    for item in excluded:
        assert item['original_values']['Height']==0
        assert item['original_line']==(RAW/'abalone.data').read_text().splitlines()[item['row_id']-1]


def test_target_reference_and_units_are_metadata_only(prepared):
    _,metadata=prepared
    h,g,a=[metadata['datasets'][k] for k in ['hydrochem_cations','gemas_major6','abalone_log4']]
    assert (h['reference'],h['target'],h['transform_plan'])==('Ca','Na/Ca','alr')
    assert (g['reference'],g['target'],g['transform_plan'])==('Al','Ca/Al','alr')
    assert (a['reference'],a['target'],a['transform_plan'])==(None,'ShuckedWeight','log')
    assert not a['source_scaling_statement']['applied_back_conversion']
    assert a['source_scaling_statement']['stated_divisor']==200
    assert all('Published scaled' in s for s in a['units'].values())
    assert set(h['units'])==set(h['feature_order'])
    assert set(g['units'])==set(g['feature_order'])
    assert set(a['units'])==set(a['feature_order'])
    assert 'no closure' in metadata['preparation']


def test_offline_rerun_without_audit_directories_is_byte_identical_and_never_downloads(prepared,tmp_path,monkeypatch):
    output,metadata=prepared
    before={str(p.relative_to(output)):p.read_bytes() for p in output.rglob('*') if p.is_file()}
    def forbidden(*args,**kwargs): raise AssertionError('Network used despite cached audited bytes')
    monkeypatch.setattr(prep.urllib.request,'urlopen',forbidden)
    new=prep.prepare(output,tmp_path/'absent_chemical',tmp_path/'absent_measurement',download=True)
    assert new==metadata
    assert before=={str(p.relative_to(output)):p.read_bytes() for p in output.rglob('*') if p.is_file()}


def test_reproducible_outputs_do_not_depend_on_destination_directory(prepared,tmp_path):
    output,metadata=prepared
    other=tmp_path/'other'
    new=prep.prepare(other,output/'raw',output/'raw')
    assert new==metadata
    for p in output.rglob('*'):
        if p.is_file(): assert p.read_bytes()==(other/p.relative_to(output)).read_bytes()


def test_corrupt_cached_source_is_not_silently_refetched(prepared,monkeypatch):
    output,_=prepared;path=output/'raw/Hydrochem.txt';path.write_bytes(b'bad audited data')
    monkeypatch.setattr(prep.urllib.request,'urlopen',lambda *a,**k:pytest.fail('Must not refetch corrupt cached data'))
    with pytest.raises(ValueError,match='hash mismatch'):
        prep.prepare(output,RAW,RAW,download=True)
    assert path.read_bytes()==b'bad audited data'


def test_corrupt_extracted_abalone_cannot_hide_behind_valid_zip(prepared):
    output,_=prepared;(output/'raw/abalone.data').write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='hash mismatch'):
        prep.prepare(output,RAW,RAW)


def test_missing_source_offline_does_not_enable_network(tmp_path,monkeypatch):
    monkeypatch.setattr(prep.urllib.request,'urlopen',lambda *a,**k:pytest.fail('Offline means no download'))
    with pytest.raises(FileNotFoundError,match='Missing audited raw source'):
        prep.prepare(tmp_path/'out',tmp_path/'none1',tmp_path/'none2')


def test_explicit_download_fetches_only_missing_sources_and_checks_every_hash(tmp_path,monkeypatch):
    calls=[];by_url={v['url']:name for name,v in prep.SOURCES.items()}
    def fetch(url,timeout):
        calls.append(url);return io.BytesIO((RAW/by_url[url]).read_bytes())
    monkeypatch.setattr(prep.urllib.request,'urlopen',fetch)
    out=tmp_path/'downloaded'
    metadata=prep.prepare(out,tmp_path/'none1',tmp_path/'none2',download=True)
    assert len(calls)==5 and metadata['datasets']['abalone_log4']['n_rows']==4175
    prep.prepare(out,tmp_path/'none1',tmp_path/'none2',download=True)
    assert len(calls)==5


def test_changed_download_is_rejected_before_raw_files_are_written(tmp_path,monkeypatch):
    monkeypatch.setattr(prep.urllib.request,'urlopen',lambda *a,**k:io.BytesIO(b'changed upstream data'))
    out=tmp_path/'reject'
    with pytest.raises(ValueError,match='hash mismatch'):
        prep.prepare(out,tmp_path/'none1',tmp_path/'none2',download=True)
    assert not out.exists()


def test_preparation_contract_matches_frozen_application_config(prepared):
    output,metadata=prepared
    config=json.loads((prep.ROOT/'configs/phase3_applications.json').read_text())
    assert Path(config['data_metadata']).name=='preparation_manifest.json'
    assert set(metadata['datasets'])=={s['id'] for s in config['datasets']}
    for spec in config['datasets']:
        item=metadata['datasets'][spec['id']]
        assert item['csv']==Path(spec['csv']).name
        assert item['feature_order']==spec['columns']
        assert item['n_rows']==spec['expected_rows']
        assert item['n_groups']==spec['expected_groups']
        assert item['transform_plan']==spec['transform']
        assert item['reference']==spec.get('reference')
        assert item['target']==spec['conditional_target']
        assert item['csv_sha256']==prep.sha256((output/item['csv']).read_bytes())
