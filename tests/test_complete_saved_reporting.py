"""Complete scopes, preserved existing results and immutable numeric inputs."""
import json
from pathlib import Path
import shutil
import pandas as pd
import pytest
import complete_saved_reporting as complete
import reproduce_saved_tables as tables


@pytest.fixture
def isolated(tmp_path):
    root=tmp_path/'inputs';root.mkdir()
    for name in ['saved_summaries','configs']:
        shutil.copytree(complete.ROOT/name,root/name)
    shutil.copyfile(complete.ROOT/'export_manifest.json',root/'export_manifest.json')
    return root


def test_complete_numeric_scope_and_old_csv_bytes(isolated,tmp_path,monkeypatch):
    monkeypatch.setattr(tables,'ROOT',isolated)
    out=tmp_path/'outputs';receipt=tables.build(out)
    assert len(receipt['output_sha256'])==23
    prior=json.loads((isolated/'configs/previous_saved_csv_sha256.json').read_text())
    assert len(prior)==12
    for name,expected in prior.items():
        assert complete.sha(out/name)==expected
    counts={'interval_summary.csv':108,'paired_comparisons.csv':81,
            'outer_diagnostics.csv':9,'bootstrap_diagnostics.csv':9,'interval_exclusions.csv':108}
    for name,count in counts.items():
        assert len(pd.read_csv(out/'complete_intervals'/name))==count
    for name,count in zip(complete.SIMULATION_FILES,[65,520,585,9,9,6]):
        assert len(pd.read_csv(out/'complete_simulations'/name))==count
    interval=pd.read_csv(out/'complete_intervals/interval_summary.csv')
    no_intervals=interval[interval.method.eq('bootstrap_t')&interval.n.eq(100)]
    assert len(no_intervals)==9 and no_intervals.available.eq(0).all()
    assert no_intervals.conditional_coverage.isna().all()
    assert no_intervals.mean_length.isna().all()
    pairs=pd.read_csv(out/'complete_intervals/paired_comparisons.csv')
    assert (pairs.both_available+pairs.left_only_available+pairs.right_only_available+pairs.neither_available).eq(500).all()
    assert receipt['new_fits'] is False and receipt['data_downloads'] is False


@pytest.mark.parametrize('relative',[
    'saved_summaries/interval_records/inference_task_intervals.csv.gz',
    'saved_summaries/complete_simulations/primary_paired_performance.csv',
    'configs/phase2_simulations.json',
])
def test_modified_numeric_bytes_are_rejected(isolated,tmp_path,relative):
    p=isolated/relative;p.write_bytes(p.read_bytes()+b'\n')
    with pytest.raises(ValueError,match='Changed numeric input'):
        complete.build(isolated,tmp_path/'outputs')


def test_missing_required_hash_is_rejected(isolated):
    p=isolated/'export_manifest.json';m=json.loads(p.read_text())
    m['entries']=[entry for entry in m['entries'] if entry['file']!=complete.required_inputs()[0]]
    p.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='absent from the export manifest'):
        complete.verified_inputs(isolated)


def test_freshly_hashed_duplicate_setting_is_still_invalid(isolated):
    p=isolated/'saved_summaries/complete_simulations/settings.csv';f=pd.read_csv(p)
    f.loc[1,'cell_id']=f.loc[0,'cell_id'];f.to_csv(p,index=False)
    with pytest.raises(ValueError,match='setting scope'):
        complete.validate_simulations(isolated)


def test_unavailable_pair_denominator_cannot_exceed_plan(isolated):
    p=isolated/'saved_summaries/complete_simulations/primary_paired_performance.csv';f=pd.read_csv(p)
    f.loc[0,'mean_mse_pairs']=f.loc[0,'planned_pairs']+1;f.to_csv(p,index=False)
    with pytest.raises(ValueError,match='paired available count'):
        complete.validate_simulations(isolated)
