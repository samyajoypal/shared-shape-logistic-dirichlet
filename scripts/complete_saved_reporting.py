"""Verify and regenerate complete numeric interval and simulation summaries.

Interval summaries are recomputed from the saved outer-sample records.
Simulation summaries are checked saved estimates, not recomputed model fits.
No document generation, data download or fitting is performed.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import interval_reporting

ROOT = Path(__file__).resolve().parents[1]
SIMULATION_FILES = ('settings.csv', 'primary_paired_performance.csv',
    'primary_numerical_diagnostics.csv', 'primary_numerical_totals.csv',
    'conditional_risk_changed_methods.csv', 'conditional_risk_changed_pairs.csv')
INTERVAL_FILES = ('inference_task_intervals.csv.gz', 'inference_outer_diagnostics.csv.gz',
                  'extraction_manifest.json', 'aggregate.json', 'production_configuration.json')
CONFIG_FILES = ('configs/phase2_simulations.json', 'configs/phase2_gaussian_control.json')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def required_inputs():
    return (list(CONFIG_FILES)
            + ['saved_summaries/interval_records/'+name for name in INTERVAL_FILES]
            + ['saved_summaries/complete_simulations/'+name for name in SIMULATION_FILES]
            + ['saved_summaries/conditional_risk/all_cell_paired_risks.csv'])


def verified_inputs(root):
    manifest = json.loads((root/'export_manifest.json').read_text())
    entries = manifest['entries']
    lookup = {entry['file']:entry['export_sha256'] for entry in entries}
    if len(lookup) != len(entries):
        raise ValueError('Duplicate exported file identity')
    required = required_inputs()
    if set(required)-set(lookup):
        raise ValueError('A required numeric input is absent from the export manifest')
    hashes = {}
    for relative in required:
        value = sha(root/relative)
        if value != lookup[relative]:
            raise ValueError('Changed numeric input: '+relative)
        hashes[relative] = value
    return hashes


def validate_simulations(root):
    folder = root/'saved_summaries/complete_simulations'
    frames = {name:pd.read_csv(folder/name) for name in SIMULATION_FILES}
    config = json.loads((root/CONFIG_FILES[0]).read_text())
    control = json.loads((root/CONFIG_FILES[1]).read_text())
    cells = config['cells']
    settings = frames['settings.csv']
    if (len(cells) != 65 or settings.setting.tolist() != list(range(1,66))
            or settings.cell_id.tolist() != [cell['id'] for cell in cells]
            or int(settings.repetitions.sum()) != 10500):
        raise ValueError('The complete primary setting scope differs')
    for _, row in settings.iterrows():
        cell = cells[int(row.setting)-1]
        for column in settings.columns:
            if column in ('setting','cell_id'):
                continue
            expected = cell.get(column)
            if expected is None:
                if not pd.isna(row[column]):
                    raise ValueError('Unexpected setting metadata')
            elif row[column] != expected:
                raise ValueError('Setting metadata differs from the configuration')
    methods = config['primary_methods'] + ['bounded_skew_normal',control['method']]
    if len(set(methods)) != 9:
        raise ValueError('Expected nine distinct procedures')
    comparators = set(methods)-{'shared_ld'}
    diagnostics = frames['primary_numerical_diagnostics.csv']
    pairs = frames['primary_paired_performance.csv']
    if (len(diagnostics)!=585 or len(pairs)!=520
            or diagnostics.duplicated(['cell_id','method']).any()
            or pairs.duplicated(['cell_id','comparator']).any()):
        raise ValueError('Duplicate or omitted method/setting rows')
    for number, cell in enumerate(cells,1):
        d = diagnostics[diagnostics.cell_id.eq(cell['id'])]
        p = pairs[pairs.cell_id.eq(cell['id'])]
        if (set(d.method)!=set(methods) or set(p.comparator)!=comparators
                or not d.setting.eq(number).all() or not p.setting.eq(number).all()):
            raise ValueError('Changed method or setting identity')
        planned = {method:(min(cell['repetitions'],config['secondary']['bounded_skew_repetitions_per_cell'])
                           if method=='bounded_skew_normal' else cell['repetitions']) for method in methods}
        for _, row in d.iterrows():
            if row.planned != planned[row.method]:
                raise ValueError('Changed planned fit denominator')
        for _, row in p.iterrows():
            if row.planned_pairs != planned[row.comparator]:
                raise ValueError('Changed planned pair denominator')
            for metric in ('test_log_density','mean_rmse','weight_rmse','mean_mse','weight_mse'):
                count = row[metric+'_pairs']
                estimate, se = row[metric+'_difference'], row[metric+'_mcse']
                if count != int(count) or not 0 <= count <= row.planned_pairs:
                    raise ValueError('Invalid paired available count')
                if count and not np.isfinite(estimate):
                    raise ValueError('Available paired estimate is not finite')
                if count>1 and (not np.isfinite(se) or se<0):
                    raise ValueError('Invalid paired Monte Carlo standard error')
    integer_columns = [c for c in diagnostics if c not in ('setting','cell_id','method')]
    counts = diagnostics[integer_columns].to_numpy()
    if not np.isfinite(counts).all() or np.any(counts<0) or np.any(counts != np.floor(counts)):
        raise ValueError('Invalid numerical diagnostic count')
    identities = ((diagnostics.recorded+diagnostics.missing,diagnostics.planned),
                  (diagnostics.evaluation_available+diagnostics.unavailable,diagnostics.planned),
                  (diagnostics.score_checked+diagnostics.score_unknown,diagnostics.recorded),
                  (diagnostics.score_passed+diagnostics.score_failed,diagnostics.score_checked),
                  (diagnostics.boundary_checked+diagnostics.boundary_unknown,diagnostics.recorded))
    if any(not np.array_equal(left,right) for left,right in identities):
        raise ValueError('Diagnostic denominators do not reconcile')
    totals = frames['primary_numerical_totals.csv'].set_index('method').sort_index()
    summed = diagnostics.groupby('method')[integer_columns].sum().sort_index()
    pd.testing.assert_frame_equal(totals[summed.columns],summed,check_dtype=False)
    changed = frames['conditional_risk_changed_methods.csv']
    changed_pairs = frames['conditional_risk_changed_pairs.csv']
    expected_cells = {f'core_ld_d6_k3_n250_h{h}':count for h,count in [('0p2',199),('0p5',199),('1p0',192)]}
    if (len(changed)!=9 or len(changed_pairs)!=6
            or set(changed.cell_id)!=set(expected_cells) or set(changed_pairs.cell_id)!=set(expected_cells)
            or changed.duplicated(['cell_id','method']).any()
            or changed_pairs.duplicated(['cell_id','left_method','right_method']).any()):
        raise ValueError('Changed common-score risk scope differs')
    for cell,count in expected_cells.items():
        a = changed[changed.cell_id.eq(cell)]
        b = changed_pairs[changed_pairs.cell_id.eq(cell)]
        if (set(a.method)!={'shared_ld','full_ld','gaussian_shared_correlation'}
                or not a.available.eq(count).all() or not b.paired_available.eq(count).all()
                or not b.left_method.eq('shared_ld').all()
                or set(b.right_method)!={'full_ld','gaussian_shared_correlation'}
                or not a.subset.eq('all_methods_score_eligible').all()
                or not b.subset.eq('all_methods_score_eligible').all()):
            raise ValueError('Common-score risk denominators or methods differ')
    existing = pd.read_csv(root/'saved_summaries/conditional_risk/all_cell_paired_risks.csv')
    selected = existing[existing.cell_id.isin(expected_cells) & existing.left_method.eq('shared_ld')
                        & existing.subset.eq('all_methods_score_eligible')]
    keys = ['cell_id','left_method','right_method','subset']
    pd.testing.assert_frame_equal(changed_pairs.sort_values(keys).reset_index(drop=True),
        selected.sort_values(keys).reset_index(drop=True),check_dtype=False,rtol=2e-13,atol=2e-13)
    return frames


def build(root, output):
    root,output = Path(root),Path(output)
    hashes = verified_inputs(root)
    frames = validate_simulations(root)
    interval_dir,simulation_dir = output/'complete_intervals',output/'complete_simulations'
    if interval_dir.exists() or simulation_dir.exists():
        raise FileExistsError('Choose a fresh complete-report output directory')
    receipt = interval_reporting.run(root/'saved_summaries/interval_records',interval_dir)
    simulation_dir.mkdir(parents=True)
    outputs = {'complete_intervals/'+name:value for name,value in receipt['output_sha256'].items()}
    for name in SIMULATION_FILES:
        path = simulation_dir/name
        path.write_bytes((root/'saved_summaries/complete_simulations'/name).read_bytes())
        outputs['complete_simulations/'+name] = sha(path)
    if any(sha(root/path)!=value for path,value in hashes.items()):
        raise ValueError('A verified input changed during reporting')
    manifest = {'input_sha256':hashes,'output_sha256':outputs,'source_sha256':sha(__file__),
                'interval_source_sha256':sha(Path(interval_reporting.__file__)),
                'new_fits':False,'data_downloads':False,
                'intervals':'Recomputed from all54000 interval records and4500 outer records',
                'simulations':'Verified saved numerical estimates; full fitted checkpoints are not bundled',
                'simulation_rows':{name:len(frame) for name,frame in frames.items()}}
    (output/'complete_reporting_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(build(ROOT,args.output),indent=2))
