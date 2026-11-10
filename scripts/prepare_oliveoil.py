"""Acquire/convert the documented CRAN pdfCluster olive-oil dataset.

Default conversion is offline if the original is present. --download fetches
the public source; it never replaces an existing source file. No R execution,
pseudocounts, imputation, or outcome-driven feature selection is performed.
"""
from pathlib import Path
import argparse
import hashlib
import json
import urllib.request
import numpy as np
import rdata

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://raw.githubusercontent.com/cran/pdfCluster/master/data/oliveoil.rda'
EXPECTED_SHA256 = '1a3f94224a24023865ab6a0fc283154b99949e782fab39d0bdedc01655774669'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    raw = ROOT/'data/oliveoil_original.rda'
    if not raw.exists():
        if not args.download:
            raise SystemExit('Raw file absent. Run with --download to fetch the public dataset.')
        raw.write_bytes(urllib.request.urlopen(URL, timeout=60).read())
    if hashlib.sha256(raw.read_bytes()).hexdigest() != EXPECTED_SHA256:
        raise ValueError('Source checksum changed; inspect the new upstream version before analysis.')
    frame = rdata.read_rda(raw)['oliveoil']
    assert frame.shape == (572,10)
    frame.to_csv(ROOT/'data/oliveoil.csv',index=False)
    selected = ['palmitic','linoleic','oleic']
    x = frame[selected].to_numpy(dtype=float)
    assert np.isfinite(x).all() and (x > 0).all()
    provenance = {
        'retrieved_date':'2026-11-07', 'source_url':URL,
        'documentation_url':'https://search.r-project.org/CRAN/refmans/pdfCluster/html/oliveoil.html',
        'scientific_selection_source':'https://www.aocs.org/resource/individual-commodity-oils-and-fats/',
        'source_citation':'Forina et al. (2008), V-PARVUS, as credited by pdfCluster; see also Forina et al. (1983), Classification of olive oils from their fatty acid composition.',
        'license_note':'Retain upstream data attribution. Data documentation does not specify a separate dataset license; public distribution terms must be checked before repository release.',
        'raw_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),
        'csv_sha256':hashlib.sha256((ROOT/'data/oliveoil.csv').read_bytes()).hexdigest(),
        'rows':len(frame), 'primary_features':selected, 'reference':'oleic',
        'selection_reason':'Three major fatty acids selected from chemical context before fitting any models. Oleic is the major constituent and fixed reference. This is a three-part subcomposition, not the full eight-acid composition.',
        'coordinates':['log(palmitic/oleic)','log(linoleic/oleic)'],
        'zeros_in_primary_features':int((x==0).sum()),
        'minimum_primary_measurements':dict(zip(selected,x.min(axis=0).tolist())),
        'full_data_caution':'Some other acids have instrument-limit zeros; they are retained in the raw table but not analysed. No pseudocount is added.',
        'labels':'macro.area is held out from fitting and starts; k=3 fixed from the documented three macro-areas; labels used only for external ARI.',
        'reference_sensitivity':'Diagonal location-scale LD with different coordinate scales is not generally invariant to changing the ALR reference. Do not choose the reference by held-out performance.',
        'density_comparison':'DMM is fitted to the actual three-part composition using its original ALR coordinates. All held-out log densities are compared in original ALR measure. Standardizing ALR before a DMM fit changes its family and is not used here.'
    }
    (ROOT/'data/oliveoil_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps({'rows':len(frame),'features':selected,'positive':True}))


if __name__ == '__main__':
    main()
