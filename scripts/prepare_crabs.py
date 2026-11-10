"""Convert the public MASS crab measurements without executing R code."""
from pathlib import Path
import argparse
import hashlib
import json
import urllib.request
import rdata

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://raw.githubusercontent.com/cran/MASS/master/data/crabs.rda'
SHA256 = 'deb6b7b2a844e60c867741a6fc11478d45e7052e9d2c3c79de59a8d1966c5e52'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    raw = ROOT / 'data/crabs_original.rda'
    if args.download and not raw.exists():
        with urllib.request.urlopen(URL, timeout=40) as response:
            raw.write_bytes(response.read())
    if hashlib.sha256(raw.read_bytes()).hexdigest() != SHA256:
        raise ValueError('Source hash changed; inspect before changing the frozen data.')
    frame = rdata.read_rda(raw)['crabs']
    assert frame.shape == (200, 8)
    assert (frame[['FL','RW','CL','CW','BD']].to_numpy() > 0).all()
    target = ROOT / 'data/crabs.csv'
    frame.to_csv(target, index=False)
    metadata = json.loads((ROOT / 'data/crabs_provenance.json').read_text())
    metadata['source_sha256'] = SHA256
    metadata['csv_sha256'] = hashlib.sha256(target.read_bytes()).hexdigest()
    (ROOT / 'data/crabs_provenance.json').write_text(json.dumps(metadata, indent=2)+'\n')


if __name__ == '__main__':
    main()
