"""Fetch missing public data, verify exact source bytes, and prepare study tables.

Download-only datasets are not included in the repository. The network is used
only when this command is explicitly run. Existing files must match checksums.
No fitting, imputation, scaling or selection by outcomes occurs here.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
import urllib.request

ROOT=Path(__file__).resolve().parents[1]


def fetch(url,path,expected):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    raw=path.read_bytes() if path.exists() else urllib.request.urlopen(url,timeout=60).read()
    if hashlib.sha256(raw).hexdigest()!=expected:
        raise ValueError('Source changed; do not silently change the scientific data: '+url)
    if not path.exists():path.write_bytes(raw)
    return raw


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets',choices=['all','original','panel','benchmarks'],default='all');a=p.parse_args()
    if a.datasets in ['all','original']:
        for name in ['prepare_oliveoil','prepare_crabs']:
            subprocess.run([sys.executable,str(ROOT/'scripts'/f'{name}.py'),'--download'],check=True)
    if a.datasets in ['all','panel']:
        subprocess.run([sys.executable,str(ROOT/'scripts/prepare_extended_application_data.py'),'--download'],check=True)
    if a.datasets in ['all','benchmarks']:
        m=json.loads((ROOT/'data/benchmark_sources.json').read_text())['datasets']['faithful']
        raw=fetch(m['download_url'],ROOT/m['original_file'],m['original_sha256']).decode()
        fields={}
        for name in ['eruptions','waiting']:
            match=re.search(r'\b'+name+r'\s*=\s*c\(([^)]*)\)',raw)
            if not match:raise ValueError('Unexpected source structure')
            fields[name]=[x.strip() for x in match.group(1).split(',')]
            if len(fields[name])!=272 or not all(re.fullmatch(r'\d+(?:\.\d+)?',v) for v in fields[name]):raise ValueError('Unexpected data')
        target=ROOT/'data/faithful.csv'
        with target.open('w',newline='') as f:
            w=csv.writer(f);w.writerow(fields);w.writerows(zip(*fields.values()))
        if hashlib.sha256(target.read_bytes()).hexdigest()!=m['analysis_table_sha256']:raise ValueError('Prepared data differs')
    print('Requested data sources verified and prepared.')


if __name__=='__main__':main()
