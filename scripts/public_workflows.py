"""Portable helpers for fresh reproductions, without historical machine receipts.

A fresh conditional-risk report binds the provided simulation and Gaussian runs
before evaluating them. This does not alter selected settings, formulas or
fit selection, and is not a new scientific preregistration.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')


def bind_conditional_risk(primary,gaussian,output):
    import simulation_conditional_mean_risk as risk
    primary,gaussian,output=map(lambda p:Path(p).resolve(),(primary,gaussian,output))
    config=json.loads((ROOT/'configs/simulation_conditional_mean_risk.json').read_text())
    p=json.loads((primary/'manifest.json').read_text());g=json.loads((gaussian/'manifest.json').read_text())
    rule=ROOT/'docs/public_analysis_rules.json'
    config.update(primary_manifest_sha256=digest(primary/'manifest.json'),
                  gaussian_manifest_sha256=digest(gaussian/'manifest.json'),
                  primary_configuration_sha256=p['configuration_sha256'],protocol_sha256=digest(rule))
    risk.validate_scope(config,p,g)
    binding=output.parent/(output.name+'_binding');binding.mkdir(parents=True,exist_ok=False)
    config_path=binding/'config.json';write(config_path,config)
    receipt=binding/'rules_receipt.json'
    write(receipt,{'config_sha256':digest(config_path),'protocol_sha256':digest(rule),
                  'artifact_role':'public_reproduction_rules','historical_preregistration_receipt':False,
                  'new_conditional_mean_risk_outcomes_inspected':False,'new_model_fits':False,
                  'interpretation':'Bindings for a fresh rerun; published historical outcomes are already known.'})
    risk.freeze(primary,gaussian,output,config_path,rule,receipt)
    return output


def run_application_range(run,start,stop):
    for i in range(start,stop):
        subprocess.run([sys.executable,str(ROOT/'scripts/phase2_applications.py'),
                        '--output',str(run),'--task-index',str(i)],check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    r=sub.add_parser('bind-risk');r.add_argument('--primary',type=Path,required=True)
    r.add_argument('--gaussian',type=Path,required=True);r.add_argument('--output',type=Path,required=True)
    a=sub.add_parser('applications');a.add_argument('--run',type=Path,required=True);a.add_argument('--task-range',required=True)
    args=p.parse_args()
    if args.command=='bind-risk':print(bind_conditional_risk(args.primary,args.gaussian,args.output))
    else:
        start,stop=map(int,args.task_range.split(':'))
        if not 0<=start<stop:raise ValueError('Expected half-open START:STOP')
        run_application_range(args.run,start,stop)


if __name__=='__main__':main()
