"""Small synthetic installation check; not part of the scientific study.

Uses different labelled demo seeds and one start for speed. Its output is not
publication evidence and must not be combined with the fixed study summaries.
"""
from pathlib import Path
import argparse
import json
import numpy as np
from threadpoolctl import threadpool_limits
import phase2_simulations as simulation
import run_confirmatory_experiments as models
import gaussian_shared_correlation as gaussian
ROOT=Path(__file__).resolve().parents[1]

def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    config=json.loads((ROOT/'configs/phase2_simulations.json').read_text())
    spec=simulation.generating_parameters(config['cells'][0],config)
    x,_=simulation.generate_sample(100,spec,np.random.default_rng(10801))
    test,_=simulation.generate_sample(100,spec,np.random.default_rng(10802))
    center=x.mean(0);scale=x.std(0);y=(x-center)/scale;z=(test-center)/scale
    partitions=models.make_partitions(y,2,10803,n_starts=1)
    settings={**models.DEFAULT_SETTINGS,'n_starts':1,'maxiter':300,'refinement_maxiter':300,
              'package_maxiter':200,'start_timeout_seconds':30}
    fits={};rows=[]
    with threadpool_limits(limits=1):
        for method in ['shared_ld','full_ld','gaussian_full','fmvmm_dmm']:
            fit=models.fit_method(method,y,2,10803,settings,partitions,
                warm_start=fits.get('shared_ld') if method=='full_ld' else None)
            logpdf=fit.evaluate(z)[0]-np.log(scale).sum()
            if not np.isfinite(logpdf).all():raise ValueError('Nonfinite demo density')
            fits[method]=fit;rows.append({'method':method,'test_log_density':float(logpdf.mean()),
                'score_converged':fit.record.get('score_converged'),'boundary_hit':fit.record.get('boundary_hit')})
        fit=gaussian.fit(y,2,10803,partitions=partitions,n_starts=1,maxiter=300,polish_maxiter=300)
        if not np.isfinite(fit.evaluate(z)[0]).all():raise ValueError('Nonfinite Gaussian-control density')
        rows.append({'method':gaussian.METHOD,'score_converged':fit.record.get('score_converged')})
    result={'purpose':'installation-only synthetic smoke','scientific_study':False,
            'dmm_interpretation':'Induced inverse-ALR baseline on standardized synthetic observations, not natural compositional data.',
            'fits':rows,'finite':True}
    (output/'smoke_result.json').write_text(json.dumps(result,indent=2)+'\n');return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'artifacts/smoke')
    a=p.parse_args();print(json.dumps(run(a.output)))
if __name__=='__main__':main()
