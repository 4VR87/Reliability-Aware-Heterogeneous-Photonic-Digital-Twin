from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import beta

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data'

def interval(k,n):
    if not n:return [None,None]
    return [float(beta.ppf(.025,k,n-k+1)) if k else 0.,
            float(beta.ppf(.975,k+1,n-k)) if k<n else 1.]

def summarize(df):
    out=[]
    for method,g in df.groupby('method',sort=False):
        n=len(g);na=int(g.authorized.sum());nb=int(g.unsafe.sum())
        both=g[g.authorized & g.oracle_available].copy()
        delta=(both.heat_w-both.oracle_heat_w)*1000
        if len(delta):
            rng=np.random.default_rng(20261025)
            boot=np.mean(np.asarray(delta)[rng.integers(len(delta),size=(2000,len(delta)))],axis=1)
            heat_ci=np.quantile(boot,[.025,.975]).tolist()
        else:heat_ci=[None,None]
        out.append({'method':method,'n':n,'authorized_n':na,'unsafe_n':nb,
           'availability':na/n,'availability_ci':interval(na,n),
           'unsafe_authorization':nb/n,'unsafe_authorization_ci':interval(nb,n),
           'conditional_failure':nb/na if na else None,'conditional_failure_ci':interval(nb,na),
           'mean_heat_mw':float(g.heat_w.mean()*1000),'heat_excess_mw':float(delta.mean()) if len(delta) else None,
           'heat_excess_ci_mw':heat_ci,'oracle_matched_n':len(both),
           'temperature_failures':int(g.temperature_failure.sum()),'power_failures':int(g.power_failure.sum()),
           'leakage_failures':int(g.leakage_failure.sum())})
    return out

def main():
    result={}
    for prefix in ['primary','refreshed']:
        d=pd.read_csv(DATA/f'{prefix}_decisions.csv')
        result[prefix]=summarize(d)
        cov=pd.read_csv(DATA/f'{prefix}_coverage.csv');cc=[]
        for method,g in cov.groupby('method'):
            accepted=g[g.authorized]
            cc.append({'method':method,'uniform_coverage':float(g.uniform_covered.mean()),
                       'uniform_coverage_ci':interval(int(g.uniform_covered.sum()),len(g)),
                       'random_coverage':float(g.random_covered.mean()),
                       'chosen_coverage_given_authorized':float(accepted.chosen_covered.mean()) if len(accepted) else None})
        result[prefix+'_coverage']=cc
        sens=pd.read_csv(DATA/f'{prefix}_sensitivity.csv')
        result[prefix+'_sensitivity']=[{'kind':kind,'setting':float(setting),
           'availability':float(g.authorized.mean()),'unsafe_authorization':float(g.unsafe.mean()),
           'conditional_failure':float(g.unsafe.sum()/g.authorized.sum()) if g.authorized.any() else None,
           'quantile':float(g['quantile'].iloc[0])} for (kind,setting),g in sens.groupby(['method','setting'])]
    result['dense_reference']=summarize(pd.read_csv(DATA/'main_decisions.csv'))
    for key in ['noise_half','noise_double','optical_only','matched_full']:
        p=DATA/f'{key}_decisions.csv'
        if p.exists():result[key]=summarize(pd.read_csv(p))
    stale=pd.read_csv(DATA/'primary_stale_interface.csv')
    result['stale']=[{'method':method,'authorized_n':int(g.authorized.sum()),'unsafe_n':int(g.unsafe.sum()),
          'unsafe_authorization':float(g.unsafe.mean()),'conditional_failure':float(g.unsafe.sum()/g.authorized.sum()) if g.authorized.any() else None,
          **{key+'_failures':int(g[key+'_failure'].sum()) for key in ['temperature','power','leakage']}}
          for method,g in stale.groupby('method')]
    # Recompute every main declaration directly from retained selected margins.
    for prefix in ['primary','refreshed']:
        d=pd.read_csv(DATA/f'{prefix}_decisions.csv')
        actual=d.authorized & (d[['temperature_margin','power_margin','leakage_margin']]<0).any(axis=1)
        assert np.array_equal(actual,d.unsafe)
        assert ((d.method=='Oracle') & d.unsafe).sum()==0
        fits=pd.read_csv(DATA/f'{prefix}_devices.csv')
        result[prefix+'_fit_successes']=int(fits.success.sum())
    result['physics_validation']=json.loads((DATA/'physics_validation.json').read_text())
    result['primary_thresholds']=json.loads((DATA/'primary_thresholds.json').read_text())
    result['refreshed_thresholds']=json.loads((DATA/'refreshed_thresholds.json').read_text())
    (DATA/'results_summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({k:result[k] for k in ['primary','primary_coverage','refreshed','stale']},indent=2))

if __name__=='__main__':main()
