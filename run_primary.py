"""Prospectively separated primary study with a physics-generated action library."""
from pathlib import Path
from dataclasses import replace
from time import perf_counter
import json,sys
import numpy as np
import pandas as pd
from scipy.stats import norm
from twin_core import *

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data'
ALPHAS=[.01,.025,.05,.1,.2];CAL_N=[49,99,199,399]

def unit(seed,changed=False):
    plant=draw_plant(seed)
    if changed:plant=replace(plant,gi=plant.gi*.65,eta=plant.eta*.85)
    obs=commission(plant,seed+(1_000_000 if changed else 0))
    ambient=float(np.random.default_rng(seed+77).uniform(25,45))
    start=perf_counter();p,c,fit=fit_twin(obs);fit_seconds=perf_counter()-start
    a=proposed_library(p,ambient)
    start=perf_counter();mean,scale=predict(p,c,a,ambient);prediction_seconds=perf_counter()-start
    truth=plant_outputs(plant,a,ambient)[0];score=(mean-truth)/scale
    point=int(np.random.default_rng(seed+12345).integers(len(a)))
    info={'seed':seed,'ambient':ambient,**plant_dict(plant),**fit,'fit_seconds':fit_seconds,
          'prediction_seconds':prediction_seconds,'candidate_count':len(a),
          'maxscore':float(score.max()),'pointscore':float(score[point].max()),
          'random_action_index':point,'oracle_index':oracle(truth)}
    for j,v in enumerate(p):info['fit_p'+str(j)]=float(v)
    return plant,obs,p,c,a,mean,scale,truth,score,info

def row(seed,name,idx,a,truth,mean,scale,oi,extra=None):
    accepted=idx>=0
    d={'seed':seed,'method':name,'authorized':accepted,'unsafe':bool(accepted and np.any(truth[idx]<0)),
       'index':idx,'candidate_count':len(a),'oracle_available':oi>=0,
       'oracle_heat_w':float(a[oi].sum()) if oi>=0 else np.nan,
       'heat_w':float(a[idx].sum()) if accepted else np.nan}
    for j,key in enumerate(['temperature','power','leakage']):
        d[key+'_failure']=bool(accepted and truth[idx,j]<0)
        d[key+'_margin']=float(truth[idx,j]) if accepted else np.nan
        d[key+'_predicted_margin']=float(mean[idx,j]) if accepted else np.nan
    if extra:d.update(extra)
    return d

def run(changed=False):
    prefix='refreshed' if changed else 'primary'
    start=perf_counter();cal=[];calobs=[];calfit=[];calcov=[]
    for i,seed in enumerate(range(70000,70399)):
        plant,obs,p,c,a,m,s,t,sc,info=unit(seed,changed)
        cal.append(info);calobs.append(obs);calfit.append(p);calcov.append(c)
        if (i+1)%100==0:print(prefix,'calibration',i+1,flush=True)
    cal=pd.DataFrame(cal);cal.to_csv(DATA/f'{prefix}_calibration.csv',index=False)
    q=quantile(cal.maxscore,.05);qp=quantile(cal.pointscore,.05);qg=float(norm.ppf(1-.05/3))
    qa={alpha:quantile(cal.maxscore,alpha) for alpha in ALPHAS}
    qn={n:quantile(cal.maxscore.iloc[:n],.05) for n in CAL_N}
    settings={'q_uniform':q,'q_point':qp,'q_gaussian':qg,'alpha':.05,
              'alpha_quantiles':qa,'size_quantiles':qn,'calibration_count':len(cal),
              'qualification_responses':int(cal.candidate_count.sum()),
              'max_candidates':int(cal.candidate_count.max()),'mean_candidates':float(cal.candidate_count.mean())}
    (DATA/f'{prefix}_thresholds.json').write_text(json.dumps(settings,indent=2));print(prefix,settings,flush=True)
    decisions=[];coverage=[];infoall=[];sensitivity=[];stale=[];observations=[];fits=[];covs=[]
    for i,seed in enumerate(range(80000,81000)):
        plant,obs,p,c,a,m,s,t,sc,info=unit(seed,changed);oi=info['oracle_index']
        infoall.append(info);observations.append(obs);fits.append(p);covs.append(c)
        fixed=np.flatnonzero(np.all(m-np.array([2.,.15,.004])>=0,axis=1))
        ids={'Plugin':select(m,s,0.),'Gaussian':select(m,s,qg),'Pointwise':select(m,s,qp),
             'Fixed margin':int(fixed[0]) if len(fixed) else -1,'Proposed':select(m,s,q),'Oracle':oi}
        for name,idx in ids.items():
            decisions.append(row(seed,name,idx,a,t,m,s,oi))
            z={'Plugin':0.,'Gaussian':qg,'Pointwise':qp,'Proposed':q}.get(name)
            if z is not None:
                coverage.append({'seed':seed,'method':name,'uniform_covered':bool(sc.max()<=z),
                      'random_covered':bool(sc[info['random_action_index']].max()<=z),
                      'chosen_covered':bool(idx>=0 and sc[idx].max()<=z),'authorized':idx>=0})
        for alpha,z in qa.items():sensitivity.append(row(seed,'alpha',select(m,s,z),a,t,m,s,oi,{'setting':alpha,'quantile':z}))
        for n,z in qn.items():sensitivity.append(row(seed,'calibration_size',select(m,s,z),a,t,m,s,oi,{'setting':n,'quantile':z}))
        if not changed:
            fault=replace(plant,gi=plant.gi*.65,eta=plant.eta*.85)
            it=[idx for idx in ids.values() if idx>=0]
            if it:
                ftruth=plant_outputs(fault,a[it],info['ambient'])[0];lookup={idx:ftruth[j] for j,idx in enumerate(it)}
            for name,idx in ids.items():
                if name=='Oracle':continue
                stale.append({'seed':seed,'method':name,'authorized':idx>=0,
                    'unsafe':bool(idx>=0 and np.any(lookup[idx]<0)),
                    **{key+'_failure':bool(idx>=0 and lookup[idx][j]<0) for j,key in enumerate(['temperature','power','leakage'])}})
        if seed in [80000,80001,80002,80003,80004]:
            np.savez_compressed(DATA/f'{prefix}_example_{seed}.npz',actions=a,mean=m,scale=s,truth=t,
                        state=p,covariance=c,observations=obs,ambient=info['ambient'])
        if (i+1)%200==0:print(prefix,'test',i+1,flush=True)
    pd.DataFrame(decisions).to_csv(DATA/f'{prefix}_decisions.csv',index=False)
    pd.DataFrame(coverage).to_csv(DATA/f'{prefix}_coverage.csv',index=False)
    pd.DataFrame(infoall).to_csv(DATA/f'{prefix}_devices.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(DATA/f'{prefix}_sensitivity.csv',index=False)
    if stale:pd.DataFrame(stale).to_csv(DATA/'primary_stale_interface.csv',index=False)
    np.savez_compressed(DATA/f'{prefix}_commissioning.npz',calibration_observations=calobs,
            calibration_fitted=calfit,calibration_covariance=calcov,test_observations=observations,
            test_fitted=fits,test_covariance=covs,probes=probes())
    timing={'wall_seconds':perf_counter()-start,'calibration_devices':399,'test_devices':1000}
    (DATA/f'{prefix}_timing.json').write_text(json.dumps(timing,indent=2));print(prefix,timing,flush=True)

if __name__=='__main__':run(changed=len(sys.argv)>1 and sys.argv[1]=='refreshed')
