"""Run calibration, independent tests and prespecified sensitivities."""
from pathlib import Path
from dataclasses import replace
from time import perf_counter
import json,sys
import numpy as np
import pandas as pd
from scipy.stats import norm
from twin_core import *

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data';DATA.mkdir(exist_ok=True)
A=action_library();ALPHAS=[.01,.025,.05,.10,.20]
CAL_N=[49,99,199,399]
# Nested sublibraries selected by original indices, independent of outcomes.
SUBSETS={n:np.unique(np.linspace(0,len(A)-1,n,dtype=int)) for n in [64,256,1024,len(A)]}

def profile(seed,noise_scale=1.,temperatures=True):
    plant=draw_plant(seed);obs=commission(plant,seed,noise_scale)
    ambient=float(np.random.default_rng(seed+77).uniform(25,45))
    t0=perf_counter();p,c,fit=fit_twin(obs,noise_scale,temperatures);fit_time=perf_counter()-t0
    t0=perf_counter();mean,scale=predict(p,c,A,ambient);predict_time=perf_counter()-t0
    truth=plant_outputs(plant,A,ambient)[0]
    score=(mean-truth)/scale
    point=int(np.random.default_rng(seed+12345).integers(len(A)))
    info={'seed':seed,'ambient':ambient,**plant_dict(plant),**fit,
          'fit_seconds':fit_time,'prediction_seconds':predict_time,
          'maxscore':float(score.max()),'pointscore':float(score[point].max()),
          'random_action_index':point,'oracle_index':oracle(truth),
          'bound_mw':reduced_infeasibility_bound(p,ambient)}
    for j,v in enumerate(p):info['fit_p'+str(j)]=float(v)
    return plant,obs,p,c,mean,scale,truth,score,info

def decision(seed,method,idx,truth,mean,scale,oracle_id,meta=None):
    accepted=idx>=0;bad=accepted and bool(np.any(truth[idx]<0))
    row={'seed':seed,'method':method,'authorized':accepted,'unsafe':bad,'index':idx,
         'oracle_available':oracle_id>=0,'oracle_index':oracle_id,
         'heat_w':float(A[idx].sum()) if accepted else np.nan,
         'oracle_heat_w':float(A[oracle_id].sum()) if oracle_id>=0 else np.nan}
    for j,key in enumerate(['temperature','power','leakage']):
        row[key+'_failure']=bool(accepted and truth[idx,j]<0)
        row[key+'_margin']=float(truth[idx,j]) if accepted else np.nan
        row[key+'_predicted_margin']=float(mean[idx,j]) if accepted else np.nan
        row[key+'_scale']=float(scale[idx,j]) if accepted else np.nan
    if meta:row.update(meta)
    return row

def run_main():
    t0=perf_counter();cal_info=[];cal_subset=[];observations=[];fitted=[];covariances=[]
    print('Calibration started',flush=True)
    for i,seed in enumerate(range(20000,20399)):
        plant,obs,p,c,m,s,t,sc,info=profile(seed)
        cal_info.append(info);observations.append(obs);fitted.append(p);covariances.append(c)
        cal_subset.append({n:float(sc[ids].max()) for n,ids in SUBSETS.items()})
        if (i+1)%100==0:print('Calibration',i+1,flush=True)
    cal=pd.DataFrame(cal_info);cal.to_csv(DATA/'calibration_devices.csv',index=False)
    cal_sub=pd.DataFrame(cal_subset);cal_sub.to_csv(DATA/'calibration_subset_scores.csv',index=False)
    q=quantile(cal.maxscore,.05);qp=quantile(cal.pointscore,.05)
    qs={alpha:quantile(cal.maxscore,alpha) for alpha in ALPHAS}
    qn={n:quantile(cal.maxscore.iloc[:n],.05) for n in CAL_N}
    qsub={n:quantile(cal_sub[n],.05) for n in SUBSETS}
    thresholds={'alpha':.05,'q_uniform':q,'q_point':qp,'q_gaussian':float(norm.ppf(1-.05/3)),
                'alpha_quantiles':qs,'size_quantiles':qn,'subset_quantiles':qsub,
                'candidate_actions':len(A),'temperature_limit_c':LIMITS[0],
                'power_minimum_mw':LIMITS[1],'leakage_maximum':LIMITS[2]}
    (DATA/'thresholds.json').write_text(json.dumps(thresholds,indent=2))
    print('Frozen quantiles',thresholds,flush=True)
    rows=[];test_info=[];sensitivity=[];shift=[];coverage=[]
    for i,seed in enumerate(range(30000,31000)):
        plant,obs,p,c,m,s,t,sc,info=profile(seed)
        test_info.append(info);observations.append(obs);fitted.append(p);covariances.append(c)
        o=info['oracle_index']
        fixed=np.all(m-np.array([2.,.15,.004])>=0,axis=1)
        fixed_id=int(np.flatnonzero(fixed)[0]) if np.any(fixed) else -1
        ids={'Plugin':select(m,s,0.),'Gaussian':select(m,s,thresholds['q_gaussian']),
             'Pointwise':select(m,s,qp),'Fixed margin':fixed_id,
             'Proposed':select(m,s,q),'Oracle':o}
        for method,idx in ids.items():
            rows.append(decision(seed,method,idx,t,m,s,o))
            if method!='Oracle':
                z={'Plugin':0.,'Gaussian':thresholds['q_gaussian'],'Pointwise':qp,'Proposed':q}.get(method)
                if z is not None:
                    coverage.append({'seed':seed,'method':method,'uniform_covered':bool(sc.max()<=z),
                         'random_covered':bool(sc[info['random_action_index']].max()<=z),
                         'chosen_covered':bool(idx>=0 and sc[idx].max()<=z),'authorized':idx>=0})
        for alpha,z in qs.items():
            sensitivity.append(decision(seed,'alpha',select(m,s,z),t,m,s,o,{'setting':alpha,'quantile':z}))
        for n,z in qn.items():
            sensitivity.append(decision(seed,'calibration_size',select(m,s,z),t,m,s,o,{'setting':n,'quantile':z}))
        for n,z in qsub.items():
            ix=SUBSETS[n];local=select(m[ix],s[ix],z);idx=int(ix[local]) if local>=0 else -1
            tor=oracle(t[ix]);oi=int(ix[tor]) if tor>=0 else -1
            sensitivity.append(decision(seed,'library_size',idx,t,m,s,oi,{'setting':len(ix),'quantile':z}))
        # Interfaces change after commissioning, so these are frozen stale decisions.
        changed=replace(plant,gi=plant.gi*.65,eta=plant.eta*.85)
        selected=[idx for idx in ids.values() if idx>=0]
        if selected:
            tm=plant_outputs(changed,A[selected],info['ambient'])[0]
            lookup={idx:tm[j] for j,idx in enumerate(selected)}
        for method,idx in ids.items():
            if method=='Oracle':continue
            out={'seed':seed,'method':method,'authorized':idx>=0,
                 'unsafe':bool(idx>=0 and np.any(lookup[idx]<0))}
            for j,key in enumerate(['temperature','power','leakage']):out[key+'_failure']=bool(idx>=0 and lookup[idx][j]<0)
            shift.append(out)
        if seed in [30000,30001,30002]:
            np.savez_compressed(DATA/f'example_{seed}.npz',actions=A,mean=m,scale=s,truth=t,
                                state=p,covariance=c,observations=obs,ambient=info['ambient'])
        if (i+1)%100==0:print('Test',i+1,flush=True)
    pd.DataFrame(test_info).to_csv(DATA/'test_devices.csv',index=False)
    pd.DataFrame(rows).to_csv(DATA/'main_decisions.csv',index=False)
    pd.DataFrame(coverage).to_csv(DATA/'coverage.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(DATA/'sensitivities.csv',index=False)
    pd.DataFrame(shift).to_csv(DATA/'interface_change_stress.csv',index=False)
    np.savez_compressed(DATA/'commissioning_observations_and_fits.npz',observations=observations,
                        fitted=fitted,covariances=covariances,calibration_seeds=np.arange(20000,20399),
                        test_seeds=np.arange(30000,31000),probes=probes(),actions=A)
    timing={'main_wall_seconds':perf_counter()-t0,'calibration_devices':399,'test_devices':1000,
            'qualification_action_evaluations':399*len(A)}
    (DATA/'timing.json').write_text(json.dumps(timing,indent=2));print(timing,flush=True)

def run_ablation(name,noise_scale=1.,temperatures=True):
    # Disjoint independent units for a fresh scale/telemetry configuration.
    base=40000 if name in ['optical_only','matched_full'] else 50000+int(noise_scale*1000)
    cal=[];calinfo=[]
    for seed in range(base,base+199):
        plant,obs,p,c,m,s,t,score,info=profile(seed,noise_scale,temperatures)
        cal.append(info['maxscore']);calinfo.append(info)
    q=quantile(cal,.05);out=[];inf=[]
    for seed in range(base+10000,base+10300):
        plant,obs,p,c,m,s,t,score,info=profile(seed,noise_scale,temperatures)
        inf.append(info)
        for method,z in [('Proposed',q),('Plugin',0.),('Gaussian',float(norm.ppf(1-.05/3)))]:
            out.append(decision(seed,method,select(m,s,z),t,m,s,info['oracle_index'],
                       {'configuration':name,'noise_scale':noise_scale,'quantile':q,'uniform_covered':score.max()<=z}))
    pd.DataFrame(out).to_csv(DATA/(name+'_decisions.csv'),index=False)
    pd.DataFrame(calinfo).to_csv(DATA/(name+'_calibration.csv'),index=False)
    pd.DataFrame(inf).to_csv(DATA/(name+'_devices.csv'),index=False)
    print(name,'quantile',q,flush=True)

if __name__=='__main__':
    mode=sys.argv[1] if len(sys.argv)>1 else 'main'
    if mode=='main':run_main()
    elif mode=='ablation':run_ablation('optical_only',temperatures=False)
    elif mode=='noise':
        run_ablation('noise_half',noise_scale=.5)
        run_ablation('noise_double',noise_scale=2.)
    else:raise ValueError(mode)
