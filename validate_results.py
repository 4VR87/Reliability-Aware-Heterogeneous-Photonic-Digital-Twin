"""Replay both primary populations from retained observations/states and seeds.

This is a reproducibility audit of the implemented synthetic model, not a
second physical model or independent experimental validation.
"""
from pathlib import Path
from dataclasses import replace
import json
import numpy as np
import pandas as pd
from twin_core import (draw_plant, commission, proposed_library, predict,
                       plant_outputs, select, oracle, quantile)

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'

def close(a,b):
    np.testing.assert_allclose(a,b,rtol=2e-7,atol=2e-8,equal_nan=True)

def audit(prefix):
    changed=prefix=='refreshed'
    saved=np.load(DATA/f'{prefix}_commissioning.npz')
    cal=pd.read_csv(DATA/f'{prefix}_calibration.csv')
    test=pd.read_csv(DATA/f'{prefix}_devices.csv')
    decisions=pd.read_csv(DATA/f'{prefix}_decisions.csv').set_index(['seed','method'])
    coverage=pd.read_csv(DATA/f'{prefix}_coverage.csv').set_index(['seed','method'])
    th=json.loads((DATA/f'{prefix}_thresholds.json').read_text())
    stale=pd.read_csv(DATA/'primary_stale_interface.csv').set_index(['seed','method']) if not changed else None
    assert set(cal.seed).isdisjoint(set(test.seed))
    assert len(cal)==399 and len(test)==1000
    close(th['q_uniform'],quantile(cal.maxscore,.05))
    close(th['q_point'],quantile(cal.pointscore,.05))
    for alpha,q in th['alpha_quantiles'].items():close(q,quantile(cal.maxscore,float(alpha)))
    for n,q in th['size_quantiles'].items():close(q,quantile(cal.maxscore.iloc[:int(n)],.05))
    checked=0
    for part,frame in [('calibration',cal),('test',test)]:
        for j,record in enumerate(frame.itertuples(index=False)):
            seed=int(record.seed);plant=draw_plant(seed)
            if changed:plant=replace(plant,gi=plant.gi*.65,eta=plant.eta*.85)
            obs=commission(plant,seed+(1_000_000 if changed else 0))
            close(obs,saved[f'{part}_observations'][j])
            p=saved[f'{part}_fitted'][j];cov=saved[f'{part}_covariance'][j]
            a=proposed_library(p,record.ambient)
            assert 15<=len(a)<=105
            assert len(a)==record.candidate_count
            assert np.all(np.diff(a.sum(axis=1))>=-1e-11)
            mean,scale=predict(p,cov,a,record.ambient)
            truth=plant_outputs(plant,a,record.ambient)[0]
            assert np.isfinite(mean).all() and np.isfinite(scale).all() and (scale>0).all()
            sc=(mean-truth)/scale
            close(sc.max(),record.maxscore)
            close(sc[int(record.random_action_index)].max(),record.pointscore)
            if part=='test':
                fixed=np.flatnonzero(np.all(mean-np.array([2.,.15,.004])>=0,axis=1))
                choices={'Plugin':select(mean,scale,0.),'Gaussian':select(mean,scale,th['q_gaussian']),
                         'Pointwise':select(mean,scale,th['q_point']),
                         'Fixed margin':int(fixed[0]) if len(fixed) else -1,
                         'Proposed':select(mean,scale,th['q_uniform']),'Oracle':oracle(truth)}
                for name,ix in choices.items():
                    row=decisions.loc[(seed,name)]
                    assert int(row['index'])==ix
                    assert bool(row.authorized)==(ix>=0)
                    assert bool(row.unsafe)==bool(ix>=0 and np.any(truth[ix]<0))
                    if ix>=0:
                        close(a[ix].sum(),row.heat_w)
                        close(truth[ix],row[['temperature_margin','power_margin','leakage_margin']].to_numpy(dtype=float))
                    z={'Plugin':0.,'Gaussian':th['q_gaussian'],'Pointwise':th['q_point'],'Proposed':th['q_uniform']}.get(name)
                    if z is not None:
                        cr=coverage.loc[(seed,name)]
                        assert bool(cr.uniform_covered)==bool(sc.max()<=z)
                        assert bool(cr.random_covered)==bool(sc[int(record.random_action_index)].max()<=z)
                        assert bool(cr.chosen_covered)==bool(ix>=0 and sc[ix].max()<=z)
                    if stale is not None and name!='Oracle':
                        sr=stale.loc[(seed,name)]
                        fault=replace(plant,gi=plant.gi*.65,eta=plant.eta*.85)
                        bad=ix>=0 and np.any(plant_outputs(fault,a[ix:ix+1],record.ambient)[0]<0)
                        assert bool(sr.unsafe)==bool(bad)
                checked+=len(choices)
        print(prefix,part,'replayed',len(frame),flush=True)
    assert int(cal.candidate_count.sum())==th['qualification_responses']
    return {'calibration_units_replayed':len(cal),'test_units_replayed':len(test),
            'policy_decisions_replayed':checked,'qualification_responses':int(cal.candidate_count.sum()),
            'observations_scores_candidates_coverage_and_stale_outcomes':'passed'}

if __name__=='__main__':
    result={prefix:audit(prefix) for prefix in ['primary','refreshed']}
    result['scope']='Computational replay from stored fitted states; model fitting is not repeated.'
    (DATA/'reproducibility_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
