"""Coupled thermal/optical synthetic plant and deliberately reduced circuit twin.

Temperatures are in degrees C, conductances W/K, heat W, optical power mW.
No measured hardware dataset is represented by this model.
"""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.optimize import least_squares

TREF = 25.0
BETA = 0.0035
KAPPA = np.array([0.32, 0.30])
NOISE = np.array([0.25, 0.25, 0.03, 0.03])
FLOORS = np.array([0.35, 0.035, 0.0015])
LIMITS = np.array([75.0, 4.5, 0.02])
PNOM = np.array([np.log(.0035), np.log(.004), np.log(.04), 0., 0., np.log(.82)])
PRIOR_SD = np.array([.8, .6, .6, 100., .25, .6])
LOW = np.array([np.log(.0009), np.log(.0015), np.log(.012), -np.pi-1., np.log(.65), np.log(.35)])
HIGH = np.array([np.log(.012), np.log(.009), np.log(.1), np.pi+1., np.log(1.5), np.log(.999)])
CAPACITY = np.array([.006, .0008, .0008, .04])

@dataclass
class Plant:
    gi: float
    gh: float
    gs: float
    phase: float
    ks: float
    eta: float
    beta: float
    chi_i: float
    chi_h: float
    gamma: float
    arm_ratio: float

def draw_plant(seed):
    r = np.random.default_rng(seed)
    return Plant(gi=.0035*np.exp(.30*r.normal()), gh=.004*np.exp(.15*r.normal()),
                 gs=.04*np.exp(.20*r.normal()), phase=r.uniform(-np.pi,np.pi),
                 ks=np.exp(.06*r.normal()), eta=1/(1+np.exp(-(1.51635+.28*r.normal()))),
                 beta=np.clip(BETA+.0005*r.normal(), .0015, .0055),
                 chi_i=r.uniform(.02,.12), chi_h=r.uniform(0,.04),
                 gamma=r.normal(0,.00025), arm_ratio=r.uniform(.92,1.))

def action_library(step=.003):
    q = np.round(np.arange(.09,.16001,.005), 6)
    h = np.round(np.arange(0,.06001,step), 6)
    a=np.stack(np.meshgrid(q,h,h,indexing='ij'),axis=-1).reshape(-1,3)
    # Stable order supplies a deterministic tie-break after minimum total heat.
    return a[np.lexsort((a[:,2],a[:,1],a[:,0],a.sum(axis=1)))]

def probes():
    return np.array([[q,h1,h2] for q in [.075,.09,.105]
                     for h1,h2 in [(0.,0.),(.018,0.),(0.,.018),(.024,.012)]])

def proposed_library(p,ambient):
    """Fixed physics recipe; it sees commissioning data through p, never truth.

    For each source level: source-only plus the first reachable dark-fringe
    branch on either heater at three fixed target offsets. At most 105 actions.
    """
    r=np.linalg.inv(thermal_matrix(*np.exp(p[:3])))
    v=np.exp(p[4])*(KAPPA[0]*r[1,:3]-KAPPA[1]*r[2,:3])
    out=[]
    for q in np.round(np.arange(.09,.16001,.005),6):
        base=p[3]+np.exp(p[4])*(KAPPA[0]-KAPPA[1])*(ambient-TREF)+v[0]*q
        out.append((q,0.,0.))
        for arm in [1,2]:
            for offset in [-.12,0.,.12]:
                options=[]
                for branch in range(-3,4):
                    h=(np.pi+2*np.pi*branch+offset-base)/v[arm]
                    if 0<=h<=.0600000001:options.append(float(h))
                if options:
                    a=[float(q),0.,0.];a[arm]=min(options);out.append(tuple(a))
    a=np.unique(np.round(out,12),axis=0)
    return a[np.lexsort((a[:,2],a[:,1],a[:,0],a.sum(axis=1)))]

def heat_vector(actions):
    a=np.atleast_2d(actions)
    return np.column_stack([a,np.zeros(len(a))])

def thermal_matrix(gi,gh,gs):
    gi,gh,gs=np.broadcast_arrays(gi,gh,gs)
    k=np.zeros(gi.shape+(4,4),dtype=float)
    edges=[(0,3,gi),(1,3,gh),(2,3,1.08*gh),(0,1,.00025),
           (0,2,.00020),(1,2,.00060)]
    for i,j,g in edges:
        k[...,i,i]+=g;k[...,j,j]+=g;k[...,i,j]-=g;k[...,j,i]-=g
    k[...,3,3]+=gs
    return k

def twin_state(p,actions,ambient):
    gi,gh,gs=np.exp(p[:3])
    k=thermal_matrix(gi,gh,gs)
    theta=np.linalg.solve(k,heat_vector(actions).T).T
    return ambient+theta

def source_power(q,temp,beta=BETA):
    return 10.*np.maximum(q-.04,0)/.10*np.exp(-beta*(temp-TREF))

def optical_response(temp,actions,phase,ks,eta,beta=BETA,gamma=0.,arm_ratio=1.):
    tr=np.asarray(temp)-TREF
    phi=phase+ks*(KAPPA[0]*tr[:,1]-KAPPA[1]*tr[:,2])+gamma*(tr[:,1]**2-tr[:,2]**2)
    src=source_power(np.atleast_2d(actions)[:,0],temp[:,0],beta)
    bright=eta*src*(1+arm_ratio**2-2*arm_ratio*np.cos(phi))/4
    dark=eta*src*(1+arm_ratio**2+2*arm_ratio*np.cos(phi))/4
    leakage=dark/np.maximum(bright+dark,1e-14)
    return bright,dark,leakage,phi

def twin_outputs(p,actions,ambient):
    t=twin_state(p,actions,ambient)
    bright,dark,leak,phi=optical_response(t,actions,p[3],np.exp(p[4]),np.exp(p[5]))
    margins=np.column_stack([LIMITS[0]-t.max(axis=1),bright-LIMITS[1],LIMITS[2]-leak])
    obs=np.column_stack([t[:,0],t[:,1],bright,dark])
    return margins,obs,t,phi

def plant_state(plant,actions,ambient,return_residual=False):
    q=heat_vector(actions)
    t=np.linalg.solve(thermal_matrix(plant.gi,plant.gh,plant.gs),q.T).T
    for it in range(80):
        gi=plant.gi/(1+plant.chi_i*(t[:,0]-t[:,3])/30.)
        gh=plant.gh/(1+plant.chi_h*(t[:,1]+t[:,2])/60.)
        k=thermal_matrix(gi,gh,np.full(len(t),plant.gs))
        new=np.linalg.solve(k,q[...,None])[...,0]
        delta=np.max(np.abs(new-t))
        t=.8*new+.2*t
        if delta<2e-10:break
    if delta>=1e-7:raise RuntimeError(f'Nonlinear thermal fixed point failed: {delta}')
    if return_residual:
        gi=plant.gi/(1+plant.chi_i*(t[:,0]-t[:,3])/30.)
        gh=plant.gh/(1+plant.chi_h*(t[:,1]+t[:,2])/60.)
        res=np.einsum('nij,nj->ni',thermal_matrix(gi,gh,np.full(len(t),plant.gs)),t)-q
        return ambient+t,float(np.max(np.abs(res)))
    return ambient+t

def plant_outputs(plant,actions,ambient):
    t=plant_state(plant,actions,ambient)
    bright,dark,leak,phi=optical_response(t,actions,plant.phase,plant.ks,plant.eta,
                        plant.beta,plant.gamma,plant.arm_ratio)
    margins=np.column_stack([LIMITS[0]-t.max(axis=1),bright-LIMITS[1],LIMITS[2]-leak])
    obs=np.column_stack([t[:,0],t[:,1],bright,dark])
    return margins,obs,t,phi

def commission(plant,seed,noise_scale=1.):
    r=np.random.default_rng(seed+6_000_000)
    exact=plant_outputs(plant,probes(),TREF)[1]
    return exact+r.normal(size=exact.shape)*NOISE*noise_scale

def fit_twin(observations,noise_scale=1.,temperatures=True):
    a=probes();sd=NOISE*noise_scale
    columns=np.arange(4) if temperatures else np.array([2,3])
    def residual(p):
        obs=twin_outputs(p,a,TREF)[1]
        likelihood=((obs-observations)/sd)[:,columns].ravel()
        return np.r_[likelihood,(p-PNOM)/PRIOR_SD]
    # A fixed coarse phase scan prevents the zero derivative at an intensity extremum.
    starts=[]
    for phase in np.linspace(-np.pi,np.pi,25,endpoint=False):
        p=PNOM.copy();p[3]=phase;starts.append((float(residual(p)@residual(p)),p))
    starts.sort(key=lambda x:x[0])
    fits=[least_squares(residual,p,bounds=(LOW,HIGH),method='trf',
                       ftol=1e-9,xtol=1e-9,gtol=1e-8,max_nfev=100) for _,p in starts[:2]]
    f=min(fits,key=lambda x:x.cost)
    precision=f.jac.T@f.jac
    covariance=np.linalg.inv(precision)
    return f.x,covariance,{'cost':float(2*f.cost),'success':bool(f.success),
           'nfev':int(f.nfev),'jac_condition':float(np.linalg.cond(precision))}

def predict(p,covariance,actions,ambient,structured_scale=True):
    mean=twin_outputs(p,actions,ambient)[0]
    jac=np.empty(mean.shape+(6,))
    eps=2e-5
    for j in range(6):
        plus=p.copy();minus=p.copy();plus[j]+=eps;minus[j]-=eps
        jac[...,j]=(twin_outputs(plus,actions,ambient)[0]-twin_outputs(minus,actions,ambient)[0])/(2*eps)
    var=np.einsum('nji,ik,njk->nj',jac,covariance,jac)
    # Floors reflect unmodelled discrepancy only through a prespecified scale;
    # the order-statistic calibration, not the covariance, gives coverage.
    if structured_scale:
        # Physics-shaped discrepancy scales are frozen before calibration.
        # They are normalizers, not a claim that discrepancy is Gaussian.
        t=twin_state(p,actions,ambient);tr=t-TREF
        gi,gh,gs=np.exp(p[:3]);k=thermal_matrix(gi,gh,gs)
        theta=t-ambient
        edge_i=gi*(theta[:,0]-theta[:,3])/30.
        edge_h=gh*(theta[:,1]+theta[:,2])/60.
        dk_i=thermal_matrix(-edge_i,np.zeros(len(t)),np.zeros(len(t)))
        dk_h=thermal_matrix(np.zeros(len(t)),-edge_h,np.zeros(len(t)))
        # Remove the fixed lateral edges: this is the derivative of vertical
        # interface conductances only, not a second thermal graph.
        lateral=thermal_matrix(0.,0.,0.)
        dk_i-=lateral;dk_h-=lateral
        dt_i=-np.linalg.solve(k,np.einsum('nij,nj->ni',dk_i,theta).T).T
        dt_h=-np.linalg.solve(k,np.einsum('nij,nj->ni',dk_h,theta).T).T
        dominant=np.argmax(t,axis=1)
        var[:,0]+=(.03*dt_i[np.arange(len(t)),dominant])**2+(.02*dt_h[np.arange(len(t)),dominant])**2
        bright,dark,leak,phi=optical_response(t,actions,p[3],np.exp(p[4]),np.exp(p[5]))
        total=bright+dark
        quadratic=(tr[:,1]**2-tr[:,2]**2)*.00025
        var[:,1]+=(tr[:,0]*bright*.0005)**2+(total*.5*np.sin(phi)*quadratic)**2
        var[:,2]+=(.5*np.sin(phi)*quadratic)**2
        # Retain curvature at an interferometric extremum, where the first
        # derivative of port power vanishes although phase error still matters.
        pj=np.empty((len(t),6))
        for j in range(6):
            plus=p.copy();minus=p.copy();plus[j]+=eps;minus[j]-=eps
            pj[:,j]=(twin_outputs(plus,actions,ambient)[3]-twin_outputs(minus,actions,ambient)[3])/(2*eps)
        phase_var=np.maximum(np.einsum('ni,ij,nj->n',pj,covariance,pj),0)+quadratic**2
        curvature=phase_var**2/8.
        var[:,2]+=curvature;var[:,1]+=total**2*curvature
    scale=np.sqrt(np.maximum(var,0)+FLOORS**2)
    return mean,scale

def quantile(scores,alpha):
    x=np.asarray(scores);rank=int(np.ceil((len(x)+1)*(1-alpha)))
    return np.inf if rank>len(x) else max(0.,float(np.sort(x)[rank-1]))

def select(mean,scale,q):
    safe=np.all(mean-q*scale>=0,axis=1)
    ids=np.flatnonzero(safe)
    return int(ids[0]) if len(ids) else -1

def oracle(truth):
    ids=np.flatnonzero(np.all(truth>=0,axis=1))
    return int(ids[0]) if len(ids) else -1

def reduced_infeasibility_bound(p,ambient):
    # Optimistic bound ignores heater heat and grants perfect interference.
    r00=np.linalg.inv(thermal_matrix(*np.exp(p[:3])))[0,0]
    cap=min(.16,(LIMITS[0]-ambient)/r00)
    if cap<.09:return 0.
    optimum=min(cap,.04+1/(BETA*r00))
    return float(np.exp(p[5])*source_power(optimum,ambient+r00*optimum))

def plant_dict(p):return asdict(p)
