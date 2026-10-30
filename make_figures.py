from pathlib import Path
from io import BytesIO
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Polygon, PathPatch
from matplotlib.path import Path as MPath
from twin_core import *

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data';FIG=ROOT/'figures';FIG.mkdir(exist_ok=True)
S=json.loads((DATA/'results_summary.json').read_text())
BLUE='#1675A9';GREEN='#008D71';ORANGE='#D16B16';RED='#B74A42';PURPLE='#8555A5';GRAY='#65727C'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.titlesize':13,
 'axes.labelsize':12,'legend.fontsize':10,'axes.spines.top':False,'axes.spines.right':False,
 'savefig.facecolor':'white','figure.facecolor':'white','axes.titlepad':10,
 'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})

def save(fig,name):
    for ext in ['png','pdf','svg']:
        b=BytesIO();fig.savefig(b,format=ext,dpi=300,bbox_inches='tight',pad_inches=.08)
        (FIG/(name+'.'+ext)).write_bytes(b.getvalue())
    plt.close(fig)

def box(ax,x,y,w,h,text,fc='#EEF4F8',ec='#3C556B',fs=12):
    p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.04,rounding_size=.07',facecolor=fc,edgecolor=ec,lw=1.5)
    ax.add_patch(p);ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs);return p

def arrow(ax,start,end,color=GRAY,lw=1.6,style='-|>'):
    ax.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':style,'color':color,'lw':lw,'shrinkA':3,'shrinkB':3})

def architecture():
    fig,axs=plt.subplots(1,2,figsize=(12,5.8),gridspec_kw={'width_ratios':[1,1.2]})
    ax=axs[0];ax.set(xlim=(0,8),ylim=(0,8));ax.axis('off');ax.set_title('(a) Heterogeneous integration concept',loc='left')
    for xy,w,h,fc,label in [((.6,1.1),6.7,1.1,'#B8C4D0','Heat spreader and ambient sink'),
       ((.8,2.35),6.3,1.0,'#DCE6EE',''),
       ((1.0,3.5),2.8,.35,'#E5C486','Bond layer'),
       ((1.0,4.0),2.8,.9,'#BBD9D0','III-V gain chiplet')]:
        ax.add_patch(Rectangle(xy,w,h,facecolor=fc,edgecolor='#3C556B',lw=1.4))
        ax.text(xy[0]+w/2,xy[1]+h/2,label,ha='center',va='center',fontsize=10 if h>.5 else 9)
    ax.add_patch(Rectangle((4.6,3.45),2.4,.25,facecolor=ORANGE,edgecolor='none'))
    ax.text(5.8,3.98,'Thermal actuator',ha='center',fontsize=11)
    ax.text(2.1,2.54,'Silicon PIC',ha='center',va='center',fontsize=10)
    for x in [1.45,3.35]:arrow(ax,(x,4.0),(x,3.30),RED)
    arrow(ax,(6.8,3.45),(6.8,2.05),RED)
    arrow(ax,(4.0,1.1),(4.0,.3),RED)
    ax.text(4.45,.45,'Heat removal',fontsize=10)
    ax.plot([3.8,4.15,4.5,6.6],[4.45,4.45,2.9,2.9],color=BLUE,lw=3)
    ax.text(5.1,5.25,'Optical transition',ha='center',fontsize=11,color=BLUE)
    arrow(ax,(5.1,5.05),(4.13,4.4),BLUE)
    ax.text(.9,6.75,'Two interfaces, two uncertainties',fontsize=12,fontweight='bold')
    ax.text(.9,5.98,'Bond conductance controls temperature.\nOptical coupling controls delivered power.',fontsize=11,linespacing=1.6)
    ax.text(.9,7.5,'Conceptual stack; dimensions are not to scale.',fontsize=9,color=GRAY)
    ax=axs[1];ax.set(xlim=(0,10),ylim=(0,8));ax.axis('off');ax.set_title('(b) Circuit and monitored constraints',loc='left')
    box(ax,.2,4.0,1.7,1.0,'III-V\nsource',fc='#E4F2EA')
    ax.plot([1.95,3.0],[4.5,4.5],color=BLUE,lw=3)
    for upper in [True,False]:
        yy=5.65 if upper else 3.35
        verts=[(3.,4.5),(3.45,4.5),(3.45,yy),(4.,yy),(7.,yy),(7.55,yy),(7.55,4.5),(8.,4.5)]
        codes=[MPath.MOVETO,MPath.CURVE4,MPath.CURVE4,MPath.CURVE4,MPath.LINETO,MPath.CURVE4,MPath.CURVE4,MPath.CURVE4]
        ax.add_patch(PathPatch(MPath(verts,codes),fill=False,color=BLUE,lw=3))
    ax.plot([8.,9.5],[4.5,5.1],color=GREEN,lw=2.8)
    ax.plot([8.,9.5],[4.5,3.9],color=GRAY,lw=2.8)
    for x,y,label in [(4.5,5.9,'Heater 1'),(4.5,2.82,'Heater 2')]:
        ax.add_patch(Rectangle((x,y),1.65,.25,facecolor=ORANGE))
        ax.text(x+.825,y+.51 if y>4 else y-.3,label,ha='center',fontsize=10)
    for x,y,n in [(1.05,5.35,'T0'),(5.95,5.65,'T1')]:
        ax.add_patch(Circle((x,y),.24,color=RED));ax.text(x,y,n,ha='center',va='center',color='white',fontsize=9)
    ax.text(9.6,5.3,'Bright',ha='right',fontsize=10,color=GREEN)
    ax.text(9.6,3.6,'Dark',ha='right',fontsize=10,color=GRAY)
    box(ax,.45,.5,8.8,1.4,'Authorize only when all limits pass\nTemperature ≤ 75 °C    Bright power ≥ 4.5 mW\nDark-port fraction ≤ 2%',fc='#F4F7F9',fs=11)
    ax.text(4.8,7.25,'Temperature + two-port optical telemetry',ha='center',fontsize=11)
    fig.subplots_adjust(wspace=.12);save(fig,'figure_1_architecture')

def flowchart():
    fig,axs=plt.subplots(1,2,figsize=(11.5,6.6))
    for ax in axs:ax.set(xlim=(0,10),ylim=(0,12));ax.axis('off')
    ax=axs[0];ax.set_title('(a) Qualification before deployment',loc='left')
    blocks=[(10.1,'Freeze physics model, scales\nand candidate-generation recipe'),
            (7.75,'Commission independent\ncalibration devices'),
            (5.4,'Generate each candidate set\nQualify all three margins'),
            (3.05,'One maximum score per device\nAcross candidates and constraints'),
            (.7,'Freeze the conformal quantile\nRecord population and scope')]
    for y,txt in blocks:box(ax,.65,y,8.4,1.25,txt,fs=11)
    for top,bottom in zip(blocks[:-1],blocks[1:]):arrow(ax,(4.85,top[0]),(4.85,bottom[0]+1.25))
    ax=axs[1];ax.set_title('(b) A new device operating episode',loc='left')
    for y,txt in [(10.1,'Measure 12 commissioning settings\nFit positive thermal and optical twin'),
                  (7.9,'Generate ≤ 105 candidate settings\nPredict joint lower margin bounds')]:
        box(ax,.65,y,8.4,1.25,txt,fs=11)
    arrow(ax,(4.85,10.1),(4.85,9.15))
    diamond=Polygon([[4.85,7.3],[8.0,6.05],[4.85,4.8],[1.7,6.05]],closed=True,facecolor='#FFF6DD',edgecolor='#9A7D30',lw=1.5)
    ax.add_patch(diamond);ax.text(4.85,6.05,'Any candidate has\nthree nonnegative\nmargin bounds?',ha='center',va='center',fontsize=10.5)
    arrow(ax,(4.85,7.9),(4.85,7.3))
    box(ax,.3,2.85,4.3,1.25,'Choose minimum heat\nAuthorize service',fc='#E4F2EA',fs=10.5)
    box(ax,5.15,2.85,4.3,1.25,'Withhold service\nNo feasible setting',fc='#F9E9E6',fs=10.5)
    arrow(ax,(2.85,5.25),(2.45,4.1),GREEN);arrow(ax,(6.85,5.25),(7.3,4.1),RED)
    ax.text(1.9,4.8,'Yes',color=GREEN,fontsize=10);ax.text(7.65,4.8,'No',color=RED,fontsize=10)
    box(ax,.65,.45,8.4,1.25,'Interface population changes?\nStop using the old qualification\nRecommission and requalify',fc='#EDF1F5',fs=10.5)
    arrow(ax,(2.45,2.85),(4.6,1.7));arrow(ax,(7.3,2.85),(5.1,1.7))
    fig.subplots_adjust(wspace=.13);save(fig,'figure_2_flowchart')

def transient():
    z=np.load(DATA/'physical_transients.npz');fig,axs=plt.subplots(1,2,figsize=(11.5,4.2))
    for j,label in enumerate(['III-V source','Silicon arm 1','Silicon arm 2','Spreader']):axs[0].plot(z['time'],z['temperature'][:,j],label=label,lw=2)
    axs[0].set(xlabel='Time after heat step (s)',ylabel='Temperature (°C)',title='(a) Coupled thermal response',xlim=(0,25))
    axs[0].axvline(S['physics_validation']['settle_1percent_seconds'],color=GRAY,ls='--',lw=1)
    axs[0].legend(frameon=False,fontsize=9)
    axs[1].plot(z['time'],z['bright'],color=GREEN,label='Bright port',lw=2)
    axs[1].plot(z['time'],z['dark'],color=GRAY,label='Dark port',lw=2)
    axs[1].plot(z['time'],z['bright']+z['dark'],color=BLUE,ls='--',label='Two-port sum',lw=1.8)
    axs[1].set(xlabel='Time after heat step (s)',ylabel='Optical power (mW)',title='(b) Thermal and optical coupling',xlim=(0,25))
    axs[1].legend(frameon=False,fontsize=9)
    for ax in axs:ax.grid(alpha=.17)
    fig.tight_layout(w_pad=2);save(fig,'figure_3_physics')

def comparisons():
    vals=S['primary'];names=[v['method'] for v in vals];colors=[GRAY,ORANGE,PURPLE,'#A68E39',BLUE,GREEN]
    labels=[x.replace('Fixed margin','Fixed\nmargin') for x in names]
    fig,axs=plt.subplots(1,2,figsize=(11.5,4.4))
    for ax,key,cikey,title in [(axs[0],'availability','availability_ci','(a) Service availability'),
                (axs[1],'unsafe_authorization','unsafe_authorization_ci','(b) Unsafe authorizations')]:
        y=np.array([v[key] for v in vals])*100
        ci=np.array([v[cikey] for v in vals])*100
        ax.bar(np.arange(len(vals)),y,color=colors,width=.64)
        ax.errorbar(np.arange(len(vals)),y,yerr=np.array([y-ci[:,0],ci[:,1]-y]),fmt='none',color='black',capsize=3,lw=1)
        ax.set_xticks(np.arange(len(vals)),labels,rotation=25,ha='right')
        ax.set_ylabel('Independent devices (%)');ax.set_title(title,loc='left');ax.grid(axis='y',alpha=.15)
        ax.set_ylim(0,82 if key=='availability' else 9)
        for i,v in enumerate(y):ax.text(i,ci[i,1]+(1.5 if key=='availability' else .25),f'{v:.2f}',ha='center',fontsize=9)
    fig.tight_layout(w_pad=2);save(fig,'figure_4_reliability')

def coverage():
    names=['Gaussian','Pointwise','Proposed'];d={x['method']:x for x in S['primary_coverage']}
    fig,axs=plt.subplots(1,2,figsize=(11.5,4.3));x=np.arange(3);w=.32
    for k,key in enumerate(['random_coverage','uniform_coverage']):
        axs[0].bar(x+(k-.5)*w,[100*d[n][key] for n in names],width=w,color=[GRAY,BLUE][k],label=['One random setting','Entire candidate set'][k])
    axs[0].axhline(95,color=RED,ls='--',lw=1.4,label='95% target')
    axs[0].set(xticks=x,xticklabels=names,ylabel='Device coverage (%)',ylim=(0,108),title='(a) Different coverage questions')
    axs[0].legend(frameon=False,fontsize=9,loc='lower right');axs[0].grid(axis='y',alpha=.15)
    scores=np.sort(pd.read_csv(DATA/'primary_devices.csv').maxscore)
    axs[1].step(scores,np.arange(1,len(scores)+1)/len(scores),color=BLUE,lw=2,label='Test maximum score')
    th=S['primary_thresholds']
    for n,key,c in [('Pointwise','q_point',PURPLE),('Gaussian','q_gaussian',ORANGE),('Proposed','q_uniform',GREEN)]:
        axs[1].axvline(th[key],ls='--',lw=1.5,color=c,label=f'{n}: {th[key]:.2f}')
    axs[1].axhline(.95,color=RED,ls=':',lw=1)
    axs[1].set(xlabel='Maximum normalized adverse error',ylabel='Empirical cumulative fraction',xlim=(0,5),ylim=(0,1.02),title='(b) Action-uniform score distribution')
    axs[1].legend(frameon=False,fontsize=8.5,loc='lower right');axs[1].grid(alpha=.15)
    fig.tight_layout(w_pad=2);save(fig,'figure_5_coverage')

def sensitivity():
    df=pd.DataFrame(S['primary_sensitivity']);fig,axs=plt.subplots(1,2,figsize=(11.5,4.2))
    for ax,kind,title,xlabel in [(axs[0],'alpha','(a) Reliability and availability','Target α (%)'),
                               (axs[1],'calibration_size','(b) One nested calibration realization','Calibration devices')]:
        sub=df[df.kind==kind].sort_values('setting');xx=sub.setting*(100 if kind=='alpha' else 1)
        ax.plot(xx,sub.availability*100,'o-',color=BLUE,label='Availability',lw=2)
        ax.set(xlabel=xlabel,ylabel='Authorized devices (%)',title=title,ylim=(45,65));ax.grid(alpha=.15)
        other=ax.twinx()
        if kind=='alpha':
            other.plot(xx,sub.unsafe_authorization*100,'s--',color=RED,label='Unsafe authorization',lw=1.5)
            other.set(ylabel='Unsafe devices (%)',ylim=(0,2.))
        else:
            other.plot(xx,sub['quantile'],'s--',color=PURPLE,label='Calibration quantile',lw=1.5)
            other.set(ylabel='Calibrated multiplier',ylim=(1.8,2.7))
        lines=ax.get_lines()+other.get_lines();ax.legend(lines,[l.get_label() for l in lines],frameon=False,fontsize=9,loc='upper left')
    fig.tight_layout(w_pad=2);save(fig,'figure_6_sensitivity')

def cost_telemetry():
    fig,axs=plt.subplots(1,2,figsize=(11.5,4.3));th=S['primary_thresholds']
    values=[399*6615,th['qualification_responses']]
    bars=axs[0].bar([0,1],values,color=[GRAY,BLUE],width=.55)
    axs[0].set(xticks=[0,1],xticklabels=['Dense grid\n6615 / device','Physics proposal\n≤ 105 / device'],yscale='log',ylim=(1e4,1e7),ylabel='Qualification response evaluations',title='(a) Acquisition burden for 399 devices')
    for b,v in zip(bars,values):axs[0].text(b.get_x()+b.get_width()/2,v*1.2,f'{v:,}',ha='center',fontsize=10)
    vals=[next(v for v in S[k] if v['method']=='Proposed') for k in ['optical_only','matched_full']]
    y=np.array([v['availability'] for v in vals])*100;ci=np.array([v['availability_ci'] for v in vals])*100
    axs[1].bar([0,1],y,color=[ORANGE,BLUE],width=.55)
    axs[1].errorbar([0,1],y,yerr=[y-ci[:,0],ci[:,1]-y],fmt='none',color='black',capsize=4)
    axs[1].set(xticks=[0,1],xticklabels=['Two optical\nchannels','Two optical + two\ntemperature channels'],ylim=(0,75),ylabel='Authorized devices (%)',title='(b) Matched dense-grid telemetry ablation')
    for j,v in enumerate(vals):axs[1].text(j,y[j]+8,f"{y[j]:.2f}%\nUnsafe: {v['unsafe_n']}/300",ha='center',fontsize=10)
    for ax in axs:ax.grid(axis='y',alpha=.15)
    fig.tight_layout(w_pad=2);save(fig,'figure_7_budget_and_telemetry')

def lifecycle():
    fig,axs=plt.subplots(1,2,figsize=(11.5,4.5))
    p=next(v for v in S['primary'] if v['method']=='Proposed');r=next(v for v in S['refreshed'] if v['method']=='Proposed')
    avail=[p['availability'],p['availability'],r['availability']];unsafe=[0,p['availability'],r['unsafe_authorization']]
    xx=np.arange(3);w=.32
    axs[0].bar(xx-w/2,np.array(avail)*100,width=w,color=BLUE,label='Authorized')
    axs[0].bar(xx+w/2,np.array(unsafe)*100,width=w,color=RED,label='Unsafe authorization')
    axs[0].set(xticks=xx,xticklabels=['Nominal','Stale\ninterfaces','Recommissioned\nand requalified'],ylabel='Independent devices (%)',ylim=(0,70),title='(a) Qualification has a limited scope')
    axs[0].legend(frameon=False,fontsize=9);axs[0].grid(axis='y',alpha=.15)
    chosen=None
    for seed in [80000,80001,80002,80003,80004]:
        z=np.load(DATA/f'primary_example_{seed}.npz');idx=select(z['mean'],z['scale'],S['primary_thresholds']['q_uniform'])
        if idx>=0:chosen=(seed,z,idx);break
    if chosen is None:raise RuntimeError('No predefined example authorizes')
    seed,z,idx=chosen;true=z['truth'];temperature=75-true[:,0];power=4.5+true[:,1];leak=.02-true[:,2]
    sc=axs[1].scatter(temperature,power,c=leak*100,cmap='viridis',s=25,vmin=0,vmax=8,alpha=.8)
    axs[1].scatter(temperature[idx],power[idx],marker='*',s=190,c=ORANGE,edgecolor='black',label='Authorized setting',zorder=5)
    axs[1].axvline(75,color=RED,ls='--',lw=1);axs[1].axhline(4.5,color=RED,ls='--',lw=1)
    axs[1].set(xlabel='Maximum temperature (°C)',ylabel='Bright-port power (mW)',title=f'(b) One candidate set, device {seed}')
    axs[1].legend(frameon=False,fontsize=8.5,loc='lower right');axs[1].grid(alpha=.15)
    cb=fig.colorbar(sc,ax=axs[1],fraction=.045,pad=.03,extend='max');cb.set_label('Dark-port fraction (%)',fontsize=10)
    (DATA/'figure_example_seed.json').write_text(json.dumps({'seed':seed,'selected_index':int(idx)}))
    fig.tight_layout(w_pad=2);save(fig,'figure_8_lifecycle')

if __name__=='__main__':
    for f in [architecture,flowchart,transient,comparisons,coverage,sensitivity,cost_telemetry,lifecycle]:f()
    print('Created eight figures in PNG, PDF and editable SVG formats.')
