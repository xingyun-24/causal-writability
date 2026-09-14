"""Regenerate F1, F2, and F4 with reserved title space after visual QA."""
import argparse, csv, json
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

INK='#272B33'; SECONDARY='#66707A'; INVALID='#C8CBCB'; LOW='#E76F61'; HIGH='#4C72B0'

mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Helvetica Neue','Helvetica','Arial','Liberation Sans','DejaVu Sans'],'font.size':7,'axes.titlesize':8.5,'axes.titleweight':'semibold','axes.labelsize':7.5,'xtick.labelsize':6.3,'ytick.labelsize':6.3,'legend.fontsize':6.5,'axes.edgecolor':INK,'axes.labelcolor':INK,'xtick.color':'#626970','ytick.color':'#626970','text.color':INK,'axes.facecolor':'white','figure.facecolor':'white','axes.grid':True,'grid.color':'#DDE0E2','grid.linewidth':.5,'grid.alpha':.8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white','savefig.bbox':None,'savefig.pad_inches':.04})

def rows(path):
    with Path(path).open(newline='', encoding='utf-8') as f: return list(csv.DictReader(f))
def num(row, key): return float(row[key])
def save(fig, out, stem):
    for ext in ('svg','pdf','png'):
        fig.savefig(out/f'{stem}.{ext}', dpi=600, bbox_inches=None)
    plt.close(fig)
def finish(fig, title):
    fig.suptitle(title, y=.985, fontsize=12, color=INK)
    fig.tight_layout(rect=(.0,.035,1,.93), pad=.8)

def f1(kit):
    data=rows(kit/'behavior_rollouts.csv'); fig,axes=plt.subplots(2,2,figsize=(3862/600,2419/600),sharex=True,sharey=True)
    for ri,band in enumerate(('low','high')):
        for ci,regime in enumerate(('Short','Long')):
            ax=axes[ri,ci]; group=[r for r in data if r['true_gravity_band']==band and r['history_regime']==regime]
            for r in group:
                valid=r['valid']=='True'; color=np.array([num(r,'future_r'),num(r,'future_g'),num(r,'future_b')])/255 if valid else INVALID
                ax.scatter(num(r,'hue_u'),num(r,'gravity_hat_e3'),s=5,c=[color],alpha=.55,marker='o' if valid else 'x',linewidths=.35)
            xs=sorted({num(r,'hue_u') for r in group}); med=[np.median([num(r,'gravity_hat_e3') for r in group if num(r,'hue_u')==x and r['valid']=='True']) for x in xs]
            ax.plot(xs,med,c=INK,lw=1,marker='o',ms=2); ax.axhspan(.003,.025,color='#F3D6D1',alpha=.45); ax.axhspan(.025,.045,color='#F8F6F1',alpha=.55); ax.axhspan(.045,.067,color='#D8E1F0',alpha=.45)
            ax.set_title(regime,fontsize=9,pad=7); ax.text(.02,.95,f"invalid: {sum(r['valid']!='True' for r in group)}/{len(group)}",transform=ax.transAxes,va='top',fontsize=6.2,color=INK)
            if ci==0: ax.set_ylabel(f"{band.title()}-gravity history\nDecoded gravity $\\hat g$",fontsize=8)
            if ri==1: ax.set_xlabel('Input cue (red → blue)',fontsize=8)
            ax.grid(color='#D9DEE3',lw=.45)
    finish(fig,'Free fall: appearance cue and observed history jointly shape decoded gravity')
    fig.text(.5,.012,'Raw rollouts and per-hue medians are descriptive; point color is measured future RGB.',ha='center',fontsize=6.2,color=SECONDARY)
    save(fig,kit/'figures','freefall_behavior')

def f2(kit):
    data=rows(kit/'coordinate_predictions.csv'); summary=json.loads((kit/'coordinate_summary.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(3844/600,1831/600))
    for j,rank in enumerate((1,2)):
        ax=axes[j]
        for band,color in (('low',LOW),('high',HIGH)):
            for split,marker in (('train','o'),('heldout','s')):
                group=[r for r in data if r['direction']==band and r['split']==split and int(r['coordinate'])==rank]
                ax.scatter([num(r,'observed_score') for r in group],[num(r,'predicted_score') for r in group],s=12,c=color,alpha=.45 if split=='train' else .82,marker=marker,label=f'{band} {split}')
        own=[r for r in data if int(r['coordinate'])==rank]; lo=min(min(num(r,'observed_score'),num(r,'predicted_score')) for r in own); hi=max(max(num(r,'observed_score'),num(r,'predicted_score')) for r in own)
        ax.plot([lo,hi],[lo,hi],c=INK,lw=.8); ax.set_title(f'Causal coordinate $z_{rank}$',fontsize=10,pad=7); ax.set_xlabel(f'Observed $z_{rank}$'); ax.set_ylabel(f'Fit-only predicted $z_{rank}$'); ax.grid(color='#D9DEE3',lw=.45)
    axes[0].text(.03,.95,'$\\hat z^{(r)}(g)=\\beta_0^{(r)}+\\beta_g^{(r)}g$',transform=axes[0].transAxes,va='top',fontsize=8)
    axes[1].legend(frameon=False,fontsize=6,loc='upper left',ncol=2)
    finish(fig,f"Free fall: direction-specific coordinate prediction (joint $R^2$: low {summary['models']['low']['heldout_joint_R2']:.3f}, high {summary['models']['high']['heldout_joint_R2']:.3f})")
    save(fig,kit/'figures','freefall_state_geometry')

def f4(kit):
    data=rows(kit/'layer_scan.csv'); fig,axes=plt.subplots(1,2,figsize=(3848/600,1768/600),sharey=True)
    for ax,band,color in zip(axes,('low','high'),(LOW,HIGH)):
        for site in range(30):
            group=[r for r in data if r['direction']==band and int(r['site_index'])==site and r['valid']=='True']
            ys=[num(r,'normalized_recovery_g') for r in group]
            ax.plot([site]*len(ys),ys,'o',ms=1.1,color=color,alpha=.12)
        means=[]
        for site in range(30):
            ys=[num(r,'normalized_recovery_g') for r in data if r['direction']==band and int(r['site_index'])==site and r['valid']=='True']
            means.append(np.mean(ys))
        ax.plot(range(30),means,c=color,lw=1.7); ax.axvspan(-.5,1.5,color='#F2C990',alpha=.30); ax.axhspan(.9,1.1,color='#CFE9E5',alpha=.75); ax.axhline(1,c='#16877A',lw=.8)
        ax.set_title(f'Target {band} | $L_{{50}}$=1',fontsize=9,pad=7); ax.set_xlabel('Residual site (after zero-based block)',fontsize=8); ax.grid(color='#D9DEE3',lw=.45)
    axes[0].set_ylabel('Normalized gravity recovery $R^g$',fontsize=8)
    finish(fig,'Free fall: direct writeability closes over an early depth interval')
    save(fig,kit/'figures','freefall_writeability')

def main():
    p=argparse.ArgumentParser();p.add_argument('kit',type=Path);a=p.parse_args(); f1(a.kit); f2(a.kit); f4(a.kit)
if __name__=='__main__': main()
