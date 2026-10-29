#!/usr/bin/env python3
"""Regenerate all four manuscript figures from analytical formulas and exact
saved record weights. Run from any directory; defaults to the repository root.
Figure 3 uses the full-domain result of Appendix D and true policy endpoints.
"""
import argparse, json, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator

BLUE='#1261A0'; ORANGE='#C25D24'; INK='#202020'; GRAY='#777777'; GREEN='#31735A'

def save(fig, stem, out):
    for suffix in ('pdf','png','svg'):
        fig.savefig(out/(stem+'.'+suffix),dpi=360,bbox_inches='tight',pad_inches=.06)
    plt.close(fig)

def panel(ax,label,title):
    ax.text(0,1.045,label,transform=ax.transAxes,ha='left',va='bottom',fontweight='bold',fontsize=11)
    ax.set_title(title,pad=10)

def boxed(ax,minor=True):
    for spine in ax.spines.values():
        spine.set_visible(True);spine.set_color('black');spine.set_linewidth(.9)
    ax.tick_params(which='both',direction='in',top=True,right=True)
    if minor:
        ax.xaxis.set_minor_locator(AutoMinorLocator(2));ax.yaxis.set_minor_locator(AutoMinorLocator(2))

from src.policies import probability_frontier, magic_n

def plot_full_domain(root,out):
    """Full-source-domain ceiling and exact fixed-instrument frontier."""
    from src.mixed_domain import response_gap as response
    cs=np.linspace(.001,.999,599)
    lo=np.zeros_like(cs);hi=np.ones_like(cs)
    # Bisection of the threshold equation from Appendix D.
    for _ in range(60):
        mid=(lo+hi)/2;positive=response(mid,cs)>0
        hi=np.where(positive,mid,hi);lo=np.where(positive,lo,mid)
    boundary=(lo+hi)/2;cplot=np.r_[0,cs,1];pplot=np.r_[1,boundary,1]
    fig,axes=plt.subplots(1,2,figsize=(7.1,3.3),layout='constrained');ax=axes[0]
    ax.fill_between(cplot,.6,pplot,color='#F0F0F0')
    ax.fill_between(cplot,pplot,1,color='#D6E5EF')
    ax.plot(cplot,pplot,color=INK,lw=1.3)
    ax.text(.58,.922,'Selected-event advantage',ha='center',fontsize=9,color=BLUE)
    ax.text(.5,.655,'Selected event does not exceed\nthe three-mixing bound',ha='center',fontsize=9,color=INK)
    ax.annotate(r'$F_4=F_{3,\mathrm{opt}}$',xy=(.83,float(np.interp(.83,cplot,pplot))),xytext=(.55,.735),
                fontsize=9,arrowprops=dict(arrowstyle='-',lw=.7,color=INK))
    ax.plot(.3,.9,marker='o',ms=3.8,color=BLUE,zorder=5)
    ax.annotate(r'$p=0.9,\ c=0.3$',xy=(.3,.9),xytext=(.035,.82),fontsize=8.2,
                arrowprops=dict(arrowstyle='-',lw=.6,color=BLUE),color=BLUE)
    ax.set(xlim=(0,1),ylim=(.6,1),xlabel=r'Pure-component overlap $c$',ylabel=r'Source Bloch length $p$')
    panel(ax,'(a)','Mixed-source separation');boxed(ax)
    ax=axes[1]
    sources=[(1.0,BLUE,'POLICY_p1_c0_5.npz'),(.9,ORANGE,'POLICY_p0_9_c0_5.npz')]
    for p_source,col,npzname in sources:
        d=np.load(root/'data'/npzname)
        tj=d['three_raw_J'];tp=np.trace(tj,axis1=1,axis2=2).real;tn=magic_n(tj)
        for pw,nw,ls in [(d['p'],d['n'],'-'),(tp,tn,'--')]:
            _,maxf,knots=probability_frontier(pw,nw,[])
            targets=np.unique(np.r_[np.linspace(.6,maxf,650),knots[(knots>=.6)&(knots<=maxf)],maxf])
            vals,_,_=probability_frontier(pw,nw,targets)
            mark='o' if p_source==1 else 's'
            ax.plot(targets,vals,color=col,ls=ls,lw=1.7,marker=mark,markevery=140,ms=2.6,mfc='white')
            ax.plot(maxf,vals[-1],marker=mark,ms=3.8,color=col,zorder=4)
            ax.plot([maxf,maxf],[vals[-1],0],color=col,ls=':',lw=.7)
    ax.set(xlim=(.6,.9),ylim=(0,.305),xlabel=r'Target Bell fidelity $F_{\rm tar}$',ylabel='Accepted probability per supplied trial')
    panel(ax,'(b)',r'Fixed circuits at $c=1/2$');boxed(ax)
    h1=[Line2D([],[],color=BLUE,marker='o',ms=3,label=r'$p=1$'),Line2D([],[],color=ORANGE,marker='s',ms=3,label=r'$p=0.9$')]
    h2=[Line2D([],[],color=INK,ls='-',label=r'$U_4$, midpoint $h_+$'),Line2D([],[],color=INK,ls='--',label=r'$U_3$, opposite $h_-$')]
    l1=ax.legend(handles=h1,loc='upper right',fontsize=8.6,ncol=2,columnspacing=.9,handlelength=1.6);ax.add_artist(l1)
    ax.legend(handles=h2,loc='upper right',bbox_to_anchor=(1,.9),fontsize=8.4,handlelength=2.5)
    save(fig,'FIGURE3',out)
    results=root/'results';results.mkdir(exist_ok=True)
    np.savetxt(results/'figure3_boundary.csv',np.column_stack([cplot,pplot]),
               delimiter=',',fmt='%.16g',
               header='Appendix D: 3*p*c = M(p,c)*(3+p^2*(2*c^2-1)-p*c); endpoint limits are 1.\nc,p_crit')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project-root',type=Path,default=ROOT)
    ap.add_argument('--output-dir',type=Path,default=ROOT/'figures')
    ap.add_argument('--figures',default='1,2,3,4',help='Comma-separated figure numbers to regenerate')
    args=ap.parse_args();root=args.project_root.resolve();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    plt.style.use(Path(__file__).with_name('photonic_journal.mplstyle'))
    selected={int(x) for x in args.figures.split(',')}
    if not selected <= {1,2,3,4}: raise ValueError('Choose figure numbers 1 through 4')

    if 1 in selected:
        # Figure 1: logical modes and polarization encoding.
        fig,axes=plt.subplots(2,1,figsize=(7.1,4.6),gridspec_kw={'height_ratios':[1.15,1]},layout='constrained')
        ax=axes[0];ax.set_xlim(-.85,7.55);ax.set_ylim(-.65,5.15);ax.axis('off')
        ax.text(-.8,5,'(a)',fontweight='bold',fontsize=11);ax.text(-.25,5,'Logical-mode circuit',fontsize=11)
        for i in range(5):
            y=4-i;ax.plot([.15,6.65],[y,y],color=INK,lw=.85,zorder=1)
            ax.text(-.03,y,rf'${i}$',ha='right',va='center')
            ax.text(7.02,y,('1','1','0','0','1')[i],ha='center',va='center',bbox=dict(boxstyle='square,pad=.16',ec=INK,fc='white',lw=.7),fontsize=9)
        ax.text(-.65,3.5,'A',ha='center',va='center');ax.plot([-.42,-.42],[3.05,3.95],color=INK,lw=.7)
        ax.text(-.65,1.5,'B',ha='center',va='center');ax.plot([-.42,-.42],[1.05,1.95],color=INK,lw=.7)
        ax.text(-.65,0,'H',ha='center',va='center');ax.text(7.02,4.53,'Counts',ha='center',fontsize=9)
        for x,i,j,name in [(1.25,0,2,'02'),(2.15,1,3,'13'),(3.9,1,4,'14'),(5.65,0,1,'01')]:
            lo,hi=4-j,4-i
            boundaries=[lo]+[y for y in range(lo+1,hi)]+[hi]
            for a,b in zip(boundaries[:-1],boundaries[1:]):
                ax.plot([x,x],[a+(0 if a==lo else .075),b-(0 if b==hi else .075)],color=BLUE,lw=1.5,zorder=2)
            for y in [lo,hi]:
                ax.add_patch(Rectangle((x-.09,y-.10),.18,.20,ec=BLUE,fc='white',lw=1.2,zorder=3))
                ax.plot([x-.06,x+.06],[y-.065,y+.065],color=BLUE,lw=.9,zorder=4)
            label_y=(lo+hi)/2+(.30 if name in ('02','13') else 0)
            ax.text(x+.13,label_y,rf'$B_{{{name}}}$',ha='left',va='center',fontsize=9,color=BLUE,
                    bbox=dict(fc='white',ec='none',pad=.8))
        for x,l in [(1.7,r'$T=1/2$'),(3.9,r'$T=1/3$'),(5.65,r'$T=1/2$')]:ax.text(x,-.47,l,ha='center',va='center',fontsize=9)
        for a,b,l in [(.9,2.5,'Data mixing'),(3.45,4.5,'Helper mixing'),(5.1,6.3,'Final mixing')]:
            ax.plot([a,b],[4.62,4.62],color=GRAY,lw=.6);ax.text((a+b)/2,4.69,l,ha='center',va='bottom',fontsize=9)
        ax=axes[1];ax.axis('off');ax.set_xlim(-.85,7.55);ax.set_ylim(-.75,3.2)
        ax.text(-.8,3.05,'(b)',fontweight='bold',fontsize=11);ax.text(-.25,3.05,'Equivalent polarization implementation',fontsize=11)
        for y,l in [(2,r'$a$: H/V'),(1,r'$b$: H/V'),(0,r'$h$: V')]:
            ax.plot([.25,5.45 if y in [1,2] else 6.6],[y,y],color=INK,lw=.85);ax.text(.1,y,l,ha='right',va='center',fontsize=9)
        for x,lo,hi,label,T in [(1.55,1,2,'NPBS',r'$T_H=T_V=1/2$'),(3.35,0,2,'PPBS',r'$T_H=1,\ T_V=1/3$')]:
            segments=[(lo,hi)] if lo==1 else [(0,.91),(1.09,2)]
            for a,b in segments:ax.plot([x,x],[a,b],color=BLUE,lw=1.5)
            for y in [lo,hi]:
                ax.add_patch(Rectangle((x-.11,y-.12),.22,.24,fc='white',ec=BLUE,lw=1.1,zorder=4))
                ax.plot([x-.08,x+.08],[y-.08,y+.08],color=BLUE,lw=.9,zorder=5)
            ax.text(x,2.42,label,ha='center',fontsize=9)
            ax.text(x,-.42,T,ha='center',fontsize=9)
        x=4.65
        ax.add_patch(Polygon([(x-.10,1.78),(x+.11,1.90),(x+.10,2.22),(x-.11,2.10)],ec=BLUE,fc='white',lw=1.2,zorder=4))
        ax.text(x,2.42,'HWP',ha='center',fontsize=9);ax.text(x,1.53,r'$22.5^\circ$',ha='center',fontsize=9)
        for y in [1,2]:
            ax.add_patch(Rectangle((5.45,y-.18),.58,.36,fc='white',ec=INK,lw=.8,zorder=4));ax.text(5.74,y,'H/V',ha='center',va='center',fontsize=8,zorder=5)
            for dy in [-.16,.16]:ax.plot([6.03,6.23,6.65],[y,y+dy,y+dy],color=INK,lw=.85)
        ax.text(7,2.16,'0: 1',ha='center',va='center',fontsize=9);ax.text(7,1.84,'1: 1',ha='center',va='center',fontsize=9)
        ax.text(7,1.16,'2: 0',ha='center',va='center',fontsize=9);ax.text(7,.84,'3: 0',ha='center',va='center',fontsize=9)
        ax.text(7,0,'4: 1',ha='center',va='center',fontsize=9);ax.text(7,2.62,'Mode: count',ha='center',fontsize=9)
        ax.text(5.7,-.5,r'$h_H$: uncoupled vacuum',ha='center',fontsize=9,color=GRAY)
        save(fig,'FIGURE1',out)

    if 2 in selected:
        # Figure 2: pure-source fidelity and conditional Bell error.
        fig,axes=plt.subplots(1,2,figsize=(7.1,2.9),layout='constrained')
        c=np.linspace(0,1,501);f=(1+c+c*c)/(2-c+2*c*c);ax=axes[0]
        ax.plot(c,f,color=BLUE,label=r'$U_4$: selected event')
        ax.plot(c,(1+c)/2,color=INK,ls='--',label='Three-mixing bound')
        ax.plot(c,(1+c*c)/2,color=GRAY,ls=':',label='Bare ceiling')
        ax.set(xlim=(0,1),ylim=(.49,1.01),xlabel=r'Wave-packet overlap $c$',ylabel='Conditional Bell fidelity')
        panel(ax,'(a)','Pure-source fidelity');boxed(ax);ax.legend(loc='lower right',fontsize=8.5,handlelength=2.5)
        eps=np.geomspace(1e-5,.45,300);cc=np.sqrt(1-2*eps);err=(1-cc)**2/(2-cc+2*cc*cc);ax=axes[1]
        ax.loglog(eps,err,color=BLUE,label=r'$U_4$: exact')
        ax.loglog(eps,(1-cc)/2,color=INK,ls='--',label='Three-mixing minimum')
        # The dotted asymptote uses the same color as the exact error curve.
        ax.loglog(eps,eps**2/3,color=BLUE,ls=':',lw=1.1,label=r'Asymptote $\epsilon^2/3$')
        ax.set(xlabel=r'Bare Bell error $\epsilon$',ylabel='Conditional Bell error',xlim=(1e-5,.5),ylim=(2e-11,1))
        panel(ax,'(b)','Quadratic conditional error');boxed(ax,False);ax.legend(loc='upper left',fontsize=8.5,handlelength=2.5)
        save(fig,'FIGURE2',out)

    if 3 in selected:
        plot_full_domain(root,out)

    if 4 in selected:
        # Figure 4: physical-comparison curves from Appendix F.
        d=json.loads((root/'data/FIGURE4_PLOT_INPUTS.json').read_text());fig,axes=plt.subplots(1,2,figsize=(7.1,4.0))
        fig.subplots_adjust(left=.085,right=.99,bottom=.32,top=.90,wspace=.29)
        entries=[('bare_curve',GRAY,':','Bare frontier'),('three_attainer_curve',INK,'--','Three-mixing circuit'),('four_curve',BLUE,'-',r'Fixed $U_4$ frontier'),('common_center_curve',ORANGE,'-.','Gaussian: common center'),('independent_center_curve',ORANGE,'-','Gaussian: independent centers')]
        handles=[]
        for j,ax in enumerate(axes):
            for key,color,ls,label in entries:
                ar=np.array(d[key]);line,=ax.plot(ar[:,0],ar[:,1],color=color,ls=ls,lw=1.8 if key=='four_curve' else 1.5,label=label)
                if j==0:handles.append(line)
                if key in ['bare_curve','three_attainer_curve','four_curve']:
                    ax.plot(ar[-1,0],ar[-1,1],marker='o',ms=3.5,color=color)
            two=d['two_helper_point']['P_attempt'];ideal=d['rank_one_point']['P_attempt']
            line,=ax.plot([.7,1],[two,two],color=GREEN,ls=(0,(4,2,1,2)),lw=1.5,label='Two helpers: feasible rate')
            star=ax.scatter([1],[ideal],color=INK,marker='D',s=27,zorder=5,label='Ideal rank-one projection')
            ax.plot([1],[two],marker='s',ms=3.5,color=GREEN)
            ax.plot([1],[d['independent_center_curve'][-1][1]],marker='o',ms=3.5,color=ORANGE)
            if j==0:handles.extend([line,star])
            ax.set_xlabel(r'Target Bell fidelity $F_{\rm tar}$');boxed(ax,False)
        ax=axes[0];ax.set(xlim=(.7,1.007),ylim=(0,.5),ylabel='Accepted probability per source attempt')
        ax.set_xticks([.7,.8,.9,1]);ax.xaxis.set_minor_locator(AutoMinorLocator(5));ax.yaxis.set_minor_locator(AutoMinorLocator(2));panel(ax,'(a)','Comparison overview')
        common=np.array(d['common_center_curve']);ax.plot([common[-1,0],1],[common[-1,1],0],color=ORANGE,ls='-.',lw=1.5)
        ax=axes[1];ax.set(xlim=(.955,1.0015),ylim=(.0008,.3),yscale='log',ylabel='Accepted probability per source attempt');ax.set_xticks([.96,.97,.98,.99,1]);ax.xaxis.set_minor_locator(AutoMinorLocator(2));panel(ax,'(b)','High fidelity (log scale)')
        fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.53,.005),ncol=2,columnspacing=1.5,handlelength=3.2,fontsize=8.8)
        save(fig,'FIGURE4',out)
    print('Figures written to', out)

if __name__=='__main__': main()
