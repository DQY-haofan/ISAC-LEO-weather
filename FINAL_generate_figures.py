"""
=============================================================================
ISAC 2026 — FINAL Figure Generator (v5)
=============================================================================
Generates: fig2_crb_theory, fig3_identifiability, fig4_itu_validation,
           fig5_crb_rate_tradeoff

Usage:  python FINAL_generate_figures.py [--no-data]
  --no-data : skip Fig 4 (requires OpenSat4Weather netCDF files)

Requirements: numpy matplotlib scipy [netCDF4 for Fig 4]
=============================================================================
"""
import numpy as np, os, sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import matplotlib.gridspec as gridspec

# ========================= CONFIG =========================
DATA_DIR = r"F:\Github-repositories\ISAC-LEO-weather\OpenSat4Weather_data"
FIG_DIR  = r"F:\Github-repositories\ISAC-LEO-weather\conference_figures"
os.makedirs(FIG_DIR, exist_ok=True)
SKIP_DATA = '--no-data' in sys.argv
W = 7.16  # IEEE double-column width in inches

plt.rcParams.update({
    'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
    'font.size':8,'axes.labelsize':9,'legend.fontsize':7,
    'figure.dpi':300,'savefig.dpi':300,'savefig.bbox':'tight',
    'savefig.pad_inches':0.03,'axes.linewidth':0.6,
    'lines.linewidth':1.0,'lines.markersize':4,
    'grid.linewidth':0.4,'grid.alpha':0.3,'text.usetex':False,
})

# =================== ITU-R P.838-3 ===================
_ka=np.array([-5.33980,-0.35351,-0.23789,-0.94158])
_kb=np.array([-0.10008,1.26970,0.86036,0.64552])
_kc=np.array([1.13098,0.45400,0.15354,0.16817])
_km,_kn=0.83433,0.14298
_aa=np.array([-0.14318,0.29591,0.32177,-5.37610,16.1721])
_ab=np.array([1.82442,0.77564,0.63773,-0.96230,-3.29980])
_ac=np.array([-0.55187,0.19822,0.13164,1.47828,3.43990])
_am,_an=0.67849,-1.95537

def itu_ka(f):
    lf=np.log10(f)
    k=10**(np.sum(_ka*np.exp(-((lf-_kb)/_kc)**2))+_km*lf+_kn)
    a=np.sum(_aa*np.exp(-((lf-_ab)/_ac)**2))+_am*lf+_an
    return k,a

def spec_att(f,R): k,a=itu_ka(f); return k*max(R,0)**a
def rain_att(f,R,L): return spec_att(f,R)*L
def gas_att(f,rho=7.5): return 0.005*(f/10)**2.1+0.001*rho*(f/10)**1.8
def cloud_att(f,M=0.3): return 0.0005*f**1.95*M

def eff_path(R,hR,el):
    er=np.radians(max(el,5)); Ls=hR/np.sin(er); LG=Ls*np.cos(er)
    sa=spec_att(12.2,R)
    r=1/(1+0.78*np.sqrt(LG*sa/12.2)-0.38*(1-np.exp(-2*LG)))
    return Ls*np.clip(r,0.01,1.0)

def crb_multi(fs,R,L,sig,Np=1):
    J=sum(((lambda k,a: k*a*max(R,.01)**(a-1)*L)(*itu_ka(f)))**2 for f in fs)
    return sig**2/max(J,1e-30)

def joint_fim(fs,R,rho,M,G,Lr,Lg,Lc,sig,d=1e-5):
    th=[R,rho,M,G]; n=4
    def At(f,t):
        k,a=itu_ka(f)
        return k*max(t[0],.01)**a*Lr+gas_att(f,t[1])*Lg+cloud_att(f,t[2])*Lc+t[3]
    J=np.zeros((n,n))
    for f in fs:
        g=np.zeros(n)
        for p in range(n):
            tp,tm=list(th),list(th); h=max(abs(th[p])*d,d)
            tp[p]+=h; tm[p]-=h; g[p]=(At(f,tp)-At(f,tm))/(2*h)
        J+=np.outer(g,g)/sig**2
    return J

# ============== Hassibi 2003 rate (CORRECTED) ==============
def hassibi_rate(gamma, eta, Nsym):
    """Spectral efficiency with imperfect CSI.
    Hassibi & Hochwald, IEEE TIT 2003, eq. (7):
      gamma_eff = gamma^2 * eta*Nsym / (1 + gamma*eta*Nsym)
      C = (1-eta) * log2(1 + gamma_eff)
    """
    T = eta * Nsym
    gamma_eff = gamma**2 * T / (1 + gamma * T)
    return (1 - eta) * np.log2(1 + max(gamma_eff, 0))

# ========================= FIG 2 =========================
def fig2():
    print("Fig 2: CRB theory...")
    fig=plt.figure(figsize=(W,2.6)); gs=gridspec.GridSpec(1,2,wspace=0.35)
    Rs=np.linspace(1,100,200); L=3.0; sig=1.0

    ax=fig.add_subplot(gs[0,0])
    for fs,lab,ls in [([12.2],'Single freq. (12.2 GHz)','-'),
        ([10.7,11.2,11.7,12.2,12.7],'Ku-band (5 freq.)','-'),
        ([10.7,11.2,11.7,12.2,12.7,18,19,20],'Ku+Ka (8 freq.)','--')]:
        ax.semilogy(Rs,[np.sqrt(crb_multi(fs,R,L,sig)) for R in Rs],ls=ls,lw=1.2,label=lab)
    ax.semilogy(Rs,[np.sqrt(crb_multi([10.7,11.2,11.7,12.2,12.7],R,L,2.0)) for R in Rs],
                ':',lw=1,alpha=.7,color='red',label=r'Ku, $\sigma$=2 dB')
    ax.semilogy(Rs,Rs*.1,'k--',lw=.6,alpha=.4,label='10% rel.')
    ax.semilogy(Rs,Rs*.5,'k:',lw=.6,alpha=.4,label='50% rel.')
    ax.set(xlabel='Rain rate $R$ (mm/h)',ylabel='CRB RMSE of $\\hat{R}$ (mm/h)',
           xlim=[1,100],ylim=[.01,100])
    ax.legend(loc='upper left',fontsize=6.5); ax.grid(True,which='both')
    ax.text(.03,.03,'(a)',transform=ax.transAxes,fontsize=9,fontweight='bold')

    ax=fig.add_subplot(gs[0,1])
    Ls=np.linspace(.5,8,100); fk=[10.7,11.2,11.7,12.2,12.7]
    for Rv,c in [(5,'#1f77b4'),(10,'#ff7f0e'),(20,'#2ca02c'),(50,'#d62728')]:
        ax.semilogy(Ls,[np.sqrt(crb_multi(fk,Rv,l,sig)) for l in Ls],color=c,lw=1.2,
                    label=f'$R$={Rv}')
    Lg=3.1/np.sin(np.radians(38))
    ax.axvline(Lg,color='gray',ls=':',lw=.8); ax.text(Lg+.1,.015,'GEO 38°',fontsize=6,color='gray')
    ax.axvspan(3,3.5,alpha=.08,color='blue'); ax.text(3.05,50,'LEO\nzenith',fontsize=6,color='blue',alpha=.6)
    ax.set(xlabel='Eff. rain path $L_{\\mathrm{eff}}$ (km)',ylabel='CRB RMSE of $\\hat{R}$ (mm/h)',
           xlim=[.5,8],ylim=[.01,100])
    ax.legend(fontsize=6.5); ax.grid(True,which='both')
    ax.text(.03,.03,'(b)',transform=ax.transAxes,fontsize=9,fontweight='bold')

    for ext in ('pdf','png'): fig.savefig(os.path.join(FIG_DIR,f'fig2_crb_theory.{ext}'))
    plt.close(fig); print("  OK")

# ========================= FIG 3 =========================
def fig3():
    print("Fig 3: Identifiability...")
    fig=plt.figure(figsize=(W,2.6)); gs=gridspec.GridSpec(1,2,wspace=.35,width_ratios=[1.1,1])
    Lr,Lg,Lc=3.0,10.0,2.0; fp=np.linspace(10,25,300)

    ax=fig.add_subplot(gs[0,0])
    ax.semilogy(fp,[spec_att(f,20)*Lr for f in fp],'b-',lw=1.5,label='Rain ($R$=20)')
    ax.semilogy(fp,[gas_att(f,7.5)*Lg for f in fp],'r--',lw=1.2,label='Gas ($\\rho_{wv}$=7.5)')
    ax.semilogy(fp,[cloud_att(f,.3)*Lc for f in fp],'g:',lw=1.2,label='Cloud ($M_c$=0.3)')
    ax.semilogy(fp,[1]*len(fp),'k-.',lw=.8,alpha=.5,label='Gain $G$ (flat)')
    ax.axvspan(10.7,12.7,alpha=.08,color='blue'); ax.text(11,.005,'Ku',fontsize=7,color='blue')
    ax.axvspan(17.7,20.2,alpha=.08,color='orange'); ax.text(18.2,.005,'Ka',fontsize=7,color='orange')
    ax.set(xlabel='Frequency (GHz)',ylabel='Attenuation (dB)',xlim=[10,25],ylim=[.003,50])
    ax.legend(loc='upper left',fontsize=6.5); ax.grid(True,which='both')
    ax.text(.03,.03,'(a)',transform=ax.transAxes,fontsize=9,fontweight='bold')

    ax=fig.add_subplot(gs[0,1])
    ff=np.linspace(10.7,12.7,20); sig=1.0
    cfgs=[('$R$ only',[0]),('$R + G$',[0,3]),('$R + \\rho_{wv}$',[0,1]),
          ('$R + M_c$',[0,2]),('$R + \\rho + M$',[0,1,2]),('All unkn.',[0,1,2,3])]
    labs,vals=[],[]
    for lb,idx in cfgs:
        J=joint_fim(ff,20,7.5,.3,0,Lr,Lg,Lc,sig)
        Js=J[np.ix_(idx,idx)]
        try: rel=np.sqrt(abs(np.linalg.inv(Js)[0,0]))/20*100
        except: rel=1e5
        labs.append(lb); vals.append(rel)
    cols=['#2ca02c','#4daf4a','#ff7f00','#e6ab02','#e41a1c','#984ea3']
    yp=np.arange(len(labs))
    bars=ax.barh(yp,vals,color=cols,height=.55,edgecolor='black',linewidth=.3)
    ax.set_xscale('log'); ax.set_xlabel('CRB RMSE / $R$ (%)')
    ax.set_yticks(yp); ax.set_yticklabels(labs,fontsize=7); ax.invert_yaxis()
    ax.axvline(100,color='red',ls=':',lw=.8,alpha=.7)
    for b,v in zip(bars,vals):
        c='black' if v<100 else 'red'
        t=f'{v:.1f}%' if v<100 else f'{v:.0f}%'
        ax.text(v*1.4,b.get_y()+b.get_height()/2,t,va='center',fontsize=6.5,color=c)
    ax.set_xlim([.5,2e5]); ax.grid(True,which='major',axis='x')
    ax.text(5,5.6,'Identifiable',fontsize=6.5,color='green',style='italic')
    ax.text(500,5.6,'Unidentifiable',fontsize=6.5,color='red',style='italic')
    ax.text(.03,.03,'(b)',transform=ax.transAxes,fontsize=9,fontweight='bold')

    for ext in ('pdf','png'): fig.savefig(os.path.join(FIG_DIR,f'fig3_identifiability.{ext}'))
    plt.close(fig); print("  OK")

# ========================= FIG 4 =========================
def fig4():
    print("Fig 4: ITU validation (requires data)...")
    import netCDF4 as nc
    from scipy.interpolate import interp1d

    ds_s=nc.Dataset(os.path.join(DATA_DIR,'sml_data_2022.nc'))
    ds_r=nc.Dataset(os.path.join(DATA_DIR,'radar_along_sml_data_2022.nc'))
    rsl=np.ma.filled(ds_s.variables['rsl'][:],np.nan).astype(float); rsl[rsl<-100]=np.nan
    st=ds_s.variables['time'][:].astype(float)
    el=np.array(ds_s.variables['satellite_elevation'][:],dtype=float)
    d0=np.ma.filled(ds_s.variables['deg0l'][:],np.nan).astype(float); d0[d0<0]=np.nan
    rad=np.ma.filled(ds_r.variables['rainfall_amount'][:],np.nan).astype(float); rad[rad<0]=np.nan
    rt=ds_r.variables['time'][:].astype(float)
    ns,nst=rsl.shape; sidx=np.clip(np.searchsorted(st,rt),0,nst-1)

    def rbl(a,w=360,s=30):
        n=len(a); x,y=[],[]
        for i in range(0,n,s):
            c=a[max(0,i-w//2):min(n,i+w//2)]; v=c[~np.isnan(c)]
            if len(v)>50: x.append(i); y.append(np.percentile(v,97))
        if len(x)<2: return np.full(n,np.nan)
        return interp1d(x,y,fill_value='extrapolate')(np.arange(n))

    sub=np.linspace(0,ns-1,40,dtype=int); aa,ar,al=[],[],[]
    for ci,si in enumerate(sub):
        if ci%10==0: print(f"  SML {ci+1}/40...")
        bl=rbl(rsl[si,:])
        for t in range(len(rt)):
            s=sidx[t]; rv=rsl[si,s]; bv=bl[s]
            if np.isnan(rv) or np.isnan(bv): continue
            at=bv-rv
            if at<-2: continue
            rr=rad[t,si]
            if np.isnan(rr): continue
            rr*=12; dv=d0[si,s]
            if np.isnan(dv) or dv<100: dv=3000
            hr=dv/1000+.36
            Le=eff_path(rr,hr,el[si]) if rr>.1 else hr/np.sin(np.radians(el[si]))
            aa.append(max(at,0)); ar.append(rr); al.append(Le)
    aa,ar,al=np.array(aa),np.array(ar),np.array(al)
    ap=np.array([rain_att(12.2,r,l) for r,l in zip(ar,al)])
    ds_s.close(); ds_r.close()
    print(f"  Pairs: {len(aa):,}, RR>1: {np.sum(ar>1):,}")

    fig=plt.figure(figsize=(W,2.5)); gs=gridspec.GridSpec(1,3,wspace=.38)
    m=ar>.5

    # (a) scatter
    ax=fig.add_subplot(gs[0,0])
    ax.scatter(ap[m],aa[m],c=np.log10(np.maximum(ar[m],.1)),cmap='YlOrRd',s=1.5,alpha=.25,
               rasterized=True,vmin=-.5,vmax=2)
    ax.plot([0,15],[0,15],'k--',lw=.8,alpha=.5)
    be=np.arange(0,12,1); bx,by,bs=[],[],[]
    for i in range(len(be)-1):
        mb=(ap[m]>=be[i])&(ap[m]<be[i+1])
        if np.sum(mb)>20: bx.append((be[i]+be[i+1])/2); by.append(np.mean(aa[m][mb])); bs.append(np.std(aa[m][mb]))
    ax.errorbar(bx,by,bs,fmt='bs-',ms=4,lw=1.2,capsize=2,label='Binned mean ± std',zorder=5)
    v=(ap>.1)&(aa>0)&np.isfinite(ap)&np.isfinite(aa)
    ax.text(.05,.92,f'$r$ = {np.corrcoef(ap[v],aa[v])[0,1]:.3f}\nRMSE = {np.sqrt(np.mean((aa[v]-ap[v])**2)):.2f} dB',
            transform=ax.transAxes,fontsize=7,bbox=dict(boxstyle='round,pad=.3',fc='white',alpha=.9))
    ax.set(xlabel='Predicted atten. (dB)',ylabel='Measured atten. (dB)',xlim=[0,15],ylim=[0,15],aspect='equal')
    ax.legend(loc='lower right',fontsize=6); ax.grid(True)
    ax.text(.03,.03,'(a)',transform=ax.transAxes,fontsize=9,fontweight='bold')

    # (b) A vs R
    ax=fig.add_subplot(gs[0,1])
    rb=[.5,1,2,5,8,12,20,35,60,100]; rc,am,asd,pm=[],[],[],[]
    for i in range(len(rb)-1):
        mb=(ar>=rb[i])&(ar<rb[i+1])
        if np.sum(mb)>30:
            rc.append(np.mean(ar[mb])); am.append(np.mean(aa[mb]))
            asd.append(np.std(aa[mb])); pm.append(np.mean(ap[mb]))
    ax.errorbar(rc,am,asd,fmt='rs-',ms=4,lw=1,capsize=2,label='Measured')
    ax.plot(rc,pm,'b^--',ms=4,lw=1,label='ITU-R P.838+P.618')
    Rth=np.linspace(.5,100,200)
    ax.plot(Rth,[rain_att(12.2,R,eff_path(R,3.5,38)) for R in Rth],'b:',lw=.8,alpha=.5,label='P.838 cont.')
    ax.set(xlabel='Rain rate $R$ (mm/h)',ylabel='Attenuation (dB)',xlim=[0,80],ylim=[0,12])
    ax.legend(fontsize=6); ax.grid(True)
    ax.text(.03,.03,'(b)',transform=ax.transAxes,fontsize=9,fontweight='bold')

    # (c) CCDF
    ax=fig.add_subplot(gs[0,2])
    for (lo,hi),lb,cl in [((0,.5),'Dry','#1f77b4'),((.5,2),'0.5–2','#2ca02c'),
                            ((2,10),'2–10','#ff7f0e'),((10,50),'10–50','#d62728'),
                            ((50,300),'>50','#9467bd')]:
        mr=(ar>=lo)&(ar<hi)&(aa>=0)
        if np.sum(mr)>50:
            v=np.sort(aa[mr]); cdf=np.arange(1,len(v)+1)/len(v)
            ix=np.linspace(0,len(v)-1,min(500,len(v)),dtype=int)
            ax.plot(v[ix],1-cdf[ix],color=cl,lw=1,label=lb)
    ax.set(xlabel='Attenuation (dB)',ylabel='CCDF $P(A>x)$',xlim=[0,15],ylim=[1e-4,1])
    ax.set_yscale('log'); ax.legend(fontsize=6); ax.grid(True,which='both')
    ax.text(.03,.03,'(c)',transform=ax.transAxes,fontsize=9,fontweight='bold')

    for ext in ('pdf','png'): fig.savefig(os.path.join(FIG_DIR,f'fig4_itu_validation.{ext}'))
    plt.close(fig); print("  OK")

# ========================= FIG 5 =========================
def fig5():
    print("Fig 5: CRB-rate tradeoff + multi-link...")
    fig=plt.figure(figsize=(W,2.8)); gs=gridspec.GridSpec(1,2,wspace=.38)
    etas=np.linspace(.01,.5,200)
    f0=12.2; SNR0_dB=10; SNR0=10**(SNR0_dB/10); Nsym=302; L0=3.0

    # (a) Pareto frontier
    ax=fig.add_subplot(gs[0,0])
    for R,col,mk in [(5,'#1f77b4','o'),(10,'#ff7f0e','s'),(20,'#2ca02c','^'),(50,'#d62728','D')]:
        k,a=itu_ka(f0); AdB=k*R**a*L0; gamma=SNR0/10**(AdB/10)
        crbs,ses=[],[]
        for eta in etas:
            # CRB: sigma^2(eta) = C_dB / (gamma * eta * Nsym)
            C_dB=(10/np.log(10))**2*2
            sig2=C_dB/(max(gamma,.01)*max(eta*Nsym,1))
            dA=k*a*max(R,.01)**(a-1)*L0
            crbs.append(np.sqrt(sig2/max(dA**2,1e-30)))
            # Rate: Hassibi 2003 corrected formula
            ses.append(hassibi_rate(gamma, eta, Nsym))
        ax.plot(ses,crbs,color=col,lw=1.2,label=f'$R$={R} mm/h')
        for em in [.05,.1,.2]:
            i=np.argmin(np.abs(etas-em))
            ax.plot(ses[i],crbs[i],marker=mk,color=col,ms=5,
                    markeredgecolor='black',markeredgewidth=.3,zorder=5)

    # Annotate eta on R=20
    k20,a20=itu_ka(f0); gamma20=SNR0/10**(k20*20**a20*L0/10)
    for em in [.05,.1,.2]:
        C_dB=(10/np.log(10))**2*2
        sig2=C_dB/(max(gamma20,.01)*max(em*Nsym,1))
        dA=k20*a20*20**(a20-1)*L0
        cv=np.sqrt(sig2/max(dA**2,1e-30))
        sv=hassibi_rate(gamma20,em,Nsym)
        ax.annotate(f'$\\eta$={em}',xy=(sv,cv),xytext=(5,-6),textcoords='offset points',
                    fontsize=6,color='#2ca02c')

    ax.annotate('',xy=(3.2,.5),xytext=(1,.05),
                arrowprops=dict(arrowstyle='->',color='gray',lw=1.2,connectionstyle='arc3,rad=.2'))
    ax.text(2.6,.07,'Less pilot',fontsize=5.5,color='gray',style='italic',ha='center')
    ax.text(.7,.5,'More pilot',fontsize=5.5,color='gray',style='italic',ha='center')
    ax.set(xlabel='Spectral efficiency (bits/s/Hz)',ylabel='CRB RMSE of $\\hat{R}$ (mm/h)',
           xlim=[0,4],ylim=[.01,100])
    ax.set_yscale('log'); ax.legend(loc='upper right',fontsize=6.5); ax.grid(True,which='both')
    ax.text(.03,.03,'(a)',transform=ax.transAxes,fontsize=9,fontweight='bold')

    # (b) Multi-link
    ax=fig.add_subplot(gs[0,1])
    Ns=np.array([1,2,5,10,20,50,100,215])
    for lb,R,eta,cl,mk in [('$R$=10, $\\eta$=0.1',10,.1,'#1f77b4','o'),
                             ('$R$=20, $\\eta$=0.1',20,.1,'#2ca02c','s'),
                             ('$R$=20, $\\eta$=0.05',20,.05,'#d62728','^'),
                             ('$R$=50, $\\eta$=0.1',50,.1,'#ff7f0e','D')]:
        gamma_r=SNR0/10**(itu_ka(f0)[0]*R**itu_ka(f0)[1]*L0/10)
        C_dB=(10/np.log(10))**2*2
        sig2_1=C_dB/(max(gamma_r,.01)*max(eta*Nsym,1))
        k,a=itu_ka(f0); dA=k*a*max(R,.01)**(a-1)*L0
        crb1=sig2_1/max(dA**2,1e-30)
        ax.loglog(Ns,[np.sqrt(crb1/N) for N in Ns],marker=mk,color=cl,ms=4,lw=1,label=lb)
    ax.loglog(Ns,2/np.sqrt(Ns),'k--',lw=.8,alpha=.4,label=r'$\propto 1/\sqrt{N}$')
    ax.set(xlabel='Number of links $N$',ylabel='CRB RMSE of $\\hat{R}$ (mm/h)',
           xlim=[.8,300],ylim=[.001,10])
    ax.set_xticks(Ns); ax.set_xticklabels(['1','2','5','10','20','50','100','215'],fontsize=7)
    ax.legend(fontsize=6); ax.grid(True,which='both')
    ax.text(.03,.03,'(b)',transform=ax.transAxes,fontsize=9,fontweight='bold')

    for ext in ('pdf','png'): fig.savefig(os.path.join(FIG_DIR,f'fig5_crb_rate_tradeoff.{ext}'))
    plt.close(fig); print("  OK")

# ========================= MAIN =========================
if __name__=='__main__':
    print("="*60); print("ISAC 2026 — Figure Generator"); print("="*60)
    fig2(); fig3()
    if not SKIP_DATA:
        try: fig4()
        except Exception as e: print(f"  Fig 4 SKIPPED: {e}")
    else: print("Fig 4: SKIPPED (--no-data)")
    fig5()
    print("="*60)
    print(f"Output: {FIG_DIR}")
    for f in sorted(os.listdir(FIG_DIR)):
        if f.startswith('fig'): print(f"  {f}")
    print("="*60)
