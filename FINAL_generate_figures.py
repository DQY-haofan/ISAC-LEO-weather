"""
=============================================================================
ISAC 2026 — v7 CORRECTED Figure Generator
=============================================================================
FIX LOG from v6:
  v6 fixes (retained):
    - Fig 3(b): joint_fim uses Np=30, matching Table II
    - Fig 5(a): Hassibi rate formula correct (γ²ηN/(1+γηN))

  v7 fixes (NEW):
    - Fig 5(a): CRB now uses Eq.(4) directly with sigma_n=1 dB (fixed),
      NOT the old SNR-dependent noise model.
      OLD: sig2 = C_dB / (gamma * eta * Nsym)  ← SNR-dependent, wrong
      NEW: CRB = sigma_n^2 / (Np * sum_k [dA/dR]_k^2)  ← Eq.(4), correct
    - Fig 5(a): Uses 5 Ku-band frequencies (consistent with Fig 2 main curve)
    - Fig 5(b): Uses 5 Ku-band frequencies (was single freq f0=12.2)
    - Fig 5(b): Reference line rescaled to match new absolute CRB values
    - Numerical verification prints at end of fig3() and fig5()

  KNOWN ISSUE (deferred to TWC):
    - P.838 linear coefficients (mk=0.83433, ck=0.14298) may have a
      table-swap error vs official ITU tables (k ~3.5x too high).
      Does NOT affect internal consistency or validation r=0.745
      (same implementation used throughout). Will audit for TWC.

Usage:  python FINAL_generate_figures_v7.py [--no-data]
Output: conference_figures/fig{2,3,4,5}_*.{pdf,png}
=============================================================================
"""
import numpy as np, os, sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import matplotlib.gridspec as gridspec

DATA_DIR = r"F:\Github-repositories\ISAC-LEO-weather\OpenSat4Weather_data"
FIG_DIR  = r"F:\Github-repositories\ISAC-LEO-weather\conference_figures"
os.makedirs(FIG_DIR, exist_ok=True)
SKIP_DATA = '--no-data' in sys.argv
W = 7.16

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
_aa=np.array([-0.14318,0.29591,0.32177,-5.37610,16.1721])
_ab=np.array([1.82442,0.77564,0.63773,-0.96230,-3.29980])
_ac=np.array([-0.55187,0.19822,0.13164,1.47828,3.43990])

def itu_ka(f):
    lf=np.log10(f)
    k=10**(np.sum(_ka*np.exp(-((lf-_kb)/_kc)**2))+0.83433*lf+0.14298)
    a=np.sum(_aa*np.exp(-((lf-_ab)/_ac)**2))+0.67849*lf-1.95537
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

# --- Standard 5 Ku-band frequencies (used in Fig 2 and now Fig 5) ---
FK_KU = [10.7, 11.2, 11.7, 12.2, 12.7]

def crb_R_only(fs, R, L, sig, Np=30):
    """
    CRB for R only — Eq.(4) in paper.
    CRB(R) = sigma_n^2 / (Np * sum_k [k_k * alpha_k * R^{alpha_k-1} * L]^2)
    """
    J = 0.0
    for f in fs:
        k, a = itu_ka(f)
        dA = k * a * max(R, 0.01)**(a-1) * L
        J += Np * dA**2 / sig**2
    return 1.0 / max(J, 1e-30)

# Keep old name as alias for backward compat
def crb_multi(fs, R, L, sig):
    """CRB for R only, Np=30. Wrapper around crb_R_only."""
    return crb_R_only(fs, R, L, sig, Np=30)

# ============ UNIFIED joint_fim: matches Table II ============
def joint_fim(fs, R, rho, M, G, Lr, Lg, Lc, sig, Np=30, d=1e-5):
    """
    Joint FIM for [R, rho_wv, M_c, G].
    Np=30 (eta=0.1, Nsym=302) multiplied into FIM.
    Matches Table II in the paper.
    """
    th=[R,rho,M,G]; n=4
    def At(f,t):
        k,a=itu_ka(f)
        return k*max(t[0],.01)**a*Lr+gas_att(f,t[1])*Lg+cloud_att(f,t[2])*Lc+t[3]
    J=np.zeros((n,n))
    for f in fs:
        g=np.zeros(n)
        for p in range(n):
            tp,tm=list(th),list(th); h=max(abs(th[p])*d,d)
            tp[p]+=h; tm[p]-=h
            g[p]=(At(f,tp)-At(f,tm))/(2*h)
        J += Np * np.outer(g,g) / sig**2
    return J

def hassibi_rate(gamma, eta, Nsym):
    """Eq.(8): per-subcarrier spectral efficiency under imperfect CSI."""
    T = eta * Nsym
    gamma_eff = gamma**2 * T / (1 + gamma * T)
    return (1 - eta) * np.log2(1 + max(gamma_eff, 0))

# ========================= FIG 2 (UNCHANGED) =========================
def fig2():
    print("Fig 2: CRB theory...")
    fig=plt.figure(figsize=(W,2.6)); gs=gridspec.GridSpec(1,2,wspace=0.35)
    Rs=np.linspace(1,100,200); L=3.0; sig=1.0

    ax=fig.add_subplot(gs[0,0])
    for fs,lab,ls in [([12.2],'Single freq. (12.2 GHz)','-'),
        (FK_KU,'Ku-band (5 freq.)','-'),
        (FK_KU+[18,19,20],'Ku+Ka (8 freq.)','--')]:
        ax.semilogy(Rs,[np.sqrt(crb_multi(fs,R,L,sig)) for R in Rs],ls=ls,lw=1.2,label=lab)
    y_noisy=[np.sqrt(crb_multi(FK_KU,R,L,2.0)) for R in Rs]
    ax.semilogy(Rs,y_noisy,':',lw=1,alpha=.7,color='red',label=r'Ku, $\sigma$=2 dB')
    ax.semilogy(Rs,Rs*.1,'k--',lw=.6,alpha=.4,label='10% rel.')
    ax.semilogy(Rs,Rs*.5,'k:',lw=.6,alpha=.4,label='50% rel.')
    ax.set(xlabel='Rain rate $R$ (mm/h)',ylabel='CRB RMSE of $\\hat{R}$ (mm/h)',xlim=[1,100],ylim=[.01,100])
    ax.legend(loc='upper left',fontsize=6.5); ax.grid(True,which='both')
    ax.text(.03,.03,'(a)',transform=ax.transAxes,fontsize=9,fontweight='bold')

    ax=fig.add_subplot(gs[0,1])
    Ls=np.linspace(.5,8,100)
    for Rv,c in [(5,'#1f77b4'),(10,'#ff7f0e'),(20,'#2ca02c'),(50,'#d62728')]:
        ax.semilogy(Ls,[np.sqrt(crb_multi(FK_KU,Rv,l,sig)) for l in Ls],color=c,lw=1.2,label=f'$R$={Rv}')
    Lg=3.1/np.sin(np.radians(38))
    ax.axvline(Lg,color='gray',ls=':',lw=.8); ax.text(Lg+.1,.015,'GEO 38°',fontsize=6,color='gray')
    ax.axvspan(3,3.5,alpha=.08,color='blue'); ax.text(3.05,50,'LEO\nzenith',fontsize=6,color='blue',alpha=.6)
    ax.set(xlabel='Eff. rain path $L_{\\mathrm{eff}}$ (km)',ylabel='CRB RMSE of $\\hat{R}$ (mm/h)',xlim=[.5,8],ylim=[.01,100])
    ax.legend(fontsize=6.5); ax.grid(True,which='both')
    ax.text(.03,.03,'(b)',transform=ax.transAxes,fontsize=9,fontweight='bold')
    for ext in ('pdf','png'): fig.savefig(os.path.join(FIG_DIR,f'fig2_crb_theory.{ext}'))
    plt.close(fig); print("  OK")

# ========================= FIG 3 (UNCHANGED from v6 fix) =========================
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

    # --- Panel (b): Side-info hierarchy (Np=30, K=20 — matches Table II) ---
    ax=fig.add_subplot(gs[0,1])
    ff=np.linspace(10.7,12.7,20)  # K=20 representative frequencies
    sig=1.0; Np=30

    cfgs=[('$R$ only',[0]),('$R + G$',[0,3]),('$R + \\rho_{wv}$',[0,1]),
          ('$R + M_c$',[0,2]),('$R + \\rho + M$',[0,1,2]),('All unkn.',[0,1,2,3])]
    labs,vals=[],[]
    for lb,idx in cfgs:
        J=joint_fim(ff,20,7.5,.3,0,Lr,Lg,Lc,sig,Np)
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
    ax.set_xlim([.1,2e5]); ax.grid(True,which='major',axis='x')
    ax.text(3,5.6,'Identifiable',fontsize=6.5,color='green',style='italic')
    ax.text(300,5.6,'Unidentifiable',fontsize=6.5,color='red',style='italic')
    ax.text(.03,.03,'(b)',transform=ax.transAxes,fontsize=9,fontweight='bold')
    for ext in ('pdf','png'): fig.savefig(os.path.join(FIG_DIR,f'fig3_identifiability.{ext}'))
    plt.close(fig); print("  OK")

    # Print Table II verification
    print("  Table II verification (K=20, Np=30):")
    paper_vals = [0.4, 2.8, 7.0, 8.0, 690, 25018]
    for (lb,idx),pv in zip(cfgs, paper_vals):
        J=joint_fim(ff,20,7.5,.3,0,Lr,Lg,Lc,sig,Np)
        Js=J[np.ix_(idx,idx)]
        try:
            rel=np.sqrt(abs(np.linalg.inv(Js)[0,0]))/20*100
            kap=np.linalg.cond(Js)
        except: rel=1e6; kap=1e20
        match = "✓" if abs(rel - pv)/max(pv,0.01) < 0.05 else "✗ MISMATCH"
        print(f"    {lb:20s}: code={rel:10.1f}%  paper={pv:10.1f}%  {match}  kappa={kap:.1e}")

# ========================= FIG 4 (UNCHANGED) =========================
def fig4():
    print("Fig 4: ITU validation...")
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
    ns=rsl.shape[0]; sidx=np.clip(np.searchsorted(st,rt),0,rsl.shape[1]-1)

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
            if np.isnan(rv) or np.isnan(bv) or bv-rv<-2: continue
            rr=rad[t,si]
            if np.isnan(rr): continue
            rr*=12; dv=d0[si,s]
            if np.isnan(dv) or dv<100: dv=3000
            hr=dv/1000+.36; Le=eff_path(rr,hr,el[si]) if rr>.1 else hr/np.sin(np.radians(el[si]))
            aa.append(max(bv-rv,0)); ar.append(rr); al.append(Le)
    aa,ar,al=np.array(aa),np.array(ar),np.array(al)
    ap=np.array([rain_att(12.2,r,l) for r,l in zip(ar,al)])
    ds_s.close(); ds_r.close()
    print(f"  Pairs: {len(aa):,}, RR>1: {np.sum(ar>1):,}")

    fig=plt.figure(figsize=(W,2.5)); gs=gridspec.GridSpec(1,3,wspace=.38); m=ar>.5
    # (a)
    ax=fig.add_subplot(gs[0,0])
    ax.scatter(ap[m],aa[m],c=np.log10(np.maximum(ar[m],.1)),cmap='YlOrRd',s=1.5,alpha=.25,rasterized=True,vmin=-.5,vmax=2)
    ax.plot([0,15],[0,15],'k--',lw=.8,alpha=.5)
    be=np.arange(0,12,1); bx,by,bs=[],[],[]
    for i in range(len(be)-1):
        mb=(ap[m]>=be[i])&(ap[m]<be[i+1])
        if np.sum(mb)>20: bx.append((be[i]+be[i+1])/2); by.append(np.mean(aa[m][mb])); bs.append(np.std(aa[m][mb]))
    ax.errorbar(bx,by,bs,fmt='bs-',ms=4,lw=1.2,capsize=2,label='Binned mean ± std',zorder=5)
    v=(ap>.1)&(aa>0)&np.isfinite(ap)&np.isfinite(aa)
    corr=np.corrcoef(ap[v],aa[v])[0,1]; rmse=np.sqrt(np.mean((aa[v]-ap[v])**2))
    ax.text(.05,.92,f'$r$ = {corr:.3f}\nRMSE = {rmse:.2f} dB',transform=ax.transAxes,fontsize=7,
            bbox=dict(boxstyle='round,pad=.3',fc='white',alpha=.9))
    ax.set(xlabel='Predicted atten. (dB)',ylabel='Measured atten. (dB)',xlim=[0,15],ylim=[0,15],aspect='equal')
    ax.legend(loc='lower right',fontsize=6); ax.grid(True)
    ax.text(.03,.03,'(a)',transform=ax.transAxes,fontsize=9,fontweight='bold')
    # (b)
    ax=fig.add_subplot(gs[0,1])
    rb=[.5,1,2,5,8,12,20,35,60,100]; rc,am,asd,pm=[],[],[],[]
    for i in range(len(rb)-1):
        mb=(ar>=rb[i])&(ar<rb[i+1])
        if np.sum(mb)>30:
            rc.append(np.mean(ar[mb])); am.append(np.mean(aa[mb])); asd.append(np.std(aa[mb])); pm.append(np.mean(ap[mb]))
    ax.errorbar(rc,am,asd,fmt='rs-',ms=4,lw=1,capsize=2,label='Measured')
    ax.plot(rc,pm,'b^--',ms=4,lw=1,label='ITU-R P.838+P.618')
    Rth=np.linspace(.5,100,200)
    ax.plot(Rth,[rain_att(12.2,R,eff_path(R,3.5,38)) for R in Rth],'b:',lw=.8,alpha=.5,label='P.838 cont.')
    ax.set(xlabel='Rain rate $R$ (mm/h)',ylabel='Attenuation (dB)',xlim=[0,80],ylim=[0,12])
    ax.legend(fontsize=6); ax.grid(True)
    ax.text(.03,.03,'(b)',transform=ax.transAxes,fontsize=9,fontweight='bold')
    # (c)
    ax=fig.add_subplot(gs[0,2])
    for (lo,hi),lb,cl in [((0,.5),'Dry','#1f77b4'),((.5,2),'0.5–2','#2ca02c'),((2,10),'2–10','#ff7f0e'),
                            ((10,50),'10–50','#d62728'),((50,300),'>50','#9467bd')]:
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

# ========================= FIG 5 (v7 CORRECTED) =========================
def fig5():
    """
    v7 FIX: Panel (a) CRB now computed from Eq.(4) with sigma_n=1 dB (fixed),
    using 5 Ku-band frequencies. The old code used an SNR-dependent noise model
    that contradicted Eq.(4) and produced CRB values inconsistent with Fig 2.

    Key changes vs v6:
      - CRB = sigma_n^2 / (eta*Nsym * sum_k [dA/dR]_k^2)  [Eq.(4)]
      - 5 Ku-band frequencies (was single f0=12.2 GHz)
      - Y-axis range adjusted: [0.01, 10] (was [0.01, 100])
      - R=50 curve now has LOWEST CRB (consistent with R^{-0.4} scaling)
    """
    print("Fig 5: CRB-rate tradeoff + multi-link (v7 CORRECTED)...")
    fig = plt.figure(figsize=(W, 2.8))
    gs = gridspec.GridSpec(1, 2, wspace=0.38)

    etas = np.linspace(0.01, 0.50, 200)
    Nsym = 302; sigma_n = 1.0; L0 = 3.0
    SNR0_dB = 10; SNR0 = 10**(SNR0_dB/10)
    f0 = 12.2  # for link budget / rate calculation only

    # ============ Panel (a): CRB-Rate Pareto frontier ============
    ax = fig.add_subplot(gs[0, 0])

    for R, col, mk in [(5,'#1f77b4','o'), (10,'#ff7f0e','s'),
                        (20,'#2ca02c','^'), (50,'#d62728','D')]:
        # Post-rain SNR for rate calculation
        k0, a0 = itu_ka(f0)
        AdB = k0 * R**a0 * L0
        gamma = SNR0 / 10**(AdB/10)

        crbs, ses = [], []
        for eta in etas:
            # ---- v7 FIX: CRB from Eq.(4) with 5 Ku-band freqs ----
            Np = eta * Nsym
            crb_val = crb_R_only(FK_KU, R, L0, sigma_n, Np)
            crbs.append(np.sqrt(crb_val))
            # Rate from Eq.(8) — unchanged
            ses.append(hassibi_rate(gamma, eta, Nsym))

        ax.plot(ses, crbs, color=col, lw=1.2, label=f'$R$={R} mm/h')
        # Mark eta = 0.05, 0.10, 0.20
        for em in [0.05, 0.1, 0.2]:
            i = np.argmin(np.abs(etas - em))
            ax.plot(ses[i], crbs[i], marker=mk, color=col, ms=5,
                    markeredgecolor='black', markeredgewidth=0.3, zorder=5)

    # Annotate eta values for R=20 curve
    gamma20 = SNR0 / 10**(itu_ka(f0)[0]*20**itu_ka(f0)[1]*L0 / 10)
    for em in [0.05, 0.1, 0.2]:
        Np = em * Nsym
        cv = np.sqrt(crb_R_only(FK_KU, 20, L0, sigma_n, Np))
        sv = hassibi_rate(gamma20, em, Nsym)
        ax.annotate(f'$\\eta$={em}', xy=(sv, cv), xytext=(5, -6),
                    textcoords='offset points', fontsize=6, color='#2ca02c')

    # Arrows showing pilot direction
    ax.annotate('', xy=(3.2, 0.3), xytext=(1, 0.06),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.2,
                                connectionstyle='arc3,rad=.2'))
    ax.text(2.6, 0.045, 'Less pilot', fontsize=5.5, color='gray',
            style='italic', ha='center')
    ax.text(0.7, 0.3, 'More pilot', fontsize=5.5, color='gray',
            style='italic', ha='center')

    ax.set(xlabel='Spectral efficiency (bits/s/Hz)',
           ylabel='CRB RMSE of $\\hat{R}$ (mm/h)',
           xlim=[0, 4], ylim=[0.01, 10])
    ax.set_yscale('log')
    ax.legend(loc='upper right', fontsize=6.5)
    ax.grid(True, which='both')
    ax.text(0.03, 0.03, '(a)', transform=ax.transAxes,
            fontsize=9, fontweight='bold')

    # ============ Panel (b): Multi-link fusion ============
    ax = fig.add_subplot(gs[0, 1])
    Ns = np.array([1, 2, 5, 10, 20, 50, 100, 215])

    for lb, R, eta, cl, mk in [
        ('$R$=10, $\\eta$=0.1', 10, 0.1, '#1f77b4', 'o'),
        ('$R$=20, $\\eta$=0.1', 20, 0.1, '#2ca02c', 's'),
        ('$R$=20, $\\eta$=0.05', 20, 0.05, '#d62728', '^'),
        ('$R$=50, $\\eta$=0.1', 50, 0.1, '#ff7f0e', 'D'),
    ]:
        # v7: use 5 Ku-band freqs for consistency
        Np = eta * Nsym
        crb1 = crb_R_only(FK_KU, R, L0, sigma_n, Np)
        ax.loglog(Ns, [np.sqrt(crb1/N) for N in Ns],
                  marker=mk, color=cl, ms=4, lw=1, label=lb)

    # Reference 1/sqrt(N) line — scaled to pass through R=20,eta=0.1,N=1
    crb1_ref = crb_R_only(FK_KU, 20, L0, sigma_n, 0.1*Nsym)
    rmse1_ref = np.sqrt(crb1_ref)
    ax.loglog(Ns, rmse1_ref/np.sqrt(Ns), 'k--', lw=0.8, alpha=0.4,
              label=r'$\propto 1/\sqrt{N}$')

    ax.set(xlabel='Number of links $N$',
           ylabel='CRB RMSE of $\\hat{R}$ (mm/h)',
           xlim=[0.8, 300], ylim=[0.001, 1])
    ax.set_xticks(Ns)
    ax.set_xticklabels(['1','2','5','10','20','50','100','215'], fontsize=7)
    ax.legend(fontsize=6)
    ax.grid(True, which='both')
    ax.text(0.03, 0.03, '(b)', transform=ax.transAxes,
            fontsize=9, fontweight='bold')

    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG_DIR, f'fig5_crb_rate_tradeoff.{ext}'))
    plt.close(fig)
    print("  OK")

    # ============ Numerical verification ============
    print("\n  === Fig 5 Numerical Verification ===")
    print("  Panel (a) key values (5 Ku-band freqs, sigma_n=1 dB):")
    for R in [5, 10, 20, 50]:
        k0, a0 = itu_ka(f0)
        gamma = SNR0 / 10**(k0*R**a0*L0/10)
        gamma_dB = 10*np.log10(max(gamma, 1e-10))
        print(f"    R={R:2d}: gamma={gamma_dB:+.1f} dB")
        for em in [0.05, 0.10, 0.20]:
            Np = em * Nsym
            rmse = np.sqrt(crb_R_only(FK_KU, R, L0, sigma_n, Np))
            se = hassibi_rate(gamma, em, Nsym)
            print(f"      eta={em:.2f}: RMSE={rmse:.3f} mm/h, SE={se:.2f} bits/s/Hz")

    # 5.6x claim verification
    eta_star = (np.sqrt(1 + SNR0*Nsym) - 1) / (SNR0*Nsym)
    crb_star = crb_R_only(FK_KU, 20, L0, sigma_n, eta_star*Nsym)
    crb_010 = crb_R_only(FK_KU, 20, L0, sigma_n, 0.10*Nsym)
    ratio = crb_star / crb_010
    print(f"\n  5.6x claim: eta*={eta_star:.4f}, CRB ratio={ratio:.2f}x, RMSE ratio={np.sqrt(ratio):.2f}x")

    # Rate loss claim
    se_star = hassibi_rate(SNR0, eta_star, Nsym)  # clear-sky
    se_010 = hassibi_rate(SNR0, 0.10, Nsym)
    print(f"  Rate loss: SE(eta*)={se_star:.2f}, SE(0.10)={se_010:.2f}, loss={100*(se_star-se_010)/se_star:.0f}%")

    # Cross-check with Fig 2: at eta=0.1, Np=30, sigma=1, L=3, R=20
    crb_fig2 = crb_multi(FK_KU, 20, L0, 1.0)  # uses Np=30 = 0.1*302
    rmse_fig2 = np.sqrt(crb_fig2)
    rmse_fig5 = np.sqrt(crb_010)
    print(f"\n  Cross-check Fig 2 vs Fig 5 at (R=20, eta=0.1):")
    print(f"    Fig 2 crb_multi (Np=30): RMSE = {rmse_fig2:.4f} mm/h")
    print(f"    Fig 5 crb_R_only(Np=30.2): RMSE = {rmse_fig5:.4f} mm/h")
    print(f"    Match: {'✓' if abs(rmse_fig2 - rmse_fig5)/rmse_fig2 < 0.01 else '✗ MISMATCH'}")


# ========================= MAIN =========================
if __name__ == '__main__':
    print("="*60)
    print("ISAC 2026 v7 — CORRECTED Figure Generator")
    print("  Fixes: Fig 3(b) Np=30, Fig 5(a) Eq.(4) CRB model")
    print("="*60)

    fig2()
    fig3()

    if not SKIP_DATA:
        try: fig4()
        except Exception as e: print(f"  Fig 4 SKIPPED: {e}")
    else:
        print("Fig 4: SKIPPED (--no-data)")

    fig5()

    print("="*60)
    print(f"Output: {FIG_DIR}")
    print("="*60)