"""
=============================================================================
ISAC 2026 Conference Paper — Figure Generation v2
=============================================================================
Fixes: path length model, CRB calibration, event selection, richer figures

Output: ./conference_figures/  (4 code figures; Fig 1 = system diagram, hand-drawn)
  fig2_crb_theory.pdf        — CRB vs SNR + frequency diversity (pure theory)
  fig3_identifiability.pdf   — Side-information hierarchy + spectral signatures
  fig4_itu_validation.pdf    — Model validation with OpenSat4Weather (FIXED)
  fig5_multilink_crb.pdf     — Multi-link diversity + CRB attainability
=============================================================================
"""

import netCDF4 as nc
import numpy as np
import os
from datetime import datetime, timedelta
from scipy.interpolate import interp1d

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, LogLocator
import matplotlib.gridspec as gridspec

# =============================================================================
# CONFIG
# =============================================================================
DATA_DIR = r"F:\Github-repositories\ISAC-LEO-weather\OpenSat4Weather_data"
FIG_DIR = r"F:\Github-repositories\ISAC-LEO-weather\conference_figures"
os.makedirs(FIG_DIR, exist_ok=True)
BASE_TIME = datetime(2022, 8, 1, 0, 0, 0)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 8,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.03,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'lines.linewidth': 1.0,
    'lines.markersize': 4,
    'grid.linewidth': 0.4,
    'grid.alpha': 0.3,
    'text.usetex': False,
})

SINGLE_COL = 3.5
DOUBLE_COL = 7.16

# =============================================================================
# ITU-R P.838-3 MODEL
# =============================================================================
kH_a = np.array([-5.33980, -0.35351, -0.23789, -0.94158])
kH_b = np.array([-0.10008, 1.26970, 0.86036, 0.64552])
kH_c = np.array([1.13098, 0.45400, 0.15354, 0.16817])
kH_mk, kH_ck = 0.83433, 0.14298
aH_a = np.array([-0.14318, 0.29591, 0.32177, -5.37610, 16.1721])
aH_b = np.array([1.82442, 0.77564, 0.63773, -0.96230, -3.29980])
aH_c = np.array([-0.55187, 0.19822, 0.13164, 1.47828, 3.43990])
aH_ma, aH_ca = 0.67849, -1.95537

def itu_k_alpha(f_ghz):
    logf = np.log10(f_ghz)
    log_k = np.sum(kH_a * np.exp(-((logf - kH_b) / kH_c)**2)) + kH_mk * logf + kH_ck
    alpha = np.sum(aH_a * np.exp(-((logf - aH_b) / aH_c)**2)) + aH_ma * logf + aH_ca
    return 10**log_k, alpha

def specific_atten(f_ghz, R):
    k, a = itu_k_alpha(f_ghz)
    return k * np.power(np.maximum(R, 0), a)

def rain_atten_dB(f_ghz, R, L_km):
    return specific_atten(f_ghz, R) * L_km

# ITU-R P.618-14: Effective slant path length for earth-space links
def effective_path_itu618(R001, h_rain_km, elev_deg, lat_deg=43.6):
    """
    ITU-R P.618-14 Step 6-7: horizontal reduction factor and vertical adjustment.
    R001: rain rate exceeded 0.01% of the time (mm/h)
    h_rain_km: rain height (km), from ITU-R P.839 or ERA5 deg0l + 0.36 km
    elev_deg: elevation angle (degrees)
    Returns: effective path length L_eff (km)
    """
    elev_rad = np.radians(np.maximum(elev_deg, 5))
    # Slant path below rain height
    L_s = (h_rain_km) / np.sin(elev_rad)   # km
    L_G = L_s * np.cos(elev_rad)            # horizontal projection

    # Horizontal reduction factor r (P.618 Eq. 32)
    r = 1.0 / (1.0 + 0.78 * np.sqrt(L_G * specific_atten(12.2, R001) / 12.2) - 0.38 * (1 - np.exp(-2 * L_G)))
    r = np.clip(r, 0.01, 1.0)

    L_eff = L_s * r
    return L_eff

# CRB computation
def compute_crb_R(f_ghz_arr, R_true, L_km, sigma_dB, N_obs=1):
    """
    CRB for R from attenuation measurement at multiple frequencies.
    sigma_dB: measurement noise std in dB per observation.
    """
    J = 0.0
    for f in f_ghz_arr:
        k, alpha = itu_k_alpha(f)
        dA_dR = k * alpha * np.maximum(R_true, 0.01)**(alpha - 1) * L_km
        J += N_obs * dA_dR**2 / sigma_dB**2
    return 1.0 / np.maximum(J, 1e-30)

# Joint FIM with nuisance parameters
def compute_joint_fim(f_arr, R, rho_wv, M_cloud, G_off, L_rain, L_gas, L_cloud,
                       sigma_dB, N_obs, delta=1e-5):
    theta = [R, rho_wv, M_cloud, G_off]
    N_theta = len(theta)
    N_f = len(f_arr)

    def total_atten(f, th):
        k, a = itu_k_alpha(f)
        A_rain = k * max(th[0], 0.01)**a * L_rain
        A_gas = (0.005 * (f/10)**2.1 + 0.001 * th[1] * (f/10)**1.8) * L_gas
        A_cloud = 0.0005 * f**1.95 * th[2] * L_cloud
        return A_rain + A_gas + A_cloud + th[3]

    J = np.zeros((N_theta, N_theta))
    for f in f_arr:
        grad = np.zeros(N_theta)
        for p in range(N_theta):
            tp = list(theta); tm = list(theta)
            h = max(abs(theta[p]) * delta, delta)
            tp[p] += h; tm[p] -= h
            grad[p] = (total_atten(f, tp) - total_atten(f, tm)) / (2 * h)

        for i in range(N_theta):
            for j in range(N_theta):
                J[i, j] += N_obs * grad[i] * grad[j] / sigma_dB**2

    return J

print("=" * 70)
print("ISAC 2026 Conference Figures v2")
print("=" * 70)

# =============================================================================
# FIGURE 2: CRB Theoretical Results (double column, 2 panels)
# =============================================================================
print("\nGenerating Fig 2: CRB theory...")

fig2 = plt.figure(figsize=(DOUBLE_COL, 2.6))
gs2 = gridspec.GridSpec(1, 2, wspace=0.35)

# --- Panel (a): CRB vs Rain Rate for different configurations ---
ax2a = fig2.add_subplot(gs2[0, 0])

R_sweep = np.linspace(1, 100, 200)
L_km = 3.0  # representative effective path
sigma_dB = 1.0  # 1 dB RSL noise (typical for SML receivers)
N_obs_1min = 60  # 1-min integration at ~1 Hz

# Config 1: Single frequency (12.2 GHz)
crb_single = [np.sqrt(compute_crb_R([12.2], R, L_km, sigma_dB, N_obs_1min)) for R in R_sweep]

# Config 2: Ku-band full (10.7, 11.2, 11.7, 12.2, 12.7 GHz — 5 channels)
f_ku = [10.7, 11.2, 11.7, 12.2, 12.7]
crb_ku_full = [np.sqrt(compute_crb_R(f_ku, R, L_km, sigma_dB, N_obs_1min)) for R in R_sweep]

# Config 3: Ku + Ka (add 18.0, 19.0, 20.0 GHz)
f_ku_ka = [10.7, 11.2, 11.7, 12.2, 12.7, 18.0, 19.0, 20.0]
crb_ku_ka = [np.sqrt(compute_crb_R(f_ku_ka, R, L_km, sigma_dB, N_obs_1min)) for R in R_sweep]

# Config 4: Wider noise (sigma = 2 dB)
crb_noisy = [np.sqrt(compute_crb_R(f_ku, R, L_km, 2.0, N_obs_1min)) for R in R_sweep]

ax2a.semilogy(R_sweep, crb_single, 'b-', linewidth=1.2, label='Single freq. (12.2 GHz)')
ax2a.semilogy(R_sweep, crb_ku_full, 'r-', linewidth=1.2, label='Ku-band (5 freq.)')
ax2a.semilogy(R_sweep, crb_ku_ka, 'g--', linewidth=1.2, label='Ku+Ka (8 freq.)')
ax2a.semilogy(R_sweep, crb_noisy, 'r:', linewidth=1.0, alpha=0.7, label=r'Ku-band, $\sigma$=2 dB')
ax2a.semilogy(R_sweep, R_sweep * 0.1, 'k--', linewidth=0.6, alpha=0.4, label='10% relative')
ax2a.semilogy(R_sweep, R_sweep * 0.5, 'k:', linewidth=0.6, alpha=0.4, label='50% relative')

ax2a.set_xlabel('Rain rate $R$ (mm/h)')
ax2a.set_ylabel('CRB RMSE of $\\hat{R}$ (mm/h)')
ax2a.legend(loc='upper left', fontsize=6.5, ncol=1)
ax2a.grid(True, which='both', alpha=0.3)
ax2a.set_xlim([1, 100])
ax2a.set_ylim([0.01, 100])
ax2a.xaxis.set_minor_locator(AutoMinorLocator())
ax2a.text(0.03, 0.03, '(a)', transform=ax2a.transAxes, fontsize=9, fontweight='bold')

# --- Panel (b): CRB vs Effective Path Length ---
ax2b = fig2.add_subplot(gs2[0, 1])

L_sweep = np.linspace(0.5, 8, 100)
R_fixed_vals = [5, 10, 20, 50]
colors_R = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for R_val, col in zip(R_fixed_vals, colors_R):
    crb_L = [np.sqrt(compute_crb_R(f_ku, R_val, L, sigma_dB, N_obs_1min)) for L in L_sweep]
    ax2b.semilogy(L_sweep, crb_L, color=col, linewidth=1.2, label=f'$R$={R_val} mm/h')

# Mark typical GEO Ku-band path at 38° elevation
L_geo = 3.0 / np.sin(np.radians(38))
ax2b.axvline(x=L_geo, color='gray', linestyle=':', linewidth=0.8)
ax2b.text(L_geo + 0.1, 0.015, f'GEO\n38° elev.', fontsize=6, color='gray')

# LEO range
ax2b.axvspan(3.0, 3.5, alpha=0.08, color='blue')
ax2b.text(3.05, 50, 'LEO\nzenith', fontsize=6, color='blue', alpha=0.6)

ax2b.set_xlabel('Effective rain path $L_{\\mathrm{eff}}$ (km)')
ax2b.set_ylabel('CRB RMSE of $\\hat{R}$ (mm/h)')
ax2b.legend(loc='upper right', fontsize=6.5)
ax2b.grid(True, which='both', alpha=0.3)
ax2b.set_xlim([0.5, 8])
ax2b.set_ylim([0.01, 100])
ax2b.xaxis.set_minor_locator(AutoMinorLocator())
ax2b.text(0.03, 0.03, '(b)', transform=ax2b.transAxes, fontsize=9, fontweight='bold')

fig2.savefig(os.path.join(FIG_DIR, 'fig2_crb_theory.pdf'))
fig2.savefig(os.path.join(FIG_DIR, 'fig2_crb_theory.png'), dpi=300)
print("  Saved fig2_crb_theory")
plt.close(fig2)

# =============================================================================
# FIGURE 3: Identifiability Analysis (double column, 2 panels)
# =============================================================================
print("Generating Fig 3: Identifiability...")

fig3 = plt.figure(figsize=(DOUBLE_COL, 2.6))
gs3 = gridspec.GridSpec(1, 2, wspace=0.35, width_ratios=[1.1, 1])

# --- Panel (a): Spectral signatures ---
ax3a = fig3.add_subplot(gs3[0, 0])

f_plot = np.linspace(10, 25, 300)
R_val = 20
L_r, L_g, L_c = 3.0, 10.0, 2.0

A_rain = [specific_atten(f, R_val) * L_r for f in f_plot]
A_gas = [(0.005 * (f/10)**2.1 + 0.001 * 7.5 * (f/10)**1.8) * L_g for f in f_plot]
A_cloud = [0.0005 * f**1.95 * 0.3 * L_c for f in f_plot]
A_flat = [1.0] * len(f_plot)  # gain offset = 1 dB everywhere

ax3a.semilogy(f_plot, A_rain, 'b-', linewidth=1.5, label=f'Rain ($R$={R_val} mm/h)')
ax3a.semilogy(f_plot, A_gas, 'r--', linewidth=1.2, label='Gas ($\\rho_{wv}$=7.5 g/m³)')
ax3a.semilogy(f_plot, A_cloud, 'g:', linewidth=1.2, label='Cloud ($M_c$=0.3 g/m³)')
ax3a.semilogy(f_plot, A_flat, 'k-.', linewidth=0.8, alpha=0.5, label='Gain offset $G$ (flat)')

# Shade Ku and Ka bands
ax3a.axvspan(10.7, 12.7, alpha=0.08, color='blue')
ax3a.axvspan(17.7, 20.2, alpha=0.08, color='orange')
ax3a.text(11.0, 0.005, 'Ku', fontsize=7, color='blue')
ax3a.text(18.2, 0.005, 'Ka', fontsize=7, color='orange')

ax3a.set_xlabel('Frequency (GHz)')
ax3a.set_ylabel('Attenuation (dB)')
ax3a.legend(loc='upper left', fontsize=6.5)
ax3a.grid(True, which='both', alpha=0.3)
ax3a.set_xlim([10, 25])
ax3a.set_ylim([0.003, 50])
ax3a.text(0.03, 0.03, '(a)', transform=ax3a.transAxes, fontsize=9, fontweight='bold')

# --- Panel (b): Side-information hierarchy ---
ax3b = fig3.add_subplot(gs3[0, 1])

# Compute CRB for each configuration using joint FIM
f_full = np.linspace(10.7, 12.7, 20)
sigma_dB_j = 1.0
N_obs_j = 60

configs_data = []
config_list = [
    ('$R$ only', {'R': True}),
    ('$R + G$', {'R': True, 'G': True}),
    ('$R + \\rho_{wv}$', {'R': True, 'rho': True}),
    ('$R + M_c$', {'R': True, 'M': True}),
    ('$R + \\rho + M$', {'R': True, 'rho': True, 'M': True}),
    ('All unkn.', {'R': True, 'rho': True, 'M': True, 'G': True}),
]

for label, flags in config_list:
    # Build subset FIM
    full_theta = [20.0, 7.5, 0.3, 0.0]
    param_idx = []
    if flags.get('R'): param_idx.append(0)
    if flags.get('rho'): param_idx.append(1)
    if flags.get('M'): param_idx.append(2)
    if flags.get('G'): param_idx.append(3)

    J_full = compute_joint_fim(f_full, 20.0, 7.5, 0.3, 0.0,
                                L_r, L_g, L_c, sigma_dB_j, N_obs_j)
    J_sub = J_full[np.ix_(param_idx, param_idx)]
    try:
        J_inv = np.linalg.inv(J_sub)
        rmse_R = np.sqrt(abs(J_inv[0, 0]))
        rel = rmse_R / 20 * 100
    except:
        rel = 1e5

    configs_data.append((label, rel))

labels = [c[0] for c in configs_data]
values = [c[1] for c in configs_data]
colors_bar = ['#2ca02c', '#4daf4a', '#ff7f00', '#e6ab02', '#e41a1c', '#984ea3']

y_pos = np.arange(len(labels))
bars = ax3b.barh(y_pos, values, color=colors_bar, height=0.55,
                  edgecolor='black', linewidth=0.3)

ax3b.set_xscale('log')
ax3b.set_xlabel('CRB RMSE / $R$ (%)')
ax3b.set_yticks(y_pos)
ax3b.set_yticklabels(labels, fontsize=7)
ax3b.invert_yaxis()
ax3b.axvline(x=100, color='red', linestyle=':', linewidth=0.8, alpha=0.7)

# Annotate values
for bar, val in zip(bars, values):
    x_pos = val * 1.4 if val < 100 else val * 1.4
    color = 'black' if val < 100 else 'red'
    txt = f'{val:.1f}%' if val < 100 else f'{val:.0f}%'
    ax3b.text(x_pos, bar.get_y() + bar.get_height()/2, txt,
              va='center', fontsize=6.5, color=color)

ax3b.set_xlim([0.5, 2e5])
ax3b.grid(True, which='major', axis='x', alpha=0.3)
ax3b.text(0.03, 0.03, '(b)', transform=ax3b.transAxes, fontsize=9, fontweight='bold')

# Add "identifiable" / "unidentifiable" regions
ax3b.text(5, 5.6, 'Identifiable', fontsize=6.5, color='green', style='italic')
ax3b.text(500, 5.6, 'Unidentifiable', fontsize=6.5, color='red', style='italic')

fig3.savefig(os.path.join(FIG_DIR, 'fig3_identifiability.pdf'))
fig3.savefig(os.path.join(FIG_DIR, 'fig3_identifiability.png'), dpi=300)
print("  Saved fig3_identifiability")
plt.close(fig3)

# =============================================================================
# LOAD OPENSAT4WEATHER DATA (for Figs 4-5)
# =============================================================================
print("\nLoading OpenSat4Weather data...")

ds_sml = nc.Dataset(os.path.join(DATA_DIR, 'sml_data_2022.nc'))
rsl_ma = ds_sml.variables['rsl'][:]
sml_time = ds_sml.variables['time'][:].astype(float)
elev_arr = np.array(ds_sml.variables['satellite_elevation'][:], dtype=float)
deg0l_ma = ds_sml.variables['deg0l'][:]

rsl = rsl_ma.filled(np.nan).astype(float) if isinstance(rsl_ma, np.ma.MaskedArray) else rsl_ma.astype(float)
rsl[rsl < -100] = np.nan
deg0l = deg0l_ma.filled(np.nan).astype(float) if isinstance(deg0l_ma, np.ma.MaskedArray) else deg0l_ma.astype(float)
deg0l[deg0l < 0] = np.nan

ds_rad = nc.Dataset(os.path.join(DATA_DIR, 'radar_along_sml_data_2022.nc'))
rad_ma = ds_rad.variables['rainfall_amount'][:]
rad_time = ds_rad.variables['time'][:].astype(float)
rad = rad_ma.filled(np.nan).astype(float) if isinstance(rad_ma, np.ma.MaskedArray) else rad_ma.astype(float)
rad[rad < 0] = np.nan

n_sml = rsl.shape[0]
n_sml_time = rsl.shape[1]
n_rad_time = len(rad_time)
F_GHZ = 12.2

# Align timestamps
sml_idx_for_rad = np.searchsorted(sml_time, rad_time)
sml_idx_for_rad = np.clip(sml_idx_for_rad, 0, n_sml_time - 1)

print(f"  SML: {rsl.shape}, Radar: {rad.shape}")

# =============================================================================
# PREPROCESSING: Compute attenuation with FIXED path model
# =============================================================================
print("Computing attenuation pairs (FIXED path model)...")

# Rolling baseline: 6-hour P97 (faster + more stable than 24h)
WINDOW_H = 6
STEP = 30  # compute every 30 min

def rolling_baseline(arr, window_min=360, step=30):
    n = len(arr)
    anchors_x = []
    anchors_y = []
    half = window_min // 2
    for i in range(0, n, step):
        lo, hi = max(0, i - half), min(n, i + half)
        chunk = arr[lo:hi]
        valid = chunk[~np.isnan(chunk)]
        if len(valid) > 50:
            anchors_x.append(i)
            anchors_y.append(np.percentile(valid, 97))
    if len(anchors_x) < 2:
        return np.full(n, np.nan)
    f = interp1d(anchors_x, anchors_y, kind='linear', fill_value='extrapolate')
    return f(np.arange(n))

# Process 40 SMLs for better statistics
sml_subset = np.linspace(0, n_sml - 1, 40, dtype=int)

all_atten = []
all_rr = []
all_L_eff = []
all_sml_idx_list = []

for cnt, si in enumerate(sml_subset):
    if cnt % 10 == 0:
        print(f"  SML {cnt+1}/{len(sml_subset)}...")

    baseline = rolling_baseline(rsl[si, :])

    for rt in range(n_rad_time):
        st = sml_idx_for_rad[rt]
        rsl_val = rsl[si, st]
        base_val = baseline[st]
        if np.isnan(rsl_val) or np.isnan(base_val):
            continue

        atten = base_val - rsl_val
        if atten < -2:
            continue

        rr_accum = rad[rt, si]
        if np.isnan(rr_accum):
            continue
        rr = rr_accum * 12  # mm/h

        # Rain height from deg0l + 0.36 km (ITU-R P.839)
        d0 = deg0l[si, st]
        if np.isnan(d0) or d0 < 100:
            d0 = 3000
        h_rain_km = d0 / 1000.0 + 0.36

        # FIXED: Use ITU-R P.618 effective path
        if rr > 0.1:
            L_eff = effective_path_itu618(rr, h_rain_km, elev_arr[si])
        else:
            L_eff = h_rain_km / np.sin(np.radians(elev_arr[si]))

        all_atten.append(max(atten, 0))
        all_rr.append(rr)
        all_L_eff.append(L_eff)
        all_sml_idx_list.append(si)

all_atten = np.array(all_atten)
all_rr = np.array(all_rr)
all_L_eff = np.array(all_L_eff)

# Predicted attenuation
all_atten_pred = np.array([rain_atten_dB(F_GHZ, r, L) for r, L in zip(all_rr, all_L_eff)])

print(f"  Total pairs: {len(all_atten):,}, RR>1: {np.sum(all_rr>1):,}, RR>5: {np.sum(all_rr>5):,}")

# =============================================================================
# FIGURE 4: ITU Model Validation (double column, 3 panels)
# =============================================================================
print("Generating Fig 4: ITU validation (FIXED)...")

fig4 = plt.figure(figsize=(DOUBLE_COL, 2.5))
gs4 = gridspec.GridSpec(1, 3, wspace=0.38, width_ratios=[1, 1, 1])

# --- Panel (a): Scatter: predicted vs measured ---
ax4a = fig4.add_subplot(gs4[0, 0])

mask_rain = all_rr > 0.5
atten_m = all_atten[mask_rain]
atten_p = all_atten_pred[mask_rain]
rr_f = all_rr[mask_rain]

sc = ax4a.scatter(atten_p, atten_m, c=np.log10(np.maximum(rr_f, 0.1)),
                   cmap='YlOrRd', s=1.5, alpha=0.25, rasterized=True,
                   vmin=-0.5, vmax=2)

ax4a.plot([0, 15], [0, 15], 'k--', linewidth=0.8, alpha=0.5)

# Binned mean ± std
bin_edges = np.arange(0, 12, 1)
bin_x, bin_y, bin_std = [], [], []
for i in range(len(bin_edges)-1):
    mask_b = (atten_p >= bin_edges[i]) & (atten_p < bin_edges[i+1])
    if np.sum(mask_b) > 20:
        bin_x.append((bin_edges[i] + bin_edges[i+1]) / 2)
        bin_y.append(np.mean(atten_m[mask_b]))
        bin_std.append(np.std(atten_m[mask_b]))

bin_x, bin_y, bin_std = np.array(bin_x), np.array(bin_y), np.array(bin_std)
ax4a.errorbar(bin_x, bin_y, yerr=bin_std, fmt='bs-', markersize=4,
               linewidth=1.2, capsize=2, label='Binned mean ± std', zorder=5)

# R^2 and RMSE for binned data
valid_b = (atten_p > 0.1) & (atten_m > 0) & np.isfinite(atten_p) & np.isfinite(atten_m)
rmse_all = np.sqrt(np.mean((atten_m[valid_b] - atten_p[valid_b])**2))
corr = np.corrcoef(atten_p[valid_b], atten_m[valid_b])[0, 1]

ax4a.text(0.05, 0.92, f'$r$ = {corr:.3f}\nRMSE = {rmse_all:.2f} dB',
          transform=ax4a.transAxes, fontsize=7,
          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

ax4a.set_xlabel('Predicted atten. (dB)')
ax4a.set_ylabel('Measured atten. (dB)')
ax4a.set_xlim([0, 15])
ax4a.set_ylim([0, 15])
ax4a.set_aspect('equal')
ax4a.legend(loc='lower right', fontsize=6)
ax4a.grid(True, alpha=0.3)
ax4a.text(0.03, 0.03, '(a)', transform=ax4a.transAxes, fontsize=9, fontweight='bold')

# --- Panel (b): Attenuation vs Rain Rate (binned) ---
ax4b = fig4.add_subplot(gs4[0, 1])

rr_bins = [0.5, 1, 2, 5, 8, 12, 20, 35, 60, 100]
bin_rr_c, bin_atten_mean, bin_atten_std, bin_pred_mean = [], [], [], []

for i in range(len(rr_bins)-1):
    mask_b = (all_rr >= rr_bins[i]) & (all_rr < rr_bins[i+1])
    if np.sum(mask_b) > 30:
        rc = np.mean(all_rr[mask_b])
        bin_rr_c.append(rc)
        bin_atten_mean.append(np.mean(all_atten[mask_b]))
        bin_atten_std.append(np.std(all_atten[mask_b]))
        bin_pred_mean.append(np.mean(all_atten_pred[mask_b]))

bin_rr_c = np.array(bin_rr_c)
bin_atten_mean = np.array(bin_atten_mean)
bin_atten_std = np.array(bin_atten_std)
bin_pred_mean = np.array(bin_pred_mean)

ax4b.errorbar(bin_rr_c, bin_atten_mean, yerr=bin_atten_std, fmt='rs-',
               markersize=4, linewidth=1.0, capsize=2, label='Measured (mean±std)')
ax4b.plot(bin_rr_c, bin_pred_mean, 'b^--', markersize=4, linewidth=1.0,
           label='ITU-R P.838 + P.618')

# Theoretical curve
R_th = np.linspace(0.5, 100, 200)
L_eff_th = [effective_path_itu618(R, 3.5, 38.0) for R in R_th]
A_th = [rain_atten_dB(F_GHZ, R, L) for R, L in zip(R_th, L_eff_th)]
ax4b.plot(R_th, A_th, 'b:', linewidth=0.8, alpha=0.5, label='P.838 (continuous)')

ax4b.set_xlabel('Rain rate $R$ (mm/h)')
ax4b.set_ylabel('Attenuation (dB)')
ax4b.legend(fontsize=6, loc='upper left')
ax4b.grid(True, alpha=0.3)
ax4b.set_xlim([0, 80])
ax4b.set_ylim([0, 12])
ax4b.xaxis.set_minor_locator(AutoMinorLocator())
ax4b.yaxis.set_minor_locator(AutoMinorLocator())
ax4b.text(0.03, 0.03, '(b)', transform=ax4b.transAxes, fontsize=9, fontweight='bold')

# --- Panel (c): Attenuation CDF for different rain rate ranges ---
ax4c = fig4.add_subplot(gs4[0, 2])

rr_ranges = [(0, 0.5), (0.5, 2), (2, 10), (10, 50), (50, 300)]
rr_labels = ['Dry ($R$<0.5)', '0.5-2', '2-10', '10-50', '>50 mm/h']
rr_colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd']

for (lo, hi), lab, col in zip(rr_ranges, rr_labels, rr_colors):
    mask_r = (all_rr >= lo) & (all_rr < hi) & (all_atten >= 0)
    if np.sum(mask_r) > 50:
        vals = np.sort(all_atten[mask_r])
        cdf = np.arange(1, len(vals)+1) / len(vals)
        # Subsample for smooth plot
        idx_sub = np.linspace(0, len(vals)-1, min(500, len(vals)), dtype=int)
        ax4c.plot(vals[idx_sub], 1-cdf[idx_sub], color=col, linewidth=1.0, label=lab)

ax4c.set_xlabel('Attenuation (dB)')
ax4c.set_ylabel('CCDF $P(A > x)$')
ax4c.set_yscale('log')
ax4c.set_xlim([0, 15])
ax4c.set_ylim([1e-4, 1])
ax4c.legend(fontsize=6, loc='upper right')
ax4c.grid(True, which='both', alpha=0.3)
ax4c.xaxis.set_minor_locator(AutoMinorLocator())
ax4c.text(0.03, 0.03, '(c)', transform=ax4c.transAxes, fontsize=9, fontweight='bold')

fig4.savefig(os.path.join(FIG_DIR, 'fig4_itu_validation.pdf'))
fig4.savefig(os.path.join(FIG_DIR, 'fig4_itu_validation.png'), dpi=300)
print("  Saved fig4_itu_validation")
plt.close(fig4)

# =============================================================================
# FIGURE 5: Multi-link Diversity + CRB Attainability (double column, 2 panels)
# =============================================================================
print("Generating Fig 5: Multi-link + CRB attainability...")

fig5 = plt.figure(figsize=(DOUBLE_COL, 2.6))
gs5 = gridspec.GridSpec(1, 2, wspace=0.35)

# --- Panel (a): CRB attainability from data ---
ax5a = fig5.add_subplot(gs5[0, 0])

# MLE estimator: R_hat = (A_meas / (k * L_eff))^(1/alpha)
k_itu, alpha_itu = itu_k_alpha(F_GHZ)

rain_mask = (all_rr > 1) & (all_atten > 0.1) & (all_L_eff > 0.1)
rr_est_data = all_rr[rain_mask]
atten_est = all_atten[rain_mask]
L_est = all_L_eff[rain_mask]

R_hat = np.power(np.maximum(atten_est, 0.001) / (k_itu * L_est), 1.0 / alpha_itu)

# Bin and compute RMSE
bins_R = [1, 3, 5, 8, 12, 18, 25, 40, 60, 100]
bc, mse_emp, crb_th = [], [], []

for i in range(len(bins_R)-1):
    m = (rr_est_data >= bins_R[i]) & (rr_est_data < bins_R[i+1])
    if np.sum(m) < 30:
        continue
    R_c = np.mean(rr_est_data[m])
    rmse = np.sqrt(np.mean((R_hat[m] - rr_est_data[m])**2))
    L_c = np.mean(L_est[m])

    # CRB with calibrated noise
    # Noise std from clear-sky RSL variability ≈ 1.0 dB
    crb_val = np.sqrt(compute_crb_R([F_GHZ], R_c, L_c, 1.0, 60))

    bc.append(R_c)
    mse_emp.append(rmse)
    crb_th.append(crb_val)

bc, mse_emp, crb_th = np.array(bc), np.array(mse_emp), np.array(crb_th)

ax5a.semilogy(bc, mse_emp, 'rs-', markersize=5, linewidth=1.2, label='MLE RMSE (data)')
ax5a.semilogy(bc, crb_th, 'b^--', markersize=5, linewidth=1.2, label='CRB ($\\sigma$=1 dB)')

# CRB with worse noise
crb_2dB = [np.sqrt(compute_crb_R([F_GHZ], R, 3.0, 2.0, 60)) for R in bc]
ax5a.semilogy(bc, crb_2dB, 'b:', markersize=3, linewidth=0.8, alpha=0.6, label='CRB ($\\sigma$=2 dB)')

# Multi-freq CRB
crb_mf = [np.sqrt(compute_crb_R(f_ku, R, 3.0, 1.0, 60)) for R in bc]
ax5a.semilogy(bc, crb_mf, 'g--', markersize=3, linewidth=1.0, label='CRB (5 freq., $\\sigma$=1 dB)')

R_line = np.linspace(1, 100, 100)
ax5a.semilogy(R_line, R_line * 0.1, 'k:', linewidth=0.5, alpha=0.4)
ax5a.semilogy(R_line, R_line * 0.5, 'k-.', linewidth=0.5, alpha=0.4)
ax5a.text(80, 10, '10%', fontsize=6, alpha=0.5, rotation=25)
ax5a.text(80, 50, '50%', fontsize=6, alpha=0.5, rotation=25)

ax5a.set_xlabel('Rain rate $R$ (mm/h)')
ax5a.set_ylabel('RMSE of $\\hat{R}$ (mm/h)')
ax5a.legend(loc='upper left', fontsize=6)
ax5a.grid(True, which='both', alpha=0.3)
ax5a.set_xlim([1, 100])
ax5a.set_ylim([0.05, 200])
ax5a.xaxis.set_minor_locator(AutoMinorLocator())
ax5a.text(0.03, 0.03, '(a)', transform=ax5a.transAxes, fontsize=9, fontweight='bold')

# --- Panel (b): Multi-link diversity gain ---
ax5b = fig5.add_subplot(gs5[0, 1])

rng = np.random.default_rng(42)
mean_rr_t = np.nanmean(rad, axis=1) * 12

# For different rain rate thresholds, show diversity gain
thresholds = [('$R$>1 mm/h', 1, '#1f77b4'),
              ('$R$>5 mm/h', 5, '#d62728'),
              ('$R$>10 mm/h', 10, '#2ca02c')]

n_links_test = [1, 2, 5, 10, 20, 50, 100, 215]

for label_thr, thr, col in thresholds:
    heavy_t = np.where(mean_rr_t > thr)[0]
    if len(heavy_t) < 20:
        continue

    rmse_list = []
    for n_l in n_links_test:
        errors = []
        for _ in range(300):
            t_idx = rng.choice(heavy_t)
            sml_choice = rng.choice(n_sml, size=min(n_l, n_sml), replace=False)
            rr_true = rad[t_idx, sml_choice] * 12
            rr_true = rr_true[~np.isnan(rr_true)]
            if len(rr_true) == 0:
                continue
            R_spatial = np.mean(rr_true)
            if R_spatial < 0.1:
                continue
            # Add estimation noise
            R_noisy = rr_true + rng.normal(0, max(R_spatial * 0.3, 0.5), len(rr_true))
            R_est = np.mean(np.maximum(R_noisy, 0))
            errors.append((R_est - R_spatial)**2)

        if len(errors) > 10:
            rmse_list.append(np.sqrt(np.mean(errors)))
        else:
            rmse_list.append(np.nan)

    ax5b.loglog(n_links_test, rmse_list, 'o-', color=col, markersize=4,
                 linewidth=1.0, label=label_thr)

# Reference line
ax5b.loglog(n_links_test, 3.0 / np.sqrt(n_links_test), 'k--',
             linewidth=0.8, alpha=0.4, label=r'$\propto 1/\sqrt{N}$')

ax5b.set_xlabel('Number of links $N$')
ax5b.set_ylabel('RMSE of $\\hat{R}$ (mm/h)')
ax5b.legend(fontsize=6, loc='upper right')
ax5b.grid(True, which='both', alpha=0.3)
ax5b.set_xticks([1, 2, 5, 10, 20, 50, 100, 215])
ax5b.set_xticklabels(['1', '2', '5', '10', '20', '50', '100', '215'], fontsize=7)
ax5b.text(0.03, 0.03, '(b)', transform=ax5b.transAxes, fontsize=9, fontweight='bold')

fig5.savefig(os.path.join(FIG_DIR, 'fig5_multilink_crb.pdf'))
fig5.savefig(os.path.join(FIG_DIR, 'fig5_multilink_crb.png'), dpi=300)
print("  Saved fig5_multilink_crb")
plt.close(fig5)

# =============================================================================
# Cleanup
# =============================================================================
ds_sml.close()
ds_rad.close()
ds_rg.close()

print("\n" + "=" * 70)
print("ALL FIGURES GENERATED")
print("=" * 70)
print(f"Output: {FIG_DIR}")
for f in sorted(os.listdir(FIG_DIR)):
    if f.startswith('fig') and (f.endswith('.pdf') or f.endswith('.png')):
        print(f"  {f}")

print(f"""
Conference paper figure plan:
  Fig 1: System diagram (hand-drawn, TikZ/PPT)
  Fig 2: CRB theory — (a) CRB vs R, (b) CRB vs path length     [THEORY]
  Fig 3: Identifiability — (a) spectral signatures, (b) bar chart [THEORY]
  Fig 4: ITU validation — (a) scatter, (b) A vs R, (c) CCDF      [DATA]
  Fig 5: Multi-link — (a) CRB attainability, (b) diversity gain   [DATA+THEORY]
""")