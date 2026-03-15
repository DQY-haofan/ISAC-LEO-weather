"""
OpenSat4Weather Dataset Explorer v3 — FIXED
=============================================
Fixes: masked array handling, radar axis, scale_factor double-apply
"""

import netCDF4 as nc
import numpy as np
import os
from datetime import datetime, timedelta

DATA_DIR = r"F:\Github-repositories\ISAC-LEO-weather\OpenSat4Weather_data"
base_time = datetime(2022, 8, 1, 0, 0, 0)

def safe_strings(var):
    try:
        return [str(x) for x in var[:].tolist()]
    except:
        return [f"item_{i}" for i in range(var.shape[0])]

def read_masked(var):
    """Read variable, let netCDF4 handle scale/mask, convert to float with NaN."""
    data = var[:]
    if isinstance(data, np.ma.MaskedArray):
        return data.filled(np.nan).astype(float)
    return np.array(data, dtype=float)

# =============================================================================
print("=" * 80)
print("PART 1: FILE INVENTORY")
print("=" * 80)
files = {}
for f in sorted(os.listdir(DATA_DIR)):
    if f.endswith('.nc'):
        fp = os.path.join(DATA_DIR, f)
        files[f] = fp
        print(f"  {f:45s} {os.path.getsize(fp)/1e6:.1f} MB")

# =============================================================================
print("\n" + "=" * 80)
print("PART 2: SML DATA")
print("=" * 80)

ds = nc.Dataset(files['sml_data_2022.nc'])

# Let netCDF4 auto-mask and auto-scale (default behavior)
n_sml = ds.dimensions['sml_id'].size
n_time = ds.dimensions['time'].size
print(f"  Dimensions: {n_sml} SMLs × {n_time} time steps")

time_arr = ds.variables['time'][:]  # minutes since 2022-08-01
print(f"  Time: {base_time} to {base_time + timedelta(minutes=float(time_arr[-1]))}")

# Metadata
print(f"\n--- Metadata ---")
print(f"  IDs: {safe_strings(ds.variables['sml_id'])[:5]}")

lat = read_masked(ds.variables['site_0_lat'])
lon = read_masked(ds.variables['site_0_lon'])
alt = read_masked(ds.variables['site_0_alt'])
print(f"  Lat:  [{np.nanmin(lat):.3f}, {np.nanmax(lat):.3f}]")
print(f"  Lon:  [{np.nanmin(lon):.3f}, {np.nanmax(lon):.3f}]")
print(f"  Alt:  [{np.nanmin(alt):.0f}, {np.nanmax(alt):.0f}] m")

elev = read_masked(ds.variables['satellite_elevation'])
azim = read_masked(ds.variables['satellite_azimuth'])
print(f"  Elevation: [{np.nanmin(elev):.2f}°, {np.nanmax(elev):.2f}°], mean={np.nanmean(elev):.2f}°")
print(f"  Azimuth:   [{np.nanmin(azim):.2f}°, {np.nanmax(azim):.2f}°]")

sats = safe_strings(ds.variables['satellite'])
print(f"  Satellites: {list(set(sats))}")
print(f"  Hardware:   {list(set(safe_strings(ds.variables['hardware'])))}")

# RSL — use netCDF4 auto masking
print(f"\n--- RSL ---")
rsl_ma = ds.variables['rsl'][:]  # masked array, auto-scaled
print(f"  Type: {type(rsl_ma)}, shape={rsl_ma.shape}, dtype={rsl_ma.dtype}")

if isinstance(rsl_ma, np.ma.MaskedArray):
    n_masked = np.ma.count_masked(rsl_ma)
    n_valid = rsl_ma.count()
    print(f"  Masked (fill): {n_masked:,} ({n_masked/rsl_ma.size*100:.1f}%)")
    print(f"  Valid:          {n_valid:,} ({n_valid/rsl_ma.size*100:.1f}%)")
    rsl_valid = rsl_ma.compressed()
else:
    rsl_float = rsl_ma.astype(float)
    n_bad = np.sum((rsl_float < -200) | (rsl_float > 100))  # physically impossible
    print(f"  Physically invalid (< -200 or > 100 dBm): {n_bad:,}")
    rsl_valid = rsl_float[(rsl_float >= -200) & (rsl_float <= 100)]

print(f"  Valid RSL: min={np.min(rsl_valid):.2f}, max={np.max(rsl_valid):.2f}, "
      f"mean={np.mean(rsl_valid):.2f}, std={np.std(rsl_valid):.2f} dBm")

# Percentiles
pcts = [1, 5, 25, 50, 75, 95, 99]
vals = np.percentile(rsl_valid, pcts)
print(f"  Percentiles: " + ", ".join([f"P{p}={v:.2f}" for p, v in zip(pcts, vals)]))

# Per-SML stats
print(f"\n--- Per-SML RSL Summary (first 10) ---")
print(f"  {'#':<5} {'elev°':<8} {'N_valid':<10} {'RSL_mean':<10} {'RSL_min':<10} {'RSL_max':<10} {'range':<8}")

# Convert to float NaN array for easier per-row stats
rsl_float = rsl_ma.filled(np.nan).astype(float) if isinstance(rsl_ma, np.ma.MaskedArray) else rsl_ma.astype(float)
# Filter out unphysical values
rsl_float[rsl_float < -200] = np.nan
rsl_float[rsl_float > 100] = np.nan

for i in range(min(10, n_sml)):
    row = rsl_float[i, :]
    v = row[~np.isnan(row)]
    if len(v) > 0:
        print(f"  {i:<5} {elev[i]:<8.2f} {len(v):<10} {np.mean(v):<10.2f} {np.min(v):<10.2f} {np.max(v):<10.2f} {np.max(v)-np.min(v):<8.2f}")

# Overall per-SML availability and dynamic range
avail = np.sum(~np.isnan(rsl_float), axis=1)
print(f"\n--- Availability ---")
print(f"  Per-SML valid steps: min={np.min(avail)}, max={np.max(avail)}, mean={np.mean(avail):.0f} / {n_time}")
print(f"  >80%: {np.sum(avail > 0.8*n_time)}, >50%: {np.sum(avail > 0.5*n_time)}, >20%: {np.sum(avail > 0.2*n_time)}")

ranges = []
for i in range(n_sml):
    v = rsl_float[i, :]; v = v[~np.isnan(v)]
    ranges.append(np.ptp(v) if len(v) > 100 else 0)
ranges = np.array(ranges)
print(f"  RSL dynamic range: min={np.min(ranges):.2f}, max={np.max(ranges):.2f}, "
      f"mean={np.mean(ranges):.2f}, median={np.median(ranges):.2f} dB")

# deg0l
print(f"\n--- 0°C Isotherm ---")
deg0l_ma = ds.variables['deg0l'][:]
if isinstance(deg0l_ma, np.ma.MaskedArray):
    d0 = deg0l_ma.compressed().astype(float)
else:
    d0 = deg0l_ma.astype(float)
    d0 = d0[d0 > -1000]  # filter fill
print(f"  Stats (m AGL): min={np.min(d0):.0f}, max={np.max(d0):.0f}, mean={np.mean(d0):.0f}, median={np.median(d0):.0f}")

ds.close()

# =============================================================================
print("\n" + "=" * 80)
print("PART 3: RAIN GAUGE DATA")
print("=" * 80)

ds = nc.Dataset(files['rg_data_2022.nc'])
n_rg = ds.dimensions['id'].size
n_rg_time = ds.dimensions['time'].size
print(f"  {n_rg} gauges × {n_rg_time} time steps")

rg_time = ds.variables['time'][:]
dt = np.median(np.diff(rg_time[:200].astype(float)))
print(f"  Time: {base_time} to {base_time + timedelta(minutes=float(rg_time[-1]))}, interval={dt:.0f} min")

rain_ma = ds.variables['rainfall_amount'][:]
rain = rain_ma.filled(np.nan).astype(float) if isinstance(rain_ma, np.ma.MaskedArray) else rain_ma.astype(float)
rain_flat = rain[~np.isnan(rain)]
rain_nz = rain_flat[rain_flat > 0]

print(f"\n--- Rainfall ---")
print(f"  Valid: {len(rain_flat):,} / {rain.size:,} ({len(rain_flat)/rain.size*100:.1f}%)")
print(f"  Non-zero: {len(rain_nz):,} ({len(rain_nz)/len(rain_flat)*100:.2f}%)")
if len(rain_nz) > 0:
    rr = rain_nz / (dt / 60)  # accumulation / interval_hours = mm/h
    print(f"  Rain rate (mm/h): mean={np.mean(rr):.2f}, median={np.median(rr):.2f}, "
          f"p95={np.percentile(rr,95):.2f}, max={np.max(rr):.2f}")
ds.close()

# =============================================================================
print("\n" + "=" * 80)
print("PART 4: RADAR DATA")
print("=" * 80)

ds = nc.Dataset(files['radar_along_sml_data_2022.nc'])

print(f"\n--- Dimensions ---")
for dname, d in ds.dimensions.items():
    print(f"  {dname:20s} size={len(d)}")

print(f"\n--- Variables ---")
for vname, v in ds.variables.items():
    print(f"  {vname:30s} shape={v.shape}  dtype={v.dtype}")
    for a in v.ncattrs():
        print(f"    @{a} = {v.getncattr(a)}")

rad_time = ds.variables['time'][:]
print(f"\n  Time: {base_time} to {base_time + timedelta(minutes=float(rad_time[-1]))}")
print(f"  Steps: {len(rad_time)}, interval={np.median(np.diff(rad_time[:200].astype(float))):.0f} min")

rad_ma = ds.variables['rainfall_amount'][:]
print(f"  Raw type: {type(rad_ma)}, shape={rad_ma.shape}")

rad = rad_ma.filled(np.nan).astype(float) if isinstance(rad_ma, np.ma.MaskedArray) else rad_ma.astype(float)
# Filter unphysical
rad[rad < 0] = np.nan

# Determine axes: shape is (44064, 215) → (time, sml_id)
# So axis=0 is time, axis=1 is sml_id
TIME_AXIS = 0
SML_AXIS = 1
print(f"  Axes: dim0=time({rad.shape[0]}), dim1=sml_id({rad.shape[1]})")

rad_flat = rad[~np.isnan(rad)]
rad_nz = rad_flat[rad_flat > 0]
print(f"  Valid: {len(rad_flat):,}, Non-zero: {len(rad_nz):,} ({len(rad_nz)/len(rad_flat)*100:.2f}%)")

if len(rad_nz) > 0:
    rr_rad = rad_nz * 12  # 5-min accum → mm/h
    print(f"  Rain rate (mm/h): mean={np.mean(rr_rad):.2f}, median={np.median(rr_rad):.2f}, "
          f"p95={np.percentile(rr_rad,95):.2f}, max={np.max(rr_rad):.2f}")

# =============================================================================
print("\n" + "=" * 80)
print("PART 5: RAIN EVENTS")
print("=" * 80)

# Spatial-mean rain per time step: average across SMLs (axis=1)
mean_rain_per_t = np.nanmean(rad, axis=SML_AXIS)  # shape: (n_time,)
mean_rr_per_t = mean_rain_per_t * 12  # mm/h

# Filter out NaN/negative
mean_rr_per_t = np.where(np.isnan(mean_rr_per_t), 0, mean_rr_per_t)
mean_rr_per_t = np.where(mean_rr_per_t < 0, 0, mean_rr_per_t)

print(f"  Mean rain rate stats: max={np.max(mean_rr_per_t):.2f} mm/h")
for thresh in [0.5, 1, 2, 5, 10]:
    cnt = np.sum(mean_rr_per_t > thresh)
    print(f"  Steps > {thresh} mm/h: {cnt} ({cnt/len(mean_rr_per_t)*100:.2f}%)")

# Also check per-SML max rain rate
print(f"\n--- Per-SML Peak Rain Rate ---")
max_rr_per_sml = np.nanmax(rad, axis=TIME_AXIS) * 12  # max over time, per SML
print(f"  Per-SML max RR: min={np.nanmin(max_rr_per_sml):.2f}, max={np.nanmax(max_rr_per_sml):.2f}, "
      f"mean={np.nanmean(max_rr_per_sml):.2f} mm/h")
for thresh in [5, 10, 20, 50]:
    cnt = np.sum(max_rr_per_sml > thresh)
    print(f"  SMLs with peak RR > {thresh} mm/h: {cnt}")

# Top 10 rainiest moments
top_idx = np.argsort(mean_rr_per_t)[-10:][::-1]
print(f"\n--- Top 10 Rainiest Moments (spatial mean) ---")
for rank, idx in enumerate(top_idx):
    t = base_time + timedelta(minutes=float(rad_time[idx]))
    # Also get max across SMLs at this time
    max_at_t = np.nanmax(rad[idx, :]) * 12
    print(f"  {rank+1:2d}. {t}  mean={mean_rr_per_t[idx]:.2f}  max_sml={max_at_t:.2f} mm/h")

# Event detection
print(f"\n--- Rain Events (spatial-mean > 0.5 mm/h) ---")
mask = mean_rr_per_t > 0.5
events = []
in_ev = False
for i in range(len(mask)):
    if mask[i] and not in_ev:
        s = i; in_ev = True
    elif not mask[i] and in_ev:
        e = i - 1
        dur = float(rad_time[e] - rad_time[s])
        peak = np.max(mean_rr_per_t[s:e+1])
        avg = np.mean(mean_rr_per_t[s:e+1])
        # Also max across all SMLs during event
        max_any = np.nanmax(rad[s:e+1, :]) * 12
        events.append((s, e, dur, peak, avg, max_any))
        in_ev = False
if in_ev:
    e = len(mask) - 1
    dur = float(rad_time[e] - rad_time[s])
    peak = np.max(mean_rr_per_t[s:e+1])
    avg = np.mean(mean_rr_per_t[s:e+1])
    max_any = np.nanmax(rad[s:e+1, :]) * 12
    events.append((s, e, dur, peak, avg, max_any))

events.sort(key=lambda x: x[5], reverse=True)  # sort by max across SMLs
print(f"  Total events: {len(events)}")
print(f"\n{'#':<4} {'Start':<22} {'End':<22} {'Dur(h)':<8} {'Mean_RR':<10} {'Peak_mean':<12} {'Max_any_SML':<12}")
for i, (s, e, dur, peak, avg, max_any) in enumerate(events[:20]):
    ts = base_time + timedelta(minutes=float(rad_time[s]))
    te = base_time + timedelta(minutes=float(rad_time[e]))
    print(f"  {i+1:<2} {str(ts):<22} {str(te):<22} {dur/60:<8.1f} {avg:<10.2f} {peak:<12.2f} {max_any:<12.2f}")

ds.close()

# =============================================================================
print("\n" + "=" * 80)
print("PART 6: RSL-RAIN PREVIEW (Heaviest Event)")
print("=" * 80)

if events:
    ev = events[0]  # heaviest by max_any_SML

    ds_sml = nc.Dataset(files['sml_data_2022.nc'])
    ds_rad = nc.Dataset(files['radar_along_sml_data_2022.nc'])

    sml_time = ds_sml.variables['time'][:]
    rsl_ma = ds_sml.variables['rsl'][:]
    rsl_f = rsl_ma.filled(np.nan).astype(float) if isinstance(rsl_ma, np.ma.MaskedArray) else rsl_ma.astype(float)
    rsl_f[rsl_f < -200] = np.nan

    rad_time_ev = ds_rad.variables['time'][:]
    rad_ma_ev = ds_rad.variables['rainfall_amount'][:]
    rad_ev = rad_ma_ev.filled(np.nan).astype(float) if isinstance(rad_ma_ev, np.ma.MaskedArray) else rad_ma_ev.astype(float)
    rad_ev[rad_ev < 0] = np.nan

    elev_arr = read_masked(ds_sml.variables['satellite_elevation'])

    # Event window ±2h
    ev_s_min = float(rad_time_ev[ev[0]]) - 120
    ev_e_min = float(rad_time_ev[ev[1]]) + 120

    sml_idx = np.where((sml_time >= ev_s_min) & (sml_time <= ev_e_min))[0]
    rad_idx = np.where((rad_time_ev >= ev_s_min) & (rad_time_ev <= ev_e_min))[0]

    ts = base_time + timedelta(minutes=float(rad_time_ev[ev[0]]))
    te = base_time + timedelta(minutes=float(rad_time_ev[ev[1]]))
    print(f"\n  Event: {ts} to {te}")
    print(f"  Max rain (any SML): {ev[5]:.2f} mm/h")
    print(f"  Window ±2h: {len(sml_idx)} SML steps, {len(rad_idx)} radar steps")

    # Find SMLs with highest rain during event
    # rad shape: (time, sml_id) → rad[rad_idx, :] gives (window_time, 215)
    rad_window = rad_ev[rad_idx, :]  # (time_window, 215)
    max_rr_per_sml_event = np.nanmax(rad_window, axis=0) * 12  # per SML peak

    top5 = np.argsort(max_rr_per_sml_event)[-5:][::-1]

    print(f"\n--- Top 5 SMLs by peak rain during event ---")
    print(f"  {'SML#':<6} {'elev°':<8} {'peak_RR':<10} {'RSL_base':<10} {'RSL_min':<10} {'max_atten':<10}")

    for si in top5:
        peak_rr = max_rr_per_sml_event[si]

        # RSL during event
        rsl_chunk = rsl_f[si, sml_idx]
        rsl_v = rsl_chunk[~np.isnan(rsl_chunk)]

        if len(rsl_v) < 10:
            continue

        # Baseline from pre-event dry period
        pre_idx = np.where((sml_time >= ev_s_min) & (sml_time < float(rad_time_ev[ev[0]])))[0]
        pre_rsl = rsl_f[si, pre_idx]
        pre_v = pre_rsl[~np.isnan(pre_rsl)]

        if len(pre_v) > 10:
            baseline = np.percentile(pre_v, 95)
        else:
            baseline = np.percentile(rsl_v, 95)

        rsl_min = np.min(rsl_v)
        max_atten = baseline - rsl_min

        print(f"  {si:<6} {elev_arr[si]:<8.2f} {peak_rr:<10.2f} {baseline:<10.2f} {rsl_min:<10.2f} {max_atten:<10.2f}")

    ds_sml.close()
    ds_rad.close()

# =============================================================================
print("\n" + "=" * 80)
print("DONE — Copy everything above to Claude")
print("=" * 80)