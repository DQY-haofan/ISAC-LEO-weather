"""
OpenSat4Weather Dataset Explorer
=================================
Run this script locally to explore the three NetCDF files.
Copy the FULL console output back to Claude for next-step planning.

Requirements: pip install netCDF4 numpy matplotlib
"""

import netCDF4 as nc
import numpy as np
import os
from datetime import datetime, timedelta

DATA_DIR = r"F:\Github-repositories\ISAC-LEO-weather\OpenSat4Weather_data"

# =============================================================================
# PART 1: File inventory
# =============================================================================
print("=" * 80)
print("PART 1: FILE INVENTORY")
print("=" * 80)

files = {}
for fname in os.listdir(DATA_DIR):
    if fname.endswith('.nc'):
        fpath = os.path.join(DATA_DIR, fname)
        size_mb = os.path.getsize(fpath) / 1e6
        files[fname] = fpath
        print(f"  {fname:45s} {size_mb:.1f} MB")

print(f"\nTotal files: {len(files)}")

# =============================================================================
# PART 2: SML data (satellite microwave link RSL)
# =============================================================================
print("\n" + "=" * 80)
print("PART 2: SML DATA — sml_data_2022.nc")
print("=" * 80)

ds_sml = nc.Dataset(files['sml_data_2022.nc'])

print("\n--- Dimensions ---")
for dim_name, dim in ds_sml.dimensions.items():
    print(f"  {dim_name:30s} size = {len(dim)}")

print("\n--- Variables ---")
for var_name, var in ds_sml.variables.items():
    shape_str = str(var.shape) if var.shape else "scalar"
    dtype = var.dtype
    # Get attributes
    attrs = {a: var.getncattr(a) for a in var.ncattrs()} if var.ncattrs() else {}
    print(f"  {var_name:30s} shape={shape_str:20s} dtype={dtype}")
    for ak, av in attrs.items():
        print(f"    @{ak} = {av}")

# Time range
time_var = ds_sml.variables['time'][:]
t_min, t_max = np.nanmin(time_var), np.nanmax(time_var)
base_time = datetime(2022, 8, 1, 0, 0, 0)  # UTC
t_start = base_time + timedelta(minutes=float(t_min))
t_end = base_time + timedelta(minutes=float(t_max))
print(f"\n--- Time ---")
print(f"  Range: {t_min:.0f} to {t_max:.0f} minutes from {base_time}")
print(f"  => {t_start} to {t_end}")
print(f"  Total time steps: {len(time_var)}")
dt = np.diff(time_var[:100])
print(f"  Sampling interval (first 100): min={np.min(dt):.1f}, max={np.max(dt):.1f}, median={np.median(dt):.1f} min")

# SML metadata
n_sml = ds_sml.dimensions['sml_id'].size
print(f"\n--- SML Metadata ({n_sml} links) ---")

# IDs
sml_ids = nc.chartostring(ds_sml.variables['sml_id'][:]) if ds_sml.variables['sml_id'].dtype.kind in ('S', 'U', 'O') else ds_sml.variables['sml_id'][:]
try:
    sml_ids = nc.chartostring(ds_sml.variables['sml_id'][:])
except:
    sml_ids = ds_sml.variables['sml_id'][:]
print(f"  Sample IDs: {list(sml_ids[:5])}")

# Coordinates
for vname in ['site_0_lat', 'site_0_lon', 'site_0_alt']:
    if vname in ds_sml.variables:
        v = ds_sml.variables[vname][:]
        print(f"  {vname:25s}: min={np.nanmin(v):.4f}, max={np.nanmax(v):.4f}, mean={np.nanmean(v):.4f}")

# Satellite geometry
for vname in ['satellite_azimuth', 'satellite_elevation']:
    if vname in ds_sml.variables:
        v = ds_sml.variables[vname][:]
        print(f"  {vname:25s}: min={np.nanmin(v):.2f}, max={np.nanmax(v):.2f}, mean={np.nanmean(v):.2f}")

# Satellite names
if 'satellite' in ds_sml.variables:
    try:
        sat_names = nc.chartostring(ds_sml.variables['satellite'][:])
        unique_sats = np.unique(sat_names)
        print(f"  Satellites: {list(unique_sats)}")
    except:
        print(f"  Satellites: (could not decode)")

# Hardware
if 'hardware' in ds_sml.variables:
    try:
        hw = nc.chartostring(ds_sml.variables['hardware'][:])
        unique_hw = np.unique(hw)
        print(f"  Hardware versions: {list(unique_hw)}")
    except:
        print(f"  Hardware: (could not decode)")

# RSL statistics
print(f"\n--- RSL (Received Signal Level) ---")
rsl = ds_sml.variables['rsl'][:]  # shape: (sml_id, time)
print(f"  Shape: {rsl.shape}")
print(f"  dtype: {rsl.dtype}")

# Handle masked arrays
if hasattr(rsl, 'mask'):
    total_cells = rsl.size
    valid_cells = np.count_nonzero(~rsl.mask)
    print(f"  Total cells: {total_cells:,}")
    print(f"  Valid cells: {valid_cells:,} ({valid_cells/total_cells*100:.1f}%)")
    print(f"  Missing: {total_cells - valid_cells:,} ({(total_cells-valid_cells)/total_cells*100:.1f}%)")
    rsl_flat = rsl.compressed()
else:
    nan_count = np.count_nonzero(np.isnan(rsl))
    print(f"  NaN count: {nan_count:,} / {rsl.size:,} ({nan_count/rsl.size*100:.1f}%)")
    rsl_flat = rsl[~np.isnan(rsl)]

print(f"  Valid RSL stats: min={np.min(rsl_flat):.2f}, max={np.max(rsl_flat):.2f}, "
      f"mean={np.mean(rsl_flat):.2f}, std={np.std(rsl_flat):.2f} dBm")

# Per-SML availability
print(f"\n--- Per-SML Data Availability ---")
avail_per_sml = []
for i in range(min(n_sml, rsl.shape[0])):
    row = rsl[i, :]
    if hasattr(row, 'mask'):
        valid = np.count_nonzero(~row.mask)
    else:
        valid = np.count_nonzero(~np.isnan(row))
    avail_per_sml.append(valid)

avail_arr = np.array(avail_per_sml)
n_time = rsl.shape[1]
print(f"  Per-SML valid timesteps: min={np.min(avail_arr)}, max={np.max(avail_arr)}, "
      f"mean={np.mean(avail_arr):.0f} / {n_time} total")
print(f"  SMLs with >80% availability: {np.count_nonzero(avail_arr > 0.8*n_time)} / {n_sml}")
print(f"  SMLs with >50% availability: {np.count_nonzero(avail_arr > 0.5*n_time)} / {n_sml}")

# 0-degree isotherm
if 'deg0l' in ds_sml.variables:
    deg0l = ds_sml.variables['deg0l'][:]
    if hasattr(deg0l, 'compressed'):
        deg0l_flat = deg0l.compressed()
    else:
        deg0l_flat = deg0l[~np.isnan(deg0l)]
    print(f"\n--- 0°C Isotherm Height (deg0l) ---")
    print(f"  Shape: {deg0l.shape}")
    print(f"  Stats: min={np.min(deg0l_flat):.0f}, max={np.max(deg0l_flat):.0f}, "
          f"mean={np.mean(deg0l_flat):.0f} m AGL")

ds_sml.close()

# =============================================================================
# PART 3: Rain gauge data
# =============================================================================
print("\n" + "=" * 80)
print("PART 3: RAIN GAUGE DATA — rg_data_2022.nc")
print("=" * 80)

ds_rg = nc.Dataset(files['rg_data_2022.nc'])

print("\n--- Dimensions ---")
for dim_name, dim in ds_rg.dimensions.items():
    print(f"  {dim_name:30s} size = {len(dim)}")

print("\n--- Variables ---")
for var_name, var in ds_rg.variables.items():
    shape_str = str(var.shape) if var.shape else "scalar"
    print(f"  {var_name:30s} shape={shape_str:20s} dtype={var.dtype}")

# Time
rg_time = ds_rg.variables['time'][:]
rg_t_start = base_time + timedelta(minutes=float(np.nanmin(rg_time)))
rg_t_end = base_time + timedelta(minutes=float(np.nanmax(rg_time)))
print(f"\n--- Time ---")
print(f"  Range: {rg_t_start} to {rg_t_end}")
print(f"  Time steps: {len(rg_time)}")
rg_dt = np.diff(rg_time[:100])
print(f"  Sampling interval: min={np.min(rg_dt):.1f}, max={np.max(rg_dt):.1f}, median={np.median(rg_dt):.1f} min")

# Gauge metadata
n_rg = ds_rg.dimensions['id'].size
print(f"\n--- {n_rg} Rain Gauges ---")

for vname in ['lat', 'lon', 'elev']:
    if vname in ds_rg.variables:
        v = ds_rg.variables[vname][:]
        print(f"  {vname:10s}: min={np.nanmin(v):.4f}, max={np.nanmax(v):.4f}")

# Rainfall stats
rain = ds_rg.variables['rainfall_amount'][:]
print(f"\n--- Rainfall Data ---")
print(f"  Shape: {rain.shape}")

if hasattr(rain, 'mask'):
    rain_flat = rain.compressed()
    valid_pct = rain_flat.size / rain.size * 100
else:
    rain_flat = rain[~np.isnan(rain)]
    valid_pct = rain_flat.size / rain.size * 100

print(f"  Valid data: {valid_pct:.1f}%")
print(f"  Rainfall stats (6-min accumulation, mm):")
print(f"    min={np.min(rain_flat):.3f}, max={np.max(rain_flat):.3f}, mean={np.mean(rain_flat):.4f}")

# Non-zero rain
rain_nonzero = rain_flat[rain_flat > 0]
print(f"  Non-zero entries: {len(rain_nonzero):,} / {len(rain_flat):,} ({len(rain_nonzero)/len(rain_flat)*100:.2f}%)")
if len(rain_nonzero) > 0:
    print(f"  Non-zero stats: mean={np.mean(rain_nonzero):.3f}, median={np.median(rain_nonzero):.3f}, "
          f"max={np.max(rain_nonzero):.3f} mm/6min")
    # Convert to rain rate (mm/h)
    rr_nonzero = rain_nonzero * 10  # 6-min accumulation * 10 = mm/h
    print(f"  Rain rate (mm/h): mean={np.mean(rr_nonzero):.2f}, median={np.median(rr_nonzero):.2f}, "
          f"p95={np.percentile(rr_nonzero, 95):.2f}, max={np.max(rr_nonzero):.2f}")

ds_rg.close()

# =============================================================================
# PART 4: Radar data along SML paths
# =============================================================================
print("\n" + "=" * 80)
print("PART 4: RADAR DATA — radar_along_sml_data_2022.nc")
print("=" * 80)

ds_rad = nc.Dataset(files['radar_along_sml_data_2022.nc'])

print("\n--- Dimensions ---")
for dim_name, dim in ds_rad.dimensions.items():
    print(f"  {dim_name:30s} size = {len(dim)}")

print("\n--- Variables ---")
for var_name, var in ds_rad.variables.items():
    shape_str = str(var.shape) if var.shape else "scalar"
    print(f"  {var_name:30s} shape={shape_str:20s} dtype={var.dtype}")

# Time
rad_time = ds_rad.variables['time'][:]
rad_t_start = base_time + timedelta(minutes=float(np.nanmin(rad_time)))
rad_t_end = base_time + timedelta(minutes=float(np.nanmax(rad_time)))
print(f"\n--- Time ---")
print(f"  Range: {rad_t_start} to {rad_t_end}")
print(f"  Time steps: {len(rad_time)}")

# Radar rainfall along SML paths
rad_rain = ds_rad.variables['rainfall_amount'][:]
print(f"\n--- Radar Rainfall along SML paths ---")
print(f"  Shape: {rad_rain.shape}")

if hasattr(rad_rain, 'mask'):
    rad_flat = rad_rain.compressed()
    rad_valid_pct = rad_flat.size / rad_rain.size * 100
else:
    rad_flat = rad_rain[~np.isnan(rad_rain)]
    rad_valid_pct = rad_flat.size / rad_rain.size * 100

print(f"  Valid data: {rad_valid_pct:.1f}%")
if len(rad_flat) > 0:
    print(f"  Stats (5-min accumulation, mm): min={np.min(rad_flat):.4f}, max={np.max(rad_flat):.4f}, "
          f"mean={np.mean(rad_flat):.5f}")
    rad_nonzero = rad_flat[rad_flat > 0]
    print(f"  Non-zero: {len(rad_nonzero):,} / {len(rad_flat):,} ({len(rad_nonzero)/len(rad_flat)*100:.2f}%)")
    if len(rad_nonzero) > 0:
        rr_rad = rad_nonzero * 12  # 5-min to mm/h
        print(f"  Rain rate (mm/h): mean={np.mean(rr_rad):.2f}, median={np.median(rr_rad):.2f}, "
              f"p95={np.percentile(rr_rad, 95):.2f}, max={np.max(rr_rad):.2f}")

ds_rad.close()

# =============================================================================
# PART 5: Rain event identification
# =============================================================================
print("\n" + "=" * 80)
print("PART 5: RAIN EVENT IDENTIFICATION")
print("=" * 80)

# Reopen radar data for event finding
ds_rad = nc.Dataset(files['radar_along_sml_data_2022.nc'])
rad_rain = ds_rad.variables['rainfall_amount'][:]
rad_time = ds_rad.variables['time'][:]

# Compute spatial-mean rain rate per time step
# Average across all SMLs for each time step
if hasattr(rad_rain, 'mask'):
    mean_rain_per_time = np.array([
        np.mean(rad_rain[:, t].compressed()) if rad_rain[:, t].compressed().size > 0 else 0
        for t in range(rad_rain.shape[1])
    ])
else:
    mean_rain_per_time = np.nanmean(rad_rain, axis=0)

# Convert to rain rate (mm/h) from 5-min accumulation
mean_rr = mean_rain_per_time * 12

# Find heavy rain periods (mean rain rate > 2 mm/h)
heavy_idx = np.where(mean_rr > 2)[0]
print(f"\nTime steps with spatial-mean rain rate > 2 mm/h: {len(heavy_idx)} / {len(mean_rr)}")

# Find top 10 rainiest moments
top_idx = np.argsort(mean_rr)[-10:][::-1]
print(f"\n--- Top 10 Rainiest Time Steps ---")
print(f"{'Rank':<6} {'Time (UTC)':<22} {'Mean RR (mm/h)':<16} {'Minute offset':<14}")
for rank, idx in enumerate(top_idx):
    t = base_time + timedelta(minutes=float(rad_time[idx]))
    print(f"  {rank+1:<4} {str(t):<22} {mean_rr[idx]:<16.2f} {rad_time[idx]:<14.0f}")

# Identify distinct rain events (clusters of consecutive rainy time steps)
rain_mask = mean_rr > 0.5  # > 0.5 mm/h threshold
event_starts = []
event_ends = []
in_event = False
for i in range(len(rain_mask)):
    if rain_mask[i] and not in_event:
        event_starts.append(i)
        in_event = True
    elif not rain_mask[i] and in_event:
        event_ends.append(i - 1)
        in_event = False
if in_event:
    event_ends.append(len(rain_mask) - 1)

print(f"\n--- Rain Events (threshold > 0.5 mm/h) ---")
print(f"Total events: {len(event_starts)}")

# Show top events by duration and intensity
events = []
for s, e in zip(event_starts, event_ends):
    duration_min = float(rad_time[e] - rad_time[s])
    peak_rr = np.max(mean_rr[s:e+1])
    mean_rr_event = np.mean(mean_rr[s:e+1])
    t_start_ev = base_time + timedelta(minutes=float(rad_time[s]))
    t_end_ev = base_time + timedelta(minutes=float(rad_time[e]))
    events.append((t_start_ev, t_end_ev, duration_min, peak_rr, mean_rr_event, s, e))

# Sort by peak rain rate
events.sort(key=lambda x: x[3], reverse=True)
print(f"\n--- Top 10 Events by Peak Rain Rate ---")
print(f"{'#':<4} {'Start (UTC)':<22} {'End (UTC)':<22} {'Dur (h)':<8} {'Peak RR':<10} {'Mean RR':<10}")
for i, ev in enumerate(events[:10]):
    dur_h = ev[2] / 60
    print(f"  {i+1:<2} {str(ev[0]):<22} {str(ev[1]):<22} {dur_h:<8.1f} {ev[3]:<10.2f} {ev[4]:<10.2f}")

ds_rad.close()

# =============================================================================
# PART 6: RSL-Rain correlation preview (one heavy event)
# =============================================================================
print("\n" + "=" * 80)
print("PART 6: RSL-RAIN CORRELATION PREVIEW")
print("=" * 80)

if events:
    # Take the heaviest event
    best_event = events[0]
    print(f"\nAnalyzing heaviest event: {best_event[0]} to {best_event[1]}")
    print(f"  Peak rain rate: {best_event[3]:.2f} mm/h")
    
    # Reopen SML and radar
    ds_sml = nc.Dataset(files['sml_data_2022.nc'])
    ds_rad = nc.Dataset(files['radar_along_sml_data_2022.nc'])
    
    rsl = ds_sml.variables['rsl'][:]
    sml_time = ds_sml.variables['time'][:]
    rad_rain = ds_rad.variables['rainfall_amount'][:]
    rad_time = ds_rad.variables['time'][:]
    
    # Event window (extend by 2h each side for baseline)
    ev_s_min = float(rad_time[best_event[5]]) - 120
    ev_e_min = float(rad_time[best_event[6]]) + 120
    
    # Find SML time indices in window
    sml_mask = (sml_time >= ev_s_min) & (sml_time <= ev_e_min)
    sml_idx = np.where(sml_mask)[0]
    
    rad_mask_t = (rad_time >= ev_s_min) & (rad_time <= ev_e_min)
    rad_idx = np.where(rad_mask_t)[0]
    
    print(f"  SML time steps in window: {len(sml_idx)}")
    print(f"  Radar time steps in window: {len(rad_idx)}")
    
    # Pick a well-sampled SML
    if len(sml_idx) > 0:
        # Find SML with best coverage in this window
        best_sml = -1
        best_valid = 0
        for i in range(rsl.shape[0]):
            chunk = rsl[i, sml_idx]
            if hasattr(chunk, 'mask'):
                valid = np.count_nonzero(~chunk.mask)
            else:
                valid = np.count_nonzero(~np.isnan(chunk))
            if valid > best_valid:
                best_valid = valid
                best_sml = i
        
        print(f"\n  Best SML for this event: index {best_sml}, {best_valid}/{len(sml_idx)} valid points")
        
        if best_sml >= 0:
            # Get elevation for this SML
            if 'satellite_elevation' in ds_sml.variables:
                elev = ds_sml.variables['satellite_elevation'][best_sml]
                print(f"  Satellite elevation: {float(elev):.2f}°")
            
            # RSL time series during event
            rsl_event = rsl[best_sml, sml_idx]
            if hasattr(rsl_event, 'mask'):
                rsl_valid = ~rsl_event.mask
                rsl_vals = rsl_event.data[rsl_valid]
            else:
                rsl_valid = ~np.isnan(rsl_event)
                rsl_vals = rsl_event[rsl_valid]
            
            if len(rsl_vals) > 0:
                # Estimate clear-sky baseline (max RSL in dry period)
                rsl_sorted = np.sort(rsl_vals)
                baseline = np.percentile(rsl_vals, 95)  # 95th percentile as baseline
                attenuation = baseline - rsl_vals  # dB attenuation
                
                print(f"\n  RSL during event:")
                print(f"    Baseline (P95): {baseline:.2f} dBm")
                print(f"    Min RSL: {np.min(rsl_vals):.2f} dBm")
                print(f"    Max attenuation: {baseline - np.min(rsl_vals):.2f} dB")
                print(f"    Mean attenuation: {np.mean(attenuation):.2f} dB")
            
            # Radar rain for same SML
            if len(rad_idx) > 0:
                rad_event = rad_rain[best_sml, rad_idx]
                if hasattr(rad_event, 'mask'):
                    rad_vals = rad_event.compressed()
                else:
                    rad_vals = rad_event[~np.isnan(rad_event)]
                
                rr_event = rad_vals * 12  # to mm/h
                print(f"\n  Radar rain rate along this SML path:")
                print(f"    Max: {np.max(rr_event):.2f} mm/h")
                print(f"    Mean: {np.mean(rr_event):.2f} mm/h")
    
    ds_sml.close()
    ds_rad.close()

# =============================================================================
# PART 7: Summary for Claude
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY — KEY NUMBERS FOR NEXT STEP")
print("=" * 80)
print("""
Copy everything above this line and paste it to Claude.
Key questions Claude will answer:
1. Is the data quality sufficient for CRB validation?
2. Which rain events are best for the conference paper figure?
3. How to compute attenuation from RSL and compare with ITU-R P.838?
""")
