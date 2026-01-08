import spikeinterface
print("SpikeInterface version:", spikeinterface.__version__)


# === Shijia AIND export with template metrics merge + best-channel templates + T2P + analysis_meta + diagnostics ===
import os
import glob
import json
import time
import argparse
import platform
import numpy as np
import pandas as pd
import spikeinterface as si

# -------- Command-line arguments --------
parser = argparse.ArgumentParser(
    description="Extract AIND-format outputs from spike sorting results."
)
parser.add_argument(
    '--session', '-s',
    type=str,
    default=None,
    help="Path to a specific session output folder to process "
         "(e.g., /path/to/aind_output_scratch/20251207-SL412-Reward1_g0_output/). "
         "If not provided, discovers sessions from AIND_INPUT_BASE_DIR."
)
parser.add_argument(
    '--skip-existing',
    action='store_true',
    default=False,
    help="Skip sessions that already have an AIND export (analysis_meta.json exists)."
)
args = parser.parse_args()

# -------- Base folder paths --------
# Allow overriding the default locations via environment variables so the
# pipeline script can drive which directory of recording sessions to process.
input_base_dir  = os.environ.get(
    'AIND_INPUT_BASE_DIR',
    '/n/netscratch/bsabatini_lab/Lab/shunnnli/spikesorting/aind_input/todo'
)
output_base_dir = os.environ.get(
    'AIND_OUTPUT_BASE_DIR',
    '/n/netscratch/bsabatini_lab/Lab/shunnnli/spikesorting/aind_output_scratch'
)
skip_existing = args.skip_existing

# -------- Discover raw-recording subfolders and derive session names --------
all_raw_folders = []
session_names   = []

if args.session:
    # --session mode: process a single specified output folder directly
    session_path = args.session.rstrip('/')
    if not os.path.isdir(session_path):
        raise ValueError(f"Session path does not exist: {session_path}")
    
    # Derive session name from folder name (strip _output suffix if present)
    folder_name = os.path.basename(session_path)
    if folder_name.endswith('_output'):
        session_name = folder_name[:-7]  # remove '_output'
    else:
        session_name = folder_name
    
    # For --session mode, we set output_base_dir to the parent and use a dummy raw folder
    output_base_dir = os.path.dirname(session_path)
    all_raw_folders = [session_path]  # Use session path as placeholder for raw folder
    session_names = [session_name]
    print(f"[--session mode] Processing single session: {session_name}")
    print(f"  Output folder: {session_path}")
else:
    # Default mode: discover sessions from input_base_dir
    for item in sorted(os.listdir(input_base_dir)):
        full_path = os.path.join(input_base_dir, item)
        if not os.path.isdir(full_path):
            continue

        if item.endswith('_imec0') or item.endswith('_imec1'):
            all_raw_folders.append(full_path)
            session_names.append('_'.join(item.split('_')[:-1]))
        else:
            for sd in sorted(os.listdir(full_path)):
                if ('imec0' in sd or 'imec1' in sd) and os.path.isdir(os.path.join(full_path, sd)):
                    all_raw_folders.append(os.path.join(full_path, sd))
                    session_names.append(item)

# -------- Helpers --------
def find_experiment_files(preproc_path):
    files = glob.glob(os.path.join(preproc_path, 'block0_imec*.ap_recording1*.json'))
    files.sort()
    return files

def _parse_ap_meta(meta_path):
    out_raw = {}
    if meta_path and os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if "=" in line:
                    k, v = line.split("=", 1)
                    out_raw[k.strip()] = v.strip()
    out_sum = {}
    if "imSampRate" in out_raw:
        try: out_sum["sampling_rate_hz"] = float(out_raw["imSampRate"])
        except: pass
    if "nSavedChans" in out_raw:
        try: out_sum["n_saved_channels"] = int(out_raw["nSavedChans"])
        except: pass
    if "fileTimeSecs" in out_raw:
        try: out_sum["file_time_secs"] = float(out_raw["fileTimeSecs"])
        except: pass
    if "fileName" in out_raw:
        out_sum["ap_bin_path"] = out_raw["fileName"]
    return {"raw_meta": out_raw, "summary": out_sum}

def check_template_metrics_merge(aind_folder):
    tsv_tm  = os.path.join(aind_folder, "template_metrics.tsv")
    tsv_ci  = os.path.join(aind_folder, "cluster_info.tsv")

    print(f"\n[diagnostics] AIND folder: {aind_folder}")

    if not os.path.exists(tsv_ci):
        print("❌ cluster_info.tsv not found.")
        return
    ci = pd.read_csv(tsv_ci, sep="\t")
    print(f"✅ cluster_info.tsv rows={len(ci)}, cols={len(ci.columns)}")

    if not os.path.exists(tsv_tm):
        print("⚠️ template_metrics.tsv not found (merge likely skipped).")
        candidate_cols = [c for c in ci.columns if any(k in c.lower() for k in
                             ["trough", "peak", "half", "width", "ptp", "duration", "repolar"])]
        if candidate_cols:
            print(f"   • Found template-like columns in cluster_info: {candidate_cols[:8]}")
        else:
            print("   • No template-like columns detected in cluster_info.")
        return

    tm = pd.read_csv(tsv_tm, sep="\t")
    print(f"✅ template_metrics.tsv rows={len(tm)}, cols={len(tm.columns)}")
    key_cols = {"global_unit_ids", "unit_id", "unit_ids"}
    tm_metric_cols = [c for c in tm.columns if c not in key_cols]
    present = [c for c in tm_metric_cols if c in ci.columns]
    missing = [c for c in tm_metric_cols if c not in ci.columns]
    if present:
        print(f"🎉 Template metrics present in cluster_info (examples): {present[:8]}")
    else:
        print("❌ No template-metric columns found in cluster_info.")
    if missing:
        print(f"ℹ️ Some template-metric columns not found in cluster_info (first few): {missing[:8]}")
    key_in_tm  = "global_unit_ids" if "global_unit_ids" in tm.columns else ("unit_id" if "unit_id" in tm.columns else None)
    key_in_ci  = "global_unit_ids" if "global_unit_ids" in ci.columns else None
    if key_in_tm and key_in_ci:
        covered = tm[key_in_tm].nunique()
        in_ci   = ci[key_in_ci].nunique()
        print(f"🔗 Join keys — template_metrics unique={covered}, cluster_info unique={in_ci}")
    else:
        print(f"⚠️ Could not compare join keys (key_in_tm={key_in_tm}, key_in_ci={key_in_ci})")

# --- internal helpers for templates access and T2P ---
def _fetch_templates_array(sa):
    """
    Try several access paths across SI 0.102.x.
    Return a numpy array with shape (n_units, n_samples, n_channels) or
    (n_units, n_channels, n_samples). Also return string axis_mode:
      - 'usc' = (units, samples, channels)
      - 'ucs' = (units, channels, samples)
    """
    try:
        sa.load_extension('templates')
    except Exception:
        sa.compute(['templates'], n_jobs=4, progress_bar=False)
    ext = sa.get_extension('templates')
    arr = None
    # 1) get_data -> dict['templates']
    try:
        data = ext.get_data()
        if isinstance(data, dict) and isinstance(data.get('templates', None), np.ndarray):
            arr = data['templates']
    except Exception:
        pass
    # 2) get_all_templates / get_templates
    if arr is None and hasattr(ext, 'get_all_templates'):
        try: arr = ext.get_all_templates()
        except Exception: pass
    if arr is None and hasattr(ext, 'get_templates'):
        try: arr = ext.get_templates()
        except Exception: pass
    if not (isinstance(arr, np.ndarray) and arr.ndim == 3):
        return None, None

    # Determine axis order using explicit parameters from SpikeInterface,
    # NOT by guessing from array shape (which is unreliable).
    n_units_arr, a1, a2 = arr.shape
    
    # Try to get expected n_samples from templates extension params
    n_samples_expected = 210
    try:
        params = ext.params
        if 'nbefore' in params and 'nafter' in params:
            n_samples_expected = params['nbefore'] + params['nafter']
    except Exception:
        pass
    
    # Try to get expected n_channels from the recording (default 384 for Neuropixels)
    n_channels_expected = 384
    try:
        n_channels_expected = sa.get_num_channels()
    except Exception:
        pass
    
    # Debug output to help diagnose issues
    # print(f"   [templates] shape={arr.shape}, n_samples_expected={n_samples_expected}, n_channels_expected={n_channels_expected}")
    
    # Determine axis order based on explicit parameters
    if n_samples_expected is not None:
        if a1 == n_samples_expected:
            # print(f"   [templates] Determinedaxis order: (units, samples, channels), shape={arr.shape}")
            return arr, 'usc'  # (units, samples, channels)
        elif a2 == n_samples_expected:
            # print(f"   [templates] Determined axis order: (units, channels, samples), shape={arr.shape}")
            return arr, 'ucs'  # (units, channels, samples)
    
    if n_channels_expected is not None:
        if a2 == n_channels_expected:
            # print(f"   [templates] Determined axis order: (units, samples, channels), shape={arr.shape}")
            return arr, 'usc'  # (units, samples, channels)
        elif a1 == n_channels_expected:
            # print(f"   [templates] Determined axis order: (units, channels, samples), shape={arr.shape}")
            return arr, 'ucs'  # (units, channels, samples)
    
    # Fallback: assume SpikeInterface's default format is (units, samples, channels)
    # This is the standard format in SI 0.100+
    print(f"   [templates] WARNING: Could not determine axis order from params, assuming 'usc' (units, samples, channels)")
    return arr, 'usc'

def _ptp_per_channel(template_unit, axis_mode):
    """Return per-channel peak-to-peak amplitude vector for one unit template."""
    if axis_mode == 'usc':   # 'usc' -> (samples, channels)
        # template_unit shape: (n_samples, n_channels); max/min along axis=0
        return (template_unit.max(axis=0) - template_unit.min(axis=0))
    else:                     # 'ucs' -> (channels, samples)
        # template_unit shape: (n_channels, n_samples); max/min along axis=1
        return (template_unit.max(axis=1) - template_unit.min(axis=1))

def _extract_best_channel_waveform(templates_arr, unit_index, best_ch, axis_mode):
    if axis_mode == 'usc':   # (units, samples, channels)
        return templates_arr[unit_index, :, best_ch]
    else:                     # 'ucs' -> (units, channels, samples)
        return templates_arr[unit_index, best_ch, :]

def _trough_to_peak_ms(wf_1d, fs_hz):
    """Compute trough-to-peak latency (ms) on a 1-D waveform."""
    if wf_1d is None or len(wf_1d) < 3:
        return np.nan
    i_trough = int(np.argmin(wf_1d))
    # peak after trough; if none, use absolute peak
    if i_trough < len(wf_1d) - 1:
        i_peak_rel = int(np.argmax(wf_1d[i_trough:]))
        i_peak = i_trough + i_peak_rel
    else:
        i_peak = int(np.argmax(wf_1d))
    dt_samples = max(i_peak - i_trough, 0)
    return (dt_samples / float(fs_hz)) * 1000.0

# -------- Main processing loop --------
for raw_rec, session_name in zip(all_raw_folders, session_names):
    print(f"\n=== Session: {session_name} ===")

    if args.session:
        # --session mode: use the provided path directly
        baseFolder = args.session.rstrip('/')
        print(f"→ using output folder (--session): {baseFolder}")
    else:
        # Default mode: match session output folder by name
        matches = [d for d in os.listdir(output_base_dir)
                   if session_name in d 
                   and os.path.isdir(os.path.join(output_base_dir, d))]
        if not matches:
            print(f"⚠️  No output folder found for '{session_name}' in {output_base_dir}, skipping.")
            continue

        output_folder = sorted(matches)[0]
        baseFolder    = os.path.join(output_base_dir, output_folder)
        print(f"→ using output folder: {output_folder}")

    # Prepare AIND output dir
    AIND_folder = os.path.join(baseFolder, f'AIND_{session_name}')
    # If we've already exported this session (based on presence of analysis_meta.json),
    # skip re-running the full AIND export to save time, unless the caller explicitly
    # disables skipping via the AIND_SKIP_EXISTING_EXPORT environment variable.
    sentinel_path = os.path.join(AIND_folder, "analysis_meta.json")
    if skip_existing and os.path.exists(sentinel_path):
        print(f"[skip] AIND export already exists for '{session_name}' at {AIND_folder}; skipping.")
        continue
    os.makedirs(AIND_folder, exist_ok=True)

    # define subfolders
    preProcessed = os.path.join(baseFolder, 'preprocessed')
    postProcessed= os.path.join(baseFolder, 'postprocessed')
    spikes       = os.path.join(baseFolder, 'spikesorted')
    curated      = os.path.join(baseFolder, 'curated')

    experiment_files = find_experiment_files(preProcessed)
    if not experiment_files:
        print("⚠️  No experiment JSONs found under:", preProcessed)
        continue

    # accumulators
    total_units = 0
    all_spike_times    = []
    all_spike_clusters = []
    unit_labels_combined      = []
    qm_combined_with_global_ids = []
    tm_all_segments = []       # template metrics per segment (GLOBAL-keyed)
    uloc_all_segments = []     # unit locations per segment (GLOBAL-keyed, optional)

    # best-channel outputs
    best_templates_all = []    # list of 1-D arrays (best-channel template)
    best_templates_meta = []   # dicts with global_unit_ids, peak_channel, segment_name, n_samples, trough_to_peak_ms

    global_unit_counter = 1

    for i, expf in enumerate(experiment_files, start=1):
        print(f" • [{i}/{len(experiment_files)}] {expf}")
        exp_base = os.path.splitext(os.path.basename(expf))[0]
        zarr_path = os.path.join(postProcessed, exp_base + '.zarr')
        if not os.path.exists(zarr_path):
            print("   – Missing zarr:", zarr_path)
            continue

        # load sorting & spikes with SI 0.102.x API
        sorting_analyzer = si.load_sorting_analyzer(zarr_path, load_extensions=False)
        sorting_curated  = si.load(os.path.join(curated, exp_base))
        spike_extractor  = si.load(os.path.join(spikes,   exp_base))

        unit_ids    = sorting_curated.get_unit_ids()
        labels      = sorting_curated.get_property('decoder_label')
        default_qc_vals = sorting_curated.get_property('default_qc')
        fs_hz    = sorting_curated.get_sampling_frequency()
        print(f"   → {len(unit_ids)} units")

        # quality metrics
        sorting_analyzer.load_extension('quality_metrics')
        qm = sorting_analyzer.get_extension('quality_metrics').get_data()
        qm_df = pd.DataFrame(qm)
        if "unit_ids" not in qm_df.columns:
            qm_df['unit_ids'] = unit_ids  # LOCAL IDs (keep this column)

        # From Tom, added 20250911: add unit locations (x, y in µm; typically x = lateral, y = depth)
        sorting_analyzer.load_extension('unit_locations')
        unit_locs = sorting_analyzer.get_extension('unit_locations').get_data()  # shape (n_units, 2) or (n_units, 3)
        try:
            # common case: numpy array
            qm_df['x_um'] = unit_locs[:, 0]
            qm_df['y_um'] = unit_locs[:, 1]
        except Exception:
            # fallback if SI returns a DataFrame-like
            uloc_df = pd.DataFrame(unit_locs)
            if 'x_um' in uloc_df.columns and 'y_um' in uloc_df.columns:
                qm_df['x_um'] = uloc_df['x_um'].to_numpy()
                qm_df['y_um'] = uloc_df['y_um'].to_numpy()
            else:
                # last resort: 'x'/'y' names
                qm_df['x_um'] = uloc_df['x'].to_numpy()
                qm_df['y_um'] = uloc_df['y'].to_numpy()
            
        # unit locations (µm) — optional
        try:
            sorting_analyzer.load_extension('unit_locations')
            uloc = sorting_analyzer.get_extension('unit_locations').get_data()
            uloc_df = pd.DataFrame(uloc)
            if 'unit_id' not in uloc_df.columns:
                uloc_df = uloc_df.reset_index().rename(columns={'index':'unit_id'})
            uloc_df['unit_id'] = uloc_df['unit_id'].astype(int)
        except Exception as e:
            uloc_df = None
            print(f"   (info) unit_locations not found: {e}")

        # LOCAL → GLOBAL mapping for this block
        map_df = pd.DataFrame({
            'unit_ids': unit_ids,
            'global_unit_ids': np.arange(global_unit_counter, global_unit_counter + len(unit_ids))
        })

        # attach global ids to QC rows
        qm_df = qm_df.merge(map_df, on='unit_ids', how='left')
        qm_combined_with_global_ids.append(qm_df)

        # labels table (cluster_group.tsv). Always export labels; add default_qc
        # column when available.
        labels_df = pd.DataFrame({
            'global_unit_ids': map_df['global_unit_ids'],
            'labels': labels
        })
        labels_df['default_qc'] = np.asarray(default_qc_vals)
        unit_labels_combined.append(labels_df)

        # spikes (sample indices) per local unit, tagged with GLOBAL id
        for local_id, global_id in zip(map_df['unit_ids'], map_df['global_unit_ids']):
            try:
                stimes = spike_extractor.get_unit_spike_train(local_id)  # samples
                all_spike_times.extend(stimes)
                all_spike_clusters.extend([global_id] * len(stimes))
            except ValueError:
                pass

        # ---- per-segment TEMPLATE METRICS (Tom-style) ----
        try:
            try:
                sorting_analyzer.load_extension('template_metrics')
            except Exception:
                sorting_analyzer.compute(['templates', 'template_metrics'], n_jobs=4, progress_bar=False)
            tm_seg = sorting_analyzer.get_extension('template_metrics').get_data()
            tm_seg_df = pd.DataFrame(tm_seg)
            if 'unit_id' not in tm_seg_df.columns:
                tm_seg_df = tm_seg_df.reset_index().rename(columns={'index': 'unit_id'})
            tm_seg_df['unit_id'] = tm_seg_df['unit_id'].astype(int)
            # map LOCAL -> GLOBAL; keep ONLY global key here
            tm_seg_df = tm_seg_df.merge(map_df.rename(columns={'unit_ids':'unit_id'}),
                                        on='unit_id', how='left')
            tm_seg_df.drop(columns=['unit_id'], inplace=True)
            tm_all_segments.append(tm_seg_df)
        except Exception as e:
            print(f"   (info) template_metrics unavailable for this segment: {e}")
            tm_seg_df = None  # important for best-channel fallback below

        # ---- per-segment unit_locations mapped to GLOBAL (optional) ----
        if uloc_df is not None:
            uloc_df = uloc_df.merge(map_df.rename(columns={'unit_ids':'unit_id'}), on='unit_id', how='left')
            keep_cols = ['global_unit_ids']
            for col in ['x','y','x_um','y_um','z','z_um']:
                if col in uloc_df.columns: keep_cols.append(col)
            uloc_all_segments.append(uloc_df[keep_cols])

        # === BEST-CHANNEL TEMPLATES + T2P per segment ===
        # 1) Fetch templates array
        templates_arr, axis_mode = _fetch_templates_array(sorting_analyzer)
        print(f"   [best-channel] templates_arr shape={templates_arr.shape}, axis_mode={axis_mode}")
        if templates_arr is None:
            print("   (info) templates array not available; skipping best-channel export for this segment.")
        else:
            # 2) Determine best/peak channel per LOCAL unit
            peak_col = None
            if tm_seg_df is not None:
                for cand in ('peak_channel', 'best_channel', 'max_ptp_channel'):
                    if cand in tm_seg_df.columns:
                        peak_col = cand
                        break

            # Build LOCAL -> GLOBAL map with peak channel if present
            local_ids = pd.Series(unit_ids, name='unit_id')
            map_local = local_ids.to_frame().merge(
                map_df.rename(columns={'unit_ids':'unit_id'}), on='unit_id', how='left'
            )

            if peak_col is not None:
                # Recover peak channel per GLOBAL from tm_seg_df
                # (tm_seg_df currently only has global key; re-merge the original tm with local ids)
                try:
                    sorting_analyzer.load_extension('template_metrics')
                    tm_full = sorting_analyzer.get_extension('template_metrics').get_data()
                    tm_full = pd.DataFrame(tm_full)
                    if 'unit_id' not in tm_full.columns:
                        tm_full = tm_full.reset_index().rename(columns={'index':'unit_id'})
                    tm_full['unit_id'] = tm_full['unit_id'].astype(int)
                    map_local = map_local.merge(tm_full[['unit_id', peak_col]], on='unit_id', how='left')
                except Exception:
                    peak_col = None  # fall back to compute from PTP
                    print("   (info) peak channel column not retrievable from extension; falling back to PTP argmax.")

            # If no peak_col information, compute channel PTP and pick argmax
            if peak_col is None:
                # templates_arr shape: 'usc' = (units, S, C) or 'ucs' = (units, C, S)
                # We assume the unit order follows sorting_curated.get_unit_ids()
                for u_idx, u_local in enumerate(unit_ids):
                    unit_template = templates_arr[u_idx, :, :]  # (S, C) or (C, S)
                    ptp_vec = _ptp_per_channel(unit_template, axis_mode)
                    best_ch = int(np.argmax(ptp_vec))
                    map_local.loc[map_local['unit_id'] == u_local, 'peak_channel'] = best_ch

            # 3) Extract best-channel 1-D waveform + compute T2P, n_samples
            # Create lookup from LOCAL unit -> (GLOBAL id, peak_channel)
            local_to_global = dict(zip(map_local['unit_id'].tolist(), map_local['global_unit_ids'].tolist()))
            local_to_peak   = dict(zip(map_local['unit_id'].tolist(), map_local['peak_channel'].astype(int).tolist()))

            # n_samples from templates
            if axis_mode == 'usc':  # (units, samples, channels)
                n_samples_seg = int(templates_arr.shape[1])
            else:                   # 'ucs' = (units, channels, samples)
                n_samples_seg = int(templates_arr.shape[2])

            for u_idx, u_local in enumerate(unit_ids):
                g_uid = int(local_to_global.get(u_local, -1))
                if g_uid < 0:  # skip unmapped
                    continue
                best_ch = int(local_to_peak.get(u_local, 0))
                wf_1d = _extract_best_channel_waveform(templates_arr, u_idx, best_ch, axis_mode).astype(np.float32)
                t2p_ms = _trough_to_peak_ms(wf_1d, fs_hz)

                # print(f"   [best-channel] peak_channel={best_ch}, n_samples={n_samples_seg}, template_shape={templates_arr.shape}")

                best_templates_all.append(wf_1d)
                best_templates_meta.append({
                    'global_unit_ids': g_uid,
                    'peak_channel': best_ch,
                    'segment_name': exp_base,
                    'n_samples': n_samples_seg,
                    'trough_to_peak_ms': float(t2p_ms),
                })

        global_unit_counter += len(unit_ids)
        total_units += len(unit_ids)

    print(f" ✅ Total units: {total_units}")
    # if len(best_templates_meta) > 0:
    #     peak_channels = [m.get('peak_channel') for m in best_templates_meta if 'peak_channel' in m]
    #     unique_peaks = sorted(set(peak_channels)) if peak_channels else []
    #     print(f"   [best-channel] unique peak_channel values: {unique_peaks}")

    # -------- Concatenate per-session tables --------
    unit_labels_df = (pd.concat(unit_labels_combined, ignore_index=True)
                      if unit_labels_combined else pd.DataFrame())
    qm_combined_df = (pd.concat(qm_combined_with_global_ids, ignore_index=True)
                      if qm_combined_with_global_ids else pd.DataFrame())
    print("Keys in cluster_group.tsv:", unit_labels_df.columns.tolist())
    print("Keys in cluster_info.tsv:", qm_combined_df.columns.tolist())

    # -------- Save the three core artifacts --------
    spike_times    = np.array(all_spike_times)      # samples
    spike_clusters = np.array(all_spike_clusters)   # GLOBAL unit ids
    sort_idx       = np.argsort(spike_times)
    spike_times    = spike_times[sort_idx]
    spike_clusters = spike_clusters[sort_idx]

    np.save(os.path.join(AIND_folder, 'spike_times.npy'),    spike_times)
    np.save(os.path.join(AIND_folder, 'spike_clusters.npy'), spike_clusters)
    unit_labels_df.to_csv(os.path.join(AIND_folder, 'cluster_group.tsv'),
                         sep='\t', index=False)

    # -------- Stitch template metrics across segments (GLOBAL-keyed) --------
    if len(tm_all_segments) > 0:
        tm_all_df = pd.concat(tm_all_segments, ignore_index=True).drop_duplicates(subset=['global_unit_ids'])
        tm_all_df.to_csv(os.path.join(AIND_folder, "template_metrics.tsv"), sep="\t", index=False)
    else:
        tm_all_df = None

    # -------- Optional: stitch unit locations similarly --------
    if len(uloc_all_segments) > 0:
        uloc_all_df = pd.concat(uloc_all_segments, ignore_index=True).drop_duplicates(subset=['global_unit_ids'])
    else:
        uloc_all_df = None

    # -------- Build best_channel_templates.npy and its metadata --------
    if len(best_templates_all) > 0:
        # normalize lengths to the most common (pad with zeros or truncate)
        lengths = pd.Series([len(x) for x in best_templates_all])
        mode_len = int(lengths.mode().iloc[0])
        def _fixlen(x, L):
            if len(x) == L: return x
            if len(x) > L:  return x[:L]
            out = np.zeros(L, dtype=np.float32); out[:len(x)] = x; return out
        best_arr = np.vstack([_fixlen(x, mode_len)[None, :] for x in best_templates_all]).astype(np.float32)
        np.save(os.path.join(AIND_folder, "best_channel_templates.npy"), best_arr)

        best_meta_df = pd.DataFrame(best_templates_meta)
        # keep the latest row per global_unit_ids (in case of duplicates across segments)
        best_meta_df = best_meta_df.sort_values('segment_name').drop_duplicates(subset=['global_unit_ids'], keep='last')
        # NOTE: you asked not to keep an extra TSV; we only use this df to merge onto cluster_info
    else:
        best_meta_df = None

    # Check unique peak_channel values
    # print(f"   [best-channel] unique peak_channel values: {best_meta_df['peak_channel'].unique()}")

    # -------- Final cluster_info.tsv = QC + template metrics + best-channel info (deduped merges) --------
    merged = qm_combined_df.copy()  # contains 'unit_ids' (LOCAL) + 'global_unit_ids'

    # 1) Template metrics (GLOBAL-keyed) — avoid bringing unit_id/unit_ids again
    if tm_all_df is not None and not tm_all_df.empty:
        tm_cols_add = [c for c in tm_all_df.columns if c not in {'unit_id', 'unit_ids'}]
        merged = merged.merge(tm_all_df[tm_cols_add], on='global_unit_ids', how='left')

    # 2) Unit locations — only add if x_um/y_um not already present from the per-segment QC step
    if uloc_all_df is not None and not uloc_all_df.empty and not ({'x_um','y_um'} <= set(merged.columns)):
        if {'x_um','y_um'}.issubset(uloc_all_df.columns):
            loc_use = uloc_all_df[['global_unit_ids','x_um','y_um']]
        elif {'x','y'}.issubset(uloc_all_df.columns):
            loc_use = uloc_all_df[['global_unit_ids','x','y']].rename(columns={'x':'x_um','y':'y_um'})
        else:
            loc_use = None
        if loc_use is not None:
            merged = merged.merge(loc_use, on='global_unit_ids', how='left')

    # 3) Best-channel metadata — drop conflicting names first to prevent _x/_y
    if best_meta_df is not None and not best_meta_df.empty:
        add_cols = ['global_unit_ids', 'peak_channel', 'segment_name', 'n_samples', 'trough_to_peak_ms']
        add_cols = [c for c in add_cols if c in best_meta_df.columns]
        # drop any existing versions (except the key) so the merge won't create _x/_y
        merged = merged.drop(columns=[c for c in add_cols if c != 'global_unit_ids' and c in merged.columns],
                             errors='ignore')
        merged = merged.merge(best_meta_df[add_cols], on='global_unit_ids', how='left')

    # 4) Sanity: warn if any suffixed dup columns slipped in
    suffixed = [c for c in merged.columns if c.endswith('_x') or c.endswith('_y')]
    if suffixed:
        print(f"⚠️ Duplicated columns with suffixes detected (first 10): {suffixed[:10]}")

    # 5) Ensure global_unit_ids is the LAST column
    cols = merged.columns.tolist()
    if 'global_unit_ids' in cols:
        cols.remove('global_unit_ids')
        cols.append('global_unit_ids')
        merged = merged[cols]

    merged.to_csv(os.path.join(AIND_folder, 'cluster_info.tsv'), sep='\t', index=False)
    print("🎯 Wrote: spike_times.npy, spike_clusters.npy, cluster_group.tsv, cluster_info.tsv (+ template_metrics.tsv, best_channel_templates.npy)")
    print(f"   [best-channel] unique peak_channel values: {merged['peak_channel'].unique()}")

    # -------- analysis_meta.json (provenance; complements ap.meta) --------
    ap_meta_candidates = glob.glob(os.path.join(raw_rec, "*ap.meta"))
    ap_meta_path = ap_meta_candidates[0] if ap_meta_candidates else None

    analysis_meta = {
        "notebook": "extract_aind_output.ipynb",
        "export_time_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "paths": {
            "raw_folder": raw_rec,
            "export_folder": AIND_folder,
            "ap_meta_path": ap_meta_path,
            "preprocessed": preProcessed,
            "postprocessed": postProcessed,
            "spikesorted": spikes,
            "curated": curated,
        },
        "recording_meta": _parse_ap_meta(ap_meta_path),
        "waveform_params": {
            "computed_extensions_used": ["templates", "template_metrics"],   # used per segment
            "fallback_compute": False
        },
        "aggregation": {
            "global_unit_count": int(merged["global_unit_ids"].nunique()) if not merged.empty else 0,
            "segments_count": int(len(experiment_files)),
        },
        "environment": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "spikeinterface_version": getattr(si, "__version__", "unknown"),
        }
    }

    with open(os.path.join(AIND_folder, "analysis_meta.json"), "w", encoding="utf-8") as f:
        json.dump(analysis_meta, f, indent=2)
    print(f"🧾 Saved analysis_meta.json in {AIND_folder}")

    # -------- Diagnostics: confirm template_metrics merged into cluster_info --------
    check_template_metrics_merge(AIND_folder)



# ============================== Move output files to the parent directory ==============================
import shutil

parent = os.path.dirname(input_base_dir)
for d in os.listdir(input_base_dir):
    d_path = os.path.join(input_base_dir, d)
    if os.path.isdir(d_path):
        dname = d.rstrip('/')
        dest_path = os.path.join(parent, dname)
        if not os.path.exists(dest_path):
            shutil.move(d_path, dest_path)
            print(f"moved {dname} to {dest_path}")
        else:
            print(f"Skipping {dname}: {dest_path} already exists")