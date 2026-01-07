import argparse
import glob
import json
import os
import platform
import shutil
import time

import numpy as np
import pandas as pd
import spikeinterface as si


def find_experiment_files(preproc_path: str):
    files = glob.glob(os.path.join(preproc_path, "block0_imec*.ap_recording1*.json"))
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
        try:
            out_sum["sampling_rate_hz"] = float(out_raw["imSampRate"])
        except Exception:
            pass
    if "nSavedChans" in out_raw:
        try:
            out_sum["n_saved_channels"] = int(out_raw["nSavedChans"])
        except Exception:
            pass
    if "fileTimeSecs" in out_raw:
        try:
            out_sum["file_time_secs"] = float(out_raw["fileTimeSecs"])
        except Exception:
            pass
    if "fileName" in out_raw:
        out_sum["ap_bin_path"] = out_raw["fileName"]
    return {"raw_meta": out_raw, "summary": out_sum}


def _fetch_templates_array(sa):
    """
    Try several access paths across SpikeInterface versions.
    Return a numpy array with shape (n_units, n_samples, n_channels) if possible,
    or (n_units, n_channels, n_samples). Also return string axis_mode: 'ucs' or 'usc'.
    """
    try:
        sa.load_extension("templates")
    except Exception:
        sa.compute(["templates"], n_jobs=4, progress_bar=False)
    ext = sa.get_extension("templates")
    arr = None
    try:
        data = ext.get_data()
        if isinstance(data, dict) and isinstance(data.get("templates", None), np.ndarray):
            arr = data["templates"]
    except Exception:
        pass
    if arr is None and hasattr(ext, "get_all_templates"):
        try:
            arr = ext.get_all_templates()
        except Exception:
            pass
    if arr is None and hasattr(ext, "get_templates"):
        try:
            arr = ext.get_templates()
        except Exception:
            pass
    if not (isinstance(arr, np.ndarray) and arr.ndim == 3):
        return None, None

    a1, a2 = arr.shape[1], arr.shape[2]
    if a1 <= 512 and a2 > 32:
        return arr, "usc"  # (units, channels, samples)
    return arr, "ucs"  # (units, samples, channels)


def _ptp_per_channel(template_unit, axis_mode):
    if axis_mode == "ucs":
        return template_unit.max(axis=0) - template_unit.min(axis=0)
    return template_unit.max(axis=1) - template_unit.min(axis=1)


def _extract_best_channel_waveform(templates_arr, unit_index, best_ch, axis_mode):
    if axis_mode == "ucs":
        return templates_arr[unit_index, :, best_ch]
    return templates_arr[unit_index, best_ch, :]


def _trough_to_peak_ms(wf_1d, fs_hz):
    if wf_1d is None or len(wf_1d) < 3:
        return np.nan
    i_trough = int(np.argmin(wf_1d))
    if i_trough < len(wf_1d) - 1:
        i_peak = i_trough + int(np.argmax(wf_1d[i_trough:]))
    else:
        i_peak = int(np.argmax(wf_1d))
    dt_samples = max(i_peak - i_trough, 0)
    return (dt_samples / float(fs_hz)) * 1000.0


def _safe_rmtree(path: str):
    if os.path.islink(path) or os.path.isfile(path):
        os.unlink(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)


def _copytree_follow_symlinks(src: str, dst: str, force: bool):
    """
    Copy directory tree following symlinks (copying actual content, not symlinks).
    If destination exists, always remove it first to ensure a complete, fresh copy.
    """
    if os.path.exists(dst):
        _safe_rmtree(dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    # Follow symlinks by copying the referenced content (symlinks=False)
    shutil.copytree(src, dst, symlinks=False)


def regenerate_aind_for_session(base_folder: str, session_name: str, force: bool):
    preprocessed = os.path.join(base_folder, "preprocessed")
    postprocessed = os.path.join(base_folder, "postprocessed")
    spikes = os.path.join(base_folder, "spikesorted")
    curated = os.path.join(base_folder, "curated")

    experiment_files = find_experiment_files(preprocessed)
    if not experiment_files:
        print(f"  [skip] No experiment JSONs found under: {preprocessed}")
        return None

    aind_folder = os.path.join(base_folder, f"AIND_{session_name}")
    if force and os.path.exists(aind_folder):
        _safe_rmtree(aind_folder)
    os.makedirs(aind_folder, exist_ok=True)

    total_units = 0
    all_spike_times = []
    all_spike_clusters = []
    unit_labels_combined = []
    qm_combined_with_global_ids = []
    tm_all_segments = []
    uloc_all_segments = []

    best_templates_all = []
    best_templates_meta = []

    global_unit_counter = 1

    for i, expf in enumerate(experiment_files, start=1):
        print(f"   • [{i}/{len(experiment_files)}] {os.path.basename(expf)}")
        exp_base = os.path.splitext(os.path.basename(expf))[0]
        zarr_path = os.path.join(postprocessed, exp_base + ".zarr")
        if not os.path.exists(zarr_path):
            print(f"     – Missing zarr: {zarr_path}")
            continue

        sorting_analyzer = si.load_sorting_analyzer(zarr_path, load_extensions=False)
        sorting_curated = si.load(os.path.join(curated, exp_base))
        spike_extractor = si.load(os.path.join(spikes, exp_base))

        unit_ids = sorting_curated.get_unit_ids()
        fs_hz = sorting_curated.get_sampling_frequency()

        def _get_prop(name):
            try:
                return sorting_curated.get_property(name)
            except Exception:
                return None

        decoder_label = _get_prop("decoder_label")
        default_qc = _get_prop("default_qc")
        decoder_prob = _get_prop("decoder_probability")

        print(f"     → {len(unit_ids)} units")

        # quality metrics (many columns, may or may not include default_qc)
        sorting_analyzer.load_extension("quality_metrics")
        qm = sorting_analyzer.get_extension("quality_metrics").get_data()
        qm_df = pd.DataFrame(qm)
        if "unit_ids" not in qm_df.columns:
            qm_df["unit_ids"] = unit_ids

        # unit locations in QC table (optional)
        try:
            sorting_analyzer.load_extension("unit_locations")
            unit_locs = sorting_analyzer.get_extension("unit_locations").get_data()
            try:
                qm_df["x_um"] = unit_locs[:, 0]
                qm_df["y_um"] = unit_locs[:, 1]
            except Exception:
                uloc_df = pd.DataFrame(unit_locs)
                if "x_um" in uloc_df.columns and "y_um" in uloc_df.columns:
                    qm_df["x_um"] = uloc_df["x_um"].to_numpy()
                    qm_df["y_um"] = uloc_df["y_um"].to_numpy()
        except Exception:
            pass

        # unit locations (optional, for merging)
        try:
            sorting_analyzer.load_extension("unit_locations")
            uloc = sorting_analyzer.get_extension("unit_locations").get_data()
            uloc_df = pd.DataFrame(uloc)
            if "unit_id" not in uloc_df.columns:
                uloc_df = uloc_df.reset_index().rename(columns={"index": "unit_id"})
            uloc_df["unit_id"] = uloc_df["unit_id"].astype(int)
        except Exception:
            uloc_df = None

        # LOCAL → GLOBAL mapping for this segment
        map_df = pd.DataFrame(
            {
                "unit_ids": unit_ids,
                "global_unit_ids": np.arange(global_unit_counter, global_unit_counter + len(unit_ids)),
            }
        )

        qm_df = qm_df.merge(map_df, on="unit_ids", how="left")
        qm_combined_with_global_ids.append(qm_df)

        # cluster_group.tsv (labels + default_qc)
        labels_df = pd.DataFrame({"global_unit_ids": map_df["global_unit_ids"]})
        if decoder_label is not None:
            labels_df["labels"] = decoder_label
        if default_qc is not None:
            labels_df["default_qc"] = np.asarray(default_qc)
        if decoder_prob is not None:
            labels_df["decoder_prob"] = np.asarray(decoder_prob)
        unit_labels_combined.append(labels_df)

        # spikes (sample indices) per local unit, tagged with GLOBAL id
        for local_id, global_id in zip(map_df["unit_ids"], map_df["global_unit_ids"]):
            try:
                stimes = spike_extractor.get_unit_spike_train(local_id)
                all_spike_times.extend(stimes)
                all_spike_clusters.extend([int(global_id)] * len(stimes))
            except Exception:
                pass

        # template metrics (optional)
        try:
            try:
                sorting_analyzer.load_extension("template_metrics")
            except Exception:
                sorting_analyzer.compute(["templates", "template_metrics"], n_jobs=4, progress_bar=False)
            tm_seg = sorting_analyzer.get_extension("template_metrics").get_data()
            tm_seg_df = pd.DataFrame(tm_seg)
            if "unit_id" not in tm_seg_df.columns:
                tm_seg_df = tm_seg_df.reset_index().rename(columns={"index": "unit_id"})
            tm_seg_df["unit_id"] = tm_seg_df["unit_id"].astype(int)
            tm_seg_df = tm_seg_df.merge(map_df.rename(columns={"unit_ids": "unit_id"}), on="unit_id", how="left")
            tm_seg_df.drop(columns=["unit_id"], inplace=True)
            tm_all_segments.append(tm_seg_df)
        except Exception:
            tm_seg_df = None

        # unit locations mapped to GLOBAL (optional)
        if uloc_df is not None:
            uloc_df = uloc_df.merge(map_df.rename(columns={"unit_ids": "unit_id"}), on="unit_id", how="left")
            keep_cols = ["global_unit_ids"]
            for col in ["x", "y", "x_um", "y_um", "z", "z_um"]:
                if col in uloc_df.columns:
                    keep_cols.append(col)
            uloc_all_segments.append(uloc_df[keep_cols])

        # best-channel templates (optional)
        templates_arr, axis_mode = _fetch_templates_array(sorting_analyzer)
        if templates_arr is not None:
            peak_col = None
            if tm_seg_df is not None:
                for cand in ("peak_channel", "best_channel", "max_ptp_channel"):
                    if cand in tm_seg_df.columns:
                        peak_col = cand
                        break

            # local -> global map
            map_local = pd.DataFrame({"unit_id": unit_ids}).merge(
                map_df.rename(columns={"unit_ids": "unit_id"}), on="unit_id", how="left"
            )

            if peak_col is None:
                for u_idx, u_local in enumerate(unit_ids):
                    unit_template = templates_arr[u_idx, :, :]  # either (S,C) or (C,S), axis_mode handles
                    ptp_vec = _ptp_per_channel(unit_template, axis_mode)
                    best_ch = int(np.argmax(ptp_vec))
                    map_local.loc[map_local["unit_id"] == u_local, "peak_channel"] = best_ch
            else:
                # if available, try to recover from extension (best effort)
                try:
                    sorting_analyzer.load_extension("template_metrics")
                    tm_full = pd.DataFrame(sorting_analyzer.get_extension("template_metrics").get_data())
                    if "unit_id" not in tm_full.columns:
                        tm_full = tm_full.reset_index().rename(columns={"index": "unit_id"})
                    tm_full["unit_id"] = tm_full["unit_id"].astype(int)
                    map_local = map_local.merge(tm_full[["unit_id", peak_col]], on="unit_id", how="left")
                    map_local = map_local.rename(columns={peak_col: "peak_channel"})
                except Exception:
                    for u_idx, u_local in enumerate(unit_ids):
                        unit_template = templates_arr[u_idx, :, :]
                        ptp_vec = _ptp_per_channel(unit_template, axis_mode)
                        best_ch = int(np.argmax(ptp_vec))
                        map_local.loc[map_local["unit_id"] == u_local, "peak_channel"] = best_ch

            local_to_global = dict(zip(map_local["unit_id"].tolist(), map_local["global_unit_ids"].tolist()))
            local_to_peak = dict(zip(map_local["unit_id"].tolist(), map_local["peak_channel"].astype(int).tolist()))

            n_samples_seg = int(templates_arr.shape[1]) if axis_mode == "ucs" else int(templates_arr.shape[2])

            for u_idx, u_local in enumerate(unit_ids):
                g_uid = int(local_to_global.get(u_local, -1))
                if g_uid < 0:
                    continue
                best_ch = int(local_to_peak.get(u_local, 0))
                wf_1d = _extract_best_channel_waveform(templates_arr, u_idx, best_ch, axis_mode).astype(np.float32)
                t2p_ms = _trough_to_peak_ms(wf_1d, fs_hz)
                best_templates_all.append(wf_1d)
                best_templates_meta.append(
                    {
                        "global_unit_ids": g_uid,
                        "peak_channel": best_ch,
                        "segment_name": exp_base,
                        "n_samples": n_samples_seg,
                        "trough_to_peak_ms": float(t2p_ms),
                    }
                )

        global_unit_counter += len(unit_ids)
        total_units += len(unit_ids)

    if total_units == 0:
        print("  [skip] No units found in any segment.")
        return None

    # Concatenate per-session tables
    unit_labels_df = pd.concat(unit_labels_combined, ignore_index=True) if unit_labels_combined else pd.DataFrame()
    qm_combined_df = pd.concat(qm_combined_with_global_ids, ignore_index=True) if qm_combined_with_global_ids else pd.DataFrame()

    # Save core artifacts
    spike_times = np.array(all_spike_times)
    spike_clusters = np.array(all_spike_clusters)
    sort_idx = np.argsort(spike_times)
    spike_times = spike_times[sort_idx]
    spike_clusters = spike_clusters[sort_idx]

    np.save(os.path.join(aind_folder, "spike_times.npy"), spike_times)
    np.save(os.path.join(aind_folder, "spike_clusters.npy"), spike_clusters)
    unit_labels_df.to_csv(os.path.join(aind_folder, "cluster_group.tsv"), sep="\t", index=False)

    # Stitch template metrics across segments
    tm_all_df = None
    if len(tm_all_segments) > 0:
        tm_all_df = pd.concat(tm_all_segments, ignore_index=True).drop_duplicates(subset=["global_unit_ids"])
        tm_all_df.to_csv(os.path.join(aind_folder, "template_metrics.tsv"), sep="\t", index=False)

    # Stitch unit locations
    uloc_all_df = None
    if len(uloc_all_segments) > 0:
        uloc_all_df = pd.concat(uloc_all_segments, ignore_index=True).drop_duplicates(subset=["global_unit_ids"])

    # Best-channel templates
    best_meta_df = None
    if len(best_templates_all) > 0:
        lengths = pd.Series([len(x) for x in best_templates_all])
        mode_len = int(lengths.mode().iloc[0])

        def _fixlen(x, L):
            if len(x) == L:
                return x
            if len(x) > L:
                return x[:L]
            out = np.zeros(L, dtype=np.float32)
            out[: len(x)] = x
            return out

        best_arr = np.vstack([_fixlen(x, mode_len)[None, :] for x in best_templates_all]).astype(np.float32)
        np.save(os.path.join(aind_folder, "best_channel_templates.npy"), best_arr)

        best_meta_df = pd.DataFrame(best_templates_meta)
        best_meta_df = best_meta_df.sort_values("segment_name").drop_duplicates(subset=["global_unit_ids"], keep="last")

    # cluster_info.tsv = QC + template metrics + best-channel info
    merged = qm_combined_df.copy()
    if tm_all_df is not None and not tm_all_df.empty:
        tm_cols_add = [c for c in tm_all_df.columns if c not in {"unit_id", "unit_ids"}]
        merged = merged.merge(tm_all_df[tm_cols_add], on="global_unit_ids", how="left")

    if uloc_all_df is not None and not uloc_all_df.empty and not ({"x_um", "y_um"} <= set(merged.columns)):
        if {"x_um", "y_um"}.issubset(uloc_all_df.columns):
            loc_use = uloc_all_df[["global_unit_ids", "x_um", "y_um"]]
        elif {"x", "y"}.issubset(uloc_all_df.columns):
            loc_use = uloc_all_df[["global_unit_ids", "x", "y"]].rename(columns={"x": "x_um", "y": "y_um"})
        else:
            loc_use = None
        if loc_use is not None:
            merged = merged.merge(loc_use, on="global_unit_ids", how="left")

    if best_meta_df is not None and not best_meta_df.empty:
        add_cols = ["global_unit_ids", "peak_channel", "segment_name", "n_samples", "trough_to_peak_ms"]
        add_cols = [c for c in add_cols if c in best_meta_df.columns]
        merged = merged.drop(
            columns=[c for c in add_cols if c != "global_unit_ids" and c in merged.columns], errors="ignore"
        )
        merged = merged.merge(best_meta_df[add_cols], on="global_unit_ids", how="left")

    # Ensure global_unit_ids last
    cols = merged.columns.tolist()
    if "global_unit_ids" in cols:
        cols.remove("global_unit_ids")
        cols.append("global_unit_ids")
        merged = merged[cols]

    merged.to_csv(os.path.join(aind_folder, "cluster_info.tsv"), sep="\t", index=False)

    # analysis_meta.json
    ap_meta_candidates = glob.glob(os.path.join(base_folder, "*.ap.meta"))
    ap_meta_path = ap_meta_candidates[0] if ap_meta_candidates else None
    analysis_meta = {
        "script": "regenerate_aind_from_outputs.py",
        "export_time_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "paths": {
            "session_output_folder": base_folder,
            "export_folder": aind_folder,
            "ap_meta_path": ap_meta_path,
            "preprocessed": preprocessed,
            "postprocessed": postprocessed,
            "spikesorted": spikes,
            "curated": curated,
        },
        "recording_meta": _parse_ap_meta(ap_meta_path),
        "aggregation": {"global_unit_count": int(merged["global_unit_ids"].nunique()) if not merged.empty else 0},
        "environment": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "spikeinterface_version": getattr(si, "__version__", "unknown"),
        },
    }
    with open(os.path.join(aind_folder, "analysis_meta.json"), "w", encoding="utf-8") as f:
        json.dump(analysis_meta, f, indent=2)

    return aind_folder


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate AIND_* folders from existing per-session *_output folders."
    )
    parser.add_argument(
        "--output-base-dir",
        default=os.environ.get("AIND_OUTPUT_BASE_DIR", "/n/netscratch/bsabatini_lab/Lab/shunnnli/spikesorting/aind_output_scratch"),
        help="Directory containing per-session *_output folders.",
    )
    parser.add_argument(
        "--download-base-dir",
        default="/n/netscratch/bsabatini_lab/Lab/shunnnli/spikesorting/aind_output_fordownload",
        help="Destination directory to copy AIND_* folders for download.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="If set, delete and recreate existing AIND_* folders (both in output and download dirs).",
    )
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="If set, do not copy AIND_* folders to the download directory.",
    )
    args = parser.parse_args()

    output_base_dir = args.output_base_dir.rstrip("/")
    download_base_dir = args.download_base_dir.rstrip("/")

    print("SpikeInterface version:", getattr(si, "__version__", "unknown"))
    print("Output base dir   :", output_base_dir)
    print("Download base dir :", download_base_dir)
    print("Force regenerate  :", bool(args.force))
    print("Copy to download  :", not args.no_copy)

    session_dirs = sorted(glob.glob(os.path.join(output_base_dir, "*_output")))
    session_dirs = [d for d in session_dirs if os.path.isdir(d)]
    if not session_dirs:
        raise SystemExit(f"No *_output folders found under: {output_base_dir}")

    os.makedirs(download_base_dir, exist_ok=True)

    ok = 0
    skipped = 0
    failed = 0

    for session_dir in session_dirs:
        session_dir_name = os.path.basename(session_dir)
        session_name = session_dir_name[:-7] if session_dir_name.endswith("_output") else session_dir_name
        print(f"\n=== Session folder: {session_dir_name}  (session_name={session_name}) ===")

        try:
            aind_folder = regenerate_aind_for_session(session_dir, session_name, force=args.force)
            if aind_folder is None:
                skipped += 1
                continue
            ok += 1

            if not args.no_copy:
                # Put AIND folder inside {session_name}_output in download directory
                session_download_dir = os.path.join(download_base_dir, f"{session_name}_output")
                dst = os.path.join(session_download_dir, os.path.basename(aind_folder))
                print(f"  Copying AIND folder -> {dst}")
                # Always force a fresh copy to download directory (removes existing if present)
                _copytree_follow_symlinks(aind_folder, dst, force=True)
        except Exception as e:
            failed += 1
            print(f"  [error] Failed session {session_dir_name}: {e}")

    print("\nDone.")
    print(f"  regenerated_ok={ok}, skipped={skipped}, failed={failed}")
    if failed > 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()


