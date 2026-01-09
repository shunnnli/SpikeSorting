import warnings

warnings.filterwarnings("ignore")

# GENERAL IMPORTS
import os

# this is needed to limit the number of scipy threads
# and let spikeinterface handle parallelization
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import numpy as np
from pathlib import Path
import json
import pickle
import time
from datetime import datetime, timedelta
from typing import List, Optional

# SPIKEINTERFACE
import spikeinterface as si
import spikeinterface.preprocessing as spre

from spikeinterface.core.core_tools import check_json

# AIND
from aind_data_schema.core.processing import DataProcess

URL = "https://github.com/AllenNeuralDynamics/aind-ephys-preprocessing"
VERSION = "1.0-custom"  # Mark as custom version


data_folder = Path("../data/")
scratch_folder = Path("../scratch/")
results_folder = Path("../results/")

motion_presets = spre.get_motion_presets()

# define argument parser
parser = argparse.ArgumentParser(description="Preprocess AIND Neurpixels data")

# positional arguments
denoising_group = parser.add_mutually_exclusive_group()
denoising_help = "Which denoising strategy to use. Can be 'cmr' or 'destripe'"
denoising_group.add_argument("--denoising", choices=["cmr", "destripe"], help=denoising_help)
denoising_group.add_argument("static_denoising", nargs="?", default="cmr", help=denoising_help)

filter_group = parser.add_mutually_exclusive_group()
filter_help = "Which filter to use. Can be 'highpass' or 'bandpass'"
filter_group.add_argument("--filter-type", choices=["highpass", "bandpass"], help=filter_help)
filter_group.add_argument("static_filter_type", nargs="?", default="highpass", help=filter_help)

remove_out_channels_group = parser.add_mutually_exclusive_group()
remove_out_channels_help = "Whether to remove out channels"
remove_out_channels_group.add_argument("--no-remove-out-channels", action="store_true", help=remove_out_channels_help)
remove_out_channels_group.add_argument(
    "static_remove_out_channels", nargs="?", default="true", help=remove_out_channels_help
)

remove_bad_channels_group = parser.add_mutually_exclusive_group()
remove_bad_channels_help = "Whether to remove bad channels"
remove_bad_channels_group.add_argument("--no-remove-bad-channels", action="store_true", help=remove_bad_channels_help)
remove_bad_channels_group.add_argument(
    "static_remove_bad_channels", nargs="?", default="true", help=remove_bad_channels_help
)

max_bad_channel_fraction_group = parser.add_mutually_exclusive_group()
max_bad_channel_fraction_help = (
    "Maximum fraction of bad channels to remove. If more than this fraction, processing is skipped"
)
max_bad_channel_fraction_group.add_argument(
    "--max-bad-channel-fraction", default=0.5, help=max_bad_channel_fraction_help
)
max_bad_channel_fraction_group.add_argument(
    "static_max_bad_channel_fraction", nargs="?", default=None, help=max_bad_channel_fraction_help
)

motion_correction_group = parser.add_mutually_exclusive_group()
motion_correction_help = "How to deal with motion correction. Can be 'skip', 'compute', or 'apply'"
motion_correction_group.add_argument("--motion", choices=["skip", "compute", "apply"], help=motion_correction_help)
motion_correction_group.add_argument("static_motion", nargs="?", default="compute", help=motion_correction_help)

motion_preset_group = parser.add_mutually_exclusive_group()
motion_preset_help = (
    f"What motion preset to use. Supported presets are: {', '.join(motion_presets)}."
)
motion_preset_group.add_argument(
    "--motion-preset",
    choices=motion_presets,
    help=motion_preset_help,
)
motion_preset_group.add_argument("static_motion_preset", nargs="?", default=None, help=motion_preset_help)

t_start_group = parser.add_mutually_exclusive_group()
t_start_help = (
    "Start time of the recording in seconds (assumes recording starts at 0). "
    "This parameter is ignored in case of multi-segment or multi-block recordings."
    "Default is None (start of recording)"
)
t_start_group.add_argument("static_t_start", nargs="?", default=None, help=t_start_help)
t_start_group.add_argument("--t-start", default=None, help=t_start_help)

t_stop_group = parser.add_mutually_exclusive_group()
t_stop_help = (
    "Stop time of the recording in seconds (assumes recording starts at 0). "
    "This parameter is ignored in case of multi-segment or multi-block recordings."
    "Default is None (end of recording)"
)
t_stop_group.add_argument("static_t_stop", nargs="?", default=None, help=t_stop_help)
t_stop_group.add_argument("--t-stop", default=None, help=t_stop_help)

n_jobs_group = parser.add_mutually_exclusive_group()
n_jobs_help = (
    "Number of jobs to use for parallel processing. Default is -1 (all available cores). "
    "It can also be a float between 0 and 1 to use a fraction of available cores"
)
n_jobs_group.add_argument("static_n_jobs", nargs="?", default=None, help=n_jobs_help)
n_jobs_group.add_argument("--n-jobs", default="-1", help=n_jobs_help)

params_group = parser.add_mutually_exclusive_group()
params_file_help = "Optional json file with parameters"
params_group.add_argument("static_params_file", nargs="?", default=None, help=params_file_help)
params_group.add_argument("--params-file", default=None, help=params_file_help)
params_group.add_argument("--params-str", default=None, help="Optional json string with parameters")

# ============================================================================
# NEW: Manual bad channel arguments
# ============================================================================
bad_channel_ids_help = (
    "Comma-separated list of bad channel IDs to add to auto-detected bad channels. "
    "Example: --bad-channel-ids 'AP0,AP5,AP127' or --bad-channel-ids '0,5,127'"
)
parser.add_argument("--bad-channel-ids", default=None, help=bad_channel_ids_help)

bad_channels_config_help = (
    "Path to a bad_channels.conf file with session-specific bad channel mappings. "
    "Format: session_name = channel1, channel2, channel3"
)
parser.add_argument("--bad-channels-config", default=None, help=bad_channels_config_help)


# ============================================================================
# Helper functions for manual bad channels
# ============================================================================
def parse_bad_channel_ids(bad_channel_str: str) -> List[str]:
    """
    Parse comma-separated bad channel IDs from command line.
    
    Args:
        bad_channel_str: Comma-separated string like "AP0,AP5,AP127" or "0,5,127"
    
    Returns:
        List of channel ID strings
    """
    if not bad_channel_str or bad_channel_str.strip() == "":
        return []
    return [ch.strip() for ch in bad_channel_str.split(",") if ch.strip()]


def load_bad_channels_from_config(config_path: Path, session_name: str) -> List[str]:
    """
    Load bad channel IDs from config file for a specific session.
    
    Config file format:
        # Comments start with #
        session_name = channel1, channel2, channel3
        partial_match = channel4, channel5
    
    Args:
        config_path: Path to bad_channels.conf file
        session_name: Name of the recording session
    
    Returns:
        List of bad channel IDs (empty list if no match found)
    """
    if not config_path.exists():
        print(f"\t[custom] No bad channels config found at {config_path}")
        return []
    
    bad_channels = []
    
    with open(config_path, "r") as f:
        for line in f:
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue
            
            # Parse key=value pairs
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                
                # Check for exact or partial match (case-insensitive)
                if key.lower() in session_name.lower() or session_name.lower() in key.lower():
                    channels = [ch.strip() for ch in value.split(",") if ch.strip()]
                    bad_channels.extend(channels)
                    print(f"\t[custom] Found bad channels for '{key}': {channels}")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_channels = []
    for ch in bad_channels:
        if ch not in seen:
            seen.add(ch)
            unique_channels.append(ch)
    
    return unique_channels


def get_channel_ids_from_names(recording, channel_names: List[str]) -> np.ndarray:
    """
    Convert channel names to channel IDs that SpikeInterface understands.
    
    Supports:
    - Direct channel ID matching (e.g., "AP0" if that's the actual ID)
    - Numeric index matching (e.g., "5" -> channel at index 5)
    - AP-prefix numeric matching (e.g., "AP5" -> channel at index 5)
    
    Args:
        recording: SpikeInterface recording object
        channel_names: List of channel names (e.g., ['AP0', 'AP5', '127'])
    
    Returns:
        Array of channel IDs
    """
    all_channel_ids = recording.get_channel_ids()
    
    matched_ids = []
    for name in channel_names:
        # Try direct match first
        if name in all_channel_ids:
            matched_ids.append(name)
            continue
        
        # Try numeric matching
        try:
            # Handle "AP5" format -> extract 5
            if isinstance(name, str) and name.upper().startswith("AP"):
                idx = int(name[2:])
            else:
                idx = int(name)
            
            if 0 <= idx < len(all_channel_ids):
                matched_ids.append(all_channel_ids[idx])
            else:
                print(f"\t[custom] Warning: Channel index {idx} out of range (0-{len(all_channel_ids)-1})")
        except (ValueError, IndexError):
            print(f"\t[custom] Warning: Could not find channel '{name}'")
    
    return np.array(matched_ids)


def dump_to_json_or_pickle(recording, results_folder, base_name, relative_to):
    if recording.check_serializability("json"):
        recording.dump_to_json(results_folder / f"{base_name}.json", relative_to=relative_to)
    else:
        recording.dump_to_pickle(results_folder / f"{base_name}.pkl", relative_to=relative_to)


if __name__ == "__main__":
    args = parser.parse_args()

    DENOISING_STRATEGY = args.denoising or args.static_denoising
    FILTER_TYPE = args.filter_type or args.static_filter_type
    REMOVE_OUT_CHANNELS = False if args.no_remove_out_channels else args.static_remove_out_channels == "true"
    REMOVE_BAD_CHANNELS = False if args.no_remove_bad_channels else args.static_remove_bad_channels == "true"
    MAX_BAD_CHANNEL_FRACTION = float(args.static_max_bad_channel_fraction or args.max_bad_channel_fraction)
    motion_arg = args.motion or args.static_motion
    MOTION_PRESET = args.static_motion_preset or args.motion_preset
    COMPUTE_MOTION = True if motion_arg != "skip" else False
    APPLY_MOTION = True if motion_arg == "apply" else False
    T_START = args.static_t_start or args.t_start
    if isinstance(T_START, str) and T_START == "":
        T_START = None
    T_STOP = args.static_t_stop or args.t_stop
    if isinstance(T_STOP, str) and T_STOP == "":
        T_STOP = None

    N_JOBS = args.static_n_jobs or args.n_jobs
    N_JOBS = int(N_JOBS) if not N_JOBS.startswith("0.") else float(N_JOBS)
    PARAMS_FILE = args.static_params_file or args.params_file
    PARAMS_STR = args.params_str

    # NEW: Parse manual bad channel arguments
    MANUAL_BAD_CHANNEL_IDS = parse_bad_channel_ids(args.bad_channel_ids) if args.bad_channel_ids else []
    BAD_CHANNELS_CONFIG = Path(args.bad_channels_config) if args.bad_channels_config else None

    # Use CO_CPUS env variable if available
    N_JOBS_CO = os.getenv("CO_CPUS")
    N_JOBS = int(N_JOBS_CO) if N_JOBS_CO is not None else N_JOBS

    print(f"Running preprocessing with the following parameters:")
    print(f"\tDENOISING_STRATEGY: {DENOISING_STRATEGY}")
    print(f"\tFILTER TYPE: {FILTER_TYPE}")
    print(f"\tREMOVE_OUT_CHANNELS: {REMOVE_OUT_CHANNELS}")
    print(f"\tREMOVE_BAD_CHANNELS: {REMOVE_BAD_CHANNELS}")
    print(f"\tMAX BAD CHANNEL FRACTION: {MAX_BAD_CHANNEL_FRACTION}")
    print(f"\tCOMPUTE_MOTION: {COMPUTE_MOTION}")
    print(f"\tAPPLY_MOTION: {APPLY_MOTION}")
    print(f"\tMOTION PRESET: {MOTION_PRESET}")
    print(f"\tT_START: {T_START}")
    print(f"\tT_STOP: {T_STOP}")
    print(f"\tN_JOBS: {N_JOBS}")
    # NEW: Print manual bad channel info
    if MANUAL_BAD_CHANNEL_IDS:
        print(f"\tMANUAL_BAD_CHANNEL_IDS: {MANUAL_BAD_CHANNEL_IDS}")
    if BAD_CHANNELS_CONFIG:
        print(f"\tBAD_CHANNELS_CONFIG: {BAD_CHANNELS_CONFIG}")

    if PARAMS_FILE is not None:
        print(f"\nUsing custom parameter file: {PARAMS_FILE}")
        with open(PARAMS_FILE, "r") as f:
            processing_params = json.load(f)
    elif PARAMS_STR is not None:
        processing_params = json.loads(PARAMS_STR)
    else:
        with open("params.json", "r") as f:
            processing_params = json.load(f)

    data_process_prefix = "data_process_preprocessing"

    job_kwargs = processing_params["job_kwargs"]
    job_kwargs["n_jobs"] = N_JOBS
    si.set_global_job_kwargs(**job_kwargs)

    preprocessing_params = processing_params["preprocessing"]
    preprocessing_params["denoising_strategy"] = DENOISING_STRATEGY
    preprocessing_params["remove_out_channels"] = REMOVE_OUT_CHANNELS
    preprocessing_params["remove_bad_channels"] = REMOVE_BAD_CHANNELS
    preprocessing_params["max_bad_channel_fraction"] = MAX_BAD_CHANNEL_FRACTION
    motion_params = processing_params["motion_correction"]
    motion_params["compute"] = COMPUTE_MOTION
    motion_params["apply"] = APPLY_MOTION
    if MOTION_PRESET is not None:
        motion_params["preset"] = MOTION_PRESET

    # load job files
    job_config_files = [p for p in data_folder.iterdir() if (p.suffix == ".json" or p.suffix == ".pickle" or p.suffix == ".pkl") and "job" in p.name]
    print(f"Found {len(job_config_files)} configurations")

    if len(job_config_files) > 0:
        ####### PREPROCESSING #######
        print("\n\nPREPROCESSING")
        t_preprocessing_start_all = time.perf_counter()
        preprocessing_vizualization_data = {}

        for job_config_file in job_config_files:
            datetime_start_preproc = datetime.now()
            t_preprocessing_start = time.perf_counter()
            preprocessing_notes = ""

            if job_config_file.suffix == ".json":
                with open(job_config_file, "r") as f:
                    job_config = json.load(f)
            else:
                with open(job_config_file, "rb") as f:
                    job_config = pickle.load(f)

            session_name = job_config["session_name"]
            recording_name = job_config["recording_name"]
            recording_dict = job_config["recording_dict"]
            skip_times = job_config.get("skip_times", False)
            debug = job_config.get("debug", False)

            try:
                recording = si.load_extractor(recording_dict, base_folder=data_folder)
            except:
                raise RuntimeError(
                    f"Could not find load recording {recording_name} from dict. "
                    f"Make sure mapping is correct!"
                )
            if skip_times:
                print("Resetting recording timestamps")
                recording.reset_times()

            skip_processing = False
            vizualization_file_is_json_serializable = True

            preprocessing_vizualization_data[recording_name] = {}
            preprocessing_output_process_json = results_folder / f"{data_process_prefix}_{recording_name}.json"
            preprocessing_output_folder = results_folder / f"preprocessed_{recording_name}"
            preprocessingviz_output_filename = f"preprocessedviz_{recording_name}"
            preprocessing_output_filename = f"preprocessed_{recording_name}"
            motioncorrected_output_filename = f"motioncorrected_{recording_name}"
            binary_output_filename = f"binary_{recording_name}"

            print(f"Preprocessing recording: {session_name} - {recording_name}")

            # ================================================================
            # NEW: Load manual bad channels for this session
            # ================================================================
            session_manual_bad_channels = list(MANUAL_BAD_CHANNEL_IDS)  # Start with CLI-provided channels
            
            # Add channels from config file if provided
            if BAD_CHANNELS_CONFIG is not None:
                config_bad_channels = load_bad_channels_from_config(BAD_CHANNELS_CONFIG, session_name)
                session_manual_bad_channels.extend(config_bad_channels)
            
            # Remove duplicates
            session_manual_bad_channels = list(dict.fromkeys(session_manual_bad_channels))
            
            if session_manual_bad_channels:
                print(f"\t[custom] Manual bad channels for this session: {session_manual_bad_channels}")
                preprocessing_notes += f"\n- Manual bad channels specified: {session_manual_bad_channels}"
            # ================================================================

            if (T_START is not None or T_STOP is not None):
                if recording.get_num_segments() > 1:
                    print(f"\tRecording has multiple segments. Ignoring T_START and T_STOP")
                else:
                    if T_START is None:
                        T_START = 0
                    if T_STOP is None:
                        T_STOP = recording.get_duration()
                    T_START = float(T_START)
                    T_STOP = float(T_STOP)
                    T_STOP = min(T_STOP, recording.get_duration())
                    print(f"\tOriginal recording duration: {recording.get_duration()} -- Clipping to {T_START}-{T_STOP} s")
                    start_frame = int(T_START * recording.get_sampling_frequency())
                    end_frame = int(T_STOP * recording.get_sampling_frequency() + 1)
                    recording = recording.frame_slice(start_frame=start_frame, end_frame=end_frame)

            print(f"\tDuration: {np.round(recording.get_total_duration(), 2)} s")

            preprocessing_vizualization_data[recording_name]["timeseries"] = dict()
            preprocessing_vizualization_data[recording_name]["timeseries"]["full"] = dict(
                raw=recording.to_dict(relative_to=data_folder, recursive=True)
            )
            if not recording.check_serializability("json"):
                vizualization_file_is_json_serializable = False
            # maybe a recording is from a different source and it doesn't need to be phase shifted
            if "inter_sample_shift" in recording.get_property_keys():
                recording_ps_full = spre.phase_shift(recording, **preprocessing_params["phase_shift"])
                preprocessing_vizualization_data[recording_name]["timeseries"]["full"].update(
                    dict(phase_shift=recording_ps_full.to_dict(relative_to=data_folder, recursive=True))
                )
            else:
                recording_ps_full = recording

            if FILTER_TYPE == "highpass":
                recording_filt_full = spre.highpass_filter(recording_ps_full, **preprocessing_params["highpass_filter"])
                preprocessing_vizualization_data[recording_name]["timeseries"]["full"].update(
                    dict(highpass=recording_filt_full.to_dict(relative_to=data_folder, recursive=True))
                )
                preprocessing_params["filter_type"] = "highpass"
            elif FILTER_TYPE == "bandpass":
                recording_filt_full = spre.bandpass_filter(recording_ps_full, **preprocessing_params["bandpass_filter"])
                preprocessing_vizualization_data[recording_name]["timeseries"]["full"].update(
                    dict(bandpass=recording_filt_full.to_dict(relative_to=data_folder, recursive=True))
                )
                preprocessing_params["filter_type"] = "bandpass"
            else:
                raise ValueError(f"Filter type {FILTER_TYPE} not recognized")

            if recording.get_total_duration() < preprocessing_params["min_preprocessing_duration"] and not debug:
                print(f"\tRecording is too short ({recording.get_total_duration()}s). Skipping further processing")
                preprocessing_notes += (
                    f"\n- Recording is too short ({recording.get_total_duration()}s). Skipping further processing\n"
                )
                channel_labels = None
                skip_processing = True
            else:
                # IBL bad channel detection
                _, channel_labels = spre.detect_bad_channels(
                    recording_filt_full, **preprocessing_params["detect_bad_channels"]
                )
                dead_channel_mask = channel_labels == "dead"
                noise_channel_mask = channel_labels == "noise"
                out_channel_mask = channel_labels == "out"
                print(f"\tBad channel detection:")
                print(
                    f"\t\t- dead channels - {np.sum(dead_channel_mask)}\n\t\t- noise channels - {np.sum(noise_channel_mask)}\n\t\t- out channels - {np.sum(out_channel_mask)}"
                )
                dead_channel_ids = recording_filt_full.channel_ids[dead_channel_mask]
                noise_channel_ids = recording_filt_full.channel_ids[noise_channel_mask]
                out_channel_ids = recording_filt_full.channel_ids[out_channel_mask]

                # ================================================================
                # NEW: Add manual bad channels to the detected ones
                # ================================================================
                manual_bad_channel_ids_resolved = np.array([], dtype=dead_channel_ids.dtype)
                if session_manual_bad_channels:
                    manual_bad_channel_ids_resolved = get_channel_ids_from_names(
                        recording_filt_full, session_manual_bad_channels
                    )
                    if len(manual_bad_channel_ids_resolved) > 0:
                        print(f"\t\t- manual channels - {len(manual_bad_channel_ids_resolved)}")
                # ================================================================

                all_bad_channel_ids = np.concatenate((dead_channel_ids, noise_channel_ids, out_channel_ids))

                skip_processing = False
                max_bad_channel_fraction = preprocessing_params["max_bad_channel_fraction"]
                if len(all_bad_channel_ids) >= int(max_bad_channel_fraction * recording.get_num_channels()):
                    print(f"\tMore than {max_bad_channel_fraction * 100}% bad channels ({len(all_bad_channel_ids)}). ")
                    preprocessing_notes += f"\n- Found {len(all_bad_channel_ids)} bad channels."
                    if preprocessing_params["remove_bad_channels"]:
                        skip_processing = True
                        print("\tSkipping further processing for this recording.")
                        preprocessing_notes += f" Skipping further processing for this recording.\n"
                    else:
                        preprocessing_notes += "\n"

                if not skip_processing:
                    if preprocessing_params["remove_out_channels"]:
                        print(f"\tRemoving {len(out_channel_ids)} out channels")
                        recording_rm_out = recording_filt_full.remove_channels(out_channel_ids)
                        preprocessing_notes += f"\n- Removed {len(out_channel_ids)} outside of the brain."
                    else:
                        recording_rm_out = recording_filt_full

                    recording_processed_cmr = spre.common_reference(
                        recording_rm_out, **preprocessing_params["common_reference"]
                    )

                    # ================================================================
                    # NEW: Combine auto-detected + manual bad channels for interpolation
                    # ================================================================
                    bad_channel_ids = np.concatenate((dead_channel_ids, noise_channel_ids))
                    
                    # Add manual bad channels (excluding out channels and duplicates)
                    if len(manual_bad_channel_ids_resolved) > 0:
                        # Filter out any that are already in bad_channel_ids or out_channel_ids
                        existing_bad = set(bad_channel_ids.tolist()) | set(out_channel_ids.tolist())
                        new_manual = [ch for ch in manual_bad_channel_ids_resolved if ch not in existing_bad]
                        if new_manual:
                            bad_channel_ids = np.concatenate((bad_channel_ids, np.array(new_manual)))
                            print(f"\t[custom] Added {len(new_manual)} manual bad channels for interpolation")
                            preprocessing_notes += f"\n- Added {len(new_manual)} manual bad channels."
                    # ================================================================

                    recording_interp = spre.interpolate_bad_channels(recording_rm_out, bad_channel_ids)
                    
                    # protection against short probes
                    try:
                        recording_hp_spatial = spre.highpass_spatial_filter(
                            recording_interp, **preprocessing_params["highpass_spatial_filter"]
                        )
                    except Exception as e:
                        print(f"\t[custom] Highpass spatial filter failed: {e}")
                        recording_hp_spatial = None
                    
                    preprocessing_vizualization_data[recording_name]["timeseries"]["proc"] = dict(
                        highpass=recording_rm_out.to_dict(relative_to=data_folder, recursive=True),
                        cmr=recording_processed_cmr.to_dict(relative_to=data_folder, recursive=True),
                    )
                    if recording_hp_spatial is not None:
                        preprocessing_vizualization_data[recording_name]["timeseries"]["proc"].update(
                            dict(highpass_spatial=recording_hp_spatial.to_dict(relative_to=data_folder, recursive=True))
                        )

                    denoising_strategy = preprocessing_params["denoising_strategy"]
                    if denoising_strategy == "cmr":
                        recording_processed = recording_processed_cmr
                    else:
                        # ================================================================
                        # FIX: Fall back to CMR if destripe (highpass_spatial) failed
                        # ================================================================
                        if recording_hp_spatial is not None:
                            recording_processed = recording_hp_spatial
                        else:
                            print(f"\t[custom] Destripe failed, falling back to CMR")
                            recording_processed = recording_processed_cmr
                            preprocessing_notes += "\n- Destripe failed, fell back to CMR."
                        # ================================================================

                    if preprocessing_params["remove_bad_channels"]:
                        print(f"\tRemoving {len(bad_channel_ids)} channels after {denoising_strategy} preprocessing")
                        recording_processed = recording_processed.remove_channels(bad_channel_ids)
                        preprocessing_notes += f"\n- Removed {len(bad_channel_ids)} bad channels after preprocessing.\n"

                    # save to binary to speed up downstream processing
                    recording_bin = recording_processed.save(folder=preprocessing_output_folder)

                    # motion correction
                    recording_corrected = None
                    recording_bin_corrected = None
                    if motion_params["compute"]:
                        from spikeinterface.sortingcomponents.motion import interpolate_motion

                        preset = motion_params["preset"]
                        print(f"\tComputing motion correction with preset: {preset}")

                        detect_kwargs = motion_params.get("detect_kwargs", {})
                        select_kwargs = motion_params.get("select_kwargs", {})
                        localize_peaks_kwargs = motion_params.get("localize_peaks_kwargs", {})
                        estimate_motion_kwargs = motion_params.get("estimate_motion_kwargs", {})
                        interpolate_motion_kwargs = motion_params.get("interpolate_motion_kwargs", {})

                        motion_folder = results_folder / f"motion_{recording_name}"

                        try:
                            concat_motion = False
                            if recording_processed.get_num_segments() > 1:
                                recording_bin_c = si.concatenate_recordings([recording_bin])
                                recording_processed_c = si.concatenate_recordings([recording_processed])
                                concat_motion = True
                            else:
                                recording_bin_c = recording_bin
                                recording_processed_c = recording_processed

                            recording_bin_corrected, motion_info = spre.correct_motion(
                                recording_bin_c,
                                preset=preset,
                                folder=motion_folder,
                                output_motion_info=True,
                                detect_kwargs=detect_kwargs,
                                select_kwargs=select_kwargs,
                                localize_peaks_kwargs=localize_peaks_kwargs,
                                estimate_motion_kwargs=estimate_motion_kwargs,
                                interpolate_motion_kwargs=interpolate_motion_kwargs
                            )
                            recording_corrected = interpolate_motion(
                                recording_processed_c.astype("float32"),
                                motion=motion_info["motion"],
                                **interpolate_motion_kwargs
                            )

                            # split segments back
                            if concat_motion:
                                rec_corrected_list = []
                                rec_corrected_bin_list = []
                                for segment_index in range(recording_bin.get_num_segments()):
                                    num_samples = recording_bin.get_num_samples(segment_index)
                                    if segment_index == 0:
                                        start_frame = 0
                                    else:
                                        start_frame = recording_bin.get_num_samples(segment_index - 1)
                                    end_frame = start_frame + num_samples
                                    rec_split_corrected = recording_corrected.frame_slice(
                                        start_frame=start_frame,
                                        end_frame=end_frame
                                    )
                                    rec_corrected_list.append(rec_split_corrected)
                                    rec_split_bin = recording_bin_corrected.frame_slice(
                                        start_frame=start_frame,
                                        end_frame=end_frame
                                    )
                                    rec_corrected_bin_list.append(rec_split_bin)
                                # append all segments
                                recording_corrected = si.append_recordings(rec_corrected_list)
                                recording_bin_corrected = si.append_recordings(rec_corrected_bin_list)

                            if motion_params["apply"]:
                                print(f"\tApplying motion correction")
                                recording_processed = recording_corrected
                                recording_bin = recording_bin_corrected
                        except Exception as e:
                            print(f"\tMotion correction failed:\n\t{e}")
                            recording_corrected = None
                            recording_bin_corrected = None

                    # this is used to reload the binary traces downstream
                    dump_to_json_or_pickle(
                        recording_bin,
                        results_folder,
                        binary_output_filename,
                        relative_to=results_folder
                    )

                    # this is to reload the recordings lazily            
                    dump_to_json_or_pickle(
                        recording_processed,
                        results_folder,
                        preprocessing_output_filename,
                        relative_to=results_folder
                    )

                    # this is to reload the motion-corrected recording lazily
                    if recording_corrected is not None:     
                        dump_to_json_or_pickle(
                            recording_corrected,
                            results_folder,
                            motioncorrected_output_filename,
                            relative_to=results_folder
                        )

                    recording_drift = recording_bin
                    drift_relative_folder = results_folder

            if skip_processing:
                # in this case, processed timeseries will not be visualized
                preprocessing_vizualization_data[recording_name]["timeseries"]["proc"] = None
                recording_drift = recording_filt_full
                drift_relative_folder = data_folder
                # make a dummy file if too many bad channels to skip downstream processing
                preprocessing_output_folder.mkdir()
                error_file = preprocessing_output_folder / "error.txt"
                error_file.write_text("Too many bad channels")

            # store recording for drift visualization
            preprocessing_vizualization_data[recording_name]["drift"] = dict(
                recording=recording_drift.to_dict(relative_to=drift_relative_folder, recursive=True)
            )

            if vizualization_file_is_json_serializable:            
                with open(results_folder / f"{preprocessingviz_output_filename}.json", "w") as f:
                    json.dump(check_json(preprocessing_vizualization_data), f, indent=4)
            else:
                # then dump to pickle
                with open(results_folder / f"{preprocessingviz_output_filename}.pkl", "wb") as f:
                    pickle.dump(preprocessing_vizualization_data, f)

            t_preprocessing_end = time.perf_counter()
            elapsed_time_preprocessing = np.round(t_preprocessing_end - t_preprocessing_start, 2)

            # save params in output
            preprocessing_params["recording_name"] = recording_name
            # NEW: Add manual bad channels to saved params
            preprocessing_params["manual_bad_channel_ids"] = session_manual_bad_channels
            
            if channel_labels is not None:
                preprocessing_outputs = dict(
                    channel_labels=channel_labels.tolist(),
                )
            else:
                preprocessing_outputs = dict()
            preprocessing_process = DataProcess(
                name="Ephys preprocessing",
                software_version=VERSION,  # either release or git commit
                start_date_time=datetime_start_preproc,
                end_date_time=datetime_start_preproc + timedelta(seconds=np.floor(elapsed_time_preprocessing)),
                input_location=str(data_folder),
                output_location=str(results_folder),
                code_url=URL,
                parameters=preprocessing_params,
                outputs=preprocessing_outputs,
                notes=preprocessing_notes,
            )
            with open(preprocessing_output_process_json, "w") as f:
                f.write(preprocessing_process.model_dump_json(indent=3))

        t_preprocessing_end_all = time.perf_counter()
        elapsed_time_preprocessing_all = np.round(t_preprocessing_end_all - t_preprocessing_start_all, 2)

        print(f"PREPROCESSING time: {elapsed_time_preprocessing_all}s")
