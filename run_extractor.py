#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run_extractor.py - General Purpose Voice Extractor
Author: Your Name/Organization
Version: 1.2.0 (Split segments around overlap, updated deps)

This script processes an input audio file to identify, isolate (solo segments only),
and transcribe segments of a specified target speaker.
Target speaker segments are split to remove detected speech overlap, maximizing data retention.

Key Features:
- Speaker diarization using PyAnnote Audio.
- Overlapped speech detection using PyAnnote Audio.
- Target speaker identification using Resemblyzer.
- Vocal separation using Demucs (optional pre-processing).
- Multi-model speaker verification (SpeechBrain & Resemblyzer) for refined solo segments.
- Voice activity detection (VAD) using Silero-VAD for refined solo segments.
- Transcription using OpenAI Whisper for refined solo segments.
- Detailed logging and visualization of processing stages.

Example Usage:
python run_extractor.py --input-audio "path/to/interview.wav" \
                        --reference-audio "path/to/target_speaker_sample.wav" \
                        --target-name "SpeakerName" \
                        --output-base-dir "./extraction_results"
"""

import argparse
import sys
from pathlib import Path
import time
import shutil
import logging
import json # For diarization hyperparams
import torch

# --- Bootstrap Dependencies (Call _ensure early) ---
from common import _ensure, REQ
_ensure(REQ) # Ensure all dependencies are met with specified versions

# --- Now import other common elements and pipeline functions ---
from common import (
    log, console, DEVICE, DEFAULT_OUTPUT_BASE_DIR, get_huggingface_token,
    DEFAULT_MIN_SEGMENT_SEC, DEFAULT_MAX_MERGE_GAP,
    DEFAULT_VERIFICATION_THRESHOLD,
    save_detailed_spectrograms, create_comparison_spectrograms, create_diarization_plot,
    ensure_dir_exists, safe_filename, format_duration, set_args_for_debug,
    torchaudio_version, torchvision_version # Import version strings
)

from audio_pipeline import (
    prepare_reference_audio,
    diarize_audio,
    detect_overlapped_regions,
    identify_target_speaker,
    run_demucs,
    slice_and_verify_target_solo_segments,
    transcribe_segments,
    concatenate_segments,
    HAVE_SPEECHBRAIN
)


def main(args):
    """Main orchestrator function for the voice extraction pipeline."""
    start_time_total = time.time()
    set_args_for_debug(args)
    log.info(f"[bold cyan]===== Voice Extractor Initializing (Device: {DEVICE.type.upper()}) =====[/]")
    log.info("[bold yellow]Strategy: Identify target speaker segments and SPLIT them to remove detected overlaps, keeping only solo portions.[/]")


    input_audio_p = Path(args.input_audio)
    reference_audio_p = Path(args.reference_audio)
    target_name_str = args.target_name

    if not input_audio_p.is_file():
        log.error(f"[bold red]Input audio file not found: {input_audio_p}. Exiting.[/]")
        sys.exit(1)
    if not reference_audio_p.is_file():
        log.error(f"[bold red]Reference audio file not found: {reference_audio_p}. Exiting.[/]")
        sys.exit(1)
    if not target_name_str.strip():
        log.error("[bold red]Target name cannot be empty. Exiting.[/]")
        sys.exit(1)

    # --- Setup Directories ---
    run_output_dir_name = f"{safe_filename(target_name_str)}_{input_audio_p.stem}_SOLO_Split"
    output_dir = Path(args.output_base_dir) / run_output_dir_name
    run_tmp_dir = output_dir / "__tmp_processing"
    segments_base_output_dir = output_dir / "target_segments_solo"
    transcripts_output_dir = output_dir / "transcripts_solo"
    concatenated_output_dir = output_dir / "concatenated_audio_solo"
    visualizations_output_dir = output_dir / "visualizations"

    for dir_path in [output_dir, run_tmp_dir, segments_base_output_dir,
                     transcripts_output_dir, concatenated_output_dir, visualizations_output_dir]:
        ensure_dir_exists(dir_path)

    log.info(f"Processing input: [bold cyan]{input_audio_p.name}[/]")
    log.info(f"Reference audio for '{target_name_str}': [bold cyan]{reference_audio_p.name}[/]")
    log.info(f"Run output directory: [bold cyan]{output_dir}[/]")
    log.info(f"Temporary files in: [bold cyan]{run_tmp_dir}[/]")
    log.info(f"Solo segment parameters: min_duration={args.min_duration}s, merge_gap={args.merge_gap}s")
    log.info(f"Speaker verification threshold: {args.verification_threshold:.2f}")
    if args.dry_run: log.warning("[DRY-RUN MODE ENABLED] Processing will be limited.")
    if args.disable_speechbrain: log.info("SpeechBrain verification is disabled by user.")
    if args.skip_demucs: log.info("Demucs vocal separation will be SKIPPED.")


    # --- STAGE 1: Prepare Reference & Initial Spectrogram ---
    log.info("[bold magenta]== STAGE 1: Reference Preparation & Initial Analysis ==[/]")
    processed_reference_file = prepare_reference_audio(reference_audio_p, run_tmp_dir, target_name_str)
    save_detailed_spectrograms(input_audio_p, visualizations_output_dir, "01_Original_InputAudio", target_name_str)
    save_detailed_spectrograms(processed_reference_file, visualizations_output_dir, "00_Processed_ReferenceAudio", target_name_str)

    # --- STAGE 2: Optional Vocal Separation (Demucs) ---
    source_for_diarization_osd = input_audio_p # This will be the audio fed to diarization and OSD
    demucs_vocals_file = None # This will be the audio fed to slicing IF demucs is run
    if not args.skip_demucs:
        log.info("[bold magenta]== STAGE 2: Vocal Separation (Demucs) ==[/]")
        demucs_vocals_file = run_demucs(input_audio_p, run_tmp_dir)
        save_detailed_spectrograms(demucs_vocals_file, visualizations_output_dir, "02a_Demucs_Vocals_Only", target_name_str)
        source_for_diarization_osd = demucs_vocals_file # Use cleaner vocals for diarization/OSD
        log.info(f"Using Demucs output '{demucs_vocals_file.name}' for subsequent diarization and OSD.")
    else:
        log.info(f"Using original input '{input_audio_p.name}' for diarization and OSD (Demucs skipped).")


    # --- STAGE 3: Speaker Diarization ---
    log.info("[bold magenta]== STAGE 3: Speaker Diarization ==[/]")
    diar_model_config = {"diar_model": args.diar_model, "diar_hyperparams": {}}
    if args.diar_hyperparams:
        try:
            diar_model_config["diar_hyperparams"] = json.loads(args.diar_hyperparams)
            log.info(f"Using custom diarization hyperparameters: {diar_model_config['diar_hyperparams']}")
        except json.JSONDecodeError:
            log.error(f"Invalid JSON for diarization hyperparameters: {args.diar_hyperparams}. Using defaults.")
    
    diarization_annotation = diarize_audio(source_for_diarization_osd, run_tmp_dir, args.token, diar_model_config, args.dry_run)
    if not diarization_annotation or not diarization_annotation.labels():
        log.error("[bold red]Diarization failed or produced no speaker labels. Cannot proceed. Exiting.[/]")
        if args.dry_run: log.warning("Diarization might be empty due to dry-run mode.")
        sys.exit(1)

    # --- STAGE 4: Overlapped Speech Detection ---
    log.info("[bold magenta]== STAGE 4: Overlapped Speech Detection ==[/]")
    overlap_timeline = detect_overlapped_regions(source_for_diarization_osd, run_tmp_dir, args.token, args.osd_model, args.dry_run)


    # --- STAGE 5: Identify Target Speaker ---
    log.info(f"[bold magenta]== STAGE 5: Identifying Target Speaker ('{target_name_str}') ==[/]")
    identified_target_label = identify_target_speaker(diarization_annotation, source_for_diarization_osd, processed_reference_file, target_name_str)

    # Plot diarization after identifying target, so target can be highlighted, and overlaps shown
    create_diarization_plot(diarization_annotation, identified_target_label, target_name_str,
                            visualizations_output_dir,
                            plot_title_prefix=f"03_Diarization_with_Overlap_{safe_filename(target_name_str)}",
                            overlap_timeline=overlap_timeline)
    # Also save a spectrogram of the audio that was diarized/OSD'd, showing overlaps
    save_detailed_spectrograms(source_for_diarization_osd, visualizations_output_dir,
                               "04_Source_For_Slicing_with_OverlapMarkings", target_name_str,
                               overlap_timeline=overlap_timeline)


    # --- STAGE 6: Slice, Verify Target's Refined SOLO Segments ---
    log.info(f"[bold magenta]== STAGE 6: Slice & Verify '{target_name_str}' Refined SOLO Segments ==[/]")
    if args.disable_speechbrain:
        log.info("User disabled SpeechBrain, it will not be used for verification even if available.")
        # To modify the flag in audio_pipeline module:
        import audio_pipeline
        audio_pipeline.HAVE_SPEECHBRAIN = False

    # Slicing source should be the audio we want the final segments from.
    # If Demucs was run, use its output. Otherwise, use the original input.
    slicing_source_audio = demucs_vocals_file if demucs_vocals_file and demucs_vocals_file.exists() else input_audio_p
    log.info(f"Slicing final segments from: {slicing_source_audio.name}")

    solo_segment_paths = slice_and_verify_target_solo_segments(
        diarization_annotation, identified_target_label, overlap_timeline,
        slicing_source_audio, # Audio to cut from
        processed_reference_file, target_name_str,
        segments_base_output_dir, # Base for "..._solo_verified"
        run_tmp_dir,
        args.verification_threshold,
        args.min_duration, args.merge_gap,
        output_sample_rate=int(args.output_sr),
        output_channels=1
    )

    # --- STAGE 7: Transcribe SOLO Segments ---
    log.info(f"[bold magenta]== STAGE 7: Transcribing SOLO Segments ('{target_name_str}') ==[/]")
    if solo_segment_paths:
        transcribe_segments(solo_segment_paths, transcripts_output_dir, target_name_str, "solo_verified", args.whisper_model, args.language)
    else:
        log.info(f"No refined solo segments of '{target_name_str}' to transcribe.")


    # --- STAGE 8: Concatenate SOLO Segments ---
    log.info(f"[bold magenta]== STAGE 8: Concatenating SOLO Segments ('{target_name_str}') ==[/]")
    concat_sr = int(args.output_sr)
    concatenated_solo_file = None

    if solo_segment_paths:
        concatenated_solo_file_path = concatenated_output_dir / f"{safe_filename(target_name_str)}_solo_split_concatenated.wav"
        concat_solo_success = concatenate_segments(solo_segment_paths, concatenated_solo_file_path, run_tmp_dir,
                                                   silence_duration=args.concat_silence, output_sr_concat=concat_sr)
        if concat_solo_success:
            concatenated_solo_file = concatenated_solo_file_path
            save_detailed_spectrograms(concatenated_solo_file, visualizations_output_dir,
                                       "05_Concatenated_Target_SOLO_Split", target_name_str)
    else:
        log.info(f"No refined solo segments of '{target_name_str}' to concatenate.")


    # --- STAGE 9: Final Comparison Spectrograms ---
    log.info("[bold magenta]== STAGE 9: Generating Final Comparison Spectrograms ==[/]")
    comparison_files_list = [(input_audio_p, "Original Input")]
    # Key for overlap_timeline_for_plots should be the *resolved absolute path* of the audio file
    # that the overlap_timeline corresponds to (i.e., source_for_diarization_osd)
    overlap_timeline_for_plots = {str(source_for_diarization_osd.resolve()): overlap_timeline}

    if demucs_vocals_file and demucs_vocals_file.exists():
        comparison_files_list.append((demucs_vocals_file, "Demucs Vocals Only"))
        # If OSD was on demucs_vocals_file, overlap_timeline_for_plots already has it keyed correctly above.

    if concatenated_solo_file and concatenated_solo_file.exists():
        comparison_files_list.append((concatenated_solo_file, f"{target_name_str} Segments (SOLO Split, Concatenated)"))
        # Concatenated solo file has no overlaps, so no timeline for it here.

    create_comparison_spectrograms(comparison_files_list, visualizations_output_dir, target_name_str,
                                   main_prefix="06_Audio_Processing_Stages_Overview",
                                   overlap_timeline_dict=overlap_timeline_for_plots)


    # --- Finalization ---
    if not args.keep_temp_files and run_tmp_dir.exists():
        log.info(f"Cleaning up temporary directory: {run_tmp_dir}")
        try:
            shutil.rmtree(run_tmp_dir)
        except Exception as e:
            log.warning(f"Could not remove temporary directory {run_tmp_dir}: {e}")
    else:
        log.info(f"Temporary processing files kept at: {run_tmp_dir}")

    total_duration_seconds = time.time() - start_time_total
    log.info(f"[bold green]✅ Voice Extractor SOLO (Split Overlap) processing finished successfully for '{target_name_str}'![/]")
    log.info(f"Total processing time: {format_duration(total_duration_seconds)}")
    log.info(f"All output files are located in: [bold cyan]{output_dir}[/]")
    log.info(f"  - Verified SOLO Segments: {segments_base_output_dir / (safe_filename(target_name_str) + '_solo_verified')}")
    log.info(f"  - Transcripts (SOLO): {transcripts_output_dir}")
    log.info(f"  - Concatenated Audio (SOLO): {concatenated_output_dir}")
    log.info(f"  - Visualizations: {visualizations_output_dir}")


# --- CLI Argument Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="General Purpose Voice Extractor: Identifies, isolates SOLO (non-overlapped by splitting) segments, "
                    "and transcribes speech of a target speaker. Uses PyAnnote for diarization and overlap detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    req_group = parser.add_argument_group('Required Arguments')
    req_group.add_argument("--input-audio", "-i", type=str, required=True, help="Path to the main input audio file.")
    req_group.add_argument("--reference-audio", "-r", type=str, required=True, help="Path to a clean reference audio clip of the target speaker.")
    req_group.add_argument("--target-name", "-n", type=str, required=True, help="A name for the target speaker.")

    path_group = parser.add_argument_group('Output and Path Arguments')
    path_group.add_argument("--output-base-dir", "-o", type=str, default=str(DEFAULT_OUTPUT_BASE_DIR), help="Base directory for all output.")
    path_group.add_argument("--output-sr", type=int, default=44100, help="Sample rate for final extracted and concatenated SOLO audio segments (Hz).")

    auth_group = parser.add_argument_group('Authentication Arguments')
    auth_group.add_argument("--token", "-t", type=str, default=None, help="HuggingFace User Access Token for PyAnnote models.")

    proc_group = parser.add_argument_group('Processing Control Arguments')
    proc_group.add_argument("--diar-model", type=str, default="pyannote/speaker-diarization-3.1",
                            help="PyAnnote speaker diarization model from Hugging Face (e.g., 'pyannote/speaker-diarization-3.1', 'pyannote/speaker-diarization@2.1').")
    proc_group.add_argument("--diar-hyperparams", type=str, default="{}",
                            help="JSON string of hyperparameters for the PyAnnote diarization pipeline (e.g., '{\"min_duration_on\": 0.05, \"min_duration_off\": 0.05}'). Applied to the diarization step only.")
    proc_group.add_argument("--osd-model", type=str, default="pyannote/segmentation-3.0",
                            help="PyAnnote model for Overlapped Speech Detection (e.g., 'pyannote/overlapped-speech-detection', 'pyannote/segmentation-3.0', 'pyannote/segmentation'). segmentation-3.0 is a valid segmentation model.")
    proc_group.add_argument("--whisper-model", type=str, default="base.en", help="Whisper model name for transcription.")
    proc_group.add_argument("--language", type=str, default="en", help="Language code for Whisper transcription ('auto' for detection).")
    proc_group.add_argument("--disable-speechbrain", action="store_true", help="Disable SpeechBrain for speaker verification.")
    proc_group.add_argument("--skip-demucs", action="store_true", help="Skip the Demucs vocal separation stage.")
    proc_group.add_argument("--concat-silence", type=float, default=0.5, help="Duration of silence (seconds) between concatenated SOLO segments.")

    tune_group = parser.add_argument_group('Fine-tuning Parameters for SOLO Segments')
    tune_group.add_argument("--min-duration", type=float, default=DEFAULT_MIN_SEGMENT_SEC,
                            help="Minimum duration (seconds) for a refined SOLO voice segment to be kept *after splitting and merging*.")
    tune_group.add_argument("--merge-gap", type=float, default=DEFAULT_MAX_MERGE_GAP,
                            help="Maximum gap (seconds) between target speaker's refined SOLO segments (post-splitting) to merge them.")
    tune_group.add_argument("--verification-threshold", type=float, default=DEFAULT_VERIFICATION_THRESHOLD,
                            help="Minimum speaker verification score (0.0-1.0) for a refined SOLO segment.")

    debug_group = parser.add_argument_group('Debugging and Miscellaneous')
    debug_group.add_argument("--dry-run", "-d", action="store_true", help="Limits diarization/OSD to first 60s for quick testing.")
    debug_group.add_argument("--debug", action="store_true", help="Enable verbose DEBUG level logging.")
    debug_group.add_argument("--keep-temp-files", action="store_true", help="Keep temporary processing directory.")

    parsed_args = parser.parse_args()
    set_args_for_debug(parsed_args) # Make args available for plotting error tracebacks in common.py

    if parsed_args.debug:
        log.setLevel(logging.DEBUG)
        log.debug("Debug mode enabled. Logging will be verbose.")
        if torch.cuda.is_available() and DEVICE.type == "cuda":
            log.debug(f"PyTorch version: {torch.__version__}")
            log.debug(f"Torchaudio version: {torchaudio_version}")
            log.debug(f"Torchvision version: {torchvision_version}")
            log.debug(f"CUDA device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                log.debug(f"  Device {i}: {torch.cuda.get_device_name(i)}")
                # Check memory only if CUDA is primary device to avoid errors on CPU-only systems with torch.cuda calls
                if DEVICE.type == 'cuda':
                    try:
                        log.debug(f"    Memory (Allocated/Reserved): {torch.cuda.memory_allocated(i)/1e9:.2f}GB / {torch.cuda.memory_reserved(i)/1e9:.2f}GB")
                    except Exception as e_mem:
                        log.debug(f"    Could not get CUDA memory stats for device {i}: {e_mem}")


    if not parsed_args.token:
        parsed_args.token = get_huggingface_token() # Ensure token is fetched if not provided

    try:
        main(parsed_args)
    except KeyboardInterrupt:
        log.warning("[bold yellow]\nProcess interrupted by user (Ctrl+C). Exiting.[/]")
        if not parsed_args.keep_temp_files:
            try:
                _input_audio_p_for_tmp = Path(parsed_args.input_audio)
                _target_name_str_for_tmp = parsed_args.target_name
                _run_output_dir_name_for_tmp = f"{safe_filename(_target_name_str_for_tmp)}_{_input_audio_p_for_tmp.stem}_SOLO_Split"
                _output_dir_for_tmp = Path(parsed_args.output_base_dir) / _run_output_dir_name_for_tmp
                tmp_dir_to_clean = _output_dir_for_tmp / "__tmp_processing"
                if tmp_dir_to_clean.exists():
                    log.info(f"Attempting to clean temporary directory on interrupt: {tmp_dir_to_clean}")
                    shutil.rmtree(tmp_dir_to_clean, ignore_errors=True)
            except Exception as e_tmp_clean:
                log.debug(f"Could not determine or clean tmp_dir path on interrupt: {e_tmp_clean}")
        sys.exit(130)
    except FileNotFoundError as e:
        log.error(f"[bold red][FILE NOT FOUND ERROR] {e}[/]")
        sys.exit(2)
    except RuntimeError as e:
        log.error(f"[bold red][RUNTIME ERROR] {e}[/]")
        if parsed_args.debug: log.exception("Traceback for RuntimeError:")
        sys.exit(1)
    except SystemExit as e: # To allow sys.exit() to work as intended
        sys.exit(e.code if e.code is not None else 1)
    except Exception as e:
        log.error(f"[bold red][FATAL SCRIPT ERROR] An unexpected error occurred: {e}[/]")
        if parsed_args.debug: log.exception("Full traceback for unexpected error:")
        sys.exit(1)