#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
audio_pipeline.py
Core audio processing pipeline for the Voice Extractor.
Includes diarization, speaker identification, overlap detection,
vocal separation, verification, transcription, and concatenation.
Target speaker segments are split around detected overlaps to maximize data retention.
"""

import sys
from pathlib import Path
import numpy as np
import shutil
import time
import csv
import subprocess
import os
import re # Import regex for more robust parsing

os.environ['SPEECHBRAIN_FETCH_LOCAL_STRATEGY'] = 'copy'

import torch
import soundfile as sf
import librosa
from pyannote.audio import Pipeline as PyannotePipeline
from pyannote.audio import Model as PyannoteModel
from pyannote.audio.pipelines import OverlappedSpeechDetection as PyannoteOSDPipeline
from pyannote.core import Segment, Timeline, Annotation
from resemblyzer import VoiceEncoder, preprocess_wav as resemblyzer_preprocess_wav
import whisper

try:
    from speechbrain.inference.speaker import SpeakerRecognition
    HAVE_SPEECHBRAIN = True
except ImportError:
    HAVE_SPEECHBRAIN = False

from common import (
    log, console, DEVICE,
    ff_trim, ff_slice, cos, to_mono,
    plot_verification_scores,
    DEFAULT_MIN_SEGMENT_SEC, DEFAULT_MAX_MERGE_GAP,
    ensure_dir_exists, safe_filename, format_duration
)
import ffmpeg
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, SpinnerColumn
from rich.table import Table


def prepare_reference_audio(
    reference_audio_path_arg: Path, tmp_dir: Path, target_name: str
) -> Path:
    log.info(f"Preparing reference audio for '{target_name}' from: {reference_audio_path_arg.name}")
    ensure_dir_exists(tmp_dir)
    processed_ref_filename = f"{safe_filename(target_name)}_reference_processed.wav"
    processed_ref_path = tmp_dir / processed_ref_filename
    if not reference_audio_path_arg.exists():
        raise FileNotFoundError(f"Reference audio file not found: {reference_audio_path_arg}")
    try:
        ff_trim(reference_audio_path_arg, processed_ref_path, 0, 999999, target_sr=16000, target_ac=1)
        if not processed_ref_path.exists() or processed_ref_path.stat().st_size == 0:
            raise RuntimeError("Processed reference audio file is empty or was not created.")
        log.info(f"Processed reference audio saved to: {processed_ref_path.name}")
        return processed_ref_path
    except Exception as e:
        log.error(f"Failed to process reference audio '{reference_audio_path_arg.name}': {e}")
        raise

def diarize_audio(
    input_audio_file: Path, tmp_dir: Path, huggingface_token: str,
    model_config: dict, dry_run: bool = False
) -> Annotation:
    model_name = model_config.get("diar_model", "pyannote/speaker-diarization-3.1")
    hyper_params = model_config.get("diar_hyperparams", {})
    log.info(f"Starting speaker diarization for: {input_audio_file.name} (Model: {model_name})")
    if DEVICE.type == "cuda": torch.cuda.empty_cache()
    ensure_dir_exists(tmp_dir)
    if hyper_params: log.info(f"With diarization hyperparameters: {hyper_params}")
    try:
        pipeline = PyannotePipeline.from_pretrained(model_name, use_auth_token=huggingface_token)
        if hasattr(pipeline, "to") and callable(getattr(pipeline, "to")): pipeline = pipeline.to(DEVICE)
        log.info(f"Diarization model '{model_name}' loaded to {DEVICE.type.upper()}.")
    except Exception as e:
        log.error(f"[bold red]Error loading diarization model '{model_name}': {e}[/]")
        raise RuntimeError(f"Diarization model loading failed for {model_name}") from e

    target_audio_for_processing = input_audio_file
    if dry_run:
        cut_audio_file_path = tmp_dir / f"{input_audio_file.stem}_60s_diar_dryrun.wav"
        log.warning(f"[DRY-RUN] Using first 60s for diarization. Temp: {cut_audio_file_path.name}")
        try:
            ff_trim(input_audio_file, cut_audio_file_path, 0, 60, target_sr=16000)
            target_audio_for_processing = cut_audio_file_path
        except Exception as e:
            log.error(f"Failed to create dry-run audio for diarization: {e}. Using full audio.")

    log.info(f"Running diarization on {DEVICE.type.upper()} for {target_audio_for_processing.name}...")
    try:
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), TimeElapsedColumn(), console=console) as progress:
            task = progress.add_task("Diarizing...", total=None)
            diarization_result = pipeline({"uri": "audio", "audio": str(target_audio_for_processing)}, **hyper_params)
            progress.update(task, completed=1, total=1)
        num_speakers = len(diarization_result.labels())
        total_speech_duration = diarization_result.get_timeline().duration()
        log.info(f"[green]✓ Diarization complete.[/] Found {num_speakers} labels. Total speech: {format_duration(total_speech_duration)}.")
        if num_speakers == 0: log.warning("Diarization resulted in zero speakers.")
        return diarization_result
    except RuntimeError as e:
        if "CUDA out of memory" in str(e) and DEVICE.type == "cuda":
            log.error("[bold red]CUDA out of memory during diarization![/]")
            torch.cuda.empty_cache(); log.warning("Attempting diarization on CPU (slower)...")
            try:
                pipeline = pipeline.to(torch.device("cpu"))
                log.info("Switched diarization pipeline to CPU.")
                with Progress(SpinnerColumn(),TextColumn("[progress.description]{task.description}"),TimeElapsedColumn(),console=console) as p_cpu:
                    task_cpu = p_cpu.add_task("Diarizing (CPU)...", total=None)
                    res_cpu = pipeline({"uri":"audio","audio":str(target_audio_for_processing)}, **hyper_params); p_cpu.update(task_cpu,completed=1,total=1)
                log.info(f"[green]✓ Diarization (CPU) complete.[/] Found {len(res_cpu.labels())} spk. Total speech: {format_duration(res_cpu.get_timeline().duration())}.")
                return res_cpu
            except Exception as cpu_e: raise RuntimeError("Diarization failed on GPU (OOM) and CPU.") from cpu_e
        else: raise
    except Exception as e: raise

def detect_overlapped_regions(
    input_audio_file: Path, tmp_dir: Path, huggingface_token: str,
    osd_model_name: str = "pyannote/segmentation-3.0", dry_run: bool = False
) -> Timeline:
    log.info(f"Starting OSD for: {input_audio_file.name} (OSD Model: {osd_model_name})")
    if DEVICE.type == "cuda": torch.cuda.empty_cache()
    ensure_dir_exists(tmp_dir)

    osd_pipeline_instance = None
    default_osd_hyperparameters = {
        "onset": 0.5,
        "offset": 0.5,
        "min_duration_on": 0.05,
        "min_duration_off": 0.05
    }

    try:
        if osd_model_name == "pyannote/overlapped-speech-detection":
            log.info(f"Loading dedicated OSD pipeline: '{osd_model_name}'...")
            osd_pipeline_instance = PyannotePipeline.from_pretrained(
                osd_model_name, use_auth_token=huggingface_token
            )
        elif osd_model_name.startswith("pyannote/segmentation"):
            log.info(f"Loading '{osd_model_name}' as base segmentation model for OSD pipeline...")
            segmentation_model = PyannoteModel.from_pretrained(
                osd_model_name, use_auth_token=huggingface_token
            )
            osd_pipeline_instance = PyannoteOSDPipeline(
                segmentation=segmentation_model
            )
            osd_pipeline_instance.instantiate(default_osd_hyperparameters)
            log.info(f"Instantiated OverlappedSpeechDetection pipeline (from '{osd_model_name}') with parameters: {default_osd_hyperparameters}.")
        else:
            log.warning(
                f"OSD model string '{osd_model_name}' not recognized as a specific type. "
                "Attempting to load as a generic PyannotePipeline. This may fail if it's a base model."
            )
            osd_pipeline_instance = PyannotePipeline.from_pretrained(
                osd_model_name, use_auth_token=huggingface_token
            )

        if osd_pipeline_instance is None:
            raise RuntimeError(f"Failed to load or instantiate OSD pipeline for '{osd_model_name}'. Instance is None.")

        if hasattr(osd_pipeline_instance, "to") and callable(getattr(osd_pipeline_instance, "to")):
            log.debug(f"Moving OSD pipeline for '{osd_model_name}' to {DEVICE.type.upper()}")
            osd_pipeline_instance = osd_pipeline_instance.to(DEVICE)
        elif hasattr(osd_pipeline_instance, 'segmentation_model') and hasattr(osd_pipeline_instance.segmentation_model, 'to'):
            log.debug(f"Moving OSD pipeline's segmentation model to {DEVICE.type.upper()}")
            osd_pipeline_instance.segmentation_model = osd_pipeline_instance.segmentation_model.to(DEVICE)

        log.info(f"OSD model/pipeline '{osd_model_name}' successfully prepared on {DEVICE.type.upper()}.")

    except Exception as e:
        log.error(f"[bold red]Fatal error loading/instantiating OSD model/pipeline '{osd_model_name}': {type(e).__name__} - {e}[/]")
        if "private or gated" in str(e) or "accept the user conditions" in str(e) or \
           "Could not download" in str(e) or "make sure to authenticate" in str(e) or \
           (isinstance(e, KeyError) and 'pipeline' in str(e)):
            log.error(
                "This can be due to several reasons:\n"
                f"1. The model '{osd_model_name}' is private/gated: Ensure you have accepted terms at https://hf.co/{osd_model_name} and your token is valid.\n"
                f"2. The model '{osd_model_name}' is a base model (like segmentation) and the script failed to wrap it in PyannoteOSDPipeline (this specific 'if' branch should handle it).\n"
                f"3. The model name is incorrect, does not exist, or is not compatible with this version of pyannote.audio.\n"
            )
        raise RuntimeError(f"OSD model/pipeline setup failed for '{osd_model_name}'") from e

    target_audio_for_processing = input_audio_file
    if dry_run:
        cut_audio_file_path = tmp_dir / f"{input_audio_file.stem}_60s_osd_dryrun.wav"
        log.warning(f"[DRY-RUN] Using first 60s for OSD. Temp: {cut_audio_file_path.name}")
        try:
            ff_trim(input_audio_file, cut_audio_file_path, 0, 60, target_sr=16000)
            target_audio_for_processing = cut_audio_file_path
        except Exception as e:
            log.error(f"Failed to create dry-run audio for OSD: {e}. Using full audio.")

    log.info(f"Running OSD on {DEVICE.type.upper()} for {target_audio_for_processing.name}...")
    try:
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), TimeElapsedColumn(), console=console) as progress:
            task = progress.add_task("Detecting overlaps...", total=None)
            osd_annotation = osd_pipeline_instance({"uri": "audio", "audio": str(target_audio_for_processing)})
            progress.update(task, completed=1, total=1)

        overlap_timeline = Timeline()
        if "overlap" in osd_annotation.labels():
            overlap_timeline.update(osd_annotation.label_timeline("overlap"))
        else:
            labels_from_osd = osd_annotation.labels()
            log.debug(f"OSD with '{osd_model_name}' did not directly yield 'overlap' labels. Checking other labels: {labels_from_osd}")
            found_overlap_via_other_means = False
            if osd_model_name == "pyannote/overlapped-speech-detection" and "speech" in labels_from_osd:
                log.info("Using 'speech' label from 'pyannote/overlapped-speech-detection' as overlap.")
                overlap_timeline.update(osd_annotation.label_timeline("speech"))
                found_overlap_via_other_means = True

            if not found_overlap_via_other_means:
                for label in labels_from_osd:
                    if osd_model_name.startswith("pyannote/segmentation"):
                        try:
                            count_str = label.split('_')[-1].replace('+','')
                            if count_str.isdigit() and int(count_str) >= 2:
                                overlap_timeline.update(osd_annotation.label_timeline(label))
                                found_overlap_via_other_means = True
                                log.info(f"Inferred overlap from speaker count label '{label}'.")
                                break
                        except ValueError: pass
                    elif "overlap" in label.lower():
                         overlap_timeline.update(osd_annotation.label_timeline(label))
                         found_overlap_via_other_means = True
                         log.info(f"Using label '{label}' as overlap.")
                         break

            if not overlap_timeline and not found_overlap_via_other_means and labels_from_osd:
                 log.warning(f"OSD model '{osd_model_name}' did not produce clear 'overlap' or speaker count >= 2 labels. Labels: {labels_from_osd}. Fallback: union of all segments from OSD.")
                 for label in labels_from_osd:
                     overlap_timeline.update(osd_annotation.label_timeline(label))

        overlap_timeline = overlap_timeline.support()
        total_overlap_duration = overlap_timeline.duration()
        log.info(f"[green]✓ Overlap detection complete.[/] Total overlap: {format_duration(total_overlap_duration)}.")
        if total_overlap_duration == 0: log.info("No overlapped speech detected by OSD model.")
        return overlap_timeline
    except RuntimeError as e:
        if "CUDA out of memory" in str(e) and DEVICE.type == "cuda":
            log.error("[bold red]CUDA out of memory during OSD![/]")
            torch.cuda.empty_cache(); log.warning("Attempting OSD on CPU (slower)...")
            cpu_device = torch.device("cpu")
            try:
                osd_pipeline_cpu = None
                if osd_model_name == "pyannote/overlapped-speech-detection":
                    log.info(f"Re-loading dedicated OSD pipeline '{osd_model_name}' for CPU...")
                    osd_pipeline_cpu = PyannotePipeline.from_pretrained(
                        osd_model_name, use_auth_token=huggingface_token
                    ).to(cpu_device)
                elif osd_model_name.startswith("pyannote/segmentation"):
                    log.info(f"Re-loading base model '{osd_model_name}' and wrapping in OSD pipeline for CPU...")
                    segmentation_model_cpu = PyannoteModel.from_pretrained(
                        osd_model_name, use_auth_token=huggingface_token
                    ).to(cpu_device)
                    osd_pipeline_cpu = PyannoteOSDPipeline(
                        segmentation=segmentation_model_cpu
                    )
                    osd_pipeline_cpu.instantiate(default_osd_hyperparameters)
                    log.info(f"Instantiated OSD pipeline (from '{osd_model_name}') for CPU with parameters: {default_osd_hyperparameters}.")
                    if hasattr(osd_pipeline_cpu, "to") and callable(getattr(osd_pipeline_cpu, "to")):
                        osd_pipeline_cpu = osd_pipeline_cpu.to(cpu_device)
                else:
                    log.info(f"Re-loading generic OSD pipeline '{osd_model_name}' for CPU...")
                    osd_pipeline_cpu = PyannotePipeline.from_pretrained(
                        osd_model_name, use_auth_token=huggingface_token
                    ).to(cpu_device)

                if osd_pipeline_cpu is None:
                    raise RuntimeError("Failed to create OSD pipeline for CPU fallback.")

                log.info("Switched OSD pipeline to CPU.")
                with Progress(SpinnerColumn(),TextColumn("[progress.description]{task.description}"),TimeElapsedColumn(),console=console) as p_cpu:
                    task_cpu = p_cpu.add_task("Detecting overlaps (CPU)...", total=None)
                    osd_ann_cpu = osd_pipeline_cpu({"uri":"audio","audio":str(target_audio_for_processing)}); p_cpu.update(task_cpu,completed=1,total=1)

                ov_tl_cpu = Timeline()
                if "overlap" in osd_ann_cpu.labels():
                    ov_tl_cpu.update(osd_ann_cpu.label_timeline("overlap"))
                else:
                    labels_from_osd_cpu = osd_ann_cpu.labels()
                    found_overlap_cpu = False
                    if osd_model_name == "pyannote/overlapped-speech-detection" and "speech" in labels_from_osd_cpu:
                         ov_tl_cpu.update(osd_ann_cpu.label_timeline("speech"))
                         found_overlap_cpu = True

                    if not found_overlap_cpu:
                        for label in labels_from_osd_cpu:
                            if osd_model_name.startswith("pyannote/segmentation"):
                                try:
                                    count_str = label.split('_')[-1].replace('+','')
                                    if count_str.isdigit() and int(count_str) >= 2:
                                        ov_tl_cpu.update(osd_ann_cpu.label_timeline(label)); found_overlap_cpu = True
                                        break
                                except ValueError: pass
                            elif "overlap" in label.lower():
                                ov_tl_cpu.update(osd_ann_cpu.label_timeline(label)); found_overlap_cpu = True
                                break

                    if not ov_tl_cpu and not found_overlap_cpu and labels_from_osd_cpu:
                         for label in labels_from_osd_cpu:
                             ov_tl_cpu.update(osd_ann_cpu.label_timeline(label))

                ov_tl_cpu = ov_tl_cpu.support()
                log.info(f"[green]✓ OSD (CPU) complete.[/] Total overlap: {format_duration(ov_tl_cpu.duration())}.")
                return ov_tl_cpu
            except Exception as cpu_e: raise RuntimeError(f"OSD failed on GPU (OOM) and subsequently on CPU: {cpu_e}") from cpu_e
        else: raise
    except Exception as e:
        log.error(f"An error occurred during OSD processing of {target_audio_for_processing.name}: {e}")
        raise

def identify_target_speaker(
    annotation: Annotation, input_audio_file: Path, processed_reference_file: Path, target_name: str
) -> str:
    log.info(f"Identifying '{target_name}' among diarized speakers using reference: {processed_reference_file.name}")
    try:
        encoder = VoiceEncoder(device="cuda" if DEVICE.type == "cuda" else "cpu")
    except TypeError: encoder = VoiceEncoder(); log.debug("Resemblyzer using CPU (old version or CUDA issue).")
    try:
        ref_wav_processed = resemblyzer_preprocess_wav(processed_reference_file)
        ref_embedding = encoder.embed_utterance(ref_wav_processed)
    except Exception as e: raise RuntimeError(f"Reference audio processing failed for Resemblyzer: {e}") from e
    try:
        full_input_audio_wav, sr_input = librosa.load(input_audio_file, sr=None, mono=True)
        if sr_input != 16000: full_input_audio_wav = librosa.resample(full_input_audio_wav, orig_sr=sr_input, target_sr=16000)
    except Exception as e: raise RuntimeError(f"Input audio loading failed for speaker ID: {e}") from e

    speaker_similarities = {}
    unique_speaker_labels = annotation.labels()
    if not unique_speaker_labels: raise ValueError("Diarization produced no speaker labels.")

    log.info(f"Comparing reference of '{target_name}' with {len(unique_speaker_labels)} diarized speakers.")
    for spk_label in unique_speaker_labels:
        speaker_segments_timeline = annotation.label_timeline(spk_label)
        if not speaker_segments_timeline: log.debug(f"Spk label '{spk_label}' has no speech segments. Skipping."); continue

        combined_audio_for_embedding_list = []
        current_duration_for_embedding = 0.0; MAX_EMBED_DURATION = 20.0
        for seg in speaker_segments_timeline:
            start_sample, end_sample = int(seg.start * 16000), int(seg.end * 16000)
            if start_sample >= len(full_input_audio_wav) or start_sample >= end_sample: continue
            end_sample = min(end_sample, len(full_input_audio_wav))
            audio_segment_np = full_input_audio_wav[start_sample:end_sample]
            if len(audio_segment_np) == 0: continue
            combined_audio_for_embedding_list.append(audio_segment_np)
            current_duration_for_embedding += seg.duration
            if current_duration_for_embedding >= MAX_EMBED_DURATION: break

        if combined_audio_for_embedding_list:
            try:
                concatenated_wav_np = np.concatenate(combined_audio_for_embedding_list)
                spk_embedding = encoder.embed_utterance(concatenated_wav_np)
                similarity = cos(ref_embedding, spk_embedding)
                speaker_similarities[spk_label] = similarity
            except Exception as e: log.warning(f"Error embedding for spk '{spk_label}': {e}. Score 0."); speaker_similarities[spk_label] = 0.0
        else: log.debug(f"No valid audio for spk '{spk_label}' embedding."); speaker_similarities[spk_label] = 0.0

    if not speaker_similarities: raise RuntimeError(f"Speaker similarity calculation failed for {target_name}.")
    if all(score == 0.0 for score in speaker_similarities.values()):
        log.error(f"[bold red]All speaker similarity scores are zero for '{target_name}'. Cannot reliably ID target.[/]")
        best_match_label = unique_speaker_labels[0] if unique_speaker_labels else "UNKNOWN_SPEAKER"
        max_similarity_score = 0.0
        log.warning(f"Arbitrarily assigning '{best_match_label}' due to all zero scores.")
    else:
        best_match_label = max(speaker_similarities, key=speaker_similarities.get)
        max_similarity_score = speaker_similarities[best_match_label]

    log.info(f"[green]✓ Identified '{target_name}' as label → [bold]{best_match_label}[/] (Resemblyzer sim: {max_similarity_score:.3f})[/]")
    sim_table = Table(title=f"Resemblyzer Similarities to '{target_name}'", show_lines=True, highlight=True)
    sim_table.add_column("Speaker Label", style="cyan", justify="center"); sim_table.add_column("Similarity Score", style="magenta", justify="center")
    for spk, score in sorted(speaker_similarities.items(), key=lambda item: item[1], reverse=True):
        sim_table.add_row(spk, f"{score:.4f}", style="bold yellow on bright_black" if spk == best_match_label else "")
    console.print(sim_table)
    return best_match_label

def run_demucs(input_audio_file: Path, tmp_dir: Path) -> Path:
    log.info(f"Starting vocal separation with Demucs for: {input_audio_file.name}")
    ensure_dir_exists(tmp_dir)
    demucs_output_base_dir = tmp_dir / "demucs_output"
    expected_vocals_file = demucs_output_base_dir / "htdemucs" / input_audio_file.stem / "vocals.wav"
    if expected_vocals_file.exists() and expected_vocals_file.stat().st_size > 0:
        log.info(f"Found existing Demucs vocals, skipping separation: {expected_vocals_file.name}")
        return expected_vocals_file
    log.info("Running Demucs v4 (htdemucs model) for vocal separation…")
    demucs_cmd = [sys.executable, "-m", "demucs", "-n", "htdemucs", "--two-stems", "vocals", "-o", str(demucs_output_base_dir), "--device", DEVICE.type, str(input_audio_file)]
    log.debug(f"Demucs command: {' '.join(demucs_cmd)}")
    try:
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), TimeElapsedColumn(), console=console) as progress:
            task = progress.add_task("Demucs (vocals)...", total=None)
            completed_process = subprocess.run(demucs_cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', check=False)
            progress.update(task, completed=1, total=1)
        if completed_process.returncode != 0:
            log.error(f"[bold red]Demucs failed! (RC: {completed_process.returncode})[/]\nSTDOUT: {completed_process.stdout}\nSTDERR: {completed_process.stderr}")
            raise RuntimeError(f"Demucs vocal separation failed. RC: {completed_process.returncode}")
        log.info("[green]✓ Demucs processing completed.[/]")
        if completed_process.stderr and ("Warning" in completed_process.stderr or "UserWarning" in completed_process.stderr):
             log.warning(f"Demucs warnings/info:\n{completed_process.stderr}")
    except Exception as e: raise RuntimeError(f"Error running Demucs: {e}") from e
    if not expected_vocals_file.exists() or expected_vocals_file.stat().st_size == 0:
        log.error(f"Demucs ran, but '{expected_vocals_file}' missing/empty.")
        possible_vocals = list((demucs_output_base_dir / "htdemucs" / input_audio_file.stem).glob("vocals*.wav"))
        if possible_vocals: log.warning(f"Using fallback: {possible_vocals[0]}"); return possible_vocals[0]
        raise FileNotFoundError("Demucs output vocals.wav not found.")
    log.info(f"Demucs vocals output: {expected_vocals_file.name}")
    return expected_vocals_file

def merge_nearby_segments(segments_to_merge: list[Segment], max_allowed_gap: float = DEFAULT_MAX_MERGE_GAP) -> list[Segment]:
    if not segments_to_merge: return []
    sorted_segments = sorted(list(segments_to_merge), key=lambda s: s.start)
    if not sorted_segments: return []
    merged_timeline = Timeline()
    current_merged_segment = sorted_segments[0]
    for next_segment in sorted_segments[1:]:
        if (next_segment.start <= current_merged_segment.end + max_allowed_gap) and \
           (next_segment.end > current_merged_segment.end):
            current_merged_segment = Segment(current_merged_segment.start, next_segment.end)
        elif next_segment.start > current_merged_segment.end + max_allowed_gap:
            merged_timeline.add(current_merged_segment)
            current_merged_segment = next_segment
    merged_timeline.add(current_merged_segment)
    return list(merged_timeline.support())

def filter_segments_by_duration(segments_to_filter: list[Segment], min_req_duration: float = DEFAULT_MIN_SEGMENT_SEC) -> list[Segment]:
    return [seg for seg in segments_to_filter if seg.duration >= min_req_duration]

def check_voice_activity(audio_path: Path, min_speech_ratio: float = 0.6, vad_threshold: float = 0.5) -> bool:
    try: y, sr = librosa.load(audio_path, sr=16000, mono=True) # VAD expects 16kHz
    except Exception as e: log.debug(f"VAD: Load failed {audio_path.name}: {e}. No voice."); return False
    if len(y) == 0: log.debug(f"VAD: Empty {audio_path.name}. No voice."); return False
    try:
        vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, trust_repo=True, onnx=False)
        (get_speech_timestamps, _, _, _, _) = utils; vad_model.to(DEVICE)
    except Exception as e: log.warning(f"VAD: Model load failed: {e}. Skipping VAD for {audio_path.name}, assuming active."); return True
    try:
        # Ensure audio tensor is on the correct device for VAD model
        audio_tensor = torch.FloatTensor(y).to(DEVICE)
        # Silero VAD model expects 16000, 8000 or 48000Hz. Librosa loaded at 16000Hz.
        speech_timestamps = get_speech_timestamps(audio_tensor, vad_model, sampling_rate=16000, threshold=vad_threshold)
        speech_duration_sec = sum(d['end'] - d['start'] for d in speech_timestamps) / 16000
        total_duration_sec = len(y) / 16000
        ratio = speech_duration_sec / total_duration_sec if total_duration_sec > 0 else 0.0
        log.debug(f"VAD for {audio_path.name}: Ratio {ratio:.2f} (Speech: {speech_duration_sec:.2f}s / Total: {total_duration_sec:.2f}s)")
        return ratio >= min_speech_ratio
    except Exception as e: log.warning(f"VAD: Error processing {audio_path.name}: {e}. Assuming active."); return True

def init_speaker_verification_model():
    global HAVE_SPEECHBRAIN
    if not HAVE_SPEECHBRAIN: log.warning("SpeechBrain import failed. SB verification skipped."); return None
    log.info("Initializing SpeechBrain SpeakerRecognition (spkrec-ecapa-voxceleb)...")
    if os.name == 'nt' and os.getenv('SPEECHBRAIN_FETCH_LOCAL_STRATEGY') != 'copy':
        log.warning("SPEECHBRAIN_FETCH_LOCAL_STRATEGY not 'copy'. May cause issues on Windows.")
    try:
        if DEVICE.type == "cuda": torch.cuda.empty_cache()
        model_source = "speechbrain/spkrec-ecapa-voxceleb"
        user_cache_dir = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache"))
        savedir = user_cache_dir / "voice_extractor_sb_cache" / model_source.replace("/", "_")
        ensure_dir_exists(savedir)
        model = SpeakerRecognition.from_hparams(source=model_source, savedir=str(savedir), run_opts={"device": DEVICE.type})
        model.eval()
        log.info(f"[green]✓ SpeechBrain model '{model_source}' loaded to {DEVICE.type.upper()}.[/]")
        return model
    except Exception as e:
        log.error(f"Failed to load SpeechBrain model: {e}")
        HAVE_SPEECHBRAIN = False; return None

def verify_speaker_segment(
    segment_audio_path: Path, reference_audio_path: Path, sb_verification_model
) -> tuple[float, dict]:
    scores = {"speechbrain": 0.0, "resemblyzer": 0.0, "voice_activity_factor": 0.1}
    seg_name = segment_audio_path.name

    # For SpeechBrain, it will load the audio. We assume it handles resampling if needed.
    # Reference audio is already processed to 16kHz mono.
    if sb_verification_model and HAVE_SPEECHBRAIN:
        try:
            # SpeechBrain's verify_files loads audio, so it should handle the current SR of segment_audio_path
            # (which is now output_sample_rate) and resample to its model's needs (typically 16kHz).
            ref_p, seg_p = reference_audio_path.resolve(strict=True).as_posix(), segment_audio_path.resolve(strict=True).as_posix()
            score_t, _ = sb_verification_model.verify_files(ref_p, seg_p)
            scores["speechbrain"] = score_t.item()
            log.debug(f"SpeechBrain score for {seg_name}: {scores['speechbrain']:.3f}")
        except Exception as e: log.warning(f"SpeechBrain verification failed for {seg_name}: {e}")

    # For Resemblyzer, preprocess_wav handles resampling to 16kHz.
    try:
        rz_encoder = VoiceEncoder(device="cuda" if DEVICE.type == "cuda" else "cpu")
        if not reference_audio_path.exists() or not segment_audio_path.exists(): raise FileNotFoundError("Ref/Seg audio missing for RZ.")
        ref_wav_rz = resemblyzer_preprocess_wav(reference_audio_path) # Already 16kHz
        seg_wav_rz = resemblyzer_preprocess_wav(segment_audio_path)   # Will be resampled to 16kHz if not already
        ref_emb_rz, seg_emb_rz = rz_encoder.embed_utterance(ref_wav_rz), rz_encoder.embed_utterance(seg_wav_rz)
        scores["resemblyzer"] = cos(ref_emb_rz, seg_emb_rz)
        log.debug(f"Resemblyzer score for {seg_name}: {scores['resemblyzer']:.3f}")
    except Exception as e: log.warning(f"Resemblyzer verification failed for {seg_name}: {e}")

    # check_voice_activity already loads audio at 16kHz for VAD.
    scores["voice_activity_factor"] = 1.0 if check_voice_activity(segment_audio_path) else 0.1

    final_score = (scores["speechbrain"] * 0.75 + scores["resemblyzer"] * 0.25) * scores["voice_activity_factor"] \
        if sb_verification_model and HAVE_SPEECHBRAIN and scores["speechbrain"] > 0.01 \
        else scores["resemblyzer"] * scores["voice_activity_factor"]
    log.debug(f"Final combined score for {seg_name}: {final_score:.3f}, Details: {scores}")
    return final_score, scores

def get_target_solo_timeline(
    diarization_annotation: Annotation, identified_target_label: str, overlap_timeline: Timeline
) -> Timeline:
    if not identified_target_label or identified_target_label not in diarization_annotation.labels():
        log.warning(f"Target label '{identified_target_label}' not in diarization. Cannot extract solo timeline.")
        return Timeline()
    target_speaker_timeline = diarization_annotation.label_timeline(identified_target_label)
    if not target_speaker_timeline:
        log.info(f"No speech segments for target '{identified_target_label}' in diarization.")
        return Timeline()

    final_solo_timeline = target_speaker_timeline.support().extrude(overlap_timeline.support())
    return final_solo_timeline

def slice_and_verify_target_solo_segments(
    diarization_annotation: Annotation, identified_target_label: str, overlap_timeline: Timeline,
    source_audio_file: Path, processed_reference_file: Path, target_name: str,
    output_segments_base_dir: Path,
    tmp_dir: Path, verification_threshold: float,
    min_segment_duration: float, max_merge_gap_val: float,
    output_sample_rate: int = 48000, output_channels: int = 1
) -> tuple[list[Path], list[Path]]:
    log.info(f"Refining and processing SOLO segments for '{target_name}' (label: {identified_target_label}).")
    target_solo_speech_timeline = get_target_solo_timeline(diarization_annotation, identified_target_label, overlap_timeline)
    if not target_solo_speech_timeline:
        log.warning(f"No solo speech for '{target_name}' after excluding overlaps. Skipping extraction.")
        return [], []
    log.info(f"Initial solo timeline for '{target_name}' (post-overlap subtraction) has {len(list(target_solo_speech_timeline))} sub-segments, duration: {format_duration(target_solo_speech_timeline.duration())}.")

    merged_target_solo_segments = merge_nearby_segments(list(target_solo_speech_timeline), max_merge_gap_val)
    log.info(f"After merging nearby solo sub-segments (gap <= {max_merge_gap_val}s): {len(merged_target_solo_segments)} segments.")
    duration_filtered_target_solo_segments = filter_segments_by_duration(merged_target_solo_segments, min_segment_duration)
    log.info(f"After duration filtering (>= {min_segment_duration}s): {len(duration_filtered_target_solo_segments)} final solo segments.")
    if not duration_filtered_target_solo_segments:
        log.warning(f"No solo segments for '{target_name}' after merging/duration filtering. Skipping.")
        return [], []

    solo_segments_dir = output_segments_base_dir / f"{safe_filename(target_name)}_solo_verified"
    rejected_segments_dir = output_segments_base_dir / f"{safe_filename(target_name)}_solo_rejected_for_review"
    ensure_dir_exists(solo_segments_dir)
    ensure_dir_exists(rejected_segments_dir)

    tmp_extracted_segments_dir = tmp_dir / f"__tmp_segments_solo_{safe_filename(target_name)}"
    ensure_dir_exists(tmp_extracted_segments_dir); [f.unlink() for f in tmp_extracted_segments_dir.glob("*.wav") if f.is_file()]

    sb_model = init_speaker_verification_model()
    all_sliced_solo_paths, segment_verification_scores_map = [], {}

    log.info(f"Slicing {len(duration_filtered_target_solo_segments)} candidate solo segments from '{source_audio_file.name}' at {output_sample_rate}Hz, {output_channels}ch...")
    with Progress(*Progress.get_default_columns(), console=console, transient=True) as pb_slice:
        task_slice = pb_slice.add_task("Slicing solo segments...", total=len(duration_filtered_target_solo_segments))
        for i, seg_obj in enumerate(duration_filtered_target_solo_segments):
            s_str, e_str = f"{seg_obj.start:.3f}".replace('.','p'), f"{seg_obj.end:.3f}".replace('.','p')
            base_seg_name = f"solo_seg_{i:04d}_{s_str}s_to_{e_str}s"
            tmp_seg_path = tmp_extracted_segments_dir / f"{base_seg_name}.wav"
            try:
                ff_slice(source_audio_file, tmp_seg_path, seg_obj.start, seg_obj.end,
                         target_sr=output_sample_rate, target_ac=output_channels)
                if tmp_seg_path.exists() and tmp_seg_path.stat().st_size > 0: all_sliced_solo_paths.append(tmp_seg_path)
                else: log.warning(f"Failed to create/empty slice: {tmp_seg_path.name}")
            except Exception as e: log.error(f"Failed to slice {tmp_seg_path.name}: {e}. Skipping.")
            pb_slice.update(task_slice, advance=1)

    if not all_sliced_solo_paths:
        log.warning("No solo segments successfully sliced. Skipping verification.");
        if sb_model and HAVE_SPEECHBRAIN and DEVICE.type == "cuda": torch.cuda.empty_cache();
        return [], []

    log.info(f"Verifying identity in {len(all_sliced_solo_paths)} sliced solo segments...")
    with Progress(*Progress.get_default_columns(), console=console, transient=True) as pb_verify:
        task_verify = pb_verify.add_task(f"Verifying '{target_name}' (solo)...", total=len(all_sliced_solo_paths))
        for tmp_path in all_sliced_solo_paths:
            final_score, _ = verify_speaker_segment(tmp_path, processed_reference_file, sb_model)
            segment_verification_scores_map[str(tmp_path)] = final_score
            pb_verify.update(task_verify, advance=1)
    if sb_model and HAVE_SPEECHBRAIN: del sb_model; log.debug("SB verification model cleared.")
    if DEVICE.type == "cuda": torch.cuda.empty_cache()

    num_accepted, num_rejected = plot_verification_scores({Path(k).name: v for k,v in segment_verification_scores_map.items()}, verification_threshold, output_segments_base_dir, target_name, plot_title_prefix=f"{safe_filename(target_name)}_SOLO_Verification_Scores")

    final_verified_solo_paths = []
    final_rejected_solo_paths = []
    log.info(f"Finalizing {num_accepted} verified solo segments (thresh: {verification_threshold:.2f}). Rejected: {num_rejected}.")
    with Progress(*Progress.get_default_columns(), console=console, transient=True) as pb_finalize:
        task_finalize = pb_finalize.add_task("Finalizing solo segments...", total=len(all_sliced_solo_paths))
        for tmp_path in all_sliced_solo_paths: # tmp_path is now at output_sample_rate and output_channels
            score = segment_verification_scores_map.get(str(tmp_path), 0.0)
            if score >= verification_threshold: # If ACCEPTED
                final_seg_path = solo_segments_dir / tmp_path.name
                # File is already at correct SR/channels from ff_slice, just move it.
                shutil.move(str(tmp_path), str(final_seg_path)) if tmp_path.exists() else None
                final_verified_solo_paths.append(final_seg_path)
            else: # If REJECTED
                if tmp_path.exists():
                    # File is already at correct SR/channels.
                    rejected_filename = f"{tmp_path.stem}_score_{score:.3f}.wav"
                    rejected_seg_path = rejected_segments_dir / rejected_filename
                    shutil.move(str(tmp_path), str(rejected_seg_path))
                    final_rejected_solo_paths.append(rejected_seg_path)
            pb_finalize.update(task_finalize, advance=1)

    if tmp_extracted_segments_dir.exists():
        try: shutil.rmtree(tmp_extracted_segments_dir)
        except OSError as e: log.warning(f"Could not remove tmp solo segments dir {tmp_extracted_segments_dir}: {e}")
    log.info(f"[green]✓ Extracted and verified {len(final_verified_solo_paths)} solo segments for '{target_name}'.[/]")
    if num_rejected > 0:
        log.info(f"  Rejected {num_rejected} segments saved for review in: {rejected_segments_dir}")
    return final_verified_solo_paths, final_rejected_solo_paths

def transcribe_segments(
    segment_paths: list[Path], output_transcripts_dir: Path, target_name: str,
    segment_type_tag: str, whisper_model_name: str = "large-v3", language: str = "en"
):
    if not segment_paths: log.info(f"No '{segment_type_tag}' segments for '{target_name}' to transcribe."); return
    log.info(f"Transcribing {len(segment_paths)} '{segment_type_tag}' segments for '{target_name}' using Whisper model '{whisper_model_name}'...")
    if DEVICE.type == "cuda": torch.cuda.empty_cache()
    ensure_dir_exists(output_transcripts_dir)
    try:
        log.info(f"Loading Whisper model '{whisper_model_name}' to {DEVICE.type.upper()}...")
        model = whisper.load_model(whisper_model_name, device=DEVICE)
        log.info(f"Whisper model '{whisper_model_name}' loaded.")
    except Exception as e: log.error(f"Failed to load Whisper model '{whisper_model_name}': {e}. Transcription skipped."); return

    transcription_data_for_csv, plain_text_transcript_lines = [], []
    file_prefix = f"{safe_filename(target_name)}_{safe_filename(segment_type_tag)}"
    csv_path, txt_path = output_transcripts_dir/f"{file_prefix}_trans.csv", output_transcripts_dir/f"{file_prefix}_trans.txt"

    time_pattern = re.compile(r"(\d+p\d+s_to_\d+p\d+)s")

    def get_sort_key_time(p: Path):
        try:
            match = time_pattern.search(p.stem)
            if match:
                time_part_str = match.group(1)
                start_time_str = time_part_str.split('s_to_')[0]
                return float(start_time_str.replace('p', '.'))
            return 0.0
        except: return 0.0
    sorted_segment_paths = sorted(segment_paths, key=get_sort_key_time)

    with Progress(*Progress.get_default_columns(), console=console, transient=True) as pb:
        task = pb.add_task(f"Whisper ({target_name}, {segment_type_tag})...", total=len(sorted_segment_paths))
        for wav_file in sorted_segment_paths:
            if not wav_file.exists() or wav_file.stat().st_size == 0: log.warning(f"Skipping missing/empty: {wav_file.name}"); pb.update(task, advance=1); continue
            text = "[TRANSCRIPTION ERROR]"; s_time, e_time = 0.0, 0.0
            try:
                match = time_pattern.search(wav_file.stem)
                if match:
                    time_part_str = match.group(1)
                    s_time_str_part, e_time_str_part_full = time_part_str.split('s_to_')
                    s_time = float(s_time_str_part.replace('p','.'))
                    e_time = float(e_time_str_part_full.replace('p','.'))
                else:
                    log.warning(f"Could not parse s_time/e_time from filename '{wav_file.name}' for transcript metadata. Will be 0.0.")

                opts = {"fp16": DEVICE.type=="cuda"}
                if language and language.lower() != "auto": opts["language"] = language
                result = model.transcribe(str(wav_file), **opts)
                text = result["text"].strip()
            except Exception as e:
                log.error(f"Error transcribing {wav_file.name}: {e}")
                s_time, e_time = 0.0, 0.0

            duration = librosa.get_duration(path=wav_file)
            transcription_data_for_csv.append([f"{s_time:.3f}", f"{e_time:.3f}", f"{duration:.3f}", wav_file.name, text])
            plain_text_transcript_lines.append(f"[{format_duration(s_time)} - {format_duration(e_time)}] {wav_file.name} (Dur: {duration:.2f}s):\n{text}\n---")
            pb.update(task, advance=1)

    if transcription_data_for_csv:
        try:
            with csv_path.open("w", newline='', encoding="utf-8") as f:
                writer = csv.writer(f); writer.writerow(["original_start_s", "original_end_s", "seg_duration_s", "filename", "transcript"]); writer.writerows(transcription_data_for_csv)
            log.info(f"Saved {len(transcription_data_for_csv)} transcripts to CSV: {csv_path.name}")
            txt_path.write_text("\n".join(plain_text_transcript_lines), encoding="utf-8")
            log.info(f"Saved transcripts to TXT: {txt_path.name}")
        except Exception as e: log.error(f"Failed to save transcripts: {e}")
    if DEVICE.type == "cuda": torch.cuda.empty_cache()
    log.info(f"[green]✓ Transcription completed for '{target_name}' ({segment_type_tag}).[/]")

def concatenate_segments(
    audio_segment_paths: list[Path], destination_concatenated_file: Path, tmp_dir_concat: Path,
    silence_duration: float = 0.5, output_sr_concat: int = 48000, output_channels_concat: int = 1
) -> bool:
    if not audio_segment_paths: log.warning(f"No segments to concat for {destination_concatenated_file.name}."); return False
    ensure_dir_exists(tmp_dir_concat); ensure_dir_exists(destination_concatenated_file.parent)

    time_pattern_concat = re.compile(r"(\d+p\d+s)_to_")

    def get_sort_key_concat(p: Path):
        try:
            match = time_pattern_concat.search(p.stem)
            if match:
                start_time_str = match.group(1)
                return float(start_time_str.replace('p', '.').removesuffix('s'))
            return 0.0
        except: return 0.0
    sorted_audio_paths = sorted(audio_segment_paths, key=get_sort_key_concat)

    silence_file = tmp_dir_concat/f"silence_{silence_duration}s_{output_sr_concat}hz_{output_channels_concat}ch.wav"
    if silence_duration > 0:
        try:
            if not silence_file.exists() or silence_file.stat().st_size == 0:
                channel_layout_str = 'mono' if output_channels_concat == 1 else 'stereo'
                anullsrc_description = f"anullsrc=channel_layout={channel_layout_str}:sample_rate={output_sr_concat}"
                (ffmpeg
                    .input(anullsrc_description, format='lavfi', t=str(silence_duration))
                    .output(str(silence_file), acodec='pcm_s16le', ar=str(output_sr_concat), ac=output_channels_concat)
                    .overwrite_output()
                    .run(quiet=True, capture_stdout=True, capture_stderr=True))
        except ffmpeg.Error as e: log.error(f"ffmpeg failed to create silence file: {e.stderr.decode(errors='ignore') if e.stderr else 'Err'}"); return False

    list_file = tmp_dir_concat / f"{destination_concatenated_file.stem}_concat_list.txt"
    lines = []
    valid_count = 0
    for i, audio_path in enumerate(sorted_audio_paths):
        if not audio_path.exists() or audio_path.stat().st_size == 0: log.warning(f"Segment {audio_path.name} for concat missing/empty. Skipping."); continue
        if i > 0 and silence_duration > 0 and silence_file.exists(): lines.append(f"file '{silence_file.resolve().as_posix()}'")
        lines.append(f"file '{audio_path.resolve().as_posix()}'"); valid_count += 1

    if valid_count == 0: log.warning(f"No valid segments to concat for {destination_concatenated_file.name}."); return False
    if valid_count == 1 and silence_duration == 0:
        single_path = next((p for p in sorted_audio_paths if p.exists() and p.stat().st_size > 0), None)
        if single_path:
            log.info(f"One segment to 'concat'. Copy/Re-encode {single_path.name} to {destination_concatenated_file.name}")
            try: (ffmpeg.input(str(single_path)).output(str(destination_concatenated_file), acodec='pcm_s16le', ar=output_sr_concat, ac=output_channels_concat).overwrite_output().run(quiet=True)); return True
            except ffmpeg.Error as e: log.error(f"ffmpeg single segment copy failed: {e.stderr.decode(errors='ignore') if e.stderr else 'Err'}"); return False
        else: return False

    try: list_file.write_text("\n".join(lines), encoding="utf-8")
    except Exception as e: log.error(f"Failed to write ffmpeg concat list {list_file.name}: {e}"); return False

    log.info(f"Concatenating {valid_count} segments to: {destination_concatenated_file.name}...")
    try:
        (ffmpeg.input(str(list_file), format="concat", safe=0)
               .output(str(destination_concatenated_file), acodec="pcm_s16le", ar=output_sr_concat, ac=output_channels_concat)
               .overwrite_output().run(quiet=True, capture_stdout=True, capture_stderr=True))
        log.info(f"[green]✓ Successfully concatenated segments to: {destination_concatenated_file.name}[/]")
        return True
    except ffmpeg.Error as e:
        log.error(f"ffmpeg concat failed for {destination_concatenated_file.name}: {e.stderr.decode(errors='ignore') if e.stderr else 'Err'}")
        log.debug(f"Concat list ({list_file.name}):\n" + "\n".join(lines)); return False
    finally:
        if list_file.exists(): list_file.unlink(missing_ok=True)
        if silence_duration > 0 and silence_file.exists(): silence_file.unlink(missing_ok=True)

if __name__ == '__main__':
    log.info("audio_pipeline.py executed directly (likely for testing individual functions).")
