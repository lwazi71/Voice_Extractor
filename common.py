#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
common.py
Shared configurations, constants, utilities, logging, device management,
dependency bootstrapping, and plotting for the Voice Extractor.
"""

import subprocess
import sys
import importlib
import importlib.metadata as md
import csv
import os
import getpass
from pathlib import Path
import numpy as np
import logging
import shutil

# --- Dependency Management ---
REQ = [ # These will be checked against the versions in requirements.txt if specified there
    "rich", "ffmpeg-python", "soundfile", "numpy",
    "torch>=2.7.0", "torchaudio>=2.7.0", "torchvision>=0.22.0",
    "pyannote.audio>=3.3.2", "resemblyzer",
    "openai-whisper>=20240930",
    "demucs>=4.0.1", "matplotlib", "librosa", "speechbrain>=1.0.0",
    "torchcrepe>=0.0.21", "silero-vad>=5.1.2"
]

def _ensure(pkgs):
    """Ensures specified packages are installed, installing them if not."""
    installed_pkgs_details = []
    missing_pkgs_to_install = []

    for spec in pkgs:
        name_parts_spec = spec.split("==") if "==" in spec else \
                          spec.split(">=") if ">=" in spec else \
                          spec.split("<=") if "<=" in spec else \
                          spec.split("!=") if "!=" in spec else \
                          spec.split("~=") if "~=" in spec else [spec]
        name = name_parts_spec[0]

        try:
            # Get the raw version string
            raw_version = md.version(name)
            
            # Clean the version string: remove build metadata like +cu121
            # Build metadata is separated by a '+'
            cleaned_version = raw_version.split('+')[0]
            
            installed_pkgs_details.append(f"  - Found {name} (version {raw_version}) -> Cleaned for check: {cleaned_version}")

            # Basic check for version compatibility if specified
            if len(name_parts_spec) > 1:
                req_op = ""
                req_ver_str = name_parts_spec[1] # This is the version from your REQ list
                for op in [">=", "<=", "==", "!=", "~="]:
                    if op in spec:
                        req_op = op
                        break
                
                # Ensure version strings are comparable (e.g., "2.7.0", not "2.7")
                current_ver_parts = cleaned_version.split('.')
                req_ver_parts = req_ver_str.split('.')

                # Pad with zeros if necessary for consistent comparison length (e.g. 2.7 vs 2.7.0)
                max_len = max(len(current_ver_parts), len(req_ver_parts))
                current_ver_parts.extend(['0'] * (max_len - len(current_ver_parts)))
                req_ver_parts.extend(['0'] * (max_len - len(req_ver_parts)))

                current_ver_tuple = tuple(map(int, current_ver_parts))
                req_ver_tuple = tuple(map(int, req_ver_parts))

                compatible = True
                if req_op == ">=":
                    compatible = current_ver_tuple >= req_ver_tuple
                elif req_op == "<=":
                    compatible = current_ver_tuple <= req_ver_tuple
                elif req_op == "==": # Exact match on major.minor.patch (after cleaning)
                    compatible = current_ver_tuple == req_ver_tuple
                # More complex ops like ~= and != might need more sophisticated parsing or a library
                # For now, we focus on the common ones. Pip will handle the ultimate check.

                if not compatible:
                    print(f"[Setup] Package '{name}' version {raw_version} (cleaned: {cleaned_version}) does not meet requirement {spec}. Queuing for update/reinstall.")
                    missing_pkgs_to_install.append(spec)
        except md.PackageNotFoundError:
            print(f"[Setup] Package '{name}' (from spec: {spec}) not found. Queuing for installation.")
            missing_pkgs_to_install.append(spec)
        except ValueError as ve: # Catch errors during int conversion of version parts
            print(f"[Setup] Warning: Could not parse version string for {name} ('{raw_version}' -> '{cleaned_version}'): {ve}. Will rely on pip for this package.")
            # If parsing fails even after cleaning, it's safer to just try to install/upgrade it via pip.
            if spec not in missing_pkgs_to_install: # Avoid adding duplicates
                 missing_pkgs_to_install.append(spec)


    if installed_pkgs_details and not missing_pkgs_to_install:
        print("[Setup] Checked existing packages:")
        for detail in installed_pkgs_details:
            print(detail)

    if missing_pkgs_to_install:
        unique_missing_pkgs = sorted(list(set(missing_pkgs_to_install))) # Ensure unique and sorted
        print(f"[Setup] Attempting to install/update {len(unique_missing_pkgs)} package(s): {', '.join(unique_missing_pkgs)}")
        # Add --upgrade to ensure versions are met if package exists but is old
        # Using --force-reinstall can be risky but sometimes helps with corrupted installs or ensuring exact versions.
        # For now, --upgrade should suffice.
        install_command = [sys.executable, "-m", "pip", "install", "--upgrade"] + unique_missing_pkgs
        try:
            subprocess.check_call(install_command)
            print(f"[Setup] Successfully installed/updated {', '.join(unique_missing_pkgs)}")
        except subprocess.CalledProcessError as e:
            print(f"[Setup] ERROR: Failed to install/update packages. Error: {e}")
            print(f"Command was: {' '.join(install_command)}")
            print("Please try installing them manually. Exiting.")
            sys.exit(1)
        print("[Setup] All required packages should now be installed/updated to specified versions.")
    elif not installed_pkgs_details and not missing_pkgs_to_install: # First run
        print("[Setup] No pre-existing packages detected. Attempting to install all requirements...")
        install_command = [sys.executable, "-m", "pip", "install", "--upgrade"] + pkgs
        try:
            subprocess.check_call(install_command)
            print(f"[Setup] Successfully installed all required packages.")
        except subprocess.CalledProcessError as e:
            print(f"[Setup] ERROR: Failed to install packages. Error: {e}")
            print(f"Command was: {' '.join(install_command)}")
            print("Please try installing them manually. Exiting.")
            sys.exit(1)
    else:
        print("[Setup] All required packages are already installed and meet basic version checks.")

# --- End Dependency Management ---


# Import external libraries after _ensure can run
import torch
import ffmpeg
import soundfile as sf
from rich.console import Console
from rich.logging import RichHandler
from rich.prompt import Prompt, Confirm
import matplotlib
matplotlib.use("Agg") # Use non-interactive backend for plotting
import matplotlib.pyplot as plt
import librosa
import librosa.display

# Module-level variables for global access
torchaudio_version = "N/A"
torchvision_version = "N/A"

try:
    import torchaudio
    torchaudio_version = torchaudio.__version__
except ImportError:
    pass
try:
    import torchvision
    torchvision_version = torchvision.__version__
except ImportError:
    pass


# --- Rich Console & Logging ---
console = Console(width=120)
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False, console=console)]
)
log = logging.getLogger("voice_extractor")


# --- Configuration Constants ---
DEFAULT_OUTPUT_BASE_DIR = Path("./output_runs")
SPECTROGRAM_SEC    = 60
SPEC_FIGSIZE       = (19.2, 6.0)
SPEC_DPI           = 150
HIGH_RES_NFFT      = 4096
FREQ_MAX           = 12000

DEFAULT_MIN_SEGMENT_SEC    = 1.0
DEFAULT_MAX_MERGE_GAP      = 0.25
DEFAULT_VERIFICATION_THRESHOLD = 0.69

SPEECH_BANDS = [
    (0, 300, "Sub-bass / Rumble", "lightgray"),
    (300, 1000, "Vowels & Bass / Warmth", "lightblue"),
    (1000, 3000, "Speech Formants / Clarity", "lightyellow"),
    (3000, 5000, "Consonants & Sibilance / Presence", "lightgreen"),
    (5000, 8000, "Brightness & Harmonics", "lightpink"),
    (8000, 12000, "Air & Detail", "lavender")
]

# --- Device Setup (CPU/CUDA) ---
def check_cuda():
    if not torch.cuda.is_available():
        log.warning("CUDA is not available. Processing will run on CPU and may be significantly slower.")
        return torch.device("cpu")
    try:
        _ = torch.zeros(1).cuda()
        device_count = torch.cuda.device_count()
        log.info(f"[bold green]✓ CUDA is available.[/]")
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            total_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            log.info(f"  - Device {i}: {device_name} (Total Memory: {total_mem:.2f} GB)")
        cuda_version = torch.version.cuda if hasattr(torch.version, 'cuda') and torch.version.cuda else "N/A"
        cudnn_version = torch.backends.cudnn.version() if hasattr(torch.backends, 'cudnn') and torch.backends.cudnn.is_available() else "N/A"
        log.info(f"  - PyTorch CUDA version: {cuda_version}")
        log.info(f"  - PyTorch cuDNN version: {cudnn_version}")
        torch.cuda.empty_cache()
        return torch.device("cuda")
    except Exception as e:
        log.error(f"[bold red]CUDA initialization failed: {e}[/]")
        log.warning("Falling back to CPU. This will be very slow.")
        return torch.device("cpu")

DEVICE = check_cuda()


# --- General Utility Functions ---
def to_mono(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=1).astype(np.float32) if x.ndim > 1 else x.astype(np.float32)

def cos(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)

def ff_trim(src_path: Path, dst_path: Path, start_time: float, end_time: float, target_sr: int = 16000, target_ac: int = 1):
    try:
        (
            ffmpeg.input(str(src_path), ss=start_time, to=end_time)
            .output(str(dst_path), acodec="pcm_s16le", ac=target_ac, ar=target_sr)
            .overwrite_output()
            .run(quiet=True, capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        log.error(f"ffmpeg trim failed for {dst_path.name}: {e.stderr.decode('utf8') if e.stderr else 'Unknown error'}")
        raise

def ff_slice(src_path: Path, dst_path: Path, start_time: float, end_time: float, target_sr: int, target_ac: int = 1):
    try:
        (
            ffmpeg.input(str(src_path), ss=start_time, to=end_time)
            .output(str(dst_path), acodec="pcm_s16le", ar=target_sr, ac=target_ac)
            .overwrite_output()
            .run(quiet=True, capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        log.error(f"ffmpeg slice failed for {dst_path.name}: {e.stderr.decode('utf8') if e.stderr else 'Unknown error'}")
        raise

def get_huggingface_token(token_arg: str = None) -> str:
    if token_arg:
        log.info("Using HuggingFace token from command-line argument.")
        return token_arg
    token = os.getenv("HUGGINGFACE_TOKEN")
    if token:
        log.info("Using HuggingFace token from HUGGINGFACE_TOKEN environment variable.")
        return token
    console.print("\n[bold yellow]HuggingFace User Access Token is required for PyAnnote models.[/]")
    console.print("You can create a token at: [link=https://huggingface.co/settings/tokens]https://huggingface.co/settings/tokens[/link] (read permissions are sufficient).")
    try:
        token = Prompt.ask("Enter your HuggingFace token (will not be displayed)", password=True)
    except Exception:
        token = getpass.getpass("Enter your HuggingFace token (input hidden): ")
    if not token or not token.strip():
        log.error("[bold red]No HuggingFace token provided. Exiting.[/]")
        sys.exit(1)
    token = token.strip()
    log.info("HuggingFace token provided by user.")
    try:
        if Confirm.ask("Save this token as environment variable HUGGINGFACE_TOKEN for future use? (Recommended)", default=True):
            env_var_name = "HUGGINGFACE_TOKEN"
            if os.name == "nt":
                try:
                    subprocess.run(["setx", env_var_name, token], check=True, capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
                    console.print(f"[green]Token saved as User environment variable '{env_var_name}'. You may need to restart your terminal/IDE for it to take effect.[/green]")
                except FileNotFoundError:
                    console.print(f"[yellow]'setx' command not found. Please set the environment variable '{env_var_name}' manually.[/yellow]")
                except subprocess.CalledProcessError as e:
                    console.print(f"[red]Failed to save token with 'setx': {e.stderr.decode(errors='ignore') if e.stderr else e}. Please set it manually.[/red]")
            else:
                shell_name = Path(os.environ.get("SHELL", "")).name
                rc_files = {"bash": "~/.bashrc", "zsh": "~/.zshrc", "fish": "~/.config/fish/config.fish"}
                shell_file_path_str = rc_files.get(shell_name)
                if not shell_file_path_str:
                    console.print(f"[yellow]Could not determine shell RC file for '{shell_name}'. Please add/update HUGGINGFACE_TOKEN in your shell's startup file manually.[/yellow]")
                else:
                    shell_file_path = Path(shell_file_path_str).expanduser()
                    try:
                        if shell_file_path.exists() and f'export {env_var_name}=' in shell_file_path.read_text():
                             console.print(f"[yellow]HUGGINGFACE_TOKEN already seems to be set in {shell_file_path}. Please update it manually if needed.[/yellow]")
                        else:
                            with open(shell_file_path, "a") as f:
                                f.write(f'\n# Added by Voice Extractor\nexport {env_var_name}="{token}"\n')
                            console.print(f"[green]Token appended to {shell_file_path}. Please restart your terminal or run 'source {shell_file_path_str}' to apply.[/green]")
                    except Exception as e:
                        console.print(f"[red]Failed to write to {shell_file_path}: {e}. Please set HUGGINGFACE_TOKEN manually.[/red]")
    except Exception as e:
        log.warning(f"Could not prompt to save token due to an interactive console issue: {e}. Please set HUGGINGFACE_TOKEN environment variable manually if desired.")
    return token

def format_duration(seconds: float) -> str:
    ms = int((seconds - int(seconds)) * 1000)
    s = int(seconds)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

def safe_filename(name: str, max_length: int = 200) -> str:
    import re
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '', name)
    name = name.replace(' ', '_')
    if len(name) > max_length:
        name = name[:max_length]
    return name if name else "unnamed_file"

def ensure_dir_exists(dir_path: Path):
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        log.error(f"Failed to create directory {dir_path}: {e}")
        raise

# --- Plotting Functions ---
def save_detailed_spectrograms(
    wav_path: Path, output_dir: Path, title_prefix: str, target_name: str = "TargetSpeaker",
    sample_sec: float = SPECTROGRAM_SEC, segments_to_mark: list = None, overlap_timeline = None
):
    ensure_dir_exists(output_dir)
    safe_prefix = safe_filename(title_prefix)
    if not wav_path.exists():
        log.warning(f"Audio file {wav_path} not found for spectrogram '{safe_prefix}'. Skipping.")
        return
    try:
        y, sr = librosa.load(wav_path, sr=None, mono=True, duration=sample_sec)
    except Exception as e:
        log.error(f"Failed to load audio {wav_path} for spectrogram '{safe_prefix}': {e}")
        return
    if len(y) == 0:
        log.warning(f"Audio file {wav_path} is empty. Cannot generate spectrogram '{safe_prefix}'.")
        return
    duration = librosa.get_duration(y=y, sr=sr)
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=SPEC_FIGSIZE)
    try:
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=HIGH_RES_NFFT, hop_length=HIGH_RES_NFFT // 4)), ref=np.max)
        img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='linear', hop_length=HIGH_RES_NFFT // 4, cmap='magma', ax=ax)
        ax.set_ylim(0, FREQ_MAX)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        for low, high, label, color in SPEECH_BANDS:
            if high <= FREQ_MAX:
                ax.axhspan(low, high, color=color, alpha=0.15, ec='none')
                if duration > 0:
                    ax.text(duration * 0.02, (low + high) / 2, label, verticalalignment='center', fontsize=7, color='black', bbox=dict(facecolor=color, alpha=0.3, boxstyle='round,pad=0.2'))
        
        overlap_legend_added_spec = False
        if overlap_timeline:
            for segment in overlap_timeline:
                if segment.start > duration: continue
                plot_start, plot_end = max(0, segment.start), min(duration, segment.end)
                if plot_end > plot_start:
                    ax.axvspan(plot_start, plot_end, color='gray', alpha=0.4, label='Overlap Region' if not overlap_legend_added_spec else None)
                    if not overlap_legend_added_spec: overlap_legend_added_spec = True
        
        unique_labels_plotted_spec = set()
        if segments_to_mark:
            cmap_segments = plt.cm.get_cmap('viridis', len(segments_to_mark) if len(segments_to_mark) > 0 else 1)
            for i, (start, end, label) in enumerate(segments_to_mark):
                if start > duration or end > duration or start >= end: continue
                is_target_segment_for_color = target_name.lower() in label.lower() if target_name else False
                color_val = 'orange' if is_target_segment_for_color else cmap_segments(i / len(segments_to_mark) if len(segments_to_mark) > 1 else 0.5)
                alpha_val = 0.5 if is_target_segment_for_color else 0.3
                label_for_legend = label if label not in unique_labels_plotted_spec else None
                ax.axvspan(start, end, color=color_val, alpha=alpha_val, label=label_for_legend)
                if label_for_legend: unique_labels_plotted_spec.add(label)
                ax.text(start + (end - start) / 2, FREQ_MAX * 0.95, label, horizontalalignment='center', color=color_val, fontsize=8, bbox=dict(facecolor='white', alpha=0.7, pad=0.2))
        
        plot_title = f"{target_name} - {title_prefix} (Sample: {duration:.1f}s, Max Freq: {FREQ_MAX/1000:.1f}kHz)"
        ax.set_title(plot_title, fontsize=12)
        fig.colorbar(img, ax=ax, format='%+2.0f dB', label='Amplitude (dB)')
        if ax.has_data() and (unique_labels_plotted_spec or overlap_legend_added_spec): ax.legend(loc="upper right", fontsize=8)
        plt.tight_layout()
        out_path = output_dir / f"{safe_filename(target_name)}_{safe_prefix}_linear_hires.png"
        plt.savefig(out_path, dpi=SPEC_DPI)
        log.info(f"Saved detailed spectrogram: {out_path.name}")
    except Exception as e:
        log.error(f"Error generating detailed spectrogram for {safe_prefix}: {e}")
        if args_for_debug_plotting and args_for_debug_plotting.debug: log.exception("Traceback for detailed spectrogram error:")
    finally:
        plt.close(fig)

def create_comparison_spectrograms(
    files_to_compare: list, output_dir: Path, target_name: str,
    main_prefix: str = "Audio_Stages_Comparison", sample_sec: float = SPECTROGRAM_SEC,
    overlap_timeline_dict: dict = None
):
    ensure_dir_exists(output_dir)
    if not files_to_compare: log.warning("No files provided for spectrogram comparison."); return
    valid_files = [(Path(fp) if isinstance(fp, str) else fp, title) for fp, title in files_to_compare if (Path(fp) if isinstance(fp, str) else fp) and (Path(fp) if isinstance(fp, str) else fp).exists() and (Path(fp) if isinstance(fp, str) else fp).stat().st_size > 0]
    if not valid_files: log.warning("No valid files for spectrogram comparison after checking."); return
    
    n_files = len(valid_files)
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axs = plt.subplots(n_files, 1, figsize=(19.2, 4.0 * n_files if n_files > 0 else 4.0), sharex=True, sharey=True, layout="constrained", squeeze=False)
    axs = axs.flatten()
    common_sr = None
    img_ref_for_colorbar = None # To store one of the specshow images for colorbar

    for i, (file_path, title) in enumerate(valid_files):
        current_overlap_timeline = overlap_timeline_dict.get(str(file_path.resolve())) if overlap_timeline_dict else None
        try:
            y, sr_current = librosa.load(file_path, sr=None, mono=True, duration=sample_sec)
            if common_sr is None: common_sr = sr_current
            elif common_sr != sr_current: y = librosa.resample(y, orig_sr=sr_current, target_sr=common_sr)
            if len(y) == 0:
                axs[i].set_title(f"{title} (Empty Audio)", fontsize=10); axs[i].text(0.5, 0.5, "Empty Audio", ha='center', va='center', transform=axs[i].transAxes); continue
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=HIGH_RES_NFFT, hop_length=HIGH_RES_NFFT // 4)), ref=np.max)
            duration = librosa.get_duration(y=y, sr=common_sr)
            img = librosa.display.specshow(D, sr=common_sr, x_axis='time', y_axis='linear', hop_length=HIGH_RES_NFFT // 4, ax=axs[i], cmap='magma')
            if i == 0: img_ref_for_colorbar = img # Capture first valid image for colorbar
            axs[i].set_title(f"{title} ({duration:.1f}s sample, {common_sr/1000:.1f}kHz)", fontsize=10)
            axs[i].set_ylabel("Frequency (Hz)"); axs[i].set_ylim(0, FREQ_MAX)
            for low, high, band_label, color in SPEECH_BANDS:
                if high <= FREQ_MAX: axs[i].axhspan(low, high, color=color, alpha=0.1)
            if current_overlap_timeline:
                overlap_legend_added_comp = False
                for segment in current_overlap_timeline:
                    if segment.start > duration: continue
                    plot_start, plot_end = max(0, segment.start), min(duration, segment.end)
                    if plot_end > plot_start:
                        axs[i].axvspan(plot_start, plot_end, color='dimgray', alpha=0.35, label='Overlap Region' if not overlap_legend_added_comp else None)
                        if not overlap_legend_added_comp: overlap_legend_added_comp = True
        except Exception as e:
            log.error(f"Failed to process {file_path.name} for comparison spectrogram: {e}")
            axs[i].set_title(f"{title} (Error)", fontsize=10); axs[i].text(0.5, 0.5, "Error loading/processing", ha='center', va='center', transform=axs[i].transAxes, wrap=True)
    
    if n_files > 0: axs[-1].set_xlabel("Time (s)")
    fig.suptitle(f"{target_name} - {main_prefix}", fontsize=14, y=1.02 if n_files > 1 else 1.05)
    if img_ref_for_colorbar: fig.colorbar(img_ref_for_colorbar, ax=axs.tolist(), format='%+2.0f dB', label='Amplitude (dB)', orientation='vertical', aspect=max(15, 30*n_files), pad=0.01)
    
    out_path = output_dir / f"{safe_filename(target_name)}_{safe_filename(main_prefix)}.png"
    try:
        plt.savefig(out_path, dpi=SPEC_DPI, bbox_inches='tight')
        log.info(f"Saved comparison spectrogram: {out_path.name}")
    except Exception as e:
        log.error(f"Failed to save comparison spectrogram {out_path.name}: {e}")
    finally:
        plt.close(fig)

def create_diarization_plot(
    annotation, target_speaker_label: str, target_name: str, output_dir: Path,
    plot_title_prefix: str = "Diarization_Results", overlap_timeline = None
):
    ensure_dir_exists(output_dir)
    plt.style.use('seaborn-v0_8-darkgrid')
    num_labels_in_annotation = len(annotation.labels()) if annotation and annotation.labels() else 0
    plot_height = max(4, num_labels_in_annotation * 0.8 if num_labels_in_annotation > 0 else 1 * 0.8)
    fig, ax = plt.subplots(figsize=(20, plot_height))
    speakers = list(annotation.labels()) if annotation and annotation.labels() else []

    if not speakers:
        log.warning("No speaker labels found in annotation for diarization plot.")
        ax.text(0.5, 0.5, "No speaker segments found.", ha='center', va='center')
        ax.set_title(f"{target_name} - {plot_title_prefix} (No Speaker Data)", fontsize=12)
    else:
        sorted_speakers = sorted(speakers, key=lambda spk: (spk != target_speaker_label if target_speaker_label else True, spk))
        speaker_y_pos = {spk: i for i, spk in enumerate(sorted_speakers)}
        plot_colors = plt.cm.get_cmap('tab20', len(sorted_speakers) if len(sorted_speakers) > 0 else 1)
        max_time = 0
        if annotation:
            for segment_obj, _, _ in annotation.itertracks(yield_label=True):
                if segment_obj.end > max_time: max_time = segment_obj.end
        if overlap_timeline:
            for segment_obj in overlap_timeline:
                if segment_obj.end > max_time: max_time = segment_obj.end
        
        unique_legend_labels_spk = set()
        for i, spk_label_from_list in enumerate(sorted_speakers):
            segments_for_this_speaker = []
            if annotation:
                for segment, _, label_in_track in annotation.itertracks(yield_label=True):
                    if label_in_track == spk_label_from_list: segments_for_this_speaker.append((segment.start, segment.end))
            is_target = (spk_label_from_list == target_speaker_label)
            bar_color = 'crimson' if is_target else plot_colors(i % plot_colors.N)
            display_label_base = f"Target: {target_name} ({spk_label_from_list})" if is_target else f"Other Spk ({spk_label_from_list})"
            for seg_idx, (start, end) in enumerate(segments_for_this_speaker):
                legend_label_spk = display_label_base if display_label_base not in unique_legend_labels_spk else None
                ax.barh(y=speaker_y_pos[spk_label_from_list], width=end - start, left=start, height=0.7, color=bar_color, alpha=0.8 if is_target else 0.6, edgecolor='black' if is_target else bar_color, linewidth=0.5, label=legend_label_spk)
                if legend_label_spk: unique_legend_labels_spk.add(display_label_base)
        
        overlap_legend_added_plot = False
        if overlap_timeline:
            for seg_overlap in overlap_timeline:
                ax.axvspan(seg_overlap.start, seg_overlap.end, color='gray', alpha=0.3, label='Overlapped Speech' if not overlap_legend_added_plot else None, zorder=0) # zorder=0 to be behind speaker bars
                if not overlap_legend_added_plot: overlap_legend_added_plot = True
        
        ax.set_yticks(list(speaker_y_pos.values()))
        ax.set_yticklabels([f"{target_name} ({spk})" if spk == target_speaker_label else f"Speaker {spk}" for spk in sorted_speakers])
        ax.set_xlabel("Time (seconds)"); ax.set_ylabel("Speaker")
        title_suffix = f"(Target Label: {target_speaker_label})" if target_speaker_label and target_speaker_label in speakers else "(Target Not Found)"
        ax.set_title(f"{target_name} - {plot_title_prefix} {title_suffix}", fontsize=12)
        ax.set_xlim(0, max_time * 1.02 if max_time > 0 else 10)
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        if unique_legend_labels_spk or overlap_legend_added_plot: ax.legend(loc='upper right', fontsize=9)

    out_path = output_dir / f"{safe_filename(target_name)}_{safe_filename(plot_title_prefix)}.png"
    plt.tight_layout()
    try:
        plt.savefig(out_path, dpi=150)
        log.info(f"Saved diarization visualization: {out_path.name}")
    except Exception as e:
        log.error(f"Failed to save diarization plot {out_path.name}: {e}")
        if args_for_debug_plotting and args_for_debug_plotting.debug: log.exception("Traceback for diarization plot error:")
    finally:
        plt.close(fig)

def plot_verification_scores(
    scores_dict: dict, threshold: float, output_dir: Path, target_name: str,
    plot_title_prefix: str = "Verification_Scores"
):
    ensure_dir_exists(output_dir)
    if not scores_dict: log.warning("No verification scores provided to plot."); return 0, 0
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(max(14, len(scores_dict) * 0.4), 7)) # Adjusted width scaling
    sorted_scores_list = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
    segment_names = [Path(item[0]).name for item in sorted_scores_list]
    score_values = [item[1] for item in sorted_scores_list]
    bar_colors = ['mediumseagreen' if score >= threshold else 'lightcoral' for score in score_values]
    bars = ax.bar(range(len(segment_names)), score_values, color=bar_colors)
    ax.axhline(y=threshold, color='black', linestyle='--', linewidth=1.5, label=f'Verification Threshold ({threshold:.2f})')
    accepted_count = sum(1 for s in score_values if s >= threshold)
    rejected_count = len(score_values) - accepted_count

    for bar_idx, (bar, score_val) in enumerate(zip(bars, score_values)):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{score_val:.2f}', ha='center', va='bottom', fontsize=7, rotation=45 if len(scores_dict) > 10 else 0)
    
    ax.set_xticks(range(len(segment_names)))
    ax.set_xticklabels(segment_names, rotation=60 if len(scores_dict) > 5 else 45, ha="right", fontsize=8) # Adjusted rotation
    ax.set_ylabel("Verification Score (Similarity to Reference)", fontsize=10)
    ax.set_title(f"{target_name} - {plot_title_prefix} (Accepted Solo: {accepted_count}, Rejected/Low Score: {rejected_count})", fontsize=12)
    ax.set_ylim(0, 1.05); ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.legend(loc='upper right', fontsize=9)
    out_path = output_dir / f"{safe_filename(target_name)}_{safe_filename(plot_title_prefix)}.png"
    plt.tight_layout()
    try:
        plt.savefig(out_path, dpi=150)
        log.info(f"Saved verification scores plot: {out_path.name}")
    except Exception as e:
        log.error(f"Failed to save verification scores plot {out_path.name}: {e}")
        if args_for_debug_plotting and args_for_debug_plotting.debug: log.exception("Traceback for verification plot error:")
    finally:
        plt.close(fig)
    return accepted_count, rejected_count

args_for_debug_plotting = None
def set_args_for_debug(cli_args):
    global args_for_debug_plotting
    args_for_debug_plotting = cli_args

if __name__ == '__main__':
    log.info("common.py executed directly (likely for testing).")