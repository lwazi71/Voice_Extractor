# Voice_Extractor

A tool for identifying, isolating, and transcribing clean solo segments of a target speaker from multi-speaker audio recordings. Processes audio to extract only non-overlapped speech of a specific person using modern AI techniques.

## Google Colab Version With a GUI:

[colab.research.google.com/github/ReisCook/Voice_Extractor_Colab/blob/main/Voice_Extractor_Colab.ipynb
](https://colab.research.google.com/github/ReisCook/Voice_Extractor_Colab/blob/main/Voice_Extractor_Colab.ipynb)
## Features

- **Speaker Diarization**: Identifies who is speaking when using PyAnnote
- **Overlap Detection**: Finds and removes segments with multiple speakers  
- **Target Identification**: Matches speakers to a reference sample
- **Vocal Separation**: Optional Demucs preprocessing to isolate vocals
- **Speaker Verification**: Multi-model approach (SpeechBrain & Resemblyzer)
- **Transcription**: OpenAI Whisper for accurate transcription
- **Visualization**: Spectrograms and diarization plots

## Tech Stack

- **Audio Processing**: PyAnnote Audio, Resemblyzer, Demucs
- **Speech Models**: SpeechBrain, Silero-VAD, OpenAI Whisper
- **Dependencies**: PyTorch 2.7.0+, torchaudio, torchvision, librosa
- **Output**: Verified WAV segments, transcripts (CSV/TXT), spectrograms

## Min Specs:

NVIDIA GPU with 13GB+ VRAM

## Installation



Install all required dependencies:        pip install -r requirements.txt

You'll need a Hugging Face access token which you can create at: https://huggingface.co/settings/tokens

Request access to the following gated repos:
- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/overlapped-speech-detection
- https://huggingface.co/pyannote/segmentation-3.0
- https://huggingface.co/pyannote/segmentation

# Base Command (required arguments only)
python run_extractor.py \
    --input-audio "path/to/input_audio.wav" \
    --reference-audio "path/to/target_sample.wav" \
    --target-name "TargetName" \
    --token "hf_YourHuggingFaceToken"

# Optional Arguments (add as needed)
    --output-base-dir "path/to/output_directory"  # Output directory (default: ./output_runs)
    --output-sr 44100                         # Output sample rate in Hz (default: 44100)
    --osd-model "pyannote/overlapped-speech-detection"  # Overlap detection model
    --whisper-model "large-v3"                # Transcription model (default: large-v3)
    --language "en"                           # Language for transcription (default: en)
    --min-duration 1.0                        # Minimum segment duration in seconds
    --merge-gap 0.25                          # Maximum gap between segments to merge
    --verification-threshold 0.69             # Speaker verification strictness (0-1)
    --concat-silence 0.5                      # Silence between segments in output
    --dry-run                                 # Process only first minute (testing)
    --debug                                   # Enable verbose logging
    --skip-demucs                             # Skip vocal separation
    --disable-speechbrain                     # Use only Resemblyzer verification
    --skip-rejected-transcripts               # Don't transcribe rejected segments
    --keep-temp-files                         # Keep temporary processing files
