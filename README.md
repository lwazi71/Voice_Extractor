# Voice_Extractor

A tool for identifying, isolating, and transcribing clean solo segments of a target speaker from multi-speaker audio recordings. Processes audio to extract only non-overlapped speech of a specific person using modern AI techniques.

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

Request access to the following gated repos:

https://huggingface.co/pyannote/speaker-diarization-3.1
https://huggingface.co/pyannote/overlapped-speech-detection
https://huggingface.co/pyannote/segmentation-3.0
https://huggingface.co/pyannote/segmentation

There are others - need to add the rest of the gated repo links. The error logs will let you know in the meantime.

## Dry Run (first one minute for testing):

python run_extractor.py `
    --input-audio "path/to/input_audio.wav" `
    --reference-audio "path/to/target_sample.wav" `
    --target-name "TargetName" `
    --output-base-dir "path/to/output_directory" `
    --token "hf_YourHuggingFaceToken" `
    --osd-model "pyannote/overlapped-speech-detection" `
    --dry-run `
    --debug

## Full Run (processes the entire audio file):

python run_extractor.py `
    --input-audio "path/to/input_audio.wav" `
    --reference-audio "path/to/target_sample.wav" `
    --target-name "TargetName" `
    --output-base-dir "path/to/output_directory" `
    --token "hf_YourHuggingFaceToken" `
    --osd-model "pyannote/overlapped-speech-detection" `
    --debug

## Full Run (24k Hz sample rate for Sesame CSM):

python run_extractor.py `
    --input-audio "path/to/input_audio.wav" `
    --reference-audio "path/to/target_sample.wav" `
    --target-name "TargetName" `
    --output-base-dir "path/to/output_directory" `
    --token "hf_YourHuggingFaceToken" `
    --osd-model "pyannote/overlapped-speech-detection" `
    --output-sr 24000 `
    --whisper-model "base.en" `
    --language "en" `
    --debug


## Optional Arguments:

--skip-demucs: If your full_audio.wav is already relatively clean or primarily vocals, you might skip Demucs to save time.
--disable-speechbrain: If you want to rely solely on Resemblyzer for verification (might be faster, potentially less accurate).
--concat-silence 0.25: To change the silence duration between concatenated segments (default is 0.5s).
--min-duration X: To change the minimum duration for a solo segment (default 1.0s).
--merge-gap Y: To change the maximum gap for merging segments (default 0.25s).
--verification-threshold Z: To adjust the speaker verification strictness (default 0.69).
--skip-rejected-transcripts: If you decide you don't need transcripts for the rejected segments to save time.
--keep-temp-files: If you want to inspect the __tmp_processing directory after the run.
