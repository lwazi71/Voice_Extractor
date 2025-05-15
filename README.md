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
