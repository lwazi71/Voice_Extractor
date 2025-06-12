Here’s your cleaned version of the README.md for Voice_Extractor, no weird text — fully polished, ready to paste:

⸻

Voice_Extractor

A powerful AI pipeline for identifying, isolating, and transcribing clean solo segments of a target speaker from multi-speaker audio recordings.

⸻

Team Name: FitnessGram

Members:
	•	Lwazi Mabota
	•	Resis Cook
	•	Zafar Ahmad
	•	Jian Zhou

⸻

Introduction

Use Case: Automatically extract speech segments from a target speaker in noisy, multi-speaker audio.

Purpose: Enable creation of clean speaker datasets, improve podcast/audio editing, and enhance research in speaker-based AI.

Target Users:
	•	Podcast editors and producers
	•	Speech data scientists
	•	Voice biometric researchers
	•	Transcription professionals

⸻

Problem Statement & Objective

Manual extraction of speaker-specific segments is tedious and error-prone. Our objective is to automate:
	•	Diarization: Identify who speaks when
	•	Overlap detection: Remove multi-speaker overlap
	•	Target matching: Select segments matching a reference speaker
	•	Verification: Confirm speaker match with multi-stage verification
	•	Transcription: Generate high-quality transcripts of verified segments
	•	Visualization: Provide plots for verification confidence

⸻

Model Selection & Justification

Diarization
	•	PyAnnote speaker-diarization-3.1
State-of-the-art diarization accuracy.

Overlap Detection
	•	PyAnnote overlapped-speech-detection
Enables removal of overlapping speech segments.

Speaker Matching
	•	WeSpeaker Deep r-vector (English)
Robust target speaker embedding and matching.

Speaker Verification
	•	WeSpeaker Gemini model + SpeechBrain ECAPA-TDNN
Multi-stage verification for maximum accuracy.

Vocal Separation (optional)
	•	Bandit-v2
Cinematic audio separation to isolate vocals from music/effects.

Transcription
	•	OpenAI Whisper (large-v3)
Best-in-class transcription accuracy.

⸻

System Architecture

Pipeline Flow
	•	Audio → PyAnnote diarization + overlap detection → clean speaker segments
	•	Segments → WeSpeaker target matching → candidate segments
	•	Candidates → Multi-stage verification → final verified segments
	•	(Optional) Bandit-v2 → improved isolated vocals
	•	Final segments → Whisper → transcription
	•	Outputs → WAV + CSV/TXT transcripts + visualizations

⸻

Live Demo

Google Colab Version

👉 Run on Colab

Run Locally

pip install -r requirements.txt

python run_extractor.py \
--input-audio "path/to/input_audio.wav" \
--reference-audio "path/to/target_sample.wav" \
--target-name "TargetName" \
--token "hf_YourHuggingFaceToken"


⸻

Required Model Access

Request access to the following gated repos:
	•	https://huggingface.co/pyannote/speaker-diarization-3.1
	•	https://huggingface.co/pyannote/overlapped-speech-detection
	•	https://huggingface.co/pyannote/segmentation-3.0
	•	https://huggingface.co/pyannote/segmentation

⸻

Evaluation & User Feedback

Metrics
	•	Speaker verification accuracy: > 95% on clean samples, ~88% on noisy real-world audio
	•	BLEU / WER scores (Whisper transcriptions) validated on internal podcast & meeting datasets

User Survey
	•	Segmentation quality: 4.5/5
	•	Verification correctness: 4.7/5
	•	Transcription accuracy: 4.6/5
	•	Visualization clarity: 4.8/5

Example Scenario

Input: Multi-speaker podcast (3 speakers)
Target: Host A (reference sample provided)
Output:
	•	Verified segments of Host A only (non-overlapped)
	•	Clean WAV files
	•	CSV + TXT transcriptions
	•	Verification score plots & spectrograms

⸻

Strengths
	•	High-quality speaker isolation
	•	Accurate multi-stage verification
	•	Flexible and modular pipeline
	•	Excellent visualization tools

⸻

Limitations
	•	Requires 16GB+ VRAM GPU
	•	OSD imperfect on very noisy speech
	•	Manual tuning of verification threshold required

⸻

Conclusion & Future Work

What Worked
	•	Highly accurate pipeline
	•	Modular CLI with flexible options
	•	Robust real-world performance

Improvements Needed
	•	Faster processing for long audio
	•	Improved OSD on noisy files
	•	More user-friendly GUI

Next Steps
	•	Real-time / streaming extraction
	•	Speaker diarization web explorer
	•	Multi-target speaker extraction
	•	Domain adaptation for TV, call centers, podcasting

⸻

Tech Stack
	•	AI Models:
	•	Bandit-v2 (source separation)
	•	PyAnnote (diarization & overlap detection)
	•	WeSpeaker r-vector + Gemini verification
	•	SpeechBrain ECAPA-TDNN
	•	OpenAI Whisper large-v3
	•	Silero VAD
	•	Libraries:
	•	PyTorch
	•	torchaudio
	•	torchvision
	•	librosa
	•	asteroid
	•	ffmpeg-python
	•	ray

⸻

License

MIT License
