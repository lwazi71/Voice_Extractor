---

# 2️⃣ Voice_Extractor — `README.md`

```markdown
# Voice_Extractor

A powerful AI pipeline for identifying, isolating, and transcribing clean solo segments of a target speaker from multi-speaker audio recordings.

---

## Team Name: FitnessGram

**Members:**  
- Lwazi Mabota  
- Resis Cook  
- Zafar Ahmad  
- Jian Zhou

---

## Introduction

**Use Case:** Automatically extract speech segments from a target speaker in noisy, multi-speaker audio.

**Purpose:** Enable creation of clean speaker datasets, improve podcast/audio editing, and enhance research in speaker-based AI.

**Target Users:**  
- Podcast editors and producers  
- Speech data scientists  
- Voice biometric researchers  
- Transcription professionals  

---

## Problem Statement & Objective

Manual extraction of speaker-specific segments is tedious and error-prone. Our objective is to automate:
- Diarization: Identify who speaks when
- Overlap detection: Remove multi-speaker overlap
- Target matching: Select segments matching a reference speaker
- Verification: Confirm speaker match with multi-stage verification
- Transcription: Generate high-quality transcripts of verified segments
- Visualization: Provide plots for verification confidence

---

## Model Selection & Justification

### Diarization
- PyAnnote speaker-diarization-3.1  
- State-of-the-art diarization accuracy.

### Overlap Detection
- PyAnnote overlapped-speech-detection  
- Enables removal of overlapping speech segments.

### Speaker Matching
- WeSpeaker Deep r-vector (English)  
- Robust target speaker embedding and matching.

### Speaker Verification
- WeSpeaker Gemini model + SpeechBrain ECAPA-TDNN  
- Multi-stage verification for maximum accuracy.

### Vocal Separation (optional)
- Bandit-v2  
- Cinematic audio separation to isolate vocals from music/effects.

### Transcription
- OpenAI Whisper (large-v3)  
- Best-in-class transcription accuracy.

---

## System Architecture

![System Architecture](docs/architecture_diagram.png)

### Pipeline Flow
- Audio → PyAnnote diarization + overlap detection → clean speaker segments  
- Segments → WeSpeaker target matching → candidate segments  
- Candidates → Multi-stage verification → final verified segments  
- (Optional) Bandit-v2 → improved isolated vocals  
- Final segments → Whisper → transcription  
- Outputs → WAV + CSV/TXT transcripts + visualizations

---

## Live Demo

### Google Colab Version  
👉 [Run on Colab](https://colab.research.google.com/github/ReisCook/Voice_Extractor_Colab/blob/main/Voice_Extractor_Colab.ipynb)

### Run Locally
```bash
pip install -r requirements.txt

python run_extractor.py \
--input-audio "path/to/input_audio.wav" \
--reference-audio "path/to/target_sample.wav" \
--target-name "TargetName" \
--token "hf_YourHuggingFaceToken"
