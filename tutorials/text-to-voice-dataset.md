# Building Custom Datasets to Train a Text-to-Voice Model

A step-by-step guide for creating high-quality speech datasets from scratch.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Step 1: Define Your Voice Profile](#step-1-define-your-voice-profile)
4. [Step 2: Write and Curate Your Scripts](#step-2-write-and-curate-your-scripts)
5. [Step 3: Set Up Your Recording Environment](#step-3-set-up-your-recording-environment)
6. [Step 4: Record Your Audio](#step-4-record-your-audio)
7. [Step 5: Process and Clean the Audio](#step-5-process-and-clean-the-audio)
8. [Step 6: Align Transcripts to Audio](#step-6-align-transcripts-to-audio)
9. [Step 7: Validate Dataset Quality](#step-7-validate-dataset-quality)
10. [Step 8: Structure and Package the Dataset](#step-8-structure-and-package-the-dataset)
11. [Step 9: Train a Baseline Model](#step-9-train-a-baseline-model)
12. [Step 10: Evaluate and Iterate](#step-10-evaluate-and-iterate)
13. [Reference: Tools and Libraries](#reference-tools-and-libraries)

---

## Overview

A text-to-speech (TTS) model learns to convert written text into natural-sounding audio by studying paired examples of text and its spoken recording. The quality of the dataset directly determines the quality of the voice model — garbage in, garbage out.

This tutorial walks through the full pipeline:

```
Script writing → Recording → Audio processing → Alignment → Validation → Training
```

A practical minimum for a single-speaker TTS model is roughly **1–3 hours of clean, aligned audio**. Professional-quality voices typically use 10–30 hours.

---

## Prerequisites

**Knowledge**
- Basic command-line usage
- Python 3.10+
- Familiarity with audio concepts (sample rate, bit depth, mono/stereo)

**Hardware**
- A condenser or dynamic microphone (USB condenser is fine for starters)
- Quiet room or sound-treated space
- Computer with at least 8 GB RAM

**Software** — install before starting:

```bash
pip install librosa soundfile pydub nemo_toolkit[all] resemblyzer
pip install montreal-forced-aligner   # MFA for transcript alignment
sudo apt install ffmpeg               # or brew install ffmpeg on macOS
```

---

## Step 1: Define Your Voice Profile

Before recording a single word, decide what the voice should sound like. Inconsistency between sessions is one of the biggest dataset killers.

**Document the following:**

| Parameter | Example choices |
|---|---|
| Speaker | Yourself, a hired voice actor, a synthetic pre-voice |
| Language & accent | American English, Brazilian Portuguese, etc. |
| Speaking pace | ~130–160 wpm for natural speech |
| Pitch register | Natural / no affectation |
| Emotional tone | Neutral, warm, authoritative |
| Use-case domain | Audiobooks, navigation, customer service |

**Create a reference recording.** Record a 2-minute sample reading any passage at your target pace and tone. Save it as `reference_voice.wav`. Return to this file at the start of every recording session to calibrate your voice.

---

## Step 2: Write and Curate Your Scripts

The text you record is called a **script** or **corpus**. Its linguistic diversity determines how well the model generalizes.

### 2.1 Coverage goals

A good script covers:

- All phonemes in the target language (phonetically balanced sentences)
- Common words and rare words
- Multiple sentence lengths: short (< 5 words), medium (5–15 words), long (15–30 words)
- Questions, statements, and exclamations
- Numbers, abbreviations, proper nouns, and punctuation patterns

### 2.2 Use an existing corpus as a base

You do not need to write all sentences from scratch. Publicly licensed corpora provide a strong foundation:

| Corpus | Language | Notes |
|---|---|---|
| [CMU ARCTIC](http://www.festvox.org/cmu_arctic/) | English | Phonetically balanced, 1,132 sentences |
| [LJ Speech](https://keithito.com/LJ-Speech-Dataset/) | English | 13,100 sentences from public domain books |
| [LibriTTS](https://openslr.org/60/) | English | 585 hours from LibriVox audiobooks |
| [M-AILABS](https://www.caito.de/2019/01/the-m-ailabs-speech-dataset/) | Multi-language | German, French, Spanish, Italian, and more |

For a custom domain (e.g., ebook narration), supplement a base corpus with sentences extracted from your target content.

### 2.3 Clean the scripts

```python
import re

def clean_sentence(text: str) -> str:
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove unspeakable characters
    text = re.sub(r'[^\w\s.,!?\'-]', '', text)
    # Expand common abbreviations
    abbreviations = {
        "Dr.": "Doctor",
        "Mr.": "Mister",
        "Mrs.": "Missus",
        "vs.": "versus",
        "etc.": "et cetera",
    }
    for abbr, expansion in abbreviations.items():
        text = text.replace(abbr, expansion)
    return text

with open("raw_scripts.txt") as f:
    sentences = [clean_sentence(line) for line in f if line.strip()]

# Remove duplicates and very short sentences
sentences = list({s for s in sentences if len(s.split()) >= 3})

with open("clean_scripts.txt", "w") as f:
    for s in sentences:
        f.write(s + "\n")
```

### 2.4 Format: one sentence per line

Each line in the final script file is one recording prompt. Keep sentences under ~20 seconds when spoken aloud (roughly 50 words maximum).

```
The wind swept across the empty harbor before dawn.
She opened the letter twice, reading it slowly each time.
Turn left at the next intersection and continue for two miles.
```

---

## Step 3: Set Up Your Recording Environment

### 3.1 Treat the room

Unwanted room acoustics contaminate recordings permanently — no software fully fixes a reverberant room.

**Quick treatment options:**
- Record in a closet surrounded by hanging clothes
- Use heavy curtains, rugs, bookshelves to absorb reflections
- Avoid rooms with hard parallel walls
- Turn off fans, HVAC, and any device with a spinning disk

**Test your room:** Clap once sharply and listen. If you hear a distinct decay or flutter, add more absorption.

### 3.2 Configure your DAW or recording software

Recommended free options: **Audacity**, **Reaper** (discounted license), **Ocenaudio**

Settings:
- **Sample rate:** 44,100 Hz or 22,050 Hz (TTS models typically downsample to 22,050 Hz)
- **Bit depth:** 24-bit WAV
- **Channels:** Mono
- **File format:** WAV (lossless; convert to FLAC for archiving)

### 3.3 Set gain correctly

Your microphone gain should peak between **-12 dBFS and -6 dBFS** during normal speech. Peaks above -3 dBFS cause clipping (irreversible distortion). Peaks below -24 dBFS will have a low signal-to-noise ratio.

---

## Step 4: Record Your Audio

### 4.1 Session structure

Divide recording into short sessions of **30–60 minutes** to prevent voice fatigue. Fatigue introduces subtle tonal shifts that confuse the model.

**Before each session:**
1. Hydrate — drink water, not coffee or tea
2. Do brief vocal warm-ups (hum, lip trills)
3. Listen to `reference_voice.wav` from Step 1
4. Record a short calibration phrase and compare to the reference

### 4.2 Recording workflow

Organize each recording session as a folder:

```
dataset/
  raw/
    session_01/
      001.wav
      002.wav
      ...
    session_02/
      ...
  scripts/
    clean_scripts.txt
```

Automate prompt delivery with a simple script so you never have to switch windows:

```python
# prompter.py — displays one sentence at a time
sentences = open("scripts/clean_scripts.txt").read().splitlines()
for i, sentence in enumerate(sentences, start=1):
    print(f"\n[{i}/{len(sentences)}] {sentence}")
    input("  Press ENTER when done recording...")
```

### 4.3 During recording

- Keep a consistent **mouth-to-mic distance**: 6–10 cm with a pop filter
- Re-record any take where you stumbled, coughed, or varied your tone significantly
- Do not edit within a take — record the whole sentence again from the start
- Log bad takes in a text file; do not delete them until after alignment

---

## Step 5: Process and Clean the Audio

### 5.1 Batch normalize loudness

TTS models train better when all clips have consistent loudness. Use the **EBU R128** standard (-23 LUFS) or a simpler RMS normalization:

```bash
# Normalize all WAV files in raw/ to -23 LUFS using ffmpeg-normalize
pip install ffmpeg-normalize
ffmpeg-normalize raw/**/*.wav -o processed/ -ext wav \
  --normalization-type ebu -t -23 --sample-rate 22050
```

### 5.2 Trim silence

Strip leading and trailing silence from each clip:

```python
from pydub import AudioSegment
from pydub.silence import detect_leading_silence
import os

def trim_silence(path_in: str, path_out: str, silence_thresh: int = -50) -> None:
    audio = AudioSegment.from_wav(path_in)
    start_trim = detect_leading_silence(audio, silence_threshold=silence_thresh)
    end_trim = detect_leading_silence(audio.reverse(), silence_threshold=silence_thresh)
    duration = len(audio)
    trimmed = audio[start_trim: duration - end_trim]
    trimmed.export(path_out, format="wav")

os.makedirs("trimmed", exist_ok=True)
for fname in os.listdir("processed"):
    if fname.endswith(".wav"):
        trim_silence(f"processed/{fname}", f"trimmed/{fname}")
```

### 5.3 Remove noise (optional but recommended)

```python
import noisereduce as nr
import soundfile as sf
import librosa

def denoise(path_in: str, path_out: str) -> None:
    y, sr = librosa.load(path_in, sr=None)
    # Use the first 0.5 s as a noise profile sample
    noise_sample = y[:int(sr * 0.5)]
    reduced = nr.reduce_noise(y=y, sr=sr, y_noise=noise_sample)
    sf.write(path_out, reduced, sr)
```

### 5.4 Discard outliers

Remove clips that are too short, too long, or too noisy:

```python
import librosa

MIN_DURATION = 1.0   # seconds
MAX_DURATION = 15.0  # seconds

def check_duration(path: str) -> bool:
    duration = librosa.get_duration(filename=path)
    return MIN_DURATION <= duration <= MAX_DURATION
```

---

## Step 6: Align Transcripts to Audio

Forced alignment maps each word (or phoneme) in a transcript to its exact timestamp in the audio. This is required by most TTS trainers.

### 6.1 Install and configure Montreal Forced Aligner (MFA)

```bash
conda install -c conda-forge montreal-forced-aligner
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa
```

### 6.2 Prepare the MFA input structure

MFA expects audio and matching transcript files side by side:

```
mfa_input/
  speaker_01/
    001.wav
    001.txt     # contains the sentence spoken in 001.wav
    002.wav
    002.txt
```

Generate the `.txt` files from your script:

```python
import os

sentences = open("scripts/clean_scripts.txt").read().splitlines()
os.makedirs("mfa_input/speaker_01", exist_ok=True)

for i, sentence in enumerate(sentences, start=1):
    wav_src = f"trimmed/{i:04d}.wav"
    txt_dst = f"mfa_input/speaker_01/{i:04d}.txt"
    wav_dst = f"mfa_input/speaker_01/{i:04d}.wav"
    if os.path.exists(wav_src):
        import shutil
        shutil.copy(wav_src, wav_dst)
        with open(txt_dst, "w") as f:
            f.write(sentence)
```

### 6.3 Run alignment

```bash
mfa align mfa_input/ english_us_arpa english_us_arpa mfa_output/ \
  --clean --num_jobs 4
```

MFA outputs `.TextGrid` files containing word- and phoneme-level timestamps. Most TTS frameworks consume these directly.

---

## Step 7: Validate Dataset Quality

Never feed unvalidated data into training. A small amount of bad data degrades the model disproportionately.

### 7.1 Automated checks

```python
import librosa
import numpy as np

def validate_clip(wav_path: str, transcript: str) -> dict:
    y, sr = librosa.load(wav_path, sr=22050)
    duration = librosa.get_duration(y=y, sr=sr)
    rms = np.sqrt(np.mean(y**2))
    issues = []

    if duration < 0.5:
        issues.append("too_short")
    if duration > 20.0:
        issues.append("too_long")
    if rms < 0.005:
        issues.append("too_quiet")
    if rms > 0.9:
        issues.append("clipping_risk")

    words_per_second = len(transcript.split()) / duration
    if words_per_second > 5.0:
        issues.append("unnatural_pace")

    return {"path": wav_path, "duration": duration, "rms": float(rms), "issues": issues}
```

### 7.2 Manual listening review

Sample at least **10% of clips** by ear. Listen for:
- Pronunciation errors or stumbles
- Background noise spikes
- Tone or pace inconsistency versus your reference recording
- Clipping or distortion

Flag bad clips in a `bad_clips.txt` file and remove them from the dataset.

### 7.3 Speaker consistency check

Use a speaker verification model to detect clips that sound significantly different from the rest:

```python
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import numpy as np

encoder = VoiceEncoder()
wav_paths = list(Path("trimmed").glob("*.wav"))

embeddings = [encoder.embed_utterance(preprocess_wav(p)) for p in wav_paths]
mean_embedding = np.mean(embeddings, axis=0)

SIMILARITY_THRESHOLD = 0.75
outliers = []
for path, emb in zip(wav_paths, embeddings):
    similarity = np.dot(emb, mean_embedding)
    if similarity < SIMILARITY_THRESHOLD:
        outliers.append((str(path), similarity))

for path, score in sorted(outliers, key=lambda x: x[1]):
    print(f"Low similarity ({score:.3f}): {path}")
```

Review and re-record flagged clips.

---

## Step 8: Structure and Package the Dataset

### 8.1 Standard directory layout

```
tts_dataset/
  wavs/
    0001.wav
    0002.wav
    ...
  metadata.csv
  LICENSE.txt
  SPEAKERS.txt
```

### 8.2 Create metadata.csv

The metadata file links filenames to transcripts. Different TTS frameworks expect slightly different formats; here is the LJ Speech format (pipe-separated), which is the most widely supported:

```
filename|normalized_text|raw_text
wavs/0001|The wind swept across the empty harbor before dawn.|The wind swept across the empty harbor before dawn.
wavs/0002|She opened the letter twice reading it slowly each time.|She opened the letter twice, reading it slowly each time.
```

Generate it programmatically:

```python
import csv
import os

sentences = open("scripts/clean_scripts.txt").read().splitlines()
rows = []

for i, sentence in enumerate(sentences, start=1):
    wav_path = f"wavs/{i:04d}"
    if os.path.exists(f"tts_dataset/{wav_path}.wav"):
        rows.append({
            "filename": wav_path,
            "normalized_text": sentence.lower().replace(",", "").replace(".", ""),
            "raw_text": sentence,
        })

with open("tts_dataset/metadata.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["filename", "normalized_text", "raw_text"],
                            delimiter="|")
    for row in rows:
        f.writerow(row)

print(f"Dataset contains {len(rows)} clips.")
```

### 8.3 Compute dataset statistics

```python
import librosa
import os

total_duration = 0.0
for fname in os.listdir("tts_dataset/wavs"):
    if fname.endswith(".wav"):
        total_duration += librosa.get_duration(filename=f"tts_dataset/wavs/{fname}")

hours = total_duration / 3600
print(f"Total audio: {total_duration:.1f} s ({hours:.2f} hours)")
print(f"Total clips: {len(os.listdir('tts_dataset/wavs'))}")
```

---

## Step 9: Train a Baseline Model

With a clean, aligned dataset, you can train using any major open-source TTS framework.

### 9.1 Recommended frameworks

| Framework | Architecture | Best for |
|---|---|---|
| [Coqui TTS](https://github.com/coqui-ai/TTS) | VITS, XTTS, Tacotron2 | Easiest to start; wide model support |
| [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) | FastPitch, RADTTS | Production-grade; GPU cluster support |
| [ESPnet](https://github.com/espnet/espnet) | VITS, FastSpeech2 | Research-focused; many languages |
| [Piper](https://github.com/rhasspy/piper) | VITS | Lightweight; runs on Raspberry Pi |

### 9.2 Quick-start with Coqui TTS (VITS model)

```bash
pip install TTS

# Create a config file
tts train \
  --config_path configs/vits_ljspeech.json \
  --output_path output/vits_custom/ \
  --dataset_path tts_dataset/ \
  --dataset_name ljspeech \
  --meta_file_train metadata.csv
```

A minimal config override for your dataset:

```json
{
  "datasets": [
    {
      "name": "ljspeech",
      "path": "tts_dataset/",
      "meta_file_train": "metadata.csv",
      "meta_file_val": null
    }
  ],
  "audio": {
    "sample_rate": 22050,
    "fft_size": 1024,
    "win_length": 1024,
    "hop_length": 256,
    "num_mels": 80
  },
  "batch_size": 32,
  "eval_batch_size": 16,
  "num_loader_workers": 4,
  "run_eval": true,
  "save_step": 1000,
  "epochs": 1000
}
```

### 9.3 Hardware requirements

| Dataset size | Recommended GPU | Approximate training time |
|---|---|---|
| 1 hour | RTX 3060 (12 GB) | 12–24 hours |
| 5 hours | RTX 3090 (24 GB) | 3–5 days |
| 20+ hours | A100 (40 GB) or multi-GPU | 1–2 weeks |

Use [Google Colab Pro](https://colab.research.google.com/) or [RunPod](https://runpod.io/) if you do not own a suitable GPU.

---

## Step 10: Evaluate and Iterate

### 10.1 Objective metrics

| Metric | What it measures | Target range |
|---|---|---|
| MOS (Mean Opinion Score) | Human-rated naturalness (1–5) | > 4.0 is good |
| MCD (Mel Cepstral Distortion) | Spectral similarity to ground truth | < 8 dB |
| WER (Word Error Rate via ASR) | Intelligibility | < 5% |
| RTF (Real-Time Factor) | Inference speed | < 1.0 (faster than real-time) |

### 10.2 Evaluate intelligibility with an ASR model

```python
import whisper

model = whisper.load_model("base")

def measure_wer(wav_path: str, reference_text: str) -> float:
    result = model.transcribe(wav_path)
    hypothesis = result["text"].strip().lower()
    reference = reference_text.strip().lower()
    # Simple WER: word-level edit distance / reference length
    from jiwer import wer
    return wer(reference, hypothesis)
```

### 10.3 Common failure modes and fixes

| Symptom | Likely cause | Fix |
|---|---|---|
| Robotic or buzzy voice | Insufficient data or short clips | Add more recordings; minimum 2 hours |
| Mispronunciations on proper nouns | Missing lexicon entries | Add entries to the pronunciation dictionary |
| Inconsistent prosody | Noisy training data | Re-run validation, remove outlier clips |
| Model repeats or skips words | Alignment errors | Re-run MFA; manually inspect TextGrids |
| Fast or slow delivery | Pace inconsistency in recordings | Re-record affected sessions with reference playback |

### 10.4 Iterate

Improving a TTS voice is a cycle:

```
Record more data
     ↓
Retrain (fine-tune from checkpoint, do not start from scratch)
     ↓
Evaluate with MOS + WER
     ↓
Identify weakest phonemes / words
     ↓
Add targeted sentences to cover gaps
     ↓
Repeat
```

Fine-tuning from a checkpoint is far more efficient than full retraining. Always save checkpoints every 1,000–5,000 training steps.

---

## Reference: Tools and Libraries

| Tool | Purpose | Link |
|---|---|---|
| Audacity | Free DAW for recording | https://www.audacityteam.org |
| ffmpeg | Audio conversion and normalization | https://ffmpeg.org |
| ffmpeg-normalize | Batch EBU R128 normalization | https://github.com/slhck/ffmpeg-normalize |
| pydub | Python audio manipulation | https://github.com/jiaaro/pydub |
| librosa | Python audio analysis | https://librosa.org |
| noisereduce | Spectral noise reduction | https://github.com/timsainburg/noisereduce |
| Montreal Forced Aligner | Transcript-to-audio alignment | https://montreal-forced-aligner.readthedocs.io |
| Resemblyzer | Speaker verification embeddings | https://github.com/resemble-ai/Resemblyzer |
| Coqui TTS | End-to-end TTS training | https://github.com/coqui-ai/TTS |
| OpenAI Whisper | ASR for intelligibility evaluation | https://github.com/openai/whisper |
| jiwer | Word Error Rate computation | https://github.com/jitsi/jiwer |

---

*Last updated: 2026-02-22*
