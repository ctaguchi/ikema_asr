#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# A short script to that wraps the Silero-VAD voice activity detection
# package to act as a local recognizer in ELAN.
# This code is based on the previous implementation that used Voxseg:
# https://github.com/coxchristopher/voxseg-elan/blob/main/voxseg-elan.py

import os
import os.path
import re
import shutil
import subprocess
import sys
import tempfile

import pydub
import pydub.silence
import torch


# ---------------------------------------------------------------------
# Locate ffmpeg
# ---------------------------------------------------------------------
ffmpeg = shutil.which("ffmpeg")
if not ffmpeg:
    sys.exit("ffmpeg not found in PATH.")


# ---------------------------------------------------------------------
# Read ELAN parameters from stdin
# ---------------------------------------------------------------------
params = {}
for line in sys.stdin:
    match = re.search(r'<param name="(.*?)".*?>(.*?)</param>', line)
    if match:
        params[match.group(1)] = match.group(2).strip()


# ---------------------------------------------------------------------
# Convert source audio to 16 kHz mono WAV
# ---------------------------------------------------------------------
input_dir = tempfile.TemporaryDirectory()
input_wavs_dir = os.path.join(input_dir.name, "wavs")
os.mkdir(input_wavs_dir)

tmp_wav_file = os.path.join(input_wavs_dir, "temp_input.wav")
subprocess.call([
    ffmpeg, "-y", "-v", "0",
    "-i", params["source"],
    "-ac", "1",
    "-ar", "16000",
    "-sample_fmt", "s16",
    "-acodec", "pcm_s16le",
    tmp_wav_file
])

# ---------------------------------------------------------------------
# Load Silero-VAD model
# ---------------------------------------------------------------------
torch.set_num_threads(1)
model, utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=False,
)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

# ---------------------------------------------------------------------
# Run VAD
# ---------------------------------------------------------------------
wav = read_audio(tmp_wav_file, sampling_rate=16000)
speech_probs = get_speech_timestamps(
    wav,
    model,
    sampling_rate=16000,
    # threshold=float(params.get("speech_threshold", 0.5))
)

# Convert results to start/end times in seconds
predicted_labels = [
    {"start": s["start"] / 16000.0, "end": s["end"] / 16000.0}
    for s in speech_probs
]

# ---------------------------------------------------------------------
# Adjust segment edges
# ---------------------------------------------------------------------
adjust_start_s = float(params.get("adjust_start_ms", 0)) / 1000.0
adjust_end_s = float(params.get("adjust_end_ms", 0)) / 1000.0

# ---------------------------------------------------------------------
# Optional silence-based refinement
# ---------------------------------------------------------------------
do_silence_detection = params.get("do_silence_detection", "Disable") == "Enable"

if do_silence_detection:
    audio = pydub.AudioSegment.from_wav(tmp_wav_file)

    search_window_ms = 250
    window_ms = 10
    edge_threshold_factor = 1.0 + (float(params.get("edge_threshold", 0)) / 100)
    internal_threshold_factor = 1.0 + (float(params.get("internal_threshold", 0)) / 100)

    adjusted_labels = []
    for seg in predicted_labels:
        start_ms = int(seg["start"] * 1000)
        end_ms = int(seg["end"] * 1000)
        clip = audio[start_ms:end_ms]
        avg_vol = clip.dBFS if clip.dBFS != float("-inf") else -60.0
        threshold_vol = avg_vol * edge_threshold_factor

        # Expand start backward
        new_start = max(0, start_ms - search_window_ms)
        for w in range(new_start, start_ms, window_ms):
            window_clip = audio[w:w + window_ms]
            if window_clip.dBFS <= threshold_vol:
                start_ms = w
            else:
                start_ms = w - window_ms
                break

        # Expand end forward
        new_end = min(end_ms + search_window_ms, len(audio))
        for w in range(new_end - window_ms, start_ms, -window_ms):
            window_clip = audio[w:w + window_ms]
            if window_clip.dBFS <= threshold_vol:
                end_ms = w
            else:
                end_ms = w
                break

        adjusted_labels.append({"start": start_ms, "end": end_ms})

    # Split long segments based on silence
    split_labels = []
    keep_silence_ms = 50
    for seg in adjusted_labels:
        start_ms, end_ms = seg["start"], seg["end"]
        clip = audio[start_ms:end_ms]
        avg_vol = clip.dBFS if clip.dBFS != float("-inf") else -60.0
        thresh = avg_vol * internal_threshold_factor

        nonsilence = pydub.silence.detect_nonsilent(
            clip, min_silence_len=500, silence_thresh=thresh, seek_step=10
        )
        for i, (s, e) in enumerate(nonsilence):
            if i != 0:
                s -= keep_silence_ms
            if i != len(nonsilence) - 1:
                e += keep_silence_ms
            split_labels.append({
                "start": start_ms + s,
                "end": start_ms + e
            })
    adjusted_labels = split_labels
else:
    # convert seconds to ms for consistency
    adjusted_labels = [
        {"start": int(s["start"] * 1000), "end": int(s["end"] * 1000)}
        for s in predicted_labels
    ]

# ---------------------------------------------------------------------
# Write ELAN-style output
# ---------------------------------------------------------------------
with open(params["output_segments"], "w", encoding="utf-8") as out:
    out.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    tier_name = "SileroVADOutput-Adjusted" if do_silence_detection else "SileroVADOutput"
    out.write(f'<TIER xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
              f'xsi:noNamespaceSchemaLocation="file:avatech-tier.xsd" '
              f'columns="{tier_name}">\n')

    for seg in adjusted_labels:
        start_s = (seg["start"] / 1000.0) + adjust_start_s
        end_s = (seg["end"] / 1000.0) + adjust_end_s
        out.write(f'    <span start="{start_s:.3f}" end="{end_s:.3f}"><v></v></span>\n')

    out.write("</TIER>\n")

print("RESULT: DONE.", flush=True)
