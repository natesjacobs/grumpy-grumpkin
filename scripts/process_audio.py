#!/usr/bin/env python3
"""
Grumpkin audio processor — offline backup for pre-processing audio assets.

Reads pitch/speed settings from public/audio-config.json and applies them
to all .m4a files in public/, writing results to public/processed/.

Requirements:
    pip install librosa soundfile pydub numpy

Also requires ffmpeg on your system:
    brew install ffmpeg      (macOS)
    sudo apt install ffmpeg  (Linux)

Usage:
    cd /path/to/Grumpkin
    python scripts/process_audio.py
"""

import json
import os
import sys
import tempfile

import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment


def process_audio(input_path, output_path, pitch_semitones, speed):
    """Apply pitch shift and speed change to a single audio file."""

    # Load audio via librosa (handles m4a through ffmpeg)
    y, sr = librosa.load(input_path, sr=None, mono=False)

    # librosa works with (channels, samples) for multichannel,
    # or (samples,) for mono. Normalize to always process per-channel.
    is_mono = y.ndim == 1
    if is_mono:
        y = y[np.newaxis, :]  # shape: (1, samples)

    processed_channels = []
    for ch in range(y.shape[0]):
        channel = y[ch]

        # Pitch shift (independent of speed — uses phase vocoder internally)
        if pitch_semitones != 0:
            channel = librosa.effects.pitch_shift(
                channel, sr=sr, n_steps=pitch_semitones
            )

        # Speed change (time stretch: rate > 1 = faster)
        if speed != 1.0:
            channel = librosa.effects.time_stretch(channel, rate=speed)

        processed_channels.append(channel)

    # Recombine channels
    y_out = np.array(processed_channels)
    if is_mono:
        y_out = y_out[0]  # back to (samples,)

    # Write output — soundfile can't write m4a, so round-trip through pydub
    ext = os.path.splitext(output_path)[1].lower()
    if ext in ('.m4a', '.mp3', '.aac'):
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
        # soundfile expects (samples, channels) for multichannel
        wav_data = y_out.T if y_out.ndim > 1 else y_out
        sf.write(tmp_path, wav_data, sr)
        segment = AudioSegment.from_wav(tmp_path)
        format_map = {'.m4a': 'ipod', '.mp3': 'mp3', '.aac': 'ipod'}
        segment.export(output_path, format=format_map.get(ext, 'ipod'))
        os.unlink(tmp_path)
    else:
        wav_data = y_out.T if y_out.ndim > 1 else y_out
        sf.write(output_path, wav_data, sr)


def main():
    # Resolve paths relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, 'public', 'audio-config.json')
    input_dir = os.path.join(project_root, 'public')
    output_dir = os.path.join(project_root, 'public', 'processed')

    # Load config
    try:
        with open(config_path) as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Config not found at {config_path}, using defaults.")
        config = {'pitch': 12, 'speed': 2}

    pitch = config.get('pitch', 12)
    speed = config.get('speed', 2)

    print(f"Audio config: pitch = {pitch} semitones, speed = {speed}x")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print()

    os.makedirs(output_dir, exist_ok=True)

    for i in range(1, 7):
        input_file = os.path.join(input_dir, f'{i}.m4a')
        output_file = os.path.join(output_dir, f'{i}.m4a')

        if not os.path.exists(input_file):
            print(f"  [{i}] Skipping — {i}.m4a not found")
            continue

        print(f"  [{i}] Processing {i}.m4a ...")
        try:
            process_audio(input_file, output_file, pitch, speed)
            print(f"      -> {output_file}")
        except Exception as e:
            print(f"      ERROR: {e}", file=sys.stderr)

    print()
    print("Done! Processed files are in public/processed/")
    print("To use them, copy them back to public/ replacing the originals.")


if __name__ == '__main__':
    main()
