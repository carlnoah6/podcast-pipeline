"""Audio slicing with webrtcvad + quality analysis on Modal.

Downloads audio from HuggingFace, slices with VAD, analyzes quality,
uploads segments back to HuggingFace.
"""
from __future__ import annotations

import modal

app = modal.App("podcast-slice")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("webrtcvad", "pydub", "numpy", "scipy", "huggingface_hub", "httpx")
)


@app.function(image=image, timeout=600, cpu=2.0, memory=4096)
def slice_episode(
    episode_id: str,
    audio_repo: str,
    output_repo: str,
    hf_token: str,
    min_duration: float = 3.0,
    max_duration: float = 15.0,
    aggressiveness: int = 2,
) -> dict:
    """Slice a single episode's audio into segments using VAD.

    Returns dict with episode_id, segment count, quality stats.
    """
    import io
    import json
    import struct
    import tempfile
    import wave
    from pathlib import Path

    import numpy as np
    import webrtcvad
    from huggingface_hub import HfApi, hf_hub_download
    from pydub import AudioSegment
    from scipy import signal as scipy_signal

    api = HfApi(token=hf_token)

    # Download audio from HF
    try:
        audio_path = hf_hub_download(
            repo_id=audio_repo,
            filename=f"audio/{episode_id}.m4a",
            repo_type="dataset",
            token=hf_token,
        )
    except Exception:
        return {"episode_id": episode_id, "error": "audio not found on HF", "segments": 0}

    # Convert to 16kHz mono WAV for VAD
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)

    # VAD segmentation
    vad = webrtcvad.Vad(aggressiveness)
    sample_rate = 16000
    frame_duration_ms = 30
    frame_size = int(sample_rate * frame_duration_ms / 1000) * 2  # bytes

    raw = audio.raw_data
    frames = [raw[i:i + frame_size] for i in range(0, len(raw), frame_size)]
    frames = [f for f in frames if len(f) == frame_size]

    # Detect speech segments
    is_speech = []
    for frame in frames:
        try:
            is_speech.append(vad.is_speech(frame, sample_rate))
        except Exception:
            is_speech.append(False)

    # Merge adjacent speech frames into segments
    segments = []
    in_segment = False
    start = 0
    min_silence_frames = int(300 / frame_duration_ms)  # 300ms silence to split

    silence_count = 0
    for i, speech in enumerate(is_speech):
        if speech:
            if not in_segment:
                start = i
                in_segment = True
            silence_count = 0
        else:
            if in_segment:
                silence_count += 1
                if silence_count >= min_silence_frames:
                    end = i - silence_count
                    duration = (end - start) * frame_duration_ms / 1000
                    if min_duration <= duration <= max_duration:
                        segments.append((start, end))
                    in_segment = False
                    silence_count = 0

    if in_segment:
        end = len(is_speech)
        duration = (end - start) * frame_duration_ms / 1000
        if min_duration <= duration <= max_duration:
            segments.append((start, end))

    # Extract segments and analyze quality
    results = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for idx, (start_frame, end_frame) in enumerate(segments):
            start_ms = start_frame * frame_duration_ms
            end_ms = end_frame * frame_duration_ms
            segment_audio = audio[start_ms:end_ms]

            # Quality analysis
            samples = np.array(segment_audio.get_array_of_samples(), dtype=np.float32)
            if len(samples) == 0:
                continue

            # SNR estimation
            rms = np.sqrt(np.mean(samples ** 2))
            noise_floor = np.percentile(np.abs(samples), 5)
            snr = 20 * np.log10(rms / max(noise_floor, 1e-10))

            # Spectral flatness
            freqs = np.abs(np.fft.rfft(samples))
            freqs = freqs[freqs > 0]
            if len(freqs) > 0:
                geo_mean = np.exp(np.mean(np.log(freqs + 1e-10)))
                arith_mean = np.mean(freqs)
                spectral_flatness = geo_mean / max(arith_mean, 1e-10)
            else:
                spectral_flatness = 0

            # Zero crossing rate
            zcr = np.sum(np.abs(np.diff(np.sign(samples)))) / (2 * len(samples))

            duration_s = (end_ms - start_ms) / 1000
            quality_score = snr * 0.4 + (1 - spectral_flatness) * 30 + (1 - min(zcr * 10, 1)) * 30

            seg_info = {
                "episode_id": episode_id,
                "segment_idx": idx,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "duration_s": round(duration_s, 2),
                "snr": round(float(snr), 2),
                "spectral_flatness": round(float(spectral_flatness), 4),
                "zcr": round(float(zcr), 4),
                "quality_score": round(float(quality_score), 2),
            }
            results.append(seg_info)

            # Export segment as WAV
            seg_path = Path(tmpdir) / f"{episode_id}_{idx:04d}.wav"
            segment_audio.export(str(seg_path), format="wav",
                                 parameters=["-ar", "24000", "-ac", "1"])

        # Upload segments to HF
        if results:
            # Upload analysis JSON
            analysis_path = Path(tmpdir) / f"{episode_id}_analysis.json"
            with open(analysis_path, "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            api.upload_file(
                path_or_fileobj=str(analysis_path),
                path_in_repo=f"slices/{episode_id}/analysis.json",
                repo_id=output_repo,
                repo_type="dataset",
                commit_message=f"Slice analysis: {episode_id}",
            )

            # Upload top segments (by quality score)
            top_segments = sorted(results, key=lambda x: x["quality_score"], reverse=True)[:10]
            for seg in top_segments:
                idx = seg["segment_idx"]
                seg_file = Path(tmpdir) / f"{episode_id}_{idx:04d}.wav"
                if seg_file.exists():
                    api.upload_file(
                        path_or_fileobj=str(seg_file),
                        path_in_repo=f"slices/{episode_id}/{episode_id}_{idx:04d}.wav",
                        repo_id=output_repo,
                        repo_type="dataset",
                        commit_message=f"Slice: {episode_id} seg {idx}",
                    )

    return {
        "episode_id": episode_id,
        "segments": len(results),
        "top_quality": round(results[0]["quality_score"], 2) if results else 0,
        "avg_quality": round(sum(r["quality_score"] for r in results) / len(results), 2) if results else 0,
    }
