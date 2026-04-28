"""Audio features for the Presentation Analysis scorer.

Extracts 11 acoustic features from a video's audio track. All features are
**language-agnostic** — we do NOT depend on transcription quality since our
training set mixes English and Thai clips (Whisper tiny struggles with Thai).

Features
--------
``audio_voiced_ratio``           fraction of frames with detectable pitch
``audio_speech_ratio``           fraction of audio identified as speech (non-silent)
``audio_pause_count_per_min``    number of pauses (≥ 0.5s silence) per minute
``audio_pause_mean_duration``    mean pause length in seconds
``audio_longest_pause``          longest pause in seconds
``audio_pitch_mean``             mean f0 across voiced frames (Hz)
``audio_pitch_std``              std of f0 (monotone → low, dynamic → high)
``audio_pitch_range``            (p95 − p5) of f0, robust to outliers
``audio_volume_mean``            mean RMS amplitude
``audio_volume_std``             std of RMS amplitude
``audio_volume_dynamic_range``   20 * log10(max_rms / mean_rms), in dB

Calling :func:`extract_audio_features` on a video path runs the whole pipeline
and returns a dict. On any failure (no audio track, missing library, etc.) it
returns a dict of zeros so downstream code never crashes.

Cost (typical 60-120s video, Apple M4): ~5-8s per clip (mostly librosa pyin).
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Dict

logger = logging.getLogger(__name__)


FEATURE_KEYS = (
    "audio_voiced_ratio",
    "audio_speech_ratio",
    "audio_pause_count_per_min",
    "audio_pause_mean_duration",
    "audio_longest_pause",
    "audio_pitch_mean",
    "audio_pitch_std",
    "audio_pitch_range",
    "audio_volume_mean",
    "audio_volume_std",
    "audio_volume_dynamic_range",
)


def _zeros() -> Dict[str, float]:
    return {k: 0.0 for k in FEATURE_KEYS}


def extract_audio_features(video_path: str, sr: int = 16000) -> Dict[str, float]:
    """Pull an audio clip out of ``video_path`` and compute 11 acoustic features.

    Returns a dict keyed by :data:`FEATURE_KEYS`; zeros if extraction fails.
    """
    out = _zeros()

    try:
        import numpy as np
        import librosa  # type: ignore
    except ImportError as exc:
        logger.warning("[audio] librosa not available: %s", exc)
        return out

    try:
        from moviepy.editor import VideoFileClip  # type: ignore
    except ImportError as exc:
        logger.warning("[audio] moviepy not available: %s", exc)
        return out

    tmp_wav: str = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_wav = tmp.name

        # ---- Extract audio track as 16 kHz mono PCM ----
        clip = VideoFileClip(video_path)
        if clip.audio is None:
            logger.warning("[audio] no audio track in %s", video_path)
            clip.close()
            return out
        clip.audio.write_audiofile(
            tmp_wav, fps=sr, nbytes=2, codec="pcm_s16le",
            verbose=False, logger=None,
        )
        clip.close()

        # ---- Load as numpy mono ----
        y, sr = librosa.load(tmp_wav, sr=sr, mono=True)
        duration = max(len(y) / float(sr), 0.01)

        # ---- Pitch (f0) via yin — faster than pyin and good enough. ----
        # yin returns f0 estimate per frame; NaN for unvoiced frames.
        try:
            f0 = librosa.yin(
                y, fmin=75, fmax=400, sr=sr,
                frame_length=2048, hop_length=512,
            )
            # yin returns all frames even if "unvoiced"; filter by harmonic
            # probability proxy using zero-crossing-rate to avoid noisy values.
            zcr = librosa.feature.zero_crossing_rate(y, hop_length=512)[0]
            valid_mask = (f0 > 80.0) & (f0 < 400.0) & (zcr < 0.30)
            voiced = f0[valid_mask]
            out["audio_voiced_ratio"] = float(valid_mask.sum()) / max(len(f0), 1)
            if voiced.size >= 10:
                out["audio_pitch_mean"] = float(np.mean(voiced))
                out["audio_pitch_std"] = float(np.std(voiced))
                p5, p95 = np.percentile(voiced, [5, 95])
                out["audio_pitch_range"] = float(p95 - p5)
        except Exception as exc:
            logger.warning("[audio] pitch extraction failed: %s", exc)

        # ---- RMS (volume) ----
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        if rms.size >= 10:
            mean_rms = float(np.mean(rms))
            max_rms = float(np.max(rms))
            out["audio_volume_mean"] = mean_rms
            out["audio_volume_std"] = float(np.std(rms))
            # Dynamic range in dB (max / mean). Clamp mean away from 0.
            mean_safe = max(mean_rms, 1e-5)
            out["audio_volume_dynamic_range"] = float(20.0 * np.log10(max_rms / mean_safe))

        # ---- Speech/pause detection via librosa.effects.split ----
        # top_db=30 means silence = regions 30 dB below peak. Good for most
        # speech recordings even with some background noise.
        try:
            intervals = librosa.effects.split(
                y, top_db=30, frame_length=2048, hop_length=512,
            )
            if intervals.size > 0:
                # Speech ratio = total non-silent duration / total duration.
                speech_samples = int(np.sum([b - a for a, b in intervals]))
                out["audio_speech_ratio"] = speech_samples / float(len(y))

                # Pause analysis: gaps between consecutive speech intervals.
                pauses = []
                for i in range(1, intervals.shape[0]):
                    gap_start = intervals[i - 1][1]
                    gap_end = intervals[i][0]
                    pause_sec = (gap_end - gap_start) / float(sr)
                    if pause_sec >= 0.5:  # ignore micro-pauses
                        pauses.append(pause_sec)
                if pauses:
                    out["audio_pause_count_per_min"] = len(pauses) * 60.0 / duration
                    out["audio_pause_mean_duration"] = float(np.mean(pauses))
                    out["audio_longest_pause"] = float(np.max(pauses))
        except Exception as exc:
            logger.warning("[audio] pause detection failed: %s", exc)

    except Exception as exc:
        logger.warning("[audio] extraction failed for %s: %s", video_path, exc)
    finally:
        if tmp_wav and os.path.exists(tmp_wav):
            try:
                os.unlink(tmp_wav)
            except Exception:
                pass

    logger.info(
        "[audio] voiced=%.2f speech=%.2f pauses/min=%.1f pitch_mean=%.1f "
        "pitch_std=%.1f vol_mean=%.4f vol_dr=%.1fdB",
        out["audio_voiced_ratio"], out["audio_speech_ratio"],
        out["audio_pause_count_per_min"], out["audio_pitch_mean"],
        out["audio_pitch_std"], out["audio_volume_mean"],
        out["audio_volume_dynamic_range"],
    )
    return out
