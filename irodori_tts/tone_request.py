"""Pure request-shaping helpers for the VoiceDesign (caption) generation path.

These build the exact keyword arguments that get forwarded to
``irodori_tts.inference_runtime.SamplingRequest`` for a VoiceDesign call, and
are shared by both the main "generate" button and the "試聴" (preview) button
in gradio_app_yuupro.py so their request shapes can never diverge.

Deliberately free of torch/gradio imports so it is unit-testable without the
full inference stack.
"""

from __future__ import annotations

CAPTION_UNSUPPORTED_WARNING = (
    "⚠️ このモデルはキャプション（声のトーン・感情プリセット）に対応していません。"
    "選択したトーン/感情は無視されます。VoiceDesign対応モデルを使用してください。"
)


def check_caption_support(use_caption_condition: bool, caption: str | None) -> str | None:
    """Return a warning message if a caption was requested but the loaded
    model checkpoint does not support caption conditioning at all, else None.
    """
    if caption and str(caption).strip() and not use_caption_condition:
        return CAPTION_UNSUPPORTED_WARNING
    return None


def build_vd_request_kwargs(
    *,
    text: str,
    caption: str | None,
    ref_wavs: list[str] | None = None,
    speaker_condition_supported: bool = True,
    cfg_scale_text: float = 3.0,
    cfg_scale_caption: float = 4.0,
    cfg_scale_speaker: float = 5.0,
    num_steps: int = 40,
    seed: int | None = None,
    t_schedule_mode: str = "linear",
    sway_coeff: float = -1.0,
) -> dict:
    """Build the SamplingRequest kwargs for a VoiceDesign generation call.

    The read-aloud ``text`` and the ``caption`` (tone/emotion instruction) are
    always independent: neither is derived from, nor clears, the other, and
    when a reference audio is supplied it is always included in the same
    dict alongside the caption (never one or the other).

    ``ref_wavs`` accepts the full list of reference clips (v4-Small では複数の
    参照音声を連結できる)。空リスト／None、または話者条件を持たないモデルの
    場合は参照なしモードになる。
    """
    cap = caption.strip() if caption and str(caption).strip() else None
    refs = [str(r) for r in (ref_wavs or []) if r and str(r).strip()]
    no_ref = (not refs) or (not speaker_condition_supported)
    if no_ref:
        refs = []

    return {
        "text": text,
        "caption": cap,
        "ref_wav": None,
        "ref_wavs": refs or None,
        "ref_latent": None,
        "no_ref": no_ref,
        "ref_normalize_db": -16.0,
        "ref_ensure_max": True,
        "num_candidates": 1,
        "decode_mode": "sequential",
        # None = チェックポイントの推奨値（v4-Small: 120秒 / v3以前: 30秒）
        "seconds": None,
        "max_ref_seconds": None,
        "max_text_len": None,
        "max_caption_len": None,
        "num_steps": int(num_steps),
        "seed": seed,
        "cfg_guidance_mode": "independent",
        "cfg_scale_text": float(cfg_scale_text),
        "cfg_scale_caption": float(cfg_scale_caption),
        "cfg_scale_speaker": 0.0 if no_ref else float(cfg_scale_speaker),
        "cfg_scale": None,
        "cfg_min_t": 0.5,
        "cfg_max_t": 1.0,
        "truncation_factor": None,
        "rescale_k": None,
        "rescale_sigma": None,
        "context_kv_cache": True,
        "speaker_kv_scale": None,
        "speaker_kv_min_t": None,
        "speaker_kv_max_layers": None,
        "t_schedule_mode": str(t_schedule_mode),
        "sway_coeff": float(sway_coeff),
        "trim_tail": True,
    }
