#!/usr/bin/env python3
"""
Irodori-TTS ゆうぷろカスタム V2.0.0 - ローカル版
Based on: IrodoriTTS_YuuproCustom_V2.0.0.ipynb
Original: ゆうぷろ (https://www.youtube.com/@yuupro) with Antigravity
Adapted for local use (no Google Drive)
"""

import sys
import os
import gc
import shutil
import socket
import subprocess as sp
import argparse
from pathlib import Path
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime

REPO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_DIR))
os.chdir(str(REPO_DIR))

import gradio as gr
import torch
from irodori_tts.inference_runtime import (
    RuntimeKey, SamplingRequest, clear_cached_runtime,
    default_runtime_device, download_hf_checkpoint, get_cached_runtime, save_wav,
)

# --- ローカル設定 ---
OUTPUT_DIR = str(REPO_DIR / "gradio_outputs")
REF_DIR = str(REPO_DIR / "ref_audio_presets")
DICT_FILE = str(REPO_DIR / "dictionary.txt")

for _d in [OUTPUT_DIR, REF_DIR]:
    Path(_d).mkdir(parents=True, exist_ok=True)

CODEC_REPO = "Aratako/Semantic-DACVAE-Japanese-32dim"

# --- モデル定義 ---
V4_REPO = "Aratako/Irodori-TTS-v4-Small"
V4_QUANT_REPO = "Aratako/Irodori-TTS-v4-Small-Quantized"
V3_BASE_REPO = "Aratako/Irodori-TTS-500M-v3"
V3_VD_REPO = "Aratako/Irodori-TTS-600M-v3-VoiceDesign"


@dataclass(frozen=True)
class ModelPreset:
    """UIから選べるモデル構成。

    v4-Small はクローンとボイスデザインが1つのチェックポイントに統合されているため、
    clone_source と design_source は同じになる（タブを切り替えても再ロードが起きない）。
    v3 は用途ごとにチェックポイントが分かれるため、タブ切替でモデルが載せ替わる。
    """

    clone_source: str
    design_source: str
    precision: str
    unified: bool  # True = v4系（キャプション＋参照音声の同時利用、複数参照音声に対応）
    note: str


MODEL_PRESETS = OrderedDict([
    ("v4-Small（統合・推奨）", ModelPreset(
        clone_source=V4_REPO, design_source=V4_REPO,
        precision="fp32", unified=True,
        note=(
            "**v4-Small（統合モデル / fp32・約3.1GB）** — クローンとボイスデザインが1モデルに統合され、"
            "参照音声とスタイル説明文を同時に使えます。参照音声は複数クリップを連結して最大120秒まで指定可能。"
            "<br>※同一話者の短いクリップを合計30秒程度並べるのが最も効果的です。"
            "参照が短い1クリップだけの場合は v3 のほうが似ることがあります。"
        ),
    )),
    ("v4-Small int8（低VRAM・実験的）", ModelPreset(
        clone_source=f"{V4_QUANT_REPO}/int8-weight-only",
        design_source=f"{V4_QUANT_REPO}/int8-weight-only",
        precision="bf16", unified=True,
        note=(
            "**v4-Small INT8 量子化版（約0.9GB / bf16推論）** — VRAMが少ない環境向け。"
            "初回選択時にダウンロードが走ります。"
            "<br>※torchao量子化モデルのため、環境によっては動作しない場合があります（実験的）。"
        ),
    )),
    ("v4-Small int4（最小VRAM・実験的）", ModelPreset(
        clone_source=f"{V4_QUANT_REPO}/int4-weight-only",
        design_source=f"{V4_QUANT_REPO}/int4-weight-only",
        precision="bf16", unified=True,
        note=(
            "**v4-Small INT4 量子化版（約0.85GB / bf16推論）** — 最もVRAM消費が少ない構成。"
            "初回選択時にダウンロードが走ります。"
            "<br>※CUDA tinygemmカーネルを使うため **Compute Capability 8.0以降（RTX 30系以降）のNVIDIA GPUが必須** です（実験的）。"
        ),
    )),
    ("v3（従来・短い参照音声向け）", ModelPreset(
        clone_source=V3_BASE_REPO, design_source=V3_VD_REPO,
        precision="fp32", unified=False,
        note=(
            "**v3（従来構成）** — クローンは Irodori-TTS-500M-v3、ボイスデザインは 600M-v3-VoiceDesign を使用。"
            "タブを切り替えるとモデルが載せ替わります（数十秒かかります）。"
            "<br>※参照音声は1つのみ・最大30秒。短い参照1クリップでのクローンは v4-Small より似る傾向があります。"
        ),
    )),
])

DEFAULT_MODEL = next(iter(MODEL_PRESETS))
_current_source: "tuple[str, str] | None" = None

# ffmpeg の確認
_FFMPEG_AVAILABLE = sp.run(
    ["ffmpeg", "-version"], capture_output=True
).returncode == 0
if not _FFMPEG_AVAILABLE:
    print("[warning] ffmpeg が見つかりません。話速コントロールは無効になります。")
    print("          https://ffmpeg.org/download.html からインストールしてPATHに追加してください。")

# --- 保存済み辞書の読込 ---
SAVED_DICT = ""
if os.path.exists(DICT_FILE):
    with open(DICT_FILE, "r", encoding="utf-8") as f:
        SAVED_DICT = f.read()

# --- 絵文字データ ---
EMOJI_GROUPS = [
    ("感情", [
        ("😊", "楽しげ"), ("😆", "喜び"), ("😭", "泣き"), ("😠", "怒り"), ("😲", "驚き"),
        ("🥺", "震え声"), ("😟", "心配"), ("😖", "苦しい"), ("🫣", "照れ"), ("🙄", "呆れ"),
        ("😌", "安堵"), ("🤔", "疑問"), ("😱", "悲鳴"),
    ]),
    ("話し方", [
        ("👂", "囁き"), ("😏", "からかい"), ("⏩", "早口"), ("🐢", "ゆっくり"), ("😪", "眠そう"),
        ("😰", "慌て"), ("🥴", "酔い"), ("🙏", "懇願"), ("🫶", "優しく"), ("🤐", "口塞ぎ"), ("🥵", "うめき声"),
    ]),
    ("効果音", [
        ("😮‍💨", "吐息"), ("🤭", "笑い"), ("🌬️", "息切れ"), ("😮", "息をのむ"),
        ("👅", "舐め音"), ("💋", "リップ"), ("🥤", "ゴクリ"), ("🤧", "咳"),
        ("😒", "舌打ち"), ("👌", "相槌"), ("🥱", "あくび"), ("🎵", "鼻歌"),
        ("⏸️", "間"), ("📢", "エコー"), ("📞", "電話"),
    ]),
]

# --- キャプションプリセット ---
CAPTION_PRESETS = OrderedDict([
    ("（プリセットを選択）", ""),
    ("─── 👩 女性 ───", None),
    ("🎀 落ち着いた女性", "落ち着いた女性の声で、近い距離感でやわらかく自然に話す。"),
    ("😊 明るい女性", "明るく元気な女性の声で、笑顔で話しているように。"),
    ("🌸 アニメ風女の子", "元気な女の子の声で、アニメの主人公のように話す。"),
    ("💕 ツンデレ", "ちょっと低めのツンデレな女の子の声。最初は強気だが、後半は照れて優しくなるように話す"),
    ("─── 👨 男性 ───", None),
    ("🎙️ 男性ナレーション", "低い男性の声で、落ち着いたナレーション風に話す。"),
    ("🧑 クールな男性", "クールで落ち着いた若い男性の声で、淡々と話す。"),
    ("👦 元気な男の子", "元気な男の子の声で、楽しそうに話す。"),
    ("👴 渋い中年男性", "渋い中年男性の声で、穏やかにゆっくりと話す。"),
    ("─── 🎭 その他 ───", None),
    ("🎭 ドラマチック朗読", "感情豊かに、ドラマチックに朗読する。"),
    ("📖 丁寧な朗読", "はっきりとした声で、丁寧に朗読する。"),
])

# ==========================================
# ユーティリティ関数
# ==========================================

def get_preset(model_choice):
    return MODEL_PRESETS.get(model_choice) or MODEL_PRESETS[DEFAULT_MODEL]


def _make_key(ckpt, precision):
    d = default_runtime_device()
    return RuntimeKey(
        checkpoint=ckpt, model_device=d, codec_repo=CODEC_REPO,
        model_precision=precision, codec_device=d, codec_precision="fp32",
        compile_model=False, compile_dynamic=False,
    )


def _ensure_model(source, precision):
    """チェックポイントを取得してランタイムを用意する。

    source は "owner/repo" もしくは "owner/repo/subfolder"（量子化版）。
    download_hf_checkpoint は model.safetensors に加えて同梱の tokenizer/ も取得するため、
    v4-Small では hf_hub_download ではなくこちらを使う必要がある。
    """
    global _current_source
    target = (source, precision)
    if _current_source != target:
        if _current_source is not None:
            print(f"[model] {_current_source[0]} → アンロード中...", flush=True)
            clear_cached_runtime()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        _current_source = target
    ckpt = download_hf_checkpoint(source)
    runtime, reloaded = get_cached_runtime(_make_key(ckpt, precision))
    if reloaded:
        print(f"[model] {source} ({precision}) ロード完了", flush=True)
    return runtime


def _save(result, prefix, do_save=True):
    target_dir = OUTPUT_DIR if do_save else str(REPO_DIR / "temp_outputs")
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return str(save_wav(
        Path(target_dir) / f"{prefix}_{stamp}.wav",
        result.audio.float(),
        result.sample_rate,
    ))


def apply_speed(path, rate):
    if abs(rate - 1.0) < 0.01 or not _FFMPEG_AVAILABLE:
        return path
    out = path.replace(".wav", "_speed.wav")
    sp.run(["ffmpeg", "-y", "-i", path, "-filter:a", f"atempo={rate}", out], capture_output=True)
    return out if os.path.exists(out) else path


def parse_dict(text):
    d = {}
    if not text:
        return d
    for line in text.strip().split("\n"):
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip()
        if k and v:
            d[k] = v
    return d


def apply_dict(text, dict_text):
    for k, v in parse_dict(dict_text).items():
        text = text.replace(k, v)
    return text


def save_dict_to_file(dict_text):
    with open(DICT_FILE, "w", encoding="utf-8") as f:
        f.write(dict_text or "")
    return f"✅ 辞書を保存しました: {DICT_FILE}"


# --- 参照音声プリセット ---

def list_ref_presets():
    Path(REF_DIR).mkdir(parents=True, exist_ok=True)
    return ["（選択なし）"] + sorted([f.stem for f in Path(REF_DIR).glob("*.wav")])


def save_ref_preset(audio_path, name):
    if not audio_path or not name or not name.strip():
        return gr.update(), "⚠️ 音声とプリセット名を入力してください"
    dst = Path(REF_DIR) / f"{name.strip()}.wav"
    shutil.copy2(audio_path, dst)
    return gr.update(choices=list_ref_presets()), f"✅ '{name.strip()}' を保存しました"


def delete_ref_preset(name):
    if not name or name == "（選択なし）":
        return gr.update(), "⚠️ 削除するプリセットを選択してください"
    p = Path(REF_DIR) / f"{name}.wav"
    if p.exists():
        p.unlink()
    return gr.update(choices=list_ref_presets(), value="（選択なし）"), f"✅ '{name}' を削除しました"


def load_ref_preset(name):
    if not name or name == "（選択なし）":
        return None
    p = Path(REF_DIR) / f"{name}.wav"
    return str(p) if p.exists() else None


def _on_t_schedule_mode_change(mode: str) -> object:
    return gr.update(interactive=str(mode).strip().lower() == "sway")


def _on_model_change(model_choice):
    """モデル選択に応じて、v4系のみで使える入力欄の表示を切り替える。"""
    preset = get_preset(model_choice)
    return (
        preset.note,
        gr.update(visible=preset.unified),  # b_extra_refs
        gr.update(visible=preset.unified),  # b_extra_note
        gr.update(visible=preset.unified),  # b_cap_group
        gr.update(visible=preset.unified),  # b_cfg_c
        gr.update(visible=preset.unified),  # v_extra_refs
    )


def _parse_optional_str(raw: str | None) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "" or text.lower() in {"none", "null", "off", "disable", "disabled", "base"}:
        return None
    return text


def _coerce_gradio_file_path(value: object) -> "str | None":
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, dict):
        for key in ("path", "name"):
            candidate = value.get(key)
            if candidate is not None and str(candidate).strip():
                return str(candidate)
        return None
    candidate = getattr(value, "name", None)
    if candidate is not None and str(candidate).strip():
        return str(candidate)
    text = str(value).strip()
    return text or None


def _resolve_ref_paths(main_audio, extra_files):
    """メイン参照音声＋追加参照音声を、指定順のパスリストにまとめる。

    v4-Small は各クリップを個別にエンコードしてからこの順で連結する。
    """
    paths = []
    main = _coerce_gradio_file_path(main_audio)
    if main is not None:
        paths.append(main)
    if extra_files:
        values = extra_files if isinstance(extra_files, (list, tuple)) else [extra_files]
        for value in values:
            path = _coerce_gradio_file_path(value)
            if path is not None:
                paths.append(path)
    return paths


def _supports_caption(runtime):
    return bool(getattr(runtime.model_cfg, "use_caption_condition", False))


# ==========================================
# 音声生成関数
# ==========================================

def generate_base(model_choice, text, caption, ref_audio, extra_refs, speaker_embed_file, speaker_embed_path_text, num_steps, cfg_t, cfg_c, cfg_s, seed_raw, speed, dict_text, save_audio, max_seconds, specified_seconds, dur_scale, t_schedule_mode, sway_coeff, lora_adapter_raw):
    if not text or not text.strip():
        raise gr.Error("テキストを入力してください")
    original = text.strip()
    processed = apply_dict(original, dict_text)
    preset = get_preset(model_choice)
    runtime = _ensure_model(preset.clone_source, preset.precision)
    # 追加参照音声とキャプションはv4系のみ。v3選択時はUIを隠すが、値は送られてくるので明示的に無視する。
    refs = _resolve_ref_paths(ref_audio, extra_refs if preset.unified else None)
    cap = caption.strip() if preset.unified and caption and caption.strip() else None
    if cap is not None and not _supports_caption(runtime):
        cap = None
    speaker_embed = _coerce_gradio_file_path(speaker_embed_file)
    if speaker_embed is None and speaker_embed_path_text and str(speaker_embed_path_text).strip():
        speaker_embed = str(speaker_embed_path_text).strip()
    elif speaker_embed is not None and speaker_embed_path_text and str(speaker_embed_path_text).strip():
        raise gr.Error("スピーカー埋め込みはアップロードとパス指定のどちらか一方のみ使用できます")
    if refs and speaker_embed is not None:
        raise gr.Error("参照音声とスピーカー埋め込みは同時に使用できません")
    no_ref = not refs and speaker_embed is None
    seed = None
    if seed_raw and seed_raw.strip():
        try:
            seed = int(seed_raw.strip())
        except ValueError:
            pass
    seconds = float(specified_seconds) if specified_seconds and float(specified_seconds) > 0 else None
    effective_max_seconds = float(max_seconds)
    if seconds is not None and seconds > effective_max_seconds:
        effective_max_seconds = seconds
    lora_adapter = _parse_optional_str(lora_adapter_raw)
    result = runtime.synthesize(SamplingRequest(
        text=processed, caption=cap,
        ref_wav=None, ref_wavs=refs or None,
        ref_latent=None, ref_embed=speaker_embed, no_ref=no_ref,
        ref_normalize_db=-16.0, ref_ensure_max=True,
        num_candidates=1, decode_mode="sequential",
        seconds=seconds, duration_scale=float(dur_scale),
        # None = チェックポイントの推奨値（v4-Small: 120秒 / v3以前: 30秒）
        max_ref_seconds=None, max_text_len=None, max_caption_len=None,
        max_seconds=effective_max_seconds,
        num_steps=int(num_steps), seed=seed,
        cfg_guidance_mode="independent",
        cfg_scale_text=float(cfg_t), cfg_scale_caption=float(cfg_c),
        cfg_scale_speaker=float(cfg_s),
        cfg_scale=None, cfg_min_t=0.5, cfg_max_t=1.0,
        truncation_factor=None, rescale_k=None, rescale_sigma=None,
        context_kv_cache=True,
        speaker_kv_scale=None, speaker_kv_min_t=None, speaker_kv_max_layers=None,
        t_schedule_mode=str(t_schedule_mode),
        sway_coeff=float(sway_coeff),
        trim_tail=True,
        lora_adapter=lora_adapter,
    ), log_fn=lambda m: print(m, flush=True))
    path = apply_speed(_save(result, "base", save_audio), speed)
    if speaker_embed is not None:
        mode = "埋め込みあり"
    elif len(refs) > 1:
        mode = f"参照{len(refs)}本"
    elif refs:
        mode = "参照あり"
    else:
        mode = "参照なし"
    if cap:
        mode += "/説明文あり"
    save_loc = "📁ローカル保存" if save_audio else "🗑️保存なし(一時表示)"
    info = f"🎤 {model_choice}({mode}) | Seed: {result.used_seed} | 生成: {result.total_to_decode:.1f}秒 | {save_loc}"
    if seconds is not None:
        info += f" | 指定長さ: {seconds:.1f}秒"
    elif abs(float(dur_scale) - 1.0) >= 0.01:
        info += f" | 時間スケール: {float(dur_scale):.2f}x"
    if abs(speed - 1.0) >= 0.01:
        info += f" | 話速: {speed:.1f}x" + ("" if _FFMPEG_AVAILABLE else " (ffmpeg未検出のため無効)")
    if original != processed:
        info += f"\n📖 辞書適用後: {processed}"
    return path, info


def generate_vd(model_choice, text, caption, ref_audio, extra_refs, num_steps, cfg_t, cfg_c, cfg_s, seed_raw, speed, dict_text, save_audio, t_schedule_mode, sway_coeff):
    if not text or not text.strip():
        raise gr.Error("テキストを入力してください")
    original = text.strip()
    processed = apply_dict(original, dict_text)
    preset = get_preset(model_choice)
    runtime = _ensure_model(preset.design_source, preset.precision)
    cap = caption.strip() if caption and caption.strip() else None
    # 追加参照音声はv4系のみ。v3選択時はUIを隠すが、値は送られてくるので明示的に無視する。
    refs = _resolve_ref_paths(ref_audio, extra_refs if preset.unified else None)
    no_ref = not refs or not runtime.model_cfg.use_speaker_condition_resolved
    if no_ref:
        refs = []
    seed = None
    if seed_raw and seed_raw.strip():
        try:
            seed = int(seed_raw.strip())
        except ValueError:
            pass
    result = runtime.synthesize(SamplingRequest(
        text=processed, caption=cap,
        ref_wav=None, ref_wavs=refs or None,
        ref_latent=None, no_ref=no_ref,
        ref_normalize_db=-16.0, ref_ensure_max=True,
        num_candidates=1, decode_mode="sequential",
        # None = チェックポイントの推奨値（v4-Small: 120秒 / v3以前: 30秒）
        seconds=None, max_ref_seconds=None,
        max_text_len=None, max_caption_len=None,
        num_steps=int(num_steps), seed=seed,
        cfg_guidance_mode="independent",
        cfg_scale_text=float(cfg_t), cfg_scale_caption=float(cfg_c),
        cfg_scale_speaker=0.0 if no_ref else float(cfg_s),
        cfg_scale=None, cfg_min_t=0.5, cfg_max_t=1.0,
        truncation_factor=None, rescale_k=None, rescale_sigma=None,
        context_kv_cache=True,
        speaker_kv_scale=None, speaker_kv_min_t=None, speaker_kv_max_layers=None,
        t_schedule_mode=str(t_schedule_mode),
        sway_coeff=float(sway_coeff),
        trim_tail=True,
    ), log_fn=lambda m: print(m, flush=True))
    path = apply_speed(_save(result, "vd", save_audio), speed)
    ci = "説明文あり" if cap else "説明文なし"
    if no_ref:
        speaker_mode = "参照なし"
    elif len(refs) > 1:
        speaker_mode = f"参照{len(refs)}本"
    else:
        speaker_mode = "参照あり"
    save_loc = "📁ローカル保存" if save_audio else "🗑️保存なし(一時表示)"
    info = f"🎨 {model_choice}({ci}/{speaker_mode}) | Seed: {result.used_seed} | 生成: {result.total_to_decode:.1f}秒 | {save_loc}"
    if abs(speed - 1.0) >= 0.01:
        info += f" | 話速: {speed:.1f}x" + ("" if _FFMPEG_AVAILABLE else " (ffmpeg未検出のため無効)")
    if original != processed:
        info += f"\n📖 辞書適用後: {processed}"
    return path, info


# --- 絵文字パレット ---

def create_emoji_palette(text_comp, elem_id):
    with gr.Accordion("😄 絵文字パレット（クリックでテキストに挿入）", open=False):
        for gn, emojis in EMOJI_GROUPS:
            gr.Markdown(f"**{gn}**")
            with gr.Row():
                for e, label in emojis:
                    btn = gr.Button(f"{e}{label}", size="sm", min_width=70)
                    js_code = f"""
                    function(text) {{
                        const el = document.querySelector('#{elem_id} textarea');
                        if (!el) return (text || "") + "{e}";
                        const start = el.selectionStart;
                        const end = el.selectionEnd;
                        const newText = (text || "").substring(0, start) + "{e}" + (text || "").substring(end);
                        setTimeout(() => {{
                            el.focus();
                            el.setSelectionRange(start + "{e}".length, start + "{e}".length);
                        }}, 100);
                        return newText;
                    }}
                    """
                    btn.click(fn=None, inputs=[text_comp], outputs=[text_comp], js=js_code)


# ==========================================
# Gradio UI
# ==========================================

def _get_private_ips():
    ips = []
    try:
        _, _, addr_list = socket.gethostbyname_ex(socket.gethostname())
        for ip in addr_list:
            parts = ip.split('.')
            if len(parts) != 4:
                continue
            try:
                a, b = int(parts[0]), int(parts[1])
                if a == 10 or (a == 172 and 16 <= b <= 31) or (a == 192 and b == 168):
                    ips.append(ip)
            except ValueError:
                pass
    except Exception:
        pass
    return ips


def build_ui():
    ffmpeg_status = "🎚️ 話速: 有効" if _FFMPEG_AVAILABLE else "🎚️ 話速: 無効(ffmpeg未検出)"
    speed_label_suffix = "" if _FFMPEG_AVAILABLE else " (ffmpeg未検出のため無効)"

    with gr.Blocks(title="Irodori-TTS ゆうぷろカスタム V2.0.0 [ローカル版]") as app:
        gr.Markdown("# 🎙️ Irodori-TTS ゆうぷろカスタム V2.0.0 [ローカル版]")
        gr.Markdown(
            f"🎤 ボイスクローン / 🎨 ボイスデザイン / 😄 絵文字 / "
            f"{ffmpeg_status} / 📖 辞書 / 📝 プリセット / 📁 ローカル保存モード"
        )

        m_model = gr.Radio(
            choices=list(MODEL_PRESETS.keys()), value=DEFAULT_MODEL,
            label="🧠 使用モデル",
        )
        m_model_note = gr.Markdown(MODEL_PRESETS[DEFAULT_MODEL].note)

        with gr.Tabs():
            # ===== ボイスクローン =====
            with gr.Tab("🎤 ボイスクローン"):
                gr.Markdown("### 参照音声から声をクローンして音声生成")
                with gr.Row():
                    with gr.Column(scale=3):
                        b_text = gr.Textbox(
                            label="📝 テキスト", lines=4,
                            placeholder="音声にしたいテキストを入力...", elem_id="b_text",
                        )
                        create_emoji_palette(b_text, "b_text")
                        with gr.Tabs():
                            with gr.Tab("🎤 参照音声"):
                                b_ref = gr.Audio(label="参照音声（任意）", type="filepath")
                                with gr.Accordion("📋 参照音声プリセット", open=False):
                                    with gr.Row():
                                        b_ref_dd = gr.Dropdown(
                                            choices=list_ref_presets(), value="（選択なし）",
                                            label="保存済みプリセット", scale=3,
                                        )
                                        b_ref_load = gr.Button("📂 使用", size="sm", scale=1)
                                    with gr.Row():
                                        b_ref_name = gr.Textbox(
                                            label="新規プリセット名", placeholder="名前を入力...", scale=3,
                                        )
                                        b_ref_save = gr.Button("💾 保存", size="sm", scale=1)
                                        b_ref_del = gr.Button("🗑️ 削除", size="sm", scale=1)
                                    b_ref_msg = gr.Textbox(label="ステータス", interactive=False)
                                    b_ref_load.click(load_ref_preset, [b_ref_dd], [b_ref])
                                    b_ref_save.click(save_ref_preset, [b_ref, b_ref_name], [b_ref_dd, b_ref_msg])
                                    b_ref_del.click(delete_ref_preset, [b_ref_dd], [b_ref_dd, b_ref_msg])
                                b_extra_refs = gr.File(
                                    label="➕ 追加の参照音声（任意・複数可／上の参照音声の後ろに並べた順で連結）",
                                    type="filepath", file_count="multiple",
                                    file_types=["audio"], allow_reordering=True,
                                    visible=MODEL_PRESETS[DEFAULT_MODEL].unified,
                                )
                                b_extra_note = gr.Markdown(
                                    "💡 同じ話者のノイズの少ない短いクリップを複数並べると似やすくなります"
                                    "（合計30秒程度が目安・最大120秒）。長い録音1本でも動きますが効果は未検証です。",
                                    visible=MODEL_PRESETS[DEFAULT_MODEL].unified,
                                )
                            with gr.Tab("🧬 スピーカー埋め込み"):
                                with gr.Row():
                                    b_speaker_embed_file = gr.File(
                                        label="埋め込みファイル (.speaker.safetensors, 任意)",
                                        type="filepath", file_count="single", scale=1,
                                    )
                                    b_speaker_embed_path = gr.Textbox(
                                        label="埋め込みパス (.speaker.safetensors, 任意)",
                                        value="", scale=1,
                                    )
                        with gr.Group(visible=MODEL_PRESETS[DEFAULT_MODEL].unified) as b_cap_group:
                            b_cap_preset = gr.Dropdown(
                                choices=list(CAPTION_PRESETS.keys()),
                                value="（プリセットを選択）",
                                label="📝 説明文プリセット（v4-Small は参照音声と説明文を併用できます）",
                            )
                            b_cap = gr.Textbox(
                                label="🎨 話し方・感情の説明文（任意）", lines=2,
                                placeholder="例: 深く傷つき、今にも泣き出しそうな様子。声が震えており、弱々しく話す。",
                            )
                            b_cap_preset.change(
                                lambda p: gr.update() if CAPTION_PRESETS.get(p) is None else CAPTION_PRESETS.get(p, ""),
                                inputs=[b_cap_preset], outputs=[b_cap],
                            )
                        b_speed = gr.Slider(
                            0.5, 2.0, 1.0, step=0.1,
                            label=f"🎚️ 話速{speed_label_suffix}",
                        )
                        b_cfg_t = gr.Slider(0.0, 10.0, 3.0, step=0.1, label="テキストと感情表現の効き具合 標準=3")
                        b_cfg_c = gr.Slider(
                            0.0, 10.0, 4.0, step=0.1, label="説明文の効き具合 標準=4",
                            visible=MODEL_PRESETS[DEFAULT_MODEL].unified,
                        )
                        b_save = gr.Checkbox(label="💾 生成音声をフォルダに自動保存する", value=True)
                        with gr.Accordion("⚙️ パラメータ", open=False):
                            with gr.Row():
                                b_steps = gr.Slider(1, 120, 40, step=1, label="Num Steps")
                                b_cfg_s = gr.Slider(0.0, 10.0, 5.0, step=0.1, label="CFG Speaker")
                            b_max_seconds = gr.Slider(
                                5, 120, 30, step=5,
                                label="⏱️ 最大生成時間 (秒) ※30秒超はトレーニング範囲外のため品質が低下する場合があります",
                            )
                            b_seconds = gr.Number(
                                value=0, minimum=0, maximum=300, step=0.5,
                                label="⏱️ 指定長さ (秒) ※0=自動（duration予測に従う）",
                            )
                            b_dur_scale = gr.Slider(
                                0.5, 1.5, 1.0, step=0.01,
                                label="⏱️ 生成時間スケール（Duration予測の倍率）※指定長さ=0の場合のみ有効",
                            )
                            b_seed = gr.Textbox(label="Seed（空欄=ランダム）", value="")
                            with gr.Row():
                                b_t_schedule = gr.Dropdown(
                                    label="Time Schedule",
                                    choices=["linear", "sway"],
                                    value="linear",
                                )
                                b_sway_coeff = gr.Slider(
                                    label="Sway Coeff",
                                    minimum=-1.0, maximum=1.5, value=-1.0, step=0.1,
                                    interactive=False,
                                )
                            b_lora_adapter = gr.Textbox(label="LoRA Adapter Directory (optional)", value="")
                        b_btn = gr.Button("🎵 音声を生成", variant="primary", size="lg")
                    with gr.Column(scale=2):
                        b_out = gr.Audio(label="🔈 生成音声", type="filepath")
                        b_info = gr.Textbox(label="ℹ️ 生成情報", interactive=False, lines=3)

            # ===== ボイスデザイン =====
            with gr.Tab("🎨 ボイスデザイン"):
                gr.Markdown("### テキスト指示で声をデザインして音声生成")
                with gr.Row():
                    with gr.Column(scale=3):
                        v_text = gr.Textbox(
                            label="📝 テキスト", lines=4,
                            placeholder="音声にしたいテキストを入力...", elem_id="v_text",
                        )
                        create_emoji_palette(v_text, "v_text")
                        v_preset = gr.Dropdown(
                            choices=list(CAPTION_PRESETS.keys()),
                            value="（プリセットを選択）",
                            label="📝 説明文プリセット",
                        )
                        v_cap = gr.Textbox(
                            label="🎨 作りたい音声スタイルの説明文", lines=3,
                            placeholder="どんな声で読むか指示...",
                        )
                        v_preset.change(
                            lambda p: gr.update() if CAPTION_PRESETS.get(p) is None else CAPTION_PRESETS.get(p, ""),
                            inputs=[v_preset], outputs=[v_cap],
                        )
                        v_ref = gr.Audio(
                            label="🎤 参照音声（任意、空欄=参照なしモード）",
                            type="filepath",
                        )
                        v_extra_refs = gr.File(
                            label="➕ 追加の参照音声（任意・複数可／上の参照音声の後ろに並べた順で連結）",
                            type="filepath", file_count="multiple",
                            file_types=["audio"], allow_reordering=True,
                            visible=MODEL_PRESETS[DEFAULT_MODEL].unified,
                        )
                        v_speed = gr.Slider(
                            0.5, 2.0, 1.0, step=0.1,
                            label=f"🎚️ 話速{speed_label_suffix}",
                        )
                        v_cfg_t = gr.Slider(0.0, 10.0, 3.0, step=0.1, label="テキストと感情表現の効き具合 標準=3")
                        v_cfg_c = gr.Slider(0.0, 10.0, 4.0, step=0.1, label="音声スタイルの効き具合 標準=4")
                        v_save = gr.Checkbox(label="💾 生成音声をフォルダに自動保存する", value=True)
                        with gr.Accordion("⚙️ パラメータ", open=False):
                            with gr.Row():
                                v_steps = gr.Slider(1, 120, 40, step=1, label="Num Steps")
                                v_cfg_s = gr.Slider(0.0, 10.0, 5.0, step=0.1, label="CFG Speaker")
                            v_seed = gr.Textbox(label="Seed（空欄=ランダム）", value="")
                            with gr.Row():
                                v_t_schedule = gr.Dropdown(
                                    label="Time Schedule",
                                    choices=["linear", "sway"],
                                    value="linear",
                                )
                                v_sway_coeff = gr.Slider(
                                    label="Sway Coeff",
                                    minimum=-1.0, maximum=1.5, value=-1.0, step=0.1,
                                    interactive=False,
                                )
                        v_btn = gr.Button("🎵 音声を生成", variant="primary", size="lg")
                    with gr.Column(scale=2):
                        v_out = gr.Audio(label="🔈 生成音声", type="filepath")
                        v_info = gr.Textbox(label="ℹ️ 生成情報", interactive=False, lines=3)

            # ===== 読み辞書 =====
            with gr.Tab("📖 読み辞書"):
                gr.Markdown("### 📖 読み辞書")
                gr.Markdown("誤読する単語を登録 → 生成時に自動変換。`変換前=変換後` 形式で1行1エントリ")
                dict_input = gr.Textbox(
                    label="辞書エントリ",
                    placeholder="例:\nIrodori=イロドリ\n彩音=あやね\nTTS=ティーティーエス",
                    lines=12, value=SAVED_DICT,
                )
                dict_save_btn = gr.Button("💾 辞書を保存（ローカルファイルに保存）", size="sm")
                dict_status = gr.Textbox(label="ステータス", interactive=False)
                dict_save_btn.click(save_dict_to_file, [dict_input], [dict_status])
                gr.Markdown("💡 辞書はすべてのタブの音声生成に自動で適用されます")
                gr.Markdown(f"📁 保存先: `{DICT_FILE}`")

        b_t_schedule.change(
            _on_t_schedule_mode_change, inputs=[b_t_schedule], outputs=[b_sway_coeff]
        )
        v_t_schedule.change(
            _on_t_schedule_mode_change, inputs=[v_t_schedule], outputs=[v_sway_coeff]
        )
        m_model.change(
            _on_model_change, inputs=[m_model],
            outputs=[m_model_note, b_extra_refs, b_extra_note, b_cap_group, b_cfg_c, v_extra_refs],
        )

        b_btn.click(
            generate_base,
            [m_model, b_text, b_cap, b_ref, b_extra_refs, b_speaker_embed_file, b_speaker_embed_path, b_steps, b_cfg_t, b_cfg_c, b_cfg_s, b_seed, b_speed, dict_input, b_save, b_max_seconds, b_seconds, b_dur_scale, b_t_schedule, b_sway_coeff, b_lora_adapter],
            [b_out, b_info],
        )
        v_btn.click(
            generate_vd,
            [m_model, v_text, v_cap, v_ref, v_extra_refs, v_steps, v_cfg_t, v_cfg_c, v_cfg_s, v_seed, v_speed, dict_input, v_save, v_t_schedule, v_sway_coeff],
            [v_out, v_info],
        )

        gr.Markdown("---\n📜 コード・モデル: MIT License | [Irodori-TTS](https://github.com/Aratako/Irodori-TTS)")

    return app


def main():
    parser = argparse.ArgumentParser(
        description="Irodori-TTS ゆうぷろカスタム V2.0.0 [ローカル版]"
    )
    parser.add_argument("--server-name", default="127.0.0.1", help="サーバーアドレス (デフォルト: 127.0.0.1)")
    parser.add_argument("--server-port", type=int, default=7860, help="ポート番号 (デフォルト: 7860)")
    parser.add_argument("--share", action="store_true", help="Gradio 共有リンクを有効化")
    parser.add_argument("--debug", action="store_true", help="デバッグモードを有効化")
    parser.add_argument("--watermark", action="store_true", help="silentcipher ウォーターマーク埋込を有効化（デフォルト: 無効）")
    args = parser.parse_args()

    if not args.watermark:
        os.environ["IRODORI_NO_WATERMARK"] = "1"

    print("=" * 60)
    print("🎙️ Irodori-TTS ゆうぷろカスタム V2.0.0 [ローカル版]")
    print(f"   🧠 既定モデル : {DEFAULT_MODEL}")
    print(f"   📁 出力先     : {OUTPUT_DIR}")
    print(f"   🎤 参照音声   : {REF_DIR}")
    print(f"   📖 辞書       : {DICT_FILE}")
    print(f"   🎚️ ffmpeg     : {'利用可能' if _FFMPEG_AVAILABLE else '未検出（話速無効）'}")
    print("=" * 60)
    _private_ips = _get_private_ips() if args.server_name == "0.0.0.0" else []
    print()
    print("=" * 60)
    print(f"  URL (ローカル) : http://127.0.0.1:{args.server_port}")
    for _ip in _private_ips:
        print(f"  URL (LAN)       : http://{_ip}:{args.server_port}")
    print("  終了 : Ctrl+C")
    if args.server_name == "0.0.0.0":
        print("  注意 : 全ネットワークインターフェースで受信中 (0.0.0.0)")
        print("         ファイアウォール設定を確認してください")
    print("=" * 60)
    print()

    app = build_ui()
    app.queue(default_concurrency_limit=1)
    app.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        debug=args.debug,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
