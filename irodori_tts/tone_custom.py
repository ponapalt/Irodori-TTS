"""'カスタム' モード: スライダー/選択項目から自然な1つのcaptionを合成する。

複数の調整項目（感情・明るさ・声の高さ・話速・声量・抑揚・息の多さ・感情強度）を
単純に文字列連結するのではなく、閾値ごとに自然な日本語フレーズへ変換してから
1つの矛盾のない文へ結合する。矛盾する指定（例: 「悲しい」と「明るさ最大」）が
同時に選択された場合は、最後に変更した方を優先し、警告メッセージを返す。

依存先は標準ライブラリのみ（torch/gradio 非依存）。
"""

from __future__ import annotations

from dataclasses import dataclass

# --- 感情（カスタムモードの簡易プルダウン用） ---
# (id, 表示名, フレーズ, 明るさ極性: -1=暗い系 / 0=中立 / +1=明るい系)
EMOTION_CHOICES: list[tuple[str, str, str, int]] = [
    ("none", "指定なし", "", 0),
    ("glad", "嬉しい", "喜びを隠しきれないような", 1),
    ("fun", "楽しい", "楽しそうに軽快な", 1),
    ("excited", "わくわく", "期待に胸を弾ませているような", 1),
    ("gentle", "優しい", "相手を安心させるように柔らかく優しい", 1),
    ("lonely", "寂しい", "寂しさをにじませた静かな", -1),
    ("sad", "悲しい", "悲しみをこらえるような", -1),
    ("dark", "暗い", "暗く陰鬱な雰囲気の", -1),
    ("angry", "怒り", "怒りを込めて強い", -1),
    ("anxious", "不安", "不安を感じているような", -1),
    ("surprised", "驚き", "思いがけない出来事に驚いたような", 0),
    ("serious", "真剣", "集中した真剣な", 0),
]
_EMOTION_BY_ID = {e[0]: e for e in EMOTION_CHOICES}

_THRESHOLD = 0.35


def _axis_phrase(value: float, low_word: str, high_word: str, threshold: float = _THRESHOLD) -> str:
    if value >= threshold:
        return high_word
    if value <= -threshold:
        return low_word
    return ""


def _brightness_phrase(v: float) -> str:
    return _axis_phrase(v, "暗く沈んだ調子にして", "明るく元気な調子にして")


def _pitch_phrase(v: float) -> str:
    return _axis_phrase(v, "声を低めにして", "声を高めにして")


def _speed_phrase(v: float) -> str:
    return _axis_phrase(v, "話速をゆっくりにして", "話速を速めにして")


def _volume_phrase(v: float) -> str:
    return _axis_phrase(v, "声量を小さめにして", "声量を大きめにして")


def _intonation_phrase(v: float) -> str:
    if v >= 0.66:
        return "抑揚を大きくつけて"
    if v <= 0.2:
        return "抑揚を抑えて"
    return ""


def _breathiness_phrase(v: float) -> str:
    if v >= 0.66:
        return "息を多めに含ませて"
    if v <= 0.15:
        return "息を抑えて"
    return ""


@dataclass(frozen=True)
class CustomCaptionResult:
    caption: str
    warnings: list[str]


def build_custom_caption(
    *,
    base_text: str = "",
    emotion: str = "none",
    brightness: float = 0.0,
    pitch: float = 0.0,
    speed: float = 0.0,
    volume: float = 0.0,
    intonation: float = 0.5,
    breathiness: float = 0.3,
    intensity: float = 0.5,
    last_changed: str | None = None,
) -> CustomCaptionResult:
    """Compose the 8 custom-mode axes into a single natural-language caption.

    ``last_changed`` should be one of "emotion"/"brightness"/... (whichever
    control the caller just changed) so that a contradiction between the
    emotion choice and the brightness slider can be resolved deterministically
    (last-changed wins) instead of silently picking one arbitrarily.
    """
    warnings: list[str] = []
    emo_id, emo_label, emo_phrase, emo_polarity = _EMOTION_BY_ID.get(
        emotion, _EMOTION_BY_ID["none"]
    )
    brightness_eff = float(brightness)

    contradicts = (
        emo_polarity != 0
        and abs(brightness_eff) >= _THRESHOLD
        and (emo_polarity > 0) != (brightness_eff > 0)
    )
    if contradicts:
        # Default precedence when we don't know which was touched last:
        # the explicit numeric slider wins over the coarse emotion category.
        loser = "emotion" if last_changed != "emotion" else "brightness"
        if loser == "brightness":
            warnings.append(
                f"「{emo_label}」と明るさの指定が矛盾していたため、"
                f"明るさの指定を無視して「{emo_label}」を優先しました。"
            )
            brightness_eff = 0.0
        else:
            warnings.append(
                f"「{emo_label}」と明るさの指定が矛盾していたため、"
                f"明るさの指定を優先して「{emo_label}」の指定を無視しました。"
            )
            emo_phrase = ""

    fragments = [
        emo_phrase,
        _brightness_phrase(brightness_eff),
        _pitch_phrase(pitch),
        _speed_phrase(speed),
        _volume_phrase(volume),
        _intonation_phrase(intonation),
        _breathiness_phrase(breathiness),
    ]
    fragments = [f for f in fragments if f]

    if intensity >= 0.75 and emo_phrase:
        fragments.insert(0, "感情を強く出して")
    elif intensity <= 0.25 and emo_phrase:
        fragments.insert(0, "感情を控えめにして")

    if fragments:
        body = "、".join(fragments) + "話す。"
    else:
        body = "自然な口調で話す。"

    caption = "参照音声の話者の声質と声色を保ったまま、" + body
    if base_text and base_text.strip():
        caption = caption.rstrip("。") + "。" + base_text.strip()

    return CustomCaptionResult(caption=caption, warnings=warnings)
