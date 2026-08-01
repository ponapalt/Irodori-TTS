"""Tone / emotion preset registry for voice-design captions.

This is the single source of truth for every built-in "声のトーン・感情プリセット".
Adding a new :class:`TonePreset` to :data:`TONE_PRESETS` is enough to make it show
up everywhere presets are consumed (category listings, search, dropdown choices) -
no UI code or if/switch statements need to change.

This module intentionally has zero third-party dependencies (no torch, no gradio)
so it can be imported and unit tested in isolation from the heavy inference stack.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

# --- カテゴリ ---
# UI 上でプリセットを絞り込むためのカテゴリ一覧（表示順を保持）。
CATEGORIES: OrderedDict[str, str] = OrderedDict(
    [
        ("basic", "基本"),
        ("bright", "明るい・楽しい"),
        ("kind", "優しい・好意的"),
        ("sad", "悲しい・暗い"),
        ("angry", "怒り・強い感情"),
        ("anxious", "不安・緊張"),
        ("acting", "演技・用途別"),
    ]
)

# UI の「カスタム」モードを表す擬似 ID（TONE_PRESETS には含まれない）。
CUSTOM_PRESET_ID = "custom"


@dataclass(frozen=True)
class TonePreset:
    """1つの声のトーン・感情プリセット。"""

    id: str
    name: str
    category: str
    caption: str
    description: str
    recommended_speed: float = 1.0
    recommended_volume: float = 1.0
    recommended_pitch: float = 0.0
    enabled: bool = True


def _p(
    id: str,
    name: str,
    category: str,
    caption: str,
    description: str,
    speed: float = 1.0,
    volume: float = 1.0,
    pitch: float = 0.0,
    enabled: bool = True,
) -> TonePreset:
    return TonePreset(
        id=id,
        name=name,
        category=category,
        caption=caption,
        description=description,
        recommended_speed=speed,
        recommended_volume=volume,
        recommended_pitch=pitch,
        enabled=enabled,
    )


# recommended_volume / recommended_pitch はモデル・API に音量/ピッチを直接操作する
# パラメータが存在しないため、UI 上の参考表示（および将来の拡張用メタデータ）に留まる。
# recommended_speed のみ既存の話速スライダーの初期値提案として実際に利用できる。
TONE_PRESETS: list[TonePreset] = [
    # ===== 基本 =====
    _p(
        "normal", "通常", "basic",
        "参照音声の話者の声質と声色を保ったまま、自然で聞き取りやすく、落ち着いた口調で話す。過度な感情表現は加えない。",
        "自然で落ち着いた基本トーン。過度な感情表現なし。",
        speed=1.0, volume=1.0, pitch=0.0,
    ),
    _p(
        "bright", "明るい", "basic",
        "参照音声の話者の声質と声色を保ったまま、明るく元気で親しみやすく話す。声には笑顔を感じさせ、少し高めのトーンで、ハキハキと軽快に話す。",
        "明るく親しみやすい、少し高めで軽快なトーン。",
        speed=1.05, volume=1.05, pitch=0.08,
    ),
    _p(
        "energetic", "元気", "basic",
        "参照音声の話者の声質を保ったまま、活力にあふれた大きめの声で、テンポよく勢いをつけて話す。抑揚を大きめにする。",
        "活力にあふれ、テンポよく勢いのあるトーン。",
        speed=1.15, volume=1.15, pitch=0.05,
    ),
    _p(
        "calm", "落ち着いた", "basic",
        "参照音声の話者の声質を保ったまま、穏やかで安定した声で話す。話速はややゆっくりにし、抑揚は自然に抑える。",
        "穏やかで安定した、ややゆっくりのトーン。",
        speed=0.9, volume=0.95, pitch=-0.05,
    ),
    # ===== 明るい・楽しい =====
    _p(
        "glad", "嬉しい", "bright",
        "参照音声の話者の声質を保ったまま、喜びを隠しきれないような声で話す。声を少し弾ませ、明るい抑揚をつける。",
        "喜びを隠しきれない、弾んだ明るいトーン。",
        speed=1.05, volume=1.05, pitch=0.08,
    ),
    _p(
        "fun", "楽しい", "bright",
        "参照音声の話者の声質を保ったまま、楽しそうに軽快に話す。言葉にリズムをつけ、自然な笑顔を感じる声にする。",
        "リズムよく軽快で、笑顔を感じるトーン。",
        speed=1.1, volume=1.05, pitch=0.05,
    ),
    _p(
        "excited", "わくわく", "bright",
        "参照音声の話者の声質を保ったまま、期待に胸を弾ませているように話す。少し早口で、語尾を明るく弾ませる。",
        "期待に胸を弾ませる、少し早口なトーン。",
        speed=1.15, volume=1.05, pitch=0.08,
    ),
    # ===== 優しい・好意的 =====
    _p(
        "gentle", "優しい", "kind",
        "参照音声の話者の声質を保ったまま、相手を安心させるように柔らかく優しく話す。声量は控えめで、温かい響きにする。",
        "柔らかく優しい、控えめで温かいトーン。",
        speed=0.95, volume=0.85, pitch=0.0,
    ),
    _p(
        "sweet", "甘い", "kind",
        "参照音声の話者の声質を保ったまま、柔らかく甘えるような声で話す。距離感を近くし、語尾を優しく伸ばすが、過剰に演技的にはしない。",
        "柔らかく甘えるような、距離の近いトーン。",
        speed=0.9, volume=0.85, pitch=0.03,
    ),
    _p(
        "shy", "照れ", "kind",
        "参照音声の話者の声質を保ったまま、少し恥ずかしそうに話す。声をやや小さくし、ときどき言葉に迷うような自然な間を入れる。",
        "少し恥ずかしそうで、間のある小さめの声。",
        speed=0.95, volume=0.8, pitch=0.02,
    ),
    _p(
        "relieved", "安心", "kind",
        "参照音声の話者の声質を保ったまま、包み込むような穏やかな声で話す。ゆっくりと丁寧に、聞き手を落ち着かせるようにする。",
        "包み込むような、ゆっくり丁寧なトーン。",
        speed=0.9, volume=0.85, pitch=-0.05,
    ),
    # ===== 悲しい・暗い =====
    _p(
        "lonely", "寂しい", "sad",
        "参照音声の話者の声質と声色を保ったまま、寂しさをにじませて静かに話す。声量はやや小さく、ゆっくりしたテンポで、言葉の間を少し長めに取る。わずかに声を震わせるが、泣き声にはしすぎない。",
        "寂しさをにじませ、間を長めに取る静かなトーン。",
        speed=0.85, volume=0.8, pitch=-0.05,
    ),
    _p(
        "sad", "悲しい", "sad",
        "参照音声の話者の声質を保ったまま、悲しみをこらえながら話す。声を弱くし、抑揚を控えめにして、語尾を少し落とす。",
        "悲しみをこらえる、抑揚控えめなトーン。",
        speed=0.85, volume=0.75, pitch=-0.05,
    ),
    _p(
        "tearful", "泣きそう", "sad",
        "参照音声の話者の声質を保ったまま、涙をこらえているように話す。声をわずかに震わせ、息を詰まらせるような間を自然に入れる。ただし完全な泣き声にはしない。",
        "涙をこらえる、わずかに震えるトーン。",
        speed=0.85, volume=0.75, pitch=0.0,
    ),
    _p(
        "dark", "暗い", "sad",
        "参照音声の話者の声質と声色を保ったまま、暗く陰鬱な雰囲気で話す。低めで重いトーンにし、感情の起伏と抑揚を抑え、ゆっくり冷たく話す。怒り声やうなり声にはしない。",
        "低く重い、陰鬱で抑揚の少ないトーン。",
        speed=0.85, volume=0.75, pitch=-0.1,
    ),
    _p(
        "despair", "絶望", "sad",
        "参照音声の話者の声質を保ったまま、力を失い、希望をなくしたように話す。声量を小さくし、低く重い声で、長めの間を入れる。",
        "力を失い、低く重い、長めの間のトーン。",
        speed=0.8, volume=0.7, pitch=-0.1,
    ),
    _p(
        "tired", "疲れた", "sad",
        "参照音声の話者の声質を保ったまま、疲労がにじむ声で話す。声に力を入れすぎず、やや遅いテンポで、息を多めに含ませる。",
        "疲労がにじむ、やや遅く息多めのトーン。",
        speed=0.85, volume=0.8, pitch=-0.05,
    ),
    _p(
        "sleepy", "眠そう", "sad",
        "参照音声の話者の声質を保ったまま、眠気を感じる柔らかく力の抜けた声で話す。話速は遅めにし、語尾をゆるやかに落とす。",
        "眠気を感じる、力の抜けた遅めのトーン。",
        speed=0.8, volume=0.75, pitch=-0.05,
    ),
    # ===== 怒り・強い感情 =====
    _p(
        "angry", "怒り", "angry",
        "参照音声の話者の声質を保ったまま、怒りを込めて強く話す。声量と語気を強め、短く鋭い抑揚をつける。ただし音割れするような叫び声にはしない。",
        "怒りを込めて強く、鋭い抑揚のトーン。",
        speed=1.1, volume=1.2, pitch=0.0,
    ),
    _p(
        "cold_anger", "静かな怒り", "angry",
        "参照音声の話者の声質を保ったまま、感情を押し殺した低く冷たい声で話す。話速を遅くし、一語ずつ強調する。",
        "感情を押し殺した、低く冷たいトーン。",
        speed=0.85, volume=0.9, pitch=-0.1,
    ),
    _p(
        "irritated", "苛立ち", "angry",
        "参照音声の話者の声質を保ったまま、少し不機嫌で苛立った口調で話す。テンポをやや速くし、語尾を強める。",
        "不機嫌で苛立った、やや速いトーン。",
        speed=1.1, volume=1.05, pitch=0.0,
    ),
    _p(
        "confident_strong", "強気", "angry",
        "参照音声の話者の声質を保ったまま、自信があり堂々とした強い口調で話す。声を前に出し、はっきりと言い切る。",
        "自信があり堂々とした、はっきりしたトーン。",
        speed=1.05, volume=1.1, pitch=0.0,
    ),
    _p(
        "cold", "冷たい", "angry",
        "参照音声の話者の声質を保ったまま、感情を見せず冷淡に話す。抑揚を小さくし、一定のテンポと低めの声で話す。",
        "感情を見せない、抑揚の小さい冷淡なトーン。",
        speed=0.95, volume=0.9, pitch=-0.05,
    ),
    _p(
        "intimidating", "威圧的", "angry",
        "参照音声の話者の声質を保ったまま、相手に強い圧を与えるような低く重い声で話す。ゆっくりと、一語ずつ明確に発音する。ただし叫ばない。",
        "強い圧を与える、低く重いゆっくりなトーン。",
        speed=0.85, volume=1.05, pitch=-0.1,
    ),
    # ===== 不安・緊張 =====
    _p(
        "anxious", "不安", "anxious",
        "参照音声の話者の声質を保ったまま、不安を感じているように話す。声を少し弱くし、語尾を揺らし、迷うような短い間を入れる。",
        "語尾が揺れる、不安げなトーン。",
        speed=1.0, volume=0.85, pitch=0.0,
    ),
    _p(
        "scared", "怖がっている", "anxious",
        "参照音声の話者の声質を保ったまま、恐怖を感じているように小さく震える声で話す。息を浅くし、やや早口にする。",
        "小さく震える、恐怖を感じるトーン。",
        speed=1.1, volume=0.8, pitch=0.05,
    ),
    _p(
        "nervous", "緊張", "anxious",
        "参照音声の話者の声質を保ったまま、緊張で少し硬くなった声で話す。発音を丁寧にし、不自然にならない程度に間を入れる。",
        "少し硬くなった、丁寧な発音のトーン。",
        speed=0.95, volume=0.9, pitch=0.0,
    ),
    _p(
        "hurried", "焦り", "anxious",
        "参照音声の話者の声質を保ったまま、時間に追われて焦っているように話す。話速を速め、息継ぎを短くし、語尾を急がせる。",
        "時間に追われる、速く息継ぎの短いトーン。",
        speed=1.2, volume=1.0, pitch=0.05,
    ),
    _p(
        "surprised", "驚き", "anxious",
        "参照音声の話者の声質を保ったまま、思いがけない出来事に驚いたように話す。冒頭を強くし、声の高さと抑揚を大きく変化させる。",
        "冒頭が強く、抑揚が大きく変化するトーン。",
        speed=1.1, volume=1.05, pitch=0.1,
    ),
    _p(
        "confused", "困惑", "anxious",
        "参照音声の話者の声質を保ったまま、状況を理解できず戸惑っているように話す。語尾を少し上げ、考えながら話すような間を入れる。",
        "戸惑い、考えながら話すようなトーン。",
        speed=1.0, volume=0.9, pitch=0.03,
    ),
    # ===== 演技・用途別 =====
    _p(
        "whisper", "ささやき", "acting",
        "参照音声の話者の声質を保ったまま、聞き手の近くで話すような小さなささやき声にする。息を多めに含ませ、音量を控えめにする。",
        "近くで話すような、息多めのささやき声。",
        speed=0.9, volume=0.5, pitch=0.0,
    ),
    _p(
        "secret", "内緒話", "acting",
        "参照音声の話者の声質を保ったまま、秘密を打ち明けるように声を抑えて話す。親密で少し楽しそうな雰囲気を含ませる。",
        "秘密を打ち明けるような、親密で控えめな声。",
        speed=0.95, volume=0.6, pitch=0.0,
    ),
    _p(
        "narration", "ナレーション", "acting",
        "参照音声の話者の声質を保ったまま、明瞭で聞き取りやすいナレーション調で話す。発音をはっきりさせ、安定した話速と適度な抑揚をつける。",
        "明瞭で聞き取りやすいナレーション調。",
        speed=1.0, volume=1.0, pitch=0.0,
    ),
    _p(
        "news", "ニュース", "acting",
        "参照音声の話者の声質を保ったまま、正確で落ち着いたニュース読みの口調で話す。感情を抑え、一定のテンポではっきり発音する。",
        "正確で落ち着いたニュース読みの口調。",
        speed=1.0, volume=1.0, pitch=0.0,
    ),
    _p(
        "reading", "朗読", "acting",
        "参照音声の話者の声質を保ったまま、物語を丁寧に読み聞かせるように話す。文章の意味に合わせて自然な間と抑揚をつける。",
        "物語を丁寧に読み聞かせるようなトーン。",
        speed=0.95, volume=1.0, pitch=0.0,
    ),
    _p(
        "anime", "アニメ風", "acting",
        "参照音声の話者の声質を保ったまま、感情表現と抑揚をやや大きくした、明瞭で表情豊かな演技調で話す。過剰に甲高くはしない。",
        "感情表現豊かな、表情のある演技調。",
        speed=1.05, volume=1.05, pitch=0.05,
    ),
    _p(
        "emotionless", "無感情", "acting",
        "参照音声の話者の声質を保ったまま、感情をほとんど出さず、一定の高さ、速さ、音量で淡々と話す。",
        "感情をほとんど出さない、淡々としたトーン。",
        speed=1.0, volume=0.9, pitch=0.0,
    ),
    _p(
        "mechanical", "機械的", "acting",
        "参照音声の話者の声質をある程度保ちながら、規則的で機械的なリズムで話す。抑揚を抑え、文節ごとの間隔を一定にする。",
        "規則的で機械的なリズムのトーン。",
        speed=1.0, volume=0.9, pitch=0.0,
    ),
    _p(
        "sexy", "色っぽい", "acting",
        "参照音声の話者の声質を保ったまま、落ち着いた低めの声で、大人っぽく余裕のある話し方にする。話速をやや遅くし、息を自然に含ませる。過剰な性的表現にはしない。",
        "落ち着いた低めで、大人っぽく余裕のあるトーン。",
        speed=0.85, volume=0.85, pitch=-0.05,
    ),
    _p(
        "serious", "真剣", "acting",
        "参照音声の話者の声質を保ったまま、集中した真剣な口調で話す。余計な抑揚を抑え、重要な言葉をはっきり強調する。",
        "集中した真剣な口調。重要な語を強調。",
        speed=0.95, volume=1.0, pitch=-0.05,
    ),
    _p(
        "self_assured", "自信あり", "acting",
        "参照音声の話者の声質を保ったまま、自信と余裕を感じさせる声で話す。安定した音量で、迷いなくはっきりと言い切る。",
        "自信と余裕を感じさせる、迷いのないトーン。",
        speed=1.0, volume=1.05, pitch=0.0,
    ),
]


_PRESETS_BY_ID: dict[str, TonePreset] = {p.id: p for p in TONE_PRESETS}


def get_preset(preset_id: str | None, presets: list[TonePreset] | None = None) -> TonePreset | None:
    """Look up a preset by id. Returns None if not found (or id is falsy)."""
    if not preset_id:
        return None
    if presets is None:
        return _PRESETS_BY_ID.get(preset_id)
    for p in presets:
        if p.id == preset_id:
            return p
    return None


def enabled_presets(presets: list[TonePreset] | None = None) -> list[TonePreset]:
    return [p for p in (presets if presets is not None else TONE_PRESETS) if p.enabled]


def presets_by_category(
    category_id: str, presets: list[TonePreset] | None = None
) -> list[TonePreset]:
    return [p for p in enabled_presets(presets) if p.category == category_id]


def category_label(category_id: str) -> str:
    return CATEGORIES.get(category_id, category_id)
