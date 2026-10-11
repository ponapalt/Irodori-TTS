"""ゆうぷろカスタム UI（gradio_app_yuupro.py）の補助機能。

ローカル設定・保存先/ファイル名・前回入力の復元・内部音声の整理・配色テーマ・再生音量・
Windows 終了時の接続リセット抑制をまとめる。EasyIrodoriTTS v1.2 の同名機能を、
ボイスクローン/ボイスデザインの2タブ構成に合わせて移植したもの。

生成直後の音声は常に temp_outputs（内部領域）に作り、自動保存が有効なときだけ
設定した保存先へコピーする。画面の再生・ダウンロードには、保存名と同じ名前の内部コピーを使う。
"""

from __future__ import annotations

import base64
import functools
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import uuid
from datetime import datetime
from pathlib import Path

import gradio as gr

REPO_DIR = Path(__file__).resolve().parent.parent
TEMP_DIR = REPO_DIR / "temp_outputs"
DEFAULT_EXPORT_DIR = REPO_DIR / "gradio_outputs"
SETTINGS_FILE = REPO_DIR / "yuupro_settings.json"
PREVIOUS_INPUT_FILE = REPO_DIR / "yuupro_previous_input.json"

DEFAULT_PATTERN = "$種別$_$日付$_$連番$"
FILENAME_TOKENS = ("$連番$", "$日付$", "$テキスト$", "$種別$")
DEFAULT_SETTINGS = dict(
    theme="system",
    auto_browser=False,
    restore_input=False,
    custom_output=False,
    output_dir="",
    custom_pattern=False,
    pattern=DEFAULT_PATTERN,
    txt=False,
)

_LOCK = threading.RLock()
_INPUT_LOCK = threading.Lock()
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


# ==========================================
# 設定・JSON
# ==========================================

def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        print(f"[warning] {Path(path).name} を読み込めませんでした: {exc}", flush=True)
        return None


def _atomic_write(path, data):
    path = Path(path)
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def read_settings():
    result = dict(DEFAULT_SETTINGS)
    saved = _read_json(SETTINGS_FILE)
    if isinstance(saved, dict):
        for key, default in DEFAULT_SETTINGS.items():
            if key in saved and isinstance(saved[key], type(default)):
                result[key] = saved[key]
    return result


def save_settings(values):
    """設定を検証して保存し、ステータス欄に出す文言を返す（不正な値は保存せず理由を返す）。"""
    with _LOCK:
        result = read_settings()
        for key, default in DEFAULT_SETTINGS.items():
            if key in values:
                if not isinstance(values[key], type(default)):
                    return "⚠️ 設定値が不正です。保存していません。"
                result[key] = values[key]
        if result["theme"] not in ("light", "dark", "system"):
            return "⚠️ 配色が不正です。保存していません。"
        if result["custom_output"] and not Path(result["output_dir"]).is_absolute():
            return "⚠️ 保存先はドライブ名から指定してください。保存していません。"
        if not result["pattern"].strip():
            return "⚠️ ファイル名を指定してください。保存していません。"
        _atomic_write(SETTINGS_FILE, result)
    return "設定を保存しました。ブラウザ自動起動の変更は次回起動から有効です。"


def _powershell(script, *, encoding="utf-8-sig"):
    """PowerShell を STA で実行して標準出力を返す。スクリプトは UTF-16LE の Base64 で渡す。"""
    if os.name != "nt":
        raise gr.Error("この機能は Windows のみ対応しています。")
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    proc = subprocess.run(
        ["powershell.exe", "-NoProfile", "-STA", "-ExecutionPolicy", "Bypass", "-EncodedCommand", encoded],
        capture_output=True, check=True, creationflags=_NO_WINDOW,
    )
    return proc.stdout.decode(encoding, errors="replace")


_FOLDER_DIALOG_PS = """
Add-Type -AssemblyName System.Windows.Forms
$d = New-Object System.Windows.Forms.FolderBrowserDialog
$d.Description = 'Select the output folder'
if ($d.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
    [Console]::OutputEncoding = [Text.Encoding]::UTF8
    Write-Output $d.SelectedPath
}
$d.Dispose()
"""


def pick_folder():
    selected = _powershell(_FOLDER_DIALOG_PS).strip()
    return selected or gr.skip()


# ==========================================
# 保存先・ファイル名
# ==========================================

def safe_name(value):
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(value)).strip().rstrip(". ")
    value = value[:100].rstrip(". ") or "音声"
    if re.match(r"^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)", value, re.I):
        value = "_" + value
    return value


def build_filename(settings, text, kind, ordinal, date):
    pattern = settings["pattern"] if settings["custom_pattern"] else DEFAULT_PATTERN
    tokens = {"$連番$": f"{ordinal:03}", "$日付$": date, "$テキスト$": str(text)[:40], "$種別$": kind}
    return safe_name(re.sub(r"\$(?:連番|日付|テキスト|種別)\$", lambda m: tokens[m.group()], pattern))


def filename_example(custom_pattern, pattern):
    settings = dict(custom_pattern=custom_pattern, pattern=pattern)
    return build_filename(settings, "こんにちは", "base", 1, datetime.now().strftime("%Y%m%d_%H%M%S")) + ".wav"


def _export_dir(settings):
    folder = Path(settings["output_dir"]) if settings["custom_output"] else DEFAULT_EXPORT_DIR
    if not folder.is_absolute():
        raise ValueError("保存先は絶対パスで指定してください。")
    return folder


def export_audio(source, text, settings, kind, ordinal, date):
    """保存先へ衝突なしでコピーし、{'wav': パス, 'txt': パス, 'error': 文言} を返す。"""
    folder = _export_dir(settings)
    folder.mkdir(parents=True, exist_ok=True)
    result = {}
    base = build_filename(settings, text, kind, ordinal, date)
    with _LOCK:
        index = 0
        while True:
            wav = folder / (base + (f"_{index:03}" if index else "") + ".wav")
            if wav.with_suffix(".txt").exists():
                index += 1
                continue
            try:
                # 排他作成で名前を確保する（別プロセスとも衝突しない）。
                with wav.open("xb") as dest, Path(source).open("rb") as src:
                    shutil.copyfileobj(src, dest)
                result["wav"] = str(wav)
                break
            except FileExistsError:
                index += 1
            except Exception:
                wav.unlink(missing_ok=True)
                raise
        if settings["txt"]:
            txt = wav.with_suffix(".txt")
            try:
                with txt.open("x", encoding="utf-8-sig") as stream:
                    stream.write(text)
                result["txt"] = str(txt)
            except FileExistsError:
                result["error"] = "TXTが同名で存在します。保存先を確認してください。"
            except OSError as exc:
                result["error"] = f"WAVは保存済みですがTXT保存に失敗しました: {exc}"
    return result


def publish_candidates(paths, text, kind, do_save):
    """生成済みの候補を保存名つきの内部コピーに移し、(表示用パス, 保存結果の文言リスト) を返す。

    paths は temp_outputs 内のファイルで、ここで内部コピーへ移動する。
    do_save が真なら設定どおりの保存先へも書き出す。
    """
    settings = read_settings()
    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = TEMP_DIR / f"download_{uuid.uuid4().hex}"
    display, notes = [], []
    for ordinal, path in enumerate(paths, start=1):
        name = build_filename(settings, text, kind, ordinal, date) + ".wav"
        if do_save:
            try:
                result = export_audio(path, text, settings, kind, ordinal, date)
                if result.get("wav"):
                    name = Path(result["wav"]).name
                    notes.append(result["wav"])
                if result.get("error"):
                    notes.append(result["error"])
            except Exception as exc:
                print(f"[warning] 自動保存に失敗しました: {exc}", flush=True)
                notes.append(f"自動保存に失敗しました: {exc}")
        # 候補ごとにフォルダを分けて、同名でも衝突しないようにする。
        target = folder / str(ordinal) / name
        target.parent.mkdir(parents=True)
        shutil.move(path, target)
        display.append(str(target))
    return display, notes


# ==========================================
# 前回入力の復元
# ==========================================

def remember(key, value):
    if not read_settings()["restore_input"]:
        return
    with _INPUT_LOCK:
        saved = _read_json(PREVIOUS_INPUT_FILE)
        saved = saved if isinstance(saved, dict) else {}
        if key in saved and saved[key] == value:
            return
        saved[key] = value
        _atomic_write(PREVIOUS_INPUT_FILE, saved)


def previous_input():
    if not read_settings()["restore_input"]:
        return {}
    saved = _read_json(PREVIOUS_INPUT_FILE)
    return saved if isinstance(saved, dict) else {}


def attach_input_memory(components):
    """各入力欄の変更を保存する。テキスト欄は入力のたびに保存しないよう、フォーカスが外れたときに保存する。"""
    for key, component in components.items():
        event = component.blur if isinstance(component, gr.Textbox) else component.change
        event(
            functools.partial(remember, key), inputs=component, queue=False,
            show_progress="hidden", trigger_mode="always_last", api_name=False,
        )


def restore_updates(keys, allowed=None):
    """keys の順に、保存済みの値を戻す更新を返す。allowed は {key: 許可する値の集合}。"""
    saved = previous_input()
    updates = []
    for key in keys:
        if key not in saved or (allowed and key in allowed and saved[key] not in allowed[key]):
            updates.append(gr.skip())
        else:
            updates.append(gr.update(value=saved[key]))
    return updates


# ==========================================
# 内部音声の整理
# ==========================================

def _inventory():
    root = TEMP_DIR.resolve()
    records = []
    if root.exists():
        for path in root.rglob("*.wav"):
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                continue
            stat = path.stat()
            records.append(dict(path=str(path), name=str(path.relative_to(root)), size=stat.st_size, mtime=stat.st_mtime_ns))
    records.sort(key=lambda r: r["path"])
    return records, hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()


def cleanup_preview():
    records, stamp = _inventory()
    total = sum(r["size"] for r in records) / 1024 / 1024
    listing = f"対象：{len(records)}件 / {total:.1f} MB\n" + "\n".join(r["name"] for r in records)
    return listing, stamp, gr.update(interactive=bool(records))


_RECYCLE_PS = r"""
$ErrorActionPreference = 'Stop'
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
[ComImport,Guid("43826D1E-E718-42EE-BC55-A1E261C37BFE"),InterfaceType(ComInterfaceType.InterfaceIsIUnknown)] interface IShellItem {}
[ComImport,Guid("947AAB5F-0A5C-4C13-B4D6-4BF7836FC9F8"),InterfaceType(ComInterfaceType.InterfaceIsIUnknown)] interface IFileOperation {
 void Advise(IntPtr sink,out uint cookie); void Unadvise(uint cookie); void SetOperationFlags(uint flags);
 void SetProgressMessage([MarshalAs(UnmanagedType.LPWStr)]string message); void SetProgressDialog(IntPtr dialog); void SetProperties(IntPtr props); void SetOwnerWindow(uint hwnd);
 void ApplyPropertiesToItem(IShellItem item); void ApplyPropertiesToItems(IntPtr items);
 void RenameItem(IShellItem item,[MarshalAs(UnmanagedType.LPWStr)]string name,IntPtr sink); void RenameItems(IntPtr items,[MarshalAs(UnmanagedType.LPWStr)]string name);
 void MoveItem(IShellItem item,IShellItem destination,[MarshalAs(UnmanagedType.LPWStr)]string name,IntPtr sink); void MoveItems(IntPtr items,IShellItem destination);
 void CopyItem(IShellItem item,IShellItem destination,[MarshalAs(UnmanagedType.LPWStr)]string name,IntPtr sink); void CopyItems(IntPtr items,IShellItem destination);
 void DeleteItem(IShellItem item,IntPtr sink); void DeleteItems(IntPtr items);
 void NewItem(IShellItem destination,uint attrs,[MarshalAs(UnmanagedType.LPWStr)]string name,[MarshalAs(UnmanagedType.LPWStr)]string template,IntPtr sink);
 void PerformOperations(); void GetAnyOperationsAborted([MarshalAs(UnmanagedType.Bool)]out bool aborted);
}
public class YuuproRecycle {
 [DllImport("shell32.dll",CharSet=CharSet.Unicode,PreserveSig=false)] static extern void SHCreateItemFromParsingName(string path,IntPtr ctx,ref Guid iid,out IShellItem item);
 public static void Run(string[] paths) {
  var op=(IFileOperation)Activator.CreateInstance(Type.GetTypeFromCLSID(new Guid("3AD05575-8857-4850-9277-11B85BDB8E09")));
  try {
   // ごみ箱へ送れない場合は失敗扱いにする（完全削除へのフォールバックはしない）。
   op.SetOperationFlags(0x00080000|0x00100000|0x00000040|0x00000010|0x00000400|0x00000004);
   foreach(var path in paths) { Guid id=new Guid("43826D1E-E718-42EE-BC55-A1E261C37BFE"); IShellItem item; SHCreateItemFromParsingName(path,IntPtr.Zero,ref id,out item); try { op.DeleteItem(item,IntPtr.Zero); } finally {Marshal.ReleaseComObject(item);} }
   op.PerformOperations();bool aborted;op.GetAnyOperationsAborted(out aborted);if(aborted)throw new Exception("Recycle operation was cancelled.");
  } finally {Marshal.ReleaseComObject(op);}
 }
}
'@
$paths = Get-Content -LiteralPath $ListFile -Raw -Encoding UTF8 | ConvertFrom-Json
[YuuproRecycle]::Run([string[]]$paths)
"""


def cleanup(stamp):
    records, current = _inventory()
    if current != stamp:
        return "対象が変わりました。もう一度一覧を確認してください。", None, gr.update(interactive=False)
    if not records:
        return "整理する音声はありません。", None, gr.update(interactive=False)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", dir=TEMP_DIR, encoding="utf-8", delete=False) as f:
        json.dump([r["path"] for r in records], f)
        manifest = Path(f.name)
    try:
        script = f"$ListFile = '{str(manifest).replace(chr(39), chr(39) * 2)}'\n" + _RECYCLE_PS
        _powershell(script)
    except subprocess.CalledProcessError:
        return "ごみ箱への移動を完了できませんでした。対象を再確認してください。", None, gr.update(interactive=False)
    finally:
        manifest.unlink(missing_ok=True)
    # 空になった download_* フォルダを片付ける（TEMP_DIR 自体は残す）。
    for folder in sorted((p for p in TEMP_DIR.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        try:
            folder.rmdir()
        except OSError:
            pass
    moved = sum(not Path(r["path"]).exists() for r in records)
    return f"{moved}件をごみ箱へ移動しました。", None, gr.update(interactive=False)


# ==========================================
# 配色テーマ・再生音量（フロントエンド）
# ==========================================

THEME_JS = """(choice)=>{
    if(window.yuuproThemeCleanup) window.yuuproThemeCleanup();
    const media=window.matchMedia('(prefers-color-scheme: dark)');
    const apply=()=>{
        const dark=choice==='dark'||(choice==='system'&&media.matches);
        const targets=new Set([document.documentElement,document.body,...document.querySelectorAll('gradio-app,.gradio-container')]);
        document.querySelectorAll('.gradio-container').forEach(root=>{
            for(let node=root.parentElement;node;node=node.parentElement)targets.add(node);
        });
        targets.forEach(root=>{
            if(root.classList.contains('dark')!==dark)root.classList.toggle('dark',dark);
            root.style.colorScheme=dark?'dark':'light';
        });
    };
    const observer=new MutationObserver(apply);
    observer.observe(document.body,{childList:true,subtree:true,attributes:true,attributeFilter:['class']});
    media.addEventListener('change',apply);
    apply();
    window.yuuproThemeCleanup=()=>{observer.disconnect();media.removeEventListener('change',apply);};
}"""

# 再生用の音量。保存する WAV の音量は変えない。対象のプレイヤーは class="yuupro-player"。
VOLUME_HTML = """
<div title="このブラウザに記憶します。保存する WAV の音量は変わりません">
<label for="yuupro-volume-slider">🔊 再生音量：<output id="yuupro-volume-value">75</output>％</label>
<input id="yuupro-volume-slider" type="range" min="0" max="100" step="1" value="75"
 style="width:100%" aria-label="再生音量（WAVの音量は変わりません）">
</div>
"""

VOLUME_JS = """() => {
    if (window.yuuproVolumeCleanup) window.yuuproVolumeCleanup();
    const key = 'yuupro-playback-volume';
    let volume = 0.75;
    try {
        const saved = localStorage.getItem(key);
        const number = Number(saved);
        if (saved !== null && saved.trim() !== '' && Number.isFinite(number)
            && number >= 0 && number <= 1) volume = number;
    } catch (_) { /* プライベートブラウズ等では保存できない */ }
    const controller = new AbortController();
    const options = {signal: controller.signal};
    const bound = new WeakSet();
    const sources = new WeakMap();
    const sliders = new WeakSet();
    const observedRoots = new WeakSet();
    const resetPosition = audio => {
        audio.pause();
        try { audio.currentTime = 0; } catch (_) {}
        // Gradio の再生状態にも伝える。
        audio.dispatchEvent(new Event('timeupdate'));
    };
    const players = () => {
        const found = [];
        const visit = scope => {
            found.push(...scope.querySelectorAll('audio'));
            scope.querySelectorAll('*').forEach(element => {
                if (!element.shadowRoot) return;
                if (!observedRoots.has(element.shadowRoot)) {
                    observedRoots.add(element.shadowRoot);
                    observer.observe(element.shadowRoot, {
                        childList:true, subtree:true, attributes:true, attributeFilter:['src']
                    });
                }
                visit(element.shadowRoot);
            });
        };
        document.querySelectorAll('.yuupro-player').forEach(visit);
        return found;
    };
    const apply = audio => { if (audio.volume !== volume) audio.volume = volume; };
    const refresh = () => {
        const slider = document.getElementById('yuupro-volume-slider');
        if (slider) {
            slider.value = String(Math.round(volume * 100));
            if (!sliders.has(slider)) {
                sliders.add(slider);
                slider.addEventListener('input', () => {
                    volume = Number(slider.value) / 100;
                    try { localStorage.setItem(key, String(volume)); } catch (_) {}
                    players().forEach(apply);
                    refresh();
                }, options);
            }
        }
        const label = document.getElementById('yuupro-volume-value');
        const text = String(Math.round(volume * 100));
        if (label && label.textContent !== text) label.textContent = text;
        players().forEach(audio => {
            apply(audio);
            const source = audio.getAttribute('src') || audio.querySelector('source')?.getAttribute('src') || '';
            if (sources.get(audio) !== source) { sources.set(audio, source); resetPosition(audio); }
            if (bound.has(audio)) return;
            bound.add(audio);
            audio.addEventListener('loadedmetadata', () => resetPosition(audio), options);
            // プレイヤー側の音量リセットよりも、専用スライダーの値を優先する。
            ['volumechange', 'loadedmetadata', 'play', 'emptied'].forEach(event =>
                audio.addEventListener(event, () => apply(audio), options));
        });
        document.querySelectorAll('.yuupro-player button[aria-label="Adjust volume"]')
            .forEach(button => { button.style.display = 'none'; });
    };
    const observer = new MutationObserver(refresh);
    observer.observe(document.body, {childList:true, subtree:true, attributes:true,
                                    attributeFilter:['src']});
    refresh();
    window.yuuproVolumeCleanup = () => {observer.disconnect(); controller.abort();};
}
"""


# ==========================================
# オプション設定タブ
# ==========================================

def build_options_tab():
    """「オプション設定」タブの中身を組み立て、設定項目の {名前: コンポーネント} を返す。

    gr.Tab のコンテキスト内で呼ぶこと。
    """
    settings = read_settings()
    controls = {}
    gr.Markdown("## ⚙️ オプション設定\n変更後は「設定を保存」を押してください。")
    with gr.Group():
        gr.Markdown("### 🎨 画面の配色")
        controls["theme"] = gr.Dropdown(
            [("ライト", "light"), ("ダーク", "dark"), ("PCの設定に合わせる", "system")],
            value=settings["theme"], label="配色", info="選ぶとすぐに試せます。保存すると次回にも反映します。",
        )
    with gr.Group():
        gr.Markdown("### 🚀 起動時の設定")
        controls["auto_browser"] = gr.Checkbox(settings["auto_browser"], label="起動時にブラウザを開く（次回起動から有効）")
        controls["restore_input"] = gr.Checkbox(
            settings["restore_input"],
            label="前回の入力を復元する",
            info="テキスト・説明文・各パラメータを次回起動時に戻します。参照音声は再選択してください。"
                 "ONの間、入力内容が yuupro_previous_input.json に保存されます。",
        )
    with gr.Group():
        gr.Markdown("### 📁 音声の保存先\n各タブの「生成音声をフォルダに自動保存する」がONのときの保存先です。OFFでも試聴とダウンロードはできます。")
        controls["custom_output"] = gr.Checkbox(settings["custom_output"], label=f"保存先を指定する（OFF：{DEFAULT_EXPORT_DIR.name}）")
        with gr.Row():
            controls["output_dir"] = gr.Textbox(settings["output_dir"], label="保存先フォルダ（ドライブ名から指定）", scale=4)
            pick = gr.Button("📂 フォルダを選ぶ", scale=1)
        pick.click(pick_folder, outputs=controls["output_dir"], queue=False, api_name=False)
    with gr.Group():
        gr.Markdown("### 🏷️ 保存するファイル名")
        controls["custom_pattern"] = gr.Checkbox(settings["custom_pattern"], label=f"名前の付け方を指定する（OFF：{DEFAULT_PATTERN}）")
        controls["pattern"] = gr.Textbox(settings["pattern"], label="名前の付け方")
        example = gr.Textbox(value=filename_example(settings["custom_pattern"], settings["pattern"]), label="名前の例", interactive=False)
        with gr.Row():
            for token in FILENAME_TOKENS:
                button = gr.Button(token, size="sm")
                button.click(
                    lambda value, t=token: value + ("" if not value or value.endswith(("_", " ")) else "_") + t,
                    controls["pattern"], controls["pattern"], queue=False, api_name=False,
                )
        for name in ("custom_pattern", "pattern"):
            controls[name].change(
                filename_example, [controls["custom_pattern"], controls["pattern"]], example,
                queue=False, api_name=False,
            )
        controls["txt"] = gr.Checkbox(settings["txt"], label="原文をTXTでも自動保存する（辞書適用前の文章）")
    with gr.Group():
        gr.Markdown(
            f"### 🧹 内部音声の整理\n画面の再生・ダウンロード用に作った内部音声（{TEMP_DIR.name}）を確認して、ごみ箱へ移動します。"
            f"必要な音声は先に保存してください。保存先（{DEFAULT_EXPORT_DIR.name} など）・参照音声プリセットは対象外です。"
        )
        inspect_button = gr.Button("対象の一覧と容量を確認")
        listing = gr.Textbox(label="整理対象", interactive=False, lines=5)
        stamp = gr.State(None)
        trash = gr.Button("確認した音声をごみ箱へ移動", interactive=False)
        inspect_button.click(cleanup_preview, outputs=[listing, stamp, trash], queue=False, api_name=False)
        # 生成・モデル解放と同じ concurrency_id で直列化し、生成中の音声を整理しないようにする。
        trash.click(cleanup, stamp, [listing, stamp, trash], concurrency_id="synthesis", api_name=False)
    status = gr.Markdown("保存済み")
    save_button = gr.Button("💾 設定を保存", variant="primary")
    for control in controls.values():
        control.change(lambda: "変更あり・未保存", outputs=status, queue=False, api_name=False)
    controls["theme"].change(fn=None, inputs=controls["theme"], outputs=None, js=THEME_JS, queue=False)
    names = list(controls)
    save_button.click(
        lambda *values: save_settings(dict(zip(names, values))),
        list(controls.values()), status, queue=False, api_name=False,
    )
    return controls


# ==========================================
# Windows: 終了時の接続リセットの抑制
# ==========================================

def install_connection_reset_filter():
    """asyncio の終了処理で出る、Windows 特有の接続リセット(10054)のログだけを隠す。"""
    if os.name != "nt":
        return
    import logging
    from asyncio.proactor_events import _ProactorBasePipeTransport

    teardown_code = _ProactorBasePipeTransport._call_connection_lost.__code__

    class ConnectionResetFilter(logging.Filter):
        _yuupro_teardown_filter = True

        def filter(self, record):
            if not record.exc_info:
                return True
            _, error, traceback = record.exc_info
            if not isinstance(error, ConnectionResetError) or getattr(error, "winerror", None) != 10054:
                return True
            if traceback is None:
                return True
            while traceback.tb_next is not None:
                traceback = traceback.tb_next
            # 実際のコードオブジェクトで比較し、生成・ダウンロード等で起きた同種の例外は隠さない。
            return traceback.tb_frame.f_code is not teardown_code

    logger = logging.getLogger("asyncio")
    if not any(getattr(f, "_yuupro_teardown_filter", False) for f in logger.filters):
        logger.addFilter(ConnectionResetFilter())
