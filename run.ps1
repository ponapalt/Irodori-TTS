#Requires -Version 5.1
# このファイルは UTF-8 BOM 付き (UTF-8 with BOM) で保存してください。
# BOM なしで保存すると日本語文字の文字コードエラーが発生します。

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

$RepoDir = $PSScriptRoot

Write-Host ""
Write-Host "============================================================"
Write-Host "  Irodori-TTS ゆうぷろカスタム V2.0.0 [ローカル版]"
Write-Host "============================================================"
Write-Host ""

# ============================================================
# Step 1: uv 確認
# ============================================================
Write-Host "[Step 1/4] uv を確認中..."

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "  uv が見つかりません。自動インストールします..."
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "[ERROR] uv のインストールに失敗しました。"
        Write-Host "        手動でインストールしてください:"
        Write-Host "        powershell -ExecutionPolicy ByPass -c `"irm https://astral.sh/uv/install.ps1 | iex`""
        Write-Host ""
        Read-Host "続行するには Enter キーを押してください"
        exit 1
    }
    # インストール直後は PATH に含まれていないため、既定の場所を直接追加する
    $uvBin = Join-Path $env:USERPROFILE '.local\bin'
    if (Test-Path $uvBin) {
        $env:PATH = "$uvBin;$env:PATH"
    }
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Write-Host ""
        Write-Host "[ERROR] uv のインストール後もコマンドが見つかりません。"
        Write-Host "        ターミナルを再起動してから再度お試しください。"
        Write-Host ""
        Read-Host "続行するには Enter キーを押してください"
        exit 1
    }
    Write-Host "  インストール完了。"
}
$uvVer = uv --version 2>&1
Write-Host "  $uvVer"
Write-Host ""

# ============================================================
# Step 2: ffmpeg
# ============================================================
Write-Host "[Step 2/4] ffmpeg を確認中..."

if (Get-Command ffmpeg -ErrorAction SilentlyContinue) {
    Write-Host "  ffmpeg は既にインストール済みです。スキップします。"
} else {
    Write-Host "  ffmpeg が見つかりません。winget でインストールします..."
    winget install --id Gyan.FFmpeg --source winget --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "[ERROR] ffmpeg のインストールに失敗しました。"
        Write-Host "        手動でインストールしてください: https://ffmpeg.org/download.html"
        Write-Host ""
        Read-Host "続行するには Enter キーを押してください"
        exit 1
    }
    Write-Host "  インストール完了。環境変数 PATH を再読み込みします..."
    $machinePath = [System.Environment]::GetEnvironmentVariable('PATH', [System.EnvironmentVariableTarget]::Machine)
    $userPath    = [System.Environment]::GetEnvironmentVariable('PATH', [System.EnvironmentVariableTarget]::User)
    $env:PATH    = (@($machinePath, $userPath) | Where-Object { $_ }) -join ';'
    $verifyVer   = & ffmpeg '-version' 2>&1 | Select-Object -First 1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  確認済み: $verifyVer"
    } else {
        Write-Host "  注意: ffmpeg は次回のターミナル起動後に有効になります。"
    }
}
Write-Host ""

# ============================================================
# Step 3: 仮想環境 (uv venv)
# ============================================================
Write-Host "[Step 3/4] 仮想環境を確認中..."

$venvDir = Join-Path $RepoDir '.venv'
if (Test-Path $venvDir) {
    Write-Host "  .venv は既に存在します。スキップします。"
} else {
    Write-Host "  .venv を作成します (uv venv)..."
    Set-Location $RepoDir
    uv venv --python 3.10
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "[ERROR] 仮想環境の作成に失敗しました。"
        Write-Host "        Python 3.10 以上がインストールされているか確認してください。"
        Read-Host "続行するには Enter キーを押してください"
        exit 1
    }
    Write-Host "  仮想環境を作成しました。"
}
Write-Host ""

# ============================================================
# Step 4: 依存パッケージ (uv sync)
# ============================================================
Write-Host "[Step 4/4] 依存パッケージを確認中 (uv sync)..."
Write-Host "  ※ 初回は PyTorch (CUDA 12.8) を含む全パッケージをダウンロードします"
Write-Host "  ※ CPU 環境や別の CUDA バージョンの場合は pyproject.toml を編集してください"
Write-Host "     参考: https://pytorch.org/get-started/locally/"
Write-Host ""

Set-Location $RepoDir
uv sync --no-dev --extra cu128
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[ERROR] uv sync に失敗しました。"
    Write-Host "        ネットワーク環境や CUDA バージョンを確認してください。"
    Read-Host "続行するには Enter キーを押してください"
    exit 1
}
Write-Host ""

# ============================================================
# モデルの事前ダウンロード（未取得分のみ）
# ============================================================
Write-Host "============================================================"
Write-Host "  モデルキャッシュを確認中"
Write-Host "============================================================"

$pyScript = Join-Path $env:TEMP 'irodori_dl.py'
@'
from huggingface_hub import hf_hub_download, try_to_load_from_cache
from huggingface_hub.errors import LocalEntryNotFoundError

MODELS = [
    ("ベースモデル",  "Aratako/Irodori-TTS-500M-v3",            "model.safetensors"),
    ("VoiceDesign", "Aratako/Irodori-TTS-600M-v3-VoiceDesign", "model.safetensors"),
]

missing = []
for label, repo_id, filename in MODELS:
    cached = try_to_load_from_cache(repo_id=repo_id, filename=filename)
    if isinstance(cached, str):
        print(f"  [OK]   {label} ({repo_id}) は取得済み")
    else:
        print(f"  [MISS] {label} ({repo_id}) は未取得")
        missing.append((label, repo_id, filename))

if not missing:
    print("  すべてのモデルが取得済みです。")
else:
    print("")
    print(f"  未取得のモデルをダウンロードします ({len(missing)}件)... 数分かかります")
    for i, (label, repo_id, filename) in enumerate(missing, 1):
        print(f"  [{i}/{len(missing)}] {label} ({repo_id})...")
        p = hf_hub_download(repo_id=repo_id, filename=filename)
        print(f"          完了: {p}")
    print("  ダウンロード完了！")
'@ | Set-Content $pyScript -Encoding UTF8
uv run --no-sync python $pyScript
Remove-Item $pyScript -Force -ErrorAction SilentlyContinue
Write-Host ""

# ============================================================
# アプリ起動
# ============================================================
$ServerHost = '0.0.0.0'
$ServerPort = 7860

uv run --no-sync python gradio_app_yuupro.py --server-name $ServerHost --server-port $ServerPort @args

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[ERROR] アプリの起動に失敗しました。"
    Write-Host ""
    Read-Host "続行するには Enter キーを押してください"
}
