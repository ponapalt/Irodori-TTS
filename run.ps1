#Requires -Version 5.1

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

$RepoDir    = $PSScriptRoot
$VenvDir    = Join-Path $RepoDir '.venv'
$ActivatePs = Join-Path $VenvDir 'Scripts\Activate.ps1'
$PipExe     = Join-Path $VenvDir 'Scripts\pip.exe'
$PyVenvExe  = Join-Path $VenvDir 'Scripts\python.exe'

Write-Host ""
Write-Host "============================================================"
Write-Host "  Irodori-TTS ゆうぷろカスタム V1.4.0 [ローカル版]"
Write-Host "============================================================"
Write-Host ""

# ============================================================
# Step 1: Python 確認
# ============================================================
Write-Host "[Step 1/5] Python を確認中..."

$PythonExe      = $null
$PythonBaseArgs = @()

try {
    $output = & py '-3.10' '--version' 2>&1
    if ($LASTEXITCODE -eq 0) {
        $PythonExe      = 'py'
        $PythonBaseArgs = @('-3.10')
        Write-Host "  $output"
    }
} catch {}

if (-not $PythonExe) {
    try {
        $output = & python '--version' 2>&1
        if ($LASTEXITCODE -eq 0) {
            $PythonExe      = 'python'
            $PythonBaseArgs = @()
            Write-Host "  $output"
            Write-Host "  注意: Python 3.10 を推奨します（.python-version 参照）"
        }
    } catch {}
}

if (-not $PythonExe) {
    Write-Host ""
    Write-Host "[ERROR] Python が見つかりません。Python 3.10 をインストールしてください:"
    Write-Host "        https://www.python.org/downloads/release/python-31011/"
    Write-Host ""
    Read-Host "続行するには Enter キーを押してください"
    exit 1
}
Write-Host ""

# ============================================================
# Step 2: 仮想環境
# ============================================================
Write-Host "[Step 2/5] 仮想環境を確認中..."

$venvCreated = $false
if (-not (Test-Path $ActivatePs)) {
    Write-Host "  仮想環境を作成中: $VenvDir"
    & $PythonExe @PythonBaseArgs '-m' 'venv' $VenvDir
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] 仮想環境の作成に失敗しました。"
        Read-Host "続行するには Enter キーを押してください"
        exit 1
    }
    $venvCreated = $true
    Write-Host "  作成完了。"
} else {
    Write-Host "  既存の仮想環境を使用: $VenvDir"
}

& $PyVenvExe '-m' 'pip' 'install' '--upgrade' 'pip' '--quiet'
Write-Host ""

# ============================================================
# Step 3: ffmpeg
# ============================================================
Write-Host "[Step 3/5] ffmpeg を確認中..."

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
# Step 4: PyTorch (CUDA 12.8)
# ============================================================
Write-Host "[Step 4/5] PyTorch を確認中..."

$torchVer = & $PyVenvExe '-c' 'import torch; print(torch.__version__)' 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "  PyTorch $torchVer は既にインストール済みです。スキップします。"
} else {
    Write-Host "  CUDA 12.x 対応 GPU 用の PyTorch をインストールします。"
    Write-Host "  異なる CUDA バージョンや CPU 環境の場合は手動インストールしてください:"
    Write-Host "  https://pytorch.org/get-started/locally/"
    Write-Host ""
    & $PipExe install --upgrade typing-extensions --quiet
    & $PipExe install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "[ERROR] PyTorch のインストールに失敗しました。"
        Write-Host "        ネットワーク環境や CUDA バージョンを確認してください。"
        Read-Host "続行するには Enter キーを押してください"
        exit 1
    }
}
Write-Host ""

# ============================================================
# Step 5: 依存パッケージ
# ============================================================
Write-Host "[Step 5/5] 依存パッケージを確認中..."

$reqFile    = Join-Path $RepoDir 'requirements.txt'
$tmpReqFile = Join-Path $RepoDir '.requirements_local.txt'
$hashFile   = Join-Path $VenvDir '.req_hash'

$reqHash    = (Get-FileHash $reqFile -Algorithm MD5).Hash
$storedHash = if (Test-Path $hashFile) { (Get-Content $hashFile -Raw).Trim() } else { '' }

if ($reqHash -eq $storedHash) {
    Write-Host "  依存パッケージは最新です。スキップします。"
} else {
    (Get-Content $reqFile -Encoding UTF8) |
        Where-Object { $_.Trim() -notmatch '^wandb' } |
        Set-Content $tmpReqFile -Encoding UTF8

    & $PipExe install --upgrade protobuf --quiet
    & $PipExe install --extra-index-url https://download.pytorch.org/whl/cu128 -r $tmpReqFile
    $pipErr = $LASTEXITCODE
    Remove-Item $tmpReqFile -Force -ErrorAction SilentlyContinue

    if ($pipErr -ne 0) {
        Write-Host ""
        Write-Host "[WARNING] 一部パッケージのインストールに失敗した可能性があります。"
        Write-Host "          torchcodec が Windows 環境で利用不可の場合があります。"
    } else {
        [System.IO.File]::WriteAllText($hashFile, $reqHash)
    }
}
Write-Host ""

# ============================================================
# 初回セットアップ時のみ: モデルの事前ダウンロード
# ============================================================
if ($venvCreated) {
    Write-Host "============================================================"
    Write-Host "  モデルの事前ダウンロード（任意）"
    Write-Host "  スキップすると初回起動時に自動でダウンロードされます"
    Write-Host "============================================================"
    $dlChoice = Read-Host "今すぐダウンロードしますか？ [Y/N]"
    Write-Host ""

    if ($dlChoice -imatch '^y') {
        Write-Host "  モデルをダウンロード中... 数分かかります"
        $pyScript = Join-Path $env:TEMP 'irodori_dl.py'
        @'
from huggingface_hub import hf_hub_download
print("  [1/2] ベースモデル (Irodori-TTS-500M-v2)...")
p = hf_hub_download(repo_id="Aratako/Irodori-TTS-500M-v2", filename="model.safetensors")
print(f"        完了: {p}")
print("  [2/2] VoiceDesign (Irodori-TTS-500M-v2-VoiceDesign)...")
p = hf_hub_download(repo_id="Aratako/Irodori-TTS-500M-v2-VoiceDesign", filename="model.safetensors")
print(f"        完了: {p}")
print("  ダウンロード完了！")
'@ | Set-Content $pyScript -Encoding UTF8
        & $PyVenvExe $pyScript
        Remove-Item $pyScript -Force -ErrorAction SilentlyContinue
        Write-Host ""
    }
}

# ============================================================
# アプリ起動
# ============================================================
$ServerHost = '0.0.0.0'
$ServerPort = 7860

Set-Location $RepoDir
. $ActivatePs

python gradio_app_yuupro.py --server-name $ServerHost --server-port $ServerPort @args

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[ERROR] アプリの起動に失敗しました。"
    Write-Host ""
    Read-Host "続行するには Enter キーを押してください"
}
