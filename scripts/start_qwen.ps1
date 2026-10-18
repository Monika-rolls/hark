# Start a local Qwen answer model for Hark's local mode.
#
#   .\scripts\start_qwen.ps1                      # models\qwen2.5-1.5b-instruct-q4_k_m.gguf
#   .\scripts\start_qwen.ps1 -Model models\qwen2.5-0.5b-instruct-q4_k_m.gguf
#   .\scripts\start_qwen.ps1 -Port 8080 -Ctx 4096
#
# It serves the OpenAI-compatible API the app speaks, on http://127.0.0.1:<port>/v1,
# which is where hark/server/local_llm.py looks by default. Leave it running, then pick
# "Answer: Qwen -- local (llama.cpp)" in the page.
param(
    [string]$Model = "models\qwen2.5-1.5b-instruct-q4_k_m.gguf",
    [int]$Port = 8080,
    [int]$Ctx = 4096
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

# llama-server may be on PATH, or installed by winget under Packages.
$server = (Get-Command llama-server -ErrorAction SilentlyContinue).Source
if (-not $server) {
    $server = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Recurse `
        -Filter 'llama-server.exe' -ErrorAction SilentlyContinue |
        Select-Object -First 1 -ExpandProperty FullName
}
if (-not $server) {
    throw "llama-server not found. Install it with: winget install ggml.llamacpp"
}

$modelPath = if ([System.IO.Path]::IsPathRooted($Model)) { $Model } else { Join-Path $root $Model }
if (-not (Test-Path $modelPath)) {
    throw "No model at $modelPath. Download one, e.g.:
  curl.exe -L -o models\qwen2.5-1.5b-instruct-q4_k_m.gguf https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"
}

Write-Host "llama-server : $server"
Write-Host "model        : $modelPath"
Write-Host "listening    : http://127.0.0.1:$Port/v1"
Write-Host "Leave this running, then choose 'Answer: Qwen -- local' in the page."

& $server -m $modelPath --port $Port --ctx-size $Ctx --host 127.0.0.1
