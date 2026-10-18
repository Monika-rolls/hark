# Start a local Gemma answer model for Hark's local mode.
#
#   .\scripts\start_local_llm.ps1                 # models\gemma-3-1b-it-Q4_K_M.gguf
#   .\scripts\start_local_llm.ps1 -Model models\other-model.gguf
#   .\scripts\start_local_llm.ps1 -Port 8081 -Ctx 4096
#
# It serves the OpenAI-compatible API the app speaks, on http://127.0.0.1:<port>/v1,
# which is where hark/server/local_llm.py looks by default. Leave it running, then pick
# "Answer: Gemma -- local (llama.cpp)" in the page. Port 8081 by
# default: Docker Desktop often holds 8080.
param(
    [string]$Model = "models\gemma-3-1b-it-Q4_K_M.gguf",
    [int]$Port = 8081,
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
  curl.exe -L -o models\gemma-3-1b-it-Q4_K_M.gguf https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf"
}

Write-Host "llama-server : $server"
Write-Host "model        : $modelPath"
Write-Host "listening    : http://127.0.0.1:$Port/v1"
Write-Host "Leave this running, then choose 'Answer: Gemma -- local' in the page."

& $server -m $modelPath --port $Port --ctx-size $Ctx --host 127.0.0.1
