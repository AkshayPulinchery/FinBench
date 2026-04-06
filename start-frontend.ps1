param(
  [int]$Port = 5500
)

$ErrorActionPreference = "Stop"

# Serve static files from this script's directory.
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host "Starting frontend server at http://localhost:$Port"
Write-Host "Press Ctrl+C to stop."

if (Get-Command py -ErrorAction SilentlyContinue) {
  py -m http.server $Port
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
  python -m http.server $Port
} else {
  Write-Error "Python is required to run this script. Install Python and retry."
}
