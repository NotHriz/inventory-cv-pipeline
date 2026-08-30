# ==============================================================================
# Launch the model upload watcher in the background.
#
# This starts `upload_when_ready.py` as a detached background process so it
# keeps running even after you close this terminal and leave. It watches for
# `models/inventory_model.tflite` and copies it to your configured cloud
# folder (see UPLOAD_METHOD / UPLOAD_LOCAL_DIR in `.env`) once training ends.
#
# Usage (in PowerShell, from the project folder):
#     .\start_upload_watcher.ps1
# ==============================================================================

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "====================================================="
Write-Host "  Model Upload Watcher Launcher"
Write-Host "====================================================="

# Verify the watcher script exists.
if (-not (Test-Path ".\upload_when_ready.py")) {
    Write-Host "ERROR: upload_when_ready.py not found." -ForegroundColor Red
    exit 1
}

# Compute a unique log file next to the project.
$logFile = Join-Path $PSScriptRoot "upload_watcher_output.log"

# Launch as a detached background process using Start-Process so it survives
# the current shell closing.
Write-Host "Starting background watcher..."
Write-Host "Log file: $logFile"
$proc = Start-Process `
    -FilePath "python" `
    -ArgumentList "upload_when_ready.py" `
    -WorkingDirectory $PSScriptRoot `
    -RedirectStandardOutput $logFile `
    -RedirectStandardError "$logFile.err" `
    -WindowStyle Hidden `
    -PassThru
Write-Host "Watcher launched with PID: $($proc.Id)"
Write-Host ""
Write-Host "The watcher will monitor for models\inventory_model.tflite and"
Write-Host "copy it to your configured upload destination as soon as training"
Write-Host "and export finish."
Write-Host ""
Write-Host "You can now close this window and leave. Check the log later:"
Write-Host "    Get-Content $logFile"

