$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

$drive = [System.IO.DriveInfo]::new([System.IO.Path]::GetPathRoot($PSScriptRoot))
if ($drive.AvailableFreeSpace -lt 3GB) {
    throw 'Free at least 3 GB on the project drive before installing Python and TensorFlow.'
}

# Keep the Python runtime and environment inside this project.
if (-not (Test-Path -LiteralPath '.python312/python.exe')) {
    py install --target .python312 3.12
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 download failed.' }
}
if (-not (Test-Path -LiteralPath '.venv312/Scripts/python.exe')) {
    & '.python312/python.exe' -m venv .venv312
    if ($LASTEXITCODE -ne 0) { throw 'Environment creation failed.' }
}
& '.venv312/Scripts/python.exe' -m pip install --no-cache-dir --only-binary=:all: --timeout 120 --retries 5 --progress-bar off -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Package installation failed.' }
& '.venv312/Scripts/python.exe' -c "import numpy, matplotlib.pyplot, tensorflow, sklearn, ipykernel; print('All imports passed. TensorFlow:', tensorflow.__version__)"
if ($LASTEXITCODE -ne 0) { throw 'Import verification failed.' }
Write-Host 'Setup complete. In VS Code, select .venv312/Scripts/python.exe as the notebook kernel.'
