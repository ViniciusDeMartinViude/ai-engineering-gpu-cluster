$ErrorActionPreference = 'Stop'
Write-Host 'Windows GPU:'
nvidia-smi
Write-Host 'WSL GPU:'
wsl -e nvidia-smi
Write-Host 'Docker GPU:'
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
