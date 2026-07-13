#!/usr/bin/env bash
# Build and verify the isolated WSL2 ROCm environment used for detector training.
set -euo pipefail

VENV="${OBJECTCOUNT_TRAIN_VENV:-$HOME/objectcount-train-venv}"
ROCM_WHEELS="https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2"
TORCH_WHEEL="$ROCM_WHEELS/torch-2.10.0+rocm7.2.0.lw.gitb6ee5fde-cp310-cp310-linux_x86_64.whl"
TORCHVISION_WHEEL="$ROCM_WHEELS/torchvision-0.25.0+rocm7.2.0.git82df5f59-cp310-cp310-linux_x86_64.whl"
TRITON_WHEEL="$ROCM_WHEELS/triton-3.6.0+rocm7.2.0.gitba5c1517-cp310-cp310-linux_x86_64.whl"
ULTRALYTICS_VERSION="${ULTRALYTICS_VERSION:-8.3.169}"
RUN_ULTRALYTICS_SMOKE=1

if [[ "${1:-}" == "--skip-ultralytics-smoke" ]]; then
    RUN_ULTRALYTICS_SMOKE=0
elif [[ $# -gt 0 ]]; then
    echo "usage: $0 [--skip-ultralytics-smoke]" >&2
    exit 2
fi

say() {
    printf '\n======== %s ========\n' "$*"
}

say "ROCm preflight"
command -v python3 >/dev/null
command -v rocminfo >/dev/null
python3 - <<'PY'
import sys

if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"expected WSL Python 3.10, got {sys.version.split()[0]}")
PY
rocminfo 2>/dev/null | grep 'gfx1201' >/dev/null || {
    echo "ROCm does not report the expected gfx1201 device" >&2
    exit 1
}

say "Create isolated environment"
if [[ ! -x "$VENV/bin/python" ]]; then
    python3 -m venv "$VENV"
fi
PYTHON="$VENV/bin/python"
PIP="$VENV/bin/pip"
"$PIP" install --upgrade pip wheel

say "Install pinned ROCm PyTorch"
if "$PYTHON" - <<'PY'
import torch
import torchvision
import triton

expected = (
    "2.10.0+rocm7.2.0.gitb6ee5fde",
    "0.25.0+rocm7.2.0.git82df5f59",
    "3.6.0+rocm7.2.0.gitba5c1517",
)
actual = (torch.__version__, torchvision.__version__, triton.__version__)
if actual != expected:
    raise SystemExit(f"installed ROCm versions differ: {actual!r}")
PY
then
    echo "Pinned ROCm packages are already installed."
else
    "$PIP" install --upgrade --force-reinstall \
        "$TORCH_WHEEL" \
        "$TORCHVISION_WHEEL" \
        "$TRITON_WHEEL"
fi

# The wheel bundles a generic HSA runtime. WSL needs the matching system ROCm 7.2 runtime.
TORCH_LIB="$($PYTHON -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
SYSTEM_HSA="$(find /opt/rocm/lib -maxdepth 1 -name 'libhsa-runtime64.so.1.*' -print -quit)"
if [[ -z "$SYSTEM_HSA" ]]; then
    echo "system ROCm HSA runtime was not found" >&2
    exit 1
fi
rm -f "$TORCH_LIB/libhsa-runtime64.so" "$TORCH_LIB/libhsa-runtime64.so.1"
cp "$SYSTEM_HSA" "$TORCH_LIB/libhsa-runtime64.so"
ln -s libhsa-runtime64.so "$TORCH_LIB/libhsa-runtime64.so.1"

torch_fingerprint() {
    "$PYTHON" - <<'PY'
import torch
import torchvision

print(f"torch={torch.__version__}")
print(f"torchvision={torchvision.__version__}")
print(f"hip={torch.version.hip}")
PY
}

BEFORE_TORCH="$(torch_fingerprint)"

say "Install Ultralytics without replacing ROCm packages"
"$PIP" install "ultralytics==$ULTRALYTICS_VERSION"
AFTER_TORCH="$(torch_fingerprint)"
if [[ "$BEFORE_TORCH" != "$AFTER_TORCH" ]]; then
    printf 'PyTorch changed during Ultralytics installation.\nBefore:\n%s\nAfter:\n%s\n' \
        "$BEFORE_TORCH" "$AFTER_TORCH" >&2
    exit 1
fi
"$PIP" freeze > "$VENV/pip-freeze.txt"

say "20-step forward/backward gate"
"$PYTHON" - <<'PY'
import torch

print("torch:", torch.__version__)
print("HIP:", torch.version.hip)
assert torch.cuda.is_available(), "ROCm device is not visible to PyTorch"
print("device:", torch.cuda.get_device_name(0))
torch.cuda.reset_peak_memory_stats()
x = torch.randn(1024, 1024, device="cuda", requires_grad=True)
w = torch.randn(1024, 1024, device="cuda", requires_grad=True)
for _ in range(20):
    loss = (x @ w).pow(2).mean()
    loss.backward()
    with torch.no_grad():
        x -= 1e-4 * x.grad
        w -= 1e-4 * w.grad
        x.grad = None
        w.grad = None
torch.cuda.synchronize()
print("peak_memory_mib:", round(torch.cuda.max_memory_allocated() / 1024**2, 1))
print("AMD_AUTOGRAD_GATE=PASS")
PY

if [[ "$RUN_ULTRALYTICS_SMOKE" -eq 1 ]]; then
    say "Ultralytics one-epoch smoke gate"
    SMOKE_ROOT="${OBJECTCOUNT_SMOKE_ROOT:-$PWD/out/training/smoke}"
    export OBJECTCOUNT_SMOKE_ROOT="$SMOKE_ROOT"
    "$PYTHON" - <<'PY'
import os
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import yaml
from ultralytics import YOLO
from ultralytics.utils import ROOT

smoke_root = Path(os.environ["OBJECTCOUNT_SMOKE_ROOT"]).resolve()
model_path = smoke_root / "pretrained" / "yolo11n.pt"
model_path.parent.mkdir(parents=True, exist_ok=True)
dataset_parent = smoke_root / "datasets"
dataset_root = dataset_parent / "coco8"
if not dataset_root.is_dir():
    dataset_parent.mkdir(parents=True, exist_ok=True)
    archive_path = dataset_parent / "coco8.zip"
    urlretrieve(
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip",
        archive_path,
    )
    with ZipFile(archive_path) as archive:
        for member in archive.infolist():
            destination = (dataset_parent / member.filename).resolve()
            if dataset_parent not in destination.parents and destination != dataset_parent:
                raise RuntimeError(f"unsafe COCO8 archive member: {member.filename}")
        archive.extractall(dataset_parent)

data_config = yaml.safe_load((ROOT / "cfg/datasets/coco8.yaml").read_text())
data_config["path"] = str(dataset_root)
data_path = dataset_parent / "coco8-local.yaml"
data_path.write_text(yaml.safe_dump(data_config, sort_keys=False))

model = YOLO(str(model_path))
model.train(
    data=str(data_path),
    epochs=1,
    imgsz=320,
    batch=2,
    workers=2,
    device=0,
    project=str(smoke_root),
    name="amd-rocm",
    exist_ok=True,
    seed=713,
    deterministic=True,
    amp=False,
)
print("ULTRALYTICS_SMOKE_GATE=PASS")
PY
fi

say "Environment ready"
printf 'python=%s\n' "$PYTHON"
printf '%s\n' "$AFTER_TORCH"
