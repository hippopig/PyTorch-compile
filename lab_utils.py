"""Shared helpers for the AI compiler lab."""

from __future__ import annotations

from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"


def select_device(allow_cpu: bool = False) -> torch.device:
    """Select CUDA and fail loudly unless the caller explicitly allows CPU."""

    if torch.cuda.is_available():
        return torch.device("cuda")
    if allow_cpu:
        print("[warning] CUDA is unavailable; using CPU because --allow-cpu was set.")
        return torch.device("cpu")
    raise RuntimeError(
        "CUDA is unavailable. Run inside the cuda-dev environment and verify "
        "that the NVIDIA GPU is exposed, or pass --allow-cpu for a CPU-only run."
    )


def device_report(device: torch.device) -> str:
    lines = [f"PyTorch: {torch.__version__}", f"selected device: {device}"]
    if device.type == "cuda":
        major, minor = torch.cuda.get_device_capability(device)
        lines.extend(
            [
                f"CUDA runtime bundled with PyTorch: {torch.version.cuda}",
                f"GPU: {torch.cuda.get_device_name(device)}",
                f"compute capability: sm_{major}{minor}",
                f"architectures in this PyTorch build: {torch.cuda.get_arch_list()}",
            ]
        )
    return "\n".join(lines)


def prepare_artifacts() -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACT_DIR


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)

