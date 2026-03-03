from __future__ import annotations

import hashlib
import subprocess
import time

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

UTF8 = "utf-8"


@dataclass
class ApptainerExecResult:
    output: str
    return_code: int
    timed_out: bool
    runtime: float


def image_ref_to_apptainer_uri(image_ref: str) -> str:
    """
    Convert a container image reference to an Apptainer URI.
    """
    if image_ref.startswith(("docker://", "library://", "oras://", "shub://")):
        return image_ref
    return f"docker://{image_ref}"


def image_ref_to_sif_name(image_ref: str) -> str:
    """
    Build a deterministic SIF filename from an image reference.
    """
    digest = hashlib.sha256(image_ref.encode(UTF8)).hexdigest()[:12]
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in image_ref)
    safe = safe.strip("._-") or "image"
    safe = safe[:96]
    return f"{safe}.{digest}.sif"


def get_cached_sif_path(image_ref: str, cache_dir: Path | str) -> Path:
    """
    Return the deterministic path for a cached SIF image.
    """
    cache_dir = Path(cache_dir)
    return cache_dir / image_ref_to_sif_name(image_ref)


def ensure_apptainer_image(
    image_ref: str,
    cache_dir: Path | str,
    apptainer_bin: str = "apptainer",
) -> Path:
    """
    Pull an Apptainer image to cache if needed and return its SIF path.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    sif_path = get_cached_sif_path(image_ref, cache_dir)
    if sif_path.exists():
        return sif_path

    uri = image_ref_to_apptainer_uri(image_ref)
    cmd = [apptainer_bin, "pull", str(sif_path), uri]
    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            encoding=UTF8,
            errors="replace",
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise RuntimeError(
            f"Failed to pull image {image_ref} with Apptainer: {stderr}"
        ) from exc
    return sif_path


def build_apptainer_exec_cmd(
    sif_path: Path | str,
    inner_cmd: str | Sequence[str],
    bind_mounts: Sequence[tuple[Path | str, str]] | None = None,
    workdir: str | None = None,
    apptainer_bin: str = "apptainer",
) -> list[str]:
    """
    Construct an `apptainer exec` command.
    """
    cmd = [apptainer_bin, "exec", "--cleanenv", "--writable-tmpfs"]

    for src, dst in bind_mounts or ():
        cmd.extend(["--bind", f"{Path(src).resolve()}:{dst}"])

    if workdir:
        cmd.extend(["--pwd", workdir])

    cmd.append(str(sif_path))
    if isinstance(inner_cmd, str):
        cmd.extend(["/bin/bash", "-lc", inner_cmd])
    else:
        cmd.extend(list(inner_cmd))
    return cmd


def _combine_output(stdout: str | None, stderr: str | None) -> str:
    if stdout and stderr:
        return f"{stdout}{stderr}"
    return stdout or stderr or ""


def run_apptainer_exec(
    sif_path: Path | str,
    inner_cmd: str | Sequence[str],
    bind_mounts: Sequence[tuple[Path | str, str]] | None = None,
    workdir: str | None = None,
    timeout: int | None = None,
    apptainer_bin: str = "apptainer",
) -> ApptainerExecResult:
    """
    Execute a command inside an Apptainer image.
    """
    cmd = build_apptainer_exec_cmd(
        sif_path=sif_path,
        inner_cmd=inner_cmd,
        bind_mounts=bind_mounts,
        workdir=workdir,
        apptainer_bin=apptainer_bin,
    )
    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding=UTF8,
            errors="replace",
            timeout=timeout,
            check=False,
        )
        output = _combine_output(proc.stdout, proc.stderr)
        return ApptainerExecResult(
            output=output,
            return_code=proc.returncode,
            timed_out=False,
            runtime=time.time() - start,
        )
    except subprocess.TimeoutExpired as exc:
        output = _combine_output(exc.stdout, exc.stderr)
        return ApptainerExecResult(
            output=output,
            return_code=124,
            timed_out=True,
            runtime=time.time() - start,
        )
