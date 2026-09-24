#!/usr/bin/env python3
"""Serialize Colima fstrim and watch existing SilentSwap generators."""

from __future__ import annotations

import argparse
try:
    import fcntl
except ImportError:
    fcntl = None
import shutil
import os
from pathlib import Path
import subprocess
import time
from typing import Any


LOCK_PATH = Path(
    os.environ.get(
        "SILENTSWAP_TRIM_LOCK",
        str(Path.home() / ".cache" / "silentswap_docker_trim.lock"),
    )
)
TRIM_COMMAND = ["colima", "ssh", "--", "sudo", "fstrim", "-av"]
SAMPLE_IMAGE_PREFIX = "aweaiteam/denovoswe"


def trim(*, minimum_interval: float = 15.0) -> dict[str, Any]:
    """Run one cross-process fstrim, coalescing near-simultaneous requests."""
    if fcntl is None or shutil.which("colima") is None:
        return {"command": " ".join(TRIM_COMMAND), "exit_code": 0, "stdout": "", "stderr": "", "skipped": True}
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        lock.seek(0)
        value = lock.read().strip()
        last_trim = float(value) if value else 0.0
        now = time.time()
        if now - last_trim < minimum_interval:
            return {
                "command": " ".join(TRIM_COMMAND),
                "exit_code": 0,
                "stdout": "",
                "stderr": "",
                "skipped": True,
            }
        completed = subprocess.run(
            TRIM_COMMAND,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        result = {
            "command": " ".join(TRIM_COMMAND),
            "exit_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "skipped": False,
        }
        if completed.returncode == 0:
            lock.seek(0)
            lock.truncate()
            lock.write(str(time.time()))
            lock.flush()
        return result


def sample_image_ids(docker: str = "docker") -> set[str]:
    completed = subprocess.run(
        [docker, "images", "--format", "{{.Repository}} {{.ID}}"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "docker images failed")
    return {
        line.split(maxsplit=1)[1]
        for line in completed.stdout.splitlines()
        if line.startswith(f"{SAMPLE_IMAGE_PREFIX} ")
    }


def generators_running() -> bool:
    completed = subprocess.run(
        ["ps", "-Ao", "command="],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return any(
        "python" in line
        and "expand_multi_swaps.py" in line
        and "--samples" in line
        for line in completed.stdout.splitlines()
    )


def watch(*, docker: str = "docker", poll_interval: float = 5.0) -> int:
    previous = sample_image_ids(docker)
    observed_generator = False
    while generators_running():
        observed_generator = True
        time.sleep(poll_interval)
        current = sample_image_ids(docker)
        if previous - current:
            result = trim()
            state = "skipped" if result["skipped"] else "done"
            print(f"[docker:trim:{state}] removed_images={len(previous - current)}", flush=True)
        previous = current
    if observed_generator:
        result = trim()
        state = "skipped" if result["skipped"] else "done"
        print(f"[docker:trim:{state}] final", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docker", default="docker")
    parser.add_argument("--poll-interval", type=float, default=5.0)
    args = parser.parse_args()
    return watch(docker=args.docker, poll_interval=args.poll_interval)


if __name__ == "__main__":
    raise SystemExit(main())
