#!/usr/bin/env python3
"""Download the HumanoidArena v3.1 **raw** SONIC recordings from ModelScope.

Why the raw release and not the LeRobot one: only the raw ``.npz`` carries the
64-D SONIC motion token (``encoder_latent``) and the 14-D finger commands
(``hand_action_left/right``).  The ``HumanoidArenaV3.1_SONIC40`` LeRobot release
that is already on disk stores a 40-D reference-pose action and a 2-D binary
hand switch, so it cannot supervise our 64+14 head.

Only the SONIC backend subtree is fetched::

    <task>/sonic/tw/<episode>.npz            ~113 MB each, 100 per task
    <task>/sonic/tw/videos/<episode>_front_rgb.mp4   ~9 MB each

Layout under ``--dest``::

    <task>/<episode>.npz
    <task>/videos/<episode>_front_rgb.mp4
    manifest.tsv        path<TAB>size, cached after the first listing
    download.log        append-only progress

Usage::

    python3 scripts/fetch_ha_raw.py --list-only
    python3 scripts/fetch_ha_raw.py --workers 12
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "ha_raw"

NS = "Twang2026"
REPO = "HumanoidArena_raw"
TREE = (
    f"https://www.modelscope.cn/api/v1/datasets/{NS}/{REPO}/repo/tree"
)
FILE = f"https://www.modelscope.cn/api/v1/datasets/{NS}/{REPO}/repo?Revision=master&FilePath="

# The raw release does not use one layout per task: the operator subdirectory differs
# (``sonic/tw``, ``sonic/yb``, ``sonic/zz``) and ``HOI_football_v2`` has no ``sonic/``
# at all, so the task/operator list is discovered instead of hard-coded.
TASKS = [
    "HOI_double_desk",
    "HOI_football_v2",
    "HOI_grap_cup",
    "HOI_pp_box",
    "HSI_boxing",
    "HSI_open_door",
    "HSI_sit_sofa",
    "HSI_vision_navi",
]

# Only the plain SONIC backend, never ``sonic_multicam_rerecord``: the rerecorded
# npz carry extra depth/camera streams and are 3-6x larger for the same tokens.
BACKENDS = ["sonic"]

_lock = threading.Lock()
_done = [0, 0]  # files, bytes


def _ls(root: str) -> list[dict]:
    out: list[dict] = []
    page = 1
    while True:
        q = urllib.parse.urlencode(
            {"Revision": "master", "Root": root, "PageSize": 200, "PageNumber": page}
        )
        try:
            raw = urllib.request.urlopen(f"{TREE}?{q}", timeout=90).read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return []
            raise
        files = json.loads(raw)["Data"]["Files"]
        out += files
        if len(files) < 200:
            return out
        page += 1


def _discover() -> list[str]:
    """Return ``task/backend/operator`` roots that actually contain SONIC npz."""
    roots: list[str] = []
    for task in TASKS:
        for backend in BACKENDS:
            for entry in _ls(f"/{task}/{backend}"):
                if entry.get("Type") != "tree":
                    continue
                op = entry["Path"].split("/")[-1]
                roots.append(f"{task}/{backend}/{op}")
    return roots


def build_manifest(manifest: Path) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    for root in _discover():
        blobs = [e for e in _ls(f"/{root}") if e.get("Type") == "blob"]
        videos = [e for e in _ls(f"/{root}/videos") if e.get("Type") == "blob"]
        for e in blobs + videos:
            path = e["Path"]
            size = int(e.get("Size") or 0)
            if size <= 0:
                raise SystemExit(f"no size for {path}")
            rows.append((path, size))
        print(f"  {root}: {len(blobs)} npz, {len(videos)} video", flush=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w") as fh:
        for path, size in rows:
            fh.write(f"{path}\t{size}\n")
    total = sum(s for _, s in rows)
    print(f"manifest: {len(rows)} files, {total / 1e9:.2f} GB -> {manifest}", flush=True)
    return rows


def local_path(dest: Path, path: str) -> Path:
    """``<task>/sonic/<op>/<name>.npz`` -> ``<dest>/<task>/<op>/<name>.npz``.

    Keeping the operator directory in the local layout matters because the operator
    subdir is per-task (tw/yb/zz) and is provenance we want when we later compare
    token statistics per operator.
    """
    parts = path.split("/")
    assert parts[0] in TASKS, path
    task, op, name = parts[0], parts[2], parts[-1]
    if name.endswith(".npz"):
        return dest / task / op / name
    if name.endswith(".mp4"):
        return dest / task / op / "videos" / name
    raise ValueError(f"unexpected file {path}")


def fetch_one(dest: Path, path: str, want: int, logf) -> tuple[str, bool]:
    out = local_path(dest, path)
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_suffix(out.suffix + ".part")
    for attempt in range(1, 9):
        have = part.stat().st_size if part.exists() else 0
        if have > want:
            part.unlink()
            have = 0
        if have == want:
            break
        url = FILE + urllib.parse.quote(path, safe="")
        # --speed-limit/--speed-time aborts a stalled transfer so --retry restarts.
        rc = subprocess.call(
            [
                "curl", "-sSL", "--fail", "--retry", "3", "--retry-delay", "4",
                "--retry-all-errors", "--speed-limit", "20000", "--speed-time", "60",
                "-C", "-", "-o", str(part), url,
            ]
        )
        have = part.stat().st_size if part.exists() else 0
        if have == want:
            break
        if attempt == 8:
            with _lock:
                logf.write(f"FAIL {path} {have}/{want}\n")
                logf.flush()
            return path, False
        if rc != 0:
            time.sleep(min(30, 3 * attempt))
    os.replace(part, out)
    with _lock:
        _done[0] += 1
        _done[1] += want
        n, b = _done[0], _done[1]
        if n % 20 == 0 or n == _total:
            pct = 100.0 * b / _total_bytes
            logf.write(
                f"[{time.strftime('%H:%M:%S')}] {n}/{_total} files  "
                f"{b / 1e9:.1f}/{_total_bytes / 1e9:.1f} GB ({pct:.1f}%)\n"
            )
            logf.flush()
    return path, True


_total = 0
_total_bytes = 0


def main() -> int:
    global _total, _total_bytes
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=Path, default=DEST)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--list-only", action="store_true")
    args = ap.parse_args()

    dest: Path = args.dest
    dest.mkdir(parents=True, exist_ok=True)
    manifest = dest / "manifest.tsv"

    if manifest.exists() and manifest.stat().st_size > 0 and not args.list_only:
        rows = [
            (line.split("\t")[0], int(line.split("\t")[1]))
            for line in manifest.read_text().splitlines()
            if line.strip()
        ]
        print(f"manifest cached: {len(rows)} files", flush=True)
    else:
        print("listing ModelScope ...", flush=True)
        rows = build_manifest(manifest)
        if args.list_only:
            return 0

    todo = []
    for path, size in rows:
        out = local_path(dest, path)
        if out.exists() and out.stat().st_size == size:
            continue
        todo.append((path, size))
    _total = len(todo)
    _total_bytes = sum(s for _, s in todo)
    print(
        f"missing: {len(todo)}/{len(rows)} files, {_total_bytes / 1e9:.2f} GB "
        f"({args.workers} workers)",
        flush=True,
    )
    if not todo:
        print("already complete", flush=True)
        return 0

    log = dest / "download.log"
    failed: list[str] = []
    t0 = time.time()
    with log.open("a") as logf:
        logf.write(f"\n=== start {time.strftime('%F %T')} {len(todo)} files ===\n")
        logf.flush()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(fetch_one, dest, p, s, logf) for p, s in todo]
            for fut in as_completed(futs):
                path, ok = fut.result()
                if not ok:
                    failed.append(path)
        dt = time.time() - t0
        avg = _done[1] / 1e6 / max(dt, 1e-9)
        logf.write(
            f"=== done {time.strftime('%F %T')} {_done[0]}/{_total} files, "
            f"{_done[1] / 1e9:.2f} GB in {dt / 60:.1f} min ({avg:.1f} MB/s), "
            f"{len(failed)} failed ===\n"
        )
        logf.flush()
    print(f"downloaded {_done[0]}/{_total} files, failed {len(failed)}")
    if failed:
        (dest / "failed.txt").write_text("\n".join(failed) + "\n")
        return 1
    (dest / "DOWNLOAD_OK").write_text(f"{len(rows)} files\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
