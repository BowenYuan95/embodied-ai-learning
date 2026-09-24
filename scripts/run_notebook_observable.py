#!/usr/bin/env python3
"""Execute a notebook cell by cell, with progress visible from outside.

Why this exists
---------------
``nbclient`` captures a cell's stdout into the notebook, so a single
``NotebookClient.execute()`` prints **nothing** to the terminal until the whole
notebook has finished. A notebook containing a few minutes of training is then
indistinguishable from a hang, which is exactly the failure this repository
recorded on 2026-09-24.

This runner does three things instead:

1. executes **one cell at a time** and prints a heartbeat to stderr as each
   finishes, including that cell's own stream output;
2. **saves the notebook after every cell**, so an interrupt keeps the cells that
   already succeeded instead of discarding all of them;
3. exits non-zero on the first failing cell, naming it, rather than continuing
   into cells whose prerequisites never ran.

Usage
-----
    python scripts/run_notebook_observable.py notebooks/3.8f_ablation.ipynb
    python scripts/run_notebook_observable.py NB --start-at 6 --timeout 1200
    python scripts/run_notebook_observable.py NB --no-echo    # heartbeats only
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def first_line(cell) -> str:
    for line in cell.source.splitlines():
        text = line.strip()
        if text:
            return text[:72]
    return ""


def stream_text(cell) -> str:
    parts = []
    for out in cell.get("outputs", []):
        if out.get("output_type") == "stream":
            parts.append("".join(out.get("text", "")))
        elif out.get("output_type") == "error":
            parts.append(f"{out.get('ename')}: {out.get('evalue')}\n")
    return "".join(parts).rstrip()


def error_text(cell) -> str:
    for out in cell.get("outputs", []):
        if out.get("output_type") == "error":
            return f"{out.get('ename')}: {out.get('evalue')}"
    return ""


async def run(path: Path, timeout: int, kernel: str, save_each: bool,
              start_at: int, echo: bool) -> int:
    nb = nbformat.read(path, as_version=4)
    code_indices = [i for i, c in enumerate(nb.cells) if c.cell_type == "code"]
    if not code_indices:
        print(f"{path}: no code cells", file=sys.stderr)
        return 0

    client = NotebookClient(nb, timeout=timeout, kernel_name=kernel,
                            resources={"metadata": {"path": str(path.parent.parent)}},
                            allow_errors=False)

    total = len(code_indices)
    done = 0
    t_start = time.time()
    print(f"== {path.name}: {total} code cells, timeout {timeout}s/cell ==",
          file=sys.stderr, flush=True)

    async with client.async_setup_kernel():
        for i in code_indices:
            if i < start_at:
                continue
            cell = nb.cells[i]
            t0 = time.time()
            try:
                await client.async_execute_cell(cell, i)
            except Exception as exc:                       # noqa: BLE001 - report, do not hide
                dt = time.time() - t0
                done += 1
                print(f"[{done:>2}/{total}] {dt:>8.1f}s  FAIL  cell {i}: {first_line(cell)}",
                      file=sys.stderr, flush=True)
                detail = error_text(cell) or f"{type(exc).__name__}: {exc}"
                print(f"          {detail[:400]}", file=sys.stderr, flush=True)
                if save_each:
                    nbformat.write(nb, path)
                    print(f"          saved {path.name} (cells before the failure kept)",
                          file=sys.stderr, flush=True)
                return 1

            dt = time.time() - t0
            done += 1
            print(f"[{done:>2}/{total}] {dt:>8.1f}s  OK    cell {i}: {first_line(cell)}",
                  file=sys.stderr, flush=True)
            if echo:
                body = stream_text(cell)
                if body:
                    for line in body.splitlines():
                        print(f"          | {line}", file=sys.stderr, flush=True)
            if save_each:
                nbformat.write(nb, path)

    print(f"== done in {time.time() - t_start:.0f}s, all {total} code cells OK ==",
          file=sys.stderr, flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("notebook", type=Path)
    ap.add_argument("--timeout", type=int, default=900, help="seconds per cell (default 900)")
    ap.add_argument("--kernel", default="python3")
    ap.add_argument("--start-at", type=int, default=0, help="skip cells with index below this")
    ap.add_argument("--no-save-each", dest="save_each", action="store_false",
                    help="save only after the whole notebook succeeds")
    ap.add_argument("--no-echo", dest="echo", action="store_false",
                    help="print heartbeats but not each cell's stream output")
    args = ap.parse_args()

    if not args.notebook.exists():
        print(f"no such notebook: {args.notebook}", file=sys.stderr)
        return 2
    return asyncio.run(run(args.notebook, args.timeout, args.kernel,
                           args.save_each, args.start_at, args.echo))


if __name__ == "__main__":
    raise SystemExit(main())
