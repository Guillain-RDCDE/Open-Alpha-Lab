#!/usr/bin/env python
"""Smoke-run the offline synthetic demos -- the scripts a newcomer runs first.

Each demo is run as a separate process from the repository root, exactly as CI and the
READMEs invoke it (``python studies/<id>/examples/run_synthetic_demo.py``). A demo that
exits non-zero fails the run; every demo is attempted so one break does not hide another.

    python tools/run_smoke_demos.py            # all demos
    python tools/run_smoke_demos.py 03 71      # only the demos of studies 03 and 71
    python tools/run_smoke_demos.py --list     # print the demo paths and exit

Pure stdlib.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEMOS = [
    "studies/01-overnight-anomaly/examples/run_synthetic_demo.py",
    "studies/71-ambush/examples/run_synthetic_demo.py",
    "studies/72-loaded-dice/examples/run_synthetic_demo.py",
    "studies/03-fear-gauge/examples/run_synthetic_demo.py",
    "studies/04-social-oracle/examples/run_synthetic_demo.py",
    "studies/05-twin-spread/examples/run_synthetic_demo.py",
    "studies/09-phantom-kernel/examples/run_synthetic_demo.py",
    "studies/11-vanishing-penny/examples/run_synthetic_demo.py",
    "studies/16-storm-shy/examples/run_synthetic_demo.py",
    "studies/30-house-edge/examples/run_synthetic_demo.py",
    "studies/31-trade-winds/examples/run_synthetic_demo.py",
    "studies/32-rip-tide/examples/run_synthetic_demo.py",
    "studies/33-slingshot/examples/run_synthetic_demo.py",
    "studies/34-aftershock/examples/run_synthetic_demo.py",
    "studies/35-contango/examples/run_synthetic_demo.py",
    "studies/36-greenback/examples/run_synthetic_demo.py",
    "studies/37-barometer/examples/run_synthetic_demo.py",
    "studies/38-chorus/examples/run_synthetic_demo.py",
    "studies/39-black-box/examples/run_synthetic_demo.py",
    "studies/301-triple-rsi/examples/run_synthetic_demo.py",
]


def _selected(args: list[str]) -> list[str]:
    if not args:
        return DEMOS
    want = {a.split("-")[0] for a in args}
    return [d for d in DEMOS if d.split("/")[1].split("-")[0] in want]


def main(argv: list[str]) -> int:
    if "--list" in argv:
        print("\n".join(DEMOS))
        return 0
    demos = _selected([a for a in argv if not a.startswith("-")])
    env = dict(os.environ, MPLBACKEND=os.environ.get("MPLBACKEND", "Agg"))
    failed = []
    for rel in demos:
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            print(f"  X  {rel}: missing")
            failed.append(rel)
            continue
        t0 = time.time()
        proc = subprocess.run([sys.executable, path], cwd=ROOT, env=env,
                              capture_output=True, text=True)
        dt = time.time() - t0
        if proc.returncode:
            print(f"  X  {rel} exited {proc.returncode} after {dt:.1f}s")
            print(proc.stdout[-4000:])
            print(proc.stderr[-4000:], file=sys.stderr)
            failed.append(rel)
        else:
            print(f"  ok {rel} ({dt:.1f}s)")
    print(f"\n{len(demos) - len(failed)}/{len(demos)} demos ran clean.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
