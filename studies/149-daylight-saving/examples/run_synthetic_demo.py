"""Offline, deterministic demo — Study 149 (Daylight-Saving).

No network. Builds a synthetic daily tape and shows the study's spine in one screen:
the post-DST Monday return gap only emerges when a DST effect is *planted* in the
data; on a pure random walk (dst_effect=0) it scores near zero. Run:

    python studies/149-daylight-saving/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from daylight_saving import data, strategy as st  # noqa: E402


def _add_labels(frame):
    """Add is_monday and season columns to a synthetic frame."""
    import pandas as pd
    frame = frame.copy()
    frame["is_monday"] = frame.index.dayofweek == 0
    tbl = data.dst_mondays(int(frame.index.year.min()), int(frame.index.year.max()))
    dst_map = {row["date"].normalize(): row["season"] for _, row in tbl.iterrows()}
    frame["season"] = [dst_map.get(pd.Timestamp(d).normalize(), "") for d in frame.index]
    return frame


def main() -> None:
    print("Study 149 — Daylight-Saving — synthetic positive control\n")
    print(f"{'planted DST effect':>20} | {'DST mean(bps)':>14} {'Other Mon(bps)':>15} "
          f"{'gap(bps)':>10} {'Welch-t':>8} | signal?")
    print("-" * 82)
    for effect in (0.00, -0.002, -0.005, -0.010):
        frame, truth = data.synthetic_daily(dst_effect=effect, seed=149)
        frame = _add_labels(frame)
        h = st.dst_vs_other_mondays(frame)
        n_dst = h["dst"]["n"]
        dst_m = h["dst"]["mean_bps"]
        oth_m = h["other"]["mean_bps"]
        gap   = h["diff_bps"]
        t     = h["welch_t"]
        sig   = "YES (|t|>2)" if abs(t) > 2 else "no"
        print(f"{effect*100:>18.1f}%  | {dst_m:>+14.2f} {oth_m:>+15.2f} "
              f"{gap:>+10.2f} {t:>+8.2f} | {sig}")

    print()
    print(f"DST Mondays in sample (1980-2025): {truth['n_dst']}")
    print(f"Other Mondays in sample:           {truth['n_other_monday']}")
    print()
    print("The engine only detects a gap when one is planted (effect < 0).")
    print("On the real tape (see docs/results.md), the gap is near zero — NONE.")


if __name__ == "__main__":
    main()
