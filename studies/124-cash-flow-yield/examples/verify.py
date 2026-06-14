# -*- coding: utf-8 -*-
"""Real-tape verification -- Study 124 (Cash-Flow-Yield). Regenerates docs/results.md numbers.

Reads from cached EDGAR parquets (desk-wide shared cache) and a per-study year-end price
cache (generated on first --fetch). Runs the OCF-yield and earnings-yield quintile sorts,
the hedge-vs-market comparison, and the pairwise IC test.

    python studies/124-cash-flow-yield/examples/verify.py            # cache-only
    python studies/124-cash-flow-yield/examples/verify.py --fetch    # refresh year-end prices
"""

from __future__ import annotations

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from cash_flow_yield import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    ocfy, ey, fwd = data.fetch_panel(fetch=fetch)
    if ocfy.empty:
        print("No cached data found. Re-run with --fetch to download year-end prices.")
        return

    print(f"OCF-yield panel: {ocfy.shape}  years {ocfy.index.min()}-{ocfy.index.max()}")
    print(f"Fingerprints: OCF={data.fingerprint(ocfy)}  EY={data.fingerprint(ey)}  fwd={data.fingerprint(fwd)}")
    print()

    # --- OCF-yield quintile sort ---
    q_ocfy = st.quintile_sort(ocfy, fwd)
    s_ocfy = st.summary(q_ocfy["hedge"])
    print("=== OCF-yield quintile sort (Q5 long, Q1 short) ===")
    print(q_ocfy[["Q1", "Q5", "hedge", "market"]].round(3).to_string())
    print(f"\n  hedge: mean={s_ocfy['mean']:+.3f}  vol={s_ocfy['vol']:.3f}  "
          f"Sharpe={s_ocfy['sharpe']:+.2f}  t={s_ocfy['tstat']:+.2f}  "
          f"hit={s_ocfy['hit_rate']:.0%}  n={s_ocfy['n']}")

    # --- EY quintile sort ---
    q_ey = st.quintile_sort(ey, fwd)
    s_ey = st.summary(q_ey["hedge"])
    print()
    print("=== Earnings-yield quintile sort (control) ===")
    print(f"  hedge: mean={s_ey['mean']:+.3f}  vol={s_ey['vol']:.3f}  "
          f"Sharpe={s_ey['sharpe']:+.2f}  t={s_ey['tstat']:+.2f}  "
          f"hit={s_ey['hit_rate']:.0%}  n={s_ey['n']}")

    # --- Hedge vs market ---
    hvm = st.hedge_vs_market(ocfy, fwd)
    s_hvm = st.summary(hvm["spread"])
    hvm_ey = st.hedge_vs_market(ey, fwd)
    s_hvm_ey = st.summary(hvm_ey["spread"])
    print()
    print("=== Top-quintile vs equal-weight market ===")
    print(f"  OCF top-Q: spread mean={s_hvm['mean']:+.3f}  t={s_hvm['tstat']:+.2f}  hit={s_hvm['hit_rate']:.0%}")
    print(f"  EY  top-Q: spread mean={s_hvm_ey['mean']:+.3f}  t={s_hvm_ey['tstat']:+.2f}  hit={s_hvm_ey['hit_rate']:.0%}")

    # --- Pairwise IC ---
    ic = st.pairwise_ic(ocfy, ey, fwd)
    s_ic_a = st.summary(ic["ic_a"])
    s_ic_b = st.summary(ic["ic_b"])
    s_ic_diff = st.summary(ic["ic_diff"])
    print()
    print("=== Pairwise IC: OCF yield vs Earnings yield ===")
    print(f"  OCF IC: mean={s_ic_a['mean']:+.4f}  t={s_ic_a['tstat']:+.2f}")
    print(f"  EY  IC: mean={s_ic_b['mean']:+.4f}  t={s_ic_b['tstat']:+.2f}")
    print(f"  Diff:   mean={s_ic_diff['mean']:+.4f}  t={s_ic_diff['tstat']:+.2f}")

    # --- Market benchmark ---
    mkt_mean = q_ocfy["market"].mean()
    mkt_vol = q_ocfy["market"].std()
    print()
    print(f"Market (equal-weight S&P 500): mean={mkt_mean:+.3f}  vol={mkt_vol:.3f}")

    print()
    print("Verdict: OCF-yield hedge t={:.2f}, EY hedge t={:.2f} -- NONE/NONE (see docs/results.md)".format(
        s_ocfy['tstat'], s_ey['tstat']))


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
