"""Real-tape verification — Study 148 (Lunar-Effect).  Regenerates docs/results.md numbers.

Loads (or fetches) the ^GSPC daily return history, assigns lunar labels, computes the
new-vs-full window contrast, the decade breakdown, the shuffled-label control, and the
long-new/short-full strategy Sharpe — all the numbers quoted in docs/results.md.
Network is touched only with --fetch.

    python studies/148-lunar-effect/examples/verify.py            # cache-only
    python studies/148-lunar-effect/examples/verify.py --fetch    # refresh the data
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from lunar_effect import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    frame = data.fetch_panel(fetch=fetch)
    print(f"^GSPC  {frame.index[0].date()} -> {frame.index[-1].date()}  "
          f"fp={data.fingerprint(frame)}  n={len(frame):,}")
    print()

    # --- Core contrast ---
    s = st.summarize(frame)
    print("=== Window contrast (full sample 1928-2026) ===")
    print(f"  NEW   n={s['n_new']:,}  mean={s['mean_new_bps']:+.2f} bps  "
          f"HAC t={s['t_new']:+.2f}")
    print(f"  FULL  n={s['n_full']:,}  mean={s['mean_full_bps']:+.2f} bps  "
          f"HAC t={s['t_full']:+.2f}")
    print(f"  ALL   n={s['n_total']:,}  mean={s['mean_all_bps']:+.2f} bps")
    print(f"  Contrast (NEW-FULL) = {s['contrast_bps']:+.2f} bps  "
          f"HAC t = {s['t_contrast']:+.2f}")

    # --- Modern era ---
    frame_73 = frame[frame.index.year >= 1973]
    s73 = st.summarize(frame_73)
    print()
    print("=== Post-1973 (YZZ sample period onward) ===")
    print(f"  Contrast = {s73['contrast_bps']:+.2f} bps  HAC t = {s73['t_contrast']:+.2f}")

    frame_oos = frame[frame.index.year >= 2002]
    s_oos = st.summarize(frame_oos)
    print("=== Post-2001 (out-of-sample for YZZ) ===")
    print(f"  Contrast = {s_oos['contrast_bps']:+.2f} bps  HAC t = {s_oos['t_contrast']:+.2f}")

    # --- Decade breakdown ---
    dec = st.by_decade(frame)
    print()
    print("=== Decade breakdown ===")
    print(dec.to_string())

    # --- Shuffled-label control ---
    shuffled = st.shuffled_contrast(frame, n_perms=500, seed=148)
    obs = s["contrast_bps"]
    perm_pval = (shuffled >= obs).mean()
    print()
    print(f"=== Shuffled-label control (n_perms=500) ===")
    print(f"  Observed contrast: {obs:+.2f} bps")
    print(f"  Shuffle mean={np.nanmean(shuffled):+.2f}  std={np.nanstd(shuffled):.2f}")
    print(f"  Perm p-value (obs >= shuffled): {perm_pval:.3f}")

    # --- Strategy Sharpe ---
    try:
        from quantlab import stats as qstats
        strat = st.lunar_strategy_returns(frame)
        ci = qstats.sharpe_ci_bootstrap(strat, periods_per_year=252, seed=148)
        bh_ci = qstats.sharpe_ci_bootstrap(frame["ret"], periods_per_year=252, seed=148)
        print()
        print("=== Long-new / short-full strategy ===")
        print(f"  Strategy ann. Sharpe = {ci['sharpe']:+.2f}  "
              f"CI [{ci['ci_low']:+.2f}, {ci['ci_high']:+.2f}]  "
              f"{ci['frac_negative']*100:.0f}% neg")
        print(f"  B&H ann. Sharpe      = {bh_ci['sharpe']:+.2f}")
    except ImportError:
        print("(quantlab not importable from this path — skipping strategy Sharpe)")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
