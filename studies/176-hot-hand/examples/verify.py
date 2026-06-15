"""Real-tape verification — Study 176 (Hot-Hand). Regenerates docs/results.md numbers.

Fetches (or reads from cache) the S&P 500 daily tape, computes the full streak table
across all lengths 1–6 in both directions (up/down), runs the hot-hand follow rule
and the gambler's-fallacy fade rule, applies Bonferroni correction, and sweeps costs.
Network is touched only with --fetch.

    python studies/176-hot-hand/examples/verify.py            # cache-only
    python studies/176-hot-hand/examples/verify.py --fetch    # refresh the tape
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from hot_hand import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    cache = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_cache"))
    bars = data.load_gspc(fetch=fetch, cache_dir=cache)

    if fetch:
        print(f"^GSPC  {bars.index[0].date()}..{bars.index[-1].date()}  "
              f"fp={data.fingerprint(bars)}  n={len(bars)}")
        print()

    print("=== FULL STREAK TABLE (cost=1bps) ===")
    tbl = st.streak_table(bars["close"], max_streak=st.MAX_STREAK, cost_bps=1.0)
    print(tbl[["direction", "streak_len", "n", "continuation", "binom_p",
               "excess_bps", "tstat", "net_bps"]].to_string(index=False))

    bf = st.bonferroni_summary(tbl)
    print(f"\nBonferroni ({bf['n_tests']} tests, alpha_adj={bf['alpha_bonf']:.4f}): "
          f"{bf['sig_raw']} raw, {bf['sig_bonf']} after correction")

    print("\n=== HOT-HAND (follow up-streaks, min_streak=2) ===")
    for cost in (0.0, 1.0):
        lh = st.trade_streak(bars["close"], min_streak=2, direction=1,
                              follow_streak=True, cost_bps=cost)
        s = st.summarize(lh, "ret_net" if cost > 0 else "ret_gross")
        col = "ret_net" if cost > 0 else "ret_gross"
        label = f"cost={cost:.1f}"
        print(f"  {label}: n={s['n_trades']:4d} mean={s['mean_bps']:+.2f}bps "
              f"t={s['tstat']:+.3f} win={s['win_rate']:.3f}")

    print("\n=== GAMBLER'S FALLACY (fade up-streaks, min_streak=2) ===")
    for cost in (0.0, 1.0):
        lg = st.trade_streak(bars["close"], min_streak=2, direction=1,
                              follow_streak=False, cost_bps=cost)
        s = st.summarize(lg, "ret_net" if cost > 0 else "ret_gross")
        label = f"cost={cost:.1f}"
        print(f"  {label}: n={s['n_trades']:4d} mean={s['mean_bps']:+.2f}bps "
              f"t={s['tstat']:+.3f} win={s['win_rate']:.3f}")

    print("\n=== DOWN-STREAK REVERSAL (fade down-streaks, min_streak=3) ===")
    for cost in (0.0, 0.5, 1.0, 2.0, 5.0):
        ld = st.trade_streak(bars["close"], min_streak=3, direction=-1,
                              follow_streak=False, cost_bps=cost)
        col = "ret_net" if cost > 0 else "ret_gross"
        s = st.summarize(ld, col)
        label = f"cost={cost:.1f}"
        print(f"  {label}: n={s['n_trades']:4d} mean={s['mean_bps']:+.2f}bps "
              f"t={s['tstat']:+.3f} win={s['win_rate']:.3f}")

    print("\n=== UNCONDITIONAL ===")
    df = st.label_streaks(bars["close"])
    df_valid = df.dropna(subset=["next_ret"])
    uncond_bps = float(df_valid["next_ret"].mean() * 1e4)
    n = len(df_valid)
    r = df_valid["next_ret"].to_numpy()
    mu = r.mean()
    e = r - mu
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    t_uncond = float(mu / se) if se > 0 else float("nan")
    print(f"  n={n} mean={uncond_bps:.3f}bps t={t_uncond:.3f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
