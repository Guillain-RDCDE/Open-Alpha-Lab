"""Real-tape verification — Study 167 (Hindenburg-Omen).  Regenerates docs/results.md numbers.

Reads the cached breadth frame (built by data.load_breadth(fetch=True)), runs the full
summarize() pipeline across horizons 30/60/90/120 days, and reports signal vs base returns,
crash rates, and multiple-comparison-corrected p-values.

    python studies/167-hindenburg-omen/examples/verify.py          # cache-only
    python studies/167-hindenburg-omen/examples/verify.py --fetch  # refresh data
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from hindenburg_omen import data, strategy as st  # noqa: E402

HORIZONS = (30, 60, 90, 120)


def main(fetch: bool) -> None:
    if fetch:
        print("Fetching constituent panel and SPY … (this can take a few minutes)")
        breadth = data.load_breadth(fetch=True, threshold_pct=data.THRESHOLD_PCT)
        print(f"Breadth frame: {len(breadth)} days, "
              f"{int((breadth['hindenburg']==1).sum())} raw Hindenburg days")
        print(f"Fingerprint: {data.fingerprint(breadth)}")
        print(f"Date range: {breadth.index[0].date()} → {breadth.index[-1].date()}")
        print()
    else:
        breadth = data.load_breadth(fetch=False, threshold_pct=data.THRESHOLD_PCT)

    sig = st.cluster_signals(breadth, window_days=30)
    print(f"Raw signal days : {int((breadth['hindenburg']==1).sum())}")
    print(f"Signal clusters : {int(sig.sum())}")
    print()

    fwd_sig = st.forward_returns(breadth, sig, horizons=HORIZONS)
    fwd_base = st.unconditional_forward_returns(breadth, horizons=HORIZONS, sample_every=21)

    print(f"{'horizon':>8} | {'sig n':>6} | {'sig mean%':>10} | {'base mean%':>10} | "
          f"{'Welch-t':>8} | {'p_raw':>7} | {'p_bonf':>7}")
    print("-" * 72)
    for h in HORIZONS:
        col = f"ret{h}"
        s = fwd_sig[col].dropna().to_numpy()
        b = fwd_base[col].dropna().to_numpy()
        s_mean = float(s.mean() * 100) if len(s) else float("nan")
        b_mean = float(b.mean() * 100) if len(b) else float("nan")
        t = st._welch_t(s, b)
        t_hac = st._hac_tstat(s)

        from scipy import stats as scipy_stats
        df_approx = max(len(s) + len(b) - 2, 1)
        import numpy as np
        p_raw = float(2.0 * scipy_stats.t.sf(abs(t), df=df_approx)) if np.isfinite(t) else float("nan")
        n_hyp = len(HORIZONS) * 3
        p_bonf = min(1.0, p_raw * n_hyp) if np.isfinite(p_raw) else float("nan")

        print(f"{h:>7}d | {len(s):>6} | {s_mean:>+10.2f} | {b_mean:>+10.2f} | "
              f"{t:>+8.2f} | {p_raw:>7.4f} | {p_bonf:>7.4f}")

    print()
    print("=== crash rate (≥5% drawdown within 120 days) ===")
    cr = st.crash_rate(breadth, sig, horizon=120, drawdown_threshold=-0.05)
    print(f"signal crash rate : {cr['signal_crash_rate']:.1%}  "
          f"({cr['n_crashes_after_signal']} / {cr['n_signals']})")
    print(f"base crash rate   : {cr['base_crash_rate']:.1%}")
    print(f"false-alarm rate  : {cr['false_alarm_rate']:.1%}")
    print(f"Welch-t (sig vs base proportions): {cr['welch_t']:+.2f}")

    print()
    print("=== threshold sensitivity (signal days only, 60-day horizon) ===")
    for th in data.THRESHOLD_VARIANTS:
        import pandas as pd
        breadth_th = data.load_breadth(fetch=False, threshold_pct=th)
        sig_th = st.cluster_signals(breadth_th, window_days=30)
        fwd_th = st.forward_returns(breadth_th, sig_th, horizons=(60,))
        s_arr = fwd_th["ret60"].dropna().to_numpy()
        s_mean = float(s_arr.mean() * 100) if len(s_arr) else float("nan")
        t_hac = st._hac_tstat(s_arr)
        print(f"  threshold {th:.1f}%: {int(sig_th.sum())} clusters, "
              f"60d mean {s_mean:+.2f}%, HAC t = {t_hac:+.2f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
