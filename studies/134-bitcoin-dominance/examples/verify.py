"""Real-tape verification — Study 134 (Bitcoin-Dominance). Regenerates docs/results.md numbers.

Fetches (or reads from cache) the BTC+alt daily panel, constructs the BTC dominance
proxy, tests whether falling dominance forecasts forward alt-minus-BTC returns across
three signal variants, and sweeps the horizon. Network is touched only with --fetch.

    python studies/134-bitcoin-dominance/examples/verify.py            # cache-only
    python studies/134-bitcoin-dominance/examples/verify.py --fetch    # refresh the panel
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bitcoin_dominance import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    panel = data.fetch_panel(fetch=fetch)
    print(f"Panel: {panel.index[0].date()} .. {panel.index[-1].date()}")
    print(f"Tickers: {list(panel.columns)}")
    print(f"Fingerprint: {data.fingerprint(panel)}")
    print(f"Rows: {len(panel)}")
    print()

    # Print per-ticker stats
    print("=== per-ticker stats ===")
    for col in panel.columns:
        print(f"  {col:10s}: {panel.index[0].date()} .. {panel.index[-1].date()}  "
              f"close={panel[col].iloc[-1]:.2f}")
    print()

    # Compute BTC dominance
    dom = data.compute_dominance(panel)
    print(f"BTC dominance proxy: mean={dom.mean():.3f}  min={dom.min():.3f}  "
          f"max={dom.max():.3f}  std={dom.std():.4f}")
    print()

    # Main analysis: three signal variants
    print("=== signal vs base-rate (horizon=5d, i.e. 1 week) ===")
    for lookback in (10, 20):
        sig = st.dominance_signal(panel, lookback=lookback)
        pairs = st.signal_outcome_pairs(panel, sig, horizon=5)
        alt_season = pairs[pairs["signal"] < 0]["spread_fwd"]
        btc_season = pairs[pairs["signal"] > 0]["spread_fwd"]
        base = pairs["spread_fwd"]
        reg = st.regression_tstat(pairs["spread_fwd"], pairs["signal"])

        s_alt = st.summarize(alt_season, label="alt-season (sig<0)")
        s_btc = st.summarize(btc_season, label="btc-season (sig>0)")
        s_base = st.summarize(base, label="unconditional")

        print(f"\nSignal: dom_chg_{lookback}d  (n_total={len(pairs)}, n_alt={len(alt_season)})")
        print(f"  alt-season (sig<0): mean={s_alt['mean_pct']:+.3f}%/wk  t={s_alt['tstat']:+.2f}")
        print(f"  btc-season (sig>0): mean={s_btc['mean_pct']:+.3f}%/wk  t={s_btc['tstat']:+.2f}")
        print(f"  unconditional:      mean={s_base['mean_pct']:+.3f}%/wk  t={s_base['tstat']:+.2f}")
        print(f"  regression slope:   slope={reg['slope']:+.4f}  t={reg['tstat']:+.2f}  R²={reg['r2']:.4f}")

    # Horizon sweep
    print("\n=== horizon sweep (dom_chg_10d) ===")
    sig10 = st.dominance_signal(panel, lookback=10)
    for h in (1, 5, 10, 21):
        pairs = st.signal_outcome_pairs(panel, sig10, horizon=h)
        alt_season = pairs[pairs["signal"] < 0]["spread_fwd"]
        reg = st.regression_tstat(pairs["spread_fwd"], pairs["signal"])
        s = st.summarize(alt_season)
        print(f"  horizon={h:2d}d: alt-season n={len(alt_season):4d} "
              f"mean={s['mean_pct']:+.3f}%  t={s['tstat']:+.2f}  reg_t={reg['tstat']:+.2f}")

    # Dominance percentile signal
    print("\n=== dominance percentile signal (low pct = alt-season) ===")
    pct_sig = st.dominance_percentile(panel, window=252)
    for thresh in (0.25, 0.50):
        # Signal: low dominance percentile → expect alt outperformance
        low_dom = pct_sig[pct_sig <= thresh].rename("signal") * 0  # convert to signal series
        # We need to use the full pairs approach: signal = (dom_pct - 0.5), i.e. negative = below median
        centered_sig = (pct_sig - 0.5).rename("signal")
        pairs = st.signal_outcome_pairs(panel, centered_sig, horizon=5)
        alt_season = pairs[pairs["signal"] < thresh - 0.5]["spread_fwd"]
        reg = st.regression_tstat(pairs["spread_fwd"], pairs["signal"])
        s = st.summarize(alt_season)
        print(f"  dom_pct<={thresh:.2f}: n={len(alt_season):4d} "
              f"mean={s['mean_pct']:+.3f}%/wk  t={s['tstat']:+.2f}  reg_t={reg['tstat']:+.2f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
