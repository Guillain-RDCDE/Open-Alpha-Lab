"""Real-panel verification — Study 578 (Cross-Asset-Correlation-Regime). Regenerates docs/results.md.

Reads this study's own yfinance cache (daily cross-asset ETF returns), builds the trailing mean
pairwise correlation index, splits the tape into HIGH/LOW regimes by its expanding 70th percentile,
and tests the *fragility claim*: does a HIGH-correlation regime predict LOWER forward returns and
HIGHER forward volatility on SPY? Prints every number that lands in docs/results.md, plus the
block-shuffle placebo null, the coincident-drawdown mechanism, the robustness sweep, the risk-off
overlay net of costs, and the deterministic seed-robust synthetic positive control.

    python studies/578-cross-asset-correlation-regime/examples/verify.py

Everything runs offline once the cache exists (populate it once with data.fetch_returns(fetch=True)).
No look-ahead: the regime label uses only trailing data + an expanding quantile, and the forward
window is strictly after day t; the tradable overlay enters one bar later (the single execution lag).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import pandas as pd  # noqa: E402

from cross_asset_correlation_regime import data, strategy as st  # noqa: E402

WINDOW, HORIZON, Q = 63, 21, 0.70


def main() -> None:
    rets = data.fetch_returns(fetch=False)
    if rets.empty or "SPY" not in rets.columns:
        print("Cache not found. Run once with network: data.fetch_returns(fetch=True).")
        return

    print("Study 578 — Cross-Asset-Correlation-Regime — real yfinance tape\n")
    fp = data.fingerprint(rets)
    print(f"Returns: {rets.shape}, fingerprint={fp}, "
          f"{rets.index[0].date()} -> {rets.index[-1].date()}")
    print(f"Index: trailing-{WINDOW}d mean pairwise corr | regime: expanding {Q:.0%} pctile | "
          f"forward: SPY next {HORIZON}d | as-of 2026-06-30\n")

    res = st.regime_forward_test(rets, window=WINDOW, horizon=HORIZON, q=Q, min_periods=252)
    frame = res["_df"][["regime", "fwd_ret", "fwd_vol", "avg_corr"]]
    print(f"Aligned regime/forward frame fingerprint={data.fingerprint(frame)}\n")

    print("--- Regime forward test (window=63, horizon=21, q=0.70) ---")
    print(f"  n_high={res['n_high']}  n_low={res['n_low']}")
    print(f"  fwd RETURN : HIGH {res['high_fwd_ret']*100:+.2f}%  LOW {res['low_fwd_ret']*100:+.2f}%  "
          f"spread {res['ret_spread']*100:+.2f}% (t {res['ret_t']:+.2f})  [claim: negative]")
    print(f"  fwd VOL    : HIGH {res['high_fwd_vol']*100:.1f}%  LOW {res['low_fwd_vol']*100:.1f}%  "
          f"spread {res['vol_spread']*100:+.2f}% (t {res['vol_t']:+.2f})  [claim: positive]")
    p = st.placebo_pvalue(res, n_perm=2000, block=21, seed=578)
    print(f"  block-shuffle placebo p (return spread) = {p:.3f}")
    print(f"  avg cross-asset correlation mean = {res['avg_corr_mean']:.3f}\n")

    # coincident drawdown mechanism
    ac = st.avg_pairwise_corr(rets, window=WINDOW)
    lab = st.regime_labels(ac, q=Q, min_periods=252)
    spy = rets["SPY"]
    eq = (1 + spy).cumprod()
    dd = eq / eq.cummax() - 1
    d = pd.DataFrame({"lab": lab, "dd": dd}).dropna()
    print("--- Coincident-drawdown mechanism ---")
    print(f"  mean contemporaneous SPY drawdown: HIGH {d[d.lab == 1].dd.mean()*100:+.1f}%  "
          f"LOW {d[d.lab == 0].dd.mean()*100:+.1f}%\n")

    # robustness sweep
    specs = [(63, 21, 0.70), (63, 21, 0.80), (63, 63, 0.70), (126, 21, 0.70),
             (126, 63, 0.70), (42, 21, 0.70), (63, 5, 0.70)]
    sw = st.robustness_sweep(rets, specs)
    print("--- Robustness sweep (ret_spread positive = wrong sign for the claim) ---")
    print(sw.assign(ret_spread=(sw["ret_spread"] * 100).round(2),
                    vol_spread=(sw["vol_spread"] * 100).round(2),
                    ret_t=sw["ret_t"].round(2), vol_t=sw["vol_t"].round(2)).to_string(index=False))
    print()

    # overlay
    ov = st.overlay_backtest(rets, window=WINDOW, q=Q, cost_bps=5.0)
    print("--- Risk-off overlay (flat in HIGH regime, 1-day lag, 5 bps/switch) ---")
    for k in ["overlay_gross", "overlay_net", "buy_hold"]:
        s = ov[k]
        print(f"  {k:14s}: CAGR {s['cagr']*100:+.2f}%  vol {s['vol']*100:.1f}%  "
              f"Sharpe {s['sharpe']:+.2f}  maxDD {s['maxdd']*100:.1f}%")
    print(f"  switches={ov['switches']}  frac_high={ov['frac_high']:.3f}\n")

    # synthetic positive control
    print("--- Synthetic positive control (mean over 25 seeds) ---")
    for frag in [0.0, 0.5, 0.9, 1.4]:
        sp, t = st.synthetic_mean_t(data, fragility=frag, n_seeds=25)
        print(f"  fragility {frag:+.2f} -> mean spread {sp*100:+.2f}%  mean t {t:+.2f}")


if __name__ == "__main__":
    main()
