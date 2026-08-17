"""Real-tape verification — Study 938 (Open or Close). Regenerates docs/results.md numbers.

Reads cached adjusted OHLC for SPY / IWM / EEM / EFA plus the BIL cash leg, builds the same
moving-average timing rule twice (10-month and 20-week), fills it at the next **open** and
at the next **close**, and measures the gap: annualised bps, HAC *t*, per-trade slippage,
win rate, excess-of-cash Sharpes, a block-bootstrap CI, an era cut, an opening-auction cost
sweep and the cash-split proxy sweep. Network only on ``--fetch``.

    python studies/938-open-vs-close-execution/examples/verify.py            # cache-only
    python studies/938-open-vs-close-execution/examples/verify.py --fetch    # refresh tapes
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from open_close_exec import data, strategy as st  # noqa: E402

RULES = (("M", 10, "10-month"), ("W", 20, "20-week"))
COST_BPS = 2.0
SPLIT = "2017-01-01"


def main(fetch: bool) -> None:
    if fetch:
        data.fetch()
    frames = data.load_ohlc()
    cash = frames[data.CASH]["close"]
    risk = {tk: frames[tk] for tk in data.TICKERS}

    idx = frames[data.TICKERS[0]].index
    print(f"tapes: {', '.join(data.TICKERS)} + {data.CASH} (cash)")
    print(f"window {idx[0].date()} -> {idx[-1].date()}  n={len(idx)}  "
          f"fp={data.fingerprint(frames)}   as-of {data.AS_OF}")
    print(f"one-way cost {COST_BPS:.0f} bps x NAV, both venues; long-only -> no short leg, no borrow")
    print("two rules: 10-month (lookback ~210 sessions) and 20-week (~100) -> the weekly arm is "
          "faster AND shorter-horizon, not a pure change of frequency")

    cells = []
    all_trades = []
    for freq, lb, label in RULES:
        print("\n" + "=" * 78)
        print(f"RULE: {label} moving-average filter (rebalanced at {'month' if freq == 'M' else 'week'} ends)")
        res = st.run_panel(risk, cash, tickers=list(data.TICKERS),
                           freq=freq, lookback=lb, cost_bps=COST_BPS)
        for tk in data.TICKERS:
            arms = res["per_tape"][tk]
            g = st.venue_gap(arms)
            cells.append({"rule": label, "tape": tk, **g})
            all_trades.append(arms.loc[arms["trade"] > 0, "diff"])
            print(f"  {tk}: trades={g['n_trades']:3d}  gap={g['gap_ann_bps']:+7.1f} bps/yr "
                  f"(HAC t={g['gap_t_hac']:+5.2f})  per-trade={g['gap_per_trade_bps']:+7.1f} bps "
                  f"(t={g['gap_per_trade_t']:+5.2f})  win={g['win_rate_trades']:.0%}  "
                  f"exSharpe open {g['sharpe_open']:+.3f} / close {g['sharpe_close']:+.3f}")

        pooled = res["pooled"]
        gp = st.venue_gap(pooled)
        gp_gross = st.venue_gap(pooled, net=False)
        print(f"  POOLED (equal-weight 4 tapes): gap={gp['gap_ann_bps']:+7.1f} bps/yr "
              f"(HAC t={gp['gap_t_hac']:+5.2f}) | gross {gp_gross['gap_ann_bps']:+7.1f} "
              f"(t={gp_gross['gap_t_hac']:+5.2f})")
        print(f"    exSharpe open {gp['sharpe_open']:+.3f} vs close {gp['sharpe_close']:+.3f} "
              f"(d={gp['sharpe_gap']:+.3f})   excess-of-cash CAGR {gp['cagr_open']:+.2%} vs "
              f"{gp['cagr_close']:+.2%}  (raw, cash-inclusive: {gp['cagr_open_total']:+.2%} vs "
              f"{gp['cagr_close_total']:+.2%})")

        ci = st.block_bootstrap_ci(pooled["diff"], stat="mean_ann_bps", seed=938)
        print(f"    block-bootstrap 95% CI on the gap: [{ci['ci_low']:+.1f}, {ci['ci_high']:+.1f}] bps/yr "
              f"(share < 0: {ci['frac_negative']:.1%})")

        eras = st.era_cut(pooled, split=SPLIT)
        for tag, e in eras.items():
            if e is None:
                continue
            print(f"    era {tag:5s} (n={e['n_days']}): gap {e['gap_ann_bps']:+7.1f} bps/yr "
                  f"(t={e['gap_t_hac']:+5.2f})")

        print("    opening-auction penalty sweep (PROXY, extra one-way bps on the open arm):")
        for x in (0.0, 1.0, 2.0, 5.0):
            rp = st.run_panel(risk, cash, tickers=list(data.TICKERS), freq=freq,
                              lookback=lb, cost_bps=COST_BPS, open_extra_bps=x)
            gg = st.venue_gap(rp["pooled"])
            print(f"      +{x:4.1f} bps -> gap {gg['gap_ann_bps']:+7.1f} bps/yr "
                  f"(t={gg['gap_t_hac']:+5.2f})  d-exSharpe {gg['sharpe_gap']:+.3f}")

        print("    cash-split PROXY sweep (overnight share of the T-bill accrual):")
        for row in st.cash_split_sweep(risk["SPY"], cash, freq=freq, lookback=lb, cost_bps=COST_BPS):
            print(f"      f={row['cash_overnight_frac']:.2f} -> SPY gap {row['gap_ann_bps']:+7.2f} "
                  f"bps/yr (t={row['gap_t_hac']:+.3f})")

        print("    mechanism — intraday (price-only) return on trade days:")
        for tk in data.TICKERS:
            d = st.trade_day_decomposition(res["per_tape"][tk])
            print(f"      {tk}: entries {d['n_entries']:3d} {d['entry_intraday_bps']:+7.1f} bps "
                  f"(t={d['entry_t']:+5.2f}) | exits {d['n_exits']:3d} {d['exit_intraday_bps']:+7.1f} bps "
                  f"(t={d['exit_t']:+5.2f}) | all days {d['all_intraday_bps']:+5.2f} bps | "
                  f"spread {d['spread_bps']:+7.1f} (Welch t={d['spread_t']:+5.2f})")

    # ---------------------------------------------------------------- pooled trades
    print("\n" + "=" * 78)
    gaps = np.array([c["gap_ann_bps"] for c in cells])
    ts = np.array([c["gap_t_hac"] for c in cells])
    print(f"DISPERSION across the 8 rule x tape cells: mean {gaps.mean():+.1f} bps/yr, "
          f"sd {gaps.std(ddof=1):.1f}, range [{gaps.min():+.1f}, {gaps.max():+.1f}]; "
          f"|t| >= 2 in {int((np.abs(ts) >= 2).sum())}/8 cells")

    tr = pd.concat(all_trades)
    k = int((tr > 0).sum())
    lo, hi = st.wilson_interval(k, len(tr))
    print(f"ALL {len(tr)} TRADES POOLED: mean slippage {tr.mean() * 1e4:+.2f} bps "
          f"(naive t={st.one_sample_t(tr.to_numpy()):+.2f}), open-wins {k}/{len(tr)} = "
          f"{k / len(tr):.1%}  95% Wilson [{lo:.1%}, {hi:.1%}]")
    # The fills are NOT independent: four correlated tapes flip on the same days and the two
    # rules share the window. Cluster on the trade date before believing any of the above.
    cb = st.cluster_bootstrap_trades(tr.to_numpy(), tr.index)
    print(f"  CLUSTERED on trade date ({cb['n_dates']} distinct dates for {cb['n_fills']} fills): "
          f"t={st.cluster_t(tr.to_numpy(), tr.index):+.2f}  "
          f"mean 95% CI [{cb['mean_ci'][0]:+.2f}, {cb['mean_ci'][1]:+.2f}] bps  "
          f"win-rate 95% CI [{cb['win_ci'][0]:.1%}, {cb['win_ci'][1]:.1%}]")

    # ---------------------------------------------------------------- synthetic control
    print("\n=== synthetic control (machinery proof only — never supports the stamp) ===")
    for ss in (0.0, 1.0):
        rows = [st.synthetic_detect(data.synthetic_daily(signal_strength=ss, seed=938 + 11 * s)[0])
                for s in range(8)]
        g = np.array([r["gap_ann_bps"] for r in rows])
        t = np.array([r["gap_t_hac"] for r in rows])
        tag = "null (no continuation)" if ss == 0 else "planted continuation"
        print(f"  signal_strength={ss:.0f} — {tag}: gap mean {g.mean():+7.1f} bps/yr "
              f"(sd {g.std(ddof=1):.1f}), HAC t mean {t.mean():+.2f}, "
              f"fires |t|>=2 on {int((np.abs(t) >= 2).sum())}/8 seeds")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
