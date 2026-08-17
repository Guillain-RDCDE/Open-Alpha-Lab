"""Real-tape verification — Study 939 (DRIP or Sweep). Regenerates docs/results.md.

Reconstructs each fund's per-share distribution stream from its price-only and
total-return legs, audits the reconstruction, then races two accounting policies over
the same holding: reinvest on the pay date (DRIP) vs park in BIL and reinvest at the
next quarter (or year) end. Prints the terminal-wealth gap in annualised bps, the HAC
*t* on the daily log-return difference, block-bootstrap CIs, the era and rate-regime
cuts, and the pay-lag / cost / frequency sweeps. Network only on ``--fetch``.

    python studies/939-drip-vs-sweep/examples/verify.py            # cache-only
    python studies/939-drip-vs-sweep/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from drip_sweep import data, strategy as st  # noqa: E402

PAY_LAG = data.PAY_LAG_DAYS
DRIP_C = data.DRIP_COST_BPS
SWEEP_C = data.SWEEP_COST_BPS


def build_tapes():
    """Return {ticker: (price_only, dividend_ps, cash_index)} on the common window."""
    tr = data.load_prices()
    dist = data.load_distributions()
    cash = tr[data.CASH].dropna()
    tapes = {}
    for tk in data.PAYERS:
        px = dist[tk]["close"].dropna()
        trc = tr[tk].dropna()
        divs = data.reconstruct_dividends(px, trc)
        common = px.index.intersection(cash.index)
        tapes[tk] = (px.loc[common], divs.reindex(common).fillna(0.0), cash.loc[common])
    return tr, dist, tapes


def main(fetch: bool) -> None:
    if fetch:
        data.fetch()
    tr, dist, tapes = build_tapes()

    # The data stamp fingerprints the tape the RACE actually sees (each payer's
    # price-only close + reconstructed distribution + the BIL cash leg over the study
    # window), not the raw cached frames. The shared desk cache is written by whichever
    # study fetched a ticker first, so its pre-window history can grow or shrink under
    # us; fingerprinting the raced window keeps this stamp reproducible.
    stamp = pd.concat(
        {tk: pd.concat({"px": tapes[tk][0], "div": tapes[tk][1], "cash": tapes[tk][2]},
                       axis=1) for tk in data.PAYERS}, axis=1)
    print(f"as-of {data.AS_OF}  |  fingerprint {data.fingerprint(stamp.fillna(0.0))} "
          f"(raced window {stamp.index[0].date()} -> {stamp.index[-1].date()}, "
          f"{len(stamp)} days)")
    print(f"assumptions: pay lag {PAY_LAG} calendar days (PROXY), "
          f"DRIP cost {DRIP_C} bps, sweep cost {SWEEP_C} bps (one-way x amount reinvested)")

    print("\n=== dividend reconstruction: price leg vs total-return leg (study window) ===")
    for tk in data.PAYERS:
        px, _, _ = tapes[tk]
        chk = data.dividend_reconstruction_check(
            px, tr[tk].reindex(px.index), dist[tk]["dividend"].reindex(px.index).fillna(0.0))
        print(f"  {tk:5s} {px.index[0].date()} -> {px.index[-1].date()}: "
              f"{chk['n_events_reconstructed']:3d} events reconstructed / "
              f"{chk['n_events_reported']:3d} reported, {chk['n_events_matched']:3d} matched  |  "
              f"cash/share {chk['total_ps_reconstructed']:.3f} vs {chk['total_ps_reported']:.3f} "
              f"(ratio {chk['ratio_rec_over_rep']:.4f}, corr {chk['event_corr']:.4f})")

    print("\n=== audit: a zero-lag, zero-cost DRIP must reproduce the total-return index ===")
    for tk in data.PAYERS:
        px, divs, _ = tapes[tk]
        a = st.drip_tracks_total_return(px, divs, tr[tk].dropna())
        print(f"  {tk:5s}: terminal ratio {a['terminal_ratio']:.5f}  "
              f"max |dev| {a['max_abs_dev_pct']:.3f}%  tracking {a['ann_tracking_bps']:+.2f} bps/yr")

    print("\n=== headline race: DRIP vs quarterly sweep (excess-of-cash Sharpes) ===")
    races = {}
    for tk in data.PAYERS:
        px, divs, cash = tapes[tk]
        r = st.race(px, divs, cash, pay_lag_days=PAY_LAG,
                    drip_cost_bps=DRIP_C, sweep_cost_bps=SWEEP_C, sweep_freq="Q")
        races[tk] = r
        ci = st.bootstrap_gap_ci(r["dlog"], seed=939)
        # The REALISED in-sample distribution yield (not the fund's current headline
        # yield): it is what the mechanism actually scales with.
        yld = float((divs / px).sum() / (len(px) / 252.0))
        print(f"  {tk:5s} {r['start'].date()} -> {r['end'].date()} ({r['years']:.1f} yr, "
              f"{r['drip']['n_days']} days, realised distribution yield {yld:.2%})")
        # NB both Sharpe and CAGR here are EXCESS OF CASH (the series is r - r_BIL),
        # so the CAGR column is an excess-of-cash compounding rate, not the fund's.
        print(f"        DRIP   exSharpe {r['drip']['sharpe']:+.4f}  exCAGR {r['drip']['cagr']:+.4%}  "
              f"terminal {r['terminal_drip']:,.0f}  trades {r['n_trades_drip']}")
        print(f"        SWEEP  exSharpe {r['sweep']['sharpe']:+.4f}  exCAGR {r['sweep']['cagr']:+.4%}  "
              f"terminal {r['terminal_sweep']:,.0f}  trades {r['n_trades_sweep']}")
        print(f"        gap {r['gap_bps_per_year']:+.2f} bps/yr  HAC t {r['t_hac_dlog']:+.2f}  "
              f"terminal ratio {r['terminal_ratio']:.5f}  "
              f"95% CI [{ci['ci_low']:+.2f}, {ci['ci_high']:+.2f}]  share<0 {ci['frac_negative']:.3f}")

    print("\n=== pooled (equal-weight across the three funds) ===")
    for freq in ("Q", "A"):
        cols = {}
        for tk in data.PAYERS:
            px, divs, cash = tapes[tk]
            cols[tk] = st.race(px, divs, cash, pay_lag_days=PAY_LAG, drip_cost_bps=DRIP_C,
                               sweep_cost_bps=SWEEP_C, sweep_freq=freq)["dlog"]
        pooled = pd.concat(cols, axis=1).mean(axis=1).dropna()
        ci = st.bootstrap_gap_ci(pooled, seed=939)
        print(f"  sweep_freq={freq}: gap {pooled.mean() * 252 * 1e4:+.2f} bps/yr  "
              f"HAC t {st.newey_west_t(pooled.to_numpy()):+.2f}  n={len(pooled)}  "
              f"95% CI [{ci['ci_low']:+.2f}, {ci['ci_high']:+.2f}]  share<0 {ci['frac_negative']:.3f}")

    print("\n=== monotone-in-yield check on the COMMON window (SCHD's start onward) ===")
    # The headline ordering SPY < VYM < SCHD mixes yield with WINDOW: SCHD's tape is
    # 2011-2026 while SPY/VYM run from 2007. Re-race all three on SCHD's window so the
    # only thing that differs is the fund.
    start_common = tapes["SCHD"][0].index[0]
    for tk in data.PAYERS:
        px, divs, cash = tapes[tk]
        px2 = px[px.index >= start_common]
        d2 = divs.reindex(px2.index).fillna(0.0)
        c2 = cash.reindex(px2.index)
        r = st.race(px2, d2, c2, pay_lag_days=PAY_LAG, drip_cost_bps=DRIP_C,
                    sweep_cost_bps=SWEEP_C, sweep_freq="Q")
        realised_yield = float((d2 / px2).sum() / (len(px2) / 252.0))
        print(f"  {tk:5s} realised yield {realised_yield:.2%}: "
              f"gap {r['gap_bps_per_year']:+.2f} bps/yr  t {r['t_hac_dlog']:+.2f}  "
              f"({r['gap_bps_per_year'] / (realised_yield * 100):.2f} bps per 1% of yield)")

    print("\n=== HAC lag robustness of the headline t (auto lag is the LOWEST) ===")
    for tk in data.PAYERS:
        x = races[tk]["dlog"].to_numpy()
        cells = "  ".join(f"{tag} {st.newey_west_t(x, lags=L):+.2f}"
                          for tag, L in (("auto", None), ("63", 63), ("252", 252)))
        print(f"  {tk:5s} {cells}")

    print("\n=== sweep frequency: how much does waiting a whole year cost? ===")
    for tk in data.PAYERS:
        px, divs, cash = tapes[tk]
        for row in st.frequency_sweep(px, divs, cash, pay_lag_days=PAY_LAG,
                                      drip_cost_bps=DRIP_C, sweep_cost_bps=SWEEP_C):
            print(f"  {tk:5s} {row['sweep_freq']}: gap {row['gap_bps_per_year']:+.2f} bps/yr  "
                  f"t {row['t_hac_dlog']:+.2f}  terminal ratio {row['terminal_ratio']:.5f}  "
                  f"sweep trades {row['n_trades_sweep']}")

    print("\n=== era cut (split 2016-01-01) ===")
    for tk in data.PAYERS:
        px, divs, cash = tapes[tk]
        for tag, e in st.era_cut(px, divs, cash, split="2016-01-01", pay_lag_days=PAY_LAG,
                                 drip_cost_bps=DRIP_C, sweep_cost_bps=SWEEP_C).items():
            if e is None:
                print(f"  {tk:5s} {tag:5s}: too short")
                continue
            print(f"  {tk:5s} {tag:5s} ({e['years']:.1f} yr): gap {e['gap_bps_per_year']:+.2f} bps/yr  "
                  f"t {e['t_hac_dlog']:+.2f}  ratio {e['terminal_ratio']:.5f}")

    print("\n=== rate-regime cut (calendar years by realised BIL return, threshold 2%) ===")
    for tk in data.PAYERS:
        px, divs, cash = tapes[tk]
        for tag, e in st.rate_regime_cut(px, divs, cash, threshold=0.02, pay_lag_days=PAY_LAG,
                                         drip_cost_bps=DRIP_C, sweep_cost_bps=SWEEP_C).items():
            if e is None:
                continue
            print(f"  {tk:5s} {tag:9s} ({e['n_years_calendar']} calendar yr, {e['n_days']} days): "
                  f"gap {e['gap_bps_per_year']:+.2f} bps/yr  t {e['t_hac_dlog']:+.2f}")

    print("\n=== pay-lag sweep (the ASSUMPTION the tape cannot see) ===")
    for tk in data.PAYERS:
        px, divs, cash = tapes[tk]
        rows = st.pay_lag_sweep(px, divs, cash, lags=(0, 15, 30, 45),
                                drip_cost_bps=DRIP_C, sweep_cost_bps=SWEEP_C)
        print("  " + tk + ": " + "  ".join(
            f"{r['pay_lag_days']:>2d}d {r['gap_bps_per_year']:+.2f}(t{r['t_hac_dlog']:+.2f})"
            for r in rows))

    print("\n=== cost sweep (drip_bps, sweep_bps), one-way x amount reinvested ===")
    for tk in data.PAYERS:
        px, divs, cash = tapes[tk]
        rows = st.cost_sweep(px, divs, cash, pay_lag_days=PAY_LAG)
        print("  " + tk + ": " + "  ".join(
            f"({r['drip_cost_bps']:.0f},{r['sweep_cost_bps']:.0f}) {r['gap_bps_per_year']:+.2f}"
            for r in rows))

    print("\n=== what the gap is worth in money (10,000 invested) ===")
    for tk in data.PAYERS:
        r = races[tk]
        diff = r["terminal_drip"] - r["terminal_sweep"]
        print(f"  {tk:5s}: {r['terminal_drip']:,.0f} vs {r['terminal_sweep']:,.0f} "
              f"= {diff:+,.0f} over {r['years']:.1f} yr ({diff / r['years']:+,.1f} / yr)")

    print("\n=== synthetic control (machinery proof only - never supports the stamp) ===")
    kw = dict(pay_lag_days=PAY_LAG, drip_cost_bps=DRIP_C, sweep_cost_bps=SWEEP_C)
    for freq in ("Q", "A"):
        for ss in (1.0, 0.0):
            r = st.seed_sweep(data.synthetic_daily, ss, n_seeds=8, sweep_freq=freq, **kw)
            tag = "planted premium" if ss else "null (no premium)"
            print(f"  {freq} {tag:17s}: gap mean {r['mean']:+.2f} bps/yr "
                  f"(sd {r['sd']:.2f}, se {r['se']:.2f})")

    print("\n  monotone in distribution yield (planted, quarterly sweep):")
    frames, truth = data.synthetic_panel(signal_strength=1.0, seed=939)
    for tk, y in zip(truth["tickers"], truth["div_yields"]):
        g = st.synthetic_detect(frames[tk], sweep_freq="Q", **kw)["gap_bps_per_year"]
        print(f"    yield {y:.0%}: gap {g:+.2f} bps/yr")

    print("\n  power check at MARKET-realistic parameters (5.5% premium, 3% yield, 16% vol):")
    real = dict(equity_premium=0.055, div_yield_ann=0.03, vol_ann=0.16)
    for ss in (1.0, 0.0):
        r = st.seed_sweep(data.synthetic_daily, ss, n_seeds=8, gen_kw=real,
                          sweep_freq="Q", **kw)
        tag = "planted premium" if ss else "null (no premium)"
        print(f"    {tag:17s}: gap mean {r['mean']:+.2f} bps/yr "
              f"(sd {r['sd']:.2f}, se {r['se']:.2f})")
    print("    -> at realistic parameters ONE 20-year tape cannot separate the two. "
          "The real-tape t of ~1.2 is what that looks like.")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
