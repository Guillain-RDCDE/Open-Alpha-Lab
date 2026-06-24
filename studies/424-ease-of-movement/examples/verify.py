"""Real-tape verification — Study 424 (Ease of Movement). Regenerates docs/results.md.

Reads (or fetches with --fetch) six daily ETF tapes, builds the EOM(14) long/flat timing
rule, and races its NET excess-of-cash Sharpe against buy-and-hold, an SMA(50/200) cross,
and MACD(12/26/9) — with a HAC t-stat, a sign-shuffle permutation placebo, a cost sweep,
and a period sweep. The headline tape is SPY. Network is touched only on a cache miss or
with --fetch; everything else is offline and deterministic.

    python studies/424-ease-of-movement/examples/verify.py            # cache-first
    python studies/424-ease-of-movement/examples/verify.py --fetch    # refresh the tapes

As-of pin: the headline run trims every tape to the last complete month (2026-05-31) so
no partial month enters the stamped statistics.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ease_of_movement import data, strategy as st  # noqa: E402

TICKERS = ["SPY", "QQQ", "IWM", "DIA", "EFA", "EEM"]
ASOF = "2026-05-31"
CACHE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_cache"))


def main(fetch: bool) -> None:
    if fetch:
        for t in TICKERS:
            b = data.load_real(t, fetch=True, cache_dir=CACHE)
            print(f"{t:5s} {b.index[0].date()}..{b.index[-1].date()} fp={data.fingerprint(b)}")
        print()

    print(f"=== Data stamp (trimmed to as-of {ASOF}) ===")
    for t in TICKERS:
        b = data.load_real(t, cache_dir=CACHE).loc[:ASOF]
        print(f"{t:5s} {b.index[0].date()}..{b.index[-1].date()} n={len(b)} "
              f"fp={data.fingerprint(b)}")

    spy = data.load_real("SPY", cache_dir=CACHE).loc[:ASOF]
    r = st.run_experiment(spy, period=14, long_short=False, cost_bps=1.0, n_perm=2000)
    e, g, bh = r["eom"], r["eom_gross"], r["bh"]
    sma, macd = r["sma"], r["macd"]

    print("\n=== SPY — EOM(14) long/flat, NET 1 bp, excess-of-cash ===")
    print(f"  EOM net   Sharpe={e['sharpe']:+.3f}  HAC t={e['tstat']:+.2f}  "
          f"ret={e['ann_ret']*100:+.2f}%  vol={e['ann_vol']*100:.1f}%  "
          f"maxDD={e['max_dd']*100:.1f}%  exposure={e['exposure']:.3f}  "
          f"turn={e['ann_turnover']:.1f}/yr")
    print(f"  EOM gross Sharpe={g['sharpe']:+.3f}  HAC t={g['tstat']:+.2f}")
    print(f"  Buy&Hold  Sharpe={bh['sharpe']:+.3f}  ret={bh['ann_ret']*100:+.2f}%  "
          f"maxDD={bh['max_dd']*100:.1f}%")
    print(f"  SMA50/200 Sharpe={sma['sharpe']:+.3f}  HAC t={sma['tstat']:+.2f}")
    print(f"  MACD      Sharpe={macd['sharpe']:+.3f}  HAC t={macd['tstat']:+.2f}")

    print("\n=== Difference tests (positive t = EOM wins) ===")
    print(f"  EOM - B&H : ann={r['vs_bh']['mean_ann']*100:+.2f}%  HAC t={r['vs_bh']['tstat']:+.2f}")
    print(f"  EOM - SMA : ann={r['vs_sma']['mean_ann']*100:+.2f}%  HAC t={r['vs_sma']['tstat']:+.2f}")
    print(f"  EOM - MACD: ann={r['vs_macd']['mean_ann']*100:+.2f}%  HAC t={r['vs_macd']['tstat']:+.2f}")

    print("\n=== Sign-shuffle permutation placebo (gross Sharpe, 2000 draws) ===")
    p = r["perm"]
    print(f"  observed Sharpe={p['observed']:+.3f}  p-value={p['p_value']:.4f}  "
          f"null mean={p['null_mean']:+.3f}")

    print("\n=== Cost sweep (SPY long/flat, net) ===")
    for c in (0.0, 1.0, 2.0, 5.0, 10.0):
        rr = st.run_experiment(spy, cost_bps=c, n_perm=1)
        print(f"  cost={c:5.1f}bp  net Sharpe={rr['eom']['sharpe']:+.3f}  "
              f"t={rr['eom']['tstat']:+.2f}  ret={rr['eom']['ann_ret']*100:+.2f}%")

    print("\n=== Period sweep (SPY long/flat, net 1bp) ===")
    for per in (9, 14, 20, 30, 40):
        rr = st.run_experiment(spy, period=per, cost_bps=1.0, n_perm=1)
        print(f"  EOM({per:2d})  net Sharpe={rr['eom']['sharpe']:+.3f}  "
              f"t={rr['eom']['tstat']:+.2f}  vs-B&H t={rr['vs_bh']['tstat']:+.2f}")

    print("\n=== Per-instrument (EOM long/flat, net 1bp) ===")
    for t in TICKERS:
        b = data.load_real(t, cache_dir=CACHE).loc[:ASOF]
        rr = st.run_experiment(b, period=14, cost_bps=1.0, n_perm=1)
        print(f"  {t:5s} EOM Sharpe={rr['eom']['sharpe']:+.3f} t={rr['eom']['tstat']:+.2f} "
              f"| B&H Sharpe={rr['bh']['sharpe']:+.3f} | vs-B&H t={rr['vs_bh']['tstat']:+.2f}")

    print("\n=== Synthetic positive control (long/short, gross, cost 0) ===")
    for ed in (-0.10, 0.0, 0.30, 0.60, 1.00):
        d, _ = data.synthetic_panel(n_days=2500, edge=ed, seed=424)
        eom = st.ease_of_movement(d, period=14)
        pos = st.eom_position(eom, long_short=True)
        bk = st.position_returns(d, pos, cost_bps=0.0, borrow_bps_annual=0.0)
        s = st.summarize(bk, "r_strat_gross")
        print(f"  planted edge={ed:+.2f}  EOM gross Sharpe={s['sharpe']:+.3f}  t={s['tstat']:+.2f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
