"""Real-tape verification — Study 935 (Value Averaging). Regenerates docs/results.md.

Reads cached SPY (equity sleeve) and BIL (cash leg) daily total-return closes, runs
Edleson value averaging against dollar-cost averaging over every rolling 36-month
window on identical committed capital, and prints the terminal-wealth gap, the win
rate with a Wilson interval, the HAC *t* and block-bootstrap CI, both IRR measures,
the cap-binding statistics, the era cut, the four sweeps (value-path growth, cash
buffer, cost, horizon), the exposure-matched control, the calibrated random-walk
placebo, the IEF/QQQ cross-checks and the synthetic control. Network only on
``--fetch``.

    python studies/935-value-averaging/examples/verify.py            # cache-only
    python studies/935-value-averaging/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from value_avg import data, strategy as st  # noqa: E402

SLEEVE = "SPY"
CASH = "BIL"
HORIZON = 36          # months in one accumulation programme
BUFFER = 6.0          # cash buffer, in monthly contributions (ASSUMPTION — swept)
GROWTH = 0.0          # value-path growth rate (ASSUMPTION — swept); 0 = Edleson linear
COST_BPS = 1.0        # one-way, x NAV (ETF-realistic)
BASE = dict(buffer_mult=BUFFER, growth_ann=GROWTH, cost_bps=COST_BPS)


def legs(px, ticker):
    a = px[ticker].dropna()
    c = px[CASH].dropna()
    k = a.index.intersection(c.index)
    return a.loc[k], c.loc[k], k


def line(tag, s):
    print(f"  {tag:16s} n={s['n_windows']:4d}  gap={s['gap_mean_cents']:+7.3f}c  "
          f"t={s['t_hac']:+6.2f}  VA wins {s['va_win_rate']:6.1%}  "
          f"cap binds {s['bind_window_rate']:5.1%}  "
          f"equity {s['va_invested_frac']:.3f}/{s['dca_invested_frac']:.3f}")


def main(fetch: bool) -> None:
    if fetch:
        data.fetch()
    px = data.load_prices()
    a, c, k = legs(px, SLEEVE)

    print(f"{SLEEVE} vs {CASH}: {k[0].date()} -> {k[-1].date()}  n={len(k)}  "
          f"fp={data.fingerprint(px[[SLEEVE, CASH]].loc[k])}")
    print(f"as-of {data.AS_OF}  |  horizon {HORIZON}m  buffer {BUFFER:.0f}xC  "
          f"value-path growth {GROWTH:.0%}  cost {COST_BPS:.0f} bp one-way")

    df = st.rolling_race(a, c, HORIZON, **BASE)
    s = st.summarise(df, HORIZON)

    print("\n=== headline: VA minus DCA terminal wealth, cents per dollar contributed ===")
    print(f"  windows {len(df)}  ({df.index[0].date()} -> {df['end'].iloc[-1].date()} last valuation)")
    print(f"  mean gap {s['gap_mean_cents']:+.3f}c   median {s['gap_median_cents']:+.3f}c   "
          f"sd {s['gap_sd_cents']:.3f}c")
    print(f"  HAC t (lag {HORIZON}) {s['t_hac']:+.2f}   non-overlapping t {s['t_nonoverlap']:+.2f} "
          f"(n={s['n_nonoverlap']})   bootstrap 95% CI [{s['boot_lo']:+.3f}, {s['boot_hi']:+.3f}]")
    print(f"  VA wins {s['va_win_rate']:.1%} of windows  Wilson [{s['win_lo']:.1%}, {s['win_hi']:.1%}]")
    print(f"  excess-of-cash terminal: VA {s['excess_va_cents']:+.2f}c vs DCA "
          f"{s['excess_dca_cents']:+.2f}c per dollar contributed")
    print(f"  whole-programme IRR: VA {s['irr_prog_va']:+.4f} vs DCA {s['irr_prog_dca']:+.4f}")
    print(f"  equity-only IRR (Edleson's metric): VA {s['irr_eq_va']:+.4f} vs DCA "
          f"{s['irr_eq_dca']:+.4f}  -> VA 'wins' by {(s['irr_eq_va']-s['irr_eq_dca'])*100:+.2f} pp/yr")
    print(f"  mean equity weight: VA {s['va_invested_frac']:.3f} vs DCA {s['dca_invested_frac']:.3f}")
    print(f"  dispersion ratio (VA/DCA excess-of-cash sd): {s['gap_dispersion_ratio']:.3f}")

    print("\n=== the cash the rule demands ===")
    binds = df[df["va_bind_months"] > 0]
    print(f"  cap binds in {len(binds)}/{len(df)} windows ({s['bind_window_rate']:.1%}), "
          f"{s['bind_month_rate']:.2%} of all rebalance months")
    print(f"  worst SINGLE-MONTH unfunded call {s['worst_month_shortfall']:.2f} x monthly contribution; "
          f"worst PROGRAMME-TOTAL shortfall {s['worst_prog_shortfall']:.2f} x "
          f"(binding months across all windows: {s['bind_months_total']})")
    print(f"  binding windows start: {[d.strftime('%Y-%m') for d in binds.index]}")
    print(f"  trades: VA {s['va_trades']:.1f} vs DCA {s['dca_trades']:.1f}; notional "
          f"VA {s['va_notional']:.2f} vs DCA {s['dca_notional']:.2f}; cost per dollar "
          f"VA {s['va_cost_cents']:.4f}c vs DCA {s['dca_cost_cents']:.4f}c")
    print(f"  worst window {df['gap_cents'].min():+.2f}c (start {df['gap_cents'].idxmin().date()});  "
          f"best {df['gap_cents'].max():+.2f}c (start {df['gap_cents'].idxmax().date()})")

    print("\n=== era cut (window start before / after 2016-01-01) ===")
    for tag, e in st.era_cut(df, "2016-01-01", HORIZON).items():
        if e:
            line(tag, e)

    print("\n=== sweep: value-path growth rate (an ASSUMPTION, not tape) ===")
    for g in (0.0, 0.04, 0.08, 0.12):
        line(f"g={g:.0%}", st.summarise(st.rolling_race(a, c, HORIZON, buffer_mult=BUFFER,
                                                        growth_ann=g, cost_bps=COST_BPS), HORIZON))

    print("\n=== sweep: cash buffer (an ASSUMPTION, not tape) ===")
    print("  the equity-only IRR column is the point: it barely moves while the buffer -- and")
    print("  therefore the actual programme -- changes completely. That is why it is the wrong metric.")
    for b in (0.0, 3.0, 6.0, 12.0, 24.0):
        sb = st.summarise(st.rolling_race(a, c, HORIZON, buffer_mult=b,
                                          growth_ann=GROWTH, cost_bps=COST_BPS), HORIZON)
        line(f"buffer={b:.0f}xC", sb)
        print(f"    equity-only IRR VA {sb['irr_eq_va']:.4f} vs DCA {sb['irr_eq_dca']:.4f}  |  "
              f"whole-programme IRR VA {sb['irr_prog_va']:.4f} vs DCA {sb['irr_prog_dca']:.4f}")

    print("\n=== sweep: one-way cost ===")
    for cb in (0.0, 1.0, 5.0, 25.0):
        line(f"cost={cb:.0f}bp", st.summarise(st.rolling_race(a, c, HORIZON, buffer_mult=BUFFER,
                                                              growth_ann=GROWTH, cost_bps=cb), HORIZON))

    print("\n=== sweep: horizon ===")
    for h in (24, 36, 60, 120):
        s_h = st.summarise(st.rolling_race(a, c, h, **BASE), h)
        line(f"H={h}m", s_h)
        if h >= 60:
            n_ind = s_h["n_nonoverlap"]
            print(f"    (CAVEAT: only {n_ind} independent {h}-month "
                  f"programme{'' if n_ind == 1 else 's'} exist{'s' if n_ind == 1 else ''} "
                  f"in this sample, and the HAC lag truncation ({h}) is "
                  f"{h / s_h['n_windows']:.0%} of the window count, so this t reads as "
                  f"direction, not as a size-correct test.)")

    print("\n=== exposure-matched control (DCA dialled to VA's own equity weight) ===")
    print("  NOTE: lambda is fitted IN-SAMPLE on this very tape, so the +gap it produces is an")
    print("  in-sample residual, not an out-of-sample result. The placebo below is fitted the")
    print("  same way on each synthetic path, so the comparison is like-for-like.")
    em = st.exposure_matched_race(a, c, HORIZON, **BASE)
    print(f"  lambda={em['dca_scale']:.4f}  residual exposure gap {em['exposure_gap']:+.4f}")
    print(f"  gap {em['gap_mean_cents']:+.3f}c  HAC t {em['t_hac']:+.2f}  VA wins "
          f"{em['va_win_rate']:.1%}  bootstrap CI [{em['boot_lo']:+.3f}, {em['boot_hi']:+.3f}]")

    print("\n=== placebo: the same exposure-matched race on a CALIBRATED random walk (12 paths) ===")
    lr = np.log(a).diff().dropna()
    drift = float(lr.mean() * 252)
    vol = float(lr.std(ddof=1) * np.sqrt(252))
    yrs = (k[-1] - k[0]).days / 365.25
    cash_rate = float((c.iloc[-1] / c.iloc[0]) ** (1.0 / yrs) - 1.0)
    print(f"  calibration: log-drift {drift:.4f}, vol {vol:.4f}, cash {cash_rate:.4f}, {yrs:.1f}y")
    gaps = []
    for seed in range(2000, 2012):
        p, _ = data.synthetic_daily(n_years=19, signal_strength=0.0, seed=seed,
                                    drift_ann=drift, vol_ann=vol, cash_rate_ann=cash_rate)
        e = st.exposure_matched_race(p["asset"], p["cash"], HORIZON, tol=0.004,
                                     max_iter=7, **BASE)
        gaps.append(e["gap_mean_cents"])
        print(f"    seed {seed}: gap {e['gap_mean_cents']:+.3f}c  t {e['t_hac']:+.2f}")
    print(f"  placebo gap: mean {np.mean(gaps):+.3f}c  sd {np.std(gaps, ddof=1):.3f}  "
          f"range [{min(gaps):+.3f}, {max(gaps):+.3f}]")
    z = (em["gap_mean_cents"] - np.mean(gaps)) / np.std(gaps, ddof=1)
    n_above = int(sum(g >= em["gap_mean_cents"] for g in gaps))
    print(f"  the real tape's exposure-matched gap sits at z={z:+.2f} of the placebo spread")
    print(f"  distribution-free read: {n_above}/{len(gaps)} no-predictability paths beat it "
          f"(one-sided p = {(n_above + 1) / (len(gaps) + 1):.3f})")

    print("\n=== cross-checks: other sleeves ===")
    for tk in ("IEF", "QQQ"):
        aa, cc, _ = legs(px, tk)
        line(tk, st.summarise(st.rolling_race(aa, cc, HORIZON, **BASE), HORIZON))

    print("\n=== synthetic control (machinery proof, never supports the stamp) ===")
    pl, _ = data.synthetic_daily(n_years=30, signal_strength=1.0, seed=935)
    line("planted wobble", st.synthetic_detect(pl, HORIZON, buffer_mult=BUFFER,
                                               growth_ann=GROWTH, cost_bps=COST_BPS))
    pl12, _ = data.synthetic_daily(n_years=12, signal_strength=1.0, seed=935)
    em_pl = st.exposure_matched_race(pl12["asset"], pl12["cash"], HORIZON,
                                     tol=0.01, max_iter=6, **BASE)
    print(f"  planted, exposure-matched : gap {em_pl['gap_mean_cents']:+.3f}c  t {em_pl['t_hac']:+.2f}")
    det, _ = data.synthetic_daily(n_years=12, signal_strength=0.0, seed=935, vol_ann=0.0)
    em_det = st.exposure_matched_race(det["asset"], det["cash"], HORIZON,
                                      tol=0.005, max_iter=8, **BASE)
    print(f"  zero-vol null, matched    : gap {em_det['gap_mean_cents']:+.3f}c (must be ~0)")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
