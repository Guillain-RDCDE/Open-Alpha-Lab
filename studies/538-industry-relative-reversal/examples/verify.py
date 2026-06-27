"""Real-tape verification -- Study 538 (Industry-Relative-Reversal).

Reads (cache-first) the monthly close panel for the fixed sector basket, builds the RAW
one-month reversal (Study 329) and the INDUSTRY-RELATIVE reversal (Hameed-Mian 2015), and
reports both loser-minus-winner spreads head-to-head: HAC t, the skip=1 (bid-ask-bounce)
variant, the placebo industry-label-shuffle null, the beta decomposition, the sub-period
split, and the turnover/cost sweep. Also runs the synthetic positive control. Network is
touched only with --fetch.

    python studies/538-industry-relative-reversal/examples/verify.py          # cache-only
    python studies/538-industry-relative-reversal/examples/verify.py --fetch  # refresh panel
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from industry_relative_reversal import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    prices = data.fetch_monthly(fetch=fetch) if fetch else data.load_real()
    sectors = data.sector_series(prices.columns)
    print(f"Panel: {prices.shape[0]} months x {prices.shape[1]} tickers")
    print(f"Range: {prices.index[0].date()} -> {prices.index[-1].date()}")
    print(f"Sectors: {sectors.value_counts().to_dict()}")
    print(f"Fingerprint (last row): {data.fingerprint(prices)}\n")

    # --- RAW vs INDUSTRY-RELATIVE (skip=0) -----------------------------------
    raw0 = st.quintile_returns(st.raw_signal(prices, skip=0), prices, q=0.20)
    irr0 = st.quintile_returns(st.industry_relative_signal(prices, sectors, skip=0), prices, q=0.20)

    s_raw = st.summarize(raw0["spread"])
    s_irr = st.summarize(irr0["spread"])

    print("=== skip=0: RAW vs INDUSTRY-RELATIVE one-month reversal ===")
    print(f"Portfolio months: raw {len(raw0)}  irr {len(irr0)}  avg names/quintile: {irr0['n_loser'].mean():.0f}")
    print(f"RAW spread: {s_raw['mean']*1e4:+.1f}bps/mo = {s_raw['mean']*12*100:+.2f}%/yr  HAC t={s_raw['tstat']:+.2f}  hit={s_raw['hit_rate']:.3f}")
    print(f"IRR spread: {s_irr['mean']*1e4:+.1f}bps/mo = {s_irr['mean']*12*100:+.2f}%/yr  HAC t={s_irr['tstat']:+.2f}  hit={s_irr['hit_rate']:.3f}")

    beta_raw, alpha_raw = st.beta_alpha(raw0["spread"], raw0["market"])
    beta_irr, alpha_irr = st.beta_alpha(irr0["spread"], irr0["market"])
    print(f"RAW spread beta={beta_raw:.3f}  alpha={alpha_raw*100:+.2f}%/yr")
    print(f"IRR spread beta={beta_irr:.3f}  alpha={alpha_irr*100:+.2f}%/yr")

    # --- skip=1: bid-ask-bounce-safe variant ---------------------------------
    raw1 = st.quintile_returns(st.raw_signal(prices, skip=1), prices, q=0.20)
    irr1 = st.quintile_returns(st.industry_relative_signal(prices, sectors, skip=1), prices, q=0.20)
    s_raw1 = st.summarize(raw1["spread"])
    s_irr1 = st.summarize(irr1["spread"])
    print(f"\n=== skip=1 (one-month gap; bid-ask-bounce-safe) ===")
    print(f"RAW spread: {s_raw1['mean']*1e4:+.1f}bps/mo  HAC t={s_raw1['tstat']:+.2f}")
    print(f"IRR spread: {s_irr1['mean']*1e4:+.1f}bps/mo  HAC t={s_irr1['tstat']:+.2f}")

    # --- placebo: industry-label shuffle null --------------------------------
    placebo = st.placebo_irr_tstats(prices, sectors, n_shuffles=200, q=0.20, seed=538)
    real_t = s_irr["tstat"]
    p_val = float((placebo >= real_t).mean())
    print(f"\n=== Placebo: industry-label shuffle null (200 shuffles) ===")
    print(f"Shuffled IRR t: mean={placebo.mean():+.2f}  sd={placebo.std():.2f}  "
          f"q05={np.quantile(placebo,0.05):+.2f}  q95={np.quantile(placebo,0.95):+.2f}")
    print(f"Real IRR t={real_t:+.2f}  placebo p(shuffled >= real)={p_val:.3f}")

    # --- sub-periods (IRR spread) --------------------------------------------
    print("\n=== Sub-period decay (IRR spread, skip=0) ===")
    for lab, a, b in [("1990-2002", "1990", "2002"), ("2003-2014", "2003", "2014"),
                      ("2015-2026", "2015", "2026")]:
        s = st.summarize(irr0["spread"][a:b])
        print(f"  {lab}: {s['mean']*1e4:+.1f}bps/mo  t={s['tstat']:+.2f}  n={s['n']}")

    # --- turnover & cost sweep (IRR) -----------------------------------------
    print(f"\n=== Turnover & costs (IRR spread, skip=0) ===")
    print(f"Avg IRR loser-leg one-way turnover: {irr0['turn'].mean():.1%}")
    print(f"Break-even one-way cost (excl borrow): {st.break_even_cost(irr0):.1f} bps")
    for c in [0, 5, 10, 20]:
        s = st.summarize(st.net_spread(irr0, c))
        print(f"  net @ {c:2d} bps one-way (+50bps/yr borrow): {s['mean']*1e4:+.1f}bps/mo  t={s['tstat']:+.2f}")

    # --- synthetic positive control ------------------------------------------
    print("\n=== Synthetic positive control (within-industry reversal knob) ===")
    print("  RAW sees a diluted signal; IRR isolates the planted within-industry reversal.")
    for rev in [0.0, 0.04, 0.08]:
        p, sec, _ = data.synthetic_panel(within_reversal=rev, n_months=250, seed=538)
        rs = st.quintile_returns(st.raw_signal(p, skip=0), p, q=0.20)
        is_ = st.quintile_returns(st.industry_relative_signal(p, sec, skip=0), p, q=0.20)
        sr = st.summarize(rs["spread"])
        si = st.summarize(is_["spread"])
        print(f"  within_reversal={rev:+.2f}: RAW t={sr['tstat']:+.2f} ({sr['mean']*12*100:+.1f}%/yr)  "
              f"IRR t={si['tstat']:+.2f} ({si['mean']*12*100:+.1f}%/yr)")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
