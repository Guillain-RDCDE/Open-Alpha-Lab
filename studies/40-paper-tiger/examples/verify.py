"""Reproduce the real-data headline run (docs/results.md) — GEM on SPY/EFA/AGG, 2003–today.

    python examples/verify.py            # cache-only (offline); prints if cache present
    python examples/verify.py --fetch    # download SPY/EFA/AGG/^IRX monthly from Yahoo, then run

Rebuilds Antonacci's Global Equities Momentum book, charges 20 bp per switch, and prints it next to
the only benchmarks that matter — buy-and-hold SPY and a plain 60/40 — in **both Sharpe conventions**
(raw and excess-of-cash), plus the test that isolates the timing skill (the spanning alpha of GEM on
its own ingredients), the decay, the crisis cushion and the portfolio-sleeve test docs/results.md cites.
"""

from __future__ import annotations

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import pandas as pd  # noqa: E402

from paper_tiger import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    prices, tbill = data.fetch_etfs(fetch=fetch)
    if prices.empty:
        print("No cached real data. Re-run with --fetch (needs network) to download SPY/EFA/AGG/^IRX.")
        return

    gem = st.gem_returns(prices, tbill, cost_bps=20.0)
    gem0 = st.gem_returns(prices, tbill, cost_bps=0.0)
    bh = st.buy_and_hold(prices, "US").loc[gem.index]
    b40 = st.blend(prices).loc[gem.index]
    rf = tbill.reindex(gem.index)

    def line(nm, x):
        s, sx = st.summary(x), st.summary(x, rf=rf)
        print(f"  {nm:24} Sharpe {s['sharpe']:5.2f} (excess {sx['sharpe']:5.2f})  CAGR {s['cagr']*100:5.1f}%  "
              f"vol {s['vol_ann']*100:4.1f}%  maxDD {s['max_drawdown']*100:5.0f}%  skew {s['skew']:+.2f}")

    start = prices.dropna().index.min().date()
    print(f"\nGEM on SPY/EFA/AGG, {start} - {gem.index[-1].date()} ({len(gem)} months, {st.n_switches(prices, tbill)} switches)\n")
    line("GEM gross", gem0)
    line("GEM net (20bp/switch)", gem)
    line("Buy & Hold SPY", bh)
    line("60/40 SPY/AGG", b40)

    # the test that isolates the timing skill: spanning alpha on the book's own ingredients
    rets = st.to_returns(prices)
    fac = pd.DataFrame({"SPY": rets["US"], "AGG": rets["BOND"]}).sub(tbill, axis=0).loc[gem.index]
    sp = st.spanning_alpha((gem - rf).dropna(), fac)
    print(f"\nSpanning alpha (GEM net excess on SPY/AGG excess): {sp['alpha_bps_m']:.0f} bp/mo "
          f"({sp['alpha_ann_pct']:.1f}%/yr)  HAC t={sp['tstat']:.2f}  "
          f"betas SPY {sp['betas']['SPY']:.2f} / AGG {sp['betas']['AGG']:.2f}  R²={sp['r2']:.2f}")
    sp40 = st.spanning_alpha((gem - rf).dropna(), (b40 - rf).dropna().rename("60/40").to_frame())
    print(f"Spanning alpha (GEM net excess on 60/40 excess):   {sp40['alpha_bps_m']:.0f} bp/mo  "
          f"HAC t={sp40['tstat']:.2f}  beta {sp40['betas']['60/40']:.2f}")

    # book-level inference in both conventions (reuse the desk engine)
    try:
        from quantlab.analytics import mean_tstat_hac, sharpe_with_se
        ex = (gem - rf).dropna()
        lo_x, hac_x = sharpe_with_se(gem, periods_per_year=12, rf=rf), mean_tstat_hac(ex)
        lo_r, hac_r = sharpe_with_se(gem, periods_per_year=12), mean_tstat_hac(gem)
        spy_x = sharpe_with_se(bh, periods_per_year=12, rf=rf)
        print(f"\nBook-level inference (what an always-invested book passes anyway):")
        print(f"  excess-of-cash: Lo Sharpe t={lo_x['tstat']:.2f} · HAC mean {hac_x['mean_bps']:.0f}bps t={hac_x['tstat']:.2f}"
              f"   (SPY itself: Lo t={spy_x['tstat']:.2f})")
        print(f"  raw:            Lo Sharpe t={lo_r['tstat']:.2f} · HAC mean {hac_r['mean_bps']:.0f}bps t={hac_r['tstat']:.2f}")
    except Exception:
        pass

    # decay
    print("Decay (sub-period Sharpe, raw / excess):",
          f"2004-2013 {st.summary(gem.loc['2004':'2013'])['sharpe']:.2f} / {st.summary(gem.loc['2004':'2013'], rf=rf)['sharpe']:.2f} · "
          f"2014-2026 {st.summary(gem.loc['2014':'2026'])['sharpe']:.2f} / {st.summary(gem.loc['2014':'2026'], rf=rf)['sharpe']:.2f}")

    # crisis cushion: worst-decile SPY months
    crisis = bh <= bh.quantile(0.10)
    print(f"Crisis (worst-decile SPY months, n={int(crisis.sum())}): "
          f"GEM {gem[crisis].mean()*100:.2f}%/mo vs SPY {bh[crisis].mean()*100:.2f}%/mo · "
          f"corr {gem[crisis].corr(bh[crisis]):+.2f}")

    # portfolio sleeve
    mix = 0.7 * b40 + 0.3 * gem
    s40, sm = st.summary(b40, rf=rf), st.summary(mix, rf=rf)
    print(f"Sleeve: 60/40 excess Sharpe {s40['sharpe']:.2f} maxDD {s40['max_drawdown']*100:.0f}% → "
          f"+30% GEM excess Sharpe {sm['sharpe']:.2f} maxDD {sm['max_drawdown']*100:.0f}%")

    try:
        from quantlab import repro
        fp_in = prices.join(tbill)
        print(f"\nas-of {gem.index[-1].date()} · inputs fingerprint {repro.fingerprint(fp_in)}")
    except Exception:
        pass


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
