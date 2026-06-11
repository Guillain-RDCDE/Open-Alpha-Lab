"""Reproduce the real-data headline run (docs/results.md) — GEM on SPY/EFA/AGG, 2003–today.

    python examples/verify.py            # cache-only (offline); prints if cache present
    python examples/verify.py --fetch    # download SPY/EFA/AGG/^IRX monthly from Yahoo, then run

Rebuilds Antonacci's Global Equities Momentum book, charges 20 bp per switch, and prints it next to
the only benchmarks that matter — buy-and-hold SPY and a plain 60/40 — plus the inference, the
post-publication decay, the crisis cushion and the portfolio-sleeve test that docs/results.md cites.
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

    def line(nm, x):
        s = st.summary(x)
        print(f"  {nm:24} Sharpe {s['sharpe']:5.2f}  CAGR {s['cagr']*100:5.1f}%  vol {s['vol_ann']*100:4.1f}%  "
              f"maxDD {s['max_drawdown']*100:5.0f}%  skew {s['skew']:+.2f}")

    start = prices.dropna().index.min().date()
    print(f"\nGEM on SPY/EFA/AGG, {start} - {gem.index[-1].date()} ({len(gem)} months, {st.n_switches(prices, tbill)} switches)\n")
    line("GEM gross", gem0)
    line("GEM net (20bp/switch)", gem)
    line("Buy & Hold SPY", bh)
    line("60/40 SPY/AGG", b40)

    # inference (reuse the desk engine)
    try:
        from quantlab.analytics import mean_tstat_hac, sharpe_with_se
        lo = sharpe_with_se(gem, periods_per_year=12)
        hac = mean_tstat_hac(gem)
        print(f"\nInference: Lo Sharpe t={lo['tstat']:.2f} · HAC mean {hac['mean_bps']:.0f}bps t={hac['tstat']:.2f}")
    except Exception:
        pass

    # decay
    print("Decay (sub-period Sharpe):",
          f"2004-2013 {st.summary(gem.loc['2004':'2013'])['sharpe']:.2f} · "
          f"2014-2026 {st.summary(gem.loc['2014':'2026'])['sharpe']:.2f}")

    # crisis cushion: worst-decile SPY months
    crisis = bh <= bh.quantile(0.10)
    print(f"Crisis (worst-decile SPY months, n={int(crisis.sum())}): "
          f"GEM {gem[crisis].mean()*100:.2f}%/mo vs SPY {bh[crisis].mean()*100:.2f}%/mo · "
          f"corr {gem[crisis].corr(bh[crisis]):+.2f}")

    # portfolio sleeve
    mix = 0.7 * b40 + 0.3 * gem
    s40, sm = st.summary(b40), st.summary(mix)
    print(f"Sleeve: 60/40 Sharpe {s40['sharpe']:.2f} maxDD {s40['max_drawdown']*100:.0f}% → "
          f"+30% GEM Sharpe {sm['sharpe']:.2f} maxDD {sm['max_drawdown']*100:.0f}%")

    try:
        from quantlab import repro
        fp_in = prices.join(tbill)
        print(f"\nas-of {gem.index[-1].date()} · inputs fingerprint {repro.fingerprint(fp_in)}")
    except Exception:
        pass


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
