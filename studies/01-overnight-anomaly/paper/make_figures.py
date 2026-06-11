"""Generate the paper's figures (PDF) from the package + cached data.

Run from the repo root:  python paper/make_figures.py
Outputs to paper/figures/*.pdf. Reproducible: same data -> same figures.
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from quantlab import analytics, bayes, data, decompose, simulate, universe  # noqa: E402

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"figure.figsize": (7, 4.3), "font.size": 10, "savefig.bbox": "tight"})


def _spy():
    ohlc = data.fetch("SPY", mode="split_only")
    return ohlc, decompose.decompose(ohlc)


def fig_alpha_decay(dec):
    rs = analytics.rolling_sharpe(dec["r_overnight"])
    fig, ax = plt.subplots()
    ax.plot(rs.index, rs.values, lw=1.5, color="#2c7fb8")
    ax.axhline(0, color="k", lw=.6)
    ax.axhline(0.5, color="grey", ls="--", lw=.8)
    ax.set_ylabel("Trailing 5y annualised Sharpe")
    ax.set_xlabel("Date")
    ax.set_title("SPY overnight: post-publication alpha decay")
    ax.grid(alpha=.3)
    fig.savefig(os.path.join(FIG, "alpha_decay.pdf"))
    plt.close(fig)


def fig_manipulator(dec, ohlc):
    drift = dec["r_overnight"].mean() * 1e4
    vol = dec["r_close_close"].std(ddof=1) * 1e4
    adv_usd = float(ohlc["Volume"].tail(252).mean()) * float(ohlc["Close"].tail(252).mean())
    sizes = [10 ** e for e in range(6, 12)]
    sw = simulate.pnl_vs_capital(drift, adv_usd, vol, sizes_usd=sizes)
    be = simulate.breakeven_capital(drift, adv_usd, vol)
    fig, ax = plt.subplots()
    ax.plot(sizes, sw["net_bps_on_capital"].values, marker="o", color="#b2182b")
    ax.axhline(0, color="k", lw=.8, ls="--")
    ax.axvline(be, color="grey", lw=.8, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("Capital deployed (USD, log scale)")
    ax.set_ylabel("Net edge (bps on capital)")
    ax.set_title("The steelman manipulator's P&L vs scale (SPY)")
    ax.annotate(f"break-even ≈ ${be/1e6:.0f}M", xy=(be, 0), xytext=(be, sw['net_bps_on_capital'].min()*0.4),
                fontsize=8, ha="center")
    ax.grid(alpha=.3)
    fig.savefig(os.path.join(FIG, "manipulator_pnl.pdf"))
    plt.close(fig)


def fig_cross_section():
    path = "_cache/cross_section_per_symbol.parquet"
    if os.path.exists(path):
        per = pd.read_parquet(path)
    else:  # fallback: recompute on whatever panel is cached
        # Survivorship bias opted into explicitly: current membership projected
        # backwards — the paper states the caveat next to the figure.
        syms = universe.sp500_symbols(allow_survivorship_bias=True)
        per = universe.cross_section_decompose(
            universe.download_panel(syms, allow_survivorship_bias=True),
            allow_survivorship_bias=True,
        )["per_symbol"]
    fig, ax = plt.subplots()
    bins = np.linspace(-1.5, 1.5, 41)
    ax.hist(per["overnight_sharpe"].astype(float), bins=bins, alpha=.6, label="overnight", color="#2c7fb8")
    ax.hist(per["intraday_sharpe"].astype(float), bins=bins, alpha=.6, label="intraday", color="#fdae61")
    ax.axvline(per["overnight_sharpe"].median(), color="#2c7fb8", lw=1.5, ls="--")
    ax.axvline(per["intraday_sharpe"].median(), color="#d95f02", lw=1.5, ls="--")
    ax.set_xlabel("Annualised Sharpe")
    ax.set_ylabel(f"# stocks (N={len(per)})")
    ax.set_title("Firm-level night vs day Sharpe across S&P 500 members")
    ax.legend()
    ax.grid(alpha=.3)
    fig.savefig(os.path.join(FIG, "cross_section.pdf"))
    plt.close(fig)


def fig_posterior():
    grid = bayes.posterior_sensitivity(priors=(0.1, 0.3, 0.5, 0.7, 0.9), lr_scales=(0.5, 1.0, 2.0))
    fig, ax = plt.subplots()
    for col in grid.columns:
        ax.plot(grid.index, grid[col].values, marker="o", label=col)
    ax.plot(grid.index, grid.index, color="grey", lw=.8, ls=":", label="posterior = prior")
    ax.set_xlabel("Prior P(manipulation)")
    ax.set_ylabel("Posterior P(manipulation)")
    ax.set_title("Manipulation posterior vs prior, by evidence strength")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)
    fig.savefig(os.path.join(FIG, "posterior.pdf"))
    plt.close(fig)


def main():
    ohlc, dec = _spy()
    fig_alpha_decay(dec)
    fig_manipulator(dec, ohlc)
    fig_cross_section()
    fig_posterior()
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
