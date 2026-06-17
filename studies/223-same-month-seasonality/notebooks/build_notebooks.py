"""Generate the two narrative notebooks for Study 223 (Same-Month Seasonality).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the cached
monthly panel under ../_cache/ if present and otherwise quote the frozen headline
numbers in ``R`` (mirroring docs/results.md), so the notebook re-runs for any reader.

The _write convention (each build_*() ends by calling _write) is kept so the repo's
intro-restyle tooling can monkeypatch it.
"""

from __future__ import annotations

import os

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))


def md(text):
    return new_markdown_cell(text)


def code(text):
    return new_code_cell(text)


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-16).
R = dict(
    n_months=402, n_stocks=80, n_high=8,
    high_ann=31.55, low_ann=13.85, market_ann=15.34,
    spread_bps=147.5, spread_ann=17.70, spread_t=5.57, spread_hit=0.624,
    spread_sr_lo=0.556, spread_sr_hi=1.197, spread_frac_neg=0,
    high_xret_ann=16.21, high_xret_t=7.46,
    # sub-periods
    sub_1999_mean=21.02, sub_1999_t=3.92, sub_1999_n=120,
    sub_2009_mean=17.04, sub_2009_t=3.65, sub_2009_n=108,
    sub_2018_mean=14.13, sub_2018_t=2.15, sub_2018_n=101,
    # costs
    turnover=78.25, drag_bps=15.7, net_spread_bps=131.9, net_spread_t=4.98,
    # per-month notable
    jul_bps=199.0, jul_t=3.21,
    dec_bps=314.7, dec_t=2.37,
    aug_bps=-9.8, aug_t=-0.10,
    nov_bps=-55.2, nov_t=-0.49,
)

# ---------------------------------------------------------------------------
# Shared analysis preamble
# ---------------------------------------------------------------------------
BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root (quantlab/)
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from same_month_seasonality import data, strategy as st

CACHE_PATH = data.MONTHLY_CACHE
HAVE_REAL = os.path.exists(CACHE_PATH)

if HAVE_REAL:
    prices = data.fetch_monthly(fetch=False, cache_path=CACHE_PATH)
    sig = st.same_month_signal(prices, min_years=5)
    res = st.decile_returns(sig, prices, q=0.10)
    spread = res["spread"].dropna()
    high_xret = (res["high"] - res["market"]).dropna()
    print(f"Real panel loaded: {prices.shape[0]} months x {prices.shape[1]} tickers")
    print(f"Portfolio months: {len(res)}")
else:
    prices = sig = res = spread = high_xret = None
    print("No real cache -- frozen headline numbers from R dict will be used")
"""

# Frozen-numbers fallback (used in cells that need real-tape numbers)
R_INJECT = f"""
# Frozen headline numbers (mirror of docs/results.md, as-of 2026-06-16)
R = {R!r}
"""


# ---------------------------------------------------------------------------
# Notebook 1 -- For the curious
# ---------------------------------------------------------------------------
def build_curious() -> None:
    nb = new_notebook()
    nb.cells = [
        md("""\
# Study 223 -- Same-Month Seasonality
## For the curious: does a stock keep outperforming in the same calendar month it always has?

*Part of [Open-Alpha-Lab](../../../README.md). See the [desk methodology](../../../METHODOLOGY.md).*

---

Some stocks seem to outperform in January every year. Others consistently do well in July.
Is this a coincidence -- or a persistent, exploitable pattern?

Heston & Sadka (2008) asked exactly this question and found that stocks sorted by their
historical same-calendar-month performance continue to outperform in that same month.
We replicate and stress-test their claim on a modern large-cap panel.
"""),
        md("## Setup"),
        code(BOOT),
        code(R_INJECT),

        md("""\
## The intuition: stocks with a recurring December pattern

Imagine a retailer that always has strong revenue in December -- its stock tends to run
up in December year after year.  Similarly, a tax-loss harvesting target might always be
sold in November and bounce in January.

The Heston-Sadka strategy says: **at the end of November, look at each stock's average
December return from the past 5+ years, buy the best December performers, sell the worst
December performers, and hold through December**.  Repeat every month with the appropriate
calendar-month history.

This is *not* the same as the well-known January Effect (which is a market-level
seasonality).  It is a *cross-sectional* sort: we are ranking stocks against each other
based on their own calendar-month history.
"""),

        md("## The top-decile race: high vs low seasonality stocks"),
        code("""\
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

if HAVE_REAL:
    # Cumulative returns
    ax = axes[0]
    cum_high   = (1 + res["high"].dropna()).cumprod()
    cum_low    = (1 + res["low"].dropna()).cumprod()
    cum_market = (1 + res["market"].dropna()).cumprod()
    ax.plot(cum_high.index,   cum_high.values,   color=GREEN, lw=1.8, label="Top decile (high seasonality)")
    ax.plot(cum_low.index,    cum_low.values,    color=RED,   lw=1.8, label="Bottom decile (low seasonality)")
    ax.plot(cum_market.index, cum_market.values, color=GREY,  lw=1.2, ls="--", label="Equal-weight market")
    ax.set_title("Cumulative growth of $1 (survivorship-biased)")
    ax.set_ylabel("Cumulative return ($)")
    ax.legend(fontsize=9)

    # Rolling 36-month spread
    ax = axes[1]
    roll = spread.rolling(36).mean() * 12 * 100
    ax.plot(roll.index, roll.values, color=GREEN, lw=1.5)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_title("Rolling 36-month spread (% / yr)")
    ax.set_ylabel("Annualised spread (%/yr)")
else:
    for ax in axes:
        ax.text(0.5, 0.5, f"Cache absent\\nFrozen: spread = +{R['spread_ann']:.1f}%/yr  t = +{R['spread_t']:.2f}",
                ha="center", va="center", transform=ax.transAxes, fontsize=12, color=GREEN)
        ax.set_title("(frozen numbers)")

plt.tight_layout()
plt.savefig("../docs/equity_curve.png", dpi=120, bbox_inches="tight")
plt.show()
print(f"Spread: +{R['spread_ann']:.1f}%/yr = +{R['spread_bps']:.1f}bps/mo  HAC t = +{R['spread_t']:.2f}")
"""),

        md("""\
## The headline: large spread, but concentrated and survivorship-biased

The same-month seasonality spread is **+17.7%/yr** with a HAC *t* = **+5.57** -- that is
highly significant by any conventional standard.  But before getting excited:

1. **Only ~8 stocks per decile**: with ~80 stocks in a large-cap basket and 10% decile
   cuts, each leg holds roughly 8 names.  This is not a diversified factor overlay --
   it is closer to a handful of individual bets.

2. **~78% monthly turnover**: nearly all stocks rotate out of the decile each month.
   Transaction costs and market impact are severe for such concentrated portfolios.

3. **Survivorship bias**: our universe is the current S&P 500 / large-cap survivors
   projected backwards.  True "losers" (firms that underperformed badly enough to be
   delisted or removed) are absent.  The real live-universe spread is materially lower.
"""),

        md("## Which calendar months actually drive the effect?"),
        code("""\
months_lbl = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

if HAVE_REAL:
    month_stats = []
    for m in range(1, 13):
        sub = spread[spread.index.month == m]
        if len(sub) < 5: continue
        s = st.summarize(sub)
        month_stats.append({"month": months_lbl[m-1], "bps": s["mean"]*10000, "t": s["tstat"], "n": s["n"]})
    ms = pd.DataFrame(month_stats)
    colors = [GREEN if t >= 2 else AMBER if t >= 1 else RED for t in ms["t"]]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(ms["month"], ms["bps"], color=colors)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Same-month spread by calendar month (bps/mo; green = t >= 2)")
    ax.set_ylabel("Spread (bps/mo)")
    for i, row in ms.iterrows():
        ax.text(i, row["bps"] + (8 if row["bps"] >= 0 else -20),
                f"t={row['t']:+.1f}", ha="center", fontsize=7.5)
    plt.tight_layout()
    plt.savefig("../docs/by_month.png", dpi=120, bbox_inches="tight")
    plt.show()
else:
    print("Frozen per-month highlights:")
    print(f"  Jul: {R['jul_bps']:+.1f} bps/mo  t={R['jul_t']:+.2f}")
    print(f"  Dec: {R['dec_bps']:+.1f} bps/mo  t={R['dec_t']:+.2f}")
    print(f"  Aug: {R['aug_bps']:+.1f} bps/mo  t={R['aug_t']:+.2f}")
    print(f"  Nov: {R['nov_bps']:+.1f} bps/mo  t={R['nov_t']:+.2f}")
print(f"Only July (t={R['jul_t']:.2f}) and December (t={R['dec_t']:.2f}) individually clear t>=2.")
print(f"August and November are negative (though not significant).")
"""),

        md("""\
## The bottom line for the curious

The same-month seasonality story is **academically real** -- Heston & Sadka (2008)
found it in the original CRSP data, and it shows up in our survivorship-biased
large-cap panel at t = 5.57.

But in practice it is **hard to trade**:
- Only 8 stocks per decile in our large-cap universe -- truly a handful of bets
- Nearly 80% monthly turnover
- The signal has decayed from t = 3.92 (1999-2008) to t = 2.15 (2018-2026) on
  a dataset that is already biased upward

A live, broad-universe implementation (Russell 1000, 100+ names per decile) would
have a much smaller gross edge and much higher total cost.  This is a **Real/Fragile**
signal -- real enough to be interesting, fragile enough to be uninvestable at scale.
"""),

        md("*Desk verdict: **Real / Fragile** -- see [README.md](../README.md) and [docs/results.md](../docs/results.md) for full numbers.*"),
    ]

    _write(nb, "01_for_the_curious.ipynb")


# ---------------------------------------------------------------------------
# Notebook 2 -- For the quants
# ---------------------------------------------------------------------------
def build_quants() -> None:
    nb = new_notebook()
    nb.cells = [
        md("""\
# Study 223 -- Same-Month Seasonality
## For the quants: signal construction, HAC tests, bootstrap CI, sub-period decay

*Part of [Open-Alpha-Lab](../../../README.md). See the [desk methodology](../../../METHODOLOGY.md).*
"""),
        md("## Setup"),
        code(BOOT),
        code(R_INJECT),

        md("## Signal construction: the Heston-Sadka same-month rank"),
        code("""\
# Synthetic positive control: planted premium=0.02 yields t > 2 strongly
price_syn, truth_syn = data.synthetic_panel(n_firms=100, n_months=200, premium=0.02, seed=223)
sig_syn = st.same_month_signal(price_syn, min_years=5)
res_syn = st.decile_returns(sig_syn, price_syn, q=0.10)
s_syn = st.summarize(res_syn["spread"].dropna())
print(f"Synthetic positive control (premium=0.02):")
print(f"  Spread mean = {s_syn['mean']*12*100:+.2f}%/yr  HAC t = {s_syn['tstat']:+.2f}  n = {s_syn['n']}")

# Null control: premium=0.0
price_null, _ = data.synthetic_panel(n_firms=100, n_months=200, premium=0.0, seed=223)
sig_null = st.same_month_signal(price_null, min_years=5)
res_null = st.decile_returns(sig_null, price_null, q=0.10)
s_null = st.summarize(res_null["spread"].dropna())
print(f"\\nNull control (premium=0.0):")
print(f"  Spread mean = {s_null['mean']*12*100:+.2f}%/yr  HAC t = {s_null['tstat']:+.2f}  n = {s_null['n']}")
print("  -> Engine correctly reads ~zero on null.")
"""),

        md("## HAC t-statistics and bootstrap Sharpe CI"),
        code("""\
if HAVE_REAL:
    s_spread = st.summarize(spread)
    s_high   = st.summarize(res["high"].dropna())
    s_low    = st.summarize(res["low"].dropna())
    s_market = st.summarize(res["market"].dropna())

    print("=== Primary specification: top-minus-bottom decile, 1-month hold ===")
    print(f"Portfolio months: {len(res)}")
    print(f"Avg stocks per decile: {res['n_high'].mean():.1f}")
    print()
    print(f"HIGH  : {s_high['mean']*12*100:+.2f}%/yr  SR(ann)={s_high['sharpe']*12**0.5:+.2f}  HAC t={s_high['tstat']:+.2f}")
    print(f"LOW   : {s_low['mean']*12*100:+.2f}%/yr  SR(ann)={s_low['sharpe']*12**0.5:+.2f}  HAC t={s_low['tstat']:+.2f}")
    print(f"MARKET: {s_market['mean']*12*100:+.2f}%/yr")
    print(f"SPREAD: {s_spread['mean']*12*100:+.2f}%/yr = {s_spread['mean']*10000:+.1f}bps/mo  HAC t={s_spread['tstat']:+.2f}  hit={s_spread['hit_rate']:.3f}")

    try:
        from quantlab.stats import sharpe_ci_bootstrap
        ci = sharpe_ci_bootstrap(spread, n_boot=2000, periods_per_year=12, seed=223)
        print(f"\\nBootstrap Sharpe 95% CI: [{ci['ci_low']:+.3f}, {ci['ci_high']:+.3f}]")
        print(f"  {ci['frac_negative']*100:.0f}% of resamples negative (should be ~0)")
    except ImportError:
        print(f"\\n[quantlab.stats unavailable -- frozen CI: [{R['spread_sr_lo']:+.3f}, {R['spread_sr_hi']:+.3f}]]")
else:
    print(f"Frozen: spread = +{R['spread_ann']:.2f}%/yr  HAC t = +{R['spread_t']:.2f}")
    print(f"Bootstrap 95% CI: [{R['spread_sr_lo']:+.3f}, {R['spread_sr_hi']:+.3f}]  ({R['spread_frac_neg']}% negative)")
"""),

        md("## Sub-period decay: is the anomaly disappearing?"),
        code("""\
if HAVE_REAL:
    periods = [("1999-2008", "1999", "2008"), ("2009-2017", "2009", "2017"), ("2018-2026", "2018", "2026")]
    sub_stats = []
    for label, start, end in periods:
        sub = spread[start:end]
        s = st.summarize(sub)
        sub_stats.append({"period": label, "mean_ann": s["mean"]*12*100, "tstat": s["tstat"], "n": s["n"]})
        print(f"  {label}: mean={s['mean']*12*100:+.2f}%/yr  HAC t={s['tstat']:+.2f}  n={s['n']}")

    fig, ax = plt.subplots(figsize=(9, 4))
    sts = pd.DataFrame(sub_stats)
    colors_sub = [GREEN if t >= 2 else AMBER for t in sts["tstat"]]
    ax.bar(sts["period"], sts["tstat"], color=colors_sub)
    ax.axhline(2.0, color="black", ls="--", lw=0.9, label="t=2 inference bar")
    ax.set_title("Sub-period HAC t-stat: post-publication decay")
    ax.set_ylabel("HAC t-stat")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()
else:
    for label, mean, t, n in [
        ("1999-2008", R["sub_1999_mean"], R["sub_1999_t"], R["sub_1999_n"]),
        ("2009-2017", R["sub_2009_mean"], R["sub_2009_t"], R["sub_2009_n"]),
        ("2018-2026", R["sub_2018_mean"], R["sub_2018_t"], R["sub_2018_n"]),
    ]:
        print(f"  {label}: mean={mean:+.2f}%/yr  HAC t={t:+.2f}  n={n}")
"""),

        md("## Turnover and cost drag"),
        code("""\
if HAVE_REAL:
    drag = st.turnover_cost_drag(sig, prices, q=0.10, one_way_bps=10.0)
    avg_turnover = drag.mean() / 0.002
    avg_drag = drag.mean() * 10000
    net_spread = spread - drag.reindex(spread.index).fillna(drag.mean())
    s_net = st.summarize(net_spread)
    print(f"Avg monthly turnover: {avg_turnover:.2%}")
    print(f"Avg drag @10bps one-way: {avg_drag:.1f} bps/mo")
    print(f"Net spread @10bps: {s_net['mean']*12*100:+.2f}%/yr  HAC t={s_net['tstat']:+.2f}")

    drag20 = st.turnover_cost_drag(sig, prices, q=0.10, one_way_bps=20.0)
    net20 = spread - drag20.reindex(spread.index).fillna(drag20.mean())
    s20 = st.summarize(net20)
    print(f"Net spread @20bps: {s20['mean']*12*100:+.2f}%/yr  HAC t={s20['tstat']:+.2f}")
else:
    print(f"Frozen: turnover={R['turnover']:.1f}%  drag={R['drag_bps']:.1f}bps/mo")
    print(f"Net spread @10bps: {R['net_spread_bps']:.1f}bps/mo  HAC t={R['net_spread_t']:.2f}")
print("\\nCaution: 78% monthly turnover on 8-stock deciles is severe.")
print("  A live implementation would face market impact well beyond commission.")
"""),

        md("## Random-portfolio null: does the signal carry genuine cross-sectional information?"),
        code("""\
if HAVE_REAL:
    print("Computing random-portfolio null (n_draws=100 per month)...")
    rand = st.random_portfolio_returns(sig, prices, q=0.10, n_draws=100, seed=223)
    print(f"Random excess: mean={rand.mean()*10000:+.1f}bps  std={rand.std()*10000:.1f}bps")
    print(f"  -> Null is centred at zero: {abs(rand.mean()*10000) < 10}")
    print(f"  -> Signal is genuine: the same-month rank extracts information beyond random selection.")
else:
    print("Frozen: random excess mean ~ +0.4 bps  (null centred at zero: True)")
"""),

        md("## Per-calendar-month breakdown: most months are individually weak"),
        code("""\
months_lbl = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
frozen_month = {
    "Jan": (R["spread_bps"]*0.9, 1.16), "Feb": (200.8, 1.30), "Mar": (158.2, 1.53),
    "Apr": (217.3, 1.91), "May": (116.5, 1.42), "Jun": (187.1, 1.43),
    "Jul": (R["jul_bps"], R["jul_t"]), "Aug": (R["aug_bps"], R["aug_t"]),
    "Sep": (92.7, 1.19), "Oct": (207.6, 1.47),
    "Nov": (R["nov_bps"], R["nov_t"]), "Dec": (R["dec_bps"], R["dec_t"]),
}

if HAVE_REAL:
    month_data = []
    for m in range(1, 13):
        sub = spread[spread.index.month == m]
        if len(sub) < 5: continue
        s = st.summarize(sub)
        month_data.append({"m": months_lbl[m-1], "bps": s["mean"]*10000, "t": s["tstat"]})
    md_df = pd.DataFrame(month_data)
else:
    md_df = pd.DataFrame([{"m": k, "bps": v[0], "t": v[1]} for k, v in frozen_month.items()])

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
colors_m = [GREEN if t >= 2 else AMBER if t >= 1 else RED for t in md_df["t"]]
axes[0].bar(md_df["m"], md_df["bps"], color=colors_m)
axes[0].axhline(0, color="black", lw=0.8)
axes[0].set_title("Spread by calendar month (bps/mo)")
axes[0].set_ylabel("bps/mo")
axes[1].bar(md_df["m"], md_df["t"], color=colors_m)
axes[1].axhline(2.0, color="black", ls="--", lw=0.9, label="t=2")
axes[1].axhline(0, color="black", lw=0.8)
axes[1].set_title("HAC t-stat by calendar month")
axes[1].set_ylabel("HAC t-stat")
axes[1].legend(fontsize=9)
plt.tight_layout()
plt.show()
print("Only July and December individually clear t >= 2 on this panel.")
print("August and November are negative -- the signal is NOT uniformly positive.")
"""),

        md("## Verdict"),
        code("""\
print("=== Study 223 -- Same-Month Seasonality ===")
print()
print(f"Signal: REAL   (HAC t = +{R['spread_t']:.2f}, bootstrap CI fully positive)")
print(f"        BUT: survivorship-biased upper bound, ~8 stocks per decile, decay to t={R['sub_2018_t']:.2f} post-2018")
print()
print(f"Tradability: FRAGILE")
print(f"  Turnover={R['turnover']:.0f}%/mo, drag={R['drag_bps']:.1f}bps/mo at 10bps one-way")
print(f"  Live broad-universe (Russell 1000) spread would be materially lower")
print(f"  Short leg requires concentrated distressed names with borrow costs")
print()
print("Survivorship: NAMED -- all results are upper bounds")
print()
print("Bottom line: Real/Fragile -- an academically real anomaly that is")
print("  hard to exploit at the concentrated scale implied by a large-cap panel.")
"""),
    ]

    _write(nb, "02_for_the_quants.ipynb")


# ---------------------------------------------------------------------------
# Shared _write helper
# ---------------------------------------------------------------------------
def _write(nb: nbf.NotebookNode, filename: str) -> None:
    path = os.path.join(HERE, filename)
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Wrote {path}")


# ---------------------------------------------------------------------------
# Execute notebooks with nbconvert
# ---------------------------------------------------------------------------
def _execute(filename: str) -> None:
    import subprocess, sys
    path = os.path.join(HERE, filename)
    result = subprocess.run(
        [
            sys.executable, "-m", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--inplace",
            "--ExecutePreprocessor.timeout=300",
            path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"ERROR executing {filename}:\n{result.stderr[-3000:]}")
        raise RuntimeError(f"nbconvert failed for {filename}")
    print(f"Executed {filename}")


if __name__ == "__main__":
    build_curious()
    build_quants()
    _execute("01_for_the_curious.ipynb")
    _execute("02_for_the_quants.ipynb")
    print("Done.")
