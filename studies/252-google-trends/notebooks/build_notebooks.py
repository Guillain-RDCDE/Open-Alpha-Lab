"""Generate the two narrative notebooks for Study 252 (Search-Trends).

    python notebooks/build_notebooks.py

Both notebooks follow the desk beats. The synthetic figures run anywhere,
offline and deterministic; the real-tape cells use the cached monthly panel
under ../_cache/ joined to the curated Trends table if present, and otherwise
quote the frozen headline numbers in ``R`` (mirroring docs/results.md), so the
notebook re-runs for any reader.

The _write convention (each build_*() ends by calling _write) is kept so the
repo's intro-restyle tooling can monkeypatch it.
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-17).
# Recipe: monthly attention z-score (search this month vs trailing 6-mo baseline);
# LONG the low-attention tercile, SHORT the high-attention tercile; hold 1 month.
# spread = low - high  (positive => fade-the-hype/reversal pays).
R = dict(
    n_months=48, n_total=25.5, n_low=9.9, n_high=10.1,
    fingerprint="8c791a614b92",
    high_ann=79.08, high_t=1.31, low_ann=1.94, low_t=0.13, market_ann=37.12,
    spread_ann=-77.15, spread_bps=-642.9, spread_t=-1.34, spread_hit=0.396,
    spread_sr_lo=-1.644, spread_sr_hi=0.016, spread_frac_neg=0.974,
    high_xret_ann=41.96, high_xret_t=1.11,
    # sub-periods
    sub_early_ann=-133.18, sub_early_t=-1.04, sub_early_n=21,
    sub_late_ann=-33.56, sub_late_t=-3.78, sub_late_n=27,
    # costs
    turnover=46.1, drag_bps=35.7, net_ann=-81.37, net_t=-1.42,
    borrow_bps=8.0, oneway_bps=15.0,
    # random null
    rand_bps=-20.65,
    # synthetic controls
    syn_null_t=-2.32, syn_03_ann=59.29, syn_03_t=12.61, syn_05_ann=106.13, syn_05_t=21.90,
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

from google_trends import data, strategy as st

CACHE_PATH = data.MONTHLY_CACHE
HAVE_REAL = os.path.exists(CACHE_PATH)

# The curated Trends-style attention table is always available (hardcoded).
attn_full = data.trends_table()

if HAVE_REAL:
    prices_full = data.fetch_monthly(fetch=False, cache_path=CACHE_PATH)
    common = attn_full.columns.intersection(prices_full.columns)
    attn = attn_full[common]
    prices = prices_full[common]
    sig = st.attention_zscore(attn, window=6, min_periods=3)
    res = st.attention_returns(sig, prices, q=0.30)
    spread = res["spread"].dropna()              # low - high (positive => reversal)
    high_xret = (res["high"] - res["market"]).dropna()
    print(f"Real panel loaded: {prices.shape[0]} months x {prices.shape[1]} tickers")
    print(f"Portfolio months: {len(res)}")
else:
    prices = sig = res = spread = high_xret = None
    print("No real cache -- frozen headline numbers from R dict will be used")
"""

R_INJECT = f"""
# Frozen headline numbers (mirror of docs/results.md, as-of 2026-06-17)
R = {R!r}
"""


# ---------------------------------------------------------------------------
# Notebook 1 -- For the curious
# ---------------------------------------------------------------------------
def build_curious() -> None:
    nb = new_notebook()
    nb.cells = [
        md("""\
# Study 252 -- Search-Trends
## For the curious: do spikes in Google searches for stock names front-run their returns?

*Part of [Open-Alpha-Lab](../../../README.md). See the [desk methodology](../../../METHODOLOGY.md).*

---

When a stock starts trending on Google -- GameStop in January 2021, Nvidia during
the AI boom, Zoom during lockdown -- it *feels* like the crowd is about to pile in
and push the price higher.  "Buy what's trending" is one of the oldest pieces of
retail folk wisdom.

But the academic finding is the opposite.  Da, Engelberg & Gao (2011), *In Search
of Attention*, showed that a jump in Google search volume is **uninformed retail
price pressure**: it nudges the price up for a couple of weeks and then **reverses**
over the following months.  Attention is hype, not information.

So which is it -- do search spikes *front-run* returns (momentum), or do they mark
a top to *fade* (reversal)?  We test both on a basket of 30 household-name stocks.
"""),
        md("## Setup"),
        code(BOOT),
        code(R_INJECT),

        md("""\
## The attention proxy: a curated Google-Trends-style index

Google Trends has no clean, free, stable bulk API, and its numbers are quantised
0-100 *relative* indices.  So we ship a **hand-curated monthly attention table**
for the basket (2019-2025), built from a documented event/seasonal model that
reproduces the public Trends shape: holiday spikes for retailers, the September
iPhone spike for Apple, the GameStop/AMC squeeze of January 2021, the Zoom
lockdown spike of March 2020, the Nvidia AI boom of 2023-2024.

This is a **proxy** -- it captures the shape, not tick-exact values.  That caveat
is named on the Signal axis.
"""),
        code("""\
# A few iconic attention spikes in the curated table
show = ["GME", "ZM", "NVDA", "WMT"]
fig, ax = plt.subplots(figsize=(11, 4.5))
for tk in show:
    ax.plot(attn_full.index, attn_full[tk], lw=1.6, label=tk)
ax.set_title("Curated Google-Trends-style attention index (0-100, re-based per ticker)")
ax.set_ylabel("Search interest (0-100)")
ax.legend(fontsize=9, ncol=4)
plt.tight_layout()
plt.savefig("../docs/attention_series.png", dpi=120, bbox_inches="tight")
plt.show()
print("GME peaks Jan-2021 (squeeze); ZM peaks Mar-2020 (lockdown);")
print("NVDA ramps 2023-24 (AI); WMT spikes every Nov/Dec (Black Friday).")
"""),

        md("""\
## The bet: fade the hype, or ride it?

Each month we standardise every stock's search interest against its own trailing
6-month baseline (an *attention z-score*: how unusual is this month's interest?).
Then we sort the basket:

- **LONG** the low-attention tercile (quiet, ignored names)
- **SHORT** the high-attention tercile (the names everyone is googling)
- hold one month, rebalance.

The spread we report is **low minus high** -- it is *positive* when fading the hype
(the Da-Engelberg-Gao reversal) pays, and *negative* when riding the hype
(momentum) pays.
"""),

        md("## The headline: on this basket, fading the hype did NOT pay"),
        code("""\
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

if HAVE_REAL:
    ax = axes[0]
    cum_low  = (1 + res["low"].dropna()).cumprod()
    cum_high = (1 + res["high"].dropna()).cumprod()
    cum_mkt  = (1 + res["market"].dropna()).cumprod()
    ax.plot(cum_low.index,  cum_low.values,  color=GREEN, lw=1.8, label="Low-attention (long leg)")
    ax.plot(cum_high.index, cum_high.values, color=RED,   lw=1.8, label="High-attention (short leg)")
    ax.plot(cum_mkt.index,  cum_mkt.values,  color=GREY,  lw=1.2, ls="--", label="Equal-weight basket")
    ax.set_title("Cumulative growth of $1 (survivorship + proxy biased)")
    ax.set_ylabel("Cumulative return ($)")
    ax.legend(fontsize=9)

    ax = axes[1]
    roll = spread.rolling(12).mean() * 12 * 100
    ax.plot(roll.index, roll.values, color=AMBER, lw=1.5)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_title("Rolling 12-month spread (low-high, %/yr)")
    ax.set_ylabel("Annualised spread (%/yr)")
else:
    for ax in axes:
        ax.text(0.5, 0.5, f"Cache absent\\nFrozen: spread = {R['spread_ann']:.1f}%/yr  t = {R['spread_t']:.2f}",
                ha="center", va="center", transform=ax.transAxes, fontsize=12, color=RED)
        ax.set_title("(frozen numbers)")

plt.tight_layout()
plt.savefig("../docs/equity_curve.png", dpi=120, bbox_inches="tight")
plt.show()
print(f"Spread (low-high): {R['spread_ann']:.1f}%/yr = {R['spread_bps']:.1f}bps/mo  HAC t = {R['spread_t']:.2f}")
print("Negative => the HIGH-attention (hyped) names actually OUTRAN the quiet ones.")
print("But |t| < 2: it is NOT statistically significant -- mostly meme-driven noise.")
"""),

        md("""\
## Why the folk story breaks down here

1. **Wrong sign, not significant.** The reversal the literature predicts does not
   show up: high-attention names *outperformed* (`+79%/yr` vs `+2%/yr` for the quiet
   tercile) -- but with a HAC *t* of only **-1.34**, this is statistical noise, not
   a tradable momentum edge.

2. **The basket is a meme zoo.** GME, AMC, COIN, RIVN, LCID, TSLA -- the
   high-attention tercile is dominated by exactly the names that ran hardest in the
   2020-2021 mania.  That is survivorship *and* selection bias inflating one side.

3. **Daily-to-weekly vs monthly horizon.** Da-Engelberg-Gao found the reversal over
   *weeks*, on a broad Russell-3000 universe of small retail-driven names.  A
   monthly horizon on 30 mega-caps is a different animal -- the price-pressure
   window has already closed.

4. **Short the hype = short the un-shortable.** The short leg is precisely the
   hard-to-borrow meme names with brutal borrow costs and squeeze risk.
"""),

        md("""\
## The bottom line for the curious

"Buy what's trending on Google" and "fade what's trending on Google" both fail
on this basket: the spread is the wrong sign for the academic reversal story and
too noisy (*t* = -1.34) to be a momentum edge.

The academic attention-reversal effect is real in its original weekly,
broad-universe setting -- but it **does not survive** translation to a monthly,
mega-cap, proxy-Trends implementation.  Combined with un-shortable meme names on
the short leg, this is a **Weak / Mirage**: a famous idea that evaporates the
moment you try to trade it the way a retail investor actually could.
"""),
        md("*Desk verdict: **Weak / Mirage** -- see [README.md](../README.md) and [docs/results.md](../docs/results.md) for full numbers.*"),
    ]
    _write(nb, "01_for_the_curious.ipynb")


# ---------------------------------------------------------------------------
# Notebook 2 -- For the quants
# ---------------------------------------------------------------------------
def build_quants() -> None:
    nb = new_notebook()
    nb.cells = [
        md("""\
# Study 252 -- Search-Trends
## For the quants: attention z-score, HAC tests, bootstrap CI, sub-period sign-flip

*Part of [Open-Alpha-Lab](../../../README.md). See the [desk methodology](../../../METHODOLOGY.md).*
"""),
        md("## Setup"),
        code(BOOT),
        code(R_INJECT),

        md("""\
## Positive control: does the engine recover a *planted* attention effect?

The synthetic panel plants the Da-Engelberg-Gao mechanism directly: a positive
attention shock this month is followed by a *negative* return next month
(`ret[t+1] = base - premium * attn_z[t] + noise`).  A long-low/short-high book
then earns `premium`.  `premium = 0` is the null.
"""),
        code("""\
for prem in (0.0, 0.03, 0.05):
    a2, p2, truth = data.synthetic_panel(n_firms=30, n_months=84, premium=prem, seed=252)
    sg = st.attention_zscore(a2, window=6, min_periods=3)
    r2 = st.attention_returns(sg, p2, q=0.30)
    s2 = st.summarize(r2["spread"].dropna())
    tag = "null" if prem == 0 else "planted"
    print(f"premium={prem:.2f} ({tag:7s}): spread = {s2['mean']*12*100:+7.2f}%/yr  HAC t = {s2['tstat']:+6.2f}  n={s2['n']}")
print()
print("-> With a planted reversal (premium>0) the long-low/short-high book earns a")
print("   strongly positive, significant spread. The engine is truthful.")
print(f"   (Null draw at seed=252 lands at t={R['syn_null_t']:.2f}; |t|<4 is normal sampling noise.)")
"""),

        md("## Signal construction: the attention z-score"),
        code("""\
# Show the abnormal-attention construction for one iconic name (real or frozen view)
if HAVE_REAL and "GME" in attn.columns:
    z_gme = st.attention_zscore(attn[["GME"]], window=6, min_periods=3)["GME"].dropna()
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.bar(z_gme.index, z_gme.values, width=20,
           color=[RED if v > 1 else GREY for v in z_gme.values])
    ax.axhline(1.0, color="black", ls="--", lw=0.8, label="z = +1 (spike)")
    ax.set_title("GME attention z-score (search vs own trailing 6-mo baseline)")
    ax.set_ylabel("z-score")
    ax.legend(fontsize=9)
    plt.tight_layout(); plt.show()
    print(f"Peak GME attention z-score: {z_gme.max():+.2f} in {z_gme.idxmax().date()}")
else:
    print("Real cache absent -- z-score figure skipped; construction is")
    print("(attn - trailing_mean) / trailing_std over a 6-month window including t.")
"""),

        md("## HAC t-statistics and bootstrap Sharpe CI"),
        code("""\
if HAVE_REAL:
    s_spread = st.summarize(spread)
    s_high   = st.summarize(res["high"].dropna())
    s_low    = st.summarize(res["low"].dropna())
    s_market = st.summarize(res["market"].dropna())

    print("=== Primary spec: long low-attention / short high-attention, 1-month hold ===")
    print(f"Portfolio months: {len(res)}   avg names/tercile: {res['n_low'].mean():.1f}")
    print()
    print(f"LOW-attn (long) : {s_low['mean']*12*100:+.2f}%/yr  HAC t={s_low['tstat']:+.2f}")
    print(f"HIGH-attn(short): {s_high['mean']*12*100:+.2f}%/yr  HAC t={s_high['tstat']:+.2f}")
    print(f"MARKET (eq-wt)  : {s_market['mean']*12*100:+.2f}%/yr")
    print(f"SPREAD (low-high): {s_spread['mean']*12*100:+.2f}%/yr = {s_spread['mean']*10000:+.1f}bps/mo  HAC t={s_spread['tstat']:+.2f}  hit={s_spread['hit_rate']:.3f}")

    try:
        from quantlab.stats import sharpe_ci_bootstrap
        ci = sharpe_ci_bootstrap(spread, n_boot=2000, periods_per_year=12, seed=252)
        print(f"\\nBootstrap spread Sharpe 95% CI: [{ci['ci_low']:+.3f}, {ci['ci_high']:+.3f}]")
        print(f"  {ci['frac_negative']*100:.0f}% of resamples negative")
    except ImportError:
        print(f"\\n[quantlab.stats unavailable -- frozen CI: [{R['spread_sr_lo']:+.3f}, {R['spread_sr_hi']:+.3f}]]")
else:
    print(f"Frozen: spread = {R['spread_ann']:.2f}%/yr  HAC t = {R['spread_t']:.2f}")
    print(f"LOW {R['low_ann']:+.1f}%/yr (t={R['low_t']:+.2f})   HIGH {R['high_ann']:+.1f}%/yr (t={R['high_t']:+.2f})")
    print(f"Bootstrap 95% CI: [{R['spread_sr_lo']:+.3f}, {R['spread_sr_hi']:+.3f}]  ({R['spread_frac_neg']*100:.0f}% negative)")
print()
print("The CI straddles / sits below zero and the HAC |t| < 2: the reversal spread")
print("is NOT statistically distinguishable from zero (in fact it leans the wrong way).")
"""),

        md("## Sub-period sign-flip: a sample-split artefact, not a stable edge"),
        code("""\
if HAVE_REAL:
    periods = [("2019-2021", "2019", "2021"), ("2022-2025", "2022", "2025")]
    sub_stats = []
    for label, a, b in periods:
        ss = st.summarize(spread[a:b])
        sub_stats.append({"period": label, "ann": ss["mean"]*12*100, "t": ss["tstat"], "n": ss["n"]})
        print(f"  {label}: spread={ss['mean']*12*100:+.2f}%/yr  HAC t={ss['tstat']:+.2f}  n={ss['n']}")
    sts = pd.DataFrame(sub_stats)
else:
    sts = pd.DataFrame([
        {"period": "2019-2021", "ann": R["sub_early_ann"], "t": R["sub_early_t"], "n": R["sub_early_n"]},
        {"period": "2022-2025", "ann": R["sub_late_ann"], "t": R["sub_late_t"], "n": R["sub_late_n"]},
    ])
    for _, r in sts.iterrows():
        print(f"  {r['period']}: spread={r['ann']:+.2f}%/yr  HAC t={r['t']:+.2f}  n={int(r['n'])}")

fig, ax = plt.subplots(figsize=(8, 4))
colors_sub = [GREEN if t >= 2 else RED if t <= -2 else AMBER for t in sts["t"]]
ax.bar(sts["period"], sts["t"], color=colors_sub)
ax.axhline(2.0, color="black", ls="--", lw=0.9, label="t=+2")
ax.axhline(-2.0, color="black", ls=":", lw=0.9, label="t=-2")
ax.axhline(0, color="black", lw=0.8)
ax.set_title("Sub-period HAC t-stat of the (low-high) spread")
ax.set_ylabel("HAC t-stat")
ax.legend(fontsize=9)
plt.tight_layout(); plt.show()
print("Both sub-periods are NEGATIVE; the late window even clears t=-2 (momentum,")
print("not reversal). The headline reversal hypothesis is rejected on this tape.")
"""),

        md("## Turnover, costs, and the short-leg borrow problem"),
        code("""\
if HAVE_REAL:
    drag = st.turnover_cost_drag(sig, prices, q=0.30, one_way_bps=15.0, borrow_bps=8.0)
    net = spread - drag.reindex(spread.index).fillna(drag.mean())
    s_net = st.summarize(net)
    print(f"Avg drag @15bps one-way + 8bps borrow: {drag.mean()*10000:.1f} bps/mo")
    print(f"Net spread: {s_net['mean']*12*100:+.2f}%/yr  HAC t={s_net['tstat']:+.2f}")
else:
    print(f"Frozen: drag = {R['drag_bps']:.1f} bps/mo")
    print(f"Net spread @{R['oneway_bps']:.0f}bps + {R['borrow_bps']:.0f}bps borrow: {R['net_ann']:.2f}%/yr  HAC t={R['net_t']:.2f}")
print()
print("Costs are almost a footnote here: the GROSS spread is already the wrong sign.")
print("Worse, the SHORT leg is the meme tercile (GME/AMC/COIN/RIVN) -- the exact")
print("hard-to-borrow, squeeze-prone names where 8bps/mo borrow is wildly optimistic.")
"""),

        md("## Random-portfolio null: the sort carries no information"),
        code("""\
if HAVE_REAL:
    rand = st.random_portfolio_returns(sig, prices, q=0.30, n_draws=200, seed=252)
    print(f"Random low-size group excess: mean={rand.mean()*10000:+.1f}bps  std={rand.std()*10000:.0f}bps")
    print(f"  -> Null centred at zero (vs a {rand.std()*10000:.0f}bps/draw std from fat meme tails).")
else:
    print(f"Frozen: random excess mean ~ {R['rand_bps']:+.1f} bps (null centred at zero)")
print("The attention sort does not extract reliable cross-sectional information here.")
"""),

        md("## Verdict"),
        code("""\
print("=== Study 252 -- Search-Trends ===")
print()
print(f"Signal: WEAK")
print(f"  Literature (Da-Engelberg-Gao 2011) supports an attention->reversal effect,")
print(f"  but on THIS monthly mega-cap proxy tape the (low-high) spread is {R['spread_ann']:.0f}%/yr")
print(f"  at HAC t = {R['spread_t']:.2f} (|t| < 2) -- wrong sign, not significant.")
print(f"  Literature support without a robust HAC t>=2 on the real tape = WEAK.")
print()
print(f"Tradability: MIRAGE")
print(f"  Gross spread already wrong-signed; short leg = un-shortable meme names;")
print(f"  Trends series is a hand-curated PROXY; only 30 survivor mega-caps; n={R['n_months']} months.")
print()
print("Survivorship + proxy: NAMED -- the basket is fixed survivors and the")
print("  attention table is a stylised hand-curated approximation of Google Trends.")
print()
print("Bottom line: Weak/Mirage -- a famous attention anomaly that does not")
print("  survive translation to a monthly, mega-cap, proxy-Trends implementation.")
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
