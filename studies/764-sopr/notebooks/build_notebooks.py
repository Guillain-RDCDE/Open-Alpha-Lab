"""Generate the two narrative notebooks for Study 764 (SOPR).

    python notebooks/build_notebooks.py

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the cached
BTC monthly series under ../_cache/ if present and otherwise quote the frozen headline
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-07-13).
# Derived from the curated SOPR series joined to BTC-USD monthly closes.
R = dict(
    n_months=141, n_sopr=150, aligned=142,
    # predictive regression: next-month BTC log-return on SOPR stretch
    reg_slope=0.9528, reg_t=1.32, reg_r2=0.013,
    # horse race vs price momentum
    horse_sopr_t=0.18, horse_price_t=1.04,
    # per-band next-month return (%/mo), n, hit
    band_greed_mo=8.87, band_greed_n=44, band_greed_hit=0.59,
    band_neutral_mo=4.75, band_neutral_n=67,
    band_cap_mo=2.60, band_cap_n=30, band_cap_hit=0.53,
    # ">1 / <1" regime timing rule vs buy-and-hold
    tim_share=0.620, tim_turnover=0.255,
    gross_ann=60.6, gross_t=2.66,
    timing_ann=59.7, timing_t=2.61, timing_sr=0.98,
    bh_ann=67.0, bh_t=2.66, bh_sr=0.93,
    excess_ann=-7.3, excess_t=-0.73,
    # threshold sensitivity (net %/yr)
    th99_ann=59.5, th100_ann=59.7, th101_ann=54.9,
    # placebo: shuffle SOPR in time
    placebo_real_mo=-0.61, placebo_mean_mo=-2.25, placebo_std_mo=0.83, placebo_p=0.975,
    # synthetic controls
    syn_slope=1.286, syn_t=5.75, syn_null_t=-0.21,
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

from sopr import data, strategy as st

CACHE_PATH = data.BTC_CACHE
HAVE_REAL = os.path.exists(CACHE_PATH)

if HAVE_REAL:
    df = data.joined_real(fetch=False, cache_path=CACHE_PATH)
    reg = st.predictive_regression(df)
    pos = st.timing_signal(df, thresh=1.0)
    bt = st.backtest_timing(df, pos, cost_bps=30.0)
    print(f"Real tape: {len(df)} aligned months  {df.index[0].date()} -> {df.index[-1].date()}")
else:
    df = reg = pos = bt = None
    print("No real BTC cache -- frozen headline numbers from R dict will be used")
"""

R_INJECT = f"""
# Frozen headline numbers (mirror of docs/results.md, as-of 2026-07-13)
R = {R!r}
"""


# ---------------------------------------------------------------------------
# Notebook 1 -- For the curious
# ---------------------------------------------------------------------------
def build_curious() -> None:
    nb = new_notebook()
    nb.cells = [
        md("""\
# Study 764 -- SOPR 🔗
## For the curious: does on-chain SOPR time Bitcoin capitulation and greed?

*Part of [Open-Alpha-Lab](../../../README.md). See the [desk methodology](../../../METHODOLOGY.md).*

![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)
![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)
![Single-survivor: Named](https://img.shields.io/badge/Single--survivor-Named-8b949e?style=flat-square)

---

"When SOPR is above 1 the market is in profit -- stay long; when it drops below 1
holders are capitulating -- step aside." It is one of crypto's most quoted
on-chain rules. **SOPR** -- the Spent Output Profit Ratio -- is the average
"sale price / cost basis" of every coin that moves on-chain: above 1 the movers
are selling at a profit, below 1 at a loss. The famous chart-lore says the line
**1** acts as *support* in bull markets and *resistance* in bear markets.

We test it honestly. The catch is that SOPR is computed from realised
profit/loss against *past* prices -- so a down month mechanically drags SOPR
below 1. That makes SOPR close to a re-label of "was last month up or down." The
real questions: does SOPR predict *next* month's return, and does the ">1 / <1"
rule beat simply **holding** BTC?

> ⚠️ **Not investment advice.** Research and education only. See the [LICENSE](../../../LICENSE).
> The SOPR series here is a **labelled proxy** digitised from the public Glassnode chart,
> not a live feed. Companion: [02_for_the_quants](02_for_the_quants.ipynb).
"""),
        md("## Setup"),
        code(BOOT),
        code(R_INJECT),

        md("""\
## The oscillator everyone points at

Plot SOPR and you get a tidy line hugging **1**: it pokes above 1 in the bull
runs of 2017, 2021 and 2024, and knifes below 1 at the capitulation lows
(2018-12, the 2020 covid crash, the 2022 FTX bottom). It *looks* like a clean
regime switch in hindsight -- which is exactly the trap with a level that is
mechanically tied to recent price.
"""),
        code("""\
sp = data.sopr_series()
fig, ax1 = plt.subplots(figsize=(11, 5))
ax1.plot(sp.index, sp.values, color=GREEN, lw=1.6, label="Adjusted SOPR (proxy)")
ax1.axhline(1.0, color=RED, ls="--", lw=1.0, label="profit/loss line (SOPR = 1)")
ax1.set_ylabel("SOPR", color=GREEN)
ax2 = ax1.twinx()
if HAVE_REAL:
    ax2.plot(df.index, df["price"].values, color=GREY, lw=1.3, alpha=0.8, label="BTC-USD")
    ttl = "SOPR vs BTC price -- real tape"
else:
    ax2.text(0.5, 0.5, "BTC cache absent -- SOPR shown alone",
             ha="center", va="center", transform=ax2.transAxes, fontsize=11, color=GREY)
    ttl = "SOPR (BTC cache absent)"
ax2.set_ylabel("BTC-USD", color=GREY)
ax2.set_yscale("log")
ax2.grid(False)
ax1.set_title(ttl)
ax1.legend(fontsize=8, loc="upper left")
plt.tight_layout()
plt.savefig("../docs/sopr_vs_price.png", dpi=120, bbox_inches="tight")
plt.show()
print("SOPR dips below 1 at the lows and rises above 1 in the bulls IN HINDSIGHT.")
print("The question is whether today's SOPR predicts NEXT month's return.")
"""),

        md("""\
## The honest test: does this month's SOPR predict next month's return?

Forget the picture. The tradable claim is momentum: if SOPR is comfortably above
1 this month, BTC should keep rising *next* month. We regress next-month BTC
return on this month's SOPR stretch (log of SOPR vs the profit/loss line at 1).
"""),
        code("""\
if HAVE_REAL:
    print("Predictive regression  r(t+1) = a + b * SOPR_stretch(t)")
    print(f"  slope b = {reg['slope_sopr']:+.4f}   HAC t = {reg['t_sopr']:+.2f}   R^2 = {reg['r2']:.3f}   n = {reg['n']}")
else:
    print(f"Frozen: slope b = {R['reg_slope']:+.4f}  HAC t = {R['reg_t']:+.2f}  R^2 = {R['reg_r2']:.3f}  n = {R['n_months']}")
print()
print("The slope has the right (positive, momentum) sign -- but it is")
print("statistically indistinguishable from zero (|t| < 2, R^2 ~ 0.01).")
print("SOPR stretch does NOT reliably lead next-month BTC returns.")
"""),

        md("""\
## "But greed months really do run hotter!" -- the grain of truth

There *is* a whiff here. Sort months by SOPR band and the average next-month
return lines up in the folk direction -- greed months beat capitulation months.
That monotone ordering is why the indicator refuses to die. But look at *how* the
rule uses it below, and the grain of truth evaporates.
"""),
        code("""\
if HAVE_REAL:
    tab = st.state_forward_stats(df, high=1.02, low=0.98)
    print(tab.to_string())
else:
    print("greed        : %+.2f%%/mo  n=%d" % (R['band_greed_mo'], R['band_greed_n']))
    print("neutral      : %+.2f%%/mo  n=%d" % (R['band_neutral_mo'], R['band_neutral_n']))
    print("capitulation : %+.2f%%/mo  n=%d" % (R['band_cap_mo'], R['band_cap_n']))
print()
print("Greed > neutral > capitulation, in the folk order -- BUT the gap is not")
print("significant, and (next cell) it dies once you control for price momentum:")
print("'greed months' are largely just 'months right after price went up.'")
"""),

        md("""\
## The timing rule doesn't beat just holding -- it *loses*

The literal rule: hold BTC while SOPR >= 1, step to cash when it drops below.
On a single asset that ~150x'd, the honest benchmark is **buy-and-hold**. The
problem: SOPR drops below 1 exactly at the capitulation lows -- which are the
months that most often *rebound*. So the rule sells the bottom and buys back
higher.
"""),
        code("""\
fig, ax = plt.subplots(figsize=(10, 5))
if HAVE_REAL:
    eq_net = (1 + bt["net"]).cumprod()
    eq_bh  = (1 + bt["bh"]).cumprod()
    ax.plot(eq_net.index, eq_net.values, color=GREEN, lw=1.8, label="SOPR >1/<1 timing (net 30bps)")
    ax.plot(eq_bh.index,  eq_bh.values,  color=GREY,  lw=1.5, ls="--", label="Buy-and-hold BTC")
    ax.set_yscale("log")
    s_net = st.summarize(bt["net"]); s_bh = st.summarize(bt["bh"])
    tim_ann, tim_sr = s_net["mean"]*1200, s_net["sharpe"]*12**0.5
    bh_ann, bh_sr = s_bh["mean"]*1200, s_bh["sharpe"]*12**0.5
    share = st.time_in_market(pos)
else:
    tim_ann, tim_sr = R["timing_ann"], R["timing_sr"]
    bh_ann, bh_sr = R["bh_ann"], R["bh_sr"]
    share = R["tim_share"]
    ax.text(0.5, 0.5, f"BTC cache absent\\nFrozen: timing {tim_ann:.0f}%/yr < buy-hold {bh_ann:.0f}%/yr",
            ha="center", va="center", transform=ax.transAxes, fontsize=12, color=AMBER)
ax.set_title("Growth of $1: SOPR >1/<1 timing vs buy-and-hold BTC (price-only)")
ax.set_ylabel("Cumulative ($, log)")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("../docs/equity_curve.png", dpi=120, bbox_inches="tight")
plt.show()
print(f"Timing rule: {tim_ann:+.1f}%/yr  SR={tim_sr:.2f}  (long {share:.0%} of months)")
print(f"Buy-and-hold: {bh_ann:+.1f}%/yr  SR={bh_sr:.2f}")
print("The rule GIVES UP ~7%/yr versus simply holding. A slightly smoother ride,")
print("bought by forfeiting part of a 150x. That is not market timing.")
"""),

        md("""\
## The bottom line for the curious

- SOPR / price *co-movement* is real, but it is **mechanical** (SOPR is realised
  profit against past prices), not a leading indicator.
- SOPR stretch does **not** reliably predict next-month BTC returns (HAC *t* =
  +1.3, R^2 ~ 0.01), and the whiff it has dies against plain price momentum.
- The monotone band ordering (greed > capitulation) is the grain of truth -- but
  it is not significant and not tradable.
- The ">1 / <1" timing rule **loses ~7%/yr to buy-and-hold** by sitting in cash
  through capitulation months that tend to rebound.

This is a textbook **None / Mirage**: a beloved on-chain gauge whose apparent
edge is hindsight regime-labelling on a single survivor.
"""),
        md("*Desk verdict: **None / Mirage** -- see [README.md](../README.md) and [docs/results.md](../docs/results.md) for full numbers.*"),
    ]
    _write(nb, "01_for_the_curious.ipynb")


# ---------------------------------------------------------------------------
# Notebook 2 -- For the quants
# ---------------------------------------------------------------------------
def build_quants() -> None:
    nb = new_notebook()
    nb.cells = [
        md("""\
# Study 764 -- SOPR 🔗
## For the quants: momentum regression, price horse race, per-band returns, >1/<1 timing vs buy-and-hold, time-shuffle placebo

*Part of [Open-Alpha-Lab](../../../README.md). See the [desk methodology](../../../METHODOLOGY.md).*

![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)
![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)
![Single-survivor: Named](https://img.shields.io/badge/Single--survivor-Named-8b949e?style=flat-square)

The headline numbers are frozen in `R` (mirror of [docs/results.md](../docs/results.md),
as-of 2026-07-13); the cells recompute them live from the cached BTC tape when present.
"""),
        md("## Setup"),
        code(BOOT),
        code(R_INJECT),

        md("""\
## Positive control: the engine detects a planted momentum SOPR->price link

> 💡 **In plain words:** before trusting the harness on the real tape, we feed it
> fake data where we *know* the answer. If we plant "high SOPR this month -> higher
> return next month," the regression must find it; if we plant nothing, it must
> read ~zero. It does both. So a null on the real tape means *no signal*, not a
> broken tool.
"""),
        code("""\
# Positive control: beta=2.0 plants last-month SOPR stretch POSITIVELY into this-
# month price return (the momentum mechanism). SOPR's real stretch is tiny (~0.02
# in log), so a strong planted effect needs a large beta. Regression should
# recover a clearly positive slope.
df_syn, truth = data.synthetic_series(beta=2.0, seed=764)
reg_syn = st.predictive_regression(df_syn)
print(f"Positive control (beta=2.0): slope = {reg_syn['slope_sopr']:+.3f}  HAC t = {reg_syn['t_sopr']:+.2f}  n = {reg_syn['n']}")

# Null control: beta=0.0 -> SOPR is an independent mean-reverting series
df_null, _ = data.synthetic_series(beta=0.0, seed=764)
reg_null = st.predictive_regression(df_null)
print(f"Null control (beta=0.00):    slope = {reg_null['slope_sopr']:+.3f}  HAC t = {reg_null['t_sopr']:+.2f}")
print("\\n-> Engine reads strongly positive on a planted momentum link, ~zero on null. It is truthful.")
"""),

        md("## Real tape: predictive regression of next-month return on SOPR stretch"),
        code("""\
if HAVE_REAL:
    r = st.predictive_regression(df)
    print("r(t+1) = a + b*SOPR_stretch(t)")
    print(f"  slope={r['slope_sopr']:+.4f}  HAC t={r['t_sopr']:+.2f}  R^2={r['r2']:.4f}  n={r['n']}")
else:
    print(f"  slope={R['reg_slope']:+.4f}  HAC t={R['reg_t']:+.2f}  R^2={R['reg_r2']:.4f}  n={R['n_months']}")
print("\\nThe slope is the right (positive) sign for the momentum story but does")
print("NOT clear |t| >= 2. SOPR stretch is not a robust leading indicator.")
"""),

        md("""\
## Horse race: does SOPR add anything beyond BTC's own momentum?

> 💡 **In plain words:** SOPR is built from recent price action, so it might just
> be echoing "price went up." We put BTC's own one-month momentum in the same
> regression -- if SOPR's *t* collapses, it was never adding information.
"""),
        code("""\
if HAVE_REAL:
    rc = st.predictive_regression(df, add_price_control=True)
    print("r(t+1) = a + b*SOPR_stretch(t) + c*price_momentum(t)")
    print(f"  SOPR slope b: HAC t = {rc['t_sopr']:+.2f}")
    print(f"  price-mom slope c: HAC t = {rc['t_price']:+.2f}")
else:
    print(f"  SOPR slope: HAC t = {R['horse_sopr_t']:+.2f}")
    print(f"  price-mom slope: HAC t = {R['horse_price_t']:+.2f}")
print("\\nWith price momentum in the regression, SOPR's t-stat collapses toward 0.")
print("Its faint directional whiff is not incremental to the price trend.")
"""),

        md("## Per-band forward returns: monotone but insignificant"),
        code("""\
if HAVE_REAL:
    tab = st.state_forward_stats(df, high=1.02, low=0.98)
    print(tab.to_string())
else:
    print("greed        : %+.2f%%/mo  hit=%.2f  n=%d" % (R['band_greed_mo'], R['band_greed_hit'], R['band_greed_n']))
    print("neutral      : %+.2f%%/mo            n=%d" % (R['band_neutral_mo'], R['band_neutral_n']))
    print("capitulation : %+.2f%%/mo  hit=%.2f  n=%d" % (R['band_cap_mo'], R['band_cap_hit'], R['band_cap_n']))
print("\\nThe ordering is monotone in the folk direction (greed > neutral > capit.),")
print("but the gap is not significant (regression t=1.3) and dies in the horse race.")
"""),

        md("## The '>1 / <1' regime timing rule vs buy-and-hold (net of costs)"),
        code("""\
if HAVE_REAL:
    s_net = st.summarize(bt["net"]); s_gross = st.summarize(bt["gross"]); s_bh = st.summarize(bt["bh"])
    print(f"Time in market: {st.time_in_market(pos):.1%}   avg turnover: {st.turnover(pos):.3f}/mo")
    print(f"GROSS timing: {s_gross['mean']*1200:+.1f}%/yr  SR={s_gross['sharpe']*12**0.5:+.2f}  HAC t={s_gross['tstat']:+.2f}")
    print(f"NET   timing: {s_net['mean']*1200:+.1f}%/yr  SR={s_net['sharpe']*12**0.5:+.2f}  HAC t={s_net['tstat']:+.2f}")
    print(f"BUY-HOLD:     {s_bh['mean']*1200:+.1f}%/yr  SR={s_bh['sharpe']*12**0.5:+.2f}  HAC t={s_bh['tstat']:+.2f}")
    excess = (bt['net'] - bt['bh'])
    se = st.summarize(excess)
    print(f"\\nTiming minus buy-hold: {se['mean']*1200:+.1f}%/yr  HAC t={se['tstat']:+.2f}")
else:
    print(f"Time in market: {R['tim_share']:.1%}   avg turnover: {R['tim_turnover']:.3f}/mo")
    print(f"GROSS timing: {R['gross_ann']:+.1f}%/yr  HAC t={R['gross_t']:+.2f}")
    print(f"NET   timing: {R['timing_ann']:+.1f}%/yr  SR={R['timing_sr']:+.2f}  HAC t={R['timing_t']:+.2f}")
    print(f"BUY-HOLD:     {R['bh_ann']:+.1f}%/yr  SR={R['bh_sr']:+.2f}  HAC t={R['bh_t']:+.2f}")
    print(f"\\nTiming minus buy-hold: {R['excess_ann']:+.1f}%/yr  HAC t={R['excess_t']:+.2f}")
print("\\nThe rule is out of the market 38% of the time and LOSES ~7%/yr to holding")
print("(HAC t ~ -0.7). It buys a slightly higher Sharpe by forfeiting return -- on a")
print("150x asset that is a bad trade, not an edge.")
"""),

        md("## Threshold sensitivity: no winning knob"),
        code("""\
if HAVE_REAL:
    for th in (0.99, 1.00, 1.01):
        p = st.timing_signal(df, thresh=th)
        b = st.backtest_timing(df, p, cost_bps=30.0)
        s = st.summarize(b["net"])
        print(f"thresh={th:.2f}: net {s['mean']*1200:+.1f}%/yr  SR={s['sharpe']*12**0.5:+.2f}  long {st.time_in_market(p):.0%}")
    s_bh = st.summarize(bt["bh"])
    print(f"buy-hold  : {s_bh['mean']*1200:+.1f}%/yr  SR={s_bh['sharpe']*12**0.5:+.2f}")
else:
    print(f"thresh=0.99: net {R['th99_ann']:+.1f}%/yr")
    print(f"thresh=1.00: net {R['th100_ann']:+.1f}%/yr  (the folk value)")
    print(f"thresh=1.01: net {R['th101_ann']:+.1f}%/yr")
    print(f"buy-hold   : {R['bh_ann']:+.1f}%/yr")
print("\\nEvery threshold trails buy-and-hold. There is no value of the knob at which")
print("the rule wins -- the signature of a mirage, not a robust edge.")
"""),

        md("""\
## Placebo: shuffle SOPR in time

> 💡 **In plain words:** if we scramble the SOPR values so they no longer line up
> with the right months, how well does the rule do? A *random* long/flat schedule
> that is long ~62% of the time still loses to buy-and-hold, because missing part
> of a strong uptrend costs money. If the real rule's result sits inside that
> random cloud, SOPR added nothing.
"""),
        code("""\
if HAVE_REAL:
    pl = st.placebo_edge(df, n_shuffles=2000)
    real_mo, mean_mo, std_mo, pval = pl['real_edge_mo']*100, pl['placebo_mean_mo']*100, pl['placebo_std_mo']*100, pl['p_value']
else:
    real_mo, mean_mo, std_mo, pval = R['placebo_real_mo'], R['placebo_mean_mo'], R['placebo_std_mo'], R['placebo_p']
print(f"Real rule edge over buy-hold : {real_mo:+.2f}%/mo")
print(f"Placebo edge (shuffled SOPR) : {mean_mo:+.2f}%/mo  (std {std_mo:.2f}pp)")
print(f"Two-sided empirical p        : {pval:.3f}")
print("\\nA random 62%-long schedule loses ~2.3%/mo to holding; the real SOPR rule loses")
print("LESS (~0.6%/mo), so SOPR does time better than a coin flip. But it STILL loses to")
print("buy-and-hold, and its edge is nowhere near the tail (p~0.98). 'Beats random,")
print("loses to holding' = a weak, untradable regime filter.")
"""),

        md("## Cost & lag honesty"),
        code("""\
print("Honesty checklist:")
print(" - Execution lag: SOPR known at month-end t, position held for month t+1 (1-month lag).")
print(" - Costs: 30 bps one-way charged on every flip (|delta position|) x NAV. Long/flat -> no borrow.")
print(" - Returns: PRICE-ONLY (BTC pays no yield); same basis for timing and buy-hold.")
print(" - Proxy label: the SOPR series is a DIGITISED PROXY of the public Glassnode aSOPR chart,")
print("   hardcoded in sopr/data.py -- NOT a live feed. Named as a proxy, never under a real-tape banner.")
print(" - Partial month: the join stops at 2026-06-30 (last SOPR month), dropping the in-progress July bar.")
print(" - Single-survivor: BTC is the one crypto that ~150x'd and SOPR is DERIVED from its own")
print("   on-chain spending. The regime thresholds are fitted to ~four cycle turns. NAMED on Signal axis.")
"""),

        md("## Verdict"),
        code("""\
print("=== Study 764 -- SOPR ===")
print()
print("Signal: NONE")
print(f"  SOPR stretch does not robustly predict next-month BTC returns: HAC t = {R['reg_t']:+.2f}")
print(f"  (R^2 ~ {R['reg_r2']:.2f}); in a horse race vs price momentum the SOPR slope is t = {R['horse_sopr_t']:+.2f}.")
print(f"  The band ordering is monotone (greed {R['band_greed_mo']:+.1f} > capit. {R['band_cap_mo']:+.1f} %/mo) but no t clears 2.")
print()
print("Tradability: MIRAGE")
print(f"  The '>1 / <1' rule LOSES {R['excess_ann']:+.1f}%/yr to buy-and-hold (HAC t = {R['excess_t']:+.2f}) and")
print(f"  trails at every threshold. The placebo shows it beats a coin flip (p = {R['placebo_p']:.2f}) yet still")
print("  can't beat holding. Any CAGR is just long exposure to a 150x survivor, minus 38% time in cash.")
print()
print("Single-survivor: NAMED -- BTC is the surviving moonshot; SOPR is derived from its own spending.")
print()
print("Bottom line: None/Mirage -- a beloved on-chain gauge that is hindsight")
print("regime-labelling on a single survivor, with no incremental predictive content.")
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
