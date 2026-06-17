"""Generate the two narrative notebooks for Study 293 (MVRV-Ratio).

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


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-17).
# Derived from the curated MVRV series joined to BTC-USD monthly closes.
R = dict(
    n_months=140, n_mvrv=149,
    # predictive regression: next-month BTC log-return on MVRV stretch
    reg_slope=-0.0067, reg_t=-0.15, reg_r2=0.000,
    # horse race vs price momentum
    horse_mvrv_t=-0.70, horse_price_t=1.88,
    # per-band next-month return (%/mo) and n
    band_over_mo=-6.70, band_over_n=4, band_over_hit=0.25,
    band_neutral_mo=6.39, band_neutral_n=111,
    band_under_mo=4.97, band_under_n=25,
    # contrarian timing rule (cash when over-heated) vs buy-and-hold
    tim_share=0.972, tim_turnover=0.029,
    timing_ann=71.4, timing_t=2.89, timing_sr=1.03,
    bh_ann=69.2, bh_t=2.74, bh_sr=0.97,
    excess_ann=2.2, excess_t=0.59,
    # band-threshold sensitivity (net %/yr)
    high30_ann=60.1, high40_ann=71.5,
    # synthetic positive control
    syn_slope=-0.474, syn_t=-9.60, syn_null_t=0.52,
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

from mvrv_ratio import data, strategy as st

CACHE_PATH = data.BTC_CACHE
HAVE_REAL = os.path.exists(CACHE_PATH)

if HAVE_REAL:
    df = data.joined_real(fetch=False, cache_path=CACHE_PATH)
    reg = st.predictive_regression(df)
    pos = st.timing_signal(df, high=3.5, low=1.0)
    bt = st.backtest_timing(df, pos, cost_bps=30.0)
    print(f"Real tape: {len(df)} aligned months  {df.index[0].date()} -> {df.index[-1].date()}")
else:
    df = reg = pos = bt = None
    print("No real BTC cache -- frozen headline numbers from R dict will be used")
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
# Study 293 -- MVRV-Ratio
## For the curious: does on-chain MVRV time Bitcoin tops and bottoms?

*Part of [Open-Alpha-Lab](../../../README.md). See the [desk methodology](../../../METHODOLOGY.md).*

---

"Sell when MVRV is over 3.5, buy when it's under 1." It is one of crypto's most
beloved on-chain rules. **MVRV** -- market value divided by *realized* value
(every coin priced at the moment it last moved on-chain) -- is sold as a
contrarian valuation gauge: high MVRV means euphoria and an over-valued top, low
MVRV means capitulation and a bottom.

We test it honestly. The trouble is that realized value is a slow average of
*past* prices, so MVRV and price are mechanically linked -- and Bitcoin has only
had about four cycles, so the "sell" band is calibrated to a handful of tops. The
real questions: does MVRV predict *next* month's return, and does a contrarian
rule beat simply **holding** BTC?
"""),
        md("## Setup"),
        code(BOOT),
        code(R_INJECT),

        md("""\
## The MVRV oscillator everyone points at

Plot MVRV and you get a tidy oscillator: it spikes above ~3.5 at the 2017 and
2021 tops and sinks below ~1.0 at the 2018 and 2022 bottoms. It *looks* like a
perfect market timer in hindsight -- which is exactly the problem with a band
drawn after the fact through four extremes.
"""),
        code("""\
mv = data.mvrv_series()
fig, ax1 = plt.subplots(figsize=(11, 5))
ax1.plot(mv.index, mv.values, color=GREEN, lw=1.8, label="MVRV ratio")
ax1.axhline(3.5, color=RED, ls="--", lw=1.0, label="over-heated (>=3.5)")
ax1.axhline(1.0, color=AMBER, ls="--", lw=1.0, label="under-valued (<=1.0)")
ax1.set_ylabel("MVRV ratio", color=GREEN)
ax2 = ax1.twinx()
if HAVE_REAL:
    ax2.plot(df.index, df["price"].values, color=GREY, lw=1.3, alpha=0.8, label="BTC-USD")
    ttl = "MVRV ratio vs BTC price -- real tape"
else:
    ax2.text(0.5, 0.5, "BTC cache absent -- MVRV shown alone",
             ha="center", va="center", transform=ax2.transAxes, fontsize=11, color=GREY)
    ttl = "MVRV ratio (BTC cache absent)"
ax2.set_ylabel("BTC-USD", color=GREY)
ax2.set_yscale("log")
ax2.grid(False)
ax1.set_title(ttl)
ax1.legend(fontsize=8, loc="upper left")
plt.tight_layout()
plt.savefig("../docs/mvrv_vs_price.png", dpi=120, bbox_inches="tight")
plt.show()
print("MVRV spikes at tops and sinks at bottoms IN HINDSIGHT. The question is")
print("whether today's MVRV predicts NEXT month's return -- out of sample.")
"""),

        md("""\
## The honest test: does this month's MVRV predict next month's return?

Forget the picture. The tradable claim is contrarian: if MVRV is *stretched*
high this month, BTC should fall *next* month. We regress next-month BTC return
on this month's MVRV stretch (log of MVRV vs its mid-band).
"""),
        code("""\
if HAVE_REAL:
    print("Predictive regression  r(t+1) = a + b * MVRV_stretch(t)")
    print(f"  slope b = {reg['slope_mvrv']:+.4f}   HAC t = {reg['t_mvrv']:+.2f}   R^2 = {reg['r2']:.3f}   n = {reg['n']}")
else:
    print(f"Frozen: slope b = {R['reg_slope']:+.4f}  HAC t = {R['reg_t']:+.2f}  R^2 = {R['reg_r2']:.3f}  n = {R['n_months']}")
print()
print("The slope has the right (negative, contrarian) sign -- but it is")
print("statistically indistinguishable from zero (|t| < 2, R^2 ~ 0).")
print("MVRV stretch does NOT lead next-month BTC returns.")
"""),

        md("""\
## "But the over-heated band nailed the tops!" -- on four months

The one place MVRV looks predictive is the over-heated band: months with
MVRV >= 3.5 are followed by an average **-6.7%** the next month. Impressive --
until you count how many months actually triggered it.
"""),
        code("""\
if HAVE_REAL:
    tab = st.state_forward_stats(df, high=3.5, low=1.0)
    print(tab.to_string())
    over_mo = tab.loc["over-heated", "mean"]*100; over_n = int(tab.loc["over-heated", "n"])
else:
    print("over-heated  : %+.2f%%/mo  n=%d" % (R['band_over_mo'], R['band_over_n']))
    print("neutral      : %+.2f%%/mo  n=%d" % (R['band_neutral_mo'], R['band_neutral_n']))
    print("under-valued : %+.2f%%/mo  n=%d" % (R['band_under_mo'], R['band_under_n']))
    over_mo, over_n = R['band_over_mo'], R['band_over_n']
print()
print(f"The over-heated band fires only {over_n} times in the whole sample.")
print("That is the 2017 and 2021 tops and nothing else -- anecdote, not evidence.")
print("And the under-valued 'buy' band does NOT beat ordinary neutral months.")
"""),

        md("""\
## The timing rule is just buy-and-hold wearing a costume

A charitable contrarian rule: hold BTC, but step to cash whenever MVRV is
over-heated. On a single asset that 1000x'd, the honest benchmark is
**buy-and-hold**. Below, the rule earns a fortune in absolute terms -- because it
is long 97% of the time -- and barely differs from simply holding.
"""),
        code("""\
fig, ax = plt.subplots(figsize=(10, 5))
if HAVE_REAL:
    eq_net = (1 + bt["net"]).cumprod()
    eq_bh  = (1 + bt["bh"]).cumprod()
    ax.plot(eq_net.index, eq_net.values, color=GREEN, lw=1.8, label="MVRV contrarian timing (net 30bps)")
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
    ax.text(0.5, 0.5, f"BTC cache absent\\nFrozen: timing SR={tim_sr:.2f} ~ buy-hold SR={bh_sr:.2f}",
            ha="center", va="center", transform=ax.transAxes, fontsize=12, color=AMBER)
ax.set_title("Growth of $1: MVRV contrarian timing vs buy-and-hold BTC (price-only)")
ax.set_ylabel("Cumulative ($, log)")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("../docs/equity_curve.png", dpi=120, bbox_inches="tight")
plt.show()
print(f"Timing rule: {tim_ann:+.1f}%/yr  SR={tim_sr:.2f}  (long {share:.0%} of months)")
print(f"Buy-and-hold: {bh_ann:+.1f}%/yr  SR={bh_sr:.2f}")
print("The 'edge' over holding is ~+2%/yr and not significant. It is buy-and-hold")
print("that occasionally blinks to cash -- not a market timer.")
"""),

        md("""\
## The bottom line for the curious

- The MVRV / price *co-movement* is real but it is a **mechanical link** (realized
  value is a slow average of past price), not a leading indicator.
- MVRV stretch does **not** predict next-month BTC returns (HAC *t* < 2, R^2 ~ 0).
- The "sell the over-heated top" band looks sharp but fires on **four months** in
  twelve years -- the 2017 and 2021 tops.
- The contrarian timing rule is **97% buy-and-hold** and its edge over simply
  holding is statistically zero.

This is a textbook **None / Mirage**: a beloved on-chain gauge whose apparent
edge is hindsight band-fitting on a single survivor.
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
# Study 293 -- MVRV-Ratio
## For the quants: contrarian regression, price-momentum horse race, per-band returns, timing vs buy-and-hold

*Part of [Open-Alpha-Lab](../../../README.md). See the [desk methodology](../../../METHODOLOGY.md).*
"""),
        md("## Setup"),
        code(BOOT),
        code(R_INJECT),

        md("## Positive control: the engine detects a planted contrarian MVRV->price link"),
        code("""\
# Synthetic positive control: beta=0.50 plants last-month MVRV stretch NEGATIVELY
# into this-month price return (the contrarian mechanism). The regression should
# recover a strongly negative slope.
df_syn, truth = data.synthetic_series(beta=0.50, seed=293)
reg_syn = st.predictive_regression(df_syn)
print(f"Positive control (beta=0.50): slope = {reg_syn['slope_mvrv']:+.3f}  HAC t = {reg_syn['t_mvrv']:+.2f}  n = {reg_syn['n']}")

# Null control: beta=0.0 -> MVRV is an independent mean-reverting series
df_null, _ = data.synthetic_series(beta=0.0, seed=293)
reg_null = st.predictive_regression(df_null)
print(f"Null control (beta=0.00):     slope = {reg_null['slope_mvrv']:+.3f}  HAC t = {reg_null['t_mvrv']:+.2f}")
print("\\n-> Engine reads strongly negative on a planted contrarian link, ~zero on null. It is truthful.")
"""),

        md("## Real tape: predictive regression of next-month return on MVRV stretch"),
        code("""\
if HAVE_REAL:
    r = st.predictive_regression(df)
    print(f"r(t+1) = a + b*MVRV_stretch(t)")
    print(f"  slope={r['slope_mvrv']:+.4f}  HAC t={r['t_mvrv']:+.2f}  R^2={r['r2']:.4f}  n={r['n']}")
else:
    print(f"  slope={R['reg_slope']:+.4f}  HAC t={R['reg_t']:+.2f}  R^2={R['reg_r2']:.4f}  n={R['n_months']}")
print("\\nThe slope is the right (negative) sign for the contrarian story but does")
print("NOT clear |t| >= 2. MVRV stretch is not a leading indicator of BTC returns.")
"""),

        md("## Horse race: does MVRV add anything beyond BTC's own momentum?"),
        code("""\
if HAVE_REAL:
    rc = st.predictive_regression(df, add_price_control=True)
    print("r(t+1) = a + b*MVRV_stretch(t) + c*price_momentum(t)")
    print(f"  MVRV slope b: HAC t = {rc['t_mvrv']:+.2f}")
    print(f"  price-mom slope c: HAC t = {rc['t_price']:+.2f}")
else:
    print(f"  MVRV slope: HAC t = {R['horse_mvrv_t']:+.2f}")
    print(f"  price-mom slope: HAC t = {R['horse_price_t']:+.2f}")
print("\\nWith price momentum in the regression, MVRV's t-stat stays ~0. Whatever")
print("little contrarian whiff it carries is not incremental to the price trend.")
"""),

        md("## Per-band forward returns: the 'sell' band rests on n=4"),
        code("""\
if HAVE_REAL:
    tab = st.state_forward_stats(df, high=3.5, low=1.0)
    print(tab.to_string())
else:
    print("over-heated  : %+.2f%%/mo  hit=%.2f  n=%d" % (R['band_over_mo'], R['band_over_hit'], R['band_over_n']))
    print("neutral      : %+.2f%%/mo            n=%d" % (R['band_neutral_mo'], R['band_neutral_n']))
    print("under-valued : %+.2f%%/mo            n=%d" % (R['band_under_mo'], R['band_under_n']))
print("\\nThe over-heated band's negative forward return is real but rests on FOUR")
print("months (2017 + 2021 tops). The under-valued 'buy' band does not beat neutral.")
print("With n=4 the 'sell the top' edge is anecdote, not a testable signal.")
"""),

        md("## Contrarian timing rule vs buy-and-hold (net of costs)"),
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
    print(f"NET   timing: {R['timing_ann']:+.1f}%/yr  SR={R['timing_sr']:+.2f}  HAC t={R['timing_t']:+.2f}")
    print(f"BUY-HOLD:     {R['bh_ann']:+.1f}%/yr  SR={R['bh_sr']:+.2f}  HAC t={R['bh_t']:+.2f}")
    print(f"\\nTiming minus buy-hold: {R['excess_ann']:+.1f}%/yr  HAC t={R['excess_t']:+.2f}")
print("\\nThe rule is 97% buy-and-hold. Its +2%/yr edge over holding is NOT significant")
print("(HAC t ~ 0.6) and comes entirely from sidestepping a handful of post-top months.")
"""),

        md("## Band-threshold sensitivity: a knob, not a signal"),
        code("""\
if HAVE_REAL:
    for h in (3.0, 3.5, 4.0):
        p = st.timing_signal(df, high=h, low=1.0)
        b = st.backtest_timing(df, p, cost_bps=30.0)
        s = st.summarize(b["net"])
        print(f"high={h:.1f}: net {s['mean']*1200:+.1f}%/yr  SR={s['sharpe']*12**0.5:+.2f}  long {st.time_in_market(p):.0%}")
    s_bh = st.summarize(bt["bh"])
    print(f"buy-hold : {s_bh['mean']*1200:+.1f}%/yr  SR={s_bh['sharpe']*12**0.5:+.2f}")
else:
    print(f"high=3.0: net {R['high30_ann']:+.1f}%/yr  (trails buy-hold)")
    print(f"high=3.5: net {R['timing_ann']:+.1f}%/yr  (the cherry-picked sweet spot)")
    print(f"high=4.0: net {R['high40_ann']:+.1f}%/yr  (~ buy-hold, ~99% in market)")
    print(f"buy-hold: {R['bh_ann']:+.1f}%/yr")
print("\\nTighten the band and it trails buy-hold; loosen it and it converges to buy-hold.")
print("The 'edge' is a threshold-tuning artefact, the signature of a mirage.")
"""),

        md("## Cost & lag honesty"),
        code("""\
print("Honesty checklist:")
print(" - Execution lag: MVRV known at month-end t, position held for month t+1 (1-month lag).")
print(" - Costs: 30 bps one-way charged on every flip (|delta position|) x NAV. Long/flat -> no borrow.")
print(" - Returns: PRICE-ONLY (BTC pays no yield); same basis for timing and buy-hold.")
print(" - Single-survivor: BTC is the one crypto that 1000x'd and MVRV is DERIVED from its")
print("   own price path (realized cap = slow average of past prices). The contrarian bands")
print("   are fitted to ~four cycle turns. NAMED on the Signal axis.")
"""),

        md("## Verdict"),
        code("""\
print("=== Study 293 -- MVRV-Ratio ===")
print()
print(f"Signal: NONE")
print(f"  MVRV stretch does not predict next-month BTC returns: HAC t = {R['reg_t']:+.2f} (R^2 ~ 0).")
print(f"  In a horse race vs price momentum the MVRV slope is t = {R['horse_mvrv_t']:+.2f}. The only")
print(f"  suggestive number -- the over-heated band's {R['band_over_mo']:+.1f}%/mo -- rests on n = {R['band_over_n']} months.")
print()
print(f"Tradability: MIRAGE")
print(f"  The contrarian 'cash when over-heated' rule is {R['tim_share']:.0%} buy-and-hold; its")
print(f"  {R['excess_ann']:+.1f}%/yr edge over holding is insignificant (HAC t = {R['excess_t']:+.2f}) and flips")
print(f"  sign with the band threshold. Any CAGR is just long exposure to a 1000x survivor.")
print()
print("Single-survivor: NAMED -- BTC is the surviving moonshot; MVRV is derived from its price.")
print()
print("Bottom line: None/Mirage -- a beloved on-chain gauge that is hindsight")
print("band-fitting on a single survivor, with zero incremental predictive content.")
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
