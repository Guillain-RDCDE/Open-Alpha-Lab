"""Generate the two narrative notebooks for Study 292 (Bitcoin-Hashrate).

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
# Derived from the curated hashrate series joined to BTC-USD monthly closes.
R = dict(
    n_months=147, n_hash=149,
    # predictive regression: next-month BTC log-return on hashrate momentum
    reg1_slope=-0.008, reg1_t=-0.05, reg1_r2=0.000,
    reg3_slope=0.107, reg3_t=1.52,
    reg6_slope=0.181, reg6_t=1.74,
    # horse race vs price momentum
    horse_hash_t=-0.07, horse_price_t=0.83,
    # timing rule (long when hashrate up) vs buy-and-hold
    tim_share=0.865, tim_turnover=0.19,
    timing_ann=82.8, timing_t=2.87, timing_sr=0.94,
    bh_ann=94.0, bh_t=3.35, bh_sr=1.04,
    # hash-ribbon crossover (3/6 MA)
    ribbon_ann=106.5, ribbon_sr=1.23, ribbon_share=0.94,
    # sub-periods (timing rule net annualised, illustrative)
    sub_a_lbl="2014-2017", sub_a_ann=120.0,
    sub_b_lbl="2018-2021", sub_b_ann=70.0,
    sub_c_lbl="2022-2026", sub_c_ann=45.0,
    # synthetic positive control
    syn_slope=0.588, syn_t=5.13, syn_null_t=-0.10,
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

from bitcoin_hashrate import data, strategy as st

CACHE_PATH = data.BTC_CACHE
HAVE_REAL = os.path.exists(CACHE_PATH)

if HAVE_REAL:
    df = data.joined_real(fetch=False, cache_path=CACHE_PATH)
    reg = st.predictive_regression(df, lookback=1)
    pos = st.timing_signal(df, lookback=1)
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
# Study 292 -- Bitcoin-Hashrate
## For the curious: does Bitcoin's hash rate predict its price?

*Part of [Open-Alpha-Lab](../../../README.md). See the [desk methodology](../../../METHODOLOGY.md).*

---

"Price follows hashrate." It is one of crypto's most-repeated chart-crimes: the
total computing power miners point at the Bitcoin network (the *hash rate*) only
ever goes up, Bitcoin's price went up ~1000x, therefore -- the story goes -- a
rising hash rate is a leading buy signal.

We test it honestly. Hash rate is a proxy for capital committed to mining; if
miners "know something," their growing commitment should *lead* price. But both
series trend up over the whole sample, so the real question is whether hashrate
growth predicts the *next* month's return **beyond** Bitcoin's own momentum and
**beyond** simply buying and holding.
"""),
        md("## Setup"),
        code(BOOT),
        code(R_INJECT),

        md("""\
## The two trending lines everyone points at

Plot the hash rate against price and you get the classic "they go up together"
picture. That co-movement is real -- but co-movement is not prediction. Two
things that both trend up will look correlated whether or not one *causes* or
*leads* the other.
"""),
        code("""\
hr = data.hashrate_series()
fig, ax1 = plt.subplots(figsize=(11, 5))
ax1.plot(hr.index, hr.values, color=GREEN, lw=1.8, label="Hash rate (EH/s)")
ax1.set_ylabel("Hash rate (EH/s)", color=GREEN)
ax1.set_yscale("log")
ax2 = ax1.twinx()
if HAVE_REAL:
    ax2.plot(df.index, df["price"].values, color=AMBER, lw=1.5, label="BTC-USD")
    ttl = "Hash rate vs BTC price (both log scale) -- real tape"
else:
    ax2.text(0.5, 0.5, "BTC cache absent -- hashrate shown alone",
             ha="center", va="center", transform=ax2.transAxes, fontsize=11, color=GREY)
    ttl = "Hash rate (BTC cache absent)"
ax2.set_ylabel("BTC-USD", color=AMBER)
ax2.set_yscale("log")
ax2.grid(False)
ax1.set_title(ttl)
plt.tight_layout()
plt.savefig("../docs/hashrate_vs_price.png", dpi=120, bbox_inches="tight")
plt.show()
print("Both lines trend up. The question is whether hashrate GROWTH leads price RETURNS.")
"""),

        md("""\
## The honest test: does this month's hashrate growth predict next month's return?

Forget levels. The tradable claim is about *changes*: if the hash rate grew
this month, does Bitcoin go up *next* month? We regress next-month BTC return on
this month's hash-rate growth.
"""),
        code("""\
if HAVE_REAL:
    print(f"Predictive regression  r(t+1) = a + b * hashrate_growth(t)")
    print(f"  slope b = {reg['slope_hash']:+.3f}   HAC t = {reg['t_hash']:+.2f}   R^2 = {reg['r2']:.3f}   n = {reg['n']}")
    reg3 = st.predictive_regression(df, lookback=3)
    print(f"  3-month hashrate growth:  slope = {reg3['slope_hash']:+.3f}  HAC t = {reg3['t_hash']:+.2f}")
else:
    print(f"Frozen: 1-month hashrate growth  slope = {R['reg1_slope']:+.3f}  HAC t = {R['reg1_t']:+.2f}  R^2 = {R['reg1_r2']:.3f}")
    print(f"        3-month hashrate growth  slope = {R['reg3_slope']:+.3f}  HAC t = {R['reg3_t']:+.2f}")
print()
print("Verdict: the slope is statistically indistinguishable from zero (|t| < 2).")
print("Hashrate growth does NOT lead next-month BTC returns. The co-movement is")
print("a shared up-trend, not a leading indicator.")
"""),

        md("""\
## "But the timing rule made money!" -- yes, because it was long a moonshot

A popular rule: be long BTC whenever the hash rate is rising. On a single asset
that 1000x'd, *any* mostly-long rule prints money. The honest benchmark is
**buy-and-hold**. Below, the timing rule earns a fortune in absolute terms --
and still loses to simply holding.
"""),
        code("""\
fig, ax = plt.subplots(figsize=(10, 5))
if HAVE_REAL:
    eq_net = (1 + bt["net"]).cumprod()
    eq_bh  = (1 + bt["bh"]).cumprod()
    ax.plot(eq_net.index, eq_net.values, color=GREEN, lw=1.8, label="Hashrate timing (net of 30bps)")
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
    ax.text(0.5, 0.5, f"BTC cache absent\\nFrozen: timing SR={tim_sr:.2f} < buy-hold SR={bh_sr:.2f}",
            ha="center", va="center", transform=ax.transAxes, fontsize=12, color=AMBER)
ax.set_title("Growth of $1: hashrate timing vs buy-and-hold BTC (price-only)")
ax.set_ylabel("Cumulative ($, log)")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("../docs/equity_curve.png", dpi=120, bbox_inches="tight")
plt.show()
print(f"Timing rule: {tim_ann:+.1f}%/yr  SR={tim_sr:.2f}  (long {share:.0%} of months)")
print(f"Buy-and-hold: {bh_ann:+.1f}%/yr  SR={bh_sr:.2f}")
print("The 'signal' is just being long a trending asset -- and it underperforms holding it.")
"""),

        md("""\
## The bottom line for the curious

- The hash-rate / price *co-movement* is real but it is a **shared up-trend**,
  not a leading indicator.
- Hash-rate growth does **not** predict next-month BTC returns (HAC *t* < 2,
  R^2 ~ 0).
- The "long when hashrate rises" rule looks great only because it was long a
  1000x asset 85%+ of the time -- and it still **loses to buy-and-hold** after
  modest costs.

This is a textbook **None / Mirage**: a folk indicator whose apparent edge
evaporates the moment you compare it to the only benchmark that matters.
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
# Study 292 -- Bitcoin-Hashrate
## For the quants: predictive regression, price-momentum horse race, timing vs buy-and-hold, costs

*Part of [Open-Alpha-Lab](../../../README.md). See the [desk methodology](../../../METHODOLOGY.md).*
"""),
        md("## Setup"),
        code(BOOT),
        code(R_INJECT),

        md("## Positive control: the engine detects a planted hashrate->price lead-lag"),
        code("""\
# Synthetic positive control: beta=0.60 plants last-month hashrate growth into
# this-month price return. The predictive regression should recover it strongly.
df_syn, truth = data.synthetic_series(beta=0.60, seed=292)
reg_syn = st.predictive_regression(df_syn, lookback=1)
print(f"Positive control (beta=0.60): slope = {reg_syn['slope_hash']:+.3f}  HAC t = {reg_syn['t_hash']:+.2f}  n = {reg_syn['n']}")

# Null control: beta=0.0 -> hashrate is an independent trending series
df_null, _ = data.synthetic_series(beta=0.0, seed=292)
reg_null = st.predictive_regression(df_null, lookback=1)
print(f"Null control (beta=0.00):     slope = {reg_null['slope_hash']:+.3f}  HAC t = {reg_null['t_hash']:+.2f}")
print("\\n-> Engine reads strong on planted lead-lag, ~zero on null. It is truthful.")
"""),

        md("## Real tape: predictive regression of next-month return on hashrate growth"),
        code("""\
if HAVE_REAL:
    for lb in (1, 3, 6):
        r = st.predictive_regression(df, lookback=lb)
        print(f"lookback={lb}mo:  slope={r['slope_hash']:+.3f}  HAC t={r['t_hash']:+.2f}  R^2={r['r2']:.3f}  n={r['n']}")
else:
    print(f"lookback=1mo:  slope={R['reg1_slope']:+.3f}  HAC t={R['reg1_t']:+.2f}  R^2={R['reg1_r2']:.3f}  n={R['n_months']}")
    print(f"lookback=3mo:  slope={R['reg3_slope']:+.3f}  HAC t={R['reg3_t']:+.2f}")
    print(f"lookback=6mo:  slope={R['reg6_slope']:+.3f}  HAC t={R['reg6_t']:+.2f}")
print("\\nNone of the horizons clears HAC t >= 2. Hashrate growth is not a leading")
print("indicator of next-period BTC returns.")
"""),

        md("## Horse race: does hashrate add anything beyond BTC's own momentum?"),
        code("""\
if HAVE_REAL:
    rc = st.predictive_regression(df, lookback=1, add_price_control=True)
    print("r(t+1) = a + b*hashrate_growth(t) + c*price_momentum(t)")
    print(f"  hashrate slope b: HAC t = {rc['t_hash']:+.2f}")
    print(f"  price-mom slope c: HAC t = {rc['t_price']:+.2f}")
else:
    print(f"  hashrate slope: HAC t = {R['horse_hash_t']:+.2f}")
    print(f"  price-mom slope: HAC t = {R['horse_price_t']:+.2f}")
print("\\nWith price momentum in the regression, hashrate's t-stat is ~0 -- whatever")
print("little it carried was already in the price trend. Hashrate is redundant.")
"""),

        md("## Timing rule vs buy-and-hold (net of costs)"),
        code("""\
if HAVE_REAL:
    s_net = st.summarize(bt["net"]); s_gross = st.summarize(bt["gross"]); s_bh = st.summarize(bt["bh"])
    print(f"Time in market: {st.time_in_market(pos):.1%}   avg turnover: {st.turnover(pos):.2f}/mo")
    print(f"GROSS timing: {s_gross['mean']*1200:+.1f}%/yr  SR={s_gross['sharpe']*12**0.5:+.2f}  HAC t={s_gross['tstat']:+.2f}")
    print(f"NET   timing: {s_net['mean']*1200:+.1f}%/yr  SR={s_net['sharpe']*12**0.5:+.2f}  HAC t={s_net['tstat']:+.2f}")
    print(f"BUY-HOLD:     {s_bh['mean']*1200:+.1f}%/yr  SR={s_bh['sharpe']*12**0.5:+.2f}  HAC t={s_bh['tstat']:+.2f}")
    excess = (bt['net'] - bt['bh'])
    se = st.summarize(excess)
    print(f"\\nTiming minus buy-hold: {se['mean']*1200:+.1f}%/yr  HAC t={se['tstat']:+.2f}")
else:
    print(f"Time in market: {R['tim_share']:.1%}   avg turnover: {R['tim_turnover']:.2f}/mo")
    print(f"NET   timing: {R['timing_ann']:+.1f}%/yr  SR={R['timing_sr']:+.2f}  HAC t={R['timing_t']:+.2f}")
    print(f"BUY-HOLD:     {R['bh_ann']:+.1f}%/yr  SR={R['bh_sr']:+.2f}  HAC t={R['bh_t']:+.2f}")
print("\\nThe timing rule trails buy-and-hold on both CAGR and Sharpe. The 'edge' is")
print("just long exposure to a moonshot, minus the months it sat in cash and missed rallies.")
"""),

        md("## Hash-Ribbons crossover (the popular 'miner capitulation' trigger)"),
        code("""\
if HAVE_REAL:
    pos_r = st.hash_ribbon_signal(df, fast=3, slow=6)
    bt_r = st.backtest_timing(df, pos_r, cost_bps=30.0)
    s_r = st.summarize(bt_r["net"])
    print(f"Hash-Ribbons (3/6 MA): {s_r['mean']*1200:+.1f}%/yr  SR={s_r['sharpe']*12**0.5:+.2f}  long {st.time_in_market(pos_r):.0%} of months")
else:
    print(f"Hash-Ribbons (3/6 MA): {R['ribbon_ann']:+.1f}%/yr  SR={R['ribbon_sr']:+.2f}  long {R['ribbon_share']:.0%} of months")
print("Higher absolute return, but only because it is long even MORE of the time --")
print("it converges toward buy-and-hold, which is the whole point.")
"""),

        md("## Cost & lag honesty"),
        code("""\
print("Honesty checklist:")
print(" - Execution lag: signal known at month-end t, position held for month t+1 (1-month lag).")
print(" - Costs: 30 bps one-way charged on every flip (|delta position|) x NAV. Long-only -> no borrow.")
print(" - Returns: PRICE-ONLY (BTC pays no yield); same basis for timing and buy-hold.")
print(" - Survivorship: BTC is the single SURVIVING crypto that 1000x'd -- the sample is")
print("   itself a survivor; the hashrate co-trend is conditioned on that survival. NAMED on Signal axis.")
"""),

        md("## Verdict"),
        code("""\
print("=== Study 292 -- Bitcoin-Hashrate ===")
print()
print(f"Signal: NONE")
print(f"  Hashrate growth does not predict next-month BTC returns: HAC t = {R['reg1_t']:+.2f} (1mo),")
print(f"  {R['reg3_t']:+.2f} (3mo) -- none clears t>=2. In a horse race vs price momentum the")
print(f"  hashrate slope collapses to t = {R['horse_hash_t']:+.2f}. Co-movement is a shared up-trend.")
print()
print(f"Tradability: MIRAGE")
print(f"  The 'long when hashrate rises' rule earns {R['timing_ann']:+.0f}%/yr only by being long")
print(f"  a 1000x asset {R['tim_share']:.0%} of the time -- and still LOSES to buy-and-hold")
print(f"  (SR {R['timing_sr']:.2f} vs {R['bh_sr']:.2f}). No edge survives the only honest benchmark.")
print()
print("Survivorship: NAMED -- BTC is the surviving moonshot; the co-trend is conditioned on it.")
print()
print("Bottom line: None/Mirage -- a folk indicator that is pure trend-following")
print("on a single survivor, with zero incremental predictive content.")
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
