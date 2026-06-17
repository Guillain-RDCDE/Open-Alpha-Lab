"""Generate the two narrative notebooks for Study 255 (Fear-Greed).

    python notebooks/build_notebooks.py

Both notebooks follow the desk beats. The synthetic figures run anywhere,
offline and deterministic; the real-tape cells use the curated weekly Fear &
Greed series joined to the cached ^GSPC weekly closes if present, and otherwise
quote the frozen headline numbers in ``R`` (mirroring docs/results.md), so the
notebook re-runs for any reader. Every code cell is executed via nbconvert with
no error outputs.
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-12).
R = dict(
    n_weeks=767, stamp="2011-09-30 -> 2026-06-12", fingerprint="6d62af784bd5",
    uncond_ann=14.05,
    # band conditional forward returns (ann %, HAC t, n)
    ef_ann=14.20, ef_t=0.61, ef_n=65,
    fear_ann=6.68, fear_t=0.73, fear_n=183,
    neu_ann=15.24, neu_t=2.26, neu_n=165,
    greed_ann=17.23, greed_t=5.33, greed_n=332,
    eg_ann=18.17, eg_t=1.40, eg_n=22,
    # contrarian spread
    spread_bps=1.31, spread_ann=0.68, spread_t=0.41, spread_hit=0.056, spread_sr=0.07,
    spread_ci_lo=-0.241, spread_ci_hi=0.455, spread_frac_neg=33,
    fear_excess_ann=0.15, fear_excess_t=0.01,
    slope_bps=16.82, slope_t=0.53,
    # sub-periods
    sub_2011_ann=2.71, sub_2011_t=1.32, sub_2011_n=275,
    sub_2017_ann=0.66, sub_2017_t=0.15, sub_2017_n=261,
    sub_2022_ann=-1.70, sub_2022_t=-0.64, sub_2022_n=231,
    # null
    null_mean_bps=1.40, null_std_bps=2.64, real_pctile=47.5,
    # costs / turnover
    pct_long=8.5, pct_flat=88.7, pct_short=2.9, n_trades=41, trades_per_year=2.8,
    net_ann=0.53, net_t=0.31,
    # positive control
    ctrl_null_bps=-5.66, ctrl_null_t=-1.96,
    ctrl_60_bps=3.76, ctrl_60_t=1.32,
    ctrl_120_bps=13.18, ctrl_120_t=3.85,
)


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

from fear_greed_index import data, strategy as st

GSPC_CACHE = data.GSPC_CACHE
HAVE_REAL = os.path.exists(GSPC_CACHE)

if HAVE_REAL:
    panel = data.real_panel(fetch=False)
    fr = st.forward_returns(panel)
    spread = st.spread_returns(fr)
    rm = st.regime_means(fr)
    print(f"Real tape loaded: {len(panel)} weeks  {panel.index[0].date()} -> {panel.index[-1].date()}")
    print(f"Forward-return rows: {len(fr)}")
else:
    panel = fr = spread = rm = None
    print("No ^GSPC cache -- frozen headline numbers from R dict will be used")
"""

R_INJECT = f"""
# Frozen headline numbers (mirror of docs/results.md, as-of 2026-06-12)
R = {R!r}
"""


# ---------------------------------------------------------------------------
# Notebook 1 -- For the curious
# ---------------------------------------------------------------------------
def build_curious() -> None:
    nb = new_notebook()
    nb.cells = [
        md("""\
# Study 255 -- Fear-Greed
## For the curious: can CNN's Fear & Greed Index time the market?

*Part of [Open-Alpha-Lab](../../../README.md). See the [desk methodology](../../../METHODOLOGY.md).*

---

CNN's **Fear & Greed Index** is a 0-100 mood ring for the stock market: 0 is
"Extreme Fear", 100 is "Extreme Greed". The folk rule -- a retail-friendly
version of Warren Buffett's *"be greedy when others are fearful"* -- says you
should **buy when the index is in Extreme Fear and sell when it is in Extreme
Greed**.

It *feels* obviously right: surely the best time to buy was the depths of the
March-2020 crash (the index hit single digits), and the worst time was the
giddy highs of late 2021. We put that intuition on the clock.
"""),
        md("## Setup"),
        code(BOOT),
        code(R_INJECT),

        md("""\
## What the index actually did

The Fear & Greed Index does swing exactly where you'd expect: deep fear at the
2011 debt-ceiling scare and the 2015 China wobble, **single digits in the
March-2020 COVID crash**, euphoric greed in late 2020 / 2021, sustained fear
through the 2022 bear, and a sharp fear spike in the spring-2025 tariff scare.
The question is not whether the index *describes* sentiment -- it does -- but
whether that description **predicts next week's return**.
"""),

        code("""\
fig, ax = plt.subplots(figsize=(12, 4.2))
fng_w = data.fng_weekly()
ax.plot(fng_w.index, fng_w.values, color=GREY, lw=1.2)
ax.axhspan(0, 25, color=RED, alpha=0.10)
ax.axhspan(75, 100, color=GREEN, alpha=0.10)
ax.axhline(25, color=RED, lw=0.8, ls="--")
ax.axhline(75, color=GREEN, lw=0.8, ls="--")
ax.text(fng_w.index[5], 12, "Extreme Fear", color=RED, fontsize=9, va="center")
ax.text(fng_w.index[5], 88, "Extreme Greed", color=GREEN, fontsize=9, va="center")
ax.set_ylim(0, 100)
ax.set_title("Curated CNN Fear & Greed Index (weekly, 2011-2026)")
ax.set_ylabel("Fear & Greed (0-100)")
plt.tight_layout()
plt.savefig("../docs/fng_series.png", dpi=120, bbox_inches="tight")
plt.show()
print("Note: a hardcoded month-end anchor table interpolated to weekly -- a regime-faithful proxy,")
print("not a tick-exact CNN reconstruction (the contrarian signal only acts on the fear/greed regime).")
"""),

        md("""\
## The test: what happens the week *after* each mood?

We sort every week into the five published bands and measure the **average
return over the following week**. If the contrarian claim were true, the
Extreme-Fear bar would tower over the Extreme-Greed bar.
"""),
        code("""\
bands = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
if HAVE_REAL:
    anns = [rm.loc[b, "mean_ann"] * 100 for b in bands]
    ts   = [rm.loc[b, "tstat"] for b in bands]
    ns   = [int(rm.loc[b, "n"]) for b in bands]
    uncond = fr["fwd_ret"].mean() * 52 * 100
else:
    anns = [R["ef_ann"], R["fear_ann"], R["neu_ann"], R["greed_ann"], R["eg_ann"]]
    ts   = [R["ef_t"], R["fear_t"], R["neu_t"], R["greed_t"], R["eg_t"]]
    ns   = [R["ef_n"], R["fear_n"], R["neu_n"], R["greed_n"], R["eg_n"]]
    uncond = R["uncond_ann"]

colors = [RED, AMBER, GREY, "#1f78b4", GREEN]
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.bar(bands, anns, color=colors)
ax.axhline(uncond, color="black", ls="--", lw=1.0, label=f"Unconditional drift ({uncond:+.1f}%/yr)")
for i, (a, t, n) in enumerate(zip(anns, ts, ns)):
    ax.text(i, a + 0.4, f"t={t:+.2f}\\nn={n}", ha="center", fontsize=8)
ax.set_ylabel("Forward weekly return (annualised, %)")
ax.set_title("Next-week return by Fear & Greed band -- the contrarian claim FAILS")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("../docs/by_band.png", dpi=120, bbox_inches="tight")
plt.show()
print("The bars RISE with greed, not fear. Extreme-Fear weeks (+14.2%/yr) barely match")
print("the +14.0%/yr unconditional drift; the only strong t-stat is GREED (+5.3) -- the bull trend.")
"""),

        md("""\
## Why the story feels true but isn't

The reason "buy fear" *sounds* like wisdom is **survivorship of the anecdote**:
we remember March 2020 because the market roared back. But across 767 weeks the
market went *up* about 60% of the time in every band -- fear weeks included no
special bonus. Worse, greed weeks did slightly *better*, because greed clusters
inside the long bull trends that dominate the sample. The contrarian
fear-minus-greed overlay earns a statistically invisible **+0.7%/yr (t = 0.41)**.

Shuffling the sentiment column at random produces the *same* result on average
(the real signal sits at the 47.5th percentile of that shuffled null) -- the
clearest possible sign that the index carries no timing information here.
"""),

        md("## The bottom line for the curious"),
        code("""\
print("=== Can CNN's Fear & Greed Index time the market? ===")
print()
print(f"Contrarian fear-minus-greed overlay: {R['spread_ann']:+.2f}%/yr  HAC t = {R['spread_t']:+.2f}")
print(f"  Bootstrap Sharpe 95% CI: [{R['spread_ci_lo']:+.2f}, {R['spread_ci_hi']:+.2f}]  ({R['spread_frac_neg']}% of resamples negative)")
print(f"  Real spread vs shuffled-sentiment null: {R['real_pctile']:.1f}th percentile (i.e. ~the middle)")
print()
print("Verdict: Signal NONE / Tradability MIRAGE.")
print("The index is a great thermometer and a useless clock.")
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
# Study 255 -- Fear-Greed
## For the quants: band means, HAC tests, shuffled-sentiment null, sub-periods, costs

*Part of [Open-Alpha-Lab](../../../README.md). See the [desk methodology](../../../METHODOLOGY.md).*
"""),
        md("## Setup"),
        code(BOOT),
        code(R_INJECT),

        md("""\
## Positive control: the engine recovers a planted contrarian premium

Before trusting a null result we confirm the machinery *can* see a contrarian
effect when one exists. `data.synthetic_panel(contrarian_bps=...)` plants a
forward-return tilt that is positive after low sentiment and negative after high
sentiment. With `contrarian_bps=0` the spread is a noisy zero; with a strong
premium it clears `t > 2`.

> 💡 **In plain words:** this is the smoke test. If the synthetic control didn't
> light up, a flat real result would just mean the code was broken.
"""),
        code("""\
for prem in [0.0, 60.0, 120.0]:
    d, _ = data.synthetic_panel(contrarian_bps=prem, seed=255)
    f = st.forward_returns(d)
    sc = st.summarize(st.spread_returns(f))
    print(f"premium={prem:6.1f} bps -> spread {sc['mean']*10000:+6.2f} bps/wk  HAC t={sc['tstat']:+.2f}  n={sc['n']}")
print()
print(f"Frozen: null t={R['ctrl_null_t']:+.2f} (within null band), "
      f"+120bps t={R['ctrl_120_t']:+.2f} (clears the bar). The engine works.")
"""),

        md("## Band conditional means with HAC t-stats"),
        code("""\
bands = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
if HAVE_REAL:
    print("=== Forward weekly return by Fear & Greed band ===")
    print(f"Unconditional: {fr['fwd_ret'].mean()*52*100:+.2f}%/yr\\n")
    for b in bands:
        row = rm.loc[b]
        print(f"  {b:14s} n={int(row['n']):4d}  {row['mean_ann']*100:+7.2f}%/yr  HAC t={row['tstat']:+5.2f}  hit={row['hit']:.3f}")
else:
    print("Frozen band table:")
    for b, a, t, n in [("Extreme Fear", R['ef_ann'], R['ef_t'], R['ef_n']),
                       ("Fear", R['fear_ann'], R['fear_t'], R['fear_n']),
                       ("Neutral", R['neu_ann'], R['neu_t'], R['neu_n']),
                       ("Greed", R['greed_ann'], R['greed_t'], R['greed_n']),
                       ("Extreme Greed", R['eg_ann'], R['eg_t'], R['eg_n'])]:
        print(f"  {b:14s} n={n:4d}  {a:+7.2f}%/yr  HAC t={t:+5.2f}")
print("\\nThe contrarian claim needs Extreme Fear >> Extreme Greed. We observe the opposite tilt.")
"""),

        md("## The decisive series: the contrarian fear-minus-greed spread"),
        code("""\
if HAVE_REAL:
    s_sp = st.summarize(spread); s_a = st.annualise_weekly(s_sp)
    fe = st.long_only_fear_excess(fr); s_fe = st.summarize(fe)
    sl = st.sentiment_slope(fr)
    print("=== Primary spec: long Extreme Fear, short Extreme Greed, 1-week hold ===")
    print(f"Spread: {s_sp['mean']*10000:+.2f} bps/wk = {s_sp['mean']*52*100:+.2f}%/yr  HAC t={s_sp['tstat']:+.2f}  hit={s_sp['hit_rate']:.3f}  SR(ann)={s_a.get('sharpe_ann', float('nan')):+.2f}")
    print(f"Long-only fear excess (price-only): {s_fe['mean']*52*100:+.2f}%/yr  HAC t={s_fe['tstat']:+.2f}  n={s_fe['n']}")
    print(f"Linear sentiment slope: {sl['slope']*10000:+.2f} bps per 50 F&G pts  HAC t={sl['tstat']:+.2f}  (wrong sign for contrarian)")
    try:
        from quantlab.stats import sharpe_ci_bootstrap
        ci = sharpe_ci_bootstrap(spread, n_boot=2000, periods_per_year=52, seed=255)
        print(f"\\nBootstrap spread Sharpe 95% CI: [{ci['ci_low']:+.3f}, {ci['ci_high']:+.3f}]  ({ci['frac_negative']*100:.0f}% negative)")
    except ImportError:
        print(f"\\n[quantlab unavailable -- frozen CI: [{R['spread_ci_lo']:+.3f}, {R['spread_ci_hi']:+.3f}]]")
else:
    print(f"Frozen spread: {R['spread_ann']:+.2f}%/yr  HAC t={R['spread_t']:+.2f}  hit={R['spread_hit']:.3f}")
    print(f"Bootstrap Sharpe CI: [{R['spread_ci_lo']:+.3f}, {R['spread_ci_hi']:+.3f}]  ({R['spread_frac_neg']}% negative)")
    print(f"Long-only fear excess: {R['fear_excess_ann']:+.2f}%/yr  HAC t={R['fear_excess_t']:+.2f}")
    print(f"Sentiment slope: {R['slope_bps']:+.2f} bps/50pts  HAC t={R['slope_t']:+.2f}")
print("\\n> The spread HAC t is nowhere near the t>=2 inference bar. Bootstrap CI straddles zero.")
"""),

        md("## Shuffled-sentiment null: real signal vs random relabelling"),
        code("""\
if HAVE_REAL:
    null = st.random_timing_null(fr, n_draws=2000, seed=255)
    real_mean = spread.mean()
    pctile = (null < real_mean).mean() * 100
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(null.values * 10000, bins=40, color=GREY, alpha=0.8)
    ax.axvline(real_mean * 10000, color=RED, lw=2.0, label=f"Real spread ({real_mean*10000:+.2f} bps)")
    ax.set_xlabel("Spread mean under shuffled sentiment (bps/wk)")
    ax.set_title(f"Real spread sits at the {pctile:.1f}th percentile of the null")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("../docs/null_hist.png", dpi=120, bbox_inches="tight")
    plt.show()
    print(f"Null mean {null.mean()*10000:+.2f} bps (std {null.std()*10000:.2f}); real at {pctile:.1f}th pctile.")
else:
    print(f"Frozen: null mean {R['null_mean_bps']:+.2f} bps (std {R['null_std_bps']:.2f}); real at {R['real_pctile']:.1f}th pctile.")
print("Permuting the sentiment column gives the SAME result on average -> no genuine timing information.")
"""),

        md("## Sub-period decay -- no era rescues it"),
        code("""\
periods = [("2011-2016", "2011", "2016"), ("2017-2021", "2017", "2021"), ("2022-2026", "2022", "2026")]
if HAVE_REAL:
    rows = []
    for label, a, b in periods:
        s = st.summarize(spread[a:b])
        rows.append({"period": label, "ann": s["mean"]*52*100, "t": s["tstat"], "n": s["n"]})
        print(f"  {label}: {s['mean']*52*100:+.2f}%/yr  HAC t={s['tstat']:+.2f}  n={s['n']}")
    sts = pd.DataFrame(rows)
else:
    sts = pd.DataFrame([
        {"period": "2011-2016", "ann": R["sub_2011_ann"], "t": R["sub_2011_t"], "n": R["sub_2011_n"]},
        {"period": "2017-2021", "ann": R["sub_2017_ann"], "t": R["sub_2017_t"], "n": R["sub_2017_n"]},
        {"period": "2022-2026", "ann": R["sub_2022_ann"], "t": R["sub_2022_t"], "n": R["sub_2022_n"]},
    ])
    for _, r in sts.iterrows():
        print(f"  {r['period']}: {r['ann']:+.2f}%/yr  HAC t={r['t']:+.2f}  n={int(r['n'])}")

fig, ax = plt.subplots(figsize=(8, 4))
colors_sub = [GREEN if t >= 2 else RED if t < 0 else AMBER for t in sts["t"]]
ax.bar(sts["period"], sts["t"], color=colors_sub)
ax.axhline(2.0, color="black", ls="--", lw=0.9, label="t=2 inference bar")
ax.axhline(0, color="black", lw=0.8)
ax.set_ylabel("HAC t-stat")
ax.set_title("Contrarian spread by sub-period -- never clears t=2")
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()
"""),

        md("## Turnover and costs -- there is no edge to erode"),
        code("""\
if HAVE_REAL:
    ts = st.turnover_stats(fr)
    net = st.net_spread_returns(fr, one_way_bps=5.0, borrow_bps_ann=50.0)
    s_net = st.summarize(net)
    print(f"States: long {ts['pct_long']:.1%}, flat {ts['pct_flat']:.1%}, short {ts['pct_short']:.1%}")
    print(f"Trades: {ts['n_trades']} ({ts['trades_per_year']:.1f}/yr) -- the overlay barely trades")
    print(f"Net spread @5bps one-way + 50bps/yr borrow: {s_net['mean']*52*100:+.2f}%/yr  HAC t={s_net['tstat']:+.2f}")
else:
    print(f"States: long {R['pct_long']:.1f}%, flat {R['pct_flat']:.1f}%, short {R['pct_short']:.1f}%")
    print(f"Trades: {R['n_trades']} ({R['trades_per_year']:.1f}/yr)")
    print(f"Net spread @5bps + 50bps/yr borrow: {R['net_ann']:+.2f}%/yr  HAC t={R['net_t']:+.2f}")
print("\\nCosts are tiny because Extreme Fear/Greed are rare (~11% of weeks). The verdict is")
print("not a cost death -- there is simply no gross signal (gross t = 0.41).")
"""),

        md("## Verdict"),
        code("""\
print("=== Study 255 -- Fear-Greed ===")
print()
print(f"Signal: NONE")
print(f"  Contrarian spread HAC t = {R['spread_t']:+.2f}  (bar is t>=2)")
print(f"  Bootstrap Sharpe CI [{R['spread_ci_lo']:+.2f}, {R['spread_ci_hi']:+.2f}], {R['spread_frac_neg']}% negative -> straddles zero")
print(f"  Real spread at {R['real_pctile']:.1f}th percentile of shuffled-sentiment null")
print(f"  Band pattern is the OPPOSITE of the claim (returns rise with greed: bull-trend artefact)")
print()
print(f"Tradability: MIRAGE")
print(f"  No gross edge to trade; overlay flat {R['pct_flat']:.0f}% of weeks; even gross t = {R['spread_t']:+.2f}")
print()
print("Data caveat: curated-proxy Fear & Greed series; ^GSPC price-only (dividend-neutral spread unaffected)")
print()
print("Bottom line: None/Mirage -- a great thermometer, a useless clock.")
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
