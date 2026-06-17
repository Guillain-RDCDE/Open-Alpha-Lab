"""Generate the two narrative notebooks for Study 295 (Stablecoin-Supply).

    python notebooks/build_notebooks.py

Both notebooks follow the desk beats. The synthetic figures run anywhere,
offline and deterministic; the real-tape cells use the cached BTC monthly tape
under ../_cache/ joined with the curated stablecoin-supply series if present,
and otherwise quote the frozen headline numbers in ``R`` (mirroring
docs/results.md), so the notebook re-runs for any reader, online or offline.

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
R = dict(
    n_months=101, months_long=77, months_flat=24,
    btc_ann=41.2, btc_sr=0.61, btc_t=1.51, btc_dd=-73.0,
    timed_ann=49.5, timed_sr=0.85, timed_t=2.09, timed_dd=-50.0,
    exc_ann=8.3, exc_t=0.72, exc_hit=0.129,
    exc_ci_lo=-0.423, exc_ci_hi=0.899, exc_frac_neg=0.213,
    # predictive regression
    lag1_beta=0.0417, lag1_t=1.59, lag1_r2=0.047, lag1_corr=0.217, lag1_n=99,
    lag0_beta=0.0532, lag0_t=2.15, lag0_r2=0.077, lag0_corr=0.277, lag0_n=100,
    # sub-periods (lagged regression)
    sub1_label="2018-2020", sub1_beta=0.0984, sub1_t=3.05, sub1_n=34,
    sub2_label="2021-2022", sub2_beta=0.0308, sub2_t=0.75, sub2_n=22,
    sub3_label="2023-2026", sub3_beta=0.0002, sub3_t=0.01, sub3_n=39,
    # turnover / cost
    turnover_pct=8.9,
    net10_ann=49.4, net10_t=2.09, net25_ann=49.3, net25_t=2.08, net50_ann=49.0, net50_t=2.07,
    # synthetic positive control
    syn_null_t=-0.13, syn_live_t=3.48, syn_live_beta=0.0386,
    btc_fp="41dff5f08708", panel_fp="ae8e27f23d83",
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

from stablecoin_supply import data, strategy as st

CACHE_PATH = data.BTC_CACHE
HAVE_REAL = os.path.exists(CACHE_PATH)

if HAVE_REAL:
    btc = data.fetch_btc_monthly(fetch=False, cache_path=CACHE_PATH)
    panel = data.build_real_panel(btc)
    rets = st.timed_returns(panel, window=1, threshold=0.0, one_way_bps=0.0)
    excess = (rets["timed"] - rets["btc"]).dropna()
    print(f"Real panel loaded: {len(panel)} months  {panel.index[0].date()} -> {panel.index[-1].date()}")
    print(f"Long {int(rets['weight'].sum())} / flat {int((rets['weight']==0).sum())} months")
else:
    btc = panel = rets = excess = None
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
# Study 295 -- Stablecoin-Supply
## For the curious: does stablecoin supply growth fuel the next leg of Bitcoin?

*Part of [Open-Alpha-Lab](../../../README.md). See the [desk methodology](../../../METHODOLOGY.md).*

---

Stablecoins -- USDT, USDC, DAI -- are dollars parked on-chain.  The popular
**"dry powder"** thesis says this is cash waiting on the sidelines: when the
*total stablecoin supply grows*, new money has entered crypto and is about to be
deployed into Bitcoin -- so supply growth should **lead** the next leg up.

It is a seductive story.  We test it honestly: does last month's supply growth
*predict* next month's BTC return, or does supply just grow *at the same time* as
prices rise (in which case it forecasts nothing you can trade)?
"""),
        md("## Setup"),
        code(BOOT),
        code(R_INJECT),

        md("""\
## The two tapes: stablecoin supply and Bitcoin

We join a curated monthly **total stablecoin supply** series ($bn) with the
BTC-USD monthly tape.  The supply series tells the sector's story: a tiny ~$2bn
in 2018, the 2020-21 DeFi/bull mint wave to ~$180bn, the **May-2022 UST/Luna
collapse** and 2022-23 contraction, then the 2024-26 re-expansion past $250bn.
"""),
        code("""\
fig, ax1 = plt.subplots(figsize=(11, 5))
supply = data.supply_series()
ax1.plot(supply.index, supply.values, color=GREEN, lw=2, label="Stablecoin supply ($bn)")
ax1.set_ylabel("Stablecoin supply ($bn)", color=GREEN)
ax1.tick_params(axis="y", labelcolor=GREEN)

if HAVE_REAL:
    btc_px = btc.reindex(supply.index).dropna()
    ax2 = ax1.twinx()
    ax2.plot(btc_px.index, btc_px.values, color=GREY, lw=1.4, ls="--", label="BTC-USD")
    ax2.set_ylabel("BTC-USD ($)", color=GREY)
    ax2.set_yscale("log")
    ax2.grid(False)
ax1.set_title("Stablecoin supply vs Bitcoin -- they rise together (a warning sign)")
ax1.axvline(pd.Timestamp("2022-05-31"), color=RED, lw=1, ls=":", alpha=.7)
ax1.text(pd.Timestamp("2022-05-31"), supply.max()*0.5, "  UST collapse", color=RED, fontsize=9)
plt.tight_layout()
plt.savefig("../docs/supply_vs_btc.png", dpi=120, bbox_inches="tight")
plt.show()
print("Supply and BTC clearly co-move -- but co-movement is NOT a tradable forecast.")
"""),

        md("""\
## The naive timing rule looks great -- because it just rides Bitcoin

The dry-powder rule: **at each month-end, if supply grew this month, hold BTC
next month; otherwise sit in cash.**  (Strict one-month lag -- no peeking.)
"""),
        code("""\
fig, ax = plt.subplots(figsize=(11, 5))
if HAVE_REAL:
    cum_btc   = (1 + rets["btc"]).cumprod()
    cum_timed = (1 + rets["timed"]).cumprod()
    ax.plot(cum_btc.index,   cum_btc.values,   color=GREY,  lw=1.6, ls="--", label="Buy-and-hold BTC")
    ax.plot(cum_timed.index, cum_timed.values, color=GREEN, lw=1.8, label="Supply-timed (long/flat)")
    ax.set_yscale("log")
    ax.set_ylabel("Growth of $1 (log)")
    ax.legend(fontsize=10)
    ax.set_title("Supply-timed vs buy-and-hold BTC -- timed looks smoother, but...")
else:
    ax.text(0.5, 0.5, f"Cache absent\\nFrozen: timed SR={R['timed_sr']:.2f} vs BTC SR={R['btc_sr']:.2f}\\nbut excess t = +{R['exc_t']:.2f} (NOT significant)",
            ha="center", va="center", transform=ax.transAxes, fontsize=12, color=AMBER)
    ax.set_title("(frozen numbers)")
plt.tight_layout()
plt.show()
print(f"BTC   : {R['btc_ann']:+.1f}%/yr  SR={R['btc_sr']:.2f}  HAC t={R['btc_t']:+.2f}  maxDD={R['btc_dd']:.0f}%")
print(f"TIMED : {R['timed_ann']:+.1f}%/yr  SR={R['timed_sr']:.2f}  HAC t={R['timed_t']:+.2f}  maxDD={R['timed_dd']:.0f}%")
print(f"EXCESS (timed - BTC): {R['exc_ann']:+.1f}%/yr  HAC t = +{R['exc_t']:.2f}  <- NOT significant")
"""),

        md("""\
## The catch: the "edge" is just being long Bitcoin in a bull market

The timed strategy is long BTC in **77 of 101 months**.  During a once-in-a-
generation BTC bull run, *any* mostly-long rule looks good.  The honest question
is whether it beats simply **buying and holding** -- and the excess return is
**+8.3%/yr with HAC *t* = +0.72**.  That is statistically indistinguishable from
zero.  The bootstrap 95% CI on the excess Sharpe is **[-0.42, +0.90]**, with 21%
of resamples negative -- it straddles zero.
"""),

        md("""\
## Why supply growth forecasts nothing: reflexivity

Here is the heart of it.  We regress BTC's return on supply growth two ways:

- **Contemporaneous** (this month's supply growth vs this month's BTC return):
  HAC *t* = **+2.15** -- significant.  Supply and price move *together*.
- **Lagged** (last month's supply growth vs this month's BTC return -- the only
  thing you can actually trade): HAC *t* = **+1.59** -- below the t >= 2 bar.

When the *contemporaneous* link is stronger than the *lagged* one, that is the
textbook signature of a **coincident, reflexive series**: stablecoins mint when
money is already flooding in and prices are already rising.  Supply growth is a
mirror of today's enthusiasm, not a window into tomorrow's.
"""),
        code("""\
labels = ["Contemporaneous\\n(this month, NOT tradable)", "Lagged\\n(next month, tradable)"]
tvals  = [R["lag0_t"], R["lag1_t"]]
colors = [GREEN if t >= 2 else AMBER for t in tvals]
fig, ax = plt.subplots(figsize=(8.5, 4.5))
ax.bar(labels, tvals, color=colors)
ax.axhline(2.0, color="black", ls="--", lw=0.9, label="t = 2 inference bar")
ax.set_ylabel("HAC t-stat on supply-growth slope")
ax.set_title("Contemporaneous beats lagged = coincident, not leading")
ax.legend(fontsize=9)
for i, t in enumerate(tvals):
    ax.text(i, t + 0.05, f"t={t:+.2f}", ha="center", fontsize=10)
plt.tight_layout()
plt.savefig("../docs/reflexivity.png", dpi=120, bbox_inches="tight")
plt.show()
print("The tradable (lagged) signal clears no inference bar; the significant link is non-tradable.")
"""),

        md("""\
## And what little there was has decayed to nothing

Splitting the lagged regression by era shows the apparent edge was a
**2018-2020 artefact** that has since vanished -- as the stablecoin sector
matured and the dry-powder thesis became common knowledge.
"""),
        code("""\
subs = [(R["sub1_label"], R["sub1_t"]), (R["sub2_label"], R["sub2_t"]), (R["sub3_label"], R["sub3_t"])]
fig, ax = plt.subplots(figsize=(8.5, 4))
colors_s = [GREEN if t >= 2 else AMBER if t >= 1 else RED for _, t in subs]
ax.bar([s[0] for s in subs], [s[1] for s in subs], color=colors_s)
ax.axhline(2.0, color="black", ls="--", lw=0.9, label="t = 2")
ax.set_ylabel("Lagged-regression HAC t-stat")
ax.set_title("Sub-period decay: t=+3.05 (2018-20) -> t=+0.01 (2023-26)")
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()
for lbl, t in subs:
    print(f"  {lbl}: lagged HAC t = {t:+.2f}")
"""),

        md("""\
## The bottom line for the curious

The dry-powder story is **half true and entirely untradable**:

- Stablecoin supply genuinely **co-moves** with Bitcoin (contemporaneous t = +2.15).
- But the only thing you can trade -- *last* month's supply growth predicting
  *next* month's return -- clears no bar (t = +1.59), adds no significant return
  over just holding BTC (excess t = +0.72), and has **decayed to zero** since 2020.
- The pretty timed equity curve is BTC's bull-market beta in disguise.

This is a **Weak / Mirage** signal: a reflexive, coincident series dressed up as
a leading indicator.
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
# Study 295 -- Stablecoin-Supply
## For the quants: lagged vs contemporaneous HAC regressions, bootstrap CI, sub-period decay

*Part of [Open-Alpha-Lab](../../../README.md). See the [desk methodology](../../../METHODOLOGY.md).*
"""),
        md("## Setup"),
        code(BOOT),
        code(R_INJECT),

        md("## Positive control: the engine recovers a planted dry-powder premium"),
        code("""\
# Synthetic positive control: planted premium=0.04 -> lagged slope t > 2 strongly
df_live, truth = data.synthetic_panel(n_months=200, premium=0.04, seed=295)
r_live = st.predictive_regression(df_live, window=1, lag=1)
print(f"Synthetic positive control (premium=0.04):")
print(f"  Lagged slope = {r_live['beta']:+.4f}  HAC t = {r_live['tstat']:+.2f}  n = {r_live['n']}")

# Null control: premium=0.0
df_null, _ = data.synthetic_panel(n_months=200, premium=0.0, seed=295)
r_null = st.predictive_regression(df_null, window=1, lag=1)
print(f"\\nNull control (premium=0.0):")
print(f"  Lagged slope = {r_null['beta']:+.4f}  HAC t = {r_null['tstat']:+.2f}  n = {r_null['n']}")
print("  -> Engine reads strong on planted signal, ~zero on the null.")
"""),

        md("## Primary specification: supply-timed vs buy-and-hold (HAC + bootstrap)"),
        code("""\
if HAVE_REAL:
    s_btc   = st.summarize(rets["btc"])
    s_timed = st.summarize(rets["timed"])
    s_exc   = st.summarize(excess)
    print("=== Supply-timed (long when supply grew last month) vs buy-and-hold BTC ===")
    print(f"Months: {len(rets)}  (long {int(rets['weight'].sum())} / flat {int((rets['weight']==0).sum())})")
    print(f"BTC   : {s_btc['mean']*12*100:+.1f}%/yr  SR(ann)={s_btc['sharpe']*12**0.5:+.2f}  HAC t={s_btc['tstat']:+.2f}  maxDD={s_btc['max_drawdown']*100:.0f}%")
    print(f"TIMED : {s_timed['mean']*12*100:+.1f}%/yr  SR(ann)={s_timed['sharpe']*12**0.5:+.2f}  HAC t={s_timed['tstat']:+.2f}  maxDD={s_timed['max_drawdown']*100:.0f}%")
    print(f"EXCESS: {s_exc['mean']*12*100:+.1f}%/yr  HAC t={s_exc['tstat']:+.2f}  hit={s_exc['hit_rate']:.3f}")
    try:
        from quantlab.stats import sharpe_ci_bootstrap
        ci = sharpe_ci_bootstrap(excess, n_boot=2000, periods_per_year=12, seed=295)
        print(f"\\nExcess Sharpe 95% CI: [{ci['ci_low']:+.3f}, {ci['ci_high']:+.3f}]  ({ci['frac_negative']*100:.0f}% negative)")
    except ImportError:
        print(f"\\n[quantlab.stats unavailable -- frozen CI: [{R['exc_ci_lo']:+.3f}, {R['exc_ci_hi']:+.3f}]]")
else:
    print(f"Frozen: BTC {R['btc_ann']:+.1f}%/yr (t={R['btc_t']:+.2f}), timed {R['timed_ann']:+.1f}%/yr (t={R['timed_t']:+.2f})")
    print(f"Excess (timed - BTC): {R['exc_ann']:+.1f}%/yr  HAC t = +{R['exc_t']:.2f}  hit={R['exc_hit']:.3f}")
    print(f"Excess Sharpe 95% CI: [{R['exc_ci_lo']:+.3f}, {R['exc_ci_hi']:+.3f}]  ({R['exc_frac_neg']*100:.0f}% negative)")
print("\\nThe timed series beats BTC on paper but the EXCESS is not significant -- it is beta timing.")
"""),

        md("## Reflexivity: lagged (tradable) vs contemporaneous (not tradable)"),
        code("""\
if HAVE_REAL:
    r1 = st.predictive_regression(panel, window=1, lag=1)
    r0 = st.predictive_regression(panel, window=1, lag=0)
    print(f"LAGGED (t+1, tradable)          : beta={r1['beta']:+.4f}  HAC t={r1['tstat']:+.2f}  R2={r1['r2']:.4f}  corr={r1['corr']:+.3f}  n={r1['n']}")
    print(f"CONTEMPORANEOUS (t, NOT tradable): beta={r0['beta']:+.4f}  HAC t={r0['tstat']:+.2f}  R2={r0['r2']:.4f}  corr={r0['corr']:+.3f}  n={r0['n']}")
    t_lag, t_con = r1["tstat"], r0["tstat"]
else:
    print(f"Frozen LAGGED          : beta={R['lag1_beta']:+.4f}  HAC t=+{R['lag1_t']:.2f}  R2={R['lag1_r2']:.3f}  n={R['lag1_n']}")
    print(f"Frozen CONTEMPORANEOUS : beta={R['lag0_beta']:+.4f}  HAC t=+{R['lag0_t']:.2f}  R2={R['lag0_r2']:.3f}  n={R['lag0_n']}")
    t_lag, t_con = R["lag1_t"], R["lag0_t"]

fig, ax = plt.subplots(figsize=(8, 4.2))
bars = ["Contemporaneous\\n(not tradable)", "Lagged t+1\\n(tradable)"]
tv = [t_con, t_lag]
ax.bar(bars, tv, color=[GREEN if t >= 2 else AMBER for t in tv])
ax.axhline(2.0, color="black", ls="--", lw=0.9, label="t=2")
ax.set_ylabel("HAC t on supply-growth slope")
ax.set_title("Contemporaneous > lagged => coincident, reflexive series")
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()
print("Significant link is contemporaneous (non-tradable); tradable lagged link clears no bar.")
"""),

        md("## Sub-period decay of the lagged (tradable) regression"),
        code("""\
if HAVE_REAL:
    sub_stats = []
    for label, a, b in [("2018-2020","2018","2020"), ("2021-2022","2021","2022"), ("2023-2026","2023","2026")]:
        r = st.predictive_regression(panel[a:b], window=1, lag=1)
        sub_stats.append((label, r["tstat"], r["n"]))
        print(f"  {label}: beta={r['beta']:+.4f}  HAC t={r['tstat']:+.2f}  n={r['n']}")
else:
    sub_stats = [(R["sub1_label"], R["sub1_t"], R["sub1_n"]),
                 (R["sub2_label"], R["sub2_t"], R["sub2_n"]),
                 (R["sub3_label"], R["sub3_t"], R["sub3_n"])]
    for label, t, n in sub_stats:
        print(f"  {label}: HAC t={t:+.2f}  n={n}")

fig, ax = plt.subplots(figsize=(8.5, 4))
labels = [s[0] for s in sub_stats]; tvals = [s[1] for s in sub_stats]
ax.bar(labels, tvals, color=[GREEN if t >= 2 else AMBER if t >= 1 else RED for t in tvals])
ax.axhline(2.0, color="black", ls="--", lw=0.9, label="t=2")
ax.set_ylabel("Lagged-regression HAC t-stat")
ax.set_title("Post-2020 decay: the dry-powder edge is gone")
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()
print("The lagged predictive power was a 2018-2020 artefact; it has decayed to t~0.")
"""),

        md("## Turnover and cost drag -- not the binding constraint"),
        code("""\
if HAVE_REAL:
    to = st.turnover(panel)
    print(f"Avg monthly switch rate: {to:.2%}")
    for bps in (10, 25, 50):
        rn = st.timed_returns(panel, one_way_bps=bps)
        sn = st.summarize(rn["timed_net"])
        print(f"  Net timed @{bps}bps one-way: {sn['mean']*12*100:+.1f}%/yr  HAC t={sn['tstat']:+.2f}")
else:
    print(f"Frozen: switch rate {R['turnover_pct']:.1f}%/mo")
    print(f"  Net @10bps: {R['net10_ann']:+.1f}%/yr  t={R['net10_t']:+.2f}")
    print(f"  Net @25bps: {R['net25_ann']:+.1f}%/yr  t={R['net25_t']:+.2f}")
    print(f"  Net @50bps: {R['net50_ann']:+.1f}%/yr  t={R['net50_t']:+.2f}")
print("\\nCosts barely dent the timed series (supply grows most months -> few switches).")
print("The binding constraint is the ABSENCE of a tradable signal, not transaction cost.")
"""),

        md("## Verdict"),
        code("""\
print("=== Study 295 -- Stablecoin-Supply ===")
print()
print(f"Signal: WEAK")
print(f"  Tradable lagged slope HAC t = +{R['lag1_t']:.2f}  (< 2 inference bar)")
print(f"  Significant link is CONTEMPORANEOUS (t = +{R['lag0_t']:.2f}) = reflexive, non-tradable")
print(f"  Decayed: t = +{R['sub1_t']:.2f} (2018-20) -> +{R['sub3_t']:.2f} (2023-26)")
print(f"  Synthetic positive control confirms the engine works (planted t = +{R['syn_live_t']:.2f})")
print()
print(f"Tradability: MIRAGE")
print(f"  Timed beats BTC on paper (SR {R['timed_sr']:.2f} vs {R['btc_sr']:.2f}) but")
print(f"  EXCESS over buy-and-hold HAC t = +{R['exc_t']:.2f}, bootstrap CI [{R['exc_ci_lo']:+.2f}, {R['exc_ci_hi']:+.2f}] straddles zero")
print(f"  It is BTC beta-timing in a single bull market, not alpha")
print()
print("Reflexivity / data caveat: NAMED -- coincident series; curated public-aggregate supply;")
print("  single-survivor BTC spot (price-only == total-return); strict one-month lag.")
print()
print("Bottom line: Weak/Mirage -- a reflexive, coincident series masquerading as a leading signal.")
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
