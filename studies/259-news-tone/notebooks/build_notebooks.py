"""Generate the two narrative notebooks for Study 259 (News-Tone).

    python notebooks/build_notebooks.py

This script writes BOTH notebooks and then executes them in place with nbconvert.
The synthetic figures run anywhere, offline and deterministic; the real-tape cells
use the cached ^GSPC series under ../_cache/ if present and otherwise quote the
frozen headline numbers in ``R`` (mirroring docs/results.md), so the notebook
re-runs for any reader.

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
    n_days=4780, win_start="2007-01-03", win_end="2025-12-31", fp="c3d8b303cf23",
    # naive (hindsight-curated) next-day regression
    nd_beta=0.00209, nd_t=5.74, nd_r2=0.0169,
    sd_t=5.42,  # same-day mechanical fit
    contemp_corr=0.125,
    # naive sign-timing strategy (the mirage)
    sign_ann=35.0, sign_t=9.49, sign_sr=1.78, sign_hit=0.554,
    sign_net_t=9.46, sign_net_ann=34.9,
    tilt_ann=14.6, tilt_t=6.18, tilt_sr=1.11,
    buyhold_ann=10.3, buyhold_t=2.65, buyhold_sr=0.52,
    # the two killers
    demean_t=0.00, demean_r2=0.000,
    priormonth_ann=3.8, priormonth_t=0.97, priormonth_sr=0.19,
    # sub-periods of the naive (mirage) sign strat
    sub1_ann=33.9, sub1_t=4.11, sub2_ann=25.7, sub2_t=5.69, sub3_ann=43.9, sub3_t=7.00,
    # synthetic controls
    syn_null_t=0.90, syn_null_sr=0.31, syn_g2_t=7.29, syn_g2_sr=2.59,
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

from news_tone import data, strategy as st

CACHE_PATH = data.GSPC_CACHE
HAVE_REAL = os.path.exists(CACHE_PATH)

if HAVE_REAL:
    df_real = data.build_real_panel(fetch=False)
    print(f"Real panel loaded: {len(df_real)} trading days "
          f"({df_real.index[0].date()} -> {df_real.index[-1].date()})")
else:
    df_real = None
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
# Study 259 -- News-Tone
## For the curious: does today's news mood move *tomorrow's* tape?

*Part of [Open-Alpha-Lab](../../../README.md). See the [desk methodology](../../../METHODOLOGY.md).*

---

Every morning the financial press has a *mood*. Some days the headlines drip
risk-off dread; other days everything is a melt-up. It is tempting to believe
that if you could distil that mood into a single **news-tone** score, you could
front-run the market: buy when the news is good, sell when it is grim.

We build a curated daily news-tone proxy (a z-scored read of the macro headline
flow, deeply negative in 2008 and March-2020, positive in the 2017 and 2021
melt-ups) and ask the only question that matters for a trader: **does today's
tone forecast *tomorrow's* S&P 500 return?**
"""),
        md("## Setup"),
        code(BOOT),
        code(R_INJECT),

        md("""\
## The seductive chart: tone lines up with the market

First, the chart that sells the idea. Plot the news-tone proxy against the
market. When tone is negative, the market is falling; when tone is positive, it
is climbing. Looks like a money machine.

**But watch the trap**: this is *same-day* comovement. Of course bad-news days
are down days -- the news *describes* the move; it does not *precede* it. The
contemporaneous correlation is mechanical, not tradable.
"""),
        code("""\
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

if HAVE_REAL:
    ax = axes[0]
    cum = (1 + df_real["ret"]).cumprod()
    ax.plot(cum.index, cum.values, color=GREY, lw=1.2, label="^GSPC (growth of $1)")
    ax.set_ylabel("Cumulative return ($)")
    ax2 = ax.twinx()
    ax2.fill_between(df_real.index, df_real["tone"].values, 0,
                     color=GREEN, alpha=.25, label="News tone")
    ax2.set_ylabel("News-tone proxy (z)")
    ax.set_title("News tone vs the market (same-day, seductive)")
    ax.legend(loc="upper left", fontsize=9)

    ax = axes[1]
    ax.scatter(df_real["tone"], df_real["ret"] * 100, s=4, color=GREY, alpha=.3)
    ax.set_xlabel("Today's tone (z)"); ax.set_ylabel("Today's return (%)")
    corr = df_real["tone"].corr(df_real["ret"])
    ax.set_title(f"SAME-DAY: corr = {corr:+.3f}  (mechanical)")
else:
    for ax in axes:
        ax.text(0.5, 0.5, f"Cache absent\\nFrozen contemporaneous corr = {R['contemp_corr']:+.3f}",
                ha="center", va="center", transform=ax.transAxes, fontsize=12, color=GREY)
        ax.set_title("(frozen numbers)")

plt.tight_layout()
plt.savefig("../docs/tone_vs_market.png", dpi=120, bbox_inches="tight")
plt.show()
print(f"Same-day tone~return correlation: {R['contemp_corr']:+.3f} -- positive but MECHANICAL.")
print("Bad-news days ARE down days. This is not a forecast; it is a description.")
"""),

        md("""\
## The naive next-day strategy looks fantastic -- and that is the warning sign

Now the trade: at today's close, go **long** tomorrow if today's tone is
positive, **short** if it is negative. Hold one day. Repeat.

On the curated proxy this prints a glorious **+35%/yr at *t* ~ 9.5, Sharpe
~1.8** -- crushing the +10%/yr buy-and-hold. If that sounds too good to be a
free lunch off a public-mood index, it is.
"""),
        code("""\
fig, ax = plt.subplots(figsize=(10, 5))

if HAVE_REAL:
    sig = st.tone_sign_signal(df_real)
    gross = st.timing_returns(df_real, sig)
    s = st.summarize(gross)
    eq = (1 + gross).cumprod()
    bh = (1 + df_real.loc[gross.index, "fwd_ret"]).cumprod()
    ax.plot(eq.index, eq.values, color=GREEN, lw=1.8, label=f"Tone sign-timing (SR={s['sharpe_ann']:.2f})")
    ax.plot(bh.index, bh.values, color=GREY, lw=1.2, ls="--", label="Buy & hold ^GSPC")
    ax.set_yscale("log"); ax.set_ylabel("Growth of $1 (log)")
    ax.set_title("Naive tone sign-timing vs buy-and-hold")
    ax.legend(fontsize=9)
    print(f"Naive sign-timing: {s['mean_ann']*100:+.1f}%/yr  HAC t={s['tstat']:+.2f}  SR={s['sharpe_ann']:.2f}  hit={s['hit_rate']:.3f}")
else:
    ax.text(0.5, 0.5, f"Cache absent\\nFrozen: {R['sign_ann']:+.1f}%/yr  t={R['sign_t']:+.2f}  SR={R['sign_sr']:.2f}",
            ha="center", va="center", transform=ax.transAxes, fontsize=12, color=GREEN)
    ax.set_title("(frozen numbers)")

plt.tight_layout()
plt.savefig("../docs/equity_curve.png", dpi=120, bbox_inches="tight")
plt.show()
print(f"Buy & hold for comparison: {R['buyhold_ann']:+.1f}%/yr  SR={R['buyhold_sr']:.2f}")
"""),

        md("""\
## The two tests that demolish it

A *t* of 9.5 is far too clean. Two honest checks expose the mirage.

**1. Demean within the month.** Our tone proxy is a *monthly* read, held
constant across each month -- and it was hand-curated *knowing* which months
were good or bad. So "tone predicts tomorrow" is really just "this whole month
drifts up/down, and every day inside it inherits that drift." Subtract each
month's own average return and the predictability vanishes to **exactly zero**.

**2. Use last month's tone instead of this month's.** A real trader at the start
of a month does *not* know the month's eventual mood. If we lag the tone by one
month (only information available *before* the month begins), the sign strategy
collapses to **+3.8%/yr at *t* ~ 1.0, Sharpe 0.19** -- statistically
indistinguishable from the market's drift.
"""),
        code("""\
labels = ["Naive\\n(hindsight tone)", "Within-month\\ndemeaned", "Prior-month\\ntone (honest)"]

if HAVE_REAL:
    # naive next-day regression
    nd = st.predictive_regression(df_real, target_col="fwd_ret")
    # within-month demeaned target
    d2 = df_real.copy()
    d2["ym"] = d2.index.to_period("M")
    d2["fwd_dm"] = d2["fwd_ret"] - d2.groupby("ym")["fwd_ret"].transform("mean")
    dm = st.predictive_regression(d2[["tone", "fwd_dm"]].rename(columns={"fwd_dm": "fwd_ret"}),
                                  target_col="fwd_ret")
    # prior-month tone sign strategy
    monthly = data.news_tone_monthly().shift(1)
    keys = df_real.index + pd.offsets.MonthEnd(0)
    d3 = df_real.copy()
    d3["tone_lag"] = monthly.reindex(keys).to_numpy()
    d3 = d3.dropna(subset=["tone_lag", "fwd_ret"])
    g_lag = (np.sign(d3["tone_lag"]) * d3["fwd_ret"]).dropna()
    s_lag = st.summarize(g_lag)
    tstats = [nd["tstat"], dm["tstat"], s_lag["tstat"]]
    print(f"Naive next-day reg t           = {nd['tstat']:+.2f}  (r2={nd['r2']:.4f})")
    print(f"Within-month demeaned reg t    = {dm['tstat']:+.2f}  (r2={dm['r2']:.6f})  <- collapses to ZERO")
    print(f"Prior-month sign strategy t    = {s_lag['tstat']:+.2f}  (ann={s_lag['mean_ann']*100:+.1f}%, SR={s_lag['sharpe_ann']:.2f})  <- no edge")
else:
    tstats = [R["nd_t"], R["demean_t"], R["priormonth_t"]]
    print(f"Naive next-day reg t        = {R['nd_t']:+.2f}")
    print(f"Within-month demeaned t     = {R['demean_t']:+.2f}  <- ZERO")
    print(f"Prior-month sign strategy t = {R['priormonth_t']:+.2f}  <- no edge")

fig, ax = plt.subplots(figsize=(8.5, 4.5))
colors = [GREEN if t >= 2 else RED for t in tstats]
ax.bar(labels, tstats, color=colors)
ax.axhline(2.0, color="black", ls="--", lw=0.9, label="t = 2 inference bar")
ax.axhline(0, color="black", lw=0.8)
ax.set_ylabel("HAC t-stat (next-day)")
ax.set_title("The edge evaporates the moment hindsight is removed")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("../docs/honesty_bars.png", dpi=120, bbox_inches="tight")
plt.show()
"""),

        md("""\
## The bottom line for the curious

The news-tone story is a **mirage**:

- The eye-catching same-day correlation is *mechanical* -- the news describes
  the move, it does not predict it.
- The eye-catching next-day strategy (*t* ~ 9.5) is built on a **hindsight-
  curated** monthly proxy. Demean within the month and the signal is **exactly
  zero**. The strategy was only ever harvesting each month's known drift.
- Give a real trader only the information they would actually have had
  (last month's tone), and the edge falls to **Sharpe 0.19, *t* ~ 1.0** --
  no better than buying and holding.

Aggregate news tone does **not** move the next day's tape in any way you could
trade. This is a **None / Mirage**: no real day-ahead signal, and the apparent
one dissolves under the lightest honesty check.
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
# Study 259 -- News-Tone
## For the quants: predictive regressions, the within-month placebo, lagged-tone tradability

*Part of [Open-Alpha-Lab](../../../README.md). See the [desk methodology](../../../METHODOLOGY.md).*
"""),
        md("## Setup"),
        code(BOOT),
        code(R_INJECT),

        md("""\
## Positive control: the engine finds a signal when one is planted

Before trusting any real-tape number we confirm the machinery is truthful on the
deterministic synthetic tape (`data.synthetic_daily`). The DGP plants a linear
link `ret_{t+1} = gamma * z(tone_t) + noise`. At `gamma = 0` the tone is a
persistent-but-useless AR(1) series (the null); at `gamma > 0` the next-day
predictability is real.

> 💡 **In plain words:** we feed the code data where we *know* the answer. On
> pure-noise tone it must read ~zero; with a planted link it must light up. Only
> then do we believe what it says about the real market.
"""),
        code("""\
print("Synthetic control (n_days=2000, seed=259):")
for gamma in [0.0, 0.002, 0.004]:
    d, _ = data.synthetic_daily(n_days=2000, gamma=gamma, seed=259)
    reg = st.predictive_regression(d, target_col="fwd_ret")
    sig = st.tone_sign_signal(d)
    s = st.summarize(st.timing_returns(d, sig))
    tag = "NULL " if gamma == 0 else "planted"
    print(f"  gamma={gamma:.3f} [{tag}]: reg t={reg['tstat']:+.2f}  r2={reg['r2']:.4f}  | sign SR={s['sharpe_ann']:.2f}  t={s['tstat']:+.2f}")
print("\\n-> Null reads ~zero (t<1); planted gamma lights up monotonically. Engine is truthful.")
"""),

        md("""\
## Same-day vs next-day: the mechanical-vs-tradable gap

Two regressions of return on tone z-score. The *same-day* fit (`ret`) is
mechanical -- bad-news days are down days. The *next-day* fit (`fwd_ret`) is the
only one a trader could act on. On the curated proxy *both* look significant,
which is the first red flag: a low-frequency hindsight tone leaks its forward
drift.
"""),
        code("""\
if HAVE_REAL:
    sd = st.predictive_regression(df_real, target_col="ret")
    nd = st.predictive_regression(df_real, target_col="fwd_ret")
    print("SAME-DAY  (ret ~ tone):     beta={:+.5f}  HAC t={:+.2f}  r2={:.4f}  n={}".format(sd["beta"], sd["tstat"], sd["r2"], sd["n"]))
    print("NEXT-DAY  (fwd_ret ~ tone): beta={:+.5f}  HAC t={:+.2f}  r2={:.4f}  n={}".format(nd["beta"], nd["tstat"], nd["r2"], nd["n"]))
else:
    print(f"SAME-DAY  HAC t = {R['sd_t']:+.2f}")
    print(f"NEXT-DAY  HAC t = {R['nd_t']:+.2f}  (beta={R['nd_beta']:+.5f}, r2={R['nd_r2']:.4f})")
print("\\nBoth significant -> suspicious. A genuine forward signal should NOT")
print("be roughly as strong as the mechanical contemporaneous one.")
"""),

        md("""\
## The within-month placebo: where the t-stat goes to die

The tone proxy is constant within each calendar month and was curated *knowing*
each month's outcome. So the regression is really exploiting the month-level
mean. Subtract each month's own mean forward return and re-run: if any genuine
*day-to-day* information existed, the slope would survive. It does not.
"""),
        code("""\
if HAVE_REAL:
    d2 = df_real.copy()
    d2["ym"] = d2.index.to_period("M")
    d2["fwd_dm"] = d2["fwd_ret"] - d2.groupby("ym")["fwd_ret"].transform("mean")
    dm = st.predictive_regression(d2[["tone", "fwd_dm"]].rename(columns={"fwd_dm": "fwd_ret"}),
                                  target_col="fwd_ret")
    print(f"Within-month-demeaned next-day reg: beta={dm['beta']:+.8f}  HAC t={dm['tstat']:+.2f}  r2={dm['r2']:.8f}")
else:
    print(f"Within-month-demeaned next-day reg: HAC t = {R['demean_t']:+.2f}  r2 = {R['demean_r2']:.3f}")
print("\\n-> EXACTLY ZERO. 100% of the apparent edge was the month-level drift")
print("   that the hindsight-curated proxy already 'knew'. No daily forecast power.")
"""),

        md("""\
## The honest tradable: lag the tone by a month

A trader on the first day of a month does not know that month's mood. Replace
this month's tone with *last* month's (knowable in advance) and run the sign
strategy. This is the only version that respects the information set.
"""),
        code("""\
if HAVE_REAL:
    monthly_lag = data.news_tone_monthly().shift(1)
    keys = df_real.index + pd.offsets.MonthEnd(0)
    d3 = df_real.copy()
    d3["tone_lag"] = monthly_lag.reindex(keys).to_numpy()
    d3 = d3.dropna(subset=["tone_lag", "fwd_ret"])
    g_lag = (np.sign(d3["tone_lag"]) * d3["fwd_ret"]).dropna()
    s_lag = st.summarize(g_lag)
    bh = st.summarize(df_real["fwd_ret"])
    print(f"Prior-month sign-timing : {s_lag['mean_ann']*100:+.1f}%/yr  HAC t={s_lag['tstat']:+.2f}  SR={s_lag['sharpe_ann']:.2f}  hit={s_lag['hit_rate']:.3f}")
    print(f"Buy & hold benchmark    : {bh['mean_ann']*100:+.1f}%/yr  HAC t={bh['tstat']:+.2f}  SR={bh['sharpe_ann']:.2f}")
else:
    print(f"Prior-month sign-timing : {R['priormonth_ann']:+.1f}%/yr  HAC t={R['priormonth_t']:+.2f}  SR={R['priormonth_sr']:.2f}")
    print(f"Buy & hold benchmark    : {R['buyhold_ann']:+.1f}%/yr  HAC t={R['buyhold_t']:+.2f}  SR={R['buyhold_sr']:.2f}")
print("\\n-> The honest version is no better than the market's drift. t < 2. No edge.")
"""),

        md("""\
## Costs: even the mirage is cost-cheap (because it barely trades)

For completeness, the naive sign-timing strategy nets almost nothing in costs --
the monthly tone flips sign only a handful of times per year, so turnover is
tiny. Cost is *not* what kills this idea; **hindsight** is. We charge a one-way
cost on |position change| x NAV, with short days paying a daily borrow.
"""),
        code("""\
if HAVE_REAL:
    sig = st.tone_sign_signal(df_real)
    gross = st.timing_returns(df_real, sig)
    net = st.net_of_costs(sig, gross, one_way_bps=1.0, borrow_bps_per_day=0.04)
    sg, sn = st.summarize(gross), st.summarize(net)
    flips = int(sig.diff().abs().gt(0).sum())
    print(f"Naive sign-timing  gross: {sg['mean_ann']*100:+.1f}%/yr  t={sg['tstat']:+.2f}")
    print(f"Naive sign-timing  net  : {sn['mean_ann']*100:+.1f}%/yr  t={sn['tstat']:+.2f}  (sign flips: {flips})")
    print("\\nGross ~ net: turnover is trivial. The +35%/yr is real in-sample -- and")
    print("entirely a hindsight artifact, as the demean and lag tests proved.")
else:
    print(f"Naive sign-timing gross: {R['sign_ann']:+.1f}%/yr t={R['sign_t']:+.2f}")
    print(f"Naive sign-timing net  : {R['sign_net_ann']:+.1f}%/yr t={R['sign_net_t']:+.2f}")
"""),

        md("## Sub-periods of the naive (mirage) strategy: stable *because* it is an artifact"),
        code("""\
if HAVE_REAL:
    sig = st.tone_sign_signal(df_real)
    gross = st.timing_returns(df_real, sig)
    for a, b in [("2007", "2012"), ("2013", "2018"), ("2019", "2025")]:
        s = st.summarize(gross[a:b])
        print(f"  {a}-{b}: {s['mean_ann']*100:+.1f}%/yr  HAC t={s['tstat']:+.2f}  n={s['n']}")
else:
    print(f"  2007-2012: {R['sub1_ann']:+.1f}%/yr  t={R['sub1_t']:+.2f}")
    print(f"  2013-2018: {R['sub2_ann']:+.1f}%/yr  t={R['sub2_t']:+.2f}")
    print(f"  2019-2025: {R['sub3_ann']:+.1f}%/yr  t={R['sub3_t']:+.2f}")
print("\\nUniformly 'significant' across sub-periods -- exactly what a hindsight")
print("artifact looks like. Stability here is a symptom, not a virtue.")
"""),

        md("## Verdict"),
        code("""\
print("=== Study 259 -- News-Tone ===")
print()
print("Signal: NONE")
print(f"  Naive next-day reg t = +{R['nd_t']:.2f} looks strong, BUT:")
print(f"    - within-month demeaned t = {R['demean_t']:+.2f}  (the apparent edge is 100% month-drift)")
print(f"    - prior-month (honest) sign strat t = {R['priormonth_t']:+.2f}, SR = {R['priormonth_sr']:.2f}  (no edge)")
print("    - tone proxy is hindsight-curated => upper bound, and even that dissolves")
print()
print("Tradability: MIRAGE")
print("  The only 'profitable' version uses information no trader had in real time.")
print("  Honest, information-respecting version is indistinguishable from buy-and-hold.")
print()
print("Proxy/survivorship: NAMED -- curated tone can only OVER-state predictability.")
print()
print("Bottom line: None/Mirage -- aggregate news tone does not move the next day's")
print("  tape in any tradable way. The headline t-stat is a hindsight artifact.")
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
    import subprocess
    import sys
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
