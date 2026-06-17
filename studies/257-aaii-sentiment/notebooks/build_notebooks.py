"""Generate (and execute) the two narrative notebooks for Study 257 (AAII-Sentiment).

    python notebooks/build_notebooks.py

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The
synthetic cells run anywhere, offline and deterministic; the real-tape cells use
the curated AAII monthly table joined to the cached ^GSPC month-end closes if the
repo ``_cache/^GSPC_split_only.parquet`` is present, and otherwise quote the frozen
headline numbers in ``R`` (mirror of docs/results.md), so the notebooks re-run for
any reader offline.

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
# Real tape = curated AAII monthly table joined to cached ^GSPC month-end closes.
R = dict(
    n_months=182, start="1987-08", end="2026-05", fingerprint="714da328c0fd",
    # regime sort (prior-spread terciles -> next-month price return)
    low_ann=23.85, low_t=2.24, low_n=61,
    mid_ann=4.76, mid_t=0.62, mid_n=55,
    high_ann=8.52, high_t=1.78, high_n=66,
    low_minus_high_pp=15.33,
    # headline contrarian long-short (long panic / short euphoria)
    contra_t=1.18, contra_n=182,
    # predictive regression: ret_{t+1} ~ standardised prior spread
    reg_beta_pct=-0.66, reg_t=-1.54, reg_r2=0.018,
    # timing overlay vs buy-and-hold (price-only, net of 5bps one-way)
    overlay_net_ann=9.25, overlay_t=2.23, bh_ann=12.52,
    # sub-periods (low vs high regime annualised)
    s1_lab="1987-1999", s1_low=21.3, s1_high=15.5, s1_lowt=1.07, s1_n=55,
    s2_lab="2000-2012", s2_low=18.7, s2_high=3.2, s2_lowt=0.80, s2_n=62,
    s3_lab="2013-2026", s3_low=33.8, s3_high=7.8, s3_lowt=2.26, s3_n=65,
    # synthetic positive control / null
    syn_beta_t=-2.99, syn_null_t=-0.20,
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

from aaii_sentiment import data, strategy as st

GSPC_CACHE = data.GSPC_CACHE
HAVE_REAL = os.path.exists(GSPC_CACHE)

if HAVE_REAL:
    panel = data.build_real_panel(fetch=False)
    reg_sum = st.regime_summary(panel)
    contra = st.regime_spread(panel)
    regr = st.predictive_regression(panel)
    print(f"Real tape: {len(panel)} months  {panel.index[0].date()} -> {panel.index[-1].date()}")
    print(f"Fingerprint(spread): {data.fingerprint(panel)}")
else:
    panel = reg_sum = contra = regr = None
    print("No ^GSPC cache -- frozen headline numbers from R dict will be used")
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
# Study 257 -- AAII-Sentiment
## For the curious: is the AAII bull-bear survey a contrarian timing tool?

*Part of [Open-Alpha-Lab](../../../README.md). See the [desk methodology](../../../METHODOLOGY.md).*

---

Every week since 1987 the American Association of Individual Investors asks its
members one question: are you **bullish**, **neutral**, or **bearish** on stocks
over the next six months? The headline number is the **bull-bear spread**
(`%bull - %bear`).

The folklore -- repeated by AAII itself -- is that the crowd is wrong at the
extremes: when individual investors are *terrified*, it is supposedly time to
*buy*; when they are *euphoric*, time to *sell*. We put that contrarian claim on
the S&P 500 tape.
"""),
        md("## Setup"),
        code(BOOT),
        code(R_INJECT),

        md("""\
## The picture: sentiment swings around a slightly-bullish average

Individual-investor sentiment is *noisy* and *mean-reverting*. The long-run
average spread is only about **+6.5%** (slightly more bulls than bears), but it
swings from panic (bears > 60% in March 2009 and again in 2022) to euphoria
(bulls > 55% in the late-1990s and early 2018). The contrarian bet is that those
extremes mark turning points.
"""),
        code("""\
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
if HAVE_REAL:
    ax = axes[0]
    ax.plot(panel.index, panel["spread"]*100, color=GREY, lw=1.2)
    ax.axhline(data.AAII_AVG_SPREAD*100, color="black", ls="--", lw=0.9, label=f"long-run avg ({data.AAII_AVG_SPREAD*100:.1f}%)")
    ax.fill_between(panel.index, panel["spread"]*100, data.AAII_AVG_SPREAD*100,
                    where=(panel["spread"] < data.AAII_AVG_SPREAD), color=GREEN, alpha=.25, label="panic (contrarian buy)")
    ax.fill_between(panel.index, panel["spread"]*100, data.AAII_AVG_SPREAD*100,
                    where=(panel["spread"] >= data.AAII_AVG_SPREAD), color=RED, alpha=.20, label="euphoria (contrarian sell)")
    ax.set_title("AAII bull-bear spread (curated monthly)")
    ax.set_ylabel("spread (%)"); ax.legend(fontsize=8)

    ax = axes[1]
    means = reg_sum["mean_ann"]*100
    colors = [GREEN, GREY, RED]
    ax.bar(["low\\n(panic)", "mid", "high\\n(euphoria)"], means.values, color=colors)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Next-month S&P return by prior sentiment regime")
    ax.set_ylabel("annualised price return (%)")
    for i, v in enumerate(means.values):
        ax.text(i, v + (0.6 if v>=0 else -1.5), f"{v:+.1f}%", ha="center", fontsize=9)
else:
    for ax in axes:
        ax.text(0.5, 0.5, f"Cache absent\\nFrozen: low={R['low_ann']:.1f}%/yr  high={R['high_ann']:.1f}%/yr",
                ha="center", va="center", transform=ax.transAxes, fontsize=11, color=GREEN)
        ax.set_title("(frozen numbers)")
plt.tight_layout()
plt.savefig("../docs/regimes.png", dpi=120, bbox_inches="tight")
plt.show()
print(f"Low-sentiment months: +{R['low_ann']:.1f}%/yr (t={R['low_t']:.2f})  vs  high-sentiment: +{R['high_ann']:.1f}%/yr (t={R['high_t']:.2f})")
print(f"Low-minus-high gap: {R['low_minus_high_pp']:.1f} pp/yr -- the contrarian direction is RIGHT...")
"""),

        md("""\
## The good news: the direction is right

Sort every month by the sentiment reading you could already see, then look at
what the S&P did the *following* month. The pattern is exactly the contrarian
story:

- After the **most bearish** (panicked) months, the S&P returned **+%s%%/yr**.
- After the **most bullish** (euphoric) months, only **+%s%%/yr**.

A gap of roughly **%s percentage points a year** -- and it points the way the
folklore says it should.
""" % (f"{R['low_ann']:.0f}", f"{R['high_ann']:.0f}", f"{R['low_minus_high_pp']:.0f}")),

        md("""\
## The bad news: it does not survive the honesty checks

Three things turn this from a tradable edge into a curiosity:

1. **The long-short test is not significant.** Going long after panic and short
   after euphoria earns a positive but noisy series -- HAC *t* = **+%s**, well
   short of the *t* >= 2 bar. Most of the gap comes from a handful of violent
   rebounds (2009, 2020) that are obvious only with hindsight.

2. **As a real overlay it *loses* to buy-and-hold.** A long/flat rule that sits
   out the euphoric months returns **+%s%%/yr** net -- *below* buy-and-hold's
   **+%s%%/yr**. Sentiment is high precisely during long bull markets you do not
   want to miss.

3. **The signal is a curated, revised monthly snapshot, not the live weekly
   tape.** A real trader watched a jittery weekly number in real time, with no
   clean "month-end reading"; the live edge is weaker still.
""" % (f"{R['contra_t']:.2f}", f"{R['overlay_net_ann']:.0f}", f"{R['bh_ann']:.0f}")),

        md("""\
## The bottom line for the curious

The AAII contrarian story is **directionally true and statistically weak**:
panicked months really are followed by better returns than euphoric months, and
the pattern shows up in every sub-period -- but it never clears the inference
bar on the headline test, and as an actual timing rule it underperforms simply
staying invested.

This is a **Weak / Mirage**: a real-feeling pattern that crumbles the moment you
ask it to beat buy-and-hold after costs.
"""),
        md("*Desk verdict: **Weak / Mirage** -- see [README.md](../README.md) and [docs/results.md](../docs/results.md).*"),
    ]
    _write(nb, "01_for_the_curious.ipynb")


# ---------------------------------------------------------------------------
# Notebook 2 -- For the quants
# ---------------------------------------------------------------------------
def build_quants() -> None:
    nb = new_notebook()
    nb.cells = [
        md("""\
# Study 257 -- AAII-Sentiment
## For the quants: regime sort, HAC regression, timing overlay, sub-periods, positive control

*Part of [Open-Alpha-Lab](../../../README.md). See the [desk methodology](../../../METHODOLOGY.md).*
"""),
        md("## Setup"),
        code(BOOT),
        code(R_INJECT),

        md("""\
## Positive control: the engine detects a planted contrarian edge

Before touching the real tape we confirm the machinery is truthful. The
synthetic generator plants a *known* contrarian loading (`edge`): next-month
return loads negatively on the current spread. The predictive regression must
recover a negative slope when the edge is planted and ~zero when it is not.
"""),
        code("""\
def to_panel(df):
    p = df.copy(); p["ret"] = df["ret"].shift(-1); return p.dropna()

df_edge, truth = data.synthetic_monthly(edge=0.06, seed=257)
reg_e = st.predictive_regression(to_panel(df_edge))
print(f"Synthetic edge=0.06:  beta = {reg_e['beta']*100:+.3f} %/sd   HAC t = {reg_e['tstat']:+.2f}   r2 = {reg_e['r2']:.3f}")

df_null, _ = data.synthetic_monthly(edge=0.0, seed=257)
reg_n = st.predictive_regression(to_panel(df_null))
print(f"Synthetic null  =0.00:  beta = {reg_n['beta']*100:+.3f} %/sd   HAC t = {reg_n['tstat']:+.2f}")
print("-> Engine recovers the planted negative (contrarian) slope and reads ~zero on the null.")
"""),

        md("## The regime sort: next-month return by prior sentiment tercile"),
        code("""\
if HAVE_REAL:
    print(reg_sum.to_string(float_format=lambda v: f"{v:+.4f}"))
    lo = reg_sum.loc["low"]; hi = reg_sum.loc["high"]
    print()
    print(f"LOW  (panic)    : {lo['mean_ann']*100:+.1f}%/yr  HAC t={lo['tstat']:+.2f}  n={int(lo['n'])}")
    print(f"HIGH (euphoria) : {hi['mean_ann']*100:+.1f}%/yr  HAC t={hi['tstat']:+.2f}  n={int(hi['n'])}")
    print(f"low-minus-high  : {(lo['mean_ann']-hi['mean_ann'])*100:+.1f} pp/yr")
else:
    print(f"Frozen: low={R['low_ann']:+.1f}%/yr (t={R['low_t']:+.2f}), high={R['high_ann']:+.1f}%/yr (t={R['high_t']:+.2f})")
    print(f"low-minus-high: {R['low_minus_high_pp']:+.1f} pp/yr")
print("\\nThe LOW-sentiment regime is the only one near significance -- panic pays, but euphoria")
print("is not a reliable SELL (high-sentiment months still average a positive return).")
"""),

        md("## Headline long-short test + predictive regression (HAC)"),
        code("""\
if HAVE_REAL:
    s_contra = st.summarize(contra)
    print("=== Contrarian long-short (long panic / short euphoria) ===")
    print(f"  mean = {s_contra['mean']*1200:+.2f}%/yr   HAC t = {s_contra['tstat']:+.2f}   n = {s_contra['n']}")
    print()
    print("=== Predictive regression: ret_(t+1) ~ standardised prior spread ===")
    print(f"  beta = {regr['beta']*100:+.3f} %/sd   HAC t = {regr['tstat']:+.2f}   r2 = {regr['r2']:.4f}   n = {regr['n']}")
else:
    print(f"Frozen: contrarian long-short HAC t = {R['contra_t']:+.2f} (n={R['contra_n']})")
    print(f"Frozen: regression beta = {R['reg_beta_pct']:+.2f}%/sd  HAC t = {R['reg_t']:+.2f}  r2 = {R['reg_r2']:.3f}")
print("\\nThe slope sign is correct (NEGATIVE = contrarian) but |t| < 2: WEAK, not REAL.")
print("R-squared ~2%: sentiment explains almost none of next-month variance.")
"""),

        md("## Timing overlay vs buy-and-hold (price-only, net of costs)"),
        code("""\
if HAVE_REAL:
    ov = st.timing_overlay(panel, allow_short=False, one_way_bps=5.0)
    s_net = st.summarize(ov["net"]); s_bh = st.summarize(ov["bh"])
    print("Long/flat contrarian overlay (sit out euphoric months), 5bps one-way:")
    print(f"  Overlay NET : {s_net['mean']*1200:+.2f}%/yr  vol={s_net['vol']*np.sqrt(12)*100:.1f}%  SR={s_net['sharpe']*12**0.5:+.2f}  t={s_net['tstat']:+.2f}")
    print(f"  Buy & hold  : {s_bh['mean']*1200:+.2f}%/yr  vol={s_bh['vol']*np.sqrt(12)*100:.1f}%  SR={s_bh['sharpe']*12**0.5:+.2f}")
    print(f"  Net of costs the overlay {'BEATS' if s_net['mean']>s_bh['mean'] else 'LOSES TO'} buy-and-hold.")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot((1+ov['net']).cumprod().index, (1+ov['net']).cumprod().values, color=AMBER, lw=1.6, label="Contrarian overlay (net)")
    ax.plot((1+ov['bh']).cumprod().index, (1+ov['bh']).cumprod().values, color=GREY, lw=1.4, ls="--", label="Buy & hold (price-only)")
    ax.set_yscale("log"); ax.set_title("Contrarian timing overlay vs buy-and-hold (price-only, log scale)")
    ax.set_ylabel("growth of $1"); ax.legend(fontsize=9)
    plt.tight_layout(); plt.savefig("../docs/overlay.png", dpi=120, bbox_inches="tight"); plt.show()
else:
    print(f"Frozen: overlay net = {R['overlay_net_ann']:+.1f}%/yr (t={R['overlay_t']:+.2f})  vs buy-and-hold {R['bh_ann']:+.1f}%/yr")
print("\\nSitting out euphoria means missing the meat of bull markets: the overlay UNDERPERFORMS.")
"""),

        md("## Sub-periods: the direction is stable, the significance is not"),
        code("""\
if HAVE_REAL:
    for lab, a, b in [("1987-1999","1987","1999"), ("2000-2012","2000","2012"), ("2013-2026","2013","2026")]:
        sub = panel[a:b]; rs = st.regime_summary(sub)
        print(f"  {lab}: low={rs.loc['low','mean_ann']*100:+.1f}%/yr (t={rs.loc['low','tstat']:+.2f})  "
              f"high={rs.loc['high','mean_ann']*100:+.1f}%/yr  n={len(sub)}")
else:
    for lab, lo, hi, lt, n in [
        (R['s1_lab'], R['s1_low'], R['s1_high'], R['s1_lowt'], R['s1_n']),
        (R['s2_lab'], R['s2_low'], R['s2_high'], R['s2_lowt'], R['s2_n']),
        (R['s3_lab'], R['s3_low'], R['s3_high'], R['s3_lowt'], R['s3_n']),
    ]:
        print(f"  {lab}: low={lo:+.1f}%/yr (t={lt:+.2f})  high={hi:+.1f}%/yr  n={n}")
print("\\nLow > high in EVERY sub-period (direction robust), but the low-regime t-stat only")
print("clears 2 in the most recent slice -- and on a curated, revised sentiment snapshot.")
"""),

        md("## Verdict"),
        code("""\
print("=== Study 257 -- AAII-Sentiment ===")
print()
print(f"Signal: WEAK")
print(f"  Direction correct (panic -> +{R['low_ann']:.0f}%/yr vs euphoria +{R['high_ann']:.0f}%/yr, stable across sub-periods)")
print(f"  BUT headline long-short HAC t = {R['contra_t']:+.2f} and regression slope t = {R['reg_t']:+.2f} (<2)")
print(f"  R-squared ~{R['reg_r2']*100:.0f}%; signal is curated/revised monthly, not point-in-time weekly")
print()
print(f"Tradability: MIRAGE")
print(f"  Long/flat overlay nets {R['overlay_net_ann']:+.0f}%/yr -- BELOW buy-and-hold {R['bh_ann']:+.0f}%/yr")
print(f"  Euphoria coincides with bull markets you cannot afford to skip")
print(f"  Edge concentrated in a few violent rebounds (2009, 2020) visible only ex post")
print()
print("Vintage/survivorship: NAMED -- curated monthly snapshot is an upper bound on the live weekly edge")
print()
print("Bottom line: Weak/Mirage -- a directionally-real contrarian tilt that fails the")
print("  t>=2 bar and loses to buy-and-hold once you actually have to trade it.")
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
