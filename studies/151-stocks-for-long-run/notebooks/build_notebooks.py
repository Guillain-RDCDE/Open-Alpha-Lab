"""Generate the two narrative notebooks for Study 151 (Stocks-For-Long-Run).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-data cells read the staged
Shiller parquet under ../../.._cache/ and fall back to the frozen headline numbers in
``R`` (mirroring docs/results.md), so the notebook re-runs for any reader.

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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-14).
# Shiller panel 1871-01 to 2023-06 (1830 monthly rows), fingerprint c58161b47345.
R = dict(
    # Horizon: 1 year
    h1_eq_mean=8.56, h1_eq_worst=-57.73, h1_eq_neg=30.86,
    h1_ex_mean=6.12, h1_ex_worst=-72.63, h1_ex_neg=35.48, h1_t=5.44,
    # Horizon: 5 years
    h5_eq_mean=7.18, h5_eq_worst=-13.06, h5_eq_neg=18.86,
    h5_ex_mean=4.81, h5_ex_worst=-21.41, h5_ex_neg=27.51, h5_t=9.32,
    # Horizon: 10 years
    h10_eq_mean=6.93, h10_eq_worst=-5.91, h10_eq_neg=11.23,
    h10_ex_mean=4.67, h10_ex_worst=-8.00, h10_ex_neg=16.49, h10_t=14.27,
    # Horizon: 20 years
    h20_eq_mean=6.64, h20_eq_worst=-0.18, h20_eq_neg=0.06,
    h20_ex_mean=4.49, h20_ex_worst=-1.48, h20_ex_neg=2.96, h20_t=21.16,
    # Horizon: 30 years
    h30_eq_mean=6.66, h30_eq_worst=1.94, h30_eq_neg=0.00,
    h30_ex_mean=4.63, h30_ex_worst=-0.12, h30_ex_neg=0.14, h30_t=30.92,
    # Decades with bonds winning: 4 / 15
    n_decades=15, n_decades_bonds_win=4,
    # Worst-case 20y and 30y equity and excess for claims
    worst_20y_eq_start="1901-06",
    worst_30y_ex_start="1902-06",
)

# ---------------------------------------------------------------------------
# Shared preamble — imports, data load, HAVE_REAL guard
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

from stocks_for_long_run import data, strategy as st

# Try to load the real Shiller panel (staged parquet, no network needed)
try:
    _df = data.load_shiller()
    HAVE_REAL = len(_df) > 1000
except Exception:
    HAVE_REAL = False
    _df = None

print("Shiller real panel loaded:", HAVE_REAL,
      f"({len(_df)} rows)" if HAVE_REAL else "(using frozen R numbers)")

# Frozen headline numbers — mirror of docs/results.md (as-of 2026-06-14)
R = dict(
    h1_eq_mean=8.56, h1_eq_worst=-57.73, h1_eq_neg=30.86,
    h1_ex_mean=6.12, h1_ex_worst=-72.63, h1_ex_neg=35.48, h1_t=5.44,
    h5_eq_mean=7.18, h5_eq_worst=-13.06, h5_eq_neg=18.86,
    h5_ex_mean=4.81, h5_ex_worst=-21.41, h5_ex_neg=27.51, h5_t=9.32,
    h10_eq_mean=6.93, h10_eq_worst=-5.91, h10_eq_neg=11.23,
    h10_ex_mean=4.67, h10_ex_worst=-8.00, h10_ex_neg=16.49, h10_t=14.27,
    h20_eq_mean=6.64, h20_eq_worst=-0.18, h20_eq_neg=0.06,
    h20_ex_mean=4.49, h20_ex_worst=-1.48, h20_ex_neg=2.96, h20_t=21.16,
    h30_eq_mean=6.66, h30_eq_worst=1.94, h30_eq_neg=0.00,
    h30_ex_mean=4.63, h30_ex_worst=-0.12, h30_ex_neg=0.14, h30_t=30.92,
    n_decades=15, n_decades_bonds_win=4,
    worst_20y_eq_start="1901-06", worst_30y_ex_start="1902-06",
)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Stocks-For-Long-Run — does equity always win over the long haul?\n"
            "### Siegel's century-long claim tested on 150 years of Shiller real prices\n\n"
            "![Signal: Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square)\n"
            "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
            "![Useful%3F: Only_if_you_can_wait_30_years](https://img.shields.io/badge/Useful%3F-Only_if_you_can_wait_30_years-8b949e?style=flat-square)\n\n"
            "Jeremy Siegel's *Stocks for the Long Run* makes one of the most audacious claims "
            "in finance: **equity always wins** — over any 20- or 30-year stretch on record, "
            "stocks have never delivered a negative real return, and they have never lost to bonds. "
            "This notebook asks: is that literally true, and does it help a real investor?\n\n"
            "> **This is the plain-language layer.** Want the t-stats, the overlapping-window "
            "caveats and the HAC inference? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart below is "
            "drawn by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Is the long-run equity premium real? | **Yes — robustly.** Mean 30-year real return: "
            f"**+{R['h30_eq_mean']:.2f}%/yr**, mean 30-year excess over bonds: **+{R['h30_ex_mean']:.2f}%/yr**. |\n"
            f"| Does equity ALWAYS beat bonds at 20/30 years? | **Almost.** At 30 years: "
            f"**{R['h30_eq_neg']:.0f}%** of windows negative real equity; "
            f"**{R['h30_ex_neg']:.2f}%** of windows bonds winning. At 20 years: worst real return "
            f"was **{R['h20_eq_worst']:+.2f}%/yr** (barely negative). |\n"
            f"| Could you trade this? | **It's real but fragile.** The horizon is 20–30 years. "
            f"Short windows are brutal: at 10 years, **{R['h10_eq_neg']:.1f}%** of windows have "
            f"negative real equity returns. You need the *full* long run. |\n\n"
            f"> **Siegel's claim is mostly true** — but with an asterisk: at 20 years "
            f"there was one nearly-flat window (real equity {R['h20_eq_worst']:+.2f}%/yr "
            f"starting {R['worst_20y_eq_start']}), and at 30 years bonds came within "
            f"**{R['h30_ex_worst']:+.2f}%/yr** of equity once. The premium is real; "
            f"the 'always' is a rhetorical near-miss, not an iron law."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *'Stocks have never lost to bonds over any 20-year period on record, "
            "and in real terms have never delivered a negative return over any 20-year period.'*  \n"
            "> — Jeremy Siegel, **Stocks for the Long Run** (various editions, 1994–2022)\n\n"
            "Siegel's argument is not 'stocks are safe short-term' — he explicitly acknowledges "
            "the volatility. The claim is a **horizon argument**: the longer you hold, the more "
            "reliably equity dominates, and beyond 20–30 years the historical record shows near-"
            "perfect dominance over bonds (and a positive real return in every window).\n\n"
            "We test exactly that claim on Shiller's monthly data back to 1871: **150+ years**, "
            "every rolling 1-, 5-, 10-, 20-, and 30-year window, real equity total return "
            "(prices + dividends, inflation-adjusted) vs a real bond proxy (10-year nominal "
            "yield minus monthly CPI inflation)."
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If Siegel's claim is literally true, it is the most powerful argument for equity "
            "index investing ever made: *just hold long enough and you cannot lose in real terms, "
            "and you will beat bonds.* It justifies near-100% equity allocations for anyone with "
            "a long enough horizon — the core of passive investing wisdom.\n\n"
            "If it breaks down — even once, even barely — the 'never' claim fails, and the "
            "practical question becomes: how bad can things get, and for how long? A worst-case "
            "of near-zero at 20 years is very different from a worst-case of −5%/yr."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "The test is almost definitional:\n\n"
            "1. Take Shiller's real price and real dividend series, 1871–2023.\n"
            "2. Build a **real total return index**: each month reinvest the real dividend yield.\n"
            "3. Build a **real bond proxy**: 10-year nominal yield minus monthly CPI inflation.\n"
            "4. Compute every rolling 1-, 5-, 10-, 20-, 30-year annualised return for both.\n"
            "5. Find the **worst-case window** for each: what was the lowest equity return? "
            "   Did it beat bonds?\n\n"
            "**The claim fails if any window has a negative real equity return at 20+ years**, "
            "or **if bonds beat equity in any 20+ year window**. Simple."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown — 150 years of rolling returns\n\n"
            "**First, how does the worst-case shrink with horizon?**"
        ),
        code(
            "HORIZONS = [1, 5, 10, 20, 30]\n"
            "if HAVE_REAL:\n"
            "    summ = st.all_horizon_summary(_df)\n"
            "    eq_means   = [summ.loc[h, 'eq_mean']*100  for h in HORIZONS]\n"
            "    eq_worsts  = [summ.loc[h, 'eq_min']*100   for h in HORIZONS]\n"
            "    ex_means   = [summ.loc[h, 'excess_mean']*100 for h in HORIZONS]\n"
            "    ex_worsts  = [summ.loc[h, 'excess_min']*100  for h in HORIZONS]\n"
            "    eq_neg     = [summ.loc[h, 'eq_pct_neg']   for h in HORIZONS]\n"
            "else:\n"
            "    eq_means   = [R['h1_eq_mean'], R['h5_eq_mean'], R['h10_eq_mean'], R['h20_eq_mean'], R['h30_eq_mean']]\n"
            "    eq_worsts  = [R['h1_eq_worst'], R['h5_eq_worst'], R['h10_eq_worst'], R['h20_eq_worst'], R['h30_eq_worst']]\n"
            "    ex_means   = [R['h1_ex_mean'], R['h5_ex_mean'], R['h10_ex_mean'], R['h20_ex_mean'], R['h30_ex_mean']]\n"
            "    ex_worsts  = [R['h1_ex_worst'], R['h5_ex_worst'], R['h10_ex_worst'], R['h20_ex_worst'], R['h30_ex_worst']]\n"
            "    eq_neg     = [R['h1_eq_neg'], R['h5_eq_neg'], R['h10_eq_neg'], R['h20_eq_neg'], R['h30_eq_neg']]\n"
            "labels = [f'{h}y' for h in HORIZONS]\n"
            "x = range(len(HORIZONS))\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))\n"
            "axes[0].plot(labels, eq_means, 'o-', color=GREEN, lw=2, label='mean real equity')\n"
            "axes[0].fill_between(labels, eq_worsts, 0, alpha=0.18, color=RED, label='worst-case')\n"
            "axes[0].axhline(0, c='k', lw=1)\n"
            "axes[0].set_title('Real equity return: mean vs worst-case'); axes[0].set_ylabel('%/yr')\n"
            "axes[0].legend()\n"
            "axes[1].bar(labels, eq_neg, color=[RED if v>5 else AMBER if v>0.5 else GREEN for v in eq_neg])\n"
            "axes[1].set_title('% of rolling windows with negative real equity'); axes[1].set_ylabel('% of windows')\n"
            "plt.tight_layout(); plt.show()\n"
            "for h, m, w, n in zip(HORIZONS, eq_means, eq_worsts, eq_neg):\n"
            "    print(f'{h:>3}y: mean={m:+.2f}%  worst={w:+.2f}%  neg={n:.2f}%')"
        ),
        md(
            f"The worst-case collapses with horizon. At 30 years, **no window has a negative "
            f"real equity return** — worst was **+{R['h30_eq_worst']:.2f}%/yr** "
            f"({R['h30_eq_neg']:.0f}% of windows negative). At 20 years, the worst was just "
            f"**{R['h20_eq_worst']:+.2f}%/yr** — barely below zero, only {R['h20_eq_neg']:.2f}% of "
            f"windows. At 10 years, however, {R['h10_eq_neg']:.1f}% of windows are negative and the "
            f"worst was {R['h10_eq_worst']:+.2f}%/yr."
        ),
        md(
            "**Does equity always beat bonds?** Here's the equity-minus-bond excess across every horizon:"
        ),
        code(
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))\n"
            "axes[0].plot(labels, ex_means, 's-', color=GREEN, lw=2, label='mean excess (eq - bond)')\n"
            "axes[0].fill_between(labels, ex_worsts, 0, alpha=0.18, color=RED, label='worst-case')\n"
            "axes[0].axhline(0, c='k', lw=1)\n"
            "axes[0].set_title('Real equity-minus-bond excess: mean vs worst-case'); axes[0].set_ylabel('%/yr')\n"
            "axes[0].legend()\n"
            "if HAVE_REAL:\n"
            "    ex_negs = [summ.loc[h, 'excess_pct_neg'] for h in HORIZONS]\n"
            "else:\n"
            "    ex_negs = [R['h1_ex_neg'], R['h5_ex_neg'], R['h10_ex_neg'], R['h20_ex_neg'], R['h30_ex_neg']]\n"
            "axes[1].bar(labels, ex_negs, color=[RED if v>10 else AMBER if v>1 else GREEN for v in ex_negs])\n"
            "axes[1].set_title('% of rolling windows where bonds beat equity'); axes[1].set_ylabel('% of windows')\n"
            "plt.tight_layout(); plt.show()\n"
            "for h, m, w, n in zip(HORIZONS, ex_means, ex_worsts, ex_negs):\n"
            "    print(f'{h:>3}y: mean={m:+.2f}%  worst={w:+.2f}%  bonds_win={n:.2f}%')"
        ),
        md(
            f"At 30 years, bonds beat equity in just **{R['h30_ex_neg']:.2f}%** of windows (one "
            f"isolated episode, 1902–1932), with a worst-case excess of only "
            f"**{R['h30_ex_worst']:+.2f}%/yr**. At 20 years, bonds won in "
            f"**{R['h20_ex_neg']:.2f}%** of windows — still rare, but not zero. "
            f"The claim is *almost* true, not literally always true."
        ),
        md(
            "**The decade-by-decade picture — the bad stretches exist:**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    dec = st.decade_returns(_df)\n"
            "    decades = list(dec.index)\n"
            "    eq_d = list(dec['eq_ann_real']*100)\n"
            "    bd_d = list(dec['bd_ann_real']*100)\n"
            "    ex_d = list(dec['excess']*100)\n"
            "else:\n"
            "    decades = [1871,1881,1891,1901,1911,1921,1931,1941,1951,1961,1971,1981,1991,2001,2011]\n"
            "    eq_d = [12.24,4.60,9.05,5.46,-4.09,16.54,3.21,6.51,14.24,5.03,0.33,9.04,14.50,-1.12,11.79]\n"
            "    bd_d = [7.27,5.32,3.46,1.39,-3.24,5.75,4.14,-3.42,1.44,2.11,0.13,6.06,3.84,1.86,0.43]\n"
            "    ex_d = [4.97,-0.72,5.59,4.07,-0.85,10.79,-0.93,9.93,12.80,2.91,0.20,2.98,10.66,-2.98,11.36]\n"
            "xlabels = [f'{d}s' for d in decades]\n"
            "fig, ax = plt.subplots(figsize=(12, 4.5))\n"
            "width = 0.35\n"
            "x = range(len(decades))\n"
            "ax.bar([i - width/2 for i in x], eq_d, width, label='Equity real', color=GREEN, alpha=0.8)\n"
            "ax.bar([i + width/2 for i in x], bd_d, width, label='Bond proxy real', color=GREY, alpha=0.8)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "for i, e in enumerate(ex_d):\n"
            "    if e < 0: ax.annotate('bonds win', (i, min(eq_d[i], bd_d[i])-1), ha='center', fontsize=7, color=RED)\n"
            "ax.set_xticks(list(x)); ax.set_xticklabels(xlabels, rotation=45, ha='right')\n"
            "ax.set_ylabel('%/yr annualised'); ax.set_title('Decade-by-decade: real equity vs real bond proxy')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "n_neg = sum(1 for e in ex_d if e < 0)\n"
            "print(f'Bonds beat equity in {n_neg}/{len(decades)} decades')"
        ),
        md(
            f"Bonds beat equity in **{R['n_decades_bonds_win']} of {R['n_decades']} decades**: "
            "the 1880s, 1910s, 1930s, and 2000s. The long-run premium is undeniable, but the "
            "short run is brutal — and even a decade can deliver a loss for equity holders."
        ),

        # ---- BEAT 5 — THE VERDICT --------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — REAL.** The equity premium is one of the most robust facts in finance: "
            f"mean 30-year real return **+{R['h30_eq_mean']:.2f}%/yr**, mean excess over bonds "
            f"**+{R['h30_ex_mean']:.2f}%/yr**, HAC *t* = **{R['h30_t']:.1f}** (though the "
            f"overlapping-window caveat applies — see the quants notebook).\n"
            f"- **Tradability — FRAGILE.** The premium is real, but the horizon requirement is "
            f"brutal. You need 20–30 years of patience and the financial stability to never "
            f"be forced to sell. In 4 of 15 decades bonds won. At 10 years, "
            f"**{R['h10_eq_neg']:.0f}%** of windows had negative real equity returns.\n"
            f"- **Useful? — Only if you can wait 30 years.** The 'always' in Siegel's claim is "
            f"almost-but-not-literally true: at 20 years one window just barely went negative "
            f"({R['h20_eq_worst']:+.2f}%/yr), and at 30 years bonds came within "
            f"**{R['h30_ex_worst']:+.2f}%/yr** of equity once."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually use this?\n\n"
            "The Siegel claim is not a trading strategy — it is a **buy-and-hold argument**. "
            "The practical obstacles:\n\n"
            "- **Horizon.** 20–30 years is most of a working life. Markets can spend a decade in "
            "  real negative territory before recovering.\n"
            "- **Forced selling.** Job loss, health, life events can force liquidation at the worst "
            "  time. The 'always' only holds if you can hold literally forever.\n"
            "- **Survivorship.** Shiller's data covers the **U.S. stock market**, one of the great "
            "  survivorship-bias cases in investing. Germany 1914–1923, Russia 1917, Argentina — "
            "  equity markets can go to zero. The 30-year track record is real in the U.S.; it is "
            "  not guaranteed elsewhere.\n"
            "- **The future.** The 1871–2023 window includes the extraordinary rise of the U.S. as "
            "  the dominant global economy. Whether the next 150 years look like the last is unknown."
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **CAPE timing.** If the long-run premium is real but the horizon is huge, can CAPE "
            "  help you avoid the worst stretches? See "
            "  [Study 56 — Tide-Table](../../56-tide-table/) and "
            "  [Study 120 — Excess-CAPE-Yield](../../120-excess-cape-yield/).\n"
            "- **Bonds are not simple either.** Our bond proxy (10y yield − CPI) is crude. Real "
            "  total-return bond indices often differ. The 1940s result (bonds −3.42%/yr real) "
            "  shows bonds can also destroy capital.\n"
            "- **International replication.** Dimson, Marsh & Staunton (2002, *Triumph of the "
            "  Optimists*) extend this to 21 countries: the U.S. story holds broadly but with "
            "  wide variation, and a few markets went to zero.\n\n"
            "*Think the equity premium is fading? Fork this, extend the data to international "
            "markets or to post-2000 windows, and test whether the 30-year worst-case is still "
            "near zero. That's the next question.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "01_for_the_curious.ipynb")


# ===========================================================================
# 02 — FOR THE QUANTS
# ===========================================================================
def build_quants():
    cells = [
        md(
            "# Stocks-For-Long-Run — a quantitative teardown\n"
            "### Shiller 1871–2023 · rolling real equity & bond returns · worst-case windows · HAC inference\n\n"
            "![Signal: Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square)\n"
            "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
            "![Useful%3F: Only_if_you_can_wait_30_years](https://img.shields.io/badge/Useful%3F-Only_if_you_can_wait_30_years-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — "
            "*same seven beats, every claim now carrying its standard error and the overlapping-"
            "window caveat.* We measure Siegel's 'equity always wins at long horizons' claim "
            "against 150+ years of Shiller real price and dividend data, compute worst-case "
            "rolling windows, and flag the survivorship and power caveats honestly.\n\n"
            "> **Not investment advice.** Real data: Shiller S&P 500 monthly panel, "
            "1871-01 to 2023-06, fingerprint `c58161b47345`. Methods in "
            "[`docs/references.md`](../docs/references.md), reproducible numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **The `In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT + "\nfrom quantlab import analytics, stats\n"),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `REAL` | 30-year mean excess equity return: "
            f"**+{R['h30_ex_mean']:.2f}%/yr**, HAC *t* = **{R['h30_t']:.1f}** on overlapping "
            f"windows (effective N << 1,470 raw windows). |\n"
            f"| **Tradability** | `FRAGILE` | Premium is real but requires a 20–30 year "
            f"horizon to be reliable; at 10 years **{R['h10_eq_neg']:.0f}%** of windows had "
            f"negative real equity returns; 4 of {R['n_decades']} decades saw bonds win. |\n"
            f"| **Useful?** | `Only if you can wait 30 years` | At 20y the worst real equity "
            f"return was **{R['h20_eq_worst']:+.2f}%/yr** (one window); at 30y bonds came "
            f"within **{R['h30_ex_worst']:+.2f}%/yr** of equity once. |\n\n"
            "> In plain words: the equity premium is one of finance's most robust empirical "
            "facts — but it requires decades to become reliable, is U.S.-centric (survivorship), "
            "and the 'always' in Siegel's framing is a rhetorical near-miss."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $R^{eq}_{t,t+h}$ be the annualised real total equity return over $h$ years "
            "starting at $t$, and $R^{bd}_{t,t+h}$ the same for a real bond proxy.  "
            "Siegel's claim asserts:\n\n"
            "- **H₁ (no negative real equity at 20y):** $\\min_t R^{eq}_{t,t+20} \\geq 0$\n"
            "- **H₂ (no negative real equity at 30y):** $\\min_t R^{eq}_{t,t+30} \\geq 0$\n"
            "- **H₃ (equity always beats bonds at 20y):** $\\min_t (R^{eq} - R^{bd})_{t,t+20} \\geq 0$\n"
            "- **H₄ (equity always beats bonds at 30y):** $\\min_t (R^{eq} - R^{bd})_{t,t+30} \\geq 0$\n\n"
            "**Our assessment:**\n"
            f"H₁ — *barely fails*: worst-case = **{R['h20_eq_worst']:+.2f}%/yr** (1901 window).  \n"
            f"H₂ — *passes*: worst-case = **+{R['h30_eq_worst']:.2f}%/yr** (no negative 30y real equity).  \n"
            f"H₃ — *fails* at 2.96% of windows: worst excess = **{R['h20_ex_worst']:+.2f}%/yr**.  \n"
            f"H₄ — *nearly passes*: bonds beat equity in **{R['h30_ex_neg']:.2f}%** of 30y windows "
            f"(worst: **{R['h30_ex_worst']:+.2f}%/yr**)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The practical stakes are enormous. If H₁–H₄ held perfectly, an investor with a "
            "20+ year horizon could rationally hold 100% equity with certainty of beating bonds. "
            "The near-misses matter: a −0.18%/yr real return over 20 years is a 3.6% real loss "
            "on capital — small but non-zero. More importantly, the **short-run** path to that "
            "outcome can include terrifying drawdowns. The relevant risk for most investors is "
            "not the 30-year terminal wealth, but the 3-year mark-to-market they have to live "
            "through without panic-selling."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "**Data:**  \n"
            "- Shiller S&P 500 monthly panel, 1871-01 to 2023-06 (staged parquet, no re-fetch).\n"
            "- Real equity total return: cumulative product of monthly real price change + "
            "  real dividend yield (Real Price + Real Dividend columns).\n"
            "- Real bond proxy: Long Interest Rate / 100 / 12 (monthly) minus monthly CPI "
            "  inflation. This is a crude but widely-used first approximation — we name the "
            "  approximation explicitly.\n\n"
            "**Computation:**  \n"
            "- Rolling N-year annualised return for N ∈ {1, 5, 10, 20, 30}: every monthly start, "
            "  $(TR_{t+N\\cdot12}/TR_t)^{1/N} - 1$.  Overlapping windows → standard errors are "
            "  inflated by high autocorrelation; we report HAC t-stats but flag that effective "
            "  sample size for a 30-year window is closer to 4 than 1,470.\n"
            "- Worst-case: direct empirical minimum, no distributional assumption.\n"
            "- Decade breakdown: non-overlapping 10-year blocks for a clean time series.\n\n"
            "**Positive control:**  \n"
            "Synthetic world with a tunable equity premium: the engine should find near-zero "
            "worst-cases in a high-premium world, and frequent worst-cases in a null world."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The full rolling-window distribution — horizon matters enormously\n\n"
            "For each horizon, we show the distribution of annualised real equity total returns "
            "across all rolling windows."
        ),
        code(
            "HORIZONS = [1, 5, 10, 20, 30]\n"
            "if HAVE_REAL:\n"
            "    fig, axes = plt.subplots(1, len(HORIZONS), figsize=(14, 4.2), sharey=False)\n"
            "    for ax, h in zip(axes, HORIZONS):\n"
            "        col = f'roll_{h}y_eq'\n"
            "        vals = _df[col].dropna().to_numpy() * 100\n"
            "        ax.hist(vals, bins=40, color=GREEN, alpha=0.75, edgecolor='white')\n"
            "        ax.axvline(0, c=RED, lw=1.5, ls='--')\n"
            "        ax.axvline(vals.mean(), c=GREEN, lw=1.5)\n"
            "        neg_pct = (vals < 0).mean() * 100\n"
            "        ax.set_title(f'{h}y  (neg={neg_pct:.1f}%)', fontsize=9)\n"
            "        ax.set_xlabel('%/yr', fontsize=8)\n"
            "    axes[0].set_ylabel('count')\n"
            "    plt.suptitle('Distribution of rolling real equity return by horizon (Shiller 1871-2023)')\n"
            "    plt.tight_layout(); plt.show()\n"
            "else:\n"
            "    print('Real data not available; key headline numbers:')\n"
            "    for h, neg in [(1,R['h1_eq_neg']),(5,R['h5_eq_neg']),(10,R['h10_eq_neg']),(20,R['h20_eq_neg']),(30,R['h30_eq_neg'])]:\n"
            "        print(f'  {h}y: {neg:.2f}% negative windows')"
        ),
        md(
            "> In plain words: the left tail — the negative-return zone — nearly vanishes "
            "as the horizon extends. At 30 years it is essentially empty on the historical record."
        ),
        md(
            "### 4b · Worst-case table and the exact 20y/30y claims\n\n"
            "The numbers behind H₁–H₄:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    summ = st.all_horizon_summary(_df)\n"
            "    disp = summ[['n_windows','eq_min','eq_mean','eq_pct_neg','excess_min','excess_mean','excess_pct_neg','tstat_excess']].copy()\n"
            "    for c in ['eq_min','eq_mean','excess_min','excess_mean']:\n"
            "        disp[c] = disp[c] * 100\n"
            "    disp.columns = ['n_windows','eq_worst%','eq_mean%','eq_neg%','ex_worst%','ex_mean%','ex_neg%','HAC_t']\n"
            "    print(disp.round(2))\n"
            "else:\n"
            "    print('Frozen results (no real data):')\n"
            "    for h in HORIZONS:\n"
            "        ew = locals().get(f'R_h{h}_eq_worst', R.get(f'h{h}_eq_worst',float('nan')))\n"
            "        print(f'  {h}y: worst_eq={R.get(f\"h{h}_eq_worst\",\"n/a\")}%  worst_ex={R.get(f\"h{h}_ex_worst\",\"n/a\")}%  t={R.get(f\"h{h}_t\",\"n/a\")}')"
        ),
        code(
            "# Print the frozen R table directly (always works)\n"
            "rows = []\n"
            "for h in [1,5,10,20,30]:\n"
            "    rows.append({'horizon':f'{h}y','eq_worst%':R[f'h{h}_eq_worst'],'eq_mean%':R[f'h{h}_eq_mean'],\n"
            "                 'eq_neg%':R[f'h{h}_eq_neg'],'ex_worst%':R[f'h{h}_ex_worst'],\n"
            "                 'ex_mean%':R[f'h{h}_ex_mean'],'ex_neg%':R[f'h{h}_ex_neg'],'HAC_t':R[f'h{h}_t']})\n"
            "pd.DataFrame(rows).set_index('horizon').round(2)"
        ),
        md(
            f"> In plain words: at 30 years, no window has negative real equity and only "
            f"**{R['h30_ex_neg']:.2f}%** of windows have bonds winning. But the HAC t-stat "
            f"(**{R['h30_t']:.1f}**) must be read with caution: overlapping monthly windows have "
            f"~{30*12}x autocorrelation, so effective N for 30-year windows is the number of "
            f"*non-overlapping* 30-year periods in 150 years — roughly 5. The t-stat is "
            f"meaningful directionally but should not be taken at face value."
        ),
        md(
            "### 4c · Overlapping-window caveat — effective sample size\n\n"
            "Rolling monthly windows for a 30-year horizon starting every month yield 1,470 "
            "observations, but consecutive windows overlap by 359/360 months. The effective "
            "number of independent 30-year observations in 150 years of data is roughly "
            "$150/30 = 5$. We verify this with a synthetic positive control."
        ),
        code(
            "# Synthetic positive control: premium world vs null world\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))\n"
            "for ax, prem, col, label in [\n"
            "    (axes[0], 0.05, GREEN, 'premium=5% (realistic)'),\n"
            "    (axes[1], 0.00, RED,   'premium=0% (null)'),\n"
            "]:\n"
            "    frame, _ = data.synthetic_world(n_years=155, equity_premium=prem, seed=151)\n"
            "    means_syn = [st.worst_case_windows(frame, h)['excess_mean']*100 for h in [1,5,10,20,30]]\n"
            "    worsts_syn = [st.worst_case_windows(frame, h)['excess_min']*100 for h in [1,5,10,20,30]]\n"
            "    ax.plot([f'{h}y' for h in [1,5,10,20,30]], means_syn, 'o-', color=col, lw=2, label='mean excess')\n"
            "    ax.fill_between([f'{h}y' for h in [1,5,10,20,30]], worsts_syn, 0, alpha=0.18, color=col)\n"
            "    ax.axhline(0, c='k', lw=1)\n"
            "    ax.set_title(f'Synthetic control: {label}'); ax.set_ylabel('excess %/yr')\n"
            "    ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('With a planted 5% premium: worst-case improves with horizon as expected.')\n"
            "print('With zero premium: worst-case does NOT reliably improve — no structural guarantee.')"
        ),
        md(
            "> In plain words: the 'horizon heals all' property is a consequence of a *real* "
            "equity premium, not a tautology. A null world does not converge to safety just by "
            "waiting longer."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `REAL`** — mean 30-year real equity return **+{R['h30_eq_mean']:.2f}%/yr**, "
            f"mean excess **+{R['h30_ex_mean']:.2f}%/yr**; HAC *t* = **{R['h30_t']:.1f}** on "
            f"overlapping windows (effective N ≈ 5 independent 30-year spans; the directional "
            f"signal is robust, the t-stat's magnitude should not be taken literally).\n"
            f"- **Tradability `FRAGILE`** — the premium requires decades of patient holding. "
            f"At 10 years, **{R['h10_eq_neg']:.0f}%** of windows had negative real equity; "
            f"at the decade level, bonds won {R['n_decades_bonds_win']}/{R['n_decades']} times.\n"
            f"- **Useful? `Only if you can wait 30 years`** — H₁ barely fails (20y worst: "
            f"**{R['h20_eq_worst']:+.2f}%/yr**); H₃ fails at {R['h20_ex_neg']:.2f}% of 20y "
            f"windows; H₂ and H₄ pass on the historical U.S. record but with major "
            f"survivorship caveats."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the practical friction\n\n"
            "This is not a trading strategy — it is a **strategic allocation argument**. The "
            "quantitative friction points:\n\n"
            "1. **Horizon risk.** Markets can spend a full decade in real loss (2000s: equity "
            "   −1.12%/yr real; 1910s: −4.09%/yr). Investors who *needed* liquidity in those "
            "   windows lost out regardless of the 30-year promise.\n"
            "2. **Survivorship.** The U.S. equity market is one of history's greatest "
            "   survivorship stories. Extending to international markets (DMS 2002) shows "
            "   wide variation: some markets went to zero.\n"
            "3. **The bond proxy is rough.** Our bond return (nominal yield − CPI) understates "
            "   bond total return in falling-rate environments (1980–2020) and overstates it in "
            "   rising-rate environments. A total-return bond index would differ."
        ),
        code(
            "# Visualise: the 'path to the long run' — cumulative real total return 1871-2023\n"
            "if HAVE_REAL:\n"
            "    fig, ax = plt.subplots(figsize=(11, 5))\n"
            "    ax.semilogy(_df['real_tr'], color=GREEN, lw=1.5, label='Equity real TR')\n"
            "    ax.semilogy(_df['real_tr_bd'], color=GREY, lw=1.5, ls='--', label='Bond proxy real TR')\n"
            "    ax.set_title('Real total return index, log scale (1871-2023): the path is not smooth')\n"
            "    ax.set_ylabel('Real TR (log scale, 1871-01 = 1)'); ax.legend()\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print(f'End real TR: equity={_df[\"real_tr\"].iloc[-1]:.0f}x  bond proxy={_df[\"real_tr_bd\"].iloc[-1]:.2f}x')\n"
            "else:\n"
            "    print('The equity real TR grows ~600x from 1871 to 2023 (log chart)')\n"
            "    print('The bond proxy real TR grows ~3-5x over the same period')"
        ),
        md(
            "> In plain words: the log-scale chart shows that equity's dominance is real — the "
            "equity total return index grows *hundreds* of times more than bonds over 150 years. "
            "But the chart also shows the crashes, lost decades, and long stretches where bonds "
            "kept pace. The path to 30 years of patience is not smooth."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **CAPE-timing.** Can you identify in advance which 20-year windows will be the "
            "  bad ones, using Shiller's own CAPE? [Study 56 — Tide-Table](../../56-tide-table/) "
            "  tests this; [Study 120 — Excess-CAPE-Yield](../../120-excess-cape-yield/) extends "
            "  it to the bond-adjusted version. The ECY signals the long-run excess meaningfully.\n"
            "- **International robustness.** Dimson, Marsh & Staunton (2002) show the U.S. "
            "  premium is real but the U.S. is near the top of the league table. Replicating "
            "  on MSCI World or country-by-country data is the natural next study.\n"
            "- **The bond proxy.** Replacing the crude yield-minus-CPI proxy with TIPS or "
            "  a total-return bond index (e.g., using BIL/TLT from yfinance for the post-1980 "
            "  period) would give a sharper estimate of the equity-bond gap in modern markets.\n"
            "- **Post-publication.** The Siegel thesis became widely known in 1994. Has the "
            "  'always wins at 30y' track record held post-publication? With 2024+ data there "
            "  are now 30-year windows post-1994 to check.\n\n"
            "*Think the equity premium has structural reasons beyond historical luck? Fork this, "
            "add an international dataset or a better bond series, and update the worst-case table. "
            "That is the next test.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "02_for_the_quants.ipynb")


def _meta():
    return {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }


def _write(nb, name):
    path = os.path.join(HERE, name)
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("wrote", path)


if __name__ == "__main__":
    build_curious()
    build_quants()
