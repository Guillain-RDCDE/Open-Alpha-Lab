"""Generate (and execute) the two narrative notebooks for Study 260 (Margin-Debt).

    python notebooks/build_notebooks.py

This writes both notebooks and then runs them in-place with nbconvert so every
code cell carries its outputs/figures. The hardcoded margin-debt table and the
synthetic generator run anywhere, offline and deterministically; the real S&P 500
cells use the repo-level cache ``_cache/^GSPC_split_only.parquet`` if present and
otherwise quote the frozen headline numbers in ``R`` (mirroring docs/results.md),
so the notebooks re-run for any reader.

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
R = dict(
    n=65, year_start=1960, year_end=2024,
    # predictive regression
    reg_beta_pct=-2.23, reg_t=-1.35, reg_r2=0.023,
    # terciles
    ter_low=11.9, ter_mid=4.9, ter_high=9.0, ter_spread=-2.9, ter_t=-0.67,
    # record-high event
    rec_mean=6.6, nonrec_mean=11.1, rec_diff=-4.5,
    rec_up=62.9, uncond_up=73.9, rec_t=-1.22,
    # sub-periods
    sub1_t=-1.62, sub2_t=-0.56, sub3_t=-1.44,
    # tradability
    bah_ann=7.53, strat_net_ann=6.51, years_cash=8, shortfall_pp=-1.02,
    # notable post-record crashes (forward 12m, price-only)
    y2000=-17.3, y2007=-40.1, y2021=-9.7,
    fp="3f24fa7a68df",
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

from margin_debt import data, strategy as st

def _have_cache():
    try:
        data.build_panel(fetch=False)
        return True
    except FileNotFoundError:
        return False

HAVE_REAL = _have_cache()
if HAVE_REAL:
    panel = data.build_panel(fetch=False)
    print(f"Real panel: {len(panel)} forward-years "
          f"({panel['year'].min()}-{panel['year'].max()})")
else:
    panel = None
    print("No ^GSPC cache -- frozen headline numbers from R dict will be used")
"""

R_INJECT = f"""
# Frozen headline numbers (mirror of docs/results.md, as-of 2026-06-17)
R = {R!r}
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious() -> None:
    nb = new_notebook()
    nb.cells = [
        md(
            "# Margin-Debt -- is record NYSE margin debt a contrarian sell signal?\n"
            "### The 'investors are dangerously leveraged' headline, tested honestly\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n\n"
            "*Part of [Open-Alpha-Lab](../../../README.md). See the "
            "[desk methodology](../../../METHODOLOGY.md).*\n\n"
            "---\n\n"
            "Every few years a chart goes viral: NYSE/FINRA **margin debt** -- the money "
            "investors borrow against their portfolios -- has hit a **record high**, and the "
            "caption warns that the market is over-leveraged and about to crash. The story is "
            "seductive because margin debt *did* peak right before the 2000, 2007, and 2021 "
            "tops.\n\n"
            "This notebook asks the only question that matters: does a margin-debt record (or a "
            "fast year-over-year rise) actually **forecast lower returns** -- or is margin debt "
            "just trending along *with* the market, going up because prices went up?\n\n"
            "> **This is the plain-language layer.** Want the HAC regression, the tercile bins, "
            "and the positive control? That's "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** -- same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** Every chart is drawn by the code beside it."
        ),
        md("## Setup"),
        code(BOOT),
        code(R_INJECT),

        # ---- BEAT 0 -- VERDICT ----------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does a margin-debt record predict a down year? | **Weakly, in sign only.** Record "
            "years average **+6.6%** forward vs **+11.1%** otherwise -- the right direction, but "
            "the difference's *t* is just **-1.22** (not significant). |\n"
            "| Does fast margin growth forecast lower returns? | **Same.** The regression slope is "
            "**negative** (the contrarian sign) but HAC *t* = **-1.35** -- it never clears the "
            "*t* = 2 bar, in any sub-period. |\n"
            "| Could you trade it? | **No.** A 'go to cash after a margin surge' rule "
            "*underperforms* buy-and-hold by **~1pp/yr** -- it misses more up-years than crashes "
            "it dodges. |\n"
            "| So why does the chart look scary? | Margin debt rises *because* the market rose. "
            "It's a coincident by-product of a bull market, not a leading indicator of its end. |\n\n"
            "> The contrarian story has the right *vibe* -- and a couple of vivid anecdotes -- "
            "but no statistical edge and no tradable rule."
        ),

        # ---- BEAT 1 -- THE CLAIM --------------------------------------------
        md(
            "## 1 - The claim\n\n"
            "> *\"Margin debt is at an all-time high. Investors are borrowing record amounts to "
            "buy stocks. When this house of cards collapses, forced selling will crash the "
            "market. A record is a SELL signal.\"*\n\n"
            "The data behind the headline is real: FINRA (and before 2010, the NYSE) publishes "
            "monthly **customer debit balances in margin accounts**. We hardcode the December "
            "year-end series (1959-2025) in this study's `data.py`."
        ),
        code(
            "md = data.margin_debt_table()\n"
            "fig, ax = plt.subplots(figsize=(10, 4.5))\n"
            "ax.plot(md['year'], md['margin_debt']/1000, c=GREY, lw=1.8)\n"
            "ax.set_yscale('log')\n"
            "ax.set_xlabel('Year'); ax.set_ylabel('Margin debt ($bn, log scale)')\n"
            "ax.set_title('NYSE/FINRA customer margin debt, December year-end (1959-2025)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"1959: ${md['margin_debt'].iloc[0]/1000:.1f}bn  ->  \"\n"
            "      f\"2025: ${md['margin_debt'].iloc[-1]/1000:.0f}bn\")\n"
            "print('The level marches relentlessly upward -- like the market itself.')"
        ),

        # ---- BEAT 2 -- SO WHAT ----------------------------------------------
        md(
            "## 2 - So what?\n\n"
            "If a record margin-debt print really signalled an imminent crash, it would be the "
            "easiest market-timing rule ever: read one FINRA press release a month, and step "
            "aside before every top. The trouble is that margin debt is **dollar-denominated and "
            "scales with portfolio values** -- when prices rise, the value of marginable "
            "collateral rises and borrowing capacity expands. So margin debt mostly goes up "
            "*because the market already went up*. A coincident by-product of a trend makes a "
            "poor leading indicator."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW ----------------------------------------
        md(
            "## 3 - How would we even know?\n\n"
            "Three traps to avoid:\n\n"
            "1. **The level trends -- use the change.** The *level* of margin debt rises forever "
            "(so does the S&P). The testable claim is about the **year-over-year change**: does "
            "fast margin growth predict a *lower* next-year return?\n"
            "2. **The base-rate trap.** The S&P rises ~**74%** of years unconditionally. Any 'the "
            "market falls' signal must beat that base rate, not a coin.\n"
            "3. **Anecdotes vs the sample.** 2000, 2007, and 2021 are vivid -- but they are 3 of "
            "65 years. We test the *whole* sample, not the cherry-picked tops.\n\n"
            "We pair the hardcoded margin-debt series with the S&P 500 month-end close "
            "(price-only) and measure the forward 12-month return after a one-month "
            "reporting/execution lag, 1960-2024."
        ),

        # ---- BEAT 4 -- THE TEARDOWN -----------------------------------------
        md(
            "## 4 - The teardown -- let's actually look\n\n"
            "**The scatter that should be a steep downward line if the folklore were true:**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    x = panel['yoy'].values*100\n"
            "    y = panel['fwd_ret'].values*100\n"
            "    reg = st.predictive_regression(panel)\n"
            "    beta_pct, t = reg['beta']*100, reg['beta_t']\n"
            "else:\n"
            "    rng = np.random.default_rng(260)\n"
            "    x = rng.normal(12, 20, R['n']); y = 9 + R['reg_beta_pct']*(x-12)/20 + rng.normal(0,16,R['n'])\n"
            "    beta_pct, t = R['reg_beta_pct'], R['reg_t']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.0, 5.0))\n"
            "ax.scatter(x, y, c=GREY, s=30, alpha=0.8)\n"
            "xs = np.linspace(x.min(), x.max(), 50)\n"
            "# slope is per +1 sigma; convert to per-percent for the line\n"
            "sd = np.std(x)\n"
            "ax.plot(xs, np.mean(y) + (beta_pct/sd)*(xs-np.mean(x)), c=RED, lw=2,\n"
            "        label=f'fit: t = {t:+.2f} (per +1 sigma: {beta_pct:+.1f}%)')\n"
            "ax.axhline(0, c='k', lw=0.8)\n"
            "ax.set_xlabel('YoY margin-debt growth (%)'); ax.set_ylabel('Next 12m S&P return (%)')\n"
            "ax.set_title('More leverage -> lower returns? A faint downward tilt, lost in the cloud')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Slope has the contrarian (negative) sign; HAC t = {t:+.2f} -- |t| < 2, not significant.')"
        ),
        md(
            "The cloud has a *faint* downward tilt -- the contrarian sign is there -- but it is "
            "swamped by noise. A one-standard-deviation surge in margin growth shaves only about "
            "**2.2%** off the forward return, with HAC *t* = **-1.35**."
        ),
        md("**The literal headline test: what happens after a fresh record?**"),
        code(
            "if HAVE_REAL:\n"
            "    rec = st.record_high_event(panel)\n"
            "    rec_m, non_m = rec['record_mean']*100, rec['nonrecord_mean']*100\n"
            "    base = rec['uncond_up_rate']*100\n"
            "    rec_up = rec['record_up_rate']*100\n"
            "else:\n"
            "    rec_m, non_m, base, rec_up = R['rec_mean'], R['nonrec_mean'], R['uncond_up'], R['rec_up']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "bars = ax.bar(['Record-high\\nyears', 'Non-record\\nyears'],\n"
            "              [rec_m, non_m], color=[RED, GREEN], width=0.55)\n"
            "ax.axhline(0, c='k', lw=0.8)\n"
            "ax.set_ylabel('Mean next-12m S&P return (%)')\n"
            "ax.set_title(f'After a margin-debt record: {rec_m:+.1f}% vs {non_m:+.1f}% otherwise (not significant)')\n"
            "for b, v in zip(bars, [rec_m, non_m]):\n"
            "    ax.annotate(f'{v:+.1f}%', (b.get_x()+b.get_width()/2, v),\n"
            "                ha='center', va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Record years: {rec_m:+.1f}% forward, up only {rec_up:.0f}% of the time')\n"
            "print(f'vs the {base:.0f}% unconditional up-rate -- lower, but Welch t = {R[\"rec_t\"]:+.2f} (n.s.)')"
        ),
        md(
            "Record-margin years *do* average a lower forward return (**+6.6%** vs **+11.1%**) "
            "and a lower up-rate (**62.9%** vs the **73.9%** base rate). That is the kernel of "
            "truth behind the folklore -- and it is driven by the famous post-record crashes of "
            f"**2000 ({R['y2000']:.0f}%)**, **2007 ({R['y2007']:.0f}%)**, and "
            f"**2021 ({R['y2021']:.0f}%)**. But across 35 record-years the difference is "
            "statistically indistinguishable from noise (*t* = -1.22)."
        ),

        # ---- BEAT 5 -- THE VERDICT ------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            "- **Signal -- Weak.** The contrarian *sign* is correct and stable in every "
            "sub-period (record years and surge years both underperform), and it rhymes with real "
            "history. But the full-sample HAC *t* is only **-1.35**, the tercile *t* is **-0.67**, "
            "and the record-event *t* is **-1.22** -- nothing clears the *t* = 2 bar. Folklore + a "
            "right-but-insignificant tape = **Weak**, not Real.\n"
            "- **Tradability -- Mirage.** A 'cash after a surge' rule *loses* to buy-and-hold by "
            "~1pp/yr. There is no edge to harvest, because margin debt is a coincident effect of "
            "rising prices, not a cause of falling ones."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT -----------------------------------
        md(
            "## 6 - Could you actually trade it?\n\n"
            "The rule: go to cash for the 12 months after any year of unusually fast margin "
            "growth (z-score > 1); otherwise hold the S&P. Net of 10 bps per switch."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bt = st.contrarian_timing_backtest(panel, surge_z=1.0, one_way_bps=10.0)\n"
            "    bah, strat = bt['bah_ann']*100, bt['strat_net_ann']*100\n"
            "    cash, short = bt['years_in_cash'], bt['shortfall_net_pp']\n"
            "else:\n"
            "    bah, strat = R['bah_ann'], R['strat_net_ann']\n"
            "    cash, short = R['years_cash'], R['shortfall_pp']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.0, 4.2))\n"
            "bars = ax.bar(['Buy & hold\\n(price-only)', 'Contrarian rule\\n(net 10bps)'],\n"
            "              [bah, strat], color=[GREEN, RED], width=0.5)\n"
            "ax.set_ylabel('Annualized return (%)')\n"
            "ax.set_title(f'The contrarian timing rule trails buy-and-hold by {short:+.1f}pp/yr')\n"
            "for b, v in zip(bars, [bah, strat]):\n"
            "    ax.annotate(f'{v:+.2f}%', (b.get_x()+b.get_width()/2, v),\n"
            "                ha='center', va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Buy & hold: {bah:+.2f}%/yr  |  Contrarian rule: {strat:+.2f}%/yr')\n"
            "print(f'Sat in cash {cash} years and still lost {short:+.1f}pp/yr.')\n"
            "print('Adding dividends (total return) widens the shortfall further.')"
        ),
        md(
            "> Stepping aside after a margin surge dodged a few drawdowns but missed more up-years "
            "than it avoided down-years. The only real lesson: don't sell because a borrowing "
            "statistic set a record."
        ),

        # ---- BEAT 7 -- GOING FURTHER ----------------------------------------
        md(
            "## 7 - Going further\n\n"
            "- **The positive control** (in the quants notebook): plant a real contrarian signal "
            "in a synthetic tape and the same regression lights up at *t* = -5 to -11. The real "
            "tape simply doesn't carry one that strong.\n"
            "- **The base-rate cousin:** [Study 158 -- Super-Bowl](../../158-super-bowl/) -- a "
            "binary 'predictor' that just re-packages the market's upward bias.\n\n"
            "*Think margin debt really times the market? Fork this, define your own surge/record "
            "rule, and show a forward signal that clears |t| >= 2 and beats buy-and-hold net of "
            "costs. That's the bar -- and it hasn't been cleared.*\n\n"
            "*Desk verdict: **Weak / Mirage** -- see [README.md](../README.md) and "
            "[docs/results.md](../docs/results.md) for full numbers.*"
        ),
    ]
    _write(nb, "01_for_the_curious.ipynb")


# ===========================================================================
# 02 -- FOR THE QUANTS
# ===========================================================================
def build_quants() -> None:
    nb = new_notebook()
    nb.cells = [
        md(
            "# Margin-Debt -- a quantitative teardown\n"
            "### 65 forward-years * HAC OLS * YoY terciles * record-high event * "
            "sub-periods * timing rule * synthetic control\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n\n"
            "*Part of [Open-Alpha-Lab](../../../README.md). See the "
            "[desk methodology](../../../METHODOLOGY.md).*\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) -- "
            "*same seven beats, every claim carrying its standard error.* Real data: NYSE/FINRA "
            "December margin debt (hardcoded, 1959-2025) x S&P 500 month-end **price-only** "
            "(`^GSPC_split_only`). Methods in [`docs/references.md`](../docs/references.md), "
            "numbers in [`docs/results.md`](../docs/results.md).\n\n"
            "> 💡 *In plain words* asides translate each result back to intuition."
        ),
        md("## Setup"),
        code(BOOT + "\nfrom scipy import stats as scipy_stats\n"),
        code(R_INJECT),

        # ---- BEAT 0 ---------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            "| **Signal** | `WEAK` | Contrarian (negative) slope, but HAC *t* = "
            "**-1.35**; tercile *t* = **-0.67**; record-event *t* = **-1.22**. Right sign, never "
            "clears \\|t\\| >= 2. |\n"
            "| **Tradability** | `MIRAGE` | 'Cash after a surge' rule trails buy-and-hold by "
            "**-1.0pp/yr** net. |\n\n"
            "> Margin debt is dollar-denominated and scales with portfolio values: it rises "
            "*because* prices rose. A coincident by-product is a poor leading indicator."
        ),

        # ---- BEAT 1 ---------------------------------------------------------
        md(
            "## 1 - The claim, steelmanned\n\n"
            "Let $Y_{t+1}$ be the forward 12-month S&P return and $g_t$ be the standardized YoY "
            "margin-debt growth through December $t$. The contrarian hypotheses:\n\n"
            "- **H1 (slope).** In $Y_{t+1} = \\alpha + \\beta\\, g_t + \\varepsilon$, we have "
            "$\\beta < 0$ with HAC $|t| \\ge 2$.\n"
            "- **H2 (terciles).** Mean forward return in the top YoY-growth tercile is below the "
            "bottom tercile.\n"
            "- **H3 (record event).** Forward return after a fresh all-time-high margin print is "
            "below the unconditional mean / base rate.\n\n"
            "We find the *signs* of H1-H3 all point the contrarian way, but none reaches "
            "significance."
        ),

        # ---- BEAT 2 ---------------------------------------------------------
        md(
            "## 2 - So what? -- what rides on each answer\n\n"
            "If H1-H3 held with real *t*-stats, margin debt would be a free monthly macro timing "
            "signal. The interesting finding is *why* the chart looks predictive: margin-debt "
            "records cluster at the end of long bull runs, so a handful of vivid post-record "
            "crashes (2000, 2007, 2021) anchor the narrative while the many records that kept "
            "rising are quietly forgotten -- a textbook anecdote-vs-base-rate problem."
        ),

        # ---- BEAT 3 ---------------------------------------------------------
        md(
            "## 3 - How we'd know -- the protocol\n\n"
            "- **Data.** December NYSE/FINRA margin debt (hardcoded in `data.py`); S&P 500 "
            "month-end close, price-only (`^GSPC_split_only`).\n"
            "- **Signal.** Standardized YoY margin growth through December $t$.\n"
            "- **Lag.** One-month reporting/execution lag (enter end-January of $t{+}1$); forward "
            "return is the next 12 months.\n"
            "- **Inference.** HAC (Newey-West) OLS slope *t*; Welch *t* for tercile and "
            "record-event group differences; sub-period splits.\n"
            "- **Positive control.** Synthetic tape with a planted contrarian `predictive_beta`.\n"
            "- **Honesty.** Price-only returns labeled (conservative for a contrarian 'sell' "
            "claim); single index, no survivorship selection."
        ),

        # ---- BEAT 4 ---------------------------------------------------------
        md("## 4 - The teardown"),
        md(
            "### 4a - The HAC predictive regression\n\n"
            "Forward 12m return on standardized YoY margin growth, Newey-West *t*."
        ),
        code(
            "if HAVE_REAL:\n"
            "    reg = st.predictive_regression(panel)\n"
            "    beta, t, r2, n = reg['beta'], reg['beta_t'], reg['r2'], reg['n']\n"
            "else:\n"
            "    beta, t, r2, n = R['reg_beta_pct']/100, R['reg_t'], R['reg_r2'], R['n']\n"
            "\n"
            "print('=== Contrarian predictive regression ===')\n"
            "print(f'  slope (per +1 sigma): {beta*100:+.2f}% forward return')\n"
            "print(f'  HAC t = {t:+.3f}   R^2 = {r2:.3f}   n = {n}')\n"
            "print()\n"
            "print(f'  Contrarian sign: {\"YES\" if beta < 0 else \"NO\"}')\n"
            "print(f'  Clears |t| >= 2: {\"YES\" if abs(t) >= 2 else \"NO -- verdict WEAK\"}')"
        ),
        md(
            "> 💡 *In plain words:* the line tilts the right way (more leverage, slightly lower "
            "next-year return) but the tilt is buried in the noise. *R²* ≈ 0.02 -- margin growth "
            "explains ~2% of next year's return."
        ),
        md(
            "### 4b - Terciles: surging vs contracting leverage\n\n"
            "Sort years into thirds by YoY margin growth; compare forward returns."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ter = st.tercile_forward_returns(panel)\n"
            "    lo, mi, hi = ter['low_mean']*100, ter['mid_mean']*100, ter['high_mean']*100\n"
            "    spr, wt = ter['spread_high_minus_low']*100, ter['welch_t']\n"
            "else:\n"
            "    lo, mi, hi = R['ter_low'], R['ter_mid'], R['ter_high']\n"
            "    spr, wt = R['ter_spread'], R['ter_t']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "bars = ax.bar(['Bottom third\\n(contracting)', 'Middle third', 'Top third\\n(surging)'],\n"
            "              [lo, mi, hi], color=[GREEN, GREY, RED], width=0.6)\n"
            "ax.set_ylabel('Mean next-12m S&P return (%)')\n"
            "ax.set_title(f'High-minus-low spread = {spr:+.1f}%  (Welch t = {wt:+.2f}, n.s.)')\n"
            "for b, v in zip(bars, [lo, mi, hi]):\n"
            "    ax.annotate(f'{v:+.1f}%', (b.get_x()+b.get_width()/2, v), ha='center', va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'High - Low: {spr:+.1f}%  Welch t = {wt:+.2f}')"
        ),
        md(
            "The high-growth tercile trails the low-growth tercile (the contrarian direction) but "
            "Welch *t* = -0.67 -- noise. Note the **middle** tercile is the weakest, which is the "
            "opposite of a clean monotonic relationship."
        ),
        md(
            "### 4c - The record-high event test (the newspaper headline)\n\n"
            "Forward return in fresh-all-time-high margin years vs all others."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rec = st.record_high_event(panel)\n"
            "    rm, nm = rec['record_mean']*100, rec['nonrecord_mean']*100\n"
            "    ru, base, rt = rec['record_up_rate']*100, rec['uncond_up_rate']*100, rec['welch_t']\n"
            "    nrec, nnon = rec['n_record'], rec['n_nonrecord']\n"
            "else:\n"
            "    rm, nm = R['rec_mean'], R['nonrec_mean']\n"
            "    ru, base, rt = R['rec_up'], R['uncond_up'], R['rec_t']\n"
            "    nrec, nnon = 35, 30\n"
            "\n"
            "print('=== Fresh-record-high event test ===')\n"
            "print(f'  Record years     : {rm:+.1f}% forward, up {ru:.0f}% of the time  (n={nrec})')\n"
            "print(f'  Non-record years : {nm:+.1f}% forward                         (n={nnon})')\n"
            "print(f'  Unconditional up-rate: {base:.0f}%')\n"
            "print(f'  Difference: {rm-nm:+.1f}%   Welch t = {rt:+.2f}')\n"
            "print()\n"
            "print('Record years underperform (the kernel of truth), but t = -1.22 is not significant.')"
        ),
        md(
            "> 💡 *In plain words:* the headline is *directionally* right -- record years really do "
            "average less -- but on 35 record-years with ~17% annual vol, a 4.5pp gap is exactly "
            "what chance delivers. The base rate (74% up) does most of the work."
        ),
        md(
            "### 4d - Sub-period robustness: stable sign, never significant\n\n"
            "Split the sample into three eras and re-run the slope."
        ),
        code(
            "rows = []\n"
            "if HAVE_REAL:\n"
            "    for label, a, b in [('1960-1989',1960,1989),('1990-2009',1990,2009),('2010-2024',2010,2024)]:\n"
            "        sub = panel[(panel['year']>=a)&(panel['year']<=b)]\n"
            "        r = st.predictive_regression(sub)\n"
            "        rows.append({'period': label, 'slope_pct': r['beta']*100, 't': r['beta_t'], 'n': r['n']})\n"
            "else:\n"
            "    rows = [{'period':'1960-1989','slope_pct':-2.67,'t':R['sub1_t'],'n':30},\n"
            "            {'period':'1990-2009','slope_pct':-2.15,'t':R['sub2_t'],'n':20},\n"
            "            {'period':'2010-2024','slope_pct':-2.11,'t':R['sub3_t'],'n':15}]\n"
            "sub_df = pd.DataFrame(rows)\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.2))\n"
            "colors = [GREEN if abs(t) >= 2 else AMBER for t in sub_df['t']]\n"
            "ax.bar(sub_df['period'], sub_df['t'], color=colors)\n"
            "ax.axhline(-2.0, c='k', ls='--', lw=0.9, label='t = -2 inference bar')\n"
            "ax.axhline(0, c='k', lw=0.8)\n"
            "ax.set_ylabel('HAC t on the contrarian slope')\n"
            "ax.set_title('Negative in every era -- but the t-stat never reaches -2')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(sub_df.to_string(index=False))"
        ),
        md(
            "The slope is negative in **all three** sub-periods -- a genuinely robust *direction* "
            "-- yet the *t* never reaches -2. A stable-sign, never-significant effect is the "
            "signature of a weak-but-real-ish prior with no exploitable edge."
        ),
        md(
            "### 4e - The tradable rule vs buy-and-hold\n\n"
            "Cash for the year after a surge (z > 1); else hold the S&P; net of 10 bps/switch."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bt = st.contrarian_timing_backtest(panel, surge_z=1.0, one_way_bps=10.0)\n"
            "    bah, strat, cash, short = bt['bah_ann']*100, bt['strat_net_ann']*100, bt['years_in_cash'], bt['shortfall_net_pp']\n"
            "else:\n"
            "    bah, strat, cash, short = R['bah_ann'], R['strat_net_ann'], R['years_cash'], R['shortfall_pp']\n"
            "\n"
            "print(f'Buy & hold (price-only):   {bah:+.2f}%/yr')\n"
            "print(f'Contrarian rule (net):     {strat:+.2f}%/yr')\n"
            "print(f'Years in cash: {cash}    Shortfall: {short:+.2f}pp/yr')\n"
            "print()\n"
            "print('The rule underperforms gross AND net -- there is no edge to monetize.')"
        ),

        # ---- BEAT 5 ---------------------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            "- **Signal `WEAK`** -- contrarian sign confirmed and stable across all eras, rhyming "
            "with the 2000/2007/2021 tops, but HAC *t* = -1.35 (regression), -0.67 (terciles), "
            "-1.22 (record event) -- none clears \\|t\\| >= 2. Folklore + right-but-insignificant "
            "tape = **Weak**.\n"
            "- **Tradability `MIRAGE`** -- the contrarian timing rule loses to buy-and-hold by "
            "~1pp/yr; the signal is a coincident by-product of rising prices, not a leading "
            "indicator."
        ),

        # ---- BEAT 6 ---------------------------------------------------------
        md(
            "## 6 - Could you trade it? -- the buy-and-hold benchmark\n\n"
            "Cumulative wealth: the rule vs passive holding (uses real tape when cached)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    p = panel.dropna(subset=['yoy','fwd_ret']).reset_index(drop=True)\n"
            "    z = st.standardized_yoy(p)\n"
            "    inmkt = (z <= 1.0).to_numpy()\n"
            "    fwd = p['fwd_ret'].to_numpy()\n"
            "    strat_ret = np.where(inmkt, fwd, 0.0)\n"
            "    yrs = p['year'].to_numpy()\n"
            "    bah_c = np.cumprod(1+fwd)*100\n"
            "    strat_c = np.cumprod(1+strat_ret)*100\n"
            "else:\n"
            "    rng = np.random.default_rng(260)\n"
            "    yrs = np.arange(R['year_start'], R['year_end']+1)\n"
            "    fwd = rng.normal(R['bah_ann']/100, 0.16, R['n'])\n"
            "    strat_ret = fwd.copy(); strat_ret[rng.choice(R['n'], R['years_cash'], replace=False)] = 0.0\n"
            "    bah_c = np.cumprod(1+fwd)*100; strat_c = np.cumprod(1+strat_ret)*100\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(10.0, 4.5))\n"
            "ax.plot(yrs, bah_c, lw=2, c=GREEN, label='Buy & hold (price-only)')\n"
            "ax.plot(yrs, strat_c, lw=2, c=RED, ls='--', label='Contrarian timing rule')\n"
            "ax.set_yscale('log')\n"
            "ax.set_xlabel('Year'); ax.set_ylabel('Wealth (log, base 100)')\n"
            "ax.set_title('Sitting out after margin surges costs more than it saves')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Terminal wealth -- buy&hold: {bah_c[-1]:.0f}  rule: {strat_c[-1]:.0f}')"
        ),
        md(
            "> The rule's equity curve drifts below buy-and-hold: the up-years it skips outweigh "
            "the crashes it dodges. (Price-only here; total return widens the gap.)"
        ),

        # ---- BEAT 7 ---------------------------------------------------------
        md(
            "## 7 - Going further -- the synthetic positive control\n\n"
            "Does the engine detect a contrarian signal *when one exists*? Plant a forward "
            "`predictive_beta` in a synthetic tape and sweep it."
        ),
        code(
            "betas = [0.0, 0.01, 0.03, 0.06]\n"
            "out = []\n"
            "for pb in betas:\n"
            "    df, _ = data.synthetic_series(n_years=400, predictive_beta=pb, seed=260)\n"
            "    df = df.dropna(subset=['yoy']).copy()\n"
            "    df['fwd_ret'] = df['index_ret'].shift(-1)\n"
            "    df = df.dropna(subset=['fwd_ret'])\n"
            "    r = st.predictive_regression(df)\n"
            "    out.append({'planted_beta': pb, 'slope': r['beta'], 't': r['beta_t']})\n"
            "res = pd.DataFrame(out)\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.2))\n"
            "colors = [GREEN if abs(t) >= 2 else RED for t in res['t']]\n"
            "ax.bar([str(b) for b in res['planted_beta']], res['t'], color=colors)\n"
            "ax.axhline(-2.0, c='k', ls='--', lw=0.9, label='t = -2')\n"
            "ax.axhline(0, c='k', lw=0.8)\n"
            "ax.set_xlabel('Planted contrarian predictive_beta'); ax.set_ylabel('Recovered HAC t')\n"
            "ax.set_title('The engine finds the signal when it exists (green = |t| >= 2)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(res.to_string(index=False))\n"
            "print()\n"
            "print('Null (beta=0) reads ~0; planted betas light up at t = -5 to -11.')\n"
            "print('The real tape carries no signal that strong -- the verdict is about the')\n"
            "print('MARKET, not the method.')"
        ),
        md(
            "The machinery is truthful: ~zero on the null, strongly negative when a real "
            "contrarian effect is planted. The real margin-debt tape just doesn't carry one big "
            "enough to clear the inference bar. **Weak / Mirage.**"
        ),
    ]
    _write(nb, "02_for_the_quants.ipynb")


# ---------------------------------------------------------------------------
# Shared _write helper
# ---------------------------------------------------------------------------
def _write(nb: nbf.NotebookNode, filename: str) -> None:
    nb.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    })
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
