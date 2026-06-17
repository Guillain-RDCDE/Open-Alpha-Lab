"""Generate the two narrative notebooks for Study 297 (March-Madness).

    python notebooks/build_notebooks.py

This writes AND executes both notebooks (nbconvert, in-process). The hardcoded
NCAA tournament-window table and the synthetic generator run anywhere, offline and
deterministically; the real ^GSPC cells use the repo-level cache
(_cache/^GSPC_split_only.parquet) if present, and otherwise quote the frozen
headline numbers in ``R`` (mirroring docs/results.md), so the notebook re-runs for
any reader offline.

The _write convention (each build_*() ends by calling _write) is kept so the repo's
intro-restyle tooling can monkeypatch it. _execute() runs nbconvert with no
--allow-errors: every code cell must produce a non-null execution_count and zero
error outputs.
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
    n=10330,
    years=41.0,
    fp="2d2855ee9407",
    n_in=503,
    frac_in_pct=4.9,
    in_mean_bps=4.65,
    out_mean_bps=4.23,
    diff_bps=0.42,
    in_hac_t=1.01,
    out_hac_t=4.16,
    welch_t=0.086,
    in_vol_pct=16.8,
    out_vol_pct=18.2,
    vol_ratio=0.849,
    levene_p=0.295,
    perm_pct=52.9,
    bah_cagr_pct=9.48,
    strat_gross_cagr_pct=9.52,
    strat_net_cagr_pct=9.50,
    excess_net_pp=0.02,
    excess_net_t=-0.101,
    n_switches=80,
    sub1_diff_bps=-3.93,
    sub1_t=-0.61,
    sub2_diff_bps=4.79,
    sub2_t=0.66,
    year_start=1985,
    year_end=2025,
)

R_CELL = """\
# Frozen headline numbers -- mirror of docs/results.md (as-of 2026-06-17)
R = dict(
    n=10330, years=41.0, fp="2d2855ee9407",
    n_in=503, frac_in_pct=4.9,
    in_mean_bps=4.65, out_mean_bps=4.23, diff_bps=0.42,
    in_hac_t=1.01, out_hac_t=4.16, welch_t=0.086,
    in_vol_pct=16.8, out_vol_pct=18.2, vol_ratio=0.849, levene_p=0.295,
    perm_pct=52.9,
    bah_cagr_pct=9.48, strat_gross_cagr_pct=9.52, strat_net_cagr_pct=9.50,
    excess_net_pp=0.02, excess_net_t=-0.101, n_switches=80,
    sub1_diff_bps=-3.93, sub1_t=-0.61, sub2_diff_bps=4.79, sub2_t=0.66,
    year_start=1985, year_end=2025,
)
"""

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root (quantlab/)
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from march_madness import data, strategy as st

def _have_real():
    try:
        data.fetch_daily()   # cache-only (fetch=False)
        return True
    except (FileNotFoundError, RuntimeError):
        return False

HAVE_REAL = _have_real()
print("Real ^GSPC cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# March-Madness -- does the NCAA tournament distract the market lower?\n"
            "### A three-week sports distraction, tested honestly, in plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Busted%3F: Yes](https://img.shields.io/badge/Busted%3F-Yes-8b949e?style=flat-square)\n\n"
            "Every March, the same headline returns: the NCAA basketball tournament costs "
            "American employers *billions* in lost productivity while half the office streams "
            "buzzer-beaters at their desks. The natural Wall-Street corollary writes itself -- "
            "if the whole country is distracted for three weeks, the stock market must drift, "
            "sag, or get choppy. This notebook asks the only question that matters: does the "
            "S&P 500 actually behave differently during the bracket, or is March Madness "
            "just March?\n\n"
            "> **This is the plain-language layer.** Want the HAC t-stat, the variance test, "
            "and the permutation null? That's "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** -- same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart is drawn "
            "by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT + "\n" + R_CELL),

        # ---- BEAT 0 -- VERDICT ----------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Are returns lower during the tournament? | **No.** In-window days earn "
            "**+4.65 bps/day** -- *above* the +4.23 bps the rest of the year earns. The "
            "gap is +0.42 bps, the wrong sign for the folklore, with t = 0.09. |\n"
            "| Is the market more chaotic (volatile)? | **No.** In-window volatility is "
            "**16.8%** vs 18.2% outside -- a ratio of 0.85. The bracket window is *calmer*. |\n"
            "| Are these particular days special at all? | **No.** A random 5% of days beats "
            "the tournament mean **53%** of the time -- it sits at the median of pure chance. |\n"
            "| Could you trade it? | **No.** 'Hold cash during the bracket' ties buy-and-hold "
            "(+0.02pp/yr) and you pay commissions for the privilege. |\n\n"
            "> March Madness distracts office workers, not the people who set S&P 500 futures "
            "prices. There is no market in the madness."
        ),

        # ---- BEAT 1 -- THE CLAIM --------------------------------------------
        md(
            "## 1 - The claim\n\n"
            "> *\"During the three weeks of March Madness, America stops working and starts "
            "watching basketball. Volume thins, attention wanders, and the market drifts "
            "lower -- a distraction you can see in the tape.\"*\n\n"
            "The cultural anchor is the annual Challenger, Gray & Christmas press release "
            "estimating billions in lost workplace productivity each March. It's catnip for "
            "a market-folklore take. But lost desk-hours are not the same as a mispriced "
            "index -- and that gap is exactly what we test."
        ),

        # ---- BEAT 2 -- SO WHAT ----------------------------------------------
        md(
            "## 2 - So what?\n\n"
            "If a sports tournament reliably pushed the S&P down for three weeks, it would be "
            "a free calendar trade: short in mid-March, cover in early April, every year. "
            "Markets would be very strange indeed if a basketball bracket -- known years in "
            "advance, on the calendar for everyone -- left a persistent, harvestable footprint."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW ----------------------------------------
        md(
            "## 3 - How would we even know?\n\n"
            "Three traps to avoid:\n\n"
            "1. **The direction.** The claim is *lower* returns. We must check the sign, not "
            "just the magnitude -- and report it honestly even when it points the wrong way.\n"
            "2. **Tiny window.** Each tournament is ~3 weeks; over 40 brackets (2020 was "
            "cancelled) that's only ~500 trading days -- about 5% of the tape. With ~1%/day "
            "noise, the standard error on the in-window mean is ~0.5 bps, so only a big drag "
            "would ever be visible.\n"
            "3. **Is it special?** Any random 5% of days will look a little high or a little "
            "low. The right question is whether *these* days are unusual versus a random "
            "window of the same size.\n\n"
            "We test on ^GSPC daily price returns, 1985-2025 (the 64-team bracket era), "
            "with the tournament windows hardcoded in `data.py`."
        ),

        # ---- BEAT 4 -- THE TEARDOWN -----------------------------------------
        md(
            "## 4 - The teardown -- let's actually look\n\n"
            "**First, meet the windows:** every tournament's first-round Thursday through "
            "championship Monday, hardcoded, and the share of trading days they cover."
        ),
        code(
            "tbl = data.tournament_table()\n"
            "print('Brackets covered:', len(tbl), '(1985-2025; 2020 cancelled)')\n"
            "if HAVE_REAL:\n"
            "    ret, mask = data.fetch_daily()\n"
            "    n_in = int(mask.sum()); n_tot = len(ret); frac = mask.mean()\n"
            "else:\n"
            "    n_in, n_tot, frac = R['n_in'], R['n'], R['frac_in_pct']/100\n"
            "print(f'In-tournament trading days: {n_in} / {n_tot} = {frac:.1%} of the tape')\n"
            "print()\n"
            "print(tbl.head(3).to_string(index=False))\n"
            "print('...')\n"
            "print(tbl.tail(3).to_string(index=False))"
        ),
        md("**In-window vs outside -- the headline comparison.**"),
        code(
            "if HAVE_REAL:\n"
            "    rt = st.tournament_return_test(ret, mask)\n"
            "    in_bps, out_bps, diff = rt['in_mean_bps'], rt['out_mean_bps'], rt['diff_bps']\n"
            "    welch = rt['welch_t']\n"
            "else:\n"
            "    in_bps, out_bps = R['in_mean_bps'], R['out_mean_bps']\n"
            "    diff, welch = R['diff_bps'], R['welch_t']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.0, 4.3))\n"
            "bars = ax.bar(['In tournament\\n(~3 wks x 40)', 'Rest of the year'],\n"
            "              [in_bps, out_bps], color=[AMBER, GREY], width=0.55)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('Mean daily return (bps)')\n"
            "ax.set_title(f'In-window earns MORE, not less: {diff:+.2f} bps/day  (Welch t = {welch:+.2f})')\n"
            "for b, v in zip(bars, [in_bps, out_bps]):\n"
            "    ax.annotate(f'{v:+.2f}', (b.get_x()+b.get_width()/2, v),\n"
            "                ha='center', va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'In-tournament: {in_bps:+.2f} bps/day  |  Outside: {out_bps:+.2f} bps/day')\n"
            "print(f'Difference: {diff:+.2f} bps/day -- the WRONG sign for a distraction drag.')"
        ),
        md(
            "The distraction story predicts in-window returns *below* the rest of the tape. "
            "Instead they're a hair **higher** (+0.42 bps/day), and the difference "
            "(Welch t = +0.09) is statistically indistinguishable from zero. The folklore is "
            "not just unsupported -- the only non-zero direction it shows is backwards."
        ),
        md("**Is the market more chaotic during the bracket?**"),
        code(
            "if HAVE_REAL:\n"
            "    vt = st.tournament_vol_test(ret, mask)\n"
            "    in_vol, out_vol, ratio = vt['in_vol_ann']*100, vt['out_vol_ann']*100, vt['vol_ratio']\n"
            "else:\n"
            "    in_vol, out_vol, ratio = R['in_vol_pct'], R['out_vol_pct'], R['vol_ratio']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.0, 4.3))\n"
            "bars = ax.bar(['In tournament', 'Rest of the year'], [in_vol, out_vol],\n"
            "              color=[AMBER, GREY], width=0.55)\n"
            "ax.set_ylabel('Annualised volatility (%)')\n"
            "ax.set_title(f'No chaos: in/out vol ratio = {ratio:.2f} (the window is CALMER)')\n"
            "for b, v in zip(bars, [in_vol, out_vol]):\n"
            "    ax.annotate(f'{v:.1f}%', (b.get_x()+b.get_width()/2, v),\n"
            "                ha='center', va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'In-window vol {in_vol:.1f}% vs outside {out_vol:.1f}% -- ratio {ratio:.2f}')\n"
            "print('The madness-is-chaos corollary fails too: the window is slightly calmer.')"
        ),
        md(
            "The 'madness = chaos' corollary fails as well. In-window volatility is **16.8%** "
            "vs **18.2%** outside -- a ratio of 0.85. If anything, the bracket weeks are a "
            "touch calmer than the rest of the year."
        ),

        # ---- BEAT 5 -- THE VERDICT ------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            "- **Signal -- None.** In-window mean **+0.42 bps/day** (Welch t = **+0.09**) -- "
            "the wrong sign and indistinguishable from zero. Volatility ratio **0.85** "
            "(calmer, not chaotic). A random window of the same size beats it **53%** of the "
            "time. Returns are price-only on a single broad index, so survivorship isn't a "
            "factor -- there's simply nothing here.\n"
            "- **Tradability -- Mirage.** Holding cash during the bracket ties buy-and-hold "
            "(+0.02pp/yr after a one-day lag and costs). No vehicle, no edge.\n"
            "- **Busted.** The productivity-loss headlines are real; the market impact is not. "
            "The marginal price-setter in S&P 500 futures is not filling out a bracket."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT -----------------------------------
        md(
            "## 6 - Could you actually trade it?\n\n"
            "The most generous trade: hold cash during the tournament window, stay long the "
            "index otherwise (entering/exiting one day late, paying a small cost each switch)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    av = st.tournament_avoidance(ret, mask, lag=1, cost_bps=1.0)\n"
            "    bah, net, exc, t = (av['bah_cagr'], av['strat_net_cagr'],\n"
            "                        av['excess_net_cagr'], av['excess_net_hac_t'])\n"
            "else:\n"
            "    bah, net = R['bah_cagr_pct']/100, R['strat_net_cagr_pct']/100\n"
            "    exc, t = R['excess_net_pp']/100, R['excess_net_t']\n"
            "\n"
            "print(f'Buy & hold (price-only):     {bah:.2%}/yr')\n"
            "print(f'Avoidance, net of costs:     {net:.2%}/yr')\n"
            "print(f'Edge from sitting out March: {exc*100:+.2f} pp/yr  (HAC t = {t:+.2f})')\n"
            "print()\n"
            "print('A statistical tie -- and the index drifts UP, so stepping out of ~5% of')\n"
            "print('days that were marginally positive on average gives back what it saves.')"
        ),
        md(
            "The avoidance strategy is a dead heat with buy-and-hold. Because the index drifts "
            "up and the in-window days were marginally *positive*, sitting them out forgoes "
            "about as much as it avoids -- and the commissions finish the job. There is no "
            "trade in the bracket."
        ),

        # ---- BEAT 7 -- GOING FURTHER ----------------------------------------
        md(
            "## 7 - Going further\n\n"
            "- **The positive control:** the companion notebook plants a downward tournament "
            "drag into a synthetic tape and shows the same machinery detects it once it's big "
            "enough. The real tape has nothing like it.\n"
            "- **Sister folklore:** the [Super-Bowl Indicator (Study 158)](../../158-super-bowl/) "
            "and [Mercury-Retrograde (Study 164)](../../164-mercury-retrograde/) are the same "
            "in-window-vs-out discipline applied to other omens.\n\n"
            "*Think the bracket really moves the tape? Fork this, pick your own window "
            "definition, and show an in/out gap that clears |t| = 2 on the real ^GSPC tape. "
            "That's the bar -- and the data points the other way.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "01_for_the_curious.ipynb")


# ===========================================================================
# 02 -- FOR THE QUANTS
# ===========================================================================
def build_quants():
    cells = [
        md(
            "# March-Madness -- a quantitative teardown\n"
            "### ^GSPC daily 1985-2025 * in/out HAC t * Levene variance * "
            "random-placement permutation * lagged+cost avoidance * power\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Busted%3F: Yes](https://img.shields.io/badge/Busted%3F-Yes-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) -- "
            "*same seven beats, every claim carrying its standard error.* We test whether the "
            "NCAA tournament window changes the mean daily return (Newey-West HAC t-stat) or "
            "the variance (Levene), run a random-placement permutation null, build a lagged + "
            "cost-charged avoidance strategy, and close with a power calculation and a "
            "synthetic positive control.\n\n"
            "> **Not investment advice.** Real data: ^GSPC daily **price** return "
            "(split-adjusted, no dividends), 1985-2025; tournament windows hardcoded in "
            "`data.py`. Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md)."
        ),
        code(BOOT + "\n" + R_CELL + "\nfrom scipy import stats as scipy_stats\n"),

        # ---- BEAT 0 ---------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            "| **Signal** | `NONE` | In-window mean **+0.42 bps/day** (Welch t = **+0.09**, "
            "wrong sign); vol ratio **0.85** (Levene p = **0.30**); permutation p = **0.53**. |\n"
            "| **Tradability** | `MIRAGE` | Lagged + cost-charged avoidance ties buy-and-hold "
            "(**+0.02 pp/yr**, HAC t = **−0.10**). |\n"
            "| **Busted?** | `YES` | The sub-period in/out gap flips sign (−3.9 bps then "
            "+4.8 bps) -- the signature of a non-effect. |\n"
        ),

        # ---- BEAT 1 ---------------------------------------------------------
        md(
            "## 1 - The claim, steelmanned\n\n"
            "Let $r_t$ be the daily ^GSPC return and $T_t \\in \\{0,1\\}$ the in-tournament "
            "indicator. The hypotheses:\n\n"
            "- **H1 (mean drag).** $E[r_t \\mid T_t=1] < E[r_t \\mid T_t=0]$, with the "
            "difference clearing $|t| \\ge 2$ under a HAC standard error.\n"
            "- **H2 (chaos).** $\\mathrm{Var}[r_t \\mid T_t=1] > \\mathrm{Var}[r_t \\mid T_t=0]$ "
            "(vol ratio meaningfully > 1).\n"
            "- **H3 (specialness).** The in-window mean is unusually low versus a random "
            "window of the same size (low permutation p).\n\n"
            "We reject H1, H2, and H3 on the real tape. H1 even points the wrong way."
        ),

        # ---- BEAT 2 ---------------------------------------------------------
        md(
            "## 2 - So what? -- what rides on each answer\n\n"
            "March Madness is a textbook *attention/distraction* story. If H1-H3 held it would "
            "be a clean, pre-scheduled calendar trade. The interesting finding is how a "
            "culturally enormous event leaves *no* fingerprint on a deep, liquid index -- a "
            "useful reminder that 'everyone is watching' and 'prices move' are different claims."
        ),

        # ---- BEAT 3 ---------------------------------------------------------
        md(
            "## 3 - How we'd know -- the protocol\n\n"
            "- **Data.** ^GSPC daily close-to-close price return (split-adjusted, no "
            "dividends), 1985-2025; tournament windows (first-round Thu -> championship Mon) "
            "hardcoded in `data.py`. 2020 absent (tournament cancelled).\n"
            "- **Label.** `in_tournament` boolean per trading day (~5% of the tape).\n"
            "- **Mean test.** Newey-West HAC t on each group's mean; Welch t on the difference.\n"
            "- **Variance test.** Annualised vol ratio + Levene equality-of-variance test.\n"
            "- **Permutation.** 5,000 random subsets of the same size; one-sided p = fraction "
            "with mean <= the real in-window mean.\n"
            "- **Tradability.** Hold-cash-in-window strategy, **one-day execution lag**, "
            "**1 bp one-way cost** per switch; gross and net reported.\n"
            "- **Positive control.** Synthetic tape with a planted in-window drag.\n"
        ),

        # ---- BEAT 4 ---------------------------------------------------------
        md("## 4 - The teardown"),
        md(
            "### 4a - Mean return: in-window vs outside, with HAC t\n\n"
            "The headline test. The distraction claim needs a negative difference that clears "
            "|t| = 2."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ret, mask = data.fetch_daily()\n"
            "    rt = st.tournament_return_test(ret, mask)\n"
            "    n_in, n_out = rt['n_in'], rt['n_out']\n"
            "    in_bps, out_bps, diff = rt['in_mean_bps'], rt['out_mean_bps'], rt['diff_bps']\n"
            "    in_t, out_t, welch = rt['in_t'], rt['out_t'], rt['welch_t']\n"
            "    fp = data.fingerprint(ret)\n"
            "else:\n"
            "    n_in, n_out = R['n_in'], R['n']-R['n_in']\n"
            "    in_bps, out_bps, diff = R['in_mean_bps'], R['out_mean_bps'], R['diff_bps']\n"
            "    in_t, out_t, welch = R['in_hac_t'], R['out_hac_t'], R['welch_t']\n"
            "    fp = R['fp']\n"
            "\n"
            "print(f'Data fingerprint: {fp}  (expect {R[\"fp\"]})')\n"
            "print(f'n in-window: {n_in}   n outside: {n_out}')\n"
            "print(f'In-window mean:  {in_bps:+.2f} bps/day   (HAC t {in_t:+.2f})')\n"
            "print(f'Outside mean:    {out_bps:+.2f} bps/day   (HAC t {out_t:+.2f})')\n"
            "print(f'Difference:      {diff:+.2f} bps/day      Welch t {welch:+.3f}')\n"
            "print()\n"
            "print('The difference is positive (wrong sign for a distraction drag) and |t|~0.')"
        ),
        md(
            "> In-window returns are **+0.42 bps/day** *above* the rest of the tape, "
            "Welch t = **+0.09**. The full-tape outside mean is itself a healthy +4.2 bps/day "
            "with HAC t = +4.2 (that's just the equity drift). The tournament adds nothing -- "
            "and what little it adds is the opposite of the folklore."
        ),
        md(
            "### 4b - The 'chaos' corollary: variance ratio + Levene test\n\n"
            "Does volatility rise during the bracket?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    vt = st.tournament_vol_test(ret, mask)\n"
            "    in_vol, out_vol = vt['in_vol_ann']*100, vt['out_vol_ann']*100\n"
            "    ratio, lev_p = vt['vol_ratio'], vt['levene_p']\n"
            "    in_r = ret[mask].to_numpy()*100; out_r = ret[~mask].to_numpy()*100\n"
            "else:\n"
            "    in_vol, out_vol = R['in_vol_pct'], R['out_vol_pct']\n"
            "    ratio, lev_p = R['vol_ratio'], R['levene_p']\n"
            "    rng = np.random.default_rng(297)\n"
            "    in_r = rng.normal(0, in_vol/np.sqrt(252), R['n_in'])\n"
            "    out_r = rng.normal(0, out_vol/np.sqrt(252), 4000)\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.boxplot([in_r, out_r], tick_labels=['In tournament', 'Outside'],\n"
            "           patch_artist=True, showfliers=False,\n"
            "           boxprops=dict(facecolor=AMBER, alpha=0.5),\n"
            "           medianprops=dict(color='black', lw=2))\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('Daily return (%)')\n"
            "ax.set_title(f'Vol ratio {ratio:.2f} (<1 = calmer)   Levene p = {lev_p:.3f}')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'In-window vol {in_vol:.1f}% vs outside {out_vol:.1f}%  ->  ratio {ratio:.3f}')\n"
            "print(f'Levene equality-of-variance p = {lev_p:.3f} -- no significant difference.')"
        ),
        md(
            "Vol ratio **0.85** (< 1 -- *calmer*), Levene p = **0.30**. The chaos corollary "
            "fails: the tournament weeks are statistically as volatile as any other, if "
            "anything slightly less."
        ),
        md(
            "### 4c - Random-placement permutation null\n\n"
            "Are *these* ~5% of days special? Draw random windows of the same size and locate "
            "the real in-window mean in the distribution."
        ),
        code(
            "if HAVE_REAL:\n"
            "    perm = st.permutation_null(ret, mask, n_perm=5000, seed=297)\n"
            "    pct = perm['pct_worse_or_equal']*100\n"
            "    in_mean = perm['in_mean_bps']\n"
            "    rng = np.random.default_rng(297)\n"
            "    vals = ret.dropna().to_numpy(); n_tot=len(vals); n_in=int(mask.sum())\n"
            "    draws = np.array([vals[rng.choice(n_tot, n_in, replace=False)].mean()*1e4\n"
            "                      for _ in range(3000)])\n"
            "else:\n"
            "    pct = R['perm_pct']; in_mean = R['in_mean_bps']\n"
            "    rng = np.random.default_rng(297)\n"
            "    draws = rng.normal(R['out_mean_bps'], 100/np.sqrt(R['n_in']), 3000)\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=40, color=GREY, alpha=0.7, label='Random windows (same size)')\n"
            "ax.axvline(in_mean, c=RED, lw=2.5, label=f'Real in-window mean: {in_mean:+.2f} bps')\n"
            "ax.set_xlabel('Mean daily return (bps)'); ax.set_ylabel('Count')\n"
            "ax.set_title(f'The tournament window sits at the {pct:.0f}th percentile -- i.e. the median')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Permutation p (random mean <= real): {pct:.1f}%')\n"
            "print('The real window is indistinguishable from a random 5% of trading days.')"
        ),
        md(
            "> The real in-window mean lands at the **53rd percentile** of the random-null "
            "distribution -- dead center. There is nothing distinctive about the calendar days "
            "the tournament happens to occupy."
        ),
        md(
            "### 4d - Sub-period stability: the sign flips\n\n"
            "Split the tape at its midpoint and re-run the in/out mean test."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for lo, hi in [(1985, 2004), (2005, 2025)]:\n"
            "        sub = ret[(ret.index.year>=lo) & (ret.index.year<=hi)]\n"
            "        m = data.is_tournament(pd.DatetimeIndex(sub.index))\n"
            "        r = st.tournament_return_test(sub, m)\n"
            "        rows.append({'period': f'{lo}-{hi}', 'n_in': r['n_in'],\n"
            "                     'diff_bps': round(r['diff_bps'],2), 'welch_t': round(r['welch_t'],2)})\n"
            "    sub_df = pd.DataFrame(rows)\n"
            "else:\n"
            "    sub_df = pd.DataFrame({\n"
            "        'period': ['1985-2004','2005-2025'],\n"
            "        'n_in': [253, 250],\n"
            "        'diff_bps': [R['sub1_diff_bps'], R['sub2_diff_bps']],\n"
            "        'welch_t': [R['sub1_t'], R['sub2_t']]})\n"
            "\n"
            "print(sub_df.to_string(index=False))\n"
            "print()\n"
            "print('The in/out gap is NEGATIVE in the first half and POSITIVE in the second,')\n"
            "print('each well inside noise -- the fingerprint of a non-effect, not a regime.')"
        ),
        md(
            "### 4e - Power: what drag could n~500 in-window days detect?\n\n"
            "Minimum detectable in/out gap at |t| = 2."
        ),
        code(
            "vol_daily = 0.011   # ~1.1%/day\n"
            "n_in = R['n_in']\n"
            "se_in = vol_daily / np.sqrt(n_in)          # SE of in-window mean\n"
            "mde_bps = 2 * se_in * 1e4                   # |t|=2 threshold, in bps/day\n"
            "print(f'In-window days: {n_in},  daily vol ~{vol_daily:.1%}')\n"
            "print(f'SE of in-window mean: {se_in*1e4:.2f} bps/day')\n"
            "print(f'Min detectable in/out gap at |t|=2: ~{mde_bps:.1f} bps/day')\n"
            "print(f'   = ~{mde_bps*13:.0f} bps over a 13-trading-day window')\n"
            "print()\n"
            "print(f'Observed gap: {R[\"diff_bps\"]:+.2f} bps/day -- and POSITIVE.')\n"
            "print('A real distraction drag of a few bps/window would be invisible here.')"
        ),

        # ---- BEAT 5 ---------------------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            "- **Signal `NONE`** -- in/out gap +0.42 bps/day (Welch t = +0.09, wrong sign), "
            "vol ratio 0.85 (Levene p = 0.30), permutation p = 0.53, sub-period sign flip. "
            "Price-only, single index -- survivorship is not in play.\n"
            "- **Tradability `MIRAGE`** -- lagged + cost-charged avoidance ties buy-and-hold.\n"
            "- **Busted** -- the productivity-loss headlines don't translate into index moves."
        ),

        # ---- BEAT 6 ---------------------------------------------------------
        md(
            "## 6 - Could you trade it? -- the lagged, cost-charged benchmark\n\n"
            "Hold cash in the window (one-day execution lag), long otherwise, 1 bp one-way "
            "cost per switch."
        ),
        code(
            "if HAVE_REAL:\n"
            "    av = st.tournament_avoidance(ret, mask, lag=1, cost_bps=1.0)\n"
            "    bah, gross, net = av['bah_cagr'], av['strat_gross_cagr'], av['strat_net_cagr']\n"
            "    exc, t, nsw = av['excess_net_cagr'], av['excess_net_hac_t'], av['n_switches']\n"
            "    eq_bah = (1+av['bah_ret']).cumprod(); eq_str = (1+av['strat_ret']).cumprod()\n"
            "    years = eq_bah.index\n"
            "else:\n"
            "    bah, gross, net = R['bah_cagr_pct']/100, R['strat_gross_cagr_pct']/100, R['strat_net_cagr_pct']/100\n"
            "    exc, t, nsw = R['excess_net_pp']/100, R['excess_net_t'], R['n_switches']\n"
            "    rng = np.random.default_rng(297)\n"
            "    idx = pd.bdate_range('1985-01-02', periods=R['n'])\n"
            "    eq_bah = pd.Series((1+rng.normal(bah/252,0.011,R['n'])).cumprod(), index=idx)\n"
            "    eq_str = pd.Series((1+rng.normal(net/252,0.011,R['n'])).cumprod(), index=idx)\n"
            "    years = idx\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(10.0, 4.5))\n"
            "ax.plot(years, eq_bah.values, lw=2, c=GREEN, label=f'Buy & hold: {bah:.2%}/yr')\n"
            "ax.plot(years, eq_str.values, lw=1.5, c=RED, ls='--', label=f'Avoidance (net): {net:.2%}/yr')\n"
            "ax.set_yscale('log'); ax.set_xlabel('Year')\n"
            "ax.set_ylabel('Growth of $1 (log)')\n"
            "ax.set_title(f'A dead heat: excess {exc*100:+.2f} pp/yr (HAC t {t:+.2f}), {nsw} switches')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Buy & hold: {bah:.2%}/yr  |  Avoidance net: {net:.2%}/yr')\n"
            "print(f'Excess: {exc*100:+.2f} pp/yr (HAC t {t:+.2f}) over {nsw} round-trips -- no edge.')"
        ),
        md(
            "> The two curves overlay almost perfectly. Sitting out ~5% of (marginally "
            "positive) days in an upward-drifting index is a wash, and the costs tip it "
            "negative. No tradable signal."
        ),

        # ---- BEAT 7 ---------------------------------------------------------
        md(
            "## 7 - Going further -- the synthetic positive control\n\n"
            "Does the machinery detect a tournament drag *when one exists*? Plant a per-day "
            "in-window penalty into the offline synthetic tape and sweep its size."
        ),
        code(
            "planted = [0.0, -5.0, -15.0, -30.0]\n"
            "rows = []\n"
            "for bp in planted:\n"
            "    s_ret, s_mask, _ = data.synthetic_daily(n_years=41, madness_penalty_bp=bp, seed=297)\n"
            "    r = st.tournament_return_test(s_ret, s_mask)\n"
            "    rows.append({'planted_bp_day': bp, 'diff_bps': round(r['diff_bps'],2),\n"
            "                 'welch_t': round(r['welch_t'],2)})\n"
            "res = pd.DataFrame(rows)\n"
            "\n"
            "colors = [GREEN if abs(row['welch_t'])>=2 else RED for _, row in res.iterrows()]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.bar([str(b) for b in planted], res['welch_t'], color=colors)\n"
            "ax.axhline(-2, c='k', ls=':', lw=1, label='|t| = 2')\n"
            "ax.axhline(2, c='k', ls=':', lw=1)\n"
            "ax.set_xlabel('Planted in-window drag (bps/day)'); ax.set_ylabel('Welch t')\n"
            "ax.set_title('The engine finds a drag once it is big enough (green = |t|>=2)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(res.to_string(index=False))\n"
            "print()\n"
            "print('Real tape: diff +0.42 bps/day, Welch t +0.09 -- nowhere near, wrong sign.')"
        ),
        md(
            "The engine cleanly detects a planted drag once it reaches ~20-30 bps/day. The "
            "real tape shows +0.42 bps/day with t = +0.09. The verdict is a statement about "
            "**the market**, not the method: March Madness leaves no footprint on the index."
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


def _execute(name):
    """Execute a notebook in-place with nbconvert; no --allow-errors."""
    from nbconvert.preprocessors import ExecutePreprocessor

    path = os.path.join(HERE, name)
    nb = nbf.read(path, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": HERE}})
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("executed", path)


if __name__ == "__main__":
    build_curious()
    build_quants()
    _execute("01_for_the_curious.ipynb")
    _execute("02_for_the_quants.ipynb")
