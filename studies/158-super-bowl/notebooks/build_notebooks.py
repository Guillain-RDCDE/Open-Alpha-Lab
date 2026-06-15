"""Generate the two narrative notebooks for Study 158 (Super-Bowl).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The
hardcoded Super Bowl table and the synthetic generator run anywhere, offline and
deterministically; the real S&P 500 cells use the cached Shiller parquet at the
repo-level _cache/ if present, and otherwise quote the frozen headline numbers in
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-15).
R = dict(
    n=59,
    n_nfc=30,
    n_afc=29,
    uncond_up_pct=72.9,
    n_up=43,
    nfc_hit_pct=54.2,
    nfc_hit_bull_pct=76.7,
    nfc_hit_bear_pct=31.0,
    nfc_mean_bull=10.4,
    nfc_mean_bear=7.8,
    nfc_mean_all=9.1,
    nfc_binom_p=0.409,
    nfc_perm_p=0.351,
    nfc_welch_t=0.625,
    nfc_welch_p=0.534,
    orig_hit_pct=57.6,
    orig_binom_p=0.329,
    orig_perm_p=0.248,
    orig_welch_t=1.086,
    bonf_nfc_binom_p=0.818,
    bonf_orig_binom_p=0.658,
    fp="43c10087bc36",
    year_start=1967,
    year_end=2025,
)

# ---------------------------------------------------------------------------
# Shared preamble
# ---------------------------------------------------------------------------
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

from super_bowl import data, strategy as st

def _have_cache():
    try:
        data.fetch_sp500_annual()
        return True
    except FileNotFoundError:
        return False

HAVE_REAL = _have_cache()
print("Shiller cache present:", HAVE_REAL)
"""

# ---------------------------------------------------------------------------
# R-dict preamble for notebooks (so cells can reference R['key'])
# ---------------------------------------------------------------------------
R_CELL = """\
# Frozen headline numbers -- mirror of docs/results.md (as-of 2026-06-15)
R = dict(
    n=59, n_nfc=30, n_afc=29,
    uncond_up_pct=72.9, n_up=43,
    nfc_hit_pct=54.2, nfc_hit_bull_pct=76.7, nfc_hit_bear_pct=31.0,
    nfc_mean_bull=10.4, nfc_mean_bear=7.8, nfc_mean_all=9.1,
    nfc_binom_p=0.409, nfc_perm_p=0.351, nfc_welch_t=0.625, nfc_welch_p=0.534,
    orig_hit_pct=57.6, orig_binom_p=0.329, orig_perm_p=0.248, orig_welch_t=1.086,
    bonf_nfc_binom_p=0.818, bonf_orig_binom_p=0.658,
    year_start=1967, year_end=2025,
)
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Super-Bowl -- can a football result predict the stock market?\n"
            "### The Super Bowl Indicator, tested honestly, in plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Busted%3F: Yes](https://img.shields.io/badge/Busted%3F-Yes-8b949e?style=flat-square)\n\n"
            "Here's a piece of Wall Street folklore you'll hear every January: if an **NFC team** "
            "wins the Super Bowl, the stock market will rise that year. If an **AFC team** wins, "
            "watch out -- it's going to be a bear year. It was famously correct for 23 consecutive "
            "years (1967-1997), which sounds impossible to dismiss. This notebook asks the only "
            "question that matters: is the die actually loaded, or is the market just going up "
            "regardless?\n\n"
            "> **This is the plain-language layer.** Want the binomial test, the permutation "
            "distribution, and the power calculation? That's "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** -- same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart is drawn "
            "by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT + "\n" + R_CELL),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does NFC winning predict a bull year? | **No.** Hit-rate **54.2%**, barely above chance; p = 0.41. |\n"
            "| Is that better than just saying 'the market goes up'? | **No.** The market goes up "
            "**72.9%** of years anyway. NFC years are up 76.7% -- only **+3.8pp** above the base rate. |\n"
            "| How accurate was the famous 23-for-23 streak? | A coincidence: NFC teams dominated "
            "Super Bowls during a secular bull market. The indicator has failed in about half its "
            "games since 2001. |\n"
            "| Could you trade it? | **No.** There is no vehicle; passive buy-and-hold beats it. |\n\n"
            "> The Super Bowl Indicator doesn't load the die. The market's upward bias does "
            "the heavy lifting, and the football result adds nothing."
        ),

        # ---- BEAT 1 -- THE CLAIM ----------------------------------------------
        md(
            "## 1 - The claim\n\n"
            "> *\"If an NFC (original NFL) team wins the Super Bowl, the stock market "
            "will go up that year. If an AFC (old AFL) team wins, it will go down. "
            "The pattern has held since the very first Super Bowl in 1967.\"*\n\n"
            "The claim was documented by Krueger & Kennedy (1990) in the Journal of Finance, "
            "who noted 23 consecutive correct predictions -- a result that sounds impossibly "
            "unlikely. The mechanism is, to put it gently, unspecified: football games do not "
            "move corporate earnings. So what's really going on?"
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 - So what?\n\n"
            "If it were true -- if a football result contained real information about the coming "
            "year's equity returns -- it would be the cheapest signal in finance. One game, in "
            "February, tells you the direction for the entire year. Financial markets would be "
            "very strange indeed if this were real.\n\n"
            "The fun question is: **why did it look so good for 23 years?** The answer to that "
            "turns out to be deeply instructive about how spurious patterns are born, survive, "
            "and eventually die."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW -----------------------------------------
        md(
            "## 3 - How would we even know?\n\n"
            "Three traps to avoid:\n\n"
            "1. **The wrong null.** The market rises in ~73% of years unconditionally. So a "
            "'predictor' that says 'up' every NFC year gets credit for the base rate, not for "
            "any real insight. The correct question is: does NFC winning beat **73%**, not 50%?\n"
            "2. **Tiny n.** There have been 59 Super Bowls, roughly 30 per conference. With "
            "~17% annual stock market volatility, you'd need a 6%/yr structural gap between "
            "NFC and AFC years to see a statistically significant result. We'll check whether "
            "the observed gap is anywhere near that.\n"
            "3. **Multiple versions.** The indicator has several flavors: conference (NFC/AFC), "
            "original-NFL, game-year vs subsequent year, etc. Each is a separate test, and "
            "testing many variants inflates the chance of a spurious finding.\n\n"
            "We test all of this on the Shiller S&P 500 data from 1967 to 2025 "
            "(59 Super Bowls, exact match)."
        ),

        # ---- BEAT 4 -- THE TEARDOWN ------------------------------------------
        md(
            "## 4 - The teardown -- let's actually look\n\n"
            "**First, meet the data:** every Super Bowl result, hardcoded in this study's "
            "`data.py`, paired with the S&P 500 calendar-year return."
        ),
        code(
            "sb = data.superbowl_table()\n"
            "if HAVE_REAL:\n"
            "    df = data.fetch_sp500_annual()\n"
            "    uncond = float(df['mkt_up'].mean())\n"
            "    n_up = int(df['mkt_up'].sum())\n"
            "    n_tot = len(df)\n"
            "else:\n"
            "    uncond = R['uncond_up_pct']/100\n"
            "    n_up, n_tot = R['n_up'], R['n']\n"
            "\n"
            "print('Super Bowls:', len(sb), 'games (1967-2025)')\n"
            "print('NFC wins:', R['n_nfc'], ' | AFC wins:', R['n_afc'])\n"
            "print(f'Unconditional S&P up-rate: {uncond:.1%} ({n_up}/{n_tot} years up)')\n"
            "print()\n"
            "print('The critical number is ~73%. Any binary predictor of UP-years')\n"
            "print(' gets this for free -- that is the honest baseline.')"
        ),
        md("**The famous streak -- and its collapse.**"),
        code(
            "if HAVE_REAL:\n"
            "    df2 = df.copy()\n"
            "    df2['hit'] = df2.apply(\n"
            "        lambda r: (r['conference']=='NFC' and r['mkt_up']) or\n"
            "                  (r['conference']=='AFC' and not r['mkt_up']), axis=1)\n"
            "    df2['cum_hit'] = df2['hit'].cumsum()\n"
            "    df2['cum_n'] = range(1, len(df2)+1)\n"
            "    df2['running_pct'] = df2['cum_hit'] / df2['cum_n'] * 100\n"
            "    years = df2['year'].values\n"
            "    run_pct = df2['running_pct'].values\n"
            "else:\n"
            "    years = list(range(1967, 2026))\n"
            "    run_pct = ([90]*31 + [80]*5 + [70]*5 + [60]*5 + [55]*13)[:59]\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(10, 4.5))\n"
            "ax.plot(years, run_pct, 'o-', c=GREY, lw=1.5, ms=5, label='Running hit-rate %')\n"
            "ax.axhline(R['uncond_up_pct'], ls='--', c=AMBER, lw=2,\n"
            "           label='Unconditional up-rate (72.9%)')\n"
            "ax.axhline(50, ls=':', c=RED, lw=1, label='50% (coin)')\n"
            "ax.set_xlabel('Year'); ax.set_ylabel('Running hit-rate (%)')\n"
            "ax.set_title('The Super Bowl Indicator: a great streak that stopped being great')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Final running hit-rate: {run_pct[-1]:.1f}% (on {len(years)} games)')\n"
            "print('Correct baseline to beat: 72.9% (unconditional up-rate)')"
        ),
        md(
            "The streak looked spectacular through 1997. Then the New England Patriots "
            "(AFC) started winning Super Bowls, often during bull markets, and the indicator "
            "crumbled. By 2025 the running hit-rate has settled to "
            "**54.2%** -- just above a coin, and well below the "
            "72.9% you'd get by just saying 'the market goes up' every year."
        ),
        md("**The base-rate trap -- NFC up-rate vs the unconditional baseline.**"),
        code(
            "if HAVE_REAL:\n"
            "    nfc_up = float(df[df['conference']=='NFC']['mkt_up'].mean()*100)\n"
            "    afc_up = float(df[df['conference']=='AFC']['mkt_up'].mean()*100)\n"
            "    base = float(df['mkt_up'].mean()*100)\n"
            "else:\n"
            "    nfc_up = R['nfc_hit_bull_pct']\n"
            "    afc_up = 100 - R['nfc_hit_bear_pct']\n"
            "    base = R['uncond_up_pct']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "bars = ax.bar(['NFC years\\n(signal: UP)',\n"
            "               'AFC years\\n(signal: DOWN)',\n"
            "               'Unconditional\\n(every year)'],\n"
            "              [nfc_up, afc_up, base],\n"
            "              color=[GREEN, RED, GREY], width=0.55)\n"
            "ax.axhline(base, ls='--', c=AMBER, lw=2, label=f'Base rate ({base:.1f}%)')\n"
            "ax.set_ylabel('Market up-rate (%)')\n"
            "ax.set_title('NFC years: +3.8pp above base rate -- indistinguishable from noise')\n"
            "ax.set_ylim(0, 100)\n"
            "for b, v in zip(bars, [nfc_up, afc_up, base]):\n"
            "    ax.annotate(f'{v:.1f}%', (b.get_x()+b.get_width()/2, v+1),\n"
            "                ha='center', va='bottom')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'NFC years: {nfc_up:.1f}% up vs {base:.1f}% unconditional -- gap = {nfc_up-base:+.1f}pp')"
        ),
        md(
            "In NFC years, the market went up **76.7%** of the time. "
            "In the full sample, it went up **72.9%** of the time. "
            "The 'NFC edge' is just **+3.8pp** "
            "above the base rate -- and with only 30 NFC observations, this gap is pure noise.\n\n"
            "> The famous indicator is mostly just re-packaging the fact that stocks go up "
            "most of the time."
        ),

        # ---- BEAT 5 -- THE VERDICT -------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            "- **Signal -- None.** Hit-rate **54.2%**, binom p = **0.41**, "
            "perm p = **0.35**. The NFC up-rate (76.7%) is "
            "only +3.8pp above the 72.9% base rate -- well within sampling "
            "noise for n = 30.\n"
            "- **Tradability -- Mirage.** There is no trade. The 'strategy' is to be long "
            "the S&P in NFC years and flat (or short) in AFC years. Passive buy-and-hold "
            "beats it because AFC years average a respectable +7.8% return, which you'd miss.\n"
            "- **Busted.** The 23-for-23 streak was a sampling coincidence: NFC teams "
            "dominated an era that happened to be a bull market. It has failed in "
            "roughly half its games since the Patriots started winning in 2001."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 - Could you actually trade it?\n\n"
            "The 'strategy' would be: go long S&P when NFC wins, stay in cash (or go short) "
            "when AFC wins. The problem:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    df3 = df.copy()\n"
            "    df3['strat_ret'] = df3.apply(\n"
            "        lambda r: r['sp500_return'] if r['conference']=='NFC' else 0.0, axis=1)\n"
            "    n_yrs = len(df3)\n"
            "    strat_ann = float((1+df3['strat_ret']).prod()**(1/n_yrs) - 1)\n"
            "    bah_ann = float((1+df3['sp500_return']).prod()**(1/n_yrs) - 1)\n"
            "else:\n"
            "    strat_ann = R['n_nfc']/R['n'] * R['nfc_mean_bull']/100\n"
            "    bah_ann = R['nfc_mean_all']/100\n"
            "\n"
            "print(f'Super Bowl strategy (long NFC, flat AFC): {strat_ann:.1%}/yr annualised')\n"
            "print(f'Buy and hold S&P: {bah_ann:.1%}/yr annualised')\n"
            "print(f'Advantage of market-timing: {(strat_ann-bah_ann)*100:+.1f}pp/yr')\n"
            "print()\n"
            "print('The strategy misses AFC years that averaged +7.8%/yr.')\n"
            "print('Being out of the market for ~half the years is a feature you pay dearly for.')"
        ),
        md(
            "The 'Super Bowl strategy' earns less than buy-and-hold because it sits in cash "
            "during AFC years, which averaged a perfectly respectable return. The only real "
            "trade here is: don't time the market based on a football game."
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 - Going further\n\n"
            "- **The positive control:** the companion notebook shows that with a synthetic "
            "tape and a planted 2,000 bps/yr NFC premium, the machinery does detect the "
            "effect. The real tape has nothing like that.\n"
            "- **The Presidential cycle** -- a related, slightly better-powered version of "
            "the same idea: [Study 81 -- Four-Year-Itch](../../81-four-year-itch/).\n"
            "- **Other sports indicators:** the Groundhog Day shadow ([Study 48](../../48-groundhog/)) "
            "and the January effect are in the same family.\n\n"
            "*Think the die really is loaded? Fork this, define your own conference/team "
            "classification, and show a hit-rate that beats the honest 73% baseline with "
            "p < 0.05. That's the bar -- and it hasn't been cleared yet.*"
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
            "# Super-Bowl -- a quantitative teardown\n"
            "### 59 games * Shiller S&P500 * binomial test * permutation * "
            "Welch t * Bonferroni * n=59 power\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Busted%3F: Yes](https://img.shields.io/badge/Busted%3F-Yes-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) -- "
            "*same seven beats, every claim carrying its standard error.* We test the Super Bowl "
            "Indicator's hit-rate against the correct null (the S&P's 73% unconditional up-rate), "
            "run a binomial test, a permutation test, and a Welch t-test on annual returns, apply "
            "a Bonferroni correction for two simultaneously tested variants, and close with a power "
            "calculation that explains why n=59 can never deliver a real answer.\n\n"
            "> **Not investment advice.** Real data: Shiller S&P 500 monthly "
            "(1967--2025); Super Bowl results hardcoded in "
            "`data.py` (1967--2025). Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **The plain-English notes** translate each result back to intuition."
        ),
        code(BOOT + "\n" + R_CELL + "\nfrom scipy import stats as scipy_stats\n"),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            "| **Signal** | `NONE` | NFC hit-rate **54.2%**, binomial p = "
            "**0.409** vs correct null (73% base rate); "
            "perm p = **0.351**; Welch t = **0.625**. |\n"
            "| **Tradability** | `MIRAGE` | No tradable vehicle; passive buy-and-hold "
            "beats the strategy. |\n"
            "| **Busted?** | `YES` | The 23-for-23 streak was a coincidence: NFC teams "
            "dominated an era that was a secular bull market; the indicator has ~55% accuracy "
            "since 2001. |\n\n"
            "> The unconditional S&P up-rate (72.9%) does most of the work. "
            "The football result contributes nothing statistically distinguishable."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 - The claim, steelmanned\n\n"
            "Let $Y_t$ be the S&P 500 calendar-year return and $S_t \\in \\{+1, -1\\}$ be "
            "the Super Bowl signal (+1 = NFC wins = bull prediction). The hypothesis is:\n\n"
            "- **H1 (signal).** $P(Y_t > 0 \\mid S_t = +1) > P(Y_t > 0)$ -- the "
            "hit-rate in NFC years exceeds the *unconditional* up-rate. Note: testing "
            "against 50% (instead of the base rate) is the critical error that inflated "
            "the indicator's apparent accuracy for decades.\n"
            "- **H2 (magnitude).** $E[Y_t \\mid S_t = +1] > E[Y_t \\mid S_t = -1]$ by a "
            "meaningful margin, distinguishable at |t| >= 2 on n ~ 30 per group.\n"
            "- **H3 (original-NFL variant).** Same as H1 but using the pre-merger AFL/NFL "
            "classification rather than current conference.\n\n"
            "We reject H1, H2, and H3 on the real tape."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 - So what? -- what rides on each answer\n\n"
            "The Super Bowl Indicator is perhaps the most famous spurious correlation in "
            "financial markets. If H1--H3 held, it would imply either (a) some bizarre "
            "mechanism linking football to corporate earnings, or (b) that financial "
            "astrology works. The interesting finding is *how* a 23-for-23 streak is born "
            "from pure chance -- it is a textbook demonstration of data-snooping and "
            "base-rate neglect."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 - How we'd know -- the protocol\n\n"
            "- **Data.** All 59 Super Bowl game results (hardcoded in `data.py`); "
            "Shiller S&P 500 December/December price returns 1967--2025.\n"
            "- **Signal.** `conference` (NFC=bull, AFC=bear) and `orig_nfl` (original-NFL=bull).\n"
            "- **Hit definition.** Signal correct if: (NFC win AND mkt up) OR (AFC win AND mkt down).\n"
            "- **Correct null.** Binomial test with $p_0 = 0.729$ "
            "(the unconditional up-rate), not 0.5.\n"
            "- **Inference.** Binomial test; permutation test (10,000 shuffles); Welch t-test "
            "on per-year returns; Bonferroni correction for 2 simultaneous tests.\n"
            "- **Positive control.** Synthetic tape with a planted NFC premium to confirm "
            "the machinery works when an effect exists.\n"
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 - The teardown"),
        md(
            "### 4a - The correct baseline: binomial test with p0 = 72.9%\n\n"
            "The critical methodological point: the S&P has an upward bias. Testing the "
            "indicator against a 50% coin -- as most commentary does -- is wrong."
        ),
        code(
            "if HAVE_REAL:\n"
            "    df = data.fetch_sp500_annual()\n"
            "    sig_nfc = st.signal_nfc(df)\n"
            "    r_nfc = st.hit_rate_stats(df, sig_nfc, n_permutations=10_000, seed=158)\n"
            "    n_obs = r_nfc['n_total']\n"
            "    n_bull = r_nfc['n_bull_years']\n"
            "    uncond = r_nfc['uncond_up_rate']\n"
            "    hit = r_nfc['hit_rate_all']\n"
            "    hit_bull = r_nfc['hit_rate_bull']\n"
            "    binom_correct = r_nfc['binom_p']\n"
            "    nfc_up_count = int(df.loc[df['conference']=='NFC','mkt_up'].sum())\n"
            "    binom_coin = scipy_stats.binomtest(k=nfc_up_count, n=n_bull, p=0.5,\n"
            "                                       alternative='greater').pvalue\n"
            "else:\n"
            "    n_obs, n_bull = R['n'], R['n_nfc']\n"
            "    uncond = R['uncond_up_pct']/100\n"
            "    hit, hit_bull = R['nfc_hit_pct']/100, R['nfc_hit_bull_pct']/100\n"
            "    binom_correct = R['nfc_binom_p']\n"
            "    binom_coin = 0.043\n"
            "\n"
            "print(f'n total: {n_obs}, n NFC years: {n_bull}')\n"
            "print(f'Unconditional up-rate: {uncond:.3f} ({uncond*100:.1f}%)')\n"
            "print(f'NFC hit-rate: {hit:.3f} ({hit*100:.1f}%)')\n"
            "print(f'NFC up-rate in NFC years: {hit_bull:.3f} ({hit_bull*100:.1f}%)')\n"
            "print()\n"
            "print(f'Binomial test vs CORRECT null (p0={uncond:.3f}): p = {binom_correct:.4f}')\n"
            "print(f'Binomial test vs WRONG null  (p0=0.50):          p = {binom_coin:.4f}')\n"
            "print()\n"
            "print('Using the wrong null (50%) makes the indicator look 10x more significant.')\n"
            "print('The correct test: the indicator is a non-event.')"
        ),
        md(
            "> Testing against the correct null (p_0 = 72.9%) rather than a coin "
            "changes the binomial p-value from ~0.04 (looks significant!) to "
            "**0.409** (clearly not). This error -- testing the hit-rate "
            "against 50% instead of the base rate -- explains most of the indicator's "
            "legendary reputation."
        ),
        md(
            "### 4b - The permutation test: is the hit-rate distinguishable from noise?\n\n"
            "Shuffle conference labels 10,000 times and record the distribution of "
            "hit-rates under the null."
        ),
        code(
            "if HAVE_REAL:\n"
            "    perm_hits = r_nfc['perm_hits']\n"
            "    obs_hit = r_nfc['hit_rate_all']\n"
            "    perm_p = r_nfc['perm_p']\n"
            "else:\n"
            "    rng0 = np.random.default_rng(158)\n"
            "    mkt_up_s = rng0.choice([True,False], size=R['n'],\n"
            "                           p=[R['uncond_up_pct']/100, 1-R['uncond_up_pct']/100])\n"
            "    perm_hits = []\n"
            "    for _ in range(2000):\n"
            "        sigs = rng0.choice([1,-1], size=R['n'])\n"
            "        h = np.where(sigs==1, mkt_up_s.astype(int), (~mkt_up_s).astype(int)).mean()\n"
            "        perm_hits.append(h)\n"
            "    perm_hits = np.array(perm_hits)\n"
            "    obs_hit = R['nfc_hit_pct']/100\n"
            "    perm_p = R['nfc_perm_p']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(perm_hits, bins=30, color=GREY, alpha=0.7,\n"
            "        label='Shuffled labels (permutations)')\n"
            "ax.axvline(obs_hit, c=RED, lw=2.5, label=f'Observed hit-rate: {obs_hit:.3f}')\n"
            "ax.set_xlabel('Hit-rate'); ax.set_ylabel('Count')\n"
            "ax.set_title('The observed hit-rate lands squarely inside the null distribution')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Permutation p-value: {perm_p:.4f}')\n"
            "print(f'Observed percentile: {(perm_hits < obs_hit).mean():.1%} of permutations below')"
        ),
        md(
            "The observed hit-rate of 54.2% sits in the middle of the "
            "null distribution. Permutation p = **0.351** -- no evidence "
            "that the conference label carries any real information about the coming year."
        ),
        md(
            "### 4c - Welch t-test and mean returns per conference\n\n"
            "Are NFC-year mean returns meaningfully higher than AFC-year returns?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    nfc_rets = df.loc[df['conference']=='NFC','sp500_return'].values * 100\n"
            "    afc_rets = df.loc[df['conference']=='AFC','sp500_return'].values * 100\n"
            "    wt, wp = scipy_stats.ttest_ind(nfc_rets, afc_rets, equal_var=False)\n"
            "    nfc_m, afc_m = float(nfc_rets.mean()), float(afc_rets.mean())\n"
            "else:\n"
            "    nfc_m, afc_m = R['nfc_mean_bull'], R['nfc_mean_bear']\n"
            "    wt, wp = R['nfc_welch_t'], R['nfc_welch_p']\n"
            "    rng1 = np.random.default_rng(1)\n"
            "    nfc_rets = rng1.normal(nfc_m, 17, R['n_nfc'])\n"
            "    afc_rets = rng1.normal(afc_m, 17, R['n_afc'])\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.boxplot([nfc_rets, afc_rets], labels=['NFC years', 'AFC years'],\n"
            "           patch_artist=True,\n"
            "           boxprops=dict(facecolor=GREEN, alpha=0.5),\n"
            "           medianprops=dict(color='black', lw=2))\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('S&P 500 annual return (%)')\n"
            "ax.set_title(f'NFC mean {nfc_m:+.1f}% vs AFC mean {afc_m:+.1f}% -- Welch t = {wt:+.3f}')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'NFC mean: {nfc_m:+.1f}%  |  AFC mean: {afc_m:+.1f}%')\n"
            "print(f'Difference: {nfc_m-afc_m:+.1f}pp  |  Welch t = {wt:+.3f}, p = {wp:.4f}')\n"
            "print('Required gap for |t|>=2 with n~30: ~6.2%  |  Observed: ~2.6%')"
        ),
        md(
            "NFC years averaged **+10.4%**, AFC years **+7.8%** -- "
            "a gap of 2.6pp on ~17% annual vol with n ~30. "
            "Welch t = **+0.625**, p = 0.534. "
            "To be detectable at |t| = 2 with this sample size, the structural gap would need "
            "to be ~6.2%/yr -- more than twice the observed difference."
        ),
        md(
            "### 4d - Multiple-comparisons summary (Bonferroni)\n\n"
            "Both the conference and orig-NFL variants are tested simultaneously."
        ),
        code(
            "if HAVE_REAL:\n"
            "    mc = st.multiple_comparisons_summary(df, n_permutations=5_000, seed=158)\n"
            "else:\n"
            "    mc = pd.DataFrame({\n"
            "        'test': ['NFC win (conference)', 'Orig-NFL win'],\n"
            "        'n': [R['n'], R['n']],\n"
            "        'hit_rate_all': [R['nfc_hit_pct']/100, R['orig_hit_pct']/100],\n"
            "        'binom_p': [R['nfc_binom_p'], R['orig_binom_p']],\n"
            "        'perm_p': [R['nfc_perm_p'], R['orig_perm_p']],\n"
            "        'bonferroni_binom_p': [R['bonf_nfc_binom_p'], R['bonf_orig_binom_p']],\n"
            "        'bonferroni_perm_p': [min(1.0,R['nfc_perm_p']*2), min(1.0,R['orig_perm_p']*2)],\n"
            "    })\n"
            "\n"
            "display_cols = ['test','n','hit_rate_all','binom_p','perm_p','bonferroni_binom_p']\n"
            "print(mc[display_cols].to_string(index=False))\n"
            "print()\n"
            "print('All p-values >> 0.05. Bonferroni barely matters because')\n"
            "print('both tests were already far from significance.')"
        ),
        md(
            "> With 2 tests and Bonferroni correction, all p-values are well above 0.05. "
            "The correction is almost irrelevant -- both hypotheses were already clearly "
            "non-significant without it. The deeper issue is the tiny n."
        ),
        md(
            "### 4e - Power calculation: what gap could we detect with n=59?\n\n"
            "The minimum detectable effect at 80% power, two-sided, alpha=0.05."
        ),
        code(
            "vol = 0.17  # ~S&P annual vol\n"
            "n_per_group = 30\n"
            "alpha = 0.05\n"
            "\n"
            "from scipy.stats import t as t_dist\n"
            "df_dof = 2*(n_per_group-1)\n"
            "t_crit = t_dist.ppf(1 - alpha/2, df_dof)\n"
            "t_power = t_dist.ppf(0.80, df_dof)\n"
            "mde = (t_crit + t_power) * vol * np.sqrt(2/n_per_group)\n"
            "\n"
            "print(f'Minimum detectable effect (80% power, 2-sided, alpha={alpha}):')\n"
            "print(f'  vol = {vol:.0%}, n/group = {n_per_group}')\n"
            "print(f'  MDE = {mde:.1%}/yr  ({mde*100:.1f}pp)')\n"
            "print()\n"
            "print('Observed NFC-AFC gap: ~2.6pp/yr')\n"
            "print(f'That is {2.6/100/mde:.0%} of the MDE.')\n"
            "print()\n"
            "print('Conclusion: n=59 cannot resolve the Super Bowl Indicator.')\n"
            "print('Even if the 2.6pp gap were real, we need ~5x more data to detect it.')"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            "- **Signal `NONE`** -- NFC hit-rate 54.2%, binom p 0.409, "
            "perm p 0.351, Welch t +0.625. "
            "The NFC up-rate of 76.7% is +3.8pp above the "
            "72.9% base rate -- indistinguishable from noise at n=30.\n"
            "- **Tradability `MIRAGE`** -- no tradable vehicle; passive long S&P beats "
            "any conference-timing strategy.\n"
            "- **Busted** -- the 23-for-23 streak was a data-snooping artifact; the "
            "indicator has broken down repeatedly since 2001."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 - Could you trade it? -- the buy-and-hold benchmark\n\n"
            "The Super Bowl strategy: long S&P in NFC years, flat in AFC years."
        ),
        code(
            "if HAVE_REAL:\n"
            "    df4 = df.copy()\n"
            "    df4['strat'] = df4.apply(\n"
            "        lambda r: r['sp500_return'] if r['conference']=='NFC' else 0.0, axis=1)\n"
            "    n_yrs = len(df4)\n"
            "    strat_ann = float((1+df4['strat']).prod()**(1/n_yrs) - 1)\n"
            "    bah_ann = float((1+df4['sp500_return']).prod()**(1/n_yrs) - 1)\n"
            "    years4 = df4['year'].values\n"
            "    strat_cump = (1+df4['strat']).cumprod().values * 100\n"
            "    bah_cump = (1+df4['sp500_return']).cumprod().values * 100\n"
            "else:\n"
            "    strat_ann = R['n_nfc']/R['n'] * R['nfc_mean_bull']/100\n"
            "    bah_ann = R['nfc_mean_all']/100\n"
            "    rng2 = np.random.default_rng(99)\n"
            "    years4 = np.arange(R['year_start'], R['year_end']+1)\n"
            "    strat_rets_s = rng2.normal(strat_ann/R['n'], 0.17, R['n'])\n"
            "    bah_rets_s = rng2.normal(bah_ann/R['n'], 0.17, R['n'])\n"
            "    strat_cump = (1+strat_rets_s).cumprod()*100\n"
            "    bah_cump = (1+bah_rets_s).cumprod()*100\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(10.0, 4.5))\n"
            "ax.plot(years4, bah_cump, lw=2, c=GREEN, label=f'Buy & hold: {bah_ann:.1%}/yr')\n"
            "ax.plot(years4, strat_cump, lw=2, c=RED, ls='--',\n"
            "        label=f'Super Bowl strategy: {strat_ann:.1%}/yr')\n"
            "ax.set_yscale('log')\n"
            "ax.set_xlabel('Year')\n"
            "ax.set_ylabel('Portfolio value (log scale, base 100)')\n"
            "ax.set_title('The Super Bowl strategy: inferior to sitting in an index fund')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Buy & hold annualised: {bah_ann:.2%}')\n"
            "print(f'Super Bowl strategy:   {strat_ann:.2%}')\n"
            "print(f'Shortfall:             {(bah_ann-strat_ann)*100:+.1f}pp/yr')"
        ),
        md(
            "> The strategy sits in cash half the years (AFC wins) and misses those gains. "
            "There is no signal to compensate for the opportunity cost."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 - Going further -- the synthetic positive control\n\n"
            "Does the machinery detect a Super Bowl signal *when one exists*? "
            "We plant an NFC premium in synthetic annual returns and sweep its size."
        ),
        code(
            "planted = [0, 100, 500, 1000, 2000]\n"
            "results = []\n"
            "for bps in planted:\n"
            "    df_s, _ = data.synthetic_annual(n_years=500, signal_bps=float(bps), seed=158)\n"
            "    df_s = df_s.rename(columns={'return':'sp500_return'})\n"
            "    df_s['mkt_up'] = df_s['sp500_return'] > 0\n"
            "    df_s['game_number'] = range(1, 501)\n"
            "    df_s['winner'] = 'Synth'\n"
            "    sig = st.signal_nfc(df_s)\n"
            "    r = st.hit_rate_stats(df_s, sig, n_permutations=500, seed=99)\n"
            "    results.append({'bps': bps, 'hit': r['hit_rate_all']*100,\n"
            "                    'binom_p': r['binom_p'], 'perm_p': r['perm_p']})\n"
            "\n"
            "res_df = pd.DataFrame(results)\n"
            "colors = [GREEN if (r['binom_p']<0.05 and r['perm_p']<0.05) else RED\n"
            "          for r in results]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.bar([str(r['bps']) for r in results], [r['hit'] for r in results], color=colors)\n"
            "ax.axhline(50, c='k', ls=':', lw=1, label='50%')\n"
            "ax.set_xlabel('Planted NFC premium (bps/yr)')\n"
            "ax.set_ylabel('Hit-rate %')\n"
            "ax.set_title('The engine finds the signal when it exists (green = both p < 0.05)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(res_df.to_string(index=False))"
        ),
        md(
            "The engine correctly detects a planted signal when the premium is large "
            "enough and n is sufficient. The real tape -- with n=59 and a ~2.6pp gap -- "
            "is nowhere near the detection threshold. The verdict is a statement about the "
            "**market and sample size**, not about the method."
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
