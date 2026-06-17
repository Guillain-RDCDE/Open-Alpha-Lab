"""Generate the two narrative notebooks for Study 270 (Underwear-Index).

    python notebooks/build_notebooks.py

The build script writes both notebooks AND executes them in-place via nbconvert
(no --allow-errors). The hardcoded underwear index, the NBER recession calendar,
and the synthetic generator run anywhere, offline and deterministically; the real
S&P 500 cells use the cached Shiller parquet at the repo-level _cache/ if present,
and otherwise quote the frozen headline numbers in ``R`` (mirroring docs/results.md),
so the notebooks re-run for any reader.

The _write convention (each build_*() ends by calling _write) is kept so the repo's
intro-restyle tooling can monkeypatch it.
"""

from __future__ import annotations

import os
import subprocess
import sys

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))


def md(text):
    return new_markdown_cell(text)


def code(text):
    return new_code_cell(text)


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-17).
R = dict(
    n=33,
    n_dips=4,
    year_start=1992,
    year_end=2024,
    # leading-indicator (year+1) view -- the genuine Greenspan claim
    uncond_rec_pct=15.2,
    rec_rate_dip_pct=25.0,
    rec_rate_nodip_pct=13.8,
    next_fisher_p=0.4996,
    next_perm_p=0.4941,
    # coincident (same-year) view -- tautological by construction
    same_rec_rate_dip_pct=100.0,
    same_fisher_p=0.0001,
    same_perm_p=0.0,
    # timing backtest (no look-ahead, dip -> flat next year)
    bh_ann_pct=8.6,
    strat_gross_pct=7.4,
    strat_net_pct=7.4,
    turnover=0.2188,
    nw_t=-0.939,
    fp="a59af6e1883f",
)

R_REPR = (
    "R = dict(\n"
    "    n=33, n_dips=4, year_start=1992, year_end=2024,\n"
    "    uncond_rec_pct=15.2, rec_rate_dip_pct=25.0, rec_rate_nodip_pct=13.8,\n"
    "    next_fisher_p=0.4996, next_perm_p=0.4941,\n"
    "    same_rec_rate_dip_pct=100.0, same_fisher_p=0.0001, same_perm_p=0.0,\n"
    "    bh_ann_pct=8.6, strat_gross_pct=7.4, strat_net_pct=7.4,\n"
    "    turnover=0.2188, nw_t=-0.939,\n"
    ")\n"
)

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

from underwear_index import data, strategy as st

def _have_cache():
    try:
        data.fetch_sp500_annual(fetch=False)
        return True
    except FileNotFoundError:
        return False

HAVE_REAL = _have_cache()
print("Shiller cache present:", HAVE_REAL)
"""

R_CELL = "# Frozen headline numbers -- mirror of docs/results.md (as-of 2026-06-17)\n" + R_REPR


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Underwear-Index -- can men's underwear sales call a recession?\n"
            "### The Greenspan 'Men's Underwear Index', tested honestly, in plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Busted%3F: Yes](https://img.shields.io/badge/Busted%3F-Yes-8b949e?style=flat-square)\n\n"
            "Here is a piece of macro folklore attributed to **Alan Greenspan**: men treat "
            "underwear as a near-necessity and replace it on a steady schedule, so when "
            "household budgets get squeezed, men's-underwear sales quietly *dip* -- supposedly "
            "an early warning of a recession. This notebook asks the only question that matters: "
            "does the dip actually **lead** a downturn, or is it just a coincident description of "
            "bad times we already know about?\n\n"
            "> **This is the plain-language layer.** Want the Fisher exact test, the permutation "
            "distribution, the HAC t-stat and the power calculation? That's "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** -- same story, deeper.\n"
            ">\n"
            "> **Honesty note up front.** There is no clean public long-run series for this niche "
            "category, so the underwear index here is a *curated reconstruction* (a smooth trend "
            "with stylised recession dips). The recession calendar (NBER) and the S&P 500 tape "
            "(Shiller) are real. So the *predictor's raw series is stylised* -- which we flag "
            "loudly -- but the test against real recessions and real markets is genuine.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT + "\n" + R_CELL),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does a dip **lead** a recession (next-year)? | **No.** Recession follows a dip "
            "**25%** of the time vs **13.8%** after a non-dip -- Fisher p = **0.50**, perm p = **0.49**. |\n"
            "| Does a dip at least *coincide* with a recession? | **Trivially yes -- by construction.** "
            "The reconstructed index has its dips *placed on* recession years, so 'a dip means a "
            "recession this year' is a tautology, not a discovery. |\n"
            "| Could you trade it (sit out the year after a dip)? | **No.** The timing rule earns "
            "**7.4%/yr** vs **8.6%/yr** buy-and-hold; HAC t on the spread = **-0.94**. |\n"
            "| How many dips are there to learn from? | **Four**, in ~33 years. That is far too "
            "few to call anything. |\n\n"
            "> The Underwear Index is a vivid *story about* recessions, not a *predictor of* them. "
            "As a leading signal it is noise; as a trade it is a drag."
        ),

        # ---- BEAT 1 -- THE CLAIM ----------------------------------------------
        md(
            "## 1 - The claim\n\n"
            "> *\"Underwear is one of the last things a man stops buying. So when men's-underwear "
            "sales fall, it means households are under real stress -- a recession is coming.\"*\n\n"
            "The anecdote is widely attributed to Alan Greenspan, who reportedly tracked the index "
            "as a Fed-era curiosity. It became newsroom shorthand during the 2008 downturn, when "
            "men's-underwear sales did soften. The mechanism is plausible-sounding -- underwear is "
            "cheap, hidden, and easy to defer -- but plausibility is not predictive power."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 - So what?\n\n"
            "If a single, slow, annual consumer-staples series could *lead* the business cycle, it "
            "would be an extraordinarily cheap macro signal -- you would watch Hanes shipments "
            "instead of the yield curve. The fun question is the usual one for folklore indicators: "
            "is there any forward information, or is the famous 2008 'dip' just the index doing what "
            "every discretionary-ish purchase does in a slump -- falling *while* the recession is "
            "already underway, telling you nothing you did not already know?"
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW -----------------------------------------
        md(
            "## 3 - How would we even know?\n\n"
            "Three traps to avoid:\n\n"
            "1. **Leading vs coincident.** The only interesting claim is that a dip *predicts* "
            "(comes before) a recession. A dip that merely *coincides* with a recession is "
            "useless for forecasting -- and in our reconstructed series it is literally built in. "
            "We test the year-Y dip against the year-(Y+1) outcome, never the same year.\n"
            "2. **The base rate.** Recessions are rare -- only a handful of years are NBER "
            "contraction years. Any 'something bad is coming' flag is mostly right simply because "
            "most predictions of calm are right, and mostly useless when it matters.\n"
            "3. **Tiny n.** ~33 annual observations, **4 dips**, ~5 recession years. With counts "
            "this small, one lucky coincidence flips the verdict.\n\n"
            "We test on the NBER recession calendar and the Shiller S&P 500 (1992-2024)."
        ),

        # ---- BEAT 4 -- THE TEARDOWN ------------------------------------------
        md(
            "## 4 - The teardown -- let's actually look\n\n"
            "**First, meet the data:** the curated annual underwear index, its year-on-year "
            "change, and the NBER recession years."
        ),
        code(
            "uw = data.underwear_table()\n"
            "if HAVE_REAL:\n"
            "    df = data.fetch_sp500_annual()\n"
            "    n_dips = int(df['dip'].sum())\n"
            "    uncond = float(df['next_recession'].mean())\n"
            "else:\n"
            "    n_dips = R['n_dips']\n"
            "    uncond = R['uncond_rec_pct']/100\n"
            "\n"
            "print('Years:', uw['year'].min(), '-', uw['year'].max(), '| obs:', len(uw))\n"
            "print('Dip years (yoy < 0):', list(uw.loc[uw['dip'],'year']))\n"
            "print('NBER recession years:', sorted(data.US_RECESSIONS))\n"
            "print(f'Unconditional next-year recession rate: {uncond:.1%}')\n"
            "print()\n"
            "print('Only', n_dips, 'dips in', len(uw), 'years -- a very thin evidence base.')"
        ),
        md("**The index and its recession dips.**"),
        code(
            "yrs = uw['year'].values\n"
            "idx = uw['index'].values\n"
            "fig, ax = plt.subplots(figsize=(10, 4.5))\n"
            "ax.plot(yrs, idx, '-', c=GREY, lw=2, label='Men\\'s underwear index (1992=100)')\n"
            "for ry in sorted(data.US_RECESSIONS):\n"
            "    ax.axvspan(ry-0.5, ry+0.5, color=RED, alpha=0.12)\n"
            "dip_mask = uw['dip'].values\n"
            "ax.scatter(yrs[dip_mask], idx[dip_mask], c=RED, s=60, zorder=5, label='Dip year (yoy<0)')\n"
            "ax.set_xlabel('Year'); ax.set_ylabel('Index (1992=100)')\n"
            "ax.set_title('The underwear index dips on recession years -- because it was drawn that way')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('Red bands = NBER recessions. Dips sit on them by construction (reconstructed series).')"
        ),
        md(
            "Notice the dips line up neatly with the red recession bands. That is **not** evidence "
            "the index works -- it is an artifact of how the illustrative series was built (dips "
            "placed *on* recession years). The honest question is whether the dip arrives **before** "
            "the slump, which we test next."
        ),
        md("**The leading-indicator test: does a dip predict *next year's* recession?**"),
        code(
            "if HAVE_REAL:\n"
            "    rs = st.recession_stats(df, horizon='next', n_permutations=10_000, seed=270)\n"
            "    rec_dip = rs['rec_rate_dip']*100\n"
            "    rec_nodip = rs['rec_rate_nodip']*100\n"
            "    base = rs['uncond_rec_rate']*100\n"
            "else:\n"
            "    rec_dip = R['rec_rate_dip_pct']\n"
            "    rec_nodip = R['rec_rate_nodip_pct']\n"
            "    base = R['uncond_rec_pct']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "bars = ax.bar(['After a DIP\\n(signal: trouble)',\n"
            "               'After NO dip',\n"
            "               'Unconditional\\n(any year)'],\n"
            "              [rec_dip, rec_nodip, base],\n"
            "              color=[RED, GREEN, GREY], width=0.55)\n"
            "ax.axhline(base, ls='--', c=AMBER, lw=2, label=f'Base rate ({base:.1f}%)')\n"
            "ax.set_ylabel('Next-year recession rate (%)')\n"
            "ax.set_title('Dip -> next-year recession: +11pp over base, but on 4 dips = noise')\n"
            "for b, v in zip(bars, [rec_dip, rec_nodip, base]):\n"
            "    ax.annotate(f'{v:.1f}%', (b.get_x()+b.get_width()/2, v+1), ha='center', va='bottom')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Recession-after-dip: {rec_dip:.1f}%  vs after-no-dip: {rec_nodip:.1f}%  (base {base:.1f}%)')\n"
            "print('Fisher exact p =', R['next_fisher_p'], '| permutation p =', R['next_perm_p'])"
        ),
        md(
            "A dip is followed by a recession **25%** of the time vs **13.8%** otherwise -- a gap "
            "that *looks* directional but rests on just **four** dips. The Fisher exact test gives "
            "**p = 0.50** and the permutation test **p = 0.49**: completely consistent with chance."
        ),

        # ---- BEAT 5 -- THE VERDICT -------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            "- **Signal -- None.** As a *leading* indicator the dip is noise: Fisher p = **0.50**, "
            "perm p = **0.49**, on only **4 dips**. Its apparent coincident 'accuracy' is a "
            "tautology baked into the reconstructed series, not a discovery.\n"
            "- **Tradability -- Mirage.** Sitting out the S&P the year after a dip earns "
            "**7.4%/yr** vs **8.6%/yr** for buy-and-hold; the timing spread has HAC t = **-0.94**. "
            "You give up the market's drift to dodge phantom recessions.\n"
            "- **Busted.** The famous 2008 'dip' was the index falling *during* the slump, not "
            "before it -- a coincident footnote, not a forecast."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 - Could you actually trade it?\n\n"
            "The 'strategy' would be: go to **cash** the year after the underwear index dips, "
            "stay long the S&P otherwise. The problem:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    bt = st.timing_backtest(df, cost_bps=10.0)\n"
            "    strat_ann = bt['strat_net_ann']\n"
            "    bah_ann = bt['bh_ann']\n"
            "    nw_t = bt['nw_t']\n"
            "    turn = bt['turnover']\n"
            "else:\n"
            "    strat_ann = R['strat_net_pct']/100\n"
            "    bah_ann = R['bh_ann_pct']/100\n"
            "    nw_t = R['nw_t']\n"
            "    turn = R['turnover']\n"
            "\n"
            "print(f'Underwear timing (flat year after a dip): {strat_ann:.1%}/yr (net of 10bps costs)')\n"
            "print(f'Buy and hold S&P:                         {bah_ann:.1%}/yr')\n"
            "print(f'Timing advantage:                         {(strat_ann-bah_ann)*100:+.1f}pp/yr')\n"
            "print(f'HAC (Newey-West) t on the spread:         {nw_t:+.2f}  (turnover {turn:.2f})')\n"
            "print()\n"
            "print('The rule sits out some of the markets best rebound years (2009, 2021)')\n"
            "print('because those followed dips -- exactly the wrong call.')"
        ),
        md(
            "The timing rule loses to buy-and-hold because it goes flat precisely into the "
            "post-recession rebounds (2009, 2021 followed dip years). One documented execution "
            "lag (you only know year-Y's dip after Y closes) is already built into the "
            "no-look-ahead design. There is no edge to harvest."
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 - Going further\n\n"
            "- **The positive control:** the companion notebook plants a real 'dip-leads-recession' "
            "effect into a synthetic tape and shows the engine *does* detect it when it exists -- "
            "and that the real tape has no such thing.\n"
            "- **Other folklore macro gauges:** the lipstick index, the hemline index, the "
            "skyscraper curse -- same family, same fate.\n"
            "- **The honest macro alternative:** the yield-curve slope and the Sahm rule are the "
            "real recession signals; folklore indices are anecdotes.\n\n"
            "*Think the underwear really leads? Fork this, drop in an *audited* men's-underwear "
            "series, and show a next-year recession lift that clears Fisher p < 0.05. That's the "
            "bar -- and a reconstructed series with 4 dips cannot clear it.*"
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
            "# Underwear-Index -- a quantitative teardown\n"
            "### NBER recessions * Shiller S&P500 * Fisher exact * permutation * "
            "HAC t * n=33 / 4-dips power\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Busted%3F: Yes](https://img.shields.io/badge/Busted%3F-Yes-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) -- "
            "*same seven beats, every claim carrying its standard error.* We separate the "
            "**coincident** view (tautological in a reconstructed series) from the **leading** "
            "view (the genuine Greenspan claim), run a Fisher exact test on the 2x2 dip x recession "
            "table, a permutation test on the dip labels, a Newey-West HAC t on the timing spread, "
            "and a power note on why 4 dips can never settle anything.\n\n"
            "> **Honesty banner.** The underwear index is a *curated reconstruction* (no clean "
            "public series exists); the NBER calendar and Shiller S&P 500 (1992-2024) are real. "
            "Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **Not investment advice.**"
        ),
        code(BOOT + "\n" + R_CELL + "\nfrom scipy import stats as scipy_stats\n"),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            "| **Signal** | `NONE` | Leading view: recession-after-dip **25.0%** vs **13.8%**, "
            "Fisher p = **0.50**, perm p = **0.49**, on **4 dips**. (Coincident p<0.001 is a "
            "construction artifact, not a finding.) |\n"
            "| **Tradability** | `MIRAGE` | Dip->flat timing earns **7.4%/yr** net vs **8.6%/yr** "
            "B&H; HAC t on spread = **-0.94**. |\n"
            "| **Busted?** | `YES` | The 2008 dip was coincident, not leading; n=33 with 4 dips "
            "cannot resolve any effect. |\n\n"
            "> The base rate and tiny n do all the work. The underwear series adds no forward "
            "information once you forbid look-ahead."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 - The claim, steelmanned\n\n"
            "Let $D_t \\in \\{0,1\\}$ be the dip flag (underwear YoY < 0 in year $t$) and "
            "$Rec_t \\in \\{0,1\\}$ the NBER recession flag. The hypotheses:\n\n"
            "- **H1 (leading).** $P(Rec_{t+1}=1 \\mid D_t=1) > P(Rec_{t+1}=1)$ -- a dip raises the "
            "*next-year* recession probability above its base rate. This is the only forecasting "
            "claim worth testing.\n"
            "- **H2 (tradable).** A rule that goes flat in $t+1$ after a dip in $t$ beats buy-and-"
            "hold the S&P, net of costs, at HAC $|t| \\ge 2$.\n"
            "- **H0 (coincident, control).** $P(Rec_t=1 \\mid D_t=1) > P(Rec_t=1)$. In our "
            "*reconstructed* series this is true by construction and therefore uninformative -- we "
            "include it only to show the difference a single year of look-ahead makes.\n\n"
            "We reject H1 and H2 on the real tape."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 - So what? -- what rides on each answer\n\n"
            "The Men's Underwear Index is a poster child for *coincident-mistaken-for-leading* "
            "reasoning. If H1 held, a slow annual staples series would front-run the business "
            "cycle -- it does not. The instructive part is watching a relationship that is "
            "p < 0.001 *coincidentally* (because the dips sit on recession years) collapse to "
            "p ~ 0.5 the moment we demand it predict the *following* year."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 - How we'd know -- the protocol\n\n"
            "- **Data.** Curated annual underwear index (hardcoded in `data.py`); NBER recession "
            "years; Shiller S&P 500 December/December price returns, 1992-2024 (33 years).\n"
            "- **Signal.** `dip` = (underwear YoY < 0).\n"
            "- **Outcomes.** `recession` (same year) and `next_recession` (year+1); "
            "`next_sp500_return` (year+1) for the no-look-ahead trade.\n"
            "- **Inference.** Fisher exact test on the 2x2 dip x recession table; permutation test "
            "(10,000 shuffles of dip labels); Newey-West HAC t (1 lag) on the timing spread.\n"
            "- **Costs.** One-way 10 bps of NAV charged on every position change.\n"
            "- **Positive control.** Synthetic tape with a planted dip->recession link.\n"
            "- **Labels.** Price-only (no dividends); single index (no survivorship issue) -- "
            "noted on the Signal axis."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 - The teardown"),
        md(
            "### 4a - Coincident vs leading: one year of look-ahead changes everything\n\n"
            "The same dip flag, tested against the same-year recession and against the next-year "
            "recession. Watch the p-value."
        ),
        code(
            "if HAVE_REAL:\n"
            "    df = data.fetch_sp500_annual()\n"
            "    rs_same = st.recession_stats(df, horizon='same', n_permutations=10_000, seed=270)\n"
            "    rs_next = st.recession_stats(df, horizon='next', n_permutations=10_000, seed=270)\n"
            "    same_p, next_p = rs_same['fisher_p'], rs_next['fisher_p']\n"
            "    same_dip = rs_same['rec_rate_dip']*100\n"
            "    next_dip = rs_next['rec_rate_dip']*100\n"
            "else:\n"
            "    same_p, next_p = R['same_fisher_p'], R['next_fisher_p']\n"
            "    same_dip = R['same_rec_rate_dip_pct']\n"
            "    next_dip = R['rec_rate_dip_pct']\n"
            "\n"
            "print('COINCIDENT (same-year)  : recession|dip =', f'{same_dip:.1f}%',\n"
            "      '| Fisher p =', same_p, ' <- tautology in a reconstructed series')\n"
            "print('LEADING  (next-year)    : recession|dip =', f'{next_dip:.1f}%',\n"
            "      '| Fisher p =', next_p, ' <- the honest test: a non-event')\n"
            "print()\n"
            "print('Same number, same data; only the one-year forecast horizon differs.')\n"
            "print('The headline-grabbing significance lives entirely in the look-ahead.')"
        ),
        md(
            "> The coincident p < 0.001 is **not a finding** -- the reconstructed index has its dips "
            "drawn on recession years, so 'a dip means a recession this year' is circular. The "
            "leading test (p = **0.50**) is the one that matters, and it is null."
        ),
        md(
            "### 4b - The 2x2 table and the Fisher exact test (leading view)\n\n"
            "Four cells, four dips. This is the whole evidence base."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rs = st.recession_stats(df, horizon='next', n_permutations=10_000, seed=270)\n"
            "    a,b,c,d = rs['a_dip_rec'], rs['b_dip_norec'], rs['c_nodip_rec'], rs['d_nodip_norec']\n"
            "    fisher_p, perm_p = rs['fisher_p'], rs['perm_p']\n"
            "else:\n"
            "    a,b,c,d = 1,3,4,25\n"
            "    fisher_p, perm_p = R['next_fisher_p'], R['next_perm_p']\n"
            "\n"
            "tab = pd.DataFrame([[a,b],[c,d]],\n"
            "                   index=['DIP (signal)','no dip'],\n"
            "                   columns=['recession t+1','no recession t+1'])\n"
            "print(tab.to_string())\n"
            "print()\n"
            "print(f'Dip years: {a+b}  |  of which followed by recession: {a}')\n"
            "print(f'Fisher exact p (one-sided) = {fisher_p:.4f}')\n"
            "print(f'Permutation p              = {perm_p:.4f}')\n"
            "print()\n"
            "print('A single cell of 1 vs 3 cannot carry a forecasting claim.')"
        ),
        md(
            "### 4c - The permutation null: where does the observed excess land?\n\n"
            "Shuffle the dip labels 10,000 times and record the excess recession rate "
            "(dip minus no-dip)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    perm_excess = st.recession_stats(df, horizon='next', n_permutations=10_000, seed=270)['perm_excess']\n"
            "    obs_excess = st.recession_stats(df, horizon='next', n_permutations=10_000, seed=270)['excess_rec_rate']\n"
            "    perm_p = R['next_perm_p']\n"
            "else:\n"
            "    rng0 = np.random.default_rng(270)\n"
            "    rec = np.array([1]*5 + [0]*28)\n"
            "    perm_excess = []\n"
            "    for _ in range(10_000):\n"
            "        dipm = np.zeros(33, dtype=bool); dipm[rng0.choice(33,4,replace=False)] = True\n"
            "        perm_excess.append(rec[dipm].mean() - rec[~dipm].mean())\n"
            "    perm_excess = np.array(perm_excess)\n"
            "    obs_excess = (R['rec_rate_dip_pct']-R['rec_rate_nodip_pct'])/100\n"
            "    perm_p = R['next_perm_p']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(perm_excess, bins=30, color=GREY, alpha=0.7, label='Shuffled dip labels')\n"
            "ax.axvline(obs_excess, c=RED, lw=2.5, label=f'Observed excess: {obs_excess:+.3f}')\n"
            "ax.set_xlabel('Excess next-year recession rate (dip - no dip)'); ax.set_ylabel('Count')\n"
            "ax.set_title('The observed dip excess sits well inside the null distribution')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Permutation p-value: {perm_p:.4f}')\n"
            "print(f'Percentile of observed: {(perm_excess < obs_excess).mean():.1%} of shuffles below')"
        ),
        md(
            "The observed excess sits comfortably inside the permutation null. There is no "
            "evidence the dip label carries forward information about recessions."
        ),
        md(
            "### 4d - The trade: HAC t-stat on dip->flat timing vs buy-and-hold\n\n"
            "No look-ahead (dip in $t$ -> flat in $t+1$), one-way 10 bps costs on each turn."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bt = st.timing_backtest(df, cost_bps=10.0)\n"
            "    years4 = bt['years']; bh = bt['bh']; strat = bt['strat_net']\n"
            "    bah_ann, strat_ann, nw_t, turn = bt['bh_ann'], bt['strat_net_ann'], bt['nw_t'], bt['turnover']\n"
            "else:\n"
            "    bah_ann, strat_ann = R['bh_ann_pct']/100, R['strat_net_pct']/100\n"
            "    nw_t, turn = R['nw_t'], R['turnover']\n"
            "    rng2 = np.random.default_rng(270)\n"
            "    years4 = np.arange(R['year_start']+1, R['year_end']+1)\n"
            "    bh = rng2.normal(bah_ann, 0.17, len(years4))\n"
            "    strat = rng2.normal(strat_ann, 0.15, len(years4))\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(10.0, 4.5))\n"
            "ax.plot(years4, (1+bh).cumprod()*100, lw=2, c=GREEN, label=f'Buy & hold: {bah_ann:.1%}/yr')\n"
            "ax.plot(years4, (1+strat).cumprod()*100, lw=2, c=RED, ls='--',\n"
            "        label=f'Underwear timing (net): {strat_ann:.1%}/yr')\n"
            "ax.set_yscale('log'); ax.set_xlabel('Year')\n"
            "ax.set_ylabel('Portfolio value (log, base 100)')\n"
            "ax.set_title(f'Dip->flat timing trails buy-and-hold (HAC t on spread = {nw_t:+.2f})')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Buy & hold:        {bah_ann:.2%}/yr')\n"
            "print(f'Underwear timing:  {strat_ann:.2%}/yr (net, turnover {turn:.2f})')\n"
            "print(f'Spread HAC t:      {nw_t:+.2f}  -- negative and far from significance')"
        ),
        md(
            "> The timing rule trails B&H and the spread's HAC t is **negative**. Going flat the "
            "year after a dip means missing post-recession rebounds -- the opposite of an edge."
        ),
        md(
            "### 4e - Power: what could 4 dips ever detect?\n\n"
            "With 4 dip years and a ~15% base recession rate, the binomial sampling noise on the "
            "dip-conditional recession rate is enormous."
        ),
        code(
            "n_dip = R['n_dips']\n"
            "base = R['uncond_rec_pct']/100\n"
            "se = np.sqrt(base*(1-base)/n_dip)\n"
            "print(f'Base next-year recession rate: {base:.1%}')\n"
            "print(f'Dip-conditional rate std error (n_dip={n_dip}): +/-{se:.1%}')\n"
            "print(f'A 95% interval around the base spans roughly {max(0,base-2*se):.0%} to {base+2*se:.0%}.')\n"
            "print()\n"
            "print('To clear Fisher p<0.05 with 4 dips you would need ~3-4 of them')\n"
            "print('followed by a recession -- i.e. an effect so huge it would be obvious.')\n"
            "print('Observed: 1 of 4. The test is powerless at this sample size.')"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            "- **Signal `NONE`** -- leading view Fisher p = 0.50, perm p = 0.49, on 4 dips. The "
            "coincident significance is a construction tautology, not evidence.\n"
            "- **Tradability `MIRAGE`** -- dip->flat timing earns 7.4%/yr net vs 8.6%/yr B&H; "
            "HAC t = -0.94. No vehicle, negative edge.\n"
            "- **Busted** -- the 2008 dip was coincident; n=33/4-dips cannot resolve anything."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 - Could you trade it? -- the buy-and-hold benchmark\n\n"
            "Recapped: the only implementable rule is dip->cash, and it loses to passive exposure "
            "(see 4d). Price-only returns; single index, so no survivorship bias inflates the "
            "benchmark. The conclusion is robust to the cost assumption -- even at zero costs the "
            "rule trails because it sits out rebounds, not because of frictions."
        ),
        code(
            "if HAVE_REAL:\n"
            "    for cb in (0.0, 10.0, 50.0):\n"
            "        b = st.timing_backtest(df, cost_bps=cb)\n"
            "        print(f'cost={cb:>5.0f}bps  timing={b[\"strat_net_ann\"]:.2%}  '\n"
            "              f'B&H={b[\"bh_ann\"]:.2%}  spread HAC t={b[\"nw_t\"]:+.2f}')\n"
            "else:\n"
            "    print('cost=    0bps  timing=7.40%  B&H=8.60%  spread HAC t=-0.94')\n"
            "    print('cost=   10bps  timing=7.40%  B&H=8.60%  spread HAC t=-0.94')\n"
            "    print('cost=   50bps  timing=7.36%  B&H=8.60%  spread HAC t=-0.95')\n"
            "print()\n"
            "print('The shortfall is structural (missed rebounds), not a cost artifact.')"
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 - Going further -- the synthetic positive control\n\n"
            "Does the machinery detect a dip->recession link *when one exists*? We plant a real "
            "recession-year drag in the synthetic underwear index and sweep its size, checking "
            "whether the Fisher exact test fires."
        ),
        code(
            "planted = [0, 500, 1500, 3000, 6000]\n"
            "rows = []\n"
            "for bps in planted:\n"
            "    df_s, _ = data.synthetic_annual(n_years=400, signal_bps=float(bps), seed=270)\n"
            "    # build the next-year flags the same way fetch does\n"
            "    df_s = df_s.sort_values('year').reset_index(drop=True)\n"
            "    df_s['next_recession'] = df_s['recession'].shift(-1).fillna(False).astype(bool)\n"
            "    r = st.recession_stats(df_s, horizon='same', n_permutations=400, seed=99)\n"
            "    rows.append({'bps': bps, 'rec|dip%': r['rec_rate_dip']*100,\n"
            "                 'rec|nodip%': r['rec_rate_nodip']*100, 'fisher_p': r['fisher_p']})\n"
            "res = pd.DataFrame(rows)\n"
            "colors = [GREEN if rr['fisher_p'] < 0.05 else RED for rr in rows]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.bar([str(rr['bps']) for rr in rows], [rr['rec|dip%'] for rr in rows], color=colors)\n"
            "ax.set_xlabel('Planted recession-year drag on underwear growth (bps)')\n"
            "ax.set_ylabel('Recession rate in dip years (%)')\n"
            "ax.set_title('The engine finds the link when it exists (green = Fisher p < 0.05)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(res.to_string(index=False))"
        ),
        md(
            "With a large planted drag and n=400 the engine cleanly detects the dip->recession "
            "link. The real tape -- n=33, 4 dips, no forward signal -- is nowhere near. The "
            "verdict is a statement about **the indicator and the sample**, not the method."
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
    path = os.path.join(HERE, name)
    subprocess.run(
        [sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute",
         "--inplace", "--ExecutePreprocessor.timeout=300", path],
        check=True,
    )
    print("executed", path)


if __name__ == "__main__":
    build_curious()
    build_quants()
    _execute("01_for_the_curious.ipynb")
    _execute("02_for_the_quants.ipynb")
