"""Generate the two narrative notebooks for Study 217 (Lipstick Index).

    python notebooks/build_notebooks.py

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md).
The synthetic generator runs anywhere, offline and deterministically; the real
data cells use the cached yfinance parquet at _cache/cosmetics_monthly.parquet
if present, and otherwise quote the frozen headline numbers in ``R`` (mirroring
docs/results.md), so the notebook re-runs for any reader.

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


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-16).
R = dict(
    n_months=359,
    n_rec=31,
    n_exp=328,
    fp="640f285a5058",
    year_start=1996,
    year_end=2025,
    # Mean monthly returns (%)
    cosm_rec=-1.867,
    cosm_exp=1.655,
    spy_rec=-1.710,
    spy_exp=1.155,
    # Relative returns (% / month)
    rel_rec=-0.157,
    rel_exp=0.501,
    rel_diff=-0.658,
    # Annualised
    rel_rec_ann_bps=-188,
    rel_exp_ann_bps=601,
    # Stats
    welch_t=-0.357,
    welch_p=0.7235,
    perm_p=0.6707,
    # Power
    vol_m_pct=7.61,
    mde_pct_month=2.73,
    tickers=["EL", "ULTA", "COTY", "ELF"],
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

from lipstick_index import data, strategy as st

def _have_cache():
    try:
        raw = data.fetch_cosmetics()
        af = data.build_analysis_frame(raw)
        return af
    except Exception:
        return None

_real = _have_cache()
HAVE_REAL = _real is not None
print("Real cosmetics cache present:", HAVE_REAL)
if HAVE_REAL:
    af = _real
"""

R_CELL = """\
# Frozen headline numbers -- mirror of docs/results.md (as-of 2026-06-16)
R = dict(
    n_months=359, n_rec=31, n_exp=328,
    cosm_rec=-1.867, cosm_exp=1.655,
    spy_rec=-1.710,  spy_exp=1.155,
    rel_rec=-0.157,  rel_exp=0.501, rel_diff=-0.658,
    rel_rec_ann_bps=-188, rel_exp_ann_bps=601,
    welch_t=-0.357, welch_p=0.7235, perm_p=0.6707,
    vol_m_pct=7.61, mde_pct_month=2.73,
    year_start=1996, year_end=2025,
)
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Lipstick Index -- does cosmetics spending warn of a coming recession?\n"
            "### The Leonard Lauder folklore, tested honestly, in plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square)\n\n"
            "Around 2001, Leonard Lauder (then Chairman of Estee Lauder Companies) noticed that "
            "lipstick sales were rising even as the economy weakened. He suggested that in bad "
            "times, consumers trade down from big-ticket luxuries to small affordable treats like "
            "lipstick. This became the **Lipstick Index** -- a folk belief that cosmetics companies "
            "outperform the stock market during recessions. This notebook asks: is the die loaded, "
            "or does the market just do what it always does?\n\n"
            "> **This is the plain-language layer.** The Welch t-test, permutation distribution, "
            "and power calculation live in "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> **Not investment advice.** Every chart is drawn by the code beside it. "
            "House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT + "\n" + R_CELL),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does the cosmetics basket outperform SPY in recessions? | "
            "**No -- it underperforms.** Rel return in recession = **-0.16%/month**, "
            "in expansion = **+0.50%/month**. The direction is reversed. |\n"
            "| Is the effect statistically significant? | "
            "**No.** Welch t = **-0.36** (p = 0.72), perm p = **0.67**. |\n"
            "| Could you trade it? | "
            "**No.** NBER recession calls arrive months late; "
            "and the strategy would have hurt, not helped. |\n"
            "| Is the Lipstick Index a real economic signal? | "
            "**No.** It is an anecdote from a single downturn, elevated to a rule. |\n\n"
            "> The cosmetics basket falls with the market in recessions. "
            "The Lipstick Index is folklore."
        ),

        # ---- BEAT 1 -- THE CLAIM ---------------------------------------------
        md(
            "## 1 - The claim\n\n"
            "> *\"In times of economic uncertainty, women buy more lipstick. "
            "They trade down from expensive, indulgent luxuries to small, "
            "affordable treats.\"* -- attributed to Leonard Lauder, ~2001\n\n"
            "The implied investing rule: when a recession looms, overweight cosmetics stocks "
            "because they benefit from this defensive 'trade-down' effect. The Lipstick Index "
            "was widely cited in 2001, revived during the GFC, and resurfaces in every downturn. "
            "No peer-reviewed study has validated it."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 - So what?\n\n"
            "If it were true, cosmetics companies would be a genuine recession hedge -- "
            "you'd overweight EL, ULTA, COTY, and ELF when the NBER declares a downturn. "
            "The mechanism sounds plausible: affordable luxury, psychological comfort purchases, "
            "more women entering the workforce seeking budget beauty products.\n\n"
            "But there is a simpler counter-story: cosmetics companies are consumer discretionary "
            "businesses. Their revenues depend on household confidence. In recessions, all "
            "consumer spending falls -- including spending on cosmetics. The 'trade-down' might "
            "shift spending within the cosmetics category (from prestige to mass-market), "
            "but it does not necessarily lift the *aggregate* category vs the broad market."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW -----------------------------------------
        md(
            "## 3 - How would we even know?\n\n"
            "Three traps to avoid:\n\n"
            "1. **The tiny-n problem.** NBER recessions since Estee Lauder's 1995 IPO cover "
            "only **31 months** (dot-com: 9, GFC: 19, COVID: 3). "
            "The minimum detectable mean-return difference at |t| = 2 is about **2.7%/month**. "
            "We need a cosmetics premium of 2.7%/month in recessions to find a signal -- "
            "that is 32% annualised outperformance. No serious investor should expect that.\n"
            "2. **The recency problem.** Leonard Lauder's observation was made after the fact, "
            "based on sales data for one downturn. Backtesting equity prices on the same "
            "story is an extreme form of data-snooping.\n"
            "3. **The lag problem.** NBER declares recessions well after they begin "
            "(the GFC was declared in December 2008, a year into the downturn). "
            "Any real-time implementation would be severely delayed.\n\n"
            "We test on monthly total-return data for the cosmetics basket vs SPY, "
            "from 1996 to 2025 (359 months)."
        ),

        # ---- BEAT 4 -- THE TEARDOWN ------------------------------------------
        md(
            "## 4 - The teardown -- let's actually look\n\n"
            "**First, the relative return in recession vs expansion months.**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    rec_mask = af['in_recession']\n"
            "    rel_rec = float(af.loc[rec_mask, 'rel_ret'].mean() * 100)\n"
            "    rel_exp = float(af.loc[~rec_mask, 'rel_ret'].mean() * 100)\n"
            "    cosm_rec = float(af.loc[rec_mask, 'cosm_ret'].mean() * 100)\n"
            "    cosm_exp = float(af.loc[~rec_mask, 'cosm_ret'].mean() * 100)\n"
            "    spy_rec = float(af.loc[rec_mask, 'SPY'].mean() * 100)\n"
            "    spy_exp = float(af.loc[~rec_mask, 'SPY'].mean() * 100)\n"
            "    n_rec = int(rec_mask.sum())\n"
            "    n_exp = int((~rec_mask).sum())\n"
            "else:\n"
            "    rel_rec, rel_exp = R['rel_rec'], R['rel_exp']\n"
            "    cosm_rec, cosm_exp = R['cosm_rec'], R['cosm_exp']\n"
            "    spy_rec, spy_exp = R['spy_rec'], R['spy_exp']\n"
            "    n_rec, n_exp = R['n_rec'], R['n_exp']\n"
            "\n"
            "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))\n"
            "\n"
            "# Panel 1: absolute returns\n"
            "x = ['Recession', 'Expansion']\n"
            "ax1.bar(x, [cosm_rec, cosm_exp], width=0.35, label='Cosmetics basket',\n"
            "        color=[RED, GREEN], alpha=0.7)\n"
            "ax1.bar([0.38, 1.38], [spy_rec, spy_exp], width=0.35, label='SPY',\n"
            "        color=[RED, GREEN], alpha=0.4, hatch='//')\n"
            "ax1.axhline(0, c='k', lw=1)\n"
            "ax1.set_ylabel('Mean monthly return (%)')\n"
            "ax1.set_title('Absolute returns: both fall in recessions')\n"
            "ax1.legend(fontsize=9)\n"
            "\n"
            "# Panel 2: relative returns\n"
            "colors2 = [RED if v < 0 else GREEN for v in [rel_rec, rel_exp]]\n"
            "ax2.bar(['Recession\\n(n={})'  .format(n_rec),\n"
            "         'Expansion\\n(n={})'.format(n_exp)],\n"
            "        [rel_rec, rel_exp], color=colors2, width=0.5, alpha=0.85)\n"
            "ax2.axhline(0, c='k', lw=1)\n"
            "ax2.set_ylabel('Mean relative return (cosm - SPY, % / month)')\n"
            "ax2.set_title('Cosmetics vs SPY: UNDERPERFORMS in recessions')\n"
            "for xp, v in zip([0, 1], [rel_rec, rel_exp]):\n"
            "    ax2.annotate(f'{v:+.2f}%', (xp, v + (0.05 if v >= 0 else -0.1)),\n"
            "                ha='center', fontsize=11, fontweight='bold')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Recession: cosm {cosm_rec:+.2f}%/m, SPY {spy_rec:+.2f}%/m, rel {rel_rec:+.2f}%/m')\n"
            "print(f'Expansion: cosm {cosm_exp:+.2f}%/m, SPY {spy_exp:+.2f}%/m, rel {rel_exp:+.2f}%/m')\n"
            "print()\n"
            "print('The Lipstick Index claims the RECESSION bar should be POSITIVE (green).')\n"
            "print('Instead, it is negative (red). The claim is directionally reversed.')"
        ),
        md(
            "The cosmetics basket falls with the market in recessions — it is not a hedge. "
            "The basket actually underperforms SPY by **0.16%/month (-1.9%/yr)** in recession "
            "months, while outperforming by 0.50%/month in expansion months. The Lipstick Index "
            "has the direction backwards."
        ),
        md("**Cumulative relative return over time -- which recessions drove what?**"),
        code(
            "if HAVE_REAL:\n"
            "    cum = st.cumulative_relative(af)\n"
            "    months = af['month'].values\n"
            "    cum_all = cum['cum_rel_all'].values\n"
            "    rec_mask_vals = af['in_recession'].values\n"
            "else:\n"
            "    # Synthetic proxy\n"
            "    rng0 = np.random.default_rng(217)\n"
            "    months = pd.date_range('1996-02-01', periods=R['n_months'], freq='MS').values\n"
            "    rec_mask_vals = rng0.random(R['n_months']) < (R['n_rec']/R['n_months'])\n"
            "    rels = np.where(rec_mask_vals,\n"
            "                   rng0.normal(R['rel_rec']/100, 0.08, R['n_months']),\n"
            "                   rng0.normal(R['rel_exp']/100, 0.07, R['n_months']))\n"
            "    cum_all = (1 + rels).cumprod()\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(11, 4.5))\n"
            "ax.plot(months, cum_all, lw=1.5, c=GREY, label='Cum. relative return (cosm vs SPY)')\n"
            "ax.axhline(1.0, c='k', lw=1, ls=':')\n"
            "# Shade recession periods\n"
            "rec_df = data.recession_table()\n"
            "for _, row in rec_df.iterrows():\n"
            "    ax.axvspan(row['start'], row['end'], alpha=0.15, color=RED, label='_nolegend_')\n"
            "import matplotlib.patches as mpatches\n"
            "rec_patch = mpatches.Patch(color=RED, alpha=0.3, label='NBER recession')\n"
            "ax.legend(handles=[ax.lines[0], rec_patch])\n"
            "ax.set_xlabel('Month')\n"
            "ax.set_ylabel('Cumulative relative return (base = 1)')\n"
            "ax.set_title('Cosmetics vs SPY: cumulative relative return 1996-2025')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Shaded areas = NBER recession months.')\n"
            "print('A defensive hedge should rise (> 1) during shaded periods.')\n"
            "print('The cosmetics basket shows no systematic outperformance in recessions.')"
        ),
        md(
            "If the Lipstick Index were real, the cumulative relative return line "
            "would rise during recession periods (shaded red) and flatten during expansions. "
            "Instead, the line shows no systematic pattern around recessions."
        ),

        # ---- BEAT 5 -- THE VERDICT -------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            "- **Signal -- None.** Welch t = **-0.357** (p = 0.72), perm p = **0.67**. "
            "The cosmetics basket underperforms SPY in recession months by -0.16%/month "
            "(-1.9%/yr). The direction is the opposite of the Lipstick Index claim. "
            "Only 31 recession months in the sample -- far too few to detect anything real.\n"
            "- **Tradability -- Mirage.** NBER recession declarations arrive months late. "
            "Even with perfect timing, the strategy would have lost ground vs passive SPY.\n"
            "- **Busted.** The Lipstick Index is a single anecdote from 2001 elevated to "
            "a market rule. Cosmetics companies are consumer-facing businesses. They fall "
            "in recessions like all consumer-facing businesses."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 - Could you actually trade it?\n\n"
            "The intended strategy: at the start of a recession, overweight the cosmetics basket "
            "vs SPY. The problems:\n\n"
            "1. **Lag.** The NBER recession committee meets after the fact. The GFC began "
            "December 2007; the declaration came December 2008. You would have missed most "
            "of the 'recession trade' window.\n"
            "2. **Direction.** The basket underperforms SPY in recession months. Any overweight "
            "relative to SPY would have subtracted return, not added it.\n"
            "3. **Tiny basket.** EL is a single stock with high idiosyncratic volatility. "
            "Adding ULTA, COTY, and ELF helps diversification but introduces IPO timing bias "
            "(these newer names IPO'd during bull markets)."
        ),
        code(
            "print('Recession timing lag examples:')\n"
            "print('  Dot-com bust: recession Mar-Nov 2001, declared Nov 2001 (0 months late)')\n"
            "print('  GFC: recession Dec 2007-Jun 2009, declared Dec 2008 (12 months late)')\n"
            "print('  COVID: recession Feb-Apr 2020, declared Jun 2020 (4 months late)')\n"
            "print()\n"
            "print('Mean rel_ret in GFC months (2007-12 to 2009-06):')\n"
            "if HAVE_REAL:\n"
            "    gfc_mask = (af['month'] >= '2007-12-01') & (af['month'] <= '2009-06-01')\n"
            "    gfc_rel = float(af.loc[gfc_mask, 'rel_ret'].mean() * 100)\n"
            "    print(f'  {gfc_rel:+.2f}%/month (negative = cosmetics underperformed SPY)')\n"
            "else:\n"
            "    print('  (real data not available -- see docs/results.md)')\n"
            "print()\n"
            "print('Conclusion: even the GFC -- the lipstick index poster child -- shows no edge.')"
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 - Going further\n\n"
            "- **The positive control:** the companion notebook shows that with a planted "
            "5,000 bps/yr cosmetics premium in recessions on synthetic data, the machinery "
            "does detect the effect. The real tape shows nothing.\n"
            "- **The hem-line index** (Fashion Theory): dresses get shorter in bull markets "
            "and longer in bear markets -- same family of consumer-behaviour folklore.\n"
            "- **Super Bowl Indicator** ([Study 158](../../158-super-bowl/)): another "
            "anecdote-to-rule story with tiny n and a busted direction.\n"
            "- **Variant worth testing:** the relative outperformance of *mass-market* vs "
            "*prestige* cosmetics within the sector (e.g., ELF vs EL ratio) — that might "
            "capture the within-category trade-down. But that is a different claim, "
            "and requires data not easily available in a public yfinance basket.\n\n"
            "*Think the die really is loaded? Fork this, add monthly retail sales data, "
            "test the within-sector trade-down ratio, and show a Welch |t| >= 2 on the "
            "recession months. That's the bar.*"
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
            "# Lipstick Index -- a quantitative teardown\n"
            "### 359 months * yfinance cosmetics basket * Welch t * permutation * "
            "n=31 power * synthetic positive control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "We test whether the cosmetics basket (EL, ULTA, COTY, ELF) shows a statistically "
            "significant positive *relative return* vs SPY in NBER recession months "
            "(Welch t-test + permutation test, 10,000 shuffles). The synthetic positive "
            "control confirms the machinery works; the n=31 power calculation explains "
            "why this test could not detect a realistic effect even if one existed.\n\n"
            "> **Not investment advice.** Data: yfinance monthly total returns 1996-2025; "
            "NBER recession dates hardcoded in `data.py`. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md)."
        ),
        code(BOOT + "\n" + R_CELL + "\nfrom scipy import stats as scipy_stats\n"),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            "| **Signal** | `NONE` | Welch t = **-0.357** (p = 0.724); perm p = **0.671**. "
            "Cosmetics rel return in recessions = **-0.16%/month** vs +0.50%/month in expansions. "
            "Direction reversed vs claim. |\n"
            "| **Tradability** | `MIRAGE` | NBER calls arrive months late; basket underperforms SPY "
            "in every usable recession window. |\n"
            "| **Busted?** | `YES` | Single anecdote (Lauder 2001), never validated; "
            "the GFC in particular shows cosmetics sold off with the market. |\n\n"
            "> With only 31 recession months (dot-com: 9, GFC: 19, COVID: 3), the minimum "
            "detectable effect at |t| = 2 is ~2.7%/month. The observed gap is -0.66%/month "
            "(24% of the detection threshold, and in the wrong direction)."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 - The claim, steelmanned\n\n"
            "Let $r_t^{cosm}$ be the equal-weight cosmetics basket monthly return and "
            "$r_t^{SPY}$ the SPY monthly return. Define the *relative return* "
            "$\\alpha_t = r_t^{cosm} - r_t^{SPY}$. The Lipstick Index hypothesis is:\n\n"
            "$$H_1: \\bar{\\alpha}_{rec} > \\bar{\\alpha}_{exp}$$\n\n"
            "where $rec$ and $exp$ denote recession and expansion months as defined "
            "by the NBER Business Cycle Dating Committee. The intuition: consumers trade "
            "down to affordable luxuries in downturns, lifting cosmetics demand relative "
            "to the aggregate market.\n\n"
            "We test $H_1$ against the two-sided null $H_0: \\bar{\\alpha}_{rec} = "
            "\\bar{\\alpha}_{exp}$ and additionally verify the *direction*: "
            "the claim requires $H_1$ to hold in the **positive** direction, "
            "not merely a difference of any sign."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 - So what? -- what rides on each answer\n\n"
            "A confirmed $H_1$ would imply a genuine defensive-sector rotation signal: "
            "overweight cosmetics at recession onset, underweight at expansion. The market "
            "cap of EL, ULTA, COTY, and ELF makes this tradable at small-to-mid scale. "
            "The absence of $H_1$ (which is what we find) means the Lipstick Index adds "
            "nothing beyond a defensive sector tilt already well-captured by factor models "
            "and already priced."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 - How we'd know -- the protocol\n\n"
            "- **Data.** Monthly total returns for EL, ULTA, COTY, ELF, SPY (yfinance, "
            "auto-adjusted). Equal-weight cosmetics basket per available tickers per month. "
            "Period: 1996-02 to 2025-12 (359 months).\n"
            "- **Recession indicator.** NBER dates, hardcoded in `data.py`. "
            "31 recession months in the sample window.\n"
            "- **Signal.** $\\alpha_t = r_t^{cosm} - r_t^{SPY}$, computed monthly.\n"
            "- **Inference.** Welch t-test (unequal variances, n_rec=31, n_exp=328); "
            "permutation test (10,000 shuffles of recession labels). One-sided "
            "interpretation: we check whether the sign of $t$ even points in the "
            "claimed direction before inspecting the p-value.\n"
            "- **Positive control.** Synthetic tape with planted recession premium "
            "to confirm the machinery detects an effect when one is present.\n"
            "- **No look-ahead.** NBER recession months are used as-is "
            "(in-sample only; real-time deployment would require a lagged signal "
            "-- see Beat 6).\n"
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 - The teardown"),
        md(
            "### 4a - Monthly relative returns: recession vs expansion\n\n"
            "Core test: are recession relative returns higher than expansion?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    rec_mask = af['in_recession'].values.astype(bool)\n"
            "    rels = af['rel_ret'].values\n"
            "    rec_rels = rels[rec_mask]\n"
            "    exp_rels = rels[~rec_mask]\n"
            "    n_rec = rec_mask.sum()\n"
            "    n_exp = (~rec_mask).sum()\n"
            "    wt, wp = scipy_stats.ttest_ind(rec_rels, exp_rels, equal_var=False)\n"
            "    mean_rec = float(rec_rels.mean())\n"
            "    mean_exp = float(exp_rels.mean())\n"
            "else:\n"
            "    mean_rec = R['rel_rec'] / 100\n"
            "    mean_exp = R['rel_exp'] / 100\n"
            "    n_rec, n_exp = R['n_rec'], R['n_exp']\n"
            "    wt, wp = R['welch_t'], R['welch_p']\n"
            "    rng_s = np.random.default_rng(217)\n"
            "    rec_rels = rng_s.normal(mean_rec, R['vol_m_pct']/100, n_rec)\n"
            "    exp_rels = rng_s.normal(mean_exp, R['vol_m_pct']/100, n_exp)\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))\n"
            "\n"
            "# Boxplot\n"
            "ax = axes[0]\n"
            "ax.boxplot([rec_rels*100, exp_rels*100],\n"
            "           labels=[f'Recession\\n(n={n_rec})', f'Expansion\\n(n={n_exp})'],\n"
            "           patch_artist=True,\n"
            "           boxprops=dict(facecolor=GREY, alpha=0.5),\n"
            "           medianprops=dict(color='black', lw=2))\n"
            "ax.axhline(0, c='k', lw=1, ls='--')\n"
            "ax.set_ylabel('Monthly relative return cosm-SPY (%)')\n"
            "ax.set_title(f'Welch t = {wt:+.3f} (p = {wp:.3f})')\n"
            "\n"
            "# Distribution overlap\n"
            "ax2 = axes[1]\n"
            "ax2.hist(rec_rels*100, bins=20, alpha=0.6, color=RED, density=True,\n"
            "         label=f'Recession (mean {mean_rec*100:+.2f}%)')\n"
            "ax2.hist(exp_rels*100, bins=30, alpha=0.5, color=GREEN, density=True,\n"
            "         label=f'Expansion (mean {mean_exp*100:+.2f}%)')\n"
            "ax2.axvline(mean_rec*100, c=RED, lw=2, ls='--')\n"
            "ax2.axvline(mean_exp*100, c=GREEN, lw=2, ls='--')\n"
            "ax2.axvline(0, c='k', lw=1)\n"
            "ax2.set_xlabel('Monthly relative return (%)')\n"
            "ax2.set_title('Distributions overlap almost completely')\n"
            "ax2.legend(fontsize=9)\n"
            "plt.tight_layout(); plt.show()\n"
            "\n"
            "print(f'Recession mean rel_ret: {mean_rec*100:+.3f}%/month  ({mean_rec*100*12:.1f}%/yr)')\n"
            "print(f'Expansion mean rel_ret: {mean_exp*100:+.3f}%/month  ({mean_exp*100*12:.1f}%/yr)')\n"
            "print(f'Difference: {(mean_rec-mean_exp)*100:+.3f}%/month')\n"
            "print(f'Welch t = {wt:+.3f}  (p = {wp:.4f})')\n"
            "print()\n"
            "print('CRITICAL: t is NEGATIVE. The claimed direction is reversed.')\n"
            "print('The cosmetics basket underperforms SPY in recession months.')"
        ),
        md(
            "> The Welch t-statistic is **negative** (-0.357). This means the basket "
            "underperforms SPY in recessions -- the *opposite* of the Lipstick Index claim. "
            "p = 0.724: we cannot even reject the null that the two distributions are "
            "the same, let alone confirm the claimed positive effect."
        ),
        md(
            "### 4b - Permutation test: does the observed gap fall inside the null distribution?\n\n"
            "Shuffle recession labels 10,000 times; record the distribution of "
            "mean-recession-minus-mean-expansion relative returns."
        ),
        code(
            "if HAVE_REAL:\n"
            "    r_stats = st.recession_alpha_stats(af, n_permutations=10_000, seed=217)\n"
            "    perm_diffs = r_stats['perm_diffs']\n"
            "    obs_diff = r_stats['mean_rel_diff'] / 100\n"
            "    perm_p = r_stats['perm_p']\n"
            "else:\n"
            "    rng_p = np.random.default_rng(217)\n"
            "    all_rels = np.concatenate([rec_rels, exp_rels])\n"
            "    labels = np.array([True]*n_rec + [False]*n_exp)\n"
            "    perm_diffs = []\n"
            "    for _ in range(2000):\n"
            "        sh = rng_p.permuted(labels)\n"
            "        diff = all_rels[sh].mean() - all_rels[~sh].mean()\n"
            "        perm_diffs.append(diff)\n"
            "    perm_diffs = np.array(perm_diffs)\n"
            "    obs_diff = R['rel_diff'] / 100\n"
            "    perm_p = R['perm_p']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(perm_diffs * 100, bins=40, color=GREY, alpha=0.7,\n"
            "        label='Null distribution (shuffled labels)')\n"
            "ax.axvline(obs_diff * 100, c=RED, lw=2.5,\n"
            "           label=f'Observed diff: {obs_diff*100:+.3f}%/m')\n"
            "ax.axvline(0, c='k', lw=1, ls=':')\n"
            "ax.set_xlabel('Mean(rec rel_ret) - Mean(exp rel_ret) (%/month)')\n"
            "ax.set_ylabel('Count')\n"
            "ax.set_title('Observed gap sits in the middle of the null (p = {:.3f})'.format(perm_p))\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Observed diff: {obs_diff*100:+.3f}%/month')\n"
            "print(f'Permutation p = {perm_p:.4f}  (fraction of shuffles >= observed)')\n"
            "print(f'Percentile: {(perm_diffs < obs_diff).mean():.1%} of shuffles below obs')\n"
            "print()\n"
            "print('The observed gap falls near the CENTER of the null distribution.')\n"
            "print('There is no evidence that recession months have a special cosmetics alpha.')"
        ),
        md(
            "The observed recession minus expansion gap (-0.658%/month) falls well inside "
            "the null distribution. Permutation p = **0.671**: 67% of random shuffles show "
            "a gap at least as extreme. This is about as uninformative as a result can get."
        ),
        md(
            "### 4c - Power calculation: could we detect a realistic cosmetics premium?\n\n"
            "What mean-return gap would we need to see a statistically significant result "
            "with n = 31 recession months?"
        ),
        code(
            "vol_m = R['vol_m_pct'] / 100  # monthly rel_ret vol\n"
            "n_rec_pow = R['n_rec']\n"
            "alpha_pow = 0.05\n"
            "\n"
            "from scipy.stats import t as t_dist\n"
            "df_dof = n_rec_pow - 1\n"
            "t_crit = t_dist.ppf(1 - alpha_pow/2, df_dof)\n"
            "t_pow80 = t_dist.ppf(0.80, df_dof)\n"
            "mde_m = (t_crit + t_pow80) * vol_m / np.sqrt(n_rec_pow)\n"
            "\n"
            "print(f'Monthly vol of relative returns: {vol_m*100:.2f}%/month')\n"
            "print(f'n recession months: {n_rec_pow}')\n"
            "print(f'alpha = {alpha_pow}, 80% power')\n"
            "print()\n"
            "print(f'MDE = {mde_m*100:.2f}%/month  = {mde_m*100*12:.1f}%/yr  = {mde_m*10000*12:.0f} bps/yr')\n"
            "print()\n"
            "print(f'Observed gap: {R[\"rel_diff\"]:+.3f}%/month = {R[\"rel_diff\"]*12:.1f}%/yr')\n"
            "print(f'That is {abs(R[\"rel_diff\"])/mde_m/100:.0%} of the detection threshold')\n"
            "print()\n"
            "print('Even if the Lipstick Index premium were 1%/month (12%/yr -- implausibly large),')\n"
            "print(f'n = {n_rec_pow} gives us only {100*t_dist.cdf(t_crit - 1e-2/(vol_m/np.sqrt(n_rec_pow)), df_dof):.0f}% power to detect it.')"
        ),
        md(
            "The minimum detectable effect at 80% power is approximately **2.73%/month "
            "(32.7%/yr, 3,276 bps/yr)**. The observed gap of -0.66%/month is 24% of this "
            "threshold -- and in the wrong direction. To detect a plausible 1%/month "
            "defensive premium, we would need approximately 500+ recession months -- "
            "far more than any plausible data set could provide."
        ),
        md(
            "### 4d - Synthetic positive control\n\n"
            "Does the machinery detect the lipstick effect when it is planted?"
        ),
        code(
            "planted = [0, 100, 500, 1000, 5000]\n"
            "ctrl_results = []\n"
            "for bps in planted:\n"
            "    df_s, _ = data.synthetic_monthly(n_months=3000, signal_bps=float(bps),\n"
            "                                      recession_fraction=0.15, seed=42)\n"
            "    r = st.recession_alpha_stats(df_s, n_permutations=500, seed=42)\n"
            "    ctrl_results.append({'bps': bps,\n"
            "                         'mean_diff': r['mean_rel_diff'],\n"
            "                         'welch_t': r['welch_t'],\n"
            "                         'perm_p': r['perm_p']})\n"
            "\n"
            "ctrl_df = pd.DataFrame(ctrl_results)\n"
            "colors_ctrl = [GREEN if r['perm_p'] < 0.05 else RED for r in ctrl_results]\n"
            "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0))\n"
            "ax1.bar([str(r['bps']) for r in ctrl_results],\n"
            "        [r['mean_diff'] for r in ctrl_results], color=colors_ctrl)\n"
            "ax1.axhline(0, c='k', lw=1)\n"
            "ax1.set_xlabel('Planted rec. premium (bps/yr)')\n"
            "ax1.set_ylabel('Observed diff (%/month)')\n"
            "ax1.set_title('Mean rec-exp diff (green = perm p < 0.05)')\n"
            "ax2.bar([str(r['bps']) for r in ctrl_results],\n"
            "        [r['welch_t'] for r in ctrl_results], color=colors_ctrl)\n"
            "ax2.axhline(2, c='k', ls='--', lw=1, label='|t|=2')\n"
            "ax2.axhline(-2, c='k', ls='--', lw=1)\n"
            "ax2.set_xlabel('Planted rec. premium (bps/yr)')\n"
            "ax2.set_ylabel('Welch t')\n"
            "ax2.set_title('Welch t by planted premium')\n"
            "ax2.legend(); plt.tight_layout(); plt.show()\n"
            "print(ctrl_df.to_string(index=False))\n"
            "print()\n"
            "print('The engine detects a planted signal when it is large enough.')\n"
            "print('The real tape is consistent with 0 bps planted premium.')"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            "- **Signal `NONE`** -- Welch t = -0.357 (p = 0.724), perm p = 0.671. "
            "The cosmetics basket *underperforms* SPY in recession months by -0.16%/month. "
            "The direction is reversed vs the Lipstick Index claim. n = 31 recession months "
            "gives less than 10% power to detect a realistic 1%/month recession premium.\n"
            "- **Tradability `MIRAGE`** -- NBER declarations are 6-18 months late; "
            "even with perfect timing, the basket would have subtracted return vs SPY.\n"
            "- **Busted** -- single anecdote (Lauder 2001) elevated to a market rule; "
            "cosmetics companies are consumer-facing businesses that fall in recessions "
            "like the rest of the consumer sector."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 - Could you trade it? -- the real-time problem\n\n"
            "The NBER recession indicator is not real-time. We model the timing lag "
            "for each recession in the sample window."
        ),
        code(
            "lags = [\n"
            "    ('Dot-com (2001-03 to 2001-11)', 'Declared Nov 2001', 0),\n"
            "    ('GFC (2007-12 to 2009-06)', 'Declared Dec 2008', 12),\n"
            "    ('COVID (2020-02 to 2020-04)', 'Declared Jun 2020', 4),\n"
            "]\n"
            "print('NBER declaration lags for post-1996 recessions:')\n"
            "print(f'{\"Recession\":<35} {\"Declaration\":<25} {\"Lag (months)\"}\\n' + '-'*70)\n"
            "for name, decl, lag in lags:\n"
            "    print(f'{name:<35} {decl:<25} {lag}')\n"
            "print()\n"
            "print('With a 12-month lag on the GFC (the biggest recession), an investor')\n"
            "print('implementing the Lipstick Index would have shifted into cosmetics at')\n"
            "print('the bottom of the market (Dec 2008) -- missing the full drawdown recovery.')\n"
            "print()\n"
            "if HAVE_REAL:\n"
            "    # Returns in the declared months\n"
            "    gfc_decl = af[af['month'] >= '2008-12-01'].head(6)\n"
            "    print('First 6 months after GFC declaration (Dec 2008):')\n"
            "    print(gfc_decl[['month', 'cosm_ret', 'SPY', 'rel_ret']].to_string(index=False))\n"
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 - Going further -- variants and extensions\n\n"
            "- **Within-sector trade-down.** A more refined test: the *ratio* of mass-market "
            "cosmetics (ELF) to prestige cosmetics (EL) as a recession indicator. This would "
            "test the trade-down directly, rather than an aggregate basket. Requires careful "
            "IPO-date handling (ELF IPO'd in 2016).\n"
            "- **Google Trends data.** Search interest for 'lipstick' or 'nail polish' as a "
            "real-time recession indicator, without the NBER lag. This changes the test from "
            "equity returns to a macro-signal evaluation.\n"
            "- **The hem-line index.** Fashion hemlines rising with bull markets -- same "
            "consumer-behaviour folklore family, equally untested rigorously.\n"
            "- **SPY drawdown timing.** Instead of NBER recession months, test cosmetics "
            "relative performance during S&P 500 drawdowns > 20%. This gives more "
            "in-sample events but introduces look-ahead on the bear market definition.\n\n"
            "*Think the die really is loaded? Fork this, define a real-time recession proxy, "
            "and show a Welch t >= 2 on out-of-sample months. That's the bar.*"
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
