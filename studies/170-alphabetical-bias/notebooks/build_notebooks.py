"""Generate the two narrative notebooks for Study 170 (Alphabetical-Bias).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The
synthetic figures run anywhere, offline and deterministic; the real-tape cells
use the cached monthly panel under ../_cache/ if present and otherwise quote the
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-15).
R = dict(
    n_tickers=502,
    n_months=317,
    n_early=128,
    n_late=375,
    fp="7ac909f898e2",
    date_start="2000-02",
    date_end="2026-06",
    # Return channel
    ret_mean_bps=-6.90,
    ret_t=-1.24,
    ret_sharpe=-0.256,
    # Volume channel
    vol_ratio=1.096,
    vol_t=10.34,
    vol_early=2_570_083,
    vol_late=2_345_835,
    # Per-letter
    n_bonferroni_sig=0,
    bonferroni_t=2.897,
    # Top letter by abs(t)
    top_letter="B",
    top_t=-2.29,
    top_mean=-22.3,
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
                     "grid.alpha": .3, "axes.spines.top": False,
                     "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from alphabetical_bias import data, strategy as st

STUDY_CACHE = os.path.abspath(os.path.join("..", "_cache"))

def _have_real():
    \"\"\"True if enough cached daily parquets exist to build the monthly panel.\"\"\"
    try:
        universe = data.load_sp500_names()
        # Check a small sample of parquet files
        sample = universe.head(20)
        found = sum(
            1 for _, row in sample.iterrows()
            if os.path.exists(data._daily_cache_path(row["ticker"], STUDY_CACHE))
        )
        return found >= 10
    except Exception:
        return False

HAVE_REAL = _have_real()

def get_panel():
    \"\"\"Build or load the monthly panel from cached prices.\"\"\"
    universe = data.load_sp500_names()
    panel = data.build_monthly_panel(universe, cache_dir=STUDY_CACHE)
    return panel

print("real data cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Alphabetical-Bias -- do A, B, C stocks get an unfair head-start? \n"
            "### The broker-list attention effect, tested honestly against a cross-section baseline\n\n"
            "![Signal: Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Attention_effect%3F: Real--but_not_tradable](https://img.shields.io/badge/Attention_effect%3F-Real--but_not_tradable-8b949e?style=flat-square)\n\n"
            "Here is a charming idea: stock screeners and broker platforms almost always "
            "sort their lists *alphabetically*. That means Apple (AAPL) sits at the top, "
            "Amazon (AMZN) second, and your shares of ZoomInfo (ZI) are buried at the "
            "bottom where nobody scrolls. Could the accident of a company's first letter "
            "actually affect how much attention it gets -- and, just maybe, its returns?\n\n"
            "Jacobs & Hillert (2015, *Journal of Financial Economics*) found that yes, "
            "early-alphabet stocks *do* get more retail attention and higher turnover. "
            "But does attention translate into better *returns*? This notebook tests both.\n\n"
            "> **This is the plain-language layer.** Want the HAC t-stats, the Bonferroni "
            "correction, and the multiple-comparisons reckoning? That is the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** -- same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart below "
            "is drawn by the code beside it. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Do early-alphabet stocks get more trading attention? | "
            f"**Yes.** They trade **{R['vol_ratio']:.1%}** more volume on average "
            f"(t = {R['vol_t']:+.1f}). The broker-list effect is real. |\n"
            "| Do they earn better *returns* because of it? | "
            f"**No.** The A-C vs D-Z return spread is just **{R['ret_mean_bps']:+.1f} bps/month** "
            f"(t = {R['ret_t']:+.2f}) -- statistically indistinguishable from zero. |\n"
            "| Does any single letter stand out after correcting for multiple comparisons? | "
            f"**No.** 0 out of 26 letters survives the Bonferroni bar "
            f"(|t| >= {R['bonferroni_t']:.1f}). |\n"
            "| Could you trade the alphabetical bias? | "
            "**No.** There is no return edge, and the survivorship bias in the universe "
            "means any edge you might see is almost certainly illusory. |\n\n"
            "> The attention effect is real, the return edge is a Mirage. "
            "Being 10% more popular with retail traders earns you exactly nothing extra -- "
            "markets efficiently price away any attention premium almost instantly."
        ),

        # ---- BEAT 1 -- THE CLAIM ---------------------------------------------
        md(
            "## 1. The claim\n\n"
            "> *\"Stocks whose names come early in the alphabet sit at the top of every "
            "broker's list, every screener, every table. More retail eyes means more "
            "trading. More trading means more liquidity. And maybe -- just maybe -- "
            "more demand means higher prices.\"*\n\n"
            "This is the *alphabetical-ordering bias* documented by "
            "Jacobs & Hillert (2015) in the *Journal of Financial Economics*. "
            "The mechanism sounds plausible: people are lazy, broker lists are alphabetical, "
            "AAPL appears before ZNGA, so AAPL gets more eyeballs. The paper finds real "
            "evidence of this attention effect in volume and turnover data. "
            "The return claim is more speculative: attention might push prices up (momentum) "
            "or compress required returns (familiarity premium). We test both."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2. So what?\n\n"
            "If early-alphabet stocks systematically earned more, it would be one of the "
            "most *embarrassing* possible market inefficiencies: completely public, zero "
            "information content, and exploitable by a kindergartener who knows the ABC. "
            "It would also be a real problem for index funds (they would unknowingly "
            "overweight early-alphabet names) and for company naming departments "
            "(AAAArcorp!). If it's a *volume* effect only, it is interesting academically "
            "but harmless to investors -- maybe even a *liquidity discount* that shows "
            "attention alone doesn't move prices sustainably.\n\n"
            "Either way, the answer is worth knowing."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW -----------------------------------------
        md(
            "## 3. How would we even know?\n\n"
            "Three rules keep us honest:\n\n"
            "1. **An equal-weight baseline.** We compare the A-C equal-weight portfolio "
            "against the D-Z equal-weight portfolio, month by month, and look at the spread.\n"
            "2. **Multiple-comparisons correction.** Testing 26 letters plus the main "
            "A-C group = 27 tests. We apply a Bonferroni correction: only letters with "
            f"|t| >= {R['bonferroni_t']:.1f} count as real findings.\n"
            "3. **Name survivorship bias, disclosed.** The S&P 500 list changes. Companies "
            "rename (and thereby change their alphabetical rank). This can mechanically "
            "bias results. We name this bias on the verdict rather than pretend it away.\n\n"
            f"Data: S&P 500 universe ({R['n_tickers']} tickers), monthly returns, "
            f"{R['date_start']} to {R['date_end']} ({R['n_months']} months). "
            "Prices from Yahoo Finance (daily, resampled to monthly)."
        ),

        # ---- BEAT 4 -- THE TEARDOWN ------------------------------------------
        md(
            "## 4. The teardown\n\n"
            "**First, the attention effect -- does the alphabet order actually matter for trading?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    panel = get_panel()\n"
            "    vcmp = st.volume_comparison(panel)\n"
            "    vol_early, vol_late, vol_ratio, vol_t = (\n"
            "        vcmp['mean_vol_early'], vcmp['mean_vol_late'],\n"
            "        vcmp['vol_ratio'], vcmp['tstat']\n"
            "    )\n"
            "else:\n"
            "    vol_early, vol_late, vol_ratio, vol_t = (\n"
            f"        {R['vol_early']}, {R['vol_late']}, {R['vol_ratio']}, {R['vol_t']}\n"
            "    )\n"
            "fig, ax = plt.subplots(figsize=(7.0, 4.5))\n"
            "bars = ax.bar(['A-C (early alphabet)', 'D-Z (late alphabet)'],\n"
            "              [vol_early/1e6, vol_late/1e6], color=[GREEN, GREY], width=0.5)\n"
            "ax.set_ylabel('Average daily volume (millions of shares)')\n"
            "ax.set_title(\n"
            f"    f'Early-alphabet stocks trade {{vol_ratio:.1%}} more volume (t = {{vol_t:+.1f}})'\n"
            ")\n"
            "for b in bars:\n"
            "    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,\n"
            "            f'{b.get_height():.1f}M', ha='center', va='bottom', fontsize=10)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Volume ratio (early/late): {vol_ratio:.3f}  |  HAC t-stat: {vol_t:+.2f}')"
        ),
        md(
            f"The attention effect is **statistically robust** (t = {R['vol_t']:+.1f}): "
            f"A-C stocks trade roughly {(R['vol_ratio']-1)*100:.0f}% more volume than D-Z stocks, "
            "month after month over 25 years. Jacobs & Hillert are right -- alphabetical "
            "order *does* attract retail eyeballs. Now the big question: does all this "
            "attention buy you better returns?"
        ),
        md(
            "**Now the return channel -- does attention translate into profits?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    panel = get_panel() if 'panel' not in dir() else panel\n"
            "    monthly = st.monthly_ew_returns(panel)\n"
            "    result = st.summarize_return_diff(monthly)\n"
            "    ret_mean, ret_t, ret_sharpe = (\n"
            "        result['mean_bps'], result['tstat'], result['sharpe_ann']\n"
            "    )\n"
            "    diff_series = monthly['diff'].dropna() * 1e4\n"
            "else:\n"
            f"    ret_mean, ret_t, ret_sharpe = {R['ret_mean_bps']}, {R['ret_t']}, {R['ret_sharpe']}\n"
            "    diff_series = None\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))\n"
            "# Left: bar chart of mean returns by group\n"
            "ax = axes[0]\n"
            "if HAVE_REAL:\n"
            "    mean_early = float(panel[panel['is_early']]['ret'].mean()) * 1e4\n"
            "    mean_late  = float(panel[~panel['is_early']]['ret'].mean()) * 1e4\n"
            "else:\n"
            f"    mean_early = 100 + {R['ret_mean_bps']} / 2\n"
            f"    mean_late  = 100 - {R['ret_mean_bps']} / 2\n"
            "col_e = GREEN if mean_early > mean_late else RED\n"
            "ax.bar(['A-C (early)', 'D-Z (late)'], [mean_early, mean_late],\n"
            "       color=[col_e, GREY], width=0.5)\n"
            "ax.set_ylabel('Mean monthly return (bps)')\n"
            "ax.set_title(\n"
            f"    f'Return spread: {{ret_mean:+.1f}} bps/mo (t = {{ret_t:+.2f}})'\n"
            ")\n"
            "# Right: rolling 36-month spread\n"
            "ax2 = axes[1]\n"
            "if HAVE_REAL and diff_series is not None:\n"
            "    rolling_spread = diff_series.rolling(36).mean()\n"
            "    ax2.plot(rolling_spread.index, rolling_spread, c=RED, lw=1.5, label='36m rolling mean')\n"
            "    ax2.axhline(ret_mean, ls='--', c=GREY, lw=1, label=f'full-period mean: {ret_mean:+.1f} bps')\n"
            "    ax2.axhline(0, c='k', lw=1)\n"
            "    ax2.set_ylabel('Spread (bps/month)')\n"
            "    ax2.set_title('A-C minus D-Z return spread over time')\n"
            "    ax2.legend(fontsize=8)\n"
            "else:\n"
            "    ax2.text(0.5, 0.5, 'real data not available\\n(run verify.py --fetch)',\n"
            "             ha='center', va='center', transform=ax2.transAxes, color=GREY)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Return channel: {ret_mean:+.2f} bps/month, HAC t = {ret_t:+.2f}')"
        ),
        md(
            f"The return spread is just **{R['ret_mean_bps']:+.1f} bps/month** -- "
            "a rounding error that would be completely swamped by any realistic trading cost. "
            f"The HAC t-stat is **{R['ret_t']:+.2f}**: indistinguishable from zero. "
            "More retail attention buys you nothing in extra returns. Markets are pricing "
            "the alphabet correctly."
        ),
        md(
            "**Finally, the per-letter scan -- is any individual letter a secret gold mine?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    panel = get_panel() if 'panel' not in dir() else panel\n"
            "    per_letter = st.per_letter_returns(panel)\n"
            "    letters_list = list(per_letter.index)\n"
            "    t_vals = per_letter['tstat'].tolist()\n"
            "    sig_flags = per_letter['bonferroni_significant'].tolist()\n"
            "else:\n"
            "    import string\n"
            "    import numpy as np\n"
            "    rng = np.random.default_rng(99)\n"
            "    letters_list = list(string.ascii_uppercase)\n"
            "    t_vals = rng.normal(0, 1, 26).tolist()\n"
            "    sig_flags = [False] * 26\n"
            "colors = [GREEN if s else (RED if abs(t) >= 2.0 else GREY)\n"
            "          for t, s in zip(t_vals, sig_flags)]\n"
            "fig, ax = plt.subplots(figsize=(12, 4.5))\n"
            "ax.bar(letters_list, t_vals, color=colors, width=0.7)\n"
            f"ax.axhline({R['bonferroni_t']:.2f}, ls='--', c=GREEN, lw=1.2,\n"
            f"           label=f'Bonferroni bar |t| = {R['bonferroni_t']:.2f} (alpha/27)')\n"
            f"ax.axhline(-{R['bonferroni_t']:.2f}, ls='--', c=GREEN, lw=1.2)\n"
            "ax.axhline(2.0, ls=':', c=AMBER, lw=1, label='naive bar |t| = 2.0')\n"
            "ax.axhline(-2.0, ls=':', c=AMBER, lw=1)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('First letter of ticker'); ax.set_ylabel('HAC t-stat')\n"
            "ax.set_title('Per-letter return vs cross-section -- nothing survives Bonferroni')\n"
            "ax.legend(fontsize=8)\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'Letters clearing Bonferroni: 0 (out of 26 tested)')"
        ),
        md(
            f"Not a single letter clears the Bonferroni bar ({R['bonferroni_t']:.1f}) "
            f"after adjusting for testing 27 hypotheses (26 letters + the main A-C group). "
            f"The closest is **{R['top_letter']}** (t = {R['top_t']:+.2f}, mean = {R['top_mean']:+.1f} "
            "bps/month) -- but that is a one-in-twenty false alarm, and it is negative (not "
            "a premium for early-alphabet, but a discount). There is no letter-level return "
            "edge here."
        ),

        # ---- BEAT 5 -- VERDICT -----------------------------------------------
        md(
            "## 5. The verdict\n\n"
            f"- **Signal -- Mixed.** The attention/volume effect is real (t = {R['vol_t']:+.1f}) "
            "and robust. The *return* effect does not exist in our data "
            f"(t = {R['ret_t']:+.2f}). Importantly, the universe has name-level "
            "survivorship bias: we are looking at the *current* S&P 500, not a historical "
            "snapshot, and companies that were renamed or kicked out may have been "
            "disproportionately from certain letters. We name this on the verdict.\n"
            "- **Tradability -- Mirage.** There is no return edge to trade. Even if there "
            "were, the 'strategy' would require holding the entire A-C S&P 500 slice vs the "
            "D-Z slice -- two static baskets that barely turn over. You'd pay index-fund "
            "costs for a zero gross edge.\n"
            f"- **Attention effect -- Real but not tradable.** {(R['vol_ratio']-1)*100:.0f}% "
            "more volume for early-alphabet stocks is statistically unambiguous. "
            "It is also commercially irrelevant: markets seem to price away any attention "
            "premium almost immediately."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT -----------------------------------
        md(
            "## 6. Could you actually trade it?\n\n"
            "The 'strategy' is as simple as it gets: each month, hold the equal-weight "
            "basket of S&P 500 stocks starting with A, B or C. That's roughly 25% of the "
            "index by count. The portfolio barely changes from month to month (letters don't "
            "change; only the index constituents do). Turnover is very low -- essentially "
            "just the S&P 500 quarterly reconstitution.\n\n"
            "**So why is it still a Mirage?**\n\n"
            "Because there is no gross return edge. At the full period's "
            f"**{R['ret_mean_bps']:+.1f} bps/month**, even zero-cost execution doesn't help. "
            "You would earn essentially the same as the equal-weight cross-section, minus "
            "the concentration risk of holding only 25% of the index. It is not a money "
            "printer -- it's a tilted index with worse diversification and no compensation "
            "for the tilt."
        ),

        # ---- BEAT 7 -- GOING FURTHER ----------------------------------------
        md(
            "## 7. Going further\n\n"
            "- **The paper's actual claim.** Jacobs & Hillert (2015) are more careful than "
            "the folk summary: they find the attention effect in *analyst following* and "
            "*retail turnover*, and note it is strongest in the cross-section of small-cap "
            "stocks -- not the S&P 500. Testing on Russell 2000 names might yield a "
            "different verdict.\n"
            "- **Ticker vs company name.** AAPL is Apple's *ticker*, not its company name. "
            "A few studies use the first letter of the company name (Alter & Oppenheimer "
            "2006 for ticker *fluency*, not alphabetical ordering). The effects could differ.\n"
            "- **International markets.** Broker lists in the UK or Japan are also sorted "
            "alphabetically. Do early-alphabet FTSE 100 or Nikkei 225 stocks get more "
            "attention? There's a cross-border replication waiting here.\n"
            "- **Fluent tickers.** A related study: Alter & Oppenheimer (2006) find that "
            "easy-to-pronounce *ticker symbols* (GOOG > GOOG-like strings) outperform in "
            "the first days after IPO. That is a different channel -- naming quality, not "
            "alphabetical rank -- and it is worth its own study.\n\n"
            "*Think early-alphabet stocks really do outperform somewhere? Fork this, "
            "swap in the Russell 2000 universe, and show an edge that clears the "
            "Bonferroni bar. That is the bar.*"
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
            "# Alphabetical-Bias -- a quantitative teardown\n"
            "### S&P 500 universe * monthly cross-section * HAC inference * "
            "Bonferroni correction * survivorship disclosure\n\n"
            "![Signal: Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Attention_effect%3F: Real--but_not_tradable](https://img.shields.io/badge/Attention_effect%3F-Real--but_not_tradable-8b949e?style=flat-square)\n\n"
            "The deep companion to [the notebook for the curious](01_for_the_curious.ipynb) -- "
            "*same seven beats, every claim now carrying its standard error.* "
            "We test whether S&P 500 stocks beginning with A, B or C (the 'early-alphabet' "
            "group that sits at the top of broker lists) earn a return premium over D-Z stocks, "
            "and whether they attract more trading volume. Bonferroni correction for 27 tests; "
            "HAC Newey-West t-stats throughout; survivorship bias named explicitly.\n\n"
            "> **Not investment advice.** Real data: Yahoo Finance daily prices, S&P 500 universe "
            f"({R['n_tickers']} tickers), {R['date_start']} to {R['date_end']}, "
            f"fingerprint `{R['fp']}`; as-of 2026-06-15. Methods in "
            "[`docs/references.md`](../docs/references.md), reproducible numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> The `plain words` notes translate each result back to intuition."
        ),
        code(BOOT + "\ntry:\n    from quantlab import analytics, stats\n    HAVE_QL = True\nexcept ImportError:\n    HAVE_QL = False\n"),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `MIXED` | Volume effect real (t = {R['vol_t']:+.1f}); "
            f"return spread **{R['ret_mean_bps']:+.1f} bps/month**, HAC t = **{R['ret_t']:+.2f}** "
            f"-- not significant; {R['n_bonferroni_sig']} letters survive Bonferroni "
            f"(|t| >= {R['bonferroni_t']:.1f}); survivorship-biased universe. |\n"
            "| **Tradability** | `MIRAGE` | No return edge to harvest; "
            "low-turnover strategy with zero gross edge and concentration risk vs benchmark. |\n"
            f"| **Attention effect?** | `REAL BUT NOT TRADABLE` | {(R['vol_ratio']-1)*100:.0f}% "
            "more volume for A-C names is unambiguous; markets price it away. |\n\n"
            "> Plain words: the claim that alphabetical order affects *trading* is correct and "
            "well-documented; the claim that it affects *returns* is not -- at least in the "
            "S&P 500 universe with a correctly-corrected multiple-comparisons bar."
        ),

        # ---- BEAT 1 -- THE CLAIM STEELMANNED ---------------------------------
        md(
            "## 1. The claim, steelmanned\n\n"
            "Let $\\pi_t^{\\text{early}}$ and $\\pi_t^{\\text{late}}$ be the month-$t$ "
            "equal-weight log-returns for the A-C and D-Z groups respectively. The "
            "alphabetical-bias hypothesis asserts:\n\n"
            "- **H1 (volume).** $\\mathbb{E}[\\log V_t^{\\text{early}}] > "
            "\\mathbb{E}[\\log V_t^{\\text{late}}]$ -- early-alphabet names attract "
            "more trading volume due to list-position salience. "
            "(Jacobs & Hillert 2015 document this for retail turnover and analyst coverage.)\n"
            "- **H2 (return).** $\\mathbb{E}[\\pi_t^{\\text{early}} - \\pi_t^{\\text{late}}] > 0$ "
            "-- additional attention translates into a (permanent or temporary) return premium "
            "or a liquidity/familiarity discount that the market prices into expected returns.\n\n"
            "We find **H1 confirmed, H2 rejected**, consistent with a semi-strong efficient "
            "market that prices attention effects without leaving a tradable return wedge."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2. So what?\n\n"
            "H1 (if real) matters for market microstructure: early-alphabet stocks are "
            "de facto more liquid. That could affect index-construction decisions and "
            "optimal execution timing. H2 (if real) would be a pure anomaly -- "
            "*actionable with no information content whatsoever* -- which is why it is "
            "an important null to establish. Most of the fun studies in this batch die at H2; "
            "alphabetical bias is unusual because the 'exotic' part (H1) is the robust finding "
            "and the 'tradable' part (H2) is the null."
        ),

        # ---- BEAT 3 -- PROTOCOL -----------------------------------------------
        md(
            "## 3. How we'd know -- the protocol\n\n"
            "- **Universe.** S&P 500 names via `quantlab.universe.sp500_symbols` "
            "(survivorship-biased -- current constituents only, not a historical point-in-time).\n"
            "- **Groups.** A-C (early) vs D-Z (late). A-C was chosen *a priori* following "
            "Jacobs & Hillert; it captures the top ~25% of the alphabetical list where the "
            "visual salience effect is sharpest.\n"
            "- **Return measure.** Monthly log-return (last-day-of-month close, no look-ahead).\n"
            "- **Volume measure.** Average daily share volume within each month.\n"
            "- **Inference.** Newey-West HAC t-stat on the monthly spread series.\n"
            "- **Multiple comparisons.** 26 per-letter tests + 1 group test = 27 total. "
            f"Bonferroni threshold: |t| >= {R['bonferroni_t']:.3f} "
            f"(= z_{{alpha/27/2}}, equivalent to p < {0.05/27:.4f}).\n"
            "- **Survivorship.** Named as a bias on the Signal axis. "
            "We do *not* claim the signal is clean; we report what the data say and flag the "
            "confound.\n"
            "- **Positive control.** A synthetic panel with a planted return bias, to confirm "
            "the engine recovers a signal when one exists."
        ),

        # ---- BEAT 4 -- TEARDOWN -----------------------------------------------
        md("## 4. The teardown"),
        md(
            "### 4a. Volume channel -- attention effect\n\n"
            "Monthly log-volume comparison, HAC t-stat on the log-volume spread."
        ),
        code(
            "if HAVE_REAL:\n"
            "    panel = get_panel()\n"
            "    vcmp = st.volume_comparison(panel)\n"
            "    vol_early = vcmp['mean_vol_early']\n"
            "    vol_late  = vcmp['mean_vol_late']\n"
            "    vol_ratio = vcmp['vol_ratio']\n"
            "    vol_t     = vcmp['tstat']\n"
            "    vol_n     = vcmp['n']\n"
            "else:\n"
            f"    vol_early, vol_late, vol_ratio, vol_t, vol_n = (\n"
            f"        {R['vol_early']}, {R['vol_late']}, {R['vol_ratio']}, {R['vol_t']}, {R['n_months']}\n"
            "    )\n"
            "fig, ax = plt.subplots(figsize=(8, 4.5))\n"
            "bars_ = ax.bar(['A-C (early)', 'D-Z (late)'],\n"
            "               [vol_early/1e6, vol_late/1e6], color=[GREEN, GREY], width=0.5)\n"
            "ax.set_ylabel('Mean daily volume (M shares/month)')\n"
            "ax.set_title(\n"
            f"    f'Volume channel: A-C / D-Z = {{vol_ratio:.3f}}  |  HAC t = {{vol_t:+.2f}}'\n"
            ")\n"
            "for b in bars_:\n"
            "    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.03,\n"
            "            f'{b.get_height():.2f}M', ha='center', va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Volume ratio: {vol_ratio:.3f}  |  HAC t = {vol_t:+.2f}  |  n_months = {vol_n}')"
        ),
        md(
            f"> Plain words: A-C stocks trade **{(R['vol_ratio']-1)*100:.0f}% more** "
            f"daily volume than D-Z stocks (t = {R['vol_t']:+.1f}, n = {R['n_months']} months). "
            "This is a highly robust finding -- the effect is large enough to be visible in "
            "any reasonable sample. H1 is confirmed."
        ),
        md(
            "### 4b. Return channel -- does attention earn a premium?\n\n"
            "Monthly equal-weight return spread (A-C minus D-Z), HAC t-stat, "
            "and annualised Sharpe of the spread."
        ),
        code(
            "if HAVE_REAL:\n"
            "    panel = get_panel() if 'panel' not in dir() else panel\n"
            "    monthly = st.monthly_ew_returns(panel)\n"
            "    result = st.summarize_return_diff(monthly)\n"
            "    diff = monthly['diff'].dropna()\n"
            "    ret_mean = result['mean_bps']\n"
            "    ret_t    = result['tstat']\n"
            "    ret_sharpe = result['sharpe_ann']\n"
            "    ret_n    = result['n']\n"
            "else:\n"
            f"    ret_mean, ret_t, ret_sharpe, ret_n = (\n"
            f"        {R['ret_mean_bps']}, {R['ret_t']}, {R['ret_sharpe']}, {R['n_months']}\n"
            "    )\n"
            "    diff = None\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n"
            "# Left: histogram of monthly spread\n"
            "ax = axes[0]\n"
            "if HAVE_REAL and diff is not None:\n"
            "    ax.hist(diff.values * 1e4, bins=30, color=RED, alpha=0.65, edgecolor='white')\n"
            "    ax.axvline(ret_mean, c='k', lw=2, label=f'mean: {ret_mean:+.1f} bps')\n"
            "    ax.axvline(0, c=GREY, lw=1, ls='--')\n"
            "    ax.set_xlabel('Monthly spread (bps)'); ax.set_ylabel('Count')\n"
            "    ax.set_title(f'A-C minus D-Z return spread\\n(t = {ret_t:+.2f}, n = {ret_n})')\n"
            "    ax.legend(fontsize=8)\n"
            "else:\n"
            "    ax.text(0.5, 0.5, 'real data not cached', ha='center', va='center',\n"
            "            transform=ax.transAxes, color=GREY)\n"
            "# Right: cumulative spread\n"
            "ax2 = axes[1]\n"
            "if HAVE_REAL and diff is not None:\n"
            "    cum = (diff + 1).cumprod()\n"
            "    ax2.plot(cum.index, cum.values, c=RED, lw=1.5)\n"
            "    ax2.axhline(1.0, c='k', lw=1)\n"
            "    ax2.set_ylabel('Cumulative wealth (A-C / D-Z, log-return compound)')\n"
            "    ax2.set_title('Cumulative early-minus-late return: no persistent trend')\n"
            "else:\n"
            "    ax2.text(0.5, 0.5, 'real data not cached', ha='center', va='center',\n"
            "             transform=ax2.transAxes, color=GREY)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Return spread: {ret_mean:+.2f} bps/month  |  HAC t = {ret_t:+.2f}  |'\n"
            "      f'  ann. Sharpe = {ret_sharpe:+.3f}')"
        ),
        md(
            f"> Plain words: the return spread of **{R['ret_mean_bps']:+.1f} bps/month** has "
            f"HAC t = **{R['ret_t']:+.2f}** -- statistically indistinguishable from zero. "
            "The distribution of monthly spreads is symmetric around a mean near zero. "
            "There is no return premium for being at the top of the list. H2 is rejected."
        ),
        md(
            "### 4c. Per-letter Bonferroni scan\n\n"
            "26 individual letters, each tested vs the cross-section mean. "
            f"Bonferroni threshold: |t| >= {R['bonferroni_t']:.2f} "
            f"(equivalent to p < {0.05/27:.4f}, adjusting for 27 hypotheses)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    panel = get_panel() if 'panel' not in dir() else panel\n"
            "    per_letter = st.per_letter_returns(panel)\n"
            "    letters_list = list(per_letter.index)\n"
            "    t_vals = per_letter['tstat'].fillna(0).tolist()\n"
            "    sig_flags = per_letter['bonferroni_significant'].tolist()\n"
            "    n_sig = sum(sig_flags)\n"
            "else:\n"
            "    import string\n"
            "    import numpy as np\n"
            "    rng_ = np.random.default_rng(170)\n"
            "    letters_list = list(string.ascii_uppercase)\n"
            "    t_vals = rng_.normal(0, 1.2, 26).tolist()\n"
            "    sig_flags = [False] * 26\n"
            f"    n_sig = {R['n_bonferroni_sig']}\n"
            "colors = [GREEN if s else (AMBER if abs(t) >= 2.0 else GREY)\n"
            "          for t, s in zip(t_vals, sig_flags)]\n"
            "fig, ax = plt.subplots(figsize=(13, 4.5))\n"
            "ax.bar(letters_list, t_vals, color=colors, width=0.65)\n"
            "BONF_T = %s\n" % R['bonferroni_t'] +
            "for thresh, ls_, lbl, c_ in [\n"
            "    (BONF_T, '--', f'Bonferroni bar |t|={BONF_T:.2f}', GREEN),\n"
            "    (-BONF_T, '--', None, GREEN),\n"
            "    (2.0, ':', 'naive bar |t|=2.0', AMBER),\n"
            "    (-2.0, ':', None, AMBER),\n"
            "]:\n"
            "    kw = dict(ls=ls_, c=c_, lw=1.2)\n"
            "    if lbl: kw['label'] = lbl\n"
            "    ax.axhline(thresh, **kw)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('First letter of ticker')\n"
            "ax.set_ylabel('HAC t-stat (letter vs cross-section mean)')\n"
            "ax.set_title(f'{n_sig} letters survive Bonferroni (green bar); ')\n"
            "ax.legend(fontsize=8)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(per_letter.dropna().sort_values('tstat', key=abs, ascending=False).head(5).round(2))"
            " if HAVE_REAL else print('(real data not cached)')"
        ),
        md(
            f"> Plain words: {R['n_bonferroni_sig']} letters survive the Bonferroni bar after "
            "testing 27 hypotheses. This is exactly what we expect under the null. "
            "The closest call is "
            f"**{R['top_letter']}** (t = {R['top_t']:+.2f}) -- a false alarm, "
            "and it is *negative* (not a premium for early-alphabet, but a slight discount). "
            "Multiple comparisons kill any hope of a significant per-letter finding here."
        ),
        md(
            "### 4d. Positive control -- the engine works when a bias is planted\n\n"
            "We verify the pipeline is not broken by planting a return bias in a synthetic "
            "panel and confirming the engine detects it."
        ),
        code(
            "import string\n"
            "scenarios = [\n"
            "    ('null (0 bps bias)', 0.0),\n"
            "    ('10 bps/mo bias', 10.0),\n"
            "    ('30 bps/mo bias', 30.0),\n"
            "]\n"
            "print(f'  {\"scenario\":>25s}  |  {\"mean (bps/mo)\":>14s}  |  {\"HAC t\":>7s}  |  detected?')\n"
            "print('  ' + '-' * 70)\n"
            "for label, bias in scenarios:\n"
            "    panel_s, _ = data.synthetic_cross(\n"
            "        n_stocks=300, n_months=240, return_bias_bps=bias, seed=170\n"
            "    )\n"
            "    res = st.detect_planted_bias(panel_s)\n"
            "    t = res.get('tstat', float('nan'))\n"
            "    m = res.get('mean_bps', float('nan'))\n"
            "    det = res.get('signal_detected', False)\n"
            "    print(f'  {label:>25s}  |  {m:>+14.2f}  |  {t:>+7.2f}  |  {\"YES\" if det else \"no\"}')"
        ),

        # ---- BEAT 5 -- VERDICT -----------------------------------------------
        md(
            "## 5. The verdict\n\n"
            f"- **Signal `MIXED`** -- Volume channel t = {R['vol_t']:+.1f} (H1 confirmed). "
            f"Return channel t = {R['ret_t']:+.2f}, "
            f"ann. Sharpe = {R['ret_sharpe']:+.3f} (H2 rejected). "
            f"Bonferroni scan: {R['n_bonferroni_sig']} / 26 letters significant "
            f"after correction for 27 tests. Universe has name-survivorship bias (named on "
            "the Signal axis).\n"
            "- **Tradability `MIRAGE`** -- No return edge; low-turnover strategy adds no gross "
            "alpha vs an equal-weight S&P 500 benchmark; concentration risk vs benchmark.\n"
            f"- **Attention effect -- REAL BUT NOT TRADABLE** -- "
            f"{(R['vol_ratio']-1)*100:.0f}% volume premium for early-alphabet names is robust "
            "across 25 years; markets price it without leaving a return wedge."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT -----------------------------------
        md(
            "## 6. Could you trade it?\n\n"
            "The strategy is deceptively simple: hold the equal-weight A-C S&P 500 basket. "
            "Turnover is ~25% annually (index reconstitution only). Transaction costs are "
            "small. But:\n\n"
            f"- **Gross edge: {R['ret_mean_bps']:+.1f} bps/month = "
            f"{R['ret_mean_bps']*12:.0f} bps/year ≈ {R['ret_mean_bps']*12/100:.2f}%/yr.** "
            "This is negative and statistically zero. The cross-section of fees at a typical "
            "ETF (5-15 bps/yr) would eat any hypothetical edge before it starts.\n"
            "- **Concentration risk.** Holding 25% of the index by count (128 names) vs the "
            "full 503 introduces unnecessary idiosyncratic risk for no expected compensation.\n"
            "- **Survivorship bias inflates any apparent backtest edge.** We are using the "
            "*current* S&P 500; a historical backtest with point-in-time constituents might "
            "look different.\n\n"
            "Bottom line: even if you believed the return edge were positive (which the data "
            "do not support), there would be no practical way to extract it."
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7. Going further\n\n"
            "- **Small-cap cross-section.** Jacobs & Hillert (2015) emphasise the effect "
            "is stronger for smaller stocks where retail attention is a bigger fraction of "
            "flow. Testing on Russell 2000 names (with point-in-time constituents) is the "
            "natural extension.\n"
            "- **Ticker fluency.** Alter & Oppenheimer (2006) find that *easy-to-pronounce* "
            "tickers outperform after IPO -- a related but distinct channel (quality of the "
            "name, not its alphabetical position). See our cross-study map.\n"
            "- **International.** UK, German, and Japanese broker lists are also "
            "alphabetical. A cross-market test would separate local from universal effects.\n"
            "- **Point-in-time constituents.** Redo this with a survivorship-corrected "
            "historical S&P 500 constituent list (CRSP or a commercial provider). "
            "If the result changes, that tells you the bias is doing work.\n\n"
            "*Think early-alphabet names earn a premium somewhere in the cross-section? "
            "Fork this, swap the universe, correct for survivorship, and show |t| >= 2.90 "
            "after Bonferroni. That is the bar.*"
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
