"""Generate the two narrative notebooks for Study 335 (Buzz-Sentiment-ETF).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the cached daily
parquets under ../_cache/ if present and otherwise quote the frozen headline numbers in
``R`` (mirroring docs/results.md), so the notebook re-runs for any reader, online or off.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-18,
# tape trimmed to the last full month, 2021-03-04 → 2026-05-29, fp 52101bc91bce).
R = dict(
    n_days=1316,
    buzz_total=74.8, buzz_cagr=11.3, buzz_vol=32.7, buzz_sharpe=0.49, buzz_dd=-56.9,
    spy_total=115.7, spy_cagr=15.9, spy_vol=16.9, spy_sharpe=0.96, spy_dd=-24.5,
    beta=1.59, alpha_ann=-9.7, alpha_t=-1.26, r2=0.68,
    ci_lo=-24.5, ci_hi=4.8, p_pos=9.2,
    active_bps=-0.05, active_t=-0.01, info_ratio=-0.005,
    overlay_sharpe=0.52, overlay_switches=135,
    # synthetic controls (n_days=3000 tape, as shown in beat 7)
    null_t=-0.61, pos_t=2.99, pos_alpha=10.1,
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

from buzz_sentiment_etf import data, strategy as st

def load_real_or_none():
    \"\"\"Return the real (BUZZ, SPY) tape if cached, else None (offline fallback).\"\"\"
    try:
        df = data.load_real(fetch=False)
        df = df[df.index <= "2026-05-31"]   # drop the partial month, as pinned
        return df
    except Exception as e:
        print("Real cache unavailable -> showing the SYNTHETIC tape:", type(e).__name__)
        return None

REAL = load_real_or_none()
HAVE_REAL = REAL is not None
print("real BUZZ/SPY cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Buzz-Sentiment-ETF — does an AI that reads the crowd beat the market? 📣\n"
            "### A real, clickable 'AI social-sentiment' ETF, raced against a plain index fund\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![AI_sentiment_beats_the_market%3F: Busted](https://img.shields.io/badge/AI_sentiment_beats_the_market%3F-Busted-8b949e?style=flat-square)\n\n"
            "There's an ETF — **BUZZ** — whose whole pitch is that an AI reads social media, news "
            "and online chatter, scores which big US stocks the crowd is most bullish on, and holds "
            "the 75 most-loved names. It went viral at launch when a celebrity promoted it. The "
            "promise is irresistible: let a machine read the room and pick the winners. This "
            "notebook asks the only question that matters — *did it actually beat just buying the "
            "market?*\n\n"
            "> 📓 **This is the plain-language layer.** Want the alpha regression, the t-stats and "
            "the bootstrap? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool: every chart is drawn by "
            "the code beside it. When the real BUZZ/SPY cache isn't present the cells fall back to a "
            "clearly-labelled synthetic tape. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ------------------------------------------------
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Did the AI basket beat the market? | **No.** Over its whole life BUZZ returned "
            f"~+{R['buzz_cagr']:.0f}%/yr vs SPY's ~+{R['spy_cagr']:.0f}%/yr — about **5 points a "
            "year less**. |\n"
            "| Was it at least *smarter* risk for the money? | **No — worse.** Roughly **double the "
            f"swings** (vol {R['buzz_vol']:.0f}% vs {R['spy_vol']:.0f}%), about **half** the "
            f"risk-adjusted return, and a brutal **{R['buzz_dd']:.0f}%** crash vs SPY's "
            f"{R['spy_dd']:.0f}%. |\n"
            "| Is there any real skill (alpha) in it? | **No.** Strip out the fact that it's just a "
            f"bigger bet on the same market, and what's left is **{R['alpha_ann']:.0f}%/yr of "
            f"*negative* alpha** — and not even reliably negative (it's noise). |\n"
            "| So what is it, really? | A **high-beta, high-fee index fund wearing an AI costume.** "
            "The story sold; the performance didn't. |\n\n"
            "> BUZZ isn't a fraud — it really does hold a basket built from sentiment data. It just "
            "doesn't *work*: 'an AI reads the crowd' is a great story, and the tape says the crowd, "
            "read by a machine, is not where market-beating returns come from."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Harness the wisdom of the crowd. The BUZZ NextGen AI US Sentiment Leaders Index "
            "uses artificial intelligence to analyse millions of online sources — social media, "
            "news, blogs — and identify the 75 most positively-perceived US large-cap stocks.\"*\n\n"
            "— VanEck, on the Social Sentiment ETF (ticker **BUZZ**)\n\n"
            "The intuition is seductive. Markets are crowds; sentiment moves prices; an AI can read "
            "sentiment faster and wider than any human. So a fund that systematically buys whatever "
            "the crowd loves *ought* to ride those waves. It launched in March 2021, right in the "
            "meme-stock frenzy, and a celebrity pump sent money pouring in."
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "Two things make this worth a careful look. First, it's a **real product** — not a "
            "back-tested idea on a blog, but a fund real people put real money into on the strength "
            "of the 'AI reads the crowd' story. Second, the desk has already torn down the *raw "
            "signals* underneath it — [r/WallStreetBets mention counts](../../254-wsb-mentions/) and "
            "[aggregate Twitter mood](../../256-twitter-mood/), both **None/Mirage**. The honest "
            "question now is the one a buyer actually faces: *forget the theory — did the shrink-"
            "wrapped fund deliver?* And the lesson, if it didn't, transfers to every 'AI-powered' "
            "ETF you'll ever be pitched."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "There's a trap to dodge first: in a rising market, a fund that takes **bigger** bets "
            "than the index will out-return it on the way up — and that looks like genius until the "
            "first big drop. That's not skill, it's just leverage in disguise (**beta**). So:\n\n"
            "1. **Race it against SPY**, the plain S&P 500 fund — same money, same period, who ends "
            "up ahead on a *risk-adjusted* basis?\n"
            "2. **Strip out the beta.** Statistically remove 'it's just a bigger market bet' and see "
            "if any genuine skill (**alpha**) is left. *This* is the number that decides it.\n"
            "3. **Sanity-check the engine** on a fake tape where we *plant* a known amount of skill, "
            "to prove our test can find skill when it's really there.\n\n"
            "Tape: BUZZ and SPY daily, BUZZ's full live history (since March 2021)."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the simplest possible test: $1 in each, who's ahead?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    lvl = REAL / REAL.iloc[0] * 100\n"
            "    summ = st.net_of_fee_summary(REAL)\n"
            "    bc, sc = summ.loc['buzz','cagr_pct'], summ.loc['spy','cagr_pct']\n"
            "    bd, sd = summ.loc['buzz','max_dd_pct'], summ.loc['spy','max_dd_pct']\n"
            "    tag = 'REAL TAPE — BUZZ vs SPY'\n"
            "else:\n"
            "    # offline fallback: a synthetic NULL (alpha=0) — dressed-up beta, labelled as such\n"
            "    syn, _ = data.synthetic_pair(alpha_bps=0.0)\n"
            "    lvl = syn / syn.iloc[0] * 100\n"
            "    summ = st.net_of_fee_summary(syn)\n"
            "    bc, sc = summ.loc['buzz','cagr_pct'], summ.loc['spy','cagr_pct']\n"
            "    bd, sd = summ.loc['buzz','max_dd_pct'], summ.loc['spy','max_dd_pct']\n"
            "    tag = 'SYNTHETIC NULL (alpha=0) — illustration only, NOT the real tape'\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.6))\n"
            "ax.plot(lvl.index, lvl['buzz'], color=RED, lw=1.8, label='BUZZ (AI sentiment)')\n"
            "ax.plot(lvl.index, lvl['spy'], color=GREEN, lw=1.8, label='SPY (plain index)')\n"
            "ax.set_ylabel('growth of 100'); ax.legend(loc='upper left')\n"
            "ax.set_title(tag)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'CAGR  BUZZ {bc:+.1f}%/yr   SPY {sc:+.1f}%/yr')\n"
            "print(f'Worst drawdown  BUZZ {bd:.1f}%   SPY {sd:.1f}%')"
        ),
        md(
            f"On the real tape BUZZ ends **well behind** SPY (~+{R['buzz_cagr']:.0f}%/yr vs "
            f"~+{R['spy_cagr']:.0f}%/yr) and falls **much harder** when the market drops "
            f"(**{R['buzz_dd']:.0f}%** vs {R['spy_dd']:.0f}%). Out of the gate that already looks "
            "bad — but maybe it's just taking bigger swings. Let's check whether the swings are at "
            "least *paying* for themselves."
        ),
        md(
            "**Now the decisive one: strip out the beta and look for real skill.** We remove the "
            "'it's just a bigger market bet' part and measure what's left — the *alpha*."
        ),
        code(
            "if HAVE_REAL:\n"
            "    a = st.capm_alpha(st.daily_returns(REAL))\n"
            "    beta, alpha_ann, alpha_t = a['beta'], a['alpha_ann_pct'], a['tstat']\n"
            "else:\n"
            f"    beta, alpha_ann, alpha_t = {R['beta']}, {R['alpha_ann']}, {R['alpha_t']}\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "a1.bar(['BUZZ market exposure\\n(beta)'], [beta], color=GREY, width=.5)\n"
            "a1.axhline(1, ls='--', c='k', lw=1); a1.set_ylabel('beta vs SPY')\n"
            "a1.set_title(f'A {beta:.2f}x bet on the SAME market')\n"
            "a2.bar(['Skill left over\\n(alpha)'], [alpha_ann], color=RED, width=.5)\n"
            "a2.axhline(0, c='k', lw=1); a2.set_ylabel('alpha (%/yr)')\n"
            "a2.set_title(f'Alpha {alpha_ann:+.1f}%/yr  (t={alpha_t:+.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Beta {beta:.2f}  ->  BUZZ is a leveraged bet on the index.')\n"
            "print(f'Alpha {alpha_ann:+.1f}%/yr (t={alpha_t:+.2f})  ->  no positive skill; if anything, negative.')"
        ),
        md(
            f"There it is. BUZZ's beta is **{R['beta']:.2f}** — it's a *{R['beta']:.2f}-times bet on "
            "the very same market** it's supposedly out-smarting. And once you remove that, the "
            f"skill left over is **{R['alpha_ann']:.0f}%/yr — *negative***. (The *t*-stat of "
            f"{R['alpha_t']:.2f} is small, so it's not reliably negative either — it's just **no "
            "edge**.) The 'AI reads the crowd' machine added nothing you couldn't get, cheaper, from "
            "an index fund."
        ),

        # ---- BEAT 5 — THE VERDICT --------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** No positive alpha. After stripping beta, what's left is "
            f"{R['alpha_ann']:.0f}%/yr at *t* = {R['alpha_t']:.2f} — indistinguishable from "
            "(or worse than) zero skill.\n"
            f"- **Tradability — Mirage.** BUZZ lagged SPY by ~5%/yr with **double the vol**, about "
            f"**half** the risk-adjusted return, and a **{R['buzz_dd']:.0f}%** drawdown. It's "
            "high-beta beta plus a fee, not an edge.\n"
            "- **'AI sentiment beats the market'? — Busted.** The real product, over its real life, "
            "delivered a worse outcome on every risk-adjusted measure than just buying the index."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "You literally *can* — it's a real ETF, one click. The question is whether you'd *want* "
            "to. Could a little market-timing on top rescue it — hold whichever of BUZZ/SPY has the "
            "stronger recent trend?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    ov = st.sentiment_overlay(REAL)\n"
            "    ov_sharpe = ov['sharpe_net']\n"
            "    sp_sharpe = st.excess_sharpe(st.daily_returns(REAL))['sharpe_spy']\n"
            "    bz_sharpe = st.excess_sharpe(st.daily_returns(REAL))['sharpe_buzz']\n"
            "else:\n"
            f"    ov_sharpe, sp_sharpe, bz_sharpe = {R['overlay_sharpe']}, {R['spy_sharpe']}, {R['buzz_sharpe']}\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "b = ax.bar(['Buy & hold\\nSPY', '20d momentum\\nswitch (net)', 'Buy & hold\\nBUZZ'],\n"
            "           [sp_sharpe, ov_sharpe, bz_sharpe], color=[GREEN, AMBER, RED], width=.55)\n"
            "ax.set_ylabel('Sharpe (excess of cash)'); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_title('Nothing beats simply holding SPY')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'SPY {sp_sharpe:.2f}   overlay {ov_sharpe:.2f}   BUZZ {bz_sharpe:.2f}')"
        ),
        md(
            f"No rescue. Plain SPY has the best Sharpe (**{R['spy_sharpe']:.2f}**); the timing "
            f"overlay just spends costs flipping between the two legs and lands in the middle "
            f"(~{R['overlay_sharpe']:.2f}); BUZZ alone is worst (**{R['buzz_sharpe']:.2f}**). The "
            "honest 'could you trade it?' answer: yes, and you'd be choosing to do worse than the "
            "index you were already free to buy."
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The raw signals underneath.** [Study 254 — WSB-Mentions](../../254-wsb-mentions/) "
            "and [Study 256 — Twitter-Mood](../../256-twitter-mood/) test the sentiment data BUZZ "
            "packages — both dead on arrival. This study shows the *product* inherits the verdict.\n"
            "- **The thematic-ETF pattern.** Hyped, narrative-driven ETFs launched near peak "
            "attention tend to *underperform* afterward (Ben-David et al. 2023). BUZZ, sold *on* "
            "social-media buzz, is almost a parody of the pattern — a fork worth testing across the "
            "whole 'AI/theme ETF' shelf.\n"
            "- **Is the fee the whole story?** No — the negative alpha is far bigger than the "
            "expense ratio. The basket construction, not just the fee, is the drag. Fork this and "
            "attribute it to sector tilts vs name selection.\n\n"
            "*Think a different sentiment ETF, or a different benchmark, changes the verdict? Fork "
            "this notebook, swap the ticker, and show alpha clearing t = 2 on a real tape.*"
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
            "# Buzz-Sentiment-ETF — a quantitative teardown 🔬\n"
            "### Real ETF tape · CAPM alpha + HAC *t* · block-bootstrap CI · information ratio · "
            "excess-vs-excess Sharpe · synthetic positive control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![AI_sentiment_beats_the_market%3F: Busted](https://img.shields.io/badge/AI_sentiment_beats_the_market%3F-Busted-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We grade the VanEck Social "
            "Sentiment ETF (BUZZ) the way an active product must be graded: not on return but on "
            "**alpha** — the CAPM intercept of BUZZ's excess return on SPY's — net of beta and the "
            "fee, with a Newey-West *t* and a block-bootstrap CI.\n\n"
            "> ⚠️ **Not investment advice.** Real data: Yahoo daily total-return bars, BUZZ + SPY, "
            "2021-03-04 → 2026-05-29 (as-of 2026-06-18, partial month dropped); the offline core and "
            "tests run on a deterministic synthetic pair. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition — so "
            "this notebook still reads even if you skim the maths."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Real-tape alpha **{R['alpha_ann']:.1f}%/yr**, HAC *t* = "
            f"**{R['alpha_t']:.2f}**; bootstrap P(α>0) = **{R['p_pos']:.0f}%**; 95% CI "
            f"[{R['ci_lo']:.1f}, {R['ci_hi']:.1f}]%/yr. Active-return IR ≈ {R['info_ratio']:.2f}. |\n"
            f"| **Tradability** | `MIRAGE` | BUZZ Sharpe **{R['buzz_sharpe']:.2f}** vs SPY "
            f"**{R['spy_sharpe']:.2f}**; β = **{R['beta']:.2f}**, vol {R['buzz_vol']:.0f}% vs "
            f"{R['spy_vol']:.0f}%, DD {R['buzz_dd']:.0f}% vs {R['spy_dd']:.0f}%. Dressed-up beta. |\n"
            f"| **AI sentiment beats the market?** | `BUSTED` | CAGR {R['buzz_cagr']:.1f}%/yr vs "
            f"{R['spy_cagr']:.1f}%/yr — worse return *and* worse risk. |\n\n"
            "> 💡 In plain words: BUZZ is a high-beta, concentrated bet on the same index, plus a "
            "fee. Once you account for that exposure there is no skill left — the alpha point "
            "estimate is negative and statistically a wash. A great story, a closet-index outcome."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r^{BUZZ}_t, r^{SPY}_t$ be daily excess (of cash) returns and fit the CAPM:\n\n"
            "$$r^{BUZZ}_t = \\alpha + \\beta\\, r^{SPY}_t + \\varepsilon_t.$$\n\n"
            "- **H₁ (skill / alpha).** $\\alpha > 0$ with HAC *t* $\\geq 2$ — the AI basket adds "
            "return the market exposure does not explain.\n"
            "- **H₂ (risk-adjusted win).** BUZZ's excess-of-cash Sharpe exceeds SPY's.\n"
            "- **H₃ (active edge).** The information ratio of the active return $r^{BUZZ}-r^{SPY}$ "
            "is positive and significant.\n\n"
            "We find **H₁ rejected** (α < 0, |t| < 2 — no positive skill), **H₂ rejected** (lower "
            "Sharpe, far deeper drawdown), **H₃ rejected** (IR ≈ 0). The verdict is `NONE` on the "
            "Signal axis because the inference bar requires a *t* ≥ 2 **for** an effect, in the "
            "claimed (positive) direction — which the tape does not deliver."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "Sentiment moves prices *transiently* (Tetlock 2007; Da-Engelberg-Gao 2011) — that is "
            "well established and not in dispute. The open question is whether a **monthly-"
            "rebalanced, long-only 75-name basket** can convert that transient pressure into "
            "durable, fee-surviving alpha. The raw-signal studies on this desk "
            "([254](../../254-wsb-mentions/), [256](../../256-twitter-mood/)) said no; the product "
            "test asks whether the wrapper does any better. Naming the mechanism — *high beta "
            "masquerading as skill* — is worth more than the binary, because it is the exact trap "
            "every 'AI/theme ETF' sets."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Unit of analysis.** The two NAV tapes (auto-adjusted, total-return-style), inner-"
            "joined and rebased to 100. A *holding* race — both legs fully invested — so the core "
            "needs **no execution lag**; the optional timing overlay carries one lag, documented.\n"
            "- **Inference.** CAPM regression with **Newey-West HAC** covariance on the intercept; "
            "the alpha *t* is the inference-bar statistic. A **circular block bootstrap** "
            "(block = 20d) CI on the annualised alpha and P(α > 0).\n"
            "- **Risk-adjusted race.** Excess-of-cash Sharpe for each leg (excess-vs-excess, never "
            "raw-vs-excess) and the information ratio of the active return.\n"
            "- **Positive control.** A deterministic synthetic pair with a tunable planted alpha — "
            "proving the engine recovers skill when it is present and reads 'none' on a dressed-up-"
            "beta null.\n\n"
            "Tape: BUZZ + SPY daily, BUZZ's full live history."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The CAPM regression — beta is the whole story\n\n"
            "Scatter BUZZ's daily excess return against SPY's; the slope is beta, the intercept "
            "(×252) is the annual alpha."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rets = st.daily_returns(REAL); a = st.capm_alpha(rets)\n"
            "    rb, rs = rets['buzz'].to_numpy()*100, rets['spy'].to_numpy()*100\n"
            "    tag = 'REAL TAPE'\n"
            "else:\n"
            "    syn, _ = data.synthetic_pair(alpha_bps=0.0)\n"
            "    rets = st.daily_returns(syn); a = st.capm_alpha(rets)\n"
            "    rb, rs = rets['buzz'].to_numpy()*100, rets['spy'].to_numpy()*100\n"
            "    tag = 'SYNTHETIC NULL (alpha=0) — illustration, NOT the real tape'\n"
            "fig, ax = plt.subplots(figsize=(8.8, 5.0))\n"
            "ax.scatter(rs, rb, s=8, alpha=.35, color=GREY)\n"
            "xs = np.linspace(rs.min(), rs.max(), 50)\n"
            "ax.plot(xs, a['beta']*xs + a['alpha_bps_day']/100, color=RED, lw=2,\n"
            "        label=f\"beta={a['beta']:.2f}, alpha={a['alpha_ann_pct']:+.1f}%/yr\")\n"
            "ax.axhline(0, c='k', lw=.6); ax.axvline(0, c='k', lw=.6)\n"
            "ax.set_xlabel('SPY daily excess return (%)'); ax.set_ylabel('BUZZ daily excess return (%)')\n"
            "ax.legend(loc='upper left'); ax.set_title(f'CAPM fit — {tag}')\n"
            "plt.tight_layout(); plt.show()\n"
            "print({k: round(v,3) for k,v in a.items()})"
        ),
        md(
            f"> 💡 In plain words: the cloud is steep — **β = {R['beta']:.2f}** — so BUZZ amplifies "
            f"the market. The line's intercept (the alpha) sits **below zero** "
            f"({R['alpha_ann']:.1f}%/yr) at HAC *t* = {R['alpha_t']:.2f}. The R² of {R['r2']:.2f} "
            "says ~two-thirds of BUZZ's daily moves are *just SPY, scaled up*. **H₁ rejected.**"
        ),
        md(
            "### 4b · The block-bootstrap CI on alpha — is it even reliably negative?\n\n"
            "i.i.d. resampling would destroy the contemporaneous beta relationship; we resample "
            "contiguous 20-day blocks so beta and autocorrelation survive."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ci = st.block_bootstrap_alpha_ci(st.daily_returns(REAL), n_boot=2000, seed=335)\n"
            "else:\n"
            f"    ci = dict(lo={R['ci_lo']}, med={R['alpha_ann']}, hi={R['ci_hi']}, p_positive={R['p_pos']/100})\n"
            "fig, ax = plt.subplots(figsize=(9.0, 3.4))\n"
            "ax.hlines(0, ci['lo'], ci['hi'], color=GREY, lw=8, alpha=.5)\n"
            "ax.plot([ci['med']], [0], 'o', color=RED, ms=12, label=f\"median {ci['med']:+.1f}%/yr\")\n"
            "ax.axvline(0, ls='--', c='k', lw=1)\n"
            "ax.set_yticks([]); ax.set_xlabel('annualised alpha (%/yr)')\n"
            "ax.set_title(f\"95% bootstrap CI on alpha: [{ci['lo']:+.1f}, {ci['hi']:+.1f}]  ·  P(alpha>0)={ci['p_positive']*100:.0f}%\")\n"
            "ax.legend(loc='upper right')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"CI [{ci['lo']:+.1f}, {ci['hi']:+.1f}] %/yr   P(alpha>0)={ci['p_positive']*100:.1f}%\")"
        ),
        md(
            f"> 💡 In plain words: the CI is **[{R['ci_lo']:.1f}, {R['ci_hi']:.1f}]%/yr** — its mass "
            f"sits at and below zero, with only a **{R['p_pos']:.0f}%** bootstrap chance the alpha is "
            "positive. We can't certify it's *significantly negative* (the upper edge nudges past "
            "zero), but we can rule out the thing being claimed: **there is no positive alpha.** "
            "That is exactly what `NONE` means — not 'it loses for sure', but 'the edge it promised "
            "is indistinguishable from absent'."
        ),
        md(
            "### 4c · The risk-adjusted race and the information ratio\n\n"
            "Return alone flatters a high-beta fund in a bull tape. Put both legs on the same clock "
            "(excess of cash) and read the active edge directly."
        ),
        code(
            "if HAVE_REAL:\n"
            "    es = st.excess_sharpe(st.daily_returns(REAL))\n"
            "    summ = st.net_of_fee_summary(REAL)\n"
            "else:\n"
            "    syn, _ = data.synthetic_pair(alpha_bps=0.0)\n"
            "    es = st.excess_sharpe(st.daily_returns(syn)); summ = st.net_of_fee_summary(syn)\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "a1.bar(['BUZZ','SPY'], [es['sharpe_buzz'], es['sharpe_spy']], color=[RED, GREEN], width=.5)\n"
            "a1.set_ylabel('Sharpe (ex-cash)'); a1.set_title('SPY ~2x the Sharpe of BUZZ')\n"
            "a2.bar(['BUZZ','SPY'], [summ.loc['buzz','max_dd_pct'], summ.loc['spy','max_dd_pct']],\n"
            "       color=[RED, GREEN], width=.5)\n"
            "a2.set_ylabel('max drawdown (%)'); a2.set_title('...and a far deeper hole')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"Sharpe BUZZ {es['sharpe_buzz']:.2f}  SPY {es['sharpe_spy']:.2f}  \"\n"
            "      f\"active IR {es['info_ratio']:+.3f} (t={es['active_tstat']:+.2f})\")"
        ),
        md(
            f"> 💡 In plain words: BUZZ's Sharpe (**{R['buzz_sharpe']:.2f}**) is about **half** SPY's "
            f"(**{R['spy_sharpe']:.2f}**), with a drawdown nearly **2.3×** as deep. The active "
            f"(BUZZ−SPY) return has an information ratio of **{R['info_ratio']:.2f}** at *t* = "
            f"{R['active_t']:.2f} — zero, statistically. **H₂ and H₃ rejected.**"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — alpha {R['alpha_ann']:.1f}%/yr at HAC *t* = {R['alpha_t']:.2f}, "
            f"bootstrap P(α>0) = {R['p_pos']:.0f}%, IR ≈ {R['info_ratio']:.2f}. No positive,"
            " statistically real skill — the bar (|t| ≥ 2 *for* the effect) is not cleared.\n"
            f"- **Tradability `MIRAGE`** — β = {R['beta']:.2f}; Sharpe {R['buzz_sharpe']:.2f} vs "
            f"{R['spy_sharpe']:.2f}; vol {R['buzz_vol']:.0f}% vs {R['spy_vol']:.0f}%; DD "
            f"{R['buzz_dd']:.0f}% vs {R['spy_dd']:.0f}%. Dressed-up beta plus a fee.\n"
            f"- **AI sentiment beats the market? `BUSTED`** — CAGR {R['buzz_cagr']:.1f}%/yr vs "
            f"{R['spy_cagr']:.1f}%/yr; worse return and worse risk on the real product's real life."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — and the overlay that can't save it\n\n"
            "It's a one-click ETF, so 'tradable' isn't the issue — *worth holding* is. A momentum "
            "switch (own whichever leg has the stronger trailing 20-day return; signal at *t*'s "
            "close earns *t+1*; 5 bps one-way switching cost) can't beat just holding SPY."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ov = st.sentiment_overlay(REAL, cost_bps=5.0)\n"
            "    es = st.excess_sharpe(st.daily_returns(REAL))\n"
            "    sp, bz, ovn = es['sharpe_spy'], es['sharpe_buzz'], ov['sharpe_net']\n"
            "    nsw = ov['n_switches']\n"
            "else:\n"
            f"    sp, bz, ovn, nsw = {R['spy_sharpe']}, {R['buzz_sharpe']}, {R['overlay_sharpe']}, {R['overlay_switches']}\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.bar(['SPY (hold)','20d switch (net)','BUZZ (hold)'], [sp, ovn, bz],\n"
            "       color=[GREEN, AMBER, RED], width=.55)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('Sharpe (ex-cash)')\n"
            "ax.set_title(f'The overlay ({nsw} switches) lands between the legs — below SPY')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'SPY {sp:.2f}   overlay {ovn:.2f} ({nsw} switches)   BUZZ {bz:.2f}')"
        ),
        md(
            f"> 💡 In plain words: the overlay's ~{R['overlay_switches']} switches just burn costs to "
            f"land between SPY and BUZZ (~{R['overlay_sharpe']:.2f} Sharpe), below plain SPY "
            f"({R['spy_sharpe']:.2f}). There is no configuration of this product, active or passive, "
            "that beats the index it races. The cost axis is moot — the gross was already losing."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further — the synthetic positive control\n\n"
            "Is the *engine* a faithful alpha detector, or would it print 'none' on anything? Plant "
            "a known alpha and confirm it is recovered (and that a pure dressed-up-beta null is not "
            "mistaken for skill)."
        ),
        code(
            "rows = []\n"
            "for ab, lab in [(0.0,'Null (alpha=0)'), (4.0,'Planted +10%/yr')]:\n"
            "    syn, tr = data.synthetic_pair(n_days=3000, alpha_bps=ab, seed=335)\n"
            "    a = st.capm_alpha(st.daily_returns(syn))\n"
            "    rows.append((lab, tr['alpha_bps_ann'], a['alpha_ann_pct'], a['tstat']))\n"
            "dfm = pd.DataFrame(rows, columns=['tape','planted %/yr','recovered %/yr','t'])\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.2))\n"
            "col = [GREEN if abs(t) >= 2 else GREY for t in dfm['t']]\n"
            "ax.bar(dfm['tape'], dfm['t'], color=col, width=.5)\n"
            "ax.axhline(2, ls='--', c=GREY); ax.axhline(-2, ls='--', c=GREY); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('recovered alpha t-stat'); ax.set_title('Faithful detector: null below |2|, planted alpha above')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(dfm.round(2).to_string(index=False))"
        ),
        md(
            f"The engine reads **no alpha** on the dressed-up-beta null (*t* = {R['null_t']:.2f}) and "
            f"**recovers a planted alpha** at *t* = {R['pos_t']:.2f} — so it can bank skill when "
            "skill is there. (A synthetic control is a *machinery* proof, never market evidence; the "
            "`NONE` stamp is earned on the real tape alone.) The real-tape negative result is "
            "therefore a statement about **BUZZ**: an AI read of the crowd, shrink-wrapped into a "
            "75-name basket, did not beat the market it listens to.\n\n"
            "For the raw signals this fund packages, see "
            "[Study 254 — WSB-Mentions](../../254-wsb-mentions/) and "
            "[Study 256 — Twitter-Mood](../../256-twitter-mood/) — both also dead on arrival."
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
