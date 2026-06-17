"""Generate the two narrative notebooks for Study 251 (Crypto-Reversal).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
controls run anywhere, offline and deterministic; the real-tape cells use the cached crypto
panel under ../_cache/ if present and otherwise quote the frozen headline numbers in ``R``
(mirroring docs/results.md), so the notebook re-runs for any reader.
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
    n_tokens=20, n_days=2722, ls_days=2700,
    start="2019-01-01", end="2026-06-14", fp="6ff538381541", turnover=37.8,
    # gross 21-day arms
    rev_ann=-61.6, rev_sharpe=-0.826, rev_vol=75, rev_dd=-99.9, rev_t=-2.10,
    mom_ann=+61.6, mom_sharpe=+0.826, mom_vol=75, mom_dd=-95.7, mom_t=+2.10, mom_cagr=20.6,
    shuf_ann=+1.6, shuf_sharpe=+0.035, shuf_dd=-74.8, shuf_t=+0.10,
    # Fama-MacBeth (the paper's apparatus)
    fm_slope_bps=+7.15, fm_t=+2.44, rev_prem_ann=-26, mom_prem_ann=+26,
    t_mom_vs_shuffle=1.8,
    # momentum cost sweep (ann %, sharpe, t)
    mc=[(0, 61.6, 0.826, 2.10), (10, 47.8, 0.641, 1.62), (30, 20.2, 0.271, 0.68),
        (50, -7.4, -0.100, -0.25), (100, -76.5, -1.023, -2.55)],
    # reversal lookback sweep (days, ann %, sharpe, t)
    lb=[(7, -84.8, -1.293, -3.34), (14, -90.4, -1.366, -3.47), (21, -61.6, -0.826, -2.10),
        (42, -66.4, -0.905, -2.39), (63, -56.6, -0.791, -2.16)],
    # sub-era reversal (name, gross sharpe, gross t, net50 sharpe, net50 ann %)
    era=[("2019-2020", -1.221, -1.65, -2.177, -153.1),
         ("2021-2022", -0.313, -0.41, -1.045, -101.7),
         ("2023-2026", -1.193, -1.94, -2.391, -138.8)],
    # synthetic controls
    syn_rev_sh=0.956, syn_rev_t=2.39, syn_mom_sh=-0.956, syn_fm_t=-2.40,
    null_rev_sh=0.316, null_rev_t=0.78, null_fm_t=-0.81,
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))           # the study package
sys.path.insert(0, os.path.abspath("../../.."))     # repo root (quantlab/)
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False,
                     "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from crypto_reversal import data, strategy as st

LOOKBACK, SKIP, NG = 21, 1, 5

R = dict(
    n_tokens=20, n_days=2722, ls_days=2700,
    start="2019-01-01", end="2026-06-14", fp="6ff538381541", turnover=37.8,
    rev_ann=-61.6, rev_sharpe=-0.826, rev_vol=75, rev_dd=-99.9, rev_t=-2.10,
    mom_ann=+61.6, mom_sharpe=+0.826, mom_vol=75, mom_dd=-95.7, mom_t=+2.10, mom_cagr=20.6,
    shuf_ann=+1.6, shuf_sharpe=+0.035, shuf_dd=-74.8, shuf_t=+0.10,
    fm_slope_bps=+7.15, fm_t=+2.44, rev_prem_ann=-26, mom_prem_ann=+26, t_mom_vs_shuffle=1.8,
)

def _have_cache():
    return os.path.exists(data._cache_path())

HAVE_REAL = _have_cache()
print("real crypto panel cache present:", HAVE_REAL)
"""

LOAD_REAL = """\
if HAVE_REAL:
    panel = data.fetch_panel(fetch=False)
    sig   = st.trailing_signal(panel, lookback=LOOKBACK, skip=SKIP)
    rev   = st.long_short_returns(panel, sig, leg="reversal", cost_bps=0.0)
    mom   = st.long_short_returns(panel, sig, leg="momentum", cost_bps=0.0)
    shuf  = st.long_short_returns(panel, st.shuffle_signal(sig, seed=42), leg="reversal", cost_bps=0.0)
    s_rev, s_mom, s_shuf = st.summary(rev), st.summary(mom), st.summary(shuf)
    fm    = st.fama_macbeth(panel, sig)
    turn  = st.avg_turnover(sig, leg="reversal")
    print(f"panel: {panel.shape[1]} tokens x {panel.shape[0]} days, fp={data.fingerprint(panel)}")
else:
    panel = sig = rev = mom = shuf = None
    s_rev  = dict(ann_return=R["rev_ann"]/100, sharpe=R["rev_sharpe"], vol_ann=R["rev_vol"]/100, max_drawdown=R["rev_dd"]/100, tstat=R["rev_t"])
    s_mom  = dict(ann_return=R["mom_ann"]/100, sharpe=R["mom_sharpe"], vol_ann=R["mom_vol"]/100, max_drawdown=R["mom_dd"]/100, tstat=R["mom_t"])
    s_shuf = dict(ann_return=R["shuf_ann"]/100, sharpe=R["shuf_sharpe"], vol_ann=0.45, max_drawdown=R["shuf_dd"]/100, tstat=R["shuf_t"])
    fm     = dict(avg_slope_bps=R["fm_slope_bps"], tstat_nw=R["fm_t"], reversal_premium_ann=R["rev_prem_ann"]/100)
    turn   = R["turnover"]/100
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Crypto-Reversal -- does \"buy the losers\" really beat momentum in crypto?\n"
            "### A 2026 paper says short-term reversal prices the crypto cross-section. We checked, in plain English.\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Reversal_or_momentum%3F: Momentum](https://img.shields.io/badge/Reversal_or_momentum%3F-Momentum-8b949e?style=flat-square)\n\n"
            "A May-2026 paper -- the \"QuantNest-7\" model (Babayev & Aliyev, SSRN 6818558) -- makes a "
            "striking claim: in crypto, **past losers beat past winners**. Rank tokens by their last "
            "21 days of return, buy the worst, short the best, and you allegedly earn +151%/yr. This "
            "*reverses* the famous Liu-Tsyvinski-Wu (2022) result that crypto, like stocks, has "
            "**momentum** (winners keep winning). Two respected papers, opposite signs. Who's right? "
            "We can't rebuild the paper's on-chain factors (they need paid data and an undisclosed "
            "recipe) -- but the reversal-vs-momentum question is pure price data, and *that* we can "
            "settle on free Yahoo! Finance closes.\n\n"
            "> **This is the plain-language layer.** Want the Fama-MacBeth regressions, HAC t-stats "
            "and synthetic controls? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** -- same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart below is drawn "
            "by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT + "\n" + LOAD_REAL),

        # ---- BEAT 0 -- VERDICT ----------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Do past losers beat past winners (the paper's claim)? | **No -- the opposite.** On 20 "
            f"liquid tokens, buying losers / shorting winners *loses* {abs(R['rev_ann']):.0f}%/yr "
            f"(t = {R['rev_t']:+.2f}). It's **momentum** that works here. |\n"
            "| Does that match the paper or its rival? | **The rival.** The sign flips back to "
            f"Liu et al. (2022) momentum: a Fama-MacBeth slope of +{R['fm_slope_bps']:.1f} bps/day "
            f"(t = {R['fm_t']:+.2f}) says recent *winners* earn more. |\n"
            "| So is momentum a money machine? | **No.** The book rebuilds every single day "
            f"(~{R['turnover']:.0f}% turnover/day). Momentum's +{R['mom_ann']:.0f}%/yr gross is gone "
            f"by ~50 bps of trading cost and deeply negative at the >1% spreads the paper itself "
            "admits crypto has. |\n"
            "| Could you trade either side? | **Mirage.** Whichever way the sign points, daily "
            "rebalancing of a long-short crypto book at real spreads destroys it -- and you usually "
            "can't even short the small-cap leg. |\n\n"
            "> The paper's headline doesn't survive free data: crypto's liquid cross-section is "
            "priced by *momentum*, not reversal -- and that momentum is a paper profit, not a "
            "tradable one."
        ),

        # ---- BEAT 1 -- THE CLAIM --------------------------------------------
        md(
            "## 1 -- The claim\n\n"
            "> *'Cryptocurrencies have fundamentals. A seven-factor model with essentially zero "
            "overfitting prices their returns -- and, contrary to earlier work, the cross-section "
            "is driven by short-term **reversal**: past losers outperform past winners at the "
            "21-day horizon (t = 6.19, +151%/yr).'*\n\n"
            "The economic story is plausible: crypto is dominated by emotional retail traders who "
            "overreact, so prices overshoot and snap back. The rival story (Liu et al. 2022) is "
            "equally plausible: crypto trends because attention and momentum-chasing feed on "
            "themselves. Both can't be the dominant effect. We test the exact recipe: rank by "
            "trailing 21-day return (1-day skip), form quintiles, and compare a **reversal** "
            "long-short (long losers, short winners) against its mirror, **momentum**."
        ),

        # ---- BEAT 2 -- SO WHAT ----------------------------------------------
        md(
            "## 2 -- So what?\n\n"
            "This is not hair-splitting. A factor model is the lens institutions use to decompose "
            "performance, hedge, and tell skill from luck. If the sign of the dominant style factor "
            "is *wrong*, every downstream attribution is wrong. And the two factors prescribe "
            "**opposite trades**: the reversal model tells you to buy the month's biggest crypto "
            "loser; the momentum model tells you to short it. Getting the sign right is the whole "
            "game -- and a paper claiming \"essentially zero overfitting\" while reporting "
            f"+151%/yr quintile spreads on a *survivor-only* universe deserves a hard look."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW ----------------------------------------
        md(
            "## 3 -- How would we even know?\n\n"
            "Three checks keep us honest:\n\n"
            "1. **Reversal vs momentum, head to head.** They are mirror images -- if one earns +X, "
            "the other earns −X gross. We report both and let the sign decide.\n"
            "2. **The shuffle control.** We randomly scramble which token gets which trailing-return "
            "rank each day. If the spread survives the scramble, it was never about the ranking.\n"
            "3. **The synthetic control.** We build a fake market with a *planted* reversal to prove "
            "our engine can find reversal when it truly exists -- so a negative result on real data "
            "means real momentum, not a broken tool.\n\n"
            "And the killer for tradability: this signal rebalances **daily**, so we charge it "
            "realistic crypto trading costs and watch what's left."
        ),

        # ---- BEAT 4 -- THE TEARDOWN -----------------------------------------
        md(
            "## 4 -- The teardown -- let's actually look\n\n"
            "**The cumulative P&L of the three arms. If the paper were right, the green "
            "(reversal) line would climb.**"
        ),
        code(
            "fig, ax = plt.subplots(figsize=(9.5, 5.2))\n"
            "if HAVE_REAL:\n"
            "    for series, lbl, c in [(rev, 'Reversal (buy losers)', GREEN),\n"
            "                           (mom, 'Momentum (buy winners)', RED),\n"
            "                           (shuf, 'Shuffle (null)', GREY)]:\n"
            "        cum = (1 + series).cumprod()\n"
            "        ax.plot(cum.index, cum, lw=1.7, c=c, label=lbl,\n"
            "                ls='--' if c == GREY else '-')\n"
            "    ax.set_yscale('log')\n"
            "else:\n"
            "    ax.text(0.5, 0.5, 'Cache not available -- run examples/verify.py --fetch',\n"
            "            ha='center', va='center', transform=ax.transAxes)\n"
            "ax.set_ylabel('Cumulative long-short wealth (log scale)')\n"
            "ax.set_title('Crypto 21-day cross-section: reversal sinks, momentum climbs')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
        ),
        md(
            f"The reversal arm (green) -- the paper's recommended trade -- *loses* steadily: "
            f"{R['rev_ann']:.0f}%/yr, Sharpe {R['rev_sharpe']:.2f}. Its mirror, momentum (red), "
            f"climbs (+{R['mom_ann']:.0f}%/yr gross). The shuffle control (grey) goes nowhere, "
            "confirming the signal is in the *ranking*, not the machinery. On free, liquid crypto "
            "data the paper's headline is upside down."
        ),
        md("**The scorecard -- Sharpe and annual return by arm:**"),
        code(
            "labels  = ['Reversal\\n(paper says BUY)', 'Momentum\\n(mirror)', 'Shuffle\\n(null)']\n"
            "sharpes = [s_rev['sharpe'], s_mom['sharpe'], s_shuf['sharpe']]\n"
            "anns    = [s_rev['ann_return']*100, s_mom['ann_return']*100, s_shuf['ann_return']*100]\n"
            "colors  = [GREEN, RED, GREY]\n"
            "fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))\n"
            "for i, (vals, title, fmt) in enumerate([(sharpes, 'Sharpe ratio', '+.2f'),\n"
            "                                        (anns, 'Annual return (%)', '+.0f')]):\n"
            "    bars = ax[i].bar(labels, vals, color=colors, width=0.6)\n"
            "    ax[i].axhline(0, c='k', lw=1)\n"
            "    for b, v in zip(bars, vals):\n"
            "        ax[i].annotate(f'{v:{fmt}}', (b.get_x()+b.get_width()/2, v),\n"
            "                       ha='center', va='bottom' if v >= 0 else 'top', fontsize=9)\n"
            "    ax[i].set_title(title)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"Reversal: {s_rev['ann_return']:+.1%}/yr (t={s_rev['tstat']:+.2f})\")\n"
            "print(f\"Momentum: {s_mom['ann_return']:+.1%}/yr (t={s_mom['tstat']:+.2f})\")\n"
        ),
        md(
            "Reversal and momentum are exact mirrors gross, so the bars are symmetric. The point is "
            "the **sign**: the profitable side is momentum, the paper bet on reversal. But before "
            "anyone celebrates momentum, beat 6 asks the only question that matters for a "
            "daily-rebalanced strategy: *what does it cost to run?*"
        ),

        # ---- BEAT 5 -- VERDICT ----------------------------------------------
        md(
            "## 5 -- The verdict\n\n"
            f"- **Signal -- NONE (for the reversal claim).** Buying losers loses "
            f"{abs(R['rev_ann']):.0f}%/yr (t = {R['rev_t']:+.2f}); the Fama-MacBeth slope is "
            f"*positive* (+{R['fm_slope_bps']:.1f} bps/day, t = {R['fm_t']:+.2f}) -- recent winners "
            "win. The paper's reversal does not replicate on free data.\n"
            "- **Tradability -- MIRAGE.** Even the winning momentum side rebalances daily "
            f"(~{R['turnover']:.0f}% turnover/day) and dies after realistic crypto costs.\n"
            "- **Reversal or momentum? -- MOMENTUM.** The Liu et al. (2022) sign holds on liquid "
            "survivor tokens -- the opposite of the paper's claim."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT -----------------------------------
        md(
            "## 6 -- Could you actually trade it?\n\n"
            "Take the *winning* side -- momentum -- and charge it realistic one-way crypto trading "
            "costs. With the book turning over almost completely every day, costs are not a detail; "
            "they are the whole story."
        ),
        code(
            "MC = [(0, 61.6, 2.10), (10, 47.8, 1.62), (30, 20.2, 0.68), (50, -7.4, -0.25), (100, -76.5, -2.55)]\n"
            "if HAVE_REAL:\n"
            "    MC = []\n"
            "    for c in [0, 10, 30, 50, 100]:\n"
            "        s = st.summary(st.long_short_returns(panel, sig, leg='momentum', cost_bps=c))\n"
            "        MC.append((c, s['ann_return']*100, s['tstat']))\n"
            "costs = [m[0] for m in MC]; anns = [m[1] for m in MC]\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.6))\n"
            "ax.plot(costs, anns, 'o-', c=RED, lw=2)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.fill_between(costs, anns, 0, where=[a < 0 for a in anns], color=RED, alpha=0.15)\n"
            "ax.set_xlabel('one-way trading cost (bps)'); ax.set_ylabel('Momentum net return (%/yr)')\n"
            "ax.set_title('The winning side is a mirage: momentum dies by ~50 bps')\n"
            "for c, a, *_ in MC:\n"
            "    ax.annotate(f'{a:+.0f}%', (c, a), textcoords='offset points', xytext=(0, 8), ha='center', fontsize=8)\n"
            "plt.tight_layout(); plt.show()\n"
            "print('The paper itself notes crypto spreads can exceed 1% (100 bps) per trade.')\n"
        ),
        md(
            f"At 10 bps momentum still earns +{R['mc'][1][1]:.0f}%/yr; by ~50 bps it's negative; at "
            f"the paper's own >100 bps crypto spread it's {R['mc'][4][1]:.0f}%/yr. And this is the "
            "*easy* version -- it ignores that shorting the small-cap winner leg is often impossible "
            "(no borrow) and that market impact in thin alts is brutal. The gross spread is a "
            "backtest artefact; the net spread is a loss."
        ),

        # ---- BEAT 7 -- GOING FURTHER ----------------------------------------
        md(
            "## 7 -- Going further\n\n"
            "- **Where reversal might really live.** Classical short-term reversal is a small-cap, "
            "illiquid phenomenon. Our 20 tokens are the most *liquid survivors* -- so the paper's "
            "reversal could exist in the dead/tiny tail we can't see. But that tail is exactly where "
            "spreads exceed 1% and there's no borrow: untradable either way.\n"
            "- **Survivorship.** Our universe is today's survivors projected backward (no LUNA, no "
            "FTT) -- which *flatters* the momentum (winners-survive) side. The paper concedes the "
            "same bias inflates *its* numbers. Honest reversal/momentum research needs a "
            "point-in-time universe with delistings.\n"
            "- **Slower signals, fewer trades.** A monthly-rebalanced version would cut turnover "
            "~20x. Whether any net edge survives at lower frequency is the natural follow-up.\n\n"
            "*Think the paper's reversal is real somewhere? Fork this, add a point-in-time universe "
            "with dead tokens, or test a monthly rebalance. The engine and controls are here.*"
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
            "# Crypto-Reversal -- a quantitative teardown\n"
            "### Public crypto panel - 21-day cross-section - Fama-MacBeth - shuffle & synthetic controls - turnover cost\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Reversal_or_momentum%3F: Momentum](https://img.shields.io/badge/Reversal_or_momentum%3F-Momentum-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) -- "
            "*same seven beats, every claim now carrying its standard error.* We rebuild the "
            "QuantNest-7 paper's price-only REV factor on a free 20-token yfinance panel "
            "(2019-2026), run the paper's own Fama-MacBeth apparatus, and stress the sign and the "
            "tradability against shuffle/synthetic controls and crypto turnover costs.\n\n"
            "> **Not investment advice.** Real data: 20 liquid non-stablecoin tokens, daily closes "
            f"2019-01-01 to 2026-06-14 (panel n=2,722, long-short n=2,700). "
            "Methods in [`docs/references.md`](../docs/references.md), reproducible numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **The `> in plain words` notes** translate each result back to intuition."
        ),
        code(BOOT + "\n" + LOAD_REAL),

        # ---- BEAT 0 ---------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Reversal LS {R['rev_ann']:.0f}%/yr (t = {R['rev_t']:+.2f}); "
            f"Fama-MacBeth slope **+{R['fm_slope_bps']:.2f} bps/day**, NW *t* = **{R['fm_t']:+.2f}** "
            "(winners win). Paper's REV *t* = +6.19 does not replicate -- sign flips to momentum. |\n"
            f"| **Tradability** | `MIRAGE` | ~{R['turnover']:.0f}% one-way turnover/day; momentum "
            f"(the profitable side) {R['mc'][3][1]:+.0f}%/yr at 50 bps, {R['mc'][4][1]:+.0f}%/yr at "
            "100 bps; short leg often unborrowable. |\n"
            "| **Reversal or momentum?** | `MOMENTUM` | Momentum sign at every horizon 7→63d; "
            "never clears \\|*t*\\| ≥ 2 era-by-era; survivorship-flattered. |\n\n"
            "> in plain words: the paper says crypto reverses; the public cross-section says it "
            "trends. The trend (momentum) is statistically marginal gross and a loss net of the "
            "turnover this daily signal demands."
        ),

        # ---- BEAT 1 ---------------------------------------------------------
        md(
            "## 1 -- The claim, steelmanned\n\n"
            "Let $r_{i,t}$ be token $i$'s return and $c_{i,t} = \\prod_{k=2}^{22}(1+r_{i,t-k})-1$ its "
            "trailing 21-day return (1-day skip). The paper's REV exposure is $-c_{i,t}$ "
            "(negated, so losers score high). The long-short is the top-minus-bottom quintile on "
            "REV, i.e. **long losers, short winners**. Hypotheses:\n\n"
            "- **H1 (sign).** The REV premium is *positive*: $\\mathbb{E}[r^{\\text{rev}}] > 0$ "
            "with HAC $t \\geq 2$ -- losers beat winners.\n"
            "- **H2 (Fama-MacBeth).** A daily cross-sectional regression "
            "$r_{i,t} = a_t + \\lambda_t z_{i,t} + \\varepsilon_{i,t}$ on the standardized trailing "
            "return $z$ has a *negative* average slope $\\bar\\lambda$ (high past return → low future "
            "return). The paper reports the negated form with $t = 6.19$.\n"
            "- **H3 (tradable).** The spread survives realistic transaction costs.\n\n"
            "We find **H1 and H2 reject with the opposite sign** ($\\bar\\lambda > 0$, momentum), and "
            "**H3 fails for the momentum side too** once daily turnover is costed."
        ),

        # ---- BEAT 2 ---------------------------------------------------------
        md(
            "## 2 -- So what? -- what rides on each answer\n\n"
            "The QuantNest-7 paper positions itself against Liu, Tsyvinski & Wu (2022), whose "
            "three-factor model uses **momentum**. The sign of the short-horizon factor is therefore "
            "the load-bearing empirical claim of the paper. If it flips on a public panel, the "
            "paper's central novelty -- \"reversal, not momentum\" -- is unsupported out of its "
            "(survivor, paid-data) sample, and its \"zero overfitting\" framing reads as in-sample "
            "fit to a short, survivor-biased history rather than a robust law of crypto pricing."
        ),

        # ---- BEAT 3 ---------------------------------------------------------
        md(
            "## 3 -- How we'd know -- the protocol\n\n"
            "- **Universe.** 20 liquid non-stablecoin tokens with deep yfinance history "
            "(survivors -- caveat in beat 7). Quintiles of ~4 names.\n"
            "- **Signal.** Trailing 21-day return, 1-day skip; ranked cross-sectionally each day; "
            "one execution lag (signal at $t-1$ earns the $t-1\\to t$ return).\n"
            "- **Arms.** Reversal (long losers/short winners), momentum (mirror), shuffle (per-day "
            "cross-sectional permutation of the signal).\n"
            "- **Inference.** HAC (Newey-West) *t* on the long-short mean; Fama-MacBeth daily "
            "cross-sectional slope with NW *t* -- the paper's Table-8 apparatus.\n"
            "- **Cost.** One-way bps on daily turnover; swept 0→100 bps.\n"
            "- **Controls.** Synthetic planted-reversal (positive) and null (negative) tapes to "
            "prove the engine recovers reversal when present."
        ),

        # ---- BEAT 4 ---------------------------------------------------------
        md("## 4 -- The teardown"),
        md(
            "### 4a -- Fama-MacBeth: the paper's own apparatus, the opposite sign\n\n"
            "Each day we z-score the cross-section of trailing returns and regress that day's "
            "returns on it. The average slope is the factor premium; its NW *t* is the headline."
        ),
        code(
            "print(f\"Fama-MacBeth avg slope: {fm['avg_slope_bps']:+.2f} bps/day\")\n"
            "print(f\"Newey-West t-stat:      {fm['tstat_nw']:+.2f}\")\n"
            "print(f\"Implied reversal premium: {fm['reversal_premium_ann']:+.1%}/yr \"\n"
            "      f\"({'reversal' if fm['reversal_premium_ann'] > 0 else 'MOMENTUM'} wins)\")\n"
            "print()\n"
            "print(f\"{'Arm':16s} | {'Ann':>8s} | {'Sharpe':>7s} | {'MaxDD':>8s} | {'HAC t':>6s}\")\n"
            "print('-'*56)\n"
            "for lbl, s in [('Reversal', s_rev), ('Momentum', s_mom), ('Shuffle (null)', s_shuf)]:\n"
            "    print(f\"{lbl:16s} | {s['ann_return']:+7.1%} | {s['sharpe']:+7.3f} | \"\n"
            "          f\"{s['max_drawdown']:+7.1%} | {s['tstat']:+6.2f}\")\n"
        ),
        md(
            f"> in plain words: a *positive* slope (+{R['fm_slope_bps']:.1f} bps/day, "
            f"t = {R['fm_t']:+.2f}) means tokens with higher past returns earn higher future returns "
            "-- **momentum**. The paper reports the negated REV with t = +6.19 (reversal); on free "
            "data the sign is reversed. The shuffle control sits at ~0, so this is the ranking "
            "talking, not the leg structure."
        ),
        md(
            "### 4b -- Lookback sweep: momentum at every horizon\n\n"
            "The paper's Table 11 reported reversal *strongest at short horizons*. We sweep the "
            "reversal long-short across 7→63-day lookbacks; a negative bar means momentum won."
        ),
        code(
            "LB = [(7, -84.8, -3.34), (14, -90.4, -3.47), (21, -61.6, -2.10), (42, -66.4, -2.39), (63, -56.6, -2.16)]\n"
            "if HAVE_REAL:\n"
            "    LB = []\n"
            "    for lb in [7, 14, 21, 42, 63]:\n"
            "        s = st.summary(st.long_short_returns(panel, st.trailing_signal(panel, lookback=lb), leg='reversal'))\n"
            "        LB.append((lb, s['ann_return']*100, s['tstat']))\n"
            "lbs = [x[0] for x in LB]; anns = [x[1] for x in LB]\n"
            "fig, ax = plt.subplots(figsize=(9, 4.3))\n"
            "bars = ax.bar([str(l) for l in lbs], anns, color=RED, width=0.6)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "for b, (l, a, t) in zip(bars, LB):\n"
            "    ax.annotate(f't={t:+.2f}', (b.get_x()+b.get_width()/2, a), ha='center', va='top', fontsize=8)\n"
            "ax.set_xlabel('lookback (days)'); ax.set_ylabel('Reversal LS return (%/yr)')\n"
            "ax.set_title('Reversal loses at every horizon -- i.e. momentum wins at every horizon')\n"
            "plt.tight_layout(); plt.show()\n"
        ),
        md(
            "> in plain words: reversal is negative for *all* lookbacks, strongest (most negative) at "
            "the short 7-14 day end -- the mirror image of the paper, which found reversal strongest "
            "there. Crypto's liquid cross-section trends at every horizon we test."
        ),
        md(
            "### 4c -- Synthetic controls: the engine is correct\n\n"
            "Negative reversal on real data is only meaningful if the engine *can* find reversal "
            "when it exists. We plant one."
        ),
        code(
            "rows = []\n"
            "for strength, lbl in [(1.0, 'Planted reversal'), (0.0, 'Null (no signal)')]:\n"
            "    sp, _ = data.synthetic_panel(signal_strength=strength, seed=213)\n"
            "    res = st.compare_arms(sp, cost_bps=0.0)\n"
            "    rows.append((lbl, res['reversal']['sharpe'], res['reversal']['tstat'],\n"
            "                 res['momentum']['sharpe'], res['fama_macbeth']['tstat_nw']))\n"
            "print(f\"{'Tape':18s} | {'Rev Sharpe':>10s} | {'Rev t':>6s} | {'Mom Sharpe':>10s} | {'FM t':>6s}\")\n"
            "print('-'*60)\n"
            "for r in rows:\n"
            "    print(f'{r[0]:18s} | {r[1]:+10.3f} | {r[2]:+6.2f} | {r[3]:+10.3f} | {r[4]:+6.2f}')\n"
        ),
        md(
            "On the planted tape the reversal arm is positive and significant and the FM slope is "
            "negative (correct reversal detection); on the null tape nothing significant appears. "
            "So the real-tape negative reversal is genuine **momentum**, not a broken engine."
        ),

        # ---- BEAT 5 ---------------------------------------------------------
        md(
            "## 5 -- The verdict\n\n"
            f"- **Signal `NONE`** (for the reversal claim) -- reversal LS {R['rev_ann']:.0f}%/yr "
            f"(t = {R['rev_t']:+.2f}); FM slope +{R['fm_slope_bps']:.2f} bps/day (t = {R['fm_t']:+.2f}). "
            "H1 and H2 reject with the *opposite* sign. The paper's reversal is not present on a "
            "public panel; momentum is (marginally).\n"
            f"- **Tradability `MIRAGE`** -- ~{R['turnover']:.0f}% one-way turnover/day. Momentum, the "
            f"only profitable side, is {R['mc'][3][1]:+.0f}%/yr at 50 bps and {R['mc'][4][1]:+.0f}%/yr "
            "at 100 bps (the paper's own crypto spread); short leg often unborrowable. No tradable "
            "spread.\n"
            "- **Reversal or momentum? `MOMENTUM`** -- the Liu et al. (2022) sign, at every horizon "
            "-- but survivorship-flattered and sub-bar era-by-era, so not a REAL momentum stamp "
            "either; just the resolution of the sign dispute."
        ),

        # ---- BEAT 6 ---------------------------------------------------------
        md(
            "## 6 -- Could you trade it? -- turnover and cost\n\n"
            "The signal reforms daily on a 21-day lookback, so turnover is near-total. Cost is the "
            "whole game."
        ),
        code(
            "MC = [(0, 61.6, 2.10), (10, 47.8, 1.62), (30, 20.2, 0.68), (50, -7.4, -0.25), (100, -76.5, -2.55)]\n"
            "if HAVE_REAL:\n"
            "    MC = []\n"
            "    for c in [0, 10, 30, 50, 100]:\n"
            "        s = st.summary(st.long_short_returns(panel, sig, leg='momentum', cost_bps=c))\n"
            "        MC.append((c, s['ann_return']*100, s['tstat']))\n"
            "costs = [m[0] for m in MC]; anns = [m[1] for m in MC]; ts = [m[2] for m in MC]\n"
            "fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "ax[0].plot(costs, anns, 'o-', c=RED, lw=2); ax[0].axhline(0, c='k', lw=1)\n"
            "ax[0].set_xlabel('one-way cost (bps)'); ax[0].set_ylabel('Momentum net return (%/yr)')\n"
            "ax[0].set_title(f'Turnover ~{turn:.0%}/day: momentum dies by ~50 bps')\n"
            "ax[1].plot(costs, ts, 's-', c=GREY, lw=2); ax[1].axhline(2, ls='--', c=GREEN)\n"
            "ax[1].axhline(-2, ls='--', c=RED); ax[1].axhline(0, c='k', lw=1)\n"
            "ax[1].set_xlabel('one-way cost (bps)'); ax[1].set_ylabel('HAC t-stat')\n"
            "ax[1].set_title('Significance evaporates with cost')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'one-way turnover: {turn:.1%}/day  (~{turn*365:.0f}%/yr)')\n"
        ),
        md(
            "> in plain words: gross momentum looks tradable, but it requires replacing almost the "
            "entire book every day. At crypto's real costs the edge is gone before you account for "
            "the impossibility of shorting thin alts. This is the textbook fate of a high-turnover, "
            "short-horizon anomaly (Novy-Marx & Velikov 2016)."
        ),

        # ---- BEAT 7 ---------------------------------------------------------
        md(
            "## 7 -- Going further & the honest caveats\n\n"
            "- **Survivorship.** The universe is today's liquid survivors projected backward (no "
            "LUNA/FTT). This *flatters momentum* (the winners are the survivors) and is the same "
            "bias the paper concedes inflates its own factor returns. A point-in-time universe with "
            "delistings is the proper fix -- and would most plausibly weaken the momentum side, not "
            "rescue reversal.\n"
            "- **Liquidity tail.** Classical short-term reversal is a small-cap, high-cost "
            "phenomenon. Our 20 liquid tokens may simply not contain it. But the small/illiquid "
            "tail where reversal could live is precisely where the >1% spreads make it untradable -- "
            "so the tradability verdict is robust to this caveat.\n"
            "- **The non-reproducible factors.** The paper's genuinely novel claims (on-chain "
            "*quality* t = 9.80, *value* t = 8.91, perp-*funding* t = 6.74) rest on paid Coin "
            "Metrics data and undisclosed composites. We tested only the price-only slice. A "
            "follow-up using free on-chain proxies (active addresses via public explorers, Binance "
            "funding API) could probe those -- a separate study.\n"
            "- **Frequency.** A weekly or monthly rebalance cuts turnover ~5-20x. Whether *any* net "
            "edge (either sign) survives at low frequency is the natural next test.\n\n"
            "*Fork this: add a point-in-time universe with dead tokens, drop to a monthly rebalance, "
            "or wire in free on-chain proxies for the paper's quality/value factors. "
            "The engine and controls are here.*"
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
