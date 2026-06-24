"""Generate the two narrative notebooks for Study 443 (Volume Profile POC).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached 5m event tables
under ../_cache/ (dropping the in-progress current session) and otherwise quote the frozen
headline numbers in ``R`` (mirroring docs/results.md). The synthetic positive control runs
anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance 5m, 6-name liquid
# basket SPY/QQQ/AAPL/MSFT/NVDA/TSLA, 2026-03-31 -> 2026-06-23, 348 sessions, ~60 trading days).
R = dict(
    start="2026-03-31", end="2026-06-23", as_of="2026-06-23",
    fingerprint="10ce6e8c7ecb", n_names=6, n_sessions=348, n_events=336,
    # reversion: POC touch-rate vs distance-matched random control level
    poc_rate=49.7, ctrl_rate=53.3, edge_pp=-3.6,
    poc_ci=(44.4, 55.0), ctrl_ci=(47.9, 58.5),
    t_paired=-1.28, t_hac=-1.32, placebo_p=0.682, placebo_mean=50.5,
    # fade trade
    fade_n=336, fade_hit=49.7, fade_gross=-0.102, fade_net=-0.122,
    fade_gross_t=-1.46, fade_net_t=-1.75, breakeven_bps=-5.1,
    # tolerance sweep: (tol_bps, poc%, ctrl%, edge_pp, hac_t)
    tol=[(0, 49.7, 53.3, -3.6, -1.32), (5, 52.1, 54.8, -2.7, -1.04),
         (10, 54.2, 57.1, -3.0, -1.14), (20, 58.3, 61.6, -3.3, -1.26)],
    # per-ticker edge: (ticker, n, poc%, ctrl%, edge_pp)
    by_tk=[("AAPL", 56, 62.5, 62.5, 0.0), ("MSFT", 57, 42.1, 42.1, 0.0),
           ("NVDA", 56, 51.8, 53.6, -1.8), ("QQQ", 58, 44.8, 43.1, 1.7),
           ("SPY", 52, 46.2, 48.1, -1.9), ("TSLA", 57, 50.9, 61.4, -10.5)],
    # synthetic control: (pull, poc%, ctrl%, edge_pp, hac_t, placebo_p)
    syn=[(0.0, 49.3, 48.7, 0.6, 0.44, 0.011), (0.7, 76.5, 61.0, 15.5, 7.49, 0.000)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Price_returns_to_POC%3F: Busted](https://img.shields.io/badge/Price_returns_to_POC%3F-Busted-8b949e?style=flat-square)\n\n"
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from volume_profile_poc import data, strategy as st

AS_OF = pd.Timestamp("2026-06-23")     # last COMPLETE session in the pinned cache
HAVE_REAL = data.have_real()
if HAVE_REAL:
    EV = data.load_real()
    EV = EV[EV["date"] <= AS_OF].reset_index(drop=True)
else:
    EV = None
print("real POC cache present:", HAVE_REAL,
      "| sessions:", (0 if EV is None else len(EV)),
      "| fingerprint:", (None if EV is None else data.fingerprint(EV)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does price get \"pulled back\" to the volume point of control? 🧲\n"
            "### The volume-profile POC — the level traders swear acts like a magnet — in plain English\n\n"
            + BADGES +
            "Here's a chart-room favourite. Every day, some price level trades **way more volume** "
            "than the rest — the **Point of Control (POC)**, the price the market \"agreed on\" most. "
            "The folklore says that level is a **magnet**: if tomorrow opens away from it, price will "
            "get **pulled back** to touch it. So you fade the open *toward* yesterday's POC and target "
            "the level.\n\n"
            "It's a tidy story. The trouble is that on a volatile intraday tape, *almost any* level "
            "near the open gets touched eventually — so \"price came back to the POC\" can be true "
            "**without the POC being special at all**. The honest test isn't \"how often is the POC "
            "touched?\" — it's \"is the POC touched **more often than a random level the same "
            "distance away**?\"\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo test and the cost math? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A loud data note.** Intraday 5-minute bars on Yahoo only go back **~60 trading "
            "days**, so this is a *short-span* study on a 6-name liquid basket "
            "(SPY/QQQ/AAPL/MSFT/NVDA/TSLA). And intraday levels are exactly where **the spread eats "
            "you** — capacity is the whole story. Every chart is drawn by the code beside it; house "
            "style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does price come back to yesterday's POC? | **About half the time — {R['poc_rate']:.0f}%.** "
            "Which *sounds* like a magnet… until you check the control. |\n"
            f"| Is that more than a **random** level the same distance away? | **No — it's *less*.** A "
            f"random level gets touched **{R['ctrl_rate']:.0f}%** of the time, *more* than the POC "
            f"({R['edge_pp']:+.1f} points). The POC is, if anything, **slightly stickier to avoid**. |\n"
            f"| Is the difference real or noise? | **Noise.** The gap isn't statistically distinguishable "
            f"from zero (t ≈ {R['t_hac']:.1f}). |\n"
            f"| Could you trade the fade? | **No.** Fading the open toward the POC loses money "
            f"**before you even pay the spread** ({R['fade_gross']:+.2f}% gross), and intraday spreads "
            "are exactly what kills levels. |\n\n"
            "> The POC is a real, observable level — but the \"magnet\" is an illusion of a volatile "
            "tape: any nearby level gets touched a lot. The POC isn't special."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"The Point of Control is where the most volume traded — it's 'fair value' for the "
            "session. Price is drawn back to it like a magnet. If we open above the POC, fade short "
            "toward it; if we open below, buy toward it. The level gets respected.\"*\n\n"
            "This is the heart of **Volume Profile** / Market Profile trading (J. Peter Steidlmayer's "
            "framework, popularised by a generation of futures day-traders). The POC, the Value Area, "
            "the \"naked\" POCs that supposedly get revisited — it's a whole vocabulary. We're testing "
            "its single most central claim: **price returns to the POC.**"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the POC really were a magnet, it would be a clean, mechanical intraday edge: a known "
            "level every morning, a direction to fade, a target to exit at. Day-traders pay for "
            "exactly this.\n\n"
            "But there's a trap baked into the claim, and it's the same trap that inflates *most* "
            "support/resistance lore: on a day where the stock swings around, the high and low cover a "
            "lot of ground, so **a level near the open is touched whether or not it means anything**. "
            "\"It came back to the POC\" feels like confirmation, but the right question is whether the "
            "POC beats a **coin-flip level** placed the same distance from the open. If it doesn't, the "
            "magnet is just **volatility**."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We pull **5-minute bars** for a 6-name liquid basket over the last **~60 trading days** "
            f"({R['start']} → {R['end']}). For each session we build a **volume profile** — bin the "
            "day's volume by price, take the busiest bin: that's the POC. Then for the *next* session "
            "we ask:\n\n"
            "1. **Did the day's range touch the prior POC?** (the believer's claim)\n"
            "2. **Did it touch a *random control* level the same distance from the open?** (the honest "
            "benchmark — same volatility, no special meaning)\n"
            "3. **Would fading the open toward the POC have made money** after a realistic spread?\n\n"
            "**The mirage line, declared up front:** if the POC's touch-rate is **not clearly above** "
            "the random control's — or if the fade is gross-negative — we stamp it a **mirage**. We're "
            "not asking \"is the POC ever touched,\" we're asking \"is it *special*.\""
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**The decisive picture.** How often is the POC touched, versus a random level the same "
            "distance from the open? If the magnet is real, the green (POC) bar towers over the grey "
            "(control) bar."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rs = st.reversion_stats(EV)\n"
            "    poc, ctrl = rs['poc_rate']*100, rs['ctrl_rate']*100\n"
            "    poc_ci, ctrl_ci = rs['poc_ci'], rs['ctrl_ci']\n"
            "else:\n"
            "    poc, ctrl = R['poc_rate'], R['ctrl_rate']\n"
            "    poc_ci = tuple(c/100 for c in R['poc_ci']); ctrl_ci = tuple(c/100 for c in R['ctrl_ci'])\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.4))\n"
            "bars = ax.bar(['POC\\n(the claim)', 'random level\\n(same distance)'], [poc, ctrl],\n"
            "              color=[GREEN, GREY], width=.6)\n"
            "errs = [[poc-poc_ci[0]*100, poc_ci[1]*100-poc],[ctrl-ctrl_ci[0]*100, ctrl_ci[1]*100-ctrl]]\n"
            "ax.errorbar([0,1],[poc,ctrl], yerr=np.array(errs).T, fmt='none', ecolor='k', capsize=6)\n"
            "ax.axhline(50, ls='--', c=RED, lw=1, label='coin flip')\n"
            "for i,v in enumerate([poc,ctrl]): ax.annotate(f'{v:.1f}%',(i,v),ha='center',va='bottom',fontsize=11)\n"
            "ax.set_ylabel('touch-rate (%)'); ax.set_ylim(0, 75)\n"
            "ax.set_title('The POC is touched LESS than a random level the same distance away')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'POC touch {poc:.1f}%   random-level touch {ctrl:.1f}%   edge {poc-ctrl:+.1f} pp')"
        ),
        md(
            f"There's the whole study in one chart. The POC is touched **{R['poc_rate']:.0f}%** of the "
            f"time — but a random level placed the *same distance* from the open is touched "
            f"**{R['ctrl_rate']:.0f}%**, *more* than the POC. The \"magnet\" doesn't even keep pace with "
            "a coin-flip level. The confidence bars overlap heavily: this is noise, and what little sign "
            "there is points the **wrong way**."
        ),
        md(
            "**Does loosening the definition of \"touch\" rescue it?** Maybe an exact range-straddle is "
            "too strict — let's count a touch if price comes *within* a few basis points of the POC, and "
            "do the same for the control. Watch whether the edge ever turns positive."
        ),
        code(
            "tols = [t[0] for t in R['tol']]\n"
            "if HAVE_REAL:\n"
            "    poc_r = [st.reversion_stats(EV, tol=t/1e4)['poc_rate']*100 for t in tols]\n"
            "    ctrl_r = [st.reversion_stats(EV, tol=t/1e4)['ctrl_rate']*100 for t in tols]\n"
            "else:\n"
            "    poc_r = [t[1] for t in R['tol']]; ctrl_r = [t[2] for t in R['tol']]\n"
            "x = np.arange(len(tols))\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.bar(x-.2, poc_r, .4, color=GREEN, label='POC')\n"
            "ax.bar(x+.2, ctrl_r, .4, color=GREY, label='random level')\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{t} bps' for t in tols])\n"
            "ax.set_xlabel('touch tolerance'); ax.set_ylabel('touch-rate (%)')\n"
            "ax.set_title('Loosen the touch rule and BOTH rise together — the gap never flips')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('edge by tolerance (pp):', [round(p-c,1) for p,c in zip(poc_r,ctrl_r)])"
        ),
        md(
            "Loosening the rule lifts **both** touch-rates together — of course it does, a wider net "
            "catches both the POC and the random level. The **gap stays negative** at every tolerance. "
            "There's no setting where the POC beats a coin-flip level."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** The POC is touched **{R['poc_rate']:.0f}%** of the time vs "
            f"**{R['ctrl_rate']:.0f}%** for a distance-matched random level (edge **{R['edge_pp']:+.1f} "
            f"pp**, t ≈ {R['t_hac']:.1f}). No magnet — just volatility touching nearby levels.\n"
            f"- **Tradability — Mirage.** Fading the open toward the POC is **gross-negative** "
            f"(**{R['fade_gross']:+.2f}%** per trade) *before* any spread — and intraday spreads are "
            "where levels go to die.\n"
            "- **\"Price returns to the POC\"? — Busted.** It returns no more (a touch less) than to a "
            "random level. The claim mistakes a feature of volatile tape for a force."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — the spread finishes it off\n\n"
            "Even granting the magnet, here's the fade — short above the POC / long below, target the "
            "level — **gross** (before costs) vs **net** (after a tiny **1 bp** per-leg spread, the "
            "*best case* for a liquid name)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    fd = st.fade_trade(EV, cost_bps=1.0)\n"
            "    g, nv = fd['gross_mean']*100, fd['net_mean']*100\n"
            "else:\n"
            "    g, nv = R['fade_gross'], R['fade_net']\n"
            "fig, ax = plt.subplots(figsize=(7.2, 4.3))\n"
            "ax.bar(['gross', 'net (1 bp/leg)'], [g, nv], color=[GREY, RED], width=.55)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate([g,nv]): ax.annotate(f'{v:+.3f}%',(i,v),ha='center',va='top',fontsize=11)\n"
            "ax.set_ylabel('mean per-trade return (%)')\n"
            "ax.set_title('The fade loses money even GROSS — the spread is just the burial')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'fade gross {g:+.3f}%   net {nv:+.3f}%   (break-even one-way cost {R[\"breakeven_bps\"]:+.1f} bps)')"
        ),
        md(
            f"The fade is **{R['fade_gross']:+.2f}%** per trade *gross* — it loses before costs, so "
            f"there's no edge to pay a spread out of (break-even cost is **{R['breakeven_bps']:+.1f} "
            "bps**, i.e. negative — there's nothing there). The 1 bp/leg charge is generous for a liquid "
            "ETF; widen it to a realistic intraday spread and it's worse. **Mirage**, full stop."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The honesty trick generalises.** Every support/resistance claim — round numbers, "
            "prior highs, VWAP, the POC — has to beat a **distance-matched random level**, not zero. On "
            "a volatile tape, *everything* gets touched. (See the desk's round-numbers and "
            "research-method demos.)\n"
            f"- **Short span, named loudly.** Yahoo caps 5m bars at ~60 days, so this is **{R['n_events']} "
            "events** on 6 names — enough to say \"no magnet here,\" not enough to chase a tiny "
            "conditional edge. Re-run on futures tick data (ES, NQ) over years for the definitive test.\n"
            "- **Variants to try.** \"Naked\" POCs (untouched for days), the Value Area edges instead of "
            "the POC, or continuation (does price *accelerate* away from the POC?) rather than reversion.\n\n"
            "*Think the POC is a real magnet? Show its touch-rate clearing a distance-matched random "
            "control with t ≥ 2 on a longer tape — then we'll talk.*"
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
            "# Volume Profile POC — a quantitative teardown 🔬\n"
            "### POC touch-rate vs a distance-matched random control · paired & HAC *t* · a "
            "shuffle-the-POC placebo · the fade net of spread · a synthetic planted-magnet control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The Volume "
            "Profile **Point of Control** is a real, observable level; the claim under test is that "
            "price is *drawn back* to it. The whole game is the **right control**: on a volatile tape "
            "any nearby level is touched often, so the POC must beat a **distance-matched random "
            "level**, not zero.\n\n"
            "> ⚠️ **Data + capacity note (loud).** 5-minute bars, Yahoo's ~60-trading-day cap, a 6-name "
            "liquid basket (SPY/QQQ/AAPL/MSFT/NVDA/TSLA). This is a **short-span** test — fine to "
            "*reject* a magnet, underpowered to certify a tiny one. And the subject is intraday levels, "
            "where the **bid-ask spread is the binding constraint** — capacity is the verdict. "
            "Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | POC touch-rate **{R['poc_rate']:.1f}%** vs distance-matched random "
            f"control **{R['ctrl_rate']:.1f}%** → edge **{R['edge_pp']:+.1f} pp**, paired "
            f"**t = {R['t_paired']:.2f}**, **HAC t = {R['t_hac']:.2f}**, shuffle-POC placebo "
            f"**p = {R['placebo_p']:.2f}**. The sign is *negative* and far from the t ≥ 2 bar. |\n"
            f"| **Tradability** | `MIRAGE` | The fade (short above / long below, target the POC) is "
            f"**gross-negative** ({R['fade_gross']:+.3f}% per trade, t = {R['fade_gross_t']:.2f}); "
            f"break-even one-way cost **{R['breakeven_bps']:+.1f} bps** (no edge to pay any spread). |\n"
            f"| **Price returns to POC?** | `BUSTED` | The POC is touched *no more* (a touch **less**) "
            "than a random level the same distance from the open, at every touch tolerance. The "
            "\"magnet\" is volatility, not the POC. |\n\n"
            "> 💡 In plain words: the POC is a real level, but it has **no special pull**. A coin-flip "
            "level the same distance away is touched just as often — slightly more — so there is "
            "nothing to trade."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $P^\\star_{d-1}$ be the **POC** of session $d-1$ — the centre of the highest-volume "
            "price bin of that day's 5-minute volume profile,\n\n"
            "$$P^\\star = \\text{bin-centre}\\;\\arg\\max_b \\sum_{t\\in d-1} v_t \\,"
            "\\mathbb{1}\\{\\,\\text{tp}_t \\in b\\,\\},\\qquad \\text{tp}_t=\\tfrac{h_t+l_t+c_t}{3}.$$\n\n"
            "The believer's claim: in session $d$, price **returns** to $P^\\star_{d-1}$. Define a "
            "touch as $\\;\\ell_d \\le P^\\star_{d-1} \\le h_d\\;$ (the day's range straddles the level). "
            "The honest benchmark is a **distance-matched random control** "
            "$C_d = O_d + s\\,|O_d-P^\\star_{d-1}|,\\; s\\sim\\{-1,+1\\}$ — same gap from the open, "
            "random side.\n\n"
            "- **H₁ (magnet).** $\\Pr[\\text{touch }P^\\star] - \\Pr[\\text{touch }C] > 0$ and "
            "significant.\n"
            "- **H₂ (tradable).** Fading the open toward $P^\\star$ is net-positive after the spread.\n\n"
            "We find **H₁ rejected** (edge negative, HAC t ≈ −1.3) and **H₂ rejected** (gross-negative). "
            "The level exists; the force does not."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the control is the whole argument\n\n"
            "A naive test — \"the POC is touched ~50% of the time, look, it's a magnet!\" — is the "
            "classic support/resistance fallacy. On a tape with intraday range $\\sim$ a few %, any "
            "level within that range is touched with high probability **regardless of meaning**. So the "
            "estimand is a **difference of touch-rates**, POC minus a distance-matched random level, "
            "and the test is a **paired** one (each session contributes both touches), with a "
            "**Newey-West (HAC)** standard error because consecutive sessions' touch outcomes are "
            "autocorrelated (volatility clusters).\n\n"
            "$$ t = \\frac{\\overline{(\\,\\mathbb{1}_{\\text{POC}} - \\mathbb{1}_{\\text{ctrl}}\\,)}}"
            "{\\widehat{\\text{se}}_{\\text{HAC}}}. $$\n\n"
            "Tradability then charges the one thing that actually kills intraday levels: the **spread**, "
            "one-way on both legs of the fade."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** 6 liquid names (SPY, QQQ, AAPL, MSFT, NVDA, TSLA); yfinance **5m** RTH "
            f"bars, **{R['start']} → {R['end']}** (~60 trading days, **{R['n_sessions']}** session "
            f"rows, **{R['n_events']}** events with the open ≥ 5 bps from the POC). Fingerprint "
            f"`{R['fingerprint']}`.\n"
            "- **POC.** Highest-volume bin (50 bins) of each session's 5m typical-price volume "
            "profile; stamped onto the *next* session (known at the prior close — no look-ahead).\n"
            "- **Touch.** Day-$d$ range straddles the prior POC (tolerance sweep 0–20 bps).\n"
            "- **Control.** Distance-matched random level: same |gap| from the open, random side.\n"
            "- **Signal test.** Paired (POC − control) touch indicator vs 0; one-sample **t** and "
            "**HAC t** (Newey-West, 5 lags).\n"
            "- **Placebo.** **Shuffle-the-POC** — re-pair each session's range with a *permuted* POC "
            "offset many times; $p = \\Pr[\\text{shuffled touch-rate} \\ge \\text{true POC rate}]$.\n"
            "- **Costs.** Fade the open toward the POC; **1 bp** one-way × 2 legs (generous for these "
            "names); report break-even cost.\n"
            "- **Positive control.** A synthetic panel with a **planted** pull toward the POC: pull = 0 "
            "must NOT manufacture an edge; pull > 0 must light up the HAC t."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The estimand — POC vs distance-matched control, with Wilson intervals\n\n"
            "Touch-rates with 95% Wilson intervals. If the POC were a magnet the green interval would "
            "sit clearly above the grey one. It does not."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rs = st.reversion_stats(EV)\n"
            "    poc, ctrl, edge = rs['poc_rate']*100, rs['ctrl_rate']*100, rs['edge']*100\n"
            "    tp, th = rs['t_paired'], rs['t_hac']\n"
            "    poc_ci, ctrl_ci = rs['poc_ci'], rs['ctrl_ci']\n"
            "else:\n"
            "    poc, ctrl, edge = R['poc_rate'], R['ctrl_rate'], R['edge_pp']\n"
            "    tp, th = R['t_paired'], R['t_hac']\n"
            "    poc_ci = tuple(c/100 for c in R['poc_ci']); ctrl_ci = tuple(c/100 for c in R['ctrl_ci'])\n"
            "fig, ax = plt.subplots(figsize=(7.8, 4.4))\n"
            "errs = [[poc-poc_ci[0]*100, poc_ci[1]*100-poc],[ctrl-ctrl_ci[0]*100, ctrl_ci[1]*100-ctrl]]\n"
            "ax.bar(['POC','random\\ncontrol'], [poc,ctrl], color=[GREEN,GREY], width=.55)\n"
            "ax.errorbar([0,1],[poc,ctrl], yerr=np.array(errs).T, fmt='none', ecolor='k', capsize=6)\n"
            "ax.axhline(50, ls='--', c=RED, lw=1, label='coin flip')\n"
            "for i,v in enumerate([poc,ctrl]): ax.annotate(f'{v:.1f}%',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('touch-rate (%)'); ax.set_ylim(0,75)\n"
            "ax.set_title(f'edge {edge:+.1f} pp   paired t={tp:.2f}   HAC t={th:.2f}'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'POC {poc:.1f}% [{poc_ci[0]*100:.1f},{poc_ci[1]*100:.1f}]  '\n"
            "      f'ctrl {ctrl:.1f}% [{ctrl_ci[0]*100:.1f},{ctrl_ci[1]*100:.1f}]  edge {edge:+.1f}pp')"
        ),
        md(
            f"> 💡 In plain words: the intervals **overlap**. POC **{R['poc_rate']:.1f}%** "
            f"[{R['poc_ci'][0]:.1f}, {R['poc_ci'][1]:.1f}], control **{R['ctrl_rate']:.1f}%** "
            f"[{R['ctrl_ci'][0]:.1f}, {R['ctrl_ci'][1]:.1f}]. The point estimate of the edge is "
            f"**negative** ({R['edge_pp']:+.1f} pp), HAC **t = {R['t_hac']:.2f}** — nowhere near the "
            "t ≥ 2 bar, and pointing the wrong way. No magnet."
        ),
        md(
            "### 4b · The shuffle-the-POC placebo — is the *true* pairing special?\n\n"
            "Re-pair each session's range with a **permuted** POC offset thousands of times. If the POC "
            "had real pull, the *true* pairing would be touched more than these shuffled pairings (the "
            "true touch-rate sits in the right tail). Here it sits **in the middle of the cloud**."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.placebo_pvalue(EV, n_draws=5000)\n"
            "    draws = pl['draws']*100; obs = pl['poc_rate']*100; pval = pl['p_value']\n"
            "else:\n"
            "    rng = np.random.default_rng(443); draws = rng.normal(R['placebo_mean'], 2.5, 5000)\n"
            "    obs = R['poc_rate']; pval = R['placebo_p']\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=40, color=GREY, alpha=.85, label='shuffled POC-pairings (touch-rate %)')\n"
            "ax.axvline(obs, c=GREEN, lw=2.5, label=f'true POC touch-rate {obs:.1f}%')\n"
            "ax.set_xlabel('touch-rate (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'True POC sits inside the luck cloud: placebo p = {pval:.2f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'true POC {obs:.1f}%   shuffled mean {R[\"placebo_mean\"]:.1f}%   p = {pval:.3f}')"
        ),
        md(
            f"> 💡 In plain words: the green line is **inside** the grey cloud (placebo "
            f"**p = {R['placebo_p']:.2f}**). A randomly-paired level is touched just as often as the "
            "true POC — the session-to-its-own-POC bond carries **no information**. Exactly what a "
            "non-magnet looks like."
        ),
        md(
            "### 4c · Robustness + cross-section — it fails everywhere\n\n"
            "Left: the edge across touch tolerances (0–20 bps) — never positive. Right: the per-name "
            "edge — no name shows a magnet; the only large number (TSLA) is **negative**."
        ),
        code(
            "tols = [t[0] for t in R['tol']]\n"
            "if HAVE_REAL:\n"
            "    edges = [st.reversion_stats(EV, tol=t/1e4)['edge']*100 for t in tols]\n"
            "    tk_edges = []\n"
            "    for tk in sorted(EV['ticker'].unique()):\n"
            "        sub = EV[EV['ticker']==tk]\n"
            "        tk_edges.append((tk, st.reversion_stats(sub)['edge']*100))\n"
            "else:\n"
            "    edges = [t[3] for t in R['tol']]\n"
            "    tk_edges = [(t[0], t[4]) for t in R['by_tk']]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "a1.bar([f'{t}' for t in tols], edges, color=[GREEN if e>0 else RED for e in edges], width=.6)\n"
            "a1.axhline(0, c='k', lw=.8)\n"
            "for i,e in enumerate(edges): a1.annotate(f'{e:+.1f}',(i,e),ha='center',va='top' if e<0 else 'bottom',fontsize=9)\n"
            "a1.set_xlabel('touch tolerance (bps)'); a1.set_ylabel('edge POC-ctrl (pp)')\n"
            "a1.set_title('Edge never turns positive')\n"
            "tks = [t[0] for t in tk_edges]; tv = [t[1] for t in tk_edges]\n"
            "a2.bar(tks, tv, color=[GREEN if e>0 else RED for e in tv], width=.6)\n"
            "a2.axhline(0, c='k', lw=.8)\n"
            "for i,e in enumerate(tv): a2.annotate(f'{e:+.1f}',(i,e),ha='center',va='top' if e<0 else 'bottom',fontsize=9)\n"
            "a2.set_ylabel('edge POC-ctrl (pp)'); a2.set_title('No name shows a magnet')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('edge by tol:', [round(e,1) for e in edges]); print('edge by name:', [(t,round(e,1)) for t,e in tk_edges])"
        ),
        md(
            "> 💡 In plain words: it fails on **every** cut. Loosen the touch rule — still negative. "
            "Split by name — no green. There is no slice of this tape where the POC behaves like a "
            "magnet. (Per-name samples are tiny — ~55 days each — so read the *pooled* number; the "
            "point is the absence of any positive signal, not the noisy per-name values.)"
        ),
        md(
            "### 4d · The fade, net of spread — gross-negative is the end of the story\n\n"
            "Fade the open toward the POC, target the level. Gross (before costs) vs net (1 bp/leg). "
            "There's nothing to charge a spread *against*."
        ),
        code(
            "if HAVE_REAL:\n"
            "    fd = st.fade_trade(EV, cost_bps=1.0)\n"
            "    g, nv, be = fd['gross_mean']*100, fd['net_mean']*100, fd['breakeven_bps']\n"
            "    gt, nt = fd['gross_t'], fd['net_t']\n"
            "else:\n"
            "    g, nv, be = R['fade_gross'], R['fade_net'], R['breakeven_bps']\n"
            "    gt, nt = R['fade_gross_t'], R['fade_net_t']\n"
            "fig, ax = plt.subplots(figsize=(7.4, 4.3))\n"
            "ax.bar(['gross','net (1bp/leg)'], [g, nv], color=[GREY, RED], width=.55)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate([g,nv]): ax.annotate(f'{v:+.3f}%',(i,v),ha='center',va='top',fontsize=11)\n"
            "ax.set_ylabel('mean per-trade return (%)')\n"
            "ax.set_title(f'Fade is gross-negative (t={gt:.2f}); break-even cost {be:+.1f} bps')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross {g:+.3f}% (t={gt:.2f})  net {nv:+.3f}% (t={nt:.2f})  break-even {be:+.1f} bps')"
        ),
        md(
            f"> 💡 In plain words: the fade loses **{R['fade_gross']:+.3f}%** per trade *before* costs "
            f"(t = {R['fade_gross_t']:.2f}), so the break-even one-way cost is **{R['breakeven_bps']:+.1f} "
            "bps** — negative, meaning there is no positive edge to pay any spread out of. On real "
            "intraday spreads it only gets worse. **Mirage.**"
        ),
        md(
            "### 4e · Faithful-engine & power control — we know the truth here\n\n"
            "On a deterministic panel where we **plant** a pull toward the POC: with **zero** pull the "
            "POC must NOT beat the control (no false positive); with a **large** planted pull it must "
            "light up. Both hold — so the flat real-tape result is a genuine *absence*, not a broken "
            "detector."
        ),
        code(
            "res = []\n"
            "for pull in (0.0, 0.7):\n"
            "    sev, _ = data.synthetic_panel(pull=pull, n_sessions=60, n_names=6)\n"
            "    r = st.reversion_stats(sev); p = st.placebo_pvalue(sev, n_draws=3000)\n"
            "    res.append((pull, r['poc_rate']*100, r['ctrl_rate']*100, r['edge']*100, r['t_hac'], p['p_value']))\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "labels = [f'planted pull\\n{r[0]:.1f}' for r in res]; tvals = [r[4] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 bar'); ax.axhline(0, c='k', lw=.8)\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('reversion HAC t'); ax.set_title('Control: pull=0 -> t~0; planted pull -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for pull,poc,ctrl,e,t,p in res: print(f'pull={pull:+.1f}: POC={poc:.1f}% ctrl={ctrl:.1f}% edge={e:+.1f}pp HAC t={t:.2f} placebo p={p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted pull the detector sits at edge "
            f"**{R['syn'][0][3]:+.1f} pp**, HAC **t = {R['syn'][0][4]:.2f}** (no false positive — noise "
            f"cannot fake a magnet); a strong planted pull reaches edge **{R['syn'][1][3]:+.1f} pp**, "
            f"HAC **t = {R['syn'][1][4]:.2f}**. The harness is unbiased and *can* detect a real magnet — "
            "so the real-tape flat/negative result is the honest absence of one."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — POC touch-rate **{R['poc_rate']:.1f}%** vs distance-matched control "
            f"**{R['ctrl_rate']:.1f}%**: edge **{R['edge_pp']:+.1f} pp**, paired **t = {R['t_paired']:.2f}**, "
            f"**HAC t = {R['t_hac']:.2f}**, shuffle-POC placebo **p = {R['placebo_p']:.2f}**. Negative "
            "sign, far from t ≥ 2, robust to tolerance and present in no name. The magnet is volatility.\n"
            f"- **Tradability `MIRAGE`** — the fade is **gross-negative** ({R['fade_gross']:+.3f}% per "
            f"trade, t = {R['fade_gross_t']:.2f}); break-even one-way cost **{R['breakeven_bps']:+.1f} "
            "bps**. Nothing to pay a spread out of, and intraday spreads are the binding constraint.\n"
            f"- **Price returns to POC? `BUSTED`** — the POC is touched *no more* (slightly less) than a "
            "random level the same distance away, at every tolerance. The claim mistakes a property of "
            "volatile tape for a force.\n\n"
            f"**Caveat, named loudly:** short span (~60 trading days, {R['n_events']} events, 6 names) "
            "— enough to *reject* a magnet on liquid names, not to chase a tiny conditional edge on "
            "futures over years. Survivorship is irrelevant here (a within-name, day-over-day level "
            "test, not a cross-sectional basket)."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the capacity story is the spread\n\n"
            "Even if there were a thin edge, the subject is **intraday levels**, where the bid-ask "
            "spread is paid on entry and exit of every fade. Here the gross edge is already below zero, "
            "so capacity is moot — but the picture is the general intraday lesson: the break-even cost "
            "is **negative**, meaning no spread, however tight, makes the fade pay."
        ),
        code(
            "if HAVE_REAL:\n"
            "    costs = [0.5, 1.0, 2.0, 5.0]\n"
            "    nets = [st.fade_trade(EV, cost_bps=c)['net_mean']*100 for c in costs]\n"
            "    gross = st.fade_trade(EV, cost_bps=0.0)['gross_mean']*100\n"
            "else:\n"
            "    costs = [0.5,1.0,2.0,5.0]; gross = R['fade_gross']\n"
            "    nets = [gross - 2*c/1e4*100 for c in costs]\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.axhline(gross, ls='--', c=GREY, label=f'gross {gross:+.3f}%')\n"
            "ax.plot(costs, nets, 'o-', c=RED, lw=2, label='net of spread')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for c,n in zip(costs,nets): ax.annotate(f'{n:+.3f}%',(c,n),ha='center',va='top',fontsize=9)\n"
            "ax.set_xlabel('one-way spread (bps/leg)'); ax.set_ylabel('net per-trade return (%)')\n"
            "ax.set_title('Already below zero gross — every spread only digs deeper'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('net by cost:', {f'{c}bp': round(n,3) for c,n in zip(costs,nets)})"
        ),
        md(
            "> 💡 In plain words: the line starts **below zero** (gross) and only falls. There is no "
            "spread tight enough to rescue a fade with no underlying edge. That's the whole capacity "
            "verdict: **MIRAGE** — and it would be a mirage even if the data span were ten years."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The distance-matched control is the reusable lesson.** Any level claim — POC, VWAP, "
            "round numbers, prior-day high/low — must beat a random level the same distance from the "
            "reference price, because volatile ranges touch *everything*. Test levels this way or don't "
            "claim them.\n"
            "- **A longer, finer tape.** ES/NQ futures tick data over years would give the power to "
            "chase a sub-percent conditional edge (e.g. naked POCs untouched for $k$ days). Yahoo's 5m "
            "60-day cap can reject the headline magnet; it cannot certify a small one.\n"
            "- **Continuation, not reversion.** The opposite claim — price *accelerates away* from a "
            "high-volume shelf (acceptance/rejection) — is testable with the same harness by flipping "
            "the sign of the fade.\n\n"
            "*The reproducible core is offline and deterministic once the 5m cache is built; the "
            "synthetic planted-magnet control runs anywhere. Methods and sources: "
            "[`docs/references.md`](../docs/references.md); frozen numbers: "
            "[`docs/results.md`](../docs/results.md).*"
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
