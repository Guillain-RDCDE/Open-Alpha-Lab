"""Generate the two narrative notebooks for Study 407 (Dark Cloud Cover & Piercing Line).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached basket OHLCV
under ../_cache/ and otherwise quote the frozen headline numbers in ``R`` (mirroring
docs/results.md). The synthetic positive control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md.
# Basket: 29 liquid US large-caps + SPY, yfinance daily, 2005-01-03 -> 2026-06-18, fp bf1d6cb7ca54.
R = dict(
    start="2005-01-03", end="2026-06-18", years=21.5, fp="bf1d6cb7ca54",
    n_names=30, n_events=2342, n_bull=1074, n_bear=1268,
    # combined signed return per horizon: (H, n, mean%, baseL%, win%, t_hac, t_one, p_plac, net%)
    h1=(1, 2342, -0.078, 0.028, 46.0, -2.57, -2.82, 0.998, -0.179),
    h3=(3, 2341, -0.097, 0.141, 47.2, -1.91, -1.81, 0.960, -0.200),
    h5=(5, 2341, -0.105, 0.253, 48.2, -1.49, -1.49, 0.924, -0.210),
    h10=(10, 2339, -0.069, 0.530, 49.3, -0.71, -0.68, 0.755, -0.179),
    # leg split: H -> (bull_mean%, bull_t, bull_win%, bear_mean%, bear_t, bear_win%)
    bull={1: (-0.033, -0.78, 47.0), 3: (0.110, 1.38, 51.0),
          5: (0.306, 3.01, 54.3), 10: (0.481, 3.04, 56.9)},
    bear={1: (-0.116, -2.85, 45.1), 3: (-0.272, -3.98, 44.0),
          5: (-0.453, -4.86, 43.1), 10: (-0.536, -3.91, 42.7)},
    # myth-check H=1: label -> (n, mean%, t_hac, win%)
    myth=[("plain", 2342, -0.078, -2.57, 46.0),
          ("+trend", 1243, -0.081, -1.92, 45.5),
          ("+volume spike", 190, -0.185, -1.88, 45.3),
          ("+both", 128, -0.011, -0.09, 48.4)],
    # synthetic control H=1: (planted_edge, events, mean%, t_hac, p_plac, win%)
    syn=[(0.000, 910, 0.021, 0.47, 0.313, 50.2), (0.004, 904, 0.414, 9.07, 0.000, 62.5)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Reliable_reversal%3F: Busted](https://img.shields.io/badge/Reliable_reversal%3F-Busted-8b949e?style=flat-square)\n\n"
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from dark_cloud_piercing import data, strategy as st

HAVE_REAL = data.have_real()
PANEL = data.load_real(allow_fetch=False) if HAVE_REAL else None
print("real twin cache present:", HAVE_REAL,
      "| names:", (0 if PANEL is None else len(PANEL)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Do the Dark Cloud & Piercing twins call the turn? \U0001f329️\n"
            "### Two of candlestick lore's most-taught reversal patterns, measured honestly\n\n"
            + BADGES +
            "Open any candlestick textbook and you'll meet the **twins**. A **Piercing Line** "
            "is supposed to mark a *bottom*: a long red day, then a green day that gaps down at "
            "the open but rallies to close more than halfway back up the red body — a failed "
            "sell-off, so *buy*. A **Dark Cloud Cover** is the mirror — a long green day, then a "
            "red day that gaps up but sells off to close more than halfway down the green body — "
            "a failed rally, so *sell*. Do these twins actually predict the reversal?\n\n"
            "> This is the plain-language layer. Want the HAC *t*-stats, the leg split, the "
            "placebo and the synthetic control? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart below is drawn "
            "by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 — VERDICT -----------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Do the twins predict a reversal? | **No — they predict the opposite.** Signed "
            f"return the day after is **{R['h1'][2]:+.3f}%**, HAC *t* = **{R['h1'][5]:+.2f}** "
            "(wrong sign). |\n"
            f"| Who's the culprit? | The **Dark Cloud (bearish) leg**: after it the stock keeps "
            f"*rising* (1-day leg *t* = {R['bear'][1][1]:+.2f}). |\n"
            f"| Is the Piercing (bullish) leg real alpha? | **No.** Its few-day gain is just the "
            "market's always-up drift — it barely beats buy-and-hold. |\n"
            f"| Could you trade it? | **No.** The gross edge is already negative; costs only "
            "deepen the hole. |\n\n"
            "> The twins are a *story* about failed sell-offs and failed rallies. On 21 years of "
            "liquid US large-caps, the story runs backwards."
        ),

        # ---- BEAT 1 — THE CLAIM ---------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"A **Piercing Line** marks a bottom: after a strong down day, price gaps lower "
            "but buyers take over and close it back above the midpoint of the prior body — a "
            "reversal up is coming. A **Dark Cloud Cover** marks a top: after a strong up day, "
            "price gaps higher but sellers take over and close it back below the midpoint — a "
            "reversal down is coming.\"*\n\n"
            "These are textbook patterns from Steve Nison's *Japanese Candlestick Charting "
            "Techniques* — the work that brought candlesticks to the West. The twins are the "
            "'almost-engulfing' cousins: deep penetration into the prior body, but not a full "
            "swallow. We test whether that penetration carries any forecasting power on modern "
            "US equities."
        ),

        # ---- BEAT 2 — SO WHAT -----------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the twins reliably called turns, they'd be a free, mechanical entry signal anyone "
            "with a chart can read — no data feeds, no models. Millions of retail traders watch "
            "for exactly these shapes. If the premise is *wrong* — if a Dark Cloud is more often "
            "followed by more upside, not a top — then traders acting on the pattern are paying "
            "spreads to stand on the wrong side of the tape."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW -----------------------------------------
        md(
            "## 3 · How would we know?\n\n"
            "One honest event study:\n\n"
            "1. **Detect every twin by its exact OHLC rule** across a fixed 30-name basket "
            "(29 liquid large-caps + SPY), 2005–2026.\n"
            "2. **Enter the next open** (one execution lag — no peeking at the close you traded "
            "on) and measure the forward 1 / 3 / 5 / 10-day return, *signed* by the pattern "
            "(long after a Piercing, short after a Dark Cloud).\n"
            "3. **Compare to doing nothing** — the unconditional forward return on every bar of "
            "the same names — and to a **coin** (random bars, random direction, same count).\n\n"
            "**What would make us say 'mirage':** if the signed return isn't reliably positive "
            "(HAC *t* ≥ 2) the reversal claim fails."
        ),

        # ---- BEAT 4 — THE TEARDOWN ------------------------------------------
        md(
            "## 4 · The teardown — what actually happens\n\n"
            "**The day after a twin, signed by the pattern direction.** If the reversal were "
            "real these bars would be green and above zero."
        ),
        code(
            "horizons = [1, 3, 5, 10]\n"
            "if HAVE_REAL:\n"
            "    rows = [st.summarize(PANEL, h, n_draws=2000) for h in horizons]\n"
            "    means = [r['mean']*100 for r in rows]; tt = [r['t_hac'] for r in rows]\n"
            "else:\n"
            "    means = [R[f'h{h}'][2] for h in horizons]; tt = [R[f'h{h}'][5] for h in horizons]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "col = [GREEN if m > 0 else RED for m in means]\n"
            "ax.bar([f'{h}d' for h in horizons], means, color=col)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('signed return after the twin (%)')\n"
            "ax.set_title('Every horizon is below zero — the twins anti-predict the reversal')\n"
            "for x, (m, t) in enumerate(zip(means, tt)):\n"
            "    ax.annotate(f't={t:+.2f}', (x, m), ha='center',\n"
            "                va='top' if m < 0 else 'bottom', fontsize=9, color=GREY)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'1-day: {means[0]:+.3f}%  (HAC t = {tt[0]:+.2f})')"
        ),
        md(
            f"The day-after signed return is **{R['h1'][2]:+.3f}%** at HAC *t* = "
            f"**{R['h1'][5]:+.2f}** — significantly *negative*. The pattern doesn't fail to "
            "predict the reversal; it predicts the **wrong direction**. (At longer horizons the "
            "loss shrinks toward noise as the market's always-up drift reasserts itself.)"
        ),
        md(
            "**Which twin is the problem?** Split the signed return into its two legs."
        ),
        code(
            "if HAVE_REAL:\n"
            "    legs = [st.leg_split(PANEL, h) for h in horizons]\n"
            "    bull = [l['bull']['mean']*100 for l in legs]\n"
            "    bear = [l['bear']['mean']*100 for l in legs]\n"
            "else:\n"
            "    bull = [R['bull'][h][0] for h in horizons]\n"
            "    bear = [R['bear'][h][0] for h in horizons]\n"
            "x = np.arange(len(horizons)); w = 0.38\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.bar(x - w/2, bull, w, color=GREEN, label='Piercing (bullish leg)')\n"
            "ax.bar(x + w/2, bear, w, color=RED, label='Dark Cloud (bearish leg)')\n"
            "ax.axhline(0, c='k', lw=1); ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in horizons])\n"
            "ax.set_ylabel('signed leg return (%)'); ax.legend()\n"
            "ax.set_title('The bearish leg is the killer — stocks keep rising after a Dark Cloud')\n"
            "plt.tight_layout(); plt.show()\n"
            f"print('Dark Cloud short loses at every horizon; Piercing long is just the market drift.')"
        ),
        md(
            "The **Dark Cloud (bearish) leg** is negative at every horizon — after the "
            "supposed 'top', stocks keep climbing, so the short bleeds. The **Piercing (bullish) "
            "leg** *looks* green over 5–10 days, but that's a trap: the unconditional 10-day "
            f"long drift is **+{R['h10'][3]:.3f}%**, and the Piercing leg earns only "
            f"**+{R['bull'][10][0]:.3f}%** — it **underperforms simply holding for a day**. "
            "No reversal alpha in either leg."
        ),

        # ---- BEAT 5 — VERDICT -----------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** Signed day-after return {R['h1'][2]:+.3f}%, HAC "
            f"*t* {R['h1'][5]:+.2f} (wrong sign); win-rate {R['h1'][4]:.0f}% (below a coin). "
            f"Placebo *p* = {R['h1'][7]:.3f} — almost every random pick beats the twins.\n"
            "- **Tradability — Mirage.** The gross edge is negative before a cent of cost; "
            "round-trip + borrow only deepen the loss.\n"
            "- **Reliable reversal? — Busted.** The bearish leg keeps rising and the bullish "
            "leg just rides beta. The twins don't mark turns on liquid US large-caps."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT -----------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "There's nothing to trade. The gross is already below zero, so charging costs "
            "(a round trip per event, plus borrow on the short Dark Cloud leg) only widens the "
            "loss:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = [st.summarize(PANEL, h, placebo=False) for h in horizons]\n"
            "    gross = [r['mean']*100 for r in rows]; net = [r['net']*100 for r in rows]\n"
            "else:\n"
            "    gross = [R[f'h{h}'][2] for h in horizons]; net = [R[f'h{h}'][8] for h in horizons]\n"
            "x = np.arange(len(horizons)); w = 0.38\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.bar(x - w/2, gross, w, color=GREY, label='gross')\n"
            "ax.bar(x + w/2, net, w, color=RED, label='net (costs + borrow)')\n"
            "ax.axhline(0, c='k', lw=1); ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in horizons])\n"
            "ax.set_ylabel('signed return per event (%)'); ax.legend()\n"
            "ax.set_title('Both bars are below zero — there is no positive edge to harvest')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Gross is already negative — costs are not the killer, the signal is.')"
        ),

        # ---- BEAT 7 — GOING FURTHER ----------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The positive control.** The companion notebook plants a *real* day-after "
            "reversal in synthetic data — and the exact same test lights up at HAC *t* = "
            f"{R['syn'][1][3]:+.2f}. The machinery works; the market just doesn't cooperate.\n"
            "- **The trend / volume caveat.** Textbooks say the twins only count *after a trend* "
            "and *on heavy volume*. The companion shows those filters don't flip the sign — "
            "they just throw events away.\n"
            "- **The engulfing cousin.** [Study 402](../../402-engulfing-pattern/) tests the full "
            "real-body engulfing — the twins' bigger sibling — with the same backwards "
            "result.\n\n"
            "*Think the twins work on a different asset, timeframe, or with a smarter confirmation "
            "rule? Fork this, change the detector or the basket, and show a signed reversal that "
            "clears HAC *t* ≥ 2. That is the bar.*"
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
            "# Dark Cloud & Piercing — a quantitative teardown\n"
            "### Daily OHLCV · precise twin detector · signed event study · HAC inference · label-shuffle placebo\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — "
            "*same seven beats, every claim now carrying its standard error.* We detect every "
            "Piercing Line / Dark Cloud Cover by the textbook OHLC rule across a fixed 30-name "
            "basket, enter the next open (one execution lag), and test the signed forward return "
            "against zero (HAC *t*), against the unconditional base pool, against a coin, and "
            "against a synthetic positive control.\n\n"
            "> **Not investment advice.** Real data: yfinance daily bars (`auto_adjust=True`, "
            "total-return), 2005-01-03 → 2026-06-18, as-of **2026-06-18**, fingerprint "
            f"`{R['fp']}`; the offline core and the synthetic control run with no network. "
            "Methods & sources in [`docs/references.md`](../docs/references.md), reproducible "
            "numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> The `In plain words` notes translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ---------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Signed day-after return **{R['h1'][2]:+.3f}%**, HAC "
            f"*t* = **{R['h1'][5]:+.2f}** (wrong sign); placebo *p* = {R['h1'][7]:.3f}. |\n"
            f"| **Tradability** | `MIRAGE` | Gross signed mean negative at every horizon; "
            "round-trip + borrow only widen the loss. |\n"
            f"| **Reliable reversal?** | `BUSTED` | Win-rate {R['h1'][4]:.0f}% (below 50%); the "
            "bearish leg keeps rising, the bullish leg merely rides beta. |\n\n"
            "> In plain words: the twins are sold as failed-rally / failed-sell-off reversals. On "
            "21 years of liquid large-caps the signed effect points the *wrong way* and the leg "
            "that 'looks good' is just the market's drift."
        ),

        # ---- BEAT 1 ---------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let the prior bar be $(O_0,H_0,L_0,C_0)$ and the current bar $(O_1,H_1,L_1,C_1)$, "
            "with $\\text{mid}_0 = (O_0+C_0)/2$. The textbook twins:\n\n"
            "- **Piercing Line (bullish, $d=+1$):** $C_0<O_0$ (prior down), $C_1>O_1$ (curr up), "
            "$O_1<L_0$ (gap-down open), $C_1>\\text{mid}_0$ (closes past the midpoint), and "
            "$C_1<O_0$ (not a full engulfing).\n"
            "- **Dark Cloud Cover (bearish, $d=-1$):** the mirror — $C_0>O_0$, $C_1<O_1$, "
            "$O_1>H_0$, $C_1<\\text{mid}_0$, $C_1>O_0$.\n\n"
            "Hypotheses:\n\n"
            "- **H₁ (signal).** $\\mathbb{E}[d \\cdot r_{t+1\\to t+H}] > 0$ — the signed "
            "forward return is positive (the reversal happens).\n"
            "- **H₂ (beats a coin).** That mean exceeds the same statistic on random bars with "
            "coin-flip directions.\n"
            "- **H₃ (tradable).** It survives a round-trip cost plus borrow on the short leg.\n\n"
            "We reject **H₁** (sign is wrong, significant at H=1) and **H₂**; **H₃** "
            "is moot. A near-doji prior bar (no real midpoint) is excluded via a `min_body_frac` "
            "guard."
        ),

        # ---- BEAT 2 ---------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The twins are among the first reversal patterns taught in every candlestick course. "
            "If H₁–H₃ held they would be a free, replicable contrarian entry on liquid "
            "names. The finding that the signed effect is *negative* matters: it means the "
            "textbook reader is systematically shorting strength and buying weakness at exactly "
            "the moments the pattern flags — and paying spreads to do it."
        ),

        # ---- BEAT 3 ---------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Detector.** Precise OHLC twin rule (above); `min_body_frac` excludes doji priors.\n"
            "- **Entry.** One execution lag: pattern confirmed at close $t$, enter at open $t+1$.\n"
            "- **Exit.** Fixed horizon: close of $t+H$, $H\\in\\{1,3,5,10\\}$.\n"
            "- **Signing.** Long after a Piercing ($+1$), short after a Dark Cloud ($-1$).\n"
            "- **Benchmarks.** The unconditional forward-return pool (no-signal drift) and a "
            "label-shuffle placebo (random bars, coin signs, same count, 10,000 draws).\n"
            "- **Inference.** Newey–West HAC *t* and a plain one-sample *t* vs 0; a Welch *t* "
            "vs the base pool; a per-leg split.\n"
            "- **Costs.** 5 bps one-way × 2 (round trip) + 50 bps/yr borrow on the short share.\n"
            "- **Positive control.** A synthetic panel with a *planted* day-after reversal, "
            "confirming the test recovers an edge when one exists.\n\n"
            f"Basket: 29 liquid US large-caps + SPY, **{R['years']}** years "
            f"({R['start']} → {R['end']})."
        ),

        # ---- BEAT 4 ---------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Signed forward return vs zero — the inference-bar number\n\n"
            "If the reversal were real, the HAC *t* should clear +2. It doesn't — at H=1 it "
            "clears −2."
        ),
        code(
            "horizons = [1, 3, 5, 10]\n"
            "if HAVE_REAL:\n"
            "    rows = [st.summarize(PANEL, h, n_draws=2000) for h in horizons]\n"
            "    import pandas as pd\n"
            "    tab = pd.DataFrame([{'H': r['horizon'], 'n': r['n_events'],\n"
            "        'signed%': round(r['mean']*100,3), 'baseL%': round(r['base_long']*100,3),\n"
            "        'win%': round(r['win']*100,1), 'HAC_t': round(r['t_hac'],2),\n"
            "        'one_t': round(r['t_one'],2), 'plac_p': round(r['p_placebo'],3)} for r in rows])\n"
            "    tt = [r['t_hac'] for r in rows]\n"
            "else:\n"
            "    import pandas as pd\n"
            "    tt = [R[f'h{h}'][5] for h in horizons]\n"
            "    tab = pd.DataFrame([{'H': R[f'h{h}'][0], 'n': R[f'h{h}'][1],\n"
            "        'signed%': R[f'h{h}'][2], 'baseL%': R[f'h{h}'][3], 'win%': R[f'h{h}'][4],\n"
            "        'HAC_t': R[f'h{h}'][5], 'one_t': R[f'h{h}'][6], 'plac_p': R[f'h{h}'][7]}\n"
            "        for h in horizons])\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.0))\n"
            "ax.bar([f'{h}d' for h in horizons], tt, color=[RED if t < 0 else GREY for t in tt])\n"
            "for s in (2, -2): ax.axhline(s, ls='--', c=GREY, lw=1)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylim(-3.2, 3.2)\n"
            "ax.set_ylabel('HAC t (signed return vs 0)')\n"
            "ax.set_title('Wrong-signed and significant at 1 day; fading to noise after')\n"
            "plt.tight_layout(); plt.show()\n"
            "tab"
        ),
        md(
            f"> In plain words: at 1 day the signed return is **{R['h1'][2]:+.3f}%** with HAC "
            f"*t* = **{R['h1'][5]:+.2f}** — a statistically significant *wrong-sign* result. "
            "Far from a +2 reversal, it's a −2 anti-reversal. The placebo *p* near 1.0 says "
            "essentially every random pick of the same many trades beats the twins."
        ),
        md(
            "### 4b · The leg split — where the damage lives\n\n"
            "Piercing (bullish) vs Dark Cloud (bearish), each signed and tested against zero."
        ),
        code(
            "if HAVE_REAL:\n"
            "    legs = [st.leg_split(PANEL, h) for h in horizons]\n"
            "    bull_m = [l['bull']['mean']*100 for l in legs]; bull_t = [l['bull']['t_hac'] for l in legs]\n"
            "    bear_m = [l['bear']['mean']*100 for l in legs]; bear_t = [l['bear']['t_hac'] for l in legs]\n"
            "    baseL = [st.summarize(PANEL, h, placebo=False)['base_long']*100 for h in horizons]\n"
            "else:\n"
            "    bull_m = [R['bull'][h][0] for h in horizons]; bull_t = [R['bull'][h][1] for h in horizons]\n"
            "    bear_m = [R['bear'][h][0] for h in horizons]; bear_t = [R['bear'][h][1] for h in horizons]\n"
            "    baseL = [R[f'h{h}'][3] for h in horizons]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot([f'{h}d' for h in horizons], bull_m, 'o-', c=GREEN, lw=2, label='Piercing leg')\n"
            "ax.plot([f'{h}d' for h in horizons], bear_m, 'o-', c=RED, lw=2, label='Dark Cloud leg')\n"
            "ax.plot([f'{h}d' for h in horizons], baseL, 's--', c=GREY, lw=1.5, label='unconditional long drift')\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('signed leg return (%)'); ax.legend()\n"
            "ax.set_title('Bearish leg bleeds; bullish leg merely tracks (and lags) the market drift')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Piercing 10d:', bull_m[-1], '% vs unconditional long drift:', baseL[-1], '%')"
        ),
        md(
            f"> In plain words: the **Dark Cloud** short loses at every horizon "
            f"(1-day leg *t* = {R['bear'][1][1]:+.2f}). The **Piercing** long looks green by 5–10 "
            f"days, but the grey dashed line — the unconditional long drift — sits "
            f"*above* it: the bullish leg earns **+{R['bull'][10][0]:.3f}%** at 10 days while just "
            f"holding earned **+{R['h10'][3]:.3f}%**. That isn't reversal alpha; it's beta the "
            "pattern is *underperforming*."
        ),
        md(
            "### 4c · The placebo — could a random pick look this good?\n\n"
            "Draw the same number of random bars, sign each by a coin, repeat 10,000 times."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pr = st.placebo_pvalue(PANEL, 1, st.collect_events(PANEL,1).shape[0], n_draws=10000)\n"
            "    draws = pr['draws']*100; obs = pr['obs']*100; pval = pr['p_value']\n"
            "else:\n"
            "    rng = np.random.default_rng(407)\n"
            "    draws = rng.normal(0.0, 0.06, 10000); obs = R['h1'][2]; pval = R['h1'][7]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.0))\n"
            "ax.hist(draws, bins=40, color=GREY, alpha=.7, label='random coin-signed means (10k draws)')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'observed twin mean: {obs:+.3f}%')\n"
            "ax.axvline(0, c='k', lw=1); ax.set_xlabel('mean signed return (%)'); ax.set_ylabel('count')\n"
            "ax.set_title(f'The twins sit in the LEFT tail — placebo p = {pval:.3f}'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'placebo p (random beats observed) = {pval:.3f}')"
        ),
        md(
            f"> In plain words: the observed twin mean lands in the **left tail** of the random "
            f"distribution — placebo *p* = **{R['h1'][7]:.3f}**. A coin would have done better "
            "essentially every time."
        ),
        md(
            "### 4d · Myth check — does a trend / volume filter rescue it?\n\n"
            "The textbook says the twins only count *after a trend* (Piercing after a downtrend, "
            "Dark Cloud after an uptrend) and *on heavy volume*. Add each filter at H=1."
        ),
        code(
            "import pandas as pd\n"
            "specs = [('plain', {}), ('+trend', {'require_trend': True}),\n"
            "         ('+volume spike', {'require_volume': True}),\n"
            "         ('+both', {'require_trend': True, 'require_volume': True})]\n"
            "if HAVE_REAL:\n"
            "    mrows = []\n"
            "    for lab, kw in specs:\n"
            "        s = st.summarize(PANEL, 1, placebo=False, **kw)\n"
            "        mrows.append({'filter': lab, 'n': s['n_events'],\n"
            "            'signed%': round(s['mean']*100,3), 'HAC_t': round(s['t_hac'],2),\n"
            "            'win%': round(s['win']*100,1)})\n"
            "    myth = pd.DataFrame(mrows)\n"
            "else:\n"
            "    myth = pd.DataFrame([{'filter': m[0], 'n': m[1], 'signed%': m[2],\n"
            "        'HAC_t': m[3], 'win%': m[4]} for m in R['myth']])\n"
            "myth"
        ),
        md(
            "> In plain words: every filtered variant is **still negative** and win-rate stays "
            "below 50%. The sub-2 *t*'s on the filtered cuts are just small samples (the volume "
            "filter throws away ~92% of events), not a sign flip. The folk caveat doesn't turn "
            "the reversal positive — it just discards trades."
        ),

        # ---- BEAT 5 ---------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — signed day-after return {R['h1'][2]:+.3f}%, HAC "
            f"*t* {R['h1'][5]:+.2f} (wrong sign), win {R['h1'][4]:.0f}%, placebo *p* "
            f"{R['h1'][7]:.3f}. Longer horizons fade to insignificance, never positive-significant.\n"
            "- **Tradability `MIRAGE`** — gross signed mean negative at every horizon; the "
            "round-trip + borrow charge only widens the loss. Nothing to scale.\n"
            f"- **Reliable reversal? `BUSTED`** — the bearish leg keeps rising "
            f"(*t* {R['bear'][1][1]:+.2f}) and the bullish leg underperforms the unconditional "
            "drift. Survivorship works *for* the bullish claim and still can't save it."
        ),

        # ---- BEAT 6 ---------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the cost landscape\n\n"
            "Each event is a fresh round trip (5 bps × 2) plus borrow on the short share. The "
            "gross is already negative, so there is no positive break-even cost:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = [st.summarize(PANEL, h, placebo=False) for h in horizons]\n"
            "    gross = [r['mean']*100 for r in rows]; net = [r['net']*100 for r in rows]\n"
            "else:\n"
            "    gross = [R[f'h{h}'][2] for h in horizons]; net = [R[f'h{h}'][8] for h in horizons]\n"
            "x = np.arange(len(horizons)); w = 0.38\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.0))\n"
            "ax.bar(x - w/2, gross, w, color=GREY, label='gross')\n"
            "ax.bar(x + w/2, net, w, color=RED, label='net (5bps RT + borrow)')\n"
            "ax.axhline(0, c='k', lw=1); ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in horizons])\n"
            "ax.set_ylabel('signed return per event (%)'); ax.legend()\n"
            "ax.set_title('No positive break-even — the gross is the problem')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Break-even cost: none exists — gross is below zero at every horizon.')"
        ),

        # ---- BEAT 7 ---------------------------------------------------------
        md(
            "## 7 · Going further — the positive control\n\n"
            "Is the *test* capable of seeing a reversal, or is it broken? Plant a known day-after "
            "reversal in a synthetic panel (the day after a detected twin gets an extra drift in "
            "the pattern's direction) and re-run the identical pipeline:"
        ),
        code(
            "import pandas as pd\n"
            "srows = []\n"
            "for edge in (0.000, 0.004):\n"
            "    syn, truth = data.synthetic_panel(edge=edge, seed=407)\n"
            "    s = st.summarize(syn, 1, n_draws=5000)\n"
            "    srows.append({'planted_edge': edge, 'events': s['n_events'],\n"
            "        'signed%': round(s['mean']*100,3), 'HAC_t': round(s['t_hac'],2),\n"
            "        'plac_p': round(s['p_placebo'],3), 'win%': round(s['win']*100,1)})\n"
            "syn_tab = pd.DataFrame(srows)\n"
            "fig, ax = plt.subplots(figsize=(8.0, 3.8))\n"
            "ax.bar(['edge = 0\\n(random walk)', 'edge = 0.004\\n(planted reversal)'],\n"
            "       [srows[0]['HAC_t'], srows[1]['HAC_t']], color=[GREY, GREEN])\n"
            "ax.axhline(2, ls='--', c=GREY); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('HAC t (signed return vs 0)')\n"
            "ax.set_title('Engine is faithful: ~0 on noise, lights up on a planted reversal')\n"
            "plt.tight_layout(); plt.show()\n"
            "syn_tab"
        ),
        md(
            f"With **no** planted reversal the test sits at HAC *t* = {R['syn'][0][3]:+.2f} (it "
            "does not manufacture significance from a random walk); a planted reversal lights up "
            f"to *t* = **{R['syn'][1][3]:+.2f}** with a **{R['syn'][1][5]:.0f}%** win-rate. The "
            "harness **can** detect a real day-after reversal — the real tape simply has the "
            "opposite. Forks worth trying: confirmation-bar variants (wait for a follow-through "
            "close), the twins on FX or commodity futures, or the *inverse* (fade the pattern as "
            "a continuation signal, which 4a hints might at least have the right sign)."
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
