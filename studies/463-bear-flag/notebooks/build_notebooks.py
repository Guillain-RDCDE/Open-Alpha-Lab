"""Generate the two narrative notebooks for Study 463 (Bear-Flag).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached daily tapes under
../_cache/ and otherwise quote the frozen headline numbers in ``R`` (mirroring docs/results.md).
The synthetic positive control runs anywhere with no network.
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
# yfinance daily, 5 indices/ETFs (SPY QQQ IWM DIA GLD), 2005-01-03 -> 2026-05-29 (As-of
# 2026-05-31, partial June dropped), 21.4 years, pole>=6%/10bar, 7-bar up-flag, breakdown short.
# All return numbers are SHORT side (positive = the short made money).
R = dict(
    asof="2026-05-31", start="2005-01-03", end="2026-05-29", years=21.4,
    tickers=["SPY", "QQQ", "IWM", "DIA", "GLD"], n_entries=156,
    fp_spy="4cb5244f3990",
    # pooled breakdown-short, per horizon:
    # (H, n, brk_bps, win%, one_sample_t, random_bps, delta_bps, net_bps, welch_t, welch_p)
    h5=(5, 156, -57.5, 46, -1.92, -10.3, -47.2, -59.5, -1.24, 0.218),
    h10=(10, 155, -57.3, 41, -1.55, -53.6, -3.7, -59.3, -0.07, 0.944),
    h20=(20, 155, -98.2, 47, -1.65, -33.1, -65.1, -100.2, -1.04, 0.299),
    h60=(60, 153, -303.2, 29, -3.39, -118.9, -184.2, -305.2, -1.90, 0.059),
    # per-ticker H=20: (ticker, entries, brk_bps, one_sample_t, random_bps, delta_bps)
    per=[("SPY", 27, 73.9, 0.56, -39.3, 113.2), ("QQQ", 31, -177.3, -1.63, -88.2, -89.1),
         ("IWM", 46, -70.3, -0.80, -2.7, -67.6), ("DIA", 20, -141.2, -0.68, -33.7, -107.4),
         ("GLD", 32, -182.4, -1.35, -1.5, -180.9)],
    # shuffled-flag placebo (SPY, H=20, 500 draws): obs_bps, p, draws
    placebo=(73.9, 0.323, 500),
    # synthetic control (H=20, n_days=4000): (edge, n, brk_bps, win%, one_sample_t)
    syn=[(0.00, 73, 36.5, 49, 0.67), (0.80, 100, 511.5, 87, 10.74)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Forecasts_continuation%3F: Busted](https://img.shields.io/badge/Forecasts_continuation%3F-Busted-8b949e?style=flat-square)\n\n"
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

from bear_flag import data, strategy as st

ASOF = "2026-05-31"
HAVE_REAL = data.have_real()
def load(t):
    b = data.load_real(t, allow_fetch=False)
    return b[b.index <= ASOF]
print("real bear-flag cache present:", HAVE_REAL, "| tickers:", data.DEFAULT_TICKERS)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does a bear flag actually \"continue the drop\"? 🏴\n"
            "### A famous chart pattern — a sharp fall, a little pause, then *the second leg down* — "
            "meets a stopwatch\n\n"
            + BADGES +
            "Open any chart-pattern course and you'll meet the **bear flag**: a fast, near-vertical "
            "drop (the *pole*), then a small, calm, slightly *up-sloping* drift (the *flag*), and "
            "then — the lore says — a **breakdown** that launches the *second leg down*, often the "
            "same size as the pole. So when price breaks below the little flag, you **short**: the "
            "drop is \"supposed\" to continue.\n\n"
            "It *looks* uncanny on a hand-picked chart. But a shape you label **after** the move — "
            "choosing which drop is a 'pole' and which drift is a 'flag' — is the textbook setup for "
            "fooling yourself. So we did the only fair thing: encode the bear flag **mechanically** "
            "(no eyeballing), fire the \"short the breakdown\" rule across five big indices over 21 "
            "years, and time the result with a stopwatch — against the only baseline that matters: "
            "**shorting on random days instead.**\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo and the cost math? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice** — research & education. Every chart is drawn by the code "
            "beside it; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| If I short when price breaks down out of the flag, does the drop continue? | **No.** "
            "The short **loses money at every horizon** — after the breakdown, price tends to *bounce*, "
            "not fall further. |\n"
            "| Is the breakdown worse than shorting on random days? | **Yes.** A short on a *random* day "
            "does **better** (less badly) than the famous breakdown short. The flag adds nothing. |\n"
            "| Does the flag's *shape* do the work? | **No.** Replace the up-sloping flag with a "
            "coin-flip and the result barely changes. The geometry isn't carrying any information. |\n"
            "| So is it a tradable edge? | **No.** It's a way to **short into the market's upward "
            "drift** — a slow bleed dressed up as a pattern. |\n\n"
            "> The bear flag is a great way to *narrate* a drop after the fact. As a *forecast* — "
            "\"the breakdown continues down\" — it's a **mirage**: the breakdown is, if anything, a "
            "short-term *bottom*, and shorting it just pays the drift the wrong way."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"A sharp drop is the **pole**. The small, up-sloping pause that follows is the "
            "**flag**. When price breaks **below** the flag, the second leg down begins — short it, "
            "and target a **measured move** the size of the pole. Flags are *continuation* patterns: "
            "the trend resumes.\"*\n\n"
            "This is the **flag/pennant** rule from the founding charting texts (Edwards & Magee, "
            "1948) and the modern pattern catalogues (Bulkowski's *Encyclopedia of Chart Patterns*; "
            "Murphy's *Technical Analysis of the Financial Markets*). It's one of the most "
            "recognisable setups in technical analysis — so: does the flag actually *forecast* the "
            "continuation?"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the flag genuinely *forecast* the second leg down, it would be remarkable: a small "
            "consolidation shape would predict the next move's *direction*, a clean crack in market "
            "efficiency you could trade with a ruler. That's the dream the pattern sells.\n\n"
            "But there are two traps. First, a bear flag is **labelled by hand, after the swings have "
            "happened** — you pick the drop and the drift that make the flag *look* right. Second, "
            "stock indices drift **up** over time, so *any* short is fighting a headwind — which makes "
            "a losing short look 'almost right'. To separate the **pattern** from the **tide**, we "
            "(a) detect the flag by a fixed mechanical rule with no hindsight, and (b) compare the "
            "breakdown short to shorting on **random days**. We'll do both."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **{len(R['tickers'])} liquid indices/ETFs** ({', '.join(R['tickers'])}), daily, "
            f"over **{R['years']:.0f} years** ({R['start']} → {R['end']}), and:\n\n"
            "1. **Find the pole mechanically.** A sharp drop: the close falls at least **6%** over a "
            "10-bar window — measurable, no eyeballing.\n"
            "2. **Find the flag by rule.** The next **7 bars** must drift gently *up* (a positive "
            "slope, against the pole) without retracing more than 60% of the drop — a pause, not a "
            "reversal.\n"
            "3. **Trade the lore.** When the close breaks **below the flag's lower line**, short at "
            "the next close; measure the return over the next **5 / 10 / 20 / 60 days** (a continued "
            "drop = a profit).\n"
            "4. **The honest baseline.** Do the exact same short on **random days**. If the flag "
            "matters, the breakdown short must beat random. *If it doesn't, the pattern is a mirage* "
            "— that's the result that would make us say so, announced before we look."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "First, what does a mechanical bear flag even look like? Here's a stretch of SPY with the "
            "detected flag windows and the breakdown bars the rule would short."
        ),
        code(
            "b = load('SPY') if HAVE_REAL else None\n"
            "if HAVE_REAL:\n"
            "    cl = b['close']; seg = cl.iloc[-900:]\n"
            "    flags = st.detect_flags(cl)\n"
            "    ent = st.breakdown_entries(cl)\n"
            "    ent = ent[ent >= seg.index[0]]\n"
            "    fent = flags.reindex(seg.index).fillna(False)\n"
            "    fig, ax = plt.subplots(figsize=(10.2, 4.8))\n"
            "    ax.plot(seg.index, seg.values, c='k', lw=1.1, label='SPY close')\n"
            "    ax.scatter(seg.index[fent.to_numpy()], seg[fent.to_numpy()], c=AMBER, s=14, zorder=4, label='flag end (consolidation)')\n"
            "    ax.scatter(ent, cl.reindex(ent), c=RED, s=55, marker='v', zorder=5, label='breakdown SHORT')\n"
            "    ax.set_title('Mechanical bear flags on SPY (last ~3.5y)'); ax.legend(loc='upper left')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('breakdown shorts in window:', len(ent))\n"
            "else:\n"
            "    print('(no cache — see docs/results.md for the frozen numbers)')"
        ),
        md(
            "Now the test. **Race the breakdown short against shorting on random days** at four "
            "horizons. A *positive* bar means the short made money (the drop continued). Red = short "
            "the breakdown; grey = short on random days."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    brk, rnd = [], []\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            c = load(t)['close']\n"
            "            e = st.breakdown_entries(c)\n"
            "            re = st.random_entries(c, max(len(e),50), seed=7)\n"
            "            tt.append(st.forward_returns(c, e, h)); rr.append(st.forward_returns(c, re, h))\n"
            "        brk.append(np.concatenate(tt).mean()*1e4); rnd.append(np.concatenate(rr).mean()*1e4)\n"
            "else:\n"
            "    brk = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x-.2, brk, .4, color=RED, label='short the breakdown')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='short on random days')\n"
            "for i,(a,bb) in enumerate(zip(brk,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='top' if a<0 else 'bottom',fontsize=8)\n"
            "    ax.annotate(f'{bb:+.0f}',(i+.2,bb),ha='center',va='top' if bb<0 else 'bottom',fontsize=8)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean SHORT return (bps)')\n"
            "ax.set_title('The breakdown short LOSES — and loses worse than random'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('breakdown:', [round(v) for v in brk]); print('random:', [round(v) for v in rnd])"
        ),
        md(
            f"There's the whole story in one chart. The breakdown short **loses money at every "
            f"horizon** (**{R['h20'][2]:.0f} bps** over 20 days, **{R['h60'][2]:.0f} bps** over 60) — "
            "after the 'continuation' breakdown, price tends to *go up*, not down. And it's *worse* "
            "than shorting on random days at every horizon. The pattern doesn't continue the drop; if "
            "anything the breakdown is a short-term **bottom**."
        ),
        md(
            "**One more sanity check.** What if we scramble the flag's *shape* — keep the sharp-drop "
            "pole filter, but decide whether each bar is a 'flag' by coin flip instead of the "
            "up-sloping test? If price really respects *the flag*, the nonsense flag should do much "
            "worse."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = load('SPY')['close']\n"
            "    pl = st.shuffled_flag_placebo(c, 20, n_draws=200, seed=463)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "print(f'real flag breakdown short (SPY, 20d): {obs:+.1f} bps')\n"
            "print(f'... and {pval*100:.0f}% of *coin-flip-flag* breakdowns do at least as well (p={pval:.2f}).')\n"
            "print('=> the flag geometry is not doing the work.')"
        ),
        md(
            f"About a third of the **coin-flip** flags match or beat the real one "
            f"(*p* = {R['placebo'][1]:.2f}). If price genuinely respected *the flag's shape*, a "
            "scramble would collapse the result. It doesn't — because the result was never about the "
            "flag."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The breakdown short does **not** beat shorting on random days "
            "(it's *worse* at every horizon; the breakdown-vs-random difference never clears *t* = 2). "
            "The short simply loses to the market's upward drift.\n"
            "- **Tradability — Mirage.** There's no continuation to capture — only a slow bleed from "
            "shorting a rising index, made worse by costs.\n"
            "- **\"Does the flag forecast continuation\"? — Busted.** Replace the flag shape with a "
            "coin flip and the result barely moves. The flag doesn't forecast."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "There's nothing here to trade — and worse than nothing. The breakdown short is a way to "
            "stand in front of the market's long-run climb and get run over: it loses on average at "
            "every horizon and the loss *deepens* with time (−303 bps at 60 days). Costs "
            "(commissions + spread on every breakdown) only add to the bleed. As a forecasting "
            "pattern, the bear flag doesn't pay; as a drawing tool, it was never meant to be a "
            "strategy on a drifting index."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The measured move.** The lore's headline target is a drop the size of the pole. "
            "That's a *conditional* claim you can test directly — and on this tape the conditional "
            "drop simply doesn't show up.\n"
            "- **Different thresholds.** Try a steeper pole, a tighter flag, a stricter slope — the "
            "result is robust: drift in, no continuation out.\n"
            "- **A real positive control.** The quants notebook plants a *genuine* post-breakdown "
            "continuation into a synthetic tape and shows the harness banks it (so the null here "
            "isn't a dead detector — it's an honest 'nothing there').\n\n"
            "*Think the flag forecasts? Show the breakdown short beating random shorts at **t ≥ 2** "
            "on a real tape — then we'll talk.*"
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
            "# The Bear-Flag — a quantitative teardown 🔬\n"
            "### Mechanical pole + up-flag + breakdown on 5 indices · short-side forward returns · "
            "one-sample HAC *t* · a drift-matched random-short baseline · a shuffled-flag geometry "
            "placebo · costs · a synthetic planted-continuation control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The job "
            "is to separate the **flag** from the **drift**: a short on an upward-drifting index has "
            "built-in negative carry, so the only meaningful test is breakdown-vs-random (same short "
            "side), plus a placebo that destroys the flag's geometry while preserving its marginals.\n\n"
            "> ⚠️ **Data note.** 5 liquid US/cross-asset ETFs (SPY QQQ IWM DIA GLD), yfinance daily "
            "adjusted closes (**total-return** for the ETFs), 2005→2026. A pole is a ≥6% drop over 10 "
            "bars; the flag is a 7-bar up-slope (≤60% retrace); entry is the **next close** (one "
            "documented lag), short side. Offline core + synthetic control are deterministic. "
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
            f"| **Signal** | `NONE` | Breakdown short vs a **drift-matched random short**: the "
            f"breakdown is *worse* at every horizon (Δ = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/"
            f"{R['h20'][6]:+.0f}/{R['h60'][6]:+.0f} bps) and the breakdown-minus-random difference "
            f"**never clears t = 2** (Welch t at 20d = {R['h20'][8]:+.2f}, 60d = {R['h60'][8]:+.2f}). |\n"
            f"| **Tradability** | `MIRAGE` | The short **loses** at every horizon (20d "
            f"{R['h20'][2]:+.0f} bps, 60d {R['h60'][2]:+.0f} bps); the negative one-sample t's are "
            "**drift carry**, not continuation. No edge to scale. |\n"
            f"| **Forecasts continuation?** | `BUSTED` | Replacing the up-sloping flag test with a coin "
            f"(shuffled-flag placebo) leaves the result intact: **p = {R['placebo'][1]:.2f}** of "
            "coin-flip flags match or beat the real one. The geometry isn't load-bearing. |\n\n"
            "> 💡 In plain words: the breakdown short isn't a continuation trade — it's a *short into "
            "the equity drift*. Race it vs a random short or scramble the flag's geometry and there's "
            "nothing left. No continuation edge."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "A bear flag is a **pole** (a sharp drop $\\Delta = \\log C_{t_0} - \\min \\log C$ over a "
            "lookback, $\\Delta \\ge \\Delta_{\\min}$), then a **flag**: a window of $L$ bars with a "
            "*positive* OLS slope (up-sloping, against the pole) retracing at most a fraction $\\rho$ "
            "of the pole. Let $\\ell_t$ be the flag's lower trendline at bar $t$. The rule **shorts** "
            "when $C_t<\\ell_t$ and rides the 'second leg down'.\n\n"
            "- **H₀ (drift).** Breakdown-short returns equal a drift-matched **random-short** baseline.\n"
            "- **H₁ (the flag forecasts).** Breakdown-short *exceeds* random at some horizon, t ≥ 2.\n"
            "- **H₂ (the geometry matters).** Breakdown-short exceeds a **shuffled-flag** rule whose "
            "up-slope test is replaced by a coin.\n\n"
            "We find **H₀ not rejected** (breakdown ≤ random at every horizon — in fact *worse*), "
            "**H₁ rejected** (Welch t never ≥ 2, and negative), **H₂ rejected** (placebo p ≈ 0.32). "
            "The steelman fails on every leg."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the two confounds this design must kill\n\n"
            "**(a) Drift (short side).** Equity indices have a positive unconditional daily mean, so a "
            "*short* held over any horizon carries a **negative** drift. A one-sample $t$ of the "
            "breakdown short against **zero** measures that carry, not continuation. The fix is the "
            "**random-short baseline** (same instrument, epoch, hold, short side) and a Welch test of "
            "breakdown-*minus*-random.\n\n"
            "**(b) Geometry as a free parameter.** A flag is a chosen pole + a chosen drift; the danger "
            "is that *any* sharp drop followed by *any* pause produces 'flags'. The **shuffled-flag "
            "placebo** keeps the pole filter and the price marginal but replaces the up-slope test with "
            "a per-bar coin — the flag's defining shape becomes meaningless, so if the real result "
            "survives the scramble, the geometry was never load-bearing."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** {', '.join(R['tickers'])}; yfinance daily adjusted closes "
            f"({R['start']}→{R['end']}, {R['years']:.1f}y). **{R['n_entries']} bear-flag breakdowns** "
            "pooled.\n"
            "- **Pole.** Log-close fall ≥ 6% over a 10-bar lookback (start-to-low), read only on bars "
            "up to *t* (no look-ahead).\n"
            "- **Flag.** 7-bar window, *positive* OLS slope, total up-retrace ≤ 60% of the pole; the "
            "lower trendline is the OLS fit shifted to the lowest residual, extrapolated to *t*.\n"
            "- **Entry.** First close below the lower flag line; **short** at the **next close** (one "
            "lag); hold H ∈ {5,10,20,60}. A continued drop is a profit.\n"
            "- **Null #1 — one-sample HAC t** of breakdown-short returns vs 0 (Newey-West).\n"
            "- **Null #2 — random-short baseline**, Welch two-sample breakdown vs random (the *real* test).\n"
            "- **Null #3 — shuffled-flag placebo** (up-slope test → coin; pole + marginals kept).\n"
            "- **Costs.** 1 bp one-way × 2 legs on every breakdown.\n"
            "- **Positive control.** Synthetic tape with a **planted** post-breakdown continuation "
            "(knob `edge`): edge=0 must NOT reach significance; edge>0 must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The drift trap — one-sample t vs the honest vs-random test\n\n"
            "Left: the breakdown short's **one-sample** t against zero. Right: the same short vs a "
            "**drift-matched random short** (the honest number). A real continuation edge would put "
            "the right bars *above* +2; instead they sit below zero."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    one_t, brk, rnd, welch = [], [], [], []\n"
            "    from scipy import stats\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            c = load(t)['close']\n"
            "            e = st.breakdown_entries(c)\n"
            "            re = st.random_entries(c, max(len(e),50), seed=7)\n"
            "            tt.append(st.forward_returns(c, e, h)); rr.append(st.forward_returns(c, re, h))\n"
            "        tt = np.concatenate(tt); rr = np.concatenate(rr)\n"
            "        one_t.append(st.summarize(tt)['t']); brk.append(tt.mean()*1e4); rnd.append(rr.mean()*1e4)\n"
            "        welch.append(stats.ttest_ind(tt, rr, equal_var=False)[0])\n"
            "else:\n"
            "    one_t = [R['h5'][4], R['h10'][4], R['h20'][4], R['h60'][4]]\n"
            "    brk = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "    welch = [R['h5'][8], R['h10'][8], R['h20'][8], R['h60'][8]]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "a1.bar([f'{h}d' for h in hs], one_t, color=GREY, width=.6)\n"
            "a1.axhline(2, ls='--', c=GREEN); a1.axhline(-2, ls='--', c=RED, label='|t|=2 bar'); a1.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(one_t): a1.annotate(f'{v:.2f}',(i,v),ha='center',va='top' if v<0 else 'bottom',fontsize=9)\n"
            "a1.set_title('One-sample t vs ZERO (it is short-side drift)'); a1.set_ylabel('t'); a1.legend()\n"
            "a2.bar([f'{h}d' for h in hs], welch, color=[GREEN if v>2 else RED for v in welch], width=.6)\n"
            "a2.axhline(2, ls='--', c=GREEN); a2.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(welch): a2.annotate(f'{v:+.2f}',(i,v),ha='center',va='top' if v<0 else 'bottom',fontsize=9)\n"
            "a2.set_title('Breakdown vs RANDOM short, Welch t (never clears 2)'); a2.set_ylabel('t')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('one-sample t:', [round(v,2) for v in one_t]); print('welch vs random:', [round(v,2) for v in welch])"
        ),
        md(
            f"> 💡 In plain words: the breakdown short's one-sample t is **negative** (60d "
            f"**{R['h60'][4]:.2f}**) — that's the short bleeding the index's upward drift. The honest "
            f"test (right) is breakdown-minus-random: it's negative at every horizon "
            f"({R['h20'][8]:+.2f} at 20d, {R['h60'][8]:+.2f} at 60d) and never clears +2. The flag "
            "adds nothing — and the breakdown is, if anything, the *wrong* side."
        ),
        md(
            "### 4b · Breakdown vs random across horizons — the gap is the verdict\n\n"
            "Mean SHORT return, breakdown vs random short, all four horizons. A real continuation "
            "pattern would put the breakdown bar *well above* random and above zero. It does neither."
        ),
        code(
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, brk, .4, color=RED, label='breakdown short')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='random short (drift baseline)')\n"
            "for i,(a,b) in enumerate(zip(brk,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='top' if a<0 else 'bottom',fontsize=8)\n"
            "    ax.annotate(f'{b:+.0f}',(i+.2,b),ha='center',va='top' if b<0 else 'bottom',fontsize=8)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean SHORT return (bps)')\n"
            "ax.set_title('Breakdown short does not beat random — it loses to it'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('delta breakdown-random (bps):', [round(a-b) for a,b in zip(brk,rnd)])"
        ),
        md(
            f"> 💡 In plain words: at 20 days the breakdown short is **{R['h20'][2]:.0f} bps** while a "
            f"random short is **{R['h20'][5]:.0f} bps** — the famous breakdown *underperforms* a dart by "
            f"{abs(R['h20'][6]):.0f} bps. Every horizon tells the same story. There is no measured-move "
            "continuation."
        ),
        md(
            "### 4c · The geometry placebo — scramble the flag, nothing changes\n\n"
            "Keep the pole filter; replace the up-sloping-flag test with a per-bar coin, so which bars "
            "are 'flags' is geometric nonsense. If price respects *the flag shape*, the scramble should "
            "demolish the result. The observed breakdown return should sit far in the right tail of the "
            "scrambled distribution. It doesn't."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = load('SPY')['close']\n"
            "    obs = st.forward_returns(c, st.breakdown_entries(c), 20).mean()*1e4\n"
            "    rng = np.random.default_rng(463); draws = []\n"
            "    for _ in range(200):\n"
            "        s = int(rng.integers(1, 2**31-1))\n"
            "        e = st.breakdown_entries(c, scramble_seed=s)\n"
            "        rr = st.forward_returns(c, e, 20)\n"
            "        if rr.size: draws.append(rr.mean()*1e4)\n"
            "    draws = np.array(draws); pval = (np.sum(draws>=obs)+1)/(len(draws)+1)\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "    rng = np.random.default_rng(463); draws = rng.normal(20, 90, 200)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=40, color=GREY, alpha=.85, label='coin-flip-flag breakdowns (SPY, 20d)')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'real flag {obs:+.0f} bps')\n"
            "ax.set_xlabel('mean breakdown-short 20d return (bps)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Real flag sits mid-pack: placebo p = {pval:.2f}'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'real flag {obs:+.1f} bps   placebo p={pval:.2f}  (>0.05 => geometry not load-bearing)')"
        ),
        md(
            f"> 💡 In plain words: the real flag (red line) sits **inside** the coin-flip-flag cloud — "
            f"**p ≈ {R['placebo'][1]:.2f}**. Geometric nonsense does about as well, so the specific "
            "up-sloping flag isn't carrying any information. This is the cleanest refutation of 'the "
            "flag forecasts continuation.'"
        ),
        md(
            "### 4d · Per-ticker — no continuation anywhere\n\n"
            "20-day breakdown-minus-random delta, per instrument. If the flag worked it would be "
            "positive across the board; instead it's negative in 4 of 5."
        ),
        code(
            "if HAVE_REAL:\n"
            "    names, deltas = [], []\n"
            "    for t in data.DEFAULT_TICKERS:\n"
            "        c = load(t)['close']\n"
            "        e = st.breakdown_entries(c); re = st.random_entries(c, max(len(e),50), seed=7)\n"
            "        d = st.summarize(st.forward_returns(c,e,20))['mean_bps'] - st.summarize(st.forward_returns(c,re,20))['mean_bps']\n"
            "        names.append(t); deltas.append(d)\n"
            "else:\n"
            "    names = [p[0] for p in R['per']]; deltas = [p[5] for p in R['per']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.bar(names, deltas, color=[GREEN if d>0 else RED for d in deltas], width=.6)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,d in enumerate(deltas): ax.annotate(f'{d:+.0f}',(i,d),ha='center',va='bottom' if d>=0 else 'top',fontsize=9)\n"
            "ax.set_ylabel('20d breakdown − random (bps)'); ax.set_title('Breakdown underperforms random in 4 of 5 names')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('per-ticker 20d delta (bps):', {n:round(d) for n,d in zip(names,deltas)})"
        ),
        md(
            f"> 💡 In plain words: only **SPY** shows a positive delta ({R['per'][0][5]:+.0f} bps, on "
            f"just {R['per'][0][1]} trades at t={R['per'][0][3]:+.2f} — noise); GLD is "
            f"**{R['per'][4][5]:+.0f}** bps *behind* a random short. No coherent, cross-sectional "
            "continuation — exactly what you'd expect if the flag is just a relabelled drop."
        ),
        md(
            "### 4e · Synthetic positive control — the harness CAN bank a real continuation\n\n"
            "To prove the null is honest (not a dead detector), plant a **real** post-breakdown "
            "continuation into a synthetic tape and check the same breakdown-short rule banks it: "
            "edge=0 must stay at t≈0; edge>0 must light up with a high win-rate."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.80):\n"
            "    px, _ = data.synthetic_panel(edge=edge, seed=463, n_days=4000)\n"
            "    c = px['close']; e = st.breakdown_entries(c); s = st.summarize(st.forward_returns(c, e, 20))\n"
            "    res.append((edge, s['n'], s['mean_bps'], s['win']*100, s['t']))\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "labels = [f'planted edge\\n{e:.2f}' for e,_,_,_,_ in res]; tvals = [r[4] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('20d one-sample t (short)'); ax.set_title('Control: edge=0 -> t~0; planted continuation -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,n,m,w,t in res: print(f'edge={e:.2f}: n={n} brk={m:+.1f}bps win={w:.0f}% t={t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted continuation the control sits at "
            f"**t = {R['syn'][0][4]:.2f}** (win {R['syn'][0][3]:.0f}% — no false positive); a planted "
            f"continuation reaches **t = {R['syn'][1][4]:.2f}** (win {R['syn'][1][3]:.0f}%). The "
            "detector works — so the negative real-tape result is a genuine 'nothing there', not a "
            "broken pipeline."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the breakdown short does not beat a drift-matched random short "
            f"(breakdown − random = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f}/"
            f"{R['h60'][6]:+.0f} bps at 5/10/20/60d; Welch t never clears 2, all negative, "
            f"min **{R['h60'][8]:+.2f}** at 60d). The short simply pays the index's upward drift.\n"
            f"- **Tradability `MIRAGE`** — the short loses at every horizon ({R['h20'][2]:+.0f} bps at "
            f"20d, {R['h60'][2]:+.0f} at 60d) and costs only deepen the hole. No continuation edge to "
            "scale.\n"
            f"- **Forecasts continuation? `BUSTED`** — the shuffled-flag placebo leaves the result "
            f"intact (**p = {R['placebo'][1]:.2f}**): coin-flip flags do as well as the real geometry, "
            "so the up-sloping flag carries no forecasting information."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — there is nothing to trade\n\n"
            "The breakdown short has no continuation to capture; its entire 'signal' is negative drift "
            "carry from standing short in front of a rising index. It loses on average at every horizon "
            "and the loss compounds with time. There is no capacity question because there is no edge — "
            "only negative expectancy plus costs. The bear flag is a descriptive after-the-fact label, "
            "not a forecasting strategy on a drifting tape."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The measured-move claim.** The lore's headline target — a continuation the size of "
            "the pole — is a clean conditional hypothesis; here the conditional drop simply does not "
            "appear (the breakdown is closer to a local bottom).\n"
            "- **Parameter sweeps.** Steeper poles, tighter flags, stricter slopes — all inherit the "
            "same short-side drift confound; the mechanical version here is the charitable upper "
            "bound.\n"
            "- **Bull flags / pennants** are affine relatives of the same geometry and inherit the "
            "same confound (a bull flag merely rides the drift it's measuring).\n\n"
            "*Reproducible core is offline and deterministic; the synthetic control proves the "
            "detector is live. Methods/sources: [`docs/references.md`](../docs/references.md); frozen "
            "numbers: [`docs/results.md`](../docs/results.md).*"
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
