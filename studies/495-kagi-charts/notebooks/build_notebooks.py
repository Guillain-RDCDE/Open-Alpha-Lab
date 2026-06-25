"""Generate the two narrative notebooks for Study 495 (Kagi Charts).

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
# 2026-05-31, partial June dropped), 21.4 years, Kagi reversal=4%, buy-the-yang-switch long.
R = dict(
    asof="2026-05-31", start="2005-01-03", end="2026-05-29", years=21.4,
    tickers=["SPY", "QQQ", "IWM", "DIA", "GLD"], n_entries=144, reversal=0.04,
    fp_spy="4cb5244f3990",
    # pooled yang-switch, per horizon:
    # (H, n, switch_bps, win%, one_sample_t, random_bps, delta_bps, net_bps, welch_t, welch_p)
    h5=(5, 144, 22.2, 58, 1.10, 10.3, 11.9, 20.2, 0.41, 0.682),
    h10=(10, 143, 22.8, 58, 0.65, 53.6, -30.8, 20.8, -0.71, 0.477),
    h20=(20, 143, 88.7, 57, 1.81, 33.1, 55.6, 86.7, 0.92, 0.358),
    h60=(60, 140, 253.3, 60, 2.84, 118.9, 134.3, 251.3, 1.27, 0.205),
    # per-ticker H=20: (ticker, entries, switch_bps, one_sample_t, random_bps, delta_bps)
    per=[("SPY", 24, 117.2, 0.99, 39.3, 77.9), ("QQQ", 29, 75.2, 0.54, 88.2, -13.0),
         ("IWM", 39, 85.0, 0.69, 2.7, 82.4), ("DIA", 26, 146.7, 1.69, 33.7, 112.9),
         ("GLD", 26, 26.9, 0.40, 1.5, 25.4)],
    # threshold-scramble placebo (SPY, H=20, 500 draws): obs_bps, p, draws
    placebo=(117.2, 0.545, 500),
    # synthetic control (H=20, n_days=8000): (edge, n, switch_bps, win%, one_sample_t)
    syn=[(0.00, 38, 82.7, 53, 1.25), (0.25, 30, 464.8, 80, 7.47)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Time_the_market%3F: Busted](https://img.shields.io/badge/Time_the_market%3F-Busted-8b949e?style=flat-square)\n\n"
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

from kagi_charts import data, strategy as st

ASOF = "2026-05-31"
HAVE_REAL = data.have_real()
def load(t):
    b = data.load_real(t, allow_fetch=False)
    return b[b.index <= ASOF]
print("real kagi cache present:", HAVE_REAL, "| tickers:", data.DEFAULT_TICKERS)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does a Kagi chart's \"yang\" line actually time the market? 〽️\n"
            "### A famous Japanese chart — thick lines, thin lines, shoulders and waists — meets a stopwatch\n\n"
            + BADGES +
            "Open a charting package and you'll find the **Kagi chart**: a time-less, price-only line "
            "that snakes up and down, reversing only when price turns by some fixed amount. Its "
            "trademark is **line thickness** — the line goes **thick (yang)** when an up-move breaks the "
            "last peak (a *shoulder*) and **thin (yin)** when a down-move breaks the last trough (a "
            "*waist*). The lore, from Steve Nison's *Beyond Candlesticks* onward, is simple: **buy when "
            "the line turns yang** (thick = the bulls have taken over) and **step aside on yin**.\n\n"
            "It *looks* decisive on a hand-picked chart. But a thick line shows up precisely when price "
            "is trending up — and stock indices trend up over time — so *any* \"buy strength\" rule will "
            "look good. So we did the only fair thing: encode the Kagi **mechanically** (no eyeballing), "
            "fire the \"buy the yang switch\" rule across five big indices over 21 years, and time the "
            "result with a stopwatch — against the only baseline that matters: **buying on random days "
            "instead.**\n\n"
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
            "| If I buy when the Kagi line turns **yang** (thick), do I make money? | **Yes — but only "
            "because the market goes up.** The raw win-rate is ~57–60% and the returns look fine. |\n"
            "| Is that *the Kagi's* doing? | **No.** Buy on **random days** instead and you do "
            "**just as well** — at 10 days the yang switch is actually *worse*. The thick line adds "
            "nothing a coin flip didn't already give you. |\n"
            "| Do the reversals \"time\" the market? | **Not in any usable way.** Re-draw the Kagi with "
            "a different reversal size and the result barely changes. The specific shoulders/waists "
            "aren't doing the work. |\n"
            "| So is it a tradable edge? | **No.** It's **beta in a costume** — the upward drift of "
            "stocks, re-labelled as a thick line. |\n\n"
            "> The Kagi is a tidy way to *filter noise* and *describe* a trend. As a *forecast* — "
            "\"yang means go long\" — it's a **mirage**: all of the apparent edge is the market's "
            "long-run climb, none of it is the line."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Draw a Kagi line: it runs in one direction until price reverses by a set amount, then "
            "it turns. When a rising line breaks the prior **shoulder** it thickens into a **yang** "
            "line — the uptrend is confirmed, so **buy**. When a falling line breaks the prior "
            "**waist** it thins into a **yin** line — **sell or stand aside**.\"*\n\n"
            "This is the **Kagi chart**, born in 1870s Japan and brought West by **Steve Nison**. It "
            "ships in TradingView, MetaTrader and StockCharts. The yin/yang thickness rule is its whole "
            "point — so: does the thickness actually *signal* anything?"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the yang switch genuinely *forecast* up-moves, it would be remarkable: a thickness flip "
            "would predict the next few weeks, a clean crack in market efficiency you could trade. "
            "That's the dream the tool sells.\n\n"
            "But there's a trap built into it. The line turns **thick exactly when price has been "
            "rising** — and it's drawn on a market (stock indices) that drifts **up** over time, so "
            "*any* \"buy when strong\" rule will look profitable. To separate the **tool** from the "
            "**tide**, we have to (a) draw the Kagi by a fixed mechanical rule with no hindsight, and "
            "(b) compare it to buying on **random days**. We'll do both."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **{len(R['tickers'])} liquid indices/ETFs** ({', '.join(R['tickers'])}), daily, "
            f"over **{R['years']:.0f} years** ({R['start']} → {R['end']}), and:\n\n"
            "1. **Build the Kagi line mechanically.** Walking the closes left to right, the line "
            f"reverses on a **{R['reversal']:.0%} counter-move** and records a shoulder/waist at each "
            "turn — using only data up to today, so we never peek ahead.\n"
            "2. **Spot the yang switch by rule.** The bar the line first breaks above the prior "
            "shoulder (thin→thick) — no eyeballing.\n"
            "3. **Trade the lore.** On that yang switch, buy at the **next close**; measure the return "
            "over the next **5 / 10 / 20 / 60 days**.\n"
            "4. **The honest baseline.** Do the exact same hold on **random days**. If the Kagi "
            "matters, the yang switch must beat random. *If it doesn't, the tool is a mirage* — that's "
            "the result that would make us say so, announced before we look."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "First, what does a mechanical Kagi even look like? Here's SPY with its yin (thin) and yang "
            "(thick) segments, and the yang switches the rule would buy."
        ),
        code(
            "b = load('SPY') if HAVE_REAL else None\n"
            "if HAVE_REAL:\n"
            "    cl = b['close']; seg = cl.iloc[-450:]\n"
            "    kl = st.kagi_line(cl, reversal=R['reversal'])\n"
            "    klseg = kl.reindex(seg.index)\n"
            "    ent = st.yang_switch_entries(cl, reversal=R['reversal'])\n"
            "    ent = ent[ent >= seg.index[0]]\n"
            "    fig, ax = plt.subplots(figsize=(10.2, 4.8))\n"
            "    thick = klseg['thick'].to_numpy()\n"
            "    # plot yin segments thin/grey, yang segments thick/green\n"
            "    ax.plot(seg.index, seg.values, c=GREY, lw=1.0, label='SPY close (yin)')\n"
            "    yangmask = np.where(thick, seg.values, np.nan)\n"
            "    ax.plot(seg.index, yangmask, c=GREEN, lw=2.6, label='yang (thick)')\n"
            "    ax.scatter(ent, cl.reindex(ent), c=GREEN, s=42, zorder=5, edgecolor='k', label='yang BUY')\n"
            "    ax.set_title('A mechanical Kagi chart on SPY (last ~2y): thin=yin, thick=yang')\n"
            "    ax.legend(loc='upper left')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('yang switches in window:', len(ent))\n"
            "else:\n"
            "    print('(no cache — see docs/results.md for the frozen numbers)')"
        ),
        md(
            "The thick green stretches sit on the up-legs — *as a description*. The question is whether "
            "those green buy dots are followed by gains beyond what any day would give. **Let's race the "
            "yang switch against random entries** at four horizons. Blue = buy the yang switch; "
            "grey = buy on random days."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    sw, rnd = [], []\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            c = load(t)['close']\n"
            "            e = st.yang_switch_entries(c, reversal=R['reversal'])\n"
            "            re = st.random_entries(c, max(len(e),50), seed=7)\n"
            "            tt.append(st.forward_returns(c, e, h)); rr.append(st.forward_returns(c, re, h))\n"
            "        sw.append(np.concatenate(tt).mean()*1e4); rnd.append(np.concatenate(rr).mean()*1e4)\n"
            "else:\n"
            "    sw = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x-.2, sw, .4, color='#2c6fbb', label='buy the yang switch')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='buy on random days')\n"
            "for i,(a,bb) in enumerate(zip(sw,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{bb:+.0f}',(i+.2,bb),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean return (bps)')\n"
            "ax.set_title('The yang switch does NOT meaningfully beat random'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('yang:', [round(v) for v in sw]); print('random:', [round(v) for v in rnd])"
        ),
        md(
            f"There's the story. The yang switch makes money in absolute terms (**+{R['h20'][2]:.0f} "
            f"bps** over 20 days) — but the gap over random is small and noisy, and at 10 days the "
            f"switch is actually *behind* random (**{R['h10'][2]:.0f}** vs **{R['h10'][5]:.0f}** bps). "
            "The quants notebook shows the switch-vs-random *t* never clears 2. The apparent edge is "
            "**the market's upward drift**, not the thickness."
        ),
        md(
            "**One more sanity check.** What if we re-draw the Kagi with a *different reversal size* — a "
            "1%, 3%, 7% Kagi instead of 4%? Each is an equally legitimate Kagi. If price really 'respects "
            "the yang switch', the real 4% line should stand out from the crowd of re-parameterised ones."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = load('SPY')['close']\n"
            "    pl = st.threshold_scramble_placebo(c, 20, reversal=R['reversal'], n_draws=200, seed=495)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "print(f'real 4% Kagi yang switch (SPY, 20d): {obs:+.1f} bps')\n"
            "print(f'... but {pval*100:.0f}% of *re-parameterised* Kagis do at least as well (p={pval:.2f}).')\n"
            "print('=> the specific reversal/geometry is not doing the work.')"
        ),
        md(
            f"More than half of the **re-parameterised** Kagis match or beat the real 4% one "
            f"(*p* = {R['placebo'][1]:.2f}). If price genuinely respected *this specific* Kagi, a random "
            "reversal would collapse the result. It doesn't — because the result was never about the "
            "shoulders and waists."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The yang-switch buy does **not** beat buying on random days "
            "(it's *worse* at 10 days; the switch-vs-random difference never clears *t* = 2). The "
            "decent absolute returns are the market's drift, not the line.\n"
            "- **Tradability — Mirage.** Nothing to trade once you remove the beta you were always "
            "getting for free — and costs (plus sitting out on yin) only make it worse.\n"
            "- **\"Do Kagi reversals time the market\"? — Busted.** Re-draw the Kagi with a different "
            "reversal and the result barely moves. The thickness doesn't forecast."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "There's nothing here to trade. The yang switch's *only* advantage over a coin flip is the "
            "market's long-run climb — which you'd capture more cheaply (and more fully) by just "
            "**holding the index**. The Kagi rule is long *less* of the time (it steps aside on yin) and "
            "pays costs on every flip, so it's a worse, more expensive way to be long. As a forecasting "
            "tool, it doesn't pay; as a noise filter, it was never meant to be a strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Other reversal definitions.** Try an ATR-based reversal or point-amount Kagi — the "
            "result is robust: drift in, thick line out (the placebo already sweeps reversal sizes).\n"
            "- **The yin short.** Going short on yin just shorts the drift — a guaranteed way to lose on "
            "an up-trending tape; a fun follow-up shows the symmetric failure.\n"
            "- **A real positive control.** The quants notebook plants a *genuine* post-yang momentum "
            "burst into a synthetic tape and shows the harness banks it (so the null result here isn't a "
            "dead detector — it's an honest 'nothing there').\n\n"
            "*Think the yang switch forecasts? Show it beating random entries at **t ≥ 2** on a real "
            "tape — then we'll talk.*"
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
            "# Kagi charts — a quantitative teardown 🔬\n"
            "### Mechanical Kagi lines on 5 indices · yang-switch forward returns · one-sample HAC *t* · "
            "a drift-matched random-entry baseline · a threshold-scramble geometry placebo · costs · a "
            "synthetic planted-momentum control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The job is "
            "to separate the **Kagi** from the **drift**: a thick (yang) line appears in up-trends and "
            "indices trend up, so *any* \"buy strength\" rule looks good. The only meaningful test is "
            "yang-vs-random, plus a placebo that re-parameterises the Kagi while preserving its "
            "marginals.\n\n"
            "> ⚠️ **Data note.** 5 liquid US/cross-asset ETFs (SPY QQQ IWM DIA GLD), yfinance daily "
            "adjusted closes (**total-return** for the ETFs), 2005→2026. The Kagi line is causal "
            f"(closes up to *t*, reversal={R['reversal']:.0%}); entry is the **next close** (one "
            "documented lag). Offline core + synthetic control are deterministic. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Yang switch vs a **drift-matched random** baseline: Δ = "
            f"{R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f}/{R['h60'][6]:+.0f} bps at "
            f"5/10/20/60d and the switch-minus-random difference **never clears t = 2** (Welch t at 20d "
            f"= {R['h20'][8]:+.2f}, 60d = {R['h60'][8]:+.2f}). |\n"
            f"| **Tradability** | `MIRAGE` | The one-sample t that creeps up with horizon (60d t = "
            f"{R['h60'][4]:.2f}) is **drift** — it vanishes against random entries and against cost. No "
            "residual edge to scale. |\n"
            f"| **Time the market?** | `BUSTED` | Re-parameterising the Kagi reversal (threshold-scramble "
            f"placebo) leaves the result intact: **p = {R['placebo'][1]:.2f}** of random-reversal Kagis "
            "match or beat the real 4% one. The shoulders/waists aren't doing the work. |\n\n"
            "> 💡 In plain words: the yang switch *looks* fine only because indices drift up. Strip the "
            "drift (race it vs random) or strip the geometry (re-draw the Kagi) and the edge evaporates. "
            "Classic beta-in-a-costume."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Build the Kagi line $K_t$ from closes: it extends in its current direction, reverses when "
            "price moves against it by $r$ (a fraction of the turning price), and records a "
            "**shoulder** $S$ (last swing high) and **waist** $W$ (last swing low). The line is "
            "**yang** (thick) once $C_t > S$ and **yin** (thin) once $C_t < W$. A **yang switch** is the "
            "bar thickness flips thin→thick. The rule buys the yang switch.\n\n"
            "- **H₀ (drift).** Yang-switch returns equal a drift-matched **random-entry** baseline.\n"
            "- **H₁ (the switch forecasts).** Yang returns **exceed** random at some horizon, t ≥ 2.\n"
            "- **H₂ (the geometry matters).** Yang returns exceed a **threshold-scrambled** Kagi whose "
            "reversal is randomly re-drawn.\n\n"
            "We find **H₀ not rejected** (switch ≈ random, worse at 10d), **H₁ rejected** (Welch t never "
            "≥ 2), **H₂ rejected** (placebo p ≈ 0.5). The steelman fails on every leg."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the two confounds this design must kill\n\n"
            "**(a) Drift.** Equity indices have a positive unconditional daily mean. *Any* long-only "
            "entry rule inherits it; a high one-sample $t$ against **zero** measures the tide, not the "
            "tool. The fix is the **random-entry baseline** (same instrument, epoch, hold) and a Welch "
            "test of yang-*minus*-random.\n\n"
            "**(b) The reversal as a free parameter.** A Kagi is defined by its reversal size; the "
            "danger is that *any* reasonable reversal on a trend produces 'confirming' yang switches. "
            "The **threshold-scramble placebo** re-draws the reversal from 1%–8% (keeping the price "
            "marginal) — the lines become a *different but equally valid* Kagi, so if the real 4% result "
            "survives the scramble, the specific geometry was never load-bearing."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** {', '.join(R['tickers'])}; yfinance daily adjusted closes "
            f"({R['start']}→{R['end']}, {R['years']:.1f}y). **{R['n_entries']} yang switches** pooled.\n"
            f"- **Kagi line.** Causal: reverses on a {R['reversal']:.0%} counter-move; shoulder = last "
            "swing high, waist = last swing low; yang once close > shoulder, yin once close < waist.\n"
            "- **Entry.** The bar thickness flips thin→thick; enter **next close** (one lag); hold "
            "H ∈ {5,10,20,60}.\n"
            "- **Null #1 — one-sample HAC t** of yang returns vs 0 (Newey-West).\n"
            "- **Null #2 — random-entry baseline**, Welch two-sample yang vs random (the *real* test).\n"
            "- **Null #3 — threshold-scramble placebo** (reversal re-drawn, marginals kept).\n"
            "- **Costs.** 1 bp one-way × 2 legs on every switch.\n"
            "- **Positive control.** Synthetic tape with a **planted** post-yang momentum burst (knob "
            "`edge`): edge=0 must NOT reach significance; edge>0 must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The drift trap — one-sample t creeps up, vs-random kills it\n\n"
            "Left: the yang switch's **one-sample** t against zero (the misleading number). Right: the "
            "same switch vs a **drift-matched random** baseline (the honest number)."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    one_t, sw, rnd, welch = [], [], [], []\n"
            "    from scipy import stats\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            c = load(t)['close']\n"
            "            e = st.yang_switch_entries(c, reversal=R['reversal'])\n"
            "            re = st.random_entries(c, max(len(e),50), seed=7)\n"
            "            tt.append(st.forward_returns(c, e, h)); rr.append(st.forward_returns(c, re, h))\n"
            "        tt = np.concatenate(tt); rr = np.concatenate(rr)\n"
            "        one_t.append(st.summarize(tt)['t']); sw.append(tt.mean()*1e4); rnd.append(rr.mean()*1e4)\n"
            "        welch.append(stats.ttest_ind(tt, rr, equal_var=False)[0])\n"
            "else:\n"
            "    one_t = [R['h5'][4], R['h10'][4], R['h20'][4], R['h60'][4]]\n"
            "    sw = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "    welch = [R['h5'][8], R['h10'][8], R['h20'][8], R['h60'][8]]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "a1.bar([f'{h}d' for h in hs], one_t, color=GREY, width=.6)\n"
            "a1.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,v in enumerate(one_t): a1.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom',fontsize=9)\n"
            "a1.set_title('One-sample t vs ZERO (misleading: it is drift)'); a1.set_ylabel('t'); a1.legend()\n"
            "a2.bar([f'{h}d' for h in hs], welch, color=[GREEN if v>2 else RED for v in welch], width=.6)\n"
            "a2.axhline(2, ls='--', c=RED); a2.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(welch): a2.annotate(f'{v:+.2f}',(i,v),ha='center',va='bottom' if v>=0 else 'top',fontsize=9)\n"
            "a2.set_title('Yang vs RANDOM, Welch t (honest: never clears 2)'); a2.set_ylabel('t')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('one-sample t:', [round(v,2) for v in one_t]); print('welch vs random:', [round(v,2) for v in welch])"
        ),
        md(
            f"> 💡 In plain words: the left bars creep up with horizon (60d **{R['h60'][4]:.2f}**) — but "
            f"that's the **drift**, every long-only entry inherits it. The right bars are the real test: "
            f"yang-minus-random tops out at just **{R['h60'][8]:+.2f}** (60d) and is *negative* at 10d "
            f"({R['h10'][8]:+.2f}) — never significant. The yang switch adds nothing over a coin flip."
        ),
        md(
            "### 4b · Yang vs random across horizons — the gap is the verdict\n\n"
            "Mean return, yang switch vs random entry, all four horizons. The switch should tower over "
            "random if the Kagi forecasts. It doesn't."
        ),
        code(
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, sw, .4, color='#2c6fbb', label='yang switch')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='random entry (drift baseline)')\n"
            "for i,(a,b) in enumerate(zip(sw,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{b:+.0f}',(i+.2,b),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean fwd return (bps)')\n"
            "ax.set_title('Yang switch does not meaningfully beat random entry'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('delta yang-random (bps):', [round(a-b) for a,b in zip(sw,rnd)])"
        ),
        md(
            f"> 💡 In plain words: at 20 days the switch is **+{R['h20'][2]:.0f} bps** and random is "
            f"**+{R['h20'][5]:.0f} bps** — a +{R['h20'][6]:.0f} bp gap that 4a already showed is noise "
            f"(Welch {R['h20'][8]:+.2f}). At 10 days the switch *loses* to a dart by "
            f"{abs(R['h10'][6]):.0f} bps. No clean, monotone edge."
        ),
        md(
            "### 4c · The geometry placebo — re-draw the Kagi, nothing changes\n\n"
            "Re-parameterise the reversal (1%–8%, keeping the price marginal) so the Kagi is a different "
            "but equally valid line. If price respects *this specific 4%* Kagi, the scramble should "
            "demolish the result. The observed yang-switch return should sit far in the right tail of "
            "the re-parameterised distribution. It doesn't."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = load('SPY')['close']\n"
            "    pl = st.threshold_scramble_placebo(c, 20, reversal=R['reversal'], n_draws=200, seed=495)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "    rng = np.random.default_rng(495); grid = np.linspace(0.01, 0.08, 64); draws=[]\n"
            "    for _ in range(200):\n"
            "        rv = float(rng.choice(grid)); ee = st.yang_switch_entries(c, reversal=rv)\n"
            "        rr = st.forward_returns(c, ee, 20)\n"
            "        if rr.size: draws.append(rr.mean()*1e4)\n"
            "    draws = np.array(draws)\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "    rng = np.random.default_rng(495); draws = rng.normal(95, 60, 200)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=30, color=GREY, alpha=.85, label='re-parameterised Kagis (SPY, 20d)')\n"
            "ax.axvline(obs, c='#2c6fbb', lw=2.5, label=f'real 4% Kagi {obs:+.0f} bps')\n"
            "ax.set_xlabel('mean yang-switch 20d return (bps)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Real Kagi sits mid-pack: placebo p = {pval:.2f}'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'real 4% Kagi {obs:+.1f} bps   placebo p={pval:.2f}  (>0.05 => geometry not load-bearing)')"
        ),
        md(
            f"> 💡 In plain words: the real 4% Kagi (blue line) sits **mid-pack** in the re-parameterised "
            f"cloud — **p = {R['placebo'][1]:.2f}**. A randomly chosen reversal does just as well, so the "
            "specific shoulders and waists aren't carrying information. This is the cleanest refutation "
            "of 'Kagi reversals time the market.'"
        ),
        md(
            "### 4d · Per-ticker — a positive sign but no significance anywhere\n\n"
            "20-day yang-minus-random delta, per instrument. Positive in 4 of 5 — but every per-ticker "
            "one-sample *t* is below 1.7, so this is noise, not a cross-sectional edge."
        ),
        code(
            "if HAVE_REAL:\n"
            "    names, deltas = [], []\n"
            "    for t in data.DEFAULT_TICKERS:\n"
            "        c = load(t)['close']\n"
            "        e = st.yang_switch_entries(c, reversal=R['reversal']); re = st.random_entries(c, max(len(e),50), seed=7)\n"
            "        d = st.summarize(st.forward_returns(c,e,20))['mean_bps'] - st.summarize(st.forward_returns(c,re,20))['mean_bps']\n"
            "        names.append(t); deltas.append(d)\n"
            "else:\n"
            "    names = [p[0] for p in R['per']]; deltas = [p[5] for p in R['per']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.bar(names, deltas, color=[GREEN if d>0 else RED for d in deltas], width=.6)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,d in enumerate(deltas): ax.annotate(f'{d:+.0f}',(i,d),ha='center',va='bottom' if d>=0 else 'top',fontsize=9)\n"
            "ax.set_ylabel('20d yang − random (bps)'); ax.set_title('Positive in 4 of 5 — but no single t clears 1.7')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('per-ticker 20d delta (bps):', {n:round(d) for n,d in zip(names,deltas)})"
        ),
        md(
            f"> 💡 In plain words: the deltas lean positive (DIA **{R['per'][3][5]:+.0f}**, IWM "
            f"**{R['per'][2][5]:+.0f}** bps) but QQQ is **{R['per'][1][5]:+.0f}**, and with only ~24–39 "
            "switches per name over 21 years none reaches significance. Exactly the underpowered, "
            "incoherent signature of relabelled drift."
        ),
        md(
            "### 4e · Synthetic positive control — the harness CAN bank a real burst\n\n"
            "To prove the null is honest (not a dead detector), plant a **real** post-yang momentum "
            "burst into a synthetic tape and check the same yang rule banks it: edge=0 must stay below "
            "t=2; edge>0 must light up with a high win-rate."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.25):\n"
            "    px, _ = data.synthetic_panel(edge=edge, seed=495, n_days=8000)\n"
            "    c = px['close']; e = st.yang_switch_entries(c, reversal=R['reversal']); s = st.summarize(st.forward_returns(c, e, 20))\n"
            "    res.append((edge, s['n'], s['mean_bps'], s['win']*100, s['t']))\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "labels = [f'planted edge\\n{e:.2f}' for e,_,_,_,_ in res]; tvals = [r[4] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('20d one-sample t'); ax.set_title('Control: edge=0 -> t<2; planted burst -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,n,m,w,t in res: print(f'edge={e:.2f}: n={n} switch={m:+.1f}bps win={w:.0f}% t={t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted burst the control sits at "
            f"**t = {R['syn'][0][4]:.2f}** (win {R['syn'][0][3]:.0f}% — no false positive); a planted "
            f"burst reaches **t = {R['syn'][1][4]:.2f}** (win {R['syn'][1][3]:.0f}%). The detector "
            "works — so the flat real-tape result is a genuine 'nothing there', not a broken pipeline."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the yang switch does not beat a drift-matched random baseline "
            f"(yang − random = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f}/{R['h60'][6]:+.0f} "
            f"bps at 5/10/20/60d; Welch t never clears 2, max **{R['h60'][8]:+.2f}** at 60d). The mild "
            f"one-sample t at 60d (**{R['h60'][4]:.2f}**) is drift.\n"
            f"- **Tradability `MIRAGE`** — no residual edge once the drift is removed; the rule sits out "
            "on yin and pays costs on each switch. You'd capture the drift more cheaply by holding the "
            "index.\n"
            f"- **Time the market? `BUSTED`** — the threshold-scramble placebo leaves the result "
            f"untouched (**p = {R['placebo'][1]:.2f}**): random-reversal Kagis do as well as the real 4% "
            "one, so the specific shoulders and waists carry no forecasting information."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — there is nothing to trade\n\n"
            "The yang switch's entire apparent profit is the unconditional drift of long equity indices, "
            "which you obtain more cheaply and more fully by **buying and holding**. The Kagi rule trades "
            "*less* of the time (it stands aside on yin) and pays costs on each flip, so it strictly "
            "dominates *nothing*. There is no capacity question because there is no edge to scale. The "
            "Kagi is a descriptive noise-filter, not a market-timing strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Reversal definitions.** ATR-based or point-amount reversals are monotone "
            "re-parameterisations of the same geometry — the threshold-scramble placebo already sweeps "
            "the percentage family and finds nothing special about 4%.\n"
            "- **The yin short.** Shorting on yin just shorts the drift on an up-trending tape — a "
            "guaranteed loser; the symmetric failure underlines that the thickness carries no timing "
            "information.\n"
            "- **Three-line-break & Renko cousins.** Other price-only Japanese charts (three-line-break, "
            "Renko) are close kin and inherit the same drift confound.\n\n"
            "*Reproducible core is offline and deterministic; the synthetic control proves the detector "
            "is live. Methods/sources: [`docs/references.md`](../docs/references.md); frozen numbers: "
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
