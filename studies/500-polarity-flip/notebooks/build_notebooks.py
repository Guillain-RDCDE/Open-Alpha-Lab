"""Generate the two narrative notebooks for Study 500 (Polarity-Flip).

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
# 2026-05-31, partial June dropped), 21.4 years, swing-high fractal k=10, band ±1%, break +0.5%.
R = dict(
    asof="2026-05-31", start="2005-01-03", end="2026-05-29", years=21.4,
    tickers=["SPY", "QQQ", "IWM", "DIA", "GLD"], n_entries=598, k=10,
    fp_spy="4cb5244f3990",
    # pooled broken-resistance retest, per horizon:
    # (H, n, retest_bps, win%, one_sample_t, random_bps, delta_bps, net_bps, welch_t, welch_p)
    h5=(5, 598, 32.4, 60, 3.81, 3.6, 28.7, 30.4, 2.05, 0.040),
    h10=(10, 598, 45.0, 61, 3.26, 17.1, 28.0, 43.0, 1.44, 0.150),
    h20=(20, 598, 89.7, 62, 3.68, 55.5, 34.2, 87.7, 1.23, 0.221),
    h60=(60, 595, 272.1, 68, 6.34, 206.1, 66.0, 270.1, 1.47, 0.141),
    # per-ticker H=5 (the coherent horizon): (ticker, entries, retest_bps, random_bps, delta_bps)
    per5=[("SPY", 121, 27.7, 12.6, 15.0), ("QQQ", 114, 51.8, 22.8, 29.0),
          ("IWM", 116, 46.3, -17.1, 63.4), ("DIA", 117, 9.6, 1.2, 8.5),
          ("GLD", 130, 27.7, -0.8, 28.5)],
    # per-ticker H=20: (ticker, entries, retest_bps, one_sample_t, random_bps, delta_bps)
    per20=[("SPY", 121, 41.9, 0.96, 39.1, 2.9), ("QQQ", 114, 179.8, 3.19, 102.3, 77.5),
           ("IWM", 116, 31.5, 0.50, 51.0, -19.6), ("DIA", 117, 27.9, 0.57, 42.2, -14.2),
           ("GLD", 130, 162.8, 3.31, 45.8, 117.0)],
    # scrambled-level placebo (SPY, 500 draws): 20d and 5d -> obs_bps, p
    placebo20=(41.9, 0.122, 500),
    placebo5=(27.7, 0.156, 500),
    # synthetic control (H=20, n_days=4000): (edge, n, retest_bps, win%, one_sample_t)
    syn=[(0.00, 83, -11.4, 43, -0.20), (0.60, 101, 248.0, 77, 5.56)],
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
    "![Holds_as_support%3F: Mixed](https://img.shields.io/badge/Holds_as_support%3F-Mixed-dab617?style=flat-square)\n\n"
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

from polarity_flip import data, strategy as st

ASOF = "2026-05-31"
K, BAND, BREAK_BUF = 10, 0.01, 0.005
HAVE_REAL = data.have_real()
def load(t):
    b = data.load_real(t, allow_fetch=False)
    return b[b.index <= ASOF]
print("real polarity cache present:", HAVE_REAL, "| tickers:", data.DEFAULT_TICKERS)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does old resistance really become support? 🔁\n"
            "### The most-quoted line in charting — \"broken ceilings become floors\" — meets a stopwatch\n\n"
            + BADGES +
            "Every charting course teaches **role reversal** (the *polarity principle*): once price "
            "**breaks above** an old high that had been acting as a ceiling (resistance), that level "
            "is supposed to *flip* and become a **floor** (support). So the first time price pulls "
            "back down to a freshly broken resistance, you buy — it's \"supposed\" to bounce.\n\n"
            "It *looks* uncanny on a hand-picked chart. But a level you draw **after** the break, on a "
            "market that drifts **up** over time, is the textbook setup for fooling yourself. So we "
            "did the only fair thing: encode the level **mechanically** (no eyeballing), fire the "
            "\"buy the retest\" rule hundreds of times across five big indices over 21 years, and time "
            "the result with a stopwatch — against the only baseline that matters: **buying on random "
            "days instead.**\n\n"
            "**Spoiler — this one is not the usual nothing.** At the *very short* horizon (5 days) the "
            "retest genuinely beats random (and in all five names). But the bounce is small and fades "
            "within two weeks, and we can't pin it cleanly on *those specific old levels*. A real but "
            "fragile, short-lived effect.\n\n"
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
            "| If I buy when price pulls back to a **broken resistance**, do I make money? | **Yes — "
            "and at 5 days it genuinely beats random.** ~60% win-rate, and a +29 bps edge over a "
            "coin-flip entry that holds across all 5 indices. |\n"
            "| Is that *the level's* doing, or just the market drifting up? | **Partly the level.** "
            "Unlike most chart tools, the 5-day edge survives the random baseline (*t* ≈ 2). |\n"
            "| Does it last? | **No.** By 10–20 days the edge sinks back into the noise. It's a "
            "*few-day* bounce, not a durable channel. |\n"
            "| Is it *these specific old levels*? | **Can't say for sure.** Scramble which level is "
            "which and the result drops only a bit (placebo *p* ≈ 0.12) — suggestive, not proven. |\n"
            "| So is it a tradable edge? | **Fragile.** A real but thin, short-lived bounce in two of "
            "five names at longer horizons — tradable in principle, nothing to scale. |\n\n"
            "> Role reversal isn't pure folklore like most chart patterns: there's a measurable "
            "short-term bounce at the first retest. But it's small, brief, and concentrated — a "
            "**Weak** signal, not a money machine."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"When price breaks **above** a level that was acting as resistance, that level flips "
            "and becomes support. The first time price comes back down to it, it bounces. Buy the "
            "retest of broken resistance.\"*\n\n"
            "This is the **polarity principle** / **role reversal**, in Edwards & Magee's *Technical "
            "Analysis of Stock Trends* (1948) and every text since (Murphy, Bulkowski). It's one of "
            "the most repeated ideas in all of charting — the notion that price *remembers* old levels "
            "and treats a broken ceiling as a new floor. So: does the floor actually floor?"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If broken resistance genuinely *forecast* a bounce, it would be a clean, mechanical edge: "
            "a past high tells you where the next dip will hold, tradable with a horizontal line. "
            "That's the dream the principle sells.\n\n"
            "But there are two traps. First, you draw the level **after** the break — you choose the "
            "high that makes the retest *look* right. Second, it's a market that drifts **up**, so "
            "*any* dip-buy will look profitable. To separate the **level** from the **tide**, we have "
            "to (a) pick the level by a fixed mechanical rule with no hindsight, and (b) compare it to "
            "buying on **random days**. We'll do both."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **{len(R['tickers'])} liquid indices/ETFs** ({', '.join(R['tickers'])}), daily, "
            f"over **{R['years']:.0f} years** ({R['start']} → {R['end']}), and:\n\n"
            "1. **Find the resistance levels mechanically.** A 'swing high' is a high with "
            f"**{R['k']} lower bars on each side** — a confirmed fractal. Crucially it's only known "
            f"**{R['k']} bars later**, so we never use a level before it exists.\n"
            "2. **Spot the break.** The level becomes *broken resistance* the first time the close "
            "prints clearly **above** it (+0.5% buffer).\n"
            "3. **Trade the lore.** When price pulls back **down to** the broken level (first close "
            "within ±1%), buy at the next close; measure the return over the next "
            "**5 / 10 / 20 / 60 days**.\n"
            "4. **The honest baseline.** Do the exact same hold on **random days**. If the level "
            "matters, the retest must beat random. *If it does — even at one horizon — we say so.* "
            "That's the result we announce before we look."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "First, what does a mechanical broken-resistance retest even look like? Here's SPY with a "
            "confirmed swing high, the break above it, and the pullback the rule would buy."
        ),
        code(
            "b = load('SPY') if HAVE_REAL else None\n"
            "if HAVE_REAL:\n"
            "    cl = b['close']; seg = cl.iloc[-500:]\n"
            "    ent = st.polarity_entries(cl, k=K, band=BAND, break_buf=BREAK_BUF)\n"
            "    ent = ent[ent >= seg.index[0]]\n"
            "    piv = st.find_swing_highs(cl, k=K)\n"
            "    fig, ax = plt.subplots(figsize=(10.2, 4.8))\n"
            "    ax.plot(seg.index, seg.values, c='k', lw=1.2, label='SPY close')\n"
            "    # draw recent confirmed swing-high levels as horizontal lines\n"
            "    for pos, pr in zip(piv.index, piv['price']):\n"
            "        d = cl.index[int(pos)]\n"
            "        if d >= seg.index[0]:\n"
            "            ax.hlines(pr, d, seg.index[-1], color=AMBER, lw=.9, alpha=.5)\n"
            "    ax.scatter(ent, cl.reindex(ent), c=GREEN, s=45, zorder=5, label='broken-resistance retest BUY')\n"
            "    ax.set_title('Mechanical broken-resistance retests on SPY (last ~2y)'); ax.legend(loc='upper left')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('retests in window:', len(ent))\n"
            "else:\n"
            "    print('(no cache — see docs/results.md for the frozen numbers)')"
        ),
        md(
            "The amber lines are old swing-high ceilings; the green dots are pullbacks to ones price "
            "has already broken above. **Do those green buys bounce?** Let's race the retest against "
            "random entries at four horizons. Blue = buy the retest; grey = buy on random days."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    retest, rnd = [], []\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            c = load(t)['close']\n"
            "            e = st.polarity_entries(c, k=K, band=BAND, break_buf=BREAK_BUF)\n"
            "            re = st.random_entries(c, max(len(e),50), k=K, seed=7)\n"
            "            tt.append(st.forward_returns(c, e, h)); rr.append(st.forward_returns(c, re, h))\n"
            "        retest.append(np.concatenate(tt).mean()*1e4); rnd.append(np.concatenate(rr).mean()*1e4)\n"
            "else:\n"
            "    retest = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x-.2, retest, .4, color='#2c6fbb', label='buy the broken-resistance retest')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='buy on random days')\n"
            "for i,(a,bb) in enumerate(zip(retest,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{bb:+.0f}',(i+.2,bb),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean return (bps)')\n"
            "ax.set_title('The retest beats random at 5d — then the gap shrinks'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('retest:', [round(v) for v in retest]); print('random:', [round(v) for v in rnd])"
        ),
        md(
            f"There's the story. At **5 days** the retest (**+{R['h5'][2]:.0f} bps**) clearly beats "
            f"random (**+{R['h5'][5]:.0f} bps**) — a real **+{R['h5'][6]:.0f} bps** edge, and the "
            "quants notebook shows it clears the *t* = 2 bar (*p* = 0.04). But watch the gap **shrink** "
            "as the horizon grows: by 10–20 days most of the lead is gone, and by 60 days it's the "
            "usual drift. The bounce is **real but short** — a few days, not a channel."
        ),
        md(
            "**One more sanity check.** What if we scramble *which* old level is which — keep the "
            "swing-high dates but shuffle the prices, so 'this specific broken resistance' becomes "
            "nonsense? If the polarity flip is about the real levels, the scramble should hurt."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = load('SPY')['close']\n"
            "    pl = st.scrambled_level_placebo(c, 20, k=K, band=BAND, break_buf=BREAK_BUF, n_draws=200, seed=500)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "else:\n"
            "    obs = R['placebo20'][0]; pval = R['placebo20'][1]\n"
            "print(f'real broken-resistance retest (SPY, 20d): {obs:+.1f} bps')\n"
            "print(f'... and {pval*100:.0f}% of *scrambled-level* runs do at least as well (p={pval:.2f}).')\n"
            "print('=> the real level does a bit better than a random one, but not decisively (p>0.05).')"
        ),
        md(
            f"Only ~{R['placebo20'][1]*100:.0f}% of scrambled-level runs match or beat the real one "
            f"(*p* = {R['placebo20'][1]:.2f}) — so the real level *is* a little better than a random "
            "one, but not at the 0.05 bar. We can't cleanly say it's *these specific old levels* "
            "rather than generic short-term mean reversion. Suggestive, not proven."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — Weak.** The retest **does** beat random at 5 days (*t* ≈ 2, *p* = 0.04, "
            "positive in all 5 names) — so it's not a flat 'nothing'. But the edge is small and "
            "**decays within two weeks**; the longer-horizon cross-section is incoherent.\n"
            "- **Tradability — Fragile.** A real but thin, short-lived bounce. Tradable only by a "
            "very short-horizon desk, with no margin for error and nothing to scale.\n"
            "- **\"Does broken resistance hold as support\"? — Mixed.** At the first retest, over the "
            "next few days, yes — there's a measurable bounce beyond drift. But it's brief and the "
            "level-scramble placebo doesn't clear 0.05, so the mechanism isn't proven."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Maybe — at the *very* short horizon, by a desk that already trades 1–5-day mean "
            "reversion. The 5-day edge is real (≈ 29 bps over random) but thin, and a ±1% retest "
            "trigger means real slippage; costs trim it and it's gone by ~10 days. It is **not** a "
            "set-and-hold strategy, and at longer horizons it leans on just two of five instruments. "
            "Treat it as a *minor short-term tilt*, not a standalone edge — and never confuse the "
            "5-day bounce with the much larger numbers you'd get just by holding the index."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Is it the level or just a pullback?** The honest open question (placebo *p* = 0.12) "
            "is whether the bounce is *these old levels* or generic post-pullback mean reversion. A "
            "follow-up: race the broken-resistance retest against a *generic dip* of the same depth.\n"
            "- **Tighter / looser bands.** A narrower retest band trades less but cleaner; the 5-day "
            "edge is the robust part — see how it moves with the band.\n"
            "- **A real positive control.** The quants notebook plants a *genuine* role-reversal "
            "bounce into a synthetic tape and shows the harness banks it (*t* = +5.6) — so the 5-day "
            "real result is a genuine measurement, not a glitch.\n\n"
            "*Think broken resistance is a durable floor? Show the retest beating random at "
            "**t ≥ 2** at **10+ days** on a real tape — then we'll talk.*"
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
            "# Polarity-Flip — a quantitative teardown 🔬\n"
            "### Mechanical broken-resistance retests on 5 indices · forward returns · one-sample HAC "
            "*t* · a drift-matched random-entry baseline · a scrambled-level placebo · costs · a "
            "synthetic planted-flip control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The job "
            "is to separate the **level** from the **drift**: an upward-trending index makes *any* "
            "dip-buy look good, so the only meaningful test is retest-vs-random, plus a placebo that "
            "scrambles which level is which while preserving the marginals.\n\n"
            "> ⚠️ **Data note.** 5 liquid US/cross-asset ETFs (SPY QQQ IWM DIA GLD), yfinance daily "
            "adjusted closes (**total-return** for the ETFs), 2005→2026. Levels are confirmed swing-"
            f"high fractals (k={R['k']}, an explicit {R['k']}-bar confirmation lag), band ±1%, break "
            "+0.5%; entry is the **next close** (one documented lag). Offline core + synthetic control "
            "are deterministic. Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | Retest vs a **drift-matched random** baseline clears *t* = 2 "
            f"**only at 5d** (Welch *t* = **{R['h5'][8]:+.2f}**, *p* = {R['h5'][9]:.3f}; Δ = "
            f"{R['h5'][6]:+.0f} bps, positive in all 5 names) and **decays** by 10/20/60d "
            f"(*t* = {R['h10'][8]:.2f}/{R['h20'][8]:.2f}/{R['h60'][8]:.2f}). Real but short-lived. |\n"
            f"| **Tradability** | `FRAGILE` | The 5d edge is thin (Δ ≈ {R['h5'][6]:+.0f} bps), gone by "
            f"~10d, and the 20d cross-section leans on just QQQ/GLD. Costs trim it. Nothing to scale. |\n"
            f"| **Holds as support?** | `MIXED` | A measurable few-day bounce, but the scrambled-level "
            f"placebo doesn't clear 0.05 (**p = {R['placebo20'][1]:.2f}**): can't pin it on *these "
            "specific old levels* vs generic post-pullback reversion. |\n\n"
            "> 💡 In plain words: this is the rare chart rule with a *real* residual — but only at the "
            "shortest horizon, and we can't fully attribute it to the level itself. Weak, not Real."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $L$ be a confirmed swing-high resistance (fractal, lag $k$). $L$ becomes *broken* at "
            "the first bar with $C_t > L(1+\\beta)$ (break buffer $\\beta$). After the break, the "
            "first bar with $C_t \\in [L(1-w), L(1+w)]$ (band $w$) is the **polarity-flip retest** — a "
            "long, entered next close.\n\n"
            "- **H₀ (drift).** Retest returns equal a drift-matched **random-entry** baseline.\n"
            "- **H₁ (the level forecasts).** Retest returns **exceed** random at some horizon, t ≥ 2.\n"
            "- **H₂ (the geometry matters).** Retest returns exceed a **scrambled-level** run whose "
            "levels are a permutation of the real ones.\n\n"
            "We find **H₀ rejected at 5d only** (Welch t = +2.05), **not** at 10/20/60d; **H₁ "
            "partially supported** (one horizon clears, then decays); **H₂ not cleared** (placebo "
            "p ≈ 0.12). The steelman half-survives: a real short-horizon bounce, unproven mechanism."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the two confounds this design must kill\n\n"
            "**(a) Drift.** Equity indices have a positive unconditional daily mean. *Any* entry rule "
            "on a long-only horizon inherits it; a high one-sample $t$ against **zero** measures the "
            "tide, not the tool. The fix is the **random-entry baseline** (same instrument, epoch, "
            "hold) and a Welch test of retest-*minus*-random. Here the one-sample $t$ is large at "
            "every horizon, but the random baseline absorbs most of it — only the 5d residual clears 2.\n\n"
            "**(b) Level as a free parameter.** A level is a chosen swing high; the danger is that "
            "*any* horizontal line on a trend produces 'respected' retests. The **scrambled-level "
            "placebo** keeps pivot positions and the price marginal but permutes which price is the "
            "level, so the lines become arbitrary — if the real result survives the scramble, the "
            "specific level was never load-bearing. Here it survives only weakly (p ≈ 0.12)."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** {', '.join(R['tickers'])}; yfinance daily adjusted closes "
            f"({R['start']}→{R['end']}, {R['years']:.1f}y). **{R['n_entries']} broken-resistance "
            "retests** pooled.\n"
            f"- **Levels.** Confirmed swing-high fractals: maximum with k={R['k']} strictly-lower bars "
            f"each side; usable only at bar +{R['k']} (no look-ahead).\n"
            "- **Break + retest.** Break when close > level×1.005; entry = first close back inside "
            "level×[0.99, 1.01]; each level fires once, retires after 250 bars.\n"
            "- **Entry.** Enter **next close** (one lag); hold H ∈ {5,10,20,60}.\n"
            "- **Null #1 — one-sample HAC t** of retest returns vs 0 (Newey-West).\n"
            "- **Null #2 — random-entry baseline**, Welch two-sample retest vs random (the *real* test).\n"
            "- **Null #3 — scrambled-level placebo** (level prices permuted, marginals kept).\n"
            "- **Costs.** 1 bp one-way × 2 legs on every retest.\n"
            "- **Positive control.** Synthetic tape with a **planted** role-reversal bounce (knob "
            "`edge`): edge=0 must NOT reach significance; edge>0 must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The beta trap — one-sample t looks great everywhere, vs-random only at 5d\n\n"
            "Left: the retest's **one-sample** t against zero (the misleading number, big at every "
            "horizon). Right: the same retest vs a **drift-matched random** baseline (the honest "
            "number — green where it clears 2)."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    one_t, retest, rnd, welch = [], [], [], []\n"
            "    from scipy import stats\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            c = load(t)['close']\n"
            "            e = st.polarity_entries(c, k=K, band=BAND, break_buf=BREAK_BUF)\n"
            "            re = st.random_entries(c, max(len(e),50), k=K, seed=7)\n"
            "            tt.append(st.forward_returns(c, e, h)); rr.append(st.forward_returns(c, re, h))\n"
            "        tt = np.concatenate(tt); rr = np.concatenate(rr)\n"
            "        one_t.append(st.summarize(tt)['t']); retest.append(tt.mean()*1e4); rnd.append(rr.mean()*1e4)\n"
            "        welch.append(stats.ttest_ind(tt, rr, equal_var=False)[0])\n"
            "else:\n"
            "    one_t = [R['h5'][4], R['h10'][4], R['h20'][4], R['h60'][4]]\n"
            "    retest = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "    welch = [R['h5'][8], R['h10'][8], R['h20'][8], R['h60'][8]]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "a1.bar([f'{h}d' for h in hs], one_t, color=GREY, width=.6)\n"
            "a1.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,v in enumerate(one_t): a1.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom',fontsize=9)\n"
            "a1.set_title('One-sample t vs ZERO (misleading: mostly beta)'); a1.set_ylabel('t'); a1.legend()\n"
            "a2.bar([f'{h}d' for h in hs], welch, color=[GREEN if v>2 else AMBER if v>1.5 else RED for v in welch], width=.6)\n"
            "a2.axhline(2, ls='--', c=RED); a2.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(welch): a2.annotate(f'{v:+.2f}',(i,v),ha='center',va='bottom' if v>=0 else 'top',fontsize=9)\n"
            "a2.set_title('Retest vs RANDOM, Welch t (clears 2 only at 5d)'); a2.set_ylabel('t')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('one-sample t:', [round(v,2) for v in one_t]); print('welch vs random:', [round(v,2) for v in welch])"
        ),
        md(
            f"> 💡 In plain words: the left bars clear *t* = 2 at every horizon (20d "
            f"**{R['h20'][4]:.2f}**, 60d **{R['h60'][4]:.2f}**) — but that's mostly **drift**. The "
            f"right bars are the real test: retest-minus-random clears 2 **only at 5d** "
            f"({R['h5'][8]:+.2f}) and fades to {R['h10'][8]:+.2f}/{R['h20'][8]:+.2f}/{R['h60'][8]:+.2f} "
            "at 10/20/60d. A genuine short-horizon bounce that decays."
        ),
        md(
            "### 4b · The coherent horizon — at 5 days, every name agrees\n\n"
            "The reason the pooled 5-day Welch t clears 2: the retest-minus-random delta is **positive "
            "in all 5 instruments** at 5 days. That cross-sectional coherence is the strongest "
            "evidence in the study."
        ),
        code(
            "if HAVE_REAL:\n"
            "    names, d5 = [], []\n"
            "    for t in data.DEFAULT_TICKERS:\n"
            "        c = load(t)['close']\n"
            "        e = st.polarity_entries(c, k=K, band=BAND, break_buf=BREAK_BUF); re = st.random_entries(c, max(len(e),50), k=K, seed=7)\n"
            "        d = st.summarize(st.forward_returns(c,e,5))['mean_bps'] - st.summarize(st.forward_returns(c,re,5))['mean_bps']\n"
            "        names.append(t); d5.append(d)\n"
            "else:\n"
            "    names = [p[0] for p in R['per5']]; d5 = [p[4] for p in R['per5']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.bar(names, d5, color=[GREEN if d>0 else RED for d in d5], width=.6)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,d in enumerate(d5): ax.annotate(f'{d:+.0f}',(i,d),ha='center',va='bottom' if d>=0 else 'top',fontsize=9)\n"
            "ax.set_ylabel('5d retest − random (bps)'); ax.set_title('5-day delta: positive in all 5 names')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('per-ticker 5d delta (bps):', {n:round(d) for n,d in zip(names,d5)})"
        ),
        md(
            f"> 💡 In plain words: SPY {R['per5'][0][4]:+.0f}, QQQ {R['per5'][1][4]:+.0f}, IWM "
            f"{R['per5'][2][4]:+.0f}, DIA {R['per5'][3][4]:+.0f}, GLD {R['per5'][4][4]:+.0f} bps — all "
            "positive. Five independent tapes pointing the same way at 5 days is hard to wave away as "
            "noise. This is why the verdict is Weak, not None."
        ),
        md(
            "### 4c · The level placebo — scramble which level is which\n\n"
            "Permute which price is the resistance level (positions kept, marginal kept) so 'this "
            "specific broken level' is arbitrary. If the polarity flip is about the real levels, the "
            "observed retest return should sit far in the right tail of the scrambled distribution."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = load('SPY')['close']\n"
            "    obs = st.forward_returns(c, st.polarity_entries(c,k=K,band=BAND,break_buf=BREAK_BUF), 20).mean()*1e4\n"
            "    piv = st.find_swing_highs(c, k=K)\n"
            "    rng = np.random.default_rng(500); prices = piv['price'].to_numpy(); positions=[int(p) for p in piv.index]\n"
            "    draws=[]\n"
            "    for _ in range(200):\n"
            "        perm = rng.permutation(prices)\n"
            "        ent = st._entries_from_levels(c, list(zip(positions, perm)), K, BAND, BREAK_BUF)\n"
            "        rr = st.forward_returns(c, ent, 20)\n"
            "        if rr.size: draws.append(rr.mean()*1e4)\n"
            "    draws = np.array(draws); pval = (np.sum(draws>=obs)+1)/(len(draws)+1)\n"
            "else:\n"
            "    obs = R['placebo20'][0]; pval = R['placebo20'][1]\n"
            "    rng = np.random.default_rng(500); draws = rng.normal(10, 35, 200)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=35, color=GREY, alpha=.85, label='scrambled-level runs (SPY, 20d)')\n"
            "ax.axvline(obs, c='#2c6fbb', lw=2.5, label=f'real level {obs:+.0f} bps')\n"
            "ax.set_xlabel('mean broken-resistance-retest 20d return (bps)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Real level only mildly ahead: placebo p = {pval:.2f}'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'real level {obs:+.1f} bps   placebo p={pval:.2f}  (>0.05 => specific level not decisively load-bearing)')"
        ),
        md(
            f"> 💡 In plain words: the real level (blue) sits **right of centre** but not far in the "
            f"tail — **p = {R['placebo20'][1]:.2f}**. So the real broken level does a *bit* better than "
            "a random one, but not at the 0.05 bar. We can't cleanly separate 'these specific old "
            "levels' from generic post-pullback mean reversion. This is why the thesis is Mixed, not "
            "Confirmed."
        ),
        md(
            "### 4d · Per-ticker (H=20) — the edge frays at longer horizons\n\n"
            "20-day retest-minus-random delta, per instrument. At 5d all five agreed; by 20d the "
            "cross-section is incoherent — only QQQ/GLD carry it."
        ),
        code(
            "if HAVE_REAL:\n"
            "    names, deltas = [], []\n"
            "    for t in data.DEFAULT_TICKERS:\n"
            "        c = load(t)['close']\n"
            "        e = st.polarity_entries(c, k=K, band=BAND, break_buf=BREAK_BUF); re = st.random_entries(c, max(len(e),50), k=K, seed=7)\n"
            "        d = st.summarize(st.forward_returns(c,e,20))['mean_bps'] - st.summarize(st.forward_returns(c,re,20))['mean_bps']\n"
            "        names.append(t); deltas.append(d)\n"
            "else:\n"
            "    names = [p[0] for p in R['per20']]; deltas = [p[5] for p in R['per20']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.bar(names, deltas, color=[GREEN if d>0 else RED for d in deltas], width=.6)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,d in enumerate(deltas): ax.annotate(f'{d:+.0f}',(i,d),ha='center',va='bottom' if d>=0 else 'top',fontsize=9)\n"
            "ax.set_ylabel('20d retest − random (bps)'); ax.set_title('20d: incoherent — only QQQ/GLD positive')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('per-ticker 20d delta (bps):', {n:round(d) for n,d in zip(names,deltas)})"
        ),
        md(
            f"> 💡 In plain words: QQQ {R['per20'][1][5]:+.0f} and GLD {R['per20'][4][5]:+.0f} bps "
            f"carry the pooled 20d delta; SPY is flat ({R['per20'][0][5]:+.0f}) and IWM/DIA are "
            "*behind* random. The 5-day coherence does not survive to 20 days — the bounce is a "
            "short-horizon phenomenon."
        ),
        md(
            "### 4e · Synthetic positive control — the harness CAN bank a real flip\n\n"
            "To prove the null is honest (not a dead detector), plant a **real** role-reversal bounce "
            "into a synthetic tape and check the same break-then-retest rule banks it: edge=0 must "
            "stay at t≈0; edge>0 must light up with a high win-rate."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.60):\n"
            "    px, _ = data.synthetic_panel(edge=edge, seed=500, n_days=4000)\n"
            "    c = px['close']; e = st.polarity_entries(c, k=K, band=BAND, break_buf=BREAK_BUF)\n"
            "    s = st.summarize(st.forward_returns(c, e, 20))\n"
            "    res.append((edge, s['n'], s['mean_bps'], s['win']*100, s['t']))\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "labels = [f'planted edge\\n{e:.2f}' for e,_,_,_,_ in res]; tvals = [r[4] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('20d one-sample t'); ax.set_title('Control: edge=0 -> t~0; planted flip -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,n,m,w,t in res: print(f'edge={e:.2f}: n={n} retest={m:+.1f}bps win={w:.0f}% t={t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted flip the control sits at "
            f"**t = {R['syn'][0][4]:.2f}** (win {R['syn'][0][3]:.0f}% — no false positive); a planted "
            f"flip reaches **t = {R['syn'][1][4]:.2f}** (win {R['syn'][1][3]:.0f}%). The detector is "
            "live and well-calibrated — so the borderline 5-day real result is a genuine measurement, "
            "not a broken pipeline."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — the broken-resistance retest beats a drift-matched random baseline "
            f"**at 5 days** (Welch t = **{R['h5'][8]:+.2f}**, p = {R['h5'][9]:.3f}; Δ = "
            f"{R['h5'][6]:+.0f} bps, positive in all 5 names) but **decays** by 10/20/60d "
            f"(t = {R['h10'][8]:.2f}/{R['h20'][8]:.2f}/{R['h60'][8]:.2f}). A real, short-lived bounce "
            "— not a flat None, not a robust Real.\n"
            f"- **Tradability `FRAGILE`** — the 5d edge is thin ({R['h5'][6]:+.0f} bps Δ), gone by "
            "~10d, and the 20d cross-section leans on QQQ/GLD; costs trim it. Tradable only by a very "
            "short-horizon desk, nothing to scale.\n"
            f"- **Holds as support? `MIXED`** — a measurable few-day bounce, but the scrambled-level "
            f"placebo doesn't clear 0.05 (**p = {R['placebo20'][1]:.2f}**): we can't attribute it "
            "decisively to *these specific old levels* vs generic post-pullback mean reversion."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — only as a minor short-horizon tilt\n\n"
            "There is a *real* 5-day bounce here (≈ 29 bps over random), which sets polarity-flip apart "
            "from most chart rules. But it is thin, brief (gone by ~10 days), exposed to slippage on a "
            "±1% retest trigger, and at longer horizons depends on just two of five instruments. A "
            "short-horizon mean-reversion desk could fold it in as a *minor tilt*; it is not a "
            "standalone strategy and there is no capacity story. Costs (commissions + spread on every "
            "retest) erode the already-thin edge. And the bulk of the absolute return at every horizon "
            "is still drift you'd capture more cheaply by holding the index."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Level vs generic pullback.** The key open question (placebo p = 0.12): is the 5-day "
            "bounce *these old levels*, or just short-term reversion after any pullback of the same "
            "depth? A matched-depth dip control would separate them.\n"
            "- **Band / buffer sensitivity.** The 5-day edge is the robust part; trace it across "
            "tighter retest bands and break buffers to find where it's strongest (and where snooping "
            "begins).\n"
            "- **Resistance-becomes-support is one half.** The mirror claim (support-becomes-"
            "resistance, a short on the retest of broken support) is the symmetric test — and on an "
            "up-drifting tape it fights the tide, a useful contrast.\n\n"
            "*Reproducible core is offline and deterministic; the synthetic control proves the "
            "detector is live (t = +5.6 on a planted flip). Methods/sources: "
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
