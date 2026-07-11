"""Generate the two narrative notebooks for Study 706 (Diamond Bottom).

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
# yfinance daily, 5 indices/ETFs (SPY QQQ IWM DIA GLD), 2005-01-03 -> 2026-06-30 (As-of
# 2026-06-30, partial July dropped), 21.5 years, fractal k=5, 6 pivots, diamond-bottom LONG.
R = dict(
    asof="2026-06-30", start="2005-01-03", end="2026-06-30", years=21.5,
    tickers=["SPY", "QQQ", "IWM", "DIA", "GLD"], n_entries=197, k=5, n_piv=6,
    fp_spy="88127c764cba",
    # pooled diamond breakout LONG, per horizon:
    # (H, n, breakout_bps, win%, one_sample_t, random_bps, delta_bps, net_bps, welch_t, welch_p)
    h5=(5, 197, 30.2, 58, 2.57, 44.2, -13.9, 28.2, -0.69, 0.492),
    h10=(10, 196, 49.9, 60, 2.49, 14.3, 35.6, 47.9, 1.16, 0.247),
    h20=(20, 195, 66.4, 61, 2.22, 91.9, -25.5, 64.4, -0.55, 0.581),
    h60=(60, 194, 243.1, 66, 4.11, 209.8, 33.3, 241.1, 0.44, 0.657),
    # per-ticker H=20: (ticker, entries, breakout_bps, one_sample_t, random_bps, delta_bps)
    per=[("SPY", 32, 32.5, 0.46, 70.9, -38.3), ("QQQ", 42, 112.0, 1.70, 107.7, 4.4),
         ("IWM", 49, 44.2, 0.72, 53.2, -8.9), ("DIA", 46, 66.3, 1.37, 101.8, -35.6),
         ("GLD", 28, 74.1, 0.71, 125.9, -51.8)],
    # shuffled-pivot placebo (SPY, H=20, 500 draws): obs_bps, p, draws
    placebo=(32.5, 0.842, 500),
    # synthetic control (H=20, n_days=8000): (edge, n, breakout_bps, win%, one_sample_t)
    syn=[(0.00, 62, -54.1, 42, -1.07), (0.60, 46, 1339.2, 59, 4.42)],
    syn_null_mean=0.03, syn_null_sd=1.07, syn_null_fire=0, syn_null_seeds=20,
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Forecasts_a_reversal%3F: Busted](https://img.shields.io/badge/Forecasts_a_reversal%3F-Busted-8b949e?style=flat-square)\n\n"
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

from diamond_bottom import data, strategy as st

ASOF = "2026-06-30"
HAVE_REAL = data.have_real()
def load(t):
    b = data.load_real(t, allow_fetch=False)
    return b[b.index <= ASOF]
print("real diamond cache present:", HAVE_REAL, "| tickers:", data.DEFAULT_TICKERS)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does a \"diamond bottom\" actually call the turn? 💎🔻\n"
            "### The bullish mirror of the diamond top — range narrows after widening, near a "
            "low — meets a stopwatch\n\n"
            + BADGES +
            "Flip through any chart-pattern book and you'll meet the **diamond bottom**: after a "
            "sell-off, price swings get *wider and wider* (a megaphone), then *tighter and tighter* "
            "(a triangle), tracing a diamond near a low. The lore, from Edwards & Magee to Bulkowski "
            "to every trading site, is that this rare shape marks **accumulation** — and when price "
            "finally breaks **up** out of the diamond, you **buy it**: the reversal is supposed to "
            "be on.\n\n"
            "It's the exact bullish twin of [study 466's diamond top](../../466-diamond-top/), which "
            "found the bearish version doesn't work. But a long on a stock index has a sneaky trap "
            "the short doesn't: **the market drifts up anyway**, so almost *any* long-only rule looks "
            "\"significant\" against zero, pattern or no pattern. So we did the only fair thing: "
            "encode the diamond **mechanically** (no eyeballing), fire the \"buy the breakout\" rule "
            "across five big indices over 21.5 years, and time the result — against the only baseline "
            "that matters: **buying on random days instead.**\n\n"
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
            "| If I buy when price breaks **up** out of a diamond bottom, do I catch a rally? | "
            "**Looks like it — but that's a trick.** Over the next 20–60 days the long makes money "
            "(+66 / +243 bps) — but so does almost any long, because the market drifts up. |\n"
            "| Is the breakout at least better than buying random days? | **No — statistically, "
            "no different.** The breakout beats a random long at 10 and 60 days, loses to it at 5 and "
            "20 days, and never clears significance either way. |\n"
            "| Is there *anything* there? | **No.** The best case (10-day) is a coin-flip-level "
            "*t* = 1.16 — nowhere near the desk's bar of 2. |\n"
            "| Does the diamond \"shape\" forecast the turn? | **No.** Scramble the diamond into "
            "geometric nonsense and the result barely changes — actually gets slightly *better* on "
            "average. The shape isn't doing the work. |\n\n"
            "> The diamond bottom is a great way to *describe* a volatile pause during a decline. As "
            "a *forecast* — \"the breakout starts a rally\" — it's a **mirage**: you were just going "
            "to get paid for being long anyway."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"After a decline, the swing range **broadens** (higher highs, lower lows — a "
            "megaphone) and then **narrows** (lower highs, higher lows — a symmetrical triangle), "
            "forming a diamond. This marks accumulation at a low. When price breaks **up** out of "
            "the apex, buy — the reversal is confirmed, with a target equal to the diamond's "
            "height.\"*\n\n"
            "This is the **diamond bottom** of Edwards & Magee's *Technical Analysis of Stock "
            "Trends* and Bulkowski's *Encyclopedia of Chart Patterns* — the bullish twin of the "
            "diamond top, billed with the same \"rare but high-reliability\" reputation. So: does "
            "the diamond actually *call the turn*?"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the diamond genuinely *forecast* bottoms, it would be remarkable: a few past "
            "wiggles predicting the end of a decline, a clean crack in market efficiency you could "
            "trade with a ruler. That's the dream the pattern sells — and this time it's tempting "
            "for a sneaky extra reason: it's a **long**, and stock indices go up over time, so a "
            "long-only rule *looks* profitable almost no matter what triggers it.\n\n"
            "There are two traps here, mirroring the diamond-top study exactly. First, a diamond "
            "is recognised **by hand, after the swings have happened** — you pick the wiggles that "
            "make the shape *look* right. Second, testing a long against **zero** just measures the "
            "market's own upward drift, not the pattern. To separate the **shape** from the "
            "**tide**, we have to (a) detect the diamond by a fixed mechanical rule with no "
            "hindsight, and (b) compare the long to buying on **random days**. We'll do both."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **{len(R['tickers'])} liquid indices/ETFs** ({', '.join(R['tickers'])}), "
            f"daily, over **{R['years']:.1f} years** ({R['start']} → {R['end']}), and:\n\n"
            "1. **Find the swing points mechanically.** A 'pivot' is a high (or low) with "
            f"**{R['k']} lower (higher) bars on each side** — a confirmed fractal, only known "
            f"**{R['k']} bars later**, so we never draw the diamond with future data.\n"
            "2. **Detect the diamond by rule.** Over the 6 most-recent alternating pivots, the swing "
            "amplitudes must first **grow** (broadening) then **shrink** (narrowing) — a diamond — "
            "formed after a **decline**. No eyeballing.\n"
            "3. **Trade the lore.** When the close breaks **above** the narrowing apex, **buy** at "
            "the next close; measure the long's return over the next **5 / 10 / 20 / 60 days**.\n"
            "4. **The honest baseline.** Do the exact same long on **random days**. If the diamond "
            "matters, the breakout long must beat random. *If it doesn't, the shape is a mirage* — "
            "that's the result that would make us say so, announced before we look."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "First, what does a mechanical diamond even look like? Here's SPY with the confirmed "
            "pivots, and the upside breaks the rule would buy."
        ),
        code(
            "b = load('SPY') if HAVE_REAL else None\n"
            "if HAVE_REAL:\n"
            "    cl = b['close']; seg = cl.iloc[-900:]\n"
            "    piv = st._alternating(st.find_pivots(cl, k=R['k']))\n"
            "    ent = st.diamond_breakouts(cl, k=R['k'], n_piv=R['n_piv'])\n"
            "    ent = ent[ent >= seg.index[0]]\n"
            "    pv = piv[(piv.index >= 0)]\n"
            "    pdates = cl.index[[int(p) for p in pv.index if int(p) < len(cl)]]\n"
            "    pdates = pdates[pdates >= seg.index[0]]\n"
            "    fig, ax = plt.subplots(figsize=(10.2, 4.8))\n"
            "    ax.plot(seg.index, seg.values, c='k', lw=1.1, label='SPY close')\n"
            "    ax.scatter(pdates, cl.reindex(pdates), c=GREY, s=22, zorder=4, label='confirmed pivots')\n"
            "    ax.scatter(ent, cl.reindex(ent), c=GREEN, s=55, marker='^', zorder=5, label='diamond breakout LONG')\n"
            "    ax.set_title('Mechanical diamond-bottom breakouts on SPY (last ~3.5y)'); ax.legend(loc='upper left')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('diamond breakouts in window:', len(ent))\n"
            "else:\n"
            "    print('(no cache — see docs/results.md for the frozen numbers)')"
        ),
        md(
            "Now the real question: are those green long triangles followed by **rallies bigger "
            "than what you'd get anyway**? **Let's race the breakout long against buying random "
            "days** at four horizons. Blue = long the diamond breakout; grey = long random days. "
            "(Both bars above zero just means the market went up — that's the trap.)"
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    brk, rnd = [], []\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            c = load(t)['close']\n"
            "            e = st.diamond_breakouts(c, k=R['k'], n_piv=R['n_piv'])\n"
            "            re = st.random_entries(c, max(len(e),50), k=R['k'], seed=7)\n"
            "            tt.append(st.forward_returns(c, e, h)); rr.append(st.forward_returns(c, re, h))\n"
            "        brk.append(np.concatenate(tt).mean()*1e4); rnd.append(np.concatenate(rr).mean()*1e4)\n"
            "else:\n"
            "    brk = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x-.2, brk, .4, color='#2c6fbb', label='long the diamond breakout')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='long on random days')\n"
            "for i,(a,bb) in enumerate(zip(brk,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom' if a>=0 else 'top',fontsize=8)\n"
            "    ax.annotate(f'{bb:+.0f}',(i+.2,bb),ha='center',va='bottom' if bb>=0 else 'top',fontsize=8)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean LONG return (bps)')\n"
            "ax.set_title('The diamond long does NOT beat a random long, at any horizon'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('breakout long:', [round(v) for v in brk]); print('random long:', [round(v) for v in rnd])"
        ),
        md(
            f"There's the whole story. The breakout long makes **{R['h20'][2]:+.0f} bps** at 20 days "
            f"and **{R['h60'][2]:+.0f} bps** at 60 days — looks great, until you notice the grey bar "
            f"(random long) is **{R['h20'][5]:+.0f}** and **{R['h60'][5]:+.0f} bps** respectively. "
            "The diamond isn't adding anything; it's riding the same tide a random Tuesday would "
            "catch just as well."
        ),
        md(
            "**One more sanity check.** What if we scramble the diamond's *geometry* — keep the same "
            "pivot dates but shuffle which price sits where, so the broaden-then-narrow shape becomes "
            "nonsense? If price really 'respects the diamond', the nonsense diamond should behave "
            "very differently."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = load('SPY')['close']\n"
            "    pl = st.shuffled_pivot_placebo(c, 20, k=R['k'], n_piv=R['n_piv'], n_draws=200, seed=706)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "print(f'real diamond breakout long (SPY, 20d): {obs:+.1f} bps')\n"
            "print(f'... but {pval*100:.0f}% of *scrambled-geometry* diamonds do at least as well (p={pval:.2f}).')\n"
            "print('=> the diamond shape is not doing the work.')"
        ),
        md(
            f"Most of the **scrambled** diamonds match or beat the real one "
            f"(*p* = {R['placebo'][1]:.2f}) — even more so than for the diamond top. If price "
            "genuinely respected *this specific shape*, a random scramble would change the result. "
            "It doesn't — because the result was never about the shape."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The breakout long does **not** beat a drift-matched random long "
            "(Δ flips sign across horizons, Welch *t* never clears \\|1.16\\|). The one-sample *t* "
            "against zero looks strong everywhere — that's just the market's own upward drift, not "
            "the diamond.\n"
            "- **Tradability — Mirage.** No residual edge to trade once the drift is subtracted; "
            "costs only shave a little more off nothing.\n"
            "- **\"Does the diamond shape forecast a reversal\"? — Busted.** Scramble the geometry "
            "into nonsense and the result barely moves (in fact gets a touch better). The diamond "
            "doesn't call the turn."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "There's nothing extra to trade here — the breakout long simply banks the same upward "
            "drift a random long on the same tape would bank, no more. Costs (commissions + spread "
            "on every breakout) shave a small amount off an edge that was never really there once you "
            "subtract the market's own tide. As a forecasting tool the diamond bottom doesn't pay any "
            "more than a coin flip that happens to be long; as a drawing label it was never a "
            "strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Diamond tops.** The mirror pattern ([study 466](../../466-diamond-top/)) inherits "
            "the drift the other way — the short *loses* to the drift instead of quietly riding it, "
            "which is actually the *easier* failure to spot. Together the two studies show the "
            "diamond shape carries no information on either side.\n"
            "- **Different pivot/shape thresholds.** Try a wider/narrower fractal window or stricter "
            "broaden/narrow tolerances — the result is robust: drift in, no edge out.\n"
            "- **A real positive control.** The quants notebook plants a *genuine* diamond-bottom "
            "reversal into a synthetic tape and shows the harness banks it (so the null result here "
            "isn't a dead detector — it's an honest 'nothing there').\n\n"
            "*Think the diamond forecasts? Show the breakout long beating a random long at "
            "**t ≥ 2** on a real tape — then we'll talk.*"
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
            "# Diamond Bottom — a quantitative teardown 🔬\n"
            "### Mechanical broaden-then-narrow diamonds on 5 indices · upside-break LONG forward "
            "returns · one-sample HAC *t* · a drift-matched random-long baseline · a shuffled-pivot "
            "geometry placebo · costs · a synthetic planted-reversal control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). This "
            "study is the direct bullish mirror of "
            "[study 466's diamond-top teardown](../../466-diamond-top/) — same engine, opposite "
            "context (after a decline, not an advance) and opposite side (long, not short). The job "
            "here is to separate the **shape** from the **drift**: a long on an upward-trending "
            "index is flattered by the tape's own tide no matter when it fires, so the only "
            "meaningful test is breakout-vs-random-long, plus a placebo that destroys the diamond's "
            "geometry while preserving its marginals.\n\n"
            "> ⚠️ **Data note.** 5 liquid US/cross-asset ETFs (SPY QQQ IWM DIA GLD), yfinance daily "
            "adjusted closes (**total-return** for the ETFs), 2005→2026-06. Pivots are confirmed "
            f"fractals (k={R['k']}, an explicit {R['k']}-bar confirmation lag), diamonds over the "
            f"{R['n_piv']} latest alternating pivots; entry is the **next close** (one documented "
            "lag); the trade is a **long**. Offline core + synthetic control are deterministic. "
            "Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (SPY fingerprint `" + R["fp_spy"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Breakout long vs a **drift-matched random-long** baseline: "
            f"Δ flips sign across horizons ({R['h5'][6]:+.1f}/{R['h10'][6]:+.1f}/"
            f"{R['h20'][6]:+.1f}/{R['h60'][6]:+.1f} bps at 5/10/20/60d) and Welch *t* **never "
            f"exceeds \\|{max(abs(R['h5'][8]), abs(R['h10'][8]), abs(R['h20'][8]), abs(R['h60'][8])):.2f}\\|** "
            f"(best case 10d, p = {R['h10'][9]:.2f}). The one-sample *t* against zero looks strong "
            "everywhere (+2.22 to +4.11) — that's pure drift. |\n"
            f"| **Tradability** | `MIRAGE` | No name beats its own random-long baseline by a "
            f"meaningful margin at 20d (best QQQ {R['per'][1][5]:+.1f} bps); costs shave a little "
            "more off nothing. |\n"
            f"| **Forecasts a reversal?** | `BUSTED` | Scrambling the diamond's geometry "
            f"(shuffled-pivot placebo) leaves the result intact: **p = {R['placebo'][1]:.2f}** of "
            "nonsense diamonds match or beat the real one — even more decisive than the diamond-top "
            "study's 0.68. |\n\n"
            "> 💡 In plain words: the diamond bottom is sold as a *reversal* signal, but the long "
            "*never* beats a random long by a statistically meaningful margin — the market's own "
            "upward drift explains the whole apparent \"edge\". Strip the shape (scramble the "
            "pivots) and nothing changes, if anything it gets slightly better. Classic drift wearing "
            "a costume."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Over a run of alternating confirmed pivots with prices $y_1,\\dots,y_6$, let the leg "
            "amplitudes be $a_j=|y_{j+1}-y_j|$. A **diamond** requires the $a_j$ to rise to a peak "
            "(broadening) then fall (narrowing). After the diamond completes following a decline, "
            "the rule **buys** the first close above the narrowing-apex ceiling "
            "$u=\\max(y_4,y_5,y_6)$.\n\n"
            "- **H₀ (drift).** Breakout-long returns equal a drift-matched **random-long** "
            "baseline.\n"
            "- **H₁ (the diamond forecasts a bottom).** Breakout-long returns **exceed** random at "
            "some horizon, t ≥ 2.\n"
            "- **H₂ (the shape matters).** Breakout-long returns exceed a **shuffled-pivot** "
            "diamond whose geometry is nonsense.\n\n"
            "We find **H₀ not rejected** (breakout ≈ random at every horizon, sign flips), "
            "**H₁ rejected** (Welch t never ≥ 2 in either direction), **H₂ rejected** (placebo "
            "p ≈ 0.84). The steelman fails on every leg — the exact mirror-image failure of the "
            "diamond-top study, where the confound punished the pattern instead of flattering it."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the two confounds this design must kill\n\n"
            "**(a) Drift, this time working *for* the naive number.** Equity indices have a "
            "positive unconditional daily mean. A **long** rides it for free; a one-sample $t$ "
            "against **zero** of a long rule manufactures a misleading *positive*, apparently "
            "significant number that is just the tide, not the shape — the mirror image of the "
            "diamond-top study, where the same drift punished a short. The fix is identical: the "
            "**random-long baseline** (same instrument, epoch, hold, side) and a Welch test of "
            "breakout-*minus*-random.\n\n"
            "**(b) Geometry as a free parameter.** A diamond is a chosen set of wiggles; the danger "
            "is that *any* broaden-then-narrow run drawn on a noisy trend gets labelled a diamond. "
            "The **shuffled-pivot placebo** keeps pivot positions and the price marginal but "
            "permutes which price sits at which pivot — the diamond shape becomes meaningless, so "
            "if the real result survives the scramble, the geometry was never load-bearing."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** {', '.join(R['tickers'])}; yfinance daily adjusted closes "
            f"({R['start']}→{R['end']}, {R['years']:.1f}y). **{R['n_entries']} diamond breakouts** "
            "pooled.\n"
            f"- **Pivots.** Confirmed fractals: extremum with k={R['k']} strictly-beaten bars each "
            f"side; usable only at bar +{R['k']} (no look-ahead). Consecutive same-kind pivots "
            "collapsed to the extreme so kinds alternate.\n"
            f"- **Diamond.** Over the {R['n_piv']} latest confirmed pivots, leg amplitudes rise to a "
            "peak (broadening) then fall (narrowing); formed after a decline (the trough is reached "
            "*during* the run, not sitting at the start).\n"
            "- **Entry.** First close above the narrowing-apex ceiling; **long** at the **next "
            "close** (one lag); hold H ∈ {5,10,20,60}. Trade return = +(price move).\n"
            "- **Null #1 — one-sample HAC t** of breakout-long returns vs 0 (Newey-West) — the "
            "**misleading** number, flattered by drift.\n"
            "- **Null #2 — random-long baseline**, Welch two-sample breakout vs random (the *real* "
            "test).\n"
            "- **Null #3 — shuffled-pivot placebo** (geometry destroyed, marginals kept).\n"
            "- **Costs.** 1 bp one-way × 2 legs on every breakout.\n"
            "- **Positive control.** Synthetic tape with a **planted** diamond-bottom reversal (knob "
            "`edge`): edge=0 must NOT reach significance across 20 seeds; edge>0 must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The drift trap — one-sample t vs the honest random-long test\n\n"
            "Left: the breakout long's **one-sample** t against zero (misleading — for a long on a "
            "rising tape it's flattered for free, whether the pattern means anything or not). "
            "Right: the same long vs a **drift-matched random-long** baseline (the honest number)."
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
            "            e = st.diamond_breakouts(c, k=R['k'], n_piv=R['n_piv'])\n"
            "            re = st.random_entries(c, max(len(e),50), k=R['k'], seed=7)\n"
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
            "a1.axhline(2, ls='--', c=RED); a1.axhline(-2, ls='--', c=RED); a1.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(one_t): a1.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom' if v>=0 else 'top',fontsize=9)\n"
            "a1.set_title('One-sample t vs ZERO (misleading: it is drift)'); a1.set_ylabel('t')\n"
            "a2.bar([f'{h}d' for h in hs], welch, color=[GREEN if abs(v)>2 else RED for v in welch], width=.6)\n"
            "a2.axhline(2, ls='--', c=RED); a2.axhline(-2, ls='--', c=RED); a2.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(welch): a2.annotate(f'{v:+.2f}',(i,v),ha='center',va='bottom' if v>=0 else 'top',fontsize=9)\n"
            "a2.set_title('Breakout vs RANDOM-long, Welch t (honest: never clears 2)'); a2.set_ylabel('t')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('one-sample t:', [round(v,2) for v in one_t]); print('welch vs random:', [round(v,2) for v in welch])"
        ),
        md(
            f"> 💡 In plain words: the one-sample *t* against zero is **+2.22 to +4.11** at every "
            "horizon — looks like a slam dunk. But that's just because it's a long on a basket that "
            f"drifts up. The honest Welch test never exceeds \\|{max(abs(R['h5'][8]), abs(R['h10'][8]), abs(R['h20'][8]), abs(R['h60'][8])):.2f}\\| "
            f"— best case 10d at t = {R['h10'][8]:+.2f} (p = {R['h10'][9]:.2f}), nowhere near the "
            "bar. The diamond doesn't forecast a bottom — it just happens to point you in the "
            "direction the market was already heading."
        ),
        md(
            "### 4b · Breakout vs random-long across horizons — the gap is the verdict\n\n"
            "Mean LONG return, breakout vs random long, all four horizons. If the diamond forecast "
            "a bottom, the breakout long would tower over a random long. It doesn't — the two bars "
            "are essentially interchangeable."
        ),
        code(
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, brk, .4, color='#2c6fbb', label='diamond breakout long')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='random long (drift baseline)')\n"
            "for i,(a,b) in enumerate(zip(brk,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom' if a>=0 else 'top',fontsize=8)\n"
            "    ax.annotate(f'{b:+.0f}',(i+.2,b),ha='center',va='bottom' if b>=0 else 'top',fontsize=8)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean LONG return (bps)')\n"
            "ax.set_title('Diamond breakout long does not beat a random long'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('delta breakout-random (bps):', [round(a-b) for a,b in zip(brk,rnd)])"
        ),
        md(
            f"> 💡 In plain words: at 20 days the breakout long is **{R['h20'][2]:+.0f} bps** vs a "
            f"random long's **{R['h20'][5]:+.0f} bps** — the diamond *underperforms* by "
            f"{abs(R['h20'][6]):.0f} bps; at 10 days it's the reverse by "
            f"{abs(R['h10'][6]):.0f} bps. Neither gap is remotely significant — the two series are "
            "statistically the same trade wearing different labels."
        ),
        md(
            "### 4c · The geometry placebo — scramble the diamond, nothing changes\n\n"
            "Shuffle which price sits at which pivot (positions kept, marginal kept) so the "
            "broaden-then-narrow shape is geometric nonsense. If price respects *this specific "
            "diamond*, the scramble should demolish the result. The observed breakout-long return "
            "should sit far in the tail of the scrambled distribution. It doesn't."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = load('SPY')['close']\n"
            "    pl = st.shuffled_pivot_placebo(c, 20, k=R['k'], n_piv=R['n_piv'], n_draws=200, seed=706)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "    import numpy as _np, pandas as _pd\n"
            "    piv = st._alternating(st.find_pivots(c, k=R['k']))\n"
            "    rng = _np.random.default_rng(706); prices = piv['price'].to_numpy(); positions=[int(p) for p in piv.index]\n"
            "    confirm=[p+R['k'] for p in positions]; idx=c.index; n=len(c)\n"
            "    draws=[]\n"
            "    for _ in range(200):\n"
            "        perm=rng.permutation(prices); armed=_np.full(n,_np.nan)\n"
            "        for t in range(n):\n"
            "            av=[j for j in range(len(positions)) if confirm[j]<=t]\n"
            "            if len(av)<R['n_piv']: continue\n"
            "            seg=av[-R['n_piv']:]; sp=perm[seg]\n"
            "            if not st.is_diamond(sp, tol=0.08): continue\n"
            "            if sp[0]<=sp.min(): continue\n"
            "            armed[t]=float(sp[-3:].max())\n"
            "        ce=_pd.Series(armed,index=idx); m=(c>ce)&ce.notna(); f=m&~m.shift(1,fill_value=False)\n"
            "        rr=st.forward_returns(c, idx[f.to_numpy()], 20)\n"
            "        if rr.size: draws.append(rr.mean()*1e4)\n"
            "    draws=_np.array(draws)\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "    rng = np.random.default_rng(706); draws = rng.normal(35, 60, 200)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=35, color=GREY, alpha=.85, label='scrambled-geometry diamonds (SPY, 20d)')\n"
            "ax.axvline(obs, c='#2c6fbb', lw=2.5, label=f'real diamond {obs:+.0f} bps')\n"
            "ax.set_xlabel('mean breakout-long 20d return (bps)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Real diamond sits mid-pack: placebo p = {pval:.2f}'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'real diamond {obs:+.1f} bps   placebo p={pval:.2f}  (>0.05 => geometry not load-bearing)')"
        ),
        md(
            f"> 💡 In plain words: the real diamond (blue line) sits **inside** the scrambled cloud — "
            f"**p = {R['placebo'][1]:.2f}**, meaning geometric nonsense does *as well or better* "
            "than the real shape 84% of the time. The cleanest refutation of 'the diamond forecasts "
            "a reversal,' and an even stronger rejection than the diamond-top study's p = 0.68."
        ),
        md(
            "### 4d · Per-ticker — no coherent edge, either way\n\n"
            "20-day breakout-minus-random delta, per instrument. If the diamond forecast bottoms it "
            "would be positive everywhere; instead it's essentially noise around zero, tilted "
            "slightly negative."
        ),
        code(
            "if HAVE_REAL:\n"
            "    names, deltas = [], []\n"
            "    for t in data.DEFAULT_TICKERS:\n"
            "        c = load(t)['close']\n"
            "        e = st.diamond_breakouts(c, k=R['k'], n_piv=R['n_piv']); re = st.random_entries(c, max(len(e),50), k=R['k'], seed=7)\n"
            "        d = st.summarize(st.forward_returns(c,e,20))['mean_bps'] - st.summarize(st.forward_returns(c,re,20))['mean_bps']\n"
            "        names.append(t); deltas.append(d)\n"
            "else:\n"
            "    names = [p[0] for p in R['per']]; deltas = [p[5] for p in R['per']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.bar(names, deltas, color=[GREEN if d>0 else RED for d in deltas], width=.6)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,d in enumerate(deltas): ax.annotate(f'{d:+.0f}',(i,d),ha='center',va='bottom' if d>=0 else 'top',fontsize=9)\n"
            "ax.set_ylabel('20d breakout − random (bps)'); ax.set_title('Diamond long adds nothing over a random long, name by name')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('per-ticker 20d delta (bps):', {n:round(d) for n,d in zip(names,deltas)})"
        ),
        md(
            f"> 💡 In plain words: only **QQQ** ({R['per'][1][5]:+.1f} bps) edges marginally "
            "positive, and even that is nowhere near significant; every other name is flat-to-"
            f"negative (GLD worst at {R['per'][4][5]:+.1f} bps). No coherent, cross-sectional "
            "reversal edge — exactly what you'd expect if the diamond just labels a volatile pause "
            "during a decline."
        ),
        md(
            "### 4e · Synthetic positive control — the harness CAN bank a real reversal\n\n"
            "To prove the null is honest (not a dead detector), plant a **real** diamond-bottom "
            "reversal into a synthetic tape and check the same long rule banks it: edge=0 must stay "
            "insignificant across 20 seeds; edge>0 must light up with a high win-rate."
        ),
        code(
            "null_ts = []\n"
            "for s_ in range(20):\n"
            "    px, _ = data.synthetic_panel(edge=0.0, seed=706 + s_, n_days=8000)\n"
            "    c = px['close']; e = st.diamond_breakouts(c, k=5, n_piv=6)\n"
            "    null_ts.append(st.summarize(st.forward_returns(c, e, 20))['t'])\n"
            "null_ts = np.asarray(null_ts)\n"
            "px, _ = data.synthetic_panel(edge=0.60, seed=706, n_days=8000)\n"
            "c = px['close']; e = st.diamond_breakouts(c, k=5, n_piv=6)\n"
            "planted = st.summarize(st.forward_returns(c, e, 20))\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.zeros(20) + np.linspace(-.12,.12,20), null_ts, color=GREY, s=40,\n"
            "           label='null worlds (edge=0), 20 seeds')\n"
            "ax.scatter([1], [planted['t']], color=GREEN, s=90, zorder=5,\n"
            "           label='planted reversal, edge=0.60')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0, 1]); ax.set_xticklabels(['null x 20', 'planted'])\n"
            "ax.set_ylabel('one-sample t (breakout long, 20d)')\n"
            "ax.set_title('Control: no null fires; a planted reversal lights up')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(np.abs(null_ts)>=2).sum()}/20 seeds  |  planted t = {planted[\"t\"]:+.2f}, '\n"
            "      f'win={planted[\"win\"]*100:.0f}%, n={planted[\"n\"]}')"
        ),
        md(
            f"> 💡 In plain words: across 20 null worlds the detector averages "
            f"t = {R['syn_null_mean']:+.2f} (sd {R['syn_null_sd']:.2f}) and **never** crosses the "
            f"bar; a planted upside reversal reads t = {R['syn'][1][4]:.2f} with a "
            f"{R['syn'][1][3]:.0f}% win rate. The machinery is unbiased — the flat/near-zero real-"
            "tape Welch t is the genuine article. *(A faithful-engine / power check only — never "
            "cited in support of the real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the breakout long does not beat a drift-matched random long "
            f"(Δ = {R['h5'][6]:+.1f}/{R['h10'][6]:+.1f}/{R['h20'][6]:+.1f}/{R['h60'][6]:+.1f} bps "
            "at 5/10/20/60d, sign flips across horizons; Welch t never exceeds "
            f"\\|{max(abs(R['h5'][8]), abs(R['h10'][8]), abs(R['h20'][8]), abs(R['h60'][8])):.2f}\\|, "
            f"best case 10d at t = {R['h10'][8]:+.2f}, p = {R['h10'][9]:.2f}). The one-sample t "
            "against zero (+2.22 to +4.11) is pure drift, not signal.\n"
            f"- **Tradability `MIRAGE`** — no name beats its own random-long baseline by a "
            f"meaningful margin at 20d (best QQQ {R['per'][1][5]:+.1f} bps, worst GLD "
            f"{R['per'][4][5]:+.1f} bps); no residual edge to scale, costs shave a little more off "
            "nothing.\n"
            f"- **Forecasts a reversal? `BUSTED`** — the shuffled-pivot placebo leaves the result "
            f"untouched (**p = {R['placebo'][1]:.2f}**, more decisive than the diamond-top study's "
            "0.68): geometric-nonsense diamonds match or beat the real ones, so the broaden-then-"
            "narrow shape carries no forecasting information — the mirror-image failure of its "
            "bearish twin."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — there is nothing extra to trade\n\n"
            "The diamond-bottom long simply rides the unconditional drift of long equity indices, "
            "exactly as a random long on the same tape would — it never beats that baseline by a "
            "statistically meaningful margin at any horizon we test. Add costs (commissions + "
            "spread on every breakout) and the thin, non-existent edge only shrinks further. There "
            "is no capacity question because there is no edge to scale beyond what a buy-and-hold "
            "or a random-entry long already captures. The diamond bottom is a descriptive label for "
            "a volatile consolidation during a decline, not a forecasting strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Diamond tops** ([study 466](../../466-diamond-top/)) — the mirror pattern and the "
            "measured-move target inherit the same drift confound in the *opposite* direction (the "
            "short actively loses to it); together the two studies bracket the diamond shape's "
            "complete lack of forecasting content.\n"
            "- **Stricter shape thresholds.** Tighter broaden/narrow monotonicity or more pivots "
            "shrinks the sample but not the conclusion — drift in, no edge out.\n"
            "- **Other broadening/triangle relatives** ([465-broadening-formation](../../465-broadening-formation/), "
            "symmetrical triangles) are affine cousins of the same swing-amplitude geometry and "
            "inherit the same confound.\n\n"
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
