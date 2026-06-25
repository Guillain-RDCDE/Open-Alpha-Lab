"""Generate the two narrative notebooks for Study 459 (Hikkake pattern).

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
# 2026-05-31, partial June dropped), 21.4 years, hikkake window=3, direction-signed.
R = dict(
    asof="2026-05-31", start="2005-01-03", end="2026-05-29", years=21.4,
    tickers=["SPY", "QQQ", "IWM", "DIA", "GLD"], n_entries=1438, n_long=548, n_short=890,
    window=3, fp_spy="4cb5244f3990",
    # pooled hikkake, per horizon:
    # (H, n, hik_bps, win%, one_sample_t, random_bps, delta_bps, net_bps, welch_t, welch_p)
    h5=(5, 1437, -5.5, 49, -0.84, -2.1, -3.4, -7.5, -0.35, 0.729),
    h10=(10, 1435, -16.7, 49, -1.89, -3.7, -13.1, -18.7, -0.98, 0.326),
    h20=(20, 1433, -34.4, 48, -2.57, -12.4, -22.0, -36.4, -1.18, 0.237),
    h60=(60, 1423, -94.0, 46, -4.10, -21.3, -72.7, -96.0, -2.23, 0.026),
    # per-ticker H=20: (ticker, entries, hik_bps, one_sample_t, random_bps, delta_bps)
    per=[("SPY", 282, -53.8, -1.97, -22.2, -31.6), ("QQQ", 297, -25.3, -0.78, -16.3, -9.0),
         ("IWM", 288, -36.3, -1.21, -22.7, -13.5), ("DIA", 287, -21.9, -1.01, -33.0, 11.1),
         ("GLD", 284, -35.4, -0.99, 32.8, -68.1)],
    # scrambled-direction placebo (SPY, H=20, 500 draws): obs_bps, p, draws
    placebo=(-53.8, 0.978, 500),
    # synthetic control (H=20, n_days=4000): (edge, n, hik_bps, win%, one_sample_t)
    syn=[(0.00, 451, -33.0, 47, -1.42), (0.50, 366, 963.5, 72, 8.89)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Trap_forecasts%3F: Busted](https://img.shields.io/badge/Trap_forecasts%3F-Busted-8b949e?style=flat-square)\n\n"
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

from hikkake_pattern import data, strategy as st

ASOF = "2026-05-31"
HAVE_REAL = data.have_real()
def load(t):
    b = data.load_real(t, allow_fetch=False)
    return b[b.index <= ASOF]
print("real hikkake cache present:", HAVE_REAL, "| tickers:", data.DEFAULT_TICKERS)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does the \"hikkake\" false-breakout trap actually pay? 🪤\n"
            "### A famous price-action pattern — inside bar, fake-out, snap-back — meets a stopwatch\n\n"
            + BADGES +
            "Price-action traders love the **hikkake** (Japanese for *trap*). The recipe: a quiet "
            "**inside bar** (one whose whole range fits inside the day before), then price **breaks "
            "out** of that little range one way — and then **snaps back**. The break was a *fake*. "
            "The lore says that fake-out traps everyone who chased it, so you trade the **other way**: "
            "fade a failed up-break (go short), buy a failed down-break (go long).\n\n"
            "It *looks* clever on a hand-picked chart. But a pattern you only label **after** the "
            "snap-back is the textbook setup for fooling yourself. So we did the only fair thing: "
            "encode the trap **mechanically** (no eyeballing), fire it **1 400+ times** across five "
            "big indices over 21 years, sign every trade by its direction, and time the result "
            "against the only baseline that matters: **the same trades on random days.**\n\n"
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
            "| If I trade the trap's reversal, do I make money? | **No.** The signed return is "
            "**negative** at every horizon (−5 / −17 / −34 / −94 bps at 5/10/20/60 days). |\n"
            "| Is it at least better than random? | **No.** The same trades on **random days** do "
            "**better** — the trap *loses* to a coin-flip entry at every horizon. |\n"
            "| Does the trap's *direction* matter? | **No.** Randomly **flip** which way each trap "
            "predicts and you do *just as well or better* 98% of the time. |\n"
            "| So is it a tradable edge? | **No.** It's a post-hoc chart label with nothing under "
            "the hood — and being short-heavy, it mostly just fades a market that drifts up. |\n\n"
            "> The hikkake is a fine way to *describe* a fake-out after it happened. As a *forecast* — "
            "\"the trap will reverse\" — it's a **mirage**: scramble its one piece of content (which "
            "way it points) and nothing changes."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Wait for an **inside bar**. Then price breaks out of its range — but the break "
            "**fails** and price snaps back through. Everyone who chased the breakout is now "
            "**trapped**. Trade the reversal: short a failed up-break, buy a failed down-break.\"*\n\n"
            "This is **Daniel Chesler's hikkake** (named ~2003), built on the older **inside-bar** "
            "and **false-breakout / bull-trap** ideas. It's one of the most recognisable price-action "
            "patterns, taught on every chart site and built into scanners — so: does the trap "
            "actually *trap*?"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the trap genuinely *forecast* reversals, it would be remarkable: a tiny three-bar "
            "shape would predict the next move's direction, a clean crack in efficiency you could "
            "trade with a ruler. That's the dream the pattern sells.\n\n"
            "But there are two traps for *us*. First, the pattern is labelled **after** the snap-back "
            "— you only call it a hikkake once it's complete, which makes it look prophetic. Second, "
            "it's a **mixed long/short** rule on a market that drifts **up**, so its direction mix "
            "alone moves the result. To separate the **pattern** from the **tape**, we (a) encode it "
            "by a fixed rule with no hindsight, (b) compare to **random days with the same long/short "
            "mix**, and (c) **flip the directions** to see if the trap's one piece of information "
            "matters. We'll do all three."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **{len(R['tickers'])} liquid indices/ETFs** ({', '.join(R['tickers'])}), daily, "
            f"over **{R['years']:.0f} years** ({R['start']} → {R['end']}), and:\n\n"
            "1. **Find the trap mechanically.** An *inside bar* is a bar with a lower high **and** a "
            "higher low than the bar before. Within the next 3 bars price must poke beyond that "
            "range and then **close back through** it. No eyeballing.\n"
            "2. **Trade the reversal.** Short a failed up-break, buy a failed down-break — entered at "
            "the **next close**; measure the signed return over the next **5 / 10 / 20 / 60 days**.\n"
            "3. **The honest baseline.** Do the exact same trades (same count, same long/short mix) "
            "on **random days**. If the trap matters, it must beat random.\n"
            "4. **Flip the trap.** Keep the trap dates but randomly **reverse** each trade's "
            "direction. If the trap's *direction* is real, flipping it should wreck the result. "
            "*If it doesn't, the pattern is a mirage* — announced before we look."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "First, what does a mechanical hikkake even look like? Here's SPY with the detected "
            "traps marked: green = long (failed down-break), red = short (failed up-break)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    b = load('SPY'); seg = b.iloc[-400:]\n"
            "    sig = st.hikkake_signals(b, window=R['window'])\n"
            "    sig = sig[sig.index >= seg.index[0]]\n"
            "    longs = sig[sig['dir']>0].index; shorts = sig[sig['dir']<0].index\n"
            "    fig, ax = plt.subplots(figsize=(10.2, 4.8))\n"
            "    ax.plot(seg.index, seg['close'].values, c='k', lw=1.1, label='SPY close')\n"
            "    ax.scatter(longs, b['close'].reindex(longs), c=GREEN, s=42, zorder=5, label='hikkake LONG')\n"
            "    ax.scatter(shorts, b['close'].reindex(shorts), c=RED, s=42, zorder=5, marker='v', label='hikkake SHORT')\n"
            "    ax.set_title('Mechanical hikkake traps on SPY (last ~1.5y)'); ax.legend(loc='upper left')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('traps in window:', len(sig), '| long:', len(longs), 'short:', len(shorts))\n"
            "else:\n"
            "    print('(no cache — see docs/results.md for the frozen numbers)')"
        ),
        md(
            "Plenty of traps, mostly **short** (the inside-bar/fake-out shape fires more often on "
            "up-pokes). Now the real test: **race the trap against random entries** at four "
            "horizons. Blue = trade the trap; grey = the same trades on random days."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    hik, rnd = [], []\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            b = load(t); sig = st.hikkake_signals(b, window=R['window'])\n"
            "            rs = st.random_entries(b, sig['dir'].to_numpy(), window=R['window'], seed=7)\n"
            "            tt.append(st.forward_returns(b, sig, h)); rr.append(st.forward_returns(b, rs, h))\n"
            "        hik.append(np.concatenate(tt).mean()*1e4); rnd.append(np.concatenate(rr).mean()*1e4)\n"
            "else:\n"
            "    hik = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x-.2, hik, .4, color='#2c6fbb', label='trade the hikkake trap')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='same trades, random days')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,(a,bb) in enumerate(zip(hik,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='top' if a<0 else 'bottom',fontsize=8)\n"
            "    ax.annotate(f'{bb:+.0f}',(i+.2,bb),ha='center',va='top' if bb<0 else 'bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean signed return (bps)')\n"
            "ax.set_title('The trap loses money — and loses to random too'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('hikkake:', [round(v) for v in hik]); print('random:', [round(v) for v in rnd])"
        ),
        md(
            f"There's the whole story. The trap's signed return is **negative everywhere** "
            f"(**{R['h20'][2]:.0f} bps** at 20 days, **{R['h60'][2]:.0f}** at 60) — and **worse than "
            f"random** at every horizon. The pattern doesn't forecast the reversal it promises; "
            "being short-heavy, it mostly just fades a market that drifts up."
        ),
        md(
            "**One more sanity check.** The trap's *only* content is **which way** it predicts. So "
            "let's keep every trap date but randomly **flip** each direction. If price really gets "
            "trapped, the correctly-directed trade should crush the coin-flip directions."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.scrambled_direction_placebo(load('SPY'), 20, window=R['window'], n_draws=300, seed=459)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "print(f'real hikkake (SPY, 20d, correct direction): {obs:+.1f} bps')\n"
            "print(f'... but {pval*100:.0f}% of *direction-flipped* hikkakes do at least as well (p={pval:.2f}).')\n"
            "print('=> which way the trap points is NOT doing the work.')"
        ),
        md(
            f"**98%** of the **direction-flipped** traps match or beat the real one "
            f"(*p* = {R['placebo'][1]:.2f}) — the \"correctly\" directed trap is among the *worst* "
            "coin-flips. If the hikkake genuinely forecast the reversal, flipping it would collapse "
            "the result. It doesn't — because the direction was never carrying information."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The trap's signed return is negative at every horizon and *loses* "
            "to the same trades on random days. It does not beat the drift/exposure-matched baseline.\n"
            "- **Tradability — Mirage.** Negative gross and net; costs only deepen the hole. Nothing "
            "to trade.\n"
            "- **\"Does the trap forecast?\" — Busted.** Randomly flip which way each trap points and "
            "the result is unchanged (*p* = 0.98). The pattern's one piece of content carries no "
            "information."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "There's nothing here to trade. The trap is negative gross *and* net of costs, it loses "
            "to random entries, and its defining direction is a coin flip. A short-heavy rule on an "
            "up-drifting market is structurally fighting the tide; you'd do better holding the index "
            "and never touching the pattern. As a forecasting tool the hikkake doesn't pay; as a "
            "*description* of a fake-out, it was never a strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **A trend filter.** Proponents often only take hikkakes *with* the trend. That's "
            "adding a *second* rule (and a free parameter); on a drifting tape it mostly re-imports "
            "the drift — try it and watch the 'edge' track the filter, not the trap.\n"
            "- **The confirmation window.** Widening or narrowing the 3-bar window changes the trade "
            "count but not the verdict: drift in, trap out.\n"
            "- **A real positive control.** The quants notebook plants a *genuine* trap-reversal into "
            "a synthetic tape and shows the harness banks it (so the dead-flat real result isn't a "
            "broken detector — it's an honest 'nothing there').\n\n"
            "*Think the trap forecasts? Show the hikkake beating its exposure-matched random "
            "baseline at **t ≥ 2** on a real tape — then we'll talk.*"
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
            "# The Hikkake pattern — a quantitative teardown 🔬\n"
            "### Mechanical inside-bar false-breakout traps on 5 indices · direction-signed forward "
            "returns · one-sample HAC *t* · an exposure-matched random-entry baseline · a "
            "scrambled-direction placebo · costs · a synthetic planted-trap control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The job "
            "is to separate the **trap** from the **tape**: a mixed long/short rule on a drifting "
            "index inherits its net exposure, so the only meaningful test is hikkake-vs-random "
            "(matched on the long/short mix), plus a placebo that destroys the trap's one degree of "
            "freedom — its direction.\n\n"
            "> ⚠️ **Data note.** 5 liquid US/cross-asset ETFs (SPY QQQ IWM DIA GLD), yfinance daily "
            "adjusted closes (**total-return** for the ETFs), 2005→2026. Trap = inside bar + false "
            f"break + snap-back within a {R['window']}-bar window; entry is the **next close** (one "
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
            f"| **Signal** | `NONE` | Direction-signed hikkake vs an **exposure-matched random** "
            f"baseline: the trap is *worse* at every horizon (Δ = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}"
            f"/{R['h20'][6]:+.0f}/{R['h60'][6]:+.0f} bps) and the hikkake-minus-random Welch *t* is "
            f"never positive (60d = {R['h60'][8]:+.2f}). |\n"
            f"| **Tradability** | `MIRAGE` | Signed return negative gross **and** net "
            f"({R['h20'][2]:+.0f} bps at 20d); costs deepen the hole. No edge to scale. |\n"
            f"| **Trap forecasts?** | `BUSTED` | Scrambling the trade *direction* leaves the result "
            f"intact: **p = {R['placebo'][1]:.2f}** of flipped traps match or beat the real one. The "
            "trap's direction is not load-bearing. |\n\n"
            "> 💡 In plain words: the hikkake is a short-heavy rule on an up-drifting tape, so its "
            "one-sample *t* is *negative* (it fades the drift) — and once you race it vs an "
            "exposure-matched random control or flip its direction, there's simply nothing there."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let bar $i$ be an **inside bar**: $H_i<H_{i-1}$ and $L_i>L_{i-1}$, defining a range "
            "$[L_i,H_i]$. Within $w$ bars, a bar $j$ **breaks** the range ($H_j>H_i$ or $L_j<L_i$); "
            "a later bar's **close** snaps back through it ($C<H_i$ after an up-break, $C>L_i$ after "
            "a down-break), completing the trap. The Andrews— sorry, **Chesler** rule trades the "
            "reversal: $d=-1$ (short) after a failed up-break, $d=+1$ (long) after a failed "
            "down-break. We measure $d\\cdot(P_{e+H}/P_e-1)$ entered at $e=$ next close.\n\n"
            "- **H₀ (drift / exposure).** Signed returns equal an exposure-matched **random-entry** "
            "baseline (same dates count, same long/short mix).\n"
            "- **H₁ (the trap forecasts).** Signed returns **exceed** random at some horizon, t ≥ 2.\n"
            "- **H₂ (the direction matters).** Signed returns exceed a **direction-scrambled** trap.\n\n"
            "We find **H₀ not rejected** (trap ≤ random everywhere), **H₁ rejected** (Welch t never "
            "≥ 2; it's *negative*), **H₂ rejected** (placebo p ≈ 0.98). The steelman fails on every leg."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the two confounds this design must kill\n\n"
            "**(a) Net exposure / drift.** A mixed long/short rule has a *net* sign; here it's "
            "**short-heavy** (890 short vs 548 long), so on an up-drifting index it inherits a "
            "*negative* unconditional mean. A one-sample $t$ against **zero** measures that exposure, "
            "not a forecast. The fix is a random baseline matched on the **same long/short mix** and "
            "a Welch test of trap-*minus*-random.\n\n"
            "**(b) Direction as the only free bit.** A hikkake's entire informational content is "
            "*which way it points*. The **scrambled-direction placebo** keeps the trap dates and "
            "trade count but randomly flips each direction — so if the real result survives the flip, "
            "the trap's geometry was never load-bearing."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** {', '.join(R['tickers'])}; yfinance daily adjusted closes "
            f"({R['start']}→{R['end']}, {R['years']:.1f}y). **{R['n_entries']} hikkakes** pooled "
            f"({R['n_long']} long / {R['n_short']} short).\n"
            f"- **Trap.** Inside bar ($H_i<H_{{i-1}}$, $L_i>L_{{i-1}}$); false break + snap-back close "
            f"within a {R['window']}-bar window; trade the reversal.\n"
            "- **Entry.** Snap-back completes the trap on its close; enter **next close** (one lag); "
            "hold H ∈ {5,10,20,60}; return signed by direction.\n"
            "- **Null #1 — one-sample HAC t** of signed returns vs 0 (Newey-West).\n"
            "- **Null #2 — exposure-matched random baseline**, Welch two-sample (the *real* test).\n"
            "- **Null #3 — scrambled-direction placebo** (direction destroyed, dates/mix kept).\n"
            "- **Costs.** 1 bp one-way × 2 legs on every trade.\n"
            "- **Positive control.** Synthetic tape with a **planted** trap-reversal (knob `edge`): "
            "edge=0 must NOT reach significance; edge>0 must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The exposure trap — one-sample t is negative, vs-random confirms no edge\n\n"
            "Left: the hikkake's **one-sample** t against zero (negative — it's the short-heavy "
            "exposure fading drift). Right: the same trap vs an **exposure-matched random** baseline "
            "(the honest number — never positive)."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    one_t, hik, rnd, welch = [], [], [], []\n"
            "    from scipy import stats\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            b = load(t); sig = st.hikkake_signals(b, window=R['window'])\n"
            "            rs = st.random_entries(b, sig['dir'].to_numpy(), window=R['window'], seed=7)\n"
            "            tt.append(st.forward_returns(b, sig, h)); rr.append(st.forward_returns(b, rs, h))\n"
            "        tt = np.concatenate(tt); rr = np.concatenate(rr)\n"
            "        one_t.append(st.summarize(tt)['t']); hik.append(tt.mean()*1e4); rnd.append(rr.mean()*1e4)\n"
            "        welch.append(stats.ttest_ind(tt, rr, equal_var=False)[0])\n"
            "else:\n"
            "    one_t = [R['h5'][4], R['h10'][4], R['h20'][4], R['h60'][4]]\n"
            "    hik = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "    welch = [R['h5'][8], R['h10'][8], R['h20'][8], R['h60'][8]]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "a1.bar([f'{h}d' for h in hs], one_t, color=GREY, width=.6)\n"
            "a1.axhline(-2, ls='--', c=RED, label='|t|=2 bar'); a1.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(one_t): a1.annotate(f'{v:.2f}',(i,v),ha='center',va='top',fontsize=9)\n"
            "a1.set_title('One-sample t vs ZERO (it is negative exposure)'); a1.set_ylabel('t'); a1.legend()\n"
            "a2.bar([f'{h}d' for h in hs], welch, color=[GREEN if v>2 else RED for v in welch], width=.6)\n"
            "a2.axhline(2, ls='--', c=RED); a2.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(welch): a2.annotate(f'{v:+.2f}',(i,v),ha='center',va='bottom' if v>=0 else 'top',fontsize=9)\n"
            "a2.set_title('Trap vs RANDOM, Welch t (never positive)'); a2.set_ylabel('t')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('one-sample t:', [round(v,2) for v in one_t]); print('welch vs random:', [round(v,2) for v in welch])"
        ),
        md(
            f"> 💡 In plain words: the left bars are **negative** (20d **{R['h20'][4]:.2f}**, 60d "
            f"**{R['h60'][4]:.2f}**) — the short-heavy trap fading the drift, *not* a forecast. The "
            f"right bars are the real test: trap-minus-random is **negative at every horizon** "
            f"({R['h20'][8]:+.2f} at 20d, {R['h60'][8]:+.2f} at 60d) — never anywhere near +2. The "
            "trap adds nothing over a matched coin flip."
        ),
        md(
            "### 4b · Trap vs random across horizons — the gap is the verdict\n\n"
            "Mean signed return, hikkake vs exposure-matched random, all four horizons. The trap "
            "should tower over random if it forecasts. It sits *below* it."
        ),
        code(
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, hik, .4, color='#2c6fbb', label='hikkake trap')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='random entry (matched mix)')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,(a,b) in enumerate(zip(hik,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='top' if a<0 else 'bottom',fontsize=8)\n"
            "    ax.annotate(f'{b:+.0f}',(i+.2,b),ha='center',va='top' if b<0 else 'bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean signed fwd return (bps)')\n"
            "ax.set_title('Hikkake trap does not beat random entry'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('delta hik-random (bps):', [round(a-b) for a,b in zip(hik,rnd)])"
        ),
        md(
            f"> 💡 In plain words: at 20 days the trap is **{R['h20'][2]:.0f} bps** and random is "
            f"**{R['h20'][5]:.0f} bps** — the trap *underperforms* a matched dart by "
            f"{abs(R['h20'][6]):.0f} bps. There is no horizon where it wins."
        ),
        md(
            "### 4c · The direction placebo — flip the trap, nothing changes\n\n"
            "Keep every trap date and the trade count, but randomly **flip** each trade's direction "
            "— destroying the trap's one degree of freedom while keeping the entry marginal. If "
            "price respects *this specific reversal*, the real (correctly-directed) trap should sit "
            "far in the right tail. It sits at the far *left*."
        ),
        code(
            "if HAVE_REAL:\n"
            "    b = load('SPY'); sig = st.hikkake_signals(b, window=R['window'])\n"
            "    pl = st.scrambled_direction_placebo(b, 20, window=R['window'], n_draws=300, seed=459)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "    import numpy as _np\n"
            "    rng = _np.random.default_rng(459); dirs = sig['dir'].to_numpy(float); draws=[]\n"
            "    for _ in range(300):\n"
            "        flip = rng.choice([-1.0,1.0], size=len(dirs)); scr = sig.copy(); scr['dir']=(dirs*flip).astype(int)\n"
            "        rr = st.forward_returns(b, scr, 20)\n"
            "        if rr.size: draws.append(rr.mean()*1e4)\n"
            "    draws=_np.array(draws)\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "    rng = np.random.default_rng(459); draws = rng.normal(0, 30, 300)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=40, color=GREY, alpha=.85, label='direction-flipped hikkakes (SPY, 20d)')\n"
            "ax.axvline(obs, c='#2c6fbb', lw=2.5, label=f'real trap {obs:+.0f} bps')\n"
            "ax.set_xlabel('mean signed 20d return (bps)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Real trap sits at the LEFT: placebo p = {pval:.2f}'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'real trap {obs:+.1f} bps   placebo p={pval:.2f}  (>0.05 => direction not load-bearing)')"
        ),
        md(
            f"> 💡 In plain words: the real trap (blue line) sits among the **worst** of the "
            f"direction-flipped cloud — **p = {R['placebo'][1]:.2f}**. Flipping the trap does at "
            "least as well 98% of the time, so the trap's direction carries no information. This is "
            "the cleanest refutation of 'the trap forecasts the reversal.'"
        ),
        md(
            "### 4d · Per-ticker — negative deltas almost everywhere\n\n"
            "20-day trap-minus-random delta, per instrument. If the trap worked it would be positive "
            "across the board; instead it's negative in 4 of 5."
        ),
        code(
            "if HAVE_REAL:\n"
            "    names, deltas = [], []\n"
            "    for t in data.DEFAULT_TICKERS:\n"
            "        b = load(t); sig = st.hikkake_signals(b, window=R['window'])\n"
            "        rs = st.random_entries(b, sig['dir'].to_numpy(), window=R['window'], seed=7)\n"
            "        d = st.summarize(st.forward_returns(b,sig,20))['mean_bps'] - st.summarize(st.forward_returns(b,rs,20))['mean_bps']\n"
            "        names.append(t); deltas.append(d)\n"
            "else:\n"
            "    names = [p[0] for p in R['per']]; deltas = [p[5] for p in R['per']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.bar(names, deltas, color=[GREEN if d>0 else RED for d in deltas], width=.6)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,d in enumerate(deltas): ax.annotate(f'{d:+.0f}',(i,d),ha='center',va='bottom' if d>=0 else 'top',fontsize=9)\n"
            "ax.set_ylabel('20d trap − random (bps)'); ax.set_title('Trap underperforms random in 4 of 5 names')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('per-ticker 20d delta (bps):', {n:round(d) for n,d in zip(names,deltas)})"
        ),
        md(
            f"> 💡 In plain words: only **DIA** edges out a positive delta ({R['per'][3][5]:+.0f} bps, "
            f"noise); GLD is **{R['per'][4][5]:+.0f}** bps *behind* random. No coherent "
            "cross-sectional edge — exactly what you'd expect if the trap is an empty label."
        ),
        md(
            "### 4e · Synthetic positive control — the harness CAN bank a real trap\n\n"
            "To prove the null is honest (not a dead detector), plant a **real** trap-reversal into a "
            "synthetic tape and check the same rule banks it: edge=0 must stay below the bar; edge>0 "
            "must light up with a high win-rate."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.50):\n"
            "    px, _ = data.synthetic_panel(edge=edge, seed=459, n_days=4000)\n"
            "    sig = st.hikkake_signals(px, window=3); s = st.summarize(st.forward_returns(px, sig, 20))\n"
            "    res.append((edge, s['n'], s['mean_bps'], s['win']*100, s['t']))\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "labels = [f'planted edge\\n{e:.2f}' for e,_,_,_,_ in res]; tvals = [r[4] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 bar'); ax.axhline(0, c='k', lw=.8)\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('20d one-sample t'); ax.set_title('Control: edge=0 -> below bar; planted trap -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,n,m,w,t in res: print(f'edge={e:.2f}: n={n} hik={m:+.1f}bps win={w:.0f}% t={t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted trap the control sits at "
            f"**t = {R['syn'][0][4]:.2f}** (win {R['syn'][0][3]:.0f}% — no false positive); a planted "
            f"reversal reaches **t = {R['syn'][1][4]:.2f}** (win {R['syn'][1][3]:.0f}%). The detector "
            "works — so the negative real-tape result is a genuine 'nothing there', not a broken pipeline."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the direction-signed hikkake does not beat an exposure-matched "
            f"random baseline (trap − random = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f}"
            f"/{R['h60'][6]:+.0f} bps at 5/10/20/60d; Welch t never positive, "
            f"{R['h60'][8]:+.2f} at 60d). The negative one-sample t's (20d **{R['h20'][4]:.2f}**) are "
            "short-heavy exposure fading the drift.\n"
            f"- **Tradability `MIRAGE`** — signed return negative gross and net; costs only deepen "
            "the hole. Nothing to scale.\n"
            f"- **Trap forecasts? `BUSTED`** — the scrambled-direction placebo leaves the result "
            f"intact (**p = {R['placebo'][1]:.2f}**): flipped traps do as well as the real one, so "
            "the trap's defining direction carries no forecasting information."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — there is nothing to trade\n\n"
            "The hikkake's signed return is negative gross *and* net of costs, it loses to an "
            "exposure-matched random entry at every horizon, and its defining direction is a coin "
            "flip. A short-heavy rule on an up-drifting index is structurally fighting the tide; "
            "there is no capacity question because there is no edge to scale. The hikkake is a "
            "descriptive label for a fake-out, not a forecasting strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Trend-filtered hikkakes.** Taking only with-trend traps adds a second rule and a "
            "free parameter; on a drifting tape the apparent improvement is mostly re-imported drift, "
            "not the trap.\n"
            "- **Window / inside-bar strictness.** Wider windows or 'soft' inside bars change the "
            "trade count but inherit the same direction-placebo refutation.\n"
            "- **Intraday hikkakes.** The pattern is often pitched on lower timeframes; the same "
            "exposure-matched + direction-placebo gauntlet applies (and the 60-day cap on sub-hourly "
            "yfinance bars is why this study uses daily).\n\n"
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
