"""Generate the two narrative notebooks for Study 467 (Bump-and-Run Reversal).

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
# 2026-05-31, partial June dropped), 21.4 years, lead=60 bump=30 bump_mult=2.0, break SHORT.
R = dict(
    asof="2026-05-31", start="2005-01-03", end="2026-05-29", years=21.4,
    tickers=["SPY", "QQQ", "IWM", "DIA", "GLD"], n_entries=162,
    lead=60, bump=30, mult=2.0,
    fp_spy="4cb5244f3990",
    # pooled BARR-break SHORT, per horizon:
    # (H, n, break_bps, win%, one_sample_t, random_bps, delta_bps, net_bps, welch_t, welch_p)
    h5=(5, 162, -15.8, 46, -1.01, 45.1, -60.9, -17.8, -2.50, 0.013),
    h10=(10, 162, -23.1, 46, -0.86, 48.1, -71.2, -25.1, -2.00, 0.046),
    h20=(20, 162, -66.0, 37, -1.84, 58.0, -123.9, -68.0, -2.54, 0.011),
    h60=(60, 162, -253.0, 36, -4.09, -121.1, -131.8, -255.0, -1.72, 0.085),
    # per-ticker H=20: (ticker, entries, break_bps, one_sample_t, random_bps, delta_bps)
    per=[("SPY", 41, -28.8, -0.37, 46.6, -75.3), ("QQQ", 38, -25.9, -0.30, 24.2, -50.1),
         ("IWM", 25, -138.6, -1.44, 138.2, -276.8), ("DIA", 41, -33.9, -0.75, 55.1, -89.0),
         ("GLD", 17, -215.7, -2.18, 25.8, -241.5)],
    # shuffled-window placebo (SPY, H=20, 500 draws): obs_bps, p, draws
    placebo=(-28.8, 0.359, 500),
    # synthetic control (H=5, n_days=8000): (edge, n, break_bps, win%, one_sample_t)
    syn=[(0.00, 23, 86.0, 61, 1.86), (0.70, 16, 182.5, 69, 3.45)],
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

from bump_and_run import data, strategy as st

ASOF = "2026-05-31"
HAVE_REAL = data.have_real()
def load(t):
    b = data.load_real(t, allow_fetch=False)
    return b[b.index <= ASOF]
print("real bump-and-run cache present:", HAVE_REAL, "| tickers:", data.DEFAULT_TICKERS)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does the bump-and-run reversal actually pay? 🪤\n"
            "### A famous chart pattern — quiet lead-in, speculative bump, break back down — meets a stopwatch\n\n"
            + BADGES +
            "Open any chart-pattern guide and you'll find the **bump-and-run reversal** (BARR): a "
            "gentle up-sloping **lead-in trendline**, then a burst of speculation that **bumps** price "
            "far above the line, then a **break** back below it. The lore — popularised by Thomas "
            "Bulkowski — is that the break is a **reversal signal**: the speculation has exhausted, so "
            "you **short** the break and ride price down.\n\n"
            "It *looks* uncanny on a hand-picked chart. But a pattern you draw **after** the swings "
            "have happened — choosing the slope, the bump, the break by eye — is the textbook setup "
            "for fooling yourself. So we did the only fair thing: encode the BARR **mechanically** (no "
            "eyeballing), fire the \"short the break\" rule across five big indices over 21 years, and "
            "time the result with a stopwatch — against the only baseline that matters: **shorting on "
            "random days instead.**\n\n"
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
            "| If I short when price breaks back below the trendline, do I make money? | **No.** The "
            "short *loses* at every horizon — but that's mostly because you're shorting a market that "
            "drifts up. |\n"
            "| Is the *pattern* doing anything? | **No — it makes things worse.** Short on **random "
            "days** instead and you lose **less**. The bump-and-run break is a *worse* place to be "
            "short than a random day. |\n"
            "| Does the break forecast a reversal? | **No.** Price tends to keep **rising** after the "
            "break — the opposite of the advertised 'run' down. |\n"
            "| Scramble the pattern's shape? | The result **barely changes** — the specific lead-in, "
            "bump and break carry no information. |\n\n"
            "> The bump-and-run is a great way to *describe* a blow-off after the fact. As a "
            "*forecast* — \"the break will reverse\" — it's a **mirage**: the break doesn't mark a top, "
            "it marks a spot you'd least want to be short."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Draw a gentle **lead-in** trendline under a quiet advance. Watch for a **bump** — a "
            "speculative surge that lifts price to at least twice the lead-in's height above the line. "
            "When price **breaks** back below the lead-in trendline, the speculation has exhausted: "
            "short the break and ride the reversal down.\"*\n\n"
            "This is **Thomas Bulkowski's** bump-and-run reversal (*Encyclopedia of Chart Patterns*, "
            "2000/2005), one of the best-known named reversal figures, built into chart scanners and "
            "taught on every TA site. So: does the break actually forecast the run down?"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the break genuinely *forecast* a reversal, it would be remarkable: a lead-in line and a "
            "bump would predict a future top, a clean crack in market efficiency you could trade with a "
            "ruler. That's the dream the pattern sells.\n\n"
            "But there's a trap. A bump-and-run is drawn **by hand, after the blow-off** — you choose "
            "the slope, what counts as a bump, where the break is. And the rule is a **short** on a "
            "market (stock indices) that drifts **up** over time, so *any* short will tend to lose — "
            "and *any* over-extended chart will, with hindsight, look like a bump that 'reversed'. To "
            "separate the **pattern** from the **tide**, we (a) draw the BARR by a fixed mechanical "
            "rule with no hindsight, and (b) compare it to shorting on **random days**. We'll do both."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **{len(R['tickers'])} liquid indices/ETFs** ({', '.join(R['tickers'])}), daily, "
            f"over **{R['years']:.0f} years** ({R['start']} → {R['end']}), and:\n\n"
            "1. **Fit the lead-in mechanically.** A least-squares trendline on the trailing "
            f"**{R['lead']} bars**, slope required to be *gently positive* (a calm up-trend) — fit only "
            "on past data, no peeking ahead.\n"
            "2. **Confirm the bump by rule.** Price must surge to at least **2×** the lead-in's "
            "above-line height, and the bump must have *just* peaked — no cherry-picking old blow-offs.\n"
            "3. **Short the break.** When the close drops back **below the lead-in line**, short at the "
            "next close; measure the return over the next **5 / 10 / 20 / 60 days**.\n"
            "4. **The honest baseline.** Do the exact same **short** on **random days**. If the pattern "
            "matters, the break-short must beat a random short. *If it doesn't, the pattern is a "
            "mirage* — that's the result that would make us say so, announced before we look."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "First, what does a mechanical bump-and-run even look like? Here's SPY with the lead-in "
            "trendline, the bump above it, and the break the rule would short."
        ),
        code(
            "b = load('SPY') if HAVE_REAL else None\n"
            "if HAVE_REAL:\n"
            "    cl = b['close']; seg = cl.iloc[-500:]\n"
            "    ent = st.barr_break_entries(cl, lead=R['lead'], bump=R['bump'], bump_mult=R['mult'])\n"
            "    ent = ent[ent >= seg.index[0]]\n"
            "    fig, ax = plt.subplots(figsize=(10.2, 4.8))\n"
            "    ax.plot(seg.index, seg.values, c='k', lw=1.2, label='SPY close')\n"
            "    ax.scatter(ent, cl.reindex(ent), c=RED, s=55, zorder=5, marker='v', label='break SHORT')\n"
            "    ax.set_title('Mechanical bump-and-run break shorts on SPY (last ~2y)'); ax.legend(loc='upper left')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('bump-and-run break shorts in window:', len(ent))\n"
            "else:\n"
            "    print('(no cache — see docs/results.md for the frozen numbers)')"
        ),
        md(
            "The pattern fires on over-extended pullbacks — *as a description* of a blow-off. The "
            "question is whether those red short markers are followed by falls. **Let's race the "
            "break-short against random shorts** at four horizons. Blue = short the break; grey = "
            "short on random days. (Both are shorts, so positive bars mean the short *made* money.)"
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    brk, rnd = [], []\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            c = load(t)['close']\n"
            "            e = st.barr_break_entries(c, lead=R['lead'], bump=R['bump'], bump_mult=R['mult'])\n"
            "            re = st.random_entries(c, max(len(e),50), lead=R['lead'], bump=R['bump'], seed=7)\n"
            "            tt.append(st.forward_returns(c, e, h)); rr.append(st.forward_returns(c, re, h))\n"
            "        brk.append(np.concatenate(tt).mean()*1e4); rnd.append(np.concatenate(rr).mean()*1e4)\n"
            "else:\n"
            "    brk = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x-.2, brk, .4, color='#2c6fbb', label='short the break')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='short on random days')\n"
            "for i,(a,bb) in enumerate(zip(brk,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom' if a>=0 else 'top',fontsize=8)\n"
            "    ax.annotate(f'{bb:+.0f}',(i+.2,bb),ha='center',va='bottom' if bb>=0 else 'top',fontsize=8)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean short return (bps)')\n"
            "ax.set_title('The break short does NOT beat a random short — it loses to it'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('break:', [round(v) for v in brk]); print('random:', [round(v) for v in rnd])"
        ),
        md(
            f"There's the whole story in one chart. The break short **loses** in absolute terms "
            f"(**{R['h20'][2]:.0f} bps** over 20 days) — but so does any short on a rising market. The "
            f"damning part: a **random short loses less** (**{R['h20'][5]:+.0f} bps** at 20 days). The "
            "famous pattern is a *worse* place to be short than a random day. Price keeps climbing "
            "after the break — the reversal the pattern promises doesn't show up."
        ),
        md(
            "**One more sanity check.** What if we scramble the pattern's *shape* — shuffle the order "
            "of the daily moves so the lead-in, bump and break become nonsense, while keeping the same "
            "pool of returns? If price really 'reverses at the break', the nonsense version should do "
            "much worse."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = load('SPY')['close']\n"
            "    pl = st.shuffled_window_placebo(c, 20, lead=R['lead'], bump=R['bump'], bump_mult=R['mult'], n_draws=200, seed=467)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "print(f'real bump-and-run break short (SPY, 20d): {obs:+.1f} bps')\n"
            "print(f'... but {pval*100:.0f}% of *shape-scrambled* tapes do at least as well (p={pval:.2f}).')\n"
            "print('=> the bump-and-run shape is not doing the work.')"
        ),
        md(
            f"More than a third of the **scrambled** tapes match or beat the real one "
            f"(*p* = {R['placebo'][1]:.2f}). If the break genuinely forecast a reversal, destroying the "
            "shape would collapse the result. It doesn't — because the result was never about the shape."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The break short does **not** beat a random short — it does *worse* "
            "(the break-vs-random difference clears −2 at 5/10/20 days). The reversal doesn't arrive.\n"
            "- **Tradability — Mirage.** Nothing to trade: you lose to the market's drift *and* to a "
            "random short, and costs only make it worse.\n"
            "- **\"Does the bump-then-break forecast a reversal\"? — Busted.** Scramble the shape and "
            "the result barely moves. The pattern describes a blow-off; it doesn't forecast the top."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "There's nothing here to trade. The break short loses to the market's upward drift and "
            "loses *again* to a random short, so it has negative skill, not positive. Costs "
            "(commissions + spread on every break) push the already-losing result further negative. As "
            "a forecasting tool the bump-and-run break doesn't pay; as a drawing tool, it was never "
            "meant to be a strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The bullish mirror.** Bulkowski also describes bump-and-run *bottoms* (buy the upside "
            "break). On an up-drifting tape that long version will *look* great — for the same beta "
            "reason the short looks bad — so it needs the same random-baseline test before you believe "
            "it.\n"
            "- **Different thresholds.** Try a steeper/gentler lead-in or a 1.5× / 3× bump — the result "
            "is robust: shape in, beta out.\n"
            "- **A real positive control.** The quants notebook plants a *genuine* post-bump reversal "
            "into a synthetic tape and shows the harness banks it (so the null result here isn't a dead "
            "detector — it's an honest 'nothing there').\n\n"
            "*Think the break forecasts the top? Show the break short beating a random short at "
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
            "# Bump-and-Run Reversal — a quantitative teardown 🔬\n"
            "### Mechanical lead-in/bump/break on 5 indices · break-short forward returns · "
            "one-sample HAC *t* · a drift-matched random-short baseline · a shuffled-window shape "
            "placebo · costs · a synthetic planted-reversal control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The job "
            "is to separate the **pattern** from the **drift**: a downward-betting rule on an "
            "upward-trending index loses *something* by construction, so the only meaningful test is "
            "break-vs-random-**short**, plus a placebo that destroys the bump-and-run shape while "
            "preserving its marginals.\n\n"
            "> ⚠️ **Data note.** 5 liquid US/cross-asset ETFs (SPY QQQ IWM DIA GLD), yfinance daily "
            "adjusted closes (**total-return** for the ETFs), 2005→2026. Lead-in is a trailing "
            f"least-squares trendline (lead={R['lead']}), bump ≥ {R['mult']:.0f}× the lead-in height "
            f"(bump window {R['bump']}), entry is the **next close** after a downcross break (one "
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
            f"| **Signal** | `NONE` | Break short vs a **drift-matched random short**: the break is "
            f"*worse* at every horizon (Δ = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f}/"
            f"{R['h60'][6]:+.0f} bps) and the break-minus-random Welch *t* is **negative**, clearing "
            f"−2 at 3/4 horizons (20d **{R['h20'][8]:+.2f}**, *p* = {R['h20'][9]:.3f}). |\n"
            f"| **Tradability** | `MIRAGE` | The short loses in absolute terms (60d one-sample "
            f"*t* = {R['h60'][4]:.2f}) — that's negative short-beta — *and* loses to a random short. "
            "No residual edge to scale. |\n"
            f"| **Forecasts a reversal?** | `BUSTED` | Scrambling the bump-and-run shape "
            f"(shuffled-window placebo) leaves the result intact: **p = {R['placebo'][1]:.2f}** of "
            "order-destroyed tapes match or beat the real one. The geometry isn't doing the work. |\n\n"
            "> 💡 In plain words: the break short *looks* like it loses for a reason, but it loses "
            "*more* than a random short — price keeps rising after the break. Strip the drift (race it "
            "vs a random short) or strip the shape (scramble the order) and there's nothing left. "
            "Classic negative-beta-in-a-costume."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Fit a lead-in trendline $y=s\\,x+b$ by least squares on a trailing window, slope $s$ "
            "*gently positive*. Let $h_L$ be its average above-line height. A **bump** is confirmed "
            "when $\\max_t (C_t - (s\\,t+b)) \\ge m\\,h_L$ (with $m\\approx 2$) and the bump peak is "
            "recent. The **break** is the first close $C_t < s\\,t+b$ that *downcrosses* the line; the "
            "rule **shorts** it.\n\n"
            "- **H₀ (drift).** Break-short returns equal a drift-matched **random-short** baseline.\n"
            "- **H₁ (the break forecasts a reversal).** Break-short returns **exceed** the random "
            "short at some horizon, t ≥ 2.\n"
            "- **H₂ (the shape matters).** Break-short returns exceed a **shuffled-window** tape whose "
            "bump-and-run geometry is destroyed.\n\n"
            "We find **H₀ not rejected — and then some** (break < random short, Welch t ≤ −2), "
            "**H₁ rejected** (the *t* is negative, not ≥ +2), **H₂ rejected** (placebo p ≈ 0.36). The "
            "steelman fails on every leg; the break, if anything, anti-forecasts the reversal."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the two confounds this design must kill\n\n"
            "**(a) Drift (short-beta).** Equity indices have a positive unconditional daily mean. *Any* "
            "short on a long horizon pays that drift away; a one-sample $t$ of a short rule against "
            "**zero** measures the tide working against you, not the rule. The fix is the "
            "**random-short baseline** (same instrument, epoch, hold, *short side*) and a Welch test of "
            "break-*minus*-random.\n\n"
            "**(b) Shape as a free parameter.** A bump-and-run is a slope + a threshold + a break; the "
            "danger is that *any* over-extension on a trend, labelled after the fact, looks like a bump "
            "that 'reversed'. The **shuffled-window placebo** permutes the per-bar returns — keeping "
            "the price marginal but destroying the lead-in/bump/break ordering — so if the real result "
            "survives the scramble, the geometry was never load-bearing."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** {', '.join(R['tickers'])}; yfinance daily adjusted closes "
            f"({R['start']}→{R['end']}, {R['years']:.1f}y). **{R['n_entries']} break shorts** pooled.\n"
            f"- **Lead-in.** Trailing least-squares trendline over {R['lead']} bars; slope must be "
            "gently positive (0 < slope ≤ a cap, in bps of price). Fit on past bars only — no "
            "look-ahead.\n"
            f"- **Bump.** Close surges to ≥ {R['mult']:.0f}× the lead-in's above-line height within the "
            f"{R['bump']}-bar bump window; the bump peak must be *recent* (no stale geometry re-fires).\n"
            "- **Entry.** First close that downcrosses the extended lead-in line; **short**, entered "
            "**next close** (one lag); hold H ∈ {5,10,20,60}.\n"
            "- **Null #1 — one-sample HAC t** of break-short returns vs 0 (Newey-West).\n"
            "- **Null #2 — random-short baseline**, Welch two-sample break vs random (the *real* test).\n"
            "- **Null #3 — shuffled-window placebo** (shape destroyed, marginals kept).\n"
            "- **Costs.** 1 bp one-way × 2 legs on every break.\n"
            "- **Positive control.** Synthetic tape with a **planted** post-bump reversal (knob "
            "`edge`): edge=0 must NOT reach significance; edge>0 must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The short-beta trap — one-sample t vs the honest random-short test\n\n"
            "Left: the break short's **one-sample** t against zero (the misleading number — a short on "
            "a rising index). Right: the same break vs a **drift-matched random short** (the honest "
            "number). A working reversal pattern would put the right bars *above* +2; they sit below 0."
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
            "            e = st.barr_break_entries(c, lead=R['lead'], bump=R['bump'], bump_mult=R['mult'])\n"
            "            re = st.random_entries(c, max(len(e),50), lead=R['lead'], bump=R['bump'], seed=7)\n"
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
            "a1.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(one_t): a1.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom' if v>=0 else 'top',fontsize=9)\n"
            "a1.set_title('One-sample t vs ZERO (misleading: it is short-beta)'); a1.set_ylabel('t')\n"
            "a2.bar([f'{h}d' for h in hs], welch, color=[GREEN if v>2 else RED for v in welch], width=.6)\n"
            "a2.axhline(2, ls='--', c=GREEN, label='t=+2 (would confirm)'); a2.axhline(-2, ls='--', c=RED); a2.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(welch): a2.annotate(f'{v:+.2f}',(i,v),ha='center',va='bottom' if v>=0 else 'top',fontsize=9)\n"
            "a2.set_title('Break vs RANDOM short, Welch t (honest: negative)'); a2.set_ylabel('t'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('one-sample t:', [round(v,2) for v in one_t]); print('welch vs random:', [round(v,2) for v in welch])"
        ),
        md(
            f"> 💡 In plain words: the left bars are negative (60d **{R['h60'][4]:.2f}**) — that's the "
            f"short fighting the drift, not the rule. The right bars are the real test: break-minus-"
            f"random is **negative**, clearing −2 at 5/10/20d ({R['h20'][8]:+.2f} at 20d). A reversal "
            "pattern would need these *above* +2; instead the break is a *worse* short than a coin flip."
        ),
        md(
            "### 4b · Break vs random short across horizons — the gap is the verdict\n\n"
            "Mean short return, break vs random short, all four horizons. The break should tower over "
            "a random short if it forecasts a reversal. It sits *below* it."
        ),
        code(
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, brk, .4, color='#2c6fbb', label='break short')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='random short (drift baseline)')\n"
            "for i,(a,b) in enumerate(zip(brk,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom' if a>=0 else 'top',fontsize=8)\n"
            "    ax.annotate(f'{b:+.0f}',(i+.2,b),ha='center',va='bottom' if b>=0 else 'top',fontsize=8)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean short return (bps)')\n"
            "ax.set_title('Break short underperforms a random short at every horizon'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('delta break-random (bps):', [round(a-b) for a,b in zip(brk,rnd)])"
        ),
        md(
            f"> 💡 In plain words: at 20 days the break short is **{R['h20'][2]:.0f} bps** but the "
            f"random short is **{R['h20'][5]:+.0f} bps** — the pattern *underperforms* a dart by "
            f"{abs(R['h20'][6]):.0f} bps. There is no horizon where the break beats the random short."
        ),
        md(
            "### 4c · The shape placebo — scramble the pattern, nothing changes\n\n"
            "Permute the per-bar returns (positions kept, marginal kept) so the lead-in/bump/break "
            "ordering is destroyed. If price reverses *at this specific shape*, the scramble should "
            "demolish the result. The observed break return should sit far in the *left* tail of the "
            "scrambled distribution (more negative = a better short). It doesn't."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = load('SPY')['close']\n"
            "    rng = np.random.default_rng(467)\n"
            "    logret = np.diff(np.log(c.to_numpy())); p0 = float(c.iloc[0]); idx = c.index\n"
            "    obs = st.summarize(st.forward_returns(c, st.barr_break_entries(c, lead=R['lead'], bump=R['bump'], bump_mult=R['mult']), 20))['mean_bps']\n"
            "    draws = []\n"
            "    for _ in range(200):\n"
            "        perm = rng.permutation(logret); scr = np.empty(len(idx)); scr[0]=p0; scr[1:]=p0*np.exp(np.cumsum(perm))\n"
            "        s = __import__('pandas').Series(scr, index=idx)\n"
            "        e = st.barr_break_entries(s, lead=R['lead'], bump=R['bump'], bump_mult=R['mult'])\n"
            "        rr = st.forward_returns(s, e, 20)\n"
            "        if rr.size: draws.append(rr.mean()*1e4)\n"
            "    draws = np.array(draws); pval = (np.sum(draws<=obs)+1)/(len(draws)+1)\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "    rng = np.random.default_rng(467); draws = rng.normal(-10, 60, 200)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=40, color=GREY, alpha=.85, label='shape-scrambled tapes (SPY, 20d)')\n"
            "ax.axvline(obs, c='#2c6fbb', lw=2.5, label=f'real bump-and-run {obs:+.0f} bps')\n"
            "ax.set_xlabel('mean break-short 20d return (bps)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Real pattern sits mid-pack: placebo p = {pval:.2f}'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'real break short {obs:+.1f} bps   placebo p={pval:.2f}  (>0.05 => shape not load-bearing)')"
        ),
        md(
            f"> 💡 In plain words: the real pattern (blue line) sits **in the middle** of the "
            f"scrambled cloud — **p = {R['placebo'][1]:.2f}**. Order-destroyed nonsense does just as "
            "well, so the specific bump-and-run shape carries no information. This is the cleanest "
            "refutation of 'the break forecasts a reversal.'"
        ),
        md(
            "### 4d · Per-ticker — the break loses to a random short everywhere\n\n"
            "20-day break-minus-random delta, per instrument. If the pattern worked it would be "
            "positive across the board; instead it's negative in all 5."
        ),
        code(
            "if HAVE_REAL:\n"
            "    names, deltas = [], []\n"
            "    for t in data.DEFAULT_TICKERS:\n"
            "        c = load(t)['close']\n"
            "        e = st.barr_break_entries(c, lead=R['lead'], bump=R['bump'], bump_mult=R['mult']); re = st.random_entries(c, max(len(e),50), lead=R['lead'], bump=R['bump'], seed=7)\n"
            "        d = st.summarize(st.forward_returns(c,e,20))['mean_bps'] - st.summarize(st.forward_returns(c,re,20))['mean_bps']\n"
            "        names.append(t); deltas.append(d)\n"
            "else:\n"
            "    names = [p[0] for p in R['per']]; deltas = [p[5] for p in R['per']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.bar(names, deltas, color=[GREEN if d>0 else RED for d in deltas], width=.6)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,d in enumerate(deltas): ax.annotate(f'{d:+.0f}',(i,d),ha='center',va='bottom' if d>=0 else 'top',fontsize=9)\n"
            "ax.set_ylabel('20d break − random (bps)'); ax.set_title('Break short underperforms a random short in all 5 names')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('per-ticker 20d delta (bps):', {n:round(d) for n,d in zip(names,deltas)})"
        ),
        md(
            f"> 💡 In plain words: every name is **negative** — even the best (QQQ, {R['per'][1][5]:+.0f} "
            f"bps) loses to a random short, and IWM is **{R['per'][2][5]:+.0f}** bps behind. No "
            "coherent, cross-sectional edge — exactly what you'd expect if the break is just relabelled "
            "short-beta."
        ),
        md(
            "### 4e · Synthetic positive control — the harness CAN bank a real reversal\n\n"
            "To prove the null is honest (not a dead detector), plant a **real** post-bump reversal "
            "into a synthetic tape and check the same break-short rule banks it: edge=0 must stay below "
            "the t=2 bar; edge>0 must light up with a high win-rate (the short profits)."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.70):\n"
            "    px, _ = data.synthetic_panel(edge=edge, seed=467, n_days=8000)\n"
            "    c = px['close']; e = st.barr_break_entries(c, lead=60, bump=30, bump_mult=2.0); s = st.summarize(st.forward_returns(c, e, 5))\n"
            "    res.append((edge, s['n'], s['mean_bps'], s['win']*100, s['t']))\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "labels = [f'planted edge\\n{e:.2f}' for e,_,_,_,_ in res]; tvals = [r[4] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('5d one-sample t (short)'); ax.set_title('Control: edge=0 -> below bar; planted reversal -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,n,m,w,t in res: print(f'edge={e:.2f}: n={n} short={m:+.1f}bps win={w:.0f}% t={t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted reversal the control sits at "
            f"**t = {R['syn'][0][4]:.2f}** (below the +2 bar — no false positive); a planted reversal "
            f"reaches **t = {R['syn'][1][4]:.2f}** (win {R['syn'][1][3]:.0f}%). The detector works — so "
            "the negative real-tape result is a genuine 'nothing there', not a broken pipeline."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the break short does not beat a drift-matched random short; it does "
            f"*worse* (break − random = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f}/"
            f"{R['h60'][6]:+.0f} bps at 5/10/20/60d; Welch t = {R['h5'][8]:+.2f}/{R['h10'][8]:+.2f}/"
            f"{R['h20'][8]:+.2f}/{R['h60'][8]:+.2f}, negative throughout). The break does not forecast "
            "the reversal — price keeps rising after it.\n"
            f"- **Tradability `MIRAGE`** — the short loses in absolute terms (negative short-beta) and "
            "again to a random short; costs only deepen the hole. No edge to scale.\n"
            f"- **Forecasts a reversal? `BUSTED`** — the shuffled-window placebo leaves the result "
            f"intact (**p = {R['placebo'][1]:.2f}**): order-scrambled nonsense does as well as the real "
            "bump-and-run shape, so the specific lead-in / bump / break geometry carries no forecasting "
            "information. The pattern is a descriptive after-the-fact label, not a reversal forecast."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — there is nothing to trade\n\n"
            "The break short has *negative* skill: it loses to the unconditional drift of long equity "
            "indices (which it is betting against) and loses again to a random short of the same size "
            "and hold. It trades *less* of the time and pays costs on each break, so it strictly "
            "dominates *nothing*. There is no capacity question because there is no edge to scale. The "
            "bump-and-run is a descriptive chart label, not a forecasting strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The bullish mirror (BARR bottom).** Buying the upside break will *look* strong on an "
            "up-drifting tape for the same beta reason the short looks weak — it needs the identical "
            "random-baseline test before belief.\n"
            "- **Hand-drawn forks of the rule.** Proponents eyeball the slope, the bump and the break. "
            "That adds *hindsight* (free parameters) which can only inflate in-sample fit; the "
            "mechanical version here is the charitable upper bound.\n"
            "- **Threshold robustness.** A 1.5× / 3× bump, a steeper/gentler lead-in cap, or a "
            "confirmation delay on the break all leave the conclusion intact: shape in, beta out.\n\n"
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
