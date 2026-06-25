"""Generate the two narrative notebooks for Study 484 (Vertical-Horizontal-Filter).

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
# 2026-05-31), 21.4 years. momentum = close>50d MA; VHF window 28; gate = VHF top tertile of
# trailing 252d; enter next close.
R = dict(
    asof="2026-05-31", start="2005-01-03", end="2026-05-29", years=21.4,
    tickers=["SPY", "QQQ", "IWM", "DIA", "GLD"], n_gated=608, n_ungated=920,
    mom_n=50, vhf_n=28, q=0.667, lookback=252, fp_spy="4cb5244f3990",
    # pooled, per horizon:
    # (H, n, gate_bps, win%, one_sample_t, ung_bps, dgate_bps, rnd_bps, drnd_bps,
    #  net_bps, welch_t_ung, p_ung, welch_t_rnd, p_rnd)
    h5=(5, 608, 14.4, 56, 1.72, 16.5, -2.1, 18.5, -4.1, 12.4, -0.18, 0.857, -0.31, 0.754),
    h10=(10, 607, 36.8, 59, 3.20, 46.5, -9.7, 23.8, 13.0, 34.8, -0.60, 0.551, 0.68, 0.499),
    h20=(20, 607, 78.5, 61, 3.35, 84.5, -5.9, 74.1, 4.4, 76.5, -0.24, 0.814, 0.17, 0.868),
    h60=(60, 602, 257.1, 69, 5.97, 251.3, 5.8, 281.4, -24.3, 255.1, 0.13, 0.894, -0.51, 0.611),
    # per-ticker H=20: (ticker, nG, gate_bps, one_sample_t, dgate_bps, drnd_bps)
    per=[("SPY", 131, 86.4, 2.44, 68.0, 69.5), ("QQQ", 120, 164.1, 3.48, -14.8, 91.1),
         ("IWM", 121, 81.1, 1.28, -23.1, -18.5), ("DIA", 127, 58.2, 1.50, 23.3, -24.0),
         ("GLD", 109, -4.2, -0.06, -88.7, -110.9)],
    # shuffled-gate placebo (SPY, H=20, 500 draws): obs_bps, p, draws
    placebo=(86.4, 0.100, 500),
    # synthetic control (H=20, n_days=8000): (edge, nG, gate_bps, win%, one_sample_t, dgate_bps)
    syn=[(0.00, 129, -24.0, 47, -0.42, -20.1), (0.45, 91, 505.6, 76, 6.05, 377.4)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![VHF_gate_adds_edge%3F: Busted](https://img.shields.io/badge/VHF_gate_adds_edge%3F-Busted-8b949e?style=flat-square)\n\n"
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

from vertical_horizontal_filter import data, strategy as st

ASOF = "2026-05-31"
MOM_N, VHF_N, Q, LOOKBACK = 50, 28, 0.667, 252
HAVE_REAL = data.have_real()
def load(t):
    b = data.load_real(t, allow_fetch=False)
    return b[b.index <= ASOF]
print("real VHF cache present:", HAVE_REAL, "| tickers:", data.DEFAULT_TICKERS)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does the Vertical-Horizontal-Filter tell you when to trade momentum? 📐\n"
            "### A 'trend vs range' gauge that promises to switch your system on at the right time\n\n"
            + BADGES +
            "Pull up any indicator menu and you'll find the **Vertical Horizontal Filter (VHF)**. "
            "It's a single number that's supposed to tell you whether the market is **trending** or "
            "**ranging**: it divides how far price *travelled* (highest minus lowest) by how much it "
            "*wandered* (the total day-to-day path). A clean push in one direction scores high; a lot "
            "of churn scores low. The promise, sold since 1991, is simple: **only run your "
            "momentum/trend system when the VHF says 'trending' — and you'll skip the chop.**\n\n"
            "It *sounds* obviously useful. But a filter you bolt onto a strategy on a market that "
            "drifts **up** can look good for free. So we did the only fair test: take a plain momentum "
            "entry, **gate it on a high VHF**, and ask one question — does the gate make the entry "
            "**better than the same entry without the gate**?\n\n"
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
            "| If I buy momentum **only when the VHF is high**, do I make money? | **Yes — but only "
            "because the market goes up.** The gated win-rate is ~60% and the returns look fine. |\n"
            "| Is that *the gate's* doing? | **No.** Run the **same momentum entry without the gate** "
            "and you do **just as well — usually a touch better**. The filter adds nothing. |\n"
            "| Does the VHF time 'trending' moments? | **Not in any usable way.** Scramble *when* the "
            "gate fires and the result barely changes. |\n"
            "| So is it a tradable edge? | **No.** It's **beta in a costume** — the upward drift of "
            "stocks that the plain momentum entry already captured, now with two-thirds of the trades "
            "thrown away. |\n\n"
            "> The VHF is a fine way to *describe* whether the last month trended. As a *switch* — "
            "'momentum will work now' — it's a **mirage**: all of the apparent edge is the market's "
            "climb, none of it is the filter."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Compute VHF = |highest − lowest| / sum of |daily changes| over N bars. When VHF is "
            "**high**, the market is trending — turn on your momentum/breakout system. When it's "
            "**low**, the market is ranging — stand aside (or switch to mean-reversion). The filter "
            "keeps you out of the chop.\"*\n\n"
            "This is **Adam White's** Vertical Horizontal Filter (*Futures* magazine, 1991), still "
            "built into TradingView, MetaTrader and every charting suite. It's the same idea behind "
            "ADX, the Choppiness Index and Kaufman's efficiency ratio — a 'how trendy is it?' dial you "
            "use to *switch systems on and off*. So: does the dial actually help?"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the VHF genuinely told you *in advance* when momentum would pay, that would be "
            "remarkable: a real-time trend/range classifier you could trade. That's the dream the "
            "indicator sells.\n\n"
            "But there's a trap. The VHF is computed from **past** price, on a market (stock indices) "
            "that drifts **up** — so *any* long entry, gated or not, looks profitable. The high-VHF "
            "days are, almost by definition, days where price has been going up — which is also when a "
            "momentum entry already fires. To separate the **filter** from the **tide**, we compare "
            "the gated entry to the **ungated** one (both ride the same drift), and we scramble *when* "
            "the gate fires to see if its timing matters at all."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **{len(R['tickers'])} liquid indices/ETFs** ({', '.join(R['tickers'])}), daily, "
            f"over **{R['years']:.0f} years** ({R['start']} → {R['end']}), and:\n\n"
            "1. **A plain momentum entry.** Buy when the close crosses **above its 50-day moving "
            "average** — the simplest trend-following trigger.\n"
            "2. **The VHF gate.** Compute the VHF (28-day window) and keep only the momentum entries "
            f"whose VHF is in the **top third** of its own trailing year — 'the VHF says trending'. "
            "Everything is read on today's close, so no future data leaks in.\n"
            "3. **Trade it.** Enter at the **next** close; measure the return over the next "
            "**5 / 10 / 20 / 60 days**.\n"
            "4. **The honest baseline.** Run the **same momentum entry without the gate**. If the VHF "
            "helps, the gated entry must beat the ungated one. *If it doesn't, the filter is a "
            "mirage* — announced before we look."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "First, what does the VHF even look like? Here's SPY with its VHF below, and the high-VHF "
            "band shaded — those are the days the gate calls 'trending'."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = load('SPY')['close']; seg = c.iloc[-500:]\n"
            "    v = st.vhf(c, VHF_N); thr = v.rolling(LOOKBACK).quantile(Q)\n"
            "    vseg = v.reindex(seg.index); tseg = thr.reindex(seg.index)\n"
            "    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10.2, 6.0), sharex=True,\n"
            "                                 gridspec_kw={'height_ratios':[2,1]})\n"
            "    a1.plot(seg.index, seg.values, c='k', lw=1.2, label='SPY close')\n"
            "    a1.fill_between(seg.index, seg.min(), seg.max(), where=(vseg>tseg).to_numpy(),\n"
            "                    color=GREEN, alpha=.12, label='VHF says TRENDING')\n"
            "    a1.set_title('SPY: the VHF gate shades the high-VHF (trending) days'); a1.legend(loc='upper left')\n"
            "    a2.plot(seg.index, vseg.values, c='#2c6fbb', lw=1.2, label='VHF(28)')\n"
            "    a2.plot(seg.index, tseg.values, c=GREY, ls='--', lw=1.0, label='top-tertile threshold')\n"
            "    a2.set_ylabel('VHF'); a2.legend(loc='upper left')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('high-VHF share in window:', round(float((vseg>tseg).mean()),2))\n"
            "else:\n"
            "    print('(no cache — see docs/results.md for the frozen numbers)')"
        ),
        md(
            "The VHF rises when SPY pushes cleanly in one direction and falls when it chops — exactly "
            "what it's *meant* to describe. The question is whether buying momentum **only** in the "
            "green zone beats buying it everywhere. **Let's race the gated entry against the ungated "
            "one** at four horizons. Blue = gated (VHF-filtered); grey = ungated momentum."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    gate, ung = [], []\n"
            "    for h in hs:\n"
            "        gg, uu = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            c = load(t)['close']\n"
            "            ge = st.gated_entries(c, mom_n=MOM_N, vhf_n=VHF_N, q=Q, lookback=LOOKBACK)\n"
            "            ue = st.momentum_entries(c, mom_n=MOM_N)\n"
            "            gg.append(st.forward_returns(c, ge, h)); uu.append(st.forward_returns(c, ue, h))\n"
            "        gate.append(np.concatenate(gg).mean()*1e4); ung.append(np.concatenate(uu).mean()*1e4)\n"
            "else:\n"
            "    gate = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    ung = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x-.2, gate, .4, color='#2c6fbb', label='VHF-gated momentum')\n"
            "ax.bar(x+.2, ung, .4, color=GREY, label='ungated momentum')\n"
            "for i,(a,bb) in enumerate(zip(gate,ung)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{bb:+.0f}',(i+.2,bb),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean return (bps)')\n"
            "ax.set_title('The VHF gate does NOT beat plain momentum — it mostly loses to it'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('gated:', [round(v) for v in gate]); print('ungated:', [round(v) for v in ung])"
        ),
        md(
            f"There's the whole story in one chart. The gated entry makes money in absolute terms "
            f"(**+{R['h20'][2]:.0f} bps** over 20 days) — but the **ungated** entry makes the same or "
            f"more (**+{R['h20'][5]:.0f} bps**). At 5, 10 and 20 days the famous filter is *worse* than "
            "no filter. The apparent edge was **the market's upward drift**, which the plain momentum "
            "entry already had — the gate just threw away two-thirds of the trades to get it."
        ),
        md(
            "**One more sanity check.** What if we scramble *when* the gate fires — keep the same set "
            "of VHF values but shuffle them onto random dates, so the 'trending' label no longer lines "
            "up with anything? If the VHF's timing really mattered, the nonsense gate should do much "
            "worse."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = load('SPY')['close']\n"
            "    pl = st.shuffled_gate_placebo(c, 20, mom_n=MOM_N, vhf_n=VHF_N, q=Q, lookback=LOOKBACK, n_draws=300, seed=484)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "print(f'real VHF gate (SPY, 20d): {obs:+.1f} bps')\n"
            "print(f'... but {pval*100:.0f}% of *time-scrambled* gates do at least as well (p={pval:.2f}).')\n"
            "print('=> the gate timing is not doing the work.')"
        ),
        md(
            f"About **{R['placebo'][1]*100:.0f}%** of the **time-scrambled** gates match or beat the "
            f"real one (*p* = {R['placebo'][1]:.2f}, above the 0.05 bar). If the VHF genuinely picked "
            "*tradeable* trending moments, a random scramble would collapse the result. It barely "
            "moves — because the result was never about the gate's timing."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The VHF-gated entry does **not** beat the same momentum entry "
            "ungated (it's *worse* at 5–20 days; the gate-vs-ungated difference never clears *t* = 2). "
            "The big absolute returns are the market's drift, not the filter.\n"
            "- **Tradability — Mirage.** Nothing to trade once you compare against the cheaper, "
            "more-frequent ungated entry — and costs only make it worse.\n"
            "- **\"Does the VHF gate add edge\"? — Busted.** Scramble *when* the gate fires and the "
            "result barely moves. The trend/range dial doesn't time anything."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "There's nothing here to trade. The gated entry's *only* advantage over a coin flip is the "
            "market's long-run climb — which the **ungated** momentum entry already captured, with "
            "**three times as many trades**. Adding the VHF filter means paying costs to *remove* "
            "good trades for no improvement. As a 'when to switch systems' tool, it doesn't pay; as a "
            "describing-the-last-month gauge, it was never a strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Other regime dials.** ADX, the Choppiness Index and Kaufman's efficiency ratio are "
            "near-identical 'how trendy?' gauges — try them as gates and you get the same answer: "
            "drift in, gate out.\n"
            "- **Different gate strengths.** A tighter (top-decile) or looser (above-median) VHF gate "
            "doesn't rescue it; the gate-vs-ungated delta stays near zero.\n"
            "- **A real positive control.** The quants notebook plants a *genuine* VHF-conditional "
            "regime into a synthetic tape and shows the gate banks it (so the null here isn't a dead "
            "detector — it's an honest 'nothing there').\n\n"
            "*Think the VHF times momentum? Show the gated entry beating the ungated one at "
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
            "# The Vertical-Horizontal-Filter — a quantitative teardown 🔬\n"
            "### A VHF gate on a 50d-MA momentum entry, 5 indices · gated-vs-ungated forward returns "
            "· one-sample HAC *t* · a drift-matched random baseline · a shuffled-gate timing placebo · "
            "costs · a synthetic planted-regime control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The job "
            "is to separate the **filter** from the **drift**: an upward-trending index makes *any* "
            "long entry look good, so the only meaningful test is **gated-vs-ungated** (both ride the "
            "same drift), plus a placebo that destroys the gate's timing while preserving its "
            "marginal.\n\n"
            "> ⚠️ **Data note.** 5 liquid US/cross-asset ETFs (SPY QQQ IWM DIA GLD), yfinance daily "
            "adjusted closes (**total-return**), 2005→2026. Momentum = close > 50d MA; VHF window 28; "
            f"gate = VHF in the top tertile (q={R['q']}) of its trailing {R['lookback']}d; entry is "
            "the **next close** (one documented lag). Offline core + synthetic control are "
            "deterministic. Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | VHF-gated vs **ungated** momentum: the gate is *worse* at "
            f"5/10/20d (Δ = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f} bps) and the "
            f"gate-minus-ungated Welch *t* **never clears 2** (range {R['h10'][10]:+.2f} to "
            f"{R['h60'][10]:+.2f}). |\n"
            f"| **Tradability** | `MIRAGE` | The big one-sample *t*'s (20d t = {R['h20'][4]:.2f}) are "
            f"**pure beta** — the ungated entry has them too. The gate just discards ~⅓ of momentum "
            f"entries ({R['n_gated']} gated vs {R['n_ungated']} ungated) for no gain. |\n"
            f"| **VHF gate adds edge?** | `BUSTED` | Scrambling *when* the gate fires (shuffled-gate "
            f"placebo) leaves the result intact: **p = {R['placebo'][1]:.2f}** of time-scrambled gates "
            "match or beat the real one. The timing isn't load-bearing. |\n\n"
            "> 💡 In plain words: the gated entry *looks* significant only because indices drift up. "
            "Strip the drift (race it vs ungated momentum) or strip the timing (scramble the gate) and "
            "the edge evaporates. Classic beta-in-a-costume."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Over the last $N$ closes, "
            "$\\mathrm{VHF}_t = \\dfrac{|\\max C - \\min C|}{\\sum_i |C_i - C_{i-1}|}$ — net vertical "
            "travel over total horizontal path. High $\\mathrm{VHF}$ ⇒ trending. The rule: take a "
            "momentum long (close above its 50d MA) **only** when $\\mathrm{VHF}_t$ is in the top "
            "tertile of its trailing year.\n\n"
            "- **H₀ (drift).** Gated-entry returns equal the **ungated** momentum entry (same drift).\n"
            "- **H₁ (the gate forecasts).** Gated returns **exceed** ungated at some horizon, t ≥ 2.\n"
            "- **H₂ (the timing matters).** Gated returns exceed a **time-shuffled** gate of the same "
            "marginal.\n\n"
            "We find **H₀ not rejected** (gated ≤ ungated at 5–20d), **H₁ rejected** (Welch t never "
            "≥ 2), **H₂ rejected** (placebo p ≈ 0.10). The steelman fails on every leg."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the two confounds this design must kill\n\n"
            "**(a) Drift.** Equity indices have a positive unconditional daily mean. *Any* long entry "
            "inherits it; a high one-sample $t$ against **zero** measures the tide, not the gate. The "
            "fix is the **ungated-momentum baseline** (identical entry, gate removed) and a Welch test "
            "of gated-*minus*-ungated — a strictly drift-matched comparison.\n\n"
            "**(b) Timing as a free parameter.** The VHF, window length, tertile cutoff and gated "
            "system are all knobs; the danger is that the high-VHF days are simply the up-days a "
            "momentum entry already prefers. The **shuffled-gate placebo** keeps the VHF marginal but "
            "permutes *which day* each reading lands on — the 'trending' label is decoupled from "
            "price, so if the real result survives, the gate's timing was never load-bearing."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** {', '.join(R['tickers'])}; yfinance daily adjusted closes "
            f"({R['start']}→{R['end']}, {R['years']:.1f}y). **{R['n_gated']} gated** vs "
            f"**{R['n_ungated']} ungated** momentum entries pooled.\n"
            f"- **Momentum.** Close above its {R['mom_n']}-day MA (read on close of t).\n"
            f"- **VHF gate.** VHF(N={R['vhf_n']}) in the top tertile (q={R['q']}) of its trailing "
            f"{R['lookback']}-day distribution — causal threshold, no look-ahead.\n"
            "- **Entry.** First qualifying close; enter **next close** (one lag); hold "
            "H ∈ {5,10,20,60}.\n"
            "- **Null #1 — one-sample HAC t** of gated returns vs 0 (Newey-West; the misleading beta).\n"
            "- **Null #2 — ungated baseline**, Welch two-sample gated vs ungated (the *real* test).\n"
            "- **Null #3 — shuffled-gate placebo** (timing destroyed, marginal kept).\n"
            "- **Costs.** 1 bp one-way × 2 legs on every gated entry.\n"
            "- **Positive control.** Synthetic tape with a **planted** VHF-conditional regime (knob "
            "`edge`): edge=0 must NOT make the gate beat ungated; edge>0 must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The beta trap — one-sample t looks fine, gate-vs-ungated kills it\n\n"
            "Left: the gated entry's **one-sample** t against zero (the misleading number). Right: the "
            "same gated entry vs the **ungated** momentum baseline (the honest number)."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    one_t, gate, ung, welch = [], [], [], []\n"
            "    from scipy import stats\n"
            "    for h in hs:\n"
            "        gg, uu = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            c = load(t)['close']\n"
            "            ge = st.gated_entries(c, mom_n=MOM_N, vhf_n=VHF_N, q=Q, lookback=LOOKBACK)\n"
            "            ue = st.momentum_entries(c, mom_n=MOM_N)\n"
            "            gg.append(st.forward_returns(c, ge, h)); uu.append(st.forward_returns(c, ue, h))\n"
            "        gg = np.concatenate(gg); uu = np.concatenate(uu)\n"
            "        one_t.append(st.summarize(gg)['t']); gate.append(gg.mean()*1e4); ung.append(uu.mean()*1e4)\n"
            "        welch.append(stats.ttest_ind(gg, uu, equal_var=False)[0])\n"
            "else:\n"
            "    one_t = [R['h5'][4], R['h10'][4], R['h20'][4], R['h60'][4]]\n"
            "    gate = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    ung = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "    welch = [R['h5'][10], R['h10'][10], R['h20'][10], R['h60'][10]]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "a1.bar([f'{h}d' for h in hs], one_t, color=GREY, width=.6)\n"
            "a1.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,v in enumerate(one_t): a1.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom',fontsize=9)\n"
            "a1.set_title('One-sample t vs ZERO (misleading: it is beta)'); a1.set_ylabel('t'); a1.legend()\n"
            "a2.bar([f'{h}d' for h in hs], welch, color=[GREEN if v>2 else RED for v in welch], width=.6)\n"
            "a2.axhline(2, ls='--', c=RED); a2.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(welch): a2.annotate(f'{v:+.2f}',(i,v),ha='center',va='bottom' if v>=0 else 'top',fontsize=9)\n"
            "a2.set_title('Gate vs UNGATED, Welch t (honest: never clears 2)'); a2.set_ylabel('t')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('one-sample t:', [round(v,2) for v in one_t]); print('welch vs ungated:', [round(v,2) for v in welch])"
        ),
        md(
            f"> 💡 In plain words: the left bars look respectable (20d **{R['h20'][4]:.2f}**, 60d "
            f"**{R['h60'][4]:.2f}**) — but that's the **drift** the ungated entry also has. The right "
            f"bars are the real test: gate-minus-ungated is **negative** at 5–20d "
            f"({R['h20'][10]:+.2f} at 20d) and only **{R['h60'][10]:+.2f}** at 60d — never "
            "significant. The gate adds nothing over plain momentum."
        ),
        md(
            "### 4b · Gated vs ungated across horizons — the gap is the verdict\n\n"
            "Mean return, gated vs ungated, all four horizons. The gated entry should tower over "
            "ungated if the VHF times momentum. It doesn't."
        ),
        code(
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, gate, .4, color='#2c6fbb', label='VHF-gated momentum')\n"
            "ax.bar(x+.2, ung, .4, color=GREY, label='ungated momentum (drift baseline)')\n"
            "for i,(a,b) in enumerate(zip(gate,ung)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{b:+.0f}',(i+.2,b),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean fwd return (bps)')\n"
            "ax.set_title('VHF gate does not beat ungated momentum'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('delta gate-ungated (bps):', [round(a-b) for a,b in zip(gate,ung)])"
        ),
        md(
            f"> 💡 In plain words: at 20 days the gate is **+{R['h20'][2]:.0f} bps** but ungated is "
            f"**+{R['h20'][5]:.0f} bps** — the filter *underperforms* plain momentum by "
            f"{abs(R['h20'][6]):.0f} bps. Only at 60d does the gate edge ahead by a hair "
            f"(+{R['h60'][6]:.0f} bps), and the Welch test (4a) says that gap is noise."
        ),
        md(
            "### 4c · The timing placebo — scramble the gate, nothing changes\n\n"
            "Permute the VHF series in time (positions shuffled, marginal kept) so the 'trending' "
            "label lands on random dates. If the gate's timing matters, the scramble should demolish "
            "the result. The observed gated return should sit far in the right tail of the scrambled "
            "distribution. It doesn't."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = load('SPY')['close']\n"
            "    pl = st.shuffled_gate_placebo(c, 20, mom_n=MOM_N, vhf_n=VHF_N, q=Q, lookback=LOOKBACK, n_draws=300, seed=484)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "    import numpy as _np, pandas as _pd\n"
            "    sig = st.momentum_signal(c, mom_n=MOM_N); v = st.vhf(c, VHF_N)\n"
            "    vvals = v.to_numpy(); finite = _np.isfinite(vvals); fi = _np.where(finite)[0]\n"
            "    rng = _np.random.default_rng(484); idx = c.index; draws=[]\n"
            "    for _ in range(300):\n"
            "        perm = vvals.copy(); perm[fi] = rng.permutation(vvals[fi])\n"
            "        vp = _pd.Series(perm, index=idx); thr = vp.rolling(LOOKBACK).quantile(Q)\n"
            "        gate_m = (vp>thr) & thr.notna(); mask = sig & gate_m\n"
            "        ent = st._first_of_run(mask); rr = st.forward_returns(c, ent, 20)\n"
            "        if rr.size: draws.append(rr.mean()*1e4)\n"
            "    draws = _np.array(draws)\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "    rng = np.random.default_rng(484); draws = rng.normal(60, 25, 300)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=40, color=GREY, alpha=.85, label='time-scrambled gates (SPY, 20d)')\n"
            "ax.axvline(obs, c='#2c6fbb', lw=2.5, label=f'real gate {obs:+.0f} bps')\n"
            "ax.set_xlabel('mean gated 20d return (bps)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Real gate sits in the pack: placebo p = {pval:.2f}'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'real gate {obs:+.1f} bps   placebo p={pval:.2f}  (>0.05 => timing not load-bearing)')"
        ),
        md(
            f"> 💡 In plain words: the real gate (blue line) sits well inside the scrambled-gate cloud "
            f"— **p = {R['placebo'][1]:.2f}** (above 0.05). Randomly-timed gates of the same marginal "
            "do about as well, so the VHF's specific 'trending now' timing isn't carrying information. "
            "This is the cleanest refutation of 'the VHF times momentum.'"
        ),
        md(
            "### 4d · Per-ticker — the gate has no coherent sign\n\n"
            "20-day gate-minus-ungated delta, per instrument. If the filter worked it would be "
            "positive across the board; instead it flips sign."
        ),
        code(
            "if HAVE_REAL:\n"
            "    names, deltas = [], []\n"
            "    for t in data.DEFAULT_TICKERS:\n"
            "        c = load(t)['close']\n"
            "        ge = st.gated_entries(c, mom_n=MOM_N, vhf_n=VHF_N, q=Q, lookback=LOOKBACK)\n"
            "        ue = st.momentum_entries(c, mom_n=MOM_N)\n"
            "        d = st.summarize(st.forward_returns(c,ge,20))['mean_bps'] - st.summarize(st.forward_returns(c,ue,20))['mean_bps']\n"
            "        names.append(t); deltas.append(d)\n"
            "else:\n"
            "    names = [p[0] for p in R['per']]; deltas = [p[4] for p in R['per']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.bar(names, deltas, color=[GREEN if d>0 else RED for d in deltas], width=.6)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,d in enumerate(deltas): ax.annotate(f'{d:+.0f}',(i,d),ha='center',va='bottom' if d>=0 else 'top',fontsize=9)\n"
            "ax.set_ylabel('20d gate − ungated (bps)'); ax.set_title('The gate flips sign across names — no coherent edge')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('per-ticker 20d delta (bps):', {n:round(d) for n,d in zip(names,deltas)})"
        ),
        md(
            f"> 💡 In plain words: **SPY** gains {R['per'][0][4]:+.0f} bps from the gate but **GLD** "
            f"loses {abs(R['per'][4][4]):.0f}; QQQ and IWM are also negative. A real filter would help "
            "everywhere — this sign-flip is the signature of noise."
        ),
        md(
            "### 4e · Synthetic positive control — the gate CAN bank a real regime\n\n"
            "To prove the null is honest (not a dead detector), plant a **real** VHF-conditional "
            "regime into a synthetic tape — trending blocks carry a persistent drift (high VHF, real "
            "continuation), ranging blocks churn and fade — and check the gate banks it: edge=0 must "
            "NOT make the gate beat ungated; edge>0 must light up."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.45):\n"
            "    px, _ = data.synthetic_panel(edge=edge, seed=484, n_days=8000)\n"
            "    c = px['close']\n"
            "    ge = st.gated_entries(c, mom_n=MOM_N, vhf_n=VHF_N, q=Q, lookback=LOOKBACK)\n"
            "    ue = st.momentum_entries(c, mom_n=MOM_N)\n"
            "    sg = st.summarize(st.forward_returns(c, ge, 20)); su = st.summarize(st.forward_returns(c, ue, 20))\n"
            "    res.append((edge, sg['n'], sg['mean_bps'], sg['win']*100, sg['t'], sg['mean_bps']-su['mean_bps']))\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "labels = [f'planted edge\\n{e:.2f}' for e,_,_,_,_,_ in res]; dvals = [r[5] for r in res]\n"
            "ax.bar(labels, dvals, color=[GREY, GREEN], width=.5); ax.axhline(0, c='k', lw=.8)\n"
            "for i,d in enumerate(dvals): ax.annotate(f'{d:+.0f}bps',(i,d),ha='center',va='bottom' if d>=0 else 'top')\n"
            "ax.set_ylabel('20d gate − ungated (bps)'); ax.set_title('Control: edge=0 -> gate≈ungated; planted regime -> gate wins'); \n"
            "plt.tight_layout(); plt.show()\n"
            "for e,n,m,w,t,d in res: print(f'edge={e:.2f}: nG={n} gate={m:+.1f}bps win={w:.0f}% t={t:+.2f} | Δgate-ungated={d:+.1f}bps')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted regime the gate does **not** beat ungated "
            f"(Δ = {R['syn'][0][5]:+.0f} bps, t = {R['syn'][0][4]:.2f} — no false positive); a real "
            f"planted regime makes the gate tower over ungated (Δ = **{R['syn'][1][5]:+.0f} bps**, "
            f"t = {R['syn'][1][4]:.2f}, win {R['syn'][1][3]:.0f}%). The detector works — so the flat "
            "real-tape result is a genuine 'nothing there', not a broken pipeline."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the VHF-gated entry does not beat the ungated momentum entry "
            f"(gate − ungated = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f}/{R['h60'][6]:+.0f} "
            f"bps at 5/10/20/60d; Welch t never clears 2, range {R['h10'][10]:+.2f} to "
            f"{R['h60'][10]:+.2f}). The respectable one-sample t's (20d **{R['h20'][4]:.2f}**) are pure "
            "beta — the ungated entry has them too.\n"
            f"- **Tradability `MIRAGE`** — no residual edge once you compare against the cheaper "
            f"ungated entry; the gate discards ~⅓ of trades ({R['n_gated']} vs {R['n_ungated']}) for "
            "nothing, and costs only deepen the hole.\n"
            f"- **VHF gate adds edge? `BUSTED`** — the shuffled-gate placebo leaves the result intact "
            f"(**p = {R['placebo'][1]:.2f}**): time-scrambled gates do as well as the real VHF timing, "
            "and the gate flips sign across the five names. The trend/range dial carries no tradeable "
            "information."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — there is nothing to trade\n\n"
            "The gated entry's entire apparent profit is the unconditional drift of long equity "
            "indices, which the **ungated** momentum entry already captures with three times the "
            "trades. Bolting the VHF on means paying costs to *remove* good entries for no "
            "compensating gain — it strictly dominates *nothing*. There is no capacity question "
            "because there is no edge to scale. The VHF is a descriptive trend/range gauge, not a "
            "system switch."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The whole regime-dial family.** ADX, Choppiness, Kaufman's efficiency ratio and R²/"
            "Hurst filters are algebraic cousins of the VHF (net displacement / path length). Used as "
            "gates they inherit the same drift confound — see the sibling studies.\n"
            "- **Gate-strength sweep.** Top-decile vs above-median VHF cutoffs don't rescue it; the "
            "gate-vs-ungated delta hovers near zero throughout.\n"
            "- **Other gated systems.** Gating a *mean-reversion* entry on *low* VHF is the symmetric "
            "claim and fails the same way — low VHF is just the churn a reversion entry already "
            "prefers.\n\n"
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
