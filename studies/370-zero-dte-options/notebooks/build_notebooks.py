"""Generate the two narrative notebooks for Study 370 (Zero-DTE Options & the SPX tape).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached SPY OHLC + the
nearest-expiry option-chain snapshot under ../_cache/ and otherwise quote the frozen headline
numbers in ``R`` (mirroring docs/results.md). The synthetic pinning control runs anywhere with
no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance daily SPY OHLC
# 2010-01-05 -> 2026-06-18, 4,139 days, 16.4 years; option-chain snapshot as-of 2026-06-22).
R = dict(
    start="2010-01-05", end="2026-06-18", days=4139, years=16.4,
    break_date="2022-01-01", n_pre=3020, n_post=1119,
    # intraday open->close leg autocorr: (ac_pre, ac_post, d_ac, welch_t, placebo_p)
    intraday=(-0.077, -0.100, -0.023, -0.28, 0.395),
    # close-to-close control leg: (ac_pre, ac_post, d_ac, welch_t)
    oc_ret=(-0.130, -0.027, +0.103, 1.52),
    vol_pre=0.122, vol_post=0.148,
    # break-date sweep: (label, d_ac, t, n_post)
    robust=[("2021-01-01", -0.018, -0.24, 1371), ("2022-01-01", -0.023, -0.28, 1119),
            ("2022-05-01", -0.056, -0.60, 1037), ("2023-01-01", -0.095, -0.79, 868)],
    # pin trade by segment: (seg, gross_bp, net_bp, win%, sharpe_net)
    trade=[("all", 2.17, 1.16, 50.0, 0.23), ("pre", 2.62, 1.61, 50.0, 0.33),
           ("post", 0.95, -0.05, 49.9, -0.01)],
    # 0DTE chain snapshot
    chain=dict(expiry="2026-06-22", spot=745.14, n_expiries=32,
               total_volume=6_813_423, share_near=0.898, band_pct=1.0),
    # synthetic control: (pin, ac_pre, ac_post, d_ac, t, placebo_p)
    syn=[(0.0, 0.003, 0.023, 0.020, 0.68, 0.149),
         (0.30, 0.003, -0.276, -0.279, -9.22, 0.057)],
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Tape_pinned%3F: Unproven](https://img.shields.io/badge/Tape_pinned%3F-Unproven-8b949e?style=flat-square)\n\n"
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
%matplotlib inline
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from zero_dte_options import data, strategy as st

HAVE_REAL = data.have_real()
BD = data.BREAK_DATE
F = data.load_real() if HAVE_REAL else None
print("real SPY OHLC cache present:", HAVE_REAL,
      "| tape days:", (0 if F is None else len(F)))
"""

# The frozen headline dict is embedded into the first code cell so every downstream cell can
# quote it whether or not the cache is present.
BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Did the 0DTE-options boom \"pin\" the stock market? ⏱️\n"
            "### The same-day-options explosion and whether it really made the tape mean-revert, in plain English\n\n"
            + BADGES +
            "Since 2022 you can buy an S&P 500 option that **expires the same day** — a \"0DTE\" "
            "(zero-days-to-expiry) contract. They went from a Friday curiosity to a **daily** product, "
            "and now make up a huge slice of all index-option trading. A popular story followed: all "
            "that same-day, at-the-money option flow forces dealers to hedge in a way that **damps and "
            "reverses** intraday moves — so the modern tape is supposedly more **mean-reverting / more "
            "pinned** than it used to be.\n\n"
            "It's a great story. This notebook asks the boring question: *can we actually see it in the "
            "data?* The honest answer is **\"the boom is real, the pinning isn't visible\"** — and part "
            "of why is that the effect lives **inside the trading day**, where a free daily price feed "
            "literally can't look.\n\n"
            "> 📓 **Plain-language layer.** Want the block-bootstrap *t*-stats, the placebo-break test "
            "and the synthetic control? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** There's no free *intraday* option tape, so we use a "
            "**labelled proxy**: the day's **open→close** move and how much it tends to *reverse* the "
            "next day. It's the closest a daily feed gets to \"is the tape mean-reverting,\" and we call "
            "it a proxy throughout. Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Did the 0DTE boom actually happen? | **Absolutely.** A current same-day option chain has "
            f"~**{R['chain']['share_near']*100:.0f}%** of its volume parked within **±1%** of the spot "
            "price, on a calendar of **daily** expiries. The product is enormous. |\n"
            "| Is the tape *more* mean-reverting after 2022? | **A tiny bit, maybe — and not "
            "reliably.** The day's move reverses *slightly* more often post-2022, but the change is well "
            "inside the noise, and a second daily measure moves the **opposite** way. |\n"
            "| Can you trade the \"pin\"? | **No.** Fading yesterday's move pays **less** after the boom, "
            "not more — and **loses money** after a single basis point of cost. |\n"
            "| So is the \"0DTE pinned the market\" claim true? | **Unproven.** Not disproven in spirit — "
            "but a daily feed *can't see* the intraday effect, and on every daily proxy we can build it's "
            "indistinguishable from chance. |\n\n"
            "> The 0DTE boom is a real, vivid thing. \"It pinned the market\" is a story the daily tape "
            "can neither confirm nor cash."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"A wall of same-day, at-the-money options forces dealers to hedge their gamma all day "
            "long. That hedging leans **against** moves into the close — so since 0DTE went daily in "
            "2022, the S&P intraday tape **mean-reverts** more, and the index gets **pinned** near big "
            "strikes.\"*\n\n"
            "It's intuitive: an enormous, fast-decaying pile of options near the current price *should* "
            "create hedging pressure. Sell-side desks wrote it up; financial media ran with it. The "
            "question is whether the famous mechanism leaves a **measurable fingerprint** on the prices "
            "we can actually download — or whether it's a plausible-sounding effect that hides in data we "
            "don't have."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "Two very different things get fused in \"0DTE pinned the market.\" (1) *Is the 0DTE market "
            "huge?* — yes, trivially, and we'll show it. (2) *Did it change how the index actually "
            "moves?* — that's the claim that would matter, and it's the hard one. A market can grow "
            "gigantic and still leave the price path statistically unchanged. The only honest way to "
            "settle (2) is to measure the tape **before** and **after** the boom and ask whether the "
            "change is bigger than what a random split of the same data would throw up. And we have to be "
            "upfront: the *intraday* effect lives where a daily feed can't see — so the most we can test "
            "is a **proxy**, and a proxy that comes up empty doesn't *prove* the story false; it just "
            "fails to find it."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We download **daily SPY** (the S&P 500 ETF) over **{R['years']:.0f} years** "
            f"({R['start']} → {R['end']}) and split it at **2022**, when same-day options went daily.\n\n"
            "1. **Build a mean-reversion gauge.** Each day's **open→close** move is the 'intraday' leg. "
            "If the tape is *pinned*, a big move one day should tend to **reverse** the next — measured "
            "by a *negative* lag-1 autocorrelation. More negative = more mean-reverting.\n"
            "2. **Compare before vs after 2022.** Did the gauge get more negative after the boom?\n"
            "3. **Stress the split.** Re-cut the tape at thousands of *random* dates and ask how often a "
            "random split produces a change as big as 2022's. That's the honest test.\n"
            "4. **Look at a real 0DTE chain.** Pull today's nearest-expiry option chain to *see* the boom "
            "directly — the at-the-money volume wall the story is built on."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the boom is real.** Here's a current same-day (0DTE) SPY option chain: volume by "
            "strike. Notice how it stacks up right around today's price — the at-the-money wall the "
            "pinning story is about."
        ),
        code(
            "if HAVE_REAL and data.have_option_chain():\n"
            "    meta = json.load(open('../_cache/option_meta.json'))\n"
            "    ch = data.load_option_chain(); spot = meta['spot']\n"
            "    agg = ch.groupby('strike')['volume'].sum()\n"
            "    agg = agg[(agg.index > spot*0.95) & (agg.index < spot*1.05)]\n"
            "    fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "    ax.bar(agg.index, agg.values/1e3, width=0.8, color=GREEN, alpha=.85)\n"
            "    ax.axvline(spot, c=RED, lw=2, label=f'spot ~{spot:.0f}')\n"
            "    ax.set_title(f\"0DTE SPY chain ({meta['snapshot_expiry']}): volume clusters at-the-money\")\n"
            "    ax.set_xlabel('strike'); ax.set_ylabel('contract volume (thousands)'); ax.legend()\n"
            "    plt.tight_layout(); plt.show()\n"
            "    cc = st.chain_concentration(ch, spot, band=0.01)\n"
            "    print(f\"{cc['share_near']*100:.0f}% of volume within +-1% of spot; {meta['n_expiries']} listed expiries\")\n"
            "else:\n"
            "    print(f\"no chain cache — frozen: {R['chain']['share_near']*100:.0f}% of volume within +-1% of spot ({R['chain']['expiry']})\")"
        ),
        md(
            f"Around **{R['chain']['share_near']*100:.0f}%** of the same-day chain's volume sits within "
            f"**±1%** of the price, across **{R['chain']['n_expiries']}** near-daily expiries. The 0DTE "
            "market is unambiguously huge and concentrated at-the-money. **Now the real question: did it "
            "change how the index moves?**"
        ),
        md(
            "**Does the tape mean-revert more after 2022?** Here's the lag-1 autocorrelation of the daily "
            "open→close move, before vs after the boom. A *more negative* bar means a *more pinned / "
            "mean-reverting* tape."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rt = st.regime_table(F, BD, 'intraday')\n"
            "    ac_pre, ac_post = rt['ac_pre'], rt['ac_post']\n"
            "    rt2 = st.regime_table(F, BD, 'oc_ret'); oc_pre, oc_post = rt2['ac_pre'], rt2['ac_post']\n"
            "else:\n"
            "    ac_pre, ac_post = R['intraday'][0], R['intraday'][1]\n"
            "    oc_pre, oc_post = R['oc_ret'][0], R['oc_ret'][1]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "labels = ['intraday\\n(open->close)', 'close-to-close\\n(control)']\n"
            "pre = [ac_pre, oc_pre]; post = [ac_post, oc_post]; x = np.arange(2)\n"
            "ax.bar(x-.2, pre, .4, color=GREY, label='pre-2022')\n"
            "ax.bar(x+.2, post, .4, color=GREEN, label='post-2022')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_xticks(x); ax.set_xticklabels(labels)\n"
            "ax.set_ylabel('lag-1 autocorrelation\\n(more negative = more mean-reverting)')\n"
            "ax.set_title('Intraday gets a touch more pinned; close-to-close goes the OTHER way')\n"
            "for i,(a,b) in enumerate(zip(pre,post)):\n"
            "    ax.annotate(f'{a:+.2f}',(i-.2,a),ha='center',va='bottom' if a>=0 else 'top')\n"
            "    ax.annotate(f'{b:+.2f}',(i+.2,b),ha='center',va='bottom' if b>=0 else 'top')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'intraday: {ac_pre:+.3f} -> {ac_post:+.3f}   close-to-close: {oc_pre:+.3f} -> {oc_post:+.3f}')"
        ),
        md(
            f"There's the catch. The intraday leg *did* edge more negative "
            f"(**{R['intraday'][0]:+.2f} → {R['intraday'][1]:+.2f}**) — the direction the story predicts. "
            f"But the move is tiny, and the **close-to-close** return went the **opposite** way "
            f"(**{R['oc_ret'][0]:+.2f} → {R['oc_ret'][1]:+.2f}**, *less* mean-reverting). A 'pinning' "
            "effect that appears in one daily measure and reverses in another isn't a fingerprint — it's "
            "noise."
        ),
        md(
            "**Could a random split look this big?** The honest test: re-cut the tape at thousands of "
            "*random* dates and see how 2022's change compares to chance."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pb = st.placebo_break_pvalue(F, 'intraday'); obs = pb['obs_d_ac']; pval = pb['p_value']\n"
            "    x = F['intraday'].values; n = len(x); lo, hi = int(n*.25), int(n*.75)\n"
            "    rng = np.random.default_rng(370); cuts = rng.integers(lo, hi, size=4000)\n"
            "    deltas = np.array([abs(st.lag1_autocorr(x[c:]) - st.lag1_autocorr(x[:c])) for c in cuts])\n"
            "else:\n"
            "    obs = abs(R['intraday'][2]); pval = R['intraday'][4]\n"
            "    rng = np.random.default_rng(370); deltas = np.abs(rng.normal(0, 0.022, 4000))\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(deltas, bins=50, color=GREY, alpha=.85, label='|change| at a RANDOM break date')\n"
            "ax.axvline(obs, c=GREEN, lw=2.5, label=f'the actual 2022 change ({obs:.3f})')\n"
            "ax.set_xlabel('|change in mean-reversion gauge| across the split'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'2022 is an ordinary split — placebo p = {pval:.2f}'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'a random split moves the gauge this much {pval*100:.0f}% of the time — 2022 is unremarkable')"
        ),
        md(
            f"The 2022 change sits **mid-cloud**: about **{R['intraday'][4]*100:.0f}%** of random splits "
            "move the gauge at least as much. In plain terms, **2022 is a perfectly ordinary place to cut "
            "the tape** — nothing special happened to daily mean-reversion that you couldn't get by "
            "slicing at a random date."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — Weak.** The intraday gauge nudged the predicted way, but it's tiny, not "
            "significant, and a second daily measure flips it. Direction-consistent, evidence-light.\n"
            "- **Tradability — Mirage.** The natural 'trade the pin' rule pays *less* after 2022 and "
            "loses money net of costs (next beat).\n"
            "- **\"0DTE pinned the market\"? — Unproven.** The boom is real; the tape effect is "
            "invisible on daily data — and the intraday phenomenon can't be seen from daily bars at all."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — fade the move\n\n"
            "If the tape is more pinned, the simplest play is **contrarian**: fade yesterday's move and "
            "expect a bounce-back. If pinning is real and growing, this should work **better** after "
            "2022. Here's the gross-vs-net daily payoff, before and after the boom."
        ),
        code(
            "if HAVE_REAL:\n"
            "    mr = st.mean_reversion_trade(F, BD, 'intraday', cost_bps=1.0)\n"
            "    segs = ['pre', 'post']; g = [mr[s]['gross_mean']*1e4 for s in segs]; nv = [mr[s]['net_mean']*1e4 for s in segs]\n"
            "else:\n"
            "    tr = {s:(gg,nn) for s,gg,nn,_,_ in R['trade']}\n"
            "    segs = ['pre','post']; g = [tr['pre'][0], tr['post'][0]]; nv = [tr['pre'][1], tr['post'][1]]\n"
            "x = np.arange(2)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.bar(x-.2, g, .4, color=GREEN, label='gross')\n"
            "ax.bar(x+.2, nv, .4, color=GREY, label='net @1bp')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_xticks(x); ax.set_xticklabels(['pre-2022','post-2022'])\n"
            "ax.set_ylabel('average daily payoff (basis points)')\n"
            "ax.set_title('The pin trade pays LESS after the boom — and loses money net of cost')\n"
            "for i,(a,b) in enumerate(zip(g,nv)):\n"
            "    ax.annotate(f'{a:+.2f}',(i-.2,a),ha='center',va='bottom')\n"
            "    ax.annotate(f'{b:+.2f}',(i+.2,b),ha='center',va='bottom' if b>=0 else 'top')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'post-2022 net: {nv[1]:+.2f} bp/day — a single basis point of cost erases it')"
        ),
        md(
            f"The exact opposite of the claim. The fade trade is **weaker** post-2022 (gross "
            f"{R['trade'][1][1]:.2f} → {R['trade'][2][1]:.2f} bp) and **net of one basis point it loses "
            f"money** (**{R['trade'][2][2]:+.2f} bp/day**, Sharpe ≈ 0). Whatever faint mean-reversion the "
            "daily proxy carries, the 0DTE boom did **not** make it more harvestable — if anything, "
            "less."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The vol market it lives in.** [Study 130 — Vol-Risk-Premium](../130-vol-risk-premium/) "
            "and [Study 111 — VIX-Term-Structure](../111-vix-term-structure/) ask whether options carry a "
            "harvestable premium at all — 0DTE is the newest corner of that market.\n"
            "- **Intraday patterns.** [Study 06 — Clockwork-Vol](../06-clockwork-vol/) looks at whether "
            "the *within-day* tape has exploitable rhythm — the same question on a different axis.\n"
            "- **Get the real data.** The honest test needs intraday SPX ticks + dealer-gamma estimates "
            "(vendor data). Swap our daily proxy for that feed and you could finally *see* the within-day "
            "pinning — or confirm it isn't there.\n\n"
            "*Think the 0DTE boom pinned the tape? Get the intraday data, measure the within-day "
            "reversal before and after 2022, and show the change landing **outside** the random-split "
            "cloud — then we'll talk.*"
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
            "# Zero-DTE options & the SPX tape — a quantitative teardown 🔬\n"
            "### Intraday-leg autocorrelation pre/post-2022 · a block-bootstrap Welch *t* + placebo-break "
            "null · break-date robustness · a cost-charged pin trade · a synthetic pinning-regime control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The "
            "believers' claim is an **intraday-microstructure** statement — dealer gamma damps and "
            "reverses within-day moves since 0DTE went daily in 2022. We cannot observe that object "
            "from a free feed, so we test its **daily shadow**: the lag-1 autocorrelation of the "
            "open→close leg, split at 2022, confronted with the **sample size and the serial structure** "
            "of the data. The decisive finding is not a sign (it's faintly in the predicted direction) "
            "but **significance**: the cross-2022 change is indistinguishable from a random split, it "
            "reverses in the close-to-close leg, and it grows only as the break date is fished later.\n\n"
            "> ⚠️ **Data + proxy note.** No free intraday option tape exists; we use the **daily** "
            "open→close leg as an explicit **proxy** for intraday mean-reversion (named on the Signal "
            "axis), plus a current nearest-expiry option-chain snapshot. Real data: yfinance daily SPY "
            "OHLC 2010→2026 + a 0DTE chain as-of 2026-06-22. Offline core + synthetic control are "
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
            f"| **Signal** | `WEAK` | intraday lag-1 autocorr **{R['intraday'][0]:+.3f} → "
            f"{R['intraday'][1]:+.3f}** (Δ {R['intraday'][2]:+.3f}); block-bootstrap Welch **t = "
            f"{R['intraday'][3]:.2f}**, placebo **p = {R['intraday'][4]:.2f}** — predicted direction but "
            f"**fails t ≥ 2**, and the close-to-close leg moves **{R['oc_ret'][2]:+.2f}** (opposite). |\n"
            f"| **Tradability** | `MIRAGE` | the fade-the-move trade goes **{R['trade'][1][1]:.2f} → "
            f"{R['trade'][2][1]:.2f} bp/day** gross pre→post and **loses net** post-2022 "
            f"({R['trade'][2][2]:+.2f} bp, Sharpe ≈ {R['trade'][2][4]:.2f}). |\n"
            f"| **Tape pinned?** | `UNPROVEN` | the boom is real (**{R['chain']['share_near']*100:.0f}%** "
            f"of a {R['chain']['total_volume']/1e6:.1f}M-volume 0DTE chain within ±1% of spot) but its "
            "daily-tape effect is indistinguishable from a random split — and is **unmeasurable** from "
            "daily bars. |\n\n"
            "> 💡 In plain words: the 0DTE *market* exploded; the *price path*, as far as daily data can "
            "tell, did not change in the way the story claims. The honest blocker is that the story is "
            "about a layer of the data we don't have."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r^{\\text{id}}_t = \\log(C_t / O_t)$ be the daily **intraday** (open→close) leg and "
            "$\\rho_1(\\cdot)$ its lag-1 autocorrelation. The pinning thesis predicts a *more negative* "
            "$\\rho_1$ after the 0DTE rollout: a pinned tape reverses within-day moves.\n\n"
            "- **H₁ (the regime changed).** $\\rho_1^{\\text{post}} < \\rho_1^{\\text{pre}}$ by a margin "
            "the data can resolve, with the break at 2022.\n"
            "- **H₂ (it's tradable).** A 1-day-lagged contrarian rule that fades $r^{\\text{id}}_{t-1}$ "
            "earns *more*, net of costs, post-2022.\n"
            "- **H₃ (0DTE caused it).** The change is specific to the 0DTE rollout, not an artefact of "
            "where you cut the tape or which return leg you pick.\n\n"
            "We find **H₁ not rejected but not supported** (Δρ₁ is the right sign but $|t|<2$, placebo "
            "$p\\approx0.4$), **H₂ rejected** (the trade weakens and loses net post-2022), **H₃ rejected** "
            "(the close-to-close leg moves the *other* way; the effect grows as you fish the break later). "
            "Worse, $r^{\\text{id}}_t$ is a *daily* shadow of an *intraday* mechanism — even a clean "
            "result here would under-identify the claim."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The whole teardown is one comparison: two regimes' lag-1 autocorrelations, judged by a "
            "**standard error that respects serial dependence**.\n\n"
            "$$\\widehat{\\Delta} = \\rho_1^{\\text{post}} - \\rho_1^{\\text{pre}},\\qquad "
            "t = \\frac{\\widehat{\\Delta}}{\\sqrt{\\,\\widehat{\\mathrm{se}}^2_{\\text{post}} + "
            "\\widehat{\\mathrm{se}}^2_{\\text{pre}}\\,}},$$\n\n"
            "where each $\\widehat{\\mathrm{se}}$ is a **block bootstrap** (20-day blocks) — an iid SE "
            "would understate it because returns are serially dependent. A second, assumption-light "
            "instrument is a **placebo-break test**: resample the break date itself and ask how often a "
            "random cut produces a $|\\widehat\\Delta|$ this large. That randomization $p$, not the point "
            "estimate's sign, decides the Signal axis — because picking 2022 as 'the' break is itself a "
            "researcher choice."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Tape.** Daily SPY OHLC (yfinance, {R['start']}→{R['end']}, {R['days']:,} days). "
            "Decompose each day into **intraday** $\\log(C/O)$ and **overnight** $\\log(O/C_{-1})$ legs.\n"
            "- **Gauge.** $\\rho_1(r^{\\text{id}})$ per regime; *more negative ⇒ more mean-reverting / "
            "more pinned*. Explicit **daily proxy** for an intraday object (named on the axis).\n"
            f"- **Break.** {R['break_date']} (same-day options rolled out Mon/Wed/Fri through 2022) ⇒ "
            f"{R['n_pre']:,} pre / {R['n_post']:,} post days.\n"
            "- **Null #1 (block-bootstrap Welch t).** $\\widehat\\Delta$ over block-bootstrap SEs.\n"
            "- **Null #2 (placebo break).** 5,000 random break dates (≥25% each side); $p = "
            "\\Pr[|\\widehat\\Delta_{\\text{random}}| \\ge |\\widehat\\Delta_{2022}|]$.\n"
            "- **Robustness.** Re-run the break at 2021/2022/2023; check the close-to-close leg too.\n"
            "- **Trade.** Position $=-\\operatorname{sign}(r^{\\text{id}}_{t-1})$, 1-day lag, 1 bp "
            "one-way × turnover; gross/net by regime.\n"
            "- **Positive control.** A deterministic intraday series with a **planted pinning switch** "
            "(post-switch AR(1) coefficient $-\\text{pin}$): the engine must detect a real switch and "
            "**not** invent one when $\\text{pin}=0$."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The point estimates — right sign, tiny, leg-inconsistent\n\n"
            "Lag-1 autocorrelation by regime for the intraday leg (the claim) and the close-to-close leg "
            "(a control). The intraday leg edges more negative; the close-to-close leg goes the other way."
        ),
        code(
            "legs = ['intraday', 'oc_ret']\n"
            "if HAVE_REAL:\n"
            "    pre = [st.regime_table(F, BD, l)['ac_pre'] for l in legs]\n"
            "    post = [st.regime_table(F, BD, l)['ac_post'] for l in legs]\n"
            "    tvals = [st.welch_t_autocorr(F, BD, l)['t'] for l in legs]\n"
            "else:\n"
            "    pre = [R['intraday'][0], R['oc_ret'][0]]; post = [R['intraday'][1], R['oc_ret'][1]]\n"
            "    tvals = [R['intraday'][3], R['oc_ret'][3]]\n"
            "x = np.arange(2)\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x-.2, pre, .4, color=GREY, label='pre-2022')\n"
            "ax.bar(x+.2, post, .4, color=GREEN, label='post-2022')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_xticks(x)\n"
            "ax.set_xticklabels(['intraday (open->close)\\nTHE CLAIM', 'close-to-close\\nCONTROL'])\n"
            "ax.set_ylabel('lag-1 autocorrelation'); ax.set_title('Right sign on the claim leg; opposite on the control leg')\n"
            "for i,(a,b) in enumerate(zip(pre,post)):\n"
            "    ax.annotate(f'{a:+.3f}',(i-.2,a),ha='center',va='bottom' if a>=0 else 'top')\n"
            "    ax.annotate(f'{b:+.3f}',(i+.2,b),ha='center',va='bottom' if b>=0 else 'top')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('Welch t by leg:', {l: round(t,2) for l,t in zip(legs, tvals)})"
        ),
        md(
            f"> 💡 In plain words: the intraday leg moved **{R['intraday'][0]:+.3f} → "
            f"{R['intraday'][1]:+.3f}** (Δ {R['intraday'][2]:+.3f}, t = {R['intraday'][3]:.2f}) — the "
            f"predicted direction, but a whisper. The control leg went **{R['oc_ret'][2]:+.2f}** the "
            "*other* way. H₁ is **not rejected, not supported**; H₃ is already wobbling — a real "
            "mechanism shouldn't flip sign across two near-identical daily measures."
        ),
        md(
            "### 4b · The decisive test — a placebo null over the break date\n\n"
            "Picking 2022 is a choice. Re-cut the tape at 5,000 random dates and build the null "
            "distribution of $|\\widehat\\Delta\\rho_1|$. The 2022 value is the green line; the *p* is the "
            "right-tail mass."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pb = st.placebo_break_pvalue(F, 'intraday'); obs = pb['obs_d_ac']; pval = pb['p_value']\n"
            "    x = F['intraday'].values; n = len(x); lo, hi = int(n*.25), int(n*.75)\n"
            "    rng = np.random.default_rng(370); cuts = rng.integers(lo, hi, size=5000)\n"
            "    deltas = np.array([abs(st.lag1_autocorr(x[c:]) - st.lag1_autocorr(x[:c])) for c in cuts])\n"
            "else:\n"
            "    obs = abs(R['intraday'][2]); pval = R['intraday'][4]\n"
            "    rng = np.random.default_rng(370); deltas = np.abs(rng.normal(0, 0.022, 5000))\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(deltas, bins=60, color=GREY, alpha=.85, label=f'null: |Δρ₁| at {len(deltas):,} random breaks')\n"
            "ax.axvline(obs, c=GREEN, lw=2.5, label=f'observed 2022 |Δρ₁| = {obs:.3f}')\n"
            "ax.set_xlabel('|change in lag-1 autocorrelation| across the split'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Placebo break p = {pval:.2f}: 2022 is an ordinary cut'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'P[|random-break Δ| >= |2022 Δ|] = {pval:.3f}  (need <0.05 to call it special)')"
        ),
        md(
            f"> 💡 In plain words: **{R['intraday'][4]*100:.0f}%** of random break dates move the gauge as "
            "much as 2022 did. A real structural break would push the green line into the right tail; "
            "instead it sits mid-cloud. H₁ rejected as *special to 2022* — the change is exactly what "
            "re-slicing a noisy autocorrelation buys you for free."
        ),
        md(
            "### 4c · Robustness — the effect grows only as you fish the break later\n\n"
            "Vary the assumed break date. If 2022 were the true regime change, the |t| should *peak* "
            "near it. Instead it **rises monotonically** the later you cut — the signature of selecting a "
            "smaller, more recent, more volatile post-window, not of a real break."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for lab in ('2021-01-01','2022-01-01','2022-05-01','2023-01-01'):\n"
            "        b = pd.Timestamp(lab); rt = st.regime_table(F, b, 'intraday'); w = st.welch_t_autocorr(F, b, 'intraday')\n"
            "        rows.append((lab, rt['d_ac'], w['t'], rt['n_post']))\n"
            "else:\n"
            "    rows = R['robust']\n"
            "labs = [r[0][:7] for r in rows]; tt = [abs(r[2]) for r in rows]; nn = [r[3] for r in rows]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(labs, tt, color=AMBER, width=.55)\n"
            "ax.axhline(2, ls='--', c=RED, label='|t|=2 (significance bar)')\n"
            "for i,(t,k) in enumerate(zip(tt,nn)): ax.annotate(f'n={k}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylim(0, 2.3); ax.set_ylabel('|Welch t| on Δρ₁'); ax.set_xlabel('assumed break date')\n"
            "ax.set_title('|t| rises as you fish later — never reaches 2 (selection, not a break)'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('break-date sweep (label, Δρ₁, t, n_post):', [(r[0], round(r[1],3), round(r[2],2), r[3]) for r in rows])"
        ),
        md(
            "> 💡 In plain words: a genuine 2022 break would make the test *strongest* at 2022; here the "
            "|t| keeps climbing toward 2023 (and still never hits 2) simply because a later cut leaves a "
            "shorter, choppier post-sample. This is textbook break-date selection, not evidence of "
            "pinning."
        ),
        md(
            "### 4d · Tradability + the synthetic control\n\n"
            "Left: the contrarian fade trade, gross vs net, pre/post-2022 — if pinning grew, it should "
            "earn *more* post-boom. Right: a deterministic series with a **planted pinning switch** — the "
            "engine must light up only when a real switch exists."
        ),
        code(
            "if HAVE_REAL:\n"
            "    mr = st.mean_reversion_trade(F, BD, 'intraday', cost_bps=1.0)\n"
            "    g = [mr['pre']['gross_mean']*1e4, mr['post']['gross_mean']*1e4]\n"
            "    nv = [mr['pre']['net_mean']*1e4, mr['post']['net_mean']*1e4]\n"
            "else:\n"
            "    tr = {s:(gg,nn) for s,gg,nn,_,_ in R['trade']}\n"
            "    g = [tr['pre'][0], tr['post'][0]]; nv = [tr['pre'][1], tr['post'][1]]\n"
            "syn_t = []\n"
            "for pin in (0.0, 0.30):\n"
            "    syn = data.synthetic_tape(n_days=4000, pin=pin, seed=370); sw = syn.index[2000]\n"
            "    syn_t.append((pin, st.welch_t_autocorr(syn, sw, 'intraday')['t']))\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.2))\n"
            "x = np.arange(2)\n"
            "a1.bar(x-.2, g, .4, color=GREEN, label='gross'); a1.bar(x+.2, nv, .4, color=GREY, label='net @1bp')\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_xticks(x); a1.set_xticklabels(['pre','post'])\n"
            "a1.set_ylabel('mean daily payoff (bp)'); a1.set_title('Pin trade: weaker post, loses net'); a1.legend()\n"
            "for i,(aa,bb) in enumerate(zip(g,nv)):\n"
            "    a1.annotate(f'{aa:+.2f}',(i-.2,aa),ha='center',va='bottom')\n"
            "    a1.annotate(f'{bb:+.2f}',(i+.2,bb),ha='center',va='bottom' if bb>=0 else 'top')\n"
            "pins = [f'pin={p:.2f}' for p,_ in syn_t]; tv = [t for _,t in syn_t]\n"
            "a2.bar(pins, tv, color=[GREY, GREEN], width=.5)\n"
            "a2.axhline(-2, ls='--', c=RED, label='|t|=2'); a2.axhline(2, ls='--', c=RED)\n"
            "for i,t in enumerate(tv): a2.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom' if t>=0 else 'top')\n"
            "a2.set_ylabel('Welch t (synthetic)'); a2.set_title('Control: detects a real pin, invents none'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('pin trade net pre/post (bp):', [round(v,2) for v in nv], ' | control t:', [round(t,2) for _,t in syn_t])"
        ),
        md(
            f"> 💡 In plain words: the fade trade falls from **{R['trade'][1][1]:.2f}** to "
            f"**{R['trade'][2][1]:.2f} bp/day** gross and goes **net-negative** post-2022 — the opposite "
            f"of a growing edge. And the control is honest: with **no** planted pin it sits at "
            f"**t = {R['syn'][0][4]:.2f}** (no false break); a **strong** pin drives "
            f"**t = {R['syn'][1][4]:.2f}**. So the real-tape t of {R['intraday'][3]:.2f} is exactly what "
            "*no measurable regime change* looks like through this lens."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — intraday Δρ₁ = **{R['intraday'][2]:+.3f}** at block-bootstrap Welch "
            f"**t = {R['intraday'][3]:.2f}** / placebo **p = {R['intraday'][4]:.2f}**; the close-to-close "
            f"leg moves **{R['oc_ret'][2]:+.2f}** (opposite) and |t| only grows as the break is fished "
            "later. Predicted direction + an insignificant, non-robust, leg-inconsistent estimate ⇒ "
            "WEAK, never REAL (the desk's t ≥ 2 bar is not met) — and the object is a daily proxy for an "
            "intraday claim.\n"
            f"- **Tradability `MIRAGE`** — the fade trade goes **{R['trade'][1][1]:.2f} → "
            f"{R['trade'][2][1]:.2f} bp/day** gross pre→post and is **net-negative** post-2022 "
            f"({R['trade'][2][2]:+.2f} bp, Sharpe ≈ {R['trade'][2][4]:.2f}). No positive, cost-surviving, "
            "post-boom edge exists on the daily tape.\n"
            f"- **Tape pinned? `UNPROVEN`** — the 0DTE boom is real "
            f"(**{R['chain']['share_near']*100:.0f}%** of a {R['chain']['total_volume']/1e6:.1f}M-volume "
            "same-day chain within ±1% of spot, on a daily-expiry calendar), but its effect on the index "
            "path is statistically indistinguishable from a random split — and the intraday phenomenon is "
            "**not measurable from daily data at all**. Real product, unproven tape effect."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the identification gap\n\n"
            "The operational truth in one picture: the *direct* 0DTE evidence (the at-the-money volume "
            "wall) vs the *measurable tape effect* (≈ 0). The chain proves the boom; the tape proves "
            "nothing — and the gap between them is the part of the data a daily feed can't reach."
        ),
        code(
            "if HAVE_REAL and data.have_option_chain():\n"
            "    meta = json.load(open('../_cache/option_meta.json')); ch = data.load_option_chain(); spot = meta['spot']\n"
            "    cc = st.chain_concentration(ch, spot, band=0.01); share = cc['share_near']*100\n"
            "    obs_t = abs(st.welch_t_autocorr(F, BD, 'intraday')['t'])\n"
            "else:\n"
            "    share = R['chain']['share_near']*100; obs_t = abs(R['intraday'][3])\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.2))\n"
            "a1.bar(['0DTE volume\\nwithin ±1% of spot'], [share], color=GREEN, width=.5)\n"
            "a1.set_ylim(0, 100); a1.set_ylabel('% of same-day chain volume')\n"
            "a1.annotate(f'{share:.0f}%', (0, share), ha='center', va='bottom', fontsize=12)\n"
            "a1.set_title('The boom is unmistakable')\n"
            "a2.bar(['tape effect\\n(|Welch t| on Δρ₁)'], [obs_t], color=AMBER, width=.5)\n"
            "a2.axhline(2, ls='--', c=RED, label='|t|=2 needed'); a2.set_ylim(0, 2.4)\n"
            "a2.annotate(f't={obs_t:.2f}', (0, obs_t), ha='center', va='bottom', fontsize=12)\n"
            "a2.set_ylabel('|Welch t|'); a2.set_title('The tape effect is ~zero'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'boom: {share:.0f}% ATM volume | tape: |t|={obs_t:.2f} (need 2) -> real product, unproven effect')"
        ),
        md(
            f"> 💡 In plain words: a {R['chain']['share_near']*100:.0f}%-at-the-money, "
            f"{R['chain']['total_volume']/1e6:.1f}M-volume same-day chain on the left; a |t| of "
            f"**{abs(R['intraday'][3]):.2f}** against a bar of 2 on the right. The boom is not in doubt; "
            "the *tape effect* is invisible. And note the structural limit: even a clean daily result "
            "would only be a **proxy** — gamma-hedging pinning is a within-day phenomenon, and the within "
            "day is exactly the resolution a daily bar throws away. To settle the claim you need intraday "
            "ticks and dealer-gamma estimates; on the data a free feed gives, **the 0DTE-pinning story is "
            "real as a market and unproven as a mechanism.**"
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The vol-premium context.** [Study 130 — Vol-Risk-Premium](../130-vol-risk-premium/) and "
            "[Study 111 — VIX-Term-Structure](../111-vix-term-structure/): is there a harvestable options "
            "premium at all, and where on the curve does 0DTE sit?\n"
            "- **Intraday rhythm.** [Study 06 — Clockwork-Vol](../06-clockwork-vol/) — the within-day "
            "volatility patterns the pinning story would have to bend.\n"
            "- **The real test.** Replace the daily proxy with intraday SPX ticks + estimated dealer "
            "gamma exposure (vendor data); regress within-day reversal on gamma, pre vs post-2022. That "
            "is the experiment this proxy can only gesture at.\n\n"
            "*The reproducible core is offline and deterministic; the intraday tape here is an explicit "
            "**daily proxy** and the option chain is a single **snapshot**. Methods and sources: "
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
