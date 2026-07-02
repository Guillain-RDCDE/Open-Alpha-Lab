"""Generate the two narrative notebooks for Study 562 (Block-Trade-Signal).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). This is a **synthetic-only**
study — there is no real block-trade tape a retail stack can reach — so every cell is deterministic
and offline (no real-tape banner). The dict ``R`` below mirrors docs/results.md exactly; the
notebook cells re-derive those numbers live from the deterministic panel, so a reader reproduces
them offline.
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


# Frozen synthetic headline numbers — mirror of docs/results.md (as-of 2026-06-30; seed 562;
# null panel fp cd3cabd17906, planted-edge panel fp dc3dd07ca2da).
R = dict(
    fp_null="cd3cabd17906", fp_pos="dc3dd07ca2da",
    n_stocks=24, n_events=1200, alpha_pos=0.006,
    # null world
    null_mean=-0.004, null_t=-0.04, null_hit=49.5, null_placebo=0.964,
    null_size_spread=0.14, null_size_t=0.59, null_slope_t=-0.13, null_corr=-0.004,
    # planted-modest world
    pos_mean=0.419, pos_t=4.32, pos_hit=55.0, pos_placebo=0.0005,
    pos_size_spread=0.62, pos_size_t=2.50, pos_slope_t=2.09, pos_corr=0.060,
    cost_bps=8.0, borrow_bps=100.0, hold_days=5, pos_net=0.249,
    # size-threshold sweep in the planted world: (min_block_z, mean%, t, n, hit%)
    sweep=[(0.0, 0.42, 4.32, 1200, 55.0), (1.5, 0.57, 4.49, 713, 56.9),
           (2.0, 0.71, 4.28, 439, 59.2), (3.0, 0.87, 3.09, 159, 62.3)],
    # synthetic control: (alpha, mean coattail-t over 25 seeds)
    ctrl=[(-0.020, -11.44), (-0.012, -8.13), (-0.006, -4.51), (0.0, -0.13),
          (0.006, 4.26), (0.012, 7.92), (0.020, 11.28)],
)


BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from block_trade_signal import data, strategy as st

# Two deterministic worlds share one schema (seed 562): the NULL (no block->drift) and a
# literature-modest planted edge. Synthetic-only: there is no real block tape to fetch.
NULL, _  = data.synthetic_panel(drift_alpha=0.0)
POS,  _  = data.synthetic_panel(drift_alpha=0.006)
print("synthetic-only study - no real tape.")
print("null panel fp   :", data.fingerprint(NULL), " events:", len(NULL))
print("planted panel fp:", data.fingerprint(POS),  " events:", len(POS))
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# The Block-Trade Signal — can you ride a whale's coattails? 🐋\n"
            "### When a giant trade crosses the tape, is it smart money — and can you copy it?\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n\n"
            "Here's a piece of trading-floor folklore as old as the ticker tape. Every so often a "
            "**giant block** prints — a single trade far bigger than everything around it. The story "
            "says: *that's an institution who knows something, crossing size.* If you can tell which "
            "way they went (buying or selling) and jump on the same side, you'll **ride their "
            "coattails** as the stock drifts their way.\n\n"
            "Sounds great. But the microstructure researchers who actually studied blocks found "
            "something inconvenient: most giant blocks aren't *informed* at all — they're a big fund "
            "**rebalancing**, willing to pay a temporary price concession to get it done, and the "
            "price **snaps back** afterward. Copy that and you *lose*.\n\n"
            "So which is it — informed drift, or a liquidity trap that reverts?\n\n"
            "> ⚠️ **We can't answer it on real data — and that's the point of this study.** To even "
            "*see* a block you need the intraday trade-by-trade tape *and* the quotes to tell buy "
            "from sell. A free retail feed gives you a daily *volume* number and nothing else. So we "
            "build a **synthetic** block-trade world we fully control, and use it to show what the "
            "signal would look like if it were real, if it reversed, and if it were nothing.\n"
            ">\n"
            "> 📓 **This is the plain-language layer.** Want the *t*-stats, the placebo test and the "
            "cost maths? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ----
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does copying a giant block earn a drift? | **In an honest coin-flip world, no.** With "
            f"no real block→drift relation the coattail earns **{R['null_mean']}%** per event "
            f"(*t* {R['null_t']}), hit rate **{R['null_hit']}%** — a coin. |\n"
            "| Could it ever work? | **Yes, if blocks were informed** — but it could equally "
            "*reverse* (if they're liquidity trades). The literature is split on the very sign. |\n"
            "| Can we settle it on real data? | **No.** No free feed can even *see* a classified "
            "block — you need the paid intraday tape + quotes. This study is **synthetic-only**. |\n"
            "| Could you trade it? | **No.** You can't cheaply observe the signal, and chasing a "
            "block means crossing the spread into a *dislocated* price — costs bite hard. |\n\n"
            "> The coattail is a great story with real academic backing on *both* sides (informed "
            "drift vs. liquidity reversal) — but on a free stack you can neither see it nor trade it, "
            "so it earns **Weak × Mirage**."
        ),

        # ---- BEAT 1 — THE CLAIM ----
        md(
            "## 1 · The claim\n\n"
            "> *\"A large block-trade print is an informed institution crossing size; the stock will "
            "drift in the direction of that flow.\"* — trading-floor folklore, with a theoretical "
            "backbone (Easley & O'Hara 1987: informed traders prefer to trade *big*, so trade size "
            "itself carries information).\n\n"
            "The coattail trade is simple: watch the tape for a giant print, work out whether it was "
            "buyer- or seller-initiated, take the *same* side, and hold for the drift. The whole bet "
            "is that the block *knew something*."
        ),

        # ---- BEAT 2 — SO WHAT ----
        md(
            "## 2 · So what?\n\n"
            "If blocks really are informed, the tape hands you a free signal every day — just follow "
            "the whales. But the microstructure literature says the opposite is at least as likely: "
            "Holthausen-Leftwich-Mayers and Keim-Madhavan showed that a big chunk of a block's price "
            "move is a **temporary concession** the initiator pays for immediacy, which then "
            "**reverts**. Same block, opposite trade. The desk has looked at daily *volume* signals "
            "([512](../../512-high-volume-return-premium/), [492](../../492-up-down-volume/)); this "
            "study goes at the *single classified block print* and its **direction**."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ----
        md(
            "## 3 · How would we even know?\n\n"
            "1. **Spot the block.** A print far bigger than the stock's normal trade size.\n"
            "2. **Guess its side.** Buyer- or seller-initiated, from a quote rule (imperfect — it "
            "gets the side wrong maybe 1 time in 7).\n"
            "3. **Copy it.** Take the same side, hold a few days, measure the P&L.\n"
            "4. **Repeat over thousands of blocks** and ask: does copying blocks earn a drift "
            "different from zero?\n\n"
            "The catch that forces this study to be **synthetic**: steps 1 and 2 need the "
            "*trade-by-trade tape and the quotes* — paid data. A free feed gives you one daily volume "
            "number, which is *not* a block and has *no* direction. So we build the block world in "
            "code, with a single knob that decides whether blocks are informed, reversing, or "
            "nothing — and let the machinery loose on it.\n\n"
            "*Timing:* the coattail is entered **after** the block prints (the signal is public the "
            "instant it crosses the tape) and held over the forward window. No peeking — one clean "
            "execution lag."
        ),

        # ---- BEAT 4 — THE TEARDOWN ----
        md(
            "## 4 · The teardown — the honest coin-flip world\n\n"
            "First the **null**: a world where a block tells you *nothing*. If the machinery is "
            "honest it should find no drift here."
        ),
        code(
            "cs = st.coattail_stats(NULL)\n"
            "p  = st.placebo_pvalue(NULL, n_perm=2000)\n"
            "mean, t, hit = cs['mean']*100, cs['t'], cs['hit']*100\n"
            "up = (NULL['fwd_ret'] > 0).mean()*100\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.4))\n"
            "ax.hist(NULL['fwd_ret']*100, bins=40, color=GREY, alpha=.85)\n"
            "ax.axvline(mean, c=RED, lw=2, label=f'mean coattail {mean:+.3f}%')\n"
            "ax.axvline(0, c='k', lw=1)\n"
            "ax.set_xlabel('coattail return per block %'); ax.set_ylabel('# blocks')\n"
            "ax.set_title(f'NULL world: copying blocks is a coin (t {t:+.2f}, hit {hit:.1f}%, placebo p {p:.2f})')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'coattail mean {mean:+.3f}% | t {t:+.2f} | hit {hit:.1f}% | placebo p {p:.3f}')"
        ),
        md(
            f"There it is — **nothing.** The coattail earns **{R['null_mean']}%** per block "
            f"(*t* {R['null_t']}), wins **{R['null_hit']}%** of the time (a coin), and the placebo "
            f"test puts the odds at *p* = {R['null_placebo']}. When blocks carry no information, "
            "copying them is a coin flip — exactly as it should be. This is the world the real market "
            "*might* be, and we have no real tape to rule it out."
        ),
        md(
            "**Now flip the knob on.** Plant a *modest* informed edge (correctly-signed giant blocks "
            "drift the coattail's way) and re-run — do bigger blocks drift more?"
        ),
        code(
            "sw = st.robustness_sweep(POS)\n"
            "fig, ax = plt.subplots(figsize=(8, 4.3))\n"
            "ax.bar([str(x) for x in sw.index], sw['mean']*100, color=GREEN, width=.55)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('only trade blocks at least this big (x normal size)')\n"
            "ax.set_ylabel('mean coattail return %')\n"
            "ax.set_title('Planted world: the bigger the block, the bigger the drift')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(sw.round(4).to_string())"
        ),
        md(
            "In the planted world the drift **grows with block size** — the giant prints drift most, "
            "just as the folklore claims. That's the shape a *real* informed-block signal would have. "
            "But remember: **we planted it.** The whole question is whether the real tape looks like "
            "this world or the coin-flip one — and no free data can tell us."
        ),

        # ---- BEAT 5 — VERDICT ----
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — Weak.** The idea has real academic support, but it fights an equally "
            "documented *reversal*, and we can't certify either on a real tape (synthetic-only). On "
            f"the honest null the coattail is a coin (*t* {R['null_t']}, placebo *p* {R['null_placebo']}).\n"
            "- **Tradability — Mirage.** You can't cheaply *see* a classified block, and chasing one "
            "means crossing into a dislocated price. Nothing to trade on a retail stack."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ----
        md(
            "## 6 · Could you actually trade it?\n\n"
            "No — twice over. First, you can't *observe* the signal: seeing a block and its side "
            "needs the paid intraday tape and quotes, not the daily bars a free feed gives you. "
            "Second, even in the planted world where the edge is real, **chasing a block means buying "
            "into the very print that just dislocated the price** — you pay the spread on a moved "
            "market. Those frictions eat roughly 40% of a modest edge (see the quants notebook). "
            "Great story, unreachable trade."
        ),

        # ---- BEAT 7 — GOING FURTHER ----
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The volume cousins.** [512 High-Volume-Return-Premium](../../512-high-volume-return-premium/) "
            "and [492 Up-Down-Volume](../../492-up-down-volume/) use *daily* volume — no block, no "
            "side. This study is the *single classified print*.\n"
            "- **Informed flow you *can* see.** [263 Insider-Buying](../../263-insider-buying/) uses "
            "disclosed insider trades (Form 4) — informedness with a real, free tape.\n"
            "- **The real test needs the tape.** Re-run this with genuine TAQ prints + Lee-Ready "
            "signs + survivorship-free returns — that's where the informed-vs-liquidity sign gets "
            "settled.\n\n"
            "*Think blocks really are informed? Fork this, feed it a real classified-block tape, and "
            "show the coattail clearing t = 2 the right way — after the spread you pay to chase it.*"
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
            "# The Block-Trade Signal — a quantitative teardown 🔬\n"
            "### coattail one-sample *t* · giant−small size sort · sign-shuffle placebo · signed-size slope · size-threshold sweep · costs & borrow · both-signs synthetic control\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We build a synthetic "
            "block-print event panel, form the per-event **coattail** return (copy each block's "
            "inferred side), and test whether giant blocks predict drift (informed flow) — or "
            "reverse (liquidity).\n\n"
            "> ⚠️ **Not investment advice. Synthetic-only study** — there is no free feed of "
            "*classified* block trades (intraday TAQ prints + Lee-Ready side + survivorship-free "
            f"returns), so per the desk's rubric this is capped at `WEAK`/`NONE`. Null panel fp "
            f"`{R['fp_null']}`, planted-edge panel fp `{R['fp_pos']}`, as-of 2026-06-30. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | Null-world coattail **{R['null_mean']}%**/event (one-sample "
            f"*t* **{R['null_t']}**, placebo *p* {R['null_placebo']}, hit {R['null_hit']}%) — a coin; "
            "literature-supported but sign-disputed and no real tape (synthetic-only). |\n"
            f"| **Tradability** | `MIRAGE` | Cannot cheaply observe a classified block; even a planted "
            f"edge nets **+{R['pos_net']}%**/event after {R['cost_bps']:.0f} bps/leg + borrow (gross "
            f"+{R['pos_mean']}% → net, ~40% eaten). |\n\n"
            "> 💡 In plain words: the engine works (the synthetic control proves it catches both "
            "signs), but on the honest null the block coattail is indistinguishable from a coin, and "
            "no free data can settle the informed-vs-liquidity sign."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let a block event $i$ carry inferred side $s_i \\in \\{-1,+1\\}$ and size $z_i$ "
            "(× normal trade size); the coattail P&L is $r_i$ (the forward return of taking side "
            "$s_i$).\n\n"
            "- **H₁ (informed drift).** $\\mathbb{E}[r] > 0$ at *t* ≥ 2 (copying blocks pays).\n"
            "- **H₂ (size).** giant blocks drift more than small: "
            "$\\mathbb{E}[r \\mid \\text{giant}] - \\mathbb{E}[r \\mid \\text{small}] > 0$.\n"
            "- **H₃ (robustness).** the sign of H₁ holds as we restrict to bigger blocks.\n"
            "- **H₄ (net).** H₁ survives the spread you cross to chase a dislocated print + borrow.\n\n"
            "The **counter-hypothesis** (Holthausen-Leftwich-Mayers; Keim-Madhavan) is "
            "$\\mathbb{E}[r] < 0$ — liquidity reversal. Because this is synthetic-only, we test the "
            "*machinery* on both a null and a planted world; the real-tape adjudication is out of "
            "reach."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what? — what rides on the sign\n\n"
            "The content is the **sign**. Easley-O'Hara gives informed traders a reason to trade big "
            "(H₁, `drift_alpha > 0`); Holthausen-Leftwich-Mayers and Keim-Madhavan give blocks a "
            "temporary-concession reversal (`drift_alpha < 0`). Barclay-Warner and Chakravarty add a "
            "twist even the *informed* side doesn't fully support the naive folklore — the "
            "information hides in *medium* trades, not the giant prints. So the size sort (H₂) is a "
            "genuine test, not a foregone conclusion."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Coattail.** For each block take side $s_i$, hold the window; $r_i$ is the P&L. "
            "Headline: one-sample *t* on $\\{r_i\\}$.\n"
            "- **Size sort.** Tercile tails by $z_i$: giant vs small; two-sample (Welch) *t* on the "
            "coattail spread.\n"
            "- **Placebo.** Re-sign each event's directional component against the return magnitude "
            "(2000 shuffles); read the mean coattail's tail probability.\n"
            "- **Firm-level.** OLS of $r$ on standardised $|s_i z_i|$ — bigger block → more drift?\n"
            "- **Robustness.** Re-run the coattail *t* at rising size thresholds.\n"
            "- **Frictions.** Round-trip cost per event (cross the spread twice) + a borrow on the "
            "short (seller-initiated) coattails.\n"
            "- **Positive control.** A deterministic world with a planted drift of *either* sign — "
            "averaged over 25 seeds (house rule).\n\n"
            "Timing: the coattail is entered *after* the block prints (signal public at the print) "
            "and held over the forward window `fwd_ret` already measures. One execution lag, no "
            "look-ahead."
        ),

        # ---- BEAT 4 ----
        md("## 4 · The teardown"),
        md(
            "### 4a · The coattail one-sample *t* — H₁ (null vs planted)\n\n"
            "The headline: does copying blocks earn a drift? Run both worlds side by side."
        ),
        code(
            "cn, cp = st.coattail_stats(NULL), st.coattail_stats(POS)\n"
            "pn = st.placebo_pvalue(NULL, n_perm=2000); pp = st.placebo_pvalue(POS, n_perm=2000)\n"
            "labels = ['NULL\\n(no signal)', 'PLANTED\\n(informed)']\n"
            "means = [cn['mean']*100, cp['mean']*100]; ts = [cn['t'], cp['t']]\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.3))\n"
            "bars = ax.bar(labels, means, color=[GREY, GREEN], width=.5)\n"
            "for b, t in zip(bars, ts):\n"
            "    ax.text(b.get_x()+b.get_width()/2, b.get_height(), f't {t:+.2f}', ha='center', va='bottom')\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('mean coattail return %/event')\n"
            "ax.set_title('Copying blocks: a coin at the null, a drift only when we plant one')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'NULL   : mean {cn[\"mean\"]*100:+.3f}% t {cn[\"t\"]:+.2f} hit {cn[\"hit\"]*100:.1f}% placebo p {pn:.3f}')\n"
            "print(f'PLANTED: mean {cp[\"mean\"]*100:+.3f}% t {cp[\"t\"]:+.2f} hit {cp[\"hit\"]*100:.1f}% placebo p {pp:.4f}')"
        ),
        md(
            f"> 💡 In plain words: at the **null**, H₁ is not there — coattail **{R['null_mean']}%** "
            f"(*t* {R['null_t']}), placebo *p* {R['null_placebo']}, a coin. Switch the knob on and the "
            f"engine recovers a clean **+{R['pos_mean']}%** (*t* +{R['pos_t']}, placebo *p* "
            f"{R['pos_placebo']}). The detector is faithful; the null is what an *uninformed* block "
            "world looks like."
        ),
        md(
            "### 4b · The giant − small size sort & the signed-size slope — H₂\n\n"
            "Do the *giant* blocks drift more than the small ones? (Planted world.)"
        ),
        code(
            "ss = st.size_sort(POS, 0.3); sl = st.signed_size_slope(POS)\n"
            "fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))\n"
            "ax[0].bar(['small','giant'], [ss['small_mean']*100, ss['giant_mean']*100],\n"
            "          color=[GREY, GREEN], width=.5)\n"
            "ax[0].axhline(0, c='k', lw=1); ax[0].set_ylabel('mean coattail %')\n"
            "ax[0].set_title(f'giant - small = {ss[\"spread\"]*100:+.2f}%  (t {ss[\"t\"]:+.2f})')\n"
            "x = np.abs(POS['signed_z']); x=(x-x.mean())/x.std(ddof=1); y = POS['fwd_ret']*100\n"
            "ax[1].scatter(x, y, s=6, color=GREY, alpha=.4)\n"
            "xs = np.linspace(x.min(), x.max(), 50)\n"
            "ax[1].plot(xs, (sl['slope']*xs + (y.mean()/100 - sl['slope']*x.mean()))*100, c=RED, lw=2)\n"
            "ax[1].set_xlabel('standardised |signed block size|'); ax[1].set_ylabel('coattail %')\n"
            "ax[1].set_title(f'slope t {sl[\"t\"]:+.2f} (corr {sl[\"corr\"]:+.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'size spread {ss[\"spread\"]*100:+.2f}% (t {ss[\"t\"]:+.2f}) | slope t {sl[\"t\"]:+.2f} corr {sl[\"corr\"]:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: H₂ holds in the planted world — the giant blocks beat the small by "
            f"**+{R['pos_size_spread']}%** (*t* +{R['pos_size_t']}) and the signed-size slope is "
            f"positive (*t* +{R['pos_slope_t']}). Bigger block, bigger drift — the folklore's shape. "
            "(At the null both collapse to noise: spread *t* +0.59, slope *t* −0.13.)"
        ),
        md(
            "### 4c · Robustness — H₃ (does the sign hold as blocks get bigger?)\n\n"
            "Re-run the coattail *t* at rising size thresholds, in both worlds."
        ),
        code(
            "swp = st.robustness_sweep(POS); swn = st.robustness_sweep(NULL)\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(swp.index, swp['t'], 'o-', c=GREEN, lw=2, label='planted (informed)')\n"
            "ax.plot(swn.index, swn['t'], 's--', c=GREY, lw=2, label='null (no signal)')\n"
            "ax.axhline(2, ls=':', c=RED, lw=1, label='t = 2 bar'); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('min block size (x normal)'); ax.set_ylabel('coattail one-sample t')\n"
            "ax.set_title('Planted signal holds above the bar as blocks get bigger; null hugs zero')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('planted:', [round(t,2) for t in swp['t']]); print('null   :', [round(t,2) for t in swn['t']])"
        ),
        md(
            "> 💡 In plain words: H₃ holds in the planted world — the coattail *t* stays above 2 as we "
            "restrict to giant prints (softening only as the sample shrinks), while the *per-event* "
            "drift actually **rises**. The null hugs zero throughout. A real informed-block signal "
            "would look like the green line; a liquidity-trap one would mirror it below zero."
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — null coattail *t* {R['null_t']} (a coin, placebo *p* "
            f"{R['null_placebo']}); the claim is literature-supported but sign-disputed and "
            "**uncertifiable** on a real tape (synthetic-only → capped at `WEAK`/`NONE`).\n"
            f"- **Tradability `MIRAGE`** — you cannot cheaply observe a classified block; even a "
            f"planted edge nets +{R['pos_net']}%/event after crossing the spread + borrow."
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you trade it? — costs and the borrow\n\n"
            "Chasing a block means buying *into* the print that just moved the price — you cross the "
            "spread on a dislocated market. Charge it (planted world)."
        ),
        code(
            "net = st.net_stats(POS, cost_bps=8.0, borrow_ann_bps=100.0, holding_days=5.0)\n"
            "gross = net['gross']*100; nett = net['net']*100\n"
            "fig, ax = plt.subplots(figsize=(7, 4.2))\n"
            "ax.bar(['gross','net\\n(cost+borrow)'], [gross, nett], color=[GREEN, GREY], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('mean coattail return %/event')\n"
            "eaten = (1 - nett/gross)*100 if gross else float('nan')\n"
            "ax.set_title(f'Frictions eat ~{eaten:.0f}% of a modest edge (gross {gross:+.2f}% -> net {nett:+.2f}%)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross {gross:+.3f}% -> net {nett:+.3f}% (cost {net[\"cost\"]*100:.3f}%, borrow {net[\"borrow\"]*100:.4f}%)')"
        ),
        md(
            f"> 💡 In plain words: even in the world where the edge is *real and planted*, "
            f"{R['cost_bps']:.0f} bps/leg of block-chasing cost turns +{R['pos_mean']}% gross into "
            f"**+{R['pos_net']}%** net — roughly 40% gone. On a free stack you can't even *see* the "
            "signal to begin with. `MIRAGE`."
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further — the both-signs synthetic control\n\n"
            "Is the engine a faithful detector, or does it always print ~0? Plant a block drift of "
            "*either* sign (informed drift > 0, liquidity reversal < 0) of increasing strength and "
            "watch the coattail *t* — averaged over 25 seeds so no single lucky seed can fake it."
        ),
        code(
            "alphas = [-0.020, -0.012, -0.006, 0.0, 0.006, 0.012, 0.020]\n"
            "ts = [st.synthetic_mean_t(data, drift_alpha=a, n_seeds=25) for a in alphas]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(alphas, ts, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1); ax.axhline(-2, ls='--', c=GREY, lw=1)\n"
            "ax.axhline(0, c='k', lw=1); ax.axvline(0, c='k', lw=.7, ls=':')\n"
            "ax.set_xlabel('planted drift_alpha  (< 0 = liquidity reversal, > 0 = informed drift)')\n"
            "ax.set_ylabel('mean coattail-t (25 seeds)')\n"
            "ax.set_title('The harness catches BOTH signs — flat at the null, monotone in the effect')\n"
            "plt.tight_layout(); plt.show()\n"
            "for a, t in zip(alphas, ts): print(f'alpha {a:+.3f} -> mean coattail-t {t:+.2f}')"
        ),
        md(
            "The coattail *t* is ≈ 0 at the null, climbs past +2 as informed drift grows, and falls "
            "past −2 as the liquidity reversal grows — **the engine catches both signs and stays flat "
            "at zero.** So the real-tape question is entirely about *which world we live in*, and that "
            "is exactly what no free block-trade feed can tell us. `WEAK` (literature says maybe-real, "
            "no real-tape certification, sign disputed) × `MIRAGE` (you cannot cheaply observe the "
            "signal to trade it). For the daily-volume cousins, see "
            "[512 High-Volume-Return-Premium](../../512-high-volume-return-premium/) and "
            "[492 Up-Down-Volume](../../492-up-down-volume/)."
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
