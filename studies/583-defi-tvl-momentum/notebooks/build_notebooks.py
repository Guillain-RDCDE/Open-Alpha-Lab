"""Generate the two narrative notebooks for Study 583 (DeFi-TVL-Momentum).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). This is a
**synthetic-only** study — there is no real tape (a survivorship-free, point-in-time per-protocol
TVL x token-return panel is not reachable on a free, no-key stack), so ``HAVE_REAL`` is always
False and every figure is drawn by the deterministic, offline synthetic generator. The frozen
numbers in ``R`` mirror docs/results.md and are recomputed live in the cells (they match), so the
notebooks are reproducible for any reader offline.
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


# Frozen headline numbers — mirror of docs/results.md (as-of 2026-06-30; seed 583; 60 protocols x
# 47 months; planted panel fp 0b6d95a52429, null fp 0a3c150d737e).
R = dict(
    fp_planted="0b6d95a52429", fp_null="0a3c150d737e",
    n_proto=60, n_months=47, n_rows=2820, alpha=0.05,
    ls_mean=11.8, ls_t=10.9, placebo_p=0.0005,
    slope=0.050, slope_t=11.3,
    null_mean=0.4, null_t=0.33, null_placebo=0.75, null_slope_t=0.03,
    gross_m=11.8, net_m=10.5, gross_ann=141.7, net_ann=126.5,
    cost_bps=30.0, borrow_bps=800.0,
    # robustness grid: (frac, lag, mean%, t)
    robu=[(0.2, 0, 13.3, 9.23), (0.3, 0, 11.8, 10.88), (0.4, 0, 9.3, 9.42),
          (0.2, 1, 5.3, 4.24), (0.3, 1, 4.4, 3.90), (0.4, 1, 4.2, 4.23)],
    # synthetic control: (alpha, mean ls-t over 25 seeds)
    ctrl=[(0.0, 0.30), (0.02, 4.66), (0.05, 11.19), (0.10, 22.06)],
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

from defi_tvl_momentum import data, strategy as st

# SYNTHETIC-ONLY: there is no real tape for a survivorship-free, point-in-time TVL x
# token-return panel on a free stack. fetch_panel() is empty by construction.
HAVE_REAL = not data.fetch_panel().empty        # always False here
ALPHA = 0.05                                     # planted-effect strength (positive control)
PANEL, TRUTH = data.synthetic_panel(tvl_alpha=ALPHA, seed=583)      # the planted world
NULL, _ = data.synthetic_panel(tvl_alpha=0.0, seed=583)            # the null world
print("real TVL tape present:", HAVE_REAL, "(synthetic-only study)")
print("planted panel:", PANEL.protocol.nunique(), "protocols x",
      PANEL.month.nunique(), "months  fp", data.fingerprint(PANEL))
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# DeFi-TVL-Momentum — does 'money flooding in' mean the token keeps pumping? 🌊\n"
            "### Ranking DeFi protocols by how fast cash is piling into them, in plain English\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n\n"
            "In DeFi (decentralised finance), the headline number everyone watches is **TVL** — "
            "*total value locked* — the dollar amount of crypto that people have deposited into a "
            "protocol's smart contracts. The folklore: when TVL is **rising fast**, that's *smart "
            "money* flooding in, and the protocol's token will keep **pumping**. It sounds obvious — "
            "money follows the winners, and price follows money.\n\n"
            "So the trade writes itself: every month, rank protocols by how fast their TVL grew, "
            "**buy** the ones with the biggest inflows, **short** the ones bleeding out.\n\n"
            "There's a catch we hit before we even start: **we can't get honest data to test this.** "
            "More on that below — it's the whole story.\n\n"
            "> 📓 **This is the plain-language layer.** Want the *t*-stats, the placebo test and the "
            "cost maths? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool: every chart is drawn by the "
            "code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ----
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Can we test this on real DeFi data? | **No — and that's the punchline.** An honest "
            "test needs every protocol that *died* (rug pulls, hacks, dead forks) still in the "
            "dataset. Free data sources quietly delete them. |\n"
            "| Does the idea *work* when the effect is really there? | **Yes** — on a synthetic world "
            "where we *plant* a real TVL→return link, our engine catches it cleanly (spread "
            f"**+{R['ls_mean']}%/mo**, hugely significant). |\n"
            "| Does it fire on *nothing*? | **No** — with the effect switched off, the engine prints "
            f"a flat **+{R['null_mean']}%/mo**. So it's honest: it finds a real signal and ignores "
            "noise. |\n"
            "| Could you trade it for real? | **No.** The 'short the outflows' leg is exactly the "
            "tokens crashing to zero — nobody will lend you those to short. |\n\n"
            "> The machinery is sound. The problem is **data**: without a survivorship-free, "
            "point-in-time tape, the flooding-in-money story can't be certified — so it's `WEAK`, "
            "and the trade is a `MIRAGE`."
        ),

        # ---- BEAT 1 — THE CLAIM ----
        md(
            "## 1 · The claim\n\n"
            "> *\"When TVL floods into a DeFi protocol, its token keeps outperforming.\"* — on-chain "
            "folklore, the crypto version of *'flows follow price, price follows flows.'*\n\n"
            "The appeal: TVL is **public and real-time**. Unlike a company's earnings (reported once "
            "a quarter, late), you can watch money move on-chain *right now*. So — the story goes — "
            "you have a live, leading indicator of demand that front-runs the token price. Rising "
            "TVL = demand = higher returns coming."
        ),

        # ---- BEAT 2 — SO WHAT ----
        md(
            "## 2 · So what?\n\n"
            "If it worked, it would be a genuinely new kind of edge — a *fundamental* signal you can "
            "read off the blockchain in real time, with no company filings, no waiting. The desk has "
            "tested price-based crypto folklore before "
            "([133 Crypto-Seasonality](../../133-crypto-seasonality/), "
            "[210 Crypto-Trend](../../210-crypto-trend/)); this is different — it's an **on-chain "
            "fundamental** (how much money is actually parked in the protocol), the crypto cousin of "
            "TradFi fund flows ([561 ETF-Flow-Momentum](../../561-etf-flow-momentum/))."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ----
        md(
            "## 3 · How would we even know?\n\n"
            "1. **Measure the flow.** For each protocol, how fast did its TVL grow over the past "
            "month? (Fast growth = inflows; shrinking = outflows.)\n"
            "2. **Sort.** Every month, rank all protocols by TVL growth. Buy the fastest-inflow "
            "third, short the outflow third.\n"
            "3. **Check next month.** Did the buy basket beat the short basket? Repeat for years and "
            "see if the average spread is reliably positive.\n\n"
            "**The wall we hit:** to do this *honestly* you need the protocols that later went to "
            "**zero** still in the data at the time. Free sources show you today's survivors and "
            "revise old TVL numbers after the fact — so any spread you measure is flattered by "
            "hindsight. That's why this study runs on a **synthetic world** we fully control: we can "
            "switch the effect on and off and prove the engine is honest.\n\n"
            "*Timing:* we rank on the month's close (using only past TVL) and hold the next month. No "
            "peeking at the future."
        ),

        # ---- BEAT 4 — THE TEARDOWN ----
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**When the effect is really there, does the engine catch it?** We plant a genuine "
            "TVL→return link in a synthetic world and run the sort."
        ),
        code(
            "ls = st.long_short_series(PANEL, frac=0.3)\n"
            "r = st.ls_tstat(ls)\n"
            "ls0 = st.long_short_series(NULL, frac=0.3)\n"
            "r0 = st.ls_tstat(ls0)\n"
            "fig, ax = plt.subplots(figsize=(8, 4.4))\n"
            "ax.bar(['effect PLANTED', 'null (no effect)'], [r['mean']*100, r0['mean']*100],\n"
            "       color=[GREEN, GREY], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('avg monthly long-short return %')\n"
            "ax.set_title(f'Buy-inflows minus short-outflows: +{r[\"mean\"]*100:.1f}%/mo planted vs +{r0[\"mean\"]*100:.1f}%/mo null')\n"
            "fig.suptitle('SYNTHETIC world (no real tape exists)', fontsize=9, color=GREY)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'PLANTED: spread +{r[\"mean\"]*100:.1f}%/mo (t {r[\"t\"]:+.1f})')\n"
            "print(f'NULL:    spread +{r0[\"mean\"]*100:.1f}%/mo (t {r0[\"t\"]:+.2f})')"
        ),
        md(
            f"When the effect is planted, the buy-inflows / short-outflows sort earns "
            f"**+{R['ls_mean']}%/mo** (a huge *t* +{R['ls_t']}). When we switch the effect **off**, "
            f"the same engine prints a flat **+{R['null_mean']}%/mo**. So the machinery is honest — it "
            "lights up on a real signal and shrugs at noise. The question was never *'can the engine "
            "detect it?'* It's *'can we get honest data?'* — and we can't."
        ),
        md(
            "**Is the planted signal fragile?** Re-run the sort at different basket sizes and with a "
            "one-month entry delay."
        ),
        code(
            "grid = st.robustness(PANEL, fracs=(0.2,0.3,0.4), lags=(0,1))\n"
            "labs = [f\"top/bot {int(f*100)}%\\n\"+(\"same-close\" if lg==0 else \"+1mo lag\")\n"
            "        for f,lg in zip(grid['frac'], grid['lag'])]\n"
            "fig, ax = plt.subplots(figsize=(9, 4.3))\n"
            "cols = [GREEN if t>=2 else AMBER for t in grid['t']]\n"
            "ax.bar(range(len(grid)), grid['mean']*100, color=cols, width=.6)\n"
            "ax.set_xticks(range(len(grid))); ax.set_xticklabels(labs, fontsize=8)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('avg monthly spread %')\n"
            "ax.set_title('Planted signal survives every cut — but fades if you enter a month late')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(grid.round(3).to_string(index=False))"
        ),
        md(
            "The planted signal survives every basket size. But notice: **entering a month late "
            "roughly halves it** — a realistic feature of a fast on-chain flow signal. By the time the "
            "TVL move is old news, most of the edge is gone."
        ),

        # ---- BEAT 5 — VERDICT ----
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — Weak.** The engine works on a planted world, but **there is no honest real "
            "tape** to test the actual claim (survivorship + revised-data problems). A synthetic-only "
            "study can never be certified `REAL` — it's capped at `WEAK`.\n"
            "- **Tradability — Mirage.** The short leg is the tokens crashing to zero, which nobody "
            "lets you borrow; monthly crypto-scale costs pile on top."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ----
        md(
            "## 6 · Could you actually trade it?\n\n"
            "No. Even if the signal were real, the trade means **shorting the protocols with "
            "collapsing TVL** — the rug pulls and hacked protocols on their way to zero. Those are "
            "exactly the tokens that are impossible to borrow, illiquid, and gap straight down. You'd "
            "own the winners fine, but the short half — where the folklore says the money is — is "
            "uninvestable."
        ),

        # ---- BEAT 7 — GOING FURTHER ----
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The data is the whole game.** If you can assemble a *survivorship-free, "
            "point-in-time* panel of per-protocol TVL and token returns — with the dead protocols "
            "still in it — you could run this for real. That's the missing ingredient.\n"
            "- **Cousins on the desk.** Price-based crypto folklore: "
            "[133 Crypto-Seasonality](../../133-crypto-seasonality/), "
            "[210 Crypto-Trend](../../210-crypto-trend/). The TradFi flow analogue: "
            "[561 ETF-Flow-Momentum](../../561-etf-flow-momentum/).\n\n"
            "*Think TVL-momentum is real? Fork this, plug in a survivorship-free on-chain panel, and "
            "show the buy-inflows/short-outflows spread clearing t = 2 on a real tape — net of a "
            "borrow you can actually get.*"
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
            "# DeFi-TVL-Momentum — a quantitative teardown 🔬\n"
            "### TVL-growth cross-sectional sort · one-sample *t* · label-shuffle placebo · month-clustered slope · robustness grid · costs & borrow · seed-robust synthetic control\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We rank DeFi protocols by "
            "trailing TVL growth, long the top / short the bottom, and test whether the forward "
            "token-return spread is reliably positive.\n\n"
            "> ⚠️ **Not investment advice.** This is a **synthetic-only** study: no survivorship-free, "
            "point-in-time per-protocol TVL x token-return panel is reachable on a free, no-key stack, "
            f"so `HAVE_REAL` is always False and every figure is the deterministic offline generator "
            f"(planted panel fp `{R['fp_planted']}`, null fp `{R['fp_null']}`, as-of 2026-06-30). "
            "Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | **No real tape.** Planted world: long-short **+{R['ls_mean']}%/mo** "
            f"(*t* **+{R['ls_t']}**, placebo *p* {R['placebo_p']}), slope-*t* **+{R['slope_t']}**; null "
            f"world flat (*t* +{R['null_t']}). Synthetic-only ⇒ capped at `WEAK` (never `REAL`). |\n"
            f"| **Tradability** | `MIRAGE` | Monthly rebalance × 2 legs + **{R['borrow_bps']:.0f} bps/yr "
            f"borrow** on an unborrowable short leg. Planted gross +{R['gross_m']}%/mo → net "
            f"+{R['net_m']}%/mo; on a real tape the short is uninvestable. |\n\n"
            "> 💡 In plain words: the engine is a faithful detector (proven below), but the claim "
            "cannot be certified without a survivorship-free, point-in-time real panel — which does "
            "not exist on a free stack."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $g_{i,t}$ be protocol $i$'s trailing TVL growth at month $t$ and $r_{i,t+1}$ its "
            "next-month token total return.\n\n"
            "- **H₁ (long-short).** $\\mathbb{E}[r \\mid \\text{top-}g] - \\mathbb{E}[r \\mid "
            "\\text{bottom-}g] > 0$ at *t* ≥ 2 (inflows out-earn outflows).\n"
            "- **H₂ (protocol-level).** slope of $r_{t+1}$ on the monthly-z-scored $g_t$ is $> 0$.\n"
            "- **H₃ (robustness).** the sign/size survives basket-fraction and entry-lag cuts.\n"
            "- **H₄ (net).** H₁ survives costs and the short-leg borrow.\n"
            "- **H₀ (data).** a survivorship-free, point-in-time real tape exists to test H₁–H₄.\n\n"
            "We find **H₀ fails** (no such free tape), so H₁–H₄ are tested only on a *planted* "
            "synthetic world — where H₁, H₂, H₃ all hold and H₄ is punished by the borrow. Because "
            "H₀ fails, the Signal is capped at `WEAK`."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The content is the **data wall**. A real DeFi-TVL-momentum test needs (a) the protocols "
            "that went to **zero** still present at each month (survivorship-free), and (b) TVL "
            "measured **point-in-time**, not the revised, re-categorised series free aggregators "
            "expose today. Miss (a) and the spread is flattered by hindsight; miss (b) and the "
            "ranking is look-ahead-contaminated. Neither is fixable on a no-key stack — the same wall "
            "the desk's [273 Lego-Returns](../../273-lego-returns/) / "
            "[276 Sneaker-Resale](../../276-sneaker-resale/) synthetic-only studies hit."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Signal.** $g_{i,t}$ = trailing TVL growth, z-scored across the cross-section each "
            "month.\n"
            "- **Sort.** Tercile tails (top-30% $g$ long, bottom-30% short); monthly long-short = "
            "long-mean − short-mean forward return.\n"
            "- **Inference.** One-sample *t* on the monthly long-short series; a **label-shuffle "
            "placebo** null (shuffle $g$ within each month, 2000 perms).\n"
            "- **Protocol-level.** Pooled OLS of $r_{t+1}$ on monthly-z $g_t$ with **month-clustered** "
            "(block) standard errors.\n"
            "- **Robustness.** Basket fractions {0.2, 0.3, 0.4} × entry lags {0, 1}.\n"
            "- **Frictions.** One-way cost per leg per monthly rebalance + an annual short borrow.\n"
            "- **Positive control.** Planted `tvl_alpha > 0` vs a null, averaged over 25 seeds.\n\n"
            "Timing: $g$ is trailing (fully public at the ranking close); we enter at that close and "
            "hold the next month. No look-ahead. The same-close fill is the documented convention; "
            "the `lag=1` robustness row shows a one-month-later entry roughly halves the edge."
        ),

        # ---- BEAT 4 ----
        md("## 4 · The teardown"),
        md(
            "### 4a · The long-short — H₁ (planted vs null)\n\n"
            "The monthly long-short series with its one-sample *t* and the placebo null, in both the "
            "planted and null worlds."
        ),
        code(
            "ls = st.long_short_series(PANEL, 0.3); r = st.ls_tstat(ls)\n"
            "p = st.placebo_pvalue(PANEL, 0.3, n_perm=2000, seed=583)\n"
            "ls0 = st.long_short_series(NULL, 0.3); r0 = st.ls_tstat(ls0)\n"
            "p0 = st.placebo_pvalue(NULL, 0.3, n_perm=2000, seed=583)\n"
            "fig, ax = plt.subplots(figsize=(9, 4.4))\n"
            "ax.plot(ls.index, ls.values*100, 'o-', c=GREEN, lw=1.5, ms=3, label=f'planted (t {r[\"t\"]:+.1f})')\n"
            "ax.plot(ls0.index, ls0.values*100, 'o-', c=GREY, lw=1.2, ms=3, label=f'null (t {r0[\"t\"]:+.2f})')\n"
            "ax.axhline(0, c='k', lw=1); ax.set_xlabel('month'); ax.set_ylabel('long-short return %')\n"
            "ax.set_title('Monthly buy-inflows / short-outflows return: planted vs null'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'PLANTED mean +{r[\"mean\"]*100:.1f}%/mo | t {r[\"t\"]:+.2f} | placebo p {p:.4f} | n {r[\"n\"]}')\n"
            "print(f'NULL    mean +{r0[\"mean\"]*100:.1f}%/mo | t {r0[\"t\"]:+.2f} | placebo p {p0:.3f}')"
        ),
        md(
            f"> 💡 In plain words: in the planted world H₁ **holds decisively** — spread "
            f"**+{R['ls_mean']}%/mo**, *t* +{R['ls_t']}, placebo *p* {R['placebo_p']}. In the null "
            f"world the same engine is **flat** (*t* +{R['null_t']}, placebo *p* {R['null_placebo']}). "
            "The detector is honest; the ceiling is data, not machinery."
        ),
        md(
            "### 4b · The protocol-level slope — H₂\n\n"
            "Pooled OLS of forward return on the monthly-z-scored TVL-growth signal, with "
            "month-clustered standard errors."
        ),
        code(
            "reg = st.pooled_slope(PANEL); reg0 = st.pooled_slope(NULL)\n"
            "df = PANEL.copy(); df['z'] = df.groupby('month')['tvl_growth'].transform(\n"
            "    lambda s: (s-s.mean())/s.std(ddof=1))\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.6))\n"
            "ax.scatter(df['z'], df['fwd_ret']*100, s=6, color=GREY, alpha=.35)\n"
            "xs = np.linspace(df['z'].min(), df['z'].max(), 50)\n"
            "b0 = df['fwd_ret'].mean() - reg['slope']*df['z'].mean()\n"
            "ax.plot(xs, (reg['slope']*xs + b0)*100, c=GREEN, lw=2.2)\n"
            "ax.set_xlabel('TVL-growth signal (monthly z-score)'); ax.set_ylabel('forward return %')\n"
            "ax.set_title(f'Slope {reg[\"slope\"]:+.3f}/z-unit (month-clustered t {reg[\"slope_t\"]:+.1f}) — POSITIVE = the claim')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'PLANTED slope {reg[\"slope\"]:+.3f}, clustered t {reg[\"slope_t\"]:+.2f}, n {reg[\"n\"]}')\n"
            "print(f'NULL    slope {reg0[\"slope\"]:+.3f}, clustered t {reg0[\"slope_t\"]:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: H₂ **holds** in the planted world — slope **+{R['slope']}** per "
            f"z-unit (clustered *t* +{R['slope_t']}), and ≈0 (*t* +{R['null_slope_t']}) at the null. "
            "The clustered SE stops within-month cross-sectional correlation from faking the *t*."
        ),
        md(
            "### 4c · Robustness — H₃ (basket size × entry lag)\n\n"
            "The long-short *t* over a grid of basket fractions and entry lags (planted world)."
        ),
        code(
            "grid = st.robustness(PANEL, fracs=(0.2,0.3,0.4), lags=(0,1))\n"
            "labs = [f\"{int(f*100)}% / lag{lg}\" for f,lg in zip(grid['frac'], grid['lag'])]\n"
            "fig, ax = plt.subplots(figsize=(9, 4.3))\n"
            "cols = [GREEN if t>=2 else AMBER for t in grid['t']]\n"
            "ax.bar(range(len(grid)), grid['t'], color=cols, width=.6)\n"
            "ax.set_xticks(range(len(grid))); ax.set_xticklabels(labs, rotation=10)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1, label='t = 2 bar')\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('long-short t')\n"
            "ax.set_title('Planted signal clears t=2 at every cut; the +1mo-lag rows are smaller (edge decays)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(grid.round(3).to_string(index=False))"
        ),
        md(
            "> 💡 In plain words: H₃ **holds** in the planted world — every basket cut clears *t* = 2. "
            "The `lag=1` rows are markedly smaller (*t* ~3.9–4.2 vs ~9–11): a one-month-late entry "
            "roughly **halves** the mean spread. A fast on-chain flow signal decays fast."
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — the engine banks the planted effect (H₁ *t* +{R['ls_t']}, H₂ "
            f"slope-*t* +{R['slope_t']}, H₃ survives) and stays flat at the null, but **H₀ fails**: no "
            "survivorship-free, point-in-time real tape exists on a free stack, so the claim cannot "
            "be certified `REAL` — capped at `WEAK`.\n"
            f"- **Tradability `MIRAGE`** — monthly rebalance × 2 legs + an {R['borrow_bps']:.0f} bps/yr "
            "borrow on an **unborrowable** short leg (the tokens gapping to zero); gross "
            f"+{R['gross_m']}%/mo → net +{R['net_m']}%/mo even in the planted world."
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you trade it? — costs and the borrow\n\n"
            "Charge a one-way cost per leg per monthly rebalance plus a punitive borrow on the "
            "short leg (collapsing-TVL tokens)."
        ),
        code(
            "ls = st.long_short_series(PANEL, 0.3)\n"
            "nm = st.net_mean(ls, cost_bps=30.0, borrow_ann_bps=800.0)\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.2))\n"
            "ax.bar(['gross', 'net\\n(costs+borrow)'], [nm['gross_m']*100, nm['net_m']*100],\n"
            "       color=[GREEN, GREY], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('avg monthly long-short %')\n"
            "ax.set_title(f'Even planted: gross +{nm[\"gross_m\"]*100:.1f}%/mo -> net +{nm[\"net_m\"]*100:.1f}%/mo')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross +{nm[\"gross_m\"]*100:.1f}%/mo ({nm[\"gross_ann\"]*100:.0f}%/yr) -> '\n"
            "      f'net +{nm[\"net_m\"]*100:.1f}%/mo ({nm[\"net_ann\"]*100:.0f}%/yr)')\n"
            "print('30 bps/leg x 2 legs monthly + 800 bps/yr borrow on the short leg')"
        ),
        md(
            "> 💡 In plain words: even in the *planted* world the costs bite. On a **real** tape the "
            "short leg — the rug/exploit tail — is **impossible to borrow** and gaps to zero, so the "
            "realisable net is worse than any gross. `MIRAGE`."
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further — the seed-robust synthetic positive control\n\n"
            "Is the engine a faithful detector, or does it always print a positive *t*? Sweep the "
            "planted `tvl_alpha` from 0 (null) upward and watch the long-short *t* — averaged over 25 "
            "seeds so no single lucky seed can fake it (the house rule for synthetic claims)."
        ),
        code(
            "alphas = [0.0, 0.01, 0.02, 0.05, 0.08, 0.10]\n"
            "ts = [st.synthetic_mean_t(data, tvl_alpha=a, n_seeds=25) for a in alphas]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(alphas, ts, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1, label='t = 2 bar')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted tvl_alpha (0 = null, larger = stronger effect)')\n"
            "ax.set_ylabel('mean long-short t (25 seeds)')\n"
            "ax.set_title('The harness banks a planted effect — flat at the null, past t=2 as it grows')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for a, t in zip(alphas, ts): print(f'alpha {a:+.2f} -> mean ls-t {t:+.2f}')"
        ),
        md(
            f"The long-short *t* is ≈ 0 at the null ({R['ctrl'][0][1]:+.2f} over 25 seeds) and climbs "
            f"past +2 as the planted effect grows ({R['ctrl'][2][1]:+.1f} at alpha {R['ctrl'][2][0]}). "
            "So the engine is a faithful detector — the honest conclusion is about **data "
            "availability**, not a broken test. The moment someone assembles a survivorship-free, "
            "point-in-time on-chain panel, this machinery is ready to run it. For the desk's "
            "price-based crypto folklore see [133 Crypto-Seasonality](../../133-crypto-seasonality/) "
            "and [210 Crypto-Trend](../../210-crypto-trend/); for the TradFi flow analogue see "
            "[561 ETF-Flow-Momentum](../../561-etf-flow-momentum/)."
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
