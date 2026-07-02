"""Generate the two narrative notebooks for Study 584 (Exchange-Netflows).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). This is a
**synthetic-only** study — a real BTC exchange-netflow tape is a paywalled address-clustering
product (see ../exchange_netflows/data.py) — so ``HAVE_REAL`` is always False and there is no
real-tape banner. Every computed cell runs the *deterministic synthetic world* (seed 584), offline
and reproducible; the frozen numbers in ``R`` mirror docs/results.md for the prose.
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


# Frozen headline numbers — mirror of docs/results.md (as-of 2026-06-30; deterministic synthetic
# world, seed 584, 1500 business days; null-world series fp 8c7301772d7b, fwd fp e3476cff1898).
R = dict(
    fp_series="8c7301772d7b", fp_fwd="e3476cff1898", n_days=1500, seed=584,
    # null world (bear_beta = 0) — the honest headline
    null_slope_bps=4.3, null_slope_t=0.47, null_corr=0.012,
    null_low_bps=13.3, null_high_bps=2.9, null_sort_spread_bps=10.3, null_sort_t=0.38,
    null_placebo_p=0.72,
    null_gross_bps=-1.1, null_net_bps=-6.4, null_turnover=0.49, null_net_sharpe=-0.32,
    cost_bps=10.0, borrow_bps=300.0,
    # robustness (null): (lag, slope_t, sort_t@0.2)
    win=[(1, 0.47, 0.38), (2, 0.58, 0.09), (3, 0.68, -0.48), (5, 0.53, -0.05)],
    # synthetic control: (bear_beta, mean slope-t over 25 seeds)
    ctrl=[(0.0, -0.01), (-0.0010, -1.12), (-0.0025, -2.78), (-0.0040, -4.45), (-0.0060, -6.67)],
    # single planted seed (bear_beta = -0.0025)
    plant_sort_bps=80.2, plant_sort_t=2.95, plant_placebo_p=0.005,
    plant_gross_sharpe=0.97, plant_net_sharpe=0.66,
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

from exchange_netflows import data, strategy as st

# SYNTHETIC-ONLY: a real BTC exchange-netflow tape is a paywalled address-clustering product,
# so there is NO real tape here (HAVE_REAL is always False, by design). Every computed cell runs
# the deterministic synthetic world, seed 584 — offline and reproducible.
HAVE_REAL = False
NULL, _ = data.synthetic_series(bear_beta=0.0, seed=584)        # the honest null world
print("synthetic null world:", len(NULL), "days | series fp", data.fingerprint(NULL),
      "| HAVE_REAL", HAVE_REAL)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Exchange Netflows — do the coins tell you when to sell? 🪙\n"
            "### \"Coins pouring onto exchanges = holders about to dump.\" In plain English.\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Would_the_engine_catch_it%3F: Yes](https://img.shields.io/badge/Would_the_engine_catch_it%3F-Yes-2ea44f?style=flat-square)\n\n"
            "There's a story on-chain analysts love. Every crypto coin lives at an *address*, and "
            "you can watch coins move between them. When lots of Bitcoin **flows onto exchanges** "
            "(Binance, Coinbase…), the story goes, holders are moving coins somewhere they can "
            "*sell* — so a wave of deposits is **bearish**: a sell-off is coming. When coins flow "
            "*off* exchanges into private wallets, that's **bullish**: people are hodling.\n\n"
            "It sounds airtight. So does BTC actually drop when 'net-inflows' turn positive?\n\n"
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
            "| Can we even test this for free? | **No — and that's the headline.** The 'netflow' "
            "number is a *paid product*: someone has to figure out which of millions of addresses "
            "belong to exchanges, and that labelling is exactly what Glassnode / CryptoQuant / "
            "Nansen *sell*. The raw blockchain gives you transfers, not the exchange labels. |\n"
            "| So what do we do instead? | Build a **fair simulated world** where we *know* the "
            "truth, and check two things: (a) if netflows really told you nothing, does our test "
            "correctly find nothing? (b) if we *plant* the effect, does the test catch it? |\n"
            f"| Does the test find a fake signal in the null world? | **No.** Slope of tomorrow's "
            f"BTC return on today's netflow ≈ **0** (*t* {R['null_slope_t']}), a coin-flip shuffle "
            f"test gives *p* = {R['null_placebo_p']}. Nothing there — as it should be. |\n"
            "| Could you trade it? | **No.** You can't buy the data, and even the simulated "
            "sign-timing book *loses* money after fees. |\n\n"
            "> The folklore might be true. But on a no-key retail stack there is **nothing real to "
            "test it on** — and a fair simulation of 'netflows tell you nothing' shows an honest "
            "test finding exactly nothing."
        ),

        # ---- BEAT 1 — THE CLAIM ----
        md(
            "## 1 · The claim\n\n"
            "> *\"When BTC net-inflows to exchanges rise, a sell-off is coming; when coins flow off "
            "to cold storage, it's bullish.\"* — the on-chain netflow folklore (CryptoQuant, "
            "Glassnode, Nansen).\n\n"
            "**Netflow = deposits − withdrawals.** Positive netflow means, on net, coins are moving "
            "*to* exchanges. The intuition: you only send coins to an exchange when you want to sell "
            "them (or trade), so a spike in deposits is supply about to hit the market."
        ),

        # ---- BEAT 2 — SO WHAT ----
        md(
            "## 2 · So what?\n\n"
            "If it worked, it would be a genuine *lead* indicator — you'd see the sellers coming "
            "before the price moved, and could step aside (or short) ahead of the drop. That's the "
            "dream of on-chain analysis: the blockchain is public, so *surely* it leaks intent.\n\n"
            "The desk has tested the crypto claims a free stack **can** reach — "
            "[133 Crypto-Seasonality](../../133-crypto-seasonality/), "
            "[210 Crypto-Trend](../../210-crypto-trend/), "
            "[325 Crypto-Fear-Greed](../../325-crypto-fear-greed/). This one is different: the data "
            "itself is behind a paywall."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ----
        md(
            "## 3 · How would we even know?\n\n"
            "1. **Get a netflow series.** …except you can't, for free. Labelling exchange addresses "
            "is the product these companies *sell*. So we build a **simulated** netflow + BTC world "
            "where we control the truth.\n"
            "2. **Line up the timing.** Today's netflow (known at today's close) vs the BTC return "
            "over the *next* day. No peeking at the future.\n"
            "3. **Two checks.** (a) In a *null* world where netflow means nothing, does our test "
            "correctly find nothing? (b) In a *planted* world where inflows really are bearish, does "
            "it catch it?\n\n"
            "*Timing:* the signal is the netflow at today's close; we trade tomorrow's return. One "
            "day of lag, documented."
        ),

        # ---- BEAT 4 — THE TEARDOWN ----
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**In the fair NULL world (netflow tells you nothing), do 'high-inflow' days really see "
            "worse next-day returns?**"
        ),
        code(
            "ss = st.sort_spread(NULL, lag=1, frac=0.2)\n"
            "low, high = ss['low_inflow_mean']*1e4, ss['high_inflow_mean']*1e4\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.4))\n"
            "ax.bar(['LOW inflow\\n(outflow, bullish)','HIGH inflow\\n(bearish)'], [low, high],\n"
            "       color=[GREEN, RED], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('next-day BTC return (bps)')\n"
            "ax.set_title(f'Folklore says HIGH-inflow days do worse. Gap = {ss[\"spread\"]*1e4:+.1f} bps (t {ss[\"t\"]:+.2f})')\n"
            "fig.suptitle('deterministic synthetic NULL world (netflow carries no info)', fontsize=9, color=GREY)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'low-inflow {low:+.1f} bps | high-inflow {high:+.1f} bps | spread {ss[\"spread\"]*1e4:+.1f} bps | t {ss[\"t\"]:+.2f}')"
        ),
        md(
            f"The gap is a statistically empty **{R['null_sort_spread_bps']:+.1f} bps** (*t* "
            f"{R['null_sort_t']}). In a world where netflow genuinely tells you nothing, our test "
            "correctly finds nothing — no fake signal. Good: the machinery isn't inventing edges."
        ),
        md(
            "**Is that just one lucky shuffle? A coin-flip (placebo) test:** shuffle the netflow "
            "labels against the returns thousands of times and see how often you'd get a gap this "
            "big by pure chance."
        ),
        code(
            "p = st.placebo_pvalue(NULL, lag=1, frac=0.2, n_perm=2000, seed=584)\n"
            "print(f'placebo p-value = {p:.3f}  (a real signal sits near 0; noise sits mid-range)')"
        ),
        md(
            f"*p* = **{R['null_placebo_p']}** — smack in the middle. Nothing distinguishes the real "
            "netflow ordering from a random one. Exactly what an honest engine prints when there is "
            "no effect."
        ),

        # ---- BEAT 5 — VERDICT ----
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** There's no real netflow tape to test (it's a paid product), and a "
            "fair simulation of the null shows the engine finding nothing (*t* +0.47, placebo *p* "
            "0.72). A synthetic-only study can't earn `REAL`.\n"
            "- **Tradability — Mirage.** You can't buy the data; even the *simulated* trade loses "
            "after fees.\n"
            "- **Would the engine catch it if it were real? — Yes.** Plant the folklore and the test "
            "banks it (next beat). The gap is *data*, not *code*."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ----
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Two walls. **First**, you can't get the netflow series without paying an on-chain data "
            "vendor — the whole thing rests on their private address labels. **Second**, even in the "
            "simulated null world, timing a long/short BTC book off the netflow sign *loses* money: "
            f"about **{R['null_net_bps']} bps per period** after a realistic per-trade cost and a "
            "short-borrow charge (mean net Sharpe across 25 simulated worlds ≈ "
            f"**{R['null_net_sharpe']}**). Turnover eats you alive for nothing."
        ),

        # ---- BEAT 7 — GOING FURTHER ----
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Is the test any good?** The quant notebook plants a *real* bearish-netflow effect "
            "and shows the engine catches it cleanly (slope-*t* down past −2) — proof the empty "
            "result is about missing data, not a blind detector. See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            "- **The crypto claims we *can* test.** [133 Crypto-Seasonality](../../133-crypto-seasonality/), "
            "[210 Crypto-Trend](../../210-crypto-trend/), [325 Crypto-Fear-Greed](../../325-crypto-fear-greed/) "
            "run on the price/sentiment tape a free stack reaches.\n\n"
            "*Have a real, licensed exchange-netflow tape? Fork this, drop it into `data.py`'s schema, "
            "and show the slope of forward BTC return on netflow clearing t = −2 the bearish way.*"
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
            "# Exchange Netflows — a quantitative teardown 🔬\n"
            "### netflow→forward-return slope · high-vs-low Welch *t* · label-shuffle placebo · lag/threshold sweep · costs & borrow · seed-robust synthetic control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Would_the_engine_catch_it%3F: Yes](https://img.shields.io/badge/Would_the_engine_catch_it%3F-Yes-2ea44f?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* The claim: exchange "
            "net-inflow (coins TO exchanges) is a bearish lead for BTC's forward return.\n\n"
            "> ⚠️ **Not investment advice.** **Synthetic-only:** a real BTC exchange-netflow tape is "
            "a paywalled address-clustering product (Glassnode / CryptoQuant / Nansen), so there is "
            "no real tape and a `REAL` stamp (robust *t* ≥ 2 on real data) is unreachable by "
            "construction — the study is capped at NONE. All computed cells run the deterministic "
            f"synthetic world (seed {R['seed']}, series fp `{R['fp_series']}`). Methods in "
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
            f"| **Signal** | `NONE` | No real tape (paid product). Null-world slope of forward return "
            f"on z(net-inflow) **{R['null_slope_bps']:+.1f} bps/σ** (*t* **{R['null_slope_t']}**); sort "
            f"spread **{R['null_sort_spread_bps']:+.1f} bps** (Welch *t* {R['null_sort_t']}, placebo "
            f"*p* {R['null_placebo_p']}). Synthetic-only → capped. |\n"
            f"| **Tradability** | `MIRAGE` | Data unreachable; simulated book gross **{R['null_gross_bps']} bps** "
            f"→ net **{R['null_net_bps']} bps** ({R['cost_bps']:.0f} bps/turn + {R['borrow_bps']:.0f} "
            f"bps/yr borrow), mean net Sharpe **{R['null_net_sharpe']}** (25 seeds). |\n"
            f"| **Would the engine catch it?** | `YES` | Planted `bear_beta −0.0025`: mean slope-*t* "
            f"**{R['ctrl'][2][1]}** (25 seeds); single seed sort spread **{R['plant_sort_bps']:+.0f} bps** "
            f"(*t* {R['plant_sort_t']}, placebo *p* {R['plant_placebo_p']}). |\n\n"
            "> 💡 In plain words: the harness is faithful (the control proves it), but the underlying "
            "netflow series is paywalled — so there is nothing real to certify, and a fair null is empty."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $F_t$ be the standardised exchange net-inflow on day $t$ and $r_{t\\to t+h}$ the "
            "forward BTC return over $(t, t+h]$.\n\n"
            "- **H₁ (slope / folklore).** $\\text{slope of } r_{t\\to t+h} \\text{ on } F_t < 0$ at "
            "*t* ≤ −2 (more inflow → lower forward return).\n"
            "- **H₂ (sort).** $\\mathbb{E}[r \\mid \\text{low inflow}] - \\mathbb{E}[r \\mid "
            "\\text{high inflow}] > 0$ (net-outflow days beat net-inflow days).\n"
            "- **H₃ (robustness).** The sign holds across execution lags and tail thresholds.\n"
            "- **H₄ (net).** The signal-timed book survives costs and the short borrow.\n\n"
            "**The catch is upstream of all four:** $F_t$ itself — an *exchange-labelled* on-chain "
            "quantity — is a paid clustering product. With no real $F_t$, none of H₁–H₄ can be "
            "certified on a real tape, so the study is synthetic-only and stamps `NONE`. We instead "
            "(a) confirm the null is empty and (b) prove the engine catches a *planted* effect."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what? — the data-availability wall\n\n"
            "A netflow is *defined* by which addresses belong to which exchange. Building that map "
            "means clustering millions of on-chain addresses into exchange cold/hot wallets — the "
            "proprietary moat Glassnode, CryptoQuant, Nansen and Chainalysis sell behind paid keys "
            "with no usable free history. The public chain gives **transfers, not labels**. Per the "
            "house inference bar, a study with no real tape is capped at `WEAK`/`NONE` — literature "
            "and a synthetic control are never market evidence. That places this beside the other "
            "*unreachable-data* studies while the tradable crypto tape lives in "
            "[133](../../133-crypto-seasonality/) / [210](../../210-crypto-trend/) / [325](../../325-crypto-fear-greed/)."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Signal.** Standardise net-inflow to $F_t = z(\\text{netflow})$.\n"
            "- **H₁.** OLS of $r_{t\\to t+h}$ on $F_t$; the *sign* of the slope is the folklore.\n"
            "- **H₂.** Split days into the top/bottom $q$ by $F_t$; Welch two-sample *t* on the two "
            "tails' forward returns.\n"
            "- **Placebo.** Shuffle $F_t$ against forward returns (2000 perms); tail probability of "
            "the spread.\n"
            "- **Robustness.** Sweep lags $\\{1,2,3,5\\}$ and tail fractions $\\{0.1,0.2,0.3\\}$.\n"
            "- **Frictions.** Signal-timed long/short book; one-way cost per turn + annual short "
            "borrow pro-rated over the hold.\n"
            "- **Positive control.** A planted world ($\\texttt{bear\\_beta}<0$) and a null, averaged "
            "over 25 seeds (house rule).\n\n"
            "Timing: $F_t$ is public at day-$t$ close; tested against $(t, t+h]$. One documented lag "
            "(default $h=1$); no future data enters the signal."
        ),

        # ---- BEAT 4 ----
        md("## 4 · The teardown"),
        md(
            "### 4a · The slope — H₁ (null world)\n\n"
            "Regress the forward BTC return on standardised net-inflow. A *negative* slope would be "
            "the folklore."
        ),
        code(
            "reg = st.flow_regression(NULL, lag=1)\n"
            "a = st.aligned(NULL, lag=1)\n"
            "x = a['z_flow'].to_numpy(); y = a['fwd'].to_numpy()*1e4\n"
            "fig, ax = plt.subplots(figsize=(8, 4.6))\n"
            "ax.scatter(x, y, s=10, color=GREY, alpha=.4)\n"
            "xs = np.linspace(x.min(), x.max(), 50)\n"
            "ax.plot(xs, (reg['slope']*xs + (a['fwd'].mean() - reg['slope']*a['z_flow'].mean()))*1e4, c=RED, lw=2)\n"
            "ax.set_xlabel('z(net-inflow) at t  (higher = more coins TO exchanges)')\n"
            "ax.set_ylabel('forward 1-day BTC return (bps)')\n"
            "ax.set_title(f\"slope {reg['slope']*1e4:+.1f} bps/sigma (t {reg['slope_t']:+.2f}) — folklore needs NEGATIVE\")\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"slope {reg['slope']*1e4:+.2f} bps/sigma | t {reg['slope_t']:+.2f} | corr {reg['corr']:+.4f} | n {reg['n']}\")"
        ),
        md(
            f"> 💡 In plain words: H₁ **not supported** — the slope is **{R['null_slope_bps']:+.1f} "
            f"bps/σ** (*t* {R['null_slope_t']}), a statistical zero and, if anything, the *wrong* "
            "sign. In a world where netflow carries no info, the engine correctly reads no info."
        ),
        md(
            "### 4b · The sort + placebo — H₂\n\n"
            "High-inflow vs low-inflow forward returns, with the Welch *t* and the label-shuffle "
            "placebo."
        ),
        code(
            "ss = st.sort_spread(NULL, lag=1, frac=0.2)\n"
            "p = st.placebo_pvalue(NULL, lag=1, frac=0.2, n_perm=2000, seed=584)\n"
            "fig, ax = plt.subplots(figsize=(8, 4.3))\n"
            "ax.bar(['LOW inflow (outflow)','HIGH inflow'],\n"
            "       [ss['low_inflow_mean']*1e4, ss['high_inflow_mean']*1e4], color=[GREEN, RED], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('forward 1-day return (bps)')\n"
            "ax.set_title(f\"spread {ss['spread']*1e4:+.1f} bps (Welch t {ss['t']:+.2f}, placebo p {p:.3f})\")\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"low {ss['low_inflow_mean']*1e4:+.1f} | high {ss['high_inflow_mean']*1e4:+.1f} | spread {ss['spread']*1e4:+.1f} bps | Welch t {ss['t']:+.2f} | placebo p {p:.3f}\")"
        ),
        md(
            f"> 💡 In plain words: H₂ **not supported** — the spread is **{R['null_sort_spread_bps']:+.1f} "
            f"bps** (Welch *t* {R['null_sort_t']}) and the placebo *p* = {R['null_placebo_p']} sits "
            "mid-distribution. No difference between high- and low-inflow days beyond noise."
        ),
        md(
            "### 4c · Robustness — H₃ (lags × thresholds)\n\n"
            "Does any lag or tail choice hide a signal?"
        ),
        code(
            "sw = st.robustness_sweep(NULL)\n"
            "piv = sw.pivot(index='lag', columns='frac', values='sort_t')\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "slope_ts = sw.drop_duplicates('lag').set_index('lag')['slope_t']\n"
            "ax.plot(slope_ts.index, slope_ts.values, 'o-', c=GREY, lw=2, label='slope-t (all fracs)')\n"
            "for fr in sorted(sw['frac'].unique()):\n"
            "    sub = sw[sw['frac']==fr]\n"
            "    ax.plot(sub['lag'], sub['sort_t'], 's--', lw=1.3, label=f'sort-t frac={fr}')\n"
            "ax.axhline(2, ls=':', c=RED, lw=1); ax.axhline(-2, ls=':', c=RED, lw=1)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_xlabel('execution lag (days)'); ax.set_ylabel('t-stat')\n"
            "ax.set_title('No lag or threshold clears |t| = 2 — the null stays null'); ax.legend(fontsize=8)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(sw.round(3).to_string(index=False))"
        ),
        md(
            "> 💡 In plain words: H₃ moot — every slope-*t* sits between +0.47 and +0.68, every "
            "sort-*t* well inside ±1. There is no fragile corner where a signal hides; the null is "
            "null everywhere."
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — no real tape (paid product); null-world slope-*t* "
            f"**{R['null_slope_t']}**, sort *t* {R['null_sort_t']}, placebo *p* {R['null_placebo_p']}. "
            "Synthetic-only caps the stamp.\n"
            f"- **Tradability `MIRAGE`** — data unreachable; simulated book net **{R['null_net_bps']} "
            f"bps**, mean net Sharpe {R['null_net_sharpe']}.\n"
            "- **Would the engine catch it? `YES`** — the planted control clears the bar (Beat 7)."
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you trade it? — costs and the borrow\n\n"
            "Even setting the missing data aside, sign-timing the netflow in the null world bleeds."
        ),
        code(
            "stt = st.strategy_stats(NULL, lag=1, band=0.5, cost_bps=10.0, borrow_ann_bps=300.0)\n"
            "fig, ax = plt.subplots(figsize=(7, 4.2))\n"
            "ax.bar(['gross','net\\n(cost+borrow)'], [stt['gross_mean']*1e4, stt['net_mean']*1e4],\n"
            "       color=[GREY, RED], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('mean per-period return (bps)')\n"
            "ax.set_title(f\"gross {stt['gross_mean']*1e4:+.1f} bps -> net {stt['net_mean']*1e4:+.1f} bps (turnover {stt['turnover']:.2f})\")\n"
            "plt.tight_layout(); plt.show()\n"
            "gs, ns = [], []\n"
            "for s in range(584, 584+25):\n"
            "    fr,_ = data.synthetic_series(bear_beta=0.0, seed=s)\n"
            "    d = st.strategy_stats(fr, lag=1); gs.append(d['gross_sharpe']); ns.append(d['net_sharpe'])\n"
            "print(f\"gross {stt['gross_mean']*1e4:+.1f} bps -> net {stt['net_mean']*1e4:+.1f} bps | \"\n"
            "      f\"mean gross Sharpe {np.nanmean(gs):+.2f} -> net {np.nanmean(ns):+.2f} (25 seeds)\")"
        ),
        md(
            f"> 💡 In plain words: gross ≈ 0, net **{R['null_net_bps']} bps** — the turnover cost and "
            "short borrow turn a nothing-signal into a slow loss. Mean net Sharpe "
            f"**{R['null_net_sharpe']}** across 25 seeds. `MIRAGE`."
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further — the synthetic positive control\n\n"
            "Is the engine a faithful detector, or does it always print ~0? Plant the folklore "
            "(`bear_beta < 0`) of increasing strength and watch the flow→return slope-*t* go "
            "negative — averaged over 25 seeds so no single lucky seed can fake it."
        ),
        code(
            "betas = [0.0, -0.001, -0.0025, -0.004, -0.006]\n"
            "ts = [st.synthetic_mean_t(b, n_seeds=25) for b in betas]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(betas, ts, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(-2, ls='--', c=GREY, lw=1, label='t = -2 bar (folklore)')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted bear_beta (more negative = stronger bearish coupling)')\n"
            "ax.set_ylabel('mean flow->return slope-t (25 seeds)')\n"
            "ax.set_title('The harness banks a planted folklore — flat at the null, past -2 as it grows')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for b, t in zip(betas, ts): print(f'bear_beta {b:+.4f} -> mean slope-t {t:+.2f}')\n"
            "# and a single planted seed's tradable book, for colour\n"
            "P,_ = data.synthetic_series(bear_beta=-0.0025, seed=584)\n"
            "ssP = st.sort_spread(P, lag=1, frac=0.2); pP = st.placebo_pvalue(P, lag=1, frac=0.2, n_perm=2000)\n"
            "sttP = st.strategy_stats(P, lag=1)\n"
            "print(f\"planted seed: sort spread {ssP['spread']*1e4:+.0f} bps (t {ssP['t']:+.2f}, placebo p {pP:.3f}); \"\n"
            "      f\"book gross Sharpe {sttP['gross_sharpe']:+.2f} -> net {sttP['net_sharpe']:+.2f}\")"
        ),
        md(
            f"The slope-*t* is ≈ 0 at the null and falls past −2 by `bear_beta = {R['ctrl'][2][0]}` "
            f"(mean **{R['ctrl'][2][1]}**, 25 seeds), reaching **{R['ctrl'][4][1]}** at the strong "
            "setting — and a single planted seed's book earns a real gross Sharpe "
            f"**{R['plant_gross_sharpe']}**. **So the engine is a faithful detector.** The empty "
            "real-side result is therefore a statement about **data availability**, not a blind "
            "test: the netflow tape is paywalled, so the folklore cannot be certified here. For the "
            "crypto claims a free stack *can* reach, see [133 Crypto-Seasonality](../../133-crypto-seasonality/), "
            "[210 Crypto-Trend](../../210-crypto-trend/) and [325 Crypto-Fear-Greed](../../325-crypto-fear-greed/)."
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
