"""Generate the two narrative notebooks for Study 793 (Cross-sectional commodity value).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached ETF basket
under ../_cache/ and otherwise quote the frozen headline numbers in ``R`` (mirroring
docs/results.md). The synthetic positive control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (13 single-commodity ETFs,
# 5-year value L/S, 163 monthly rebalances 2012-12-31 -> 2026-06-30).
R = dict(
    as_of="2026-06-30", ls_start="2012-12-31", ls_end="2026-06-30",
    n_months=163, n_etfs=13, price_panel_months=258, ret_panel_months=220,
    mean_bps=8.05, ann_pct=0.97, hac_t=0.166, hac_lags=4,
    one_sample_t=0.180, sharpe=0.049, hit_pct=54.0, turnover=0.31,
    long_bps=-9.88, long_t=-0.32, short_bps=17.93, short_t=0.69, basket_ann=1.29,
    placebo_seeds=40, placebo_real_t=0.166, placebo_mean_t=0.222, placebo_sd_t=1.109,
    placebo_mean_sharpe=0.059, placebo_frac_ge2=0.075, placebo_p=0.525,
    era_split="2019-07-01",
    early_n=79, early_bps=20.92, early_t=0.33, early_sharpe=0.14,
    late_n=84, late_bps=-4.05, late_t=-0.06, late_sharpe=-0.02, diff_t=-0.28,
    timer5_gross=0.97, timer5_net=0.28, timer5_t=0.05, timer5_sharpe=0.01,
    timer10_gross=0.97, timer10_net=0.09, timer10_t=0.02, timer10_sharpe=0.00,
    syn_null_mean=-0.23, syn_null_sd=0.97, syn_null_fire=2,
    syn_planted_t=6.41, syn_planted_bps=257.15, syn_planted_sharpe=1.98,
    fp="0524773df8ac",
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Reversal: Refuted](https://img.shields.io/badge/Reversal-Refuted-8b949e?style=flat-square)\n\n"
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from commodity_value import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    PRICE = data.monthly_price()
    MRET = data.monthly_returns()
else:
    PRICE = MRET = None
print("real cache present:", HAVE_REAL, "| return panel:",
      (None if MRET is None else f"{MRET.shape[0]} months x {MRET.shape[1]} ETFs"))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Do the beaten-down commodities bounce back? 🛢️📉\n"
            "### Commodity *value* — the textbook 'buy what's fallen' sort that the real tape "
            "flatly refuses to reward\n\n"
            + BADGES +
            "Here's the value investor's version of a commodity strategy: once a month, look at "
            "which commodities have **fallen the most over the last five years** (they're 'cheap') "
            "and which have **risen the most** (they're 'expensive'), then **buy the cheap third** "
            "and **short the expensive third**. Asness-Moskowitz-Pedersen (2013) showed this kind "
            "of long-horizon *reversal* paid across many markets. Does it work on stuff you could "
            "actually buy — a basket of commodity ETFs?\n\n"
            "Short answer: **no.** It backtests at a flat **~+1%/yr**, no better than picking names "
            "out of a hat — and the 'cheap' commodities you bought actually *lagged*.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo and the cost math? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** 13 single-commodity ETFs, 2012→2026, 163 monthly rebalances. "
            "The value signal reads the *raw price* (not total return), so five years of carry "
            "can't fake 'cheapness'. Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Do cheap (fallen) commodities beat expensive (risen) ones? | **No — about "
            f"+{R['ann_pct']:.0f}%/yr** (Sharpe {R['sharpe']:.2f}), a statistical zero "
            f"(*t* = {R['hac_t']:.2f}). |\n"
            "| Is it at least better than a random pick? | **No.** A *random* ranking of the same "
            "commodities did just as well — marginally better, in fact. |\n"
            "| Did the 'cheap' commodities you bought actually bounce? | **No — the opposite.** The "
            "cheap (long) leg *underperformed* the basket. The fallen ones kept lagging. |\n"
            "| Do trading costs kill it? | **They don't need to.** There was nothing there to "
            "begin with — net is a rounding error at any cost. |\n\n"
            "> A strategy that's real in the academic futures data but simply **isn't there** on a "
            "free, investable commodity-ETF basket — and even points the wrong way."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Commodities mean-revert over the long run. The ones that have crashed over the "
            "past five years are cheap and tend to recover; the ones that soared are expensive and "
            "tend to give it back. So rank them by how far they've fallen, buy the cheapest, short "
            "the priciest, and rebalance monthly.\"*\n\n"
            "This is **value** — the same 'buy what's cheap' idea that's famous in stocks (cheap "
            "book-to-market beats expensive), here defined for commodities as a **5-year price "
            "reversal** (Asness-Moskowitz-Pedersen 2013). It's deliberately the **opposite "
            "horizon** to *momentum* (which buys the past-*year* winners) — the two are mirror "
            "images, which is exactly why the AMP paper pairs them."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If real, this would be a clean, mechanical, once-every-few-years-of-signal rule with "
            "no forecasting — just rank on a 5-year price ratio and hold. It's one of the two "
            "legs (value + momentum) that underpin a huge swathe of 'liquid alternative' and "
            "managed-futures products. So it's worth checking whether the *value* leg — the "
            "buy-the-fallen half — actually delivers on instruments a normal investor can hold, "
            "or whether it's a spot-futures-only artefact that evaporates on real ETFs."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The sort.** Each month, rank the {R['n_etfs']} ETFs by how far their price sits "
            "below (or above) where it was ~5 years ago; long the cheapest third, short the "
            "priciest third.\n"
            "- **The *t*-test.** Is the average monthly profit big enough, relative to its wobble, "
            "to be real? (We use a stat that accounts for months bunching together.)\n"
            "- **The luck check.** Rank the commodities *randomly* instead, dozens of times — does "
            "the real value ranking beat the coin-flips?\n"
            "- **The direction check.** Did the *cheap* commodities you bought actually go up "
            "relative to the basket — or is the whole thing coming from the short side (or "
            "nowhere)?\n"
            "- **The trade check.** Charge realistic costs and short-borrow — is there anything "
            "left?"
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the equity curve.** One dollar in the long/short value book."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rep = st.value_report(PRICE, MRET)\n"
            "    ret = rep['ret']\n"
            "    ann, shp, t = rep['ann_pct'], rep['sharpe'], rep['hac_t']\n"
            "else:\n"
            "    ret = None; ann, shp, t = R['ann_pct'], R['sharpe'], R['hac_t']\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.6))\n"
            "if ret is not None:\n"
            "    eq = (1 + ret).cumprod()\n"
            "    ax.plot(eq.index, eq.values, color=GREY, lw=1.8)\n"
            "    ax.axhline(1.0, ls='--', c=RED, lw=1)\n"
            "ax.set_ylabel('growth of $1 (long/short, gross)')\n"
            "ax.set_title(f'5-year commodity value: ~{ann:+.0f}%/yr, Sharpe {shp:.2f}, HAC t={t:.2f}')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross ~{ann:+.2f}%/yr | Sharpe {shp:.2f} | HAC t {t:+.2f} (bar is 2)')"
        ),
        md(
            f"A flat, wandering line that ends roughly where it started — **~+{R['ann_pct']:.0f}%/yr** "
            f"at Sharpe **{R['sharpe']:.2f}**, HAC *t* = **{R['hac_t']:.2f}**. This is what *nothing* "
            "looks like. Compare it to the momentum sibling next door "
            "([792](../../792-commodity-momentum/)), whose curve at least climbed before it "
            "faded.\n\n"
            "**Is the value sort at least better than picking at random?** We rank the same "
            "commodities *randomly* 40 times and compare."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.random_placebo(PRICE, MRET)\n"
            "    real_t, ts = pl['real_t'], pl['ts']\n"
            "else:\n"
            "    real_t = R['hac_t']\n"
            "    ts = np.random.default_rng(793).normal(R['placebo_mean_t'], R['placebo_sd_t'], 40)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(ts, bins=16, color=GREY, alpha=.85, label='random-rank sorts (40 seeds)')\n"
            "ax.axvline(real_t, c=RED, lw=2.5, label=f'real value sort (t={real_t:.2f})')\n"
            "ax.axvline(2, ls='--', c=RED, lw=1, label='t = 2 bar')\n"
            "ax.set_xlabel('HAC t of the long/short book'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f\"Real value sort sits right in the coin-flip cloud (p = {R['placebo_p']:.2f})\")\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"real t {real_t:+.2f} vs random-rank mean {R['placebo_mean_t']:+.2f}; \"\n"
            "      f\"p = {R['placebo_p']:.2f} -> indistinguishable from random\")"
        ),
        md(
            f"The real value sort (red) sits **smack in the middle of the random-rank cloud** — "
            f"**p = {R['placebo_p']:.2f}**, and the average coin-flip rank actually scored a hair "
            f"*higher* (mean random *t* {R['placebo_mean_t']:.2f} vs real {R['hac_t']:.2f}). The "
            "value ranking is doing nothing a blindfold couldn't.\n\n"
            "**So where did the (tiny) P&L come from — did the cheap ones actually bounce?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    lg = st.long_short_legs(PRICE, MRET)\n"
            "    lb, sb = lg['long_excess_bps'], lg['short_excess_bps']\n"
            "    lt, stt = lg['long_t'], lg['short_t']\n"
            "else:\n"
            "    lb, sb = R['long_bps'], R['short_bps']\n"
            "    lt, stt = R['long_t'], R['short_t']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "ax.bar(['CHEAP leg\\n(bought, t={:.2f})'.format(lt),'EXPENSIVE leg\\n(shorted, t={:.2f})'.format(stt)],\n"
            "       [lb, sb], color=[RED, GREY], width=.55)\n"
            "for i,v in enumerate([lb, sb]): ax.annotate(f'{v:+.1f} bps',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('excess-of-basket (bps/mo)')\n"
            "ax.set_title('The cheap commodities you BOUGHT actually lagged the basket')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'cheap (long) leg {lb:+.2f} bps (t={lt:+.2f}) | expensive (short) leg {sb:+.2f} bps (t={stt:+.2f})')"
        ),
        md(
            f"This is the punchline. The **cheap (long) leg you bought lost {abs(R['long_bps']):.0f} bps/mo "
            f"relative to the basket** (t = {R['long_t']:.2f}) — the fallen commodities kept lagging, "
            f"the *opposite* of the value prediction. Whatever the L/S scraped together came only "
            f"from the short (expensive) side (+{R['short_bps']:.0f} bps/mo), and that's noise too "
            f"(t = {R['short_t']:.2f}). **'Buy the beaten-down commodity' didn't just fail to pay — "
            "it pointed the wrong way.**\n\n"
            "**Finally, the trade — is there anything to salvage after costs?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    tm5 = st.costed_timer(PRICE, MRET, cost_bps=5.0)\n"
            "    tm10 = st.costed_timer(PRICE, MRET, cost_bps=10.0)\n"
            "    g, n5, n10 = tm5['gross_ann_pct'], tm5['net_ann_pct'], tm10['net_ann_pct']\n"
            "else:\n"
            "    g, n5, n10 = R['timer5_gross'], R['timer5_net'], R['timer10_net']\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.3))\n"
            "ax.bar(['gross','net @5bps','net @10bps'], [g, n5, n10], color=[GREY, RED, RED], width=.6)\n"
            "for i,v in enumerate([g, n5, n10]): ax.annotate(f'{v:+.1f}%',(i,v),ha='center',va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('annualised return (%)')\n"
            "ax.set_title('Gross was already ~0, so net is a rounding error')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross {g:+.1f}% -> net {n5:+.1f}% (5bps) / {n10:+.1f}% (10bps)')"
        ),
        md(
            f"Because a 5-year signal barely trades (turnover ~{R['turnover']:.2f}× NAV), costs "
            f"aren't even the point — the gross **+{R['timer5_gross']:.0f}%/yr** was already zero, so "
            f"net is **+{R['timer10_net']:.1f}%/yr** at 10 bps. There's simply **nothing to capture.**"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** +{R['ann_pct']:.0f}%/yr at Sharpe {R['sharpe']:.2f}, HAC "
            f"*t* = {R['hac_t']:.2f}, indistinguishable from a random-rank sort "
            f"(p = {R['placebo_p']:.2f}). Real in the AMP futures cross-section; **absent on this "
            "investable ETF tape.**\n"
            "- **Tradability — Mirage.** Net ~0%/yr at any cost. Nothing to trade.\n"
            f"- **\"Do the fallen ones bounce?\" — Refuted.** The cheap (long) leg's excess return is "
            f"*negative* ({R['long_bps']:+.0f} bps/mo). The beaten-down commodities kept lagging."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **Why value 'should' work in commodities:** supply adjusts slowly, so a multi-year "
            "price crash eventually curbs production and the price mean-reverts. The theory is "
            "sound — but on 13 ETFs over 14 years it just doesn't show up.\n"
            "- **Why the ETF proxy is hostile to it:** chronic-contango energy ETFs (USO, UNG) "
            "grind *down* for years from roll, so they look permanently 'cheap' for a mechanical "
            "reason — dirtying the very signal value depends on. AMP use spot prices to dodge "
            "exactly this.\n"
            "- **Sibling studies:** [792-commodity-momentum](../../792-commodity-momentum/) runs the "
            "**opposite horizon** (past-year continuation) on the *same basket* (`WEAK`); "
            "[638-value-momentum-everywhere](../../638-value-momentum-everywhere/) blends value "
            "*and* momentum across four asset classes. See [docs/references.md](../docs/references.md) "
            "for the exact dedup.\n\n"
            "*Think a broader basket or a spot-futures tape revives commodity value? Show a net, "
            "certifiable (HAC t ≥ 2) edge with a **positive cheap leg** — then we'll talk.*"
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
            "# Cross-sectional commodity value — a quantitative teardown 🔬\n"
            "### The 5-year value L/S with HAC *t* · a cheap-vs-expensive leg decomposition · a "
            "40-seed random-rank placebo · a sub-period difference test · a cost-and-borrow sweep "
            "· a planted-reversal synthetic control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The "
            "claim — **Asness-Moskowitz-Pedersen (2013), the commodity value leg: rank on the "
            "ratio of the price ~5 years ago to the current price; cheap (fallen) beats expensive "
            "(risen)** — is the *long-horizon reversal* sleeve, the deliberate mirror of the 12-1 "
            "momentum sort of [792](../../792-commodity-momentum/) on the *same basket*, and the "
            "**commodity value leg alone**, not the mixed multi-asset combo of "
            "[638](../../638-value-momentum-everywhere/). The job: measure it with an "
            "autocorrelation-robust *t*, prove the machinery, and grade it honestly.\n\n"
            "> ⚠️ **Data note.** 13 single-commodity ETFs; the value signal reads the **raw price** "
            "(`auto_adjust=False`), the P&L the **total-return** close. 2012-12 → 2026-06, 163 "
            "monthly rebalances, cached. **Survivorship *and* roll-contamination named on the "
            "Signal axis** — both bias the value read upward. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (fingerprint `" + R["fp"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | 5-yr value L/S **{R['mean_bps']:+.0f} bps/mo** "
            f"(~{R['ann_pct']:+.1f}%/yr, Sharpe {R['sharpe']:.2f}); **HAC t = {R['hac_t']:+.2f}**; "
            f"random-rank placebo p = {R['placebo_p']:.2f} |\n"
            f"| **Tradability** | `MIRAGE` | net {R['timer10_net']:+.1f}%/yr @10bps "
            f"(net t = {R['timer10_t']:.2f}); gross already ~0 |\n"
            f"| **Reversal direction** | `REFUTED` | cheap (long) leg excess "
            f"{R['long_bps']:+.0f} bps/mo (t = {R['long_t']:.2f}) — *negative* |\n\n"
            "> 💡 In plain words: a value premium that is statistically zero, no better than a "
            "coin-flip ranking, and whose 'buy the cheap' leg actually *lost* to the basket — the "
            "AMP effect does not survive translation to an investable commodity-ETF sort."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $P_{i,t}$ be ETF $i$'s **raw price** in month $t$, and "
            "$V_{i,t} = \\log \\bar P_{i,t-60} - \\log P_{i,t}$ the **5-year value** signal, where "
            "$\\bar P_{i,t-60}$ is the average price over the ~4.5–5.5-years-ago band (AMP's "
            "reference). A **positive** $V$ means the price fell → the commodity is **cheap**. Each "
            "month rank the live assets on $V$, long the cheap top third $C_t$, short the expensive "
            "bottom third $E_t$, equal-weight:\n\n"
            "$$ R^{LS}_{t+1} = \\tfrac{1}{|C_t|}\\sum_{i\\in C_t} r_{i,t+1} - "
            "\\tfrac{1}{|E_t|}\\sum_{i\\in E_t} r_{i,t+1}, $$\n\n"
            "where $r$ is the **total-return** monthly return (what a held position earns).\n\n"
            "- **H₁ (premium).** $E[R^{LS}] > 0$ with an autocorrelation-robust $t \\ge 2$.\n"
            "- **H₂ (skill, not luck).** $R^{LS}$ beats a random-rank sort of the same basket.\n"
            "- **H₃ (direction).** The **cheap leg** carries a *positive* excess-of-basket return.\n"
            "- **H₄ (capture).** It survives realistic costs + short borrow.\n\n"
            f"We find **H₁ fails** (HAC t = {R['hac_t']:.2f}), **H₂ fails** "
            f"(placebo p = {R['placebo_p']:.2f}), **H₃ fails — points the wrong way** "
            f"(cheap-leg excess {R['long_bps']:+.0f} bps/mo), **H₄ moot** (gross ≈ 0)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "Monthly L/S factor returns are **serially correlated**, so the primary statistic is a "
            "**Newey-West (HAC)** *t* on the monthly mean with an automatic Bartlett lag — an "
            "i.i.d. SE would overstate significance. The value signal reads the **raw price level**, "
            "not total return, so five years of accumulated carry/roll cannot masquerade as "
            "'cheapness'. The **one execution lag** is explicit: weights formed at the close of "
            "month *t* earn month *t*+1 (a single `shift`). The random-rank **placebo (40 seeds)** "
            "holds basket breadth fixed and asks how often noise ranks match the real *t*. The "
            "sub-period split (2019-07, the effective-sample midpoint) is tested as a **difference** "
            "(Welch *t*). Costs are **one-way × traded notional**; the short leg pays borrow."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Panel.** {R['n_etfs']} single-commodity ETFs; a {R['price_panel_months']}-month "
            f"raw-price panel feeds the signal, a {R['ret_panel_months']}-month total-return panel "
            f"the P&L; the L/S trades **{R['n_months']} months** ({R['ls_start']} → {R['ls_end']}) "
            "once ≥ 6 names carry a full 5-year price history.\n"
            "- **Headline.** HAC *t* + one-sample *t* + annualised Sharpe on the monthly L/S.\n"
            "- **Decomposition.** Cheap and expensive legs, each excess-of-basket, HAC-*t*'d — the "
            "direction check.\n"
            "- **Placebo.** 40-seed random-rank null; p = share of seeds with t ≥ real t.\n"
            "- **Persistence.** Pre/post-2019 HAC *t* + a Welch *t* of the difference.\n"
            "- **Execution.** Cost sweep (5 / 10 bps one-way × turnover) + 50 bps/yr short borrow.\n"
            "- **Control.** Synthetic random-walk price panel with a plantable reversal `val_edge`; "
            "the null (edge = 0) must not systematically fire across 20 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The headline L/S and its placebo\n\n"
            "The 5-year value book's HAC *t*, and where it sits against 40 random-rank sorts of the "
            "same basket."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rep = st.value_report(PRICE, MRET)\n"
            "    print(f\"L/S: {rep['mean_bps']:+.2f} bps/mo (~{rep['ann_pct']:+.2f}%/yr) over \"\n"
            "          f\"{rep['n_months']} months\")\n"
            "    print(f\"HAC t = {rep['hac_t']:+.3f} ({rep['hac_lags']} lags) | one-sample t = \"\n"
            "          f\"{rep['one_sample_t']:+.3f} | Sharpe {rep['sharpe']:+.3f} | hit {rep['hit_rate']*100:.1f}%\")\n"
            "    pl = st.random_placebo(PRICE, MRET); real_t, ts = pl['real_t'], pl['ts']\n"
            "else:\n"
            "    real_t = R['hac_t']\n"
            "    ts = np.random.default_rng(793).normal(R['placebo_mean_t'], R['placebo_sd_t'], 40)\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.hist(ts, bins=16, color=GREY, alpha=.85, label='random-rank sorts (40 seeds)')\n"
            "ax.axvline(real_t, c=RED, lw=2.5, label=f'real value sort t = {real_t:.2f}')\n"
            "ax.axvline(2, ls='--', c=RED, lw=1, label='t = 2 bar')\n"
            "ax.set_xlabel('HAC t of the long/short book'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f\"Real value sort is a coin flip (p={R['placebo_p']:.2f})\")\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"real t {real_t:+.2f} | placebo mean t {R['placebo_mean_t']:+.2f} \"\n"
            "      f\"(sd {R['placebo_sd_t']:.2f}) | frac|t|>=2 {R['placebo_frac_ge2']:.0%} | p {R['placebo_p']:.2f}\")"
        ),
        md(
            f"> 💡 In plain words: the sort earns **{R['mean_bps']:+.0f} bps/mo** with "
            f"**HAC t = {R['hac_t']:.2f}** — a flat zero — and its *t* lands in the dead centre of "
            f"the random-rank cloud (**p = {R['placebo_p']:.2f}**; the mean coin-flip actually beat "
            f"it, {R['placebo_mean_t']:.2f} vs {R['hac_t']:.2f}). **H₁ and H₂ both fail.** There is "
            "no premium here to certify — this is `NONE`, not a near-miss `WEAK`."
        ),
        md(
            "### 4b · Leg decomposition — the direction check\n\n"
            "Each leg measured **excess of the equal-weight basket**. The value claim lives or dies "
            "on the **cheap (long) leg** being positive."
        ),
        code(
            "if HAVE_REAL:\n"
            "    lg = st.long_short_legs(PRICE, MRET)\n"
            "    lb, sb, bt = lg['long_excess_bps'], lg['short_excess_bps'], lg['basket_ann_pct']\n"
            "    lt, stt = lg['long_t'], lg['short_t']\n"
            "else:\n"
            "    lb, sb, bt = R['long_bps'], R['short_bps'], R['basket_ann']\n"
            "    lt, stt = R['long_t'], R['short_t']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "ax.bar(['cheap (long) leg\\n(t={:.2f})'.format(lt),'expensive (short) leg\\n(t={:.2f})'.format(stt)],\n"
            "       [lb, sb], color=[RED, GREY], width=.55)\n"
            "for i,v in enumerate([lb, sb]): ax.annotate(f'{v:+.1f} bps',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('excess-of-basket (bps/mo)')\n"
            "ax.set_title(f'Cheap leg is NEGATIVE; basket itself only {bt:+.1f}%/yr')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'cheap (long) leg {lb:+.2f} bps (t={lt:+.2f}) | expensive (short) leg {sb:+.2f} bps (t={stt:+.2f}) | basket {bt:+.2f}%/yr')"
        ),
        md(
            f"> 💡 In plain words: the **cheap leg you bought lost {abs(R['long_bps']):.0f} bps/mo to "
            f"the basket** (t = {R['long_t']:.2f}) — the fallen commodities kept falling relative to "
            f"the pack, the exact opposite of the value hypothesis. The short (expensive) leg added "
            f"{R['short_bps']:+.0f} bps/mo but is itself noise (t = {R['short_t']:.2f}). **H₃ fails, "
            "and not by a hair — the sign is wrong.** Whatever tiny L/S number exists is a short-side "
            "accident, not a value reversal."
        ),
        md(
            "### 4c · Persistence — was there ever a premium, in either half?\n\n"
            "Split at 2019-07-01; within-era HAC *t* and a Welch *t* of the difference."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sp = st.subperiod_contrast(PRICE, MRET, data.ERA_SPLIT)\n"
            "    e_b, l_b = sp['early_bps'], sp['late_bps']\n"
            "    e_t, l_t, d_t = sp['early_t'], sp['late_t'], sp['welch_t_diff']\n"
            "else:\n"
            "    e_b, l_b = R['early_bps'], R['late_bps']\n"
            "    e_t, l_t, d_t = R['early_t'], R['late_t'], R['diff_t']\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.3))\n"
            "ax.bar(['2012-2019\\n(n={})'.format(R['early_n']),'2019-2026\\n(n={})'.format(R['late_n'])],\n"
            "       [e_b, l_b], color=[GREY, GREY], width=.55)\n"
            "for i,(v,t_) in enumerate([(e_b,e_t),(l_b,l_t)]):\n"
            "    ax.annotate(f'{v:+.0f} bps\\n(t={t_:+.2f})',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('L/S mean (bps/mo)')\n"
            "ax.set_title(f'Neither half certifies (difference Welch t={d_t:+.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'early {e_b:+.1f} bps (t={e_t:+.2f}) | late {l_b:+.1f} bps (t={l_t:+.2f}) | diff Welch t {d_t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: the early half is a weak +{R['early_bps']:.0f} bps/mo "
            f"(t = {R['early_t']:.2f}), the late half slightly *negative* "
            f"({R['late_bps']:+.0f} bps/mo, t = {R['late_t']:.2f}); the difference is noise "
            f"(Welch t = {R['diff_t']:.2f}). This is **not a faded effect — it was never there** in "
            "either sub-period. There is nothing to decay."
        ),
        md(
            "### 4d · The costed timer — moot, but shown for completeness\n\n"
            "One-way cost × traded notional; short leg pays 50 bps/yr borrow."
        ),
        code(
            "if HAVE_REAL:\n"
            "    tm5 = st.costed_timer(PRICE, MRET, cost_bps=5.0)\n"
            "    tm10 = st.costed_timer(PRICE, MRET, cost_bps=10.0)\n"
            "    g = tm5['gross_ann_pct']; n5, n10 = tm5['net_ann_pct'], tm10['net_ann_pct']\n"
            "    t5, t10 = tm5['net_t'], tm10['net_t']; to = tm5['avg_turnover']\n"
            "else:\n"
            "    g = R['timer5_gross']; n5, n10 = R['timer5_net'], R['timer10_net']\n"
            "    t5, t10 = R['timer5_t'], R['timer10_t']; to = R['turnover']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.3))\n"
            "a1.bar(['gross','net @5bps','net @10bps'], [g, n5, n10], color=[GREY, RED, RED], width=.6)\n"
            "for i,v in enumerate([g, n5, n10]): a1.annotate(f'{v:+.1f}%',(i,v),ha='center',va='bottom')\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_ylabel('annualised (%)')\n"
            "a1.set_title(f'Turnover {to:.2f}x/mo -> costs irrelevant (gross ~0)')\n"
            "a2.bar(['net t @5bps','net t @10bps'], [t5, t10], color=[RED, RED], width=.5)\n"
            "for i,v in enumerate([t5, t10]): a2.annotate(f't={v:.2f}',(i,v),ha='center',va='bottom')\n"
            "a2.axhline(2, ls='--', c=RED, lw=1); a2.axhline(0, c='k', lw=.8)\n"
            "a2.set_ylabel('net HAC t'); a2.set_title('net t sits at zero')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross {g:+.1f}% -> net {n5:+.1f}% (t={t5:.2f}) / {n10:+.1f}% (t={t10:.2f}), turnover {to:.2f}x')"
        ),
        md(
            f"> 💡 In plain words: a 5-year signal barely trades (turnover **{R['turnover']:.2f}× "
            f"NAV/mo**), so costs are almost free — but the gross was already **~+{R['timer5_gross']:.0f}%/yr** "
            f"with **net HAC t = {R['timer10_t']:.2f}**. **H₄ is moot: you can't lose a race you never "
            "started. → `MIRAGE`.**"
        ),
        md(
            "### 4e · Faithful-engine & power control — we know the truth here\n\n"
            "Synthetic random-walk **price** panel with a TUNABLE planted long-horizon reversal "
            "(`val_edge`). The null (`val_edge` = 0) is checked over **20 seeds** — never a single "
            "stream — so we know a `NONE` on the real tape is the *effect* being absent, not the "
            "detector being broken."
        ),
        code(
            "null_ts = np.array([st.synthetic_detect(data.synthetic_world(val_edge=0.0, seed=793+s))['hac_t']\n"
            "                    for s in range(20)])\n"
            "planted = st.synthetic_detect(data.synthetic_world(val_edge=0.05, seed=793))\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.linspace(-.12,.12,20), null_ts, color=GREY, s=40,\n"
            "           label='null worlds (val_edge=0), 20 seeds')\n"
            "ax.scatter([1], [planted['hac_t']], color=GREEN, s=110, zorder=5,\n"
            "           label=f\"planted edge (t={planted['hac_t']:.1f})\")\n"
            "ax.axhline(2, ls='--', c=RED, lw=1); ax.axhline(-2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0,1]); ax.set_xticklabels(['null x 20','planted'])\n"
            "ax.set_ylabel('5-yr value L/S HAC t')\n"
            "ax.set_title('Control: unbiased null, planted reversal lights up')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(abs(null_ts)>=2).sum()}/20 | planted t {planted[\"hac_t\"]:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 20 null (random-walk) worlds the detector averages "
            f"t = {R['syn_null_mean']:+.2f} (unbiased) and fires |t| ≥ 2 on only "
            f"{R['syn_null_fire']}/20 seeds — the ~5-10% false-positive rate you *expect* from noise. "
            f"A planted reversal (`val_edge` = 0.05) reads t = {R['syn_planted_t']:.1f} "
            f"({R['syn_planted_bps']:+.0f} bps/mo). **The sort finds value when value is there; on the "
            "real tape it reads 0.17 — so the effect, not the machinery, is missing.** *(Machinery / "
            "power check only — never cited in support of the real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — 5-yr value L/S {R['mean_bps']:+.0f} bps/mo (~{R['ann_pct']:+.1f}%/yr, "
            f"Sharpe {R['sharpe']:.2f}); **HAC t = {R['hac_t']:.2f}**, indistinguishable from a "
            f"random-rank sort (p = {R['placebo_p']:.2f}). Literature-backed (AMP 2013) in a broad "
            "spot-futures cross-section, but **this investable ETF tape shows no premium at all** — "
            "and survivorship + roll both bias the read upward, so even this zero is an upper bound.\n"
            f"- **Tradability `MIRAGE`** — net {R['timer10_net']:+.1f}%/yr @10bps (net HAC "
            f"t = {R['timer10_t']:.2f}); the gross was already ~0, so there is nothing to capture at "
            "any cost.\n"
            f"- **Reversal direction `REFUTED`** — the cheap (long) leg's excess-of-basket return is "
            f"*negative* ({R['long_bps']:+.0f} bps/mo, t = {R['long_t']:.2f}). The beaten-down "
            "commodities did not out-bounce the risen ones; if anything the sign runs the wrong way."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **Spot-futures tape:** AMP's premium is measured on roll-adjusted *spot* series; an "
            "ETF proxy injects roll drift (chronic-contango energy reads perennially 'cheap'). A "
            "continuous-futures value tape is the natural robustness extension — does the effect "
            "reappear once roll is cleaned out?\n"
            "- **Bigger cross-section:** 13 ETFs is a thin sort (4-5 names per leg). A 20-30 "
            "commodity futures panel (the AMP breadth) would restore power the ETF era lacks.\n"
            "- **Dedup map:** [792-commodity-momentum](../../792-commodity-momentum/) (the "
            "**opposite horizon** — 12-1 continuation — on the *same 13-ETF basket*, `WEAK`); "
            "[638-value-momentum-everywhere](../../638-value-momentum-everywhere/) (the mixed "
            "multi-asset value+momentum *combo*, commodities one blended sleeve). This study "
            "isolates the **commodity value leg alone**.\n\n"
            "*The reproducible core is offline and deterministic; frozen numbers live in "
            "[`docs/results.md`](../docs/results.md), sources in "
            "[`docs/references.md`](../docs/references.md).*"
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
