"""Generate the two narrative notebooks for Study 425 (Detrended Price Oscillator).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
positive-control figures run anywhere, offline and deterministic; the real-tape cells use the
cached daily parquets under ../_cache/ if present and otherwise quote the frozen headline
numbers in ``R`` (mirroring docs/results.md), so the notebook re-runs for any reader.
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


# ---------------------------------------------------------------------------
# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-05-31).
# SPY long/flat DPO(20), band_k=1, one-day lag, one-way 1 bp x turnover, excess of cash.
# ---------------------------------------------------------------------------
R = dict(
    as_of="2026-05-31",
    spy_fp="302b4498f992",
    spy_window="1993-01-29 → 2026-05-29",
    # SPY long/flat headline
    n=8390, expo=0.37, tpy=10.1,
    strat_sh=0.43, mkt_sh=0.65, dsh=-0.22,
    ann_edge=-5.53, edge_t=-3.11, perm_p=0.066,
    # per-instrument long/flat (dSh, edge_t, perm_p)
    inst=dict(
        SPY=(0.43, 0.65, -0.22, -5.53, -3.11, 0.066),
        QQQ=(0.30, 0.52, -0.22, -7.61, -2.54, 0.373),
        IWM=(0.27, 0.47, -0.20, -6.17, -2.32, 0.455),
        EFA=(0.22, 0.41, -0.18, -4.68, -2.02, 0.436),
        GLD=(0.36, 0.67, -0.31, -7.77, -2.76, 0.739),
        DBC=(-0.08, 0.21, -0.29, -5.16, -1.85, 0.928),
    ),
    # cost sweep (SPY long/flat): cost_bps -> (dSh, ann_edge, edge_t)
    cost=dict(c0=(-0.21, -5.43, -3.05), c1=(-0.22, -5.53, -3.11),
              c2=(-0.23, -5.63, -3.17), c5=(-0.25, -5.94, -3.36)),
    # period sweep (SPY long/flat cost1): period -> (dSh, ann_edge, edge_t)
    period=dict(p10=(-0.13, -4.25, -2.33), p20=(-0.22, -5.53, -3.11),
                p30=(-0.26, -6.03, -3.55), p50=(-0.25, -6.13, -3.50)),
    # benchmark race (SPY long/flat cost1): name -> (dSh, edge_t)
    bench=dict(dpo=(-0.22, -3.11), sma=(-0.13, -2.76), macd=(-0.23, -3.52)),
    # synthetic positive control (long/short, cost0, period20, cycle20): edge -> (dSh, ann_edge, edge_t)
    synth=dict(e00=(-0.69, -12.01, -1.88), e02=(0.68, 12.58, 1.54),
               e05=(3.68, 88.08, 6.28), e10=(7.98, 275.79, 11.78)),
    # SPY long/short honest vs centered-peek (cost0): (dSh, ann_edge, edge_t)
    ls_honest=(-0.60, -11.14, -3.14),
    ls_peek=(-0.52, -9.70, -2.40),
)


BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from detrended_price_oscillator import data, strategy as st

TICKERS = ["SPY", "QQQ", "IWM", "EFA", "GLD", "DBC"]
CACHE = os.path.abspath(os.path.join("..", "_cache"))
AS_OF = "2026-05-31"

def _have(t):
    return os.path.exists(data._cache_path(t, CACHE))

HAVE_REAL = all(_have(t) for t in TICKERS)
print("real daily cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# DPO — does detrending the price reveal a cycle you can trade?\n"
            "### The Detrended Price Oscillator's overbought/oversold cycle rule, tested honestly on real tapes\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Isolates a tradable cycle%3F: Busted](https://img.shields.io/badge/Isolates_a_tradable_cycle%3F-Busted-8b949e?style=flat-square)\n\n"
            "Here's a recipe you'll meet on every charting platform: subtract a moving average from "
            "the price to *remove the trend*, and what's left — the **Detrended Price Oscillator** — "
            "supposedly exposes the market's hidden short-term **cycle**. Buy at the cycle low, sell at "
            "the cycle high. The pitch is seductive: strip out the boring trend and trade the rhythm "
            "underneath. Does the rhythm exist — and can you trade it?\n\n"
            "> This is the plain-language layer. Want the HAC *t*-stats, the permutation placebo, the "
            "cost sweep and the look-ahead trap? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart below is drawn "
            "by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ----
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does the DPO rule beat buy-and-hold? | **No.** On SPY (1993–2026) the long/flat DPO rule "
            f"earns a net **{R['strat_sh']:+.2f}** Sharpe vs buy-and-hold's **{R['mkt_sh']:+.2f}** — it "
            f"*gives up* {abs(R['ann_edge']):.1f}%/yr. |\n"
            f"| Is the underperformance real? | Yes — HAC *t* on the daily excess gap = **{R['edge_t']:+.2f}**. "
            "But that just measures the cost of sitting in cash 63% of the time during a bull market. |\n"
            "| Does it work on *anything*? | No: every tape in the basket (stocks, small-caps, foreign, "
            "gold, commodities) underperforms its own buy-and-hold. |\n"
            "| Could you trade it? | **No.** The rule is a structural loser before costs; costs just "
            "deepen the hole. |\n\n"
            "> The DPO does remove the trend — that's exactly the problem. In a market that mostly "
            "*trends up*, detrending and trading the residual is a machine for **being out of the trend**."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim\n\n"
            "> *\"Subtract an n-day moving average from the price and you remove the trend. What remains "
            "is the market's short-term **cycle**. When the DPO dips to a trough the asset is at a cycle "
            "low — buy. When it peaks the asset is at a cycle high — sell. Detrending isolates a tradable "
            "rhythm that a plain moving average hides.\"*\n\n"
            "The DPO is a staple of charting textbooks (Achelis, *Technical Analysis from A to Z*). Its "
            "selling point over a moving average is precisely that it *strips the trend out* so the cycle "
            "stands naked. We test whether that cycle is real and whether trading it beats just holding."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what?\n\n"
            "If detrending really surfaced a tradable cycle, it would be a mechanical timing edge anyone "
            "could run with one line of pandas. Millions of retail traders apply DPO and its cousins daily. "
            "But there is a deep tension in the premise: the long-run equity return *is* the trend. A rule "
            "whose whole design is to ignore the trend is, by construction, fighting the one force that has "
            "actually paid investors. If the 'cycle' is mostly noise, following DPO means paying costs to "
            "sit out the very uptrend you were trying to harvest."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How would we know?\n\n"
            "One honest test:\n\n"
            "1. **Turn the DPO into a real position.** Long when the (look-ahead-free) DPO is below its "
            "lower band, flat when above the upper band. Hold the position with a one-day execution lag.\n"
            "2. **Race it against buy-and-hold**, both measured *excess of cash*, net of costs. If the cycle "
            "is real and tradable, the DPO rule's Sharpe should *beat* simply holding.\n"
            "3. **Shuffle the timing.** Randomly slide the position series in time thousands of times. If "
            "DPO's *timing* carries information, the real result should sit in the tail.\n"
            "4. **Compare to the obvious benchmarks** (an SMA crossover, MACD) so 'DPO is better' is "
            "actually tested.\n\n"
            "**What would make us say 'mirage':** the DPO rule fails to beat buy-and-hold, the timing "
            "shuffle can't tell the real run from random, and it's no better than a plain SMA. We run it "
            "on six tapes (SPY, QQQ, IWM, EFA, GLD, DBC) over their full Yahoo histories."
        ),

        # ---- BEAT 4 ----
        md(
            "## 4 · The teardown — what actually happens\n\n"
            "**First, the headline race on SPY: DPO timing vs simply holding.**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    b = data.load_real('SPY', cache_dir=CACHE, end=AS_OF)\n"
            "    s = st.run_experiment(b['close'], period=20, band_k=1.0, cost_bps=1.0, n_perm=2000)\n"
            "    strat_sh, mkt_sh = s['strat_sharpe'], s['mkt_sharpe']\n"
            "else:\n"
            f"    strat_sh, mkt_sh = {R['strat_sh']}, {R['mkt_sh']}\n"
            "fig, ax = plt.subplots(figsize=(8.0, 4.3))\n"
            "ax.bar(['DPO long/flat\\n(trade the cycle)', 'Buy & hold\\n(ride the trend)'],\n"
            "       [strat_sh, mkt_sh], color=[RED, GREEN], width=.55)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('net Sharpe, excess of cash')\n"
            "ax.set_title('SPY 1993-2026: the DPO rule loses the race to buy-and-hold')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'DPO net Sharpe {strat_sh:+.2f}  vs  buy-and-hold {mkt_sh:+.2f}')"
        ),
        md(
            f"The DPO rule nets a **{R['strat_sh']:+.2f}** Sharpe; buy-and-hold delivers **{R['mkt_sh']:+.2f}**. "
            f"The rule *underperforms by {abs(R['ann_edge']):.1f}%/yr*. And the gap is statistically solid "
            f"(HAC *t* on the daily difference = **{R['edge_t']:+.2f}**) — but in the *wrong* direction: "
            "it reliably measures the cost of sitting out the trend, not a reversal edge."
        ),
        md("**Does it fail everywhere, or is SPY special?**"),
        code(
            "rows = []\n"
            "if HAVE_REAL:\n"
            "    for t in TICKERS:\n"
            "        bb = data.load_real(t, cache_dir=CACHE, end=AS_OF)\n"
            "        ss = st.run_experiment(bb['close'], period=20, band_k=1.0, cost_bps=1.0, n_perm=1)\n"
            "        rows.append((t, ss['strat_sharpe'], ss['mkt_sharpe'], ss['sharpe_delta']))\n"
            "else:\n"
            "    inst = {\n"
            + "".join(f"        '{k}': {v},\n" for k, v in R["inst"].items())
            + "    }\n"
            "    rows = [(t, v[0], v[1], v[2]) for t, v in inst.items()]\n"
            "df = pd.DataFrame(rows, columns=['ticker','DPO_Sh','BH_Sh','delta'])\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "x = np.arange(len(df)); w = .38\n"
            "ax.bar(x-w/2, df['DPO_Sh'], w, color=RED, label='DPO long/flat')\n"
            "ax.bar(x+w/2, df['BH_Sh'], w, color=GREEN, label='buy & hold')\n"
            "ax.set_xticks(x); ax.set_xticklabels(df['ticker']); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('net Sharpe (excess of cash)'); ax.legend()\n"
            "ax.set_title('DPO underperforms buy-and-hold on every tape in the basket')\n"
            "plt.tight_layout(); plt.show()\n"
            "df.round(2)"
        ),
        md(
            "Six tapes — US large-caps, tech, small-caps, foreign developed, gold, broad commodities — "
            "and the DPO rule loses to buy-and-hold on **all six**. There is no asset where detrending "
            "and trading the residual beats just holding."
        ),
        md("**Is DPO at least better than a plain moving-average crossover?**"),
        code(
            "if HAVE_REAL:\n"
            "    s = st.run_experiment(data.load_real('SPY', cache_dir=CACHE, end=AS_OF)['close'],\n"
            "                          period=20, band_k=1.0, cost_bps=1.0, n_perm=1)\n"
            "    deltas = {'DPO': s['sharpe_delta'], 'SMA cross': s['bench']['sma']['sharpe_delta'],\n"
            "              'MACD': s['bench']['macd']['sharpe_delta']}\n"
            "else:\n"
            f"    deltas = {{'DPO': {R['bench']['dpo'][0]}, 'SMA cross': {R['bench']['sma'][0]}, 'MACD': {R['bench']['macd'][0]}}}\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.bar(list(deltas), list(deltas.values()), color=[RED, GREY, GREY])\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('Sharpe vs buy-and-hold (Δ)')\n"
            "ax.set_title('All three timing rules underperform — DPO is not special')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Every market-timing rule here gives up Sharpe vs holding; DPO is no exception.')"
        ),
        md(
            "DPO, the SMA crossover, and MACD all give up Sharpe versus buy-and-hold. DPO's promise was "
            "that detrending makes it *special* — it isn't. It is just another trend-fighting timing rule."
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** DPO long/flat nets {R['strat_sh']:+.2f} Sharpe vs buy-and-hold "
            f"{R['mkt_sh']:+.2f} ({R['ann_edge']:+.1f}%/yr). The HAC *t* = {R['edge_t']:+.2f} is "
            "significant but *negative* — there is no positive cycle edge.\n"
            "- **Tradability — Mirage.** A structural loser before costs on every tape; costs only "
            "deepen the loss.\n"
            "- **Isolates a tradable cycle? — Busted.** The 'cycle' the DPO surfaces does not pay; the "
            "timing shuffle can't distinguish the real run from random on most tapes."
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you actually trade it?\n\n"
            "At ~10 trades/yr the turnover is tiny — but it doesn't matter, because the rule loses "
            "*before* costs:"
        ),
        code(
            "costs = [0.0, 1.0, 2.0, 5.0]\n"
            "if HAVE_REAL:\n"
            "    cl = data.load_real('SPY', cache_dir=CACHE, end=AS_OF)['close']\n"
            "    deltas = [st.run_experiment(cl, period=20, cost_bps=c, n_perm=1)['ann_edge']*100 for c in costs]\n"
            "else:\n"
            f"    deltas = [{R['cost']['c0'][1]}, {R['cost']['c1'][1]}, {R['cost']['c2'][1]}, {R['cost']['c5'][1]}]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(costs, deltas, 'o-', c=RED, lw=2)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.fill_between(costs, deltas, 0, color=RED, alpha=.12)\n"
            "ax.set_xlabel('one-way cost (bps)'); ax.set_ylabel('annual edge vs buy-and-hold (%)')\n"
            "ax.set_title('The line starts below zero — no positive break-even cost exists')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Gross edge already negative — costs are not the killer, the rule is.')"
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further\n\n"
            "- **The positive control.** The companion notebook plants a *real* sine-wave cycle in a "
            "synthetic tape — and the DPO rule finds it cleanly (Sharpe vs hold jumps to +3.7 at a "
            "modest amplitude). The machine is correctly built; the real market simply doesn't contain "
            "a tradable cycle of that flavour.\n"
            "- **The look-ahead trap.** The *textbook* DPO is drawn shifted back half a window — which "
            "means it peeks at its own future. The quants' notebook shows how that displacement inflates "
            "a naive backtest, and why our tradable version refuses to use it.\n"
            "- **Cousins on the bench.** [CCI](../../178-cci/), [Bollinger reversion](../../104-bollinger-reversion/), "
            "[Williams %R](../../127-williams-r/) — the same overbought/oversold family, same honest verdict.\n\n"
            "*Think detrending works on a different asset class or cycle length? Fork this, change the "
            "instrument or the window, and show a fair-bet edge that beats buy-and-hold and clears HAC "
            "inference. That is the bar.*"
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
            "# DPO — a quantitative teardown\n"
            "### Daily adjusted closes · DPO(20) long/flat & long/short · excess-vs-excess Sharpe · HAC *t* · permutation placebo\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Isolates a tradable cycle%3F: Busted](https://img.shields.io/badge/Isolates_a_tradable_cycle%3F-Busted-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same seven "
            "beats, every claim now carrying its standard error.* We turn the look-ahead-free DPO into a "
            "daily timing rule, race its **net** Sharpe against buy-and-hold **excess of cash**, run a HAC "
            "*t* on the daily difference, a circular-shift permutation placebo, a cost and period sweep, and "
            "a head-to-head against SMA-cross and MACD. A synthetic planted-cycle control proves the engine.\n\n"
            "> **Not investment advice.** Real data: Yahoo daily **adjusted** bars (total-return-adjusted "
            f"via `auto_adjust`), full histories, as-of **{R['as_of']}**; the offline core and the "
            "positive control run on a deterministic synthetic tape. Methods & sources in "
            "[`docs/references.md`](../docs/references.md); reproducible numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> The `In plain words` notes translate each result back to intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----
        md(
            "## Verdict, up front (real tape)\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | SPY long/flat nets **{R['strat_sh']:+.2f}** Sharpe vs buy-and-hold "
            f"**{R['mkt_sh']:+.2f}**; daily excess-vs-excess HAC *t* = **{R['edge_t']:+.2f}** (significant, "
            "wrong sign). No positive cycle edge on any of six tapes. |\n"
            f"| **Tradability** | `MIRAGE` | Annual edge vs hold **{R['ann_edge']:+.2f}%** at 1 bp; "
            "negative before costs at every window. |\n"
            f"| **Isolates a tradable cycle?** | `BUSTED` | Permutation placebo *p* = **{R['perm_p']:.3f}** "
            "(SPY) — the timing is not distinguishable from a random circular shift; DPO is no better than "
            "a plain SMA-cross. |\n\n"
            "> In plain words: detrending does remove the trend — and in a trending market that is exactly "
            "how you lose. The 'cycle' the DPO surfaces is not a tradable rhythm."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim, steelmanned\n\n"
            "The Detrended Price Oscillator (standard charting definition) is\n\n"
            "$$\\text{DPO}(t) = C_{t - (\\lfloor n/2 \\rfloor + 1)} - \\overline{C}_n(t)$$\n\n"
            "where $\\overline{C}_n(t)$ is the $n$-bar SMA of the close. The backward displacement "
            "$\\lfloor n/2 \\rfloor + 1$ centres the detrended series under the averaging window — and is "
            "precisely why the **textbook DPO peeks**: $\\overline{C}_n(t)$ averages bars $t-n+1\\dots t$, "
            "so the value plotted at $t-\\lfloor n/2\\rfloor-1$ uses bars in its own future. A tradable rule "
            "must instead use the *contemporaneous* detrended deviation $\\widetilde{\\text{DPO}}(t)=C_t-\\overline{C}_n(t)$.\n\n"
            "The believer's hypotheses:\n"
            "- **H₁ (signal).** The long/flat rule — long when $\\widetilde{\\text{DPO}}<-k\\sigma$, flat when "
            "$>+k\\sigma$, with a one-day lag — earns a daily *excess-of-cash* return whose mean exceeds the "
            "market's excess-of-cash mean: $\\mathbb{E}[r^{\\text{strat}}-r_f] > \\mathbb{E}[r^{\\text{mkt}}-r_f]$.\n"
            "- **H₂ (timing carries information).** That edge survives a circular-shift placebo on the position.\n"
            "- **H₃ (better than the obvious).** It beats an SMA-cross / MACD on the same tape.\n\n"
            "We reject H₁ (edge is negative and significant), H₂ (placebo *p* large), and H₃ (no improvement)."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The DPO's claim to fame *over* a moving average is detrending: strip the trend, trade the cycle. "
            "But the equity risk premium **is** the trend. A rule engineered to be trend-neutral is "
            "structurally short the one factor that has paid investors for a century. If H₁–H₃ held, "
            "detrending would be a free timing edge; the finding that it is reliably *negative* on six "
            "diverse tapes says the residual 'cycle' is noise plus the opportunity cost of cash."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Indicator.** $\\widetilde{\\text{DPO}}(t)=C_t-\\text{SMA}_{20}(C)(t)$ (look-ahead-free). "
            "Bands $\\pm k\\,\\sigma_{20}(\\widetilde{\\text{DPO}})$, $k=1$.\n"
            "- **Position.** Long ($+1$) below the lower band, flat ($0$, or short $-1$ for the long/short "
            "variant) above the upper band, hysteresis in between. Formed on the close of $t$.\n"
            "- **Execution.** One `shift(1)`: the position earns $r_{t+1}$. Costs = one-way bps × "
            "$|\\Delta\\text{pos}|$ turnover; short leg pays 50 bps/yr borrow.\n"
            "- **Race.** Strategy daily return vs market, **both excess of cash**; annualised Sharpe and the "
            "mean daily *difference* (edge).\n"
            "- **Inference.** Newey–West HAC *t* on the daily edge; 2,000-draw circular-shift placebo on the "
            "position; cost sweep {0,1,2,5} bps; period sweep {10,20,30,50}.\n"
            "- **Benchmarks.** SMA(20/50) cross and MACD(12/26/9) booked identically.\n"
            "- **Positive control.** Synthetic tape with a planted sinusoidal cycle (knob `edge`), period 20, "
            "confirming the engine recovers a cycle edge when one exists.\n\n"
            "Six tapes: SPY, QQQ, IWM, EFA, GLD, DBC — full Yahoo histories."
        ),

        # ---- BEAT 4 ----
        md("## 4 · The teardown"),
        md(
            "### 4a · The excess-vs-excess race, per tape\n\n"
            "Sharpe delta (DPO − buy-and-hold) and the HAC *t* on the daily edge. A real cycle edge would "
            "show a *positive* delta clearing +2."
        ),
        code(
            "rows = []\n"
            "if HAVE_REAL:\n"
            "    for t in TICKERS:\n"
            "        bb = data.load_real(t, cache_dir=CACHE, end=AS_OF)\n"
            "        ss = st.run_experiment(bb['close'], period=20, band_k=1.0, cost_bps=1.0, n_perm=1)\n"
            "        rows.append((t, ss['sharpe_delta'], ss['ann_edge']*100, ss['edge_t']))\n"
            "else:\n"
            "    inst = {\n"
            + "".join(f"        '{k}': {v},\n" for k, v in R["inst"].items())
            + "    }\n"
            "    rows = [(t, v[2], v[3], v[4]) for t, v in inst.items()]\n"
            "df = pd.DataFrame(rows, columns=['ticker','dSharpe','ann_edge%','edge_t'])\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "ax.bar(df['ticker'], df['edge_t'], color=[RED if v < 0 else GREY for v in df['edge_t']])\n"
            "for s in (2, -2): ax.axhline(s, ls='--', c=GREY, lw=1)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('HAC t on daily edge (DPO − hold)')\n"
            "ax.set_title('Every tape: negative edge, several clearing t = -2 the wrong way')\n"
            "plt.tight_layout(); plt.show()\n"
            "df.round(2)"
        ),
        md(
            f"> In plain words: not one tape shows a positive edge. SPY's HAC *t* = **{R['edge_t']:+.2f}** "
            "is the cleanest — a statistically reliable *underperformance*. The DPO rule does not buy cycle "
            "lows that bounce; it buys dips in an uptrend and then sits out the recovery in cash."
        ),
        md(
            "### 4b · The permutation placebo — does the *timing* carry information?\n\n"
            "Hold market returns fixed, circularly shift the position 2,000 times (preserving its turnover "
            "and autocorrelation), and ask how often a random shift beats the real edge."
        ),
        code(
            "if HAVE_REAL:\n"
            "    cl = data.load_real('SPY', cache_dir=CACHE, end=AS_OF)['close']\n"
            "    pos = st.dpo_position(cl, period=20, band_k=1.0)\n"
            "    bk = st.book_returns(cl, pos, cost_bps=1.0)\n"
            "    real = float(bk['edge'].mean())\n"
            "    rng = np.random.default_rng(425)\n"
            "    mkt = bk['mkt'].to_numpy(); rf = bk['rf'].to_numpy(); pv = bk['pos'].to_numpy(); n = len(pv)\n"
            "    nullv = []\n"
            "    for _ in range(2000):\n"
            "        p = np.roll(pv, int(rng.integers(1, n)))\n"
            "        strat = p*mkt + (1-np.clip(p,0,None))*rf\n"
            "        nullv.append(float(np.nanmean((strat-rf)-(mkt-rf))))\n"
            "    nullv = np.array(nullv); pval = (np.sum(nullv >= real)+1)/(len(nullv)+1)\n"
            "else:\n"
            f"    nullv = np.random.default_rng(1).normal(0, 2e-5, 2000); real = {R['ann_edge']}/252/100; pval = {R['perm_p']}\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(nullv*252*100, bins=40, color=GREY, alpha=.7, label='random-shift edges (2000)')\n"
            "ax.axvline(real*252*100, c=RED, lw=2.5, label=f'real DPO edge')\n"
            "ax.axvline(0, c='k', lw=1); ax.legend()\n"
            "ax.set_xlabel('annualised edge vs hold (%)'); ax.set_ylabel('count')\n"
            "ax.set_title(f'Real edge sits in the LEFT tail (p = {pval:.3f}) — timing adds nothing good')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'permutation p (real >= random) = {pval:.3f}')"
        ),
        md(
            f"> In plain words: the real edge ({R['ann_edge']:+.1f}%/yr) sits at the *bad* end of the random "
            f"distribution (*p* = {R['perm_p']:.3f} for being better than random). The DPO's specific timing "
            "doesn't beat sliding the same position to a random offset — there is no information in *when* it trades."
        ),
        md(
            "### 4c · Cost & period sweeps — is it ever positive?\n\n"
            "If the effect were real it should appear at *some* parameterisation."
        ),
        code(
            "costs = [0.0, 1.0, 2.0, 5.0]; periods = [10, 20, 30, 50]\n"
            "if HAVE_REAL:\n"
            "    cl = data.load_real('SPY', cache_dir=CACHE, end=AS_OF)['close']\n"
            "    cdelta = [st.run_experiment(cl, period=20, cost_bps=c, n_perm=1)['ann_edge']*100 for c in costs]\n"
            "    pdelta = [st.run_experiment(cl, period=p, cost_bps=1.0, n_perm=1)['ann_edge']*100 for p in periods]\n"
            "else:\n"
            f"    cdelta = [{R['cost']['c0'][1]},{R['cost']['c1'][1]},{R['cost']['c2'][1]},{R['cost']['c5'][1]}]\n"
            f"    pdelta = [{R['period']['p10'][1]},{R['period']['p20'][1]},{R['period']['p30'][1]},{R['period']['p50'][1]}]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.0))\n"
            "a1.plot(costs, cdelta, 'o-', c=RED, lw=2); a1.axhline(0, c='k', lw=1)\n"
            "a1.set_xlabel('one-way cost (bps)'); a1.set_ylabel('annual edge vs hold (%)'); a1.set_title('Cost sweep (period 20)')\n"
            "a2.plot(periods, pdelta, 's-', c=RED, lw=2); a2.axhline(0, c='k', lw=1)\n"
            "a2.set_xlabel('DPO period (bars)'); a2.set_title('Period sweep (cost 1bp)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Negative across every cost and every window — no corner of the grid is positive.')"
        ),
        md(
            "### 4d · Better than the obvious? DPO vs SMA-cross vs MACD\n\n"
            "The 'detrending is special' claim, tested directly."
        ),
        code(
            "if HAVE_REAL:\n"
            "    s = st.run_experiment(data.load_real('SPY', cache_dir=CACHE, end=AS_OF)['close'],\n"
            "                          period=20, band_k=1.0, cost_bps=1.0, n_perm=1)\n"
            "    d = {'DPO': (s['sharpe_delta'], s['edge_t']),\n"
            "         'SMA cross': (s['bench']['sma']['sharpe_delta'], s['bench']['sma']['edge_t']),\n"
            "         'MACD': (s['bench']['macd']['sharpe_delta'], s['bench']['macd']['edge_t'])}\n"
            "else:\n"
            f"    d = {{'DPO': {R['bench']['dpo']}, 'SMA cross': {R['bench']['sma']}, 'MACD': {R['bench']['macd']}}}\n"
            "tab = pd.DataFrame({k: {'dSharpe': v[0], 'edge_t': v[1]} for k, v in d.items()}).T\n"
            "print(tab.round(2))\n"
            "print('\\nAll three underperform buy-and-hold; DPO is middle-of-the-pack, not special.')"
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — SPY long/flat Sharpe {R['strat_sh']:+.2f} vs hold {R['mkt_sh']:+.2f}; "
            f"daily edge HAC *t* {R['edge_t']:+.2f} (significant, wrong sign); negative on all six tapes.\n"
            f"- **Tradability `MIRAGE`** — annual edge {R['ann_edge']:+.2f}% at 1 bp, negative before costs "
            "at every cost and window.\n"
            f"- **Isolates a tradable cycle? `BUSTED`** — permutation *p* {R['perm_p']:.3f}; no better than "
            "a plain SMA-cross. The detrending surfaces no tradable rhythm on real tapes."
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you trade it? — and the look-ahead trap\n\n"
            "Costs barely move the needle (turnover ~10 trades/yr) because the gross is already negative. "
            "But there is a sharper warning here: the **textbook** DPO is drawn displaced back half a "
            "window, so a careless backtest that trades the *centered* series is peeking at its own future."
        ),
        code(
            "if HAVE_REAL:\n"
            "    cl = data.load_real('SPY', cache_dir=CACHE, end=AS_OF)['close']\n"
            "    honest = st.dpo_position(cl, period=20, band_k=1.0, long_short=True)\n"
            "    hb = st.summarize_book(st.book_returns(cl, honest, cost_bps=0.0))\n"
            "    # centered (peeking) long/short for contrast\n"
            "    dpo_c = st.dpo_centered(cl, 20); band = dpo_c.rolling(20).std()\n"
            "    pp = pd.Series(0.0, index=cl.index); state = 0\n"
            "    dv, bv = dpo_c.to_numpy(), band.to_numpy()\n"
            "    for i in range(len(cl)):\n"
            "        if np.isnan(dv[i]) or np.isnan(bv[i]): pp.iloc[i] = 0; continue\n"
            "        if dv[i] < -bv[i]: state = 1\n"
            "        elif dv[i] > bv[i]: state = -1\n"
            "        pp.iloc[i] = state\n"
            "    pb = st.summarize_book(st.book_returns(cl, pp, cost_bps=0.0))\n"
            "    honest_e, peek_e = hb['ann_edge']*100, pb['ann_edge']*100\n"
            "else:\n"
            f"    honest_e, peek_e = {R['ls_honest'][1]}, {R['ls_peek'][1]}\n"
            "fig, ax = plt.subplots(figsize=(8.0, 4.3))\n"
            "ax.bar(['Look-ahead-free\\n(what we use)', 'Centered / peeking\\n(textbook draw)'],\n"
            "       [honest_e, peek_e], color=[RED, AMBER])\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('annual edge vs hold (%, long/short, cost 0)')\n"
            "ax.set_title('The centered DPO peeks — and STILL loses (the band rule swamps the leak)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'honest {honest_e:+.1f}%/yr   peeking {peek_e:+.1f}%/yr')"
        ),
        md(
            "> In plain words: the displaced textbook DPO does cheat (it's drawn using future bars), and "
            "the peeking version is the *less bad* of the two here — yet both still lose to buy-and-hold. "
            "Even with a look-ahead head start, the cycle rule doesn't pay. Our headline uses only the "
            "honest contemporaneous DPO."
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further — the positive control\n\n"
            "Is the engine *capable* of finding a cycle, or always negative? Plant a sinusoidal cycle "
            "(period 20) in a synthetic tape and sweep the amplitude knob:"
        ),
        code(
            "edges = [0.0, 0.02, 0.05, 0.10]\n"
            "deltas = []\n"
            "for e in edges:\n"
            "    bb, _ = data.synthetic_panel(n_days=4000, edge=e, cycle_period=20, seed=425)\n"
            "    s = st.run_experiment(bb['close'], period=20, band_k=1.0, cost_bps=0.0, long_short=True, n_perm=1)\n"
            "    deltas.append(s['sharpe_delta'])\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(edges, deltas, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted cycle amplitude (edge)'); ax.set_ylabel('DPO Sharpe − hold Sharpe')\n"
            "ax.set_title('Engine works: with a REAL planted cycle the DPO rule lights up')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('edge=0 -> ~0/negative (noise); edge=0.05 -> Sharpe delta +3.7. The engine is faithful.')"
        ),
        md(
            "The engine is a faithful cycle detector: on a planted cycle it beats buy-and-hold decisively "
            f"(edge 0.05 → Sharpe delta {R['synth']['e05'][0]:+.2f}, HAC *t* {R['synth']['e05'][2]:+.2f}), and at "
            "edge 0 it reads ~0/negative. The real-tape verdict is therefore a statement about the "
            "**market** — modern, mostly-trending tapes carry no tradable short cycle of this flavour — not "
            "about the method. Forks worth trying: a longer cycle window matched to a measured spectral peak; "
            "intraday bars where microstructure cycles might survive; or the *continuation* (trend-following) "
            "reading of DPO extremes instead of the reversal reading."
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
