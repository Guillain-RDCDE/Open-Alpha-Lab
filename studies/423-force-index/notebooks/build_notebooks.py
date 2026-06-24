"""Generate the two narrative notebooks for Study 423 (Force Index).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
positive control runs anywhere, offline and deterministic; the real-tape cells use the
cached daily parquets under ../_cache/ (cache-first) and otherwise quote the frozen
headline numbers in ``R`` (mirroring docs/results.md), so the notebook re-runs for any
reader. Real headline numbers live in exactly one place: the dict ``R`` below.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-23).
# Source: examples/verify.py on cached daily tapes, partial 2026-06-24 bar dropped.
R = dict(
    asof="2026-06-23",
    # SPY headline: FI(13) zero-cross long/flat, NET 1bp one-way x NAV
    spy_n=8405, spy_flips=18.0, spy_frac_in=58.4,
    fi_shp=0.30, bh_shp=0.64, fi_diff_bps=-3.450, fi_hac_t=-4.03,
    fi_ann_ret=3.27, bh_ann_ret=11.97,
    # FI period sweep (SPY, long/flat, NET 1bp)
    p2_shp=0.17, p2_t=-5.32, p2_flips=45.8,
    p13_shp=0.30, p13_t=-4.03, p13_flips=18.0,
    p50_shp=0.36, p50_t=-3.62, p50_flips=9.3,
    # Benchmark races on SPY (long/flat, NET 1bp)
    sma_shp=0.74, sma_diff=-0.722, sma_t=-0.97,
    rsi_shp=0.44, rsi_diff=-2.813, rsi_t=-3.31,
    # Cost sweep (SPY FI(13) long/flat)
    c0_shp=0.34, c0_t=-3.87, c1_shp=0.30, c1_t=-4.03,
    c2_shp=0.27, c2_t=-4.18, c5_shp=0.17, c5_t=-4.66,
    # Long/short FI(13) SPY
    ls_shp=-0.30, ls_diff=-6.983, ls_t=-4.08,
    # Permutation placebo (SPY FI(13) long/flat 1bp, 1000 draws)
    perm_gap=-0.341, perm_p=0.824,
    # Per-ticker (FI13 long/flat NET 1bp)
    tk=["SPY", "QQQ", "DIA", "IWM", "XLE", "GLD"],
    tk_fishp=[0.30, 0.24, 0.33, 0.15, 0.20, 0.64],
    tk_bhshp=[0.64, 0.52, 0.56, 0.47, 0.43, 0.64],
    tk_diff=[-3.450, -3.995, -2.734, -3.595, -3.537, -1.395],
    tk_t=[-4.03, -2.98, -2.90, -2.74, -2.21, -1.31],
    # Synthetic positive control (FI13 long/short NET 1bp, n=4000)
    syn_pe=[0.0, 0.1, 0.2, 0.4, 0.7],
    syn_fishp=[-0.31, 0.20, 0.72, 1.77, 2.92],
    syn_t=[-1.33, 0.01, 1.25, 3.46, 5.49],
    syn_null_p=0.735, syn_planted_p=0.002,
)


BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from force_index import data, strategy as st

ASOF = "2026-06-23"
TICKERS = ["SPY", "QQQ", "DIA", "IWM", "XLE", "GLD"]
CACHE = os.path.abspath(os.path.join("..", "_cache"))

def _have_cache():
    return all(os.path.exists(data._cache_path(t, CACHE)) for t in TICKERS)

HAVE_REAL = _have_cache()

def load(t):
    b = data.load_real(t, fetch=False, cache_dir=CACHE)
    return b[b.index <= ASOF]      # drop any partial as-of bar

print("real daily cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Force Index — does Elder's price-times-volume gauge call the turns?\n"
            "### Alexander Elder's Force Index zero-cross rule, tested honestly on liquid ETFs\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Flags reversals%3F: Busted](https://img.shields.io/badge/Flags_reversals%3F-Busted-8b949e?style=flat-square)\n\n"
            "Here's a recipe from one of the best-selling trading books ever written. Alexander "
            "Elder's *Force Index* multiplies a day's price change by its volume — the idea being "
            "that a big move on heavy volume is a 'forceful' move that tells you who's winning, "
            "bulls or bears. When the smoothed Force Index crosses **up** through zero, the bulls "
            "have taken over — buy. When it crosses **down**, the bears are in charge — get out. "
            "Does following that signal beat just holding the ETF?\n\n"
            "> This is the plain-language layer. Want the Sharpe race, the HAC *t*-stats, the "
            "permutation placebo and the positive control? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart below is drawn "
            "by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does the Force-Index rule beat buy-and-hold? | **No.** On SPY (1993–2026) it earns "
            f"a Sharpe of **{R['fi_shp']:.2f}** vs buy-and-hold's **{R['bh_shp']:.2f}** — "
            f"it *loses* by about **{abs(R['fi_diff_bps']):.1f} bps/day** (HAC *t* = **{R['fi_hac_t']:.2f}**). |\n"
            f"| Does it beat the *simpler* trend filter? | **No.** A plain 50/200-day moving-average "
            f"filter scores **{R['sma_shp']:.2f}** — better than buy-and-hold and far better than "
            "Force Index. The volume twist *hurts*. |\n"
            "| Does it work as a long/short? | **No, it's worse.** Shorting on the down-cross gives "
            f"a Sharpe of **{R['ls_shp']:.2f}** — negative. |\n"
            "| Could you trade it? | **No.** It's already behind buy-and-hold *before* costs; "
            "costs only widen the gap. |\n\n"
            "> The Force Index isn't random — it's *reliably worse* than doing nothing. Its "
            "zero-crosses pull you out of the market during the very up-legs that make stocks pay."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"The Force Index combines price and volume into a single number that reveals the "
            "force behind every move. When the 13-day Force Index crosses above zero, the bulls "
            "are in control — go long. When it crosses below zero, the bears have taken over — "
            "exit or go short. It flags the reversals before the crowd sees them.\"*\n\n"
            "Alexander Elder introduced the Force Index in *Trading for a Living* (1993). The "
            "intuition is genuinely appealing: a one-day jump is more meaningful if a lot of "
            "shares changed hands behind it. We test whether that intuition — turned into a "
            "mechanical zero-cross timing rule — actually beats simply holding the asset."
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the Force Index reliably called the turns, it would be a free lunch: a simple, "
            "mechanical rule that sidesteps the drawdowns of buy-and-hold while keeping the "
            "upside. Millions of retail traders learn it from Elder's books. If instead the rule "
            "*systematically* yanks you out of the market during its best up-legs, then following "
            "it isn't just useless — it's paying spread and missing return to underperform a tape "
            "you could have just held."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we know?\n\n"
            "One honest test:\n\n"
            "1. **Turn the rule into a position.** Long while the 13-day Force Index is above "
            "zero, flat while it's below. Enter at the *next* day's close (one-day lag — no "
            "peeking).\n"
            "2. **Race its Sharpe against buy-and-hold,** on the *same* tape, net of costs, "
            "comparing excess-of-cash to excess-of-cash so the part-time-in-cash rule is judged "
            "fairly.\n"
            "3. **Compare against the obvious simpler benchmark** — a 50/200 moving-average filter "
            "and an RSI filter. If the volume twist adds anything, Force Index should beat them.\n\n"
            "**What would make us say 'mirage':** if the rule's net Sharpe is below buy-and-hold "
            "with a significant *negative* t-stat on the difference. We run it on six liquid ETFs "
            "(SPY, QQQ, DIA, IWM, XLE, GLD) over their full Yahoo histories."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown — what actually happens\n\n"
            "**First: the equity curve.** Force-Index timing vs just holding SPY."
        ),
        code(
            "if HAVE_REAL:\n"
            "    b = load('SPY')\n"
            "    pos = st.fi_positions(b, period=13)\n"
            "    fi_net = st.book_returns(b, pos, cost_bps=1.0)\n"
            "    bh = st.buy_hold_returns(b)\n"
            "    fi_cum = (1 + fi_net).cumprod(); bh_cum = (1 + bh).cumprod()\n"
            "    fi_s, bh_s = st.sharpe(fi_net), st.sharpe(bh)\n"
            "    fig, ax = plt.subplots(figsize=(9.5, 4.8))\n"
            "    ax.plot(bh_cum.index, bh_cum, c=GREEN, lw=1.6, label=f'Buy & hold (Sharpe {bh_s:.2f})')\n"
            "    ax.plot(fi_cum.index, fi_cum, c=RED, lw=1.6, label=f'Force-Index timing (Sharpe {fi_s:.2f})')\n"
            "    ax.set_yscale('log'); ax.set_ylabel('growth of \\$1 (log)')\n"
            "    ax.set_title('Force-Index timing trails buy-and-hold on SPY'); ax.legend()\n"
            "    plt.tight_layout(); plt.show()\n"
            f"    print(f'Force Index Sharpe {{fi_s:.2f}}  vs  buy-and-hold {{bh_s:.2f}}')\n"
            "else:\n"
            f"    print('Force Index Sharpe {R['fi_shp']:.2f}  vs  buy-and-hold {R['bh_shp']:.2f}')"
        ),
        md(
            f"The Force-Index line ends well below buy-and-hold: a Sharpe of **{R['fi_shp']:.2f}** "
            f"against **{R['bh_shp']:.2f}**. Being out of the market roughly "
            f"**{100 - R['spy_frac_in']:.0f}%** of the time — exactly when the rule says the bears "
            "are in charge — costs more upside than it saves in drawdowns."
        ),
        md(
            "**Is Force Index even better than the *simpler* trend filter?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    r = st.run_experiment(load('SPY'), period=13, cost_bps=1.0)\n"
            "    shp = [r['fi']['sharpe_strat'], r['sma']['sharpe_strat'],\n"
            "           r['rsi']['sharpe_strat'], r['bh_sharpe']]\n"
            "else:\n"
            f"    shp = [{R['fi_shp']}, {R['sma_shp']}, {R['rsi_shp']}, {R['bh_shp']}]\n"
            "names = ['Force Index\\n(13)', 'SMA\\n50/200', 'RSI(14)\\n>50', 'Buy & hold']\n"
            "col = [RED, GREEN, AMBER, GREY]\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.bar(names, shp, color=col)\n"
            "ax.axhline(shp[-1], ls='--', c=GREY, lw=1)\n"
            "ax.set_ylabel('net Sharpe (SPY, 1bp costs)')\n"
            "ax.set_title('Force Index is the worst of the bunch — the volume twist hurts')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('The plain moving-average filter beats buy-and-hold; Force Index does not.')"
        ),
        md(
            f"The plain 50/200 moving-average filter actually *beats* buy-and-hold (**{R['sma_shp']:.2f}** "
            f"vs **{R['bh_shp']:.2f}**) — a real, if modest, trend effect. Force Index, with its "
            f"extra volume machinery, lands dead last at **{R['fi_shp']:.2f}**. The 'force' twist "
            "doesn't add information; it adds noise and churn."
        ),
        md(
            "**Does it fail everywhere, or just SPY?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    ts = []\n"
            "    for t in TICKERS:\n"
            "        r = st.run_experiment(load(t), period=13, cost_bps=1.0)\n"
            "        ts.append(r['fi_hac_t'])\n"
            "else:\n"
            f"    ts = {R['tk_t']}\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.bar(TICKERS, ts, color=[RED if v < -2 else AMBER for v in ts])\n"
            "ax.axhline(-2, ls='--', c=GREY, lw=1); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('HAC t on (Force Index − buy-and-hold) daily return')\n"
            "ax.set_title('Negative on every tape — significantly so on five of six')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Force Index trails buy-and-hold on all six ETFs.')"
        ),
        md(
            "Every single tape is negative, and five of six clear the −2 line — Force Index "
            "timing reliably *underperforms* buying and holding. Only gold (GLD) is a near-tie, "
            "and even there the rule doesn't add anything."
        ),

        # ---- BEAT 5 — VERDICT ------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** Force-Index timing scores Sharpe {R['fi_shp']:.2f} vs "
            f"buy-and-hold {R['bh_shp']:.2f} on SPY, losing by {abs(R['fi_diff_bps']):.1f} bps/day "
            f"at HAC *t* = {R['fi_hac_t']:.2f}. Negative on all six tapes.\n"
            f"- **Tradability — Mirage.** It's behind buy-and-hold *gross*; the ~{R['spy_flips']:.0f} "
            "flips/yr of costs only deepen the hole. No positive break-even cost exists.\n"
            "- **Flags reversals? — Busted.** The simpler moving-average filter beats both Force "
            "Index *and* buy-and-hold. The volume twist adds churn, not foresight."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "There's nothing to trade: the rule is behind buy-and-hold before a cent of cost, and "
            "costs make it worse. The cost sweep below just confirms there's no rescue."
        ),
        code(
            "costs = [0.0, 1.0, 2.0, 5.0]\n"
            "if HAVE_REAL:\n"
            "    b = load('SPY')\n"
            "    shp = [st.run_experiment(b, period=13, cost_bps=c)['fi_sharpe'] for c in costs]\n"
            "else:\n"
            f"    shp = [{R['c0_shp']}, {R['c1_shp']}, {R['c2_shp']}, {R['c5_shp']}]\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "ax.plot(costs, shp, 'o-', c=RED, lw=2, label='Force Index net Sharpe')\n"
            f"ax.axhline({R['bh_shp']}, ls='--', c=GREEN, lw=1.5, label='buy & hold')\n"
            "ax.set_xlabel('one-way cost (bps)'); ax.set_ylabel('net Sharpe')\n"
            "ax.set_title('Already below buy-and-hold at zero cost — costs only widen the gap')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('No break-even cost — the gross is already a loser vs holding.')"
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The positive control.** The companion notebook plants a genuine signed-volume "
            "edge in a synthetic tape — and the *same* rule finds it and beats buy-and-hold. The "
            "machinery is sound; the real market just doesn't have the structure Force Index "
            "assumes.\n"
            "- **Elder's actual system.** Elder rarely traded the Force Index alone — he paired it "
            "with his 'triple screen' trend filter. A fork could test whether the volume gauge "
            "adds anything *conditional* on a separate trend signal.\n"
            "- **FI(2) for entries.** Elder uses the 2-day Force Index for short-term timing; we "
            "tested it (see the quants notebook) and it's even worse — more churn, same lack of "
            "edge.\n\n"
            "*Think Force Index works on a different instrument or as a confirmation filter rather "
            "than a standalone signal? Fork this, change the rule, and show a net Sharpe that "
            "clears buy-and-hold with a HAC *t* ≥ 2 on the difference. That's the bar.*"
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
            "# Force Index — a quantitative teardown\n"
            "### Daily OHLCV · FI(13) zero-cross · long/flat & long/short · NET Sharpe race vs buy-and-hold · HAC inference · permutation placebo\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Flags reversals%3F: Busted](https://img.shields.io/badge/Flags_reversals%3F-Busted-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We turn Elder's 13-day "
            "Force-Index zero-cross into a daily long/flat (and long/short) timing rule and race "
            "its NET Sharpe against buy-and-hold, excess-vs-excess, with a Newey–West HAC *t* on "
            "the daily return difference, a sign-permutation placebo, a cost sweep, and the "
            "obvious simpler benchmarks (SMA, RSI).\n\n"
            "> **Not investment advice.** Real data: Yahoo daily bars, full ETF histories, "
            "total-return-adjusted close, as-of **2026-06-23** (partial 06-24 bar dropped); the "
            "offline core and the positive control run on a deterministic synthetic tape. "
            "Methods & sources in [`docs/references.md`](../docs/references.md), reproducible "
            "numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> The `In plain words` notes translate each result back to intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | FI(13) long/flat NET Sharpe **{R['fi_shp']:.2f}** vs "
            f"buy-and-hold **{R['bh_shp']:.2f}** on SPY; daily diff **{R['fi_diff_bps']:+.2f} bps**, "
            f"HAC *t* = **{R['fi_hac_t']:+.2f}**. Negative on all six tapes. |\n"
            f"| **Tradability** | `MIRAGE` | Behind buy-and-hold even gross "
            f"(Sharpe {R['c0_shp']:.2f}); ~{R['spy_flips']:.0f} flips/yr of costs widen the gap. "
            "No positive break-even. |\n"
            f"| **Flags reversals?** | `BUSTED` | A plain SMA(50/200) filter scores "
            f"**{R['sma_shp']:.2f}** (> buy-and-hold); the volume twist is strictly worse, and a "
            f"sign-permutation placebo puts the real Sharpe gap at *p* = **{R['perm_p']:.2f}** "
            "(indistinguishable from a shuffled position vector). |\n\n"
            "> In plain words: the Force Index doesn't flag reversals — its zero-crosses pull you "
            "out of the market during the up-legs that pay, and even a dumber moving-average "
            "filter beats it."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Elder's raw Force Index and its EMA-smoothed trend version:\n\n"
            "$$\\text{rawFI}_t = (C_t - C_{t-1})\\, V_t, \\qquad "
            "\\text{FI}_n(t) = \\text{EMA}_n(\\text{rawFI})$$\n\n"
            "with $n = 13$ for the trend filter ($n = 2$ for short-term triggers). The zero-cross "
            "timing rule sets position $p_t = \\mathbb{1}[\\text{FI}_{13}(t) > 0]$ (long/flat) or "
            "$p_t = \\operatorname{sign}(\\text{FI}_{13}(t))$ (long/short). Hypotheses:\n\n"
            "- **H₁ (signal).** The rule's net daily return $p_{t-1}\\,r_t - \\text{costs}$ has a "
            "higher Sharpe than buy-and-hold $r_t$, i.e. $\\mathbb{E}[r^{\\text{FI}}_t - r^{\\text{BH}}_t] > 0$.\n"
            "- **H₂ (adds value).** It beats the simpler SMA(50/200) and RSI(14) timing filters.\n"
            "- **H₃ (tradable).** A positive Sharpe gap survives realistic one-way costs.\n\n"
            "We reject H₁ (significant *negative* gap), H₂ (SMA beats it *and* buy-and-hold), and "
            "H₃ is moot."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "Force Index is one of the most widely taught price-volume oscillators in retail "
            "education (Elder's *Trading for a Living* is a perennial best-seller). If H₁–H₃ held, "
            "it would be a low-turnover (~18 flips/yr) market-timing overlay that any retail "
            "account could run. The finding that it is *significantly worse* than buy-and-hold — "
            "and worse than a plain moving-average filter — means the textbook rule systematically "
            "exits the market during the up-legs that generate the equity premium."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **FI(13).** $\\text{EMA}_{13}((C_t-C_{t-1})V_t)$; default Elder trend filter.\n"
            "- **Position.** Long/flat $p_t = \\mathbb{1}[\\text{FI}_{13}>0]$, or long/short "
            "$\\operatorname{sign}(\\text{FI}_{13})$. **One** execution lag: $p_{t-1}r_t$.\n"
            "- **Costs.** One-way turnover $|p_t - p_{t-1}|$ × NAV at `cost_bps`; shorts pay borrow "
            "(50 bps/yr) on absolute short exposure.\n"
            "- **Race.** NET Sharpe vs buy-and-hold, **excess-of-cash to excess-of-cash**; HAC "
            "(Newey–West) *t* on the *daily return difference*.\n"
            "- **Placebo.** Sign-permutation of the position vector (preserve exposure, break "
            "timing) — one-sided *p* on the Sharpe gap.\n"
            "- **Benchmarks.** SMA(50/200) and RSI(14)>50 timing filters, same engine.\n"
            "- **Positive control.** Synthetic tape with a *planted* signed-volume edge "
            "(`data.synthetic_panel`), confirming the engine recovers an edge when one exists.\n\n"
            "Six tapes: SPY, QQQ, DIA, IWM, XLE, GLD — full Yahoo histories to 2026-06-23."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Per-tape — Force Index trails buy-and-hold everywhere\n\n"
            "HAC *t* on the daily (FI − buy-and-hold) return difference. For the rule to *add* "
            "value the bars should clear **+2**; they're negative instead."
        ),
        code(
            "if HAVE_REAL:\n"
            "    recs = []\n"
            "    for t in TICKERS:\n"
            "        r = st.run_experiment(load(t), period=13, cost_bps=1.0)\n"
            "        recs.append((t, r['fi']['n_days'], r['fi_sharpe'], r['bh_sharpe'],\n"
            "                     r['fi']['mean_diff_bps'], r['fi_hac_t']))\n"
            "    tab = pd.DataFrame(recs, columns=['ticker','n','FI_shp','BH_shp','diff_bps','HAC_t'])\n"
            "else:\n"
            "    tab = pd.DataFrame({'ticker': TICKERS,\n"
            f"        'FI_shp': {R['tk_fishp']}, 'BH_shp': {R['tk_bhshp']},\n"
            f"        'diff_bps': {R['tk_diff']}, 'HAC_t': {R['tk_t']}}})\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.bar(tab['ticker'], tab['HAC_t'], color=[RED if v < -2 else AMBER for v in tab['HAC_t']])\n"
            "for s in (2, -2): ax.axhline(s, ls='--', c=GREY, lw=1)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('HAC t on (FI − buy-and-hold) daily return')\n"
            "ax.set_title('Negative on all six; significantly so on five')\n"
            "plt.tight_layout(); plt.show()\n"
            "tab.round(2)"
        ),
        md(
            "> In plain words: SPY at HAC *t* = "
            f"**{R['fi_hac_t']:+.2f}**, QQQ/DIA/IWM/XLE all below −2, only GLD a near-tie. There is "
            "no tape on which the Force-Index overlay improves on simply holding the asset."
        ),
        md(
            "### 4b · The period sweep — no FI window rescues it\n\n"
            "FI(2) (Elder's short-term trigger), FI(13) (his trend filter), FI(50) — all on SPY."
        ),
        code(
            "periods = [2, 13, 50]\n"
            "if HAVE_REAL:\n"
            "    b = load('SPY')\n"
            "    sw = [st.run_experiment(b, period=p, cost_bps=1.0) for p in periods]\n"
            "    shp = [r['fi_sharpe'] for r in sw]; tst = [r['fi_hac_t'] for r in sw]\n"
            "    bh = sw[0]['bh_sharpe']\n"
            "else:\n"
            f"    shp = [{R['p2_shp']}, {R['p13_shp']}, {R['p50_shp']}]\n"
            f"    tst = [{R['p2_t']}, {R['p13_t']}, {R['p50_t']}]; bh = {R['bh_shp']}\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.bar([f'FI({p})' for p in periods], shp, color=RED)\n"
            "ax.axhline(bh, ls='--', c=GREEN, lw=1.5, label='buy & hold')\n"
            "ax.set_ylabel('net Sharpe (SPY, 1bp)'); ax.legend()\n"
            "ax.set_title('Every FI window is below buy-and-hold; the fast FI(2) is worst')\n"
            "plt.tight_layout(); plt.show()\n"
            "for p, s, t in zip(periods, shp, tst):\n"
            "    print(f'FI({p:2d}): net Sharpe {s:.2f}, HAC t on diff {t:+.2f}')"
        ),
        md(
            f"> In plain words: the fast FI(2) churns ~{R['p2_flips']:.0f} flips/yr and lands at "
            f"Sharpe **{R['p2_shp']:.2f}** (HAC *t* = {R['p2_t']:+.2f}); even the slow FI(50) "
            f"(**{R['p50_shp']:.2f}**) trails buy-and-hold. No window is the right window."
        ),
        md(
            "### 4c · vs the simpler benchmarks — the volume twist *subtracts* value\n\n"
            "If FI's price×volume idea added information, it should beat a plain trend or momentum "
            "filter built without volume."
        ),
        code(
            "if HAVE_REAL:\n"
            "    r = st.run_experiment(load('SPY'), period=13, cost_bps=1.0)\n"
            "    rows = [('Force Index(13)', r['fi']['sharpe_strat'], r['fi']['mean_diff_bps'], r['fi']['hac_t_diff']),\n"
            "            ('SMA 50/200',      r['sma']['sharpe_strat'], r['sma']['mean_diff_bps'], r['sma']['hac_t_diff']),\n"
            "            ('RSI(14)>50',      r['rsi']['sharpe_strat'], r['rsi']['mean_diff_bps'], r['rsi']['hac_t_diff']),\n"
            "            ('Buy & hold',      r['bh_sharpe'], 0.0, float('nan'))]\n"
            "    bench = pd.DataFrame(rows, columns=['rule','Sharpe','diff_bps','HAC_t'])\n"
            "else:\n"
            "    bench = pd.DataFrame({'rule': ['Force Index(13)','SMA 50/200','RSI(14)>50','Buy & hold'],\n"
            f"        'Sharpe': [{R['fi_shp']}, {R['sma_shp']}, {R['rsi_shp']}, {R['bh_shp']}],\n"
            f"        'diff_bps': [{R['fi_diff_bps']}, {R['sma_diff']}, {R['rsi_diff']}, 0.0],\n"
            f"        'HAC_t': [{R['fi_hac_t']}, {R['sma_t']}, {R['rsi_t']}, float('nan')]}})\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.bar(bench['rule'], bench['Sharpe'], color=[RED, GREEN, AMBER, GREY])\n"
            f"ax.axhline({R['bh_shp']}, ls='--', c=GREY, lw=1)\n"
            "ax.set_ylabel('net Sharpe (SPY, 1bp)')\n"
            "ax.set_title('SMA beats buy-and-hold; Force Index is the worst rule of the four')\n"
            "plt.tight_layout(); plt.show()\n"
            "bench.round(2)"
        ),
        md(
            f"> In plain words: SMA(50/200) clears buy-and-hold (**{R['sma_shp']:.2f}** vs "
            f"**{R['bh_shp']:.2f}**, diff *t* = {R['sma_t']:+.2f} — not significantly *worse*), so "
            "a trend overlay isn't hopeless. Force Index, with its extra volume term, is the worst "
            "rule here. The volume multiplication adds churn and noise, not foresight."
        ),
        md(
            "### 4d · The sign-permutation placebo\n\n"
            "Shuffle the position vector (keep the exposure profile, destroy the timing alignment) "
            "and ask: how often does a *randomly-ordered* version of these positions beat "
            "buy-and-hold by as much as the real rule?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    pv = st.permutation_pvalue(load('SPY'), period=13, cost_bps=1.0, n_perm=1000)\n"
            "    gap, p = pv['real_gap'], pv['p_value']\n"
            "else:\n"
            f"    gap, p = {R['perm_gap']}, {R['perm_p']}\n"
            "print(f'Real FI Sharpe gap vs buy-and-hold: {gap:+.3f}')\n"
            "print(f'Sign-permutation p-value (gap >= real): {p:.3f}')\n"
            "print('The real rule is on the wrong side of zero AND no better than a shuffle —')\n"
            "print('there is no genuine timing skill to detect.')"
        ),
        md(
            f"> In plain words: the real Sharpe gap is **{R['perm_gap']:+.2f}** (negative — the rule "
            f"*loses*), and a shuffled position vector beats it about **{R['perm_p']*100:.0f}%** of "
            "the time. There is no timing information in the Force-Index zero-cross beyond its "
            "exposure profile."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — FI(13) long/flat NET Sharpe {R['fi_shp']:.2f} vs buy-and-hold "
            f"{R['bh_shp']:.2f}; daily diff {R['fi_diff_bps']:+.2f} bps, HAC *t* {R['fi_hac_t']:+.2f}; "
            "negative on all six tapes; permutation *p* = "
            f"{R['perm_p']:.2f}.\n"
            f"- **Tradability `MIRAGE`** — behind buy-and-hold even at zero cost "
            f"(Sharpe {R['c0_shp']:.2f}); long/short is *negative* ({R['ls_shp']:.2f}); no positive "
            "break-even cost.\n"
            f"- **Flags reversals? `BUSTED`** — SMA(50/200) beats both Force Index and "
            "buy-and-hold; the price×volume twist subtracts value rather than adding it."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the cost landscape & the short side\n\n"
            "There is no edge to cost against. The sweep confirms the gross is already below "
            "buy-and-hold, and the long/short version is outright negative."
        ),
        code(
            "costs = [0.0, 1.0, 2.0, 5.0]\n"
            "if HAVE_REAL:\n"
            "    b = load('SPY')\n"
            "    shp = [st.run_experiment(b, period=13, cost_bps=c)['fi_sharpe'] for c in costs]\n"
            "    tst = [st.run_experiment(b, period=13, cost_bps=c)['fi_hac_t'] for c in costs]\n"
            "    ls = st.run_experiment(b, period=13, cost_bps=1.0, long_short=True)\n"
            "    ls_shp, ls_t = ls['fi_sharpe'], ls['fi_hac_t']\n"
            "else:\n"
            f"    shp = [{R['c0_shp']}, {R['c1_shp']}, {R['c2_shp']}, {R['c5_shp']}]\n"
            f"    tst = [{R['c0_t']}, {R['c1_t']}, {R['c2_t']}, {R['c5_t']}]\n"
            f"    ls_shp, ls_t = {R['ls_shp']}, {R['ls_t']}\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.plot(costs, shp, 'o-', c=RED, lw=2, label='FI long/flat net Sharpe')\n"
            f"ax.axhline({R['bh_shp']}, ls='--', c=GREEN, lw=1.5, label='buy & hold')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('one-way cost (bps)'); ax.set_ylabel('net Sharpe')\n"
            "ax.set_title('Below buy-and-hold at every cost; long/short is negative')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Long/short FI(13): net Sharpe {ls_shp:+.2f}, HAC t on diff {ls_t:+.2f}')"
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further — the positive control\n\n"
            "Is the *engine* capable of banking a price-volume timing edge, or is it always "
            "negative? Plant a genuine signed-volume edge in a synthetic tape and sweep the knob:"
        ),
        code(
            "pes = [0.0, 0.1, 0.2, 0.4, 0.7]\n"
            "gaps = []\n"
            "for pe in pes:\n"
            "    b, _ = data.synthetic_panel(n_days=4000, planted_edge=pe, seed=423)\n"
            "    r = st.run_experiment(b, period=13, cost_bps=1.0, long_short=True)\n"
            "    gaps.append(r['fi']['sharpe_strat'] - r['fi']['sharpe_bench'])\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.plot(pes, gaps, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(0, c='k', lw=1); ax.axvline(0, ls='--', c=GREY)\n"
            "ax.set_xlabel('planted signed-volume edge'); ax.set_ylabel('FI − buy-and-hold Sharpe')\n"
            "ax.set_title('Engine works: FI recovers a planted edge when one exists')\n"
            "plt.tight_layout(); plt.show()\n"
            "for pe, g in zip(pes, gaps):\n"
            "    print(f'planted={pe:.2f}: FI − BH Sharpe gap {g:+.2f}')"
        ),
        md(
            "The engine is a faithful detector: on a tape where yesterday's signed dollar-volume "
            "flow genuinely predicts tomorrow's return, the FI rule finds it and beats buy-and-hold "
            f"(the gap rises monotonically with the planted edge; HAC *t* climbs to "
            f"**{R['syn_t'][-1]:+.2f}** at the largest plant, permutation *p* = "
            f"**{R['syn_planted_p']:.3f}**). At a planted edge of zero it loses, just like the real "
            f"tape (permutation *p* = **{R['syn_null_p']:.2f}**). The real-tape verdict is therefore "
            "a statement about the **market** — modern liquid US equities don't carry a tradable "
            "signed-volume reversal signal at the daily horizon — not about the method. Forks worth "
            "trying: Force Index as a *confirmation* filter inside Elder's triple-screen system, or "
            "on intraday bars where volume microstructure is richer."
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
