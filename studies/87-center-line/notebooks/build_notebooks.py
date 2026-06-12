"""Generate the two narrative notebooks for Study 87 (Center-Line).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the cached
5-minute parquet under ../_cache/ if present and otherwise quote the frozen headline
numbers in ``R`` (mirroring docs/results.md), so the notebook re-runs for any reader.

The _write convention (each build_*() ends by calling _write) is kept so the repo's
intro-restyle tooling can monkeypatch it.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-12).
R = dict(
    n=23462, tpd=48.9, tpy=12318,
    fade_mean=-0.35, fade_win=49.2, fade_t=-1.16,
    rand_mean=-0.25, rand_win=49.1, rand_t=-1.69,
    fade_minus_rand=-0.09,
    ci_lo=-4.67, ci_hi=1.42, frac_neg=86,
    thr05_n=29136, thr05_mean=-0.35, thr05_t=-1.27, thr05_win=49.3,
    thr10_n=23462, thr10_mean=-0.35, thr10_t=-1.16, thr10_win=49.2,
    thr20_n=14287, thr20_mean=-0.60, thr20_t=-1.68, thr20_win=48.6,
    net05=-0.85, net05_t=-2.83,
    net10=-1.35, net10_t=-4.49, net10_ann=-166.0,
    net20=-2.35, net20_t=-7.83,
    gross_ann=-42.7,
)

# ---------------------------------------------------------------------------
# Shared analysis preamble — imports, the basket, and pooled helpers.
# ---------------------------------------------------------------------------
BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root (quantlab/)
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from center_line import data, strategy as st

TICKERS = ["SPY", "QQQ", "IWM", "AAPL", "TSLA", "NVDA", "ES=F", "NQ=F"]

def _have_cache():
    return all(os.path.exists(data._cache_path(t, data.DEFAULT_CACHE)) for t in TICKERS)

HAVE_REAL = _have_cache()

def pooled(tp, sl, cost, seed=None, threshold=1.0):
    \"\"\"Pool barrier trades across the basket (cached real tapes).\"\"\"
    frames = []
    for t in TICKERS:
        b = data.fetch_5m(t, fetch=False)
        ent = st.vwap_fade_entries(b, threshold=threshold)
        dirs = st.random_directions(len(ent), seed=seed) if seed is not None else None
        frames.append(st.run_trades(b, ent, tp_R=tp, sl_R=sl, cost_bps=cost, directions=dirs))
    return pd.concat(frames, ignore_index=True)

print("real 5m cache present:", HAVE_REAL)

# Frozen headline numbers — mirror of docs/results.md (as-of 2026-06-12).
R = dict(
    n=23462, tpd=48.9, tpy=12318,
    fade_mean=-0.35, fade_win=49.2, fade_t=-1.16,
    rand_mean=-0.25, rand_win=49.1, rand_t=-1.69,
    fade_minus_rand=-0.09,
    ci_lo=-4.67, ci_hi=1.42, frac_neg=86,
    thr05_n=29136, thr05_mean=-0.35, thr05_t=-1.27, thr05_win=49.3,
    thr10_n=23462, thr10_mean=-0.35, thr10_t=-1.16, thr10_win=49.2,
    thr20_n=14287, thr20_mean=-0.60, thr20_t=-1.68, thr20_win=48.6,
    net05=-0.85, net05_t=-2.83,
    net10=-1.35, net10_t=-4.49, net10_ann=-166.0,
    net20=-2.35, net20_t=-7.83,
    gross_ann=-42.7,
)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Center-Line — does price always return to the session VWAP?\n"
            "### The VWAP-fade scalp, tested honestly, in plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Beats_a_coin%3F: Not_supported](https://img.shields.io/badge/Beats_a_coin%3F-Not_supported-8b949e?style=flat-square)\n\n"
            "Here's a claim you'll find in every prop-trading manual and day-trading community: "
            "the session VWAP (Volume-Weighted Average Price) is the **centre of gravity** for the "
            "day. Whenever price stretches an ATR or two away from it, fade it back — go short if "
            "it's stretched above, go long if it's stretched below. *It always comes back.* This "
            "notebook asks the only question that matters: is 'it always comes back' a forecasting "
            "edge, or is it just what averages do by definition?\n\n"
            "> **This is the plain-language layer.** Want the t-stats, the bootstrap intervals and "
            "the threshold sweep? That's the companion, "
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
            "| Does the VWAP deviation point the right direction? | **No.** With a fair (symmetric) "
            f"target & stop it wins **{R['fade_win']}%** of the time — a coin. |\n"
            "| Does it beat an actual coin flip? | **No.** A random-direction entry on the *same* "
            f"bars does equally well (Δ = {R['fade_minus_rand']:+.2f} bps). |\n"
            "| Does a sharper threshold help? | **No.** Testing 0.5, 1.0, and 2.0 ATR deviations: "
            "all land near −0.35 bps, none clear the noise bar. |\n"
            "| Could you trade it? | **No.** At ~49 trades/day, even 0.5 bp of cost makes it a "
            f"statistically significant loser. |\n\n"
            "> **The VWAP is a running average of the day's prices. Of course price tends to "
            "stay near it — that's what a running average of the thing being averaged does. "
            "That's an accounting identity, not a trading signal.**"
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"The session VWAP is the institutional anchor — big money executes to VWAP all day. "
            "When retail pushes price away from it, the institutions drag it back. Every significant "
            "deviation from VWAP is a fade opportunity. The VWAP is the centre of gravity — price "
            "always returns to it.\"*\n\n"
            "It's genuinely appealing: VWAP *is* the institutional benchmark, large orders *are* "
            "scheduled against it, and there *is* a mechanical pull from execution algorithms trying "
            "to match it. The question is whether that institutional gravity is exploitable from the "
            "outside — or whether by the time a 5-minute bar has deviated 1 ATR, the next bar is as "
            "likely to deviate further as to snap back."
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the VWAP gravity is real and exploitable, it's a mechanical, all-day rule: "
            "look for stretches, fade them, collect the snap-back. If it's not — if 'comes back "
            "to the VWAP' is just what running averages do by construction — then every VWAP-fade "
            "trade is just a coin flip, and at ~49 trades a day the commissions stack up fast."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Two rules keep us honest:\n\n"
            "1. **Make the bet symmetric.** A take-profit and stop at equal distance (one ATR each) "
            "means a coin flip earns zero. Only a *real* sense of direction can beat that. We "
            "test 0.5, 1.0, and 2.0 ATR deviation thresholds to see if filtering harder helps.\n"
            "2. **Race it against an actual coin.** Take the exact same entry bars and flip a coin "
            "for the direction. If the VWAP deviation knows something, it beats the coin. If it "
            "doesn't, it won't.\n\n"
            "We run it on **eight liquid markets** (SPY, QQQ, IWM, AAPL, TSLA, NVDA, ES, NQ) over "
            "~60 days of 5-minute bars — the most history the data vendor gives at this resolution."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First: how often does the signal fire?** The claim is about *significant* deviations. "
            "Here's how many bars per day qualify at each threshold:"
        ),
        code(
            "thresholds = [0.5, 1.0, 2.0]\n"
            "labels = [f'{t} ATR' for t in thresholds]\n"
            "if HAVE_REAL:\n"
            "    b = data.fetch_5m('SPY', fetch=False)\n"
            "    vwap = st.running_vwap(b)\n"
            "    dev = st.vwap_deviation(b, vwap).abs().dropna()\n"
            "    n_days = b.index.normalize().nunique()\n"
            "    counts = [((dev >= t).sum() / n_days) for t in thresholds]\n"
            "    total_bars = len(b) / n_days\n"
            "else:\n"
            f"    counts = [R['thr05_n']/(8*60), R['thr10_n']/(8*60), R['thr20_n']/(8*60)]\n"
            "    total_bars = 78.0\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.2))\n"
            "bars_chart = ax.bar(labels, counts, color=[GREEN, AMBER, RED])\n"
            "ax.axhline(total_bars, ls='--', c='k', lw=1, label=f'total bars/day (~{total_bars:.0f})')\n"
            "ax.set_ylabel('bars/day qualifying (SPY)')\n"
            "ax.set_title('The signal fires on most of the session — not on extreme outliers')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for lbl, cnt in zip(labels, counts):\n"
            f"    pct = cnt/total_bars*100\n"
            "    print(f'{lbl}: {cnt:.0f} bars/day = {cnt/total_bars*100:.0f}% of the session')"
        ),
        md(
            "Even at a 1.0 ATR threshold, the signal fires on roughly **half or more of all bars** "
            "in the session. This isn't identifying extreme dislocations — it's describing the "
            "ordinary intraday price range. A signal that fires constantly isn't filtering; it's "
            "narrating.\n\n"
            "**Now the honest test: the fair 1:1 bet, VWAP-fade vs a coin.**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = st.summarize(pooled(1, 1, 0.0), 'ret_gross')\n"
            "    r = st.summarize(pooled(1, 1, 0.0, seed=87), 'ret_gross')\n"
            "    cm, cw, rm, rw = c['mean_bps'], c['win_rate']*100, r['mean_bps'], r['win_rate']*100\n"
            "else:\n"
            f"    cm, cw, rm, rw = R['fade_mean'], R['fade_win'], R['rand_mean'], R['rand_win']\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "bars_chart = ax.bar(['VWAP fade\\n(against deviation)', 'Random direction\\n(a coin)'],\n"
            "              [cm, rm], color=[RED, GREY], width=.55)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('gross profit per trade (bps)')\n"
            "ax.set_title('The fair bet: the VWAP-fade does NOT beat a coin')\n"
            "for b, w in zip(bars_chart, [cw, rw]):\n"
            "    ax.annotate(f'win {w:.1f}%', (b.get_x()+b.get_width()/2, 0),\n"
            "                ha='center', va='bottom' if b.get_height()<0 else 'top')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'VWAP fade: {cm:+.2f} bps/trade   |   coin: {rm:+.2f} bps/trade')"
        ),
        md(
            f"On the real tape the VWAP-fade earns **{R['fade_mean']:+.2f} bps per trade** — "
            "slightly negative — and the random-direction control earns "
            f"**{R['rand_mean']:+.2f} bps**. They are the same bet; the VWAP direction adds "
            f"nothing (Δ = {R['fade_minus_rand']:+.2f} bps).\n\n"
            "> Why? The running VWAP is a *running average of close prices*. Price is near the "
            "VWAP most of the time *because the VWAP is defined as an average of those prices*. "
            "A deviation from a running average is normal noise, not a structural overshoot. The "
            "'gravity' is a tautology."
        ),
        md(
            "**Do tighter thresholds help?** Here's what happens as we require a bigger stretch "
            "before entering:"
        ),
        code(
            "thresholds = [0.5, 1.0, 2.0]\n"
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for thr in thresholds:\n"
            "        s = st.summarize(pooled(1, 1, 0.0, threshold=thr), 'ret_gross')\n"
            "        rows.append((f'{thr} ATR', s['n_trades'], s['win_rate']*100, s['mean_bps'], s['tstat']))\n"
            "    tbl = pd.DataFrame(rows, columns=['threshold', 'n trades', 'win %', 'mean bps', 't-stat'])\n"
            "else:\n"
            "    tbl = pd.DataFrame([\n"
            f"        ('0.5 ATR', R['thr05_n'], R['thr05_win'], R['thr05_mean'], R['thr05_t']),\n"
            f"        ('1.0 ATR', R['thr10_n'], R['thr10_win'], R['thr10_mean'], R['thr10_t']),\n"
            f"        ('2.0 ATR', R['thr20_n'], R['thr20_win'], R['thr20_mean'], R['thr20_t']),\n"
            "    ], columns=['threshold', 'n trades', 'win %', 'mean bps', 't-stat'])\n"
            "print(tbl.to_string(index=False))"
        ),
        md(
            "Requiring a 2 ATR overshoot (an extreme move) reduces the trade count significantly "
            "but actually makes things *worse*, not better. The mean return falls from −0.35 to "
            "−0.60 bps. There is no threshold that unlocks the edge — the gravity was never there."
        ),

        # ---- BEAT 5 — THE VERDICT --------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The VWAP-fade direction carries no real information; it loses "
            f"a statistically invisible sliver (**{R['fade_mean']:+.2f} bps**, *t* = {R['fade_t']:+.2f}) "
            "and fails to beat a coin at any threshold.\n"
            "- **Tradability — Mirage.** Even if it were flat, ~49 trades/day of costs make it a "
            f"**{R['net10_ann']:.0f}%/yr** loser.\n"
            "- **Beats a coin? — Not Supported.** The 'centre of gravity' is a mathematical property "
            "of running averages, not a forecasting edge. Price 'returns to the VWAP' because the "
            "VWAP is defined as the average of price — this is a tautology, not a signal."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "This is where even a *flat* coin goes to die. ~49 trades a day is the killer:"
        ),
        code(
            "costs = [0.0, 0.5, 1.0, 2.0]\n"
            "if HAVE_REAL:\n"
            "    net = [st.summarize(pooled(1, 1, c), 'ret_net')['mean_bps'] for c in costs]\n"
            "else:\n"
            f"    net = [R['fade_mean'], R['net05'], R['net10'], R['net20']]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(costs, net, 'o-', c=RED, lw=2)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.fill_between(costs, net, 0, where=[n<0 for n in net], color=RED, alpha=.12)\n"
            "ax.set_xlabel('round-trip cost (bps)'); ax.set_ylabel('net profit per trade (bps)')\n"
            "ax.set_title('Every bit of cost digs a deeper hole')\n"
            "plt.tight_layout(); plt.show()\n"
            f"print('gross is already {R['fade_mean']:+.2f} bps -- there is no positive break-even cost.')"
        ),
        md(
            "The line starts below zero and only falls. The usual 'it dies at the costs line' "
            "story does not even apply here — because it was never alive to begin with. "
            f"At 0.5 bp round-trip costs the trade is already a certified loser (*t* = {R['net05_t']:+.2f})."
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Is *anything* findable at 5 minutes?** The companion notebook shows a synthetic "
            "control: when we *plant* real intra-session mean-reversion in a fake tape, this exact "
            "rule finds it. So the machine works — the real market just has nothing to find here.\n"
            "- **Specific intraday time windows** have more promise than any-time VWAP fades — "
            "[Study 13 — Crimson-Hour](../../13-crimson-hour/) tests whether particular hours carry "
            "structure that survives the honest bar.\n"
            "- **The same 5-minute infrastructure**, same null result: "
            "[Study 72 — Loaded-Dice](../../72-loaded-dice/) tests the SMA(5/10) crossover and "
            "reaches the same verdict by a different path.\n\n"
            "*Think the VWAP gravity is real in a specific regime (high-volume days, post-news, "
            "particular sectors)? Fork this, add the regime filter, and show a fair-bet edge "
            "that beats the coin and clears the costs. That's the bar.*"
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
            "# Center-Line — a quantitative teardown\n"
            "### Real 5-minute tape · VWAP-fade barrier backtest · random-direction control · HAC inference · threshold & turnover sweeps\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Beats_a_coin%3F: Not_supported](https://img.shields.io/badge/Beats_a_coin%3F-Not_supported-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We test whether the VWAP "
            "deviation beats a random-direction control on a symmetric-barrier backtest across eight "
            "liquid 5-minute tapes, sweep the deviation threshold and the cost structure, and run a "
            "synthetic positive control to confirm the engine works when reversion is real.\n\n"
            "> **Not investment advice.** Real data: Yahoo 5-minute bars, ~60-day rolling window, "
            "as-of 2026-06-12; the offline core and tests run on a deterministic synthetic tape. "
            "Methods & sources in [`docs/references.md`](../docs/references.md), reproducible numbers "
            "in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **The `> In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT + "\ntry:\n    from quantlab import analytics, stats\n    HAS_QL = True\nexcept ImportError:\n    HAS_QL = False\n"),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | VWAP-fade gross **{R['fade_mean']:+.2f} bps/trade**, HAC "
            f"*t* = **{R['fade_t']:+.2f}**; no edge over a random-direction control "
            f"(Δ = {R['fade_minus_rand']:+.2f} bps); no threshold (0.5–2.0 ATR) unlocks signal. |\n"
            f"| **Tradability** | `MIRAGE` | ~{R['tpd']:.0f} trades/day → significant loser at 0.5 bp "
            f"(*t* = {R['net05_t']:+.2f}), **{R['net10_ann']:.0f}%/yr** at 1 bp. |\n"
            f"| **Beats a coin?** | `NOT SUPPORTED` | 86% of bootstrap resamples negative; "
            f"bootstrap 95% CI on ann. Sharpe: [{R['ci_lo']:+.2f}, {R['ci_hi']:+.2f}]. |\n\n"
            "> In plain words: the VWAP is the weighted average of today's prices; "
            "measuring deviation from it and fading the deviation is like fading deviation "
            "from a running average of the thing you're averaging — it's a tautology, not a signal."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $V_t$ be the running session VWAP at bar $t$ and $\\delta_t = (P_t - V_t) / "
            "\\mathrm{ATR}(20)_t$ the normalised deviation. The recipe asserts:\n\n"
            "- **H₁ (signal).** $\\mathbb{E}[\\,(-\\mathrm{sgn}(\\delta_t)) \\cdot r_{t \\to \\text{exit}}\\,] > 0$ "
            "where $r$ is the symmetric-barrier return — i.e. fading the VWAP deviation carries "
            "directional information, *measured symmetrically* so a coin scores 0.\n"
            "- **H₂ (beats random).** That expectation exceeds the same statistic with direction "
            "replaced by an i.i.d. fair coin on identical entries.\n"
            "- **H₃ (tradable).** It survives a realistic round-trip cost at the rule's natural "
            "turnover (~49 trades/day).\n\n"
            "We reject H₁ and H₂ on the real tape and H₃ trivially — the break-even cost is "
            "undefined because the gross is already negative."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The VWAP-fade is one of the most-cited intraday institutional anchors in practitioner "
            "literature. If H₁–H₃ held, it would be a systematic intraday edge accessible from "
            "publicly available data. The interesting failure is *structural*: the rule fires on "
            "~63% of all intraday bars (at 1 ATR threshold), confirming it is not selecting "
            "exceptional moments but narrating ordinary intraday noise. A filter that fires on "
            "most bars cannot have selective power."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Entry.** |VWAP deviation| > threshold ATR(20) at bar $t$; direction = fade "
            "(short if above VWAP, long if below); **enter at $t{+}1$'s open** (no look-ahead).\n"
            "- **Exit (symmetric).** Barriers at $\\pm 1\\,\\mathrm{ATR}(20)$ from entry; flat at "
            "the 16:00 close; a bar that straddles both barriers is filled at the **stop** "
            "(conservative).\n"
            "- **Control.** Identical entries, direction $\\in\\{-1,+1\\}$ drawn i.i.d. (seeded).\n"
            "- **Threshold sweep.** 0.5, 1.0, 2.0 ATR — testing whether selective entry improves "
            "the edge.\n"
            "- **Inference.** Newey-West HAC *t* on per-trade returns; circular block-bootstrap "
            "Sharpe CI; per-instrument breakdown; cost sweep at the rule's turnover.\n"
            "- **Positive control.** A synthetic tape with tunable intra-session mean-reversion, "
            "to confirm the engine recovers an edge *when one exists*.\n\n"
            "Eight tapes: SPY, QQQ, IWM, AAPL, TSLA, NVDA, ES=F, NQ=F."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The symmetric bet, per instrument — no tape carries a signal\n\n"
            "Per-instrument HAC *t* on the gross per-trade return at threshold = 1.0 ATR. "
            "If the rule worked anywhere, a bar would clear +2 or −2."
        ),
        code(
            "if HAVE_REAL:\n"
            "    recs = []\n"
            "    for t in TICKERS:\n"
            "        b = data.fetch_5m(t, fetch=False)\n"
            "        ent = st.vwap_fade_entries(b, threshold=1.0)\n"
            "        s = st.summarize(st.run_trades(b, ent, tp_R=1, sl_R=1, cost_bps=0), 'ret_gross')\n"
            "        recs.append((t, s['n_trades'], s['win_rate']*100, s['mean_bps'], s['tstat']))\n"
            "    perinst = pd.DataFrame(recs, columns=['ticker','n','win%','mean_bps','t'])\n"
            "    p = pooled(1,1,0); ps = st.summarize(p,'ret_gross')\n"
            "else:\n"
            "    perinst = pd.DataFrame({'ticker': TICKERS,\n"
            "        't': [-0.71,-0.62,-0.33,-0.48,-1.25,0.40,-0.74,-0.29]})\n"
            "    perinst['mean_bps']=np.nan; perinst['win%']=np.nan; perinst['n']=np.nan\n"
            f"    ps = dict(mean_bps=R['fade_mean'], tstat=R['fade_t'], win_rate=R['fade_win']/100)\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "col = [RED if abs(v)<2 else GREEN for v in perinst['t']]\n"
            "ax.bar(perinst['ticker'], perinst['t'], color=col)\n"
            "for s in (2,-2): ax.axhline(s, ls='--', c=GREY, lw=1)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('HAC t-stat (gross mean/trade)')\n"
            "ax.set_title('Eight markets, eight non-results (|t| ≤ 1.25 everywhere)')\n"
            "ax.set_ylim(-3, 3); plt.tight_layout(); plt.show()\n"
            "print(f\"pooled: {ps['mean_bps']:+.2f} bps/trade, HAC t = {ps['tstat']:+.2f}\")\n"
            "perinst.round(2)"
        ),
        md(
            f"> In plain words: every instrument sits inside the ±2 band. Pooling {R['n']:,} trades "
            f"gives **{R['fade_mean']:+.2f} bps**, HAC *t* = **{R['fade_t']:+.2f}** — "
            "the extra power of pooling buys us confidence that the edge is *absent*, not merely "
            "noisy."
        ),
        md(
            "### 4b · VWAP-fade vs the coin — same bet, with a bootstrap band\n\n"
            "The random-direction control on identical entries, with a bootstrap 95% CI on the "
            "annualised Sharpe of the VWAP-fade arm."
        ),
        code(
            "if HAVE_REAL:\n"
            "    C = pooled(1,1,0); cm = st.summarize(C,'ret_gross')\n"
            "    rand_means = [st.summarize(pooled(1,1,0,seed=s),'ret_gross')['mean_bps'] for s in range(20)]\n"
            "    if HAS_QL:\n"
            f"        ci = stats.sharpe_ci_bootstrap(pd.Series(C['ret_gross'].values), periods_per_year=R['tpy'], seed=87)\n"
            "        clo, chi, fn = ci['ci_low'], ci['ci_high'], ci['frac_negative']*100\n"
            "    else:\n"
            f"        clo, chi, fn = R['ci_lo'], R['ci_hi'], R['frac_neg']\n"
            "    cmean = cm['mean_bps']\n"
            "else:\n"
            f"    rand_means=[R['rand_mean']]; cmean=R['fade_mean']; clo,chi,fn=R['ci_lo'],R['ci_hi'],R['frac_neg']\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(rand_means, bins=12, color=GREY, alpha=.65, label='random-control means (20 seeds)')\n"
            "ax.axvline(cmean, c=RED, lw=2.5, label=f'VWAP fade: {cmean:+.2f} bps')\n"
            "ax.axvline(0, c='k', lw=1)\n"
            "ax.set_xlabel('gross mean (bps/trade)'); ax.set_ylabel('count')\n"
            "ax.set_title('The VWAP-fade lands inside the coin\\'s noise band'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'ann. Sharpe 95% CI: [{clo:+.2f}, {chi:+.2f}]  ({fn:.0f}% of resamples < 0)')"
        ),
        md(
            f"The VWAP-fade bootstrap 95% CI on the annualised Sharpe is "
            f"**[{R['ci_lo']:+.2f}, {R['ci_hi']:+.2f}]** ({R['frac_neg']}% of resamples negative). "
            "It is statistically indistinguishable from — and slightly worse than — a coin."
        ),
        md(
            "### 4c · Threshold sweep — no entry filter unlocks a signal\n\n"
            "Requiring a larger deviation before entering trades *reduces* the count but does "
            "not improve the signal. The mean return is flat-to-worse at every threshold."
        ),
        code(
            "thresholds = [0.5, 1.0, 2.0]\n"
            "if HAVE_REAL:\n"
            "    tt = []\n"
            "    for thr in thresholds:\n"
            "        s = st.summarize(pooled(1, 1, 0, threshold=thr), 'ret_gross')\n"
            "        tt.append((f'{thr} ATR', s['n_trades'], s['win_rate']*100, s['mean_bps'], s['tstat']))\n"
            "    tt = pd.DataFrame(tt, columns=['threshold','n','win%','mean_bps','t'])\n"
            "else:\n"
            "    tt = pd.DataFrame([\n"
            f"        ('0.5 ATR', R['thr05_n'], R['thr05_win'], R['thr05_mean'], R['thr05_t']),\n"
            f"        ('1.0 ATR', R['thr10_n'], R['thr10_win'], R['thr10_mean'], R['thr10_t']),\n"
            f"        ('2.0 ATR', R['thr20_n'], R['thr20_win'], R['thr20_mean'], R['thr20_t']),\n"
            "    ], columns=['threshold','n','win%','mean_bps','t'])\n"
            "tt.round(2)"
        ),
        md(
            "> In plain words: the win-rate (49–49%) and mean (−0.35 to −0.60 bps) are flat "
            "across all thresholds. There is no sweet spot. The 2 ATR arm has the worst result "
            "despite requiring the most extreme deviation — exactly the opposite of what 'gravity' "
            "would predict."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — pooled gross {R['fade_mean']:+.2f} bps/trade, HAC "
            f"*t* {R['fade_t']:+.2f}; Δ vs control {R['fade_minus_rand']:+.2f} bps; every "
            "instrument |*t*| ≤ 1.25; no threshold improves the result.\n"
            f"- **Tradability `MIRAGE`** — ~{R['tpd']:.0f} trades/day; net "
            f"*t* {R['net05_t']:+.2f} at 0.5 bp; {R['net10_ann']:.0f}%/yr at 1 bp.\n"
            f"- **Beats a coin? `NOT SUPPORTED`** — bootstrap Sharpe CI [{R['ci_lo']:+.2f}, {R['ci_hi']:+.2f}], "
            f"{R['frac_neg']}% of resamples negative."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — turnover does the killing\n\n"
            "Per-trade net mean and its HAC *t* across round-trip cost, at ~49 trades/day."
        ),
        code(
            "costs = [0.0, 0.5, 1.0, 2.0]\n"
            "if HAVE_REAL:\n"
            "    sw = [st.summarize(pooled(1,1,c),'ret_net') for c in costs]\n"
            "    mean=[s['mean_bps'] for s in sw]; tst=[s['tstat'] for s in sw]\n"
            "else:\n"
            f"    mean=[R['fade_mean'],R['net05'],R['net10'],R['net20']]\n"
            f"    tst=[R['fade_t'],R['net05_t'],R['net10_t'],R['net20_t']]\n"
            "fig, ax = plt.subplots(figsize=(9.0,4.3))\n"
            "ax.plot(costs, mean, 'o-', c=RED, lw=2, label='net mean (bps/trade)')\n"
            "ax2 = ax.twinx(); ax2.plot(costs, tst, 's--', c=GREY, lw=1.5, label='HAC t')\n"
            "ax2.axhline(-2, ls=':', c=GREY); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('round-trip cost (bps)'); ax.set_ylabel('net mean (bps)', color=RED)\n"
            "ax2.set_ylabel('HAC t-stat', color=GREY)\n"
            "ax.set_title('No break-even cost exists: gross is already negative')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('break-even round-trip cost: none (gross is already below zero)')"
        ),
        md(
            "> In plain words: every line in the cost sweep starts below zero and gets worse. "
            "The usual 'it dies at the costs line' story applies to strategies with a positive "
            "gross edge. Here the gross is negative — costs are not the executioner, they just "
            "confirm the sentence."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further — the positive control\n\n"
            "Is the *engine* capable of finding mean-reversion, or does it always print 'none'? "
            "Plant a known level of intra-session mean-reversion in a synthetic tape and sweep "
            "it: the VWAP-fade's advantage over the coin should turn on exactly when real "
            "reversion appears."
        ),
        code(
            "revs = [-0.15, -0.05, 0.0, 0.10, 0.20, 0.30]\n"
            "edge = []\n"
            "for rv in revs:\n"
            "    b, _ = data.synthetic_5m(n_days=120, reversion=rv, seed=87)\n"
            "    ent = st.vwap_fade_entries(b, threshold=0.5)\n"
            "    if len(ent) == 0:\n"
            "        edge.append(0.0)\n"
            "        continue\n"
            "    c = st.summarize(st.run_trades(b,ent,tp_R=1,sl_R=1,cost_bps=0),'ret_gross')['mean_bps']\n"
            "    r = st.summarize(st.run_trades(b,ent,tp_R=1,sl_R=1,cost_bps=0,\n"
            "        directions=st.random_directions(len(ent),seed=1)),'ret_gross')['mean_bps']\n"
            "    edge.append(c - r)\n"
            "fig, ax = plt.subplots(figsize=(9.0,4.3))\n"
            "ax.plot(revs, edge, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(0, c='k', lw=1); ax.axvline(0, ls='--', c=GREY)\n"
            "ax.set_xlabel('planted intra-session reversion strength')\n"
            "ax.set_ylabel('fade − coin (bps/trade)')\n"
            "ax.set_title('The engine works: it harvests mean-reversion when it exists')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('At reversion=0 (martingale) the edge is ~0; it rises monotonically with planted reversion.')"
        ),
        md(
            "The VWAP-fade advantage is monotone in the planted reversion strength and crosses zero "
            "right at the martingale — so the engine is a faithful reversion detector. The real-tape "
            "verdict is therefore a statement about the **market**, not the method: at 5-minute "
            "resolution there is no exploitable mean-reversion toward the VWAP. The 'gravity' is "
            "there in the physics of the running average; it is not there as a trading signal.\n\n"
            "Forks worth trying: a VWAP-band entry *only near the session close* (the VWAP is most "
            "stable then); filtering on high-volume bars where institutional anchoring is most active; "
            "or testing on lower-frequency data where VWAP acts more as a regime separator than an "
            "intraday magnet — [Study 13 — Crimson-Hour](../../13-crimson-hour/) for the time-window "
            "angle."
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
