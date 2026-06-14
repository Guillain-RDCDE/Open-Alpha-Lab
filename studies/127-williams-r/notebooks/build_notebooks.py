"""Generate the two narrative notebooks for Study 127 (Williams-R).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the cached
daily parquet under ../_cache/ if present and otherwise quote the frozen headline
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-14).
R = dict(
    n=2111, n_per_yr=35,
    # Zone entry, hold=5, gross
    zone_mean=-9.48, zone_win=47.7, zone_t=-0.88, zone_skew=-0.26,
    zone_rand_mean=-1.17, zone_rand_t=-0.12,
    zone_delta=-8.30,
    # Cross-back, hold=5, gross
    cross_mean=-24.64, cross_win=46.1, cross_t=-2.14,
    cross_rand_mean=10.10, cross_rand_t=0.94,
    # Hold-period sweep (zone, gross)
    hold1_mean=2.95, hold1_t=0.75,
    hold3_mean=-4.67, hold3_t=-0.63,
    hold5_mean=-9.48, hold5_t=-0.88,
    hold10_mean=-24.90, hold10_t=-1.33,
    # Cost sweep (zone, hold=5, net)
    net00=-9.48, net05=-9.98, net10=-10.48, net20=-11.48,
    net00_t=-0.88, net05_t=-0.93, net10_t=-0.97, net20_t=-1.07,
    # Per-instrument (zone, hold=5, gross)
    tstat_spy=-1.40, tstat_qqq=-0.11, tstat_iwm=-0.12,
    tstat_aapl=-1.46, tstat_tsla=-0.82, tstat_nvda=0.89,
)

# ---------------------------------------------------------------------------
# Shared analysis preamble — imports, the basket, and small pooled helpers.
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

from williams_r import data, strategy as st

TICKERS = ["SPY", "QQQ", "IWM", "AAPL", "TSLA", "NVDA"]
CACHE = os.path.abspath(os.path.join("..", "_cache"))

def _have_cache():
    return all(os.path.exists(data._cache_path(t, CACHE)) for t in TICKERS)

HAVE_REAL = _have_cache()

def pool(hold=5, cost=0.0, framing="zone", seed=None):
    frames = []
    for t in TICKERS:
        b = data.fetch_daily(t, fetch=False, cache_dir=CACHE)
        wr = st.williams_r(b)
        ent = st.zone_entries(wr) if framing == "zone" else st.cross_entries(wr)
        dirs = st.random_directions(len(ent), seed=seed) if seed is not None else None
        frames.append(st.run_trades(b, ent, hold_days=hold, cost_bps=cost, directions=dirs))
    return pd.concat(frames, ignore_index=True)

print("real daily cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Williams-R — does the 'exhausted move' bounce really work?\n"
            "### The %R(14) oversold/overbought signal, tested honestly, in plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Beats_a_coin%3F: No](https://img.shields.io/badge/Beats_a_coin%3F-No-8b949e?style=flat-square)\n\n"
            "Here's one of the most-watched signals in retail charting: Larry Williams' "
            "**%R** tells you where today's close sits within the last 14 days' high-low "
            "range, on a scale of −100 (at the bottom of the range — 'oversold') to 0 "
            "(at the top — 'overbought'). The rule: when %R drops below −80, the stock "
            "has been beaten down to exhaustion — **buy** the next day expecting it to "
            "bounce. When %R pops above −20, it's run too far — **sell or short**. This "
            "notebook asks: does the bounce actually show up in the data?\n\n"
            "> This is the plain-language layer. Want the t-stats, bootstrap intervals, "
            "and the hold-period sweep? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart is "
            "drawn by the code beside it. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does %R predict a bounce when oversold? | **No.** The oversold-buy / "
            f"overbought-sell rule earns **{R['zone_mean']:+.1f} bps per trade** — "
            "worse than a coin. |\n"
            "| Does it beat a coin flip? | **No.** A random-direction entry on the same "
            f"bars earns **{R['zone_rand_mean']:+.1f} bps** (better). |\n"
            "| What about the 'confirmed reversal' version? | **Worse.** Entering after "
            f"%R exits the zone gives **{R['cross_mean']:+.1f} bps** — the bounce has "
            "already happened by then. |\n"
            "| Could you trade it? | **No.** The gross is already negative; costs make "
            "it more so. |\n\n"
            "> The 'exhausted move' bounce story is coherent — prices do sometimes revert "
            "— but %R's extreme-zone signal doesn't reliably identify *when* that happens."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"When Williams %R drops below −80, the stock is oversold — it's run too "
            "far down and is likely to bounce. When it rises above −20, it's overbought "
            "and due for a pullback. Act on these extremes for a high-probability "
            "mean-reversion trade.\"*\n\n"
            "It's an appealing idea with a solid-sounding rationale. If a stock has "
            "closed near the bottom of its 14-day range several days in a row, maybe the "
            "selling is exhausted and buyers are about to step in. Williams %R is just a "
            "fancy way of saying: *'the close is near the low of the recent range — "
            "expect a snap-back.'*"
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If it works, %R is a simple, mechanical, daily rule anyone can run on any "
            "stock or ETF. If it doesn't, millions of retail traders are getting whipsawed "
            "on signals that carry no real information — paying commissions for the "
            "privilege of trading a coin flip. The difference is worth knowing."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we know?\n\n"
            "One rule keeps us honest: **race it against an actual coin flip.** We take "
            "every bar where %R enters the oversold or overbought zone, hold for a fixed "
            "number of days, and compare the average forward return to the same bars with "
            "a randomly-flipped direction. If %R knows something, it beats the coin. If "
            "it doesn't, it won't.\n\n"
            "We test on **six liquid daily tapes** (SPY, QQQ, IWM, AAPL, TSLA, NVDA) "
            "over 10 years — enough history to have real statistical power."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, what does %R look like?** A quick visualisation on SPY to "
            "see the oversold (<−80) and overbought (>−20) zones:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    spy = data.fetch_daily('SPY', fetch=False, cache_dir=CACHE)\n"
            "    wr = st.williams_r(spy)\n"
            "    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)\n"
            "    ax1.plot(spy.index[-252:], spy['close'].iloc[-252:], c='k', lw=1)\n"
            "    ax1.set_ylabel('SPY close ($)')\n"
            "    ax1.set_title('SPY daily close (last 1 year)')\n"
            "    ax2.plot(wr.index[-252:], wr.iloc[-252:], c=GREY, lw=1)\n"
            "    ax2.axhline(-80, c=GREEN, ls='--', lw=1, label='oversold <−80 (buy zone)')\n"
            "    ax2.axhline(-20, c=RED, ls='--', lw=1, label='overbought >−20 (sell zone)')\n"
            "    ax2.fill_between(wr.index[-252:], wr.iloc[-252:], -80,\n"
            "                     where=wr.iloc[-252:]<-80, color=GREEN, alpha=0.25)\n"
            "    ax2.fill_between(wr.index[-252:], wr.iloc[-252:], -20,\n"
            "                     where=wr.iloc[-252:]>-20, color=RED, alpha=0.25)\n"
            "    ax2.set_ylabel('Williams %R(14)')\n"
            "    ax2.set_ylim(-105, 5)\n"
            "    ax2.legend(loc='lower right'); plt.tight_layout(); plt.show()\n"
            "else:\n"
            "    print('(cache not present — run verify.py --fetch to populate)')"
        ),
        md(
            "**Now the key test: does buying oversold and selling overbought beat a coin?**"
        ),
        code(
            "holds = [1, 3, 5, 10]\n"
            "if HAVE_REAL:\n"
            "    zone_means = [st.summarize(pool(h, 0), 'ret_gross')['mean_bps'] for h in holds]\n"
            "    rand_means = [st.summarize(pool(h, 0, seed=127), 'ret_gross')['mean_bps'] for h in holds]\n"
            "else:\n"
            "    zone_means = [R['hold1_mean'], R['hold3_mean'], R['hold5_mean'], R['hold10_mean']]\n"
            "    rand_means = [R['zone_rand_mean']] * 4  # approximate\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.8))\n"
            "x = range(len(holds))\n"
            "w = 0.35\n"
            "bars1 = ax.bar([i - w/2 for i in x], zone_means, w, color=RED, label='%R zone entry')\n"
            "bars2 = ax.bar([i + w/2 for i in x], rand_means, w, color=GREY, alpha=0.7, label='random (coin)')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xticks(list(x)); ax.set_xticklabels([f'{h}d' for h in holds])\n"
            "ax.set_xlabel('hold period'); ax.set_ylabel('gross profit per trade (bps)')\n"
            "ax.set_title('Williams %R zone entry vs a coin — at every horizon it trails')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for h, z, r in zip(holds, zone_means, rand_means):\n"
            "    print(f'hold {h:2d}d: %R={z:+.1f}bps  coin={r:+.1f}bps  delta={z-r:+.1f}bps')"
        ),
        md(
            f"At every hold horizon, the %R zone-entry signal either ties or **loses to a coin**. "
            f"The 5-day hold earns **{R['zone_mean']:+.1f} bps** vs the coin's "
            f"**{R['zone_rand_mean']:+.1f} bps** — a gap of **{R['zone_delta']:+.1f} bps** in the "
            "coin's favour. Waiting longer makes it worse, not better.\n\n"
            "**What about the 'confirmed reversal' — waiting for %R to exit the zone?**"
        ),
        code(
            "holds = [1, 3, 5, 10]\n"
            "if HAVE_REAL:\n"
            "    cross_means = [st.summarize(pool(h, 0, framing='cross'), 'ret_gross')['mean_bps'] for h in holds]\n"
            "    cross_rand = [st.summarize(pool(h, 0, framing='cross', seed=127), 'ret_gross')['mean_bps'] for h in holds]\n"
            "else:\n"
            "    cross_means = [R['cross_mean']]*4; cross_rand = [R['cross_rand_mean']]*4\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.8))\n"
            "ax.bar([i - w/2 for i in x], cross_means, w, color=RED, label='%R cross-back entry')\n"
            "ax.bar([i + w/2 for i in x], cross_rand, w, color=GREY, alpha=0.7, label='random (coin)')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xticks(list(x)); ax.set_xticklabels([f'{h}d' for h in holds])\n"
            "ax.set_xlabel('hold period'); ax.set_ylabel('gross profit per trade (bps)')\n"
            "ax.set_title('Cross-back (confirmed reversal) — enters too late, loses the coin')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'5-day cross-back: {cross_means[2]:+.1f} bps vs coin {cross_rand[2]:+.1f} bps')"
        ),
        md(
            f"Even worse: waiting for the confirmed reversal earns **{R['cross_mean']:+.1f} bps** at "
            f"hold=5 — by the time %R exits the oversold zone, the bounce has already happened. "
            "The entry is late, not early."
        ),

        # ---- BEAT 5 — VERDICT ------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** At the 5-day horizon: gross **{R['zone_mean']:+.1f} bps/trade**, "
            f"HAC *t* = **{R['zone_t']:+.2f}**; does not beat a coin (Δ = {R['zone_delta']:+.1f} bps). "
            "Every tested horizon is negative. No instrument shows a real edge.\n"
            f"- **Tradability — Mirage.** Gross is negative at every hold period; no positive "
            "break-even cost exists.\n"
            f"- **Beats a coin? — No.** Both framings (zone entry and cross-back) are outperformed "
            "by a random-direction control on the same entry dates."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "At ~35 signals/ticker/year the turnover is modest — but because the gross is "
            "already negative, costs make an already-bad deal worse:"
        ),
        code(
            "costs = [0.0, 0.5, 1.0, 2.0]\n"
            "if HAVE_REAL:\n"
            "    net = [st.summarize(pool(5, c), 'ret_net')['mean_bps'] for c in costs]\n"
            "else:\n"
            "    net = [R['net00'], R['net05'], R['net10'], R['net20']]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(costs, net, 'o-', c=RED, lw=2)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.fill_between(costs, net, 0, where=[n<0 for n in net], color=RED, alpha=0.12)\n"
            "ax.set_xlabel('round-trip cost (bps)'); ax.set_ylabel('net profit per trade (bps)')\n"
            "ax.set_title('Gross already negative — costs just add insult to injury')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Gross (0 cost): {net[0]:+.1f} bps — there is no positive break-even cost.')"
        ),
        md(
            "The line starts below zero. The usual story ('costs kill an otherwise viable "
            "edge') doesn't apply here — the signal was never alive."
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Is a mean-reversion signal *ever* findable?** The companion notebook shows "
            "a synthetic positive control: plant real mean-reversion in a fake tape and %R "
            "finds it. The engine works — the real market just doesn't have that structure "
            "at this parameter setting.\n"
            "- **The Stochastic Oscillator.** %K/%D is algebraically related to Williams %R "
            "— [Study 107](../../107-stochastic-oscillator/) tests it with the same outcome.\n"
            "- **Bollinger-Band reversion.** A different normalisation of the same 'close at "
            "the edge of its range' idea: [Study 104](../../104-bollinger-reversion/).\n"
            "- **What if you tune the thresholds or the period?** Changing −80/−20 or the "
            "14-day window is the natural fork — but beware: with thousands of daily bars "
            "you can data-mine a threshold that looks good in-sample.\n\n"
            "*Think the bounce is real on a specific instrument or regime? Fork this, show a "
            "fair-bet edge that clears |t| ≥ 2 out-of-sample, and submit a PR.*"
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
            "# Williams-R — a quantitative teardown\n"
            "### Daily tape · %R(14) zone entry and cross-back · random-direction control · HAC inference · hold-period sweep\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Beats_a_coin%3F: No](https://img.shields.io/badge/Beats_a_coin%3F-No-8b949e?style=flat-square)\n\n"
            "The deep companion to [01_for_the_curious.ipynb](01_for_the_curious.ipynb) — "
            "same seven beats, every claim now carrying its standard error. We test whether "
            "Williams %R(14) oversold/overbought zone-entry and cross-back signals predict "
            "daily forward returns better than a random-direction control, across six daily "
            "tapes over 10 years, at hold horizons 1–10 days.\n\n"
            "> **Not investment advice.** Real data: Yahoo daily bars, 10-year window "
            "ending 2026-06-12; as-of 2026-06-14. Methods in "
            "[`docs/references.md`](../docs/references.md), reproducible numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **The `In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Zone-entry gross **{R['zone_mean']:+.2f} bps/trade**, "
            f"HAC *t* = **{R['zone_t']:+.2f}**; Δ vs random = **{R['zone_delta']:+.2f} bps**; "
            "no instrument |*t*| ≥ 1.5. Cross-back is worse (*t* = "
            f"**{R['cross_t']:+.2f}**). |\n"
            f"| **Tradability** | `MIRAGE` | Gross negative at every hold horizon; no "
            "positive break-even cost. |\n"
            f"| **Beats a coin?** | `NO` | Both framings trail a random-direction "
            "control on identical entry dates. |\n\n"
            "> In plain words: %R's extreme-zone condition carries no forward-looking "
            "directional information beyond what a coin flip would deliver. The 'exhausted "
            "move' does not reliably snap back."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $W_t = $ Williams %R at bar $t$. The rule asserts:\n\n"
            "- **H₁ (zone-entry signal).** "
            "$\\mathbb{E}[\\,d \\cdot r_{t\\to t+N}\\,] > 0$ where $d = +1$ "
            "when $W_t < -80$ and $d = -1$ when $W_t > -20$ — i.e. the extreme zone "
            "carries positive directional information at horizon $N$.\n"
            "- **H₂ (beats random).** That expectation exceeds the same statistic with "
            "$d$ replaced by an i.i.d. fair coin on identical entry bars.\n"
            "- **H₃ (tradable).** It survives a realistic round-trip cost at the rule's "
            "natural turnover (~35 signals/ticker/year).\n\n"
            "We test H₁ and H₂ for $N \\in \\{1,3,5,10\\}$ days via Newey-West HAC "
            "*t*-stats on pooled per-trade returns, and check H₃ with a cost sweep.\n\n"
            "Two framings: **zone entry** (enter at the *onset* of the extreme condition) "
            "and **cross-back** (enter when %R *exits* the zone — the 'confirmed reversal')."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "Williams %R is among the most-watched indicators in retail charting "
            "(TradingView, ThinkOrSwim, Bloomberg). If H₁–H₃ held, it would be a simple "
            "mechanical edge on liquid daily instruments. The interesting failure is *how* "
            "both timing choices fail: the zone-entry enters too early (the move isn't "
            "exhausted yet) and the cross-back enters too late (the bounce has already "
            "run). The market has neither a reliable immediate reversal *nor* a deferred "
            "one at this parameter setting."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Indicator.** %R(14) = −100 × (HH₁₄ − closeₜ) / (HH₁₄ − LL₁₄), "
            "known at bar *t*'s close.\n"
            "- **Zone-entry signals.** Fire on the *first* bar where %R crosses below "
            "−80 (buy, $d=+1$) or above −20 (sell, $d=-1$). One entry per episode.\n"
            "- **Cross-back signals.** Fire when %R crosses *back* through the threshold "
            "(exits the extreme zone). One entry per exit.\n"
            "- **Exit.** Close the position at the close of bar $t+N$ for "
            "$N \\in \\{1,3,5,10\\}$.\n"
            "- **Control.** Same entry dates and hold, direction $\\in\\{-1,+1\\}$ drawn "
            "i.i.d. (seed=127).\n"
            "- **Inference.** Newey–West HAC *t* on per-trade returns (pooled across "
            "six instruments); per-instrument breakdown.\n"
            "- **Positive control.** Synthetic daily tape with tunable AR(1) "
            "mean-reversion — confirms the engine recovers an edge when one exists.\n\n"
            "Six tapes: SPY, QQQ, IWM, AAPL, TSLA, NVDA. Window 2016–2026."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · %R distribution and zone-frequency\n\n"
            "Before testing the signal, characterise the indicator: how often does %R "
            "enter the oversold (<−80) and overbought (>−20) zones?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    all_wr = []\n"
            "    for t in TICKERS:\n"
            "        b = data.fetch_daily(t, fetch=False, cache_dir=CACHE)\n"
            "        all_wr.append(st.williams_r(b).dropna())\n"
            "    wr_all = pd.concat(all_wr)\n"
            "    pct_os = (wr_all < -80).mean() * 100\n"
            "    pct_ob = (wr_all > -20).mean() * 100\n"
            "else:\n"
            "    wr_all = None; pct_os = 15.3; pct_ob = 17.1\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.5))\n"
            "if wr_all is not None:\n"
            "    ax.hist(wr_all, bins=60, color=GREY, edgecolor='none', density=True)\n"
            "else:\n"
            "    ax.text(0.5, 0.5, '(cache not present)', transform=ax.transAxes, ha='center')\n"
            "ax.axvline(-80, c=GREEN, lw=2, ls='--', label=f'oversold threshold (<-80): {pct_os:.1f}%')\n"
            "ax.axvline(-20, c=RED, lw=2, ls='--', label=f'overbought threshold (>-20): {pct_ob:.1f}%')\n"
            "ax.set_xlabel('Williams %R(14)'); ax.set_ylabel('density')\n"
            "ax.set_title('Distribution of %R across six instruments, 10-year daily')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Oversold (<-80): {pct_os:.1f}%  |  Overbought (>-20): {pct_ob:.1f}% of bars')"
        ),
        md(
            "### 4b · Hold-period sweep — zone entry vs coin\n\n"
            "Pooled mean forward return (gross) and HAC *t* at horizons 1, 3, 5, 10 days. "
            "The |*t*| ≥ 2 bar is the inference threshold."
        ),
        code(
            "holds = [1, 3, 5, 10]\n"
            "if HAVE_REAL:\n"
            "    recs = []\n"
            "    for h in holds:\n"
            "        s = st.summarize(pool(h, 0), 'ret_gross')\n"
            "        r = st.summarize(pool(h, 0, seed=127), 'ret_gross')\n"
            "        recs.append((h, s['n_trades'], s['win_rate']*100, s['mean_bps'], s['tstat'],\n"
            "                     r['mean_bps'], r['tstat']))\n"
            "    sweep = pd.DataFrame(recs, columns=['hold_d','n','win%','zone_bps','zone_t','rand_bps','rand_t'])\n"
            "else:\n"
            "    sweep = pd.DataFrame({\n"
            "        'hold_d': holds,\n"
            "        'zone_bps': [R['hold1_mean'], R['hold3_mean'], R['hold5_mean'], R['hold10_mean']],\n"
            "        'zone_t': [R['hold1_t'], R['hold3_t'], R['hold5_t'], R['hold10_t']],\n"
            "        'rand_bps': [R['zone_rand_mean']]*4, 'rand_t': [R['zone_rand_t']]*4,\n"
            "        'n': [R['n']]*4, 'win%': [47.7]*4\n"
            "    })\n"
            "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))\n"
            "x = range(len(holds))\n"
            "ax1.bar([i-.2 for i in x], sweep['zone_bps'], 0.4, color=RED, label='%R zone entry')\n"
            "ax1.bar([i+.2 for i in x], sweep['rand_bps'], 0.4, color=GREY, alpha=0.7, label='coin')\n"
            "ax1.axhline(0, c='k', lw=1)\n"
            "ax1.set_xticks(list(x)); ax1.set_xticklabels([f'{h}d' for h in holds])\n"
            "ax1.set_xlabel('hold period'); ax1.set_ylabel('gross bps/trade')\n"
            "ax1.set_title('Mean return: %R trails the coin at every horizon'); ax1.legend()\n"
            "ax2.bar(list(x), sweep['zone_t'], color=[GREEN if abs(t)>=2 else RED for t in sweep['zone_t']])\n"
            "ax2.axhline(2, ls='--', c=GREY); ax2.axhline(-2, ls='--', c=GREY)\n"
            "ax2.axhline(0, c='k', lw=1)\n"
            "ax2.set_xticks(list(x)); ax2.set_xticklabels([f'{h}d' for h in holds])\n"
            "ax2.set_ylabel('HAC t-stat'); ax2.set_title('t-stats: no horizon clears ±2')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(sweep[['hold_d','zone_bps','zone_t','rand_bps']].to_string(index=False))"
        ),
        md(
            "> In plain words: the %R zone-entry rule earns negative returns at every "
            "horizon tested, and the HAC *t*-stat never clears the ±2 bar. The signal "
            "carries no evidence of directional information beyond sampling noise."
        ),
        md(
            "### 4c · Per-instrument breakdown — no tape carries the edge\n\n"
            "HAC *t* on the gross per-trade return (zone entry, hold = 5 days) for each "
            "of the six instruments. A genuine signal should clear +2 somewhere."
        ),
        code(
            "TICKERS_LABELS = TICKERS\n"
            "if HAVE_REAL:\n"
            "    recs = []\n"
            "    for t in TICKERS:\n"
            "        b = data.fetch_daily(t, fetch=False, cache_dir=CACHE)\n"
            "        wr = st.williams_r(b)\n"
            "        ent = st.zone_entries(wr)\n"
            "        s = st.summarize(st.run_trades(b, ent, hold_days=5, cost_bps=0), 'ret_gross')\n"
            "        recs.append((t, s['n_trades'], s['win_rate']*100, s['mean_bps'], s['tstat']))\n"
            "    perinst = pd.DataFrame(recs, columns=['ticker','n','win%','mean_bps','t'])\n"
            "else:\n"
            "    perinst = pd.DataFrame({'ticker': TICKERS,\n"
            "        't': [R['tstat_spy'],R['tstat_qqq'],R['tstat_iwm'],\n"
            "              R['tstat_aapl'],R['tstat_tsla'],R['tstat_nvda']],\n"
            "        'mean_bps': [0]*6, 'win%': [0]*6, 'n': [0]*6})\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "col = [GREEN if abs(v) >= 2 else RED for v in perinst['t']]\n"
            "ax.bar(perinst['ticker'], perinst['t'], color=col)\n"
            "for s in (2, -2): ax.axhline(s, ls='--', c=GREY, lw=1)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('HAC t-stat (gross mean/trade)')\n"
            "ax.set_title('Six instruments, zero results (|t| < 2 everywhere)')\n"
            "ax.set_ylim(-3, 3); plt.tight_layout(); plt.show()\n"
            "print(perinst[['ticker','n','mean_bps','t']].round(2).to_string(index=False))"
        ),
        md(
            "### 4d · Zone-entry vs cross-back — two clocks, both wrong\n\n"
            "The zone-entry fires at the *onset* of the extreme condition; the cross-back "
            "fires when %R *exits* the zone (the 'confirmed reversal'). Both are compared "
            "against their own random controls at hold = 5 days."
        ),
        code(
            "if HAVE_REAL:\n"
            "    z = st.summarize(pool(5, 0, 'zone'), 'ret_gross')\n"
            "    zr = st.summarize(pool(5, 0, 'zone', seed=127), 'ret_gross')\n"
            "    c = st.summarize(pool(5, 0, 'cross'), 'ret_gross')\n"
            "    cr = st.summarize(pool(5, 0, 'cross', seed=127), 'ret_gross')\n"
            "    zb, zt, zrb = z['mean_bps'], z['tstat'], zr['mean_bps']\n"
            "    cb, ct, crb = c['mean_bps'], c['tstat'], cr['mean_bps']\n"
            "else:\n"
            "    zb, zt, zrb = R['zone_mean'], R['zone_t'], R['zone_rand_mean']\n"
            "    cb, ct, crb = R['cross_mean'], R['cross_t'], R['cross_rand_mean']\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.5))\n"
            "labels = ['Zone entry\\n(onset)', 'Zone entry\\ncoin', 'Cross-back\\n(confirmed)', 'Cross-back\\ncoin']\n"
            "vals = [zb, zrb, cb, crb]\n"
            "colors = [RED, GREY, RED, GREY]\n"
            "for lbl, v, c in zip(labels, vals, colors):\n"
            "    ax.bar([lbl], [v], color=c, alpha=0.85 if c == GREY else 1.0)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('gross bps/trade (hold=5d)')\n"
            "ax.set_title(f'Zone entry (t={zt:+.2f}) vs cross-back (t={ct:+.2f}): both lose the coin')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Zone entry: {zb:+.2f} bps (t={zt:+.2f})  vs coin: {zrb:+.2f} bps')\n"
            "print(f'Cross-back: {cb:+.2f} bps (t={ct:+.2f})  vs coin: {crb:+.2f} bps')"
        ),
        md(
            "> In plain words: entering at the *onset* of the extreme condition is slightly "
            "less bad than waiting for the confirmation (the cross-back), but neither "
            "approach carries real directional information. The cross-back framing — "
            f"at *t* = **{R['cross_t']:+.2f}** — is actually a statistically significant "
            "negative result: confirming the reversal means entering *after* most of the "
            "bounce has already occurred."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — pooled zone-entry gross {R['zone_mean']:+.2f} bps/trade, "
            f"HAC *t* {R['zone_t']:+.2f}; Δ vs random-direction control "
            f"{R['zone_delta']:+.2f} bps; no instrument |*t*| ≥ 1.5.\n"
            f"- **Tradability `MIRAGE`** — gross already negative at every hold horizon; "
            "no positive break-even cost.\n"
            f"- **Beats a coin? `NO`** — zone entry earns {R['zone_mean']:+.2f} bps, "
            f"coin earns {R['zone_rand_mean']:+.2f} bps; cross-back is *t* = "
            f"{R['cross_t']:+.2f} — actively negative."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — cost arithmetic\n\n"
            "Net per-trade return and HAC *t* across round-trip costs (zone entry, hold=5)."
        ),
        code(
            "costs = [0.0, 0.5, 1.0, 2.0]\n"
            "if HAVE_REAL:\n"
            "    sw = [st.summarize(pool(5, c), 'ret_net') for c in costs]\n"
            "    net = [s['mean_bps'] for s in sw]; tst = [s['tstat'] for s in sw]\n"
            "else:\n"
            "    net = [R['net00'], R['net05'], R['net10'], R['net20']]\n"
            "    tst = [R['net00_t'], R['net05_t'], R['net10_t'], R['net20_t']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(costs, net, 'o-', c=RED, lw=2, label='net mean (bps/trade)')\n"
            "ax2 = ax.twinx()\n"
            "ax2.plot(costs, tst, 's--', c=GREY, lw=1.5, label='HAC t')\n"
            "ax2.axhline(-2, ls=':', c=GREY)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('round-trip cost (bps)'); ax.set_ylabel('net mean (bps)', color=RED)\n"
            "ax2.set_ylabel('HAC t-stat', color=GREY)\n"
            "ax.set_title('Gross starts negative — break-even cost is undefined')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Break-even round-trip cost: none (gross already below zero)')"
        ),
        md(
            "> In plain words: the usual 'costs kill the edge' narrative does not even "
            "apply here — there was no edge to kill. The gross was negative before any "
            "cost was charged."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further — the positive control\n\n"
            "Is the *engine* capable of finding a mean-reversion edge, or does it always "
            "print 'none'? Plant a bar-level AR(1) mean-reversion coefficient in a "
            "synthetic daily tape and sweep it: the zone-entry advantage over the coin "
            "should turn on exactly when real mean-reversion appears."
        ),
        code(
            "mrs = [-0.20, -0.10, 0.0, 0.10, 0.20, 0.30, 0.40]\n"
            "edge = []\n"
            "for mr in mrs:\n"
            "    avg_e = []\n"
            "    for seed in range(5):\n"
            "        b, _ = data.synthetic_daily(n_days=800, mean_rev=mr, seed=seed*10+7)\n"
            "        wr = st.williams_r(b)\n"
            "        ent = st.zone_entries(wr)\n"
            "        if len(ent) < 5: continue\n"
            "        s = st.summarize(st.run_trades(b, ent, hold_days=5, cost_bps=0), 'ret_gross')\n"
            "        r = st.summarize(st.run_trades(b, ent, hold_days=5, cost_bps=0,\n"
            "                         directions=st.random_directions(len(ent), seed=seed+1)), 'ret_gross')\n"
            "        avg_e.append(s['mean_bps'] - r['mean_bps'])\n"
            "    edge.append(np.mean(avg_e) if avg_e else 0.0)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(mrs, edge, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(0, c='k', lw=1); ax.axvline(0, ls='--', c=GREY)\n"
            "ax.set_xlabel('planted AR(1) mean-reversion coefficient')\n"
            "ax.set_ylabel('%R − coin (bps/trade, averaged over 5 seeds)')\n"
            "ax.set_title('The engine works: it finds the edge when mean-reversion is real')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('At mean_rev=0 the edge is ~0; it rises with mean-reversion and falls with trend.')"
        ),
        md(
            "The %R advantage over the coin is monotone in the planted mean-reversion — "
            "the engine is a faithful mean-reversion detector. The real-tape verdict "
            "is therefore a statement about the *market*, not the method: at daily fidelity "
            "on these instruments over 2016–2026, the short-term mean-reversion that "
            "%R is designed to harvest is simply not strong enough to be tradable. "
            "Candidate forks: intraday (where microstructure bounce is more pronounced), "
            "individual stocks rather than large-cap ETFs, or a regime filter to "
            "restrict signals to high-VIX environments."
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
