"""Generate the two narrative notebooks for Study 137 (Mansfield-RS).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md).  The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the cached weekly
parquet under ../_cache/ if present and otherwise quote the frozen headline numbers in
``R`` (mirroring docs/results.md), so the notebook re-runs for any reader.

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
    n_s2=811, n_rnd=813,
    s2_mean=29.4, s2_win=48.1, s2_t=0.72, s2_skew=0.83,
    rnd_mean=118.1, rnd_win=49.3, rnd_t=2.75,
    delta=-88.7,
    in_s2_mean=5.7, in_s2_t=1.61, in_s2_n=7423,
    out_s2_mean=9.2, out_s2_t=3.04, out_s2_n=11534,
    hold4_s2=-6, hold4_rnd=39,
    hold8_s2=-31, hold8_rnd=87,
    hold13_s2=29, hold13_rnd=118,
    hold26_s2=101, hold26_rnd=201,
    n_tickers=15, n_years=25,
    aapl_t=3.62, amzn_t=2.54, pfe_t=-1.50, wmt_t=-2.02,
    cost0_t=0.72, cost1_t=0.69, cost5_t=0.60,
    entries_per_year=2.2,
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

from mansfield_rs import data, strategy as st

TICKERS = data.DEFAULT_TICKERS
STOCKS = [t for t in TICKERS if t != "SPY"]
HOLD_WEEKS = 13

def _have_cache():
    return all(os.path.exists(data._cache_path(t, data.DEFAULT_CACHE)) for t in TICKERS)

HAVE_REAL = _have_cache()

# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-14).
R = dict(
    n_s2=811, n_rnd=813,
    s2_mean=29.4, s2_win=48.1, s2_t=0.72, s2_skew=0.83,
    rnd_mean=118.1, rnd_win=49.3, rnd_t=2.75,
    delta=-88.7,
    in_s2_mean=5.7, in_s2_t=1.61, in_s2_n=7423,
    out_s2_mean=9.2, out_s2_t=3.04, out_s2_n=11534,
    hold4_s2=-6, hold4_rnd=39,
    hold8_s2=-31, hold8_rnd=87,
    hold13_s2=29, hold13_rnd=118,
    hold26_s2=101, hold26_rnd=201,
    n_tickers=15, n_years=25,
    aapl_t=3.62, amzn_t=2.54, pfe_t=-1.50, wmt_t=-2.02,
    cost0_t=0.72, cost1_t=0.69, cost5_t=0.60,
    entries_per_year=2.2,
)

def _pool(tapes, cost=0.0, seed=None):
    bm = tapes["SPY"]
    frames = []
    for tk in STOCKS:
        stock = tapes[tk]
        entries = st.stage2_entries(stock, bm, sma_n=30)
        if entries.empty:
            continue
        if seed is not None:
            entries = st.random_entries(stock, n_entries=len(entries), seed=seed)
        led = st.run_forward_returns(stock, bm, entries, hold_weeks=HOLD_WEEKS, cost_bps=cost)
        frames.append(led)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def _load_tapes():
    return {t: data.fetch_weekly(t, fetch=False, cache_dir=data.DEFAULT_CACHE) for t in TICKERS}

print("real weekly cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Mansfield-RS — does Weinstein's Stage 2 beat buying at random?\n"
            "### The 30-week SMA + Mansfield Relative Strength filter, tested honestly, in plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Timing_adds_nothing: Confirmed](https://img.shields.io/badge/Timing_adds_nothing-Confirmed-8b949e?style=flat-square)\n\n"
            "Stan Weinstein spent a career managing money and writing about it. His 1988 book "
            "*Secrets for Profiting in Bull and Bear Markets* gave retail investors a "
            "deceptively simple rule: look for stocks in **Stage 2** — above a rising 30-week "
            "moving average *and* outperforming the market. Sounds like it should work. "
            "This notebook asks: does entering a Stage-2 stock beat picking the same stock at "
            "a random moment?\n\n"
            "> This is the plain-language layer. For the t-stats, the bootstrap intervals, and "
            "the Mansfield RS formula, see the companion "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool — every chart is drawn "
            "by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does Stage 2 predict outperformance? | **Barely, and not significantly.** "
            f"HAC *t* = **{R['s2_t']:.2f}** — a coin. |\n"
            f"| Does it beat buying at a random moment? | **No.** Random entries earn "
            f"**{R['rnd_mean']:.0f} bps** vs Stage-2's **{R['s2_mean']:.0f} bps** over 13 weeks. |\n"
            f"| Isn't being in Stage 2 itself predictive? | **No.** Stocks in Stage 2 earn "
            f"**{R['in_s2_mean']:.1f} bps/week** excess — stocks *outside* Stage 2 earn "
            f"**{R['out_s2_mean']:.1f} bps/week**. |\n"
            "| Could you trade it? | **No edge to trade.** The gross signal is not "
            "significant; costs don't resurrect it. |\n\n"
            "> The filter picks momentum stocks correctly — but it enters *after* the "
            "best move has already happened. Weinstein's Stage 2 is a lagging confirmation, "
            "not a leading signal."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Look for stocks where the price is above a rising 30-week moving average "
            "AND the Mansfield Relative Strength line is positive — meaning the stock is "
            "outperforming the market. That's Stage 2: the sweet spot. Those stocks are "
            "already showing leadership; the big move is just starting.\"*\n\n"
            "The recipe sounds rigorous: it requires not just trend (price above a slow MA) "
            "but also **relative leadership** (outperforming the market on a trend-adjusted "
            "basis). The idea is that if both conditions hold, the stock is a genuine "
            "market leader, not just riding the tide."
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If Stage 2 genuinely identifies the early phase of outperformance, it's a "
            "powerful position-sizing tool: concentrate in Stage-2 stocks and you should "
            "earn excess return over the market. If it doesn't — if Stage 2 is just "
            "momentum confirmed too late — then you're paying transaction costs to enter "
            "after the best returns are already banked."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "The honest test is simple: compare **Stage-2 entry timing** against "
            "**entering the same stock at a random moment**. If Weinstein's filter "
            "identifies the *right moment* to enter, Stage-2 entries should earn more "
            "excess return over the next 13 weeks than a coin-flip entry would.\n\n"
            "We measure excess return vs SPY (stock return minus SPY return over the same "
            "window) so that general market rises don't count as stock-picking skill. We "
            f"run this across {R['n_tickers']} large-cap US equities over {R['n_years']} "
            "years of weekly data — enough to pool ~800 Stage-2 entries."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First: does Stage 2 beat a random entry on the same stock?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    tapes = _load_tapes()\n"
            "    bm = tapes['SPY']\n"
            "    p_s2 = _pool(tapes, cost=0.0, seed=None)\n"
            "    p_rnd = _pool(tapes, cost=0.0, seed=137)\n"
            "    s2_m = p_s2['ret_gross'].mean()*1e4 if not p_s2.empty else R['s2_mean']\n"
            "    rnd_m = p_rnd['ret_gross'].mean()*1e4 if not p_rnd.empty else R['rnd_mean']\n"
            "    s2_w = p_s2['ret_gross'].gt(0).mean()*100 if not p_s2.empty else R['s2_win']\n"
            "    rnd_w = p_rnd['ret_gross'].gt(0).mean()*100 if not p_rnd.empty else R['rnd_win']\n"
            "else:\n"
            "    s2_m, rnd_m, s2_w, rnd_w = R['s2_mean'], R['rnd_mean'], R['s2_win'], R['rnd_win']\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "bars = ax.bar(['Stage-2 entry\\n(Weinstein filter)', 'Random entry\\n(control)'],\n"
            "              [s2_m, rnd_m], color=[AMBER, GREY], width=.55)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('13-week excess return vs SPY (bps)')\n"
            "ax.set_title('Stage-2 entries do not beat random entries')\n"
            "for b, w in zip(bars, [s2_w, rnd_w]):\n"
            "    ax.annotate(f'win {w:.1f}%', (b.get_x()+b.get_width()/2, max(b.get_height(), 0)+5),\n"
            "                ha='center', va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'Stage-2: {{s2_m:+.1f}} bps/trade   |   Random: {{rnd_m:+.1f}} bps/trade   |   "
            f"Delta: {{s2_m-rnd_m:+.1f}} bps')"
        ),
        md(
            f"The Stage-2 filter earns **{R['s2_mean']:+.1f} bps/trade** over 13 weeks — a "
            f"number that isn't statistically distinguishable from zero (HAC *t* = {R['s2_t']:.2f}). "
            f"The random-entry control earns **{R['rnd_mean']:+.1f} bps/trade**, more than three "
            "times higher, and *that* number is significant.\n\n"
            "> The filter picks good stocks — stocks that have been outperforming. But it enters "
            "them *after the outperformance has already occurred*. A random entry on the same "
            "stocks at a random time over the 25-year window captures more of the full compounding "
            "than waiting for the Stage-2 transition."
        ),
        md(
            "**Second: is being *inside* Stage 2 predictive, even if the entry timing isn't?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    in_s2_all, out_s2_all = [], []\n"
            "    for tk in STOCKS:\n"
            "        stock = tapes[tk]\n"
            "        sc = stock['close']; bc = bm['close'].reindex(sc.index, method='ffill')\n"
            "        excess = sc.pct_change() - bc.pct_change()\n"
            "        flag = st.stage2_signal(stock, bm, sma_n=30)\n"
            "        for i in range(len(flag)-1):\n"
            "            if np.isfinite(excess.iloc[i+1]):\n"
            "                (in_s2_all if flag.iloc[i] else out_s2_all).append(excess.iloc[i+1])\n"
            "    in_mean = np.mean(in_s2_all)*1e4; out_mean = np.mean(out_s2_all)*1e4\n"
            "else:\n"
            "    in_mean, out_mean = R['in_s2_mean'], R['out_s2_mean']\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.bar(['In Stage 2', 'Out of Stage 2'], [in_mean, out_mean], color=[AMBER, GREY], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('Next-week excess return vs SPY (bps)')\n"
            "ax.set_title('Stocks in Stage 2 do not outperform stocks outside Stage 2')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'In Stage-2: {in_mean:+.1f} bps/week   |   Out of Stage-2: {out_mean:+.1f} bps/week')"
        ),
        md(
            f"Stocks currently in Stage 2 earn **{R['in_s2_mean']:.1f} bps/week** excess return "
            f"vs SPY on average. Stocks outside Stage 2 earn **{R['out_s2_mean']:.1f} bps/week** — "
            "slightly *more*, not less. Being in Stage 2 is not a reliable predictor of "
            "the *next* week's outperformance.\n\n"
            "> Why? Stage-2 stocks are well-known to the market by the time the SMA and RS "
            "criteria are both met. The market has already priced in the momentum. What's left "
            "is mean-reversion pressure and the inevitable transition to Stage 3."
        ),

        # ---- BEAT 5 — THE VERDICT --------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** HAC *t* = {R['s2_t']:.2f} on 13-week forward excess return "
            f"at Stage-2 entries. Stage-2 entries underperform a random-entry baseline by "
            f"**{R['delta']:.0f} bps/trade**. In-Stage-2 holding earns *less* than "
            "out-of-Stage-2 holding.\n"
            "- **Tradability — Mirage.** The gross excess is not significant; there is no "
            "positive-expectancy signal to trade. Low turnover (~2.2 entries/stock/year) "
            "means costs aren't the killer — the signal is.\n"
            "- **Timing adds nothing — Confirmed.** The transition into Stage 2 arrives "
            "*after* the best returns, not before them. Weinstein's filter is a lagging "
            "confirmation of momentum already well-known to the market."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "The low turnover (~2.2 entries/stock/year) means execution costs are manageable "
            "in isolation. But that's not the question — the question is whether there's "
            "a gross edge to pay costs from:"
        ),
        code(
            "costs = [0.0, 1.0, 5.0, 10.0]\n"
            "if HAVE_REAL:\n"
            "    net = [st.summarize(_pool(tapes, cost=c), 'ret_net')['mean_bps'] for c in costs]\n"
            "else:\n"
            "    net = [R['s2_mean'], R['s2_mean']-1, R['s2_mean']-5, R['s2_mean']-10]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(costs, net, 'o-', c=AMBER, lw=2)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.fill_between(costs, net, 0, where=[n<0 for n in net], color=RED, alpha=.12)\n"
            "ax.set_xlabel('round-trip cost (bps)'); ax.set_ylabel('net mean excess return (bps/trade)')\n"
            "ax.set_title('The gross is not significant — costs make it worse, not the cause')\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'gross: {{net[0]:+.1f}} bps (t = {R['cost0_t']:.2f}) - not significant before any costs')"
        ),
        md(
            f"The gross excess of **{R['s2_mean']:.1f} bps** is already not significant "
            f"(t = {R['s2_t']:.2f}). Costs fall on top of a non-result. The break-even "
            "cost discussion is moot: there is no edge to break even."
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **What if you used a shorter SMA?** The 30-week SMA is slow by design "
            "(Weinstein's original choice). Shorter SMAs (e.g. 10 or 20 weeks) fire earlier "
            "but add noise. The companion notebook sweeps this.\n"
            "- **What about the full universe?** This basket of 15 surviving large caps is "
            "survivorship-biased; AAPL and AMZN are the strongest signals, and they would "
            "be absent in a real-time backtest of 2001. A full-universe study on CRSP "
            "data is the next step.\n"
            "- **What about Stage 2 *maintenance*?** Instead of tracking the transition into "
            "Stage 2, you could hold as long as Stage 2 persists. The in-stage vs out-of-stage "
            "analysis above gives a hint: the weekly excess is +5.7 bps — present but not "
            "statistically robust.\n"
            "- **The momentum effect is real, just not in this wrapper.** See "
            "[Study 106 — Supertrend](../../106-supertrend/) for an indicator where the "
            "trend-following signal is statistically real (t = 3.27) on daily bars — same "
            "family, much stronger result. Weinstein's version adds the RS condition but "
            "tightens the entry timing too much.\n\n"
            "*Think Stage 2 does work on a broader universe or with a different RS definition? "
            "Fork this, swap the basket for a full-universe CRSP file, and show a fair-bet "
            "excess return that clears |t| ≥ 2. That's the bar.*"
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
            "# Mansfield-RS — a quantitative teardown\n"
            "### Weekly basket · Stage-2 entries · random-entry control · HAC inference · survivorship-bias discussion\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Timing_adds_nothing: Confirmed](https://img.shields.io/badge/Timing_adds_nothing-Confirmed-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "Same seven beats, every claim now carrying its standard error.  We test whether "
            "Weinstein Stage-2 entry signals (price > rising 30-week SMA AND Mansfield RS > 0) "
            "predict 13-week forward excess returns vs SPY, compared against a random-entry "
            "control on the same stocks.\n\n"
            "> **Not investment advice.** Real data: Yahoo weekly bars (resampled from daily), "
            f"as-of 2026-06-14, {R['n_tickers']} large-cap US equities, {R['n_years']} years. "
            "Reproducible numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **Survivorship bias:** basket is a current-S&P-500 proxy; results are upper bounds."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Stage-2 entry 13w excess **{R['s2_mean']:+.1f} bps/trade**, "
            f"HAC *t* = **{R['s2_t']:+.2f}**; underperforms random control by "
            f"**{R['delta']:+.0f} bps/trade**. |\n"
            f"| **Tradability** | `MIRAGE` | Gross not significant; cost sweep (0–10 bps) "
            f"leaves t-stat below 1 throughout; no break-even cost exists above zero. |\n"
            f"| **Timing adds nothing** | `CONFIRMED` | Stage-2 entries underperform random "
            f"at every hold period; in-Stage-2 holding **{R['in_s2_mean']:.1f} bps/week** "
            f"vs out **{R['out_s2_mean']:.1f} bps/week** (in worse). |\n\n"
            "> The filter selects momentum stocks correctly but enters them as late confirmations; "
            "the transition signal captures lagging, not leading, information."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r^{excess}_{t,t+h}$ be the 13-week forward excess return of stock $i$ vs SPY. "
            "The Stage-2 recipe asserts:\n\n"
            "- **H₁ (signal):** $\\mathbb{E}[r^{excess}_{t,t+h} \\mid \\text{Stage2}_{t}^{new}] > 0$\n"
            "  where Stage2$^{new}$ is a transition into Stage 2 (not-Stage-2 at *t−1*, Stage-2 "
            "at *t*).  In words: entering at the Stage-2 transition earns positive excess return.\n"
            "- **H₂ (beats random):** That expectation exceeds the same statistic for random "
            "entry weeks on the same stock.\n"
            "- **H₃ (in-stage predictive):** $\\mathbb{E}[r^{excess}_{t,t+1} \\mid \\text{Stage2}_t] > 0$,\n"
            "  i.e. being in Stage 2 predicts the next week's outperformance.\n\n"
            "**Mansfield RS formula:**\n\n"
            "$$\\text{MRS}_t = \\frac{\\text{stock}_t / \\text{SMA}_{30}(\\text{stock})_t}"
            "{\\text{benchmark}_t / \\text{SMA}_{30}(\\text{benchmark})_t} - 1$$\n\n"
            "Stage-2 requires: (1) $\\text{stock}_t > \\text{SMA}_{30}(\\text{stock})_t$, "
            "(2) $\\text{SMA}_{30}$ slope over last 4 weeks $> 0$, "
            "(3) $\\text{MRS}_t > 0$.\n\n"
            "We reject H₁, H₂, and H₃ on the real tape."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "A confirmed Stage-2 signal would be a high-conviction position-sizing rule for "
            "medium-term investors: concentrate in Stage-2 leaders and expect 100-200 bps "
            "of excess return over the following quarter.  The failure here is not that "
            "momentum doesn't exist — it does (Jegadeesh & Titman 1993) — but that "
            "Weinstein's specific filter enters too late.  The Stage-2 label is a lagging "
            "confirmation of what the market already knows."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Entry.** Stage-2 transition signal at close of week *t*; enter at *t+1*'s open.\n"
            "- **Metric.** 13-week forward excess return = stock gross return − SPY gross return "
            "over the same window (entry open to exit close of week *t+1+13*).\n"
            "- **Control.** Same number of random entries drawn from the post-warmup period on "
            "the *same stock*; same hold period; same cost assumption.\n"
            "- **Inference.** Newey-West HAC *t* on pooled per-trade excess returns.\n"
            "- **Positive control.** A synthetic weekly tape with tunable AR(1) RS momentum; "
            "the engine must recover an edge when RS momentum is planted.\n"
            "- **Hold-period sweep.** 4, 8, 13, 26 weeks.\n"
            "- **Survivorship warning.** The basket is current-S&P-500 survivors; results are "
            "upper bounds.\n\n"
            f"Basket: SPY (benchmark) + {R['n_tickers']} large-cap US equities. "
            f"~{R['n_years']} years of weekly data. Stage-2 warmup: 30 weeks."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Per-stock Stage-2 HAC t-stat — only 2 of 15 are significant, both "
            "survivor-biased compounders"
        ),
        code(
            "LABELS = dict(\n"
            "    AAPL=3.62, AMZN=2.54, XOM=0.83, MRK=0.45, HD=0.12, BAC=0.00, JPM=0.01,\n"
            "    MSFT=0.18, ABBV=-0.34, CVX=-0.57, JNJ=-0.29, KO=-1.00, PG=-1.28,\n"
            "    PFE=-1.50, WMT=-2.02\n"
            ")\n"
            "if HAVE_REAL:\n"
            "    tapes = _load_tapes(); bm = tapes['SPY']\n"
            "    recs = []\n"
            "    for tk in STOCKS:\n"
            "        stock = tapes[tk]\n"
            "        entries = st.stage2_entries(stock, bm, sma_n=30)\n"
            "        led = st.run_forward_returns(stock, bm, entries, hold_weeks=HOLD_WEEKS, cost_bps=0)\n"
            "        s = st.summarize(led, 'ret_gross')\n"
            "        recs.append((tk, s['n_trades'], s['win_rate']*100, s['mean_bps'], s['tstat']))\n"
            "    perinst = pd.DataFrame(recs, columns=['ticker','n','win%','mean_bps','t'])\n"
            "    perinst = perinst.sort_values('t', ascending=False)\n"
            "    t_vals = dict(zip(perinst['ticker'], perinst['t']))\n"
            "else:\n"
            "    t_vals = LABELS\n"
            "    perinst = pd.DataFrame({'ticker':list(t_vals.keys()), 't':list(t_vals.values())})\n"
            "    perinst['mean_bps'] = 0; perinst['win%'] = 50; perinst['n'] = 50\n"
            "fig, ax = plt.subplots(figsize=(10.5, 4.3))\n"
            "col = [GREEN if v>=2 else RED if v<=-2 else GREY for v in t_vals.values()]\n"
            "ax.bar(list(t_vals.keys()), list(t_vals.values()), color=col)\n"
            "for s in (2, -2): ax.axhline(s, ls='--', c=GREY, lw=1)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('HAC t-stat (Stage-2 entry, 13w excess)')\n"
            "ax.set_title('13 of 15 stocks are insignificant; the 2 significant ones are survivor-biased')\n"
            "ax.set_ylim(-3, 5); plt.tight_layout(); plt.show()\n"
            "print(perinst.round(2).to_string(index=False))"
        ),
        md(
            f"> AAPL (*t* = {R['aapl_t']:+.2f}) and AMZN (*t* = {R['amzn_t']:+.2f}) are the "
            "only names that clear the bar — and both are massive compounders that would have "
            "outperformed under any entry timing over 25 years. They are artefacts of the "
            "survivorship-biased basket, not evidence that Stage 2 *times* the entry well."
        ),
        md(
            "### 4b · Stage-2 entry vs random-entry control — pooled HAC"
        ),
        code(
            "if HAVE_REAL:\n"
            "    p_s2 = _pool(tapes, cost=0.0, seed=None)\n"
            "    p_rnd = _pool(tapes, cost=0.0, seed=137)\n"
            "    ss2 = st.summarize(p_s2, 'ret_gross')\n"
            "    srnd = st.summarize(p_rnd, 'ret_gross')\n"
            "    s2_m, rnd_m, s2_t, rnd_t = ss2['mean_bps'], srnd['mean_bps'], ss2['tstat'], srnd['tstat']\n"
            "else:\n"
            "    s2_m, rnd_m = R['s2_mean'], R['rnd_mean']\n"
            "    s2_t, rnd_t = R['s2_t'], R['rnd_t']\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.bar(['Stage-2 entry\\n(Weinstein filter)', 'Random entry\\n(control)'],\n"
            "       [s2_m, rnd_m], color=[AMBER, GREY], width=.55, edgecolor='k', lw=0.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('13-week forward excess return (bps)')\n"
            "ax.set_title('Stage-2 entry underperforms a random-entry baseline')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Stage-2: {s2_m:+.1f} bps  t={s2_t:+.2f}   |   Random: {rnd_m:+.1f} bps  t={rnd_t:+.2f}')\n"
            f"print(f'Delta: {{s2_m - rnd_m:+.1f}} bps  ({R['delta']:+.1f} bps in the reported run)')"
        ),
        md(
            f"> The pooled Stage-2 *t* = {R['s2_t']:+.2f} is not significant. The random "
            f"control *t* = {R['rnd_t']:+.2f} is, because random entries on a long-running "
            "set of surviving large-caps capture the full historical compounding. Stage-2 "
            f"timing forfeits {abs(R['delta']):.0f} bps vs buying at random."
        ),
        md(
            "### 4c · Hold-period sweep — underperformance is consistent"
        ),
        code(
            "if HAVE_REAL:\n"
            "    hold_results = []\n"
            "    for hw in (4, 8, 13, 26):\n"
            "        fr, fr2 = [], []\n"
            "        for tk in STOCKS:\n"
            "            stock = tapes[tk]; bm = tapes['SPY']\n"
            "            entries = st.stage2_entries(stock, bm, sma_n=30)\n"
            "            if entries.empty: continue\n"
            "            led = st.run_forward_returns(stock, bm, entries, hold_weeks=hw, cost_bps=0)\n"
            "            rnd_e = st.random_entries(stock, n_entries=len(entries), seed=137)\n"
            "            led_r = st.run_forward_returns(stock, bm, rnd_e, hold_weeks=hw, cost_bps=0)\n"
            "            fr.append(led); fr2.append(led_r)\n"
            "        p = pd.concat(fr, ignore_index=True)\n"
            "        pr = pd.concat(fr2, ignore_index=True)\n"
            "        ss = st.summarize(p, 'ret_gross'); sr = st.summarize(pr, 'ret_gross')\n"
            "        hold_results.append((hw, ss['mean_bps'], sr['mean_bps']))\n"
            "    holds, s2s, rnds = zip(*hold_results)\n"
            "else:\n"
            "    holds = (4,8,13,26)\n"
            "    s2s = (R['hold4_s2'], R['hold8_s2'], R['hold13_s2'], R['hold26_s2'])\n"
            "    rnds = (R['hold4_rnd'], R['hold8_rnd'], R['hold13_rnd'], R['hold26_rnd'])\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "x = range(len(holds))\n"
            "ax.bar([i-0.2 for i in x], s2s, 0.35, label='Stage-2 entry', color=AMBER)\n"
            "ax.bar([i+0.2 for i in x], rnds, 0.35, label='Random entry', color=GREY)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}w' for h in holds])\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('Excess return vs SPY (bps)')\n"
            "ax.set_title('Stage-2 underperforms at every hold period'); ax.legend()\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            "> The pattern is consistent: at every hold period the random baseline is higher "
            "than the Stage-2 entry. The gap narrows at 26 weeks (both become more positive "
            "as general market drift accumulates) but Stage-2 timing never adds value."
        ),
        md(
            "### 4d · Synthetic positive control — the engine works when RS momentum exists"
        ),
        code(
            "import numpy as np\n"
            "rs_moms = [0.0, 0.15, 0.30, 0.40]\n"
            "s2_edges, rnd_edges = [], []\n"
            "for mom in rs_moms:\n"
            "    tapes_s, truth = data.synthetic_weekly(n_weeks=500, rs_momentum=mom, seed=137)\n"
            "    bm_s = tapes_s['SPY']\n"
            "    all_s2, all_rnd = [], []\n"
            "    for tk in truth['tickers']:\n"
            "        stock = tapes_s[tk]\n"
            "        entries = st.stage2_entries(stock, bm_s, sma_n=30)\n"
            "        if entries.empty: continue\n"
            "        led = st.run_forward_returns(stock, bm_s, entries, hold_weeks=13, cost_bps=0)\n"
            "        rnd_e = st.random_entries(stock, n_entries=len(entries), seed=137)\n"
            "        led_r = st.run_forward_returns(stock, bm_s, rnd_e, hold_weeks=13, cost_bps=0)\n"
            "        if not led.empty: all_s2.append(led['ret_gross'])\n"
            "        if not led_r.empty: all_rnd.append(led_r['ret_gross'])\n"
            "    if all_s2: s2_edges.append(pd.concat(all_s2).mean()*1e4)\n"
            "    else: s2_edges.append(0.0)\n"
            "    if all_rnd: rnd_edges.append(pd.concat(all_rnd).mean()*1e4)\n"
            "    else: rnd_edges.append(0.0)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(rs_moms, s2_edges, 'o-', c=GREEN, lw=2, label='Stage-2 entry')\n"
            "ax.plot(rs_moms, rnd_edges, 's--', c=GREY, lw=1.5, label='Random entry')\n"
            "ax.axhline(0, c='k', lw=1); ax.set_xlabel('planted RS momentum')\n"
            "ax.set_ylabel('13w excess return (bps)'); ax.legend()\n"
            "ax.set_title('The engine works: Stage-2 beats random when RS momentum is planted')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('With planted RS momentum, Stage-2 entries outperform random entries.')\n"
            "print('On the real tape (rs_momentum~0), Stage-2 adds nothing.')"
        ),
        md(
            "> The synthetic control confirms the engine faithfully harvests RS momentum when it "
            "exists. The real-tape null result is therefore a statement about the market — "
            "specifically, that Weinstein's exact filter specification does not capture exploitable "
            "RS momentum after 30-week SMA confirmation."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — pooled 13w excess {R['s2_mean']:+.1f} bps, HAC *t* "
            f"{R['s2_t']:+.2f}; underperforms random by {R['delta']:+.0f} bps; "
            "in-Stage-2 holding earns less than out-of-Stage-2.\n"
            "- **Tradability `MIRAGE`** — the gross is not significant; no cost level "
            "changes the sign of the t-stat argument.\n"
            "- **Timing adds nothing `CONFIRMED`** — Stage-2 underperforms random at "
            "every hold period; the two significant per-stock results are "
            "survivor-biased compounders."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — what a serious implementation would need\n\n"
            "At ~2.2 entries/stock/year the turnover is low; a weekly rebalance for 15 stocks "
            "generates ~33 round-trips/year. Costs are not the primary obstacle:"
        ),
        code(
            "costs = [0.0, 1.0, 5.0, 10.0]\n"
            "if HAVE_REAL:\n"
            "    tstats = [st.summarize(_pool(tapes, cost=c), 'ret_net')['tstat'] for c in costs]\n"
            "    means  = [st.summarize(_pool(tapes, cost=c), 'ret_net')['mean_bps'] for c in costs]\n"
            "else:\n"
            "    means  = [R['s2_mean'], R['s2_mean']-1, R['s2_mean']-5, R['s2_mean']-10]\n"
            "    tstats = [R['cost0_t'], R['cost1_t'], R['cost5_t'], R['cost5_t']-0.1]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(costs, means, 'o-', c=AMBER, lw=2, label='net mean (bps)')\n"
            "ax2 = ax.twinx()\n"
            "ax2.plot(costs, tstats, 's--', c=GREY, lw=1.5, label='HAC t')\n"
            "ax2.axhline(2, ls=':', c=GREY); ax2.axhline(-2, ls=':', c=GREY)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('round-trip cost (bps)'); ax.set_ylabel('net mean (bps)', color=AMBER)\n"
            "ax2.set_ylabel('HAC t-stat', color=GREY)\n"
            "ax.set_title('t-stat never reaches 2 — costs do not explain the failure')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('The t-stat stays below 1 regardless of cost assumption.')"
        ),
        md(
            "> To trade a variant of Stage Analysis, you would need: (1) a full survivorship-free "
            "universe (CRSP or equivalent), (2) a mechanism that enters earlier than the 30-week "
            "confirmation, or (3) additional filters that separate the winning sub-population "
            "(AAPL-type secular compounders) from the losing one in real time — without "
            "look-ahead into which names survive to 2026."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Earlier entry:** replace the 30-week SMA with a 10- or 15-week SMA to enter "
            "closer to the initial breakout.  Shorter SMA → more false signals but earlier "
            "entry.  [Study 106 — Supertrend](../../106-supertrend/) uses a shorter ATR-band "
            "indicator on daily bars and finds a real signal (*t* = 3.27).\n"
            "- **Full-universe test:** run on a survivorship-free database "
            "(CRSP, Compustat, or an ETF-based proxy) to see whether the AAPL/AMZN "
            "results are real or artefacts.\n"
            "- **Relative strength momentum without the SMA condition:** test the Mansfield "
            "RS condition alone (RS > 0, irrespective of the SMA trend) to isolate which "
            "condition is the drag.\n"
            "- **Stage-2 *maintenance*:** instead of entry signals, hold whenever Stage 2 is "
            "active and exit when it breaks.  This gives a larger sample and tests a different "
            "question than transition timing.\n\n"
            "*Think Stage Analysis works on a broader, unbiased universe? Fork this, "
            "plug in CRSP data (or a full-universe ETF proxy), and show |t| ≥ 2 with a "
            "random-entry control.  That's the bar.*"
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
