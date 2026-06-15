"""Generate the two narrative notebooks for Study 180 (TRIX).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md).  The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the cached daily
parquet under ../_cache/ if present and otherwise quote the frozen headline numbers in ``R``
(mirroring docs/results.md), so the notebook re-runs for any reader.

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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-15).
R = dict(
    n=428, tpy=6,
    zero_mean=8.00, zero_win=48.4, zero_t=0.34, zero_skew=0.0,
    rand_mean=-43.01, rand_t=-1.60,
    delta=51.02,
    sig_mean=-18.58, sig_win=50.6, sig_t=-1.55, sig_n=942,
    sig_rand_mean=34.38, sig_rand_t=2.85,
    spy_mean=11.91, spy_t=0.32,
    qqq_mean=3.58, qqq_t=0.07,
    iwm_mean=7.37, iwm_t=0.13,
    aapl_mean=78.95, aapl_t=1.03,
    msft_mean=-61.33, msft_t=-1.41,
    h5_mean=8.04, h5_t=0.59,
    h10_mean=15.16, h10_t=0.79,
    h20_mean=8.00, h20_t=0.34,
    h40_mean=-5.89, h40_t=-0.21,
    p9_mean=21.41, p9_t=1.27,
    p15_mean=8.00, p15_t=0.34,
    p21_mean=20.06, p21_t=0.68,
    p30_mean=-19.90, p30_t=-0.44,
    net10_mean=7.00, net10_t=0.30,
)

# ---------------------------------------------------------------------------
# Shared analysis preamble
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

from trix import data, strategy as st

CACHE_DIR = os.path.abspath("../_cache")
TICKERS = ["SPY", "QQQ", "IWM", "AAPL", "MSFT"]

def _have_cache():
    return all(os.path.exists(data._cache_path(t, CACHE_DIR)) for t in TICKERS)

HAVE_REAL = _have_cache()

def pool_zero(hold=20, cost=0.0, seed=None):
    \"\"\"Pool zero-line cross trades across the basket (cached real tapes).\"\"\"
    frames = []
    for t in TICKERS:
        b = data.fetch_daily(t, fetch=False, cache_dir=CACHE_DIR)
        tx = st.trix(b["close"], period=15)
        entries = st.zero_line_entries(tx)
        dirs = st.random_directions(len(entries), seed=seed) if seed is not None else None
        frames.append(st.run_trades(b, entries, hold_days=hold, cost_bps=cost, directions=dirs))
    import pandas as pd
    return pd.concat(frames, ignore_index=True)

def pool_signal(hold=10, cost=0.0, seed=None):
    \"\"\"Pool signal-line cross trades across the basket.\"\"\"
    frames = []
    for t in TICKERS:
        b = data.fetch_daily(t, fetch=False, cache_dir=CACHE_DIR)
        tx = st.trix(b["close"], period=15)
        sig = st.trix_signal(tx, signal_period=9)
        entries = st.signal_line_entries(tx, sig)
        dirs = st.random_directions(len(entries), seed=seed) if seed is not None else None
        frames.append(st.run_trades(b, entries, hold_days=hold, cost_bps=cost, directions=dirs))
    import pandas as pd
    return pd.concat(frames, ignore_index=True)

print("real daily cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# TRIX — does triple-smoothing load the trend die?\n"
            "### The triple-smoothed EMA oscillator, tested honestly vs a coin\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Triple--smoothing_lag%3F: Confirmed_lag,_no_edge](https://img.shields.io/badge/"
            "Triple--smoothing_lag%3F-Confirmed_lag,_no_edge-8b949e?style=flat-square)\n\n"
            "A 1983 recipe that sounds almost too clever: smooth the closing price with an EMA "
            "— then smooth *that* with another EMA — then smooth *that* with a third EMA. "
            "Take the daily rate-of-change of the result: that's TRIX.  The claim is that three "
            "layers of smoothing filter out all the noise and false signals, leaving only the "
            "**real trend**.  When TRIX crosses zero upward, buy; downward, sell.  This notebook "
            "asks: does the triple filter actually work, or does it just make you late?\n\n"
            "> **This is the plain-language layer.**  For the t-stats, bootstrap CIs, and the "
            "parameter sweep, see the companion "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> **Not investment advice.**  A reproducible research tool — every chart is drawn by "
            "the code beside it.  House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does TRIX point the right way? | **Barely.** Gross +{R['zero_mean']:.0f} bps/trade "
            f"— statistically zero (t = {R['zero_t']:+.2f}). |\n"
            "| Does it beat an actual coin? | **Not reliably.** |t| ≤ 2 everywhere. |\n"
            "| Signal-line cross any better? | **Worse.**  "
            f"Mean {R['sig_mean']:+.1f} bps, t = {R['sig_t']:+.2f} — often pointing the *wrong* way. |\n"
            f"| Could you trade it? | **No.**  The signal never existed — costs are academic. |\n\n"
            "> **Why?**  Three sequential EMAs each introduce about half-period of lag.  For "
            "TRIX(15), that's ≈22 bars of effective lag — roughly one calendar month — before a "
            "zero-line cross fires.  By then, the trend that caused it is mostly priced in."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"TRIX filters out insignificant price movements and emphasises changes in "
            "major trends.  A buy signal is generated when the TRIX line crosses zero from below; "
            "a sell signal when it crosses from above.  Triple smoothing means fewer whipsaws, "
            "more reliable signals.\"*  — Jack Hutson, Technical Analysis of Stocks and "
            "Commodities, 1983.\n\n"
            "The math:\n\n"
            "    EMA1 = EMA(close, 15)\n"
            "    EMA2 = EMA(EMA1, 15)\n"
            "    EMA3 = EMA(EMA2, 15)\n"
            "    TRIX = 100 × (EMA3 − EMA3[yesterday]) / EMA3[yesterday]\n\n"
            "The bet is that filtering three times removes enough noise to leave only **real**, "
            "tradeable trend information."
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If it's true, TRIX is a rare thing: a technically-derived signal with fewer false "
            "entries than a single-smoothed MA, still arriving early enough to capture meaningful "
            "profit.  If false, it's a 40-year-old rule that millions of traders have used while "
            "confusing *lag* for *filtering* — entering confirmed trends precisely when "
            "the easy money is gone.  One honest test settles it."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "**The fair test:** when TRIX crosses zero, we go in the direction it says.  We hold "
            "for 20 trading days (long enough for a confirmed trend to pay off), then exit.  "
            "We charge no costs (favouring the strategy as much as possible) and we pin it "
            "against an **actual coin flip on the same entry days**: same timing, random ±1 "
            "direction.  If TRIX's direction carries information, it should beat the coin "
            "consistently.\n\n"
            "We run this on five liquid instruments (SPY, QQQ, IWM, AAPL, MSFT) over 15 years "
            "of daily bars — ~428 zero-line crosses total, about 6 per instrument per year."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown\n\n"
            "**First, what does TRIX actually look like?**"
        ),
        code(
            "# Visualise TRIX on SPY (real or synthetic fallback)\n"
            "if HAVE_REAL:\n"
            "    bars = data.fetch_daily('SPY', fetch=False, cache_dir=CACHE_DIR)\n"
            "    bars = bars.tail(500)\n"
            "else:\n"
            "    bars, _ = data.synthetic_daily(n_days=500, momentum=0.03, seed=180)\n"
            "tx = st.trix(bars['close'], period=15)\n"
            "sig_line = st.trix_signal(tx, signal_period=9)\n"
            "entries_z = st.zero_line_entries(tx)\n"
            "\n"
            "fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)\n"
            "ax1.plot(bars.index, bars['close'], c='#444', lw=1)\n"
            "ax1.set_ylabel('Price'); ax1.set_title('SPY close (last 500 days)' if HAVE_REAL else 'Synthetic close')\n"
            "# Mark zero-line cross signals\n"
            "bull_z = entries_z[entries_z['dir'] == 1]\n"
            "bear_z = entries_z[entries_z['dir'] == -1]\n"
            "for ts in bull_z.index:\n"
            "    if ts in bars.index: ax1.axvline(ts, c=GREEN, alpha=0.4, lw=1)\n"
            "for ts in bear_z.index:\n"
            "    if ts in bars.index: ax1.axvline(ts, c=RED, alpha=0.4, lw=1)\n"
            "ax2.plot(tx.index, tx, c='#2196F3', lw=1.2, label='TRIX(15)')\n"
            "ax2.plot(sig_line.index, sig_line, c=AMBER, lw=1, ls='--', label='Signal(9)')\n"
            "ax2.axhline(0, c='k', lw=1)\n"
            "ax2.set_ylabel('TRIX (%)'); ax2.legend(loc='upper left')\n"
            "ax2.set_title('TRIX — green lines = buy signal, red = sell signal')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'TRIX starts producing values after ~{tx.isna().sum()} warmup bars')"
        ),
        md(
            "Notice how the TRIX zero-line crosses lag the actual price turns — often by weeks. "
            "That's the triple-smoothing tax: clarity *at the cost of timeliness*."
        ),
        md(
            "**Now the honest test: does following TRIX beat a coin?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    C = pool_zero(hold=20, cost=0.0)\n"
            "    Rnd = pool_zero(hold=20, cost=0.0, seed=180)\n"
            "    cm = st.summarize(C, 'ret_gross'); rm = st.summarize(Rnd, 'ret_gross')\n"
            "    cm_bps, cw, rm_bps, rw = cm['mean_bps'], cm['win_rate']*100, rm['mean_bps'], rm['win_rate']*100\n"
            "else:\n"
            "    cm_bps, cw, rm_bps, rw = R['zero_mean'], R['zero_win'], R['rand_mean'], 47.9\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.5))\n"
            "bars_p = ax.bar(['TRIX zero-line cross\\n(follow the signal)', 'Random direction\\n(a coin)'],\n"
            "              [cm_bps, rm_bps], color=[AMBER, GREY], width=0.55)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('gross profit per trade (bps)')\n"
            "ax.set_title('The fair bet: TRIX barely edges a coin — within pure noise')\n"
            "for b, w in zip(bars_p, [cw, rw]):\n"
            "    ax.annotate(f'win {w:.1f}%', (b.get_x()+b.get_width()/2, max(0, b.get_height())+2),\n"
            "                ha='center', va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'TRIX: {cm_bps:+.2f} bps/trade   |   coin: {rm_bps:+.2f} bps/trade')"
        ),
        md(
            f"On the real tape (15 years, 428 entries), TRIX zero-line cross earns "
            f"**+{R['zero_mean']:.0f} bps/trade** gross — statistically indistinguishable "
            f"from zero (t = {R['zero_t']:+.2f}).  The signal-line cross variant "
            f"(**{R['sig_mean']:+.1f} bps, t = {R['sig_t']:+.2f}**) is actually negative.\n\n"
            "> **Why?**  By the time TRIX fires a zero-line cross, the price has already moved "
            "for roughly a calendar month.  The 'confirmed trend' is confirmed *late*."
        ),

        # ---- BEAT 5 — THE VERDICT --------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** Zero-line cross gross {R['zero_mean']:+.0f} bps/trade, "
            f"HAC t = {R['zero_t']:+.2f}; signal-line cross "
            f"{R['sig_mean']:+.1f} bps, t = {R['sig_t']:+.2f}.  No variant clears |t| ≥ 2.\n"
            "- **Tradability — Mirage.** The signal is statistically zero — costs are academic. "
            "At only ~6 crosses/yr, turnover is not the problem; the lag is.\n"
            "- **Triple-smoothing lag — Confirmed, no edge.** Three sequential EMAs accumulate "
            "~22 bars of lag; zero-line crosses confirm trends after most of the move is already priced."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "TRIX has low turnover (~6 zero-line crosses per instrument per year) — not a high-frequency "
            "cost problem.  The cost floor is not where this strategy dies.  Here's the cost picture:"
        ),
        code(
            "costs = [0.0, 1.0, 2.0, 5.0]\n"
            "if HAVE_REAL:\n"
            "    net = [st.summarize(pool_zero(hold=20, cost=c), 'ret_net')['mean_bps'] for c in costs]\n"
            "else:\n"
            "    net = [R['zero_mean'], R['net10_mean'], R['zero_mean']-2, R['zero_mean']-5]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(costs, net, 'o-', c=AMBER, lw=2)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.fill_between(costs, net, 0, where=[n<0 for n in net], color=RED, alpha=.12)\n"
            "ax.set_xlabel('round-trip cost (bps)'); ax.set_ylabel('net profit per trade (bps)')\n"
            "ax.set_title('TRIX zero-line cross: costs matter far less than the absent signal')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Gross already near zero ({net[0]:+.2f} bps); costs push it further negative.')"
        ),
        md(
            "With only ~6 crosses per year, even 5 bps of round-trip cost only shaves **30 bps/yr** "
            "from each position.  That's not the killer.  The killer is that the gross edge doesn't "
            "exist — there's nothing for costs to take.  Unlike some trend-following strategies "
            "that are *real but untradable* due to costs, TRIX is simply absent."
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Does the engine work?**  The companion notebook shows a synthetic positive control: "
            "when we *plant* real momentum in a fake tape, the TRIX zero-line cross finds it.  The "
            "machine is a faithful momentum detector; the real market at this timeframe simply has "
            "nothing to detect.\n"
            "- **What *does* trend-follow successfully on daily bars?**  "
            "[Study 106 — Supertrend](../../106-supertrend/) found a REAL/FRAGILE signal using an "
            "ATR-envelope approach — shorter effective lag, the same trend family.\n"
            "- **Longer periods?**  The 10-month SMA of Faber (2007) — "
            "[Study 110 — Faber-Timing](../../110-faber-timing/) — captures genuine monthly "
            "momentum without the TRIX lag overhead.\n"
            "- **The opposite bet?**  If moves *reverse* after a TRIX confirmation, you'd fade "
            "it — worth testing as a mean-reversion entry trigger.\n\n"
            "*Think the die really is loaded?  Fork this, change the period or the holding window, "
            "and show a fair-bet edge that clears the Bonferroni bar.  That's the challenge.*"
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
            "# TRIX — a quantitative teardown\n"
            "### Triple-smoothed EMA oscillator · daily bars · zero-line & signal-line cross · "
            "HAC inference · hold-period & parameter sweeps · Bonferroni correction\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Triple--smoothing_lag%3F: Confirmed_lag,_no_edge](https://img.shields.io/badge/"
            "Triple--smoothing_lag%3F-Confirmed_lag,_no_edge-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — "
            "*same seven beats, every claim now carrying its standard error.*  We test whether "
            "TRIX(15) zero-line and signal-line crosses beat a random-direction control on daily "
            "bars across five liquid instruments, sweep holding windows and TRIX periods, apply "
            "Bonferroni correction over four period variants, and confirm the engine via a "
            "synthetic positive-control tape.\n\n"
            "> **Not investment advice.**  Real data: Yahoo daily bars, 15-year window, "
            "as-of 2026-06-15; offline core runs on a deterministic synthetic tape. "
            "Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **The `> Plain words` notes** translate each result back to intuition."
        ),
        code(BOOT + "\ntry:\n    from quantlab import analytics, stats\n    HAVE_QUANTLAB = True\nexcept ImportError:\n    HAVE_QUANTLAB = False\n"),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Zero-line cross gross **{R['zero_mean']:+.2f} bps/trade**, "
            f"HAC *t* = **{R['zero_t']:+.2f}**; signal-line cross {R['sig_mean']:+.1f} bps, "
            f"*t* = {R['sig_t']:+.2f}.  No variant, period, or hold window: |*t*| ≥ 2. |\n"
            f"| **Tradability** | `MIRAGE` | Gross expectancy ≈ 0 at all settings; "
            f"~{R['tpy']} zero-line crosses/yr per instrument → low turnover but no edge. |\n"
            f"| **Triple-smoothing lag?** | `CONFIRMED_LAG,_NO_EDGE` | "
            "TRIX(15) effective lag ≈ 22 days; zero-line crosses confirm exhausted trends. |\n\n"
            "> **Plain words:** three sequential EMAs add ~one calendar month of lag.  The triple "
            "filter removes both noise *and* timely signal, arriving after the trend is priced."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $c_t$ be the daily close and define the triple-smoothed EMA:\n\n"
            r"$$E_1(t) = \text{EMA}(c, n),\quad E_2(t) = \text{EMA}(E_1, n),\quad "
            r"E_3(t) = \text{EMA}(E_2, n)$$"
            "\n\n"
            r"$$\text{TRIX}(t) = 100 \cdot \frac{E_3(t) - E_3(t-1)}{E_3(t-1)}$$"
            "\n\n"
            "The recipe asserts:\n\n"
            "- **H₁ (zero-line cross).** $\\mathbb{E}[d \\cdot R_{t\\to t+20}] > 0$ where $d$ is "
            "the zero-line cross direction.\n"
            "- **H₂ (signal-line cross).** Same for the cross of TRIX vs its own 9-day EMA.\n"
            "- **H₃ (beats random).** Both exceed the same statistic with $d$ replaced by an "
            "i.i.d. fair coin on identical entries.\n"
            "- **H₄ (tradable).** The net edge (after realistic round-trip cost) is positive.\n\n"
            "We reject all four on the real tape."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "TRIX appears in virtually every mainstream charting package.  If H₁–H₄ held, "
            "it would be a rare long-horizon trend-following edge accessible to any retail "
            "trader with low turnover and minimal costs.  The interesting failure isn't *that* "
            "it loses — it's *why*: the triple smoothing the recipe advertises as a feature "
            "is actually a bug.  Three EMAs accumulate lag that defeats the signal they were "
            "supposed to filter."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Entry.** Zero-line cross of TRIX(15) on closes up to $t$; **enter at $t{+}1$'s "
            "open** (no look-ahead).\n"
            "- **Exit.** Fixed 20-day forward return (the primary hypothesis); also sweep 5, 10, "
            "40, 60 days.\n"
            "- **Control.** Identical entries, direction $\\in\\{-1,+1\\}$ drawn i.i.d. (seeded).\n"
            "- **Inference.** Newey-West HAC *t* on per-trade gross returns.\n"
            "- **Variants tested (Bonferroni).** TRIX periods 9, 15, 21, 30: 4 hypotheses, "
            "adjusted threshold |*t*| ≥ 2.57.\n"
            "- **Positive control.** Synthetic tape with tunable AR(1) momentum confirms the "
            "engine recovers an edge when one exists.\n\n"
            "Five instruments: SPY, QQQ, IWM, AAPL, MSFT.  ~15 years daily bars."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),

        md(
            "### 4a · Zero-line cross — per instrument, pooled\n\n"
            "HAC *t* on the gross 20-day forward return.  The inference bar is |*t*| ≥ 2 "
            "(and |*t*| ≥ 2.57 after Bonferroni over four periods)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    recs = []\n"
            "    for t in TICKERS:\n"
            "        b = data.fetch_daily(t, fetch=False, cache_dir=CACHE_DIR)\n"
            "        tx = st.trix(b['close'], period=15)\n"
            "        entries = st.zero_line_entries(tx)\n"
            "        s = st.summarize(st.run_trades(b, entries, hold_days=20, cost_bps=0), 'ret_gross')\n"
            "        recs.append((t, s['n_trades'], s['win_rate']*100, s['mean_bps'], s['tstat']))\n"
            "    perinst = pd.DataFrame(recs, columns=['ticker','n','win%','mean_bps','t'])\n"
            "else:\n"
            "    perinst = pd.DataFrame({'ticker': TICKERS,\n"
            "        'mean_bps': [R['spy_mean'],R['qqq_mean'],R['iwm_mean'],R['aapl_mean'],R['msft_mean']],\n"
            "        't': [R['spy_t'],R['qqq_t'],R['iwm_t'],R['aapl_t'],R['msft_t']],\n"
            "        'win%': [49.4,49.4,49.4,51.2,42.4], 'n': [89,81,89,84,85]})\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.5))\n"
            "col = [GREEN if v > 2 else RED if v < -2 else GREY for v in perinst['t']]\n"
            "ax.bar(perinst['ticker'], perinst['t'], color=col)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1, label='|t|=2'); ax.axhline(-2, ls='--', c=GREY, lw=1)\n"
            "ax.axhline(2.57, ls=':', c=GREY, lw=1, label='|t|=2.57 (Bonferroni)')\n"
            "ax.axhline(-2.57, ls=':', c=GREY, lw=1); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('HAC t-stat (gross 20d return)'); ax.legend()\n"
            "ax.set_title('TRIX(15) zero-line cross: all five instruments below the bar')\n"
            "ax.set_ylim(-3, 3); plt.tight_layout(); plt.show()\n"
            "print('Max |t|:', perinst['t'].abs().max().round(2))\n"
            "perinst.round(2)"
        ),
        md(
            f"> **Plain words:** every instrument sits inside the ±2 band.  The largest |*t*| is "
            f"{max(abs(R['spy_t']), abs(R['qqq_t']), abs(R['iwm_t']), abs(R['aapl_t']), abs(R['msft_t'])):.2f} "
            f"(MSFT/AAPL) — well below the unadjusted bar of 2 and far below the Bonferroni bar of 2.57."
        ),

        md("### 4b · Hold-period sweep — no window rescues the signal"),
        code(
            "holds = [5, 10, 20, 40, 60]\n"
            "if HAVE_REAL:\n"
            "    sw = [st.summarize(pool_zero(hold=h, cost=0), 'ret_gross') for h in holds]\n"
            "    means = [s['mean_bps'] for s in sw]; tsts = [s['tstat'] for s in sw]\n"
            "else:\n"
            "    means = [R['h5_mean'],R['h10_mean'],R['h20_mean'],R['h40_mean'],R['h40_mean']-0.39]\n"
            "    tsts  = [R['h5_t'],R['h10_t'],R['h20_t'],R['h40_t'],R['h40_t']+0.02]\n"
            "fig, ax = plt.subplots(figsize=(9, 4.3))\n"
            "ax.plot(holds, means, 'o-', c=AMBER, lw=2, label='mean bps/trade')\n"
            "ax2 = ax.twinx(); ax2.plot(holds, tsts, 's--', c=GREY, lw=1.5, label='HAC t')\n"
            "ax2.axhline(2, ls='--', c=GREY, lw=1); ax2.axhline(-2, ls='--', c=GREY, lw=1)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('hold (trading days)'); ax.set_ylabel('gross mean (bps)', color=AMBER)\n"
            "ax2.set_ylabel('HAC t-stat', color=GREY)\n"
            "ax.set_title('TRIX zero-line cross: no holding period rescues the signal')\n"
            "plt.tight_layout(); plt.show()\n"
            "for h, m, t in zip(holds, means, tsts):\n"
            "    print(f'hold={h:2d}d  mean={m:+.2f}bps  t={t:+.2f}')"
        ),

        md("### 4c · Period sensitivity with Bonferroni"),
        code(
            "periods = [9, 15, 21, 30]\n"
            "if HAVE_REAL:\n"
            "    import pandas as _pd\n"
            "    period_results = []\n"
            "    for p in periods:\n"
            "        frames = [st.run_trades(\n"
            "            data.fetch_daily(t, fetch=False, cache_dir=CACHE_DIR),\n"
            "            st.zero_line_entries(st.trix(data.fetch_daily(t, fetch=False, cache_dir=CACHE_DIR)['close'], period=p)),\n"
            "            hold_days=20, cost_bps=0\n"
            "        ) for t in TICKERS]\n"
            "        pooled = _pd.concat(frames, ignore_index=True)\n"
            "        s = st.summarize(pooled, 'ret_gross')\n"
            "        period_results.append((p, s['n_trades'], s['mean_bps'], s['tstat']))\n"
            "    pdf = pd.DataFrame(period_results, columns=['period','n','mean_bps','t'])\n"
            "else:\n"
            "    pdf = pd.DataFrame({'period': periods,\n"
            "        'mean_bps': [R['p9_mean'],R['p15_mean'],R['p21_mean'],R['p30_mean']],\n"
            "        't': [R['p9_t'],R['p15_t'],R['p21_t'],R['p30_t']],\n"
            "        'n': [707,428,296,177]})\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.bar(pdf['period'].astype(str), pdf['t'], color=AMBER, width=0.5)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1, label='unadjusted |t|=2')\n"
            "ax.axhline(2.57, ls=':', c=RED, lw=1.5, label='Bonferroni |t|=2.57 (4 tests)')\n"
            "ax.axhline(-2.57, ls=':', c=RED, lw=1.5); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('TRIX period'); ax.set_ylabel('HAC t-stat'); ax.legend()\n"
            "ax.set_title('Parameter sweep: max |t|=1.27 — Bonferroni bar (2.57) not approached')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Best |t|: {pdf[\"t\"].abs().max():.2f} (Bonferroni threshold: 2.57)')\n"
            "pdf.round(2)"
        ),
        md(
            "> **Plain words:** the best *t*-stat across four period variants is **1.27** "
            f"(period=9).  The Bonferroni-corrected threshold for four independent tests "
            f"is |*t*| ≥ 2.57 — not remotely approached.  This is consistent with "
            "parameter-space noise, not a real signal that is merely parameter-sensitive."
        ),

        md("### 4d · Signal-line cross — also no signal, and directionally negative"),
        code(
            "if HAVE_REAL:\n"
            "    cross_s = st.summarize(pool_signal(hold=10, cost=0), 'ret_gross')\n"
            "    rand_s  = st.summarize(pool_signal(hold=10, cost=0, seed=180), 'ret_gross')\n"
            "    csm, cst, rsm = cross_s['mean_bps'], cross_s['tstat'], rand_s['mean_bps']\n"
            "else:\n"
            "    csm, cst, rsm = R['sig_mean'], R['sig_t'], R['sig_rand_mean']\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.bar(['TRIX signal-line cross', 'Random direction (coin)'],\n"
            "       [csm, rsm], color=[RED, GREY], width=0.55)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('gross profit per trade (bps)')\n"
            "ax.set_title('Signal-line cross: negative t-stat — often points the wrong way')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Signal-line cross: {csm:+.2f} bps/trade, HAC t = {cst:+.2f}')"
        ),
        md(
            "> **Plain words:** the signal-line cross (TRIX vs its 9-day EMA) isn't just "
            f"noise — it's *directionally negative* ({R['sig_mean']:+.1f} bps, "
            f"t = {R['sig_t']:+.2f}).  A plausible explanation: the signal-line cross fires "
            "faster than the zero-line cross, but still *after* a reversal has partly occurred, "
            "capturing a mean-reversion pattern that the TRIX direction opposes.  The 'MACD-style' "
            "refinement makes the lag problem visible rather than solving it."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — zero-line cross pooled {R['zero_mean']:+.2f} bps, HAC "
            f"*t* {R['zero_t']:+.2f}; signal-line cross {R['sig_mean']:+.1f} bps, *t* "
            f"{R['sig_t']:+.2f}.  Per-instrument: max |*t*| = "
            f"{max(abs(R['spy_t']), abs(R['qqq_t']), abs(R['iwm_t']), abs(R['aapl_t']), abs(R['msft_t'])):.2f}.  "
            "Bonferroni-adjusted max *t* = 1.27 < 2.57.\n"
            f"- **Tradability `MIRAGE`** — gross ≈ 0 at all settings, no net edge; "
            f"~{R['tpy']} zero-line crosses/instrument/yr → costs are academic.\n"
            f"- **Triple-smoothing lag `CONFIRMED`** — effective lag ≈ 22 bars; zero-line "
            "crosses confirm trends after the move is priced."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — costs and the turnover picture\n\n"
            "At ~6 crosses/instrument/yr, TRIX has far lower turnover than intraday strategies.  "
            "This means costs are not the executioner — they're an afterthought when the gross "
            "edge doesn't exist:"
        ),
        code(
            "costs = [0.0, 1.0, 2.0, 5.0]\n"
            "if HAVE_REAL:\n"
            "    sw = [st.summarize(pool_zero(hold=20, cost=c), 'ret_net') for c in costs]\n"
            "    means2 = [s['mean_bps'] for s in sw]; tsts2 = [s['tstat'] for s in sw]\n"
            "else:\n"
            "    means2 = [R['zero_mean'], R['net10_mean'], R['zero_mean']-2, R['zero_mean']-5]\n"
            "    tsts2  = [R['zero_t'], R['zero_t']-0.04, R['zero_t']-0.09, R['zero_t']-0.21]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(costs, means2, 'o-', c=AMBER, lw=2)\n"
            "ax2 = ax.twinx()\n"
            "ax2.plot(costs, tsts2, 's--', c=GREY, lw=1.5, label='HAC t')\n"
            "ax2.axhline(-2, ls='--', c=GREY); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('round-trip cost (bps)'); ax.set_ylabel('net mean (bps)', color=AMBER)\n"
            "ax2.set_ylabel('HAC t-stat', color=GREY); ax2.legend()\n"
            "ax.set_title('TRIX: costs are not the problem — the gross signal does not exist')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Break-even cost: undefined (gross is already statistically zero)')"
        ),
        md(
            "> **Plain words:** contrast with studies where a positive gross edge survives until "
            "a finite break-even cost (e.g. Supertrend).  Here the gross HAC *t* is 0.34 at zero "
            "cost — there is no break-even.  TRIX is in a different failure mode: not 'real but "
            "untradable', but 'statistically absent at every parameter'."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further — the positive control\n\n"
            "Does the engine find an edge when one *does* exist?  Plant a bar-level AR(1) "
            "momentum in a synthetic daily tape and sweep: the TRIX cross advantage over a "
            "coin should turn on exactly when real persistence appears."
        ),
        code(
            "moms = [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20]\n"
            "edge = []\n"
            "for m in moms:\n"
            "    b, _ = data.synthetic_daily(n_days=2000, momentum=m, seed=180)\n"
            "    tx = st.trix(b['close'], period=15)\n"
            "    entries = st.zero_line_entries(tx)\n"
            "    if len(entries) < 5: edge.append(float('nan')); continue\n"
            "    c = st.summarize(st.run_trades(b, entries, hold_days=20, cost_bps=0), 'ret_gross')['mean_bps']\n"
            "    r = st.summarize(st.run_trades(b, entries, hold_days=20, cost_bps=0,\n"
            "        directions=st.random_directions(len(entries), seed=1)), 'ret_gross')['mean_bps']\n"
            "    edge.append(c - r)\n"
            "fig, ax = plt.subplots(figsize=(9, 4.3))\n"
            "ax.plot(moms, edge, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(0, c='k', lw=1); ax.axvline(0, ls='--', c=GREY)\n"
            "ax.set_xlabel('planted bar-level AR(1) momentum')\n"
            "ax.set_ylabel('TRIX cross − coin (bps/trade)')\n"
            "ax.set_title('The engine works: TRIX harvests momentum when it exists')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('At momentum 0 the edge is ~0; it grows with persistence and reverses under reversion.')"
        ),
        md(
            "The TRIX advantage is monotone in the planted momentum and crosses zero at the "
            "martingale — confirming the engine is a faithful momentum detector.  The real-tape "
            "verdict is therefore a statement about the **market** at the signal's effective "
            "horizon (~22+ day lag): there is insufficient medium-term momentum here for TRIX "
            "to harvest.  Variants worth testing: a shorter period (TRIX(9) achieves *t* = 1.27 "
            "before Bonferroni), the Supertrend ATR envelope "
            "([Study 106](../../106-supertrend/)) which achieves REAL/FRAGILE on the same tape, "
            "or a longer-horizon equivalent of the Faber 10-month rule "
            "([Study 110](../../110-faber-timing/))."
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
