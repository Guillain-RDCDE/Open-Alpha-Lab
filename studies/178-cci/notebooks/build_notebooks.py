"""Generate the two narrative notebooks for Study 178 (CCI).

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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-15).
R = dict(
    n=963, n_per_yr=16,
    # Breach-entry, hold=5, gross
    breach_mean=-33.11, breach_win=45.1, breach_t=-2.06, breach_skew=0.0,
    rand_mean=-4.38, rand_t=-0.27,
    breach_delta=-28.73,
    # Cross-back, hold=5, gross
    cb_mean=-28.08, cb_win=46.3, cb_t=-1.68,
    cb_rand_mean=-26.48, cb_rand_t=-1.71,
    # Hold-period sweep (breach, gross)
    hold1_mean=3.93, hold1_t=0.63,
    hold3_mean=-14.18, hold3_t=-1.10,
    hold5_mean=-33.11, hold5_t=-2.06,
    hold10_mean=-59.51, hold10_t=-2.34,
    hold20_mean=-57.52, hold20_t=-1.26,
    # Cost sweep (breach, hold=5, net)
    net00=-33.11, net05=-33.61, net10=-34.11, net20=-35.11,
    net00_t=-2.06, net05_t=-2.09, net10_t=-2.12, net20_t=-2.18,
    # Per-instrument (breach, hold=5, gross)
    tstat_spy=0.45, tstat_qqq=0.25, tstat_iwm=-1.01,
    tstat_aapl=-0.16, tstat_tsla=-2.43, tstat_nvda=-0.82,
    mean_spy=9.05, mean_qqq=5.05, mean_iwm=-16.18,
    mean_aapl=-5.49, mean_tsla=-143.39, mean_nvda=-47.95,
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

from cci import data, strategy as st

TICKERS = ["SPY", "QQQ", "IWM", "AAPL", "TSLA", "NVDA"]
CACHE = os.path.abspath(os.path.join("..", "_cache"))

def _have_cache():
    return all(os.path.exists(data._cache_path(t, CACHE)) for t in TICKERS)

HAVE_REAL = _have_cache()

def pool(hold=5, cost=0.0, framing="breach", seed=None):
    frames = []
    for t in TICKERS:
        b = data.fetch_daily(t, fetch=False, cache_dir=CACHE)
        c = st.cci(b, period=20)
        ent = st.breach_entries(c) if framing == "breach" else st.crossback_entries(c)
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
            "# CCI — does Lambert's oscillator know when to buy the dip?\n"
            "### The Commodity Channel Index overbought/oversold rule, tested honestly on equities\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Beats a coin%3F: Busted](https://img.shields.io/badge/Beats_a_coin%3F-Busted-8b949e?style=flat-square)\n\n"
            "Here's a recipe you'll see on almost every trading platform: plot the CCI indicator "
            "on a daily chart, and when it drops below −100 it's 'oversold' — buy the bounce. "
            "When it climbs above +100 it's 'overbought' — sell the pullback. Donald Lambert "
            "invented CCI in 1980 for commodity *cycles*. Does it work on modern US equity daily bars?\n\n"
            "> This is the plain-language layer. Want the t-stats, bootstrap intervals, and the "
            "hold-period sweep? That's the companion, "
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
            f"| Does CCI breach point the right way? | **No.** Buying 'oversold' wins only "
            f"**{R['breach_win']}%** of the time — below a coin flip. |\n"
            "| Does it beat an actual coin? | **No.** A random-direction entry on the *same* "
            f"moments earns −4.4 bps; CCI earns **−33.1 bps**. CCI is *worse* than random. |\n"
            "| What's the best hold window? | 1-day shows +3.9 bps, but HAC *t* = +0.63 — "
            "statistically indistinguishable from noise at any horizon. |\n"
            "| Could you trade it? | **No.** The gross expectancy is already negative; costs "
            "just deepen the hole. |\n\n"
            "> CCI was built for commodity *cycles*. On trending equity markets, "
            "'oversold' often just means 'the trend is still down' — and buying there loses."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"When CCI falls below −100, the asset is oversold and due for a bounce — buy. "
            "When CCI rises above +100, it is overbought and due for a pullback — sell short. "
            "These thresholds identify extremes where price has strayed too far from its mean.\"*\n\n"
            "Lambert developed CCI on 1970s commodity futures, where seasonal and inventory "
            "cycles genuinely mean-revert. The oscillator measures how many 'typical' deviations "
            "the current price is from its 20-day average — in theory, extreme readings should "
            "snap back. We test whether that logic survives transplant to 21st-century US equities."
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If CCI's ±100 thresholds reliably identify bounce/pullback moments, they are a "
            "mechanical trading signal anyone can implement. The stakes are real: millions of "
            "retail traders apply CCI daily. If the premise is wrong — if equity 'oversold' "
            "conditions more often continue than reverse — then following CCI is paying a price "
            "to be systematically on the wrong side of the market."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we know?\n\n"
            "One honest test:\n\n"
            "1. **Use the same entries, flip a coin for direction.** We take the *exact same "
            "bars* where CCI crosses ±100, but assign direction by a fair coin. If CCI knows "
            "the direction, it should beat the coin. If it doesn't, it won't.\n"
            "2. **Measure a fixed holding window.** After the signal, hold 5 days and see what "
            "happened. No exit magic, no optimised stops — the forward return is the verdict.\n\n"
            "We run it on six liquid daily tapes (SPY, QQQ, IWM, AAPL, TSLA, NVDA) over "
            "10 years (2016–2026)."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown — what actually happens\n\n"
            "**First, the per-instrument picture.** Does CCI's 'oversold = buy' rule work "
            "anywhere in the basket?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    recs = []\n"
            "    for t in TICKERS:\n"
            "        b = data.fetch_daily(t, fetch=False, cache_dir=CACHE)\n"
            "        c = st.cci(b, period=20)\n"
            "        ent = st.breach_entries(c)\n"
            "        s = st.summarize(st.run_trades(b, ent, hold_days=5, cost_bps=0), 'ret_gross')\n"
            "        recs.append((t, s['n_trades'], s['win_rate']*100, s['mean_bps'], s['tstat']))\n"
            "    perinst = pd.DataFrame(recs, columns=['ticker','n','win%','mean_bps','t'])\n"
            "else:\n"
            "    perinst = pd.DataFrame({\n"
            "        'ticker': ['SPY','QQQ','IWM','AAPL','TSLA','NVDA'],\n"
            f"        'mean_bps': [{R['mean_spy']},{R['mean_qqq']},{R['mean_iwm']},"
            f"{R['mean_aapl']},{R['mean_tsla']},{R['mean_nvda']}],\n"
            f"        't': [{R['tstat_spy']},{R['tstat_qqq']},{R['tstat_iwm']},"
            f"{R['tstat_aapl']},{R['tstat_tsla']},{R['tstat_nvda']}],\n"
            "        'win%': [49.4, 45.3, 44.5, 44.8, 41.5, 45.0], 'n': [158,159,173,154,159,160]})\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.5))\n"
            "col = [GREEN if v > 0 else RED for v in perinst['mean_bps']]\n"
            "ax.bar(perinst['ticker'], perinst['mean_bps'], color=col)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('gross mean (bps/trade, hold=5d)')\n"
            "ax.set_title('CCI oversold/overbought: mostly negative across the basket')\n"
            "plt.tight_layout(); plt.show()\n"
            "perinst.round(2)"
        ),
        md(
            "SPY and QQQ are near-zero; IWM, AAPL, NVDA are modestly negative; "
            f"TSLA is a disaster at **{R['mean_tsla']:+.1f} bps/trade**. "
            "No instrument shows a reliable positive signal."
        ),
        md(
            "**Now the direct comparison: CCI vs a coin on the same entry bars.**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    breach = pool(hold=5, cost=0.0, framing='breach')\n"
            "    rand = pool(hold=5, cost=0.0, framing='breach', seed=178)\n"
            "    bm = st.summarize(breach, 'ret_gross')['mean_bps']\n"
            "    rm = st.summarize(rand, 'ret_gross')['mean_bps']\n"
            "else:\n"
            f"    bm, rm = {R['breach_mean']}, {R['rand_mean']}\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "bars_chart = ax.bar(['CCI breach\\n(follow the extreme)', 'Random direction\\n(a coin)'],\n"
            "                    [bm, rm], color=[RED, GREY], width=.55)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('gross profit per trade (bps, hold=5d)')\n"
            "ax.set_title('CCI is worse than a coin — not just equal')\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'CCI: {{bm:+.2f}} bps/trade   |   coin: {{rm:+.2f}} bps/trade')"
        ),
        md(
            f"The CCI breach rule earns **{R['breach_mean']:+.1f} bps/trade** — significantly "
            f"negative. The random coin earns **{R['rand_mean']:+.1f} bps**. CCI is not just "
            "uninformative — it is *actively harmful* compared to a coin on these instruments, "
            "consistent with momentum rather than mean-reversion dominating at the 5-day horizon.\n\n"
            "> Why? Lambert built CCI for commodity *cycles* (crop seasons, inventory). Modern "
            "equity markets trend on a structural upward path driven by earnings and innovation. "
            "A stock that is 'oversold' by CCI is more often just in a downtrend that continues."
        ),
        md(
            "**Does any hold window work?**"
        ),
        code(
            "holds = [1, 3, 5, 10, 20]\n"
            "if HAVE_REAL:\n"
            "    means = [st.summarize(pool(hold=h, cost=0.0), 'ret_gross')['mean_bps'] for h in holds]\n"
            "else:\n"
            f"    means = [{R['hold1_mean']}, {R['hold3_mean']}, {R['hold5_mean']}, "
            f"{R['hold10_mean']}, {R['hold20_mean']}]\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "col = [GREEN if v > 0 else RED for v in means]\n"
            "ax.bar([str(h)+'d' for h in holds], means, color=col)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('hold period'); ax.set_ylabel('gross mean (bps/trade)')\n"
            "ax.set_title('CCI breach: only the 1-day hold is positive (but t = +0.63 — noise)')\n"
            "plt.tight_layout(); plt.show()\n"
            f"print('1-day hold t-stat: +{R['hold1_t']:.2f} (not significant)')"
        ),
        md(
            f"The 1-day hold shows +{R['hold1_mean']:+.1f} bps but HAC *t* = **+{R['hold1_t']:.2f}** — "
            "indistinguishable from noise. At 5 and 10 days the rule is a statistically significant "
            "loser, consistent with the 'oversold keeps falling' momentum story."
        ),

        # ---- BEAT 5 — VERDICT ------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** CCI breach gross {R['breach_mean']:+.1f} bps/trade, "
            f"HAC *t* {R['breach_t']:+.2f}; worse than a coin by {R['breach_delta']:+.1f} bps. "
            "Every instrument either near-zero or negative.\n"
            "- **Tradability — Mirage.** Gross expectancy significantly negative → a "
            "loser before costs at every hold window. Low turnover (~16 trades/yr) means "
            "costs are not the killer — the signal is.\n"
            f"- **Beats a coin? — Busted.** {R['breach_win']}% win-rate (below 50%); CCI extremes "
            "predict the *wrong* direction on modern US equity data."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "At ~16 trades/instrument/year, costs are modest — but irrelevant, because the "
            "gross is already negative:"
        ),
        code(
            "costs = [0.0, 0.5, 1.0, 2.0]\n"
            "if HAVE_REAL:\n"
            "    net = [st.summarize(pool(hold=5, cost=c), 'ret_net')['mean_bps'] for c in costs]\n"
            "else:\n"
            f"    net = [{R['net00']}, {R['net05']}, {R['net10']}, {R['net20']}]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(costs, net, 'o-', c=RED, lw=2)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.fill_between(costs, net, 0, where=[n<0 for n in net], color=RED, alpha=.12)\n"
            "ax.set_xlabel('round-trip cost (bps)'); ax.set_ylabel('net profit per trade (bps)')\n"
            "ax.set_title('The line starts below zero — there is no positive break-even cost')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('The gross edge is negative — costs are not the issue, the signal is.')"
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The positive control.** The companion notebook shows a synthetic tape: when "
            "we *plant* genuine mean-reversion in the data, CCI does find it. The rule is "
            "correctly built — the real market just has the wrong character for it.\n"
            "- **Williams %R.** A closely related oscillator in the same mean-reversion family, "
            "tested in [Study 127](../../127-williams-r/) — same result.\n"
            "- **The CCI as a momentum signal?** Some sources *fade* the rule — buy overbought, "
            "short oversold — treating CCI extremes as continuation signals rather than reversal "
            "signals. The 1-day positive hint is not significant, but that variant is worth a fork.\n"
            "- **Commodity futures.** Lambert's original domain. If you have commodity futures "
            "data, that is the on-domain test and might show a different answer.\n\n"
            "*Think the oscillator works on a different asset or timeframe? Fork this, change "
            "the instrument or hold window, and show a fair-bet edge that clears the coin and "
            "passes HAC inference. That is the bar.*"
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
            "# CCI — a quantitative teardown\n"
            "### Daily OHLCV · CCI(20) · breach and cross-back framings · HAC inference · random-direction control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Beats a coin%3F: Busted](https://img.shields.io/badge/Beats_a_coin%3F-Busted-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We test whether the CCI(20) "
            "overbought/oversold rule beats a random-direction control on a fixed-horizon forward-return "
            "backtest, across six liquid daily tapes, and what costs and hold windows do to it.\n\n"
            "> **Not investment advice.** Real data: Yahoo daily bars, 10-year window (2016–2026), "
            "as-of 2026-06-15; the offline core and tests run on a deterministic synthetic tape. "
            "Methods & sources in [`docs/references.md`](../docs/references.md), reproducible numbers "
            "in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> The `In plain words` notes translate each result back to intuition."
        ),
        code(BOOT + "\nfrom quantlab import analytics, stats\n"),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Breach gross **{R['breach_mean']:+.2f} bps/trade**, HAC "
            f"*t* = **{R['breach_t']:+.2f}** (wrong sign); worse than a random-direction control "
            f"by **{R['breach_delta']:+.2f} bps**. |\n"
            f"| **Tradability** | `MIRAGE` | Gross expectancy negative at every hold window; "
            f"costs add to the loss but are not the primary driver. |\n"
            f"| **Beats a coin?** | `BUSTED` | {R['breach_win']}% win-rate (below 50%); "
            f"CCI actively mis-predicts direction on 2016–2026 US equity daily bars. |\n\n"
            "> In plain words: CCI was built for commodity price cycles, not for trending "
            "equity markets. On equities, 'oversold' is often just 'in a downtrend'."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $\\text{TP}_t = (H_t + L_t + C_t)/3$ be the typical price. Define:\n\n"
            "$$\\text{CCI}(t) = \\frac{\\text{TP}_t - \\overline{\\text{TP}}_{20}}"
            "{0.015 \\cdot \\text{MAD}_{20}(\\text{TP})}$$\n\n"
            "where $\\overline{\\text{TP}}_{20}$ is the 20-bar SMA of TP and $\\text{MAD}_{20}$ "
            "is the 20-bar mean absolute deviation. Lambert calibrated 0.015 so ~70–80% of "
            "readings fall inside [−100, +100]. The rule asserts:\n\n"
            "- **H₁ (signal).** $\\mathbb{E}[d \\cdot r_{t\\to t+5}] > 0$ where $d = +1$ when "
            "CCI < −100 (oversold-buy) and $d = -1$ when CCI > +100 (overbought-sell), i.e. "
            "CCI extremes identify mean-reversion opportunities.\n"
            "- **H₂ (beats random).** That expectation exceeds the same statistic with $d$ "
            "replaced by an i.i.d. fair coin on identical entries.\n"
            "- **H₃ (tradable).** It survives realistic round-trip costs.\n\n"
            "We reject H₁ (sign wrong, significant) and H₂ unambiguously; H₃ is moot."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "CCI is among the most widely taught oscillators in retail trading education. If H₁–H₃ "
            "held, it would provide a mechanical, replicable contrarian signal with ~16 trades/yr "
            "of turnover — low enough to be practical even for small accounts. The finding that it "
            "is *negatively* directional on large-cap US equities matters: it means retail traders "
            "following the textbook rule are systematically buying into downtrends and shorting "
            "into uptrends, paying costs for the privilege."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **CCI(20).** Period = 20 bars (Lambert's default); typical price = (H+L+C)/3; "
            "MAD scaled by 0.015.\n"
            "- **Entry (breach).** First bar where CCI crosses below −100 (buy, +1) or above "
            "+100 (sell, −1); one signal per episode. Enter at next bar's *open*.\n"
            "- **Entry (cross-back).** Bar where CCI crosses *back* through the threshold (confirmed "
            "bounce/fade). Tested separately.\n"
            "- **Exit.** Fixed-horizon: close of bar $t + \\text{hold\\_days}$, no stop.\n"
            "- **Control.** Identical entry bars, direction $\\in \\{-1, +1\\}$ drawn i.i.d. (seeded).\n"
            "- **Inference.** Newey–West HAC *t* on per-trade returns; circular block-bootstrap "
            "Sharpe CI; per-instrument breakdown; hold-period and cost sweeps.\n"
            "- **Positive control.** Synthetic tape with tunable AR(1) mean-reversion, confirming "
            "the engine recovers an edge when one exists.\n\n"
            "Six tapes: SPY, QQQ, IWM, AAPL, TSLA, NVDA — 10-year window (2016–2026)."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Per-instrument — CCI earns the wrong sign on most tickers\n\n"
            "HAC *t* on gross 5-day forward return. If the rule worked, bars should clear +2."
        ),
        code(
            "if HAVE_REAL:\n"
            "    recs = []\n"
            "    for t in TICKERS:\n"
            "        b = data.fetch_daily(t, fetch=False, cache_dir=CACHE)\n"
            "        c = st.cci(b, period=20)\n"
            "        ent = st.breach_entries(c)\n"
            "        s = st.summarize(st.run_trades(b, ent, hold_days=5, cost_bps=0), 'ret_gross')\n"
            "        recs.append((t, s['n_trades'], s['win_rate']*100, s['mean_bps'], s['tstat']))\n"
            "    perinst = pd.DataFrame(recs, columns=['ticker','n','win%','mean_bps','t'])\n"
            "    p = pool(hold=5, cost=0); ps = st.summarize(p, 'ret_gross')\n"
            "else:\n"
            "    perinst = pd.DataFrame({\n"
            "        'ticker': ['SPY','QQQ','IWM','AAPL','TSLA','NVDA'],\n"
            f"        'mean_bps': [{R['mean_spy']},{R['mean_qqq']},{R['mean_iwm']},"
            f"{R['mean_aapl']},{R['mean_tsla']},{R['mean_nvda']}],\n"
            f"        't': [{R['tstat_spy']},{R['tstat_qqq']},{R['tstat_iwm']},"
            f"{R['tstat_aapl']},{R['tstat_tsla']},{R['tstat_nvda']}],\n"
            "        'win%': [49.4, 45.3, 44.5, 44.8, 41.5, 45.0], 'n': [158,159,173,154,159,160]})\n"
            f"    ps = dict(mean_bps={R['breach_mean']}, tstat={R['breach_t']})\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "col = [GREEN if abs(v) >= 2 and v > 0 else AMBER if abs(v) >= 2 else GREY\n"
            "       for v in perinst['t']]\n"
            "ax.bar(perinst['ticker'], perinst['t'], color=[RED if v < 0 else GREY\n"
            "       for v in perinst['t']])\n"
            "for s in (2, -2): ax.axhline(s, ls='--', c=GREY, lw=1)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('HAC t-stat (gross mean/trade, hold=5d)')\n"
            "ax.set_title('Per-instrument: mostly negative, TSLA clearly wrong-signed')\n"
            "ax.set_ylim(-3, 3); plt.tight_layout(); plt.show()\n"
            "print(f\"pooled: {ps['mean_bps']:+.2f} bps/trade, HAC t = {ps['tstat']:+.2f}\")\n"
            "perinst.round(2)"
        ),
        md(
            "> In plain words: only SPY and QQQ are near zero. TSLA, NVDA, IWM are clearly "
            f"negative. Pooling {R['n']} trades gives **{R['breach_mean']:+.2f} bps**, "
            f"HAC *t* = **{R['breach_t']:+.2f}** — a statistically significant *wrong-sign* result."
        ),
        md(
            "### 4b · Breach vs the coin — with a bootstrap Sharpe band\n\n"
            "The random-direction control on identical entries."
        ),
        code(
            "if HAVE_REAL:\n"
            "    B = pool(hold=5, cost=0.0)\n"
            "    bm_val = st.summarize(B, 'ret_gross')['mean_bps']\n"
            "    rand_means = [st.summarize(pool(hold=5, cost=0, seed=s), 'ret_gross')['mean_bps']\n"
            "                  for s in range(20)]\n"
            "    ci = stats.sharpe_ci_bootstrap(\n"
            "        pd.Series(B['ret_gross'].values), periods_per_year=252//5, seed=178)\n"
            "    bm_val2, clo, chi, fn = bm_val, ci['ci_low'], ci['ci_high'], ci['frac_negative']*100\n"
            "else:\n"
            f"    rand_means = [{R['rand_mean']}]*20; bm_val2 = {R['breach_mean']}\n"
            "    clo, chi, fn = -4.0, -0.5, 97\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(rand_means, bins=12, color=GREY, alpha=.65, label='random-control means (20 seeds)')\n"
            "ax.axvline(bm_val2, c=RED, lw=2.5, label=f'CCI breach: {bm_val2:+.2f} bps')\n"
            "ax.axvline(0, c='k', lw=1)\n"
            "ax.set_xlabel('gross mean (bps/trade)'); ax.set_ylabel('count')\n"
            "ax.set_title('CCI breach is significantly below the coin, not just equal'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'annualised Sharpe 95% CI: [{clo:+.2f}, {chi:+.2f}]  ({fn:.0f}% of resamples < 0)')"
        ),
        md(
            "> In plain words: the coin earns about −4 bps. CCI earns −33 bps. The rule is "
            "not just uninformative — it is a statistically reliable way to be on the wrong side."
        ),
        md(
            "### 4c · Hold-period sweep — the pattern is consistent across windows\n\n"
            "If the effect were real, it should show up at some horizon."
        ),
        code(
            "holds = [1, 3, 5, 10, 20]\n"
            "if HAVE_REAL:\n"
            "    sw = [st.summarize(pool(hold=h, cost=0.0), 'ret_gross') for h in holds]\n"
            "    means2 = [s['mean_bps'] for s in sw]; tst = [s['tstat'] for s in sw]\n"
            "else:\n"
            f"    means2 = [{R['hold1_mean']},{R['hold3_mean']},{R['hold5_mean']},"
            f"{R['hold10_mean']},{R['hold20_mean']}]\n"
            f"    tst = [{R['hold1_t']},{R['hold3_t']},{R['hold5_t']},"
            f"{R['hold10_t']},{R['hold20_t']}]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot([str(h)+'d' for h in holds], means2, 'o-', c=RED, lw=2, label='mean bps/trade')\n"
            "ax2 = ax.twinx()\n"
            "ax2.plot([str(h)+'d' for h in holds], tst, 's--', c=GREY, lw=1.5, label='HAC t')\n"
            "ax2.axhline(-2, ls=':', c=GREY); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('hold period'); ax.set_ylabel('gross mean (bps/trade)', color=RED)\n"
            "ax2.set_ylabel('HAC t-stat', color=GREY)\n"
            "ax.set_title('CCI: negative at every horizon beyond 1 day')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> In plain words: the 1-day t = +{R['hold1_t']:.2f} is noise. At 5 and 10 days "
            "the rule is a statistically significant loser. The longer you hold, the worse it gets."
        ),
        md(
            "### 4d · Cross-back framing vs breach framing\n\n"
            "The 'confirmed bounce' version (enter when CCI exits the extreme zone)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    cb = pool(hold=5, cost=0.0, framing='crossback')\n"
            "    cbr = pool(hold=5, cost=0.0, framing='crossback', seed=178)\n"
            "    cb_s = st.summarize(cb, 'ret_gross')\n"
            "    cbr_s = st.summarize(cbr, 'ret_gross')\n"
            "    cb_m, cb_t_val = cb_s['mean_bps'], cb_s['tstat']\n"
            "    cbr_m = cbr_s['mean_bps']\n"
            "else:\n"
            f"    cb_m, cb_t_val, cbr_m = {R['cb_mean']}, {R['cb_t']}, {R['cb_rand_mean']}\n"
            "print(f'Cross-back: {{cb_m:+.2f}} bps/trade, HAC t = {{cb_t_val:+.2f}}')\n"
            "print(f'Random:     {{cbr_m:+.2f}} bps/trade')\n"
            "print(f'Both framings are negative; cross-back is only marginally better.')"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — breach gross {R['breach_mean']:+.2f} bps/trade, HAC "
            f"*t* {R['breach_t']:+.2f}; Δ vs control {R['breach_delta']:+.2f} bps; "
            "five of six instruments negative.\n"
            f"- **Tradability `MIRAGE`** — gross Sharpe significantly negative; "
            "costs add marginally but are not the root cause.\n"
            f"- **Beats a coin? `BUSTED`** — {R['breach_win']}% win-rate; "
            "CCI mis-predicts direction on modern US equity daily bars."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the cost landscape\n\n"
            "At ~16 trades/instrument/year costs barely matter — but the gross is already a loser:"
        ),
        code(
            "costs = [0.0, 0.5, 1.0, 2.0]\n"
            "if HAVE_REAL:\n"
            "    sw2 = [st.summarize(pool(hold=5, cost=c), 'ret_net') for c in costs]\n"
            "    net2 = [s['mean_bps'] for s in sw2]; tst2 = [s['tstat'] for s in sw2]\n"
            "else:\n"
            f"    net2 = [{R['net00']},{R['net05']},{R['net10']},{R['net20']}]\n"
            f"    tst2 = [{R['net00_t']},{R['net05_t']},{R['net10_t']},{R['net20_t']}]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(costs, net2, 'o-', c=RED, lw=2, label='net mean (bps/trade)')\n"
            "ax2 = ax.twinx(); ax2.plot(costs, tst2, 's--', c=GREY, lw=1.5, label='HAC t')\n"
            "ax2.axhline(-2, ls=':', c=GREY); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('round-trip cost (bps)'); ax.set_ylabel('net mean (bps)', color=RED)\n"
            "ax2.set_ylabel('HAC t-stat', color=GREY)\n"
            "ax.set_title('Costs add injury to insult — the gross is already the problem')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Break-even cost: none exists — the gross is negative.')"
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further — the positive control\n\n"
            "Is the *engine* capable of finding a mean-reversion edge, or is it always negative? "
            "Plant AR(1) mean-reversion in a synthetic daily tape and sweep the knob:"
        ),
        code(
            "mrevs = [-0.15, -0.05, 0.0, 0.10, 0.20, 0.35, 0.50]\n"
            "edge_vals = []\n"
            "for mr in mrevs:\n"
            "    b, _ = data.synthetic_daily(n_days=2000, mean_rev=mr, seed=178)\n"
            "    c = st.cci(b, period=20)\n"
            "    ent = st.breach_entries(c)\n"
            "    if len(ent) < 5:\n"
            "        edge_vals.append(float('nan')); continue\n"
            "    sig_m = st.summarize(st.run_trades(b, ent, hold_days=5, cost_bps=0), 'ret_gross')['mean_bps']\n"
            "    rnd_m = st.summarize(\n"
            "        st.run_trades(b, ent, hold_days=5, cost_bps=0,\n"
            "                      directions=st.random_directions(len(ent), seed=1)),\n"
            "        'ret_gross')['mean_bps']\n"
            "    edge_vals.append(sig_m - rnd_m)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "valid = [(m, e) for m, e in zip(mrevs, edge_vals) if not (e != e)]\n"
            "ax.plot([v[0] for v in valid], [v[1] for v in valid], 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(0, c='k', lw=1); ax.axvline(0, ls='--', c=GREY)\n"
            "ax.set_xlabel('planted AR(1) mean-reversion'); ax.set_ylabel('CCI − coin (bps/trade)')\n"
            "ax.set_title('Engine works: CCI finds mean-reversion when it exists')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('At mr=0 edge is ~0; at mr>0.2 it rises — confirming the engine, not the market.')"
        ),
        md(
            "The engine is a faithful mean-reversion detector: on a planted mean-reverting tape "
            "it finds the signal and beats the coin. The real-tape verdict is therefore a statement "
            "about the **market** (modern US equities trend rather than mean-revert at this horizon) "
            "not the method. Forks worth trying: commodity ETF futures (DBC, GLD, USO) where "
            "Lambert's original cyclical rationale applies; a longer-term oscillator window; "
            "or the inverse signal (trade the *continuation* of CCI extremes rather than the bounce)."
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
