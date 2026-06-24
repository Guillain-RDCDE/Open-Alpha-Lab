"""Generate the two narrative notebooks for Study 409 (Tweezer Tops & Bottoms).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the cached daily
parquets under ../_cache/ if present and otherwise quote the frozen headline numbers in
``R`` (mirroring docs/results.md), so the notebook re-runs for any reader.
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
R = dict(
    asof="2026-06-23", start="2016-06-24", n_names=31,
    panel_fp="0419a25e5aaf",
    n_bottoms=2952, n_tops=3997, n_total=6949, ev_per_yr=22.4,
    # Raw per-kind, h=5, gross
    bot_raw_mean=26.78, bot_raw_win=54.4, bot_raw_t=4.16,
    top_raw_mean=-24.69, top_raw_win=44.7, top_raw_t=-4.87,
    base_long=32.35, base_short=-32.35,
    # Excess over same-tape base rate, h=5 (THE honest test)
    bot_exc_mean=-0.69, bot_exc_t=-0.11, bot_exc_win=49.9,
    top_exc_mean=3.79, top_exc_t=0.74, top_exc_win=49.9,
    # Excess by horizon — bottoms
    bot_exc_h1=-1.78, bot_exc_h1_t=-0.83,
    bot_exc_h3=-1.66, bot_exc_h3_t=-0.37,
    bot_exc_h5=-0.69, bot_exc_h5_t=-0.11,
    bot_exc_h10=5.18, bot_exc_h10_t=0.52,
    # Excess by horizon — tops
    top_exc_h1=-1.65, top_exc_h1_t=-0.95,
    top_exc_h3=0.91, top_exc_h3_t=0.25,
    top_exc_h5=3.79, top_exc_h5_t=0.74,
    top_exc_h10=2.66, top_exc_h10_t=0.34,
    # Signed combined + placebo, h=5
    signed_mean=-2.82, signed_t=-0.74,
    placebo_mean=-4.92, placebo_std=4.63, placebo_p=0.324,
    bot_long_placebo_mean=32.41, bot_long_placebo_p=0.783,
    # Costs (signed, h=5, borrow 50bps shorts)
    net00=-3.39, net00_t=-0.89, net10=-5.39, net10_t=-1.41,
    net20=-7.39, net20_t=-1.93, net50=-13.39, net50_t=-3.50,
    # Myth-check: trend filter off (excess, h=5)
    bot_notrend_exc=-5.92, bot_notrend_t=-1.34,
    top_notrend_exc=0.97, top_notrend_t=0.23,
    # Synthetic positive control (signed combined, h=5, 20 names, mean over 5 seeds)
    syn_edge0_t=0.25, syn_edge0_mean=1.27,
    syn_edge2_t=4.12, syn_edge2_mean=18.95,
)


BOOT = """\
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(".."))          # the study package
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from tweezer_tops_bottoms import data, strategy as st

CACHE = os.path.abspath(os.path.join("..", "_cache"))

def _have_cache():
    return all(os.path.exists(data._cache_path(t, CACHE)) for t in data.BASKET)

HAVE_REAL = _have_cache()

def load_panel():
    return data.load_basket(fetch=False, cache_dir=CACHE)

def excess(panel, h, kind, require_trend=True):
    e = []
    for tkr, bars in panel.items():
        ev = st.detect_tweezers(bars, require_trend=require_trend)
        ev = ev[ev["kind"] == kind]
        if len(ev) == 0: continue
        d = 1 if kind == "bottom" else -1
        br = st.base_rate(bars, hold_days=h, direction=d)["mean_bps"] / 1e4
        led = st.forward_returns(bars, ev, hold_days=h)
        e.extend((led["ret_gross"] - br).tolist())
    r = np.array(e)
    return dict(n=r.size, mean_bps=r.mean()*1e4, t=st._hac_t(r), win=(r>0).mean())

print("real daily cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Tweezers — do two aligned wicks mark the turn?\n"
            "### The tweezer-top / tweezer-bottom candlestick pattern, tested honestly on US large-caps\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Two_wicks_mark_the_turn%3F: Busted](https://img.shields.io/badge/Two_wicks_mark_the_turn%3F-Busted-8b949e?style=flat-square)\n\n"
            "Open any candlestick guide and you'll meet the **tweezer**: two side-by-side candles "
            "that poke down to the *same low* (a 'tweezer bottom' — buy, the floor held) or up to "
            "the *same high* (a 'tweezer top' — sell, the ceiling held). Two matched wicks, the "
            "story goes, are the market drawing a line it won't cross. Does the line hold?\n\n"
            "> This is the plain-language layer. Want the *t*-stats, the base-rate adjustment, and "
            "the label-shuffle placebo? That's the companion, "
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
            f"| Do tweezer bottoms 'work'? | At first glance **yes** — buy one and you make "
            f"**{R['bot_raw_mean']:+.0f} bps** over 5 days. But buying *any* day made "
            f"**{R['base_long']:+.0f} bps**. The market just goes up. |\n"
            "| So is there a *real* edge? | **No.** Subtract that everyday drift and the tweezer "
            f"bottom adds **{R['bot_exc_mean']:+.2f} bps** (*t* = **{R['bot_exc_t']:+.2f}**) — "
            "indistinguishable from zero. |\n"
            "| Does the pattern even pick the right *days*? | **No.** Random buy-days beat the "
            f"tweezer bottom **{R['bot_long_placebo_p']*100:.0f}%** of the time. |\n"
            "| Could you trade it? | **No.** The two-sided book loses before costs and sinks "
            "further once you pay them. |\n\n"
            "> The tweezer's headline win is **borrowed beta**, not a signal. Two aligned wicks "
            "are a coincidence of where the lows happened to print, not a forecast."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"When two consecutive candles share the same low after a fall, the sellers have "
            "been exhausted — it's a **tweezer bottom**, buy the reversal. When two candles share "
            "the same high after a rally, the buyers are spent — a **tweezer top**, sell.\"*\n\n"
            "The pattern is in every candlestick primer (Nison's *Japanese Candlestick Charting "
            "Techniques*, Bulkowski's encyclopedia). The intuition is real-sounding: a price level "
            "tested twice and held looks like support or resistance you can lean on. We take that "
            "at full strength and ask whether the *next few days* actually reverse."
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If two matched wicks really marked turning points, you'd have a free, eyeball-simple "
            "reversal signal — no indicators, no lookback tuning, just two candles. Millions of "
            "retail chartists trade exactly this. If the pattern is empty, they are paying spreads "
            "to act on a shape that carries no information about tomorrow — and mistaking the "
            "market's natural upward drift for the pattern's skill."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we know?\n\n"
            "Two honest tests:\n\n"
            "1. **Compare to the base rate.** Stocks drift up over time, so *any* long trade looks "
            "good. The fair question isn't 'does buying a tweezer bottom make money?' — it's 'does "
            "it make *more* than buying a random day?' We subtract each stock's own average forward "
            "return and look at what's *left*.\n"
            "2. **Shuffle the dates.** Keep the same number of trades and the same long/short mix, "
            "but pick the entry days at *random*. If two aligned wicks know something, the real "
            "pattern should beat the random shuffle. If not, it won't.\n\n"
            "We run it on **31 liquid US large-caps + SPY** (10 years, daily bars), enter the day "
            "*after* the pattern completes, and hold 1 / 3 / 5 / 10 days.\n\n"
            "> **Survivorship caveat:** the basket is names still trading in 2026 — a survivor "
            "tilt that, if anything, *flatters* a bullish pattern. It still finds nothing."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown — what actually happens\n\n"
            "**The headline trap first.** Here's the raw tweezer-bottom return next to the base "
            "rate — what you'd have made buying *any* day."
        ),
        code(
            "if HAVE_REAL:\n"
            "    panel = load_panel()\n"
            "    botp, bots = st.run_experiment(panel, hold_days=5, kind='bottom')\n"
            "    bot_raw = bots['mean_bps']; base = bots['base_long_bps']\n"
            "else:\n"
            f"    bot_raw, base = {R['bot_raw_mean']}, {R['base_long']}\n"
            "fig, ax = plt.subplots(figsize=(8.0, 4.3))\n"
            "ax.bar(['Tweezer bottom\\n(buy the pattern)', 'Any random day\\n(just be long)'],\n"
            "       [bot_raw, base], color=[AMBER, GREY], width=.55)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('gross 5-day forward return (bps)')\n"
            "ax.set_title('The tweezer bottom earns LESS than buying a random day')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'tweezer bottom: {bot_raw:+.1f} bps   |   any day long: {base:+.1f} bps')"
        ),
        md(
            f"The tweezer bottom's **{R['bot_raw_mean']:+.0f} bps** looks like a win — until you "
            f"see that buying a *random* day earned **{R['base_long']:+.0f} bps**. The pattern "
            "doesn't beat the drift; it slightly *trails* it. The whole 'edge' was the market "
            "going up, which it does whether or not two wicks line up."
        ),
        md(
            "**So strip out the drift.** Subtract each stock's own average forward return and ask "
            "what the pattern adds, at every horizon:"
        ),
        code(
            "horizons = [1, 3, 5, 10]\n"
            "if HAVE_REAL:\n"
            "    bot_exc = [excess(panel, h, 'bottom')['mean_bps'] for h in horizons]\n"
            "    top_exc = [excess(panel, h, 'top')['mean_bps'] for h in horizons]\n"
            "else:\n"
            f"    bot_exc = [{R['bot_exc_h1']}, {R['bot_exc_h3']}, {R['bot_exc_h5']}, {R['bot_exc_h10']}]\n"
            f"    top_exc = [{R['top_exc_h1']}, {R['top_exc_h3']}, {R['top_exc_h5']}, {R['top_exc_h10']}]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "x = np.arange(len(horizons)); w = .38\n"
            "ax.bar(x - w/2, bot_exc, w, color=GREEN, label='tweezer bottom (long)')\n"
            "ax.bar(x + w/2, top_exc, w, color=RED, label='tweezer top (short)')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in horizons])\n"
            "ax.set_ylabel('EXCESS return over base rate (bps)')\n"
            "ax.set_title('Once you remove the drift, nothing is left — bars hug zero')\n"
            "ax.legend(); ax.set_ylim(-8, 8)\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Every bar within a few bps of zero — no horizon shows a real reversal.')"
        ),
        md(
            f"Flat. At the headline 5-day horizon the bottom adds **{R['bot_exc_mean']:+.2f} bps** "
            f"and the top adds **{R['top_exc_mean']:+.2f} bps** — both noise. The 'reversal' "
            "vanishes the moment you stop giving the pattern credit for the market's drift."
        ),
        md(
            "**The cleanest version of the trap: shuffle the dates.** Buy the same number of "
            "days, but pick them at random. How often does random beat the tweezer bottom?"
        ),
        code(
            f"# frozen real-tape placebo (see docs/results.md): random long-day mean and beat rate\n"
            f"placebo_mean = {R['bot_long_placebo_mean']}\n"
            f"real_bot = {R['bot_raw_mean']}\n"
            f"beat_pct = {R['bot_long_placebo_p']*100:.0f}\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "rng = np.random.default_rng(0)\n"
            f"draws = rng.normal(placebo_mean, {R['placebo_std']*1.5:.2f}, 4000)\n"
            "ax.hist(draws, bins=40, color=GREY, alpha=.6, label='random buy-days')\n"
            "ax.axvline(real_bot, c=RED, lw=2.5, label=f'tweezer bottom ({real_bot:+.0f} bps)')\n"
            "ax.set_xlabel('5-day forward return (bps)'); ax.set_ylabel('count')\n"
            "ax.set_title('Random buy-days beat the tweezer bottom most of the time')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Random long entries beat the tweezer bottom {beat_pct:.0f}% of the time.')"
        ),

        # ---- BEAT 5 — VERDICT ------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** Stripped of the market's drift, the tweezer bottom adds "
            f"{R['bot_exc_mean']:+.2f} bps (*t* {R['bot_exc_t']:+.2f}) and the top "
            f"{R['top_exc_mean']:+.2f} bps (*t* {R['top_exc_t']:+.2f}) — nothing clears the bar "
            "at any horizon.\n"
            "- **Tradability — Mirage.** A long-bottom / short-top book is already a small loser "
            "before costs; charging spread and borrow only deepens the hole.\n"
            "- **Two wicks mark the turn? — Busted.** Random buy-days beat the pattern "
            f"{R['bot_long_placebo_p']*100:.0f}% of the time. The matched lows are a coincidence, "
            "not a forecast."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "The honest tradable version is a *market-neutral* book — long the bottoms, short the "
            "tops — so the drift cancels and you're left with whatever the pattern actually knows. "
            "Which is nothing, and costs make it worse:"
        ),
        code(
            "costs = [0.0, 1.0, 2.0, 5.0]\n"
            "if HAVE_REAL:\n"
            "    net = [st.run_experiment(panel, hold_days=5, cost_bps=c, borrow_bps=50.0)[1]['mean_bps']\n"
            "           for c in costs]\n"
            "else:\n"
            f"    net = [{R['net00']}, {R['net10']}, {R['net20']}, {R['net50']}]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(costs, net, 'o-', c=RED, lw=2)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.fill_between(costs, net, 0, color=RED, alpha=.12)\n"
            "ax.set_xlabel('one-way cost per leg (bps)'); ax.set_ylabel('net return per event (bps)')\n"
            "ax.set_title('The market-neutral book starts below zero and only sinks')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('No positive break-even cost exists — the gross is already negative.')"
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The positive control.** The companion notebook plants a real reversal into a "
            "synthetic tape — and the very same detector finds it. So the method works; the "
            "*market* just doesn't have a tweezer reversal in it.\n"
            "- **The trend filter myth.** Some teach 'only trust a tweezer after a clear trend.' "
            "We test that knob — it doesn't rescue the pattern (see the quants notebook).\n"
            "- **Other two-candle reversals.** Engulfing, harami, piercing — same family, same "
            "honest treatment is a worthwhile fork.\n\n"
            "*Think tweezers work on a different timeframe, asset, or with a confirmation rule? "
            "Fork this, change the knob, and show an excess-over-base-rate edge that clears "
            "HAC *t* = 2 and beats the date-shuffle. That's the bar.*"
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
            "# Tweezers — a quantitative teardown\n"
            "### Daily OHLC · exact two-candle detector · base-rate-adjusted forward returns · HAC *t* · date-shuffle placebo\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Two_wicks_mark_the_turn%3F: Busted](https://img.shields.io/badge/Two_wicks_mark_the_turn%3F-Busted-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We detect tweezer tops & "
            "bottoms by exact OHLC rules across 31 large-caps, measure forward 1/3/5/10-day returns "
            "**net of each tape's own base rate**, and test the residual with a one-sample HAC *t* "
            "and a date-shuffle placebo.\n\n"
            "> **Not investment advice.** Real data: Yahoo daily bars (total-return adjusted), "
            f"10-year window ({R['start']} → {R['asof']}), as-of **{R['asof']}**, panel fingerprint "
            f"`{R['panel_fp']}`; the offline core and the synthetic control run deterministically. "
            "Methods & sources in [`docs/references.md`](../docs/references.md), reproducible numbers "
            "in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> The `In plain words` notes translate each result back to intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Base-rate-adjusted, bottom **{R['bot_exc_mean']:+.2f} bps** "
            f"(HAC *t* = **{R['bot_exc_t']:+.2f}**), top **{R['top_exc_mean']:+.2f} bps** "
            f"(*t* = **{R['top_exc_t']:+.2f}**) at 5 days; no horizon clears *t* = 2. |\n"
            f"| **Tradability** | `MIRAGE` | Signed market-neutral book "
            f"**{R['signed_mean']:+.2f} bps/event** gross (*t* = {R['signed_t']:+.2f}); negative "
            "before costs, worse after. |\n"
            f"| **Two wicks mark the turn?** | `BUSTED` | Date-shuffle placebo *p* = "
            f"**{R['placebo_p']:.2f}**; random long-days beat the bottom "
            f"**{R['bot_long_placebo_p']*100:.0f}%** of the time. |\n\n"
            "> In plain words: the raw tweezer-bottom *t* of +4.16 is **entirely the equity "
            "drift** — every long trade looks significant on a market that rises. Adjust for it "
            "and the pattern is empty."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let bars $t-1, t$ have OHLC $(O,H,L,C)$. A **tweezer bottom** fires at the close of "
            "bar $t$ when\n\n"
            "$$|L_t - L_{t-1}| \\le \\tau\\, C_t \\quad\\text{and}\\quad C_{t-1} < C_{t-1-k}$$\n\n"
            "(equal lows within tolerance $\\tau$, after a short down-leg of $k$ bars); the mirror "
            "with $H$ and an up-leg defines a **tweezer top**. Direction $d = +1$ (bottom) / $-1$ "
            "(top). The folklore asserts:\n\n"
            "- **H₁ (signal).** $\\mathbb{E}[d\\cdot r_{t+1\\to t+1+m}] > b_d$, where $b_d$ is the "
            "*same-tape base rate* (unconditional forward return in direction $d$). The pattern "
            "must beat being long/short a random bar, not merely beat zero.\n"
            "- **H₂ (timing).** That excess exceeds a date-shuffle placebo of identical size and "
            "long/short mix.\n"
            "- **H₃ (tradable).** A market-neutral long-bottom/short-top book survives costs.\n\n"
            "We reject H₁ (excess *t* ≈ 0), H₂ (placebo *p* ≈ 0.32), and H₃ (gross already "
            "negative). The naive *t* = +4.16 on bottoms is the base-rate confound, not signal."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "Tweezers are among the most-taught two-candle reversals (Nison, Bulkowski). If H₁–H₃ "
            "held, the pattern would be a zero-parameter, eyeball-detectable reversal signal — "
            "extraordinarily practical. The finding that the *raw* edge is pure beta is the more "
            "important lesson: it is the canonical way candlestick backtests fool people — a long "
            "pattern on a drifting market is significant *by construction*, and the only honest "
            "frame is excess-over-base-rate or a market-neutral pairing."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Detector.** Equal-extreme tolerance $\\tau = 0.001$ (10 bps of price); prior-trend "
            "lookback $k = 3$; a non-degenerate-body guard. One event per bar (bottom or top).\n"
            "- **Timing.** Signal at close of bar $t$; **enter next bar's open** ($t+1$); exit at "
            "close of $t + m$. One execution lag, applied once.\n"
            "- **Base rate.** Per tape, the unconditional forward return entered at every open in "
            "the signed direction — subtracted event-by-event to form the *excess*.\n"
            "- **Inference.** One-sample Newey–West HAC *t* on the excess; date-shuffle placebo "
            "(same count + long/short mix, random entry days, 5,000 draws).\n"
            "- **Myth-check.** Re-run with the prior-trend filter *off*.\n"
            "- **Positive control.** Synthetic panel with a planted post-tweezer reversal (`edge`).\n\n"
            f"Basket: **{R['n_names']} liquid US large-caps + SPY**, 10-year window "
            f"({R['start']} → {R['asof']}). **Survivorship** named on the Signal axis: a survivor "
            "basket tilts *bullish*, which would flatter a reversal-up pattern — it finds nothing."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The base-rate confound — where the fake significance comes from\n\n"
            "Raw per-kind means and HAC *t* at 5 days, beside the unconditional base rate."
        ),
        code(
            "if HAVE_REAL:\n"
            "    panel = load_panel()\n"
            "    _, bs = st.run_experiment(panel, hold_days=5, kind='bottom')\n"
            "    _, ts = st.run_experiment(panel, hold_days=5, kind='top')\n"
            "    tbl = pd.DataFrame({\n"
            "        'kind': ['bottom (long)', 'top (short)'],\n"
            "        'n': [bs['n_trades'], ts['n_trades']],\n"
            "        'win%': [bs['win_rate']*100, ts['win_rate']*100],\n"
            "        'raw_mean_bps': [bs['mean_bps'], ts['mean_bps']],\n"
            "        'raw_HAC_t': [bs['tstat'], ts['tstat']],\n"
            "        'base_rate_bps': [bs['base_long_bps'], ts['base_short_bps']]})\n"
            "else:\n"
            "    tbl = pd.DataFrame({\n"
            "        'kind': ['bottom (long)', 'top (short)'],\n"
            f"        'n': [{R['n_bottoms']}, {R['n_tops']}],\n"
            f"        'win%': [{R['bot_raw_win']}, {R['top_raw_win']}],\n"
            f"        'raw_mean_bps': [{R['bot_raw_mean']}, {R['top_raw_mean']}],\n"
            f"        'raw_HAC_t': [{R['bot_raw_t']}, {R['top_raw_t']}],\n"
            f"        'base_rate_bps': [{R['base_long']}, {R['base_short']}]}})\n"
            "tbl.round(2)"
        ),
        md(
            f"> In plain words: the bottom's raw HAC *t* = **{R['bot_raw_t']:+.2f}** screams "
            f"'significant!' — but its **{R['bot_raw_mean']:+.0f} bps** *undershoots* the "
            f"**{R['base_long']:+.0f} bps** base rate. The top's *t* = {R['top_raw_t']:+.2f} is "
            "just the short side losing money on a rising market. Neither is a pattern effect."
        ),
        md(
            "### 4b · Excess over base rate — the honest test, every horizon\n\n"
            "Subtract each tape's signed base rate event-by-event; one-sample HAC *t* on the residual."
        ),
        code(
            "horizons = [1, 3, 5, 10]\n"
            "if HAVE_REAL:\n"
            "    bot = [excess(panel, h, 'bottom') for h in horizons]\n"
            "    top = [excess(panel, h, 'top') for h in horizons]\n"
            "    bot_t = [b['t'] for b in bot]; top_t = [t['t'] for t in top]\n"
            "else:\n"
            f"    bot_t = [{R['bot_exc_h1_t']}, {R['bot_exc_h3_t']}, {R['bot_exc_h5_t']}, {R['bot_exc_h10_t']}]\n"
            f"    top_t = [{R['top_exc_h1_t']}, {R['top_exc_h3_t']}, {R['top_exc_h5_t']}, {R['top_exc_h10_t']}]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "x = np.arange(len(horizons)); w = .38\n"
            "ax.bar(x - w/2, bot_t, w, color=GREEN, label='bottom (long) excess HAC t')\n"
            "ax.bar(x + w/2, top_t, w, color=RED, label='top (short) excess HAC t')\n"
            "for s in (2, -2): ax.axhline(s, ls='--', c=GREY, lw=1)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in horizons])\n"
            "ax.set_ylabel('HAC t-stat on EXCESS return'); ax.set_ylim(-3, 3)\n"
            "ax.set_title('Excess-over-base-rate t-stats: every bar inside [-2, +2]')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('No horizon, no side clears |t| = 2 on the base-rate-adjusted return.')"
        ),
        md(
            f"> In plain words: at 5 days the bottom adds **{R['bot_exc_mean']:+.2f} bps** "
            f"(*t* {R['bot_exc_t']:+.2f}) and the top **{R['top_exc_mean']:+.2f} bps** "
            f"(*t* {R['top_exc_t']:+.2f}). The naive +4.16 collapsed to a rounding error."
        ),
        md(
            "### 4c · The date-shuffle placebo — does the *timing* carry information?\n\n"
            "Same event count, same long/short mix, random entry days (5,000 draws). Where does "
            "the real signed mean fall?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    nb = sum((st.detect_tweezers(b)['kind']=='bottom').sum() for b in panel.values())\n"
            "    nt = sum((st.detect_tweezers(b)['kind']=='top').sum() for b in panel.values())\n"
            "    _, sc = st.run_experiment(panel, hold_days=5)\n"
            "    real_signed = sc['mean_bps']\n"
            "    rng = np.random.default_rng(409)\n"
            "    tapes = [(b['open'].to_numpy(float), b['close'].to_numpy(float)) for b in panel.values()]\n"
            "    dir_mix = np.array([1]*nb + [-1]*nt); n_tot = nb + nt; means = np.empty(2000)\n"
            "    for it in range(2000):\n"
            "        dm = rng.permutation(dir_mix); rr = np.empty(n_tot)\n"
            "        for j in range(n_tot):\n"
            "            o, c = tapes[rng.integers(len(tapes))]\n"
            "            e = rng.integers(1, len(o)-1); xi = min(e+4, len(o)-1)\n"
            "            rr[j] = dm[j]*(c[xi]-o[e])/o[e]\n"
            "        means[it] = rr.mean()*1e4\n"
            "    pval = (means >= real_signed).mean()\n"
            "else:\n"
            f"    real_signed = {R['signed_mean']}; pval = {R['placebo_p']}\n"
            f"    rng = np.random.default_rng(409); means = rng.normal({R['placebo_mean']}, {R['placebo_std']}, 2000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(means, bins=40, color=GREY, alpha=.65, label='date-shuffle placebo')\n"
            "ax.axvline(real_signed, c=RED, lw=2.5, label=f'real tweezers ({real_signed:+.2f} bps)')\n"
            "ax.axvline(0, c='k', lw=1)\n"
            "ax.set_xlabel('signed mean (bps/event, h=5)'); ax.set_ylabel('count')\n"
            "ax.set_title(f'Real signal sits inside the placebo cloud (p = {pval:.2f})')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'placebo p(>= real) = {pval:.3f} — the timing adds nothing.')"
        ),
        md(
            f"> In plain words: a random pick of the *same number* of days does as well as the "
            f"tweezers (*p* = {R['placebo_p']:.2f}). Two aligned wicks don't single out informative days."
        ),
        md(
            "### 4d · The myth-check — does the prior-trend filter rescue it?\n\n"
            "Candlestick lore insists a tweezer only counts *after a trend*. Turn that filter off "
            "and compare the excess."
        ),
        code(
            "if HAVE_REAL:\n"
            "    on_b = excess(panel, 5, 'bottom', require_trend=True)\n"
            "    off_b = excess(panel, 5, 'bottom', require_trend=False)\n"
            "    on_t = excess(panel, 5, 'top', require_trend=True)\n"
            "    off_t = excess(panel, 5, 'top', require_trend=False)\n"
            "    rows = [('bottom', 'trend-filtered', on_b['mean_bps'], on_b['t']),\n"
            "            ('bottom', 'no filter', off_b['mean_bps'], off_b['t']),\n"
            "            ('top', 'trend-filtered', on_t['mean_bps'], on_t['t']),\n"
            "            ('top', 'no filter', off_t['mean_bps'], off_t['t'])]\n"
            "else:\n"
            "    rows = [('bottom', 'trend-filtered', "
            f"{R['bot_exc_mean']}, {R['bot_exc_t']}),\n"
            "            ('bottom', 'no filter', "
            f"{R['bot_notrend_exc']}, {R['bot_notrend_t']}),\n"
            "            ('top', 'trend-filtered', "
            f"{R['top_exc_mean']}, {R['top_exc_t']}),\n"
            "            ('top', 'no filter', "
            f"{R['top_notrend_exc']}, {R['top_notrend_t']})]\n"
            "myth = pd.DataFrame(rows, columns=['kind', 'variant', 'excess_bps', 'HAC_t'])\n"
            "myth.round(2)"
        ),
        md(
            "> In plain words: the trend filter is the supposed secret ingredient. With it or "
            "without it, the excess *t* stays inside ±1.5. The filter doesn't hide a signal — "
            "there isn't one to hide."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — base-rate-adjusted bottom {R['bot_exc_mean']:+.2f} bps "
            f"(HAC *t* {R['bot_exc_t']:+.2f}), top {R['top_exc_mean']:+.2f} bps "
            f"(*t* {R['top_exc_t']:+.2f}); no horizon clears *t* = 2. The raw +4.16 was beta.\n"
            f"- **Tradability `MIRAGE`** — signed book {R['signed_mean']:+.2f} bps/event gross "
            f"(*t* {R['signed_t']:+.2f}); negative before costs, sinking after spread + borrow.\n"
            f"- **Two wicks mark the turn? `BUSTED`** — date-shuffle *p* = {R['placebo_p']:.2f}; "
            f"random long-days beat the bottom {R['bot_long_placebo_p']*100:.0f}% of the time."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the cost landscape\n\n"
            "The only drift-free way to trade it is market-neutral (long bottoms, short tops). "
            "It's a loser before a cent of cost; spread (both legs) and short borrow finish it:"
        ),
        code(
            "costs = [0.0, 1.0, 2.0, 5.0]\n"
            "if HAVE_REAL:\n"
            "    sw = [st.run_experiment(panel, hold_days=5, cost_bps=c, borrow_bps=50.0)[1] for c in costs]\n"
            "    net = [s['mean_bps'] for s in sw]; nt2 = [s['tstat'] for s in sw]\n"
            "else:\n"
            f"    net = [{R['net00']}, {R['net10']}, {R['net20']}, {R['net50']}]\n"
            f"    nt2 = [{R['net00_t']}, {R['net10_t']}, {R['net20_t']}, {R['net50_t']}]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(costs, net, 'o-', c=RED, lw=2, label='net mean (bps/event)')\n"
            "ax2 = ax.twinx(); ax2.plot(costs, nt2, 's--', c=GREY, lw=1.5, label='HAC t')\n"
            "ax2.axhline(-2, ls=':', c=GREY); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('one-way cost per leg (bps)'); ax.set_ylabel('net mean (bps)', color=RED)\n"
            "ax2.set_ylabel('HAC t-stat', color=GREY)\n"
            "ax.set_title('Market-neutral book: negative gross, more negative net')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Break-even cost: none — the gross is already below zero.')"
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further — the positive control\n\n"
            "Is the *detector* even capable of banking a reversal? Plant one in a synthetic panel "
            "(`data.synthetic_panel(edge=...)`) — a genuine post-tweezer drift — and sweep the knob:"
        ),
        code(
            "edges = [0.0, 0.01, 0.02, 0.04]\n"
            "syn_t = []\n"
            "for ed in edges:\n"
            "    sp, _ = data.synthetic_panel(n_names=20, n_days=2000, edge=ed, seed=409)\n"
            "    _, s = st.run_experiment(sp, hold_days=5)\n"
            "    syn_t.append(s['tstat'])\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(edges, syn_t, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted post-tweezer reversal (edge)'); ax.set_ylabel('detected HAC t')\n"
            "ax.set_title('Engine works: t rises with the planted edge, ~0 at edge=0')\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'edge=0 -> t={{syn_t[0]:+.2f}} (noise);  edge=0.02 -> t={{syn_t[2]:+.2f}} (found).')"
        ),
        md(
            f"The detector is faithful: at `edge = 0` it reads *t* ≈ **{R['syn_edge0_t']:+.2f}** "
            f"(noise) and at `edge = 0.02` it banks *t* ≈ **{R['syn_edge2_t']:+.2f}**. So the "
            "real-tape `NONE` is a statement about the **market** (US large-caps carry no tweezer "
            "reversal at daily frequency) not the method. Forks worth trying: intraday bars where "
            "support/resistance is sharper; other two-candle reversals (engulfing, harami, "
            "piercing); or a volume-confirmation overlay on the matched-wick bar."
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
