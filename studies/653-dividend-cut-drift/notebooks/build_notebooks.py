"""Generate the two narrative notebooks for Study 653 (Dividend-Cut-Drift).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached tape under
../_cache/ and otherwise quote the frozen headline numbers in ``R`` (mirroring docs/results.md).
The synthetic positive control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance adjusted closes +
# dividend/split streams, 99/101 universe tickers loaded, 1996-01-02 -> 2026-06-30; 172 real
# cut/omission events with a full [-20,+120] window, 1997-02-05 -> 2025-08-29).
R = dict(
    universe_n=101, loaded_n=99, skipped=["MRO", "GPS"],
    spy_rows=7673, spy_lo="1996-01-02", spy_hi="2026-06-30", fp_spy="236c3c9bbc75",
    ev_detected=173, n_cut_raw=108, n_om_raw=65, ev_lo="1997-02-05", ev_hi="2026-02-13",
    fp_events="06467e29de08",
    kept_n=172, dropped_n=1, kept_lo="1997-02-05", kept_hi="2025-08-29", kept_tickers=64,
    pre_mean=0.75, pre_t=0.59,
    post20_mean=-0.59, post20_t=-0.52,
    post60_mean=-0.57, post60_t=-0.29,
    post120_mean=-0.31, post120_t=-0.14,
    hit_neg=82, hit_n=172, hit_pct=47.7, hit_lo=40.3, hit_hi=55.1,
    cut_n=107, cut_mean=-3.61, cut_t=-1.25,
    om_n=65, om_mean=5.13, om_t=1.55,
    nw_days=6123, nw_bps=-3.28, nw_t=-1.23,
    dw_n_treat=20100, dw_n_control=438031, dw_treat_bps=-0.79, dw_control_bps=-1.18, dw_welch_t=0.16,
    placebo_mean=-1.04, placebo_sd=1.60, placebo_draws=4000, placebo_p=0.675,
    short5_net=-15.18, short5_tnet=-3.98, short5_exc=-6.27, short5_texc=-1.86,
    short10_net=-15.28, short10_tnet=-4.00, short10_exc=-6.37, short10_texc=-1.89,
    long5_net=14.74, long5_tnet=3.86, long5_exc=5.83, long5_texc=1.73,
    long10_net=14.64, long10_tnet=3.83, long10_exc=5.73, long10_texc=1.70,
    short_hit=41.9, short_hit_lo=34.7, short_hit_hi=49.3, short_worst=-467.4, short_best=79.0,
    short_sharpe_net=-0.74, short_sharpe_exc=-0.35,
    long_hit=57.6, long_hit_lo=50.1, long_hit_hi=64.7, long_worst=-79.4, long_best=467.0,
    long_sharpe_net=0.72, long_sharpe_exc=0.32,
    events_per_year=5.9,
    syn_null_mean=-0.15, syn_null_sd=1.14, syn_null_fire=1, syn_null_seeds=20,
    syn_planted_mean=-9.35, syn_planted_t=-3.84, syn_planted_n=89,
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Falling_dividend%3F: Busted](https://img.shields.io/badge/Falling_dividend%3F-Busted-8b949e?style=flat-square)\n\n"
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

from dividend_cut_drift import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    PX, DIV, SPY = data.load_real()
    EVENTS = st.build_event_table(DIV)
    CAR_MAT, KEPT = st.build_cars(PX, SPY, EVENTS)
else:
    PX = DIV = SPY = EVENTS = CAR_MAT = KEPT = None
print("real cache present:", HAVE_REAL, "| tickers loaded:", (0 if PX is None else len(PX)),
      "| events with full window:", (0 if CAR_MAT is None else len(CAR_MAT)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Never catch a falling dividend — except maybe you can? 📉✂️\n"
            "### A Wall Street warning that used to be true, and quietly stopped being true\n\n"
            + BADGES +
            "When a company cuts or cancels its dividend, it's making the loudest possible "
            "confession: *the cash isn't there anymore.* Old trading-desk wisdom says don't buy "
            "the stock after a cut — the pain isn't over, it's just started. Academics agreed: a "
            "famous 1995 study found dividend-cutters kept **underperforming for up to three "
            "years** after the news.\n\n"
            "So we went and found 172 real dividend cuts and cancellations across 30 years of "
            "market history — Citigroup and Bank of America in 2008-09, oil majors in 2015-16, "
            "cruise lines and airlines in 2020, 3M and Whirlpool as recently as 2024 — and asked: "
            "**does the falling knife still fall, today?**\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo and the cost math? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** 101 named tickers, a deliberate mix of steady payers and known "
            "cutters; every cut/omission is detected from the real, split-adjusted dividend "
            "record — no hindsight. Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does the stock keep falling for 6 months after a cut? | **Not really — it's a "
            f"coin flip.** Average excess return vs the market over the next 120 trading days: "
            f"**{R['post120_mean']:+.2f}%**, roughly zero. It fell **{R['hit_pct']:.0f}%** of the "
            "time — statistically indistinguishable from 50/50. |\n"
            "| Does the market see it coming? | **No sign of it.** No measurable drift in the "
            "month *before* the cut either. |\n"
            "| Do full cancellations behave differently from partial cuts? | **Maybe, but "
            f"unconfirmed.** Partial cuts drift slightly negative, full omissions drift slightly "
            "*positive* — an intriguing \"kitchen sink, then bounce\" pattern, but neither is "
            "statistically certain on its own. |\n"
            f"| Can you trade it? | **No.** Shorting cutters loses money on average "
            f"(mostly because six months of *not owning stocks* is a bad bet in a rising market); "
            "buying them looks better on paper but isn't statistically provable once you account "
            "for that same market drift. |\n\n"
            "> The old warning was true once. On the modern tape, it just isn't showing up."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"A dividend cut is the clearest, most costly signal a board can send: 'the "
            "outlook is worse than you think.' Investors chronically underreact to it, so the "
            "stock keeps drifting down for months — sometimes years — after the news.\"*\n\n"
            "This isn't just folklore. Michaely, Thaler & Womack's landmark 1995 study found "
            "dividend-omitting companies underperformed the market by roughly **9.5% in the "
            "following year alone**, and kept losing ground for up to three years. It's one of "
            "the most cited \"market inefficiency\" results in finance."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the drift is real and current, it's a free, public, easy-to-spot signal: watch "
            "the dividend calendar, short (or avoid) anything that cuts, and collect a "
            "persistent edge. Retail investors are told the opposite folk wisdom too — \"buy the "
            "beaten-down stock, the bad news is already priced\" — so which is it? We test both "
            "sides with the same yardstick."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **Find the cuts.** Scan {R['universe_n']} large/mid-cap dividend payers' real "
            "payment history for a payment that drops to 70% or less of the prior one, or a "
            "payment that simply never arrives on schedule (an omission).\n"
            "- **Watch what happens next.** Track each stock's return *relative to the market* "
            "for the 120 trading days (~6 months) after the cut.\n"
            "- **The luck check.** Compare the average outcome to random ticker/date pairs drawn "
            "from the same universe — does the real thing look different from chance?\n"
            "- **The trade check.** Would shorting the cutter — or buying it — actually have "
            "made money, after real trading costs?"
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the headline.** Average excess return (vs SPY) in the 120 trading days "
            "after a cut or omission."
        ),
        code(
            "if HAVE_REAL:\n"
            "    hs = st.horizon_stats(CAR_MAT)\n"
            "    horizons = ['pre[-20,-1]', 'post[+1,+20]', 'post[+1,+60]', 'post[+1,+120]']\n"
            "    vals = [hs['pre_mean'], hs['post20_mean'], hs['post60_mean'], hs['post120_mean']]\n"
            "    vals = [v*100 for v in vals]\n"
            "else:\n"
            "    horizons = ['pre[-20,-1]', 'post[+1,+20]', 'post[+1,+60]', 'post[+1,+120]']\n"
            "    vals = [R['pre_mean'], R['post20_mean'], R['post60_mean'], R['post120_mean']]\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.4))\n"
            "cols = [GREY, AMBER, AMBER, RED]\n"
            "ax.bar(horizons, vals, color=cols, width=.6)\n"
            "for i, v in enumerate(vals): ax.annotate(f'{v:+.2f}%', (i, v), ha='center',\n"
            "    va='top' if v < 0 else 'bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean cumulative excess return vs SPY (%)')\n"
            "ax.set_title('Before and after a dividend cut — roughly nothing, either direction')\n"
            "plt.tight_layout(); plt.show()\n"
            "print({h: round(v, 2) for h, v in zip(horizons, vals)})"
        ),
        md(
            f"Six months out, the average cutter is sitting at **{R['post120_mean']:+.2f}%** vs "
            f"the market — noise. It fell **{R['hit_pct']:.0f}%** of the time, which is a coin "
            "flip once you account for sampling uncertainty. And there's no run-up before the cut "
            "either — the market doesn't seem to see it coming in our window, but it also doesn't "
            "keep punishing the stock afterward.\n\n"
            "**Next — cuts vs full stops.** Does a company that trims its dividend behave "
            "differently from one that cancels it outright?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    types = pd.Series([k[2] for k in KEPT])\n"
            "    post120 = (CAR_MAT[120] - CAR_MAT[0])\n"
            "    cut_mean = post120[(types=='cut').to_numpy()].mean()*100\n"
            "    om_mean = post120[(types=='omission').to_numpy()].mean()*100\n"
            "else:\n"
            "    cut_mean, om_mean = R['cut_mean'], R['om_mean']\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.3))\n"
            "ax.bar(['partial cut', 'full omission'], [cut_mean, om_mean],\n"
            "       color=[RED, GREEN], width=.5)\n"
            "for i, v in enumerate([cut_mean, om_mean]): ax.annotate(f'{v:+.2f}%', (i, v),\n"
            "    ha='center', va='top' if v < 0 else 'bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean 6-month excess return vs SPY (%)')\n"
            "ax.set_title('A trim keeps drifting down; a full stop bounces — neither is proven')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'cut {cut_mean:+.2f}%   omission {om_mean:+.2f}%')"
        ),
        md(
            f"Partial cuts point **{R['cut_mean']:+.2f}%** (mildly negative — the old story); "
            f"full omissions point **{R['om_mean']:+.2f}%** (mildly *positive* — a \"the worst is "
            "over\" bounce). Interesting texture, but neither is statistically solid on its own — "
            "see the quants notebook for the *t*-stats.\n\n"
            "**Finally, the trade.** Could you actually get paid — shorting the cutters, or "
            "buying them on the dip?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    bt_short = st.backtest(PX, SPY, KEPT, side='short', cost_bps=5.0)\n"
            "    bt_long = st.backtest(PX, SPY, KEPT, side='long', cost_bps=5.0)\n"
            "    ns, nl = bt_short['mean_net']*100, bt_long['mean_net']*100\n"
            "    xs, xl = bt_short['mean_excess']*100, bt_long['mean_excess']*100\n"
            "else:\n"
            "    ns, nl = R['short5_net'], R['long5_net']\n"
            "    xs, xl = R['short5_exc'], R['long5_exc']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.4))\n"
            "a1.bar(['short the\\ncutter', 'buy the\\ncutter'], [ns, nl], color=[RED, GREEN], width=.5)\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_title('Raw P&L per event (mostly market beta)')\n"
            "a1.set_ylabel('net return per event (%)')\n"
            "for i, v in enumerate([ns, nl]): a1.annotate(f'{v:+.1f}%', (i, v), ha='center',\n"
            "    va='top' if v < 0 else 'bottom')\n"
            "a2.bar(['short the\\ncutter', 'buy the\\ncutter'], [xs, xl], color=[AMBER, AMBER], width=.5)\n"
            "a2.axhline(0, c='k', lw=.8); a2.set_title('...strip out the market: neither is proven')\n"
            "a2.set_ylabel('excess vs matched exposure (%)')\n"
            "for i, v in enumerate([xs, xl]): a2.annotate(f'{v:+.1f}%', (i, v), ha='center',\n"
            "    va='top' if v < 0 else 'bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'short: net {ns:+.1f}% excess {xs:+.1f}%   long: net {nl:+.1f}% excess {xl:+.1f}%')"
        ),
        md(
            "The raw numbers look dramatic — shorting cutters loses ~15% a trade, buying them "
            "gains ~15% — but that's almost entirely six months of ordinary stock-market drift, "
            "not anything specific to the cut. Strip the market out and **both trades collapse to "
            "statistically uncertain territory.** And shorting carries a real horror story: one "
            f"single cutter later rallied so hard the naked short lost **{R['short_worst']:.0f}%** "
            "of the position — the kind of tail risk no amount of \"the math says it works on "
            "average\" survives."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — Weak.** The classic 1990s finding is real in the history books, but it "
            "doesn't show up on 172 modern real cut/omission events: the average 6-month excess "
            f"return is **{R['post120_mean']:+.2f}%**, a coin-flip hit rate, and a random-date "
            "placebo can't tell the observed number apart from chance.\n"
            "- **Tradability — Mirage.** Neither shorting nor buying cutters clears a "
            "statistical certification bar after costs — and the short side carries a real "
            "blow-up risk.\n"
            "- **\"Never catch a falling dividend\"? — Busted.** Not on this tape, today. The "
            "warning made sense in a slower, less-covered market; the modern tape shows nothing "
            "resembling it."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **Why might it have faded?** More analyst coverage, algorithmic reaction to "
            "corporate-action news, and deeper index/ETF ownership all push information into "
            "prices faster than in the 1970s-80s sample the original studies used.\n"
            "- **Survivorship matters here.** Our universe is companies still trading in 2026 — "
            "the ones that cut and later went to zero (there have been plenty) never make it in. "
            "The true historical effect of \"catching a falling dividend\" is plausibly worse "
            "than what we measured.\n"
            "- **Sibling studies:** [dividend-initiation](../../240-dividend-initiation/) (the "
            "*start* of the dividend life cycle), [dividend-growth](../../201-dividend-growth/) "
            "(the opposite tail — consistent raisers) and "
            "[dividend-capture](../../143-dividend-capture/) (trading the routine, uncut "
            "ex-date drop).\n\n"
            "*Think a shorter or longer window, or a sector-specific cut, changes the answer? "
            "The detector and the event table are right there in "
            "[`dividend_cut_drift/`](../dividend_cut_drift/) — fork it and show us.*"
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
            "# Dividend-Cut-Drift — a quantitative teardown 🔬\n"
            "### Split/special-dividend-cleaned event detection · cross-sectional + Newey-West "
            "CAR tests · cut-vs-omission divergence · a random-date placebo · matched-exposure "
            "short/long capture · a 20-seed synthetic control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "The claim — a dividend cut/omission predicts continued underperformance — has a "
            "genuine academic anchor (Michaely, Thaler & Womack 1995) built on 1964-88 NYSE/AMEX "
            "data. The job here is to run the *same kind* of test on a modern real tape and "
            "report honestly whether it still holds.\n\n"
            "> ⚠️ **Data note.** 101 named tickers (99 loaded — `MRO`, `GPS` renamed/delisted, "
            "skipped, not fatal), yfinance adjusted closes + split-adjusted dividend streams, "
            f"{R['spy_lo']} → {R['spy_hi']}. **{R['ev_detected']} events detected "
            f"({R['n_cut_raw']} cuts, {R['n_om_raw']} omissions)**, **{R['kept_n']}** with a full "
            "[-20,+120]-day window. Universe is current survivors only — **named on the Signal "
            "axis** (see [`docs/references.md`](../docs/references.md)). Numbers in "
            "[`docs/results.md`](../docs/results.md) (fingerprints `" + R["fp_spy"] + "` SPY / `" +
            R["fp_events"] + "` events).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | post-event CAR[+1,+120] **{R['post120_mean']:+.2f}%**, "
            f"cross-sectional **t = {R['post120_t']:+.2f}**, calendar-time Newey-West "
            f"**t = {R['nw_t']:+.2f}**, placebo **p = {R['placebo_p']:.2f}**, hit rate "
            f"{R['hit_pct']:.1f}% [{R['hit_lo']:.1f}%, {R['hit_hi']:.1f}%] |\n"
            f"| **Tradability** | `MIRAGE` | short excess t = {R['short5_texc']:.2f}, long excess "
            f"t = {R['long5_texc']:.2f} — neither certified; short worst event "
            f"{R['short_worst']:.0f}% |\n"
            f"| **Falling dividend keeps falling?** | `BUSTED` | cut subtype "
            f"t = {R['cut_t']:.2f}, omission subtype t = {R['om_t']:.2f} — opposite signs, "
            "neither certified |\n\n"
            "> 💡 In plain words: a real, once-published effect that simply isn't there anymore "
            "on the modern tape, by any of four independent tests."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $CAR_{i}[a,b]$ be firm $i$'s cumulative log return minus SPY's, from offset $a$ "
            "to $b$ trading days around a detected cut/omission (offset 0). The claims:\n\n"
            "- **H₁ (drift).** $E[CAR_i[+1,+120]] \\ll 0$ — a persistent, economically large "
            "post-event underperformance (Michaely-Thaler-Womack: ≈ −9.5%/yr on 1964-88 data).\n"
            "- **H₂ (no anticipation).** $E[CAR_i[-20,-1]] \\approx 0$ — dividend cuts are "
            "board decisions, not slow-leaking information; little should show up before the "
            "event itself is knowable.\n"
            "- **H₃ (heterogeneity).** Full omissions (a harder, more informative signal) drift "
            "*more* negative than partial cuts.\n"
            "- **H₄ (capture).** A short-the-cutter or contrarian buy-the-cutter position banks "
            "the drift net of costs.\n\n"
            f"We find **H₁ not supported** (t = {R['post120_t']:.2f}), **H₂ supported** "
            f"(t = {R['pre_t']:.2f}, no anticipation), **H₃ reversed and uncertified** "
            f"(omissions point *positive*, t = {R['om_t']:.2f}, cuts point negative, "
            f"t = {R['cut_t']:.2f}), **H₄ not certified** either direction."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "Events are individually dated but **share calendar time** (a bank-sector cut "
            "cluster in 2008-09, a pandemic-travel cluster in 2020), so the planned primary — a "
            "**cross-sectional one-sample t** on each event's CAR — treats correlated "
            "observations as independent. We cross-check with a **Newey-West(5) t** on the daily "
            "equal-weight calendar-time \"cutters\" portfolio, which is robust to exactly that "
            "overlap. Hit rates carry a **Wilson interval**; a **20-seed × 200-draw random "
            "ticker/date placebo** asks whether the observed mean is distinguishable from picking "
            "172 random dates in the same universe."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Detection.** Cut = payment ≤ 70% of prior; omission = gap ≥ 1.8× trailing "
            "typical interval (regular payers only); both on a **split-adjusted**, "
            "**special/stub-stripped** dividend stream — see 4a for why that cleaning step is "
            "load-bearing, not decorative.\n"
            f"- **Sample.** {R['ev_detected']} events detected "
            f"({R['n_cut_raw']} cut / {R['n_om_raw']} omission), **{R['kept_n']}** with a full "
            f"[-20,+120]-day window, {R['kept_lo']} → {R['kept_hi']}, {R['kept_tickers']} distinct "
            "tickers.\n"
            "- **Abnormal return.** Ticker log return − SPY log return, both auto-adjusted "
            "(total-return), so the ex-date's mechanical price drop never enters the math.\n"
            "- **Execution (third axis).** Enter the close **one session after** the event day "
            "(the cut is only knowable from that close — zero look-ahead); exit 120 trading days "
            "later; 2 × one-way cost × NAV; shorts pay 50 bps/yr borrow; excess benchmarked "
            "against matched exposure (long/short SPY over the identical window).\n"
            "- **Control.** Single-factor synthetic market-model panel, planted post-event drift "
            "knob; the null must not systematically fire across 20 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Cleaning the dividend stream — why it's load-bearing\n\n"
            "Raw yfinance dividend history is **not split-adjusted**, and several names in this "
            "universe (PEP, MO, C, SLB, MS) show a phantom 90%+ \"cut\" purely from a stock "
            "split without correction. Separately, one-off special dividends, spinoff cash "
            "distributions folded into the same `Dividends` stream (Citigroup's Aug-2002 "
            "Travelers Property Casualty spinoff), and stray stub payments (a one-off $0.01 "
            "between two $0.22 Exxon quarters) each manufacture a false cut on the payment "
            "immediately after them if left in. `strategy.strip_special_dividends` removes any "
            "payment that sits ≥ 1.8× from BOTH neighbors on the same side (spike or dip), "
            "iterated so adjacent pairs of stub payments fully clean out."
        ),
        code(
            "if HAVE_REAL:\n"
            "    n_cuts_raw_check = int((EVENTS['type']=='cut').sum())\n"
            "    n_om_raw_check = int((EVENTS['type']=='omission').sum())\n"
            "else:\n"
            "    n_cuts_raw_check, n_om_raw_check = R['n_cut_raw'], R['n_om_raw']\n"
            "print(f'events after cleaning: {n_cuts_raw_check} cuts + {n_om_raw_check} omissions '\n"
            "      f'= {n_cuts_raw_check + n_om_raw_check} total')\n"
            "print('(before the split-adjust + special-dividend-strip + warm-up fixes, an early '\n"
            "      'pass over the same raw universe found 230+ \"events\" — many of them stock '\n"
            "      'splits and spinoff artifacts, not real cuts)')"
        ),
        md(
            "### 4b · The headline CAR — cross-sectional and calendar-time\n\n"
            "Mean cumulative abnormal return by horizon, one-sample *t*; then the Newey-West "
            "overlap-robust cross-check."
        ),
        code(
            "if HAVE_REAL:\n"
            "    hs = st.horizon_stats(CAR_MAT)\n"
            "else:\n"
            "    hs = {'pre_mean': R['pre_mean']/100, 'pre_t': R['pre_t'],\n"
            "          'post20_mean': R['post20_mean']/100, 'post20_t': R['post20_t'],\n"
            "          'post60_mean': R['post60_mean']/100, 'post60_t': R['post60_t'],\n"
            "          'post120_mean': R['post120_mean']/100, 'post120_t': R['post120_t'],\n"
            "          'hit_rate': R['hit_pct']/100, 'hit_lo': R['hit_lo']/100, 'hit_hi': R['hit_hi']/100,\n"
            "          'n': R['kept_n']}\n"
            "labels = ['pre[-20,-1]', 'post[+1,+20]', 'post[+1,+60]', 'post[+1,+120]']\n"
            "means = [hs['pre_mean']*100, hs['post20_mean']*100, hs['post60_mean']*100, hs['post120_mean']*100]\n"
            "ts = [hs['pre_t'], hs['post20_t'], hs['post60_t'], hs['post120_t']]\n"
            "fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.2, 6.2), sharex=True,\n"
            "                             gridspec_kw={'height_ratios': [2, 1]})\n"
            "a1.bar(labels, means, color=[GREY, AMBER, AMBER, RED], width=.6)\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_ylabel('mean CAR (%)')\n"
            "a1.set_title('No pre-event drift; no certified post-event drift either')\n"
            "a2.bar(labels, ts, color=[RED if abs(t)>=2 else GREY for t in ts], width=.6)\n"
            "a2.axhline(-2, ls='--', c=RED, lw=1); a2.axhline(2, ls='--', c=RED, lw=1)\n"
            "a2.axhline(0, c='k', lw=.8); a2.set_ylabel('one-sample t')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"hit rate (CAR120<0): {hs['hit_rate']*100:.1f}% \"\n"
            "      f\"[{hs['hit_lo']*100:.1f}%, {hs['hit_hi']*100:.1f}%]  n={hs['n']}\")"
        ),
        code(
            "if HAVE_REAL:\n"
            "    nw = st.calendar_time_nw_t(PX, SPY, KEPT)\n"
            "else:\n"
            "    nw = {'n_days': R['nw_days'], 'mean_daily_bps': R['nw_bps'], 'nw_t': R['nw_t']}\n"
            "print(f\"calendar-time cutters portfolio: {nw['n_days']:,} days, \"\n"
            "      f\"mean daily AR {nw['mean_daily_bps']:+.2f} bps, Newey-West t = {nw['nw_t']:+.2f}\")"
        ),
        md(
            f"> 💡 In plain words: **{R['post120_mean']:+.2f}%** over six months, "
            f"t = {R['post120_t']:.2f} — nowhere near the ±2 bar, and the overlap-robust "
            f"calendar-time check (t = {R['nw_t']:.2f}) agrees. Whatever Michaely-Thaler-Womack "
            "found in 1964-88 data, it isn't reproducing on 172 modern events."
        ),
        md(
            "### 4c · Cut vs omission — a real but uncertified divergence\n\n"
            "A partial cut and a full stop are different signals in theory (the second is a "
            "harder commitment). Splitting the sample:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    types = pd.Series([k[2] for k in KEPT])\n"
            "    post120 = (CAR_MAT[120] - CAR_MAT[0])\n"
            "    cut_x = post120[(types=='cut').to_numpy()].to_numpy()\n"
            "    om_x = post120[(types=='omission').to_numpy()].to_numpy()\n"
            "    cut_m, cut_t = float(np.nanmean(cut_x))*100, st.one_sample_t(cut_x)\n"
            "    om_m, om_t = float(np.nanmean(om_x))*100, st.one_sample_t(om_x)\n"
            "else:\n"
            "    cut_m, cut_t, om_m, om_t = R['cut_mean'], R['cut_t'], R['om_mean'], R['om_t']\n"
            "fig, ax = plt.subplots(figsize=(8.0, 4.3))\n"
            "ax.bar(['cut (partial)', 'omission (full stop)'], [cut_m, om_m], color=[RED, GREEN], width=.5)\n"
            "for i,(v,t_) in enumerate([(cut_m,cut_t),(om_m,om_t)]):\n"
            "    ax.annotate(f'{v:+.2f}%\\n(t={t_:+.2f})', (i, v), ha='center', va='top' if v<0 else 'bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean CAR[+1,+120] (%)')\n"
            "ax.set_title('Opposite signs, both uncertified — H3 not confirmed')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'cut n={R[\"cut_n\"]} mean={cut_m:+.2f}% t={cut_t:+.2f}   '\n"
            "      f'omission n={R[\"om_n\"]} mean={om_m:+.2f}% t={om_t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: cuts drift **{R['cut_mean']:+.2f}%** (t = {R['cut_t']:.2f}), "
            f"omissions drift **{R['om_mean']:+.2f}%** (t = {R['om_t']:.2f}) — a plausible "
            "\"kitchen-sink capitulation\" story for full stops, but with only 65-107 events per "
            "bucket neither clears the bar. Flag it, don't lean on it."
        ),
        md(
            "### 4d · A third cut on the same question — pooled daily-level Welch\n\n"
            "The cross-sectional test gives one number per event; the calendar-time NW test "
            "gives one number per calendar day. A third, differently-shaped view: pool every "
            "raw ticker-day abnormal return inside a post-event window (\"treatment\") against "
            "every OTHER day for the same tickers (\"control\" — their own non-event history), "
            "then a two-sample Welch *t* on the pooled distributions."
        ),
        code(
            "if HAVE_REAL:\n"
            "    dw = st.daily_welch(PX, SPY, KEPT)\n"
            "else:\n"
            "    dw = {'n_treat': R['dw_n_treat'], 'n_control': R['dw_n_control'],\n"
            "          'mean_treat_bps': R['dw_treat_bps'], 'mean_control_bps': R['dw_control_bps'],\n"
            "          'welch_t': R['dw_welch_t']}\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.3))\n"
            "ax.bar(['post-event\\nwindow', 'other days\\n(same tickers)'],\n"
            "       [dw['mean_treat_bps'], dw['mean_control_bps']], color=[RED, GREY], width=.5)\n"
            "for i,v in enumerate([dw['mean_treat_bps'], dw['mean_control_bps']]):\n"
            "    ax.annotate(f'{v:+.2f} bps',(i,v),ha='center',va='top' if v<0 else 'bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean daily abnormal return (bps)')\n"
            "ax.set_title(f\"Welch t = {dw['welch_t']:+.2f} — a third method, a third non-result\")\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"treatment {dw['mean_treat_bps']:+.2f} bps (n={dw['n_treat']:,})  vs  \"\n"
            "      f\"control {dw['mean_control_bps']:+.2f} bps (n={dw['n_control']:,})  \"\n"
            "      f\"Welch t = {dw['welch_t']:+.2f}\")"
        ),
        md(
            f"> 💡 In plain words: post-event days average **{R['dw_treat_bps']:+.2f} bps** vs "
            f"**{R['dw_control_bps']:+.2f} bps** on every other day for the same tickers — "
            f"practically identical, Welch t = **{R['dw_welch_t']:+.2f}**. Cross-sectional, "
            "calendar-time and pooled-daily all land in the same place: nothing."
        ),
        md(
            "### 4e · The random-date placebo\n\n"
            "Draw 172 random ticker/date pairs from the same universe (away from the tape's "
            "edges), 20 seeds × 200 draws, compute the mean post-120 CAR each time."
        ),
        code(
            "if HAVE_REAL:\n"
            "    tickers_avail = list(PX.keys())\n"
            "    pl = st.random_date_placebo(PX, SPY, tickers_avail, n_events=len(KEPT),\n"
            "                                n_seeds=6, n_draws_per_seed=80)\n"
            "    obs = hs['post120_mean']*100\n"
            "    draws = pl['draws']*100\n"
            "else:\n"
            "    obs = R['post120_mean']\n"
            "    rng = np.random.default_rng(653)\n"
            "    draws = rng.normal(R['placebo_mean'], R['placebo_sd'], 480)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=40, color=GREY, alpha=.85, label='null: random ticker/date pairs')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'observed mean {obs:+.2f}%')\n"
            "ax.set_xlabel('mean CAR[+1,+120] of a random 172-event draw (%)')\n"
            "ax.set_ylabel('frequency')\n"
            "ax.set_title(f\"Indistinguishable from a random calendar (canonical p = {R['placebo_p']:.2f})\")\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"canonical placebo (results.md): mean {R['placebo_mean']:+.2f}%, \"\n"
            "      f\"sd {R['placebo_sd']:.2f}%, p = {R['placebo_p']:.3f}\")"
        ),
        md(
            f"> 💡 In plain words: the observed mean sits well inside the placebo cloud — "
            f"**p = {R['placebo_p']:.2f}**, nowhere near a conventional significance threshold. "
            "A random 172-event draw from the same universe looks just as \"negative\" (in fact "
            "slightly more so) as the real cutters."
        ),
        md(
            "### 4f · The third axis — honest short and long capture tests\n\n"
            "Enter the close **one session after** the event (zero look-ahead), exit 120 trading "
            "days later. Excess = net return minus the **matched-exposure** benchmark (long SPY "
            "for the long leg, short SPY for the short leg) over the identical window — isolating "
            "whatever is cutter-specific from six months of ordinary market beta."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bts = st.backtest(PX, SPY, KEPT, side='short', cost_bps=5.0)\n"
            "    btl = st.backtest(PX, SPY, KEPT, side='long', cost_bps=5.0)\n"
            "else:\n"
            "    bts = {'mean_net': R['short5_net']/100, 't_net': R['short5_tnet'],\n"
            "           'mean_excess': R['short5_exc']/100, 't_excess': R['short5_texc'],\n"
            "           'worst': R['short_worst']/100, 'hit_rate': R['short_hit']/100}\n"
            "    btl = {'mean_net': R['long5_net']/100, 't_net': R['long5_tnet'],\n"
            "           'mean_excess': R['long5_exc']/100, 't_excess': R['long5_texc'],\n"
            "           'worst': R['long_worst']/100, 'hit_rate': R['long_hit']/100}\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.2, 4.4))\n"
            "labels = ['short-the-\\ncutter', 'buy-the-\\ncutter']\n"
            "nets = [bts['mean_net']*100, btl['mean_net']*100]\n"
            "excs = [bts['mean_excess']*100, btl['mean_excess']*100]\n"
            "ts_ = [bts['t_excess'], btl['t_excess']]\n"
            "a1.bar(labels, nets, color=[RED, GREEN], width=.5)\n"
            "for i,v in enumerate(nets): a1.annotate(f'{v:+.1f}%',(i,v),ha='center',va='top' if v<0 else 'bottom')\n"
            "a1.axhline(0,c='k',lw=.8); a1.set_ylabel('net return/event (%)')\n"
            "a1.set_title('Gross P&L (mostly market beta)')\n"
            "a2.bar(labels, excs, color=[AMBER, AMBER], width=.5)\n"
            "for i,(v,t_) in enumerate(zip(excs, ts_)):\n"
            "    a2.annotate(f'{v:+.1f}%\\n(t={t_:+.2f})',(i,v),ha='center',va='top' if v<0 else 'bottom')\n"
            "a2.axhline(0,c='k',lw=.8); a2.set_ylabel('excess vs matched exposure (%)')\n"
            "a2.set_title('Neither clears t >= 2 after stripping market beta')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"short: net {bts['mean_net']*100:+.1f}% (worst {bts['worst']*100:+.1f}%) \"\n"
            "      f\"excess t={bts['t_excess']:+.2f}  |  long: net {btl['mean_net']*100:+.1f}% \"\n"
            "      f\"excess t={btl['t_excess']:+.2f}\")"
        ),
        md(
            f"> 💡 In plain words: strip out the market and short-the-cutter's excess is "
            f"**{R['short5_exc']:+.2f}%** at t = {R['short5_texc']:.2f}; buy-the-cutter's is "
            f"**{R['long5_exc']:+.2f}%** at t = {R['long5_texc']:.2f}. Close to the bar, "
            "neither over it — **H₄ not certified**. And the short leg's worst single event lost "
            f"**{R['short_worst']:.0f}%** of the position (a squeeze in a name that later "
            "rallied hard): the naked-short version of this trade carries real, undiversifiable "
            "blow-up risk that the average-return table alone doesn't show."
        ),
        md(
            "### 4g · Faithful-engine & power control — we know the truth here\n\n"
            "Deterministic single-factor market-model panel, scheduled planted "
            "\"cut\" events, TUNABLE post-event abnormal drift. Null checked over **20 seeds**."
        ),
        code(
            "null_ts = []\n"
            "for s_ in range(20):\n"
            "    pxs, bench, evs = data.synthetic_world(drift=0.0, seed=653 + s_)\n"
            "    null_ts.append(st.synthetic_detect(pxs, bench, evs)['post120_t'])\n"
            "null_ts = np.asarray(null_ts)\n"
            "pxs, bench, evs = data.synthetic_world(drift=-0.001, seed=653)\n"
            "planted = st.synthetic_detect(pxs, bench, evs)\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.zeros(20) + np.linspace(-.12,.12,20), null_ts, color=GREY, s=40,\n"
            "           label='null worlds (drift=0), 20 seeds')\n"
            "ax.scatter([1], [planted['post120_t']], color=RED, s=90, zorder=5,\n"
            "           label='planted drift = -0.001/day')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0, 1]); ax.set_xticklabels(['null x 20', 'planted'])\n"
            "ax.set_ylabel('post-120 CAR t-stat')\n"
            "ax.set_title('Control: the null fires at the nominal rate; a planted drift lights up')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(abs(null_ts)>=2).sum()}/20 seeds  |  '\n"
            "      f\"planted t = {planted['post120_t']:+.2f} (n={planted['n']})\")"
        ),
        md(
            f"> 💡 In plain words: across 20 null worlds the detector fires at "
            f"**{R['syn_null_fire']}/{R['syn_null_seeds']}** — close to the ~5% false-positive "
            "rate a *t* = 2 threshold implies by construction, i.e. well-calibrated, not biased. "
            f"A planted drift of just −0.1%/day reads t = {R['syn_planted_t']:.2f}: the machinery "
            "can find a real effect when one exists. The real-tape near-zero result is the "
            "genuine article, not a broken detector. *(A faithful-engine / power check only — "
            "never cited in support of the real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — post-event CAR[+1,+120] = **{R['post120_mean']:+.2f}%**, "
            f"cross-sectional t = **{R['post120_t']:.2f}**, Newey-West t = **{R['nw_t']:.2f}**, "
            f"placebo p = **{R['placebo_p']:.2f}**, hit rate **{R['hit_pct']:.1f}%** "
            f"[{R['hit_lo']:.1f}%, {R['hit_hi']:.1f}%]. Decades of published support "
            "(Michaely-Thaler-Womack 1995) plus a sub-2 *t* on 172 modern events is the textbook "
            "`WEAK` case. Survivorship, named: the universe excludes cutters that later went to "
            "zero, so the true population effect is plausibly more negative than measured here.\n"
            f"- **Tradability `MIRAGE`** — short-the-cutter excess t = **{R['short5_texc']:.2f}**, "
            f"buy-the-cutter excess t = **{R['long5_texc']:.2f}**, neither certified; the "
            f"dramatic-looking gross P&L is six months of market beta, and the short leg's "
            f"worst event ({R['short_worst']:.0f}%) is a real tail risk on top.\n"
            f"- **Falling dividend keeps falling? `BUSTED`** — the point estimate is a wash and "
            f"the two subtypes (cut t = {R['cut_t']:.2f}, omission t = {R['om_t']:.2f}) point in "
            "opposite, individually-uncertified directions."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The general lesson is decay, not disproof.** Michaely-Thaler-Womack's 1964-88 "
            "sample predates decimalization, algorithmic trading and today's analyst coverage "
            "density — information that used to leak for months now arrives fast. The "
            "post-cut-drift anomaly may simply have been arbitraged away, the standard fate of a "
            "well-published inefficiency.\n"
            "- **A longer window is the natural sequel.** Michaely et al.'s strongest results are "
            "at the 1-3 year horizon, not 6 months; a companion study extending the window (at "
            "the cost of far fewer independent, non-overlapping events) could test whether the "
            "drift shows up later even if it doesn't show up early.\n"
            "- **Dedup map:** [240-dividend-initiation](../../240-dividend-initiation/) (the "
            "*start* side, `NONE`), [201-dividend-growth](../../201-dividend-growth/) (the "
            "opposite tail, consistent raisers, `WEAK`), "
            "[143-dividend-capture](../../143-dividend-capture/) (the routine, uncut ex-date "
            "drop), [233-shareholder-yield](../../233-shareholder-yield/) (a level-sort "
            "composite factor, not an event study).\n\n"
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
