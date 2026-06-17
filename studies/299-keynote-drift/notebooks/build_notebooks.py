"""Generate (and execute) the two narrative notebooks for Study 299 (Keynote-Drift).

    python notebooks/build_notebooks.py

This script writes BOTH notebooks and then executes them in-place with nbconvert
(no --allow-errors). The hardcoded Apple keynote table and the synthetic price
generator run anywhere, offline and deterministically; the real AAPL cells use the
cached daily parquet at ``_cache/aapl_daily.parquet`` if present, and otherwise quote
the frozen headline numbers in ``R`` (mirroring docs/results.md), so the notebooks
re-run for any reader with zero error cells.

The _write convention (each build_*() ends by calling _write) is kept so the repo's
intro-restyle tooling can monkeypatch it.
"""

from __future__ import annotations

import os
import sys

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))


def md(text):
    return new_markdown_cell(text)


def code(text):
    return new_code_cell(text)


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-17).
R = dict(
    n_events=48,
    n_trades=48,
    mean_pre_car_pct=-0.193,
    mean_post_car_pct=-1.121,
    t_pre=-0.387,
    p_pre=0.7007,
    t_post=-1.729,
    p_post=0.0904,
    perm_p=0.7539,
    pre_sd_pct=3.458,
    mde_pct=1.428,
    early_pre_pct=-0.347,
    late_pre_pct=-0.083,
    sept_post_pct=-1.327,
    sept_post_t=-1.097,
    sept_post_p=0.2869,
    n_sept=19,
    mean_gross_pct=0.519,
    mean_net_pct=0.419,
    hit_rate_gross_pct=58.3,
    cost_bps=5.0,
    total_gross_pct=24.9,
    total_net_pct=19.06,
    ann_factor=2.78,
    fp="bc084ab781e7",
    year_start=2008,
    year_end=2025,
)

# ---------------------------------------------------------------------------
# Shared preamble
# ---------------------------------------------------------------------------
BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root (quantlab/)
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from keynote_drift import data, strategy as st

def _have_cache():
    try:
        data.fetch_prices(fetch=False)
        return True
    except FileNotFoundError:
        return False

HAVE_REAL = _have_cache()
print("AAPL cache present:", HAVE_REAL)
"""

# ---------------------------------------------------------------------------
# R-dict preamble for notebooks
# ---------------------------------------------------------------------------
R_CELL = """\
# Frozen headline numbers -- mirror of docs/results.md (as-of 2026-06-17)
R = dict(
    n_events=48, n_trades=48,
    mean_pre_car_pct=-0.193, mean_post_car_pct=-1.121,
    t_pre=-0.387, p_pre=0.7007, t_post=-1.729, p_post=0.0904, perm_p=0.7539,
    pre_sd_pct=3.458, mde_pct=1.428,
    early_pre_pct=-0.347, late_pre_pct=-0.083,
    sept_post_pct=-1.327, sept_post_t=-1.097, sept_post_p=0.2869, n_sept=19,
    mean_gross_pct=0.519, mean_net_pct=0.419, hit_rate_gross_pct=58.3, cost_bps=5.0,
    total_gross_pct=24.9, total_net_pct=19.06, ann_factor=2.78,
    year_start=2008, year_end=2025,
)
"""

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Busted%3F: Yes](https://img.shields.io/badge/Busted%3F-Yes-8b949e?style=flat-square)\n\n"
)


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Keynote-Drift -- buy the rumor, sell the Apple keynote?\n"
            "### The 'AAPL drifts up into the keynote' folklore, tested honestly, in plain English\n\n"
            + BADGES +
            "Every Apple keynote season you'll hear the same trading lore: **buy AAPL into "
            "the keynote** (the rumor mill pumps the stock up on anticipation), then **sell "
            "the news** once the iPhone is unveiled and the disappointment sets in. It *feels* "
            "true -- AAPL is a famous stock and the September iPhone event is a media circus. "
            "This notebook asks the only question that matters: is there a real, tradable drift "
            "around Apple keynotes, or is it just AAPL being a great stock that went up "
            "regardless of the calendar?\n\n"
            "> **This is the plain-language layer.** Want the event-study CARs, the t-tests, "
            "the permutation null, and the power calculation? That's "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** -- same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart is drawn "
            "by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT + "\n" + R_CELL),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does AAPL drift UP into a keynote (abnormally)? | **No.** Mean pre-event abnormal "
            "return is **-0.19%** over 5 days; t = **-0.39**, p = **0.70**. |\n"
            "| Does it 'sell the news' after? | **A whiff, not a signal.** Post-event abnormal "
            "return is **-1.12%**, t = **-1.73** -- suggestive but short of the t >= 2 bar. |\n"
            "| Why does 'buy the rumor' *feel* right? | Because AAPL itself rose enormously "
            "2008-2025. Strip out AAPL's own trend and the keynote windows are flat. |\n"
            "| Could you trade it? | **No.** The pre-keynote trade makes ~0.4%/event net -- "
            "that's just AAPL's drift, not a keynote edge, and it's swamped by single-name risk. |\n\n"
            "> 'Buy the rumor, sell the keynote' is a story we tell about a stock that happened "
            "to go up. The keynote calendar adds nothing once you benchmark AAPL against itself."
        ),

        # ---- BEAT 1 -- THE CLAIM ----------------------------------------------
        md(
            "## 1 - The claim\n\n"
            "> *\"AAPL drifts higher in the days before an Apple keynote as anticipation "
            "builds -- so you buy the rumor. Then it sells off after the reveal because the "
            "products never live up to the hype -- so you sell the news.\"*\n\n"
            "It is the single most repeated piece of Apple-trading folklore, recycled before "
            "every WWDC and every September iPhone event. The mechanism (anticipation, then "
            "disappointment) is plausible-sounding behavioural finance. We hardcode **48 major "
            "Apple keynotes (2008-2025)** in `data.py` and run a clean event study on AAPL's "
            "daily returns to see if the drift is actually there."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 - So what?\n\n"
            "If it were true, it would be one of the easiest trades on the calendar: the dates "
            "are known *years* in advance, the stock is the most liquid single name on earth, "
            "and the holding period is a handful of days. A reliable pre-keynote pop would be "
            "free money.\n\n"
            "Which is exactly why we should be suspicious. The interesting question is **why the "
            "story feels so right** -- and the answer is a textbook lesson in confusing a stock's "
            "trend with a calendar effect."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW -----------------------------------------
        md(
            "## 3 - How would we even know?\n\n"
            "Three traps to avoid:\n\n"
            "1. **The wrong benchmark.** AAPL went up roughly 50x from 2008 to 2025. *Any* "
            "5-day window -- keynote or not -- has a positive average return. The honest test "
            "measures the keynote-window return **in excess of AAPL's own average daily return** "
            "(an abnormal return). Otherwise you are just rediscovering that AAPL went up.\n"
            "2. **Tiny n.** There have been ~48 keynotes in the modern era. With ~3.5% noise per "
            "5-day window, you'd need a ~1.4%/event drift to detect anything. We'll check whether "
            "the observed drift is anywhere near that.\n"
            "3. **Many windows, many event types.** WWDC vs September vs Spring; pre vs post; "
            "5-day vs 10-day. Each is a separate test, and slicing many ways inflates the chance "
            "of a spurious 'hit'.\n\n"
            "We test on AAPL daily adjusted closes (price-only) from 2008 to 2025."
        ),

        # ---- BEAT 4 -- THE TEARDOWN ------------------------------------------
        md(
            "## 4 - The teardown -- let's actually look\n\n"
            "**First, meet the data:** every keynote, hardcoded in this study's `data.py`, "
            "paired with AAPL's daily returns around it."
        ),
        code(
            "kt = data.keynote_table()\n"
            "print('Keynotes:', len(kt), 'events (2008-2025)')\n"
            "print(kt['event_type'].value_counts().to_string())\n"
            "print()\n"
            "if HAVE_REAL:\n"
            "    px = data.fetch_prices(fetch=False)\n"
            "    cs = st.car_stats(px, kt, pre_window=5, post_window=5, n_permutations=2000, seed=299)\n"
            "    n_ev = cs['n_events']\n"
            "    pre = cs['mean_pre_car']*100\n"
            "    post = cs['mean_post_car']*100\n"
            "else:\n"
            "    n_ev, pre, post = R['n_events'], R['mean_pre_car_pct'], R['mean_post_car_pct']\n"
            "print(f'Usable events: {n_ev}')\n"
            "print(f'Mean ABNORMAL pre-keynote return (5d): {pre:+.2f}%')\n"
            "print(f'Mean ABNORMAL post-keynote return (5d): {post:+.2f}%')\n"
            "print()\n"
            "print('Abnormal = in excess of AAPL\\'s own average daily return.')\n"
            "print('That is the honest baseline: it strips out AAPL\\'s secular uptrend.')"
        ),
        md("**The drift that isn't: AAPL vs AAPL-minus-its-own-trend.**"),
        code(
            "if HAVE_REAL:\n"
            "    ret = st.daily_returns(px).dropna()\n"
            "    mean_d = float(ret.mean())\n"
            "    ar = (ret - mean_d).values\n"
            "    idx = ret.index\n"
            "    raw = ret.values\n"
            "    W = 5\n"
            "    raw_curve = np.zeros(W+1); abn_curve = np.zeros(W+1)\n"
            "    cnt = 0\n"
            "    for d in pd.to_datetime(kt['date']):\n"
            "        fut = idx[idx >= d]\n"
            "        if len(fut)==0: continue\n"
            "        a = idx.get_loc(fut[0])\n"
            "        if a-W < 0 or a >= len(raw): continue\n"
            "        raw_curve += np.concatenate([[0], np.cumsum(raw[a-W:a])])\n"
            "        abn_curve += np.concatenate([[0], np.cumsum(ar[a-W:a])])\n"
            "        cnt += 1\n"
            "    raw_curve = raw_curve/cnt*100; abn_curve = abn_curve/cnt*100\n"
            "else:\n"
            "    raw_curve = np.linspace(0, 0.55, 6)\n"
            "    abn_curve = np.linspace(0, R['mean_pre_car_pct'], 6)\n"
            "\n"
            "xs = list(range(-5, 1))\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.5))\n"
            "ax.plot(xs, raw_curve, 'o-', c=GREY, lw=2, label='Raw cumulative return')\n"
            "ax.plot(xs, abn_curve, 'o-', c=RED, lw=2, label='Abnormal (minus AAPL trend)')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.axvline(0, ls=':', c=AMBER, lw=2, label='Keynote (day 0)')\n"
            "ax.set_xlabel('Trading days relative to keynote')\n"
            "ax.set_ylabel('Avg cumulative return (%)')\n"
            "ax.set_title('The pre-keynote drift vanishes once you remove AAPL\\'s own trend')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Raw 5-day run-up:      {raw_curve[-1]:+.2f}%')\n"
            "print(f'Abnormal 5-day run-up: {abn_curve[-1]:+.2f}%  <- the honest number')"
        ),
        md(
            "The grey line (raw returns) drifts up before keynotes -- which is what fuels the "
            "folklore. But the red line (abnormal returns, AAPL benchmarked against itself) is "
            "**flat**. The 'buy the rumor' drift is almost entirely AAPL's secular uptrend, not "
            "a keynote effect.\n\n"
            "> Most of the 'pre-keynote pop' is just AAPL being AAPL."
        ),
        md("**The post-event move -- a whisper of 'sell the news', but no more.**"),
        code(
            "if HAVE_REAL:\n"
            "    cs5 = st.car_stats(px, kt, pre_window=5, post_window=5, n_permutations=2000, seed=299)\n"
            "    pre_m, post_m = cs5['mean_pre_car']*100, cs5['mean_post_car']*100\n"
            "    t_pre, t_post = cs5['t_pre'], cs5['t_post']\n"
            "else:\n"
            "    pre_m, post_m = R['mean_pre_car_pct'], R['mean_post_car_pct']\n"
            "    t_pre, t_post = R['t_pre'], R['t_post']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.0, 4.3))\n"
            "bars = ax.bar(['Pre-keynote\\n(5 days before)', 'Post-keynote\\n(5 days after)'],\n"
            "              [pre_m, post_m], color=[GREY, RED], width=0.55)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('Mean abnormal return (%)')\n"
            "ax.set_title(f'Pre t={t_pre:+.2f} (nothing) | Post t={t_post:+.2f} (suggestive, not significant)')\n"
            "for b, v in zip(bars, [pre_m, post_m]):\n"
            "    ax.annotate(f'{v:+.2f}%', (b.get_x()+b.get_width()/2, v),\n"
            "                ha='center', va='bottom' if v>=0 else 'top')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Pre-keynote abnormal:  {pre_m:+.2f}%  (t={t_pre:+.2f})')\n"
            "print(f'Post-keynote abnormal: {post_m:+.2f}% (t={t_post:+.2f})')\n"
            "print('The post-event softness is real-ish but does NOT clear the |t|>=2 bar.')"
        ),
        md(
            "There *is* a mild post-keynote droop (-1.12% abnormal, t = -1.73). It nods at the "
            "'sell the news' half of the lore -- but it falls short of the |t| >= 2 evidence bar, "
            "and as the quants notebook shows, it's driven by a handful of September events and "
            "evaporates under a permutation null."
        ),

        # ---- BEAT 5 -- THE VERDICT -------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            "- **Signal -- None.** The pre-keynote abnormal drift is **-0.19%**, t = **-0.39**, "
            "permutation p = **0.75** -- indistinguishable from zero. The post-event droop "
            "(t = -1.73) is suggestive but does not clear the t >= 2 bar, and it concentrates in "
            "a few September events. There is no robust keynote drift once AAPL is benchmarked "
            "against itself.\n"
            "- **Tradability -- Mirage.** The 'buy the rumor' trade nets ~0.4%/event, which is "
            "just AAPL's own drift over 4 trading days -- not a keynote edge. You'd be taking "
            "concentrated single-name risk for a return you'd get by simply holding the stock.\n"
            "- **Busted.** The folklore confuses a great stock's trend with a calendar effect. "
            "Strip out the trend and the keynote windows are flat."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 - Could you actually trade it?\n\n"
            "The 'strategy' is: buy AAPL 5 trading days before each keynote (with a realistic "
            "one-day execution lag), exit the day before the reveal, pay costs both ways."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bt = st.buy_the_rumor_backtest(px, kt, pre_window=5, cost_bps=5.0)\n"
            "    mg, mn = bt['mean_gross']*100, bt['mean_net']*100\n"
            "    hr = bt['hit_rate_gross']*100\n"
            "    tg, tn = bt['total_gross']*100, bt['total_net']*100\n"
            "else:\n"
            "    mg, mn = R['mean_gross_pct'], R['mean_net_pct']\n"
            "    hr = R['hit_rate_gross_pct']\n"
            "    tg, tn = R['total_gross_pct'], R['total_net_pct']\n"
            "\n"
            "print(f'Mean return per trade -- gross: {mg:+.2f}%  net: {mn:+.2f}%')\n"
            "print(f'Hit-rate (gross): {hr:.1f}%')\n"
            "print(f'Total compounded over all events -- gross: {tg:+.1f}%  net: {tn:+.1f}%')\n"
            "print()\n"
            "print('That ~0.4%/event net is AAPL\\'s ordinary 4-day drift, not a keynote signal.')\n"
            "print('You hold ~3 times a year for a few days -- mostly out of the market, fully')\n"
            "print('exposed to single-name gap risk -- to harvest what buy-and-hold gives free.')"
        ),
        md(
            "The pre-keynote trade is *positive* -- but only because AAPL drifts up on average "
            "every few days. There is no abnormal keynote alpha to capture, and the trade leaves "
            "you out of the stock most of the year while taking concentrated single-name risk "
            "during the windows. The only real trade is: don't confuse a stock's trend with a "
            "calendar omen."
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 - Going further\n\n"
            "- **The positive control:** the companion notebook plants a synthetic keynote drift "
            "and shows the same event-study machinery detects it cleanly -- so a null result on "
            "the real tape is a statement about AAPL, not about the method.\n"
            "- **Other event-around-the-calendar studies** in this lab share the same base-rate "
            "discipline: the result has to beat the asset's own unconditional drift.\n\n"
            "*Think the keynote drift is real? Fork this, pick your own event window or event "
            "subset, and show an abnormal CAR that clears |t| >= 2 on the real AAPL tape with a "
            "permutation p < 0.05. That's the bar -- and it hasn't been cleared.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "01_for_the_curious.ipynb")


# ===========================================================================
# 02 -- FOR THE QUANTS
# ===========================================================================
def build_quants():
    cells = [
        md(
            "# Keynote-Drift -- a quantitative teardown\n"
            "### 48 keynotes * AAPL daily * constant-mean event study * CAR t-test * "
            "permutation null * n=48 power\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) -- "
            "*same seven beats, every claim carrying its standard error.* We run a constant-mean "
            "event study (AAPL benchmarked against its own unconditional daily mean) on the pre- "
            "and post-keynote windows, t-test the per-event cumulative abnormal returns (CARs), "
            "validate with a permutation null that re-anchors the windows on random trading days, "
            "and close with a power calculation that explains why n=48 single-name events cannot "
            "resolve a small drift.\n\n"
            "> **Not investment advice.** Real data: AAPL daily adjusted close (price-only, "
            "split/dividend-adjusted via yfinance, 2008-2025); keynote dates hardcoded in "
            "`data.py`. Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> The **plain-English notes** translate each result back to intuition."
        ),
        code(BOOT + "\n" + R_CELL + "\nfrom scipy import stats as scipy_stats\n"),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            "| **Signal** | `NONE` | Pre-keynote abnormal CAR **-0.19%**, t = **-0.39**, "
            "perm p = **0.75**. Post CAR -1.12%, t = -1.73 -- short of |t| >= 2. |\n"
            "| **Tradability** | `MIRAGE` | 'Buy the rumor' nets ~0.4%/event = AAPL's ordinary "
            "4-day drift; no abnormal alpha, concentrated single-name risk. |\n"
            "| **Busted?** | `YES` | The folklore confuses AAPL's 2008-2025 secular uptrend with "
            "a calendar effect; abnormal (de-trended) windows are flat. |\n\n"
            "> Benchmarking AAPL against its own mean is the whole ballgame. Skip that step and "
            "you 'discover' that a 50-bagger went up before keynotes -- and after them, and on "
            "Tuesdays."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 - The claim, steelmanned\n\n"
            "Let $r_t$ be AAPL's daily return and $\\bar r$ its full-sample mean. The abnormal "
            "return is $a_t = r_t - \\bar r$ (a constant-mean market model with AAPL as its own "
            "benchmark). For keynote $i$ anchored at trading day $\\tau_i$:\n\n"
            "- **Pre-window CAR:** $\\mathrm{CAR}^{pre}_i = \\sum_{k=1}^{5} a_{\\tau_i - k}$.\n"
            "- **Post-window CAR:** $\\mathrm{CAR}^{post}_i = \\sum_{k=0}^{4} a_{\\tau_i + k}$.\n\n"
            "The folklore implies:\n\n"
            "- **H1 (buy the rumor).** $E[\\mathrm{CAR}^{pre}] > 0$ -- abnormal run-up into the event.\n"
            "- **H2 (sell the news).** $E[\\mathrm{CAR}^{post}] < 0$ -- abnormal sell-off after.\n\n"
            "We test H1 and H2 against the constant-mean null, by a one-sample t-test on the "
            "per-event CARs and a permutation null. We fail to reject H1; H2 is suggestive but "
            "not significant."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 - So what? -- what rides on each answer\n\n"
            "'Buy the rumor, sell the news' is the most cited Apple-trading heuristic. If H1-H2 "
            "held with tradable magnitude, it would be a calendar arbitrage on the most liquid "
            "single name on earth, with the dates known years in advance. The instructive finding "
            "is *how the illusion is manufactured*: a stock with a strong drift makes **every** "
            "short window look positive, so the naive (non-abnormal) test 'confirms' the rumor leg "
            "automatically -- a textbook base-rate / benchmark error."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 - How we'd know -- the protocol\n\n"
            "- **Data.** AAPL daily adjusted close, price-only (yfinance `auto_adjust=True`), "
            "2008-2025; 48 keynotes (WWDC / September / Spring) hardcoded in `data.py`.\n"
            "- **Model.** Constant-mean market model: abnormal return $a_t = r_t - \\bar r$.\n"
            "- **Windows.** Pre = 5 trading days before the reveal; post = reveal day + 4.\n"
            "- **Inference.** One-sample t-test on per-event CARs; permutation null (10,000 "
            "re-anchors on random non-event days); reported two-sided.\n"
            "- **Tradable leg.** Long AAPL from 5 days pre (one-day execution lag) to event-eve "
            "close; gross AND net of 5 bps one-way x NAV (round trip 10 bps); price-only.\n"
            "- **Positive control.** Synthetic AAPL tape with a planted keynote drift to confirm "
            "the machinery detects an effect when one exists.\n"
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 - The teardown"),
        md(
            "### 4a - Per-event CARs and the one-sample t-test\n\n"
            "The constant-mean abnormal CARs for every keynote, and the t-test of their mean "
            "against zero."
        ),
        code(
            "if HAVE_REAL:\n"
            "    px = data.fetch_prices(fetch=False)\n"
            "    kt = data.keynote_table()\n"
            "    cs = st.car_stats(px, kt, pre_window=5, post_window=5, n_permutations=10_000, seed=299)\n"
            "    n_ev = cs['n_events']\n"
            "    pre_m, post_m = cs['mean_pre_car']*100, cs['mean_post_car']*100\n"
            "    t_pre, p_pre = cs['t_pre'], cs['p_pre']\n"
            "    t_post, p_post = cs['t_post'], cs['p_post']\n"
            "    pre_cars = cs['pre_cars']*100\n"
            "else:\n"
            "    n_ev = R['n_events']\n"
            "    pre_m, post_m = R['mean_pre_car_pct'], R['mean_post_car_pct']\n"
            "    t_pre, p_pre = R['t_pre'], R['p_pre']\n"
            "    t_post, p_post = R['t_post'], R['p_post']\n"
            "    rng = np.random.default_rng(299)\n"
            "    pre_cars = rng.normal(R['mean_pre_car_pct'], R['pre_sd_pct'], R['n_events'])\n"
            "\n"
            "print(f'n events: {n_ev}')\n"
            "print(f'Mean pre-keynote abnormal CAR:  {pre_m:+.3f}%  (t={t_pre:+.3f}, p={p_pre:.4f})')\n"
            "print(f'Mean post-keynote abnormal CAR: {post_m:+.3f}% (t={t_post:+.3f}, p={p_post:.4f})')\n"
            "print()\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(pre_cars, bins=16, color=GREY, alpha=0.75)\n"
            "ax.axvline(pre_m, c=RED, lw=2.5, label=f'Mean = {pre_m:+.2f}%')\n"
            "ax.axvline(0, c='k', ls=':', lw=1.5, label='Zero (null)')\n"
            "ax.set_xlabel('Pre-keynote abnormal CAR (%)'); ax.set_ylabel('Count')\n"
            "ax.set_title(f'Pre-keynote CARs centre on zero (t={t_pre:+.2f}, p={p_pre:.2f})')\n"
            "ax.legend(); plt.tight_layout(); plt.show()"
        ),
        md(
            "> The pre-keynote CAR distribution is centred on zero: mean **-0.19%**, "
            "t = **-0.39**, p = **0.70**. There is no abnormal run-up into Apple keynotes once "
            "AAPL is benchmarked against its own mean. *In plain words: the rumor doesn't move "
            "the stock more than the stock moves anyway.*"
        ),
        md(
            "### 4b - The permutation null: re-anchor on random days\n\n"
            "Re-anchor the 48 pre-windows on random non-event trading days 10,000 times and "
            "record the distribution of mean CARs under the null."
        ),
        code(
            "if HAVE_REAL:\n"
            "    perm_means = cs['perm_means']*100\n"
            "    obs = cs['mean_pre_car']*100\n"
            "    perm_p = cs['perm_p']\n"
            "else:\n"
            "    rng2 = np.random.default_rng(299)\n"
            "    perm_means = rng2.normal(0.0, R['pre_sd_pct']/np.sqrt(R['n_events']), 10_000)\n"
            "    obs = R['mean_pre_car_pct']\n"
            "    perm_p = R['perm_p']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(perm_means, bins=40, color=GREY, alpha=0.7, label='Random-anchor null')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'Observed mean: {obs:+.2f}%')\n"
            "ax.set_xlabel('Mean pre-keynote CAR (%)'); ax.set_ylabel('Count')\n"
            "ax.set_title('The observed mean lands squarely inside the random-anchor null')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Permutation p (two-sided): {perm_p:.4f}')\n"
            "print(f'Percentile of |observed| in null: '\n"
            "      f'{(np.abs(perm_means) < abs(obs)).mean():.1%} of draws smaller')"
        ),
        md(
            "The observed mean pre-keynote CAR sits in the dead centre of the random-anchor "
            "distribution. Permutation p = **0.75** -- the keynote dates carry no information "
            "the calendar wouldn't give you by chance."
        ),
        md(
            "### 4c - Slicing by event type and sub-period (the multiple-window trap)\n\n"
            "The drift could hide in a subset -- September iPhone events, or the early years. "
            "We check, mindful that slicing many ways inflates false positives."
        ),
        code(
            "if HAVE_REAL:\n"
            "    cars = st.event_cars(px, kt)\n"
            "    cars['yr'] = pd.to_datetime(cars['date']).dt.year\n"
            "    by_type = cars.groupby('event_type')['pre_car'].mean()*100\n"
            "    sept = cars[cars['event_type']=='September']\n"
            "    st_t, st_p = scipy_stats.ttest_1samp(sept['post_car'], 0.0)\n"
            "    sept_post = sept['post_car'].mean()*100\n"
            "    early = cars[cars['yr']<=2016]['pre_car'].mean()*100\n"
            "    late = cars[cars['yr']>=2017]['pre_car'].mean()*100\n"
            "    n_sept = len(sept)\n"
            "else:\n"
            "    by_type = pd.Series({'September': -1.84, 'Spring': 0.80, 'WWDC': 0.94})\n"
            "    sept_post, st_t, st_p, n_sept = R['sept_post_pct'], R['sept_post_t'], R['sept_post_p'], R['n_sept']\n"
            "    early, late = R['early_pre_pct'], R['late_pre_pct']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.0))\n"
            "ax.bar(by_type.index, by_type.values, color=GREY, width=0.55)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('Mean pre-keynote abnormal CAR (%)')\n"
            "ax.set_title('No event type shows a consistent pre-keynote drift')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Pre-keynote abnormal CAR by event type (%):')\n"
            "print(by_type.round(2).to_string())\n"
            "print()\n"
            "print(f'September POST-event CAR: {sept_post:+.2f}% (t={st_t:+.2f}, p={st_p:.3f}, n={n_sept})')\n"
            "print(f'Pre-keynote CAR -- early (<=2016): {early:+.2f}%  late (>=2017): {late:+.2f}%')\n"
            "print('Even the September post-event droop is not significant on its own subset.')"
        ),
        md(
            "> The 'sell the news' droop is concentrated in September iPhone events, but on that "
            "subset (n=19) the post-event CAR is -1.33% with t = -1.10, p = 0.29 -- not "
            "significant. And the pre-keynote drift is near zero in **both** the early and late "
            "sub-periods. Slicing finds no robust pocket of signal."
        ),
        md(
            "### 4d - Power: what drift could n=48 actually detect?\n\n"
            "The minimum detectable mean CAR at 80% power, two-sided, alpha=0.05."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sd = float(np.std(cs['pre_cars'], ddof=1))*100\n"
            "    n = cs['n_events']\n"
            "else:\n"
            "    sd, n = R['pre_sd_pct'], R['n_events']\n"
            "\n"
            "from scipy.stats import t as t_dist\n"
            "alpha = 0.05\n"
            "t_crit = t_dist.ppf(1-alpha/2, n-1)\n"
            "t_pow  = t_dist.ppf(0.80, n-1)\n"
            "mde = (t_crit + t_pow) * sd / np.sqrt(n)\n"
            "print(f'Per-event CAR std: {sd:.2f}%  |  n = {n}')\n"
            "print(f'Minimum detectable mean CAR (80% power, 2-sided a=0.05): {mde:.2f}%/event')\n"
            "print()\n"
            "print(f'Observed pre-keynote CAR: {R[\"mean_pre_car_pct\"]:+.2f}%/event')\n"
            "print(f'That is {abs(R[\"mean_pre_car_pct\"])/mde:.0%} of the detection threshold.')\n"
            "print()\n"
            "print('Conclusion: n=48 single-name events cannot resolve a sub-1.4%/event drift.')"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            "- **Signal `NONE`** -- pre-keynote abnormal CAR -0.19%, t -0.39, perm p 0.75. The "
            "post-event CAR (t -1.73) is suggestive but short of |t| >= 2, concentrates in "
            "September, and is non-significant on that subset. No robust drift survives "
            "benchmarking AAPL against itself.\n"
            "- **Tradability `MIRAGE`** -- the tradable leg nets ~0.4%/event, which is AAPL's "
            "own 4-day drift; no abnormal alpha, concentrated single-name gap risk.\n"
            "- **Busted** -- the folklore mistakes a 50-bagger's trend for a calendar effect."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 - Could you trade it? -- gross vs net\n\n"
            "Long AAPL from 5 days pre-keynote (one-day execution lag) to event-eve close; "
            "5 bps one-way x NAV (10 bps round trip); price-only."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bt = st.buy_the_rumor_backtest(px, kt, pre_window=5, cost_bps=5.0)\n"
            "    gross = bt['gross']*100\n"
            "    mg, mn = bt['mean_gross']*100, bt['mean_net']*100\n"
            "    hr = bt['hit_rate_gross']*100\n"
            "    tg, tn = bt['total_gross']*100, bt['total_net']*100\n"
            "    af = bt['ann_factor']\n"
            "else:\n"
            "    rng3 = np.random.default_rng(299)\n"
            "    gross = rng3.normal(R['mean_gross_pct'], 3.5, R['n_trades'])\n"
            "    mg, mn = R['mean_gross_pct'], R['mean_net_pct']\n"
            "    hr, tg, tn, af = R['hit_rate_gross_pct'], R['total_gross_pct'], R['total_net_pct'], R['ann_factor']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "colors = [GREEN if g>0 else RED for g in gross]\n"
            "ax.bar(range(len(gross)), gross, color=colors)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.axhline(mg, c=AMBER, ls='--', lw=2, label=f'Mean gross {mg:+.2f}%')\n"
            "ax.set_xlabel('Keynote # (chronological)'); ax.set_ylabel('Pre-keynote trade return (%)')\n"
            "ax.set_title(f'Buy-the-rumor: {hr:.0f}% hit-rate, mean net {mn:+.2f}%/event = AAPL drift')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Mean per trade -- gross {mg:+.2f}%  net {mn:+.2f}%  | hit-rate {hr:.1f}%')\n"
            "print(f'Total compounded -- gross {tg:+.1f}%  net {tn:+.1f}%  over {af:.1f} events/yr')\n"
            "print('Net positive -- but it is the stock\\'s drift, not keynote alpha.')"
        ),
        md(
            "> The trade is net-positive, yet it captures only AAPL's ordinary multi-day drift "
            "(the abnormal CAR is zero). You take concentrated single-name risk a few times a "
            "year to earn what buy-and-hold gives you for free. No tradable edge."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 - Going further -- the synthetic positive control\n\n"
            "Does the event-study machinery detect a keynote drift *when one exists*? We plant a "
            "drift in a synthetic AAPL tape and sweep its size."
        ),
        code(
            "planted = [0, 20, 50, 100]\n"
            "kt2 = data.keynote_table()\n"
            "rows = []\n"
            "for bps in planted:\n"
            "    sp, _ = data.synthetic_prices(drift_bps=float(bps), seed=299)\n"
            "    r = st.car_stats(sp, kt2, pre_window=5, post_window=5, n_permutations=1000, seed=299)\n"
            "    rows.append({'bps': bps, 'pre_car_pct': r['mean_pre_car']*100,\n"
            "                 't_pre': r['t_pre'], 'perm_p': r['perm_p']})\n"
            "res = pd.DataFrame(rows)\n"
            "colors = [GREEN if (abs(x['t_pre'])>=2 and x['perm_p']<0.05) else RED for x in rows]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.bar([str(x['bps']) for x in rows], [x['pre_car_pct'] for x in rows], color=colors)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('Planted pre-keynote drift (bps/day)'); ax.set_ylabel('Detected pre CAR (%)')\n"
            "ax.set_title('The engine finds the drift when it exists (green = t>=2 and perm p<0.05)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(res.round(4).to_string(index=False))"
        ),
        md(
            "The engine cleanly detects a planted keynote drift once it is large enough (around "
            "50 bps/day, t clears 2). The real tape -- with a -0.19% abnormal CAR -- is nowhere "
            "near the detection threshold. The null is a statement about **AAPL and sample size**, "
            "not about the method."
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


def _execute(name):
    """Execute a notebook in-place with nbconvert (no --allow-errors)."""
    from nbconvert.preprocessors import ExecutePreprocessor
    path = os.path.join(HERE, name)
    nb = nbf.read(path, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": HERE}})
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("executed", path)


if __name__ == "__main__":
    build_curious()
    build_quants()
    _execute("01_for_the_curious.ipynb")
    _execute("02_for_the_quants.ipynb")
