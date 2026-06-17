"""Generate (and execute) the two narrative notebooks for Study 294 (Coinbase-Rank).

    python notebooks/build_notebooks.py

This writes both notebooks AND executes them in-place via nbconvert (no
--allow-errors). The hardcoded rank-spike table and the synthetic generator run
anywhere, offline and deterministically; the real BTC/ETH cells use the cached
yfinance parquet at the study-level _cache/ if present, and otherwise quote the
frozen headline numbers in ``R`` (mirroring docs/results.md) and fall back to a
synthetic tape -- so the notebook re-runs for any reader, online or not.

The _write convention (each build_*() ends by calling _write) is kept so the
repo's intro-restyle tooling can monkeypatch it.
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-17).
R = dict(
    n_days=3142,
    n_events=15,
    beta_eth=0.605,
    aar_day0_pct=2.29,
    fwd=5,
    fwd_car_pct=-5.23,
    fwd_car_t=-1.69,
    n_neg=8,
    perm_p=0.0014,
    t_drop1=-1.27,
    mean_drop1_pct=-3.53,
    median_fwd_pct=-0.97,
    fwd3_car_pct=-2.87,
    fwd3_car_t=-1.29,
    fwd10_car_pct=-4.75,
    fwd10_car_t=-1.55,
    bt_n_trades=15,
    bt_gross_pct=3.02,
    bt_net_pct=2.37,
    bt_btc_pct=-3.02,
    bt_net_t=0.80,
    bt_hit=0.533,
    bt_excess_pct=5.39,
    bt_hold3_net_pct=2.39,
    bt_hold3_t=1.10,
    bt_hold10_net_pct=-0.40,
    bt_hold10_t=-0.08,
    fp="fcd0c13455df",
    date_start="2017-11-10",
    date_end="2026-06-17",
)

R_LITERAL = ",\n    ".join(f"{k}={v!r}" for k, v in R.items())

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

from coinbase_rank import data, strategy as st

def _have_cache():
    try:
        data.fetch_prices(fetch=False)
        return True
    except FileNotFoundError:
        return False

HAVE_REAL = _have_cache()
print("BTC/ETH cache present:", HAVE_REAL)
"""

R_CELL = "# Frozen headline numbers -- mirror of docs/results.md (as-of 2026-06-17)\nR = dict(\n    " + R_LITERAL + ",\n)\n"

# A synthetic-tape fallback used when the real cache is absent, so the figures
# always render. It plants a realistic post-spike negative drift so the story is
# intact even offline (clearly labelled as synthetic, never under a real banner).
SYNTH_FALLBACK = """\
# OFFLINE FALLBACK (no real cache): a SYNTHETIC BTC/ETH tape with a planted
# post-spike negative drift so every figure renders. This is NOT the real tape --
# the headline numbers quoted in prose come from the frozen R dict above.
_synth_df, _ = data.synthetic_panel(n_days=1500, n_events=R['n_events'],
                                    bump_bps=180.0, seed=294)
rets = _synth_df[['btc_ret', 'eth_ret']].copy()
spike_dates = _synth_df.index[_synth_df['is_event']]
tw = pd.DataFrame({'date': spike_dates, 'direction': 1, 'note': 'synthetic'})
SYNTH_BANNER = "  [SYNTHETIC fallback tape -- real numbers are in R]"
"""

REAL_LOAD = """\
prices = data.fetch_prices(fetch=False)
rets = data.prices_to_returns(prices)
tw = data.rank_spike_table()
spike_dates = tw['date']
SYNTH_BANNER = ""
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Coinbase-Rank -- does the app topping the chart mark a crypto top?\n"
            "### The 'Coinbase #1 in the App Store = sell signal', tested honestly\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Busted%3F: Mostly](https://img.shields.io/badge/Busted%3F-Mostly-dab617?style=flat-square)\n\n"
            "Crypto Twitter has a favourite omen: when the **Coinbase app hits #1** in the Apple "
            "App Store, your barber is buying Bitcoin and the top is in. It *sounds* compelling -- "
            "every famous blow-off (December 2017, May 2021, the $100k break) came with Coinbase "
            "atop the charts. This notebook asks the two questions that matter: **(1)** does crypto "
            "really weaken *after* the app spikes, beyond the usual noise? And **(2)** could you "
            "have *traded* it (shorting after the spike)?\n\n"
            "> **This is the plain-language layer.** Want the market-model abnormal returns, the "
            "permutation distribution, and the lagged short backtest? That's "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** -- same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart is drawn by "
            "the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT + "\n" + R_CELL),
        code(
            "if HAVE_REAL:\n"
            + _indent(REAL_LOAD)
            + "else:\n"
            + _indent(SYNTH_FALLBACK)
            + "print('rows:', len(rets), '| spikes:', len(tw), SYNTH_BANNER)"
        ),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does crypto weaken after a Coinbase rank spike? | **A little -- but shakily.** The "
            f"5-day forward abnormal return averages **{R['fwd_car_pct']:.1f}%** (BTC net of ETH). "
            "The direction matches the omen, and a permutation test flags it -- but the "
            f"cross-sectional t is only **{R['fwd_car_t']:.2f}** (under the |2| bar). |\n"
            "| Is it just one or two crashes? | **Largely.** Drop the single worst event and the "
            f"signal collapses (t -> **{R['t_drop1']:.2f}**). The *median* drop is only "
            f"**{R['median_fwd_pct']:.1f}%** -- the mean is dragged by a couple of big busts. "
            f"Only **{R['n_neg']}/{R['n_events']}** events were even negative -- barely a coin. |\n"
            "| Could you have shorted it? | **No.** After a one-day lag, fees and borrow, the short "
            f"nets **+{R['bt_net_pct']:.1f}%** per trade with a **{R['bt_hit']:.0%}** hit-rate and "
            f"t = **{R['bt_net_t']:.2f}** -- a coin-flip you pay to play. |\n\n"
            "> The omen points the right way, but it's a *whisper*, not a signal -- and it's "
            "un-shortable after costs."
        ),

        # ---- BEAT 1 -- THE CLAIM ----------------------------------------------
        md(
            "## 1 - The claim\n\n"
            "> *\"When the Coinbase app hits #1 on the App Store, retail FOMO has peaked and crypto "
            "is about to top. Sell when your neighbour is downloading Coinbase.\"*\n\n"
            "There's a plausible mechanism: App Store rank is a real-time proxy for *retail "
            "attention*, and retail attention famously peaks near sentiment tops (the 'dumb money' "
            "story). So unlike pure folklore, this one has a behavioural story behind it. The "
            "question is whether the story shows up in the forward returns -- or whether we just "
            "*remember* the spikes that preceded a crash."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 - So what?\n\n"
            "If a *publicly observable* signal -- anyone can check the App Store ranking -- "
            "reliably called crypto tops, it would be free money, and free money doesn't survive. "
            "Either (a) the signal is real and tradable (unlikely for something this public), "
            "(b) it's real but already in the price (a top is a top; you can't short fast enough), "
            "or (c) it's hindsight: we cherry-remember the spikes that worked. Sorting (a) from "
            "(b) from (c) is the whole job here."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW -----------------------------------------
        md(
            "## 3 - How would we even know?\n\n"
            "Three traps to avoid:\n\n"
            "1. **The whole-market trap.** When BTC sells off, *all* crypto usually sells off. We "
            "strip out the broad move: we benchmark BTC against **ETH** (a 'market model') and "
            "measure only the *abnormal* part -- BTC's forward move net of what ETH did.\n"
            "2. **The hindsight trap.** We picked the *famous* spikes -- the ones everyone "
            "remembers *because* a top followed. That biases the study **toward** the omen; it's an "
            "upper bound, not a fair forward test. Named on the Signal axis.\n"
            "3. **The execution-lag trap.** You can't short the App-Store-rank *instant*. The "
            "honest entry is the **close of the spike day**, and shorting BTC means paying **borrow** "
            "on top of fees.\n\n"
            "We test on real BTC-USD / ETH-USD daily data (2017-2025, "
            f"{R['n_events']} hardcoded rank-spike events)."
        ),

        # ---- BEAT 4 -- THE TEARDOWN ------------------------------------------
        md(
            "## 4 - The teardown -- let's actually look\n\n"
            "**First: the abnormal-return path around a spike.** For each spike we line up BTC's "
            "move (net of ETH) from one day before to ten days after, then average."
        ),
        code(
            "es = st.event_study_stats(rets, spike_dates, pre=1, post=10, fwd=5,\n"
            "                          n_permutations=2000, seed=294)\n"
            "prof = es['mean_by_offset'] * 100\n"
            "cum = prof.cumsum()\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.5))\n"
            "ax.plot(prof.index, cum.values, 'o-', c=RED, lw=2, ms=5,\n"
            "        label='Cumulative abnormal return vs ETH')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.axvline(0, c=GREY, ls=':', lw=1, label='Spike day (0)')\n"
            "ax.set_xlabel('Trading days relative to the spike (0 = spike day)')\n"
            "ax.set_ylabel('Cumulative abnormal return (%)')\n"
            "ax.set_title('After the spike, BTC drifts DOWN vs ETH -- but slowly' + SYNTH_BANNER)\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"5-day forward abnormal CAR: {es['fwd_car_pct']:+.1f}% (real headline: {R['fwd_car_pct']:+.1f}%)\")\n"
            "print(f\"Cross-sectional t: {es['fwd_car_t']:+.2f} (real headline: {R['fwd_car_t']:+.2f}) -- under |2|.\")"
        ),
        md(
            "The cumulative path bends **downward** after day 0 -- the direction the omen predicts. "
            f"On the real tape the 5-day forward abnormal return is **{R['fwd_car_pct']:.1f}%**. But "
            f"the cross-sectional t is only **{R['fwd_car_t']:.2f}** -- it points the right way "
            "without clearing the bar that would make us call it *real*."
        ),
        md("**Is it just one or two crashes?** Let's see how negative each event actually was."),
        code(
            "abn, alpha, beta = st.market_model_residuals(\n"
            "    rets, st._event_mask_from_dates(rets.index, spike_dates))\n"
            "win = st.event_window_returns(abn, spike_dates, pre=1, post=10)\n"
            "fwd_cols = [o for o in win.columns if 1 <= o <= 5]\n"
            "car = (win[fwd_cols].sum(axis=1)) * 100\n"
            "n_neg = int((car < 0).sum()); n_tot = len(car)\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "order = car.sort_values()\n"
            "ax.bar(range(len(order)), order.values,\n"
            "       color=[GREEN if v < 0 else RED for v in order.values])\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('Spike events (sorted)'); ax.set_ylabel('5-day forward abnormal return (%)')\n"
            "ax.set_title(f'{n_neg}/{n_tot} spikes were followed by weakness (green = down)' + SYNTH_BANNER)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Negative (omen worked): {n_neg}/{n_tot}  (real headline: {R[\"n_neg\"]}/{R[\"n_events\"]})')\n"
            "print(f'Median forward CAR (real): {R[\"median_fwd_pct\"]:+.1f}%  <- the typical event is tiny')\n"
            "print(f'Drop the single worst crash and t collapses to {R[\"t_drop1\"]:+.2f} (real headline).')"
        ),
        md(
            f"Only **{R['n_neg']} of {R['n_events']}** spikes were actually followed by weakness -- "
            f"barely above a coin flip. The *median* forward move is just **{R['median_fwd_pct']:.1f}%**; "
            "the negative average is carried by a couple of genuine blow-offs (Dec-2017, mid-2021). "
            f"Remove the single worst event and the t-stat collapses to **{R['t_drop1']:.2f}**. This "
            "is a *fragile* signal, not a robust one."
        ),
        md(
            "**Now the only question that pays: could you short it?** We sell BTC short at the "
            "*close of the spike day* (one honest day of lag), hold 5 days, pay fees **and borrow**, "
            "and compare to just holding BTC."
        ),
        code(
            "bt = st.rank_short_backtest(rets, spike_dates, hold_days=5,\n"
            "                            cost_bps_oneway=20.0, borrow_bps_per_day=5.0, seed=294)\n"
            "net = bt['net_mean_pct'] if HAVE_REAL else R['bt_net_pct']\n"
            "btc = bt['btc_bench_pct'] if HAVE_REAL else R['bt_btc_pct']\n"
            "hit = bt['hit_rate'] if HAVE_REAL else R['bt_hit']\n"
            "tval = bt['net_t'] if HAVE_REAL else R['bt_net_t']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.3))\n"
            "bars = ax.bar(['Short the spike\\n(lagged, net)', 'Just hold BTC'],\n"
            "              [net, btc], color=[AMBER, GREY], width=0.5)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('Mean return per trade (%)')\n"
            "ax.set_title(f'After a one-day lag: net t = {tval:+.2f}, hit-rate = {hit:.0%}' + SYNTH_BANNER)\n"
            "for b, v in zip(bars, [net, btc]):\n"
            "    ax.annotate(f'{v:+.1f}%', (b.get_x()+b.get_width()/2, v),\n"
            "                ha='center', va='bottom' if v >= 0 else 'top')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Short net per trade: {net:+.2f}% | BTC long: {btc:+.2f}% | hit-rate: {hit:.1%} | t = {tval:+.2f}')\n"
            "print('A +2% drift with a ~53% hit-rate and t<1 is a coin-flip you pay fees and borrow to play.')"
        ),
        md(
            f"The short does *make* money on average (**+{R['bt_net_pct']:.1f}%** per trade, because "
            f"BTC averaged **{R['bt_btc_pct']:.1f}%** over those windows) -- but the hit-rate is only "
            f"**{R['bt_hit']:.0%}** and the t-stat is **{R['bt_net_t']:.2f}**. That's statistically "
            "indistinguishable from luck: you'd be shorting the world's strongest-trending asset on "
            "a coin flip."
        ),

        # ---- BEAT 5 -- THE VERDICT -------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            f"- **Signal -- Weak.** The 5-day forward abnormal return ({R['fwd_car_pct']:.1f}%) "
            "points the way the omen predicts, and a permutation test flags it -- but the "
            f"cross-sectional t is only **{R['fwd_car_t']:.2f}** (under |2|), only "
            f"**{R['n_neg']}/{R['n_events']}** events are negative, and dropping one crash kills it "
            f"(t -> {R['t_drop1']:.2f}). A whisper, on a hindsight-selected list.\n"
            "- **Tradability -- Mirage.** After a one-day lag, fees and borrow, the short nets "
            f"**+{R['bt_net_pct']:.1f}%** with t = **{R['bt_net_t']:.2f}** and a "
            f"**{R['bt_hit']:.0%}** hit-rate -- a coin flip. Shorting a structurally up-trending "
            "asset on this is a way to lose money slowly.\n"
            "- **Mostly busted (as a *tradable* signal).** 'Sell when Coinbase hits #1' is a "
            "memorable story that points the right direction in aggregate, but it is **fragile** "
            "(a couple of events do the work) and **un-harvestable** after the honest frictions."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 - Could you actually trade it?\n\n"
            "Only if you could short *before* the crowd -- which you can't. Three real-world "
            "frictions stack against you: **(1)** the spike and the top are roughly simultaneous, so "
            "a daily trader is a close too late; **(2)** shorting BTC costs a round-trip fee **plus** "
            "a daily borrow fee against NAV; **(3)** the hit-rate is barely above 50%, so BTC often "
            f"keeps ripping. A 3-day hold is no better (t &approx; {R['bt_hold3_t']:.1f}); a 10-day "
            f"hold turns negative ({R['bt_hold10_t']:.1f})."
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 - Going further\n\n"
            "- **The positive control:** the companion notebook plants a known post-spike drift in "
            "a synthetic tape and shows the engine detects it cleanly -- so the real-tape 'weak, "
            "un-tradable' verdict is about the *market*, not the method.\n"
            "- **Same machinery, crypto cousin:** [Study 291 -- Doge-Tweets](../../291-doge-tweets/) "
            "applies the identical hardcoded-table event study to Elon's Dogecoin tweets.\n"
            "- **Folklore twin:** [Study 158 -- Super-Bowl](../../158-super-bowl/) is the same "
            "hindsight-table teardown for a sports indicator.\n\n"
            "*Think the App Store top is a real sell signal? Fork this, add an intraday tape and a "
            "realistic short-fill + borrow model, and show a net Sharpe that beats holding (or "
            "shorting) BTC. That's the bar -- and a one-day lag plus borrow already breaks it.*"
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
            "# Coinbase-Rank -- a quantitative teardown\n"
            "### 15 rank spikes * BTC/ETH market model * forward abnormal returns * CAR * "
            "permutation test * outlier robustness * lagged short backtest\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Busted%3F: Mostly](https://img.shields.io/badge/Busted%3F-Mostly-dab617?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) -- "
            "*same seven beats, every claim carrying its standard error.* We fit a market model "
            "`BTC = a + b*ETH` on non-event days, measure BTC's *forward* abnormal return after "
            "each Coinbase rank spike, run a distribution-free permutation test, check robustness "
            "to the single largest event, and then confront the event study with an honest, lagged, "
            "cost-and-borrow-charged short backtest.\n\n"
            "> **Not investment advice.** Real data: BTC-USD / ETH-USD daily closes "
            f"(Yahoo via yfinance, {R['date_start']}--{R['date_end']}); spike dates hardcoded in "
            "`data.py`. Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **The plain-English notes** translate each result back to intuition."
        ),
        code(BOOT + "\n" + R_CELL + "\nfrom scipy import stats as scipy_stats\n"),
        code(
            "if HAVE_REAL:\n"
            + _indent(REAL_LOAD)
            + "else:\n"
            + _indent(SYNTH_FALLBACK)
            + "print('rows:', len(rets), '| spikes:', len(tw), SYNTH_BANNER)"
        ),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | 5-day forward abnormal CAR **{R['fwd_car_pct']:.1f}%** "
            f"(right direction), but cross-sectional t **{R['fwd_car_t']:.2f}** (< |2|); only "
            f"{R['n_neg']}/{R['n_events']} negative; t -> {R['t_drop1']:.2f} ex-outlier. |\n"
            f"| **Tradability** | `MIRAGE` | Lagged short net **+{R['bt_net_pct']:.1f}%**/trade, "
            f"t = **{R['bt_net_t']:.2f}**, hit-rate **{R['bt_hit']:.0%}** -- a coin flip after "
            "costs + borrow. |\n"
            "| **Busted?** | `MOSTLY` | Direction is right but the effect is fragile (one event "
            "drives it) and un-harvestable at daily resolution on a curated list. |\n\n"
            "> The permutation p looks tiny (~0.001) only because the null CAR distribution is "
            "tight around zero and a couple of big crashes pull the mean; the **cross-sectional "
            "t** (the robust hurdle) does *not* clear 2, so we call the signal WEAK, not REAL."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 - The claim, steelmanned\n\n"
            "Let $R^{BTC}_t$ and $R^{ETH}_t$ be daily returns and let "
            "$\\hat{e}_t = R^{BTC}_t - (\\hat a + \\hat b R^{ETH}_t)$ be the market-model "
            "**abnormal** return (estimated on non-event days). For a set of spike days $E$ and a "
            "forward horizon $f$:\n\n"
            "- **H1 (signal).** The mean forward CAR "
            "$\\overline{\\text{CAR}}_{[1,f]} = \\frac{1}{|E|}\\sum_{t\\in E}\\sum_{k=1}^{f}\\hat e_{t+k} < 0$ "
            "and is statistically distinguishable from a random set of days at |t| >= 2.\n"
            "- **H2 (robustness).** H1 survives dropping the single most negative event (no one "
            "crash drives it).\n"
            "- **H3 (tradability).** A SHORT entered at the spike-day *close* (one-day lag), held "
            "$h$ days, net of fees and borrow, earns a return distinguishable from zero.\n\n"
            "We find a *directionally correct but sub-threshold* H1 (perm-significant, t < 2), "
            "**reject H2** (fragile to one event), and **reject H3** on the real tape."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 - So what? -- what rides on each answer\n\n"
            "App-Store rank is a clean, public proxy for *retail attention*, and attention-induced "
            "buying near sentiment peaks is a documented behavioural force (Barber-Odean; "
            "Da-Engelberg-Gao). So a small forward drag is *theoretically* plausible. The "
            "scientifically interesting result is the gap between a real-looking permutation p and "
            "a sub-2 cross-sectional t: the effect is **directionally consistent but not robustly "
            "estimable** on 15 hindsight-selected events -- and certainly not shortable after "
            "the frictions. 'Real-ish but un-harvestable' is the honest reading."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 - How we'd know -- the protocol\n\n"
            "- **Data.** BTC-USD / ETH-USD daily auto-adjusted closes (Yahoo); "
            f"{R['n_events']} spike dates hardcoded in `data.py`.\n"
            "- **Market model.** OLS `BTC = a + b*ETH` on **non-event** days so events don't "
            "contaminate the benchmark; abnormal = BTC - fit.\n"
            "- **Event window.** Forward CAR over [+1, +5] (the headline); day-0 reaction for "
            "context; path drawn over [-1, +10].\n"
            "- **Inference.** Permutation test (10,000 random event-day sets) -- distribution-free, "
            "robust to BTC's fat tails; plus a cross-sectional t (raw and outlier-dropped) -- the "
            "|t| >= 2 hurdle.\n"
            "- **Tradability.** One-day execution lag (short at spike-day close), hold 3/5/10 days, "
            "20 bps one-way x NAV + 5 bps/day borrow, benchmarked to BTC long.\n"
            "- **Positive control.** Synthetic tape with a planted post-spike drift.\n"
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 - The teardown"),
        md(
            "### 4a - The market model and the forward abnormal-return path\n\n"
            "Fit `BTC = a + b*ETH` on non-event days; the slope `b` tells us how much of BTC's "
            "move is just the broad crypto tide."
        ),
        code(
            "es = st.event_study_stats(rets, spike_dates, pre=1, post=10, fwd=5,\n"
            "                          n_permutations=10000, seed=294)\n"
            "print(f\"Market-model beta on ETH: {es['beta']:.3f}  (real headline: {R['beta_eth']:.3f})\")\n"
            "print(f\"Day-0 abnormal return: {es['aar_day0_pct']:+.2f}%  (the spike-day reaction)\")\n"
            "print(f\"Forward CAR[+1,+5]: {es['fwd_car_pct']:+.2f}%  | t = {es['fwd_car_t']:+.2f}\")\n"
            "print(f\"Permutation p (two-sided): {es['perm_p']:.4f}\")\n"
            "print()\n"
            "prof = es['mean_by_offset'] * 100\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "ax.bar([str(i) for i in prof.index], prof.values,\n"
            "       color=[RED if i >= 1 else GREY for i in prof.index])\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('Days relative to spike'); ax.set_ylabel('Mean abnormal return vs ETH (%)')\n"
            "ax.set_title('Abnormal-return profile: small, mostly-negative post-spike days' + SYNTH_BANNER)\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> Beta on ETH is **&approx;{R['beta_eth']:.2f}** -- the market model strips out the broad "
            f"crypto move. What's left is a forward drag of **{R['fwd_car_pct']:.1f}%** over five "
            f"days, with a day-0 reaction of **+{R['aar_day0_pct']:.1f}%** (the spike often *coincides* "
            "with a final up-thrust before the fade)."
        ),
        md(
            "### 4b - The permutation test: is the forward drag distinguishable from noise?\n\n"
            "BTC returns are fat-tailed, so a normal-theory t is fragile. We reassign the same "
            "number of event days to random in-sample positions 10,000 times and ask how often a "
            "mean forward CAR this extreme appears by chance."
        ),
        code(
            "perm = es['perm_car'] * 100\n"
            "obs = es['fwd_car_pct']\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(perm, bins=40, color=GREY, alpha=0.7, label='Random event-day sets')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'Observed forward CAR: {obs:+.1f}%')\n"
            "ax.axvline(-obs, c=RED, lw=1, ls=':')\n"
            "ax.set_xlabel('Mean forward CAR[+1,+5] (%)'); ax.set_ylabel('Count')\n"
            "ax.set_title(f'Permutation null vs observed -- p = {es[\"perm_p\"]:.4f}' + SYNTH_BANNER)\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Permutation p (two-sided): {es[\"perm_p\"]:.4f}')\n"
            "print('BUT: the permutation test is sensitive to a few large draws. The robust')\n"
            "print('cross-sectional t (next cell) is the hurdle we actually trust.')"
        ),
        md(
            f"The permutation p looks small (&approx; **{R['perm_p']:.3f}**), which is *why* the omen "
            "feels real. But a permutation mean is still driven by a couple of big crashes -- so we "
            "do not stop here. The cross-sectional t and the outlier check below are the decisive "
            "tests."
        ),
        md(
            "### 4c - Robustness: is it one or two crashes?\n\n"
            "The forward CAR is a *mean*. With 15 fat-tailed events, one blow-off can manufacture a "
            "signal. We check the cross-sectional t, the share of negative events, and what happens "
            "when we drop the single worst event."
        ),
        code(
            "abn, alpha, beta = st.market_model_residuals(\n"
            "    rets, st._event_mask_from_dates(rets.index, spike_dates))\n"
            "win = st.event_window_returns(abn, spike_dates, pre=1, post=10)\n"
            "fwd_cols = [o for o in win.columns if 1 <= o <= 5]\n"
            "car = win[fwd_cols].sum(axis=1)\n"
            "n = len(car)\n"
            "t_raw = float(car.mean() / (car.std(ddof=1) / np.sqrt(n)))\n"
            "drop1 = car.drop(car.idxmin())  # drop the most-negative (biggest crash)\n"
            "t_drop1 = float(drop1.mean() / (drop1.std(ddof=1) / np.sqrt(len(drop1))))\n"
            "print(f'n events: {n}  | negative (omen worked): {int((car<0).sum())}/{n}')\n"
            "print(f'Raw forward t:          {t_raw:+.2f}  (real headline: {R[\"fwd_car_t\"]:+.2f})')\n"
            "print(f'Drop-worst forward t:   {t_drop1:+.2f}  (real headline: {R[\"t_drop1\"]:+.2f})')\n"
            "print(f'Median forward CAR:     {car.median()*100:+.2f}%  (real: {R[\"median_fwd_pct\"]:+.1f}%)')\n"
            "print()\n"
            "print('Raw t is already under |2|; dropping the single worst crash pushes it')\n"
            "print('toward zero. The median event is nearly flat -- the signal is fragile.')"
        ),
        md(
            f"> This is the crux. The raw forward t is **{R['fwd_car_t']:.2f}** -- already under the "
            f"|2| bar -- and dropping the single worst event takes it to **{R['t_drop1']:.2f}**. "
            f"Only **{R['n_neg']}/{R['n_events']}** events were negative and the median forward move "
            f"is **{R['median_fwd_pct']:.1f}%**. **H1 is sub-threshold and H2 is rejected**: a "
            "couple of real blow-offs do most of the work."
        ),
        md(
            "### 4d - The tradability reckoning: the lagged short backtest\n\n"
            "Now the honest trade. SHORT BTC at the spike-day *close* (one-day lag), hold `h` days, "
            "pay 20 bps one-way **and** 5 bps/day borrow, benchmark to BTC long. Sweep the horizon."
        ),
        code(
            "rows = []\n"
            "for h in [3, 5, 10]:\n"
            "    bt = st.rank_short_backtest(rets, spike_dates, hold_days=h,\n"
            "                                cost_bps_oneway=20.0, borrow_bps_per_day=5.0, seed=294)\n"
            "    rows.append({'hold': h, 'net_%': bt['net_mean_pct'],\n"
            "                 'btc_long_%': bt['btc_bench_pct'],\n"
            "                 'net_t': bt['net_t'], 'hit': bt['hit_rate']})\n"
            "bt_df = pd.DataFrame(rows)\n"
            "print(bt_df.to_string(index=False))\n"
            "print()\n"
            "print('Real headline (hold=5): net %+.1f%% | t %.2f | hit %.0f%%' % (\n"
            "      R['bt_net_pct'], R['bt_net_t'], R['bt_hit']*100))\n"
            "print('Every horizon: net t <= ~1, hit-rate ~50%. No harvestable short edge.')"
        ),
        md(
            f"The short nets **+{R['bt_net_pct']:.1f}%** per trade at 5 days (t = **{R['bt_net_t']:.2f}**), "
            f"flattens by 10 days ({R['bt_hold10_net_pct']:+.1f}%, t = {R['bt_hold10_t']:.2f}), and "
            f"carries a **{R['bt_hit']:.0%}** hit-rate throughout. **H3 rejected**: this is a "
            "coin-flip short on a structurally up-trending asset, with fees and borrow on top."
        ),
        md(
            "### 4e - Why the omen still 'feels' right\n\n"
            "BTC's *unconditional* drift is strongly positive, so a short loses on average in "
            "general. The reason the spike-conditioned short is roughly break-even is that the "
            "windows genuinely contain weak BTC (the omen's grain of truth). Compare the "
            "spike-conditioned BTC long return to BTC's all-sample average over the same horizon."
        ),
        code(
            "h = 5\n"
            "bt = st.rank_short_backtest(rets, spike_dates, hold_days=h,\n"
            "                            cost_bps_oneway=20.0, borrow_bps_per_day=5.0, seed=294)\n"
            "cond_btc = bt['btc_bench_pct']\n"
            "btc = rets['btc_ret'].to_numpy()\n"
            "uncond = float(np.mean([np.prod(1+btc[i:i+h])-1 for i in range(len(btc)-h)])*100)\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.3))\n"
            "bars = ax.bar(['BTC after a spike\\n(5-day)', 'BTC any 5-day\\n(unconditional)'],\n"
            "              [cond_btc, uncond], color=[RED, GREY], width=0.5)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('Mean BTC return (%)')\n"
            "ax.set_title('Post-spike BTC is weak vs its own average -- but noisily' + SYNTH_BANNER)\n"
            "for b, v in zip(bars, [cond_btc, uncond]):\n"
            "    ax.annotate(f'{v:+.2f}%', (b.get_x()+b.get_width()/2, v),\n"
            "                ha='center', va='bottom' if v >= 0 else 'top')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Post-spike 5-day BTC: {cond_btc:+.2f}% vs unconditional {uncond:+.2f}%')\n"
            "print('The grain of truth: post-spike windows ARE weaker than average -- but not')\n"
            "print('reliably enough (t<2, ~53% hit) to short after a one-day lag and borrow.')"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            f"- **Signal `WEAK`** -- forward CAR {R['fwd_car_pct']:.1f}% (right direction), perm p "
            f"&approx; {R['perm_p']:.3f}, BUT cross-sectional t {R['fwd_car_t']:.2f} (< |2|), only "
            f"{R['n_neg']}/{R['n_events']} negative, t -> {R['t_drop1']:.2f} ex-outlier. Curated "
            "hindsight list -> upper bound, named on this axis.\n"
            f"- **Tradability `MIRAGE`** -- lagged short net +{R['bt_net_pct']:.1f}%/trade, "
            f"t {R['bt_net_t']:.2f}, hit-rate {R['bt_hit']:.0%}; a coin flip after costs + borrow.\n"
            "- **Mostly busted as a *tradable* signal** -- directionally consistent, fragile, and "
            "un-harvestable at daily resolution."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 - Could you trade it? -- the short reckoning\n\n"
            f"The lagged short nets +{R['bt_net_pct']:.1f}%/trade against a BTC long that returned "
            f"{R['bt_btc_pct']:.1f}% over the same windows; the t is {R['bt_net_t']:.2f} and the "
            "hit-rate is barely above half. Shorting the strongest-trending major asset on a "
            "sub-2-sigma, ~53%-hit signal -- with borrow ticking against you every day -- is a "
            "negative-expectancy trade dressed up as contrarian wisdom."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 - Going further -- the synthetic positive control\n\n"
            "Does the machinery detect a post-spike drift *when one exists*? We plant a known "
            "negative post-spike drift in a synthetic BTC/ETH tape and sweep its size."
        ),
        code(
            "planted = [0, 100, 200, 500, 1000]\n"
            "out = []\n"
            "for bps in planted:\n"
            "    dfx, _ = data.synthetic_panel(n_days=1500, n_events=15,\n"
            "                                  bump_bps=float(bps), seed=294)\n"
            "    ed = dfx.index[dfx['is_event']]\n"
            "    e = st.event_study_stats(dfx[['btc_ret','eth_ret']], ed, pre=1, post=10, fwd=5,\n"
            "                             n_permutations=1000, seed=99)\n"
            "    out.append({'bump_bps': bps, 'fwd_car_%': e['fwd_car_pct'],\n"
            "                't': e['fwd_car_t'], 'perm_p': e['perm_p']})\n"
            "ctrl = pd.DataFrame(out)\n"
            "colors = [GREEN if (abs(r['t']) > 2 and r['perm_p'] < 0.05) else RED\n"
            "          for _, r in ctrl.iterrows()]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.bar([str(b) for b in ctrl['bump_bps']], ctrl['fwd_car_%'], color=colors)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('Planted post-spike drift (bps/day)'); ax.set_ylabel('Detected forward CAR (%)')\n"
            "ax.set_title('The engine finds a planted drift (green = |t|>2 AND perm p<0.05)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(ctrl.to_string(index=False))"
        ),
        md(
            "The engine cleanly recovers a planted post-spike drift (and stays quiet at bump = 0). "
            "So the real-tape conclusion -- a **weak**, directionally-correct drag that is "
            "**un-tradable** after a one-day lag and borrow -- is a statement about BTC and the "
            "tiny, hindsight-selected event set, not about the method."
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "02_for_the_quants.ipynb")


def _indent(block: str, n: int = 4) -> str:
    pad = " " * n
    return "".join(pad + line if line.strip() else line
                   for line in block.splitlines(keepends=True)) + ("\n" if not block.endswith("\n") else "")


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
    """Execute the notebook in-place via nbconvert (no --allow-errors)."""
    from nbconvert.preprocessors import ExecutePreprocessor

    path = os.path.join(HERE, name)
    with open(path, encoding="utf-8") as f:
        nb = nbf.read(f, as_version=4)
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
