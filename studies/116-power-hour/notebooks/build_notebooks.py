"""Generate the two narrative notebooks for Study 116 (Power-Hour).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the cached
1-hour panel parquets under ../_cache/ if present and otherwise fall back to the frozen
headline numbers in ``R`` (mirroring docs/results.md), so the notebook re-runs for any reader.

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


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-13).
R = dict(
    # Session counts
    n_spy=723, n_qqq=723, n_iwm=723, n_pool=2169,
    # Correlation diagnostic
    corr_spy=+0.0037, t_spy=+0.10,
    corr_qqq=+0.0019, t_qqq=+0.05,
    corr_iwm=-0.0856, t_iwm=-2.31,
    corr_pool=-0.0309, t_pool=-1.44,
    # Follow arm (main claim, gross)
    follow_win=47.4, follow_mean=-1.83, follow_t=-3.62,
    # Random arm (control)
    rand_win=50.5, rand_mean=-0.49, rand_t=-0.95,
    # Fade arm (contrarian, gross)
    fade_win=52.2, fade_mean=+1.83, fade_t=+3.62,
    # Per-instrument follow arm
    spy_mean=-1.49, spy_t=-1.85,
    qqq_mean=-2.21, qqq_t=-2.29,
    iwm_mean=-1.80, iwm_t=-2.06,
    # Fade arm cost sweep
    fade_net00=+1.83, fade_t00=+3.62,
    fade_net05=+1.33, fade_t05=+2.63,
    fade_net10=+0.83, fade_t10=+1.64,
    fade_net20=-0.17, fade_t20=-0.33,
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

from power_hour import data, strategy as st

TICKERS = ["SPY", "QQQ", "IWM"]

def _have_cache():
    return all(os.path.exists(data._cache_path(t, data.DEFAULT_CACHE)) for t in TICKERS)

HAVE_REAL = _have_cache()

def pool_arm(arm, cost=0.0, seed=116):
    \"\"\"Pool ledger trades across the basket (cached real tapes).\"\"\"
    frames = []
    for t in TICKERS:
        panel = data.fetch_daily_panel(t, fetch=False)
        frames.append(st.build_ledger(panel, arm=arm, cost_bps=cost, seed=seed))
    return pd.concat(frames, ignore_index=True)

print("real 1h cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Power-Hour -- does the last bar follow the day, or flip it?\n"
            "### The 'smart-money last hour' tested honestly, in plain English\n\n"
            "![Signal: MIXED](https://img.shields.io/badge/Signal-MIXED-dab617?style=flat-square)\n"
            "![Tradability: FRAGILE](https://img.shields.io/badge/Tradability-FRAGILE-dab617?style=flat-square)\n"
            "![Continuation_claim%3F: BUSTED](https://img.shields.io/badge/Continuation_claim%3F-BUSTED-8b949e?style=flat-square)\n\n"
            "Here's a recipe you'll find in almost every day-trading course: "
            "**'The last hour is the power hour -- smart money and institutions push "
            "the day's established trend into the close.'** The prescription: at 15:00, "
            "look at how the day is going -- up or down -- and bet *in that direction* "
            "for the final hour. This notebook asks the only question that matters: "
            "does the morning *actually* point the last bar's direction?\n\n"
            "> **This is the plain-language layer.** Want the t-stats, the "
            "correlation tests and the cost maths? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** -- same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart "
            "below is drawn by the code beside it. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does the morning predict the last bar's direction? | **Backwards.** "
            f"Following the morning loses **{R['follow_mean']:+.2f} bps/session** -- the last "
            "bar *reverses* the day. |\n"
            "| Beats a random-direction coin? | **No** (for the follow arm). "
            f"A coin loses only {R['rand_mean']:+.2f} bps; the follow signal loses {R['follow_mean']:+.2f}. |\n"
            "| Is there *any* last-hour signal? | **Weak.** Going *against* the morning "
            f"(the fade) earns {R['fade_mean']:+.2f} bps gross -- a real but thin reversal effect. |\n"
            "| Could you trade it? | **Barely, if at all.** The fade breakeven is ~1 bps "
            "round-trip -- within reach for institutions but out of reach for retail. |\n\n"
            "> The 'power hour continuation' story is directionally wrong: the last bar "
            "reverses the day's trend. There IS a signal -- but it is the opposite of what "
            "every day-trading course teaches."
        ),

        # ---- BEAT 1 -- THE CLAIM ---------------------------------------------
        md(
            "## 1 - The claim\n\n"
            "> *'The last hour of the trading day -- the power hour, 15:00-16:00 ET -- "
            "is when smart money steps in to push prices in the direction of the day's "
            "established trend. If the market is up going into the last hour, it tends "
            "to finish stronger. If it is down, it tends to close weaker. Follow the trend "
            "into the power hour.'*\n\n"
            "It is an honest-sounding idea with an academic cousin: Gao et al. (2018) "
            "documented that the *first half-hour* of trading predicts the *last half-hour* "
            "on SPY (a same-day momentum effect). Day-traders generalised this to 'morning "
            "direction predicts the close' -- a broadening of the claim worth testing directly."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 - So what?\n\n"
            "Millions of retail traders use the 'power hour' framework to justify entering "
            "positions in the last 30-60 minutes of the session. If it is a real continuation "
            "effect, it is a clean daily signal requiring no intraday monitoring. If it is "
            "wrong, those traders are fighting against institutional rebalancing flows and "
            "ETF creation/redemption that are known to run in the *opposite* direction near "
            "the close. The difference is one honest test."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW -----------------------------------------
        md(
            "## 3 - How would we even know?\n\n"
            "Two rules keep us honest:\n\n"
            "1. **Split the day at 15:30** (the start of the last regular-session 1-hour bar "
            "on SPY/QQQ/IWM). Compute `morning_ret` = open-09:30 -> close-14:30 and "
            "`last_ret` = open-15:30 -> close-15:30. The signal is formed *before* the last "
            "bar opens -- no look-ahead.\n"
            "2. **Race it against a coin.** Take the exact same sessions and flip a coin for "
            "the direction. If the morning knows something, following it beats the coin. If "
            "the morning is noise, following it won't.\n\n"
            "We run it on **~723 sessions each** for SPY, QQQ, and IWM (~3 years of 1-hour "
            "bars) -- far more power than the 60-day windows available at 5-minute fidelity."
        ),

        # ---- BEAT 4 -- THE TEARDOWN ------------------------------------------
        md(
            "## 4 - The teardown -- let's actually look\n\n"
            "**First: does the morning actually correlate with the last bar?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    corr_rows = []\n"
            "    for t in TICKERS:\n"
            "        panel = data.fetch_daily_panel(t, fetch=False)\n"
            "        c = st.morning_last_corr(panel)\n"
            "        corr_rows.append((t, c['n'], c['corr'], c['tstat']))\n"
            "    # Pool\n"
            "    all_p = pd.concat([data.fetch_daily_panel(t) for t in TICKERS])\n"
            "    cp = st.morning_last_corr(all_p)\n"
            "    corr_rows.append(('Pool', cp['n'], cp['corr'], cp['tstat']))\n"
            "    corr_df = pd.DataFrame(corr_rows, columns=['ticker','n','corr','t-stat'])\n"
            "else:\n"
            f"    corr_df = pd.DataFrame({{\n"
            f"        'ticker': ['SPY','QQQ','IWM','Pool'],\n"
            f"        'n': [{R['n_spy']},{R['n_qqq']},{R['n_iwm']},{R['n_pool']}],\n"
            f"        'corr': [{R['corr_spy']},{R['corr_qqq']},{R['corr_iwm']},{R['corr_pool']}],\n"
            f"        't-stat': [{R['t_spy']},{R['t_qqq']},{R['t_iwm']},{R['t_pool']}]\n"
            f"    }})\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "cols = [RED if abs(r)>0.05 and v<0 else GREEN if abs(r)>0.05 and v>0 else GREY\n"
            "        for r,v in zip(corr_df['corr'], corr_df['t-stat'])]\n"
            "ax.bar(corr_df['ticker'], corr_df['corr'], color=cols)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('Pearson corr (morning_ret vs last_ret)')\n"
            "ax.set_title('Morning -> last-bar correlation: negative (reversal), not positive')\n"
            "plt.tight_layout(); plt.show()\n"
            "corr_df.round(4)"
        ),
        md(
            f"SPY and QQQ show essentially zero correlation -- the last bar is unpredictable "
            f"from the morning alone. IWM shows **corr = {R['corr_iwm']:.3f}** (t = {R['t_iwm']:.2f}), "
            f"which is **negative** -- the last bar reverses the morning's direction. The pooled "
            f"correlation is {R['corr_pool']:.3f} (t = {R['t_pool']:.2f}): directionally "
            f"showing reversal, not continuation.\n\n"
            f"> The continuation story starts facing the wrong direction immediately."
        ),
        md(
            "**Now the honest test: the fair comparison -- follow-morning vs a coin.**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    f_led = pool_arm('follow', cost=0.0)\n"
            "    r_led = pool_arm('random', cost=0.0)\n"
            "    f_sum = st.summarize(f_led, 'ret_gross')\n"
            "    r_sum = st.summarize(r_led, 'ret_gross')\n"
            "    fm, fw, rm, rw = f_sum['mean_bps'], f_sum['win_rate']*100, r_sum['mean_bps'], r_sum['win_rate']*100\n"
            "    # Fade arm\n"
            "    fade_led = pool_arm('fade', cost=0.0)\n"
            "    fa_sum = st.summarize(fade_led, 'ret_gross')\n"
            "    fam, faw = fa_sum['mean_bps'], fa_sum['win_rate']*100\n"
            "else:\n"
            f"    fm, fw = {R['follow_mean']}, {R['follow_win']}\n"
            f"    rm, rw = {R['rand_mean']}, {R['rand_win']}\n"
            f"    fam, faw = {R['fade_mean']}, {R['fade_win']}\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.5))\n"
            "bars_ = ax.bar(\n"
            "    ['Follow morning\\n(the claim)', 'Random direction\\n(coin)', 'Fade morning\\n(contrarian)'],\n"
            "    [fm, rm, fam], color=[RED, GREY, AMBER], width=0.55\n"
            ")\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('gross profit per session (bps)')\n"
            "ax.set_title('The claim loses to a coin; the opposite trade wins')\n"
            "for b_, w_ in zip(bars_, [fw, rw, faw]):\n"
            "    ax.annotate(f'win {w_:.1f}%', (b_.get_x()+b_.get_width()/2, 0),\n"
            "                ha='center', va='bottom' if b_.get_height() < 0 else 'top', fontsize=9)\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'Follow: {{fm:+.2f}} bps  |  Coin: {{rm:+.2f}} bps  |  Fade: {{fam:+.2f}} bps')"
        ),
        md(
            f"The follow-morning arm earns **{R['follow_mean']:+.2f} bps/session** -- "
            f"significantly *negative* (the last bar reverses the morning). A coin loses only "
            f"**{R['rand_mean']:+.2f} bps**. Going *against* the morning (the fade) earns "
            f"**+{R['fade_mean']:+.2f} bps**. The 'power hour continuation' story is not just "
            "absent -- it is directionally backwards.\n\n"
            "> Why? Near the close, institutional rebalancing, ETF creation/redemption "
            "pressure, and index-reconstitution flows tend to *absorb* the day's trend, "
            "creating mean-reversion pressure rather than continuation."
        ),

        # ---- BEAT 5 -- THE VERDICT -------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            f"- **Signal -- MIXED.** The continuation claim is clearly wrong (follow t = "
            f"{R['follow_t']:+.2f}). A *reversal* signal (fade) is statistically real "
            f"(t = {R['fade_t']:+.2f}, gross {R['fade_mean']:+.2f} bps), but it is the "
            "opposite of what the 'power hour' story teaches.\n"
            "- **Tradability -- FRAGILE.** The fade edge survives 0.5 bps round-trip "
            f"(t = {R['fade_t05']:+.2f}) but not 1 bps (t = {R['fade_t10']:+.2f}). "
            "Breakeven ~1 bps: tight for retail, marginal even for institutions.\n"
            "- **Continuation claim -- BUSTED.** Following the morning's direction into the "
            "power hour is a statistically significant way to lose money, not make it."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT? ----------------------------------
        md(
            "## 6 - Could you actually trade it?\n\n"
            "Let's take the *best* version -- the fade arm that actually has a gross edge -- "
            "and sweep the cost:"
        ),
        code(
            "costs = [0.0, 0.5, 1.0, 2.0, 5.0]\n"
            "if HAVE_REAL:\n"
            "    net_means = [st.summarize(pool_arm('fade', cost=c), 'ret_net')['mean_bps'] for c in costs]\n"
            "    net_t = [st.summarize(pool_arm('fade', cost=c), 'ret_net')['tstat'] for c in costs]\n"
            "else:\n"
            f"    net_means = [{R['fade_net00']},{R['fade_net05']},{R['fade_net10']},{R['fade_net20']},-3.17]\n"
            f"    net_t = [{R['fade_t00']},{R['fade_t05']},{R['fade_t10']},{R['fade_t20']},-6.26]\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))\n"
            "axes[0].plot(costs, net_means, 'o-', c=AMBER, lw=2)\n"
            "axes[0].axhline(0, c='k', lw=1)\n"
            "axes[0].fill_between(costs, net_means, 0, where=[n<0 for n in net_means], color=RED, alpha=.15)\n"
            "axes[0].fill_between(costs, net_means, 0, where=[n>0 for n in net_means], color=GREEN, alpha=.15)\n"
            "axes[0].set_xlabel('round-trip cost (bps)'); axes[0].set_ylabel('net mean (bps/session)')\n"
            "axes[0].set_title('Fade: breakeven near 1 bps')\n"
            "axes[1].plot(costs, net_t, 's--', c=GREY, lw=1.5)\n"
            "axes[1].axhline(2, ls=':', c=GREEN, lw=1, label='t=2 bar')\n"
            "axes[1].axhline(-2, ls=':', c=RED, lw=1); axes[1].axhline(0, c='k', lw=0.5)\n"
            "axes[1].set_xlabel('round-trip cost (bps)'); axes[1].set_ylabel('HAC t-stat')\n"
            "axes[1].set_title('Signal significance dissolves above ~1 bps')\n"
            "axes[1].legend(); plt.tight_layout(); plt.show()\n"
            "print('At 1 bps cost the fade t-stat falls below 2 -- significance lost for most traders.')"
        ),
        md(
            "The fade grosses +1.83 bps per session. A realistic round-trip for retail "
            "is 2-5 bps (spread + commission); institutional execution might achieve 0.5-1 bps. "
            "Only the latter bracket survives -- and even then the t-stat barely clears 2 "
            f"at 0.5 bps ({R['fade_t05']:+.2f}). This is a **FRAGILE** edge: "
            "thin, horizon-specific, and likely to disappear if even a small number of "
            "traders attempt to exploit it."
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 - Going further\n\n"
            "- **The original academic result.** Gao, Han, Li & Zhou (2018), *Market Intraday "
            "Momentum* (JFE), tested *first half-hour* -> *last half-hour* on SPY specifically. "
            "Our study uses a longer window (morning) which may mix the signal. Re-running with "
            "only the first 30 minutes as the predictor is a natural fork.\n"
            "- **The related intraday studies.** [Study 13 -- Crimson-Hour](../../13-crimson-hour/) "
            "decomposes the opening-candle claim; [Study 87 -- Center-Line](../../87-center-line/) "
            "tests intraday reference-level momentum. Both are related methodologically.\n"
            "- **The synthetic positive control** (in the companion notebook) confirms the engine "
            "detects continuation when it is planted: the null verdict is a statement about the "
            "*market*, not the method.\n\n"
            "*Think the continuation claim survives on a different sub-sample or timeframe? Fork "
            "this, use only the 09:30-10:00 window as the predictor, or test overnight futures "
            "sessions. That is the bar.*"
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
            "# Power-Hour -- a quantitative teardown\n"
            "### 1-hour bars * session panel * Pearson corr * random-direction control * "
            "HAC inference * cost sweep\n\n"
            "![Signal: MIXED](https://img.shields.io/badge/Signal-MIXED-dab617?style=flat-square)\n"
            "![Tradability: FRAGILE](https://img.shields.io/badge/Tradability-FRAGILE-dab617?style=flat-square)\n"
            "![Continuation_claim%3F: BUSTED](https://img.shields.io/badge/Continuation_claim%3F-BUSTED-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) "
            "-- same seven beats, every claim now carrying its standard error. We decompose "
            "the 'power hour continuation' claim into (i) a correlation test of whether "
            "morning_ret predicts last_ret, and (ii) a follow-vs-random backtest on a "
            "per-session ledger -- both on ~723 sessions of 1-hour SPY/QQQ/IWM data.\n\n"
            "> **Not investment advice.** Real data: Yahoo 1-hour bars, ~730-session window, "
            "as-of 2026-06-13; offline core and tests run on a deterministic synthetic tape. "
            "Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **'> in plain words' notes** translate each result back to intuition."
        ),
        code(BOOT + "\nfrom quantlab import analytics, stats\n"),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `MIXED` | Follow-morning gross **{R['follow_mean']:+.2f} bps/session**, "
            f"HAC t = **{R['follow_t']:+.2f}** (significantly *negative*). Fade (contrarian) "
            f"earns **{R['fade_mean']:+.2f} bps**, t = **{R['fade_t']:+.2f}**. |\n"
            f"| **Tradability** | `FRAGILE` | Fade breakeven ~1 bps round-trip; at 0.5 bps "
            f"t = {R['fade_t05']:+.2f}, at 1 bps t = {R['fade_t10']:+.2f} (below the inference bar). |\n"
            f"| **Continuation?** | `BUSTED` | Pool corr = {R['corr_pool']:+.4f} "
            f"(t = {R['t_pool']:+.2f}); IWM corr = {R['corr_iwm']:+.4f} "
            f"(t = {R['t_iwm']:+.2f}) -- directionally *reversed*. |\n\n"
            "> 'in plain words' The last bar doesn't follow the day -- it tends to reverse it. "
            "The power-hour continuation story is not merely absent, it is the wrong direction."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 - The claim, steelmanned\n\n"
            r"Let $m$ = morning_ret (09:30 open -> 14:30 close) and $l$ = last_ret "
            r"(15:30 open -> 15:30 close). The 'power hour' recipe asserts:"
            "\n\n"
            r"- **H1 (continuation).** $\rho(m, l) > 0$ -- the morning and last bar are "
            r"positively correlated. Following $\mathrm{sign}(m)$ in the last bar earns "
            "a positive expected return.\n"
            r"- **H2 (beats random).** $\mathbb{E}[\mathrm{sign}(m) \cdot l] > \mathbb{E}[\epsilon \cdot l]$ "
            r"where $\epsilon \sim \pm 1$ i.i.d. -- the morning-direction signal beats a coin.\n\n"
            "We reject H1 on SPY and QQQ (corr indistinguishable from 0), marginally on IWM "
            "(corr = -0.086, *negative*), and reject H2 strongly (pooled follow t = -3.62). "
            "The data supports a *reversal* alternative: fade t = +3.62."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 - So what?\n\n"
            "The power-hour frame is ubiquitous in retail day-trading content. If H1-H2 held, "
            "it would be a clean, once-a-day signal with a 3-year track record. The failure "
            "is directional -- the continuation story is not merely noisy, it predicts the "
            "wrong sign. The interesting question is *why*: near-close institutional "
            "rebalancing, ETF creation/redemption, and index-level MOC order flows are all "
            "known to generate end-of-day *reversal* pressure, which is what the data shows."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 - How we'd know -- the protocol\n\n"
            "- **Session split.** `morning_ret` = open of 09:30 bar -> close of 14:30 bar; "
            "`last_ret` = open of 15:30 bar -> close of 15:30 bar. Signal is formed *before* "
            "the last bar opens (no look-ahead).\n"
            "- **Correlation test.** Pearson $r$ and Fisher-z t-stat (n-2 dof), per ticker "
            "and pooled. Null: $\\rho = 0$.\n"
            "- **Trade ledger.** Enter at 15:30 open in the signal direction, exit at 15:30 "
            "close. Arms: `follow` ($d = \\mathrm{sign}(m)$), `random` ($d \\sim \\pm 1$ i.i.d.), "
            "`fade` ($d = -\\mathrm{sign}(m)$). "
            "Inference: Newey-West HAC t-stat on per-session returns.\n"
            "- **Data.** Yahoo 1-hour bars, SPY/QQQ/IWM, ~723 sessions each (2023-07-18 to "
            "2026-06-12). One trade per session per ticker; pooled n = 2,169.\n"
            "- **Positive control.** Synthetic tape with tunable `continuation` knob: "
            "confirms the engine detects the signal when planted."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 - The teardown"),
        md(
            "### 4a - Per-instrument correlation\n\n"
            "Pearson correlation between morning_ret and last_ret. If the continuation claim "
            "is right, every bar should clear +2."
        ),
        code(
            "if HAVE_REAL:\n"
            "    recs = []\n"
            "    for t in TICKERS:\n"
            "        panel = data.fetch_daily_panel(t, fetch=False)\n"
            "        c = st.morning_last_corr(panel)\n"
            "        recs.append((t, c['n'], c['corr'], c['tstat']))\n"
            "    all_p = pd.concat([data.fetch_daily_panel(t) for t in TICKERS])\n"
            "    cp = st.morning_last_corr(all_p)\n"
            "    recs.append(('Pool', cp['n'], cp['corr'], cp['tstat']))\n"
            "    cdf = pd.DataFrame(recs, columns=['ticker','n','corr','t'])\n"
            "else:\n"
            f"    cdf = pd.DataFrame({{\n"
            f"        'ticker': ['SPY','QQQ','IWM','Pool'],\n"
            f"        'n': [{R['n_spy']},{R['n_qqq']},{R['n_iwm']},{R['n_pool']}],\n"
            f"        'corr': [{R['corr_spy']},{R['corr_qqq']},{R['corr_iwm']},{R['corr_pool']}],\n"
            f"        't': [{R['t_spy']},{R['t_qqq']},{R['t_iwm']},{R['t_pool']}]\n"
            f"    }})\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "cols2 = [RED if v<-2 else AMBER if abs(v)<2 else GREEN for v in cdf['t']]\n"
            "ax.bar(cdf['ticker'], cdf['t'], color=cols2)\n"
            "for s in (2,-2): ax.axhline(s, ls='--', c=GREY, lw=1)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('Pearson t-stat (morning_ret vs last_ret)')\n"
            "ax.set_title('Continuation claim: t-stats mostly negative -- the wrong direction')\n"
            "plt.tight_layout(); plt.show()\n"
            "cdf.round(4)"
        ),
        md(
            "> 'in plain words' A bar *above* zero means the morning predicts the same-direction "
            "close (continuation). All three bars are at or *below* zero -- IWM is significantly "
            "negative (t = -2.31), meaning the last bar tends to *reverse* the morning. "
            "The continuation claim fails its first test."
        ),
        md(
            "### 4b - Follow-morning vs random-direction control\n\n"
            "Per-session gross return of the follow arm versus a random-direction coin on "
            "the same sessions. The fade (contrarian) arm is shown as a bonus."
        ),
        code(
            "if HAVE_REAL:\n"
            "    f_led = pool_arm('follow', cost=0.0)\n"
            "    r_led = pool_arm('random', cost=0.0)\n"
            "    fa_led = pool_arm('fade', cost=0.0)\n"
            "    f_s = st.summarize(f_led, 'ret_gross')\n"
            "    r_s = st.summarize(r_led, 'ret_gross')\n"
            "    fa_s = st.summarize(fa_led, 'ret_gross')\n"
            "    fm2, ft2 = f_s['mean_bps'], f_s['tstat']\n"
            "    rm2, rt2 = r_s['mean_bps'], r_s['tstat']\n"
            "    fam2, fat2 = fa_s['mean_bps'], fa_s['tstat']\n"
            "else:\n"
            f"    fm2, ft2 = {R['follow_mean']}, {R['follow_t']}\n"
            f"    rm2, rt2 = {R['rand_mean']}, {R['rand_t']}\n"
            f"    fam2, fat2 = {R['fade_mean']}, {R['fade_t']}\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.5))\n"
            "labels3 = ['Follow\\n(claim)', 'Random\\n(coin)', 'Fade\\n(contrarian)']\n"
            "means3 = [fm2, rm2, fam2]\n"
            "tstats3 = [ft2, rt2, fat2]\n"
            "cols3 = [RED, GREY, AMBER]\n"
            "bars3 = ax.bar(labels3, means3, color=cols3, width=0.55)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('gross profit per session (bps)')\n"
            "ax.set_title('Follow loses significantly; fade earns -- opposite of the claim')\n"
            "for b3, t3 in zip(bars3, tstats3):\n"
            "    ax.annotate(f't={t3:+.2f}', (b3.get_x()+b3.get_width()/2,\n"
            "                b3.get_height() if b3.get_height()>0 else b3.get_height()-0.25),\n"
            "                ha='center', va='bottom', fontsize=9)\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'Follow: {{fm2:+.2f}} bps (t={{ft2:+.2f}})  |  "
            f"Random: {{rm2:+.2f}} (t={{rt2:+.2f}})  |  Fade: {{fam2:+.2f}} (t={{fat2:+.2f}})')"
        ),
        md(
            f"> 'in plain words' The signal is real -- the issue is direction. The follow "
            f"arm loses **{R['follow_mean']:+.2f} bps/session** at HAC t = **{R['follow_t']:+.2f}** "
            f"(three sigma in the *wrong* direction). The fade earns **+{R['fade_mean']:.2f} bps** "
            f"at t = **{R['fade_t']:+.2f}**. This is *not* a NONE result -- it is a MIXED result "
            "where the direction of the effect is the opposite of the folk claim."
        ),
        md(
            "### 4c - Per-instrument breakdown (follow arm)\n\n"
            "All three ETFs show negative follow means; QQQ and IWM are individually significant."
        ),
        code(
            "if HAVE_REAL:\n"
            "    per_inst = []\n"
            "    for t in TICKERS:\n"
            "        panel = data.fetch_daily_panel(t, fetch=False)\n"
            "        s = st.summarize(st.build_ledger(panel, arm='follow', cost_bps=0), 'ret_gross')\n"
            "        per_inst.append((t, s['n'], s['win_rate']*100, s['mean_bps'], s['tstat']))\n"
            "    pi_df = pd.DataFrame(per_inst, columns=['ticker','n','win%','mean_bps','t'])\n"
            "else:\n"
            f"    pi_df = pd.DataFrame({{\n"
            f"        'ticker': ['SPY','QQQ','IWM'],\n"
            f"        'n': [{R['n_spy']},{R['n_qqq']},{R['n_iwm']}],\n"
            f"        'win%': [46.5, 46.7, 49.1],\n"
            f"        'mean_bps': [{R['spy_mean']},{R['qqq_mean']},{R['iwm_mean']}],\n"
            f"        't': [{R['spy_t']},{R['qqq_t']},{R['iwm_t']}]\n"
            f"    }})\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.0))\n"
            "cols4 = [RED if abs(v)>2 else AMBER for v in pi_df['t']]\n"
            "ax.bar(pi_df['ticker'], pi_df['t'], color=cols4)\n"
            "for s in (2,-2): ax.axhline(s, ls='--', c=GREY, lw=1)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('HAC t-stat (follow arm, gross)'); ax.set_ylim(-3.5, 1)\n"
            "ax.set_title('All three ETFs: follow arm t-stat consistently negative')\n"
            "plt.tight_layout(); plt.show()\n"
            "pi_df.round(2)"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            f"- **Signal `MIXED`** -- Follow-morning: {R['follow_mean']:+.2f} bps, "
            f"t = {R['follow_t']:+.2f}. Fade: {R['fade_mean']:+.2f} bps, "
            f"t = {R['fade_t']:+.2f}. Pool corr = {R['corr_pool']:+.4f}, "
            f"IWM corr = {R['corr_iwm']:+.4f} (reversal, not continuation).\n"
            f"- **Tradability `FRAGILE`** -- Fade gross +1.83 bps; breakeven ~1 bps round-trip; "
            f"t at 0.5 bps = {R['fade_t05']:+.2f}, at 1 bps = {R['fade_t10']:+.2f}.\n"
            "- **Continuation claim `BUSTED`** -- The morning direction predicts the *opposite* "
            "of the last bar, not the same direction."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 - Could you trade it? -- the fade cost sweep\n\n"
            "The fade arm is the only arm with a positive gross edge. Its cost curve:"
        ),
        code(
            "costs2 = [0.0, 0.5, 1.0, 2.0, 5.0]\n"
            "if HAVE_REAL:\n"
            "    sw2 = [st.summarize(pool_arm('fade', cost=c), 'ret_net') for c in costs2]\n"
            "    mn2 = [s['mean_bps'] for s in sw2]\n"
            "    ts2 = [s['tstat'] for s in sw2]\n"
            "else:\n"
            f"    mn2 = [{R['fade_net00']},{R['fade_net05']},{R['fade_net10']},{R['fade_net20']},-3.17]\n"
            f"    ts2 = [{R['fade_t00']},{R['fade_t05']},{R['fade_t10']},{R['fade_t20']},-6.26]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(costs2, mn2, 'o-', c=AMBER, lw=2, label='net mean (bps)')\n"
            "ax2b = ax.twinx()\n"
            "ax2b.plot(costs2, ts2, 's--', c=GREY, lw=1.5, label='HAC t')\n"
            "ax2b.axhline(2, ls=':', c=GREEN, lw=1); ax2b.axhline(-2, ls=':', c=RED, lw=1)\n"
            "ax2b.axhline(0, ls='-', c='k', lw=0.5)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('round-trip cost (bps)'); ax.set_ylabel('net mean (bps)', color=AMBER)\n"
            "ax2b.set_ylabel('HAC t-stat', color=GREY)\n"
            "ax.set_title('Fade: edge survives ~0.5 bps; lost by 1 bps for most traders')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Breakeven round-trip: ~1.0-1.5 bps (institutional-only territory)')"
        ),
        md(
            "> 'in plain words' The fade grosses +1.83 bps per session, but a 1 bps "
            "round-trip (tight even for institutions) cuts the t-stat to 1.64 -- below the "
            "inference bar. This is what FRAGILE means: the effect is real, but it exists "
            "only in a cost window that most traders cannot access."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 - Going further -- the positive control\n\n"
            "The synthetic `continuation` knob confirms the engine correctly detects "
            "continuation when it is planted. The real-tape verdict is about the *market*, "
            "not the method."
        ),
        code(
            "conts = [-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30]\n"
            "follow_edge = []\n"
            "for cont in conts:\n"
            "    panel, _ = data.synthetic_sessions(n_sessions=500, continuation=cont, seed=116)\n"
            "    f_s2 = st.summarize(st.build_ledger(panel, arm='follow', cost_bps=0), 'ret_gross')['mean_bps']\n"
            "    r_s2 = st.summarize(st.build_ledger(panel, arm='random', cost_bps=0, seed=1), 'ret_gross')['mean_bps']\n"
            "    follow_edge.append(f_s2 - r_s2)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(conts, follow_edge, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(0, c='k', lw=1); ax.axvline(0, ls='--', c=GREY)\n"
            "ax.set_xlabel('planted continuation coefficient')\n"
            "ax.set_ylabel('follow minus coin (bps/session)')\n"
            "ax.set_title('Engine works: follows wins when continuation > 0, loses when < 0')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('At continuation=0 the edge is ~0; positive continuation -> follow wins; negative -> follow loses.')"
        ),
        md(
            "The engine is a faithful detector: the follow-morning advantage is monotone in "
            "the planted continuation, crosses zero at the null, and goes negative under "
            "reversal (exactly what the real tape shows). The real-tape verdict is therefore "
            "a statement about the market: the last regular-session 1-hour bar on SPY/QQQ/IWM "
            "carries a *reversal* tendency, not a continuation one. Forks: restrict to "
            "first-half-hour only (the Gao et al. specification), test futures instead of "
            "ETFs, or look at ex-dividend days separately."
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
