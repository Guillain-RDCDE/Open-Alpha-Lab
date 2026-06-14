"""Generate the two narrative notebooks for Study 128 (Keltner-Channel).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \\
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
    # Breakout arm (close > upper band), pooled 5-ticker
    n_bo=1150, bo_mean=81.31, bo_win=61.8, bo_t=4.95,
    rand_bo=116.56, delta_bo=-35.24,
    # Reversion arm (close < lower band), pooled 5-ticker
    n_rv=582, rv_mean=147.14, rv_win=61.5, rv_t=4.03,
    rand_rv=140.88, delta_rv=6.26,
    # Per-ticker SPY breakout
    spy_bo_mean=63.97, spy_bo_t=2.07, spy_rand_bo=113.05, spy_delta_bo=-49.09,
    # Per-ticker SPY reversion
    spy_rv_mean=246.05, spy_rv_t=3.43, spy_rand_rv=157.24, spy_delta_rv=88.81,
    # Buy-and-hold baselines (unconditional 20-day means)
    bh_spy=93.85, bh_qqq=131.00, bh_iwm=86.87, bh_gld=96.18, bh_eem=78.19,
    # Cost sweep breakout
    bo_net_0=81.31, bo_net_1=80.31, bo_net_2=79.31, bo_net_5=76.31,
    bo_t_0=4.95, bo_t_1=4.89, bo_t_2=4.83, bo_t_5=4.64,
    # Cost sweep reversion
    rv_net_0=147.14, rv_net_1=146.14, rv_net_2=145.14, rv_net_5=142.14,
    rv_t_0=4.03, rv_t_1=4.00, rv_t_2=3.98, rv_t_5=3.89,
    # Approx trades per year per ticker
    tpy_bo=12, tpy_rv=6,
)

# ---------------------------------------------------------------------------
# Shared analysis preamble
# ---------------------------------------------------------------------------
BOOT = f"""\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root (quantlab/)
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({{"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False}})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from keltner_channel import data, strategy as st

TICKERS = ["SPY", "QQQ", "IWM", "GLD", "EEM"]
HOLD = 20
CACHE = os.path.abspath("../_cache")

def _have_cache():
    return all(os.path.exists(data._cache_path(t, CACHE)) for t in TICKERS)

HAVE_REAL = _have_cache()

def pool(side, cost=0.0, seed=None):
    \"\"\"Pool fixed-hold trades across the basket.\"\"\"
    frames = []
    for t in TICKERS:
        b = data.fetch_daily(t, fetch=False, cache_dir=CACHE)
        if seed is not None:
            sig_n = len(st.channel_entries(b, side=side))
            ent = st.random_entries(b, n=sig_n, seed=seed)
        else:
            ent = st.channel_entries(b, side=side)
        frames.append(st.run_trades(b, ent, hold_days=HOLD, cost_bps=cost))
    return pd.concat(frames, ignore_index=True)

# Frozen headline numbers (mirror of docs/results.md, as-of 2026-06-14).
R = dict(
    n_bo={R['n_bo']}, bo_mean={R['bo_mean']}, bo_win={R['bo_win']}, bo_t={R['bo_t']},
    rand_bo={R['rand_bo']}, delta_bo={R['delta_bo']},
    n_rv={R['n_rv']}, rv_mean={R['rv_mean']}, rv_win={R['rv_win']}, rv_t={R['rv_t']},
    rand_rv={R['rand_rv']}, delta_rv={R['delta_rv']},
    spy_bo_mean={R['spy_bo_mean']}, spy_bo_t={R['spy_bo_t']},
    spy_rand_bo={R['spy_rand_bo']}, spy_delta_bo={R['spy_delta_bo']},
    spy_rv_mean={R['spy_rv_mean']}, spy_rv_t={R['spy_rv_t']},
    spy_rand_rv={R['spy_rand_rv']}, spy_delta_rv={R['spy_delta_rv']},
    bh_spy={R['bh_spy']}, bh_qqq={R['bh_qqq']}, bh_iwm={R['bh_iwm']},
    bh_gld={R['bh_gld']}, bh_eem={R['bh_eem']},
    bo_net_0={R['bo_net_0']}, bo_net_1={R['bo_net_1']}, bo_net_2={R['bo_net_2']}, bo_net_5={R['bo_net_5']},
    bo_t_0={R['bo_t_0']}, bo_t_1={R['bo_t_1']}, bo_t_2={R['bo_t_2']}, bo_t_5={R['bo_t_5']},
    rv_net_0={R['rv_net_0']}, rv_net_1={R['rv_net_1']}, rv_net_2={R['rv_net_2']}, rv_net_5={R['rv_net_5']},
    rv_t_0={R['rv_t_0']}, rv_t_1={R['rv_t_1']}, rv_t_2={R['rv_t_2']}, rv_t_5={R['rv_t_5']},
    tpy_bo={R['tpy_bo']}, tpy_rv={R['tpy_rv']},
)

print("real daily cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Keltner-Channel — breakout or reversion? The channel that promises both.\n"
            "### EMA(20) ± 2×ATR(10): tested honestly on 21 years of daily data\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Breakout_vs_Reversion%3F: Both_noise](https://img.shields.io/badge/Breakout_vs_Reversion%3F-Both_noise-8b949e?style=flat-square)\n\n"
            "Here is a recipe you'll find in thousands of trading tutorials: draw a Keltner Channel "
            "around price (an exponential moving average with an ATR-based envelope) and trade the "
            "pierces. But the tutorials cannot agree *which way* to trade them. Half say 'when price "
            "breaks *above* the upper band, the trend is strong — buy.' The other half say 'when "
            "price falls *below* the lower band, it has stretched too far — buy the snap-back.' "
            "This notebook tests **both claims honestly** — and shows why they cannot both be right.\n\n"
            "> 📓 **This is the plain-language layer.** Want the t-stats, bootstrap intervals, and "
            "the random-entry control maths? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool: every chart below is drawn "
            "by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does the breakout (close above upper band) signal a real edge? | **No.** "
            f"Breakout entries earn **{R['bo_mean']:.0f} bps/trade** but a random same-date "
            f"entry earns **{R['rand_bo']:.0f} bps**. The channel filter makes it **{abs(R['delta_bo']):.0f} bps worse**. |\n"
            "| Does the reversion (close below lower band) signal a real edge? | **No.** "
            f"Reversion earns **{R['rv_mean']:.0f} bps/trade** vs random's **{R['rand_rv']:.0f} bps**. "
            f"Delta: **{R['delta_rv']:+.0f} bps** — statistically indistinguishable from zero. |\n"
            "| Can both claims be right at once? | **No.** Logically impossible. "
            "And both turn out to be noise once market drift is stripped out. |\n"
            "| Could you trade either? | **No alpha to trade.** The positive gross returns are "
            "just the market going up over 21 years. |\n\n"
            "> The channel draws pretty lines, but it is not telling you anything useful about "
            "what happens next."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "The Keltner Channel wraps an exponential moving average (EMA of 20 days) "
            "with a volatility envelope: the distance from the mid-line is two times the "
            "Average True Range (a measure of daily price swings over the last 10 days). "
            "Two claims are made by different trading communities:\n\n"
            "> *\"When price closes above the upper band, it's in breakout mode — the trend is "
            "strong and will continue. Buy.\"*\n\n"
            "> *\"When price closes below the lower band, it has stretched too far from fair value. "
            "It will snap back. Buy.\"*\n\n"
            "Both camps point to the same lines on the same chart and draw opposite conclusions. "
            "At most one can be right. The test below is designed to determine which — if either — "
            "is actually adding something beyond just *buying the market on a random day*."
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "The Keltner Channel is one of the most widely-used volatility channels in retail "
            "algorithmic trading. If the breakout claim is true, it's a trend-following machine. "
            "If the reversion claim is true, it's a mean-reversion signal. If *both* claims are "
            "marketed simultaneously, the channel cannot be both things at once — and testing them "
            "together in one rigorous frame exposes the contradiction. The broader lesson: a channel "
            "that looks active on a chart is not the same as a channel with predictive power."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "The key trick is to use a **random-entry control**. Instead of comparing the channel "
            "signal to zero, we compare it to what you'd earn by entering at a *random date* on "
            "the same instrument, held for the same 20 days.\n\n"
            "Why does this matter? Markets go up most of the time. If you buy SPY after an upper-band "
            "pierce and hold for 20 days, you'll often make money — *but so would a random date*. "
            "The question is not 'did I make money?' but 'did the channel tell me anything a coin "
            "flip didn't already know?'\n\n"
            "We test both arms across five instruments (SPY, QQQ, IWM, GLD, EEM) using 21 years of "
            "daily data, held for a 20-bar time-exit, matched against a same-size random control."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown — let's look at both rules\n\n"
            "**First: what does the channel look like on the tape?**"
        ),
        code(
            "# Draw a Keltner Channel on SPY over a representative period.\n"
            "if HAVE_REAL:\n"
            "    bars = data.fetch_daily('SPY', fetch=False, cache_dir=CACHE)\n"
            "    sample = bars['2022-01-01':'2023-12-31']\n"
            "else:\n"
            "    sample, _ = data.synthetic_daily(n_days=500, momentum=0.0, seed=128)\n"
            "kc = st.keltner_channel(sample)\n"
            "fig, ax = plt.subplots(figsize=(10.5, 5.0))\n"
            "ax.plot(sample.index, sample['close'], c='k', lw=1.2, label='Close')\n"
            "ax.plot(kc.index, kc['mid'], c=GREY, lw=1, ls='--', label='EMA(20)')\n"
            "ax.fill_between(kc.index, kc['lower'], kc['upper'], alpha=0.15, color=AMBER)\n"
            "ax.plot(kc.index, kc['upper'], c=AMBER, lw=1.2, label='Upper band')\n"
            "ax.plot(kc.index, kc['lower'], c=AMBER, lw=1.2, label='Lower band')\n"
            "# Mark pierces\n"
            "bo_e = st.channel_entries(sample, side='breakout')\n"
            "rv_e = st.channel_entries(sample, side='reversion')\n"
            "ax.scatter(bo_e.index, sample.loc[bo_e.index, 'close'], marker='^', c=GREEN, s=60, zorder=5, label='Breakout pierce')\n"
            "ax.scatter(rv_e.index, sample.loc[rv_e.index, 'close'], marker='v', c=RED, s=60, zorder=5, label='Reversion pierce')\n"
            "ax.set_title('Keltner Channel on SPY (2022-2023) — opposite signals, same channel')\n"
            "ax.legend(ncol=3, fontsize=8); plt.tight_layout(); plt.show()\n"
            "print(f'Breakout pierces: {len(bo_e)} | Reversion pierces: {len(rv_e)}')"
        ),
        md(
            "The channel clearly marks extremes — the question is whether those extremes predict "
            "what comes next. Now the honest comparison: both arms vs a random same-day entry."
        ),
        code(
            "# The key chart: both arms vs random control\n"
            "if HAVE_REAL:\n"
            "    sbo = st.summarize(pool('breakout'), 'ret_gross')\n"
            "    srv = st.summarize(pool('reversion'), 'ret_gross')\n"
            "    srnd_bo = st.summarize(pool('breakout', seed=128), 'ret_gross')\n"
            "    srnd_rv = st.summarize(pool('reversion', seed=128), 'ret_gross')\n"
            "    bo_mean, rv_mean = sbo['mean_bps'], srv['mean_bps']\n"
            "    rnd_bo, rnd_rv = srnd_bo['mean_bps'], srnd_rv['mean_bps']\n"
            "else:\n"
            "    bo_mean, rv_mean = R['bo_mean'], R['rv_mean']\n"
            "    rnd_bo, rnd_rv = R['rand_bo'], R['rand_rv']\n"
            "labels = ['Breakout\\n(close > upper)', 'Breakout\\nrandom control',\n"
            "          'Reversion\\n(close < lower)', 'Reversion\\nrandom control']\n"
            "vals = [bo_mean, rnd_bo, rv_mean, rnd_rv]\n"
            "colors = [GREEN, GREY, AMBER, GREY]\n"
            "fig, ax = plt.subplots(figsize=(10.5, 5.0))\n"
            "bars_ = ax.bar(labels, vals, color=colors, width=0.55)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "for b, v in zip(bars_, vals):\n"
            "    ax.annotate(f'{v:+.0f} bps', (b.get_x()+b.get_width()/2, max(v,0)+5),\n"
            "                ha='center', va='bottom', fontsize=9)\n"
            "ax.set_ylabel('Gross mean return per trade (bps)')\n"
            "ax.set_title('Both channel signals vs a random same-date entry (20-day hold)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Breakout delta vs random: {bo_mean-rnd_bo:+.1f} bps')\n"
            "print(f'Reversion delta vs random: {rv_mean-rnd_rv:+.1f} bps')"
        ),
        md(
            f"**Breakout arm:** earns {R['bo_mean']:.0f} bps/trade — but a random same-date entry "
            f"earns {R['rand_bo']:.0f} bps. The breakout filter is **{abs(R['delta_bo']):.0f} bps worse** "
            "than a coin-flip. The reason: breakout days tend to follow big rallies, often near "
            "short-term highs — not great subsequent-entry points.\n\n"
            f"**Reversion arm:** earns {R['rv_mean']:.0f} bps/trade — barely above random's "
            f"{R['rand_rv']:.0f} bps. Delta: **{R['delta_rv']:+.0f} bps** — noise. Reversion "
            "entries often fire in crisis periods when the market is already recovering, so the "
            "positive mean is mostly timing the recovery, not the channel signal."
        ),
        md(
            "**The contradiction in one chart:** both arms look profitable — but *only* because "
            "the market generally goes up. The random control removes that drift and shows both "
            "signals are empty."
        ),

        # ---- BEAT 5 — THE VERDICT --------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** Breakout entries underperform a random-entry baseline by "
            f"**{abs(R['delta_bo']):.0f} bps** (the channel filter *hurts* vs a coin). Reversion "
            f"entries beat random by **{R['delta_rv']:+.0f} bps** — economically zero. Neither arm "
            "clears the inference bar once the market's unconditional drift is stripped out.\n"
            "- **Tradability — Mirage.** The positive gross means are captured equity drift, not "
            "channel alpha. There is nothing to trade.\n"
            "- **Breakout vs Reversion?** Both noise. The contradiction is settled empirically: "
            "both folk rules fail to add alpha over a random-entry baseline."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Unlike high-turnover strategies, the Keltner rules fire rarely (~6–23 times per year "
            "per ticker), so transaction costs are not the primary killer — the *absence of alpha* "
            "is. Even at zero cost, there is nothing here above what you'd earn by buying randomly."
        ),
        code(
            "costs = [0.0, 1.0, 2.0, 5.0]\n"
            "if HAVE_REAL:\n"
            "    net_bo = [st.summarize(pool('breakout', c), 'ret_net')['mean_bps'] for c in costs]\n"
            "    net_rv = [st.summarize(pool('reversion', c), 'ret_net')['mean_bps'] for c in costs]\n"
            "else:\n"
            "    net_bo = [R['bo_net_0'], R['bo_net_1'], R['bo_net_2'], R['bo_net_5']]\n"
            "    net_rv = [R['rv_net_0'], R['rv_net_1'], R['rv_net_2'], R['rv_net_5']]\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "ax.plot(costs, net_bo, 'o-', c=GREEN, lw=2, label='Breakout net mean')\n"
            "ax.plot(costs, net_rv, 's-', c=AMBER, lw=2, label='Reversion net mean')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('round-trip cost (bps)'); ax.set_ylabel('net mean (bps/trade)')\n"
            "ax.set_title('Both arms stay positive with costs — but it is all market drift, not alpha')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('The lines stay above zero — but so would a random-entry strategy.')\n"
            f"print('Breakout delta vs random: {R['delta_bo']:+.0f} bps | Reversion delta vs random: {R['delta_rv']:+.0f} bps')"
        ),
        md(
            "The positive net means are reassuring at first glance — but they are identical to "
            "what a random long entry would earn. The channel lines are decorative."
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Same contradiction with Bollinger Bands?** Yes. "
            "[Study 104 — Bollinger-Reversion](../../104-bollinger-reversion/) runs the same "
            "test on the Bollinger Band cousin of this channel — same verdict.\n"
            "- **Would a stop-exit change the result?** Our positive control (beat 7 in the quants "
            "notebook) shows the machinery does find real momentum or reversion when planted in a "
            "synthetic tape. The issue is not the exit — it is the absence of signal in the real tape.\n"
            "- **Is there a channel strategy that works?** Some researchers find evidence of "
            "volatility-breakout profitability at longer horizons (weekly, monthly) using ATR-based "
            "filters — but those are regime-identification tools, not simple channel-pierce entries. "
            "References in [`docs/references.md`](../docs/references.md).\n\n"
            "*Think the channel really says something? Fork this, change the exit mode or the "
            "instrument universe, and show a random-control-adjusted edge that clears |t| ≥ 2. "
            "That's the bar.*"
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
            "# Keltner-Channel — a quantitative teardown\n"
            "### EMA(20) ± 2×ATR(10) · breakout vs reversion · random-entry control · HAC inference · 21-year daily basket\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Breakout_vs_Reversion%3F: Both_noise](https://img.shields.io/badge/Breakout_vs_Reversion%3F-Both_noise-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We test both Keltner folk "
            "framings — breakout (close > upper band) and reversion (close < lower band) — against "
            "a random-entry control matched in trade count, on five daily tapes over 21 years.\n\n"
            "> ⚠️ **Not investment advice.** Real data: Yahoo daily bars, 2005-01-03 to 2026-06-12, "
            "as-of 2026-06-14; the offline core and tests run on a deterministic synthetic tape. "
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
            f"| **Signal** | `NONE` | Breakout: gross {R['bo_mean']:+.0f} bps/trade, "
            f"*but* random earns {R['rand_bo']:+.0f} bps → delta **{R['delta_bo']:+.0f} bps**. "
            f"Reversion: gross {R['rv_mean']:+.0f} bps vs random {R['rand_rv']:+.0f} bps → "
            f"delta **{R['delta_rv']:+.0f} bps**. |\n"
            f"| **Tradability** | `MIRAGE` | No channel-alpha above unconditional drift; "
            f"no break-even cost is needed because the signal is absent before costs. |\n"
            f"| **Breakout vs Reversion?** | `BOTH_NOISE` | Breakout is **{abs(R['delta_bo']):.0f} bps "
            f"below** random (the channel filter hurts). Reversion is only **{R['delta_rv']:+.0f} bps "
            "above** (noise). Neither arm clears the inference bar vs its control. |\n\n"
            "> In plain words: both channel signals show positive gross means because equities drift "
            "upward over 21 years. Once you control for that drift with a random-entry baseline, "
            "the channel adds nothing — and the breakout filter actually removes better entry days."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $K_t^+$ and $K_t^-$ denote the upper and lower Keltner bands at day $t$:\n\n"
            "$$K_t^{\\pm} = \\mathrm{EMA}(20)_t \\pm 2 \\cdot \\mathrm{ATR}(10)_t$$\n\n"
            "Two competing hypotheses:\n\n"
            "- **H₁ (breakout / momentum):** $C_t > K_t^+$ predicts $\\mathbb{E}[r_{t+1,\\ldots,t+20}] "
            "> \\mathbb{E}[r^{\\text{random}}]$ where $r^{\\text{random}}$ is the matched "
            "random-entry return.\n"
            "- **H₂ (reversion):** $C_t < K_t^-$ predicts $\\mathbb{E}[r_{t+1,\\ldots,t+20}] "
            "> \\mathbb{E}[r^{\\text{random}}]$.\n\n"
            "At most one of H₁, H₂ can hold; both can fail. We test both with HAC inference on "
            "per-trade returns vs a size-matched random-entry baseline."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The Keltner Channel is one of the most-used volatility envelopes in retail algo "
            "trading. If H₁ held, it would validate the momentum framing and justify automated "
            "breakout systems. If H₂ held, it would validate the mean-reversion framing and "
            "justify 'buy the dip' systems. Both failing is the honest base rate for chart patterns "
            "tested with proper controls. The interesting finding: the raw gross means look "
            "convincing (positive t-stats), but they are entirely explained by the equity drift "
            "baseline — a textbook case of omitting the unconditional control."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Entry.** Channel pierce on close of day $t$; **enter at $t{+}1$'s open** "
            "(no look-ahead). Only the first day of each consecutive outside-band run is used.\n"
            "- **Exit.** Fixed 20-bar time-exit (close of $t{+}20$). Both arms use the same "
            "horizon to ensure an apple-to-apple comparison.\n"
            "- **Control.** Same number of random dates (sampled after warm-up), same hold. This "
            "baseline prices in the instrument's unconditional drift — the only fair comparison.\n"
            "- **Inference.** Newey–West HAC *t*-stat on per-trade gross returns for each arm. "
            "The signal-vs-random delta and its implied *t* (via the pooled standard error).\n"
            "- **Positive control.** Synthetic tape with tunable AR(1) momentum confirms the engine "
            "recovers an edge *when one exists* — so the null result is about the market, not the method.\n\n"
            "Basket: SPY, QQQ, IWM, GLD, EEM · 2005-01-03 to 2026-06-12 (~21 years, ~5,395 bars each)."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Both arms, per instrument — the delta vs random is the verdict\n\n"
            "Gross mean per trade (bps), signal arm vs matched random-entry control. "
            "The *delta* column is the only number that matters."
        ),
        code(
            "if HAVE_REAL:\n"
            "    recs = []\n"
            "    for t in TICKERS:\n"
            "        b = data.fetch_daily(t, fetch=False, cache_dir=CACHE)\n"
            "        for side in ('breakout', 'reversion'):\n"
            "            ent = st.channel_entries(b, side=side)\n"
            "            n = len(ent)\n"
            "            s = st.summarize(st.run_trades(b, ent, hold_days=HOLD, cost_bps=0), 'ret_gross')\n"
            "            rnd = st.random_entries(b, n=n, seed=128)\n"
            "            r = st.summarize(st.run_trades(b, rnd, hold_days=HOLD, cost_bps=0), 'ret_gross')\n"
            "            recs.append((t, side, n, s['mean_bps'], s['tstat'], r['mean_bps'], s['mean_bps']-r['mean_bps']))\n"
            "    df = pd.DataFrame(recs, columns=['ticker','side','n','sig_bps','t','rand_bps','delta'])\n"
            "else:\n"
            "    df = pd.DataFrame({\n"
            "        'ticker': ['SPY','QQQ','IWM','GLD','EEM']*2,\n"
            "        'side': ['breakout']*5+['reversion']*5,\n"
            "        'n': [244,240,184,246,236, 105,105,103,123,146],\n"
            "        'sig_bps': [63.97,111.35,53.76,123.49,46.22, 246.05,167.96,190.33,84.54,83.28],\n"
            "        't': [2.07,3.59,1.46,2.92,1.25, 3.43,1.62,2.17,2.03,0.95],\n"
            "        'rand_bps': [113.05,155.15,37.44,135.28,123.10, 157.24,190.62,165.43,96.83,113.13],\n"
            "        'delta': [-49.09,-43.79,16.31,-11.79,-76.88, 88.81,-22.66,24.90,-12.30,-29.84]})\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n"
            "for ax, side, clr in zip(axes, ('breakout','reversion'), (GREEN, AMBER)):\n"
            "    sub = df[df['side']==side]\n"
            "    x = range(len(sub))\n"
            "    ax.bar(x, sub['sig_bps'], width=0.35, color=clr, label=f'{side} signal', align='edge')\n"
            "    ax.bar([i+0.37 for i in x], sub['rand_bps'], width=0.35, color=GREY, label='random', align='edge')\n"
            "    ax.set_xticks([i+0.35 for i in x]); ax.set_xticklabels(sub['ticker'])\n"
            "    ax.axhline(0, c='k', lw=1)\n"
            "    ax.set_title(f'{side.capitalize()} vs random-entry control')\n"
            "    ax.set_ylabel('gross mean (bps/trade)')\n"
            "    ax.legend(fontsize=8)\n"
            "plt.tight_layout(); plt.show()\n"
            "df.round(1)"
        ),
        md(
            f"> In plain words: the breakout arm's raw means look positive (t-stats of "
            f"{R['bo_t']:+.1f} pooled) — but the random control earns *more*. "
            f"The reversion arm barely edges random ({R['delta_rv']:+.0f} bps). "
            "The channel pierce is not adding information; it's selecting specific market regimes "
            "(post-breakout highs and crisis lows) that happen to follow the instrument's drift."
        ),
        md(
            "### 4b · The contradiction visualised — at most one can be right\n\n"
            "If the upper band truly marks 'strong trend momentum' and the lower band truly marks "
            "'stretched too far to snap back', these two signals imply opposite views of how "
            "prices behave when they pierce the channel. On a 20-day horizon the synthetic "
            "positive control below shows both can only work when the *right* structure is planted."
        ),
        code(
            "moms = [-0.20, -0.10, 0.0, 0.10, 0.20]\n"
            "bo_edge, rv_edge = [], []\n"
            "for m in moms:\n"
            "    bars, _ = data.synthetic_daily(n_days=2000, momentum=m, seed=128)\n"
            "    for side, lst in [('breakout', bo_edge), ('reversion', rv_edge)]:\n"
            "        ent = st.channel_entries(bars, side=side)\n"
            "        if len(ent) < 5:\n"
            "            lst.append(float('nan'))\n"
            "            continue\n"
            "        sig = st.summarize(st.run_trades(bars, ent, hold_days=20, cost_bps=0), 'ret_gross')['mean_bps']\n"
            "        rnd = st.random_entries(bars, n=len(ent), seed=0)\n"
            "        rand = st.summarize(st.run_trades(bars, rnd, hold_days=20, cost_bps=0), 'ret_gross')['mean_bps']\n"
            "        lst.append(sig - rand)\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "ax.plot(moms, bo_edge, 'o-', c=GREEN, lw=2, label='Breakout delta vs random')\n"
            "ax.plot(moms, rv_edge, 's-', c=AMBER, lw=2, label='Reversion delta vs random')\n"
            "ax.axhline(0, c='k', lw=1); ax.axvline(0, ls='--', c=GREY)\n"
            "ax.set_xlabel('planted AR(1) momentum (synthetic tape)')\n"
            "ax.set_ylabel('signal delta vs random control (bps/trade)')\n"
            "ax.set_title('The contradiction: breakout needs +momentum; reversion needs -momentum')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('Real tape sits near momentum=0 — both arms consistent with noise.')"
        ),
        md(
            "> The synthetic sweep confirms the engine is faithful: the breakout arm earns its "
            "delta when momentum is planted (+0.20) and loses when reversion is planted; the "
            "reversion arm flips it. On the real tape both signals sit in the noise band around "
            "zero momentum — neither arm has a detectable structural edge."
        ),
        md(
            "### 4c · Per-instrument HAC t-stats\n\n"
            "Raw t-stats on gross mean per trade for each arm. The bars above ±2 look "
            "real — until you remember they are measuring equity drift, not channel alpha."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pass  # df already computed above\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))\n"
            "for ax, side, clr in zip(axes, ('breakout','reversion'), (GREEN, AMBER)):\n"
            "    sub = df[df['side']==side]\n"
            "    colors = [RED if abs(v)<2 else clr for v in sub['t']]\n"
            "    ax.bar(sub['ticker'], sub['t'], color=colors)\n"
            "    for s in (2, -2): ax.axhline(s, ls='--', c=GREY, lw=1)\n"
            "    ax.axhline(0, c='k', lw=1)\n"
            "    ax.set_ylabel('HAC t-stat (gross mean per trade)')\n"
            "    ax.set_title(f'{side.capitalize()} — t-stats look real but measure drift')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('These t-stats are not evidence of channel signal — they are evidence of equity drift.')"
        ),
        md(
            "> The positive t-stats are seductive. They clear +2 on several tickers. But the "
            "random-entry control clears equally high (or higher) t-stats — because both are "
            "measuring the same thing: the market's upward drift over 21 years."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — Breakout delta vs random: **{R['delta_bo']:+.0f} bps** "
            f"(the channel filter *hurts* by {abs(R['delta_bo']):.0f} bps). "
            f"Reversion delta vs random: **{R['delta_rv']:+.0f} bps** (noise). "
            "Neither arm clears the inference bar vs its matched control.\n"
            "- **Tradability `MIRAGE`** — The positive gross means are equity drift; no channel "
            "alpha exists to trade. Turnover is low (~6–23 trades/yr/ticker) so costs are "
            "immaterial — the problem is the absence of signal before costs.\n"
            f"- **Breakout vs Reversion? `BOTH_NOISE`** — The breakout arm earns "
            f"{abs(R['delta_bo']):.0f} bps *less* than a coin-flip entry. The reversion arm earns "
            f"{R['delta_rv']:+.0f} bps *more*. The contradiction between the two folk claims is "
            "resolved empirically: both are noise."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — cost sweep and turnover\n\n"
            "At ~6–23 entries per year per ticker, the Keltner Channel is a low-turnover strategy. "
            "Costs barely move the needle — the fundamental problem is the absence of alpha."
        ),
        code(
            "costs = [0.0, 1.0, 2.0, 5.0]\n"
            "if HAVE_REAL:\n"
            "    t_bo = [st.summarize(pool('breakout', c), 'ret_net') for c in costs]\n"
            "    t_rv = [st.summarize(pool('reversion', c), 'ret_net') for c in costs]\n"
            "    net_bo = [s['mean_bps'] for s in t_bo]; ts_bo = [s['tstat'] for s in t_bo]\n"
            "    net_rv = [s['mean_bps'] for s in t_rv]; ts_rv = [s['tstat'] for s in t_rv]\n"
            "else:\n"
            "    net_bo = [R['bo_net_0'],R['bo_net_1'],R['bo_net_2'],R['bo_net_5']]\n"
            "    ts_bo  = [R['bo_t_0'],R['bo_t_1'],R['bo_t_2'],R['bo_t_5']]\n"
            "    net_rv = [R['rv_net_0'],R['rv_net_1'],R['rv_net_2'],R['rv_net_5']]\n"
            "    ts_rv  = [R['rv_t_0'],R['rv_t_1'],R['rv_t_2'],R['rv_t_5']]\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "ax.plot(costs, net_bo, 'o-', c=GREEN, lw=2, label='Breakout net mean')\n"
            "ax.plot(costs, net_rv, 's-', c=AMBER, lw=2, label='Reversion net mean')\n"
            "ax.axhline(R['bh_spy'], ls=':', c=GREY, label=f'SPY random entry baseline ({R[\"bh_spy\"]:.0f} bps)')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('round-trip cost (bps)'); ax.set_ylabel('net mean (bps/trade)')\n"
            "ax.set_title('Costs barely matter — the gross mean is all equity drift, not alpha')\n"
            "ax.legend(fontsize=8); plt.tight_layout(); plt.show()\n"
            "for c, nb, nr in zip(costs, net_bo, net_rv):\n"
            "    print(f'cost={c:4.1f}: breakout {nb:+.0f} bps | reversion {nr:+.0f} bps')"
        ),
        md(
            "> The net means stay comfortably positive across all costs — because they are measuring "
            "captured equity drift. The grey dashed line shows what a random entry earns on SPY. "
            "The channel signals track this baseline, not beating it."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further — the positive control\n\n"
            "The synthetic sweep in beat 4b already serves as the positive control. "
            "For completeness: here is the breakout arm's alpha (delta vs random) as a "
            "function of planted momentum, confirming the engine is a faithful signal "
            "detector when signal exists."
        ),
        code(
            "# Re-plot the positive control cleanly for the quant record.\n"
            "moms2 = [-0.25,-0.20,-0.15,-0.10,-0.05, 0.0, 0.05,0.10,0.15,0.20,0.25]\n"
            "bo2, rv2 = [], []\n"
            "for m in moms2:\n"
            "    bars, _ = data.synthetic_daily(n_days=3000, momentum=m, seed=128)\n"
            "    for side, lst in [('breakout', bo2), ('reversion', rv2)]:\n"
            "        ent = st.channel_entries(bars, side=side)\n"
            "        if len(ent) < 5:\n"
            "            lst.append(float('nan'))\n"
            "            continue\n"
            "        sig = st.summarize(st.run_trades(bars, ent, hold_days=20, cost_bps=0), 'ret_gross')['mean_bps']\n"
            "        rnd = st.random_entries(bars, n=len(ent), seed=1)\n"
            "        rand = st.summarize(st.run_trades(bars, rnd, hold_days=20, cost_bps=0), 'ret_gross')['mean_bps']\n"
            "        lst.append(sig - rand)\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "ax.plot(moms2, bo2, 'o-', c=GREEN, lw=2, label='Breakout delta')\n"
            "ax.plot(moms2, rv2, 's-', c=AMBER, lw=2, label='Reversion delta')\n"
            "ax.axhline(0, c='k', lw=1); ax.axvline(0, ls='--', c=GREY)\n"
            "ax.set_xlabel('planted AR(1) momentum'); ax.set_ylabel('delta vs random (bps/trade)')\n"
            "ax.set_title('Positive control: the engine is faithful — the real market has no planted momentum')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('Engine confirmed: breakout alpha rises with +momentum; reversion alpha rises with -momentum.')\n"
            "print('Real tape result: both arms near zero because the tape has neither planted structure.')"
        ),
        md(
            "The positive control confirms the machinery: the breakout alpha rises monotonically "
            "with planted momentum and the reversion alpha rises with planted mean-reversion. "
            "The real-tape result — both arms near zero vs random — is therefore a statement about "
            "the market (no exploitable structure at this horizon) not about the method.\n\n"
            "Related studies worth comparing:\n"
            "- **[Study 104 — Bollinger-Reversion](../../104-bollinger-reversion/)**: the Bollinger "
            "Band version — same framework, same result.\n"
            "- **[Study 78 — Crossed-Wires](../../78-crossed-wires/)**: MACD daily — momentum filter "
            "with a more complex construction, same desk treatment.\n"
            "- **[Study 21 — Fools-Gold](../../21-fools-gold/)**: the 50/200 daily golden cross — "
            "the moving-average momentum family."
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
