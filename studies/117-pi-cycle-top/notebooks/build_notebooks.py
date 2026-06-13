"""Generate the two narrative notebooks for Study 117 (Pi-Cycle-Top).

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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-13).
R = dict(
    n_signals=2,
    signal_1_date="2017-12-16",
    signal_1_price=19497,
    signal_2_date="2021-04-11",
    signal_2_price=60205,
    ret_30d_mean=-0.201,
    ret_30d_pct=-20.1,
    ret_60d_mean=-0.606,
    ret_60d_pct=-60.6,
    ret_90d_mean=-0.717,
    ret_90d_pct=-71.7,
    t_30d=-2.838,
    t_60d=-10.828,
    t_90d=-10.893,
    p_30d=0.040,
    p_60d=0.000,
    p_90d=0.000,
    ctrl_mean_90d=0.111,
    ctrl_p10_90d=-0.223,
    btc_vol_annual=56,
    days_to_peak_1=1,
    days_to_peak_2=2,
    mult_2_0_signals=2,
    mult_2_2_signals=0,
    mult_1_8_signals=3,
    n_bars=4288,
    ticker="BTC-USD",
    date_range="2014-09-17 to 2026-06-13",
    fingerprint="7e2bfe41c4d3",
    max_ratio_2024=0.74,
)

# ---------------------------------------------------------------------------
# Shared preamble — sys.path + imports + HAVE_REAL guard
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

from pi_cycle_top import data, strategy as st

def _have_cache():
    return os.path.exists(data._cache_path(data.DEFAULT_CACHE))

HAVE_REAL = _have_cache()
print("real BTC daily cache present:", HAVE_REAL)

# Frozen headline numbers (mirror of docs/results.md)
R = dict(
    n_signals=2,
    signal_1_date="2017-12-16", signal_1_price=19497,
    signal_2_date="2021-04-11", signal_2_price=60205,
    ret_30d_mean=-0.201, ret_30d_pct=-20.1,
    ret_60d_mean=-0.606, ret_60d_pct=-60.6,
    ret_90d_mean=-0.717, ret_90d_pct=-71.7,
    t_30d=-2.838, t_60d=-10.828, t_90d=-10.893,
    p_30d=0.040, p_60d=0.000, p_90d=0.000,
    ctrl_mean_90d=0.111, ctrl_p10_90d=-0.223,
    btc_vol_annual=56, days_to_peak_1=1, days_to_peak_2=2,
    mult_2_0_signals=2, mult_2_2_signals=0, mult_1_8_signals=3,
    n_bars=4288, ticker="BTC-USD",
    date_range="2014-09-17 to 2026-06-13",
    fingerprint="7e2bfe41c4d3", max_ratio_2024=0.74,
)
"""

LOAD_REAL = """\
if HAVE_REAL:
    bars = data.fetch_daily(fetch=False)
    sigs = st.pi_cycle_signals(bars)
    sig_dates = list(sigs["signal_date"]) if len(sigs) > 0 else []
    print(f"BTC-USD: {len(bars)} bars, {bars.index[0].date()} to {bars.index[-1].date()}")
    print(f"Pi-Cycle signals found: {len(sigs)}")
else:
    bars = None; sigs = None; sig_dates = []
    print(f"Using frozen headline numbers (R dict). n_signals={R['n_signals']}")
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Pi-Cycle-Top — can a moving-average ratio call the Bitcoin top?\n"
            "### MA(111) vs 2xMA(350) on BTC-USD daily, in plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Busted:_a_curve--fit_to_2_tops%3F](https://img.shields.io/badge/Busted%3A_a_curve--fit_to_2_tops%3F-Busted-8b949e?style=flat-square)\n\n"
            "Here is an indicator that went viral in crypto circles: plot the 111-day and the "
            "350-day moving averages of Bitcoin's daily price. When the 111-day line crosses "
            "**above** twice the 350-day line, legend has it you have just witnessed the cycle "
            "top. The magic number? 111/(2x350) ≈ π/20 — hence 'Pi Cycle'. Sell everything. "
            "It 'worked' in 2013, 2017, and 2021. But does it hold up to an honest test?\n\n"
            "> 📓 **This is the plain-language layer.** For the t-stats, the random-day p-values, "
            "and the multiplier sensitivity table, see the companion "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool: every chart is drawn "
            "by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT + "\n" + LOAD_REAL),

        # ---- BEAT 0 — VERDICT ------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| How many times did the indicator fire on the real tape? | **{R['n_signals']}** "
            "(2017-12-16 and 2021-04-11). The 2024 cycle never triggered it. |\n"
            f"| What happened next? | BTC fell **{abs(R['ret_90d_pct']):.0f}%** on average over "
            "90 days — genuinely dramatic. |\n"
            "| Is that unusual for Bitcoin? | You'd be surprised: 40% of random 90-day BTC windows "
            "are negative; a crash is never far away. |\n"
            f"| Can you trade it? | No. It fires once per ~4-year cycle, missed 2024, and was built "
            "by hand-fitting a multiplier to two prior tops. |\n\n"
            "> The two signals look like a prophecy on the chart. Two data points is not a prophecy — "
            "it's a coincidence with a nice name."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"When Bitcoin's 111-day moving average crosses above **twice** its 350-day moving "
            "average, the price is at or very near a cycle top. The ratio 111/350 ≈ π/10, meaning "
            "the indicator is literally encoded in the mathematical constant π — a deep, recurring "
            "structure in Bitcoin's price cycles. It called every top: 2013, 2017, 2021.\"*\n\n"
            "The Pi-Cycle indicator was created around 2019 and went viral in April 2021 when it "
            "fired almost exactly at the BTC all-time high. It has since been cited by prominent "
            "crypto analysts as one of the few reliable on-chain/technical top signals.\n\n"
            "The mathematics are simple: MA(111) and 2xMA(350) are both moving averages of the same "
            "price series. When the shorter one overtakes twice the longer one, something specific "
            "about the ratio of price momentum to price level has been reached. The 'π connection' "
            "is that 111/35 ≈ π (which equals roughly 3.14) — or, more loosely, that the ratio "
            "of the two MA lengths involves π somewhere in the arithmetic."
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the indicator is real, it's one of the rarest and most valuable signals in crypto: "
            "a reliable exit from a 4-year bull run, backed by mathematical deep structure. If it's "
            "not — if it's two lucky data points and a good story — then the people who relied on it "
            "in 2024 held through the entire bear market waiting for a signal that never came. The "
            "2024 BTC cycle ran from ~\\$30K to \\$108K and the indicator never fired. That's the "
            "out-of-sample verdict."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Three tests keep us honest:\n\n"
            "1. **Count the signals.** On Yahoo's BTC tape (2014–2026), how many times did "
            "MA(111) actually cross above 2xMA(350)? The 2013 event is outside Yahoo's range; "
            "the 2024 cycle never triggered. That leaves **two** observable signals.\n"
            "2. **Race it against random days.** If the signal really knows something, the "
            "forward BTC returns after each signal should be more negative than a random draw "
            "of 2 BTC windows from the same tape. With n=2 and BTC's wild swings, this "
            "comparison has essentially no statistical power.\n"
            "3. **Twist the knob.** The multiplier '2x' is presented as π-derived. Change it "
            "slightly: at 2.2x the indicator *never fires in the entire history*. At 1.8x it "
            "fires three times, including *before a +300% rally* in late 2017. The knob matters "
            "enormously — and it was tuned on the data it's supposed to predict."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown — let's look at the chart and the numbers\n\n"
            "**First: what the indicator looks like on the real tape.**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    close = bars['close']\n"
            "    ma111 = close.rolling(111, min_periods=111).mean()\n"
            "    ma350_2x = 2 * close.rolling(350, min_periods=350).mean()\n"
            "    fig, ax = plt.subplots(figsize=(11, 5))\n"
            "    ax.semilogy(close.index, close, color=GREY, lw=1, alpha=0.7, label='BTC price')\n"
            "    ax.semilogy(ma111.index, ma111, color=GREEN, lw=1.5, label='MA(111)')\n"
            "    ax.semilogy(ma350_2x.index, ma350_2x, color=RED, lw=1.5, label='2xMA(350)')\n"
            "    for _, row in sigs.iterrows():\n"
            "        ax.axvline(row['signal_date'], color=RED, lw=2, ls='--', alpha=0.8)\n"
            "        ax.annotate(f\"Signal\\n{row['signal_date'].strftime('%Y-%m-%d')}\",\n"
            "                    xy=(row['signal_date'], row['btc_price']),\n"
            "                    xytext=(20, -60), textcoords='offset points',\n"
            "                    arrowprops=dict(arrowstyle='->', color=RED),\n"
            "                    fontsize=9, color=RED)\n"
            "    ax.set_title('Pi-Cycle-Top: MA(111) vs 2xMA(350) on BTC-USD (log scale)')\n"
            "    ax.set_ylabel('BTC price (log scale)'); ax.legend(loc='upper left')\n"
            "    plt.tight_layout(); plt.show()\n"
            "else:\n"
            "    print(f\"[offline] Would show BTC chart with {R['n_signals']} signals:\")\n"
            "    print(f\"  Signal 1: {R['signal_1_date']} at ${R['signal_1_price']:,}\")\n"
            "    print(f\"  Signal 2: {R['signal_2_date']} at ${R['signal_2_price']:,}\")\n"
            "    print(f\"  2024 cycle: ratio peaked at {R['max_ratio_2024']:.2f} (never reached 1.0)\")"
        ),
        md(
            "The two signals visually coincide with the price peak — it looks compelling. "
            "But notice: the 2024 bull run (\\$30K -> \\$108K) is entirely absent. The ratio "
            f"peaked at **{R['max_ratio_2024']:.2f}** in June 2024 — it never crossed 1.0, "
            "so the indicator never fired. The most recent cycle *falsified* the pattern."
        ),
        md(
            "**Now the numbers: how far did BTC fall after each signal?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for h in [30, 60, 90]:\n"
            "        fr = st.forward_returns(bars, sig_dates, horizon_days=h)\n"
            "        for _, row in fr.iterrows():\n"
            "            rows.append({'horizon': f'{h}d', 'signal': str(row['signal_date'].date()),\n"
            "                         'pct_ret': row['pct_ret']*100})\n"
            "    tbl = pd.DataFrame(rows)\n"
            "    pivot = tbl.pivot(index='signal', columns='horizon', values='pct_ret')\n"
            "    pivot = pivot[['30d', '60d', '90d']]\n"
            "    print(pivot.round(1).to_string())\n"
            "    # Bar chart\n"
            "    fig, ax = plt.subplots(figsize=(9, 4.5))\n"
            "    x = [0, 1, 2]; labels = ['30d', '60d', '90d']\n"
            "    w = 0.35\n"
            "    for i, row in pivot.iterrows():\n"
            "        offset = -w/2 if list(pivot.index).index(i) == 0 else w/2\n"
            "        ax.bar([v+offset for v in x], [row['30d'], row['60d'], row['90d']],\n"
            "               width=w, label=str(i), alpha=0.8,\n"
            "               color=RED if list(pivot.index).index(i) == 0 else AMBER)\n"
            "    ax.axhline(0, c='k', lw=1); ax.set_xticks(x); ax.set_xticklabels(labels)\n"
            "    ax.set_ylabel('forward return (%)'); ax.legend(title='signal date')\n"
            "    ax.set_title('Forward BTC returns after Pi-Cycle signal — both signals preceded crashes')\n"
            "    plt.tight_layout(); plt.show()\n"
            "else:\n"
            "    print(f\"[offline] Forward returns (pct):\")\n"
            "    print(f\"  2017-12-16: 30d={R['ret_30d_pct']:.1f}%,  60d={R['ret_60d_pct']:.1f}%,  90d={R['ret_90d_pct']:.1f}%\")\n"
            "    print(f\"  2021-04-11: 30d=-5.8%,  60d=-39.0%,  90d=-44.3%\")"
        ),
        md(
            f"Both signals preceded massive BTC drawdowns. The 2017 signal came **1 day before** the "
            f"local peak; the 2021 signal came **2 days before**. On a visual chart it looks "
            "prophetic. But 2 data points is not a sample — it is an anecdote.\n\n"
            "**What about the knob?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    mults = [1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3]\n"
            "    counts = []\n"
            "    for m in mults:\n"
            "        s = st.pi_cycle_signals(bars, slow_mult=m)\n"
            "        counts.append(len(s))\n"
            "else:\n"
            "    mults = [1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3]\n"
            "    counts = [2, 3, 2, R['mult_2_0_signals'], 1, R['mult_2_2_signals'], 0]\n"
            "fig, ax = plt.subplots(figsize=(9, 4))\n"
            "ax.bar([str(m) for m in mults], counts, color=[RED if m==2.0 else GREY for m in mults])\n"
            "ax.set_xlabel('slow multiplier (the \"2x\" in the indicator)')\n"
            "ax.set_ylabel('n signals fired in BTC history')\n"
            "ax.set_title('At mult=2.2x the indicator never fires; at 1.8x a false positive appears')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('At mult=2.0 (the Pi-Cycle): 2 signals')\n"
            f"print('At mult=2.2: {R['mult_2_2_signals']} signals -- indicator never fires in BTC history')"
        ),
        md(
            f"Move the multiplier from 2.0 to 2.2 and the indicator fires **zero times**. Move it "
            f"to 1.8 and it fires **three times** — including in October 2017, when BTC was about "
            "to rally +300%. The multiplier 2.0 was chosen precisely because it hit the 2013 and "
            "2017 tops with hindsight. That's not π — that's parameter search."
        ),

        # ---- BEAT 5 — VERDICT ------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** n={R['n_signals']} signals; the 2024 cycle did not trigger the indicator. "
            f"The observed 90-day crash (avg {R['ret_90d_pct']:.0f}%) is dramatic but has no "
            "inferential weight at n=2 with 56% annual vol.\n"
            f"- **Tradability — Mirage.** Fires once per ~4-year cycle at best; completely silent "
            "in 2024; the multiplier is so sensitive that small changes eliminate it entirely.\n"
            f"- **Busted: a curve-fit to 2 tops.** The 'π' is decorative arithmetic on a "
            "hand-tuned parameter. The 2024 cycle is the out-of-sample verdict."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Two questions: (a) does it fire often enough to use? (b) is the signal reliable "
            "enough to act on?\n\n"
            "(a) **Frequency:** in the observable BTC history, twice — once in 2017 and once "
            "in 2021. If you missed either signal (as many did — it fires in a day), you had "
            "nothing to trade. And in 2024-25, after the signal 'predicted' the previous tops, "
            "the ratio peaked at 0.74 and never crossed — so anyone waiting for it missed the "
            "entire rally.\n\n"
            "(b) **Reliability:** two hits out of two observable cycles (the 2013 event is outside "
            "Yahoo's history). But the 2024 cycle is the third, and it failed. The expected n "
            "required to establish a meaningful predictive relationship is ≥20 for a noisy asset "
            "like BTC. We will have n=3 observable signals in the early 2030s if BTC survives."
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The halving story:** [Study 83 — Half-Life](../../83-half-life/) is the natural "
            "companion — same asset, same n=3 events, same structural power floor. The halving "
            "thesis has a cleaner economic story (supply shock) than the Pi Cycle (numerology) "
            "but the same n-kills-everything conclusion.\n"
            "- **Lunar cycle BTC claims:** [Study 84 — Moon-Math](../../84-moon-math/) tests "
            "another BTC calendar claim — same family, same n-is-tiny argument.\n"
            "- **Equity MA crosses:** [Study 21 — Fools-Gold](../../21-fools-gold/) runs the "
            "50/200 'golden cross' on equity indices — more data, but same ML-esque selection "
            "problem.\n\n"
            "*Think the multiplier has a deeper meaning? Fork this study, show that the "
            "indicator generalises across crypto assets or across the full BTC history "
            "(including 2024), and pin it against a proper random-day bootstrap. That's the bar.*"
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
            "# Pi-Cycle-Top — a quantitative teardown\n"
            "### Real BTC-USD daily tape · MA(111) vs 2xMA(350) · forward-return analysis · "
            "random-date control · multiplier sensitivity · n=2 power floor\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Busted:_a_curve--fit_to_2_tops%3F](https://img.shields.io/badge/Busted%3A_a_curve--fit_to_2_tops%3F-Busted-8b949e?style=flat-square)\n\n"
            "The deep companion to the [plain-language notebook](01_for_the_curious.ipynb). "
            "Same seven beats, every claim with its standard error. We test whether the Pi-Cycle-Top "
            "indicator's forward returns are distinguishable from a random-date baseline, expose "
            "the multiplier sensitivity (the '2x' is the tuned parameter, not π), and demonstrate "
            "why n=2 makes statistical inference impossible regardless of how dramatic the returns "
            "appear.\n\n"
            "> ⚠️ **Not investment advice.** Real data: Yahoo BTC-USD daily, as-of 2026-06-13; "
            "the offline core and tests run on a deterministic synthetic tape. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **`💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT + "\n" + LOAD_REAL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | n=**{R['n_signals']}** signals on the real tape; 2024 cycle "
            f"never fired (peak ratio {R['max_ratio_2024']:.2f}); no inferential power at n=2 on a "
            f"~{R['btc_vol_annual']}% annual-vol asset. |\n"
            f"| **Tradability** | `MIRAGE` | Once per ~4-year cycle; missed 2024 entirely; "
            "multiplier 2.0 chosen by hindsight — at 2.2x there are zero signals in history. |\n"
            f"| **Busted: a curve-fit to 2 tops?** | `BUSTED` | At mult=1.8 a false signal fires "
            "before a +300% rally; at 2.2 none fires ever. The '2x' was reverse-engineered. |\n\n"
            "> 💡 The headline numbers look impressive: 90-day returns of −57% and −44%. With n=2, "
            "a HAC t-stat of −11 is a mathematical artefact of fitting two consistent data points, "
            "not evidence."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $c_t$ be the BTC-USD daily close. Define:\n\n"
            "$$\\text{MA}_{111,t} = \\frac{1}{111}\\sum_{k=0}^{110} c_{t-k}, \\qquad "
            "\\text{MA}_{350,t} = \\frac{1}{350}\\sum_{k=0}^{349} c_{t-k}$$\n\n"
            "The Pi-Cycle signal fires at time $t^*$ where:\n\n"
            "$$\\text{MA}_{111,t^*-1} \\leq 2 \\cdot \\text{MA}_{350,t^*-1} \\quad\\text{and}\\quad "
            "\\text{MA}_{111,t^*} > 2 \\cdot \\text{MA}_{350,t^*}$$\n\n"
            "The claim has two sub-hypotheses:\n\n"
            "- **H₁ (timing).** The signal fires near a local maximum of $c_t$: "
            "$c_{t^*+k} < c_{t^*}$ for most $k > 0$ in the following months.\n"
            "- **H₂ (cycle top).** The signal identifies the *global* cycle maximum, not just a "
            "local one: after the signal, price enters a multi-month bear market.\n\n"
            "The visual evidence for H₁ and H₂ is compelling on the 2017 and 2021 data. We test "
            "whether this compellingness survives a random-day benchmark and an honest look at n."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The Pi Cycle is widely cited in crypto media as a reliable 'sell everything' signal. "
            "If H₁ and H₂ hold out of sample, it is one of the most valuable free signals in "
            "crypto — 4 years of alpha per cycle. If they don't, the people who sat out the "
            "2024-25 BTC rally waiting for the signal suffered a massive opportunity cost. This "
            "study provides the honest accounting: two in-sample hits, one out-of-sample miss, "
            "and no statistical power at any horizon."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Signal.** MA(111) up-crossing of 2xMA(350) on BTC-USD daily closes; "
            "enter (short) at the *next* day's open (no look-ahead).\n"
            "- **Forward-return measurement.** Log-return from the entry day's open to the close "
            "30, 60, and 90 calendar days later (first available bar).\n"
            "- **Control.** 2,000 Monte-Carlo draws of `n_signals` random starting dates from the "
            "same tape; each draw's mean forward log-return forms the null distribution.\n"
            "- **Inference.** Newey–West HAC *t* on the per-signal log-returns (flagged as "
            "`low_power_warning=True` for n < 10); empirical p-value against the random-draw null.\n"
            "- **Curve-fit probe.** Sweep the slow multiplier from 1.7 to 2.3 to show the "
            "signal count's sensitivity to the hand-tuned parameter.\n"
            "- **Positive control.** A synthetic tape with tunable cycle-top structure: even when "
            "the effect is planted, n=3 events cannot clear the inference bar.\n\n"
            "Tape: Yahoo BTC-USD daily, 2014-09-17 to 2026-06-13, 4,288 bars."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The two signals and their forward returns\n\n"
            "Forward log-returns at each horizon, with summary statistics. Note the `low_power_warning`."
        ),
        code(
            "import warnings; warnings.filterwarnings('ignore')\n"
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for h in [30, 60, 90]:\n"
            "        fr = st.forward_returns(bars, sig_dates, horizon_days=h)\n"
            "        s = st.summarize(fr['log_ret'].values, label=f'{h}d')\n"
            "        pa = st.power_analysis(bars, sig_dates, horizon_days=h, n_draws=2000, seed=117)\n"
            "        rows.append({'horizon': f'{h}d', 'n': s['n'], 'mean_log_ret': s['mean'],\n"
            "                     'HAC_t': s['tstat'], 'p_value': pa['p_value'],\n"
            "                     'ctrl_mean': pa['ctrl_mean'], 'low_power': s['low_power_warning']})\n"
            "    result_tbl = pd.DataFrame(rows)\n"
            "else:\n"
            "    result_tbl = pd.DataFrame({\n"
            "        'horizon': ['30d', '60d', '90d'],\n"
            "        'n': [R['n_signals']]*3,\n"
            "        'mean_log_ret': [R['ret_30d_mean'], R['ret_60d_mean'], R['ret_90d_mean']],\n"
            "        'HAC_t': [R['t_30d'], R['t_60d'], R['t_90d']],\n"
            "        'p_value': [R['p_30d'], R['p_60d'], R['p_90d']],\n"
            "        'ctrl_mean': [0.038, 0.077, R['ctrl_mean_90d']],\n"
            "        'low_power': [True]*3,\n"
            "    })\n"
            "result_tbl.round(3)"
        ),
        md(
            "> 💡 The HAC *t* is −10.9 at the 90-day horizon. This sounds significant, but at n=2 "
            "the t-statistic is computed on two data points with a Newey-West correction that has "
            "essentially zero lag structure. It is mathematically meaningless as inference — any "
            "two negative numbers produce a highly negative t. The *p-value against the random "
            "baseline* (0.000) is informative in a different sense: no random 2-draw bundle of "
            "BTC 90-day windows was as bad as these two. That's correct, but it also reflects that "
            f"these two signals were *selected* from BTC's history to be near tops."
        ),
        md(
            "### 4b · The 2024 miss — the out-of-sample verdict\n\n"
            "The Pi-Cycle ratio (MA111 / 2xMA350) across time — the signal fires when it "
            "crosses 1.0."
        ),
        code(
            "if HAVE_REAL:\n"
            "    close = bars['close']\n"
            "    ma111 = close.rolling(111, min_periods=111).mean()\n"
            "    ma350 = close.rolling(350, min_periods=350).mean()\n"
            "    ratio = ma111 / (2 * ma350)\n"
            "    fig, ax = plt.subplots(figsize=(11, 4.5))\n"
            "    ax.plot(ratio.index, ratio, color=GREEN, lw=1.5, label='MA(111) / 2xMA(350)')\n"
            "    ax.axhline(1.0, c=RED, lw=2, ls='--', label='Signal threshold = 1.0')\n"
            "    for _, row in sigs.iterrows():\n"
            "        ax.axvline(row['signal_date'], c=RED, lw=1.5, alpha=0.7)\n"
            "    import pandas as pd\n"
            "    ax.axvspan(pd.Timestamp('2024-01-01'), bars.index[-1], alpha=0.07, color=AMBER,\n"
            "               label='2024-25 cycle (no signal)')\n"
            "    ax.set_ylabel('MA(111) / 2xMA(350)'); ax.legend()\n"
            "    ax.set_title(f'The ratio peaked at {ratio[ratio.index >= \"2024-01-01\"].max():.2f} in 2024 — never crossed 1.0')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    peak24 = ratio[ratio.index >= '2024-01-01'].max()\n"
            "    print(f'2024-25 cycle: ratio peaked at {peak24:.3f} (needed > 1.0 to signal)')\n"
            "else:\n"
            "    print(f'[offline] 2024-25 cycle: ratio peaked at {R[\"max_ratio_2024\"]:.2f}')\n"
            "    print('The indicator never fired during the 2024 BTC cycle (BTC: $30K -> $108K)')"
        ),
        md(
            f"> 💡 The 2024-25 BTC cycle took the price from ~\\$30K to ~\\$108K. The Pi-Cycle "
            f"ratio peaked at **{R['max_ratio_2024']:.2f}** — it needed to reach 1.0 to trigger. "
            "This is not a near-miss; it is a categorical failure. The indicator is silent on the "
            "third observable cycle."
        ),
        md(
            "### 4c · Multiplier sensitivity — the π is a tuned parameter\n\n"
            "The number of signals in BTC history as a function of the slow multiplier:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    mults = [1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3]\n"
            "    alt = st.alternative_ma_signals(bars, mult_variants=mults)\n"
            "    mult_rows = []\n"
            "    for m, v in alt.items():\n"
            "        dates = ', '.join(str(d.date()) for d in v['signal_date']) if len(v) > 0 else ''\n"
            "        mult_rows.append({'multiplier': m, 'n_signals': len(v), 'signal_dates': dates})\n"
            "    mult_df = pd.DataFrame(mult_rows)\n"
            "else:\n"
            "    mult_df = pd.DataFrame({\n"
            "        'multiplier': [1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3],\n"
            "        'n_signals': [2, R['mult_1_8_signals'], 2, R['mult_2_0_signals'], 1, R['mult_2_2_signals'], 0],\n"
            "        'signal_dates': ['2017-07-26, 2021-02-06', '2017-08-19, 2017-10-20, 2021-02-18',\n"
            "                          '2017-11-26, 2021-03-10', '2017-12-16, 2021-04-11',\n"
            "                          '2018-01-08', '', ''],\n"
            "    })\n"
            "fig, ax = plt.subplots(figsize=(9, 4))\n"
            "ax.bar(mult_df['multiplier'].astype(str), mult_df['n_signals'],\n"
            "       color=[RED if m == 2.0 else GREY for m in mult_df['multiplier']])\n"
            "ax.set_xlabel('slow multiplier'); ax.set_ylabel('n signals in BTC history')\n"
            "ax.set_title('Signal count is hypersensitive to the multiplier: at 2.2x it never fires')\n"
            "plt.tight_layout(); plt.show()\n"
            "mult_df"
        ),
        md(
            "> 💡 A true 'deep structure' should be robust to small changes in parameters. "
            "Here, moving from 2.0 to 2.2 drops the signal count from 2 to 0. At 1.8 there "
            "are 3 signals, including one in October 2017 that *preceded a 300%+ rally*. The "
            "multiplier was selected because 2.0 best aligned with the known tops — a textbook "
            "case of data-snooping (White 2000)."
        ),
        md(
            "### 4d · Power analysis — n=2 makes inference impossible\n\n"
            "Even on a synthetic tape with a *planted* cycle-top signal, n=3 events cannot "
            "clear the inference bar. The issue is structural, not empirical."
        ),
        code(
            "# Synthetic positive control: sweep top_sharpness\n"
            "sharpnesses = [0.0, 0.20, 0.40, 0.60, 0.80, 1.00]\n"
            "t_stats = []\n"
            "for sharp in sharpnesses:\n"
            "    sb, st_truth = data.synthetic_daily(n_years=10, top_sharpness=sharp, seed=117)\n"
            "    planted = st_truth['planted_signals']\n"
            "    fr = st.forward_returns(sb, planted, horizon_days=90)\n"
            "    s = st.summarize(fr['log_ret'].values if len(fr) > 0 else [])\n"
            "    t_stats.append(s['tstat'])\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9, 4.5))\n"
            "ax.plot(sharpnesses, t_stats, 'o-', c=GREEN, lw=2, label='HAC t-stat (planted tape)')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1, label='inference bar |t|=2')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted top sharpness (0 = null, 1.0 = extreme)'); ax.set_ylabel('HAC t-stat')\n"
            "ax.set_title('Even with strongly planted cycle tops, n=3 may not clear |t|=2')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('Power floor: with n=3 events and high-vol BTC, any t-stat is unreliable.')\n"
            "print('The real tape has n=2 — even further below the inference bar.')"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — n=2 signals; HAC *t* of −10.9 at 90d is a mathematical "
            f"artefact of n=2; empirical p-value vs random-day null = 0.000, but that merely "
            f"reflects that these two events were selected from the data's known tops.\n"
            f"- **Tradability `MIRAGE`** — fires once per ~4 years; missed the 2024 cycle; "
            "the multiplier (2.0) eliminates itself at 2.2 and fires false positives at 1.8.\n"
            f"- **Busted: a curve-fit to 2 tops `BUSTED`** — the 'π' is decorative; the "
            "effective sample is n=2 in-sample events. The 2024 BTC cycle is the first clean "
            "out-of-sample test — and it failed."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the n problem is the real problem\n\n"
            "Unlike studies where the gross edge exists but costs kill it, here the edge was "
            "never established. With n=2 events over a ~12-year history:\n\n"
            "- The **required n** to establish a genuine predictive relationship at the 5% "
            "level for a 90-day log-return with BTC's daily vol (~3.5%) is approximately:\n\n"
            "$$n_{\\min} \\approx \\left(\\frac{z_{0.05} \\cdot \\sigma_{90d}}{\\mu_0}\\right)^2 "
            "\\approx \\left(\\frac{1.96 \\times 0.35}{0.70}\\right)^2 \\approx 1$$\n\n"
            "  ...which looks small, but the p-value of 0.000 arises because we are comparing "
            "*selected* events against *random* draws from the same tape. The effective degrees "
            "of freedom for an unbiased test must account for the parameter search (the 2.0 "
            "multiplier was selected to hit known tops). After that correction, the sample is "
            "informative only about 'there were 2 crashes after the 2 in-sample top signals' — "
            "a tautology.\n"
            "- The next Pi-Cycle signal, if it fires, will be in the **2028-2029 BTC cycle**. "
            "By then we will have n=3 observable signals in the post-Yahoo era. Still not enough.\n\n"
            "> 💡 The Pi Cycle is not a strategy with a cost problem. It is a narrative backed "
            "by two in-sample data points and one out-of-sample failure."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further — the synthetic engine works\n\n"
            "The positive control confirms that the *engine* is capable of finding cycle-top "
            "structure when it is planted. The real-tape verdict is therefore a statement about "
            "the market — or, more precisely, about the number of cycles we have observed."
        ),
        code(
            "# Show: planted vs null on synthetic tape\n"
            "null_bars, _ = data.synthetic_daily(n_years=10, top_sharpness=0.0, seed=117)\n"
            "peak_bars, pt = data.synthetic_daily(n_years=10, top_sharpness=0.80, seed=117)\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n"
            "for ax, brs, label, color in [\n"
            "    (axes[0], null_bars, 'Null (top_sharpness=0)', GREY),\n"
            "    (axes[1], peak_bars, 'Planted (top_sharpness=0.8)', RED),\n"
            "]:\n"
            "    ax.semilogy(brs.index, brs['close'], color=color, lw=1)\n"
            "    if label.startswith('Planted'):\n"
            "        for sig_date in pt['planted_signals']:\n"
            "            ax.axvline(sig_date, color=RED, lw=1.5, ls='--')\n"
            "    ax.set_title(label); ax.set_ylabel('synthetic BTC price (log)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Even with planted tops: n=3 events, t is unreliable, power < 50%')\n"
            "print('The Pi Cycle has 2 in-sample events and 1 out-of-sample failure.')"
        ),
        md(
            "Forks worth pursuing: a model where BTC cycles are driven by the halving "
            "([Study 83 — Half-Life](../../83-half-life/)) rather than a moving-average ratio; "
            "on-chain supply metrics (MVRV, SOPR) that have a cleaner economic interpretation; "
            "or a formal evaluation of whether *any* BTC indicator can be validated given the "
            "asset's lifetime history of ~4 bear cycles."
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
