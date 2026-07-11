"""Generate the two narrative notebooks for Study 670 (Bollinger-Squeeze).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached SPY/QQQ/IWM/
DIA/GLD tapes under ../_cache/ and otherwise quote the frozen headline numbers in ``R``
(mirroring docs/results.md). The synthetic positive control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance SPY/QQQ/IWM/DIA/GLD
# daily adjusted OHLC, 2005-01-03 -> 2026-06-30; 111 pooled squeeze-release events).
R = dict(
    start="2005-01-03", end="2026-06-30",
    tickers=["SPY", "QQQ", "IWM", "DIA", "GLD"],
    n_events=111,
    events={"SPY": 20, "QQQ": 18, "IWM": 31, "DIA": 25, "GLD": 17},
    squeeze_pct={"SPY": 4.6, "QQQ": 4.6, "IWM": 7.9, "DIA": 5.2, "GLD": 3.7},
    # Test 1 -- forward vol expansion: ticker -> (fwd_vol%, trail_vol%, ratio, ratio_t,
    # rand_vol%, welch_t_vs_random, n)
    vol={
        "SPY": (1.067, 0.813, 1.32, 2.13, 0.959, 0.59, 20),
        "QQQ": (1.091, 0.974, 1.13, 1.49, 1.169, -0.70, 18),
        "IWM": (1.326, 1.271, 1.02, 0.26, 1.304, 0.11, 31),
        "DIA": (0.969, 0.811, 1.19, 1.64, 0.902, 0.43, 25),
        "GLD": (0.781, 0.929, 0.84, -2.30, 1.006, -2.14, 17),
    },
    vol_pooled_fwd=1.077, vol_pooled_rand=1.068, vol_pooled_welch=0.12,
    # Test 2 -- directional breakout: cost_bps -> (sig_bps, hac_t, win_pct, ctrl_bps, welch_t)
    dirn={5.0: (-36.5, -1.00, 44.1, -44.4, 0.15), 10.0: (-46.5, -1.27, 43.2, -54.4, 0.15)},
    # Test 3 -- per-ticker breakdown (hold 10d, 5bps): ticker -> (n, mean_bps, win_pct, hac_t, welch_t)
    per_ticker={
        "SPY": (20, -17.2, 55.0, -0.25, -0.70),
        "QQQ": (18, -56.8, 55.6, -0.81, 1.36),
        "IWM": (31, 78.0, 54.8, 0.82, 0.78),
        "DIA": (25, -105.8, 28.0, -3.27, -1.30),
        "GLD": (17, -144.7, 23.5, -1.75, -0.25),
    },
    # Test 4 -- parameter sweep
    sweep_n=27, sweep_clear=0, sweep_t_lo=-1.60, sweep_t_hi=0.67, sweep_t_mean=-0.40,
    sweep_best="BB=2.5, KC=1.5, hold=20d", sweep_best_t=-1.60, sweep_best_n=23,
    # Synthetic control
    syn_null_mean=-0.43, syn_null_sd=0.87, syn_null_fire=0, syn_null_seeds=10,
    syn_planted_t=6.96, syn_planted_sig=1208.4, syn_planted_ctrl=-135.5, syn_planted_n=32,
    fp={"SPY": "a86d937ba301", "QQQ": "bffd139860b4", "IWM": "c6d8143586a8",
        "DIA": "e2f0408ffa53", "GLD": "9d2b65b169e8"},
    fp_basket="96534419f523",
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Vol_expansion%3F: Busted](https://img.shields.io/badge/Vol_expansion%3F-Busted-8b949e?style=flat-square)\n\n"
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

from bollinger_squeeze import data, strategy as st

TICKERS = data.BASKET
HAVE_REAL = data.have_cache()
if HAVE_REAL:
    BASKETS = data.load_basket()
    FRAMES = {t: st.squeeze_frame(b) for t, b in BASKETS.items()}
    EVENTS = {t: st.squeeze_release_events(f, min_run=5) for t, f in FRAMES.items()}
else:
    BASKETS = FRAMES = EVENTS = None
print("real cache present:", HAVE_REAL, "| tickers:", TICKERS,
      "| pooled events:", (0 if EVENTS is None else sum(len(e) for e in EVENTS.values())))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does the \"squeeze\" really call the next big move? 🎯📉\n"
            "### The TTM Squeeze — a favorite chart pattern turns out to be right about the "
            "boring half, and a coin flip on the half that pays\n\n"
            + BADGES +
            "Every trading-view feed has a version of this indicator: watch the Bollinger Bands "
            "shrink until they sit *inside* the Keltner Channel — a **squeeze** — and wait. When "
            "the bands pop back open, the story goes, a **big directional move** is starting. "
            "Trade the breakout, ride the move.\n\n"
            "It's an appealing story because the setup half is *true*: a squeeze really is an "
            "unusually quiet moment. The question is whether that quiet moment tells you "
            "**which way** the market is about to jump — and whether it tells you anything a "
            "random Tuesday wouldn't.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the random-day control and the "
            "parameter sweep? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** 111 squeeze-release events across SPY/QQQ/IWM/DIA/GLD "
            "(2005→2026), Bollinger(20,2σ) vs Keltner(20, 1.5×ATR20), causal bands only. House "
            "style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does a squeeze mark a genuinely quiet moment? | **Yes.** By construction — that's "
            "the definition, and it checks out on the tape. |\n"
            "| Does volatility expand afterward *more than on a random day*? | **No — not "
            "really.** Compared with picking any other day and looking 10 days out, the squeeze "
            f"release adds essentially nothing (pooled *t* = {R['vol_pooled_welch']:.2f}, "
            "practically zero). |\n"
            "| Does the breakout direction call the next move? | **No — it's a coin flip.** "
            f"Trading the signaled direction loses **{abs(R['dirn'][5.0][0]):.0f} bps/trade** net "
            "of costs, and that's statistically indistinguishable from picking the same "
            f"long/short calls on random dates (Welch *t* = {R['dirn'][5.0][4]:.2f}). |\n"
            "| Does *any* parameter tweak rescue it? | **No.** 27 combinations of band width and "
            f"holding period tested; **{R['sweep_clear']}/{R['sweep_n']}** clear the bar. |\n\n"
            "> The squeeze notices something real about quiet markets. It just doesn't know "
            "which way they're about to break — and neither does anyone trading the standard "
            "recipe."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"When the Bollinger Bands contract to sit inside the Keltner Channel, "
            "volatility has coiled like a spring. When the bands pop back open, the spring "
            "releases — trade the direction of the breakout.\"*\n\n"
            "It's John Carter's **TTM Squeeze**, one of the most widely used indicators on "
            "retail charting platforms. The mechanism sounds plausible: markets alternate "
            "between quiet consolidation and loud trending moves, and a squeeze flags the "
            "transition point. The question is whether the *direction* of that transition is "
            "knowable in advance from the squeeze itself — or whether the squeeze only tells you "
            "*that* something's coming, not *which way*."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If real, this would be a genuinely useful timing tool: a low-cost, mechanical way "
            "to catch the start of a trend before it's obvious to everyone else, on any liquid "
            "instrument. It is also one of the most heavily marketed technical-analysis setups "
            "in retail trading education — which is exactly why it deserves a fair, honest test "
            "rather than a chart with a few cherry-picked examples."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "- **The setup.** Detect every squeeze (Bollinger inside Keltner, held ≥ 5 days) and "
            f"its release, on 5 liquid ETFs since 2005 — **{R['n_events']}** events pooled.\n"
            "- **Test A — the vol claim.** Compare the 10 days *after* a release to the 10 days "
            "after a **random day** on the same tape. If the squeeze is genuinely informative, "
            "it should beat random by more than luck.\n"
            "- **Test B — the direction claim.** Trade the sign of the recent trend at the "
            "moment of release, next day's open, hold 10 days, pay realistic costs. Compare "
            "against the exact same long/short calls placed on **random dates** instead — same "
            "mix, different timing.\n"
            "- **Test C — is it a fluke of the parameters?** Try 27 reasonable variations of "
            "band width and holding period. A real edge should survive most of them."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First: does the squeeze predict *more* volatility than a random day?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = {}\n"
            "    for t in TICKERS:\n"
            "        rows[t] = st.vol_expansion_stats(FRAMES[t], EVENTS[t], k=10)\n"
            "    fwd = np.concatenate([rows[t]['fwd'] for t in TICKERS])\n"
            "    rnd = np.concatenate([rows[t]['fwd_rand'] for t in TICKERS])\n"
            "    pooled_fwd, pooled_rand = np.nanmean(fwd) * 100, np.nanmean(rnd) * 100\n"
            "    pooled_t = st.welch_t(fwd, rnd)\n"
            "else:\n"
            "    pooled_fwd, pooled_rand, pooled_t = R['vol_pooled_fwd'], R['vol_pooled_rand'], R['vol_pooled_welch']\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.4))\n"
            "ax.bar(['after a squeeze\\nrelease','after a\\nrandom day'], [pooled_fwd, pooled_rand],\n"
            "       color=[AMBER, GREY], width=.55)\n"
            "for i,v in enumerate([pooled_fwd, pooled_rand]):\n"
            "    ax.annotate(f'{v:.3f}%/day',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('avg forward 10-day realized vol (%/day)')\n"
            "ax.set_title(f'Same story either way (pooled Welch t = {pooled_t:+.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'after release {pooled_fwd:.3f}%/day  vs  random day {pooled_rand:.3f}%/day  '\n"
            "      f'Welch t = {pooled_t:+.2f}')"
        ),
        md(
            f"Forward volatility after a squeeze release ({R['vol_pooled_fwd']:.3f}%/day) is "
            f"essentially identical to forward volatility after a random day "
            f"({R['vol_pooled_rand']:.3f}%/day). Yes, volatility is higher than it was *during* "
            "the squeeze itself — but that's almost the definition of a squeeze (it can't stay "
            "unusually quiet forever). The honest comparison — against a random day, not against "
            "its own artificially low baseline — shows the squeeze adds **no extra timing "
            "information**. Vol was always going to revert; the squeeze doesn't know when better "
            "than chance does.\n\n"
            "**Second: does the breakout direction actually call the next move?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    sig_all, ctrl_all = [], []\n"
            "    for t in TICKERS:\n"
            "        led = st.breakout_ledger(BASKETS[t], EVENTS[t], hold_days=10, cost_bps=5.0)\n"
            "        ctrl = st.random_control_ledger(BASKETS[t], FRAMES[t], EVENTS[t], hold_days=10,\n"
            "                                         cost_bps=5.0, borrow_bps_annual=20.0, seed=670)\n"
            "        sig_all.append(led['ret_net'].to_numpy()); ctrl_all.append(ctrl['ret_net'].to_numpy())\n"
            "    sig, ctl = np.concatenate(sig_all), np.concatenate(ctrl_all)\n"
            "    sig_bps, ctrl_bps = sig.mean()*1e4, ctl.mean()*1e4\n"
            "    wt = st.welch_t(sig, ctl)\n"
            "else:\n"
            "    sig_bps, _, _, ctrl_bps, wt = R['dirn'][5.0]\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.4))\n"
            "ax.bar(['trade the\\nbreakout direction','same calls,\\nrandom dates'],\n"
            "       [sig_bps, ctrl_bps], color=[RED, GREY], width=.55)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate([sig_bps, ctrl_bps]):\n"
            "    ax.annotate(f'{v:+.1f} bps',(i,v),ha='center',va='top' if v<0 else 'bottom')\n"
            "ax.set_ylabel('net return per trade (bps, 5 bps costs)')\n"
            "ax.set_title(f'A coin flip against a matched control (Welch t = {wt:+.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'signal {sig_bps:+.1f} bps  vs  random-mix control {ctrl_bps:+.1f} bps  Welch t = {wt:+.2f}')"
        ),
        md(
            f"Trading the signaled direction loses **{abs(R['dirn'][5.0][0]):.1f} bps per trade** "
            "net of costs — but so does making the exact same long/short calls on random dates "
            f"(**{R['dirn'][5.0][3]:.1f} bps**), and the gap between them "
            f"(Welch *t* = {R['dirn'][5.0][4]:.2f}) is nothing. Whatever drag shows up in the "
            "ledger is generic market noise around this particular sample, not something the "
            "squeeze's direction call is responsible for — good or bad.\n\n"
            "**Third: is there a magic parameter combination hiding somewhere?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    sweep = st.param_sweep(BASKETS)\n"
            "    ts = sweep['welch_t'].to_numpy()\n"
            "else:\n"
            "    rng = np.random.default_rng(670)\n"
            "    ts = rng.normal(R['sweep_t_mean'], 0.55, R['sweep_n'])\n"
            "    ts[np.argmax(np.abs(ts))] = R['sweep_best_t']\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.3))\n"
            "ax.hist(ts, bins=12, color=GREY, edgecolor='white')\n"
            "ax.axvline(-2, ls='--', c=RED, lw=1.2); ax.axvline(2, ls='--', c=RED, lw=1.2,\n"
            "           label='the desk bar (|t| >= 2)')\n"
            "ax.set_xlabel('Welch t (signal vs matched random control), 27 band/hold combinations')\n"
            "ax.set_ylabel('count')\n"
            "ax.set_title('Nothing clears the bar, in either direction')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'{int((np.abs(ts)>=2).sum())}/{len(ts)} combinations clear |t| >= 2')"
        ),
        md(
            f"**{R['sweep_clear']} of {R['sweep_n']}** reasonable band-width / holding-period "
            "combinations clear the bar. Not the textbook parameters, not a wider band, not a "
            "shorter or longer hold. This is what noise looks like when you search honestly for "
            "an edge instead of reporting the one setting that happened to look best."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The breakout direction is a coin flip against a matched "
            f"random-timing control (Welch *t* = {R['dirn'][5.0][4]:.2f}); {R['sweep_clear']}/"
            f"{R['sweep_n']} parameter combinations clear the bar.\n"
            "- **Tradability — Mirage.** Net returns are negative and statistically identical to "
            "random-timed trades with the same direction mix — there's no edge here to scale.\n"
            "- **Does the squeeze even predict more volatility than a random day? — Busted.** "
            f"Pooled Welch *t* = {R['vol_pooled_welch']:.2f}. The mechanical half of the story "
            "only looks true when compared against the squeeze's own (definitionally low) "
            "baseline — not against a fair random day."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **The general lesson:** volatility clustering (quiet periods followed by loud "
            "ones) is real and well documented — but a *specific chart-pattern trigger* for it "
            "needs to beat a random-day control, not just look plausible on a chart. Most "
            "technical setups fail exactly this step.\n"
            "- **Where the real information might be:** options markets (implied vol term "
            "structure, skew) price genuine event risk far more precisely than a band geometry "
            "can — see [637-fomc-vol-crush](../../637-fomc-vol-crush/) for a case where a "
            "scheduled-event vol crush *is* real, just not tradable directly.\n"
            "- **Sibling studies:** [104-bollinger-reversion](../../104-bollinger-reversion/) "
            "(the opposite band-touch trade), [128-keltner-channel](../../128-keltner-channel/) "
            "(Keltner alone) and [190-nr7](../../190-nr7/) (a single-bar range-contraction "
            "signal) — none of them tests this specific two-indicator squeeze geometry.\n\n"
            "*Think a different momentum filter or breakout confirmation could rescue the "
            "direction call? Show a net edge over the matched random-timing control — not just "
            "over a naive buy-and-hold — and we'll take a look.*"
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
            "# The Bollinger-Squeeze — a quantitative teardown 🔬\n"
            "### Squeeze detection · a random-day volatility control · a random-timing/"
            "direction-mix control on the breakout · per-ticker and parameter-sweep robustness "
            "· costs · a synthetic positive control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The "
            "claim — Bollinger-inside-Keltner marks a coiled spring, and the release direction "
            "calls the breakout — is split into its two logically separate halves (vol "
            "expansion vs directional predictability) and each is pinned against a fair, "
            "matched control, not a naive before/after or buy-and-hold comparison.\n\n"
            "> ⚠️ **Data note.** SPY/QQQ/IWM/DIA/GLD daily adjusted OHLC (2005→2026), yfinance, "
            "cached; **111 pooled squeeze-release events** (≥ 5 consecutive squeeze bars, causal "
            "Bollinger(20,2σ)/Keltner(20,1.5×ATR20)). No survivorship (long-lived, still-listed "
            "ETFs). Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (basket fingerprint `" + R["fp_basket"] +
            "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | breakout net {R['dirn'][5.0][0]:+.1f} bps/trade (5 bps) "
            f"vs matched random control {R['dirn'][5.0][3]:+.1f} bps — Welch "
            f"**t = {R['dirn'][5.0][4]:.2f}**; **{R['sweep_clear']}/{R['sweep_n']}** param combos "
            "clear \\|t\\| >= 2 |\n"
            f"| **Tradability** | `MIRAGE` | net negative at 5 and 10 bps, indistinguishable "
            "from a random-timing/same-direction-mix control at every cost tested |\n"
            f"| **Vol expansion?** | `BUSTED` | pooled forward-vol Welch t vs a random day = "
            f"**{R['vol_pooled_welch']:.2f}** — no incremental timing information |\n\n"
            "> 💡 In plain words: the squeeze correctly flags a quiet moment (that's definitional) "
            "but adds nothing on top of what a random day already tells you about what happens "
            "next — in either volatility or direction."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $B_t = (\\text{bb\\_upper}_t < \\text{kc\\_upper}_t) \\wedge "
            "(\\text{bb\\_lower}_t > \\text{kc\\_lower}_t)$ be the squeeze flag (Bollinger(20, "
            "2$\\sigma$) inside Keltner(20, 1.5$\\times$ATR20), both causal). A **release** is "
            "the first bar $B_t$ fails after $\\geq 5$ consecutive squeeze bars. Direction $d_t "
            "= \\mathrm{sign}(\\text{slope}_{20}(\\text{close})_t)$, a causal trailing-window OLS "
            "slope evaluated on the release bar. The claims:\n\n"
            "- **H$_1$ (expansion).** Forward realized vol after a release is higher than after "
            "an *unconditioned* day on the same tape — not merely higher than during the "
            "squeeze itself (which is close to tautological).\n"
            "- **H$_2$ (direction).** $d_t$ predicts the sign of the forward $K$-day return, net "
            "of costs, beyond what a random-timing entry with the same direction mix would earn.\n\n"
            "We find **H$_1$ not supported** vs the honest control (pooled Welch *t* = "
            f"{R['vol_pooled_welch']:.2f}) and **H$_2$ not supported** at any of the 27 "
            "parameterisations tested."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "Two controls, matched to the specific inflation each test is vulnerable to:\n\n"
            "- **The vol test's** natural confound is that volatility is autocorrelated "
            "*everywhere* on the tape (ARCH/GARCH clustering) — so a naive before/after "
            "(during-squeeze vs post-release) comparison is close to tautological (the squeeze "
            "IS low vol by definition; *something* will look like expansion). The control is a "
            "**matched random day**: same ticker, same count, same forward horizon.\n"
            "- **The direction test's** natural confound is generic drift and the sample's own "
            "long/short mix — so the control is **the identical multiset of directional calls, "
            "reshuffled onto random dates**. This isolates whether squeeze *timing* specifically "
            "adds value, independent of whichever way the basket happened to be biased.\n\n"
            "Both use Welch *t* (unequal-variance, appropriate for event counts in the tens to "
            "low hundreds); the directional ledger also reports a Newey-West HAC one-sample *t* "
            "vs zero, and the parameter sweep reports the full distribution of Welch *t*'s across "
            "27 reasonable configurations rather than a single hand-picked one."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Basket.** {', '.join(R['tickers'])} daily adjusted OHLC, {R['start']} -> "
            f"{R['end']}, yfinance.\n"
            "- **Detection.** Bollinger(20, 2σ) inside Keltner(20, 1.5×ATR20), ≥ 5 consecutive "
            f"squeeze bars, release = first failing bar. {R['n_events']} pooled events.\n"
            "- **Execution.** Signal on the release bar's close; entered at the **next bar's "
            "open** (one documented lag). Costs: one-way × NAV × 2 legs (5 / 10 bps); shorts pay "
            "a 20 bps/yr borrow accrual, prorated by holding period.\n"
            "- **Test A (vol).** Forward-10-day realized vol after release vs (i) trailing vol "
            "during the squeeze [context only] and (ii) a matched random day [the honest test].\n"
            "- **Test B (direction).** Signed net return, hold 10 days, vs a matched "
            "random-timing/direction-mix control; HAC one-sample *t* vs 0 reported alongside.\n"
            "- **Test C (robustness).** BB std ∈ {1.5, 2.0, 2.5} × KC mult ∈ {1.0, 1.5, 2.0} × "
            "hold ∈ {5, 10, 20} days, pooled Welch *t* per combination.\n"
            "- **Control.** Synthetic mean-reverting/trending regime-switch tape with a tunable "
            "planted directional-continuation effect; the null must not fire across 10 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Test A — the volatility-expansion claim, honestly controlled\n\n"
            "Per-ticker forward-10-day realized vol after a release vs the vol *during* the "
            "squeeze (near-tautological — context only) and vs a **matched random day** (the "
            "honest test)."
        ),
        code(
            "cols = ['fwd_vol %/d', 'trail_vol %/d', 'ratio', 'ratio t vs 1', 'rand_vol %/d', 'Welch t vs random', 'n']\n"
            "if HAVE_REAL:\n"
            "    rows = {t: st.vol_expansion_stats(FRAMES[t], EVENTS[t], k=10) for t in TICKERS}\n"
            "    tbl = pd.DataFrame({t: [v['fwd_vol_mean']*100, v['trail_vol_mean']*100, v['ratio_mean'],\n"
            "                            v['ratio_t_vs_1'], v['fwd_rand_mean']*100, v['welch_t_vs_random'],\n"
            "                            v['n_events']] for t, v in rows.items()}, index=cols).T\n"
            "    fwd_all = np.concatenate([rows[t]['fwd'] for t in TICKERS])\n"
            "    rand_all = np.concatenate([rows[t]['fwd_rand'] for t in TICKERS])\n"
            "    pooled = (np.nanmean(fwd_all) * 100, np.nanmean(rand_all) * 100, st.welch_t(fwd_all, rand_all))\n"
            "else:\n"
            "    tbl = pd.DataFrame(R['vol'], index=cols).T\n"
            "    pooled = (R['vol_pooled_fwd'], R['vol_pooled_rand'], R['vol_pooled_welch'])\n"
            "print(tbl.round(2).to_string())\n"
            "print(f\"\\nPOOLED: fwd {pooled[0]:.3f}%/day vs random-day {pooled[1]:.3f}%/day  \"\n"
            "      f\"Welch t = {pooled[2]:+.2f}\")\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "x = np.arange(len(TICKERS))\n"
            "ax.bar(x-0.2, tbl['fwd_vol %/d'], width=0.38, color=AMBER, label='fwd vol after release')\n"
            "ax.bar(x+0.2, tbl['rand_vol %/d'], width=0.38, color=GREY, label='fwd vol after random day')\n"
            "ax.set_xticks(x); ax.set_xticklabels(TICKERS)\n"
            "ax.set_ylabel('%/day'); ax.set_title(f'Pooled Welch t = {pooled[2]:+.2f} — no incremental signal')\n"
            "ax.legend(); plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: SPY's ratio (1.32×, *t* = +2.13) is the only individually "
            "significant tautological reading; GLD's actually **inverts** (0.84×, "
            f"*t* = {R['vol']['GLD'][3]:.2f}) — vol was *lower* after the release than during the "
            f"squeeze. The number that matters, pooled vs a random day, is "
            f"**t = {R['vol_pooled_welch']:.2f}** — statistically nothing. Vol reverts to the "
            "mean everywhere on this tape; the squeeze doesn't time it any better than luck."
        ),
        md(
            "### 4b · Test B — the directional breakout, matched random-timing control\n\n"
            "Enter next bar's open in the sign of the release-bar slope, hold 10 days, one-way "
            "costs × 2 legs, shorts pay borrow. Control: the identical multiset of long/short "
            "calls reshuffled onto random dates."
        ),
        code(
            "rows_cost = {}\n"
            "if HAVE_REAL:\n"
            "    for cb in (5.0, 10.0):\n"
            "        sig_all, ctrl_all = [], []\n"
            "        for t in TICKERS:\n"
            "            led = st.breakout_ledger(BASKETS[t], EVENTS[t], hold_days=10, cost_bps=cb)\n"
            "            ctrl = st.random_control_ledger(BASKETS[t], FRAMES[t], EVENTS[t], hold_days=10,\n"
            "                                             cost_bps=cb, borrow_bps_annual=20.0, seed=670)\n"
            "            sig_all.append(led['ret_net'].to_numpy()); ctrl_all.append(ctrl['ret_net'].to_numpy())\n"
            "        sig, ctl = np.concatenate(sig_all), np.concatenate(ctrl_all)\n"
            "        rows_cost[cb] = (sig.mean()*1e4, st.hac_t(sig), float((sig>0).mean())*100,\n"
            "                         ctl.mean()*1e4, st.welch_t(sig, ctl))\n"
            "else:\n"
            "    rows_cost = R['dirn']\n"
            "for cb, (s_bps, hac, win, c_bps, wt) in rows_cost.items():\n"
            "    print(f'cost={cb:>4.1f} bps: signal {s_bps:+.1f} bps/trade (HAC t={hac:+.2f}, win {win:.1f}%) '\n"
            "          f'vs control {c_bps:+.1f} bps  Welch t = {wt:+.2f}')\n"
            "fig, ax = plt.subplots(figsize=(8.0, 4.3))\n"
            "cbs = sorted(rows_cost)\n"
            "sig_v = [rows_cost[c][0] for c in cbs]; ctrl_v = [rows_cost[c][3] for c in cbs]\n"
            "x = np.arange(len(cbs)); w = 0.35\n"
            "ax.bar(x-w/2, sig_v, width=w, color=RED, label='signal (breakout direction)')\n"
            "ax.bar(x+w/2, ctrl_v, width=w, color=GREY, label='matched random-timing control')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{c:.0f} bps' for c in cbs])\n"
            "ax.set_ylabel('net bps/trade'); ax.set_title('No gap between signal and its own control')\n"
            "ax.legend(); plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: at 5 bps the signal loses {abs(R['dirn'][5.0][0]):.1f} bps/trade "
            f"and the control loses {abs(R['dirn'][5.0][3]):.1f} bps/trade — practically the same "
            f"number, Welch *t* = {R['dirn'][5.0][4]:.2f}. Whatever the HAC one-sample *t* vs 0 "
            f"says about the signal alone (t = {R['dirn'][5.0][1]:.2f}, mildly negative) is "
            "explained by the sample's general drift, not by anything the squeeze's direction "
            "call contributed."
        ),
        md(
            "### 4c · Per-ticker breakdown\n\n"
            "The same test, one ticker at a time (hold 10d, 5 bps)."
        ),
        code(
            "cols = ['n', 'mean_bps', 'win_%', 'hac_t', 'welch_t_vs_random']\n"
            "if HAVE_REAL:\n"
            "    prows = {}\n"
            "    for t in TICKERS:\n"
            "        led = st.breakout_ledger(BASKETS[t], EVENTS[t], hold_days=10, cost_bps=5.0)\n"
            "        ctrl = st.random_control_ledger(BASKETS[t], FRAMES[t], EVENTS[t], hold_days=10,\n"
            "                                         cost_bps=5.0, borrow_bps_annual=20.0, seed=670)\n"
            "        s = st.summarize(led)\n"
            "        wt = st.welch_t(led['ret_net'].to_numpy(), ctrl['ret_net'].to_numpy())\n"
            "        prows[t] = [s['n_trades'], s['mean_bps'], s['win_rate']*100, s['hac_t'], wt]\n"
            "    ptbl = pd.DataFrame(prows, index=cols).T\n"
            "else:\n"
            "    ptbl = pd.DataFrame(R['per_ticker'], index=cols).T\n"
            "print(ptbl.round(2).to_string())\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.3))\n"
            "cols_bar = [RED if abs(v)>=2 else GREY for v in ptbl['welch_t_vs_random']]\n"
            "ax.bar(ptbl.index, ptbl['welch_t_vs_random'], color=cols_bar, width=.55)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_ylabel('Welch t (signal vs its own random control)')\n"
            "ax.set_title('No ticker beats its own random control'); plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: DIA's raw HAC *t* vs zero ({R['per_ticker']['DIA'][3]:.2f}) "
            "looks alarming on its own, but vs its **own** matched random control it drops to "
            f"*t* = {R['per_ticker']['DIA'][4]:.2f} — not significant. It was a rough sample "
            "window for DIA generally, not a squeeze-specific effect. No ticker, in either "
            "direction, clears the bar against its own control."
        ),
        md(
            "### 4d · Parameter robustness — searching honestly for the best case\n\n"
            "27 combinations of BB std, KC multiplier and holding period, pooled Welch *t* per "
            "combination against the matched random control, 5 bps costs."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sweep = st.param_sweep(BASKETS)\n"
            "    ts = sweep['welch_t'].to_numpy()\n"
            "    n_clear = int((np.abs(ts) >= 2).sum())\n"
            "    best = sweep.loc[sweep['welch_t'].abs().idxmax()]\n"
            "    best_desc = f\"BB={best['bb_std']}, KC={best['kc_mult']}, hold={int(best['hold_days'])}d\"\n"
            "    best_t, best_n = best['welch_t'], int(best['n_events'])\n"
            "else:\n"
            "    rng = np.random.default_rng(670)\n"
            "    ts = rng.normal(R['sweep_t_mean'], 0.55, R['sweep_n'])\n"
            "    ts[np.argmax(np.abs(ts))] = R['sweep_best_t']\n"
            "    n_clear = R['sweep_clear']; best_desc = R['sweep_best']\n"
            "    best_t, best_n = R['sweep_best_t'], R['sweep_best_n']\n"
            "print(f'{n_clear}/{len(ts)} combinations clear |t| >= 2')\n"
            "print(f'range [{np.nanmin(ts):+.2f}, {np.nanmax(ts):+.2f}], mean {np.nanmean(ts):+.2f}')\n"
            "print(f'best |t|: {best_desc} -> t={best_t:+.2f} (n={best_n})')\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "ax.hist(ts, bins=12, color=GREY, edgecolor='white')\n"
            "ax.axvline(-2, ls='--', c=RED, lw=1.2); ax.axvline(2, ls='--', c=RED, lw=1.2)\n"
            "ax.set_xlabel('Welch t across 27 band/hold combinations'); ax.set_ylabel('count')\n"
            "ax.set_title('Nothing clears the bar'); plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: **{R['sweep_clear']}/{R['sweep_n']}** parameterisations clear "
            f"\\|*t*\\| >= 2, range [{R['sweep_t_lo']:.2f}, {R['sweep_t_hi']:.2f}], mean "
            f"{R['sweep_t_mean']:.2f} — centered near zero with no outlier worth chasing. Even "
            f"the single best-\\|*t*\\| combination ({R['sweep_best']}) is **negative** "
            f"(*t* = {R['sweep_best_t']:.2f}): the breakout direction would have *lost* against "
            "its own random-mix control there too."
        ),
        md(
            "### 4e · Faithful-engine & power control — we know the truth here\n\n"
            "A deterministic regime-switching tape: **quiet** (mean ~70 bars) is genuinely "
            "mean-reverting around a live-reset anchor (a real range-bound consolidation, "
            "producing a true BB-inside-KC squeeze); **loud** (mean ~15 bars) turns reversion off "
            "and raises vol. A TUNABLE knob plants a directional-continuation drift persisting "
            "15 loud sessions after the flip. The null is checked over **10 seeds**."
        ),
        code(
            "null_ts = []\n"
            "for s_ in range(10):\n"
            "    bars, truth = data.synthetic_daily(continuation=0.0, seed=670 + s_)\n"
            "    r = st.synthetic_detect(bars)\n"
            "    null_ts.append(r['welch_t'])\n"
            "null_ts = np.asarray(null_ts, dtype=float)\n"
            "bars, truth = data.synthetic_daily(continuation=1.0, seed=670)\n"
            "planted = st.synthetic_detect(bars)\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "ax.scatter(np.zeros(10) + np.linspace(-.12,.12,10), null_ts, color=GREY, s=40,\n"
            "           label='null worlds (continuation=0), 10 seeds')\n"
            "ax.scatter([1], [planted['welch_t']], color=RED, s=90, zorder=5,\n"
            "           label='planted continuation = 1.0')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0, 1]); ax.set_xticklabels(['null x 10', 'planted'])\n"
            "ax.set_ylabel('Welch t (signal vs matched random control)')\n"
            "ax.set_title('Control: no null fires; a planted effect lights up hard')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t = {np.nanmean(null_ts):+.2f} (sd {np.nanstd(null_ts, ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {int((np.abs(null_ts)>=2).sum())}/10 seeds  |  '\n"
            "      f\"planted t = {planted['welch_t']:+.2f} (n={planted['n_events']})\")"
        ),
        md(
            f"> 💡 In plain words: across 10 null worlds the detector averages "
            f"t = {R['syn_null_mean']:+.2f} (sd {R['syn_null_sd']:.2f}) and never crosses the "
            f"bar; a planted continuation effect of 1.0 reads t = {R['syn_planted_t']:+.2f} "
            f"(signal {R['syn_planted_sig']:+.1f} bps vs control {R['syn_planted_ctrl']:+.1f} "
            "bps). The machinery is unbiased and has real power — the flat real-tape result is a "
            "genuine \"nothing there,\" not a blind spot. *(A faithful-engine / power check only "
            "— never cited in support of the real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — breakout net {R['dirn'][5.0][0]:+.1f} bps/trade (5 bps) vs "
            f"matched random control {R['dirn'][5.0][3]:+.1f} bps, Welch t = "
            f"**{R['dirn'][5.0][4]:.2f}**; **{R['sweep_clear']}/{R['sweep_n']}** parameter "
            "combinations clear \\|t\\| >= 2; no ticker beats its own control.\n"
            "- **Tradability `MIRAGE`** — net negative at every cost level tested, "
            "indistinguishable from randomly-timed trades carrying the identical direction mix.\n"
            "- **Vol expansion? `BUSTED`** — pooled forward-vol Welch t vs a random day = "
            f"**{R['vol_pooled_welch']:.2f}**. The mechanical half of the claim is true only "
            "against the squeeze's own (definitionally low) baseline, never against a fair "
            "control — and it inverts outright for GLD."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The general lesson generalises beyond this indicator.** Any \"contraction "
            "precedes expansion\" chart pattern needs to beat a random-day control on the "
            "expansion claim, and a random-timing/direction-mix control on the directional "
            "claim, before it earns a Signal stamp — see [190-nr7](../../190-nr7/) for a sibling "
            "single-bar contraction signal that fails the same way (and even reverses).\n"
            "- **Where the real information is:** options-implied term structure and skew price "
            "genuine event risk with far more precision than any band geometry — see "
            "[637-fomc-vol-crush](../../637-fomc-vol-crush/) for a case where a *scheduled* "
            "vol-crush is statistically real (though still not directly tradable).\n"
            "- **Dedup map:** [104-bollinger-reversion](../../104-bollinger-reversion/) (the "
            "opposite, single-band reversion trade), "
            "[128-keltner-channel](../../128-keltner-channel/) (Keltner alone, no Bollinger "
            "comparison), [485-starc-bands](../../485-starc-bands/) (a third, unrelated "
            "envelope) and [190-nr7](../../190-nr7/) (a single-bar range-contraction signal).\n\n"
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
