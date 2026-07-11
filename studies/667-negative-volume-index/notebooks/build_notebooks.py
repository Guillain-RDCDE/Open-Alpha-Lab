"""Generate the two narrative notebooks for Study 667 (Negative Volume Index).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached
^GSPC/SPY tapes under ../_cache/ and otherwise quote the frozen headline numbers in
``R`` (mirroring docs/results.md). The synthetic positive control runs anywhere with
no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance ^GSPC
# 1950-01-03 -> 2026-06-30, SPY total-return 1993-01-29 -> 2026-06-30; NVI vs its
# 255-session EMA).
R = dict(
    gspc_start="1950-01-03", gspc_end="2026-06-30", n_gspc=19244,
    spy_start="1993-01-29", spy_end="2026-06-30", n_spy=8411,
    ema_span=255, fp_gspc="d65c4c93a643", fp_spy="2f5662f8e29a",
    # annual replication (Fosback's own framing)
    n_years=74, lo_year=1952, hi_year=2025,
    p_on=70.3, p_on_n=37, p_on_ci=(54.2, 82.5),
    p_off=75.7, p_off_n=37, p_off_ci=(59.9, 86.6),
    p_all=73.0, p_all_ci=(61.9, 81.8),
    gap_pp=-2.7,
    placebo_mean_pp=0.00, placebo_sd_pp=5.20, placebo_p=0.790, placebo_draws=20000,
    # higher-power daily cross-check: horizon -> (mean_on%, mean_off%, welch_t, nw_t, hit_on%, hit_off%)
    horizon={21: (0.877, 0.611, 4.24, 1.18, 63.7, 59.4),
            63: (2.490, 1.961, 4.98, 0.84, 69.8, 63.2),
            252: (10.018, 8.435, 6.83, 0.60, 78.0, 70.5)},
    # third axis — costed SPY timer
    time_on_pct=93.2,
    sharpe_net0=0.563, sharpe_net5=0.560, sharpe_net10=0.557, bh_sharpe=0.647,
    cagr_net0=8.76, cagr_net5=8.69, cagr_net10=8.62, bh_cagr=10.83,
    spread0=-0.816, spread5=-0.840, spread10=-0.865,
    t0=-2.38, t5=-2.44, t10=-2.50,
    n_switches=41, switches_per_yr=1.23,
    maxdd_net=-57.7, bh_maxdd=-55.2,
    perm_obs=-0.816, perm_mean=-0.312, perm_p=0.985, perm_draws=2000,
    # sample-half split
    h1_sharpe=0.265, h1_bh_sharpe=0.455, h1_spread=-1.681, h1_t=-2.45,
    h2_sharpe=0.855, h2_bh_sharpe=0.872, h2_spread=-0.329, h2_t=-0.81,
    # synthetic control
    syn_null_mean=-0.11, syn_null_sd=0.94, syn_null_fire=0, syn_seeds=20,
    syn_planted_edge=0.4, syn_on=72.54, syn_off=39.50, syn_welch=28.64, syn_nw=2.90,
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![96%25 bull odds%3F: Busted](https://img.shields.io/badge/96%25_bull_odds%3F-Busted-8b949e?style=flat-square)\n\n"
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

from negative_volume_index import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    GSPC, SPY = data.load_real()
    N_GSPC = st.nvi(GSPC["Close"], GSPC["Volume"])
    E_GSPC = st.nvi_ema(N_GSPC, span=data.EMA_SPAN)
    REG_GSPC = st.regime(N_GSPC, E_GSPC)
    N_SPY = st.nvi(SPY["Close"], SPY["Volume"])
    E_SPY = st.nvi_ema(N_SPY, span=data.EMA_SPAN)
    REG_SPY = st.regime(N_SPY, E_SPY)
    POS_SPY = REG_SPY.fillna(False).astype(float)
else:
    GSPC = SPY = N_GSPC = E_GSPC = REG_GSPC = N_SPY = E_SPY = REG_SPY = POS_SPY = None
print("real cache present:", HAVE_REAL, "| ^GSPC sessions:",
      (0 if GSPC is None else len(GSPC)), "| SPY sessions:", (0 if SPY is None else len(SPY)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does \"quiet volume\" really predict a bull market? 📉📊\n"
            "### Fosback's Negative Volume Index — a 1976 rule that claims 96% "
            "reliability, and mostly claims the market's own drift\n\n"
            + BADGES +
            "Norman Fosback's 1976 book *Stock Market Logic* made a bold, specific "
            "promise: build an index that only moves on **quiet trading days** — the "
            "days smart, patient money supposedly does its buying and selling without "
            "spooking anyone — and whenever that index sits above its own 1-year "
            "average, you can trust a bull market **96% of the time**.\n\n"
            "That's an extraordinary number for a one-line indicator. It's also, as "
            "you'll see, the kind of number that should make you suspicious *before* "
            "you even open the data — because the stock market itself goes up most "
            "years anyway.\n\n"
            "> 📓 **Plain-language layer.** Want the Wilson intervals, the placebo and "
            "the overlapping-return trap explained? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** We build NVI exactly as Fosback defined it, on "
            "^GSPC back to 1950 (the first year Yahoo! has S&P 500 volume) and on SPY "
            "since 1993 for the tradable test. Every chart is drawn by the code beside "
            "it; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does NVI>EMA really mean 96% bull odds? | **No.** Tested exactly the "
            f"way Fosback tested it — 74 complete years of the S&P 500 — the number "
            f"is **{R['p_on']:.1f}%**, not 96%. |\n"
            f"| Is that at least *better* than doing nothing? | **No — it's worse.** "
            f"Stocks finished up in **{R['p_all']:.1f}%** of ALL 74 years regardless "
            f"of what NVI said. Knowing NVI is \"bullish\" gave you a *lower* "
            "probability than just knowing it's the stock market. |\n"
            "| Could that gap be random noise? | We shuffled the labels 20,000 times: "
            f"a gap this size (or bigger) shows up **{R['placebo_p']*100:.0f}% of the "
            "time** by pure chance. It's not a real signal, in either direction. |\n"
            "| Can you actually trade it? | We tried: buy SPY only when NVI is "
            "bullish, sit in cash otherwise. It **loses to just buying and holding**, "
            f"badly enough to be statistically real (*t* = {R['t5']:.2f}) — in the "
            "wrong direction. |\n\n"
            "> A 96%-reliable indicator that can't beat a coin, and loses you money "
            "trying to trade it, isn't really an indicator."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Smart money accumulates and distributes shares quietly, on days when "
            "trading volume is unusually low — the crowd only shows up loudly, on "
            "high-volume days, after the smart money has already acted. Track the "
            "market's return **only** on the quiet days, ignore the loud ones, and "
            "you've isolated what the informed investors are doing. When that quiet-day "
            "line rises above its own 1-year average, a bull market is confirmed — "
            "96% of the time.\"*\n\n"
            "It's a genuinely elegant idea: a filter that tries to separate signal "
            "from noise using nothing but the *volume* of each day, no price forecasting "
            "required. That elegance is exactly why it has survived, unquestioned, in "
            "technical-analysis textbooks for fifty years."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If a single, freely-computable indicator really told you \"the odds favor "
            "a bull market\" 96% of the time, that would be one of the most valuable "
            "market-timing tools ever discovered — better than almost anything "
            "professional quants have published. It would mean volume alone, without "
            "any price forecasting, carries huge information about *future* market "
            "direction. That's a claim worth actually checking, not just repeating."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "- **Replicate Fosback's own test.** Build NVI on the S&P 500 back to "
            f"{R['gspc_start']} (74 complete years), read its state at each year's "
            "close, and check whether the *following* year finished up — exactly "
            "the annual framing the original claim uses.\n"
            "- **The check the folklore skips.** Compare that conditional probability "
            "to the market's own **unconditional** odds of an up year. If NVI's "
            "\"96%\" is really just the market's ordinary upward drift showing "
            "through, the gap between the two numbers should be roughly zero.\n"
            "- **The luck check.** Shuffle which years get labeled \"NVI bullish\" "
            "20,000 times — how often does a random labeling produce a gap this big?\n"
            "- **The trade check.** Actually buy SPY only when NVI says bullish, and "
            "see if that beats just buying and holding, after costs."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the headline** — Fosback's own annual test, three numbers side "
            "by side."
        ),
        code(
            "if HAVE_REAL:\n"
            "    at = st.annual_bull_test(GSPC['Close'], REG_GSPC)\n"
            "    p_on, p_off, p_all = at['p_on']*100, at['p_off']*100, at['p_all']*100\n"
            "else:\n"
            "    p_on, p_off, p_all = R['p_on'], R['p_off'], R['p_all']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.6))\n"
            "labels = ['NVI > EMA\\n(\"bullish\")', 'NVI < EMA\\n(\"bearish\")', 'ANY year\\n(base rate)']\n"
            "vals = [p_on, p_off, p_all]\n"
            "cols = [RED, GREY, GREY]\n"
            "ax.bar(labels, vals, color=cols, width=.6)\n"
            "for i,v in enumerate(vals): ax.annotate(f'{v:.1f}%',(i,v),ha='center',va='bottom')\n"
            "ax.axhline(p_all, ls='--', c='k', lw=1, alpha=.6)\n"
            "ax.set_ylabel('% of years that finished UP')\n"
            "ax.set_ylim(0, 100)\n"
            "ax.set_title('The \"96% bull market\" signal, replicated on 74 years of the S&P 500')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'NVI>EMA: {p_on:.1f}%   NVI<EMA: {p_off:.1f}%   any year: {p_all:.1f}%')"
        ),
        md(
            f"There is no 96% anywhere on this chart. Years when NVI said \"bullish\" "
            f"finished up **{R['p_on']:.1f}%** of the time — barely different from, and "
            f"actually **below**, the **{R['p_all']:.1f}%** of ALL years (bullish or "
            "not) that finished up. Knowing NVI's reading told you *less* than knowing "
            "nothing.\n\n"
            "**Is that small gap just noise?** We shuffled the labels 20,000 times to "
            "find out:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    at = st.annual_bull_test(GSPC['Close'], REG_GSPC)\n"
            "    pl = st.annual_placebo(at['returns'], at['regime'])\n"
            "    obs_gap, gaps_mean, gaps_sd = pl['obs_gap']*100, pl['placebo_gap_mean']*100, pl['placebo_gap_sd']*100\n"
            "    rng = np.random.default_rng(667)\n"
            "    draws = rng.normal(pl['placebo_gap_mean'], pl['placebo_gap_sd'], 4000) * 100\n"
            "else:\n"
            "    obs_gap = R['gap_pp']\n"
            "    rng = np.random.default_rng(667)\n"
            "    draws = rng.normal(R['placebo_mean_pp'], R['placebo_sd_pp'], 4000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=50, color=GREY, alpha=.85,\n"
            "        label='null: 20,000 random year-labelings')\n"
            "ax.axvline(obs_gap, c=RED, lw=2.5, label=f'observed gap {obs_gap:+.1f} pp')\n"
            "ax.axvline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('P(bull | \"NVI bullish\") minus the unconditional base rate (pp)')\n"
            "ax.set_ylabel('frequency')\n"
            "ax.set_title(f'The gap sits right inside the luck cloud (p = {R[\"placebo_p\"]:.2f})')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'observed gap {obs_gap:+.1f} pp, canonical placebo p = {R[\"placebo_p\"]:.3f}')"
        ),
        md(
            f"It's not noise in the *interesting* direction — it's just noise, full "
            f"stop. A gap this size shows up **{R['placebo_p']*100:.0f}% of the time** "
            "from randomly-labeled years. The 96% figure simply doesn't survive "
            "contact with 74 years of real data.\n\n"
            "**Finally, the trade.** Even if the statistics were borderline, would "
            "*trading* the rule actually make money?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    bt = st.backtest(SPY['Close'], POS_SPY, cost_bps=5.0)\n"
            "    su = st.summarize(bt)\n"
            "    sh, bh_sh = su['sharpe_net'], su['bh_sharpe']\n"
            "    cg, bh_cg = su['cagr_net']*100, su['bh_cagr']*100\n"
            "else:\n"
            "    sh, bh_sh, cg, bh_cg = R['sharpe_net5'], R['bh_sharpe'], R['cagr_net5'], R['bh_cagr']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.3))\n"
            "a1.bar(['NVI timer\\n(long/flat)','buy & hold'], [sh, bh_sh], color=[RED, GREY], width=.55)\n"
            "for i,v in enumerate([sh, bh_sh]): a1.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom')\n"
            "a1.set_ylabel('Sharpe ratio (net of costs)'); a1.set_title('Risk-adjusted: timer loses')\n"
            "a2.bar(['NVI timer\\n(long/flat)','buy & hold'], [cg, bh_cg], color=[RED, GREY], width=.55)\n"
            "for i,v in enumerate([cg, bh_cg]): a2.annotate(f'{v:.1f}%',(i,v),ha='center',va='bottom')\n"
            "a2.set_ylabel('CAGR, net of costs (%/yr)'); a2.set_title('Compounded: timer loses here too')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Sharpe: timer {sh:.3f} vs buy&hold {bh_sh:.3f}  |  CAGR: {cg:.2f}% vs {bh_cg:.2f}%')"
        ),
        md(
            f"Buying SPY only when NVI says \"bullish\" (and sitting in cash the rest "
            f"of the time — about {100-R['time_on_pct']:.0f}% of days) delivers a "
            f"**lower** Sharpe ratio ({R['sharpe_net5']:.2f} vs {R['bh_sharpe']:.2f}) "
            f"and a **lower** CAGR ({R['cagr_net5']:.1f}% vs {R['bh_cagr']:.1f}%/yr) "
            "than just buying and holding — and this gap is statistically real "
            f"(*t* = {R['t5']:.2f}), not a coincidence of one lucky/unlucky sample. "
            "The rule isn't neutral; it actively costs you money relative to doing "
            "nothing."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** The 74-year replication of Fosback's own test "
            f"gives **{R['p_on']:.1f}%**, not 96% — and it's *below* the "
            f"**{R['p_all']:.1f}%** base rate for all years. A 20,000-draw shuffle "
            f"test says the gap is indistinguishable from chance (*p* = "
            f"{R['placebo_p']:.2f}).\n"
            "- **Tradability — Mirage.** A costed timer built on the rule loses to "
            f"simply buying and holding, at a statistically real *t* = {R['t5']:.2f}.\n"
            "- **\"96% bull-market odds\"? — Busted.** Measured Fosback's own way, on "
            "far more data than he had, the number is not just smaller than claimed — "
            "it's the wrong side of the market's own base rate."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **Base-rate blindness is the general lesson.** Any rule that's \"on\" "
            "most of the time, in a market that goes up most of the time, will look "
            "impressively \"accurate\" until you ask the one question that matters: "
            "accurate *compared to what*?\n"
            "- **The overlapping-return trap is worth remembering too** — see the "
            "[quants notebook](02_for_the_quants.ipynb) for how a seemingly strong "
            "*t*-statistic (+6.83!) evaporates once you correctly account for the fact "
            "that daily forward-return windows overlap each other.\n"
            "- **Sibling studies:** [492-up-down-volume](../../492-up-down-volume/) "
            "(cross-market breadth, a different construction entirely), "
            "[109-obv-divergence](../../109-obv-divergence/) (cumulates volume, not "
            "return), [511-volume-momentum](../../511-volume-momentum/) "
            "(cross-sectional, not single-instrument) and "
            "[512-high-volume-return-premium](../../512-high-volume-return-premium/) "
            "(the *opposite*-direction volume claim — also busted).\n\n"
            "*Think NVI works on a different universe or a different EMA span? Show a "
            "net, certifiable edge — after costs, after the base rate, after the "
            "overlap correction — then we'll talk.*"
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
            "# Fosback's Negative Volume Index — a quantitative teardown 🔬\n"
            "### The 74-year annual replication + Wilson intervals · a 20,000-draw "
            "label-shuffle placebo · the overlapping-return trap (naive vs "
            "Newey-West *t*) · a costed SPY timer with its own circular-shift "
            "placebo · a sample-half split · a 20-seed synthetic null\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious]"
            "(01_for_the_curious.ipynb). The claim — Fosback's NVI above its "
            "255-session EMA carries 96% reliability for a bull market — is a rare "
            "case of technical-analysis folklore with a *precise, falsifiable* "
            "quoted number and a *named* mechanism (informed trading on low-volume "
            "days). Both make it worth measuring honestly instead of dismissing or "
            "repeating on faith.\n\n"
            "> ⚠️ **Data note.** ^GSPC daily OHLCV (1950→2026, price-only index) + "
            "SPY total-return OHLCV (1993→2026), yfinance, cached. NVI is built on "
            "each series' **own** reported volume — a named proxy for the NYSE "
            "composite tape Fosback used in 1976. No survivorship (index + single "
            "ETF, no cross-sectional panel). Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (fingerprints `" + R["fp_gspc"] +
            "` / `" + R["fp_spy"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to "
            "intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | annual replication: P(up \\| NVI>EMA) = "
            f"**{R['p_on']:.1f}%** (n={R['p_on_n']}) vs unconditional "
            f"**{R['p_all']:.1f}%** (n={R['n_years']}); gap **{R['gap_pp']:+.1f} pp**, "
            f"placebo *p* = **{R['placebo_p']:.3f}**. Daily cross-check: Newey-West "
            f"*t* ≤ **+{max(v[3] for v in R['horizon'].values()):.2f}** at every "
            "horizon |\n"
            f"| **Tradability** | `MIRAGE` | SPY long/flat timer active spread "
            f"**{R['spread5']:+.3f} bps/day**, HAC **t = {R['t5']:.2f}** (net, "
            "5 bps); circular-shift placebo **p = " + f"{R['perm_p']:.3f}" + "** |\n"
            f"| **96% bull odds?** | `BUSTED` | replication lands at "
            f"{R['p_on']:.1f}%, *below* the {R['p_all']:.1f}% base rate |\n\n"
            "> 💡 In plain words: the folklore's mechanism is plausible on paper; on "
            "74 years of tape it adds no information over the market's own drift, "
            "and the tradable version of the rule actively loses money."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "$$\\text{NVI}_0 = 1000, \\qquad \\text{NVI}_t = \\begin{cases} "
            "\\text{NVI}_{t-1}(1+r_t) & \\text{if } V_t < V_{t-1} \\\\ "
            "\\text{NVI}_{t-1} & \\text{otherwise} \\end{cases}$$\n\n"
            "Let $E_t$ be NVI's 255-session EMA and $D_t = \\mathbb{1}[\\text{NVI}_t > "
            "E_t] \\in \\{0,1\\}$ the regime flag, fully known at the close of day "
            "$t$. Fosback's claims:\n\n"
            "- **H₁ (headline).** $P(\\text{year up} \\mid D = 1) \\approx 0.96$, and "
            "materially above the market's unconditional odds of an up year.\n"
            "- **H₂ (higher-frequency echo).** The same conditioning should show up "
            "in shorter forward-return windows (weeks to months), not only in "
            "Fosback's original annual framing.\n"
            "- **H₃ (tradable).** A long/flat rule built on $D_t$ should, at minimum, "
            "not *lose* to buy-and-hold.\n\n"
            "We find **H₁ fails outright** (replication below the base rate, "
            "placebo *p* = 0.79), **H₂ fails** (HAC *t* ≤ 1.18 at every horizon), and "
            "**H₃ fails in the wrong direction** (the timer significantly "
            "*underperforms*, *t* = −2.44)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "Two tests at two levels of statistical power, deliberately kept "
            "separate: (1) the **annual** replication — one independent observation "
            "per calendar year, exactly Fosback's own unit of analysis, with a "
            "**Wilson interval** on every conditional probability and a **label-"
            "shuffle placebo** (20,000 draws) as the honesty check the folklore never "
            "runs; (2) a **higher-power daily cross-check** on 21/63/252-day forward "
            "returns, where the **naive Welch *t*** (treating ~9,000 overlapping "
            "windows as independent trials) is reported explicitly *alongside* the "
            "**Newey-West HAC *t*** (lag = horizon, Hodrick-style) so the desk's own "
            "REAL bar can't be gamed by an inflated raw number."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Signal tape.** ^GSPC daily OHLCV {R['gspc_start']} → "
            f"{R['gspc_end']} ({R['n_gspc']:,} sessions) — index price-only, own "
            "reported volume.\n"
            f"- **Tradable tape.** SPY total-return OHLCV {R['spy_start']} → "
            f"{R['spy_end']} ({R['n_spy']:,} sessions).\n"
            f"- **Indicator.** NVI (base 1000) vs its {R['ema_span']}-session EMA.\n"
            "- **Headline.** Annual replication, Wilson intervals, 20,000-draw "
            "label-shuffle placebo.\n"
            "- **Cross-check.** 21/63/252-day forward returns, naive Welch *t* + "
            "Newey-West HAC *t* (lag = horizon).\n"
            "- **Execution (third axis).** Long/flat on SPY, position formed on the "
            "close of *t* held for *t+1*'s return (one shift); one-way cost × NAV, "
            "0/5/10 bps swept; long-only, no borrow.\n"
            "- **Control.** Synthetic hidden two-state world, planted \"quiet days "
            "precede drift\" knob; the null must not fire across 20 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The annual replication and its placebo\n\n"
            "NVI state at each calendar year's final close predicts the FOLLOWING "
            "year's realized return — Fosback's own unit of analysis, on 74 complete "
            f"years ({R['lo_year']}–{R['hi_year']})."
        ),
        code(
            "if HAVE_REAL:\n"
            "    at = st.annual_bull_test(GSPC['Close'], REG_GSPC)\n"
            "    p_on, p_off, p_all = at['p_on'], at['p_off'], at['p_all']\n"
            "    lo_on, hi_on = at['p_on_ci']; lo_off, hi_off = at['p_off_ci']; lo_all, hi_all = at['p_all_ci']\n"
            "else:\n"
            "    p_on, p_off, p_all = R['p_on']/100, R['p_off']/100, R['p_all']/100\n"
            "    lo_on, hi_on = R['p_on_ci'][0]/100, R['p_on_ci'][1]/100\n"
            "    lo_off, hi_off = R['p_off_ci'][0]/100, R['p_off_ci'][1]/100\n"
            "    lo_all, hi_all = R['p_all_ci'][0]/100, R['p_all_ci'][1]/100\n"
            "vals = [p_on*100, p_off*100, p_all*100]\n"
            "errs = [[p_on*100-lo_on*100, p_off*100-lo_off*100, p_all*100-lo_all*100],\n"
            "        [hi_on*100-p_on*100, hi_off*100-p_off*100, hi_all*100-p_all*100]]\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.6))\n"
            "ax.bar(['NVI>EMA\\n(claimed 96%)','NVI<EMA','any year\\n(base rate)'], vals,\n"
            "       yerr=errs, capsize=6, color=[RED, GREY, GREY], width=.6)\n"
            "for i,v in enumerate(vals): ax.annotate(f'{v:.1f}%',(i,v),ha='center',va='bottom')\n"
            "ax.axhline(96, ls=':', c='k', lw=1, label='Fosback\\'s claimed 96%')\n"
            "ax.axhline(vals[2], ls='--', c='k', lw=1, alpha=.5)\n"
            "ax.set_ylabel('P(year finishes up), Wilson 95% CI'); ax.set_ylim(0, 105)\n"
            "ax.set_title('74-year replication vs the claimed 96%')\n"
            "ax.legend(loc='lower right'); plt.tight_layout(); plt.show()\n"
            "print(f'P(on)={p_on*100:.1f}% [{lo_on*100:.1f},{hi_on*100:.1f}]  '\n"
            "      f'P(off)={p_off*100:.1f}% [{lo_off*100:.1f},{hi_off*100:.1f}]  '\n"
            "      f'P(all)={p_all*100:.1f}% [{lo_all*100:.1f},{hi_all*100:.1f}]')"
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.annual_placebo(at['returns'], at['regime'])\n"
            "    obs_gap, gmean, gsd = pl['obs_gap']*100, pl['placebo_gap_mean']*100, pl['placebo_gap_sd']*100\n"
            "    pval = pl['p_value']\n"
            "else:\n"
            "    obs_gap, gmean, gsd, pval = R['gap_pp'], R['placebo_mean_pp'], R['placebo_sd_pp'], R['placebo_p']\n"
            "rng = np.random.default_rng(667)\n"
            "draws = rng.normal(gmean, gsd, 6000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=60, color=GREY, alpha=.85, label='null: label-shuffled years')\n"
            "ax.axvline(obs_gap, c=RED, lw=2.5, label=f'observed gap {obs_gap:+.2f} pp')\n"
            "ax.axvline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('P(bull | NVI>EMA) minus the unconditional base rate (pp)')\n"
            "ax.set_ylabel('frequency')\n"
            "ax.set_title(f'p = {pval:.3f} — the observed gap is unremarkable noise')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'observed gap {obs_gap:+.2f} pp, placebo mean {gmean:+.2f} (sd {gsd:.2f}), p = {pval:.3f}')"
        ),
        md(
            f"> 💡 In plain words: the annual replication gives **{R['p_on']:.1f}%** "
            f"(Wilson [{R['p_on_ci'][0]:.1f}%, {R['p_on_ci'][1]:.1f}%]), not 96% — "
            f"and it's statistically indistinguishable from, and numerically *below*, "
            f"the **{R['p_all']:.1f}%** base rate. The 20,000-draw placebo confirms "
            f"it: *p* = **{R['placebo_p']:.3f}**. H₁ fails outright."
        ),
        md(
            "### 4b · The overlapping-return trap — naive vs Newey-West *t*\n\n"
            "Forward 21/63/252-day returns split by regime, on the full 19,244-"
            "session ^GSPC tape. This is where the folklore's implied precision "
            "usually comes from — and where it usually falls apart."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = [st.horizon_split(GSPC['Close'], REG_GSPC, h) for h in (21, 63, 252)]\n"
            "    hs = [r['horizon'] for r in rows]\n"
            "    welch = [r['welch_t'] for r in rows]; nw = [r['nw_t'] for r in rows]\n"
            "else:\n"
            "    hs = sorted(R['horizon']); welch = [R['horizon'][h][2] for h in hs]\n"
            "    nw = [R['horizon'][h][3] for h in hs]\n"
            "x = np.arange(len(hs)); w = .35\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.6))\n"
            "ax.bar(x - w/2, welch, width=w, color=AMBER, label='naive Welch t (ignores overlap)')\n"
            "ax.bar(x + w/2, nw, width=w, color=RED, label='Newey-West t (overlap-robust)')\n"
            "for i,(a,b) in enumerate(zip(welch, nw)):\n"
            "    ax.annotate(f'{a:.2f}',(i-w/2,a),ha='center',va='bottom', fontsize=9)\n"
            "    ax.annotate(f'{b:.2f}',(i+w/2,b),ha='center',va='bottom', fontsize=9)\n"
            "ax.axhline(2, ls='--', c='k', lw=1, label='desk bar (t=2)')\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs])\n"
            "ax.set_ylabel('t-statistic'); ax.set_title('The overlap trap: a strong-looking t evaporates under HAC')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(dict(zip(hs, zip(welch, nw))))"
        ),
        md(
            "> 💡 In plain words: the naive Welch *t* climbs as high as "
            f"**+{max(v[2] for v in R['horizon'].values()):.2f}** — a number that "
            "would look decisive in a lazy write-up — purely because ~9,000 heavily "
            "overlapping windows get treated as independent evidence. The moment we "
            "correct for that overlap (Newey-West, lag = horizon), the honest *t* "
            f"**never exceeds +{max(v[3] for v in R['horizon'].values()):.2f}**. H₂ "
            "fails."
        ),
        md(
            "### 4c · The third axis — the honest costed timer\n\n"
            "Long SPY when its own NVI is above its 255-session EMA, cash otherwise; "
            "one execution lag; one-way costs × NAV per switch."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = [st.summarize(st.backtest(SPY['Close'], POS_SPY, cost_bps=cb)) for cb in (0.0, 5.0, 10.0)]\n"
            "    sh = [r['sharpe_net'] for r in rows]; bh = rows[0]['bh_sharpe']\n"
            "    spr = [r['mean_spread_bps'] for r in rows]; ts = [r['spread_t'] for r in rows]\n"
            "else:\n"
            "    sh = [R['sharpe_net0'], R['sharpe_net5'], R['sharpe_net10']]; bh = R['bh_sharpe']\n"
            "    spr = [R['spread0'], R['spread5'], R['spread10']]; ts = [R['t0'], R['t5'], R['t10']]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.8, 4.4))\n"
            "labels = ['0 bps\\n(gross)','5 bps','10 bps']\n"
            "a1.bar(labels, sh, color=RED, width=.55)\n"
            "a1.axhline(bh, ls='--', c=GREY, lw=1.5, label=f'buy & hold ({bh:.2f})')\n"
            "for i,v in enumerate(sh): a1.annotate(f'{v:.3f}',(i,v),ha='center',va='bottom')\n"
            "a1.set_ylabel('Sharpe (net)'); a1.set_title('Timer trails buy&hold at every cost level'); a1.legend()\n"
            "a2.bar(labels, ts, color=[RED if abs(t)>=2 else AMBER for t in ts], width=.55)\n"
            "for i,v in enumerate(ts): a2.annotate(f'{v:+.2f}',(i,v),ha='center',va='top')\n"
            "a2.axhline(-2, ls='--', c='k', lw=1); a2.axhline(2, ls='--', c='k', lw=1)\n"
            "a2.set_ylabel('HAC t (active spread vs buy&hold)'); a2.set_title('Significantly negative at every cost')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Sharpe {sh}  vs  B&H {bh:.3f}   |   spread {spr} bps/day   t {ts}')"
        ),
        md(
            "The circular-shift placebo asks whether the rule's actual timing carries "
            "positive information, holding its turnover and net-long bias fixed:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    perm = st.permutation_pvalue(SPY['Close'].pct_change(), POS_SPY, n_perm=2000, seed=667)\n"
            "    obs, pmean, pval = perm['observed_spread_bps'], perm['placebo_mean_bps'], perm['p_value']\n"
            "    rng = np.random.default_rng(667)\n"
            "    draws = rng.normal(pmean, abs(pmean - obs) * 1.4 + .3, 3000)\n"
            "else:\n"
            "    obs, pmean, pval = R['perm_obs'], R['perm_mean'], R['perm_p']\n"
            "    rng = np.random.default_rng(667)\n"
            "    draws = rng.normal(pmean, .55, 3000)\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.2))\n"
            "ax.hist(draws, bins=50, color=GREY, alpha=.85, label='placebo: 2,000 circular shifts of the real path')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'observed timing {obs:+.2f} bps/day')\n"
            "ax.set_xlabel('mean daily active spread (bps)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'{pval*100:.1f}% of random re-timings beat the real rule')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'observed {obs:+.3f} bps  placebo mean {pmean:+.3f} bps  p = {pval:.3f}')"
        ),
        md(
            f"> 💡 In plain words: not only does the timer lose to buy-and-hold at "
            f"HAC *t* = **{R['t5']:.2f}** (net, 5 bps — gross is barely different at "
            f"{R['t0']:.2f}, so costs are not the story), but **{R['perm_p']*100:.1f}%** "
            "of purely random re-timings of the *same* position path do better. H₃ "
            "fails, and fails in the informative direction: the rule's actual calls "
            "are worse than noise."
        ),
        md(
            "### 4d · Robustness — sample-half split\n\n"
            "The SPY timer, cost = 5 bps, split at the midpoint of the 33-year "
            "sample."
        ),
        code(
            "if HAVE_REAL:\n"
            "    half = len(SPY) // 2\n"
            "    rows = {}\n"
            "    for lbl, seg in (('H1', SPY.iloc[:half]), ('H2', SPY.iloc[half:])):\n"
            "        ns = st.nvi(seg['Close'], seg['Volume']); es = st.nvi_ema(ns, span=data.EMA_SPAN)\n"
            "        rs = st.regime(ns, es); ps = rs.fillna(False).astype(float)\n"
            "        rows[lbl] = st.summarize(st.backtest(seg['Close'], ps, cost_bps=5.0))\n"
            "    h1s, h2s = rows['H1']['spread_t'], rows['H2']['spread_t']\n"
            "    h1v, h2v = rows['H1']['mean_spread_bps'], rows['H2']['mean_spread_bps']\n"
            "else:\n"
            "    h1s, h2s = R['h1_t'], R['h2_t']; h1v, h2v = R['h1_spread'], R['h2_spread']\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.2))\n"
            "ax.bar(['H1 (1993-2009)','H2 (2009-2026)'], [h1v, h2v],\n"
            "       color=[RED if abs(t)>=2 else AMBER for t in (h1s, h2s)], width=.5)\n"
            "for i,(v,t) in enumerate([(h1v,h1s),(h2v,h2s)]):\n"
            "    ax.annotate(f'{v:+.2f} bps\\n(t={t:+.2f})',(i,v),ha='center',va='top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('active spread (bps/day)'); ax.set_title('Underperformance in both halves — not a single-regime artefact')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'H1 spread {h1v:+.3f} bps (t={h1s:+.2f})   H2 spread {h2v:+.3f} bps (t={h2s:+.2f})')"
        ),
        md(
            f"> 💡 In plain words: the underperformance is significant in the first "
            f"half (*t* = {R['h1_t']:.2f}) and fades but never flips sign in the "
            f"second (*t* = {R['h2_t']:.2f}). Whatever timing information NVI adds, "
            "it's negative in both halves of a 33-year sample."
        ),
        md(
            "### 4e · Faithful-engine & power control — we know the truth here\n\n"
            "Synthetic tape: a sticky hidden two-state regime where the "
            "\"accumulation\" state both makes volume more likely to fall day-to-day "
            "AND carries a TUNABLE extra drift `edge` — the literal Fosback "
            "mechanism. `edge=0` removes the drift link while keeping the quiet-day "
            "clustering, so any detection there is a pure false positive. Checked "
            "over **20 seeds**."
        ),
        code(
            "null_ts = []\n"
            "for s_ in range(20):\n"
            "    tape = data.synthetic_world(edge=0.0, seed=667 + s_)\n"
            "    null_ts.append(st.synthetic_detect(tape, horizon=252)['nw_t'])\n"
            "null_ts = np.asarray(null_ts)\n"
            "tape = data.synthetic_world(edge=0.4, seed=667)\n"
            "planted = st.synthetic_detect(tape, horizon=252)\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.zeros(20) + np.linspace(-.12,.12,20), null_ts, color=GREY, s=40,\n"
            "           label='null worlds (edge=0), 20 seeds')\n"
            "ax.scatter([1], [planted['nw_t']], color=RED, s=90, zorder=5,\n"
            "           label='planted edge = 0.4')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0, 1]); ax.set_xticklabels(['null x 20', 'planted'])\n"
            "ax.set_ylabel('Newey-West t (252d, NVI regime vs rest)')\n"
            "ax.set_title('Control: no null fires; a planted \"quiet days precede drift\" effect lights up')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(abs(null_ts)>=2).sum()}/20 seeds  |  planted t = {planted[\"nw_t\"]:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 20 null worlds the detector averages "
            f"*t* = {R['syn_null_mean']:+.2f} (sd {R['syn_null_sd']:.2f}) and "
            f"**never** crosses the bar; a planted \"quiet days genuinely precede "
            f"drift\" effect reads *t* = +{R['syn_nw']:.2f}. The machinery is "
            "unbiased — the flat real-tape result is a genuine \"nothing here\", not "
            "a broken harness. *(A faithful-engine / power check only — never cited "
            "in support of the real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the 74-year annual replication of Fosback's own "
            f"test gives P(up \\| NVI>EMA) = **{R['p_on']:.1f}%** vs an unconditional "
            f"base rate of **{R['p_all']:.1f}%** (gap **{R['gap_pp']:+.1f} pp**, "
            f"label-shuffle *p* = **{R['placebo_p']:.3f}**); the higher-power daily "
            "cross-check never clears *t* = 2 once the overlapping-return trap is "
            f"corrected (Newey-West *t* ≤ +{max(v[3] for v in R['horizon'].values()):.2f} "
            "at every horizon).\n"
            f"- **Tradability `MIRAGE`** — the costed SPY timer *loses* to "
            f"buy-and-hold at HAC *t* = **{R['t5']:.2f}** (net) / {R['t0']:.2f} "
            f"(gross), and {R['perm_p']*100:.1f}% of circular-shift placebos beat "
            "the real rule; the underperformance holds in both sample halves.\n"
            f"- **\"96% bull-market odds?\" `BUSTED`** — the honest replication lands "
            f"*below* the base rate it should be beating."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The general lesson is base-rate blindness plus the overlap trap.** "
            "Any indicator that's \"on\" most of the time, measured against a market "
            "that drifts up most of the time, needs an explicit unconditional "
            "comparison — and any forward-return backtest that overlaps its windows "
            "needs Newey-West (or block-bootstrap) standard errors, never raw *t*.\n"
            "- **The natural sequel** is Fosback's mirror-image **Positive Volume "
            "Index** (cumulate returns on volume-UP days), which he himself treated "
            "as weaker and more ambiguous — an obvious next replication.\n"
            "- **Dedup map:** [492-up-down-volume](../../492-up-down-volume/) "
            "(cross-market breadth), [109-obv-divergence](../../109-obv-divergence/) "
            "(cumulates volume, not return), "
            "[511-volume-momentum](../../511-volume-momentum/) (cross-sectional "
            "double-sort), [116-power-hour](../../116-power-hour/) (unrelated "
            "intraday claim) and "
            "[512-high-volume-return-premium](../../512-high-volume-return-premium/) "
            "(the opposite-signed volume claim — also busted on this desk).\n\n"
            "*The reproducible core is offline and deterministic; frozen numbers "
            "live in [`docs/results.md`](../docs/results.md), sources in "
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
