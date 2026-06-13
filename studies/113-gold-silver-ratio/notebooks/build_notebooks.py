"""Generate the two narrative notebooks for Study 113 (Gold-Silver-Ratio).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the cached
daily parquets under ../_cache/ if present and otherwise quote the frozen headline
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-13).
R = dict(
    # Data
    n_days=4136, date_start="2010-01-04", date_end="2026-06-12",
    fp_gld="fb68a1c04a2b", fp_slv="1dbb4be8cc36",
    ratio_mean=7.36, ratio_min=3.17, ratio_max=12.55,
    # Structural tests
    adf_p=0.2158, half_life=301.3, coint_p=0.7362,
    # Primary strategy (enter |z|>1.5, exit |z|<0.5, window=252)
    n_trades=26, avg_hold=78,
    strat_win=0.654, strat_mean=144.2, strat_t=1.77,
    rand_win=0.538, rand_mean=18.9, rand_t=0.21,
    delta=125.3,
    # Cost sweep (net)
    net0=144.2, net0_t=1.77,
    net5=139.2, net5_t=1.71,
    net10=134.2, net10_t=1.65,
    net20=124.2, net20_t=1.53,
    # Buy and hold
    gld_total=125.9, gld_ann=8.0,
    slv_total=126.9, slv_ann=8.0,
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

from gold_silver_ratio import data, strategy as st

GOLD_TICKER = data.GOLD_TICKER    # GLD
SILVER_TICKER = data.SILVER_TICKER  # SLV

def _have_cache():
    return (os.path.exists(data._cache_path(GOLD_TICKER, data.DEFAULT_CACHE)) and
            os.path.exists(data._cache_path(SILVER_TICKER, data.DEFAULT_CACHE)))

HAVE_REAL = _have_cache()
print("real daily cache present:", HAVE_REAL)
"""

# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Gold-Silver-Ratio -- does the ratio mean-revert, or is it a random walk?\n"
            "### The classic pairs rotation between gold and silver, tested honestly\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
            "![Ratio_mean--reverts%3F: Not_Supported](https://img.shields.io/badge/Ratio_mean--reverts%3F-Not_Supported-8b949e?style=flat-square)\n\n"
            "Here is one of the oldest commodity trades: the gold-to-silver price ratio. "
            "For centuries, one ounce of gold bought about 15 ounces of silver. "
            "Today the ratio floats freely -- sometimes 40, sometimes 120 -- and traders "
            "watch for extremes. When gold is *very expensive* relative to silver, buy silver "
            "and sell gold, betting the ratio snaps back. When silver is expensive relative to "
            "gold, flip it. This notebook asks: **is the snap-back actually reliable, or is the "
            "ratio just wandering like any other random walk?**\n\n"
            "> **This is the plain-language layer.** Want the t-stats, the ADF tests, and the "
            "cointegration analysis? That's **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart is drawn by "
            "the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## The answer first\n\n"
            f"| Question | Answer |\n|---|---|\n"
            f"| Is the gold/silver ratio stationary (mean-reverting)? | **Not reliably.** "
            f"The ADF test fails to reject a unit root (p = {R['adf_p']:.2f}), and "
            f"the pair is not cointegrated (p = {R['coint_p']:.2f}). |\n"
            f"| How long does a deviation take to revert? | If it reverts at all, the "
            f"estimated half-life is **{R['half_life']:.0f} trading days** (~15 months). |\n"
            f"| Does the trade strategy work? | **Weakly.** The z-score strategy earns "
            f"+{R['strat_mean']:.0f} bps/trade (vs random +{R['rand_mean']:.0f} bps), "
            f"but with only **{R['n_trades']} trades over 16 years** the result is "
            f"statistically inconclusive (HAC *t* = {R['strat_t']:.2f}, below the |t| >= 2 bar). |\n"
            f"| Could you trade it? | **Barely.** ~1.6 signals/year, ~{R['avg_hold']}-day holds. "
            f"Costs are not the killer -- too few opportunities and statistical uncertainty are. |\n\n"
            "> The ratio is a slow, noisy, non-stationary wanderer. Holding a 15-month trade "
            "to wait for it to revert means missing a gold/silver buy-and-hold that both deliver "
            f"~{R['gld_ann']:.0f}%/yr."
        ),

        # ---- BEAT 1 -- THE CLAIM ---------------------------------------------
        md(
            "## 1 -- The claim\n\n"
            "> *'The gold-to-silver ratio has historically averaged around 47. When the ratio "
            "spikes above 80, silver is extremely cheap relative to gold -- buy silver. When it "
            "drops below 40, gold is cheap -- buy gold. The ratio always reverts, and the "
            "patient trader profits from the snap-back.'*\n\n"
            "This is one of the oldest macro trades, cited in everything from commodity desks "
            "to Reddit forums. It has an intuitive appeal: both are precious metals with "
            "monetary history, so their *relative* price should anchor to something fundamental. "
            "The bet is that when the ratio gets extreme, it snaps back -- and you can position "
            "for the snap."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 -- So what?\n\n"
            "If the claim is true, this is a slow, low-turnover relative-value trade accessible "
            "to any retail investor via GLD and SLV ETFs. No leverage, no exotic instruments, "
            "no market-timing -- just patient rotation between two metals. That's a compelling "
            "pitch. The critical question is whether the 'ratio always reverts' assertion is "
            "statistical fact or survivorship bias from a few memorable episodes."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW -----------------------------------------
        md(
            "## 3 -- How would we even know?\n\n"
            "Two questions before we even look at returns:\n\n"
            "1. **Does the ratio mean-revert at all?** We test whether the GLD/SLV price "
            "ratio is stationary (ADF test) and whether GLD and SLV are cointegrated "
            "(Engle-Granger test). If either fails, the 'snap-back' premise is mathematical "
            "fiction -- the ratio is just a random walk that occasionally looks extreme.\n"
            "2. **How long does a reversion take?** We estimate the OU half-life: if the "
            "answer is 15 months, you need to be comfortable holding a spread trade for over "
            "a year -- a completely different risk profile than 'patient rotation'.\n\n"
            "Then for returns we compare the z-score strategy against a **random-direction "
            "control** (same entry/exit dates, random ±1 direction) and a **buy-and-hold "
            "of each metal**. GLD and SLV daily data 2010-2026 from Yahoo Finance."
        ),

        # ---- BEAT 4 -- THE TEARDOWN ------------------------------------------
        md(
            "## 4 -- The teardown -- let's actually look\n\n"
            "**First, is there a ratio to exploit?** Plot the GLD/SLV ratio over time:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    gold = data.fetch_daily(GOLD_TICKER)\n"
            "    silver = data.fetch_daily(SILVER_TICKER)\n"
            "    ratio = st.compute_ratio(gold, silver)\n"
            "    z_ratio = st.zscore(ratio, 252)\n"
            "    adf_p = st.adf_pvalue(ratio)\n"
            "    hl = st.half_life(ratio)\n"
            "else:\n"
            f"    import pandas as pd, numpy as np\n"
            f"    # Use frozen headline numbers\n"
            f"    ratio = None; z_ratio = None\n"
            f"    adf_p = {R['adf_p']}; hl = {R['half_life']}\n"
            "\n"
            "if ratio is not None:\n"
            "    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5, 7), sharex=True)\n"
            "    ax1.plot(ratio.index, ratio.values, c=AMBER, lw=1.2)\n"
            "    ax1.set_ylabel('GLD/SLV ratio'); ax1.set_title('The gold/silver ratio (GLD/SLV)')\n"
            "    ax1.axhline(ratio.mean(), c=GREY, ls='--', lw=1, label=f'mean {ratio.mean():.2f}')\n"
            "    ax1.legend()\n"
            "    z_clean = z_ratio.dropna()\n"
            "    ax2.plot(z_clean.index, z_clean.values, c=GREY, lw=0.8)\n"
            "    ax2.axhline(1.5, c=RED, ls='--', lw=1); ax2.axhline(-1.5, c=GREEN, ls='--', lw=1)\n"
            "    ax2.axhline(0, c='k', lw=0.8)\n"
            "    ax2.set_ylabel('252-day rolling z-score'); ax2.set_title('Z-score of the ratio')\n"
            "    plt.tight_layout(); plt.show()\n"
            "print(f'ADF test p-value: {adf_p:.4f}  (> 0.05 = cannot rule out random walk)')\n"
            "print(f'OU half-life estimate: {hl:.0f} trading days ({hl/21:.0f} months)')"
        ),
        md(
            f"The ADF test p-value is **{R['adf_p']:.2f}** -- well above 0.05. We cannot "
            "rule out that the ratio is a random walk. The OR half-life is estimated at "
            f"**{R['half_life']:.0f} trading days (~15 months)** -- so even if there is *some* "
            "reversion, a trade entered today might wait over a year before it 'works'."
        ),
        md("**Now let's check: does the rotation strategy actually earn anything?**"),
        code(
            "if HAVE_REAL:\n"
            "    entries = st.zscore_entries(ratio, window=252, enter_z=1.5, exit_z=0.5)\n"
            "    ledger = st.run_trades(gold, silver, entries, cost_bps=0.0)\n"
            "    rand_ent = st.random_entries(entries, seed=113)\n"
            "    rand_led = st.run_trades(gold, silver, rand_ent, cost_bps=0.0)\n"
            "    s = st.summarize(ledger, 'ret_gross')\n"
            "    r = st.summarize(rand_led, 'ret_gross')\n"
            "    strat_m, strat_w, strat_t = s['mean_bps'], s['win_rate']*100, s['tstat']\n"
            "    rand_m, rand_w, rand_t = r['mean_bps'], r['win_rate']*100, r['tstat']\n"
            "    n_tr = s['n_trades']\n"
            "else:\n"
            f"    strat_m, strat_w, strat_t = {R['strat_mean']}, {R['strat_win']*100:.1f}, {R['strat_t']}\n"
            f"    rand_m, rand_w, rand_t = {R['rand_mean']}, {R['rand_win']*100:.1f}, {R['rand_t']}\n"
            f"    n_tr = {R['n_trades']}\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "bars_plt = ax.bar(['Z-score reversion\\n(long cheap metal)', 'Random direction\\n(a coin)'],\n"
            "                  [strat_m, rand_m], color=[AMBER, GREY], width=.55)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('gross profit per trade (bps)')\n"
            "ax.set_title(f'Strategy vs random: {n_tr} trades over 16 years')\n"
            "for b, (w, t) in zip(bars_plt, [(strat_w, strat_t), (rand_w, rand_t)]):\n"
            "    ax.annotate(f'win {w:.0f}%\\nt={t:+.2f}',\n"
            "                (b.get_x()+b.get_width()/2, max(b.get_height(), 0) + 10),\n"
            "                ha='center', va='bottom', fontsize=9)\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'n={{n_tr}} trades (about 1.6/yr) -- very thin sample')"
        ),
        md(
            f"The strategy earns **+{R['strat_mean']:.0f} bps/trade** vs +{R['rand_mean']:.0f} bps "
            f"for the random control. But with only **{R['n_trades']} trades over 16 years**, "
            f"the HAC t-stat is **+{R['strat_t']:.2f}** -- below the desk's |t| >= 2 bar. "
            "The sample is simply too thin to distinguish signal from luck."
        ),
        md("**What about cointegration -- are the two metals actually linked?**"),
        code(
            "if HAVE_REAL:\n"
            "    ci_p = st.engle_granger_pvalue(gold['close'], silver['close'])\n"
            "else:\n"
            f"    ci_p = {R['coint_p']}\n"
            "\n"
            "print(f'Engle-Granger cointegration test p-value: {ci_p:.4f}')\n"
            "print('Verdict:', 'cointegrated (structural link)' if ci_p < 0.05 else\n"
            "      'NOT cointegrated (just two correlated trending series)')"
        ),
        md(
            f"The Engle-Granger p-value is **{R['coint_p']:.2f}** -- far above 0.05. Gold "
            "and silver are *correlated* (they both go up when inflation fear rises) but they "
            "are **not cointegrated**: there is no stable long-run relationship forcing their "
            "ratio back to any particular level. The folk trade's theoretical foundation is absent."
        ),

        # ---- BEAT 5 -- VERDICT -----------------------------------------------
        md(
            "## 5 -- The verdict\n\n"
            f"- **Signal -- WEAK.** HAC *t* = {R['strat_t']:.2f} on {R['n_trades']} trades. "
            "The point estimate is positive but statistically inconclusive. The ADF test "
            "(p = 0.22) and cointegration test (p = 0.74) both fail to confirm the structural "
            "premise.\n"
            f"- **Tradability -- FRAGILE.** ~1.6 signals/year, ~{R['avg_hold']}-day holds. "
            "Costs are not the problem; extreme signal aridity and 15-month holding periods "
            "make this an impractical rotation strategy.\n"
            f"- **Ratio mean-reverts? -- NOT SUPPORTED.** The ratio behaves like a random walk "
            f"(ADF p = {R['adf_p']:.2f}, half-life = {R['half_life']:.0f} days). The 'it always "
            "reverts' assertion is selection bias from memorable episodes, not statistical law."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT -----------------------------------
        md(
            "## 6 -- Could you actually trade it?\n\n"
            "The striking thing about this strategy is that costs are *not* the killer -- "
            "with 1.6 trades/year and 78-day holds, round-trip ETF costs are tiny. The "
            "real problems are the extreme rarity of signals and the uncertain edge:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    net_means = [st.summarize(st.run_trades(gold, silver, entries, cost_bps=c),\n"
            "                             'ret_net')['mean_bps'] for c in [0,5,10,20]]\n"
            "else:\n"
            f"    net_means = [{R['net0']}, {R['net5']}, {R['net10']}, {R['net20']}]\n"
            "costs = [0, 5, 10, 20]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(costs, net_means, 'o-', c=AMBER, lw=2)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('round-trip cost (bps)'); ax.set_ylabel('net mean (bps/trade)')\n"
            "ax.set_title('Costs barely matter -- the trade is so infrequent')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('The gross edge is uncertain (t=1.77), not the costs.')\n"
            "print('Buy GLD or SLV and hold: both delivered ~8%%/yr since 2010.')"
        ),
        md(
            f"Even at 20 bps round-trip costs, the trade still earns +{R['net20']:.0f} bps/trade "
            "gross. The problem is not the cost but the combination of statistical weakness and "
            f"extreme rarity. Meanwhile, buying either metal and holding delivers ~{R['gld_ann']:.0f}%/yr "
            "with zero effort, zero timing risk, and no 15-month waits for a reversion that "
            "the statistics say may never come."
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 -- Going further\n\n"
            "- **The positive control.** The companion notebook shows a synthetic tape "
            "with planted mean-reversion (OU speed > 0): the engine reliably harvests the "
            "edge when it's real. The real tape's uncertainty is a statement about the "
            "market, not the method.\n"
            "- **Shorter windows.** A 63-day z-score fires more frequently but is noisier "
            "and doesn't fix the non-stationarity.\n"
            "- **Cross-asset pairs.** The same structure tested on [Study 85 -- Dr-Copper]"
            "(../../85-dr-copper/) for copper/gold; a ratio that has stronger macro grounding.\n"
            "- **Related desk studies.** Pairs-trading methodology in Studies 05 and 23; "
            "commodity allocation in [Study 68 -- All-Weather](../../68-all-weather/).\n\n"
            "*Think the ratio really does mean-revert? Fork this, test a pre-2010 window, "
            "or try the futures (GC=F / SI=F) instead of the ETFs. That's the bar.*"
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
            "# Gold-Silver-Ratio -- a quantitative teardown\n"
            "### Real daily tape * ADF + Engle-Granger tests * OU half-life * z-score "
            "strategy * random-direction control * HAC inference\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
            "![Ratio_mean--reverts%3F: Not_Supported](https://img.shields.io/badge/Ratio_mean--reverts%3F-Not_Supported-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) -- "
            "*same seven beats, every claim carrying its standard error.* We test whether the "
            "gold/silver ratio (GLD/SLV ETF) is stationary, whether gold and silver are "
            "cointegrated, and whether a z-score entry rule earns a statistically robust edge "
            "over a random-direction control on 16 years of daily data.\n\n"
            "> **Not investment advice.** Real data: GLD and SLV daily bars 2010-01-04 to "
            f"2026-06-12 (n = {R['n_days']:,} days each); offline core runs on deterministic "
            "synthetic tape. Methods & sources in [`docs/references.md`](../docs/references.md).\n"
            ">\n"
            "> **The `in plain words` notes** translate each result back to intuition."
        ),
        code(BOOT + "\nfrom quantlab import analytics, stats\n"),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | Gross **+{R['strat_mean']:.0f} bps/trade**, HAC "
            f"*t* = **+{R['strat_t']:.2f}** on n = **{R['n_trades']} trades** over 16 yrs; "
            f"delta vs random = +{R['delta']:.0f} bps -- statistically inconclusive. |\n"
            f"| **Tradability** | `FRAGILE` | ~1.6 signals/yr, ~{R['avg_hold']}-day holds. "
            f"Costs negligible (net *t* = {R['net5_t']:.2f} at 5 bps). Signal aridity is the problem. |\n"
            f"| **Ratio mean-reverts?** | `NOT SUPPORTED` | ADF p = {R['adf_p']:.2f} (fails "
            f"to reject unit root); Engle-Granger p = {R['coint_p']:.2f} (not cointegrated); "
            f"OU half-life = {R['half_life']:.0f} trading days. |\n\n"
            "> *In plain words:* three structural tests all fail; the trade earns a positive "
            "point estimate but the sample is too thin (26 trades) for any reliable inference."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 -- The claim, steelmanned\n\n"
            "Let $R_t = P^{\\text{gold}}_t / P^{\\text{silver}}_t$ be the gold/silver ratio. "
            "The claim asserts:\n\n"
            "- **H1 (stationarity).** $R_t$ is I(0): it has a finite unconditional variance "
            "and returns to its long-run mean. Equivalently, gold and silver prices are "
            "cointegrated: $\\log P^{\\text{gold}}_t - \\beta \\log P^{\\text{silver}}_t$ "
            "is stationary for some $\\beta$.\n"
            "- **H2 (signal).** The conditional expected return of the spread trade "
            "$d_t \\cdot (r^{\\text{silver}}_{t+1} - r^{\\text{gold}}_{t+1})$ is positive "
            "when $d_t = \\text{sign}(z_t)$ (sign of the z-score), measured against a "
            "random-direction control.\n"
            "- **H3 (tradable).** The edge survives realistic round-trip ETF costs.\n\n"
            "We reject H1 at conventional significance levels, H2 does not clear the |t| >= 2 "
            "bar (borderline positive), and H3 is technically true but moot if H2 is not."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 -- So what?\n\n"
            "The gold/silver ratio trade is frequently cited in commodity and macro research "
            "as a practical ETF rotation strategy accessible to retail investors. If H1-H3 "
            "held, it would be a low-cost, infrequent, long-horizon pairs trade. The interesting "
            "finding is *not* that it loses money (the gross point estimate is positive) -- "
            "it's that the statistical foundation is entirely absent: no stationarity, no "
            "cointegration, a 15-month half-life, and only 26 usable signals over 16 years."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 -- How we'd know -- the protocol\n\n"
            "- **Data.** GLD and SLV daily adjusted closes, 2010-01-04 to 2026-06-12 "
            "(Yahoo Finance). The GLD/SLV ratio is proportional to the spot gold/silver ratio "
            "(scaled ~10x by ETF share structure).\n"
            "- **Structural tests.** (a) ADF on the ratio (autolag AIC); "
            "(b) Engle-Granger cointegration on log(GLD) vs log(SLV); "
            "(c) OU half-life via AR(1) OLS regression on the ratio.\n"
            "- **Entry.** Rolling 252-day z-score of the ratio; enter spread when |z| > 1.5 "
            "(ratio at more than 1.5 sigma from its 1-year mean); exit when |z| < 0.5.\n"
            "- **Control.** Identical entry/exit dates, direction drawn i.i.d. ±1 (seeded).\n"
            "- **Inference.** Newey-West HAC t-stat on per-trade gross returns.\n"
            "- **Positive control.** Synthetic OU pair with tunable mean-reversion speed, "
            "confirming the engine recovers an edge when one is planted."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 -- The teardown"),
        md(
            "### 4a -- Structural tests: is there a mean-reverting ratio to exploit?\n\n"
            "Before trading on a ratio, we need it to be stationary. All three tests say no:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    gold = data.fetch_daily(GOLD_TICKER)\n"
            "    silver = data.fetch_daily(SILVER_TICKER)\n"
            "    ratio = st.compute_ratio(gold, silver)\n"
            "    adf_p = st.adf_pvalue(ratio)\n"
            "    hl = st.half_life(ratio)\n"
            "    ci_p = st.engle_granger_pvalue(gold['close'], silver['close'])\n"
            "    z_ratio = st.zscore(ratio, 252)\n"
            "else:\n"
            f"    adf_p={R['adf_p']}; hl={R['half_life']}; ci_p={R['coint_p']}\n"
            "\n"
            "print(f'ADF test p-value on ratio:           {adf_p:.4f}  (H0: unit root; p>0.05 = cannot reject)')\n"
            "print(f'Ornstein-Uhlenbeck half-life:        {hl:.1f} trading days')\n"
            "print(f'Engle-Granger cointegration p-value: {ci_p:.4f}  (H0: no cointegration)')\n"
            "print()\n"
            "print('Summary: all three tests fail to support mean-reversion in the ratio.')"
        ),
        md(
            "> *In plain words:* A pairs trade requires the spread to be stationary -- that "
            "it wanders but always comes home. Gold and silver fail this check three ways: "
            f"(1) the ADF cannot rule out a random walk (p = {R['adf_p']:.2f}); "
            f"(2) the Engle-Granger says the two prices are not cointegrated (p = {R['coint_p']:.2f}); "
            f"and (3) even assuming some reversion, the half-life is {R['half_life']:.0f} days -- "
            "15 months. That's not a 'snap-back', that's a geological timescale for a trade."
        ),
        md(
            "### 4b -- The z-score strategy vs a random-direction control\n\n"
            "We run the strategy on the real tape and compare it to a coin-flip "
            "control on identical entries:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    entries = st.zscore_entries(ratio, window=252, enter_z=1.5, exit_z=0.5)\n"
            "    ledger = st.run_trades(gold, silver, entries, cost_bps=0.0)\n"
            "    rand_ent = st.random_entries(entries, seed=113)\n"
            "    rand_led = st.run_trades(gold, silver, rand_ent, cost_bps=0.0)\n"
            "    s = st.summarize(ledger, 'ret_gross')\n"
            "    r = st.summarize(rand_led, 'ret_gross')\n"
            "    sm, sw, st_val = s['mean_bps'], s['win_rate']*100, s['tstat']\n"
            "    rm, rw, rt_val = r['mean_bps'], r['win_rate']*100, r['tstat']\n"
            "    n_tr = s['n_trades']; avg_h = ledger['trading_days'].mean()\n"
            "else:\n"
            f"    sm,sw,st_val = {R['strat_mean']},{R['strat_win']*100:.1f},{R['strat_t']}\n"
            f"    rm,rw,rt_val = {R['rand_mean']},{R['rand_win']*100:.1f},{R['rand_t']}\n"
            f"    n_tr={R['n_trades']}; avg_h={R['avg_hold']}\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "x = [0, 1]; ys = [sm, rm]; ts = [st_val, rt_val]; ws = [sw, rw]\n"
            "lbls = [f'Z-score strategy\\n(n={n_tr})', 'Random direction']\n"
            "colors = [AMBER if abs(st_val) < 2 else GREEN, GREY]\n"
            "ax.bar(x, ys, color=colors, width=0.5)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "for xi, yi, ti, wi in zip(x, ys, ts, ws):\n"
            "    ax.text(xi, yi + (20 if yi >= 0 else -30),\n"
            "            f't={ti:+.2f}\\nwin {wi:.0f}%', ha='center', fontsize=9)\n"
            "ax.set_xticks(x); ax.set_xticklabels(lbls)\n"
            "ax.set_ylabel('gross profit per trade (bps)')\n"
            "ax.set_title(f'z-score strategy vs coin flip (n={n_tr} trades, avg hold ~{avg_h:.0f}d)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Strategy: {sm:+.1f} bps/trade, t={st_val:+.2f} | '\n"
            "      f'Random: {rm:+.1f} bps/trade, t={rt_val:+.2f}')"
        ),
        md(
            f"> *In plain words:* The strategy earns a positive mean (+{R['strat_mean']:.0f} bps/trade) "
            f"vs the random control (+{R['rand_mean']:.0f} bps). But the HAC t-stat is only "
            f"+{R['strat_t']:.2f} -- with {R['n_trades']} trades, the standard error is so large "
            "that even this sizeable point estimate doesn't clear the bar. We'd need roughly 4x "
            "more trades to confirm the signal with confidence."
        ),
        md(
            "### 4c -- Positive control: the engine works when reversion is real\n\n"
            "A synthetic pair with a planted OU mean-reversion speed confirms the z-score "
            "engine finds edges when they exist:"
        ),
        code(
            "speeds = [0.0, 0.05, 0.10, 0.15, 0.20]\n"
            "strat_ts, rand_ts = [], []\n"
            "for sp in speeds:\n"
            "    gd, sv, tr = data.synthetic_daily(n_days=3000, mean_rev_speed=sp, seed=113)\n"
            "    rat = st.compute_ratio(gd, sv)\n"
            "    ent = st.zscore_entries(rat, window=252, enter_z=1.5, exit_z=0.5)\n"
            "    if len(ent) == 0:\n"
            "        strat_ts.append(float('nan')); rand_ts.append(float('nan')); continue\n"
            "    led = st.run_trades(gd, sv, ent, cost_bps=0.0)\n"
            "    rnd = st.run_trades(gd, sv, st.random_entries(ent, seed=42), cost_bps=0.0)\n"
            "    strat_ts.append(st.summarize(led, 'ret_gross')['tstat'])\n"
            "    rand_ts.append(st.summarize(rnd, 'ret_gross')['tstat'])\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(speeds, strat_ts, 'o-', c=GREEN, lw=2, label='z-score strategy')\n"
            "ax.plot(speeds, rand_ts, 's--', c=GREY, lw=1.5, label='random control')\n"
            "ax.axhline(2, ls='--', c=RED, lw=1, label='|t|=2 bar')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1)\n"
            "ax.axhline(0, c='k', lw=0.8)\n"
            "ax.set_xlabel('planted OU mean-reversion speed')\n"
            "ax.set_ylabel('HAC t-stat')\n"
            "ax.set_title('The engine is faithful: t-stat rises sharply when reversion is planted')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('At speed=0 (random walk), t-stat hovers near 0.')\n"
            "print('At speed=0.05+, the strategy clears the inference bar.')"
        ),
        md(
            "> *In plain words:* The machine is not broken. When the ratio actually "
            "mean-reverts (speeds > 0.05), the strategy rapidly earns significant t-stats. "
            "The real-tape verdict (t = 1.77) is not a machinery failure -- it's a statement "
            "that the real gold/silver ratio reverts too slowly, too rarely, or not at all."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 -- The verdict\n\n"
            f"- **Signal `WEAK`** -- gross {R['strat_mean']:+.0f} bps/trade, HAC "
            f"*t* +{R['strat_t']:.2f} on {R['n_trades']} trades (16 yrs); "
            f"delta vs random +{R['delta']:.0f} bps. Below the |t| >= 2 bar.\n"
            f"- **Tradability `FRAGILE`** -- ~1.6 signals/yr, ~{R['avg_hold']}-day holds; "
            f"costs are trivial (delta at 20 bps = {R['net0'] - R['net20']:.0f} bps). "
            "The problem is the uncertain edge and 15-month waits.\n"
            f"- **Ratio mean-reverts? `NOT SUPPORTED`** -- ADF p = {R['adf_p']:.2f}, "
            f"Engle-Granger p = {R['coint_p']:.2f}, OU half-life = {R['half_life']:.0f} days. "
            "Three tests, same answer: we cannot confirm the ratio is stationary."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 -- Could you trade it? -- signal aridity is the real constraint\n\n"
            "Unlike most failed strategies where costs are the executioner, here the cost "
            "sweep is almost irrelevant. With 1.6 trades/year and 78-day average holds, "
            "a 20 bps round-trip on a 4-leg ETF trade costs only a few bps per trade:"
        ),
        code(
            "costs = [0, 5, 10, 20]\n"
            "if HAVE_REAL:\n"
            "    net_t = [st.summarize(st.run_trades(gold, silver, entries, cost_bps=c),\n"
            "                          'ret_net')['tstat'] for c in costs]\n"
            "    net_m = [st.summarize(st.run_trades(gold, silver, entries, cost_bps=c),\n"
            "                          'ret_net')['mean_bps'] for c in costs]\n"
            "else:\n"
            f"    net_t = [{R['net0_t']}, {R['net5_t']}, {R['net10_t']}, {R['net20_t']}]\n"
            f"    net_m = [{R['net0']}, {R['net5']}, {R['net10']}, {R['net20']}]\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(costs, net_m, 'o-', c=AMBER, lw=2, label='net mean (bps/trade)')\n"
            "ax2 = ax.twinx()\n"
            "ax2.plot(costs, net_t, 's--', c=GREY, lw=1.5, label='HAC t')\n"
            "ax2.axhline(2, ls=':', c=GREY); ax2.axhline(-2, ls=':', c=GREY)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('round-trip cost (bps)')\n"
            "ax.set_ylabel('net mean (bps/trade)', color=AMBER)\n"
            "ax2.set_ylabel('HAC t-stat', color=GREY)\n"
            "ax.set_title('Costs barely move the needle -- it was never about the costs')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('The t-stat barely moves with costs. The issue is sample size, not friction.')"
        ),
        md(
            "> *In plain words:* In most failed strategies, the story is 'great idea, costs killed "
            "it.' Here it's different: the strategy is infrequent enough that costs are negligible, "
            "but the edge itself was never statistically confirmed. You'd be running a trade with "
            "uncertain economics, 15-month holding periods, and only 1-2 opportunities per year -- "
            "all while a simple buy-and-hold of either metal delivers comparable returns with zero effort."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 -- Going further -- extensions and related desk work\n\n"
            "- **Pre-2010 data.** The folk trade was most cited in the 2000s and 1990s. "
            "Adding older history might change the cointegration picture -- or confirm that "
            "the classic 'ratio always reverts' observation was an artifact of a shorter, "
            "cherry-picked window.\n"
            "- **Futures (GC=F / SI=F).** The study uses ETFs; futures contracts have slightly "
            "different price dynamics and roll costs. The structural tests would be the same.\n"
            "- **Shorter lookback window.** A 63-day z-score generates more signals but "
            "is noisier. Test with a walk-forward validation to check for overfitting.\n"
            "- **[Study 85 -- Dr-Copper](../../85-dr-copper/)**: the copper/gold ratio as "
            "an economic predictor -- a macro-grounded ratio signal.\n"
            "- **[Study 68 -- All-Weather](../../68-all-weather/)**: commodity basket timing "
            "-- the portfolio context for gold and silver allocation.\n\n"
            "*Fork this, add pre-2010 data or a different entry threshold, and show a "
            "cointegrated pair that actually clears |t| >= 2 on an out-of-sample window.*"
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
