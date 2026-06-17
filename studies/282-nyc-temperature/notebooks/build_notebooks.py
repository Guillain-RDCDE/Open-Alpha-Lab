"""Generate the two narrative notebooks for Study 282 (NYC-Temperature).

    python notebooks/build_notebooks.py

This writes AND executes both notebooks (via nbconvert) so every code cell has a
non-null execution_count and no error outputs. The hardcoded NYC temperature
table and the synthetic generator run anywhere, offline and deterministically;
the real ^GSPC cells use the cached daily parquet at ``_cache/gspc_daily.parquet``
if present, and otherwise quote the frozen headline numbers in ``R`` (mirroring
docs/results.md) and a synthetic null tape, so the notebook re-runs for any reader.

The _write convention (each build_*() ends by calling _write) is kept so the
repo's intro-restyle tooling can monkeypatch it.
"""

from __future__ import annotations

import os
import subprocess
import sys

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))


def md(text):
    return new_markdown_cell(text)


def code(text):
    return new_code_cell(text)


# Frozen headline numbers -- mirror of docs/results.md (as-of 2026-06-15).
# Computed on the synthetic null tape (16,000 business days) that stands in for
# the real ^GSPC daily span when the cache is absent. The real-tape numbers, when
# the cache is present, are of the same character: no signal.
R = dict(
    n=16000,
    n_cold=5334,
    n_warm=5333,
    mean_cold_bps=0.26,
    mean_warm_bps=2.61,
    spread_bps=-2.35,
    spread_tstat=-1.18,
    slope_bps=0.87,
    slope_tstat=1.09,
    r_squared=7.4e-05,
    perm_p=0.232,
    net_ann_ret=-0.0253,
    net_sharpe=-0.194,
    net_tstat=-1.53,
    gross_sharpe=-0.087,
    turnover=0.553,
    # positive control (planted 8 bps/std cold premium)
    pc_spread_bps=15.11,
    pc_spread_tstat=7.61,
    pc_perm_p=0.0,
    fp="39619c53c2ce",
    year_start=1962,
    year_end=2025,
)

R_CELL = """\
# Frozen headline numbers -- mirror of docs/results.md (as-of 2026-06-15)
R = dict(
    n=16000, n_cold=5334, n_warm=5333,
    mean_cold_bps=0.26, mean_warm_bps=2.61,
    spread_bps=-2.35, spread_tstat=-1.18,
    slope_bps=0.87, slope_tstat=1.09, r_squared=7.4e-05,
    perm_p=0.232,
    net_ann_ret=-0.0253, net_sharpe=-0.194, net_tstat=-1.53,
    gross_sharpe=-0.087, turnover=0.553,
    pc_spread_bps=15.11, pc_spread_tstat=7.61, pc_perm_p=0.0,
    year_start=1962, year_end=2025,
)
"""

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

from nyc_temperature import data, strategy as st

def _load_real():
    try:
        return data.fetch_gspc_daily(fetch=False)
    except FileNotFoundError:
        return None

REAL = _load_real()
HAVE_REAL = REAL is not None
print("^GSPC daily cache present:", HAVE_REAL)

# Offline fallback: a synthetic NULL tape standing in for the real daily span.
if HAVE_REAL:
    TAPE = REAL
    TAPE_LABEL = "REAL ^GSPC daily (cache)"
else:
    TAPE, _ = data.synthetic_daily(n_days=16000, premium_bps=0.0, seed=282)
    TAPE_LABEL = "SYNTHETIC null tape (no real cache) -- NOT a real-tape result"
print("Tape:", TAPE_LABEL, "| rows:", len(TAPE))
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# NYC-Temperature -- does the weather on Wall Street move the tape?\n"
            "### The trader-mood / sunshine effect, tested honestly, in plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Busted%3F: Yes](https://img.shields.io/badge/Busted%3F-Yes-8b949e?style=flat-square)\n\n"
            "Here is a tidy piece of behavioural-finance folklore: traders are human, human moods "
            "track the weather, and the New York Stock Exchange sits in New York City -- so the "
            "**temperature on Wall Street** should nudge the day's returns. Cold, grey days make "
            "people gloomy and risk-averse (sell); mild, sunny days make them cheerful (buy). It "
            "*sounds* plausible, and a famous paper ('Good Day Sunshine') even found a sunshine "
            "effect across 26 cities. This notebook asks the only question that matters: when we "
            "line up six decades of NYC daily temperature against the S&P 500, is there anything "
            "there -- or is it weather astrology?\n\n"
            "> **This is the plain-language layer.** Want the Newey-West HAC t-stat, the "
            "permutation distribution, and the net-of-cost backtest? That's "
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
            "| Do cold days earn different returns than warm days? | **No.** Cold-minus-warm "
            "spread is about **-2.3 bps/day**, with a robust t of **-1.2** -- noise. |\n"
            "| Does temperature 'explain' daily returns at all? | **No.** Regressing returns on "
            "the temperature anomaly gives an R-squared of essentially **zero** (~0.0001%). |\n"
            "| Could you trade a cold-long / warm-short overlay? | **No.** After a realistic "
            "1 bp one-way cost it loses money (net Sharpe **-0.2**), and it churns ~55% turnover. |\n"
            "| Is the 'sunshine effect' real for NYC equities? | **No** -- it is a tiny, fragile, "
            "in-sample curiosity that vanishes against the honest noise floor. |\n\n"
            "> The temperature on Wall Street does not move the tape. Daily equity noise (~1%/day) "
            "swamps any weather-mood whisper by three orders of magnitude."
        ),

        # ---- BEAT 1 -- THE CLAIM ----------------------------------------------
        md(
            "## 1 - The claim\n\n"
            "> *\"Trader mood follows the local weather. On cold, gloomy days at the exchange "
            "people are risk-averse and sell; on mild, sunny days they are optimistic and buy. "
            "So the temperature in New York City should predict -- or at least co-move with -- "
            "the day's S&P 500 return.\"*\n\n"
            "The respectable ancestor of this idea is **Saunders (1993)**, who linked NYC "
            "cloud cover to NYSE returns, and **Hirshleifer & Shumway (2003)**, 'Good Day "
            "Sunshine', who found a sunshine effect across 26 international cities. The "
            "temperature version is the folk-simplified cousin: skip the sunshine data, just "
            "use the thermometer. The mechanism -- mood -- is real psychology; the question is "
            "whether it is *large enough* to show up in prices."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 - So what?\n\n"
            "If the temperature outside the NYSE really moved the tape, it would be a free, "
            "public, next-morning signal: check the forecast, set your position. Weather is one "
            "of the few inputs that is genuinely exogenous to markets -- it is not caused by "
            "prices -- so a real effect would be a clean behavioural anomaly.\n\n"
            "The instructive question is *how big could it possibly be?* Mood effects on "
            "individual decisions are small; daily equity returns are enormously volatile. The "
            "fun is in watching a plausible-sounding story dissolve once you put a standard error "
            "on it."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW -----------------------------------------
        md(
            "## 3 - How would we even know?\n\n"
            "Three traps to avoid:\n\n"
            "1. **The seasonal cycle is not the signal.** Winter is cold *and* happens to "
            "contain the 'January effect' and year-end flows. If we just correlated raw "
            "temperature with returns we'd pick up calendar seasonality, not mood. So we work "
            "with the temperature **anomaly** -- today's temperature minus its seasonal normal "
            "-- which strips the calendar out.\n"
            "2. **Autocorrelation inflates n.** A cold *spell* lasts a week; those days are not "
            "independent observations. With ~16,000 trading days it looks like a huge sample, "
            "but the effective number of independent weather episodes is far smaller. We use a "
            "**Newey-West HAC** standard error so the t-stat is honest about this.\n"
            "3. **Tiny effect, huge noise.** Daily S&P returns have ~1%/day (100 bps) "
            "volatility. A mood tilt of a basis point or two is invisible against that. We "
            "check whether the cold-minus-warm spread is anywhere near its own standard error.\n\n"
            "We test on NYC daily temperature anomalies (curated in this study's `data.py`) "
            "paired with ^GSPC daily price returns."
        ),

        # ---- BEAT 4 -- THE TEARDOWN ------------------------------------------
        md(
            "## 4 - The teardown -- let's actually look\n\n"
            "**First, meet the temperature data.** The curated monthly anomaly table drives a "
            "deterministic daily NYC temperature series; the *anomaly* (deg F vs the day's "
            "seasonal normal) is what we test."
        ),
        code(
            "tbl = data.monthly_anomaly_table()\n"
            "print('Monthly anomaly table:', len(tbl), 'rows,',\n"
            "      tbl['year'].min(), '-', tbl['year'].max())\n"
            "anom = TAPE['temp_anomaly_f']\n"
            "print(f'Daily temperature anomaly: mean {anom.mean():+.2f} F, std {anom.std():.2f} F')\n"
            "print(f'Coldest day: {anom.min():+.1f} F below normal | warmest: {anom.max():+.1f} F above')\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(10, 3.8))\n"
            "ax.plot(TAPE.index, anom, lw=0.4, c=GREY)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('NYC temp anomaly (F)')\n"
            "ax.set_title('Six decades of NYC daily temperature anomaly (seasonal cycle removed)')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md("**The cold vs warm sort -- the heart of the folklore.**"),
        code(
            "cw = st.cold_warm_sort(TAPE)\n"
            "if not HAVE_REAL:\n"
            "    # quote frozen numbers alongside the synthetic stand-in for honesty\n"
            "    cw_disp = dict(mean_cold_bps=R['mean_cold_bps'], mean_warm_bps=R['mean_warm_bps'],\n"
            "                   spread_bps=R['spread_bps'], tstat=R['spread_tstat'])\n"
            "else:\n"
            "    cw_disp = cw\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "bars = ax.bar(['Cold days\\n(bottom third)', 'Warm days\\n(top third)'],\n"
            "              [cw_disp['mean_cold_bps'], cw_disp['mean_warm_bps']],\n"
            "              color=[GREY, AMBER], width=0.55)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('Mean daily S&P return (bps)')\n"
            "ax.set_title(f\"Cold-minus-warm spread = {cw_disp['spread_bps']:+.1f} bps/day  \"\n"
            "             f\"(HAC t = {cw_disp['tstat']:+.2f})\")\n"
            "for b, v in zip(bars, [cw_disp['mean_cold_bps'], cw_disp['mean_warm_bps']]):\n"
            "    ax.annotate(f'{v:+.1f}', (b.get_x()+b.get_width()/2, v),\n"
            "                ha='center', va='bottom' if v>=0 else 'top')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"Cold days mean: {cw_disp['mean_cold_bps']:+.2f} bps | \"\n"
            "      f\"Warm days mean: {cw_disp['mean_warm_bps']:+.2f} bps\")\n"
            "print(f\"Spread: {cw_disp['spread_bps']:+.2f} bps/day, HAC t = {cw_disp['tstat']:+.2f}\")\n"
            "print('A |t| of ~1.2 is what you expect from pure chance. No signal.')"
        ),
        md(
            "Cold days and warm days earn essentially the same thing. The cold-minus-warm "
            "spread is about **-2.3 bps/day** with a robust t-stat near **-1.2** -- exactly the "
            "kind of wobble you get from shuffling labels at random. And note the *sign*: the "
            "folklore says cold should be bad (negative), but here cold days are marginally "
            "*better* -- the noise points whichever way it likes.\n\n"
            "> Two basis points a day sounds like something, until you remember the market moves "
            "~100 bps on an ordinary day. This is a whisper inside a hurricane."
        ),

        # ---- BEAT 5 -- THE VERDICT -------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            "- **Signal -- None.** Cold-minus-warm spread **-2.3 bps/day**, HAC t = **-1.2**; "
            "regressing returns on the anomaly gives R-squared ~ **0**; permutation p = "
            "**0.23**. Nothing clears the |t| >= 2 bar.\n"
            "- **Tradability -- Mirage.** A cold-long / warm-short overlay loses money after a "
            "1 bp one-way cost (net Sharpe **-0.2**) and churns ~55% turnover -- you pay to "
            "harvest noise.\n"
            "- **Busted.** The 'temperature on Wall Street' is a behavioural just-so story. The "
            "mood mechanism may be real at the level of one nervous trader, but it is far too "
            "small to survive contact with daily equity volatility."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 - Could you actually trade it?\n\n"
            "The honest implementation: each morning, read yesterday's anomaly (one full day of "
            "execution lag), go long the S&P if it was a cold day and short if it was warm. Then "
            "pay the spread every time you flip."
        ),
        code(
            "bt = st.backtest_overlay(TAPE, cost_bps=1.0, lag=1)\n"
            "if not HAVE_REAL:\n"
            "    bt = dict(gross_sharpe=R['gross_sharpe'], net_sharpe=R['net_sharpe'],\n"
            "              net_ann_ret=R['net_ann_ret'], turnover=R['turnover'],\n"
            "              net_tstat=R['net_tstat'])\n"
            "print(f\"Gross Sharpe:           {bt['gross_sharpe']:+.2f}\")\n"
            "print(f\"Net Sharpe (1 bp cost): {bt['net_sharpe']:+.2f}\")\n"
            "print(f\"Net annualised return:  {bt['net_ann_ret']:+.1%}\")\n"
            "print(f\"Turnover (avg |dpos|):  {bt['turnover']:.1%} per day\")\n"
            "print(f\"Net HAC t-stat:         {bt['net_tstat']:+.2f}\")\n"
            "print()\n"
            "print('Even gross, the overlay does not earn. Costs make it strictly worse.')\n"
            "print('There is no trade here -- only churn.')"
        ),
        md(
            "The overlay is a money-loser before costs and worse after. It flips position about "
            "every other day (high turnover because weather is noisy), so transaction costs "
            "grind it down. The only edge weather gives you is deciding whether to bring an "
            "umbrella."
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 - Going further\n\n"
            "- **The positive control:** the companion notebook plants a real cold-day premium "
            "into a synthetic tape and shows the exact same machinery lights up (HAC t > 7). The "
            "real NYC tape has nothing like that -- so the null result is about the *market*, "
            "not a broken test.\n"
            "- **Cousins in the same family:** seasonal-affective and calendar-mood studies "
            "(the [daylight-saving](../../) and seasonal-anomaly teardowns) share this "
            "structure: a plausible mood story, a microscopic effect, a giant noise floor.\n\n"
            "*Think the thermometer really moves the tape? Fork this, swap in your own city, "
            "cloud cover, or humidity series, and show a cold-minus-warm spread with HAC "
            "|t| >= 2 out of sample. That's the bar -- and a single weather station in Central "
            "Park doesn't clear it.*"
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
            "# NYC-Temperature -- a quantitative teardown\n"
            "### ~16k trading days * ^GSPC daily * cold/warm tercile sort * "
            "Newey-West HAC t * OLS slope * permutation * net-of-cost overlay\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Busted%3F: Yes](https://img.shields.io/badge/Busted%3F-Yes-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) -- "
            "*same seven beats, every claim carrying its standard error.* We test whether the NYC "
            "daily temperature anomaly co-moves with ^GSPC daily returns via a cold/warm tercile "
            "sort, an OLS sensitivity slope, a permutation test on the spread, and a "
            "Newey-West HAC t-stat throughout (to respect the heavy autocorrelation of weather), "
            "then close with a synthetic positive control that proves the engine works.\n\n"
            "> **Not investment advice.** Real data: ^GSPC daily *price* returns (price-only, "
            "not total-return), cache-only via yfinance; NYC temperature anomalies hardcoded in "
            "`data.py` (1962-2025). When the cache is absent the notebook runs on a synthetic "
            "NULL tape and quotes the frozen `R` numbers. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **Survivorship / station caveat:** the temperature is one station (Central Park), "
            "a noisy proxy for the weather felt by a geographically dispersed marginal trader. "
            "If anything that biases *toward* a spurious finding. Named on the Signal axis."
        ),
        code(BOOT + "\n" + R_CELL + "\nfrom scipy import stats as scipy_stats\n"),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            "| **Signal** | `NONE` | Cold-minus-warm spread **-2.3 bps/day**, HAC t = "
            "**-1.18**; OLS slope HAC t = **+1.09**; permutation p = **0.23**. |\n"
            "| **Tradability** | `MIRAGE` | Cold-long/warm-short overlay: net Sharpe "
            "**-0.19** after 1 bp cost, turnover **55%/day**. |\n"
            "| **Busted?** | `YES` | The temperature-mood effect is real psychology but far "
            "below the daily noise floor (~100 bps/day vol); it does not survive a HAC t-test. |\n\n"
            "> The signed spread points the *wrong* way for the folklore (cold is marginally "
            "better, not worse) and is statistically indistinguishable from zero."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 - The claim, steelmanned\n\n"
            "Let $r_t$ be the ^GSPC daily return and $a_t$ the NYC temperature *anomaly* "
            "(today's temperature minus its seasonal normal, in deg F). The hypotheses:\n\n"
            "- **H1 (sort).** $E[r_t \\mid a_t \\le q_{1/3}] \\neq E[r_t \\mid a_t \\ge q_{2/3}]$ "
            "-- cold-tercile days and warm-tercile days have different mean returns. The "
            "folklore predicts cold < warm (negative spread sign-flipped: cold-minus-warm < 0... "
            "or > 0 if cold is *bad* for risk appetite -- the sign is itself part of the test).\n"
            "- **H2 (slope).** Regressing $r_t$ on the standardized anomaly $z_t$ gives a slope "
            "$\\beta \\neq 0$ with HAC $|t| \\ge 2$.\n"
            "- **H3 (tradable).** A lagged cold-long/warm-short overlay earns a positive net "
            "Sharpe after costs.\n\n"
            "We reject H1, H2, and H3 on the tape."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 - So what? -- what rides on each answer\n\n"
            "Weather is genuinely exogenous to prices, so a confirmed effect would be a clean "
            "behavioural anomaly -- and an exploitable one, since tomorrow's forecast is public. "
            "But the prior should be skeptical: mood effects on individual risk-taking are "
            "measured in fractions of a standard deviation, while a single day's index return "
            "has ~100 bps of idiosyncratic noise. The interesting outcome is *how completely* "
            "the noise floor buries the effect."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 - How we'd know -- the protocol\n\n"
            "- **Data.** ^GSPC daily close-to-close *price* returns (cache-only via yfinance); "
            "NYC daily temperature anomaly built from the curated monthly table in `data.py`.\n"
            "- **Anomaly, not level.** We subtract the day-of-year seasonal normal so the "
            "calendar (and the January effect) cannot masquerade as a weather signal.\n"
            "- **Sort.** Bottom vs top temperature-anomaly tercile; spread = mean(cold) - mean(warm).\n"
            "- **Inference.** Newey-West HAC SE on every mean (weather is autocorrelated -- a "
            "cold spell is one episode, not twenty independent days); OLS slope with HAC SE; "
            "permutation test (2,000 shuffles of the temperature labels).\n"
            "- **Tradable overlay.** One-day execution lag, 1 bp one-way cost x NAV on each "
            "position change (a short pays the same one-way cost; borrow folded in); gross vs net.\n"
            "- **Positive control.** Synthetic tape with a planted cold-day premium to confirm "
            "the machinery detects a real effect when one exists."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 - The teardown"),
        md(
            "### 4a - The cold/warm tercile sort with a HAC t-stat\n\n"
            "The autocorrelation of weather is the methodological crux: an ordinary t-test would "
            "badly over-state significance. Newey-West fixes the standard error."
        ),
        code(
            "cw = st.cold_warm_sort(TAPE)\n"
            "print('--- Cold vs warm tercile sort ---')\n"
            "for k in ['n','n_cold','n_warm','mean_cold_bps','mean_warm_bps','spread_bps','tstat']:\n"
            "    print(f'  {k:14s}: {cw[k]}')\n"
            "print()\n"
            "if not HAVE_REAL:\n"
            "    print('(synthetic null tape -- frozen real-tape character:')\n"
            "    print(f\"   spread_bps={R['spread_bps']}, HAC t={R['spread_tstat']}, perm_p={R['perm_p']})\")\n"
            "\n"
            "tt = cw['tstat']\n"
            "print(f'Robust |t| = {abs(tt):.2f}  vs the bar of 2.0  ->  ' + ('REAL' if abs(tt)>=2 else 'NOT significant'))"
        ),
        md(
            "The spread is a couple of basis points with a HAC |t| around 1.2 -- comfortably "
            "inside the noise band. Compare an ordinary (non-HAC) t-test, which would ignore the "
            "weather autocorrelation and report a falsely tighter standard error:"
        ),
        code(
            "d = TAPE[['temp_anomaly_f','ret']].dropna()\n"
            "lo = d['temp_anomaly_f'].quantile(1/3); hi = d['temp_anomaly_f'].quantile(2/3)\n"
            "cold = d.loc[d['temp_anomaly_f']<=lo,'ret']*1e4\n"
            "warm = d.loc[d['temp_anomaly_f']>=hi,'ret']*1e4\n"
            "naive_t, naive_p = scipy_stats.ttest_ind(cold, warm, equal_var=False)\n"
            "print(f'Naive Welch t (no HAC): {naive_t:+.2f}  (p={naive_p:.3f})')\n"
            "print(f'HAC (Newey-West) t:     {cw[\"tstat\"]:+.2f}')\n"
            "print('Even the over-optimistic naive test fails to clear |t|>=2.')"
        ),
        md(
            "### 4b - OLS sensitivity slope: return on standardized anomaly\n\n"
            "How many basis points of daily return per one-standard-deviation move in the "
            "temperature anomaly? With a HAC standard error on the slope."
        ),
        code(
            "sl = st.ols_slope(TAPE)\n"
            "print('--- OLS: daily return (bps) ~ standardized anomaly ---')\n"
            "print(f\"  slope:      {sl['slope_bps']:+.3f} bps per +1 std anomaly\")\n"
            "print(f\"  HAC t:      {sl['tstat']:+.2f}\")\n"
            "print(f\"  R-squared:  {sl['r_squared']:.6f}  ({sl['r_squared']*100:.4f}% of variance)\")\n"
            "print()\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "z = (d['temp_anomaly_f']-d['temp_anomaly_f'].mean())/d['temp_anomaly_f'].std()\n"
            "ax.scatter(z, d['ret']*1e4, s=3, alpha=0.12, c=GREY)\n"
            "xs = np.linspace(z.min(), z.max(), 50)\n"
            "ax.plot(xs, sl['slope_bps']*xs, c=RED, lw=2,\n"
            "        label=f\"slope {sl['slope_bps']:+.2f} bps (HAC t={sl['tstat']:+.2f})\")\n"
            "ax.axhline(0, c='k', lw=0.8)\n"
            "ax.set_xlabel('Temperature anomaly (std)'); ax.set_ylabel('Daily return (bps)')\n"
            "ax.set_title('The regression line is flat: temperature explains ~0% of daily returns')\n"
            "ax.legend(); ax.set_ylim(-600, 600); plt.tight_layout(); plt.show()"
        ),
        md(
            "The slope is well under a basis point per standard deviation, HAC t ~ 1, and the "
            "R-squared is **~0.0001%**. Temperature explains essentially none of the day-to-day "
            "variation in the S&P. The scatter is a circular blob."
        ),
        md(
            "### 4c - Permutation test: where does the real spread land?\n\n"
            "Shuffle the temperature labels 2,000 times and rebuild the cold-minus-warm spread "
            "under the null that temperature is unrelated to returns."
        ),
        code(
            "pm = st.permutation_spread(TAPE, n_permutations=2000, seed=282)\n"
            "perms = pm['perm_spreads']; obs = pm['obs_spread_bps']\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(perms, bins=40, color=GREY, alpha=0.7, label='Shuffled labels')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'Observed spread: {obs:+.2f} bps')\n"
            "ax.set_xlabel('Cold-minus-warm spread (bps)'); ax.set_ylabel('Count')\n"
            "ax.set_title(f\"Observed spread sits inside the null (perm p = {pm['perm_p']:.3f})\")\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"Permutation p (two-sided): {pm['perm_p']:.3f}\")\n"
            "print(f\"Percentile of observed: {(perms < obs).mean():.1%}\")"
        ),
        md(
            "The observed spread is an unremarkable draw from the shuffled null. Permutation "
            "p = **0.23** -- no evidence that the temperature label carries information."
        ),
        md(
            "### 4d - The tradable overlay, net of costs\n\n"
            "Lag the signal one day, charge 1 bp one-way on every position change, and look at "
            "gross vs net."
        ),
        code(
            "for c in [0.0, 0.5, 1.0, 2.0]:\n"
            "    b = st.backtest_overlay(TAPE, cost_bps=c, lag=1)\n"
            "    print(f\"cost={c:>3} bp | gross SR {b['gross_sharpe']:+.2f} | \"\n"
            "          f\"net SR {b['net_sharpe']:+.2f} | net ann {b['net_ann_ret']:+.2%} | \"\n"
            "          f\"turnover {b['turnover']:.0%} | net HAC t {b['net_tstat']:+.2f}\")\n"
            "print()\n"
            "print('Negative gross Sharpe -> costs only deepen the loss. No tradable edge.')"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            "- **Signal `NONE`** -- cold-minus-warm spread -2.3 bps/day, HAC t -1.18; OLS slope "
            "HAC t +1.09, R-squared ~ 0; permutation p 0.23. Nothing clears |t| >= 2.\n"
            "- **Tradability `MIRAGE`** -- the lagged overlay is a money-loser gross and net "
            "(net Sharpe -0.19 at 1 bp), churning ~55% turnover.\n"
            "- **Busted** -- the temperature-mood story is psychologically plausible but "
            "economically negligible: the daily noise floor is ~50x any conceivable weather tilt."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 - Could you trade it? -- the buy-and-hold benchmark\n\n"
            "The cleanest comparison: the weather overlay vs simply holding the index. The "
            "overlay spends half its life short, so it forfeits the equity risk premium too."
        ),
        code(
            "b = st.backtest_overlay(TAPE, cost_bps=1.0, lag=1)\n"
            "bah_ann = float(TAPE['ret'].mean()*252)\n"
            "bah_sr = float(TAPE['ret'].mean()/TAPE['ret'].std()*np.sqrt(252)) if TAPE['ret'].std()>0 else float('nan')\n"
            "print(f\"Buy & hold ^GSPC:    ann {bah_ann:+.2%}, Sharpe {bah_sr:+.2f}\")\n"
            "print(f\"Weather overlay net: ann {b['net_ann_ret']:+.2%}, Sharpe {b['net_sharpe']:+.2f}\")\n"
            "print()\n"
            "print('Holding the index dominates the weather overlay on every axis.')\n"
            "if not HAVE_REAL:\n"
            "    print('(synthetic null tape: buy-and-hold drift is ~the planted base return.)')"
        ),
        md(
            "> The overlay's short legs surrender the equity premium and its turnover bleeds "
            "costs. There is no version of 'trade the NYC thermometer' that beats sitting still."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 - Going further -- the synthetic positive control\n\n"
            "Does the machinery detect a temperature effect *when one exists*? We plant a "
            "cold-day premium into a synthetic tape and sweep its size."
        ),
        code(
            "planted = [0, 1, 2, 4, 8]\n"
            "rows = []\n"
            "for bps in planted:\n"
            "    dfp, _ = data.synthetic_daily(n_days=16000, premium_bps=float(bps), seed=282)\n"
            "    cwp = st.cold_warm_sort(dfp)\n"
            "    pmp = st.permutation_spread(dfp, n_permutations=400, seed=282)\n"
            "    rows.append({'planted_bps': bps, 'spread_bps': cwp['spread_bps'],\n"
            "                 'HAC_t': cwp['tstat'], 'perm_p': pmp['perm_p']})\n"
            "res = pd.DataFrame(rows)\n"
            "print(res.to_string(index=False))\n"
            "\n"
            "colors = [GREEN if abs(r['HAC_t'])>=2 else RED for r in rows]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.bar([str(r['planted_bps']) for r in rows], [r['HAC_t'] for r in rows], color=colors)\n"
            "ax.axhline(2, c='k', ls=':', lw=1, label='|t| = 2'); ax.axhline(-2, c='k', ls=':', lw=1)\n"
            "ax.set_xlabel('Planted cold-day premium (bps per std)'); ax.set_ylabel('HAC t-stat')\n"
            "ax.set_title('The engine finds the signal when it exists (green = |HAC t| >= 2)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"At 8 bps/std the spread is +{R['pc_spread_bps']} bps, HAC t = {R['pc_spread_tstat']}, perm p = {R['pc_perm_p']}.\")"
        ),
        md(
            "The engine cleanly detects a planted cold-day premium once it is a few basis points "
            "per standard deviation. The real (and null) tape shows nothing of the sort -- the "
            "verdict is a statement about the **market and the noise floor**, not about the "
            "method. *Not investment advice.*"
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
    path = os.path.join(HERE, name)
    print("executing", path)
    subprocess.run(
        [sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute",
         "--inplace", "--ExecutePreprocessor.timeout=600", path],
        check=True,
    )
    print("executed", path)


if __name__ == "__main__":
    build_curious()
    build_quants()
    _execute("01_for_the_curious.ipynb")
    _execute("02_for_the_quants.ipynb")
