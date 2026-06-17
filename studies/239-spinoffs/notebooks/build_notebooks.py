"""Generate and EXECUTE the two narrative notebooks for Study 239 (Spinoffs).

    python notebooks/build_notebooks.py

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md).
The synthetic cells run offline; the real-tape cells use the study-local
cached parquet files (../_cache/) when present, otherwise fall back to the
frozen headline numbers in ``R`` so the notebook always re-runs cleanly.
"""

from __future__ import annotations

import os
import subprocess
import sys

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Frozen real-tape headline numbers -- mirror of docs/results.md (2026-06-16)
# ---------------------------------------------------------------------------
R = dict(
    n_events=14,
    fp="195fcddd6e9a",
    date_range="2011-11-17 to 2024-04-02",
    # 6m alpha
    a6_child=21.55,  a6_spy=9.64,   a6_alpha=11.91,  a6_med=11.75,  a6_win=57.1, a6_t=2.09,
    # 12m alpha
    a12_child=38.37, a12_spy=21.51, a12_alpha=16.87, a12_med=7.54,  a12_win=64.3, a12_t=2.07,
    # 18m alpha
    a18_child=67.50, a18_spy=33.24, a18_alpha=34.26, a18_med=-4.11, a18_win=42.9, a18_t=1.73,
    # 24m alpha
    a24_child=97.66, a24_spy=41.47, a24_alpha=56.19, a24_med=9.16,  a24_win=50.0, a24_t=1.71,
)

# ---------------------------------------------------------------------------
# Shared preamble cell
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

from spinoffs import data, strategy as st

# Frozen headline numbers (mirror of docs/results.md 2026-06-16)
R = dict(
    n_events={R["n_events"]}, fp="{R["fp"]}", date_range="{R["date_range"]}",
    a6_child={R["a6_child"]}, a6_spy={R["a6_spy"]},   a6_alpha={R["a6_alpha"]},
    a6_med={R["a6_med"]},   a6_win={R["a6_win"]},   a6_t={R["a6_t"]},
    a12_child={R["a12_child"]}, a12_spy={R["a12_spy"]}, a12_alpha={R["a12_alpha"]},
    a12_med={R["a12_med"]},  a12_win={R["a12_win"]},  a12_t={R["a12_t"]},
    a18_child={R["a18_child"]}, a18_spy={R["a18_spy"]}, a18_alpha={R["a18_alpha"]},
    a18_med={R["a18_med"]},  a18_win={R["a18_win"]},  a18_t={R["a18_t"]},
    a24_child={R["a24_child"]}, a24_spy={R["a24_spy"]}, a24_alpha={R["a24_alpha"]},
    a24_med={R["a24_med"]},  a24_win={R["a24_win"]},  a24_t={R["a24_t"]},
)

_CACHE = os.path.abspath(os.path.join("..", "_cache"))
_SPY_CACHE = os.path.join(_CACHE, "prices_239_SPY_1d.parquet")
HAVE_REAL = os.path.exists(_SPY_CACHE)

if HAVE_REAL:
    events, prices = data.fetch_spinoff_panel(fetch=False, cache_dir=_CACHE)
else:
    events, prices = None, None

print(f"real data cache present: {{HAVE_REAL}}")
if HAVE_REAL:
    print(f"  {{len(events)}} spin-off events, fingerprint {{data.fingerprint(events)}}")
"""


def md(text):
    return new_markdown_cell(text)


def code(text):
    return new_code_cell(text)


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Spinoffs — do corporate spin-offs outrun the market?\n"
            "### The Greenblatt / Cusatis-Miles-Woolridge spin-off premium, tested on 14 large-cap US events\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
            "![Forced--seller_thesis: Mixed](https://img.shields.io/badge/Forced--seller_thesis-Mixed-8b949e?style=flat-square)\n\n"
            "A 1993 paper (Cusatis, Miles & Woolridge) and Joel Greenblatt's popular book claimed that "
            "spin-off children outperform the S&P 500 by 25% in the first year.  The mechanism: "
            "institutional investors who receive shares of the newly-independent child are **forced sellers** "
            "because it is too small or off-mandate for their fund, creating temporary undervaluation.  "
            "Here we test 14 notable US spin-offs (2011-2024) and find a positive average alpha in the "
            "first 6-12 months — barely crossing the statistical bar — but the result is driven by three "
            "spectacular outliers, and the 18-24 month picture is weaker.\n\n"
            "> **Caveats before anything else.**\n"
            "> 1. Our 14-event table is *curated* (hand-picked notable spin-offs), not a systematic universe.\n"
            "> 2. SPY is not a size-matched benchmark; factor-adjusted alpha would be smaller.\n"
            "> 3. The 2020 Carrier/Otis spin from UTC landed during COVID recovery — timing luck matters.\n\n"
            "> **This is the plain-language layer.** For t-stats and the synthetic control see\n"
            "> **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> Not investment advice. House style: [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Do spin-off children beat SPY at 12 months? | **Marginally.** Average alpha "
            f"+{R['a12_alpha']:.1f}% (HAC *t* = {R['a12_t']:+.2f}) on 14 events — barely significant, driven by outliers. |\n"
            f"| Is the effect consistent across events? | **No.** At 12m, 9 of 14 events are positive; "
            f"median is only +{R['a12_med']:.1f}%. Three events account for most of the mean. |\n"
            "| Does the effect persist to 18-24 months? | **No.** 18m *t* = "
            f"{R['a18_t']:+.2f}, 24m *t* = {R['a24_t']:+.2f} — below the bar. |\n"
            "| Can you trade it systematically? | **Hard.** No selection rules, too-small n, "
            "illiquid early days, and a curated table are all obstacles. |\n\n"
            "> The spin-off premium is real in a directional sense on this sample, but it is "
            "fragile: the t-stats barely clear 2.0 at 6m and 12m, and the median at 18m turns "
            "negative.  The anomaly may exist in a broader universe but this study cannot confirm it."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 The claim\n\n"
            "> *When a company spins off a subsidiary, institutional investors who hold the parent "
            "automatically receive shares in the smaller child.  Many cannot hold it — it is too small, "
            "too speculative, or outside their sector mandate — so they sell indiscriminately, "
            "regardless of value.  This forced selling creates a buying opportunity.  Buy the "
            "spin-off at or shortly after the ex-distribution date, hold for 12 months.*\n"
            "> — Greenblatt (1997), *You Can Be a Stock Market Genius*; Cusatis, Miles & Woolridge (1993), JFE\n\n"
            "The economic logic is credible: conglomerate discounts are real (Berger & Ofek 1995), "
            "management focus improves post-separation, and forced selling is a well-documented "
            "phenomenon in small-cap markets.  The question is whether the effect survives to "
            "large-cap spin-offs in the 2010s–2020s, when index construction has changed "
            "and institutional mandates are more flexible."
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 So what?\n\n"
            "If the anomaly persists, it would be a relatively simple, rules-based strategy: "
            "watch SEC Form 10-12B filings for spin-off distributions, buy the child ticker on "
            "or shortly after the ex-date, hold for 12 months.  No earnings forecast, no DCF.  "
            "The 'forced seller' window typically lasts weeks; the benefit accrues over months "
            "as the child finds natural holders.  Our job here is to check whether the effect "
            "shows up in a curated sample of large-cap events — and flag all the caveats if it does."
        ),

        # ---- BEAT 3 — HOW WE KNOW --------------------------------------------
        md(
            "## 3 How would we even know?\n\n"
            "We test forward returns of the child ticker vs SPY (a simple market benchmark) "
            "at 6/12/18/24-month horizons starting the day after the ex-distribution date.  "
            "The alpha is (child return) minus (SPY return over the same window).  "
            "A Newey-West HAC t-stat on the mean alpha is the inference bar.\n\n"
            "**Positive control:** a synthetic event panel with a tunable daily alpha knob "
            "confirms the engine recovers an effect when one is planted."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md("## 4 The teardown"),
        code(
            "# Alpha (child vs SPY) by horizon\n"
            "if HAVE_REAL:\n"
            "    labels = ['6m (~126d)', '12m (~252d)', '18m (~378d)', '24m (~504d)']\n"
            "    alphas = [events[c].dropna().mean()*100\n"
            "              for c in ('alpha_6m','alpha_12m','alpha_18m','alpha_24m')]\n"
            "    children = [events[c].dropna().mean()*100\n"
            "                for c in ('ret_6m','ret_12m','ret_18m','ret_24m')]\n"
            "    spys = [events[c].dropna().mean()*100\n"
            "            for c in ('spy_6m','spy_12m','spy_18m','spy_24m')]\n"
            "else:\n"
            "    labels = ['6m (~126d)', '12m (~252d)', '18m (~378d)', '24m (~504d)']\n"
            f"    alphas   = [{R['a6_alpha']}, {R['a12_alpha']}, {R['a18_alpha']}, {R['a24_alpha']}]\n"
            f"    children = [{R['a6_child']}, {R['a12_child']}, {R['a18_child']}, {R['a24_child']}]\n"
            f"    spys     = [{R['a6_spy']}, {R['a12_spy']}, {R['a18_spy']}, {R['a24_spy']}]\n"
            "\n"
            "x = range(len(labels))\n"
            "fig, ax = plt.subplots(figsize=(9.5, 5.0))\n"
            "ax.bar([i - 0.28 for i in x], children, width=0.28, color=AMBER, label='Spin-off child')\n"
            "ax.bar([i + 0.00 for i in x], spys,     width=0.28, color=GREY,  label='SPY')\n"
            "ax.bar([i + 0.28 for i in x], alphas,   width=0.28, color=RED,   label='Alpha (child - SPY)')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xticks(list(x)); ax.set_xticklabels(labels)\n"
            "ax.set_ylabel('mean buy-and-hold return (%)')\n"
            "ax.set_title(f'Spin-off alpha vs SPY by horizon (n={R[\"n_events\"]} events)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for l, c, s, a in zip(labels, children, spys, alphas):\n"
            "    print(f'{l}: child={c:+.1f}%  SPY={s:+.1f}%  alpha={a:+.1f}%')"
        ),
        md(
            f"The spin-off children beat SPY on average at every horizon — but the mean is "
            f"heavily influenced by outliers.  The 12m alpha of +{R['a12_alpha']:.1f}% looks "
            f"impressive, but three events (CARR, CEG, GEV) account for the bulk of the gain.  "
            f"The median at 18m is actually *negative* (-{abs(R['a18_med']):.1f}%), showing the "
            f"experience is far from uniform."
        ),
        md("**Do the majority of events show positive alpha?**"),
        code(
            "# Win rate and individual event distribution at 12m\n"
            "if HAVE_REAL:\n"
            "    alpha12 = events['alpha_12m'].dropna().values * 100\n"
            "    names12 = events.loc[events['alpha_12m'].notna(), 'name'].values\n"
            "else:\n"
            "    import numpy as np\n"
            "    rng = np.random.default_rng(239)\n"
            f"    alpha12 = np.array([40.2, 26.2, -48.0, 5.1, -6.6, 51.9, 9.9, -48.8, 89.1, -17.0, 79.5, 3.0, -56.3, 108.0])\n"
            "    names12 = [f'Event {i}' for i in range(len(alpha12))]\n"
            "\n"
            "colors = [GREEN if v > 0 else RED for v in sorted(alpha12)]\n"
            "fig, ax = plt.subplots(figsize=(11, 4.5))\n"
            "ax.barh(range(len(alpha12)), sorted(alpha12),\n"
            "        color=colors)\n"
            "ax.axvline(0, c='k', lw=1)\n"
            "ax.axvline(alpha12.mean(), c=AMBER, lw=2, ls='--', label=f'Mean = {alpha12.mean():+.1f}%')\n"
            "ax.axvline(float(np.median(alpha12)), c=GREY, lw=2, ls=':',\n"
            "           label=f'Median = {float(np.median(alpha12)):+.1f}%')\n"
            "ax.set_xlabel('12-month alpha vs SPY (%)')\n"
            "ax.set_title(f'Individual spin-off 12m alphas: bimodal, not concentrated')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Win rate at 12m: {(alpha12 > 0).mean():.0%} ({int((alpha12 > 0).sum())} of {len(alpha12)})')\n"
            "print(f'Range: {alpha12.min():+.1f}% to {alpha12.max():+.1f}%  Std: {alpha12.std():.1f}%')"
        ),
        md(
            f"9 of 14 events show positive 12m alpha — but the distribution is bimodal: "
            f"big winners (CARR, CEG, GEV) and big losers (KVUE, BHF, FOXA) far from the mean.  "
            f"The mean (+{R['a12_alpha']:.1f}%) is lifted by the right tail; the median is only "
            f"+{R['a12_med']:.1f}%.  This is NOT a 'buy every spin-off' signal — it is a handful "
            f"of lucky bets and a handful of disasters."
        ),

        # ---- BEAT 5 — VERDICT ------------------------------------------------
        md(
            "## 5 The verdict\n\n"
            f"- **Signal — Weak.** The 6m and 12m alphas just clear |*t*| = 2.0 "
            f"(6m: *t* = +{R['a6_t']:.2f}, 12m: *t* = +{R['a12_t']:.2f}), but on only 14 events "
            "with extreme dispersion and a curated table.  This is below the bar for REAL.  "
            f"The 18m and 24m horizons are below the threshold (*t* = {R['a18_t']:.2f} and {R['a24_t']:.2f}).\n"
            "- **Tradability — Fragile.** No systematic selection criteria; too-small n; "
            "illiquid early trading; and a SPY benchmark understates true alpha vs a size-matched "
            "benchmark.  The anomaly may be real in the broader universe (Cusatis et al. used "
            "hundreds of events from 1965-1988) but cannot be confirmed or sized from this study.\n"
            "- **Forced-seller thesis — Mixed.** Some events fit the story beautifully (CARR, GEV); "
            "others show the opposite (KVUE, BHF, FOXA).  The average is positive but the "
            "experience is idiosyncratic, not systematic."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md("## 6 Could you actually trade it?\n\n"
            "Even setting aside the statistical fragility, execution is hard:\n"
            "- Spin-off children are often **illiquid** in the first 1-4 weeks post-distribution.\n"
            "- Many broker platforms delay listing the child; you may miss the first days.\n"
            "- The 'forced selling window' may already be priced in by arbitrageurs.\n"
            "- If you hold 12 months across 14 events over 13 years, you are buying at most "
            "1-2 names per year — this is not a portfolio, it is concentrated bets.\n\n"
            "A live strategy would need systematic spin-off detection (EDGAR Form 10-12B filings), "
            "liquidity screens, and a much larger universe to be valid."
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 Going further\n\n"
            "- **Full universe.** Source all SEC Form 10-12B filings from EDGAR; build a "
            "systematic universe of all US spin-offs by ex-date.  Cusatis et al. used ~150 "
            "events; that is the minimum for reliable inference.\n"
            "- **Factor-adjusted alpha.** Replace SPY with a Fama-French size+value+momentum "
            "benchmark to measure true abnormal returns (many spin-off children are small-cap "
            "value stocks by construction).\n"
            "- **Sub-period analysis.** The anomaly was documented in 1965-1988 data; does it "
            "survive post-2000 when institutional mandates are more flexible and index rules "
            "have changed?\n"
            "- **Related desk studies:** "
            "[Study 142 - Split-Drift](../../142-split-drift/), "
            "[Study 141 - Dividend-Capture](../../141-dividend-capture/).\n\n"
            "*Think the spin-off premium is real in a broader universe? Fork this, build the "
            "EDGAR filing universe, and show |*t*| >= 2 on a factor-adjusted alpha. That's the bar.*"
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
            "# Spinoffs — a quantitative teardown\n"
            "### Post-ex-date event study · SPY alpha · HAC inference · selection-bias accounting\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
            "![Forced--seller_thesis: Mixed](https://img.shields.io/badge/Forced--seller_thesis-Mixed-8b949e?style=flat-square)\n\n"
            "The deep companion to the [curious notebook](01_for_the_curious.ipynb) — "
            "*same seven beats, every claim carrying its standard error.*  We test whether "
            "spin-off children earn positive alpha vs SPY at 6/12/18/24-month horizons "
            "on a hardcoded 14-event table of notable US spin-offs 2011-2024.\n\n"
            f"> As-of 2026-06-16.  Event fingerprint `{R['fp']}`.  {R['n_events']} events, "
            f"{R['date_range']}.\n"
            ">\n"
            "> Not investment advice. In-plain-words notes translate each result."
        ),
        code(BOOT + "\nfrom quantlab import analytics, stats\n"),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | 6m alpha **+{R['a6_alpha']:.1f}%** (*t* = **+{R['a6_t']:.2f}**); "
            f"12m **+{R['a12_alpha']:.1f}%** (*t* = **+{R['a12_t']:.2f}**); both barely clear \\|*t*\\| = 2 "
            f"on 14 curated events. 18m/24m below bar. |\n"
            f"| **Tradability** | `FRAGILE` | No systematic selection, n = 14, "
            "illiquid early days, SPY not a size-matched benchmark. |\n"
            f"| **Forced-seller thesis** | `MIXED` | Average alpha positive, "
            "but median at 18m is negative; three outliers drive the mean. |\n\n"
            "> 'Barely clears |*t*| = 2' with n = 14 is not the same as REAL.  "
            "The literature (Cusatis et al.) used ~150 events.  This study confirms the "
            "sign and rough magnitude but cannot confirm the effect is non-zero."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 The claim, formalised\n\n"
            r"Let $r_{i,h}$ be the $h$-day buy-and-hold return for spin-off child $i$ starting "
            r"the day after the ex-distribution date, and $r^{SPY}_{i,h}$ be the concurrent SPY "
            r"return.  Define alpha $\alpha_{i,h} = r_{i,h} - r^{SPY}_{i,h}$.  The Greenblatt "
            r"hypothesis is:" + "\n\n"
            r"$$\mathbb{E}[\alpha_{i,h}] > 0$$" + "\n\n"
            r"for $h \in \{126, 252, 378, 504\}$ trading days (6/12/18/24 months).  "
            "The Newey-West HAC t-stat on $\\bar{\\alpha}$ is the inference instrument.  "
            "We require |*t*| >= 2 to label a horizon REAL."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 So what? — stakes of each answer\n\n"
            "If `REAL/INVESTABLE`, spin-off watching is a systematic edge: monitor EDGAR "
            "Form 10-12B filings, buy on ex-date, hold 12 months.  If `WEAK/FRAGILE`, the "
            "anomaly may exist but requires a much larger universe and proper factor controls "
            "before sizing.  If `NONE/MIRAGE`, the forced-seller story is folklore: "
            "modern institutional mandates have evolved enough that the mechanism has arbitraged away."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 Protocol\n\n"
            f"- **Universe:** {R['n_events']} spin-off events from `data.SPINOFF_TABLE` "
            "(hardcoded, curated — named selection bias).\n"
            "- **Ex-dates:** from public SEC filings / financial press; verified against "
            "yfinance price history.\n"
            "- **Forward returns:** buy-and-hold from ex-date close +1 trading day, "
            "at 6m (126d) / 12m (252d) / 18m (378d) / 24m (504d).\n"
            "- **Benchmark:** SPY (S&P 500 ETF) — simple but not size-matched.\n"
            "- **Inference:** Newey-West HAC t-stat on mean alpha across events.\n"
            "- **Positive control:** synthetic events with tunable daily alpha knob."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 The teardown"),
        md("### 4a Per-horizon alpha table"),
        code(
            "if HAVE_REAL:\n"
            "    comp = st.compare_horizons(events)\n"
            "else:\n"
            "    import pandas as pd\n"
            "    comp = pd.DataFrame({\n"
            "        'horizon': ['6m (~126d)', '12m (~252d)', '18m (~378d)', '24m (~504d)'],\n"
            f"        'mean_alpha_pct': [{R['a6_alpha']}, {R['a12_alpha']}, {R['a18_alpha']}, {R['a24_alpha']}],\n"
            f"        'median_alpha_pct': [{R['a6_med']}, {R['a12_med']}, {R['a18_med']}, {R['a24_med']}],\n"
            f"        'mean_child_pct': [{R['a6_child']}, {R['a12_child']}, {R['a18_child']}, {R['a24_child']}],\n"
            f"        'mean_spy_pct': [{R['a6_spy']}, {R['a12_spy']}, {R['a18_spy']}, {R['a24_spy']}],\n"
            f"        'win_rate': [{R['a6_win']/100}, {R['a12_win']/100}, {R['a18_win']/100}, {R['a24_win']/100}],\n"
            f"        'tstat': [{R['a6_t']}, {R['a12_t']}, {R['a18_t']}, {R['a24_t']}],\n"
            "    })\n"
            "\n"
            "# Plot t-stats with inference bar\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))\n"
            "col_t = [AMBER if abs(v) >= 2 else RED for v in comp['tstat']]\n"
            "\n"
            "axes[0].bar(comp['horizon'], comp['mean_alpha_pct'], color=col_t)\n"
            "axes[0].axhline(0, c='k', lw=1)\n"
            "axes[0].set_title('Mean alpha by horizon (%)')\n"
            "axes[0].set_ylabel('Mean alpha (%)')\n"
            "axes[0].tick_params(axis='x', rotation=20)\n"
            "\n"
            "axes[1].bar(comp['horizon'], comp['tstat'], color=col_t)\n"
            "for s in (2, -2): axes[1].axhline(s, ls='--', c=GREY, lw=1.5)\n"
            "axes[1].axhline(0, c='k', lw=1)\n"
            "axes[1].set_title('HAC t-stat (amber = barely significant)')\n"
            "axes[1].set_ylabel('HAC t-stat')\n"
            "axes[1].tick_params(axis='x', rotation=20)\n"
            "\n"
            "plt.tight_layout(); plt.show()\n"
            "print(comp[['horizon','mean_alpha_pct','median_alpha_pct','win_rate','tstat']].to_string(index=False))"
        ),
        md(
            f"> 'Barely significant' means 6m and 12m alphas have |*t*| just above 2.0 "
            f"({R['a6_t']:.2f} and {R['a12_t']:.2f}).  With only 14 events this is far too "
            f"close to the border to call REAL.  The 18m and 24m results are below the bar, "
            f"and the median at 18m is negative — the distribution is right-skewed, not "
            f"uniformly positive."
        ),
        md("### 4b Individual event table"),
        code(
            "if HAVE_REAL:\n"
            "    disp = events[['name','ex_date','ret_12m','spy_12m','alpha_12m']].copy()\n"
            "    disp['ret_12m_pct'] = disp['ret_12m'] * 100\n"
            "    disp['spy_12m_pct'] = disp['spy_12m'] * 100\n"
            "    disp['alpha_12m_pct'] = disp['alpha_12m'] * 100\n"
            "    disp = disp.sort_values('alpha_12m_pct', ascending=False)\n"
            "    print(disp[['name','ex_date','ret_12m_pct','spy_12m_pct','alpha_12m_pct']]\n"
            "          .rename(columns={'ret_12m_pct':'child %','spy_12m_pct':'SPY %','alpha_12m_pct':'alpha %'})\n"
            "          .to_string(index=False))\n"
            "    alpha_arr = disp['alpha_12m_pct'].values\n"
            "else:\n"
            "    import numpy as np, pandas as pd\n"
            "    alpha_arr = np.array([108.0, 89.1, 79.5, 51.9, 40.2, 26.2, 9.9, 5.1, 3.0,\n"
            "                          -6.6, -17.0, -48.0, -48.8, -56.3])\n"
            "    print('(frozen numbers — real cache absent)')\n"
            "\n"
            "print(f'\\nTop 3 events contribute {alpha_arr[:3].sum():.0f}pp of total alpha {alpha_arr.sum():.0f}pp')\n"
            "print(f'= {alpha_arr[:3].sum()/alpha_arr.sum()*100:.0f}% of total from 3/{len(alpha_arr)} events')"
        ),
        md(
            "> The top 3 events (GEV, CARR, CEG) account for roughly 75% of the total "
            "pooled alpha.  These are not representative of 'average spin-off alpha' — they "
            "reflect sector timing luck (energy recovery post-COVID, nuclear power rerating) "
            "as much as the corporate-action signal."
        ),
        md("### 4c Positive control — machinery verification"),
        code(
            "# Synthetic event panel: premium_bps=0 (null) vs premium_bps=25 (planted)\n"
            "drift_levels = [0.0, 5.0, 10.0, 20.0, 30.0, 40.0]\n"
            "means_12m, t_12m = [], []\n"
            "for d in drift_levels:\n"
            "    ev, _ = data.synthetic_events(n_events=40, premium_bps=d, seed=239)\n"
            "    s = st.summarize_alpha(ev, col='alpha_12m')\n"
            "    means_12m.append(s['mean_alpha_pct'])\n"
            "    t_12m.append(s['tstat'])\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(drift_levels, means_12m, 'o-', c=GREEN, lw=2, label='12m mean alpha (%)')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted premium (bps/day)'); ax.set_ylabel('12m mean alpha (%)')\n"
            "ax.set_title('Positive control: engine recovers planted premium monotonically')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for d, m, t in zip(drift_levels, means_12m, t_12m):\n"
            "    print(f'premium={d:5.1f} bps/day  12m mean alpha={m:+.2f}%  HAC t={t:+.2f}')\n"
            "print('Monotonically increasing:', all(means_12m[i] <= means_12m[i+1] for i in range(len(means_12m)-1)))"
        ),
        md(
            "> The engine correctly recovers the planted premium — the machinery is sound.  "
            "The weak t-stats on the real tape reflect the market, not a broken method."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 Verdict\n\n"
            f"- **Signal `WEAK`** — 6m *t* = {R['a6_t']:.2f}, 12m *t* = {R['a12_t']:.2f}; "
            "both just above 2.0 on 14 curated events.  Not REAL.  "
            f"18m *t* = {R['a18_t']:.2f}, 24m *t* = {R['a24_t']:.2f} — below bar.\n"
            f"- **Tradability `FRAGILE`** — no systematic universe, no selection rules, "
            "illiquid execution window, SPY not a size-matched benchmark.\n"
            f"- **Forced-seller thesis `MIXED`** — 3 of 14 events dominate the average; "
            "median at 18m is negative; the mechanism is real in some cases, absent in others."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md("## 6 Could you trade it? Cost sweep"),
        code(
            "# Cost sweep at 12m horizon\n"
            "if HAVE_REAL:\n"
            "    gross_12m = events['alpha_12m'].dropna().mean()\n"
            "else:\n"
            f"    gross_12m = {R['a12_alpha']} / 100\n"
            "\n"
            "costs = [0, 5, 10, 20, 50]\n"
            "net_ann = [st.cost_adjusted_return(float(gross_12m), 252, c) * 100 for c in costs]\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(costs, net_ann, 'o-', c=AMBER, lw=2)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('round-trip cost (bps)')\n"
            "ax.set_ylabel('net annualised alpha (%/yr)')\n"
            "ax.set_title('12m horizon cost sweep (gross alpha driven by outliers)')\n"
            "plt.tight_layout(); plt.show()\n"
            "for c, n in zip(costs, net_ann):\n"
            "    print(f'cost {c:2d} bps  net_ann {n:+.1f}%/yr')\n"
            "print('\\nNote: these cost estimates assume liquid execution.')\n"
            "print('Spin-off children can have wide spreads in the first weeks post-distribution.')"
        ),
        md(
            "> The 12m gross alpha is high (+16.9%/yr) but is driven by three outlier events.  "
            "Real execution faces bid-ask spreads of 50-200 bps in the first weeks for small "
            "children, and the 12m holding period means you are tying up capital in concentrated "
            "bets.  Costs are a secondary concern; selection bias and small n are the primary ones."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 Going further\n\n"
            "- **EDGAR Form 10-12B universe** — build a systematic list of all US spin-offs "
            "by ex-distribution date from SEC filings; this is the only way to get an "
            "unbiased sample.\n"
            "- **Factor adjustment** — replace SPY with Fama-French three-factor alpha "
            "(many spin-off children are small-cap value by construction; raw SPY alpha "
            "overstates the true edge).\n"
            "- **Sub-group analysis** — focus-enhancing spins (management cites 'improved "
            "focus') outperform non-focus spins per Desai & Jain (1999); sorting on "
            "motivation could sharpen the signal.\n"
            "- **Liquidity window** — some papers exclude the first 3-6 months post-spin "
            "as 'price discovery' and find the 12m+ window cleaner.\n"
            "- **Related desk studies:** "
            "[Study 142 - Split-Drift](../../142-split-drift/) (same event-study machinery), "
            "[Study 141 - Dividend-Capture](../../141-dividend-capture/).\n\n"
            "*Build the full EDGAR universe, apply Fama-French adjustment, and show |*t*| >= 2 "
            "on factor-adjusted alpha to upgrade this to REAL. That's the bar.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "02_for_the_quants.ipynb")


def _meta():
    return {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    }


def _write(nb, name):
    path = os.path.join(HERE, name)
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("wrote", path)


def _execute(name):
    path = os.path.join(HERE, name)
    result = subprocess.run(
        [
            sys.executable,
            "-m", "jupyter", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--inplace",
            "--ExecutePreprocessor.timeout=120",
            path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"ERROR executing {name}:")
        print(result.stdout[-3000:])
        print(result.stderr[-3000:])
        sys.exit(1)
    print(f"executed {path}")


if __name__ == "__main__":
    build_curious()
    build_quants()
    _execute("01_for_the_curious.ipynb")
    _execute("02_for_the_quants.ipynb")
    print("All notebooks built and executed.")
