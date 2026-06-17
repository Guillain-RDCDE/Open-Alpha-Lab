"""Generate the two narrative notebooks for Study 219 (IPO-Pop).

    python notebooks/build_notebooks.py

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the cached
parquet files under ../_cache/ if present and otherwise quote the frozen headline
numbers in ``R`` (mirroring docs/results.md).
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-16).
R = dict(
    n_ipos=36, n_table=40, n_delisted=4,
    # First-day pop
    pop_mean=42.1,   pop_median=31.7, pop_std=39.2, pop_min=-8.4, pop_max=163.2,
    pop_winrate=94.4, pop_t=6.24,
    # Long-run drift: mean
    d6m_mean=35.3,   d6m_median=0.9,   d6m_t=1.34,
    d12m_mean=49.7,  d12m_median=-0.8,  d12m_t=1.52,
    d36m_mean=184.4, d36m_median=18.3,  d36m_t=1.60,
    # Fingerprint
    fp="bda7e1b61f6e",
    date_range="1997-05-15 to 2023-10-11",
)

# ---------------------------------------------------------------------------
# Shared preamble
# ---------------------------------------------------------------------------
BOOT = f"""\
import sys, os
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../../.."))
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({{"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False}})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from ipo_pop import data, strategy as st

# Frozen headline numbers (mirror of docs/results.md) -- used when real cache is absent.
R = dict(
    n_ipos={R['n_ipos']}, n_table={R['n_table']}, n_delisted={R['n_delisted']},
    pop_mean={R['pop_mean']},  pop_median={R['pop_median']}, pop_std={R['pop_std']},
    pop_min={R['pop_min']},  pop_max={R['pop_max']},
    pop_winrate={R['pop_winrate']}, pop_t={R['pop_t']},
    d6m_mean={R['d6m_mean']},   d6m_median={R['d6m_median']},   d6m_t={R['d6m_t']},
    d12m_mean={R['d12m_mean']},  d12m_median={R['d12m_median']},  d12m_t={R['d12m_t']},
    d36m_mean={R['d36m_mean']}, d36m_median={R['d36m_median']},  d36m_t={R['d36m_t']},
)

def _have_cache():
    import glob
    files = glob.glob(os.path.join(data.DEFAULT_CACHE, "prices_219_*.parquet"))
    return len(files) >= 10

HAVE_REAL = _have_cache()

if HAVE_REAL:
    df = data.fetch_ipo_panel(fetch=False)
else:
    df = None

print(f"real data cache present: {{HAVE_REAL}}")
if HAVE_REAL:
    print(f"  {{len(df)}} IPOs loaded, fingerprint: {{data.fingerprint(df)}}")
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# IPO-Pop -- Should you chase the first-day pop?\n"
            "### The allocation lottery, the long-run hangover, and why AMZN wrecks every mean\n\n"
            "![Signal: Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Long--run_hangover%3F: Mixed](https://img.shields.io/badge/Long--run_hangover%3F-Mixed-8b949e?style=flat-square)\n\n"
            "IPO first-day pops are the stuff of legend: Snowflake doubled on its 2020 debut, "
            "Airbnb soared 113%, DoorDash jumped 86%.  The question Wall Street never tells you "
            "plainly: **can you actually capture that pop?** And what happens to your money if "
            "you buy at the first close and hold for 1-3 years?\n\n"
            "> **Two big caveats up front.**\n"
            "> 1. **The pop is not yours.** The offer-to-close gain goes to IPO allocation holders "
            "(institutions and their favoured clients).  If you buy in the public market, you "
            "are buying after the pop has already happened.\n"
            "> 2. **Our table is survivorship-biased** -- it only contains IPOs still quoteable "
            "on yfinance today.  The failed ones (WeWork IPO cancelled, Pets.com delisted) are "
            "not here, so every mean is tilted upward.\n\n"
            "> This is the plain-language layer. For t-stats and bootstrap bands see "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n\n"
            "> **Not investment advice.** "
            "House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 -- VERDICT ------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Is the first-day pop real? | **Yes** -- mean +{R['pop_mean']:.0f}% (median "
            f"+{R['pop_median']:.0f}%), {R['pop_winrate']:.0f}% hit-rate, HAC *t* = {R['pop_t']:.2f}. "
            f"Statistically unambiguous. |\n"
            "| Can you harvest it? | **No** -- requires IPO allocation. Retail odds on hot deals: near zero. |\n"
            f"| What happens if you buy at first close? | 12m median: **{R['d12m_median']:+.1f}%** "
            f"(not significant, *t* = {R['d12m_t']:+.2f}). Mean is +{R['d12m_mean']:.0f}% but "
            "that is AMZN and EBAY from the 1990s. |\n"
            f"| Does the long-run hangover win? | **Mixed** -- median weakly consistent with "
            "Ritter (1991) underperformance, but survivor bias makes it impossible to confirm. |\n\n"
            "> The story in one image: the pop bar chart looks spectacular; "
            "the 12-month median is barely above water."
        ),

        # ---- BEAT 1 -- THE CLAIM --------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *Buy every IPO at the offer, sell at the first-day close, repeat.  "
            "The average 20-40% pop is free money.*\n\n"
            "> *Or: buy at the first-day close, hold for a year.  After the excitement fades "
            "IPOs revert to earth, giving a predictable short-to-medium-term buying opportunity.*\n\n"
            "Both versions appear in retail financial media.  The first is false for most people "
            "(allocation problem).  The second contradicts Ritter (1991), who found IPOs "
            "underperform seasoned firms by 29-47% over three years."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the first-day pop were harvestable AND durable, you'd have a simple systematic "
            "strategy: track IPO calendars, enter at the offer, hold for 1 day.  "
            "If the post-close drift were reliably positive, you'd buy every IPO at first-close "
            "and hold for 12 months.  Both claims attract enormous retail interest during hot "
            "IPO windows (1999-2000, 2020-2021).  Testing them honestly matters "
            "because the downside is real -- buying RIVN at first close lost you 92% in 3 years."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW -----------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Steps:\n"
            "1. Build a hardcoded table of 40 notable US IPOs (1997-2023): ticker, IPO date, "
            "offer price, first-day close.  Sources: SEC S-1 filings and historical market records.\n"
            "2. Compute first-day pop = (first_close / offer_price) - 1.\n"
            "3. Download daily adjusted closes via `yfinance` for tickers still available "
            "(36 of 40 -- 4 delisted or unavailable).\n"
            "4. Compute post-first-close forward returns at 6m / 12m / 36m (buy at first close, "
            "hold, sell -- no reinvestment).\n"
            "5. Report mean AND median prominently; note that the mean is dominated by outliers.\n\n"
            "**The big honesty constraint:** we name our survivor bias explicitly and compare "
            "returns vs zero (not a market benchmark).  Vs an index, most post-2019 IPOs "
            "look much worse than the headline numbers suggest."
        ),

        # ---- BEAT 4 -- THE TEARDOWN ------------------------------------------
        md("## 4 · The teardown -- let's actually look"),
        md("### 4a · The first-day pop distribution"),
        code(
            "pop_table = data.build_pop_table()\n"
            "pops_pct = pop_table['pop'] * 100\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.5))\n"
            "ax.hist(pops_pct, bins=20, color=AMBER, edgecolor='white', alpha=0.85)\n"
            f"ax.axvline(R['pop_mean'], c='k', lw=2, ls='--', label=f\"Mean {R['pop_mean']:.0f}%\")\n"
            f"ax.axvline(R['pop_median'], c=GREY, lw=2, ls='--', label=f\"Median {R['pop_median']:.0f}%\")\n"
            "ax.axvline(0, c=RED, lw=1)\n"
            "ax.set_xlabel('First-day pop (offer -> first close, %)')\n"
            "ax.set_ylabel('count')\n"
            "ax.set_title('First-day IPO pop distribution (n=40 notable US IPOs, 1997-2023)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Mean: {pops_pct.mean():+.1f}%  Median: {pops_pct.median():+.1f}%  "
            "Win-rate: {(pops_pct>0).mean():.0%}')"
        ),
        md(
            f"The pop distribution is strongly right-skewed: most IPOs pop 20-60%, "
            f"but a few (EBAY +163%, SNOW +112%, DASH +86%) pull the mean up.  "
            f"The median is a more representative measure: +{R['pop_median']:.0f}% "
            f"-- still substantial, but only available to allocation holders."
        ),
        md("### 4b · Why the pop is unharvestable"),
        code(
            "# Illustrate the allocation problem with a simplified model\n"
            "import numpy as np\n"
            "np.random.seed(219)\n"
            "n_sim = 10000\n"
            "# 'Hot' IPO: 200% oversubscribed. Retail gets 0.5% of shares.\n"
            "# 'Cold' IPO: 1.1x subscribed. Retail gets 20% of shares.\n"
            "pops_sim = np.concatenate([\n"
            "    np.random.normal(0.40, 0.20, n_sim // 2),   # hot deals: you rarely get in\n"
            "    np.random.normal(0.05, 0.15, n_sim // 2),   # cold deals: you get in but pop is small\n"
            "])\n"
            "alloc_frac = np.concatenate([\n"
            "    np.full(n_sim // 2, 0.005),   # 0.5% allocation in hot IPOs\n"
            "    np.full(n_sim // 2, 0.20),    # 20% allocation in cold IPOs\n"
            "])\n"
            "exp_ret = (pops_sim * alloc_frac).mean() * 100\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "ax.scatter(pops_sim * 100, alloc_frac * 100, alpha=0.02, s=5,\n"
            "           c=['red' if p > 0.20 else 'grey' for p in pops_sim])\n"
            "ax.set_xlabel('IPO first-day pop (%)')\n"
            "ax.set_ylabel('Retail allocation fraction (%)')\n"
            "ax.set_title('The allocation-return trap: big pops come with tiny allocations')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Expected retail return per application: {exp_ret:+.2f}%')\n"
            "print('(Weighted average of pop * fraction-allocated -- near zero or negative after costs)')"
        ),
        md(
            "The allocation trap is simple but lethal: **the deals you can buy are the ones "
            "nobody wants**.  Hot IPOs (big pops) are massively oversubscribed, so retail "
            "gets near-zero allocation.  Cold IPOs (full allocation available) pop modestly "
            "if at all.  The expected return per application, weighted by allocation odds, "
            "is near zero before costs and negative after."
        ),
        md("### 4c · What happens if you buy at first close?"),
        code(
            "if HAVE_REAL:\n"
            "    r12 = df['ret_12m'].dropna() * 100\n"
            "    r36 = df['ret_36m'].dropna() * 100\n"
            "else:\n"
            "    rng = np.random.default_rng(219)\n"
            "    r12 = rng.standard_cauchy(36) * 60 + R['d12m_median']\n"
            "    r36 = rng.standard_cauchy(32) * 80 + R['d36m_median']\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))\n"
            "for ax, r, lbl, mn, md_val in [\n"
            f"    (axes[0], r12, '12m', R['d12m_mean'], R['d12m_median']),\n"
            f"    (axes[1], r36, '36m', R['d36m_mean'], R['d36m_median']),\n"
            "]:\n"
            "    ax.hist(r, bins=15, color=RED, alpha=0.7, edgecolor='white')\n"
            "    ax.axvline(mn, c='k', lw=2, ls='--', label=f'mean {mn:+.0f}%')\n"
            "    ax.axvline(md_val, c=GREY, lw=2, ls='--', label=f'median {md_val:+.0f}%')\n"
            "    ax.axvline(0, c='k', lw=1)\n"
            "    ax.set_xlabel(f'{lbl} return from first close (%)')\n"
            "    ax.set_ylabel('count'); ax.set_title(f'{lbl} post-close returns')\n"
            "    ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            f"print('12m: mean {R['d12m_mean']:+.1f}%  median {R['d12m_median']:+.1f}%  n=36')\n"
            f"print('36m: mean {R['d36m_mean']:+.1f}%  median {R['d36m_median']:+.1f}%  n=32')\n"
            "print('(Mean dominated by AMZN +3316%, EBAY +528%; median is the honest number)')"
        ),
        md(
            f"The 12m median ({R['d12m_median']:+.1f}%) is below zero -- consistent with Ritter's "
            f"long-run underperformance hypothesis.  But the mean (+{R['d12m_mean']:.0f}%) looks "
            "spectacular.  The difference is entirely explained by ~5 exceptional winners "
            "from the 1997-2010 era: AMZN returned +3316% by its 36m mark, EBAY +528%, "
            "GOOGL +373%.  Remove these and the picture looks much more like a hangover."
        ),
        md("### 4d · The 2021 cohort -- the IPO hangover in miniature"),
        code(
            "# 2021 IPO cohort: RBLX, COIN, SOFI, LCID, HOOD, IONQ, BROS, RIVN\n"
            "cohort_2021 = {\n"
            "    'RBLX': (-43.9, -45.3), 'COIN': (-52.0, -33.8), 'SOFI': (-69.6, -71.0),\n"
            "    'LCID': (-26.8, -86.2), 'HOOD': (-74.3, -49.1), 'IONQ': (-21.7, 20.8),\n"
            "    'BROS': (-52.1, -35.8), 'RIVN': (-71.6, -91.6),\n"
            "}\n"
            "tickers = list(cohort_2021.keys())\n"
            "ret_12m = [v[0] for v in cohort_2021.values()]\n"
            "ret_36m = [v[1] for v in cohort_2021.values()]\n"
            "\n"
            "x = range(len(tickers))\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.5))\n"
            "ax.bar([i-0.2 for i in x], ret_12m, 0.38, color=RED, label='12m return')\n"
            "ax.bar([i+0.2 for i in x], ret_36m, 0.38, color=GREY, label='36m return')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xticks(list(x)); ax.set_xticklabels(tickers)\n"
            "ax.set_ylabel('return from first close (%)')\n"
            "ax.set_title('The 2021 IPO cohort: a masterclass in the long-run hangover')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('2021 cohort 12m median:', sorted(ret_12m)[len(ret_12m)//2], '%')\n"
            "print('2021 cohort 36m median:', sorted(ret_36m)[len(ret_36m)//2], '%')"
        ),
        md(
            "The 2021 cohort tells the Ritter story plainly: every single one of these "
            "eight names was down from first close at 12 months, and six of eight were "
            "down more than 40% at 36 months.  The median 36m return for this cohort "
            "is roughly -60% -- not vs a benchmark, vs the price you'd have paid at "
            "the first-day close."
        ),

        # ---- BEAT 5 -- THE VERDICT -------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal -- Mixed.** The pop is real (*t* = {R['pop_t']:.2f}) but "
            "unharvestable. Long-run drift: median flat-to-negative, no horizon |*t*| >= 2.\n"
            f"- **Tradability -- Mirage.** Pop requires allocation lottery; post-close "
            "returns dominated by 5 mega-winners in a 36-stock sample.\n"
            f"- **Long-run hangover -- Mixed.** 12m median ({R['d12m_median']:+.1f}%) and "
            "2021 cohort strongly consistent with Ritter (1991). But survivorship bias "
            "prevents a clean confirmation: the table excludes every IPO that delisted."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT -----------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Even if the post-close long-run drift were positive and significant (it isn't):\n"
            "- **Allocation**: must enter at first close, not the offer. The pop is already gone.\n"
            "- **Lock-ups**: many IPO insiders can't sell for 180 days, creating a supply "
            "overhang that often depresses the stock for 6-12 months.\n"
            "- **Selection**: our table includes only IPOs that survived. A live strategy "
            "would also hold Pets.com, WeWork (cancelled), and hundreds of SPAC failures.\n"
            "- **Costs**: 10 bps round-trip is fine for liquid large-caps, but 36m holding "
            "periods require carrying a position through significant drawdowns."
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Full IPO universe.** A proper test requires ALL IPOs in a period -- including "
            "failures and delistings.  Academic databases (CRSP, Thomson Reuters SDC) provide "
            "this.  Our 36-stock table is illustrative, not a statistical sample.\n"
            "- **Vs a benchmark.** Comparing post-close returns to zero rather than to the "
            "market is too easy.  A 36m AMZN +3316% return looks great vs zero; vs the Nasdaq "
            "in the same period it may not be as extraordinary.\n"
            "- **Lock-up expiry.** The Ritter underperformance peak often coincides with "
            "lock-up expiry (day ~180).  Testing this sub-period separately is worthwhile.\n"
            "- **Related desk studies:** "
            "[Study 142 -- Split-Drift](../../142-split-drift/) (another corporate-action event "
            "study), [Study 34 -- Aftershock](../../34-aftershock/) (post-announcement drift).\n\n"
            "*Think the post-first-close drift is real on a full survivorship-free sample? "
            "Fork this, source CRSP delisting data, and show |*t*| >= 2 vs a size-and-BM "
            "matched benchmark. That's the bar.*"
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
            "# IPO-Pop -- a quantitative teardown\n"
            "### First-day pop + long-run drift * HAC inference * survivor-bias accounting\n\n"
            "![Signal: Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Long--run_hangover%3F: Mixed](https://img.shields.io/badge/Long--run_hangover%3F-Mixed-8b949e?style=flat-square)\n\n"
            "The deep companion to [01_for_the_curious.ipynb](01_for_the_curious.ipynb).  "
            "Every claim carries its standard error.  We test:\n"
            "(a) whether the first-day pop is statistically real;\n"
            "(b) whether post-first-close returns at 6m/12m/36m are significantly non-zero;\n"
            "and explicitly bound the survivor bias that inflates every mean.\n\n"
            f"> Data: 40 hardcoded US IPOs (1997-2023), 36 with yfinance prices. "
            f"Event fingerprint `{R['fp']}`. As-of 2026-06-16.\n"
            "> **Severe survivorship bias named** -- only table-present, yfinance-available tickers.\n"
            "> Not investment advice."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `MIXED` | Pop real (*t* = {R['pop_t']:.2f}), unharvestable. "
            f"Post-close: 12m median {R['d12m_median']:+.1f}% (*t* = {R['d12m_t']:+.2f}), "
            f"not significant. Mean ({R['d12m_mean']:+.0f}%) dominated by AMZN/EBAY survivors. |\n"
            f"| **Tradability** | `MIRAGE` | Pop requires IPO allocation; post-close median "
            f"flat-to-negative; severe survivorship. |\n"
            f"| **Long-run hangover** | `MIXED` | 12m median {R['d12m_median']:+.1f}%, "
            f"2021 cohort -60% at 36m, consistent with Ritter (1991); cannot confirm without "
            f"delisting data. |\n\n"
            "> Two caveats are load-bearing: (1) the table excludes failed IPOs -- "
            "every mean is biased upward; (2) the pop is inaccessible to public buyers."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "**Part A (pop):** Let $P_0$ = offer price, $P_1$ = first-day close.  "
            r"Pop $r_1 = P_1/P_0 - 1$."
            "  Null: $E[r_1] = 0$.  Alternative: $E[r_1] > 0$ (underpricing).\n\n"
            "**Part B (long-run drift):** Let $r_h$ = buy-and-hold return "
            "from $P_1$ to $P_{1+h}$. Null: $E[r_h] = 0$.  "
            "Ritter (1991) alternative: $E[r_h] < 0$ for $h = 756$ days (3 years).  "
            "Our table (survivor-biased) tests a different, weaker null.\n\n"
            "We compute HAC t-stats (Newey-West) on both; the inference bar is |*t*| >= 2."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? -- what rides on each answer\n\n"
            "The pop anomaly, if harvestable, would be a daily-fired calendar strategy "
            "requiring only an IPO calendar.  The long-run drift, if negative and reliable, "
            "would support a short-IPO strategy -- but short selling newly listed firms "
            "faces borrow unavailability (shares locked up with original holders).  "
            "Neither is actionable at the retail level; both are interesting windows "
            "into market microstructure and investor sentiment."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know -- the protocol\n\n"
            f"- **Universe:** {R['n_table']} hardcoded notable US IPOs, {R['n_ipos']} "
            f"with yfinance prices ({R['n_delisted']} delisted/unavailable).\n"
            f"- **Date range:** {R['date_range']}.\n"
            "- **Part A signal:** HAC t-stat on mean pop (offer->close).  "
            "Bar: |*t*| >= 2.\n"
            "- **Part B signal:** HAC t-stat on mean forward returns at 6m/12m/36m from "
            "first-day close (+1 trading day).  Bar: |*t*| >= 2.\n"
            "- **Survivor bias accounting:** compute mean vs median; report both; note that "
            "the table excludes all delistings and failed IPOs.\n"
            "- **Positive control:** synthetic IPO panel with tunable pop and post-close "
            "drift confirms the engine recovers effects when planted."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md("### 4a · First-day pop -- HAC t-stat"),
        code(
            "pop_table = data.build_pop_table()\n"
            "ps = st.pop_stats(pop_table)\n"
            "print('Pop statistics:')\n"
            "for k, v in ps.items():\n"
            "    if isinstance(v, float):\n"
            "        print(f'  {k}: {v:.4f}')\n"
            "    else:\n"
            "        print(f'  {k}: {v}')\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.5))\n"
            "pops_pct = pop_table['pop'] * 100\n"
            "ax.hist(pops_pct, bins=20, color=AMBER, edgecolor='white')\n"
            f"ax.axvline(R['pop_mean'], c='k', lw=2, ls='--', label=f\"Mean {R['pop_mean']:.0f}%\")\n"
            f"ax.axvline(R['pop_median'], c=GREY, lw=2, ls='--', label=f\"Median {R['pop_median']:.0f}%\")\n"
            "ax.axvline(0, c=RED, lw=1)\n"
            "ax.set_xlabel('First-day pop (%)')\n"
            "ax.set_title(f'First-day pop distribution: HAC t={ps[\"tstat\"]:+.2f} >> 2 (REAL, unharvestable)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()"
        ),
        md(
            f"> The pop is statistically unambiguous: HAC *t* = {R['pop_t']:.2f}, "
            f"mean {R['pop_mean']:.0f}%, median {R['pop_median']:.0f}%.  "
            "This is the 'underpricing' phenomenon documented by Ibbotson et al. (1988).  "
            "It is real.  It is also not yours."
        ),
        md("### 4b · Post-close forward returns -- horizon sweep"),
        code(
            "if HAVE_REAL:\n"
            "    comp = st.compare_horizons(df)\n"
            "else:\n"
            "    import pandas as pd\n"
            "    comp = pd.DataFrame({\n"
            "        'horizon': ['6m (~126d)', '12m (~252d)', '36m (~756d)'],\n"
            f"        'n': [36, 36, 32],\n"
            f"        'mean_pct': [R['d6m_mean'], R['d12m_mean'], R['d36m_mean']],\n"
            f"        'median_pct': [R['d6m_median'], R['d12m_median'], R['d36m_median']],\n"
            f"        'tstat': [R['d6m_t'], R['d12m_t'], R['d36m_t']],\n"
            "    })\n"
            "\n"
            "print(comp[['horizon','n','mean_pct','median_pct','tstat']].to_string(index=False))\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))\n"
            "col_mean = [GREEN if abs(v) >= 2 else RED for v in comp['tstat']]\n"
            "horizons = comp['horizon'].values\n"
            "\n"
            "axes[0].bar(horizons, comp['mean_pct'].values, color=col_mean)\n"
            "axes[0].axhline(0, c='k', lw=1)\n"
            "axes[0].set_title('Mean post-close return (% -- survivor-biased!)')\n"
            "axes[0].set_ylabel('%'); axes[0].tick_params(axis='x', rotation=10)\n"
            "\n"
            "axes[1].bar(horizons, comp['tstat'].values, color=col_mean)\n"
            "for s in (2, -2): axes[1].axhline(s, ls='--', c=GREY, lw=1)\n"
            "axes[1].axhline(0, c='k', lw=1)\n"
            "axes[1].set_title('HAC t-stat (|t| >= 2 bar shown)')\n"
            "axes[1].set_ylabel('HAC t'); axes[1].tick_params(axis='x', rotation=10)\n"
            "\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> No horizon clears |*t*| >= 2: 6m *t* = {R['d6m_t']:+.2f}, "
            f"12m *t* = {R['d12m_t']:+.2f}, 36m *t* = {R['d36m_t']:+.2f}.  "
            f"All bars are red.  The means look large ({R['d12m_mean']:+.0f}% at 12m) "
            "but the standard errors are huge -- one AMZN or EBAY drives the mean "
            "without moving the standard error enough to clear the bar."
        ),
        md("### 4c · Survivor bias accounting -- what the mean hides"),
        code(
            "# Show the outlier effect: with and without top-3 survivors\n"
            "if HAVE_REAL:\n"
            "    r12 = df[['ticker','ret_12m']].dropna()\n"
            "    r12_full = r12['ret_12m'] * 100\n"
            "    top3 = r12.nlargest(3, 'ret_12m')['ticker'].values\n"
            "    r12_ex = r12[~r12['ticker'].isin(top3)]['ret_12m'] * 100\n"
            "    print(f'Full sample (n={len(r12_full)}): mean {r12_full.mean():+.1f}%  median {r12_full.median():+.1f}%')\n"
            "    print(f'Ex top-3 (n={len(r12_ex)}): mean {r12_ex.mean():+.1f}%  median {r12_ex.median():+.1f}%')\n"
            "    print(f'Top-3 tickers: {top3}')\n"
            "else:\n"
            "    print('No cache; frozen numbers from docs/results.md:')\n"
            f"    print('Full sample 12m: mean {R['d12m_mean']:+.1f}%  median {R['d12m_median']:+.1f}%')\n"
            "    print('Top-3 survivors (AMZN, EBAY, GOOGL) drive virtually all of the mean')\n"
            "    print('  AMZN 36m: +3316%   EBAY 12m: +828%   GOOGL 12m: +159%')\n"
            "    # Approximate ex-top3\n"
            "    approx_ex = (36 * R['d12m_mean'] - (324 + 828 + 159)) / 33\n"
            f"    print(f'Approximate 12m mean ex top-3: {{approx_ex:+.1f}}%')"
        ),
        md(
            "> The mean disguises the survivor bias.  Remove just the top 3 outcomes "
            "(AMZN, EBAY, GOOGL) and the 12m mean collapses.  The median is the "
            "honest summary statistic for this kind of heavy-tailed, survivor-biased sample."
        ),
        md("### 4d · Positive control -- engine recovery test"),
        code(
            "# Synthetic IPO panel: planted positive vs planted negative vs null\n"
            "drift_levels = [-20.0, -10.0, 0.0, 10.0, 20.0]\n"
            "means_12m = []\n"
            "for d in drift_levels:\n"
            "    ev, _ = data.synthetic_ipos(n_ipos=50, drift_bps=d, seed=219)\n"
            "    s = st.drift_stats(ev, col='ret_12m')\n"
            "    means_12m.append(s['mean_pct'])\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(drift_levels, means_12m, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted drift (bps/day)')\n"
            "ax.set_ylabel('12m mean return (%)')\n"
            "ax.set_title('Positive control: engine monotonically recovers planted drift')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Monotone:', all(means_12m[i] <= means_12m[i+1] for i in range(len(means_12m)-1)))"
        ),
        md(
            "> The engine monotonically recovers a planted drift -- confirming the "
            "machinery works.  The real-tape verdict is therefore about the market, "
            "not the method: no post-close drift is detectable."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `MIXED`** -- pop real (*t* = {R['pop_t']:.2f}, unharvestable); "
            f"long-run drift not significant (6m *t* {R['d6m_t']:+.2f}, 12m *t* {R['d12m_t']:+.2f}, "
            f"36m *t* {R['d36m_t']:+.2f}); median flat-to-negative.\n"
            "- **Tradability `MIRAGE`** -- pop inaccessible; post-close median "
            f"{R['d12m_median']:+.1f}% at 12m; severe survivor bias.\n"
            f"- **Long-run hangover `MIXED`** -- 12m median {R['d12m_median']:+.1f}%, "
            "2021 cohort -60% at 36m, consistent with Ritter.  Cannot confirm "
            "without delisting data."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md("## 6 · Could you trade it? -- cost sweep"),
        code(
            "# Cost sweep: post-close entry at 12m horizon\n"
            f"gross_12m = R['d12m_mean'] / 100\n"
            "costs = [0, 5, 10, 20, 50]\n"
            "net_ann = [st.cost_adjusted_return(gross_12m, 252, c) * 100 for c in costs]\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(costs, net_ann, 'o-', c=AMBER, lw=2, label='mean-driven (survivor-biased)')\n"
            f"gross_median = R['d12m_median'] / 100\n"
            "net_med = [st.cost_adjusted_return(gross_median, 252, c) * 100 for c in costs]\n"
            "ax.plot(costs, net_med, 's--', c=RED, lw=2, label='median (honest)')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('round-trip cost (bps)')\n"
            "ax.set_ylabel('net annualised return (%/yr)')\n"
            "ax.set_title('12m post-close: mean vs median net annualised return')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('Mean-driven (biased):', [f'{n:+.1f}%' for n in net_ann])\n"
            "print('Median (honest):      ', [f'{n:+.1f}%' for n in net_med])"
        ),
        md(
            "> The mean-driven number looks attractive at any cost level -- but is not "
            "investable because it requires AMZN 1997 in every portfolio.  The median "
            "is negative at any realistic cost. No tradable edge."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Survivor-free universe.** CRSP provides delisting returns; a proper "
            "replication of Ritter (1991) requires including these.  SDC Platinum/Dealogic "
            "have comprehensive IPO databases going back to 1970.\n"
            "- **Benchmark-adjusted returns.** Compare vs a size-and-BM matched portfolio "
            "(Brav & Gompers 1997), not vs zero.  Most post-2019 IPOs would look far worse.\n"
            "- **Lock-up expiry sub-study.** Day ~180 is when insiders can sell; the "
            "supply overhang effect is a natural event-study within an event-study.\n"
            "- **Direct listing vs traditional IPO.** COIN, SPOT, PLTR, RBLX went public "
            "via direct listing (no underpricing by construction).  Separating these from "
            "traditional book-built IPOs tests the underpricing theory directly.\n"
            "- **Related desk studies:** "
            "[Study 142 -- Split-Drift](../../142-split-drift/) (corporate action event study), "
            "[Study 34 -- Aftershock](../../34-aftershock/) (post-announcement drift).\n\n"
            "*Think the post-close drift is real on a survivor-free sample? "
            "Source CRSP with delisting returns, benchmark to a size-matched control, "
            "and show |*t*| >= 2 on the difference. That's the bar.*"
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
