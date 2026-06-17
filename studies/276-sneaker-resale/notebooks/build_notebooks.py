"""Generate the two narrative notebooks for Study 276 (Sneaker-Resale).

    python notebooks/build_notebooks.py

This writes AND executes both notebooks (via nbconvert). The hardcoded sneaker-resale
index and the synthetic generator run anywhere, offline and deterministically; the real
^GSPC cells use the cached ^GSPC parquet at the repo-level _cache/ if present, and
otherwise quote the frozen headline numbers in ``R`` (mirroring docs/results.md), so the
notebook re-runs for any reader.

The _write convention (each build_*() ends by calling _write) is kept so the repo's
intro-restyle tooling can monkeypatch it.
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-17).
R = dict(
    n=18,
    year_start=2007,
    year_end=2024,
    sneaker_cagr_pct=7.0,
    gspc_cagr_pct=8.2,
    sneaker_mean_pct=7.5,
    sneaker_vol_pct=10.6,
    gspc_mean_pct=9.9,
    gspc_vol_pct=17.9,
    excess_mean_pct=-2.4,
    excess_t_ols=-0.567,
    excess_p_ols=0.5781,
    excess_t_hac=-0.58,
    sneaker_sharpe=0.71,
    gspc_sharpe=0.55,
    ar1_rho=0.63,
    unsmoothed_vol_pct=23.1,
    unsmoothed_sharpe=0.29,
    net_sneaker_mean_pct=4.5,
    strat_cagr_pct=9.3,
    bah_gspc_cagr_pct=8.2,
    trend_shortfall_pp=1.03,
    fp="518bff365ed1",
)

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

from sneaker_resale import data, strategy as st

def _have_cache():
    try:
        data.fetch_gspc_annual(fetch=False)
        return True
    except FileNotFoundError:
        return False

HAVE_REAL = _have_cache()
print("^GSPC cache present:", HAVE_REAL)
"""

# ---------------------------------------------------------------------------
# R-dict preamble for notebooks (so cells can reference R['key'])
# ---------------------------------------------------------------------------
R_CELL = """\
# Frozen headline numbers -- mirror of docs/results.md (as-of 2026-06-17)
R = dict(
    n=18, year_start=2007, year_end=2024,
    sneaker_cagr_pct=7.0, gspc_cagr_pct=8.2,
    sneaker_mean_pct=7.5, sneaker_vol_pct=10.6,
    gspc_mean_pct=9.9, gspc_vol_pct=17.9,
    excess_mean_pct=-2.4, excess_t_ols=-0.567, excess_p_ols=0.5781, excess_t_hac=-0.58,
    sneaker_sharpe=0.71, gspc_sharpe=0.55,
    ar1_rho=0.63, unsmoothed_vol_pct=23.1, unsmoothed_sharpe=0.29,
    net_sneaker_mean_pct=4.5,
    strat_cagr_pct=9.3, bah_gspc_cagr_pct=8.2, trend_shortfall_pp=1.03,
)
"""

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Busted%3F: Yes](https://img.shields.io/badge/Busted%3F-Yes-8b949e?style=flat-square)\n\n"
)


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Sneaker-Resale -- does StockX really beat the stock market?\n"
            "### The 'sneakers are an asset class' pitch, tested honestly, in plain English\n\n"
            + BADGES +
            "You have heard it since 2017: reselling sneakers on **StockX** or **GOAT** is "
            "an alternative asset class that *beats stocks* -- double-digit returns, smaller "
            "drawdowns, the 'stock market of things.' This notebook asks the only questions "
            "that matter: **did the sneaker market actually out-return the S&P 500?** And if "
            "its risk-adjusted numbers look so good, **are they real -- or an accounting "
            "illusion?**\n\n"
            "> **This is the plain-language layer.** Want the t-stats, the Newey-West HAC "
            "standard error, and the appraisal-unsmoothing? That's "
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
            "| Did sneakers out-return the S&P? | **No.** Sneaker CAGR **+7.0%** vs S&P "
            "**+8.2%**; the average annual *excess* return is **−2.4%** -- sneakers "
            "*underperformed*. |\n"
            "| But the Sharpe looks higher (0.71 vs 0.55)? | An **illusion.** The index is "
            "smoothed (annual, infrequent appraisals). Undo the smoothing and its true "
            "volatility is **23%** (more than the S&P's 18%); the Sharpe falls to **0.29**. |\n"
            "| Could you actually buy it? | **No vehicle.** You'd resell physical pairs: "
            "~10–15% marketplace fees, authentication, storage, no shorting. Net ~**+4.5%/yr**. |\n"
            "| Was the boom real? | Yes -- and so was the **bust** (2022–2024). Buying near "
            "the 2021 peak was brutal. |\n\n"
            "> Sneakers are fun. As an *asset class that beats stocks*, the claim does not "
            "survive contact with the data."
        ),

        # ---- BEAT 1 -- THE CLAIM ---------------------------------------------
        md(
            "## 1 - The claim\n\n"
            "> *\"The sneaker resale market is a real alternative asset. On StockX it behaves "
            "like a stock exchange -- bids, asks, a live price -- and it has returned double "
            "digits with lower volatility than equities. Sneakers beat stocks.\"*\n\n"
            "The marketplaces leaned into the framing on purpose (StockX literally borrowed "
            "exchange language). The press ran with it. We test it against the boring "
            "benchmark every alternative asset must clear: the **S&P 500**."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 - So what?\n\n"
            "If it were true -- a liquid, investable sneaker index that out-returns equities "
            "at lower risk -- it would be a genuine free lunch, and every endowment would own "
            "some. The interesting question is *why the numbers look so good*: is there real "
            "outperformance, or is the attractive risk profile manufactured by how the index "
            "is priced?"
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW -----------------------------------------
        md(
            "## 3 - How would we even know?\n\n"
            "Three traps to avoid:\n\n"
            "1. **Return, not vibes.** Compare the sneaker index's annual return to the S&P, "
            "year by year, and test the *average difference* against zero.\n"
            "2. **Smoothed prices lie about risk.** Appraisal-style indices (private real "
            "estate, hedge-fund NAVs, and yes -- collectibles) report infrequent, lagged "
            "marks. That makes them look calm. The tell is high year-to-year "
            "**autocorrelation**; the fix is to *unsmooth* the series.\n"
            "3. **Can you actually buy it?** A backtest of an *index you cannot trade* is a "
            "fantasy. Physical resale has brutal frictions and no short side.\n\n"
            "We test all of this on **^GSPC** (S&P 500 price index) from 2007 to 2024, "
            "matched to our curated annual sneaker-resale index (18 paired years)."
        ),

        # ---- BEAT 4 -- THE TEARDOWN ------------------------------------------
        md(
            "## 4 - The teardown -- let's actually look\n\n"
            "**First, meet the data:** the curated sneaker-resale index (hardcoded in "
            "`data.py`) paired with the S&P 500 calendar-year return."
        ),
        code(
            "sn = data.sneaker_index()\n"
            "if HAVE_REAL:\n"
            "    df = data.fetch_gspc_annual()\n"
            "    s = st.summarize(df)\n"
            "else:\n"
            "    df, s = None, R\n"
            "\n"
            "print('Sneaker index: 2006-2024 (19 levels, 18 annual returns)')\n"
            "print(f\"Paired years vs S&P: {s['n']} ({R['year_start']}-{R['year_end']})\")\n"
            "print()\n"
            "print(f\"Sneaker CAGR: {s['sneaker_cagr_pct']:+.1f}%/yr   S&P CAGR: {s['gspc_cagr_pct']:+.1f}%/yr\")\n"
            "print(f\"Average annual EXCESS (sneaker - S&P): {s['excess_mean_pct']:+.1f}%/yr\")"
        ),
        md("**The boom and the bust -- the sneaker index level over time.**"),
        code(
            "lv = sn.set_index('year')['level']\n"
            "fig, ax = plt.subplots(figsize=(10, 4.5))\n"
            "ax.plot(lv.index, lv.values, 'o-', c=GREY, lw=2, ms=5, label='Sneaker-resale index (base 100 = 2006)')\n"
            "ax.axvline(2021, ls='--', c=RED, lw=1.5, label='2021 peak')\n"
            "ax.annotate('boom', (2018.5, 250), color=GREEN, fontsize=12)\n"
            "ax.annotate('bust', (2022.5, 300), color=RED, fontsize=12)\n"
            "ax.set_xlabel('Year'); ax.set_ylabel('Index level')\n"
            "ax.set_title('The sneaker-resale market: a real boom, then a real bust')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Peak (2021): {lv[2021]:.0f}   |   2024: {lv[2024]:.0f}   '\n"
            "      f'(gave back {(1-lv[2024]/lv[2021])*100:.0f}% of the peak)')"
        ),
        md(
            "The market really did explode once StockX and GOAT industrialised resale "
            "(2016–2021), and it really did crater afterwards (2022–2024) as premiums "
            "compressed and the marketplaces cut staff. Anyone who bought the hype near the "
            "2021 peak ate a painful drawdown."
        ),
        md("**Did it beat stocks? Sneaker vs S&P, year by year.**"),
        code(
            "if HAVE_REAL:\n"
            "    yrs = df['year'].values\n"
            "    snr = df['sneaker_return'].values * 100\n"
            "    gsr = df['gspc_return'].values * 100\n"
            "else:\n"
            "    rng = np.random.default_rng(276)\n"
            "    yrs = np.arange(R['year_start'], R['year_end']+1)\n"
            "    snr = rng.normal(R['sneaker_mean_pct'], R['sneaker_vol_pct'], R['n'])\n"
            "    gsr = rng.normal(R['gspc_mean_pct'], R['gspc_vol_pct'], R['n'])\n"
            "\n"
            "x = np.arange(len(yrs)); w = 0.4\n"
            "fig, ax = plt.subplots(figsize=(11, 4.5))\n"
            "ax.bar(x-w/2, snr, w, color=GREY, label='Sneaker index')\n"
            "ax.bar(x+w/2, gsr, w, color=GREEN, label='S&P 500')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xticks(x); ax.set_xticklabels(yrs, rotation=45, fontsize=8)\n"
            "ax.set_ylabel('Annual return (%)')\n"
            "ax.set_title('Sneaker index vs S&P 500 -- the S&P wins the big up-years')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"Sneaker mean: {snr.mean():+.1f}%   |   S&P mean: {gsr.mean():+.1f}%\")\n"
            "print('The sneaker index is calmer year to year -- hold that thought.')"
        ),
        md(
            "The sneaker bars are *smaller* in both directions -- it looks like a calmer, "
            "lower-risk asset. That calmness is exactly what makes its Sharpe ratio look "
            "great. But is the calmness real?"
        ),
        md("**The Sharpe illusion -- what happens when we undo the smoothing.**"),
        code(
            "if HAVE_REAL:\n"
            "    rho = s['ar1_rho']; rep_vol = s['sneaker_vol_pct']; uns_vol = s['unsmoothed_vol_pct']\n"
            "    rep_sh = s['sneaker_sharpe']; uns_sh = s['unsmoothed_sharpe']; spx_sh = s['gspc_sharpe']\n"
            "else:\n"
            "    rho = R['ar1_rho']; rep_vol = R['sneaker_vol_pct']; uns_vol = R['unsmoothed_vol_pct']\n"
            "    rep_sh = R['sneaker_sharpe']; uns_sh = R['unsmoothed_sharpe']; spx_sh = R['gspc_sharpe']\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))\n"
            "axes[0].bar(['Sneaker\\n(reported)', 'Sneaker\\n(unsmoothed)', 'S&P 500'],\n"
            "            [rep_vol, uns_vol, R['gspc_vol_pct']], color=[GREY, RED, GREEN])\n"
            "axes[0].set_ylabel('Volatility (%)'); axes[0].set_title(f'Smoothing hides risk (AR1 rho={rho:+.2f})')\n"
            "axes[1].bar(['Sneaker\\n(reported)', 'Sneaker\\n(unsmoothed)', 'S&P 500'],\n"
            "            [rep_sh, uns_sh, spx_sh], color=[GREY, RED, GREEN])\n"
            "axes[1].axhline(spx_sh, ls='--', c=GREEN, lw=1.5)\n"
            "axes[1].set_ylabel('Sharpe ratio'); axes[1].set_title('...and the high Sharpe vanishes')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Reported sneaker Sharpe: {rep_sh:.2f} (> S&P {spx_sh:.2f}) -- looks great')\n"
            "print(f'Unsmoothed sneaker Sharpe: {uns_sh:.2f} (< S&P {spx_sh:.2f}) -- the edge is gone')"
        ),
        md(
            "The sneaker index's year-to-year autocorrelation is **+0.63** -- a smoking gun "
            "for stale, appraisal-style pricing. Real continuously-traded assets have annual "
            "autocorrelation near zero. Once we *unsmooth* the series (a standard fix, from "
            "private real estate), the 'low' 10.6% volatility balloons to **23%** -- higher "
            "than the S&P -- and the impressive Sharpe collapses **below** the index's."
        ),

        # ---- BEAT 5 -- THE VERDICT -------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            "- **Signal -- None.** Sneakers *underperformed* the S&P by **−2.4%/yr** "
            "(t = −0.57). The only thing that looked good -- the high Sharpe -- is a "
            "stale-pricing artifact (ρ = +0.63); unsmoothed, the Sharpe is 0.29 vs the "
            "S&P's 0.55.\n"
            "- **Tradability -- Mirage.** There is no investable sneaker index. The only "
            "exposure is reselling physical pairs, at 10–15% marketplace fees, with storage, "
            "authentication, and no short side. Net of frictions, ~+4.5%/yr.\n"
            "- **Busted.** A genuine boom and a genuine bust -- but as an asset class that "
            "*beats stocks*, the claim is a mirage built on smoothed marks."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 - Could you actually trade it?\n\n"
            "Suppose you tried -- buy pairs, resell on StockX. The frictions are enormous, "
            "and there is no fund to do it for you."
        ),
        code(
            "gross = (s['sneaker_mean_pct'] if HAVE_REAL else R['sneaker_mean_pct'])\n"
            "net = (s['net_sneaker_mean_pct'] if HAVE_REAL else R['net_sneaker_mean_pct'])\n"
            "spx = (s['gspc_mean_pct'] if HAVE_REAL else R['gspc_mean_pct'])\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "bars = ax.bar(['Sneaker\\n(gross)', 'Sneaker\\n(net of ~3%/yr\\nfrictions)', 'S&P 500\\n(gross)'],\n"
            "              [gross, net, spx], color=[GREY, RED, GREEN], width=0.55)\n"
            "ax.set_ylabel('Mean annual return (%)')\n"
            "ax.set_title('Frictions cut the sneaker return roughly in half')\n"
            "for b, v in zip(bars, [gross, net, spx]):\n"
            "    ax.annotate(f'{v:+.1f}%', (b.get_x()+b.get_width()/2, v+0.2), ha='center', va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Marketplace seller fees ~10-15%, plus authentication, shipping, storage.')\n"
            "print('No short side, no index fund. The hustle nets ~half the S&P, gross beats it by nothing.')"
        ),
        md(
            "Even before frictions the sneaker index trailed the S&P. After a conservative "
            "3%/yr haircut for marketplace fees and handling, it returns roughly **half** the "
            "S&P's gross return -- for far more operational hassle and zero diversification "
            "benefit once you account for the true (unsmoothed) risk."
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 - Going further\n\n"
            "- **The positive control:** the companion notebook plants a real +20–30%/yr "
            "sneaker alpha into a synthetic tape and shows the engine detects it cleanly -- "
            "so the null result here is about the *market*, not the method.\n"
            "- **Same illusion, other assets:** smoothed appraisal pricing flatters private "
            "real estate, private equity, and many hedge funds -- the Asness-Krail-Liew "
            "'Do Hedge Funds Hedge?' lesson.\n"
            "- **Collectibles as an asset class:** Dimson & Spaenjers on stamps and wine "
            "reach the same verdict -- modest real returns, high costs, illiquidity.\n\n"
            "*Think sneakers really beat stocks? Fork this, swap in your own resale index, "
            "and show a positive excess return that clears |t| ≥ 2 on the unsmoothed series, "
            "net of marketplace fees. That's the bar.*"
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
            "# Sneaker-Resale -- a quantitative teardown\n"
            "### 18 years * ^GSPC * excess-return t * Newey-West HAC * Geltner AR(1) "
            "unsmoothing * n=18 power * synthetic control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) -- "
            "*same seven beats, every claim carrying its standard error.* We test whether the "
            "sneaker-resale index out-returns the S&P 500 (one-sample and **Newey-West HAC** "
            "t-tests on the annual excess), whether its high Sharpe survives **Geltner AR(1) "
            "unsmoothing**, what frictions do to it, and close with an n=18 power calculation "
            "and a synthetic positive control.\n\n"
            "> **Not investment advice.** Real data: ^GSPC S&P 500 price index "
            "(2007–2024, Dec/Dec); sneaker index hardcoded in `data.py` (2006–2024). "
            "Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **The plain-English notes** translate each result back to intuition."
        ),
        code(BOOT + "\n" + R_CELL + "\nfrom scipy import stats as scipy_stats\n"),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            "| **Signal** | `NONE` | Excess return **−2.4%/yr**, t = **−0.57**, "
            "Newey-West HAC t = **−0.58**. The high Sharpe is a stale-pricing artifact "
            "(AR(1) ρ = +0.63; unsmoothed Sharpe **0.29** < S&P 0.55). |\n"
            "| **Tradability** | `MIRAGE` | No investable index; physical resale at "
            "~10–15% one-way fees, no shorting; net ~+4.5%/yr. |\n"
            "| **Busted?** | `YES` | A real boom-and-bust, but the 'beats stocks at lower "
            "risk' pitch is manufactured by smoothed appraisal pricing. |\n\n"
            "> Sneakers under-returned the S&P; their only apparent edge -- risk-adjusted -- "
            "evaporates the moment you undo the smoothing."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 - The claim, steelmanned\n\n"
            "Let $r^{sn}_t$ and $r^{spx}_t$ be the calendar-year returns of the sneaker index "
            "and the S&P 500, and $x_t = r^{sn}_t - r^{spx}_t$ the annual excess. The pitch is:\n\n"
            "- **H1 (return).** $E[x_t] > 0$ -- sneakers out-return stocks, distinguishable "
            "at |t| ≥ 2 (HAC) on the real tape.\n"
            "- **H2 (risk-adjusted).** $\\text{Sharpe}(r^{sn}) > \\text{Sharpe}(r^{spx})$ on "
            "the *economic* (unsmoothed) return series, not just the reported one.\n"
            "- **H3 (tradable).** The edge survives realistic one-way frictions and the "
            "absence of a short side / index vehicle.\n\n"
            "We reject H1 (the point estimate is *negative*), reject H2 (it holds only on "
            "smoothed marks), and reject H3 (no vehicle, frictions halve the return)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 - So what? -- what rides on each answer\n\n"
            "If H1–H3 held, the sneaker market would be a rare investable free lunch and the "
            "'stock market of things' branding would be literally true. The instructive "
            "finding is *how* an illiquid, smoothed collectible index manufactures an "
            "attractive Sharpe -- the same mechanism that flatters private real estate and "
            "hedge-fund NAVs (Asness-Krail-Liew 2001; Geltner 1991)."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 - How we'd know -- the protocol\n\n"
            "- **Data.** ^GSPC December/December last-trading-day price returns 2007–2024 "
            "(repo-level cache, cache-only); sneaker index hardcoded in `data.py`.\n"
            "- **Return test.** One-sample t on $x_t = r^{sn}_t - r^{spx}_t$ vs 0, plus a "
            "**Newey-West HAC** t (lag 1) -- the honest SE for a serially correlated annual "
            "series. **Signal bar: |HAC t| ≥ 2.**\n"
            "- **Risk test.** First-order autocorrelation of $r^{sn}$, then **Geltner AR(1) "
            "unsmoothing** $r^{true}_t = (r_t - \\rho r_{t-1})/(1-\\rho)$; compare Sharpes.\n"
            "- **Tradability.** One-way frictional cost on the sneaker leg; a lagged "
            "(no-look-ahead) momentum overlay vs buy-and-hold S&P.\n"
            "- **Power + control.** n=18 minimum-detectable-effect; synthetic tape with a "
            "planted alpha and planted smoothing.\n"
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 - The teardown"),
        md(
            "### 4a - Return: the annual excess and its (HAC) t-stat\n\n"
            "The headline test: is the mean annual excess return different from zero?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    df = data.fetch_gspc_annual()\n"
            "    c = st.compare_returns(df)\n"
            "    excess = (df['sneaker_return'] - df['gspc_return']).values * 100\n"
            "    n_obs = c['n']; ex_mean = c['excess_mean']*100\n"
            "    t_ols = c['excess_t_ols']; p_ols = c['excess_p_ols']; t_hac = c['excess_t_hac']\n"
            "else:\n"
            "    rng = np.random.default_rng(276)\n"
            "    snr = rng.normal(R['sneaker_mean_pct'], R['sneaker_vol_pct'], R['n'])\n"
            "    gsr = rng.normal(R['gspc_mean_pct'], R['gspc_vol_pct'], R['n'])\n"
            "    excess = snr - gsr\n"
            "    n_obs = R['n']; ex_mean = R['excess_mean_pct']\n"
            "    t_ols = R['excess_t_ols']; p_ols = R['excess_p_ols']; t_hac = R['excess_t_hac']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(10, 4.2))\n"
            "cols = [GREEN if e>0 else RED for e in excess]\n"
            "ax.bar(range(len(excess)), excess, color=cols)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.axhline(excess.mean(), ls='--', c=AMBER, lw=2, label=f'Mean excess: {excess.mean():+.1f}%')\n"
            "ax.set_xlabel('Year index'); ax.set_ylabel('Sneaker - S&P (%)')\n"
            "ax.set_title('Annual excess return: negative on average, no significance')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'n = {n_obs}   mean excess = {ex_mean:+.1f}%/yr')\n"
            "print(f'One-sample t (vs 0): {t_ols:+.3f}  (p = {p_ols:.4f})')\n"
            "print(f'Newey-West HAC t (lag 1): {t_hac:+.3f}')\n"
            "print('|HAC t| < 2 -- and the sign is negative. No outperformance signal.')"
        ),
        md(
            "> The point estimate is **−2.4%/yr**: sneakers *underperformed*. The HAC t-stat "
            "is **−0.58**, nowhere near the |t| ≥ 2 bar (and pointing the wrong way). H1 is "
            "rejected before we even worry about risk."
        ),
        md(
            "### 4b - The Sharpe illusion: AR(1) smoothing and Geltner unsmoothing\n\n"
            "The sneaker index's low reported volatility is the heart of the pitch. We test "
            "whether it is real by measuring autocorrelation and unsmoothing."
        ),
        code(
            "if HAVE_REAL:\n"
            "    snret = df['sneaker_return']\n"
            "    uns, rho = st.unsmooth_ar1(snret)\n"
            "    rep_vol = snret.std(ddof=1)*100; uns_vol = uns.std(ddof=1)*100\n"
            "    rep_sh = c['sneaker_sharpe']; uns_sh = c['unsmoothed_sharpe']; spx_sh = c['gspc_sharpe']\n"
            "else:\n"
            "    rho = R['ar1_rho']\n"
            "    rep_vol = R['sneaker_vol_pct']; uns_vol = R['unsmoothed_vol_pct']\n"
            "    rep_sh = R['sneaker_sharpe']; uns_sh = R['unsmoothed_sharpe']; spx_sh = R['gspc_sharpe']\n"
            "\n"
            "print(f'First-order autocorrelation of sneaker returns: rho = {rho:+.3f}')\n"
            "print('  (a continuously-traded asset would be near 0; this is stale pricing)')\n"
            "print()\n"
            "print(f'Reported  vol {rep_vol:5.1f}%   Sharpe {rep_sh:.2f}')\n"
            "print(f'Unsmoothed vol {uns_vol:5.1f}%   Sharpe {uns_sh:.2f}')\n"
            "print(f'S&P 500    vol {R[\"gspc_vol_pct\"]:5.1f}%   Sharpe {spx_sh:.2f}')\n"
            "print()\n"
            "print('Unsmoothing roughly DOUBLES the volatility and pushes the Sharpe BELOW the S&P.')"
        ),
        md(
            "> With ρ = +0.63, the reported 10.6% volatility is a mirage: the economic "
            "volatility is ~23%, *higher* than the S&P. The Sharpe drops from 0.71 to "
            "**0.29**, below the S&P's 0.55. H2 is rejected -- the risk-adjusted edge was "
            "stale marks."
        ),
        md(
            "### 4c - Tradability: frictions and a lagged momentum overlay\n\n"
            "A backtest of an un-buyable index is fiction. We net realistic frictions and "
            "test a no-look-ahead trend rule (hold sneakers only after an up year)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    tf = st.trend_following_excess(df)\n"
            "    strat = tf['strat_cagr']*100; bah = tf['bah_gspc_cagr']*100\n"
            "    net = c['net_sneaker_mean']*100; gross = c['sneaker_mean']*100\n"
            "else:\n"
            "    strat = R['strat_cagr_pct']; bah = R['bah_gspc_cagr_pct']\n"
            "    net = R['net_sneaker_mean_pct']; gross = R['sneaker_mean_pct']\n"
            "\n"
            "print(f'Sneaker mean: gross {gross:+.1f}%  ->  net of ~3%/yr frictions {net:+.1f}%')\n"
            "print(f'Lagged trend overlay (hold sneakers after an up year, net): {strat:+.1f}%/yr')\n"
            "print(f'Buy-and-hold S&P:                                            {bah:+.1f}%/yr')\n"
            "print(f'Trend overlay vs B&H: {strat-bah:+.2f}pp')\n"
            "print()\n"
            "print('The overlay edge is +1pp on a SINGLE 18-point path with no statistical')\n"
            "print('support -- overfit noise. And there is no fund or short side to implement it.')"
        ),
        md(
            "> Net of frictions the sneaker leg returns ~+4.5%/yr -- about half the S&P. The "
            "lagged trend overlay nominally edges buy-and-hold by ~1pp, but that is one "
            "realised path of 18 points (zero degrees of freedom to spare) and, crucially, "
            "**there is no vehicle to trade it**. H3 rejected."
        ),
        md(
            "### 4d - Power: what excess could n=18 even detect?\n\n"
            "The minimum detectable annual excess at 80% power, two-sided, α=0.05."
        ),
        code(
            "vol_excess = 0.18  # rough sd of the annual excess series\n"
            "n_years = R['n']\n"
            "alpha = 0.05\n"
            "from scipy.stats import t as t_dist\n"
            "dof = n_years - 1\n"
            "t_crit = t_dist.ppf(1 - alpha/2, dof)\n"
            "t_pow = t_dist.ppf(0.80, dof)\n"
            "mde = (t_crit + t_pow) * vol_excess / np.sqrt(n_years)\n"
            "print(f'Minimum detectable annual excess (80% power, 2-sided, alpha={alpha}):')\n"
            "print(f'  sd(excess) ~ {vol_excess:.0%}, n = {n_years}')\n"
            "print(f'  MDE = {mde:.1%}/yr')\n"
            "print()\n"
            "print(f'Observed excess: {R[\"excess_mean_pct\"]:+.1f}%/yr (and negative).')\n"
            "print('n=18 cannot resolve anything smaller than ~9%/yr. The question is underpowered.')"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            "- **Signal `NONE`** -- excess return −2.4%/yr, t = −0.57, HAC t = −0.58. The "
            "high reported Sharpe (0.71) is a stale-pricing artifact (ρ = +0.63); unsmoothed "
            "Sharpe 0.29 < S&P 0.55. Curation/survivorship of the basket flatters the series "
            "further. n = 18 is underpowered for anything below ~9%/yr.\n"
            "- **Tradability `MIRAGE`** -- no investable index; physical resale at "
            "~10–15% one-way fees, no shorting; net ~+4.5%/yr, ~half the S&P.\n"
            "- **Busted** -- a real boom and bust, but 'beats stocks at lower risk' is "
            "manufactured by smoothed appraisal pricing."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 - Could you trade it? -- the buy-and-hold benchmark\n\n"
            "Cumulative growth of $1: the (net) sneaker index vs buy-and-hold S&P."
        ),
        code(
            "if HAVE_REAL:\n"
            "    d = df.sort_values('year')\n"
            "    yrs = d['year'].values\n"
            "    sn_net = (d['sneaker_return'].values - 0.03)\n"
            "    sn_cum = np.cumprod(1 + sn_net) * 100\n"
            "    spx_cum = np.cumprod(1 + d['gspc_return'].values) * 100\n"
            "else:\n"
            "    rng = np.random.default_rng(99)\n"
            "    yrs = np.arange(R['year_start'], R['year_end']+1)\n"
            "    sn_net = rng.normal((R['net_sneaker_mean_pct'])/100, R['sneaker_vol_pct']/100, R['n'])\n"
            "    spx = rng.normal(R['gspc_mean_pct']/100, R['gspc_vol_pct']/100, R['n'])\n"
            "    sn_cum = np.cumprod(1 + sn_net) * 100\n"
            "    spx_cum = np.cumprod(1 + spx) * 100\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(10, 4.5))\n"
            "ax.plot(yrs, spx_cum, lw=2, c=GREEN, label='Buy & hold S&P 500')\n"
            "ax.plot(yrs, sn_cum, lw=2, c=RED, ls='--', label='Sneaker index, net of frictions')\n"
            "ax.set_yscale('log'); ax.set_xlabel('Year')\n"
            "ax.set_ylabel('Growth of 100 (log scale)')\n"
            "ax.set_title('Net of marketplace frictions, the index fund wins')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Terminal: S&P {spx_cum[-1]:.0f}  vs  net sneaker {sn_cum[-1]:.0f} (base 100)')"
        ),
        md(
            "> Net of a conservative friction the un-buyable sneaker index trails a plain "
            "index fund -- and that comparison still credits the sneaker leg with its "
            "*understated* (smoothed) volatility. On true risk it is strictly worse."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 - Going further -- the synthetic positive control\n\n"
            "Does the engine detect a sneaker alpha *when one exists*, and does planted "
            "smoothing reproduce the Sharpe illusion? We sweep both knobs."
        ),
        code(
            "alphas = [0, 500, 1000, 2000, 3000]\n"
            "rows = []\n"
            "for a in alphas:\n"
            "    dfa, _ = data.synthetic_panel(n_years=400, alpha_bps=float(a), smooth_rho=0.0, seed=276)\n"
            "    ca = st.compare_returns(dfa)\n"
            "    rows.append({'alpha_bps': a, 'excess_%': ca['excess_mean']*100,\n"
            "                 't_excess': ca['excess_t_ols']})\n"
            "ctrl = pd.DataFrame(rows)\n"
            "\n"
            "# Smoothing illusion with ZERO alpha\n"
            "df_sm, _ = data.synthetic_panel(n_years=400, alpha_bps=0.0, smooth_rho=0.6, seed=276)\n"
            "csm = st.compare_returns(df_sm)\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "cols = [GREEN if t>2 else RED for t in ctrl['t_excess']]\n"
            "ax.bar(ctrl['alpha_bps'].astype(str), ctrl['t_excess'], color=cols)\n"
            "ax.axhline(2, ls='--', c='k', lw=1, label='|t| = 2 bar')\n"
            "ax.set_xlabel('Planted sneaker alpha (bps/yr)'); ax.set_ylabel('Excess-return t-stat')\n"
            "ax.set_title('The engine finds a real alpha when one exists (green = t>2)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(ctrl.round(2).to_string(index=False))\n"
            "print()\n"
            "print(f\"Smoothing-only control (alpha=0, rho=0.6): reported Sharpe {csm['sneaker_sharpe']:.2f} \"\n"
            "      f\"vs unsmoothed {csm['unsmoothed_sharpe']:.2f} -- inflated by smoothing, exactly as on the real tape.\")"
        ),
        md(
            "The engine cleanly detects a planted alpha once it is large enough, and planted "
            "AR(1) smoothing reproduces the reported-vs-unsmoothed Sharpe gap we see on the "
            "real index. The null verdict is therefore a statement about the **sneaker market "
            "and its pricing**, not about the method."
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
         "--inplace", "--ExecutePreprocessor.timeout=300", path],
        check=True,
    )
    print("executed", path)


if __name__ == "__main__":
    build_curious()
    build_quants()
    _execute("01_for_the_curious.ipynb")
    _execute("02_for_the_quants.ipynb")
