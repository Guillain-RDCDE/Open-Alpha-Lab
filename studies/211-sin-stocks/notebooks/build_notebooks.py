"""Generate the two narrative notebooks for Study 211 (Sin-Stocks).

    python notebooks/build_notebooks.py

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the cached
parquets under ../_cache/ if present and otherwise quote the frozen headline numbers in
``R`` (mirroring docs/results.md), so the notebook re-runs for any reader.

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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-12).
R = dict(
    n=4589,
    start="2008-03-18",
    end="2026-06-12",
    fingerprint="9ce93c8cb5e4",
    # SIN_EW gross
    sin_cagr=10.60, sin_vol=20.9, sin_sharpe=0.481, sin_maxdd=-65.9,
    sin_t=2.06, sin_total_ret=527,
    # SPY gross
    spy_cagr=12.17, spy_vol=19.8, spy_sharpe=0.579, spy_maxdd=-51.5,
    spy_t=2.82, spy_total_ret=710,
    # DSI gross
    dsi_cagr=12.11, dsi_vol=19.8, dsi_sharpe=0.577, dsi_maxdd=-49.9,
    dsi_t=2.80, dsi_total_ret=702,
    # Excess return vs SPY
    excess_bps=-0.558, excess_cagr=-1.40, excess_t=-0.46,
    excess_sharpe_ci_lo=-0.539, excess_sharpe_ci_hi=0.339, excess_frac_neg=68,
    # OLS vs SPY
    ols_alpha_bps=0.242, ols_alpha_ann=0.61, ols_t_alpha=0.20,
    ols_beta=0.8245, ols_r2=0.6088,
    # OLS vs DSI
    ols_dsi_alpha_bps=0.444, ols_dsi_alpha_ann=1.13, ols_dsi_t_alpha=0.34,
    ols_dsi_beta=0.7838,
    # Per-ticker
    mo_cagr=13.64, mo_sharpe=0.59, mo_maxdd=-53.7,
    pm_cagr=12.57, pm_sharpe=0.50, pm_maxdd=-42.9,
    stz_cagr=13.35, stz_sharpe=0.42, stz_maxdd=-53.5,
    lvs_cagr=0.62, lvs_sharpe=0.01, lvs_maxdd=-98.3, lvs_vol=58.5,
    lmt_cagr=13.09, lmt_sharpe=0.52, lmt_maxdd=-50.6,
    rtx_cagr=10.95, rtx_sharpe=0.39, rtx_maxdd=-52.0,
    # Crash episodes
    c08_dd=-51.5, c08_spy=12.3, c08_sin=23.9, c08_prot=11.7,
    c11_dd=-18.6, c11_spy=13.2, c11_sin=17.4, c11_prot=4.3,
    c15_dd=-13.0, c15_spy=12.2, c15_sin=23.1, c15_prot=10.9,
    c18a_dd=-10.1, c18a_spy=11.4, c18a_sin=-2.2, c18a_prot=-13.6,
    c18b_dd=-19.3, c18b_spy=12.5, c18b_sin=11.7, c18b_prot=-0.9,
    c20_dd=-33.7, c20_spy=14.0, c20_sin=-5.4, c20_prot=-19.4,
    c22_dd=-24.5, c22_spy=12.5, c22_sin=4.3, c22_prot=-8.3,
    c25_dd=-18.8, c25_spy=11.6, c25_sin=4.0, c25_prot=-7.7,
)

# ---------------------------------------------------------------------------
# Shared preamble
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

from sin_stocks import data, strategy as st

CACHE = os.path.abspath(os.path.join("..", "_cache"))
ALL_TICKERS = data.ALL_TICKERS

def _have_cache():
    return all(os.path.exists(data._cache_path(t, CACHE)) for t in ALL_TICKERS)

HAVE_REAL = _have_cache()
print("real price cache present:", HAVE_REAL)

# Frozen results (mirror of docs/results.md)
R = dict(
    n=4589, start="2008-03-18", end="2026-06-12",
    sin_cagr=10.60, sin_vol=20.9, sin_sharpe=0.481, sin_maxdd=-65.9,
    sin_t=2.06, sin_total_ret=527,
    spy_cagr=12.17, spy_vol=19.8, spy_sharpe=0.579, spy_maxdd=-51.5,
    spy_t=2.82, spy_total_ret=710,
    dsi_cagr=12.11, dsi_vol=19.8, dsi_sharpe=0.577, dsi_maxdd=-49.9,
    dsi_t=2.80, dsi_total_ret=702,
    excess_bps=-0.558, excess_cagr=-1.40, excess_t=-0.46,
    excess_sharpe_ci_lo=-0.539, excess_sharpe_ci_hi=0.339, excess_frac_neg=68,
    ols_alpha_bps=0.242, ols_alpha_ann=0.61, ols_t_alpha=0.20,
    ols_beta=0.8245, ols_r2=0.6088,
    ols_dsi_alpha_bps=0.444, ols_dsi_alpha_ann=1.13, ols_dsi_t_alpha=0.34, ols_dsi_beta=0.7838,
    mo_cagr=13.64, mo_sharpe=0.59, pm_cagr=12.57, pm_sharpe=0.50,
    stz_cagr=13.35, stz_sharpe=0.42,
    lvs_cagr=0.62, lvs_sharpe=0.01, lvs_maxdd=-98.3, lvs_vol=58.5,
    lmt_cagr=13.09, lmt_sharpe=0.52, rtx_cagr=10.95, rtx_sharpe=0.39,
    c08_prot=11.7, c11_prot=4.3, c15_prot=10.9,
    c18a_prot=-13.6, c18b_prot=-0.9, c20_prot=-19.4, c22_prot=-8.3, c25_prot=-7.7,
)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Sin-Stocks — do vice stocks really beat the virtuous market?\n"
            "### Tobacco, alcohol, gambling, defense vs SPY and the ESG counter-portfolio\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Neglect_premium: Mixed](https://img.shields.io/badge/Neglect_premium%3F-Mixed-8b949e?style=flat-square)\n\n"
            "The \"sin stocks\" thesis is elegant in its perversity: investors and institutions that "
            "avoid tobacco, alcohol, gambling, and defense on moral grounds create a neglect premium. "
            "The stocks trade cheap because nobody wants to own them, and patient, values-agnostic "
            "investors collect the premium.  Hong & Kacperczyk (2009) documented exactly this: "
            "sin stocks outperformed by ~2.5%/yr in the US from 1965–2006.  The question is whether "
            "it still works in the ESG era — with >$40 trillion of capital explicitly screening out "
            "these stocks.\n\n"
            "> 📓 **Plain-language layer.** For the HAC t-stats, OLS alpha decomposition, and "
            "bootstrap intervals, see **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** Research tool only. "
            "House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Did the sin basket beat the S&P 500? | **No.** SIN_EW returned **{R['sin_total_ret']}%** "
            f"vs SPY's **{R['spy_total_ret']}%** since 2008. |\n"
            f"| Is risk-adjusted return (Sharpe) better? | **No.** SIN Sharpe **{R['sin_sharpe']:.3f}** "
            f"vs SPY **{R['spy_sharpe']:.3f}**. |\n"
            "| Did it beat the ESG counter-portfolio? | **No.** DSI returned 702% vs sin's 527%. |\n"
            "| Is the neglect premium still alive? | **No** — at least not robustly in 2008–2026. "
            "t = −0.46 on the excess return. |\n\n"
            "> The tobacco and defense tickers individually did fine (+10–14%/yr CAGR). "
            "The basket was wrecked by Las Vegas Sands (LVS): +0.62%/yr, vol 58.5%, max drawdown −98.3%."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Vice pays a premium. Tobacco companies, alcohol distributors, casino operators, "
            "and defense contractors are shunned by ESG mandates, endowments, and socially responsible "
            "funds — which means they trade cheap and their future returns must compensate investors "
            "willing to hold them. Buy sin stocks, collect the neglect premium, let the virtuous "
            "underperform.\"*\n\n"
            "This is the Hong & Kacperczyk (2009) hypothesis — empirically documented over 1965–2006 "
            "in the US, approximately +2.5%/yr excess return after controlling for risk factors.  The "
            "question is whether institutional exclusion has grown large enough to maintain the premium "
            "or whether it has been competed away."
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If true, sin stocks offer an ethically unconstrained investor a systematic edge — a "
            "premium paid by the moral majority that keeps the stocks undervalued.  The mechanism "
            "is clean: restricted demand + unchanged fundamentals = lower price = higher expected "
            "return.  The ESG boom of the 2010s–2020s should, if anything, have *amplified* this "
            "premium by increasing the population of mandated excluders.\n\n"
            "If false, either: (a) the premium was arbitraged away by hedge funds and unconstrained "
            "capital after the 2009 publication, (b) it was a sector/value-factor tilt that decayed "
            "when growth stocks dominated, or (c) the casino stock (LVS) simply destroyed value "
            "independently of the neglect story.  One honest backtest can tell us which."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "We build an equal-weight basket of six classic sin tickers:\n\n"
            "| Ticker | Company | Sin category |\n|---|---|---|\n"
            "| MO | Altria | Tobacco (US domestic) |\n"
            "| PM | Philip Morris International | Tobacco (international) |\n"
            "| STZ | Constellation Brands | Alcohol / beer |\n"
            "| LVS | Las Vegas Sands | Gambling / casinos |\n"
            "| LMT | Lockheed Martin | Defense |\n"
            "| RTX | RTX Corp | Defense |\n\n"
            "Then compare against:\n"
            "- **SPY** — S&P 500 ETF (the market)\n"
            "- **DSI** — iShares MSCI KLD 400 Social ETF (the explicit ESG counter-portfolio)\n\n"
            "Common history: PM's IPO on 2008-03-17 sets the start date.  All prices are "
            "total-return (dividends included) — critical for tobacco and defense stocks.  "
            "We measure CAGR, Sharpe, max drawdown, per-ticker attribution, and crash episodes."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md("## 4 · The teardown\n\n**First: cumulative wealth — $1 invested in March 2008.**"),
        code(
            "if HAVE_REAL:\n"
            "    all_prices = data.load_aligned(fetch=False, cache_dir=CACHE)\n"
            "    basket = all_prices[data.SIN_BASKET]\n"
            "    ew_ret = st.build_equal_weight(basket)\n"
            "    spy_ret = np.log(all_prices['SPY'] / all_prices['SPY'].shift(1)).dropna()\n"
            "    dsi_ret = np.log(all_prices['DSI'] / all_prices['DSI'].shift(1)).dropna()\n"
            "    idx = ew_ret.index.intersection(spy_ret.index).intersection(dsi_ret.index)\n"
            "    sin_nav = pd.Series(np.exp(np.cumsum(ew_ret.loc[idx])), index=idx)\n"
            "    spy_nav = pd.Series(np.exp(np.cumsum(spy_ret.loc[idx])), index=idx)\n"
            "    dsi_nav = pd.Series(np.exp(np.cumsum(dsi_ret.loc[idx])), index=idx)\n"
            "else:\n"
            "    dates = pd.date_range(R['start'], periods=R['n'], freq='B')\n"
            "    sin_nav = pd.Series((1 + R['sin_cagr']/100) ** (np.arange(R['n'])/252), index=dates)\n"
            "    spy_nav = pd.Series((1 + R['spy_cagr']/100) ** (np.arange(R['n'])/252), index=dates)\n"
            "    dsi_nav = pd.Series((1 + R['dsi_cagr']/100) ** (np.arange(R['n'])/252), index=dates)\n"
            "fig, ax = plt.subplots(figsize=(10, 5))\n"
            "ax.plot(sin_nav.index, sin_nav.values, color=AMBER, lw=2, label='SIN_EW (sin basket)')\n"
            "ax.plot(spy_nav.index, spy_nav.values, color=GREY,  lw=2, label='SPY (S&P 500)')\n"
            "ax.plot(dsi_nav.index, dsi_nav.values, color=GREEN, lw=2, ls='--', label='DSI (ESG)')\n"
            "ax.set_ylabel('Growth of $1 (total return, log scale)')\n"
            "ax.set_yscale('log')\n"
            "ax.set_title('Cumulative wealth: sin basket vs S&P 500 and ESG, 2008–2026')\n"
            "ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Total return: SIN={R[\"sin_total_ret\"]}%  SPY={R[\"spy_total_ret\"]}%  DSI={R[\"dsi_total_ret\"]}%')\n"
            "print(f'CAGR:         SIN={R[\"sin_cagr\"]:+.2f}%  SPY={R[\"spy_cagr\"]:+.2f}%  DSI={R[\"dsi_cagr\"]:+.2f}%')"
        ),
        md(
            f"SIN turned $1 into **\\${1 + R['sin_total_ret']/100:.2f}** vs SPY's "
            f"**\\${1 + R['spy_total_ret']/100:.2f}** and DSI's "
            f"**\\${1 + R['dsi_total_ret']/100:.2f}** — the sin basket finished last.  "
            "Far from earning the neglect premium, it surrendered 183 percentage points to the market."
        ),

        md("**Second: the basket is not homogeneous — per-ticker breakdown.**"),
        code(
            "if HAVE_REAL:\n"
            "    attr = st.sector_attribution(all_prices, basket=data.SIN_BASKET)\n"
            "    cagrs = attr['cagr_pct'].tolist()\n"
            "    sharpes = attr['sharpe_ann'].tolist()\n"
            "    labels = attr.index.tolist()\n"
            "else:\n"
            "    labels = ['MO','PM','STZ','LVS','LMT','RTX']\n"
            "    cagrs  = [R['mo_cagr'], R['pm_cagr'], R['stz_cagr'], R['lvs_cagr'], R['lmt_cagr'], R['rtx_cagr']]\n"
            "    sharpes= [R['mo_sharpe'],R['pm_sharpe'],R['stz_sharpe'],R['lvs_sharpe'],R['lmt_sharpe'],R['rtx_sharpe']]\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))\n"
            "colors = [AMBER if c > R['spy_cagr'] else (RED if c < 5 else GREY) for c in cagrs]\n"
            "axes[0].barh(labels, cagrs, color=colors)\n"
            "axes[0].axvline(R['spy_cagr'], c='k', lw=1.5, ls='--', label=f\"SPY {R['spy_cagr']:.1f}%\")\n"
            "axes[0].set_xlabel('CAGR (%/yr)'); axes[0].set_title('Per-ticker CAGR')\n"
            "axes[0].legend()\n"
            "colors2 = [AMBER if s > R['spy_sharpe'] else (RED if s < 0.1 else GREY) for s in sharpes]\n"
            "axes[1].barh(labels, sharpes, color=colors2)\n"
            "axes[1].axvline(R['spy_sharpe'], c='k', lw=1.5, ls='--', label=f\"SPY {R['spy_sharpe']:.3f}\")\n"
            "axes[1].set_xlabel('Sharpe (gross, rf=0)'); axes[1].set_title('Per-ticker Sharpe')\n"
            "axes[1].legend()\n"
            "plt.tight_layout(); plt.show()\n"
            f"print('LVS (gambling): CAGR={R['lvs_cagr']:+.2f}% vol={R['lvs_vol']}% maxDD={R['lvs_maxdd']}%')\n"
            f"print('Tobacco (MO+PM avg): CAGR~{(R['mo_cagr']+R['pm_cagr'])/2:.1f}%  Defense (LMT+RTX avg): CAGR~{(R['lmt_cagr']+R['rtx_cagr'])/2:.1f}%')"
        ),
        md(
            f"**LVS is the catastrophic outlier**: CAGR +{R['lvs_cagr']:.2f}%, vol {R['lvs_vol']:.1f}%, "
            f"max drawdown {R['lvs_maxdd']:.1f}%. The casino business was nearly wiped out by "
            "COVID (2020 shutdowns + Macau regulatory risk in 2021).  Tobacco (MO, PM: ~+13%/yr) "
            "and defense (LMT, LMT: ~+12%/yr) are solid value/income compounders — just not "
            "notably better than buying the S&P 500."
        ),

        md("**Third: crash behavior — does sin protect in downturns?**"),
        code(
            "if HAVE_REAL:\n"
            "    all_ret_df = pd.DataFrame({'SIN_EW': ew_ret.loc[idx], 'SPY': spy_ret.loc[idx]})\n"
            "    crashes = st.crash_analysis(all_ret_df, asset='SIN_EW', benchmark='SPY', threshold=-0.10)\n"
            "    prot = crashes['protection_pp'].tolist()\n"
            "    ep_labels = [str(c.date())[:7] for c in crashes['start']]\n"
            "else:\n"
            "    prot = [R['c08_prot'],R['c11_prot'],R['c15_prot'],\n"
            "            R['c18a_prot'],R['c18b_prot'],R['c20_prot'],R['c22_prot'],R['c25_prot']]\n"
            "    ep_labels = ['2008-06','2011-08','2015-08','2018-02','2018-12','2020-02','2022-02','2025-03']\n"
            "colors = [GREEN if p > 0 else RED for p in prot]\n"
            "fig, ax = plt.subplots(figsize=(10, 4.5))\n"
            "ax.bar(ep_labels, prot, color=colors)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('SIN_EW protection vs SPY (pp)')\n"
            "ax.set_title('Crash shield test: mixed — works in recessions, fails in growth-led selloffs')\n"
            "for i, (l, p) in enumerate(zip(ep_labels, prot)):\n"
            "    ax.annotate(f'{p:+.1f}', (i, p), ha='center',\n"
            "                va='top' if p < 0 else 'bottom', fontsize=9)\n"
            "plt.xticks(rotation=30, ha='right')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Green = sin basket held up better; red = sin basket fell more')"
        ),
        md(
            "The crash protection record is **mixed**: sin stocks protected in the GFC era "
            f"(+{R['c08_prot']:.1f} pp in 2008, +{R['c15_prot']:.1f} pp in 2015–16) but "
            f"failed badly in COVID ({R['c20_prot']:+.1f} pp) and 2022 ({R['c22_prot']:+.1f} pp).  "
            "Tobacco and defense are genuinely defensive against recession — but not against "
            "liquidity crises (COVID) or rate-driven bear markets where growth-stock recoveries "
            "outpace the basket."
        ),

        # ---- BEAT 5 — VERDICT ------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** The *excess return* vs SPY is **{R['excess_cagr']:+.2f}%/yr** "
            f"with t = **{R['excess_t']:+.2f}** — far below the |t| ≥ 2 bar, wrong sign.  "
            "No evidence of a neglect premium.  OLS alpha: "
            f"**{R['ols_alpha_ann']:+.2f}%/yr**, t = {R['ols_t_alpha']:+.2f} (statistically zero).\n"
            f"- **Tradability — Mirage.** SIN_EW Sharpe {R['sin_sharpe']:.3f} vs SPY {R['spy_sharpe']:.3f}; "
            f"total return {R['sin_total_ret']}% vs {R['spy_total_ret']}%.  LVS is the fatal outlier: "
            f"a {R['lvs_maxdd']:.1f}% peak-to-trough loss wrecks the basket.\n"
            "- **Neglect premium? — Mixed.** Hong-Kacperczyk was real in 1965–2006 but is "
            "not robustly confirmed in 2008–2026.  Outperformed in the value era (2009–2017), "
            "underperformed badly in the growth era (2018–2021), neutral since."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually invest in it?\n\n"
            "The six stocks are all highly liquid (combined ~$500bn market cap).  The practical "
            "barrier is portfolio construction: equal-weight across six names requires periodic "
            "rebalancing — low friction but not zero.  The basket also has high idiosyncratic "
            "concentration risk (LVS alone has 58.5% vol).\n\n"
            "The better question is whether the *sin* label is a useful portfolio criterion.  "
            "Tobacco and defense are defensive value stocks; the sin label adds gambling (LVS), "
            "which is neither defensive nor value.  A cleaner implementation might be:"
        ),
        code(
            "# Compare: what if we dropped LVS from the basket?\n"
            "if HAVE_REAL:\n"
            "    no_lvs = [t for t in data.SIN_BASKET if t != 'LVS']\n"
            "    bp_5 = all_prices[no_lvs]\n"
            "    ew_5 = st.build_equal_weight(bp_5)\n"
            "    idx5 = ew_5.index.intersection(spy_ret.index)\n"
            "    s5 = st.summarize(ew_5.loc[idx5])\n"
            "    diff5 = ew_5.loc[idx5] - spy_ret.loc[idx5]\n"
            "    sd5 = st.summarize(diff5)\n"
            "    print(f'No-LVS basket: CAGR={s5[\"cagr_pct\"]:+.2f}% Sharpe={s5[\"sharpe_ann\"]:+.3f}')\n"
            "    print(f'Excess vs SPY: {sd5[\"cagr_pct\"]:+.2f}%/yr t={sd5[\"tstat\"]:+.2f}')\n"
            "else:\n"
            "    print('No-LVS (5-stock) basket: CAGR ~+12.72%  Sharpe ~+0.652')\n"
            "    print('Excess vs SPY: +0.49%/yr  t=+0.16  (still insignificant)')"
        ),
        md(
            "Even without LVS, the five-stock basket barely outperforms (+0.49%/yr excess) "
            "with t = +0.16 — confirming that the neglect premium is not robustly present "
            "in this sample, not just dragged down by the casino outlier."
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Factor-adjusted premium.** Blitz & Fabozzi (2017) argue the premium disappears "
            "once value and profitability are controlled.  Constructing a Fama-French-5-factor "
            "adjusted alpha would settle this.\n"
            "- **Individual sectors.** Tobacco alone (MO + PM) delivered +13%/yr CAGR and "
            "Sharpe ~0.55 — above SPY.  Testing tobacco as a standalone sin factor is the "
            "most plausible version of the neglect story.\n"
            "- **European and global sin.** The US basket is narrow.  A global sin index "
            "(including BAT, BATS, Leonardo) may capture a broader neglect premium.\n"
            "- **Compare [Study 206 — Dividend-Aristocrats](../../206-dividend-aristocrats/)**: "
            "similar value/defensive tilt, similar underperformance vs SPY — confirms the pattern.\n\n"
            "*Think the neglect premium is real today? Build a Fama-French-adjusted factor model "
            "and show a robust positive alpha that clears the inference bar.*"
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
            "# Sin-Stocks — a quantitative teardown\n"
            "### Neglect premium · daily total return · HAC inference · OLS decomposition · bootstrap CI\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Neglect_premium: Mixed](https://img.shields.io/badge/Neglect_premium%3F-Mixed-8b949e?style=flat-square)\n\n"
            "The quantitative companion to [01_for_the_curious.ipynb](01_for_the_curious.ipynb). "
            "We test the Hong & Kacperczyk (2009) neglect premium on a 2008–2026 daily total-return "
            "tape: equal-weight sin basket (MO, PM, STZ, LVS, LMT, RTX) vs SPY and DSI. "
            "Inference: HAC Newey-West t-stat, block-bootstrap Sharpe CI, OLS alpha/beta vs both "
            "benchmarks, crash-episode analysis, sub-period breakdown, and a synthetic positive control.\n\n"
            "> ⚠️ **Not investment advice.** Reproducible numbers in "
            "[`docs/results.md`](../docs/results.md)."
        ),
        code(BOOT + "\nfrom quantlab import analytics, stats\n"),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Excess return **{R['excess_cagr']:+.2f}%/yr**, HAC "
            f"*t* = **{R['excess_t']:+.2f}**; bootstrap excess-Sharpe CI "
            f"[{R['excess_sharpe_ci_lo']:+.3f}, {R['excess_sharpe_ci_hi']:+.3f}] "
            f"({R['excess_frac_neg']}% of resamples negative). OLS alpha **{R['ols_alpha_ann']:+.2f}%/yr**, "
            f"*t* = **{R['ols_t_alpha']:+.2f}**. |\n"
            f"| **Tradability** | `MIRAGE` | SIN_EW CAGR {R['sin_cagr']:+.2f}% vs SPY "
            f"{R['spy_cagr']:+.2f}%; Sharpe {R['sin_sharpe']:.3f} vs {R['spy_sharpe']:.3f}. "
            f"LVS max drawdown {R['lvs_maxdd']:.1f}% wrecks the basket. |\n"
            f"| **Neglect premium?** | `MIXED` | Outperformed 2009–2017 (value era), "
            "failed 2018–2021 (growth era). Regime-dependent, not durable. |\n\n"
            f"> The basket earns a positive raw t-stat (t = {R['sin_t']:+.2f}), confirming equity "
            "exposure. But the *excess return* t = −0.46 confirms there is no neglect premium "
            "over the market in this post-publication sample."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r_{S,t}$, $r_{B,t}$ be the daily log total-returns of SIN_EW and SPY. "
            "The neglect premium hypothesis asserts:\n\n"
            "- **H₁ (neglect premium).** $\\mathbb{E}[r_{S,t} - r_{B,t}] > 0$ — positive mean "
            "excess return, robust to autocorrelation (HAC t ≥ 2).\n"
            "- **H₂ (market-neutral alpha).** Positive OLS alpha: "
            "$r_{S,t} = \\alpha + \\beta r_{B,t} + \\epsilon_t$ with $\\alpha > 0$ at *t* ≥ 2. "
            "Beta < 1 (sector/defensive tilt) without positive alpha is just a different factor "
            "loading, not a neglect premium.\n"
            "- **H₃ (ESG counter-portfolio underperforms).** DSI returns < sin returns. "
            "The ESG exclusion should depress DSI and boost sin.\n\n"
            "We fail to confirm H₁, H₂, and H₃ on the 2008–2026 tape at any conventional "
            "significance level."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "**H₁** is the economic core of Hong & Kacperczyk (2009) — the entire 'vice pays' "
            "narrative.  If H₁ fails post-2009, either the premium was arbitraged away by "
            "unconstrained capital after publication, or it was a specific-period effect tied "
            "to the value factor dominance of the 1965–2006 sample.  "
            "**H₂** tests whether what excess return exists is market-neutral (genuine neglect) "
            "or is just a value/sector tilt.  Beta = 0.825 confirms the sector tilt; alpha "
            "t = +0.20 confirms zero market-neutral effect.  "
            "**H₃** is the mirror claim: DSI (the socially responsible counter-portfolio) should "
            "underperform.  In fact DSI returned 702% vs sin's 527% — the opposite of the prediction."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Data.** Yahoo Finance daily adjusted total-return closes (`auto_adjust=True`) "
            "for MO, PM, STZ, LVS, LMT, RTX (sin basket), SPY, and DSI, aligned on inner join. "
            "Start: 2008-03-18 (PM IPO date); end: 2026-06-12; n = 4,589 daily observations.\n"
            "- **Basket construction.** Equal-weight across six tickers, daily.  "
            "A deterministic synthetic tape with a tunable planted alpha serves as the positive "
            "control — confirms the engine can detect edge when it exists.\n"
            "- **Return test.** Daily log-return series; HAC Newey-West t-stat on the mean "
            "excess return; block-bootstrap 95% CI on excess Sharpe (CBB, block = n^(1/3)).\n"
            "- **Decomposition.** OLS regression of SIN_EW vs SPY and vs DSI.\n"
            "- **Crash test.** SPY drawdown episodes ≥ 10%: compare SIN_EW total return inside "
            "each episode (start to recovery above prior peak).\n"
            "- **Attribution.** Per-ticker statistics to diagnose basket composition."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),

        md("### 4a · Gross returns — SIN_EW vs SPY vs DSI"),
        code(
            "if HAVE_REAL:\n"
            "    all_prices = data.load_aligned(fetch=False, cache_dir=CACHE)\n"
            "    basket = all_prices[data.SIN_BASKET]\n"
            "    ew_ret = st.build_equal_weight(basket)\n"
            "    spy_ret = np.log(all_prices['SPY'] / all_prices['SPY'].shift(1)).dropna()\n"
            "    dsi_ret = np.log(all_prices['DSI'] / all_prices['DSI'].shift(1)).dropna()\n"
            "    idx = ew_ret.index.intersection(spy_ret.index).intersection(dsi_ret.index)\n"
            "    ew_ret, spy_ret, dsi_ret = ew_ret.loc[idx], spy_ret.loc[idx], dsi_ret.loc[idx]\n"
            "    rows = []\n"
            "    for lbl, r in [('SIN_EW', ew_ret), ('SPY', spy_ret), ('DSI', dsi_ret)]:\n"
            "        rows.append(st.summarize(r, label=lbl))\n"
            "    df_sum = pd.DataFrame(rows).set_index('label')\n"
            "else:\n"
            "    df_sum = pd.DataFrame({\n"
            "        'cagr_pct':    [R['sin_cagr'],   R['spy_cagr'],   R['dsi_cagr']],\n"
            "        'vol_ann_pct': [R['sin_vol'],    R['spy_vol'],    R['dsi_vol']],\n"
            "        'sharpe_ann':  [R['sin_sharpe'], R['spy_sharpe'], R['dsi_sharpe']],\n"
            "        'max_dd_pct':  [R['sin_maxdd'],  R['spy_maxdd'],  R['dsi_maxdd']],\n"
            "        'tstat':       [R['sin_t'],      R['spy_t'],      R['dsi_t']],\n"
            "    }, index=['SIN_EW', 'SPY', 'DSI'])\n"
            "    ew_ret, spy_ret, dsi_ret = None, None, None\n"
            "df_sum.round(3)"
        ),
        md(
            f"> The sin basket has lower CAGR (+{R['sin_cagr']:.2f}% vs +{R['spy_cagr']:.2f}%), "
            f"lower Sharpe ({R['sin_sharpe']:.3f} vs {R['spy_sharpe']:.3f}), and a much larger "
            f"max drawdown ({R['sin_maxdd']:.1f}% vs {R['spy_maxdd']:.1f}%).  DSI (the ESG portfolio) "
            "outperforms sin on every dimension — the opposite of the neglect premium prediction."
        ),

        md("### 4b · Excess return test — H₁\n\nHAC t-stat and block-bootstrap Sharpe CI."),
        code(
            "if HAVE_REAL:\n"
            "    diff = ew_ret - spy_ret\n"
            "    s_diff = st.summarize(diff, label='SIN-SPY')\n"
            "    ci = stats.sharpe_ci_bootstrap(diff, periods_per_year=252, seed=211)\n"
            "    em, et = s_diff['mean_bps'], s_diff['tstat']\n"
            "    clo, chi, fn = ci['ci_low'], ci['ci_high'], ci['frac_negative']\n"
            "else:\n"
            "    em, et = R['excess_bps'], R['excess_t']\n"
            "    clo, chi, fn = R['excess_sharpe_ci_lo'], R['excess_sharpe_ci_hi'], R['excess_frac_neg']/100\n"
            "    diff = pd.Series(dtype=float)\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))\n"
            "if HAVE_REAL and diff is not None and len(diff) > 0:\n"
            "    axes[0].hist(diff.values * 1e4, bins=60, color=AMBER, alpha=0.7)\n"
            "    axes[0].axvline(em, c=RED, lw=2, label=f'mean={em:+.2f} bps/day')\n"
            "else:\n"
            "    axes[0].bar(['Mean'], [em], color=AMBER)\n"
            "    axes[0].axhline(0, c='k', lw=1)\n"
            "axes[0].set_xlabel('Excess return (bps/day)'); axes[0].set_ylabel('count')\n"
            "axes[0].set_title(f'SIN_EW − SPY daily excess return (mean={em:+.2f} bps, t={et:+.2f})')\n"
            "axes[0].legend()\n"
            "axes[1].plot([clo, chi], [0, 0], 'o-', c=RED, lw=3, ms=8, label='95% bootstrap CI')\n"
            "axes[1].axvline(0, c='k', lw=1, ls='--')\n"
            "axes[1].set_xlabel('Excess Sharpe (annualised)'); axes[1].set_yticks([])\n"
            "axes[1].set_title(f'Excess Sharpe CI [{clo:+.3f}, {chi:+.3f}] ({fn*100:.0f}% neg.)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'HAC t = {et:+.2f}  CI = [{clo:+.3f}, {chi:+.3f}]  {fn*100:.0f}% of resamples negative')"
        ),
        md(
            f"> HAC t = **{R['excess_t']:+.2f}** on the excess return — far below |2|.  "
            f"Bootstrap CI [{R['excess_sharpe_ci_lo']:+.3f}, {R['excess_sharpe_ci_hi']:+.3f}] "
            f"straddles zero; {R['excess_frac_neg']}% of resamples are negative.  "
            "**H₁ (neglect premium) is not confirmed** in the 2008–2026 sample."
        ),

        md("### 4c · OLS alpha/beta — H₂ (market-neutral neglect) and H₃ (ESG)"),
        code(
            "if HAVE_REAL:\n"
            "    ols_spy = st.beta_alpha_ols(ew_ret, spy_ret, label='SIN vs SPY')\n"
            "    ols_dsi = st.beta_alpha_ols(ew_ret, dsi_ret, label='SIN vs DSI')\n"
            "else:\n"
            "    ols_spy = {'alpha_daily_bps': R['ols_alpha_bps'], 'alpha_ann_pct': R['ols_alpha_ann'],\n"
            "               't_alpha': R['ols_t_alpha'], 'beta': R['ols_beta'], 'r_squared': R['ols_r2'], 'n': R['n']}\n"
            "    ols_dsi = {'alpha_daily_bps': R['ols_dsi_alpha_bps'], 'alpha_ann_pct': R['ols_dsi_alpha_ann'],\n"
            "               't_alpha': R['ols_dsi_t_alpha'], 'beta': R['ols_dsi_beta'], 'r_squared': 0.55, 'n': R['n']}\n"
            "print('OLS: SIN_EW = alpha + beta * benchmark + eps')\n"
            "for lbl, ols in [('vs SPY', ols_spy), ('vs DSI', ols_dsi)]:\n"
            "    print(f\"  {lbl}: alpha={ols['alpha_daily_bps']:+.3f} bps/day ({ols['alpha_ann_pct']:+.2f}%/yr) \"\n"
            "          f\"t={ols['t_alpha']:+.2f}  beta={ols['beta']:.4f}  R²={ols['r_squared']:.4f}\")"
        ),
        md(
            f"> Beta = **{R['ols_beta']:.3f}** vs SPY reflects the sector tilt away from high-beta "
            "tech.  Alpha = "
            f"**{R['ols_alpha_bps']:+.3f} bps/day ({R['ols_alpha_ann']:+.2f}%/yr)** with t = "
            f"**{R['ols_t_alpha']:+.2f}**: statistically zero.  The same result holds vs DSI.  "
            "H₂ (market-neutral neglect premium) is not confirmed."
        ),

        md("### 4d · Crash analysis — the defensive claim"),
        code(
            "if HAVE_REAL:\n"
            "    all_ret_df = pd.DataFrame({'SIN_EW': ew_ret, 'SPY': spy_ret})\n"
            "    crashes = st.crash_analysis(all_ret_df, threshold=-0.10)\n"
            "else:\n"
            "    crashes = pd.DataFrame({\n"
            "        'start': ['2008-06','2011-08','2015-08','2018-02','2018-12','2020-02','2022-02','2025-03'],\n"
            "        'bench_max_dd_pct': [R['c08_dd'],R['c11_dd'],R['c15_dd'],R['c18a_dd'],R['c18b_dd'],R['c20_dd'],R['c22_dd'],R['c25_dd']],\n"
            "        'protection_pp': [R['c08_prot'],R['c11_prot'],R['c15_prot'],\n"
            "                          R['c18a_prot'],R['c18b_prot'],R['c20_prot'],R['c22_prot'],R['c25_prot']],\n"
            "    })\n"
            "if 'protection_pp' in crashes.columns:\n"
            "    print(crashes[['start','bench_max_dd_pct','protection_pp']].to_string())\n"
            "    print(f'\\nMean protection: {crashes[\"protection_pp\"].mean():+.1f} pp')\n"
            "    print(f'Positive episodes: {(crashes[\"protection_pp\"] > 0).sum()}/{len(crashes)}')"
        ),
        md(
            f"> Protection is **mixed**: positive in GFC-era episodes (recession-driven), "
            f"negative in growth-led bear markets.  COVID protection: {R['c20_prot']:+.1f} pp.  "
            f"2022: {R['c22_prot']:+.1f} pp.  The crash-shield claim holds only in one specific "
            "regime (recession/value), not unconditionally."
        ),

        md("### 4e · Sub-period regime breakdown — the regime-dependence story"),
        code(
            "if HAVE_REAL:\n"
            "    periods = [\n"
            "        ('2008-03-18', '2012-12-31', 'GFC era 2008-12'),\n"
            "        ('2013-01-01', '2017-12-31', 'Value bull 2013-17'),\n"
            "        ('2018-01-01', '2021-12-31', 'Growth dominance 2018-21'),\n"
            "        ('2022-01-01', '2026-06-12', 'Post-2022'),\n"
            "    ]\n"
            "    rows_p = []\n"
            "    for start, end, lbl in periods:\n"
            "        sub_ew = ew_ret.loc[start:end]\n"
            "        sub_spy = spy_ret.loc[start:end]\n"
            "        s_sin = st.summarize(sub_ew)\n"
            "        s_spy = st.summarize(sub_spy)\n"
            "        diff_p = sub_ew - sub_spy\n"
            "        s_diff = st.summarize(diff_p)\n"
            "        rows_p.append({'period': lbl, 'sin_cagr': s_sin['cagr_pct'],\n"
            "                       'spy_cagr': s_spy['cagr_pct'], 'excess': s_diff['cagr_pct'],\n"
            "                       'excess_t': s_diff['tstat']})\n"
            "    sub_df = pd.DataFrame(rows_p).set_index('period')\n"
            "else:\n"
            "    sub_df = pd.DataFrame({\n"
            "        'sin_cagr':  [7.67, 21.97, 0.07, 11.63],\n"
            "        'spy_cagr':  [4.55, 15.69, 17.52, 12.07],\n"
            "        'excess':    [3.12, 6.28, -17.45, -0.44],\n"
            "        'excess_t':  [0.44, 1.48, -2.66, -0.05],\n"
            "    }, index=['GFC era 2008-12','Value bull 2013-17','Growth dominance 2018-21','Post-2022'])\n"
            "print(sub_df.round(2))\n"
            "fig, ax = plt.subplots(figsize=(9, 4))\n"
            "colors = [GREEN if e > 0 else RED for e in sub_df['excess']]\n"
            "ax.bar(sub_df.index, sub_df['excess'], color=colors, alpha=0.8)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('Excess CAGR vs SPY (%/yr)')\n"
            "ax.set_title('Sin basket excess return by regime: value dominance vs growth dominance')\n"
            "plt.xticks(rotation=15, ha='right')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            "> The sub-period table is the key insight: sin stocks outperformed by **+3 to +6%/yr** "
            "in value-led markets (2008–2017) and underperformed by **−17%/yr** in the growth-led "
            "period (2018–2021, t = −2.66).  The HAC t-stat for the full period (t = −0.46) "
            "reflects the near-cancellation of two very different regimes.  This is a value/sector "
            "tilt, not a stable neglect premium."
        ),

        md("### 4f · Synthetic positive control — the engine works when alpha exists"),
        code(
            "alpha_grid = [0.0, 0.5, 1.0, 2.0, 3.0]\n"
            "rows_ctrl = []\n"
            "for a in alpha_grid:\n"
            "    p_syn, _ = data.synthetic_daily(n_years=15, alpha_bps=a, seed=211)\n"
            "    sin_ret = np.log(p_syn['SIN'] / p_syn['SIN'].shift(1)).dropna()\n"
            "    spy_ret_s = np.log(p_syn['SPY'] / p_syn['SPY'].shift(1)).dropna()\n"
            "    # Planted alpha shifts SIN mean by exactly a bps/day\n"
            "    s_sin = st.summarize(sin_ret)\n"
            "    rows_ctrl.append({'planted_alpha_bps': a, 'sin_mean_bps': s_sin['mean_bps'], 'sin_t': s_sin['tstat']})\n"
            "ctrl_df = pd.DataFrame(rows_ctrl)\n"
            "fig, ax = plt.subplots(figsize=(9, 4))\n"
            "ax.plot(ctrl_df['planted_alpha_bps'], ctrl_df['sin_t'], 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1, label='|t|=2 bar')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('Planted daily alpha (bps/day)')\n"
            "ax.set_ylabel('HAC t-stat on SIN return')\n"
            "ax.set_title('Engine detects alpha when planted — not a \"returns nothing\" printer')\n"
            "ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "ctrl_df.round(3)"
        ),
        md("> The engine is correctly calibrated: planted alpha shifts the mean monotonically. "
           "The real tape's t-stat on the excess return (−0.46) reflects genuine absence of "
           "a neglect premium, not a broken test."),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — excess return {R['excess_cagr']:+.2f}%/yr, HAC *t* {R['excess_t']:+.2f} "
            f"(far below |2|, wrong sign); OLS alpha {R['ols_alpha_ann']:+.2f}%/yr, *t* {R['ols_t_alpha']:+.2f}. "
            f"Bootstrap excess-Sharpe CI [{R['excess_sharpe_ci_lo']:+.3f}, {R['excess_sharpe_ci_hi']:+.3f}] "
            f"({R['excess_frac_neg']}% of resamples negative). No neglect premium detectable.\n"
            f"- **Tradability `MIRAGE`** — SIN_EW CAGR {R['sin_cagr']:+.2f}% vs SPY "
            f"{R['spy_cagr']:+.2f}%; Sharpe {R['sin_sharpe']:.3f} vs {R['spy_sharpe']:.3f}. "
            f"LVS (−{abs(R['lvs_maxdd']):.1f}% max DD) makes the basket worse than any of its "
            "individual components on a risk-adjusted basis.\n"
            "- **Neglect premium `MIXED`** — not confirmed post-publication.  The 2009–2017 "
            "outperformance reflected a value/sector regime, not a stable neglect-driven premium."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it?\n\n"
            "The basket is liquid but the economics are wrong: no gross alpha, significant "
            "idiosyncratic risk from LVS, and a strong regime-dependence that requires "
            "predicting value vs growth cycles — which is itself a hard problem.  "
            "A cleaner implementation of the 'neglect' thesis would:\n\n"
            "1. Exclude casino stocks (pure speculation, not 'neglected value').\n"
            "2. Add a valuation screen (the neglect premium is P/E or P/B compression — "
            "   buying sin stocks when they're cheap, not always).\n"
            "3. Use a factor-adjusted framework to isolate the pure neglect alpha from "
            "   value/profitability tilts."
        ),
        code(
            "# Quantify the ER headwind (sin stocks have no ETF expense ratio in this study —\n"
            "# direct ownership — but the daily rebalancing cost is material)\n"
            "# One-way transaction cost estimate: 2 bps per trade, 6 stocks, daily rebalance\n"
            "# Annual cost ~ 2 bps * 6 * 252 ~ 30.24 bps/yr = 0.30%/yr\n"
            "one_way_bps = 2.0\n"
            "n_stocks = 6\n"
            "annual_cost = one_way_bps * n_stocks * 252 / 100\n"
            "print(f'Estimated daily-rebalance friction: {annual_cost:.2f}%/yr')\n"
            "print(f'vs gross excess return: {R[\"excess_cagr\"]:+.2f}%/yr')\n"
            "print(f'Net excess: {R[\"excess_cagr\"] - annual_cost:+.2f}%/yr')\n"
            "print(f'(Friction is an overestimate for patient weekly/monthly rebalancing)')"
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Factor adjustment.** Blitz & Fabozzi (2017) show the premium shrinks "
            "in Fama-French 3/5-factor regressions.  Build a factor-neutral sin portfolio "
            "and check if any residual alpha survives.\n"
            "- **The neglect premium by decade.** The 1965–1990 premium may be structural; "
            "the 2009+ decline may reflect convergence.  A rolling 10-year window analysis "
            "would quantify the decay rate.\n"
            "- **Individual sin sub-categories.** The tobacco sub-basket (MO + PM) delivered "
            f"~{(R['mo_cagr']+R['pm_cagr'])/2:.1f}%/yr CAGR — a plausible standalone study.  "
            "Defense alone (LMT + RTX) delivered ~+12%/yr with Sharpe ~0.46 — less so.\n"
            "- **[Study 206 — Dividend-Aristocrats](../../206-dividend-aristocrats/)** tests "
            "another value/defensive tilt with similar regime-dependence and underperformance "
            "vs SPY — the family resemblance is clear.\n\n"
            "*Think the neglect premium can be isolated with better factor control? Fork this, "
            "add a Fama-French decomposition, and show a robust market-neutral alpha.*"
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
