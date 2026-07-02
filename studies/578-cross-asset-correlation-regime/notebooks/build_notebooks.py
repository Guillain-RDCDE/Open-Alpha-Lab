"""Generate the two narrative notebooks for Study 578 (Cross-Asset-Correlation-Regime).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic figures run
anywhere, offline and deterministic; the real-tape cells use the cached returns under ../_cache/ if
present and otherwise quote the frozen headline numbers in ``R`` (mirroring docs/results.md), so the
notebooks re-run for any reader offline.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-30; tape
# 2007-01-04 -> 2026-06-26; 14 ETFs; returns fp 9dbfa917d6ba, regime/forward frame fp 0b73f61632f6).
R = dict(
    fp_rets="9dbfa917d6ba", fp_frame="0b73f61632f6",
    start="2007-01-04", end="2026-06-26", asof="2026-06-30",
    window=63, horizon=21, q=0.70,
    n_high=1792, n_low=2774,
    high_fwd_ret=1.65, low_fwd_ret=0.67, ret_spread=0.98, ret_t=6.80, placebo_p=0.048,
    high_fwd_vol=17.2, low_fwd_vol=15.6, vol_spread=1.62, vol_t=4.62,
    avg_corr_mean=0.275, corr_max=0.598,
    dd_high=-11.7, dd_low=-5.3,
    # overlay: gross/net CAGR%, Sharpe, maxDD%; buy-hold same
    ov_gross_cagr=5.81, ov_net_cagr=5.55, ov_sharpe=0.42, ov_dd=-37.5,
    bh_cagr=12.20, bh_sharpe=0.62, bh_dd=-55.2, switches=98,
    # robustness sweep: (window, horizon, q, ret_spread%, ret_t, vol_spread%, vol_t)
    sweep=[
        (63, 21, 0.70, 0.98, 6.80, 1.62, 4.62),
        (63, 21, 0.80, 1.03, 6.62, 1.03, 3.01),
        (63, 63, 0.70, 3.37, 14.81, -0.06, -0.18),
        (126, 21, 0.70, 1.50, 10.64, 0.51, 1.47),
        (126, 63, 0.70, 3.76, 16.75, -1.03, -3.34),
        (42, 21, 0.70, 0.72, 4.94, 1.57, 4.52),
        (63, 5, 0.70, 0.18, 2.36, 2.77, 6.63),
    ],
    # synthetic control: (fragility, mean spread%, mean t over 25 seeds)
    ctrl=[(0.0, 0.05, 0.37), (0.5, -0.26, -1.48), (0.9, -0.50, -2.57), (1.4, -0.78, -3.44)],
    # HIGH-regime fraction by year (real)
    frac_year=[(2008, 0.29), (2009, 0.91), (2010, 0.50), (2011, 0.15), (2012, 0.25),
               (2013, 0.57), (2014, 0.06), (2015, 0.14), (2016, 0.22), (2017, 0.09),
               (2018, 0.30), (2019, 0.00), (2020, 0.63), (2021, 0.43), (2022, 0.62),
               (2023, 0.74), (2024, 0.78), (2025, 0.45), (2026, 0.31)],
)


BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from cross_asset_correlation_regime import data, strategy as st

WINDOW, HORIZON, Q = 63, 21, 0.70

RETS = data.fetch_returns(fetch=False)          # cache-first; empty frame offline
HAVE_REAL = (not RETS.empty) and ("SPY" in RETS.columns)
# Compute the correlation index + regime once (it is the slow step) and reuse it everywhere.
if HAVE_REAL:
    AC = st.avg_pairwise_corr(RETS, window=WINDOW)
    LAB = st.regime_labels(AC, q=Q, min_periods=252)
    RES = st.regime_forward_test(RETS, window=WINDOW, horizon=HORIZON, q=Q, min_periods=252)
print("real cross-asset tape present:", HAVE_REAL,
      "" if not HAVE_REAL else f"({RETS.shape[1]} ETFs, {RETS.index.min().date()} -> {RETS.index.max().date()})")
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Cross-Asset Correlation — when everything moves together, is a crash coming? 🕸️\n"
            "### Reading the 'diversification is dead' warning light, in plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Fragility_gauge%3F: Coincident](https://img.shields.io/badge/Fragility_gauge%3F-Coincident-8b949e?style=flat-square)\n\n"
            "There's a maxim every risk manager repeats: *in a crisis, all correlations go to one.* "
            "Stocks, bonds, gold, oil — normally they zig and zag independently, but when panic hits "
            "they all crash together and your careful diversification evaporates. The folklore adds a "
            "twist: watch the **average correlation** across everything, and when it starts *rising*, "
            "take cover — the market is about to break.\n\n"
            "So we test it the honest way: take 14 exchange-traded funds spanning stocks, bonds, "
            "credit, gold, commodities, oil, real estate and emerging markets; each day measure how "
            "much they're all moving together; and ask — when they cluster, does the stock market do "
            "*worse* next month?\n\n"
            "> 📓 **This is the plain-language layer.** Want the *t*-stats, the placebo test and the "
            "cost maths? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool: every chart is drawn by the "
            "code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ----
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| When correlation is HIGH, did stocks do *worse* next month? | **No — better.** SPY's "
            f"next-month return averaged **+{R['high_fwd_ret']}%** in the high-correlation regime vs "
            f"**+{R['low_fwd_ret']}%** in the calm one. The opposite of the warning. |\n"
            f"| Is that just luck? | **No** — a shuffle test that respects the overlap puts it at "
            f"*p* = {R['placebo_p']}. The effect is real; it's just backwards. |\n"
            f"| Does correlation warn of anything? | **Yes — turbulence.** Next-month *volatility* is "
            f"higher in the high-corr regime ({R['high_fwd_vol']}% vs {R['low_fwd_vol']}%). It flags "
            "bumpiness, not losses. |\n"
            f"| Could you trade the warning? | **No.** Going to cash whenever correlation is high "
            f"*halved* your return (Sharpe {R['ov_sharpe']} vs {R['bh_sharpe']}) — you'd have sat out "
            "the rebounds. |\n\n"
            "> The folklore has the *timing* backwards. Correlations spike *during* a crash, when "
            "stocks are already down — and the market usually bounces from there. High correlation is "
            "the storm you're **in**, not the one ahead."
        ),

        # ---- BEAT 1 — THE CLAIM ----
        md(
            "## 1 · The claim\n\n"
            "> *\"When average cross-asset correlation rises, diversification is failing and a market "
            "break is near.\"* — risk-management folklore (Longin & Solnik 2001; Ang & Chen 2002).\n\n"
            "The intuition is genuinely sensible. In normal times a bond rally cushions a stock "
            "sell-off, gold hedges inflation, oil marches to its own drum — the pieces offset. In a "
            "true panic everyone sells everything to raise cash, so *everything* falls at once and the "
            "offsets disappear. If that clustering shows up *early*, a rising average correlation "
            "would be a real fragility alarm."
        ),

        # ---- BEAT 2 — SO WHAT ----
        md(
            "## 2 · So what?\n\n"
            "If true, it's a free risk light: one number, updated daily, telling you to de-risk before "
            "the drop. It's also the logic behind 'risk-parity blew up' post-mortems and countless "
            "'diversification is dead' headlines. The desk has looked at a *single* correlation pair "
            "([245 Oil-Equity](../../245-oil-equity-correlation/)) and at a *cross-sectional* "
            "correlation sort ([502 Betting-Against-Correlation](../../502-betting-against-correlation/)); "
            "here we test the **market-wide average** as a timing signal."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ----
        md(
            "## 3 · How would we even know?\n\n"
            "1. **Measure togetherness.** Each day, over the last quarter (~63 trading days), compute "
            "how correlated all 14 funds are with each other and average it — one 'how much is "
            "everything moving together' number.\n"
            "2. **Split into regimes.** Call a day HIGH if that number is unusually high versus its own "
            "past (top 30%), else LOW.\n"
            "3. **Look forward.** Did stocks (SPY) earn *less* over the next month after a HIGH day "
            "than after a LOW day? The folklore says yes.\n\n"
            "*Timing:* the correlation number uses only data up to today's close, and 'the next month' "
            "is strictly *after* today — so there's no peeking at the future."
        ),

        # ---- BEAT 4 — THE TEARDOWN ----
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**After a HIGH-correlation day vs a LOW one — what did stocks do next month?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    hi_r, lo_r = RES['high_fwd_ret']*100, RES['low_fwd_ret']*100\n"
            "    spr, t = RES['ret_spread']*100, RES['ret_t']\n"
            "    banner = f\"REAL tape (14 ETFs, {RETS.index.min().year}-{RETS.index.max().year})\"\n"
            "else:\n"
            f"    hi_r, lo_r, spr, t = {R['high_fwd_ret']}, {R['low_fwd_ret']}, {R['ret_spread']}, {R['ret_t']}\n"
            "    banner = 'frozen real numbers (offline)'\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.4))\n"
            "ax.bar(['HIGH corr\\n(fragile?)','LOW corr\\n(calm)'], [hi_r, lo_r], color=[RED, GREEN], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('SPY forward 1-month return %')\n"
            "ax.set_title(f'The alarm says HIGH corr -> worse. Here HIGH corr earned MORE (+{hi_r:.2f}% vs +{lo_r:.2f}%)')\n"
            "fig.suptitle(banner, fontsize=9, color=GREY); plt.tight_layout(); plt.show()\n"
            "print(f'HIGH +{hi_r:.2f}%  vs  LOW +{lo_r:.2f}%  ->  spread {spr:+.2f}% (t {t:+.2f})')"
        ),
        md(
            f"There it is, **backwards**. The alarm predicts the high-correlation regime is the "
            f"dangerous one. On 2007-2026 stocks earned **+{R['high_fwd_ret']}%** in that regime vs "
            f"**+{R['low_fwd_ret']}%** in the calm one — a **+{R['ret_spread']}%** edge the *wrong* way. "
            "Why? Let's see *when* correlation actually spikes."
        ),
        md(
            "**When is correlation high?** Plot the share of each year spent in the HIGH regime:"
        ),
        code(
            "fy = " + repr(R["frac_year"]) + "\n"
            "if HAVE_REAL:\n"
            "    y = pd.DataFrame({'lab': LAB}).dropna(); y['yr'] = y.index.year\n"
            "    fr = y.groupby('yr')['lab'].mean()\n"
            "    years, fracs = list(fr.index), list(fr.values)\n"
            "else:\n"
            "    years, fracs = [a for a,_ in fy], [b for _,b in fy]\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.2))\n"
            "cols = [RED if f>0.5 else GREY for f in fracs]\n"
            "ax.bar([str(y) for y in years], fracs, color=cols, width=.7)\n"
            "ax.set_ylabel('share of year in HIGH-corr regime')\n"
            "ax.set_title('Correlation spikes in the crisis years: 2009, 2020, 2022 (the crashes)')\n"
            "plt.xticks(rotation=60); plt.tight_layout(); plt.show()\n"
            "print('HIGH-regime years:', [str(y) for y,f in zip(years,fracs) if f>0.5])"
        ),
        md(
            "See the pattern? Correlation goes HIGH in **2009** (91% of the year!), **2020** (COVID), "
            "**2022** (the bond-and-stock crash) — the crisis years. But those are the years stocks "
            "were **already down** and about to **rebound**. The alarm fires at the *bottom*, not the "
            "top. High correlation is a *coincident* stress signal, and the month that follows tends "
            "to be the recovery."
        ),

        # ---- BEAT 5 — VERDICT ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** High correlation predicted *higher* stock returns next month "
            f"(+{R['ret_spread']}%, *t* +{R['ret_t']}), the wrong sign, and it holds across every "
            "window we tried. No fragility-predicts-drawdowns signal here.\n"
            "- **Tradability — Mirage.** A risk-off overlay that hides in cash during high correlation "
            "*halved* the return and lowered the Sharpe.\n"
            "- **Fragility gauge? — Coincident.** It *is* a real stress gauge — but coincident (it "
            "spikes during the crash), not leading (it doesn't warn beforehand)."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ----
        md(
            "## 6 · Could you actually trade it?\n\n"
            "The obvious trade — go to cash whenever correlation is high — sounds prudent. Let's run it "
            "(enter one day after the signal, small switching cost)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ov = st.overlay_backtest(RETS, window=WINDOW, q=Q, cost_bps=5.0)\n"
            "    onet, bh = ov['overlay_net'], ov['buy_hold']\n"
            "    o_cagr, o_sh, b_cagr, b_sh = onet['cagr']*100, onet['sharpe'], bh['cagr']*100, bh['sharpe']\n"
            "else:\n"
            f"    o_cagr, o_sh, b_cagr, b_sh = {R['ov_net_cagr']}, {R['ov_sharpe']}, {R['bh_cagr']}, {R['bh_sharpe']}\n"
            "fig, ax = plt.subplots(1, 2, figsize=(9.5, 4.0))\n"
            "ax[0].bar(['risk-off\\noverlay','buy &\\nhold'], [o_cagr, b_cagr], color=[RED, GREEN], width=.55)\n"
            "ax[0].set_title('Return / yr %'); ax[0].axhline(0, c='k', lw=1)\n"
            "ax[1].bar(['risk-off\\noverlay','buy &\\nhold'], [o_sh, b_sh], color=[RED, GREEN], width=.55)\n"
            "ax[1].set_title('Sharpe'); ax[1].axhline(0, c='k', lw=1)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'overlay: {o_cagr:+.1f}%/yr, Sharpe {o_sh:+.2f}  |  buy-hold: {b_cagr:+.1f}%/yr, Sharpe {b_sh:+.2f}')"
        ),
        md(
            f"No. Sitting out the high-correlation regime cut the drawdown (a smoother ride) but "
            f"**halved the return** ({R['ov_net_cagr']}%/yr vs {R['bh_cagr']}%/yr) and *lowered* the "
            f"Sharpe ({R['ov_sharpe']} vs {R['bh_sharpe']}) — because the regime it dodged contained "
            "the best month-ahead returns: the rebounds. The 'safe' move was the costly one."
        ),

        # ---- BEAT 7 — GOING FURTHER ----
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Coincident ≠ useless.** As a *nowcast* of stress (and of higher near-term volatility) "
            "the average-correlation gauge is genuinely informative — just don't read it as a *timing* "
            "signal for returns.\n"
            "- **A longer tape.** Our ETFs only reach back to ~2007 (one GFC, one COVID). Index-level "
            "proxies back to the 1970s might contain an episode where correlation *led* a crash — worth "
            "a look.\n\n"
            "*Think rising correlation really front-runs crashes? Fork this, add a pre-2007 proxy "
            "tape, and show the high-correlation regime earning a **negative** forward return at "
            "*t* < −2.*"
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
            "# Cross-Asset Correlation Regime — a quantitative teardown 🔬\n"
            "### average pairwise correlation · expanding-quantile regime split · two-sample *t* · block-shuffle placebo · coincident-drawdown mechanism · seven-spec sweep · risk-off overlay net of costs · synthetic control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Fragility_gauge%3F: Coincident](https://img.shields.io/badge/Fragility_gauge%3F-Coincident-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We build the mean off-diagonal "
            "pairwise correlation across a 14-ETF cross-asset panel, split the tape into HIGH/LOW "
            "regimes, and test whether the HIGH regime predicts lower forward returns and higher "
            "forward vol on SPY (the fragility claim) — or, here, the opposite on returns.\n\n"
            "> ⚠️ **Not investment advice.** Real data: yfinance daily prices, 14 ETFs, "
            f"{R['start']} → {R['end']} (as-of {R['asof']}, returns fp `{R['fp_rets']}`, regime/forward "
            f"frame fp `{R['fp_frame']}`); the offline core and tests run on a deterministic synthetic "
            "world. Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | HIGH−LOW forward-return spread **+{R['ret_spread']}%** (two-sample "
            f"*t* **+{R['ret_t']}**, block-placebo *p* {R['placebo_p']}) — the *wrong* sign, positive in "
            f"**all 7** specs. Forward-vol spread +{R['vol_spread']}% (*t* +{R['vol_t']}) is right-signed "
            "(turbulence). |\n"
            f"| **Tradability** | `MIRAGE` | Risk-off overlay net CAGR **{R['ov_net_cagr']}%** / Sharpe "
            f"**{R['ov_sharpe']}** vs buy-hold **{R['bh_cagr']}%** / **{R['bh_sharpe']}**; it sat out the "
            "rebounds. |\n"
            f"| **Fragility gauge?** | `COINCIDENT` | Mean *contemporaneous* SPY drawdown "
            f"**{R['dd_high']}%** in HIGH vs **{R['dd_low']}%** in LOW — correlation spikes *inside* the "
            "crash. |\n\n"
            "> 💡 In plain words: the engine works (the synthetic control proves it), but on this "
            "cross-asset tape the fragility claim is the wrong sign on returns and only coincident on "
            "stress. It is a real *nowcast* of stress, not a *forecast* of losses."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $\\bar\\rho_t$ be the trailing average pairwise correlation across the panel, $H_t$ "
            "the HIGH-regime indicator ($\\bar\\rho_t$ above its expanding $q$-quantile), and $r_{t\\to "
            "t+h}$ SPY's forward $h$-day return.\n\n"
            "- **H₁ (fragility / returns).** $\\mathbb{E}[r \\mid H=1] - \\mathbb{E}[r \\mid H=0] < 0$ "
            "at *t* ≤ −2 (high correlation → lower forward return).\n"
            "- **H₂ (turbulence / vol).** forward vol is higher in the HIGH regime (positive spread).\n"
            "- **H₃ (robustness).** the *sign* of H₁ is stable across window/horizon/quantile.\n"
            "- **H₄ (tradable).** a risk-off overlay on the HIGH regime beats buy-and-hold net.\n\n"
            "We find **H₁ rejected with the sign reversed** (forward return is *higher* in HIGH), **H₂ "
            "supported** (vol is higher), **H₃ rejected for the claim** (the spread is stably the "
            "*wrong* sign), and **H₄ rejected** (the overlay underperforms)."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The content is the **why**. Correlation clustering is real and well documented (Longin & "
            "Solnik 2001; Ang & Chen 2002) — but it is a property of *bear* markets, i.e. it is "
            "**coincident** with drawdowns, not a *lead*. Pollet & Wilson (2010) find average "
            "*equity* correlation forecasts market variance; the return-timing folklore is the folk "
            "extension this study puts on the tape. The mechanism that flips the sign: correlation "
            "peaks near the *bottom* of a sell-off, and the forward window catches the rebound — the "
            "same reason the [502 Betting-Against-Correlation](../../502-betting-against-correlation/) "
            "premium concentrates in correlation-spike years."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Index.** $\\bar\\rho_t$ = mean off-diagonal of the trailing-63-day pairwise "
            "correlation matrix of the 14-ETF panel (trailing only).\n"
            "- **Regime.** HIGH if $\\bar\\rho_t$ > its *expanding* 70th percentile (past data only, "
            "252-day warm-up). No look-ahead in the threshold.\n"
            "- **Forward.** SPY's forward 21-day return and annualised realised vol, measured strictly "
            "*after* day *t*.\n"
            "- **Inference.** Two-sample (Welch) *t* on HIGH vs LOW; a **block-shuffle placebo** null "
            "(21-day blocks, 2000 perms) because the forward windows overlap.\n"
            "- **Mechanism.** Contemporaneous SPY drawdown in each regime; the HIGH-regime calendar.\n"
            "- **Robustness.** Seven (window, horizon, q) specs; read the sign.\n"
            "- **Tradable.** Risk-off overlay (flat in HIGH, 1-day lag), gross and net of 5 bps/switch.\n"
            "- **Positive control.** A deterministic synthetic world with a planted fragility effect "
            "(and a null), averaged over 25 seeds (house rule).\n\n"
            "Timing: the regime label is public at day *t*'s close (trailing window + expanding "
            "quantile); the forward series is strictly after *t*; the overlay enters one bar later "
            "(the single documented execution lag). No future data enters the signal."
        ),

        # ---- BEAT 4 ----
        md("## 4 · The teardown"),
        md(
            "### 4a · The regime forward test — H₁ & H₂\n\n"
            "HIGH vs LOW forward return and vol, with the two-sample *t* and the block-shuffle placebo."
        ),
        code(
            "if HAVE_REAL:\n"
            "    p = st.placebo_pvalue(RES, n_perm=2000, block=21, seed=578)\n"
            "    hi_r, lo_r, r_spr, r_t = RES['high_fwd_ret']*100, RES['low_fwd_ret']*100, RES['ret_spread']*100, RES['ret_t']\n"
            "    hi_v, lo_v, v_spr, v_t = RES['high_fwd_vol']*100, RES['low_fwd_vol']*100, RES['vol_spread']*100, RES['vol_t']\n"
            "else:\n"
            f"    hi_r, lo_r, r_spr, r_t, p = {R['high_fwd_ret']}, {R['low_fwd_ret']}, {R['ret_spread']}, {R['ret_t']}, {R['placebo_p']}\n"
            f"    hi_v, lo_v, v_spr, v_t = {R['high_fwd_vol']}, {R['low_fwd_vol']}, {R['vol_spread']}, {R['vol_t']}\n"
            "fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))\n"
            "ax[0].bar(['HIGH','LOW'], [hi_r, lo_r], color=[RED, GREEN], width=.5); ax[0].axhline(0, c='k', lw=1)\n"
            "ax[0].set_ylabel('fwd 1m return %'); ax[0].set_title(f'Return: HIGH-LOW {r_spr:+.2f}% (t {r_t:+.2f}, placebo p {p:.3f})')\n"
            "ax[1].bar(['HIGH','LOW'], [hi_v, lo_v], color=[AMBER, GREEN], width=.5)\n"
            "ax[1].set_ylabel('fwd ann. vol %'); ax[1].set_title(f'Vol: HIGH-LOW {v_spr:+.2f}% (t {v_t:+.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'RET  HIGH +{hi_r:.2f}% | LOW +{lo_r:.2f}% | spread {r_spr:+.2f}% | t {r_t:+.2f} | placebo p {p:.3f}')\n"
            "print(f'VOL  HIGH {hi_v:.1f}% | LOW {lo_v:.1f}% | spread {v_spr:+.2f}% | t {v_t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: **H₁ rejected, sign reversed** — the forward-return spread is "
            f"**+{R['ret_spread']}%** (*t* +{R['ret_t']}, placebo *p* {R['placebo_p']}): high correlation "
            f"was followed by *higher* returns. **H₂ supported** — forward vol *is* higher in HIGH "
            f"(+{R['vol_spread']}%, *t* +{R['vol_t']}). Correlation flags turbulence, not losses."
        ),
        md(
            "### 4b · Why the sign flips — the coincident-drawdown mechanism\n\n"
            "Where is SPY, *right now*, when correlation is HIGH? Measure the contemporaneous drawdown "
            "in each regime."
        ),
        code(
            "if HAVE_REAL:\n"
            "    spy = RETS['SPY']; eq = (1+spy).cumprod(); dd = eq/eq.cummax()-1\n"
            "    d = pd.DataFrame({'lab': LAB, 'dd': dd}).dropna()\n"
            "    dd_hi, dd_lo = d[d.lab==1].dd.mean()*100, d[d.lab==0].dd.mean()*100\n"
            "    ac_series = AC.dropna()\n"
            "else:\n"
            f"    dd_hi, dd_lo = {R['dd_high']}, {R['dd_low']}\n"
            "    ac_series = None\n"
            "fig, ax = plt.subplots(figsize=(8, 4.2))\n"
            "ax.bar(['HIGH-corr regime','LOW-corr regime'], [dd_hi, dd_lo], color=[RED, GREEN], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('mean CONTEMPORANEOUS SPY drawdown %')\n"
            "ax.set_title(f'When correlation is HIGH, SPY is already down {dd_hi:.1f}% (vs {dd_lo:.1f}% in LOW)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'contemporaneous SPY drawdown -> HIGH {dd_hi:.1f}% | LOW {dd_lo:.1f}%')"
        ),
        md(
            f"> 💡 In plain words: the HIGH-correlation regime *is* the drawdown — SPY sits **{R['dd_high']}%** "
            f"below its peak on average there, vs **{R['dd_low']}%** in LOW. Correlation peaks near the "
            "*bottom*; the forward month is the rebound. That is the whole sign flip."
        ),
        md(
            "### 4c · Robustness — H₃ (does the sign hold?)\n\n"
            "The forward-return spread across seven (window, horizon, quantile) specs. A real fragility "
            "signal would be *negative* everywhere; here it is stably *positive*."
        ),
        code(
            "sweep = " + repr(R["sweep"]) + "\n"
            "if HAVE_REAL:\n"
            "    specs = [(w,h,q) for (w,h,q,*_) in sweep]\n"
            "    sw = st.robustness_sweep(RETS, specs)\n"
            "    labs = [f'{int(w)}/{int(h)}/{q:.2f}' for w,h,q in zip(sw['window'],sw['horizon'],sw['q'])]\n"
            "    rspr = list(sw['ret_spread']*100)\n"
            "else:\n"
            "    labs = [f'{w}/{h}/{q:.2f}' for (w,h,q,*_) in sweep]\n"
            "    rspr = [s for (_,_,_,s,_,_,_) in sweep]\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.2))\n"
            "cols = [RED if s>0 else GREEN for s in rspr]\n"
            "ax.bar(range(len(labs)), rspr, color=cols, width=.6)\n"
            "ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, rotation=30, ha='right')\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('HIGH-LOW fwd return spread %')\n"
            "ax.set_xlabel('window / horizon / quantile')\n"
            "ax.set_title('Wrong sign EVERYWHERE (red = positive = anti-fragility)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('ret spreads by spec:', [round(s,2) for s in rspr])"
        ),
        md(
            "> 💡 In plain words: H₃ **rejected for the claim** — the forward-return spread is "
            "**positive in all seven specs** (+0.18% to +3.76%). The fragility sign (negative) never "
            "shows up. What is stable here is the *anti*-fragility."
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — H₁ rejected (spread +{R['ret_spread']}%, *t* +{R['ret_t']}, wrong "
            f"sign; placebo *p* {R['placebo_p']}); stable positive across all specs.\n"
            f"- **Tradability `MIRAGE`** — risk-off overlay net {R['ov_net_cagr']}%/yr, Sharpe "
            f"{R['ov_sharpe']} vs buy-hold {R['bh_cagr']}%/yr, Sharpe {R['bh_sharpe']}.\n"
            "- **Fragility gauge? `COINCIDENT`** — a real stress *nowcast* (higher forward vol, deep "
            "contemporaneous drawdown), not a leading return predictor."
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you trade it? — the risk-off overlay net of costs\n\n"
            "Flat in the HIGH regime, long SPY in LOW, one-day lag, 5 bps per switch. Does dodging the "
            "'fragile' regime pay?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    ov = st.overlay_backtest(RETS, window=WINDOW, q=Q, cost_bps=5.0)\n"
            "    og, on, bh = ov['overlay_gross'], ov['overlay_net'], ov['buy_hold']\n"
            "    rows = [('overlay gross', og), ('overlay net', on), ('buy & hold', bh)]\n"
            "    tab = pd.DataFrame({k: {'CAGR%': v['cagr']*100, 'vol%': v['vol']*100,\n"
            "                            'Sharpe': v['sharpe'], 'maxDD%': v['maxdd']*100} for k,v in rows}).T\n"
            "    sw_n = ov['switches']\n"
            "else:\n"
            f"    tab = pd.DataFrame({{'overlay gross': {{'CAGR%': {R['ov_gross_cagr']}, 'vol%': 13.3, 'Sharpe': 0.44, 'maxDD%': {R['ov_dd']}}},\n"
            f"                        'overlay net': {{'CAGR%': {R['ov_net_cagr']}, 'vol%': 13.3, 'Sharpe': {R['ov_sharpe']}, 'maxDD%': {R['ov_dd']}}},\n"
            f"                        'buy & hold': {{'CAGR%': {R['bh_cagr']}, 'vol%': 19.6, 'Sharpe': {R['bh_sharpe']}, 'maxDD%': {R['bh_dd']}}}}}).T\n"
            f"    sw_n = {R['switches']}\n"
            "fig, ax = plt.subplots(1, 2, figsize=(10, 4.0))\n"
            "ax[0].bar(tab.index, tab['CAGR%'], color=[AMBER, RED, GREEN], width=.6); ax[0].set_title('CAGR %/yr'); ax[0].axhline(0,c='k',lw=1)\n"
            "ax[0].tick_params(axis='x', rotation=20)\n"
            "ax[1].bar(tab.index, tab['Sharpe'], color=[AMBER, RED, GREEN], width=.6); ax[1].set_title('Sharpe'); ax[1].axhline(0,c='k',lw=1)\n"
            "ax[1].tick_params(axis='x', rotation=20)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(tab.round(2).to_string()); print('switches:', sw_n)"
        ),
        md(
            f"> 💡 In plain words: the overlay cut the drawdown ({R['ov_dd']}% vs {R['bh_dd']}%) but at a "
            f"brutal cost to return ({R['ov_net_cagr']}%/yr vs {R['bh_cagr']}%/yr) and a *lower* Sharpe "
            f"({R['ov_sharpe']} vs {R['bh_sharpe']}) — it sat out the rebounds. `MIRAGE`."
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further — the synthetic positive control\n\n"
            "Is the engine a faithful detector, or does it always print a positive spread? Plant a "
            "*real* fragility effect (`fragility > 0`: high correlation genuinely precedes worse "
            "returns) of increasing strength and watch the HIGH−LOW spread go *negative* — averaged "
            "over 25 seeds so no single lucky seed can fake it."
        ),
        code(
            "ctrl = " + repr(R["ctrl"]) + "\n"
            "fragilities = [0.0, 0.5, 0.9, 1.4]\n"
            "res = [st.synthetic_mean_t(data, fragility=f, n_seeds=25) for f in fragilities]\n"
            "spreads = [s*100 for s,_ in res]; ts = [t for _,t in res]\n"
            "fig, ax = plt.subplots(figsize=(9, 4.3))\n"
            "ax.plot(fragilities, ts, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(-2, ls='--', c=GREY, lw=1, label='t = -2 bar (fragility)')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted fragility (larger = stronger true effect)')\n"
            "ax.set_ylabel('mean HIGH-LOW forward-return t (25 seeds)')\n"
            "ax.set_title('The harness banks a planted fragility effect — flat at the null, past -2 as it grows')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for f,(s,t) in zip(fragilities, res): print(f'fragility {f:+.2f} -> mean spread {s*100:+.2f}% | mean t {t:+.2f}')"
        ),
        md(
            "The mean *t* is ≈ 0 at the null and falls past −2 as the planted fragility grows — so the "
            "engine is a faithful detector *with the right sign convention*. The real-tape result is "
            "therefore a statement about **this cross-asset tape over 2007-2026**: rising average "
            "correlation is a coincident stress gauge that, as a forward return signal, is the *wrong "
            "sign*. For the single-pair and cross-sectional cousins see "
            "[245 Oil-Equity-Correlation](../../245-oil-equity-correlation/) and "
            "[502 Betting-Against-Correlation](../../502-betting-against-correlation/)."
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
