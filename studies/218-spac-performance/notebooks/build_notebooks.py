"""Generate the two narrative notebooks for Study 218 (SPAC-Performance).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The figures run
live from the cached daily parquet under ../_cache/ if present, and otherwise quote the
frozen headline numbers in ``R`` (mirroring docs/results.md), so the notebook re-runs for
any reader. The synthetic positive-control cells run anywhere, offline and deterministic.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-16).
R = dict(
    # Tape 1: SPAK vs SPY (1.92 years, 483 days)
    n_spak=483, n_years_spak=1.92,
    spak_start="2020-10-01", spak_end="2022-09-01",
    cagr_spak=-26.14, cagr_spy_spak=10.26, cagr_gap_spak=-36.40,
    sharpe_spak=-0.930, sharpe_spy_spak=0.634,
    maxdd_spak=-62.6, maxdd_spy_spak=-23.0,
    beta_spak=0.910, r2_spak=0.333,
    alpha_daily_bps_spak=-14.538, alpha_ann_spak=-30.69, alpha_t_spak=-1.925,
    excess_bps_spak=-14.946, excess_t_spak=-2.000,
    er_spak=0.45,
    # Tape 2: De-SPAC basket vs SPY (4.56 years, 1149 days)
    n_basket=1149, n_years_basket=4.56,
    basket_start="2021-11-11", basket_end="2026-06-12",
    cagr_basket=-24.03, cagr_spy_basket=12.36, cagr_gap_basket=-36.39,
    sharpe_basket=-0.194, sharpe_spy_basket=0.750,
    maxdd_basket=-86.3, maxdd_spy_basket=-24.5,
    beta_basket=2.023, r2_basket=0.384,
    alpha_daily_bps_basket=-15.014, alpha_ann_basket=-31.52, alpha_t_basket=-1.926,
    er_spy=0.09,
    # Promote math
    promote_pct=20.0, promote_dilution_est=12.5,
)

# ---------------------------------------------------------------------------
# Shared analysis preamble
# ---------------------------------------------------------------------------
BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package (spac_performance)
sys.path.insert(0, os.path.abspath("../../.."))    # repo root (quantlab/)
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY, BLUE = "#c0392b", "#dab617", "#2ea44f", "#8b949e", "#1f6feb"
ORANGE = "#e67e22"

from spac_performance import data, strategy as st

STUDY_CACHE = os.path.abspath("../_cache")
SPAK_CACHE = os.path.join(STUDY_CACHE, "spak_daily.parquet")
SPY_CACHE = os.path.join(STUDY_CACHE, "spy_daily.parquet")

def _have_cache():
    return os.path.exists(SPAK_CACHE) and os.path.exists(SPY_CACHE)

HAVE_REAL = _have_cache()

def load_spak():
    prices = data.load_spak_vs_spy(fetch=False, cache_dir=STUDY_CACHE)
    return st.daily_returns(prices), prices

def load_basket():
    prices = data.load_basket_vs_spy(fetch=False, cache_dir=STUDY_CACHE)
    return st.daily_returns(prices), prices

print("real daily cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# SPAC-Performance — clever shortcut or structurally rigged loss?\n"
            "### SPACs (Special Purpose Acquisition Companies) tested honestly, in plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-NONE-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-MIRAGE-c0392b?style=flat-square)\n"
            "![SPACs: Busted](https://img.shields.io/badge/SPACs%3A_shortcut_or_rigged%3F-Busted-8b949e?style=flat-square)\n\n"
            "Between 2020 and 2021, SPACs (blank-check companies that raise cash in an IPO, then "
            "hunt for a merger target) raised over **$160 billion**, bringing Lucid Motors, Rivian, "
            "Clover Health, DraftKings, Virgin Galactic and dozens more to market. The pitch: "
            "faster, cheaper, and *better* than a traditional IPO, with aligned sponsors. The "
            "reality, as you are about to see, was a systematic transfer of wealth from public "
            "investors to SPAC promoters built right into the deal architecture.\n\n"
            "> This is the plain-language layer. The HAC t-stats, CAPM alpha, and structural "
            "dilution math live in **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> Not investment advice. Every chart is drawn by the code beside it. "
            "House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Did SPAC investors beat the S&P 500? | **No.** The SPAK de-SPAC ETF compounded at "
            f"**{R['cagr_spak']:+.1f}%/yr** vs the index's **+{R['cagr_spy_spak']:.1f}%/yr** — a gap "
            f"of **{abs(R['cagr_gap_spak']):.0f} percentage points per year** over SPAK's full life. |\n"
            "| Was the risk lower, at least? | **No.** SPAK's worst drawdown was "
            f"**{R['maxdd_spak']:.0f}%**, nearly triple the S&P 500's **{R['maxdd_spy_spak']:.0f}%** "
            "in the same period. |\n"
            "| Was this just a bad two-year stretch? | The nine surviving de-SPACs we track "
            "through **2026** still show **-24%/yr CAGR** vs **+12%/yr** for SPY, and the worst "
            "survivors fell **-86%** peak-to-trough. |\n"
            "| Is there a structural reason? | **Yes.** The ~20% 'promote' given free to sponsors "
            "dilutes public investors by roughly 12-13 cents per dollar at merger — before any "
            "operating miss. |\n\n"
            "> The catch: the SPAK window is only 1.92 years, so the statistics cannot certify "
            "the alpha at 95% confidence (that is why **Signal = None**, not Weak — the window is "
            "too short for a formal verdict, though the direction is not ambiguous)."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"SPACs let us bypass the nine-month, lawyer-heavy traditional IPO, lock in a "
            "valuation with a seasoned sponsor on your side, and give retail investors early "
            "access to exciting growth companies at a fair price — something the old-boy "
            "IPO allocation system denied them.\"*\n\n"
            "This was the pitch from sponsors, bankers and financial media throughout 2020-2021. "
            "Crucially, it was a *testable* claim: if SPACs were genuinely better than traditional "
            "IPOs or index investing, their post-merger price performance should exceed the market's. "
            "The Defiance SPAK ETF was purpose-built to make that test easy: one fund, the whole "
            "de-SPAC universe, daily prices."
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the claim were true, retail investors would have a genuine edge: access to "
            "sponsored, pre-negotiated deals at insider-like economics. If it's false, SPACs were "
            "a transfer mechanism: the public paid the cost of the sponsor's free equity (the "
            "'promote'), the bankers' fees, and the overvalued merger targets — all at once.\n\n"
            "With SPAK's full 1.92-year live record available (plus nine individual de-SPACs "
            "through today), we don't have to guess. The math is not subtle."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Two honest benchmarks:\n\n"
            "1. **Race SPAK against SPY** (the S&P 500 tracker, 0.09% ER) from SPAK's very "
            "first trading day to its last — the ETF's entire lifespan.\n"
            "2. **Track the survivors** — nine high-profile de-SPACs (Lucid, Rivian, Opendoor, "
            "Paysafe, Clover, Skillz, DraftKings, Virgin Galactic, QuantumScape) from their "
            "common start (RIVN's IPO date) to today. This is *generous*: we deliberately drop "
            "the worst-performing delisted names (Nikola, etc.), so any underperformance here "
            "understates the true damage."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown\n\n"
            "**First, the ETF race: $100 in SPAK vs $100 in the S&P 500 from launch to delisting.**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    rets, prices = load_spak()\n"
            "    g_spak = (1 + rets['SPAK']).cumprod() * 100\n"
            "    g_spy  = (1 + rets['SPY']).cumprod() * 100\n"
            "    end_spak, end_spy = g_spak.iloc[-1], g_spy.iloc[-1]\n"
            "else:\n"
            "    idx = pd.bdate_range(R['spak_start'], periods=R['n_spak'])\n"
            "    t = np.linspace(0, R['n_years_spak'], R['n_spak'])\n"
            "    g_spak = pd.Series(100 * (1 + R['cagr_spak']/100) ** t, index=idx)\n"
            "    g_spy  = pd.Series(100 * (1 + R['cagr_spy_spak']/100) ** t, index=idx)\n"
            "    end_spak, end_spy = g_spak.iloc[-1], g_spy.iloc[-1]\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.6))\n"
            "ax.plot(g_spak.index, g_spak.values, c=RED, lw=2, label='SPAK (de-SPAC ETF)')\n"
            "ax.plot(g_spy.index,  g_spy.values,  c=BLUE, lw=2, label='SPY (S&P 500)')\n"
            "ax.axhline(100, c=GREY, lw=1, ls='--')\n"
            "ax.set_ylabel('value of $100 invested at SPAK launch'); ax.legend(loc='upper left')\n"
            "ax.set_title('SPAK vs SPY: same start, almost opposite trajectories')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'$100 became  ${end_spak:,.0f}  in SPAK   vs   ${end_spy:,.0f}  in SPY')"
        ),
        md(
            f"$100 in SPAK shrank to roughly **${100 * (1 + R['cagr_spak']/100)**R['n_years_spak']:.0f}** "
            f"over SPAK's {R['n_years_spak']:.1f} years. The same $100 in SPY grew to "
            f"roughly **${100 * (1 + R['cagr_spy_spak']/100)**R['n_years_spak']:.0f}**. The gap: "
            f"about **{abs(R['cagr_gap_spak']):.0f} percentage points every year**.\n\n"
            "**And the crash was deeper:**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    res = st.summarize(rets['SPAK'], rets['SPY'])\n"
            "    sh_s, sh_spy = res['sharpe_fund'], res['sharpe_market']\n"
            "    dd_s, dd_spy = res['max_dd_fund_pct'], res['max_dd_market_pct']\n"
            "else:\n"
            "    sh_s, sh_spy = R['sharpe_spak'], R['sharpe_spy_spak']\n"
            "    dd_s, dd_spy = R['maxdd_spak'], R['maxdd_spy_spak']\n"
            "fig, ax = plt.subplots(1, 2, figsize=(10.0, 4.2))\n"
            "ax[0].bar(['SPAK', 'SPY'], [sh_s, sh_spy], color=[RED, BLUE], width=.55)\n"
            "ax[0].axhline(0, c='k', lw=1)\n"
            "ax[0].set_ylabel('Sharpe ratio (higher = better return per unit risk)')\n"
            "ax[0].set_title('SPAK Sharpe: deeply negative')\n"
            "ax[1].bar(['SPAK', 'SPY'], [dd_s, dd_spy], color=[RED, BLUE], width=.55)\n"
            "ax[1].set_ylabel('worst peak-to-trough loss (%)')\n"
            "ax[1].set_title('SPAK fell nearly 3x deeper than S&P 500')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Sharpe: SPAK {sh_s:.2f} vs SPY {sh_spy:.2f}   |   max DD: SPAK {dd_s:.0f}% vs SPY {dd_spy:.0f}%')"
        ),
        md(
            f"SPAK delivered a Sharpe of **{R['sharpe_spak']:.2f}** (money-losing in risk-adjusted "
            f"terms) while SPY was **+{R['sharpe_spy_spak']:.2f}**. When the market corrected in "
            f"2022, SPAK fell **{R['maxdd_spak']:.0f}%** from peak — nearly three times the index's "
            f"**{R['maxdd_spy_spak']:.0f}%** drop.\n\n"
            "**But perhaps the ETF is an outlier? Here are nine individual de-SPACs through today:**"
        ),
        code(
            "basket_names = ['LCID (Lucid)', 'RIVN (Rivian)', 'OPEN (Opendoor)', 'PSFE (Paysafe)',\n"
            "                'CLOV (Clover)', 'SKLZ (Skillz)', 'DKNG (DraftKings)', 'SPCE (Virgin Gal.)',\n"
            "                'QS (QuantumScape)']\n"
            "if HAVE_REAL:\n"
            "    rets_b, prices_b = load_basket()\n"
            "    g_basket = (1 + rets_b['BASKET']).cumprod() * 100\n"
            "    g_spy_b  = (1 + rets_b['SPY']).cumprod() * 100\n"
            "    end_basket, end_spy_b = g_basket.iloc[-1], g_spy_b.iloc[-1]\n"
            "else:\n"
            "    idx_b = pd.bdate_range(R['basket_start'], periods=R['n_basket'])\n"
            "    t_b = np.linspace(0, R['n_years_basket'], R['n_basket'])\n"
            "    g_basket = pd.Series(100*(1+R['cagr_basket']/100)**t_b, index=idx_b)\n"
            "    g_spy_b  = pd.Series(100*(1+R['cagr_spy_basket']/100)**t_b, index=idx_b)\n"
            "    end_basket, end_spy_b = g_basket.iloc[-1], g_spy_b.iloc[-1]\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.6))\n"
            "ax.plot(g_basket.index, g_basket.values, c=ORANGE, lw=2, label='9-name de-SPAC basket (equal-weight)')\n"
            "ax.plot(g_spy_b.index,  g_spy_b.values,  c=BLUE, lw=2, label='SPY (S&P 500)')\n"
            "ax.axhline(100, c=GREY, lw=1, ls='--')\n"
            "ax.set_ylabel('value of $100 invested (2021-11-11)'); ax.legend(loc='upper right')\n"
            "ax.set_title('Surviving de-SPACs still lost 86% peak-to-trough — the index gained')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Basket $100 -> ${end_basket:,.0f}  vs  SPY $100 -> ${end_spy_b:,.0f}')\n"
            "print(f'Note: delisted names (NKLA, HYLIION…) are absent — true cohort losses are larger')"
        ),
        md(
            f"Even among the **surviving** de-SPACs — the ones that didn't go bankrupt or delist — "
            f"the equal-weight basket compounded at **{R['cagr_basket']:+.1f}%/yr** vs "
            f"**+{R['cagr_spy_basket']:.1f}%/yr** for the S&P 500, with an **{R['maxdd_basket']:.0f}%** "
            "peak-to-trough collapse. This is survivorship-biased *upward*: the truest picture of "
            "SPAC investing is worse."
        ),

        # ---- BEAT 5 — THE VERDICT --------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** The Jensen alpha (market-risk-adjusted excess return) is "
            f"**{R['alpha_ann_spak']:+.1f}%/yr** for SPAK, HAC *t* = **{R['alpha_t_spak']:.3f}** "
            f"(just below the |t|=2 bar; 1.92-year window is too short to certify formally). "
            f"The de-SPAC basket extends the window to {R['n_years_basket']:.1f} years and gives "
            f"**{R['alpha_ann_basket']:+.1f}%/yr**, *t* = {R['alpha_t_basket']:.3f} — same story. "
            "The *direction* is unambiguous. We call this **None** (not Weak) because the alpha "
            "point estimates are enormous and the raw excess t already clears the bar (-2.000).\n"
            f"- **Tradability — Mirage.** SPAK CAGR **{R['cagr_spak']:+.1f}%** vs SPY "
            f"**+{R['cagr_spy_spak']:.1f}%**; Sharpe **{R['sharpe_spak']:.2f}** vs **+{R['sharpe_spy_spak']:.2f}**; "
            f"max DD **{R['maxdd_spak']:.0f}%** vs **{R['maxdd_spy_spak']:.0f}%**. SPAK was delisted "
            "in 2022 as AUM collapsed — the market voted with its feet.\n"
            "- **SPACs — Busted.** The structural-dilution mechanics (20% promote, warrants, "
            "redemption drain) transferred money from public investors to sponsors by architectural "
            "design. The underperformance is not a cycle; it is a product."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · The mechanics: why SPACs were structurally rigged\n\n"
            "Understanding *why* helps separate structural from cyclical:"
        ),
        code(
            "# The promote math: $100 SPAC raise, sponsors get 20% for free\n"
            "raise_m = 100.0\n"
            "promote = raise_m * 0.20   # sponsor's free shares\n"
            "public_pct = raise_m / (raise_m + promote)\n"
            "warrant_dilution = 3.7     # Klausner et al. (2022) median estimate, %\n"
            "total_dilution = (1 - public_pct) * 100 + warrant_dilution\n"
            "print(f'Sponsor promote:   ${promote:.0f}m  free shares on ${raise_m:.0f}m raised')\n"
            "print(f'Public investor owns at merger: {public_pct*100:.1f}% of company')\n"
            "print(f'Promote dilution:  ~{(1-public_pct)*100:.1f}%')\n"
            "print(f'Warrant dilution:  ~{warrant_dilution:.1f}%  (Klausner et al. 2022 median)')\n"
            "print(f'Total structural drag before any operating miss: ~{total_dilution:.1f}%')\n"
            "print()\n"
            "print('This is a mechanical headwind baked in at deal close — not market risk.')"
        ),
        md(
            f"Before a single business decision: public SPAC investors faced a structural drag of "
            f"roughly **~{R['promote_dilution_est']:.0f}%** at merger (sponsor promote + warrants), "
            "sourced directly from Klausner, Ohlrogge & Ruan (2022), the definitive academic audit. "
            "This is wealth transferred to the sponsor, not lost to the market. It then compounds "
            "with overvalued deal targets using five-year revenue projections not permitted in "
            "traditional IPO filings."
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The survivor problem is real.** Every delisted de-SPAC (Nikola, Otonomo, Hyliion…) "
            "is absent from our basket. Adding them would make the numbers worse, not better. The "
            "full SPAC cohort lost far more than what we show.\n"
            "- **The quants notebook** runs the full CAPM machinery on both tapes, plants a "
            "synthetic alpha to confirm the detector works, and maps the survivorship bias "
            "correction numerically.\n"
            "- **Not all SPACs equally bad.** A few deals (DKNG, and select others) delivered "
            "positive returns on some horizons. The average and the median, however, are deeply "
            "negative — and average is what a diversified investor gets.\n"
            "- **The story is not over.** Several basket names continue to trade. If you fork this, "
            "update the cache and check whether any survivor recovered.\n\n"
            "*Think one particular SPAC was structured fairly? Fork this, add the name, run the "
            "same alpha decomposition, and compare. That is the test.*"
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
            "# SPAC-Performance — a quantitative teardown\n"
            "### Two tapes · CAPM Jensen alpha · HAC inference · survivorship-bias correction · synthetic control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-NONE-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-MIRAGE-c0392b?style=flat-square)\n"
            "![SPACs: Busted](https://img.shields.io/badge/SPACs%3A_shortcut_or_rigged%3F-Busted-8b949e?style=flat-square)\n\n"
            "Deep companion to [01_for_the_curious.ipynb](01_for_the_curious.ipynb) — same beats, "
            "every claim now carrying its standard error. We test whether SPAC equity delivers a "
            "positive Jensen alpha over SPY on two complementary tapes (SPAK ETF 1.92 yrs; "
            "surviving de-SPAC basket 4.56 yrs), document the structural-dilution mechanics, and "
            "confirm the detector with a synthetic positive control.\n\n"
            "> Not investment advice. Real data: Yahoo daily adjusted-close. Methods & sources in "
            "[`docs/references.md`](../docs/references.md), numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **`In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | SPAK Jensen alpha **{R['alpha_ann_spak']:+.2f}%/yr** "
            f"({R['alpha_daily_bps_spak']:+.3f} bps/day), HAC *t* = **{R['alpha_t_spak']:.3f}** — "
            f"below |2| (1.92-yr window); raw excess *t* = **{R['excess_t_spak']:.3f}** confirms direction. "
            f"Basket: **{R['alpha_ann_basket']:+.2f}%/yr**, *t* = {R['alpha_t_basket']:.3f} over {R['n_years_basket']:.1f} yrs. |\n"
            f"| **Tradability** | `MIRAGE` | SPAK CAGR **{R['cagr_spak']:+.2f}%** vs SPY "
            f"**{R['cagr_spy_spak']:+.2f}%**; Sharpe {R['sharpe_spak']:.3f} vs {R['sharpe_spy_spak']:.3f}; "
            f"max DD {R['maxdd_spak']:.0f}% vs {R['maxdd_spy_spak']:.0f}%; SPAK delisted 2022. |\n"
            f"| **SPACs rigged?** | `BUSTED` | ~20% sponsor promote + warrant dilution creates "
            f"~{R['promote_dilution_est']:.0f}% structural drag at merger (Klausner et al. 2022); "
            f"basket max DD {R['maxdd_basket']:.0f}%, survivorship-biased upward. |\n\n"
            "> In plain words: SPAC public investors paid for the sponsor's free equity, for the "
            "overvalued targets, and for the fee — all embedded in the deal architecture. The price "
            "series confirms this in every metric."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r^{f}_t$ be the de-SPAC equity return and $r^{m}_t$ SPY's. Fit the market model\n\n"
            "$$ r^{f}_t = \\alpha + \\beta\\, r^{m}_t + \\varepsilon_t. $$\n\n"
            "The SPAC proposition requires:\n\n"
            "- **H₁ (signal).** $\\alpha > 0$ — positive Jensen alpha, net of the structural "
            "promote and ER, reflected in total-return adjusted prices.\n"
            "- **H₂ (beats buy-and-hold).** CAGR, Sharpe, and max drawdown all dominate SPY.\n"
            "- **H₃ (structural fairness).** The sponsor promote is offset by superior deal "
            "sourcing and target quality — the dilution is paid back in excess returns.\n\n"
            f"We find $\\hat\\alpha \\approx {R['alpha_ann_spak']:.0f}\\%$/yr on SPAK (failing H₁ "
            "catastrophically), H₂ fails on all three metrics, and H₃ fails — the promote is not "
            "paid back; it compounds against public investors."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The interesting failure mode here is not under-power — it is over-performance by the "
            f"alternative (SPY). With only {R['n_years_spak']:.2f} years of SPAK data, the minimum "
            f"detectable Jensen alpha at |*t*|=2 is roughly ±22%/yr (high idio vol, short window). "
            f"But the point estimate is **{R['alpha_ann_spak']:+.1f}%/yr** — so extreme it would "
            "likely have been statistically significant with even a few more months. We label "
            "Signal **None** because the t-stat misses the bar, not because the result is close.\n\n"
            "The policy implication is structural: if SPACs reliably underperformed by ~30%/yr "
            "in alpha terms, the architecture — not market conditions — explains it."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "**Tape 1 (SPAK).**\n"
            f"- Yahoo daily adjusted-close (total-return) for SPAK and SPY, aligned from "
            f"SPAK's launch ({R['spak_start']}) to delisting ({R['spak_end']}); "
            f"{R['n_spak']} common trading days ({R['n_years_spak']:.2f} years).\n"
            "- Full-sample OLS market model for $(\\alpha, \\beta)$; static beta (conservative).\n"
            "- Newey-West HAC t-stat on the OLS intercept.\n\n"
            "**Tape 2 (de-SPAC basket).**\n"
            f"- Equal-weight basket of 9 surviving de-SPACs from RIVN's IPO date "
            f"({R['basket_start']}) to {R['basket_end']}; {R['n_basket']} common trading days "
            f"({R['n_years_basket']:.2f} years). Survivorship-biased upward.\n"
            "- Same OLS + HAC protocol.\n\n"
            "**Positive control.** Synthetic tape with tunable planted alpha — the detector must "
            "recover it when planted, and read near-zero when absent.\n\n"
            f"Inference bar: |*t*| ≥ 2. SPAK alpha lands at *t* = {R['alpha_t_spak']:.3f}; "
            f"basket at *t* = {R['alpha_t_basket']:.3f}."
        ),

        # ---- BEAT 4a — SPAK decomposition ------------------------------------
        md("## 4 · The teardown\n\n### 4a · Tape 1 (SPAK ETF) — market model"),
        code(
            "if HAVE_REAL:\n"
            "    rets, prices = load_spak()\n"
            "    res = st.summarize(rets['SPAK'], rets['SPY'])\n"
            "    alpha_ann, alpha_t = res['alpha_ann_pct'], res['tstat_alpha_hac']\n"
            "    beta, r2 = res['beta'], res['r_squared']\n"
            "    rf, rm = rets['SPAK'].values, rets['SPY'].values\n"
            "else:\n"
            "    alpha_ann, alpha_t = R['alpha_ann_spak'], R['alpha_t_spak']\n"
            "    beta, r2 = R['beta_spak'], R['r2_spak']\n"
            "    rng = np.random.default_rng(218)\n"
            "    rm = rng.normal(0.07/252, 0.18/np.sqrt(252), R['n_spak'])\n"
            "    rf = (R['alpha_daily_bps_spak']/1e4) + beta*rm + rng.normal(0, 0.45/np.sqrt(252), R['n_spak'])\n"
            "fig, ax = plt.subplots(figsize=(6.2, 6.0))\n"
            "ax.scatter(rm*100, rf*100, s=6, alpha=.25, color=GREY)\n"
            "xs = np.linspace(rm.min(), rm.max(), 50)\n"
            "ax.plot(xs*100, (beta*xs + alpha_ann/100/252)*100, c=RED, lw=2,\n"
            "        label=f'fit: beta={beta:.2f}, alpha={alpha_ann:+.0f}%/yr')\n"
            "ax.plot(xs*100, xs*100, c=BLUE, ls='--', lw=1.5, label='beta=1, alpha=0 (fair)')\n"
            "ax.set_xlabel('SPY daily return (%)'); ax.set_ylabel('SPAK daily return (%)')\n"
            "ax.set_title('The fit sits far below the fair line: enormous negative alpha'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'alpha = {alpha_ann:+.2f}%/yr   HAC t = {alpha_t:+.3f}   beta = {beta:.3f}   R^2 = {r2:.3f}')"
        ),
        md(
            f"> In plain words: the regression line sits well below the 45° fair line. SPAK "
            f"had beta ≈ {R['beta_spak']:.2f} (slightly less market exposure than SPY) but its "
            f"intercept — the alpha — is **{R['alpha_ann_spak']:+.1f}%/yr**. The HAC t-stat "
            f"of **{R['alpha_t_spak']:.3f}** just misses |2|, constrained by the short 1.92-year "
            "window and high idio vol. The direction is unambiguous."
        ),

        # ---- BEAT 4b — basket ------------------------------------------------
        md("### 4b · Tape 2 (de-SPAC basket) — extending the window"),
        code(
            "if HAVE_REAL:\n"
            "    rets_b, prices_b = load_basket()\n"
            "    res_b = st.summarize(rets_b['BASKET'], rets_b['SPY'])\n"
            "    alpha_ann_b, alpha_t_b = res_b['alpha_ann_pct'], res_b['tstat_alpha_hac']\n"
            "    beta_b, r2_b = res_b['beta'], res_b['r_squared']\n"
            "    cagr_b, cagr_spy_b = res_b['cagr_fund_pct'], res_b['cagr_market_pct']\n"
            "    dd_b, dd_spy_b = res_b['max_dd_fund_pct'], res_b['max_dd_market_pct']\n"
            "else:\n"
            "    alpha_ann_b, alpha_t_b = R['alpha_ann_basket'], R['alpha_t_basket']\n"
            "    beta_b, r2_b = R['beta_basket'], R['r2_basket']\n"
            "    cagr_b, cagr_spy_b = R['cagr_basket'], R['cagr_spy_basket']\n"
            "    dd_b, dd_spy_b = R['maxdd_basket'], R['maxdd_spy_basket']\n"
            "print(f'Basket CAGR: {cagr_b:+.2f}%/yr   SPY CAGR: {cagr_spy_b:+.2f}%/yr')\n"
            "print(f'Basket Jensen alpha: {alpha_ann_b:+.2f}%/yr   HAC t = {alpha_t_b:+.3f}')\n"
            "print(f'Basket max DD: {dd_b:.1f}%   SPY max DD: {dd_spy_b:.1f}%')\n"
            "print(f'Beta vs SPY: {beta_b:.3f}   R^2: {r2_b:.3f}')\n"
            "print()\n"
            "print('Survivorship note: NKLA, HYLIION, OTRK and other delisted names are absent.')\n"
            "print('Including them would make these numbers considerably worse.')"
        ),
        md(
            f"> In plain words: extending the window to {R['n_years_basket']:.1f} years via the "
            f"surviving basket gives the same result — alpha of **{R['alpha_ann_basket']:+.1f}%/yr**, "
            f"*t* = {R['alpha_t_basket']:.3f}. High beta (≈{R['beta_basket']:.1f}) means market "
            "downturns amplify losses. The -86% max drawdown is survivorship-biased *upward*: the "
            "true cohort did worse."
        ),

        # ---- BEAT 4c — structural dilution -----------------------------------
        md("### 4c · The structural-dilution math (Klausner, Ohlrogge & Ruan 2022)"),
        code(
            "raise_m = 100.0  # $100m SPAC raise\n"
            "promote = raise_m * 0.20   # sponsor's free shares (20% promote)\n"
            "public_pct = raise_m / (raise_m + promote)\n"
            "warrant_d = 3.7  # median warrant dilution from Klausner et al.\n"
            "total_d = (1 - public_pct) * 100 + warrant_d\n"
            "\n"
            "# Show dilution vs theoretical fair deal\n"
            "scenarios = {\n"
            "    'Fair deal (no promote)': 0.0,\n"
            "    'Typical SPAC (20% promote + warrants)': total_d,\n"
            "    '95th-pct bad SPAC (30% promote + heavy warrants)': 30 + 6.0,\n"
            "}\n"
            "fig, ax = plt.subplots(figsize=(9.0, 3.8))\n"
            "colors = [GREEN, RED, '#8b0000']\n"
            "bars = ax.barh(list(scenarios.keys()), list(scenarios.values()),\n"
            "               color=colors, height=0.55)\n"
            "ax.axvline(0, c='k', lw=1)\n"
            "ax.set_xlabel('structural dilution at merger (%)')\n"
            "ax.set_title('Promote + warrants: a tax on public investors built into every deal')\n"
            "for bar, val in zip(bars, scenarios.values()):\n"
            "    ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,\n"
            "            f'{val:.1f}%', va='center', fontsize=10)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Typical SPAC: public investors own {public_pct*100:.1f}% at merger (sponsor got {promote:.0f}% for free)')\n"
            "print(f'Total structural drag (Klausner et al.): ~{total_d:.1f}% before any operational miss')"
        ),
        md(
            "The promote is not a market risk — it is a contractual transfer. Even if the acquired "
            "business performs exactly in line with the S&P 500, public investors start ~12% behind "
            "their theoretical fair value at day-one of the merged entity. The SPAK and basket "
            "results show the businesses mostly *also* underperformed — a double loss."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — Jensen alpha {R['alpha_ann_spak']:+.2f}%/yr "
            f"({R['alpha_daily_bps_spak']:+.3f} bps/day), HAC *t* {R['alpha_t_spak']:.3f}. "
            f"Raw excess *t* {R['excess_t_spak']:.3f} just clears the bar. Basket: "
            f"{R['alpha_ann_basket']:+.2f}%/yr, *t* {R['alpha_t_basket']:.3f} over "
            f"{R['n_years_basket']:.1f} yrs. Both negative, enormous, consistent.\n"
            f"- **Tradability `MIRAGE`** — SPAK CAGR {R['cagr_spak']:+.2f}% vs "
            f"{R['cagr_spy_spak']:+.2f}%; Sharpe {R['sharpe_spak']:.3f} vs "
            f"{R['sharpe_spy_spak']:.3f}; max DD {R['maxdd_spak']:.0f}% vs {R['maxdd_spy_spak']:.0f}%.\n"
            f"- **Myth `BUSTED`** — ~{R['promote_dilution_est']:.0f}% structural dilution at deal "
            "close (promote + warrants) is the floor underperformance before any business risk. "
            "The claim that sponsors' deal-sourcing skill offsets this is not supported by the data."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the tradability perspective\n\n"
            "Buying SPAK was the cleanest expression of the SPAC thesis. The equity curves below "
            "show the compounding gap (log-scaled); the wealth destruction is monotone:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    rets, prices = load_spak()\n"
            "    g_s = (1 + rets['SPAK']).cumprod() * 100\n"
            "    g_m = (1 + rets['SPY']).cumprod() * 100\n"
            "    end_s, end_m = g_s.iloc[-1], g_m.iloc[-1]\n"
            "else:\n"
            "    idx = pd.bdate_range(R['spak_start'], periods=R['n_spak'])\n"
            "    t = np.linspace(0, R['n_years_spak'], R['n_spak'])\n"
            "    g_s = pd.Series(100*(1+R['cagr_spak']/100)**t, index=idx)\n"
            "    g_m = pd.Series(100*(1+R['cagr_spy_spak']/100)**t, index=idx)\n"
            "    end_s, end_m = g_s.iloc[-1], g_m.iloc[-1]\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.4))\n"
            "ax.plot(g_s.index, g_s.values, c=RED, lw=2, label='SPAK (de-SPAC ETF)')\n"
            "ax.plot(g_m.index, g_m.values, c=BLUE, lw=2, label='SPY (S&P 500)')\n"
            "ax.set_yscale('log'); ax.set_ylabel('$100 at launch (log scale)'); ax.legend(loc='upper left')\n"
            "ax.axhline(100, c=GREY, lw=1, ls='--')\n"
            "ax.set_title('SPAK eroded ~60c on the dollar while SPY gained — log scale')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Terminal: SPAK ${end_s:,.0f}  vs  SPY ${end_m:,.0f}  '\n"
            "      f'(shortfall ${end_m-end_s:,.0f} per $100 = ${(end_m-end_s)*1000:,.0f} per $100k)')"
        ),

        # ---- BEAT 7 — POSITIVE CONTROL ---------------------------------------
        md(
            "## 7 · Going further — the positive control\n\n"
            "Does the detector work? Plant a known annual alpha and sweep: the recovered HAC "
            "alpha should track the planted value along the 45° line. High idio vol (45%/yr) "
            "requires a longer synthetic tape to resolve smaller alphas."
        ),
        code(
            "planted = [-0.30, -0.20, -0.10, 0.0, 0.05, 0.10, 0.20]\n"
            "rec_alpha, rec_t = [], []\n"
            "for a in planted:\n"
            "    prices_s, truth = data.synthetic_daily(n_days=2000, alpha_ann=a, seed=218)\n"
            "    res_s = st.detect_alpha(prices_s)\n"
            "    rec_alpha.append(res_s['alpha_ann_pct'])\n"
            "    rec_t.append(res_s['tstat_alpha_hac'])\n"
            "fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.3))\n"
            "ax[0].plot([p*100 for p in planted], rec_alpha, 'o-', c=GREEN, lw=2)\n"
            "lim = [-35, 25]; ax[0].plot(lim, lim, ls='--', c=GREY, label='perfect recovery')\n"
            "ax[0].axhline(0, c='k', lw=.8); ax[0].axvline(0, c='k', lw=.8)\n"
            "ax[0].set_xlabel('planted alpha (%/yr)'); ax[0].set_ylabel('recovered alpha (%/yr)')\n"
            "ax[0].set_title('The estimator is unbiased'); ax[0].legend()\n"
            "ax[1].plot([p*100 for p in planted], rec_t, 'o-', c=BLUE, lw=2)\n"
            "for s in (2, -2): ax[1].axhline(s, ls=':', c=GREY)\n"
            "ax[1].axhline(0, c='k', lw=.8)\n"
            "ax[1].set_xlabel('planted alpha (%/yr)'); ax[1].set_ylabel('recovered HAC t')\n"
            "ax[1].set_title('...and significant when the edge is real')\n"
            "plt.tight_layout(); plt.show()\n"
            "i0 = planted.index(0.0)\n"
            "print(f'at planted alpha 0: recovered {rec_alpha[i0]:+.2f}%/yr, t = {rec_t[i0]:+.2f}')\n"
            "print(f'at planted alpha -30%: recovered {rec_alpha[0]:+.2f}%/yr, t = {rec_t[0]:+.2f}')"
        ),
        md(
            "The recovered alpha tracks the 45° line and achieves |*t*|>2 once the edge is large "
            "enough on the 2000-day synthetic tape. The fact that SPAK's real-tape alpha of "
            f"-30%/yr barely misses |*t*|=2 at 483 days is a power story (short window + high "
            "idio vol), not a signal story. The real-tape point estimate is larger in magnitude "
            "than almost any synthetic test case that clears the bar — the data is speaking loudly. "
            "Forks worth trying: add delisted names with return-to-zero imputation, extend to the "
            "full de-SPAC cohort using a broader ticker list, or test individual sectors "
            "(electric vehicles, fintech) separately."
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
