"""Generate the two narrative notebooks for Study 623 (IPO Long-Run Underperformance).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Track A (Ritter's published cohort table) is
hardcoded in the package, so those cells always run. Track B cells read the cached ETF tape
under ../_cache/ and otherwise quote the frozen headline numbers in ``R`` (mirroring
docs/results.md). The synthetic control runs anywhere with no network.
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


# Frozen headline numbers — mirror of docs/results.md (as-of 2026-06-30, fingerprint
# 65b022ff4f0e; Ritter Table 19 updated 2026-02-16; ETF tape Nov 2013 -> Jun 2026).
R = dict(
    as_of="2026-06-30", fingerprint="65b022ff4f0e",
    # Track A — Ritter cohorts (mean %/3yr, NW t, N-weighted %, share negative %)
    n_cohorts=45, n_ipos=9253, first_day_pct=18.9,
    raw=(22.50, 3.20, 19.13, 22.2),
    mkt=(-14.20, -2.26, -20.51, 71.1),
    style=(-8.46, -1.52, -8.87, 66.7),
    mkt_extrunc=(-12.03, -2.18), style_extrunc=(-6.60, -1.37),
    # Track B — live tape
    n_months=152, years=12.7, start="2013-11-30", end="2026-06-30",
    growth=dict(IPO=(3.10, 9.34), SPY=(5.27, 14.02), IWM=(3.25, 9.74), IWO=(3.37, 10.07)),
    rel_peak=1.45, rel_peak_date="2021-01", rel_end=0.59, rel_dd=-67.5, share_beat=50.0,
    # (spread bps/mo, spread ann %, spread t, alpha bps/mo, alpha ann %, alpha t, beta)
    vs_spy=(-16.36, -1.95, -0.31, -46.71, -5.46, -0.99, 1.293),
    vs_iwm=(8.39, 1.01, 0.17, 8.86, 1.07, 0.18, 0.994),
    vs_iwo=(4.79, 0.58, 0.12, 0.23, 0.03, 0.01, 1.055),
    two_factor=(-4.66, -0.56, -0.12, 1.091, 0.999),   # alpha bps, ann %, t, b_mkt, b_style
    # tradability: (cost bps, borrow %/yr, gross bps, gross t, net bps, net ann %, net t, maxDD %)
    overlay=[(5.0, 1.0, 16.36, 0.31, 7.63, 0.92, 0.15, -57.4),
             (10.0, 1.0, 16.36, 0.31, 7.23, 0.87, 0.14, -57.5),
             (5.0, 2.0, 16.36, 0.31, -0.70, -0.08, -0.01, -59.5)],
    # synthetic: (planted bps/mo, measured bps/mo, mean t, rejection rate %)
    syn=[(0.0, 6.31, 0.21, 5.0), (-50.0, -43.69, -1.83, 45.0), (-100.0, -93.69, -3.88, 95.0)],
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Small-growth beta?: Confirmed](https://img.shields.io/badge/Small--growth_beta%3F-Confirmed-8b949e?style=flat-square)\n\n"
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

from ipo_long_run_underperformance import data, strategy as st

COHORTS = data.ritter_cohorts()                    # Track A: always offline (hardcoded)
HAVE_REAL = data.have_real()                       # Track B: cache-first
PANEL = data.monthly_panel(data.load_tape()) if HAVE_REAL else None
print("Ritter cohorts:", len(COHORTS), "| live ETF cache present:", HAVE_REAL,
      "| months:", (0 if PANEL is None else len(PANEL)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Do IPOs really sink for years after the party? 📉\n"
            "### Ritter's famous \"long-run IPO underperformance\" — the pop is for the flippers, "
            "the drift is for the bagholders. Is it true? Can you use it?\n\n"
            + BADGES +
            "Everyone knows the IPO story: a hot company lists, the stock \"pops\" 20% on day one, "
            "CNBC celebrates. Less advertised is the academic sequel, discovered by Jay Ritter in "
            "1991: for the **three to five years after** the party, the average IPO **lags** the "
            "rest of the market. The pop goes to the institutions who got allocations; the long slow "
            "slide goes to whoever bought in the aftermarket and held.\n\n"
            "We test it two ways: Ritter's own continuously-updated scorecard of **9,253 US IPOs "
            "over 45 years**, and a **live 12.7-year experiment** — an actual ETF that does nothing "
            "but hold recent IPOs (the Renaissance IPO ETF) against the S&P 500.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, HAC lags and cost math? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ Cousins on this desk, so we don't double-count: "
            "[219-ipo-pop](../../219-ipo-pop/) is the **day-one pop**; "
            "[265-ipo-volume](../../265-ipo-volume/) uses IPO **volume as a market-timing signal**. "
            "This study is the **multi-year drift** — measured from the first closing price, never "
            "the offer price, so the pop is already out of the picture."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Did IPOs historically lag the market for 3 years? | **Yes, on the 45-year record.** "
            "The average IPO cohort lagged the market by about **−14% over 3 years** — 71% of all "
            "cohort years were negative. |\n"
            "| Is it still detectably true on a live, buyable instrument? | **Not provably.** Over "
            "12.7 years the IPO ETF lagged the S&P (**+9.3%/yr vs +14.0%/yr**) but the gap is "
            "statistically indistinguishable from noise. |\n"
            "| Is \"IPO\" even the right villain? | **Mostly no.** Compare IPOs to *other* small, "
            "fast-growing, expensive stocks and the underperformance essentially **vanishes** — in "
            "Ritter's own tables and on the live tape. The sin is the *style*, not the wrapper. |\n"
            "| Can you make money shorting new listings? | **No.** After borrow fees and costs the "
            "short earns ~**+0.9%/yr** of statistical nothing and once drew down **−57%**. |\n\n"
            "> The honest version of the legend: *\"freshly-listed stocks are usually expensive "
            "small-growth stocks, and expensive small-growth is a bad neighbourhood\"* — true, but "
            "not a money machine, and not really about IPOs."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"IPOs underperform seasoned stocks for 3-5 years after listing. The first-day pop "
            "is for the flippers; the drift is for the bagholders.\"*\n\n"
            "This is not folklore — it's **Ritter (1991)** and **Loughran & Ritter (1995, *The New "
            "Issues Puzzle*)**, two of the most-cited papers in finance, plus a scorecard Ritter "
            "still updates every year. The mechanisms offered: IPOs are *sold* (not bought) at "
            "peaks of optimism, lockup expiries flood the market with supply, and small investors "
            "systematically overpay for lottery-like growth stories."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If true and usable, you should never buy a recent IPO — and maybe short a basket of "
            "them. If true but *style-driven*, the lesson changes completely: the problem isn't the "
            "IPO stamp, it's that new listings are **small growth stocks bought expensive**, and "
            "you'd get the same medicine by avoiding that style anywhere. One claim indicts an "
            "*event*; the other indicts a *price tag*. Telling them apart is the whole game — and "
            "it's exactly the third axis of this study."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"**Track A — the 45-year scorecard.** Ritter tracks every qualifying US IPO "
            f"({R['n_ipos']:,} of them, 1980-2024) for 3 years from its **first closing price** "
            "(the pop already excluded), against the market and against **style-matched** seasoned "
            "firms (same size, same valuation). One average per cohort year = 45 observations, and "
            "we test whether their mean is really below zero (with statistics that respect the "
            "overlap between consecutive 3-year windows).\n\n"
            "**Track B — the live experiment.** Since 2013 you can *buy* Ritter's aftermarket window: "
            "the **Renaissance IPO ETF** holds new listings for ~2-3 years, then drops them. We race "
            "its monthly returns (fees included) against SPY, small caps (IWM) and — the style "
            "control — small **growth** (IWO), over **12.7 years**."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**The 45-year record.** Each bar is one IPO vintage: how the average IPO of that year "
            "did over the next 3 years, *relative to the market*."
        ),
        code(
            "x = COHORTS['bhr3_mkt_adj']\n"
            "fig, ax = plt.subplots(figsize=(10.5, 4.6))\n"
            "ax.bar(x.index, x.values, color=[RED if v < 0 else GREEN for v in x.values], width=.8)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('IPO cohort year'); ax.set_ylabel('avg 3-yr return vs market (%)')\n"
            "ax.set_title('Ritter\\'s scorecard: most IPO vintages lag the market over 3 years')\n"
            "plt.tight_layout(); plt.show()\n"
            "neg = (x < 0).mean() * 100\n"
            "print(f'{len(x)} cohorts 1980-2024: mean {x.mean():+.2f}%/3yr vs market, {neg:.1f}% negative')"
        ),
        md(
            f"A sea of red: **{R['mkt'][3]:.0f}% of the 45 vintages** lagged the market, averaging "
            f"**{R['mkt'][0]:+.1f}% over 3 years** (weighted by IPO counts: {R['mkt'][2]:+.1f}%). "
            "The quants notebook confirms this clears the significance bar even after accounting "
            "for overlapping windows. So the *history* is real.\n\n"
            "**But now the style question.** Ritter also compares each IPO to a *seasoned* stock of "
            "the same size and valuation — a small-growth doppelgänger that simply isn't a new "
            "listing. Watch the drag shrink:"
        ),
        code(
            "vals = [R['mkt'][0], R['style'][0]]\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.4))\n"
            "ax.bar(['vs the MARKET', 'vs a style TWIN\\n(same size & valuation)'], vals,\n"
            "       color=[RED, AMBER], width=.55)\n"
            "for i, v in enumerate(vals): ax.annotate(f'{v:+.1f}%', (i, v), ha='center', va='top')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('avg 3-yr abnormal return (%)')\n"
            "ax.set_title('Half the \"IPO curse\" is just the small-growth costume')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'market-adjusted {vals[0]:+.1f}%/3yr  ->  style-adjusted {vals[1]:+.1f}%/3yr')"
        ),
        md(
            f"Against its style twin, the average IPO lags by only **{R['style'][0]:+.1f}%/3yr** — "
            "and (quants notebook) that residual is **no longer statistically distinguishable from "
            "zero**. Brav & Gompers made exactly this point in 1997: small growth stocks that "
            "*didn't* just IPO do about as badly.\n\n"
            "**The live experiment.** Since 2013, $1 in the IPO ETF vs $1 in SPY vs $1 in the "
            "small-growth index fund:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    nav = np.exp(np.log1p(PANEL[['IPO','SPY','IWO']]).cumsum())\n"
            "    fig, ax = plt.subplots(figsize=(10.2, 5.0))\n"
            "    ax.plot(nav.index, nav['SPY'], c=GREY, lw=2, label=f\"SPY  ({R['growth']['SPY'][0]:.1f}x)\")\n"
            "    ax.plot(nav.index, nav['IWO'], c=AMBER, lw=2, label=f\"IWO small growth  ({R['growth']['IWO'][0]:.1f}x)\")\n"
            "    ax.plot(nav.index, nav['IPO'], c=RED, lw=2, label=f\"IPO ETF  ({R['growth']['IPO'][0]:.1f}x)\")\n"
            "    ax.set_yscale('log'); ax.set_ylabel('growth of $1 (log scale, total return)')\n"
            "    ax.set_title('12.7 live years: the IPO ETF lags SPY badly - but tracks small growth')\n"
            "    ax.legend(); plt.tight_layout(); plt.show()\n"
            "    for c in ('IPO','SPY','IWM','IWO'):\n"
            "        print(f\"{c}: {st.cumulative_growth(PANEL, c):.2f}x\")\n"
            "else:\n"
            "    print('(cache missing - frozen numbers)', {k: v[0] for k, v in R['growth'].items()})"
        ),
        md(
            f"The IPO ETF turned $1 into **{R['growth']['IPO'][0]:.1f}×** "
            f"(+{R['growth']['IPO'][1]:.1f}%/yr) while SPY made **{R['growth']['SPY'][0]:.1f}×** "
            f"(+{R['growth']['SPY'][1]:.1f}%/yr) — it even briefly reached **{R['rel_peak']:.2f}× "
            f"SPY's wealth** in the 2020-21 mania before a **{R['rel_dd']:.0f}%** relative collapse. "
            f"Yet it beat SPY in **{R['share_beat']:.0f}% of months** — a coin flip — and the small-"
            f"growth fund IWO landed at almost the same place ({R['growth']['IWO'][0]:.1f}×). The "
            "quants notebook shows the lag is **statistically noise** vs SPY (*t* ≈ −1) and exactly "
            "**zero** vs small growth. And shorting it? After borrow fees you keep "
            f"~**{R['overlay'][0][5]:+.1f}%/yr** of nothing, having survived a −57% drawdown."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — Weak.** The 45-year published record genuinely shows the drift vs the "
            "market, but the style-adjusted version doesn't clear the bar, and our live 12.7-year "
            "investable test can't certify any of it (*t* ≈ −1). The literature says real; the tape "
            "we can actually trade says *can't confirm*.\n"
            "- **Tradability — Mirage.** ~+0.9%/yr net of borrow at *t* = 0.15, negative at 2% "
            "borrow, −57% drawdown on the way. Nothing deployable.\n"
            "- **Small-growth beta, not IPO-ness? — Confirmed.** Style-match the IPOs and the curse "
            "evaporates — in Ritter's own tables and on the live tape. Don't fear the ticker "
            "confetti; fear the price tag."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **The pop and the drift are different trades.** The day-one pop "
            "([219-ipo-pop](../../219-ipo-pop/)) goes to allocants at the offer price — by "
            "measuring from the first close, this study shows what's left for everyone else.\n"
            "- **Why did the legend feel so strong?** The worst vintages (1999-2000, 2020-21) were "
            "manias — IPO *volume* peaks exactly when future returns are worst, which is the timing "
            "story of [265-ipo-volume](../../265-ipo-volume/).\n"
            "- **The actionable residue** is portfolio hygiene, not alpha: a recent IPO in your "
            "portfolio is a concentrated bet on expensive small-growth. Size it like one.\n\n"
            "*Think the drift is still alive and tradable? Show a style-adjusted alpha clearing "
            "t = 2 on an investable IPO basket net of borrow — then we'll talk.*"
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
            "# IPO Long-Run Underperformance — a quantitative teardown 🔬\n"
            "### NW-t cohort inference on 45 overlapping vintages · HAC spread & alpha tests on the "
            "live ETF tape vs three benchmarks · the style-absorption two-factor regression · "
            "borrow-and-cost sweeps on the short · a seed-averaged power analysis\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "Ritter's 3-5 year IPO drift is one of the most-cited results in finance — the job here "
            "is to separate the **published record** (real, but style-fragile) from what an "
            "**investable tape** can actually certify (nothing, at this sample size), and to say "
            "which villain — the IPO event or the small-growth style — the data indicts.\n\n"
            "> ⚠️ **Data note.** Track A = Ritter Table 19 (updated 2026-02-16), hardcoded from the "
            "source PDF (cached in `_cache/`): *published, pre-aggregated* data — by desk law it "
            "carries the literature story, never the REAL stamp. Track B = yfinance total-return "
            "tape of IPO/SPY/IWM/IWO + ^IRX, Nov 2013 → Jun 2026 (152 months), cache-first. Neither "
            "track is survivor-biased (Ritter follows IPOs to delisting; the ETF's NAV nets its "
            "losers), but the live window covers **2013+ only** — the named regime caveat. Numbers in "
            "[`docs/results.md`](../docs/results.md) (as-of " + R["as_of"] + ", fingerprint `"
            + R["fingerprint"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | Published record: market-adjusted **{R['mkt'][0]:+.2f}%/3yr**, "
            f"**NW t = {R['mkt'][1]:+.2f}** across 45 cohorts — but style-adjusted only "
            f"**{R['style'][0]:+.2f}%** at **t = {R['style'][1]:+.2f}**, and the live investable "
            f"tape shows vs-SPY alpha **t = {R['vs_spy'][5]:+.2f}**, style-matched alpha ≈ 0. "
            "Literature real; tape can't certify. |\n"
            f"| **Tradability** | `MIRAGE` | Short-IPO/long-SPY nets **{R['overlay'][0][5]:+.2f}%/yr** "
            f"(t = {R['overlay'][0][6]:+.2f}) at 5 bps + 1% borrow, **{R['overlay'][2][5]:+.2f}%/yr** "
            f"at 2% borrow, max drawdown **{R['overlay'][0][7]:.1f}%**. |\n"
            f"| **Small-growth beta?** | `CONFIRMED` | Ritter: −14.2% → −8.5% (t −2.26 → −1.52) after "
            f"style-matching; live: alpha {R['vs_spy'][4]:+.2f}%/yr vs SPY → "
            f"**{R['vs_iwo'][4]:+.2f}%/yr vs IWO**, two-factor alpha **t = {R['two_factor'][2]:+.2f}** "
            f"with style loading **{R['two_factor'][4]:.3f}**. |\n\n"
            "> 💡 In plain words: history shows the drift, but everything a trader could actually "
            "hold says \"you bought expensive small-growth\" — and there's no alpha left once you "
            "name it that."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $BHAR^{3y}_c$ be the equal-weighted average 3-year buy-and-hold abnormal return of "
            "IPO cohort $c$ (measured from the **first closing price** — the pop excluded by "
            "construction), against benchmark $b \\in \\{\\text{market}, \\text{style match}\\}$.\n\n"
            "- **H₁ (the drift exists).** $\\mathbb{E}[BHAR^{3y}] < 0$, significant under serial-"
            "correlation-robust inference (consecutive cohorts overlap by 2 years).\n"
            "- **H₂ (it lives on an investable tape).** A basket of recent IPOs (the Renaissance "
            "IPO ETF, which holds listings for ~2-3 years) shows negative HAC alpha vs its "
            "benchmark, 2013-2026.\n"
            "- **H₃ (it is IPO-ness, not style).** The drag survives style adjustment — matched "
            "seasoned small-growth firms (Track A) or IWO / a small-growth factor (Track B).\n\n"
            "We find **H₁ supported vs the market** (NW t = −2.26) **but not style-adjusted** "
            "(t = −1.52), **H₂ rejected at this sample size** (t = −0.99), **H₃ rejected** — the "
            "style absorbs essentially the whole drag. Hence WEAK / MIRAGE / CONFIRMED."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the honesty problems this design must survive\n\n"
            "1. **Overlapping windows.** Annual cohorts holding 3-year returns are serially "
            "correlated by construction → Newey-West with lag 3 across cohorts, not naive iid t.\n"
            "2. **Published aggregates ≠ our tape.** Table 19 is pre-averaged by Ritter from CRSP — "
            "we can re-frame it, but the desk's REAL stamp requires *our own* investable tape to "
            "clear t ≥ 2 (it is exactly how a literature darling gets an honest grade).\n"
            "3. **Benchmark sensitivity (Fama 1998).** Long-horizon BHARs flip sign with the "
            "benchmark — so every result is shown raw, market-adjusted and style-adjusted.\n"
            "4. **Truncation.** The 2023-24 cohorts have < 3 full years of returns — results are "
            "shown with and without them.\n"
            "5. **Costs where they belong.** The tradable expression is a *short* — it pays borrow "
            "monthly and one-way costs × NAV on the re-hedge, both legs counted; the spread is "
            "self-financing so the race is excess-vs-excess by construction; there is **no signal "
            "lag to apply** (always-on, calendar-known rule — documented, not forgotten)."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Track A.** Ritter Table 19: {R['n_cohorts']} cohort years, {R['n_ipos']:,} IPOs. "
            "One EW mean 3-yr BHAR per cohort; NW t (lag 3) on raw / market-adj / style-adj; "
            "N-weighted means and share-negative reported; ex-truncation robustness.\n"
            f"- **Track B.** Monthly total returns {R['start']} → {R['end']} ({R['n_months']} "
            "months). (i) self-financing spread IPO−bench, HAC t (lag 6); (ii) CAPM-style alpha of "
            "IPO excess on bench excess (^IRX risk-free), HAC intercept t; benches = SPY, IWM, IWO. "
            "(iii) two-factor: $r_{IPO}-r_f = \\alpha + \\beta_m (r_{SPY}-r_f) + \\beta_s "
            "(r_{IWO}-r_{SPY}) + \\epsilon$ — the live analogue of Ritter's style match.\n"
            "- **Tradability.** Short IPO / long SPY, borrow ∈ {1%, 2%}/yr charged monthly, one-way "
            "costs ∈ {5, 10} bps × re-hedge turnover, max drawdown reported.\n"
            "- **Control.** Synthetic beta-1.25 worlds with planted alpha ∈ {0, −50, −100} bps/mo, "
            "**20 seeds each** (desk law): null must reject ≈ 5%, drags must be recovered — and the "
            "power at Ritter-size is quantified, not assumed."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Track A — the cohort record under NW inference\n\n"
            "45 vintages, three benchmarks. The question: is the mean cohort BHAR below zero once "
            "the overlap is respected?"
        ),
        code(
            "rows = []\n"
            "for col, label in (('bhr3_raw','raw'), ('bhr3_mkt_adj','market-adj'), ('bhr3_style_adj','style-adj')):\n"
            "    s = st.cohort_stats(COHORTS, col, lags=3)\n"
            "    rows.append((label, s['mean_pct'], s['t_nw'], s['weighted_mean_pct'], s['share_negative']*100))\n"
            "    print(f\"{label:12s} mean {s['mean_pct']:+7.2f}%/3yr  NW t = {s['t_nw']:+.2f}  \"\n"
            "          f\"(N-wtd {s['weighted_mean_pct']:+.2f}%, {s['share_negative']*100:.1f}% negative)\")\n"
            "co2 = data.ritter_cohorts(drop_truncated=True)\n"
            "for col, label in (('bhr3_mkt_adj','market-adj'), ('bhr3_style_adj','style-adj')):\n"
            "    s = st.cohort_stats(co2, col, lags=3)\n"
            "    print(f'ex-truncated {label}: mean {s[\"mean_pct\"]:+.2f}%  NW t = {s[\"t_nw\"]:+.2f}  (n=43)')\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.4))\n"
            "labels = [r[0] for r in rows[1:]]; ts = [r[2] for r in rows[1:]]\n"
            "ax.bar(labels, ts, color=[RED, AMBER], width=.5)\n"
            "ax.axhline(-2, ls='--', c='k', lw=1, label='|t| = 2 bar')\n"
            "for i, r in enumerate(rows[1:]):\n"
            "    ax.annotate(f'{r[1]:+.1f}%/3yr\\nt={r[2]:+.2f}', (i, r[2]), ha='center', va='top')\n"
            "ax.set_ylabel('NW t across 45 cohorts'); ax.set_ylim(-3.4, 0.4)\n"
            "ax.set_title('The drift clears the bar vs the market - and loses it style-adjusted')\n"
            "ax.legend(); plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: vs the market the 45-year drift is solid "
            f"(**{R['mkt'][0]:+.1f}%/3yr, NW t = {R['mkt'][1]:+.2f}**, {R['mkt'][3]:.0f}% of vintages "
            f"negative, robust ex-truncation at t = {R['mkt_extrunc'][1]:+.2f}). Style-matched it "
            f"shrinks to **{R['style'][0]:+.1f}%** at **t = {R['style'][1]:+.2f}** — *below* the "
            "bar. Ritter's own tables carry Brav & Gompers' rebuttal inside them. And this is "
            "published, pre-aggregated data — the stamp is decided on Track B."
        ),
        md(
            "### 4b · Track B — the investable tape\n\n"
            "The relative wealth line (IPO/SPY) tells the story of *why* the legend survived: two "
            "manias and two collapses."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rel = (np.exp(np.log1p(PANEL['IPO']).cumsum()) / np.exp(np.log1p(PANEL['SPY']).cumsum()))\n"
            "    fig, ax = plt.subplots(figsize=(10.2, 4.6))\n"
            "    ax.plot(rel.index, rel.values, c=RED, lw=2)\n"
            "    ax.axhline(1.0, c=GREY, ls='--', lw=1)\n"
            "    ax.set_ylabel('IPO ETF wealth / SPY wealth'); ax.set_title(\n"
            "        f'Relative line: peak {R[\"rel_peak\"]:.2f}x (Jan 2021), ends {R[\"rel_end\"]:.2f}x '\n"
            "        f'- a {R[\"rel_dd\"]:.0f}% relative drawdown')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print(f'peak {rel.max():.2f}x ({rel.idxmax().date()})  end {rel.iloc[-1]:.2f}x  '\n"
            "          f'share of months IPO>SPY: {(PANEL[\"IPO\"]>PANEL[\"SPY\"]).mean()*100:.1f}%')\n"
            "else:\n"
            "    print('(cache missing)', 'peak', R['rel_peak'], 'end', R['rel_end'], 'dd%', R['rel_dd'])"
        ),
        md(
            "Now the inference: HAC (NW, lag 6) spreads and alphas against each benchmark, plus the "
            "two-factor style regression."
        ),
        code(
            "if HAVE_REAL:\n"
            "    print(f\"{'bench':6s} {'spread bps/mo':>14s} {'t':>6s} {'alpha bps/mo':>13s} {'t':>6s} {'beta':>6s}\")\n"
            "    tvals = {}\n"
            "    for b in ('SPY','IWM','IWO'):\n"
            "        sp = st.spread_stats(PANEL, b, lags=6); al = st.alpha_stats(PANEL, b, lags=6)\n"
            "        tvals[b] = al['alpha_t']\n"
            "        print(f\"{b:6s} {sp['mean_bps']:>+14.2f} {sp['t_nw']:>+6.2f} {al['alpha_bps']:>+13.2f} \"\n"
            "              f\"{al['alpha_t']:>+6.2f} {al['beta']:>6.3f}\")\n"
            "    tf = st.two_factor_alpha(PANEL, lags=6)\n"
            "    tvals['2-factor'] = tf['alpha_t']\n"
            "    print(f\"2-factor alpha {tf['alpha_bps']:+.2f} bps/mo  t = {tf['alpha_t']:+.2f}  \"\n"
            "          f\"beta_mkt = {tf['beta_mkt']:.3f}  beta_style = {tf['beta_style']:.3f}\")\n"
            "else:\n"
            "    tvals = {'SPY': R['vs_spy'][5], 'IWM': R['vs_iwm'][5], 'IWO': R['vs_iwo'][5], '2-factor': R['two_factor'][2]}\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "ks = list(tvals)\n"
            "ax.bar(ks, [tvals[k] for k in ks], color=[RED, GREY, AMBER, GREEN], width=.55)\n"
            "ax.axhline(-2, ls='--', c='k', lw=1, label='|t| = 2 bar'); ax.axhline(0, c='k', lw=.8)\n"
            "for i, k in enumerate(ks): ax.annotate(f't={tvals[k]:+.2f}', (i, tvals[k]), ha='center', va='bottom')\n"
            "ax.set_ylabel('HAC alpha t (lag 6)'); ax.set_ylim(-2.6, 1.0)\n"
            "ax.set_title('Live tape: no benchmark gets the IPO drag anywhere near significance')\n"
            "ax.legend(); plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: vs SPY the ETF *looks* dragged (alpha **{R['vs_spy'][3]:+.1f} "
            f"bps/mo ≈ {R['vs_spy'][4]:+.1f}%/yr**) but at **t = {R['vs_spy'][5]:+.2f}** it's noise — "
            f"and the beta is **{R['vs_spy'][6]:.2f}**, i.e. a leveraged market basket. Re-benchmark "
            f"to what it actually *is* (small growth): alpha vs IWO = **{R['vs_iwo'][3]:+.2f} bps/mo** "
            f"(t = {R['vs_iwo'][5]:+.2f}); two-factor alpha **{R['two_factor'][0]:+.2f} bps/mo** "
            f"(t = {R['two_factor'][2]:+.2f}) with a style loading of **{R['two_factor'][4]:.3f}**. "
            "The live IPO basket is small-growth beta wearing a ticker-confetti costume."
        ),
        md(
            "### 4c · Tradability — shorting the drift, net of frictions\n\n"
            "Always-on short-IPO / long-SPY, monthly re-hedge; borrow charged monthly on the short "
            "leg, one-way costs × NAV on the turnover (both legs)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = [st.short_overlay_stats(PANEL, 'SPY', cost_bps=cb, borrow_ann_pct=br, lags=6)\n"
            "            for cb, br in ((5.0, 1.0), (10.0, 1.0), (5.0, 2.0))]\n"
            "    rows = [(r['cost_bps'], r['borrow_ann_pct'], r['net_bps'], r['net_ann_pct'], r['net_t'],\n"
            "             r['worst_drawdown_pct']) for r in rows]\n"
            "else:\n"
            "    rows = [(o[0], o[1], o[4], o[5], o[6], o[7]) for o in R['overlay']]\n"
            "labels = [f'{int(c)} bps\\n{int(b)}% borrow' for c, b, *_ in rows]\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.4))\n"
            "ax.bar(labels, [r[3] for r in rows], color=[AMBER, AMBER, RED], width=.55)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i, r in enumerate(rows):\n"
            "    ax.annotate(f'{r[3]:+.2f}%/yr\\nt={r[4]:+.2f}', (i, r[3]), ha='center', va='bottom')\n"
            "ax.set_ylabel('net annualised return of the short spread (%)')\n"
            "ax.set_title('The tradable expression: statistically empty, and dead at 2% borrow')\n"
            "plt.tight_layout(); plt.show()\n"
            "for c, b, nb, na, nt, dd in rows:\n"
            "    print(f'cost={c:.0f}bps borrow={b:.0f}%: net {nb:+.2f} bps/mo ({na:+.2f}%/yr, t={nt:+.2f}), maxDD {dd:.1f}%')"
        ),
        md(
            f"> 💡 In plain words: the best case keeps **{R['overlay'][0][5]:+.2f}%/yr** at "
            f"**t = {R['overlay'][0][6]:+.2f}** — indistinguishable from zero — after having been "
            f"**{R['overlay'][0][7]:.0f}%** underwater when IPOs doubled into Feb 2021. At 2%/yr "
            f"borrow the net is **{R['overlay'][2][5]:+.2f}%/yr**. No capacity, no accessibility "
            "issue even needs discussing: there is no edge to deploy. **MIRAGE.**"
        ),
        md(
            "### 4d · Faithful-engine & power control — we know the truth here\n\n"
            "Synthetic beta-1.25 worlds with a planted monthly alpha; the HAC-alpha detector runs "
            "on **20 seeds per edge** (desk law — no single-seed baselines). The null must reject at "
            "≈ 5%; planted drags must be recovered; and the power at a Ritter-sized drag tells us "
            "how much the live t = −0.99 can and cannot say."
        ),
        code(
            "res = [st.synthetic_control(e, n_seeds=20, lags=6) for e in (0.0, -0.005, -0.010)]\n"
            "for r in res:\n"
            "    print(f\"planted {r['edge_bps']:+7.1f} bps/mo: measured {r['mean_alpha_bps']:+7.2f} bps/mo  \"\n"
            "          f\"mean t = {r['mean_t']:+.2f}  reject rate = {r['reject_rate']*100:.0f}%\")\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "labels = [f\"{r['edge_bps']:+.0f}\\nbps/mo\" for r in res]\n"
            "ax.bar(labels, [r['reject_rate']*100 for r in res], color=[GREY, AMBER, GREEN], width=.5)\n"
            "ax.axhline(5, ls='--', c='k', lw=1, label='nominal 5%')\n"
            "for i, r in enumerate(res):\n"
            "    ax.annotate(f\"{r['reject_rate']*100:.0f}%\", (i, r['reject_rate']*100), ha='center', va='bottom')\n"
            "ax.set_xlabel('planted alpha'); ax.set_ylabel('|t| >= 2 rejection rate (%)')\n"
            "ax.set_title('Unbiased under the null (5%), powered at -100 bps (95%), half-powered at Ritter-size')\n"
            "ax.legend(); plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: the machinery is honest — a zero-alpha world fires exactly "
            f"**{R['syn'][0][3]:.0f}%** of the time (nominal), a −100 bps/mo drag is caught "
            f"**{R['syn'][2][3]:.0f}%** of the time. A Ritter-sized −50 bps/mo drag is caught only "
            f"**{R['syn'][1][3]:.0f}%** of the time in 152 months, so the live tape *could* be "
            "missing a modest true drag — which is exactly why the Signal reads **WEAK** (open "
            "verdict) and not **NONE**. What the tape *does* nail is the third axis: whatever drag "
            "exists is absorbed by the style factor. *(Machinery/power proof only — never cited in "
            "support of a stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — the published record is real vs the market "
            f"(**{R['mkt'][0]:+.2f}%/3yr, NW t = {R['mkt'][1]:+.2f}**, {R['mkt'][3]:.0f}% of 45 "
            f"cohorts negative) but fails style-adjusted (**t = {R['style'][1]:+.2f}**), and the "
            f"live investable tape cannot certify it (alpha t = {R['vs_spy'][5]:+.2f} vs SPY, ≈ 0 "
            "style-matched; 45% power at Ritter-size, stated). Literature says real; this tape "
            "alone can't — WEAK by the inference bar. No survivorship on either track; live window "
            "is 2013+ only (named).\n"
            f"- **Tradability `MIRAGE`** — net short spread **{R['overlay'][0][5]:+.2f}%/yr** at "
            f"t = {R['overlay'][0][6]:+.2f} (5 bps + 1% borrow), **{R['overlay'][2][5]:+.2f}%/yr** at "
            f"2% borrow, max drawdown **{R['overlay'][0][7]:.0f}%**. Nothing survives.\n"
            f"- **Small-growth beta? `CONFIRMED`** — Ritter's own style match cuts the drag to "
            f"insignificance, and the live two-factor alpha is **{R['two_factor'][0]:+.2f} bps/mo** "
            f"(t = {R['two_factor'][2]:+.2f}) with a **{R['two_factor'][4]:.3f}** loading on the "
            "small-growth spread. Brav & Gompers (1997) win the third axis."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **Fama's warning generalises.** Long-horizon BHAR anomalies live and die by the "
            "benchmark (Fama 1998); this study is a live demonstration — the same tape reads "
            "−5.5%/yr or 0.0%/yr depending on the deflator.\n"
            "- **The vintages carry the volume story.** The catastrophic cohorts (1999-2000, "
            "2020-21) are exactly the high-volume manias — the timing signal explored in "
            "[265-ipo-volume](../../265-ipo-volume/); and the pop that never reached you is "
            "[219-ipo-pop](../../219-ipo-pop/).\n"
            "- **What would change the stamp.** A point-in-time, style-adjusted IPO panel (CRSP-"
            "grade, delisting returns included) showing a post-2000 alpha with t ≥ 2 — or another "
            "decade of ETF tape. The desk re-runs on cache refresh.\n\n"
            "*The reproducible core is offline and deterministic; Track A is hardcoded from the "
            "cached source PDF, Track B is cache-first yfinance. Methods and sources: "
            "[`docs/references.md`](../docs/references.md); frozen numbers: "
            "[`docs/results.md`](../docs/results.md).*"
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
