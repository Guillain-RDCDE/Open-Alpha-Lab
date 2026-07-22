"""Generate the two narrative notebooks for Study 790 (Composite Equity Issuance).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached EDGAR-shares x
yfinance-prices panel under ../_cache/ and otherwise quote the frozen headline numbers in ``R``
(mirroring docs/results.md). The synthetic positive control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (36-name large-cap survivor
# basket; EDGAR point-in-time CommonStockSharesOutstanding x yfinance raw+adj closes; 11 annual
# formation years 2014->2024, held next-year; as-of 2025-12-31).
R = dict(
    asof="2025-12-31", fp="6a1c28a68799",
    n_names=36, n_years=11, form_lo=2014, form_hi=2024,
    long_mean=17.35, short_mean=30.57, ls_mean=-13.22, ls_median=-15.60,
    win_pct=36, win_lo=15, win_hi=65, sharpe=-0.69,
    t=-2.29, nw_t=-2.69, placebo_p=0.9972, placebo_mean=-0.02,
    # quantile-width sweep: frac -> (ls %/yr, t, sharpe)
    width={0.2: (-17.59, -2.34, -0.71), 0.3: (-13.22, -2.29, -0.69), 0.4: (-12.18, -2.30, -0.69)},
    # era split at formation year 2020
    era_early_mean=-19.60, era_early_n=6, era_early_t=-10.16,
    era_late_mean=-5.55, era_late_n=5, era_late_t=-0.45, era_diff_t=1.13,
    cost_net=-13.92, cost_net_t=-2.41, cost_bps=10, borrow_bps=50,
    syn_null_t=0.13, syn_null_mean=0.81, syn_planted_t=7.80, syn_planted_mean=28.38,
    # per-year long/short/ls for the ladder chart
    years=[2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
    ls_by_year=[-13.4, -22.2, -19.2, -20.8, -26.6, -15.6, 4.4, 8.4, -54.7, 7.1, 7.0],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Replicates here%3F: Busted](https://img.shields.io/badge/Replicates_here%3F-Busted-8b949e?style=flat-square)\n\n"
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

from composite_issuance import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    RAW, ADJ, SH = data.load_real()
    RAW, ADJ, SH = data.drop_partial_last_year(RAW, ADJ, SH)
else:
    RAW = ADJ = SH = None
print("real cache present:", HAVE_REAL,
      "| names:", (0 if RAW is None else RAW.shape[1]))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Do companies that print stock lose to companies that buy it back? 🧾\n"
            "### Composite equity issuance — a famous Wall-Street factor that *flips sign* on a "
            "basket of big survivors\n\n"
            + BADGES +
            "There's a classic finding in finance: firms that quietly **issue** a lot of new "
            "stock over five years (secondary offerings, stock-based pay, share-funded deals) go "
            "on to **underperform**, while firms that **shrink** their share count (buybacks) go "
            "on to **outperform**. It's called the *composite equity issuance* effect "
            "(Daniel-Titman 2006), and in the full US stock market it's real.\n\n"
            "On a basket of 36 big survivors, using free public data, it comes out **backwards** "
            "— and that's the honest, interesting story of this study.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo and the cost math? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** SEC EDGAR filed share counts × yfinance prices, 36 large-cap "
            "**survivors**, 2014→2024 formation years. Every chart is drawn by the code beside "
            "it; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Do low-issuance (buyback) firms beat high-issuance (diluting) firms here? | "
            f"**No — the opposite.** The buyback basket averaged **+{R['long_mean']:.0f}%/yr** "
            f"while the diluters averaged **+{R['short_mean']:.0f}%/yr**. Betting on the factor "
            f"the textbook way *lost* **{R['ls_mean']:.0f}%/yr**. |\n"
            "| Is that just bad luck / noise? | **No — it's significant, in the wrong "
            f"direction.** *t* = {R['t']:.2f} (the bar is ±2). A random reshuffle of the signal "
            f"beats the real sort **{R['placebo_p']*100:.0f}%** of the time. |\n"
            "| So the famous factor is fake? | **No — our *basket* is the problem.** We only kept "
            "companies that *survived* to 2026. The big issuers that survived are exactly the "
            "ones whose stock printing funded growth that **worked** (think mega-cap tech). The "
            "losers were deleted before we ever looked. |\n"
            "| Could you at least trade the reverse? | **Not really.** The wrong-sign effect is "
            "almost entirely a **pre-2020** phenomenon; since 2020 it's basically gone. |\n\n"
            "> The lesson isn't \"the factor is wrong.\" It's how violently **survivorship bias** "
            "can flip a real result on a small, hand-picked basket of winners."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Watch how fast a company's market value grows versus how much its stock actually "
            "returned. The gap is equity issuance — new shares creeping in. Serial issuers "
            "underperform; net buyback-ers outperform.\"*\n\n"
            "This is **composite equity issuance** (Daniel & Titman, 2006). Over five years, take "
            "the growth in a firm's total market cap and subtract what its stock return alone "
            "would have delivered. Whatever's left is the firm quietly changing its share base — "
            "issuing (bad) or repurchasing (good). It's a cousin of our "
            "[Study 519](../../519-net-share-issuance/) (which uses the *1-year raw share-count* "
            "change); this is the broader **5-year log** version."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If it held on a basket you could actually assemble from free data, it'd be a simple, "
            "mechanical stock screen: rank your names by how much stock they've been printing, "
            "buy the repurchasers, avoid the diluters. No forecasting. The whole point of this "
            "desk, though, is to check whether a famous factor *survives contact* with a small, "
            "honestly-built, survivorship-flavoured sample — because that's the sample most people "
            "can actually build."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The measure.** For each of {R['n_names']} big survivors, at each year-end, the "
            "5-year growth in market cap minus the 5-year stock return — using **share counts as "
            "they were actually filed** with the SEC (no peeking at future filings).\n"
            "- **The sort.** Buy the third with the *lowest* issuance (buyback-ers), short the "
            "third with the *highest* (diluters), hold one year, repeat.\n"
            "- **The luck check.** Randomly shuffle which company gets which issuance number "
            "20,000 times — how often does a random label beat the real one?\n"
            "- **The mirage check.** What would make us say \"survivorship, not signal\"? A "
            "synthetic world where we *plant* the real effect and confirm our sorter would catch "
            "it pointing the *right* way — so a wrong-sign result on the real tape is the sample, "
            "not a bug."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the two baskets.** Average yearly return of the buyback-ers vs the diluters."
        ),
        code(
            "if HAVE_REAL:\n"
            "    s = st.summarize(RAW, ADJ, SH, frac=0.3, placebo=False)\n"
            "    lo, hi, ls = s['long_mean']*100, s['short_mean']*100, s['ls_mean']*100\n"
            "else:\n"
            "    lo, hi, ls = R['long_mean'], R['short_mean'], R['ls_mean']\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.4))\n"
            "ax.bar(['LOW issuance\\n(buyback-ers)','HIGH issuance\\n(diluters)'], [lo, hi],\n"
            "       color=[GREEN, RED], width=.55)\n"
            "for i,v in enumerate([lo, hi]): ax.annotate(f'{v:+.1f}%/yr',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('average next-year total return')\n"
            "ax.set_title(f'The diluters WON: long-short = {ls:+.1f}%/yr (wrong sign)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'buyback-ers {lo:+.1f}%/yr | diluters {hi:+.1f}%/yr | long-short {ls:+.1f}%/yr')"
        ),
        md(
            f"The textbook says the green bar should win. Here the **red bar wins by "
            f"{abs(R['ls_mean']):.0f} points a year** — the high-issuance survivors "
            f"(+{R['short_mean']:.0f}%/yr) crushed the buyback-ers (+{R['long_mean']:.0f}%/yr). "
            "The factor didn't just fail to show up; it **inverted**.\n\n"
            "**Was every year like that, or a couple of blowups?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    ls_s = st.long_short_series(RAW, ADJ, SH, frac=0.3)\n"
            "    yrs = [d.year for d in ls_s.index]; vals = list(ls_s['ls_ret']*100)\n"
            "else:\n"
            "    yrs, vals = R['years'], R['ls_by_year']\n"
            "fig, ax = plt.subplots(figsize=(10.0, 4.3))\n"
            "cols = [GREEN if v>0 else RED for v in vals]\n"
            "ax.bar([str(y) for y in yrs], vals, color=cols, width=.7)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('long-short return that year (%)')\n"
            "ax.set_title('Wrong-sign in 7 of 11 years — the buyback bet mostly lost')\n"
            "plt.tight_layout(); plt.show()\n"
            "print({y: round(v,1) for y,v in zip(yrs, vals)})"
        ),
        md(
            f"Red in **7 of 11** years (win-rate {R['win_pct']}%). It's not one crash dragging "
            "the average — the buyback bet lost the *majority* of years. And a random reshuffle "
            f"of the issuance labels beats the real sort **{R['placebo_p']*100:.0f}%** of the "
            "time, so the real link between issuance and returns is close to the *worst* it could "
            "be for the claim.\n\n"
            "**So is the famous factor just wrong? No — watch what our sorter does when the "
            "effect is really there.**"
        ),
        code(
            "rows = st.synthetic_control(data.synthetic_panel, edges=(0.0, 0.12), n_seeds=25)\n"
            "null_t = next(r for r in rows if r['edge']==0.0)['mean_t']\n"
            "plant_t = next(r for r in rows if r['edge']==0.12)['mean_t']\n"
            "real_t = R['t']\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.bar(['null world\\n(no effect)','planted REAL\\neffect','the actual\\nreal tape'],\n"
            "       [null_t, plant_t, real_t], color=[GREY, GREEN, RED], width=.6)\n"
            "for i,v in enumerate([null_t, plant_t, real_t]):\n"
            "    ax.annotate(f't={v:+.1f}',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "ax.axhline(2, ls='--', c='k', lw=1); ax.axhline(-2, ls='--', c='k', lw=1)\n"
            "ax.set_ylabel('long-short t-stat')\n"
            "ax.set_title('Our sorter catches a REAL effect (green, +). The real tape is red (-).')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'null t={null_t:+.2f} | planted t={plant_t:+.2f} | real tape t={real_t:+.2f}')"
        ),
        md(
            f"When we *plant* a genuine buyback edge, our sorter lights up **positive** "
            f"(t = +{R['syn_planted_t']:.1f}); with no effect it sits at zero "
            f"(t = +{R['syn_null_t']:.2f}). The real tape comes out **negative** "
            f"(t = {R['t']:.2f}). The machinery is fine and points the right way when the effect "
            "is real — so the wrong sign on the real tape is a fact about our **basket of "
            "survivors**, not a broken detector."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** The buyback-minus-diluter bet lost **{R['ls_mean']:.0f}%/yr** "
            f"at *t* = {R['t']:.2f} — significant, but the **wrong sign**. On this survivor "
            "basket the famous factor doesn't replicate; it inverts.\n"
            "- **Tradability — Mirage.** You can't trade a wrong-sign factor, and the reverse bet "
            "is a pre-2020-only survivorship artifact. Costs only make it worse.\n"
            "- **\"Does the published factor replicate here?\" — Busted.** A canonical academic "
            "result, honestly rebuilt on a small survivor basket, flips sign — a textbook "
            "small-sample / survivorship lesson, not a free lunch."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **The real fix** is a point-in-time, survivorship-free universe (CRSP/Compustat), "
            "where the delisted issuers are still in the sample and the effect shows up the right "
            "way. That's the difference between the published result and a free-data replication.\n"
            "- **Why survivors invert it:** the surviving big issuers are a *selected* set — the "
            "ones whose stock printing (stock-based comp, acquisitions) funded growth that "
            "actually worked. The issuers who diluted their way into oblivion aren't in the "
            "basket at all.\n"
            "- **Sibling studies:** [519-net-share-issuance](../../519-net-share-issuance/) (the "
            "1-year raw share-count version — also `None` here), "
            "[368-buyback-drift](../../368-buyback-drift/) (the announcement *event*), and "
            "[250-reverse-split](../../250-reverse-split/) (a corporate-action signal). See "
            "[docs/references.md](docs/references.md) for the exact dedup.\n\n"
            "*Think the factor survives on free data? Build a point-in-time survivor-free basket "
            "and show a positive, certifiable long-short after costs — then we'll talk.*"
        ),
    ]
    _write(new_notebook(cells=cells, metadata=_meta()), "01_for_the_curious.ipynb")


# ===========================================================================
# 02 — FOR THE QUANTS
# ===========================================================================
def build_quants():
    cells = [
        md(
            "# Composite equity issuance — a quantitative teardown 🔬\n"
            "### The 5-year Daniel-Titman measure on point-in-time EDGAR shares · an annual "
            "cross-sectional sort · one-sample & Newey-West *t* · a 20,000-draw label-shuffle "
            "placebo · quantile-width & era cuts · costs + borrow · a 25-seed synthetic control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The "
            "claim — **high 5-year composite equity issuers underperform net buyback-ers** "
            "(Daniel-Titman 2006) — is tested as a clean annual long-short on a 36-name large-cap "
            "**survivor** basket. The job: measure it honestly with point-in-time filed shares "
            "and one execution lag, and report the sign the data actually give.\n\n"
            "> ⚠️ **Data note.** SEC EDGAR `us-gaap:CommonStockSharesOutstanding` "
            "(`dei:EntityCommonStockSharesOutstanding` fallback), point-in-time by `filed` date, "
            "× yfinance raw + adjusted closes; December year-end grid; as-of 2025-12-31; "
            "fingerprint `" + R["fp"] + "`. **Survivorship named on the Signal axis** — the basket "
            "is current names projected back, which biases the sort *against* the claim. Methods "
            "in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | long-low/short-high L−S **{R['ls_mean']:+.2f}%/yr**, "
            f"one-sample *t* = **{R['t']:.2f}**, Newey-West *t* = **{R['nw_t']:.2f}** — "
            f"*wrong sign*; placebo *p* = {R['placebo_p']:.4f} |\n"
            f"| **Tradability** | `MIRAGE` | net of {R['cost_bps']}bps×2 + {R['borrow_bps']}bps "
            f"borrow: **{R['cost_net']:+.2f}%/yr** (*t* = {R['cost_net_t']:.2f}); the inverse is "
            f"pre-2020-only (early *t* = {R['era_early_t']:.2f}, late *t* = {R['era_late_t']:.2f}) |\n"
            f"| **Replicates here?** | `BUSTED` | synthetic control recovers a planted edge at "
            f"*t* = +{R['syn_planted_t']:.1f} (right sign); the real tape is *t* = {R['t']:.2f} |\n\n"
            "> 💡 In plain words: the factor is significant here **in the wrong direction** — a "
            "survivorship-driven sign flip, confirmed by a control that catches the real effect "
            "pointing the right way."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "For firm $i$ at formation year-end $t$, composite equity issuance is\n\n"
            "$$\\iota_{i,t} = \\log\\frac{ME_{i,t}}{ME_{i,t-5}} - r_i(t-5,t),$$\n\n"
            "where $ME = \\text{shares}\\times\\text{raw price}$ is market cap and $r_i(t-5,t) = "
            "\\log(\\text{adj}_{i,t}/\\text{adj}_{i,t-5})$ is the 5-year cumulative **total** log "
            "return. Splits cancel between the legs, leaving net equity issuance in log terms. "
            "The claims:\n\n"
            "- **H₁ (premium).** $E[r^{fwd} \\mid \\iota \\text{ low}] > E[r^{fwd} \\mid \\iota "
            "\\text{ high}]$ — buyback-ers beat diluters next year.\n"
            "- **H₂ (monotone / stable).** The spread is stable across quantile widths and "
            "sub-periods, not a tail-definition or single-regime artifact.\n"
            "- **H₃ (deployable).** It survives one-way costs × both legs + short borrow.\n\n"
            f"We find **H₁ rejected with the wrong sign** (L−S {R['ls_mean']:+.2f}%/yr, "
            f"*t* = {R['t']:.2f}, NW *t* = {R['nw_t']:.2f}), **H₂ \"stable-but-inverted\"** across "
            f"widths yet concentrated **pre-2020** (early *t* = {R['era_early_t']:.2f}, late "
            f"*t* = {R['era_late_t']:.2f}), **H₃ moot** (costs deepen the loss). The synthetic "
            "control confirms the engine recovers a genuine edge with a **positive** *t*, so this "
            "is a survivorship inversion, not a detector artifact."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "The formation windows **overlap** (each is a trailing 5 years, re-struck annually), "
            "so the annual long-short series is serially dependent. The primary statistic is a "
            "**one-sample *t*** of the L−S mean vs 0; the autocorrelation-robust cross-check is a "
            "**Newey-West (1 lag)** HAC *t*. The win-rate carries a **Wilson interval**, the "
            "**label-shuffle placebo** (20,000 draws) permutes which name carries which issuance "
            "value — preserving the marginal issuance distribution and the realised forward "
            "cross-section — and the era split (formation year 2020) is tested as a **difference** "
            "(Welch *t*), not eyeballed. With only 11 annual observations, none of this is "
            "high-powered — which is exactly the point of grading it `NONE` rather than reading a "
            "story into a small sample."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Panel.** {R['n_names']} large-cap survivors, point-in-time filed shares × "
            f"raw/adjusted closes, {R['form_lo']}→{R['form_hi']} formation years ({R['n_years']} "
            "annual rebalances).\n"
            "- **Signal.** 5-year composite issuance per name per formation year-end.\n"
            "- **Sort.** Long bottom-30% (low issuance), short top-30% (high issuance), hold "
            "*t→t+1* (one execution lag).\n"
            "- **Headline.** One-sample *t* + NW(1) *t* + Wilson win-rate + 20,000-draw placebo.\n"
            "- **Robustness.** Quantile-width sweep (0.2/0.3/0.4); era split (< 2020 vs ≥ 2020).\n"
            "- **Costs.** One-way bps × both legs × turnover + short borrow.\n"
            "- **Control.** 25-seed synthetic panel, planted-edge knob; null must not fire, a "
            "planted low-issuance edge must print a **positive** *t*."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The headline sort and its placebo\n\n"
            "Long low-issuance, short high-issuance, annual. The label-shuffle null asks whether "
            "a random relabelling of the issuance signal would have produced a spread this "
            "extreme (in the claim's favour)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    s = st.summarize(RAW, ADJ, SH, frac=0.3)\n"
            "    ls, t, nwt = s['ls_mean']*100, s['t'], s['nw_t']\n"
            "    win, wlo, whi = s['win']*100, s['win_lo']*100, s['win_hi']*100\n"
            "    pl = st.placebo_pvalue(RAW, ADJ, SH, frac=0.3, n_draws=2000)\n"
            "    obs, pmean, p = pl['obs_mean']*100, pl['placebo_mean']*100, pl['p_value']\n"
            "    print(f\"L-S {ls:+.2f}%/yr | one-sample t={t:+.2f} | NW t={nwt:+.2f}\")\n"
            "    print(f\"win-rate {win:.0f}% Wilson95 [{wlo:.0f}%,{whi:.0f}%] | placebo p={p:.4f} (2k draws)\")\n"
            "else:\n"
            "    ls, obs, pmean, p = R['ls_mean'], R['ls_mean'], R['placebo_mean'], R['placebo_p']\n"
            "    win, wlo, whi = R['win_pct'], R['win_lo'], R['win_hi']\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "rng = np.random.default_rng(790)\n"
            "# illustrative null cloud centred at the placebo mean (canonical 20k p quoted from results.md)\n"
            "null = rng.normal(R['placebo_mean'], 4.0, 4000)\n"
            "ax.hist(null, bins=50, color=GREY, alpha=.85, label='label-shuffle null (illustrative)')\n"
            "ax.axvline(R['ls_mean'], c=RED, lw=2.5, label=f\"observed L-S {R['ls_mean']:+.1f}%/yr\")\n"
            "ax.axvline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('long-short return (%/yr)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f\"Real sort sits on the WRONG side of the null: canonical p={R['placebo_p']:.4f}\")\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"canonical placebo (results.md): mean {R['placebo_mean']:+.2f}%/yr, p={R['placebo_p']:.4f}\")"
        ),
        md(
            f"> 💡 In plain words: the observed **{R['ls_mean']:+.2f}%/yr** sits *below* the "
            f"null's centre (≈{R['placebo_mean']:+.2f}%/yr); the placebo beats the real sort "
            f"**{R['placebo_p']*100:.1f}%** of the time. One-sample *t* = {R['t']:.2f}, NW *t* = "
            f"{R['nw_t']:.2f}, win-rate {R['win_pct']}% (Wilson [{R['win_lo']}%, {R['win_hi']}%]). "
            "H₁ is not just unproven — it's rejected with the wrong sign."
        ),
        md(
            "### 4b · Robustness — width sweep and the era split\n\n"
            "A real factor is stable across tail widths; a survivorship artifact can be stable "
            "*and* wrong-signed *and* concentrated in one regime."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ws = st.width_sweep(RAW, ADJ, SH, fracs=(0.2,0.3,0.4))\n"
            "    fr = [r['frac'] for r in ws]; lsw = [r['ls_mean']*100 for r in ws]; tw = [r['t'] for r in ws]\n"
            "    ec = st.era_contrast(RAW, ADJ, SH, split_year=2020)\n"
            "    e, l, et, lt = ec['early_mean']*100, ec['late_mean']*100, ec['t_early'], ec['t_late']\n"
            "else:\n"
            "    fr = list(R['width']); lsw = [R['width'][f][0] for f in fr]; tw = [R['width'][f][1] for f in fr]\n"
            "    e, l, et, lt = R['era_early_mean'], R['era_late_mean'], R['era_early_t'], R['era_late_t']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.3))\n"
            "a1.bar([f'{f:.1f}' for f in fr], lsw, color=RED, width=.55)\n"
            "for i,(v,t_) in enumerate(zip(lsw,tw)): a1.annotate(f'{v:+.1f}%\\n(t={t_:+.2f})',(i,v),ha='center',va='top')\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_xlabel('tail fraction'); a1.set_ylabel('L-S (%/yr)')\n"
            "a1.set_title('Stable across widths -- and stably WRONG-signed')\n"
            "a2.bar(['pre-2020\\n(n={})'.format(R['era_early_n']),'2020+\\n(n={})'.format(R['era_late_n'])],\n"
            "       [e, l], color=[RED, GREY], width=.55)\n"
            "for i,(v,t_) in enumerate([(e,et),(l,lt)]): a2.annotate(f'{v:+.1f}%\\n(t={t_:+.2f})',(i,v),ha='center',va='top' if v<0 else 'bottom')\n"
            "a2.axhline(0, c='k', lw=.8); a2.set_ylabel('L-S (%/yr)')\n"
            "a2.set_title('The wrong-sign spread is a PRE-2020 phenomenon')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('width:', {f: round(v,1) for f,v in zip(fr,lsw)}, '| era early', round(e,1), 'late', round(l,1))"
        ),
        md(
            f"> 💡 In plain words: the wrong sign holds at every tail width "
            f"(*t* ≈ −2.3 throughout), so it isn't a quantile-definition fluke. But it is almost "
            f"entirely **pre-2020** ({R['era_early_mean']:+.1f}%/yr, *t* = {R['era_early_t']:.2f}) "
            f"and essentially gone since ({R['era_late_mean']:+.1f}%/yr, *t* = "
            f"{R['era_late_t']:.2f}); the difference test can't certify the change (*t* = "
            f"{R['era_diff_t']:+.2f}). Not a stable inverse relationship — a regime-plus-"
            "survivorship artifact."
        ),
        md(
            "### 4c · Costs — there is nothing to deploy\n\n"
            "One-way bps × both legs × turnover + a short-leg borrow, on an already-negative "
            "spread."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = st.net_of_costs(RAW, ADJ, SH, frac=0.3, cost_bps=10.0, borrow_ann_bps=50.0)\n"
            "    g, n, nt = c['gross_mean']*100, c['net_mean']*100, c['net_t']\n"
            "else:\n"
            "    g, n, nt = R['ls_mean'], R['cost_net'], R['cost_net_t']\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.3))\n"
            "ax.bar(['gross L-S','net (10bps x2\\n+50bps borrow)'], [g, n], color=[GREY, RED], width=.5)\n"
            "for i,v in enumerate([g, n]): ax.annotate(f'{v:+.1f}%/yr',(i,v),ha='center',va='top')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('long-short (%/yr)')\n"
            "ax.set_title(f'Costs deepen the loss (net t={nt:+.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross {g:+.2f}%/yr -> net {n:+.2f}%/yr (t={nt:+.2f})')"
        ),
        md(
            f"> 💡 In plain words: gross {R['ls_mean']:+.2f}%/yr → net {R['cost_net']:+.2f}%/yr "
            f"(*t* = {R['cost_net_t']:.2f}). The long-low/short-high book loses double digits; its "
            "inverse is the pre-2020-only artifact above. **Tradability = MIRAGE** in either "
            "direction."
        ),
        md(
            "### 4d · Faithful-engine & power control — we know the truth here\n\n"
            "Deterministic cross-sectional panel with a TUNABLE planted low-issuance edge, "
            "averaged over 25 seeds. The null (edge = 0) must not fire; a planted edge must print "
            "a **positive** *t* — proving the wrong sign on the real tape is the sample, not the "
            "sorter."
        ),
        code(
            "rows = st.synthetic_control(data.synthetic_panel, edges=(0.0, 0.06, 0.12), n_seeds=25)\n"
            "edges = [r['edge'] for r in rows]; ts = [r['mean_t'] for r in rows]\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "ax.plot(edges, ts, 'o-', color=GREEN, lw=2, ms=8, label='planted worlds (25 seeds each)')\n"
            "ax.scatter([0.0],[R['t']], color=RED, s=110, zorder=5, label=f\"real tape (t={R['t']:.2f})\")\n"
            "ax.axhline(2, ls='--', c='k', lw=1); ax.axhline(-2, ls='--', c='k', lw=1); ax.axhline(0, c='k', lw=.6)\n"
            "ax.set_xlabel('planted low-issuance edge'); ax.set_ylabel('long-short t (25-seed mean)')\n"
            "ax.set_title('Engine recovers a REAL edge (+t). The real tape is -t: survivorship.')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print({round(e,2): round(t,2) for e,t in zip(edges,ts)}, '| real tape t =', R['t'])"
        ),
        md(
            f"> 💡 In plain words: the null world averages *t* = +{R['syn_null_t']:.2f} (no false "
            f"positive), and a planted edge climbs monotonically to *t* = +{R['syn_planted_t']:.1f} "
            f"— **positive**, the Daniel-Titman direction. The real tape sits at *t* = {R['t']:.2f}, "
            "the opposite side. The detector is unbiased and correctly signed, so the real-tape "
            "inversion is a property of the **survivor basket**. *(A faithful-engine / power check "
            "only — never cited in support of the real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — long-low/short-high composite issuance is "
            f"**{R['ls_mean']:+.2f}%/yr** at one-sample *t* = {R['t']:.2f} (NW *t* = "
            f"{R['nw_t']:.2f}) — significant but **wrong-signed**; the label-shuffle placebo beats "
            f"the real sort {R['placebo_p']*100:.1f}% of the time, win-rate {R['win_pct']}%. A "
            "survivorship-driven inversion (survivorship named on this axis, its direction argued "
            "against the claim), confirmed by a synthetic control that recovers a genuine edge "
            f"with a **positive** *t* = +{R['syn_planted_t']:.1f}.\n"
            f"- **Tradability `MIRAGE`** — net of costs {R['cost_net']:+.2f}%/yr (*t* = "
            f"{R['cost_net_t']:.2f}); the inverse bet is a pre-2020-only artifact (early *t* = "
            f"{R['era_early_t']:.2f}, late *t* = {R['era_late_t']:.2f}). Nothing to size either "
            "way.\n"
            "- **Replicates here? `BUSTED`** — a canonical academic factor, honestly rebuilt on a "
            "36-name survivor basket with point-in-time shares and one lag, flips sign. The effect "
            "that lives in the full CRSP/Compustat cross-section does not survive a small-survivor "
            "replication."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The fix is the universe.** A point-in-time, survivorship-free panel keeps the "
            "delisted issuers that dilute into oblivion — the exact names whose absence flips the "
            "sign here. A natural follow-up is to bolt EDGAR filings onto a delisting-inclusive "
            "identifier map and re-run.\n"
            "- **Why the survivors invert it:** the surviving high-issuers are a selected sample "
            "(dilution that funded growth that *worked* — mega-cap tech, share-financed winners). "
            "The direction of the bias is *against* the claim, which is why the sign is negative "
            "rather than merely noisy.\n"
            "- **Dedup map:** [519-net-share-issuance](../../519-net-share-issuance/) (the 1-year "
            "raw share-count change — `None`), [368-buyback-drift](../../368-buyback-drift/) (the "
            "announcement event), [250-reverse-split](../../250-reverse-split/) (a "
            "corporate-action signal). None test the 5-year log composite measure on filed "
            "shares.\n\n"
            "*The reproducible core is offline and deterministic; frozen numbers live in "
            "[`docs/results.md`](../docs/results.md), sources in "
            "[`docs/references.md`](../docs/references.md).*"
        ),
    ]
    _write(new_notebook(cells=cells, metadata=_meta()), "02_for_the_quants.ipynb")


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
