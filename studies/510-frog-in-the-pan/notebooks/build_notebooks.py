"""Generate the two narrative notebooks for Study 510 (Frog-In-The-Pan).

    python notebooks/build_notebooks.py

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic control
runs anywhere, offline and deterministic; the real-panel cells use the cached yfinance parquet
under ../_cache/ if present and otherwise quote the frozen headline numbers in ``R``.

ONE dict of real numbers (``R``) mirrors docs/results.md exactly (as-of 2026-06-26).
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


# Frozen real-panel headline numbers -- mirror of docs/results.md (as-of 2026-06-26).
R = dict(
    n_months=151, n_tickers=38, fp="e038fa42cc02",
    date_start="2012-05-18", date_end="2025-12-30", month_ends=164,
    base_mean=0.18, base_t=0.05,
    low_mean=-0.37, low_vol=15.6, low_sharpe=-0.02, low_t=-0.09, low_hit=54.3, low_dd=-43.4,
    high_mean=-4.92, high_vol=13.3, high_sharpe=-0.37, high_t=-1.32, high_hit=50.3, high_dd=-56.9,
    fip_mean=4.55, fip_t=1.01, fip_hit=57.6,
    turnover=35.0, names_leg=6.0,
    low_net_mean=-1.29, low_net_t=-0.31, low_net_sharpe=-0.08,
    placebo_p=0.558, placebo_mean=0.18,
    placebo_seeds={1: 0.550, 7: 0.545, 42: 0.540, 510: 0.573, 2024: 0.547},
    ctrl=[  # (strength, low_mean%, low_t, high_mean%, high_t, fip_gap%)
        (0.00, -0.5, -0.09, 4.4, 0.95, -4.8),
        (0.20, 13.6, 2.60, 12.2, 2.13, 1.4),
        (0.40, 40.2, 6.98, 28.9, 4.74, 11.3),
        (0.60, 77.2, 9.87, 38.0, 5.99, 39.2),
    ],
    annual={
        2013: -3.1, 2014: 11.0, 2015: 15.7, 2016: -18.3, 2017: 14.6, 2018: 1.0,
        2019: 5.1, 2020: -12.5, 2021: -11.3, 2022: -2.3, 2023: -3.5, 2024: 14.0,
        2025: -21.0,
    },
)

# ---------------------------------------------------------------------------
# Shared preamble
# ---------------------------------------------------------------------------
BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False,
                     "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from frog_in_the_pan import data, strategy as st

FRAC, COST_BPS, BORROW_BPS = 0.30, 5.0, 50.0

def _have_cache():
    return os.path.exists(os.path.join(os.path.abspath(".."), "_cache", "fip_prices.parquet"))

HAVE_REAL = _have_cache()
print("yfinance cache present:", HAVE_REAL)

if HAVE_REAL:
    prices = data.drop_partial_last_month(data.fetch_panel())
    mp = st.to_monthly(prices)
    lo = st.long_short(mp, prices=prices, id_side="low",  frac=FRAC)
    hi = st.long_short(mp, prices=prices, id_side="high", frac=FRAC)
    base = st.long_short(mp, prices=None, frac=FRAC)
    s_lo, s_hi, s_base = st.summary(lo["wml_gross"]), st.summary(hi["wml_gross"]), st.summary(base["wml_gross"])
    fip = st.fip_spread(lo, hi); s_fip = st.summary(fip)
    print(f"N months: {s_lo['n']} | LOW-ID WML: {s_lo['mean']*100:+.2f}%/yr (t {s_lo['tstat']:+.2f})"
          f" | FIP gap: {s_fip['mean']*100:+.2f}%/yr (t {s_fip['tstat']:+.2f})")
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Frog-In-The-Pan -- is momentum stronger when the trend boiled up *slowly*?\n"
            "### Da-Gurun-Warachka (2014): continuous information and momentum, tested honestly\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Does gradual beat jumpy%3F: Mixed](https://img.shields.io/badge/Does%20gradual%20beat%20jumpy%3F-Mixed-8b949e?style=flat-square)\n\n"
            "Drop a frog in boiling water and it leaps out. Warm the water slowly and it never "
            "reacts -- and boils. In 2014, Zhi Da, Umit Gurun and Mitch Warachka argued markets "
            "behave the same way: a **discrete** shock (a jump) is priced instantly, but "
            "**continuous** information -- a stock drifting up on many small green days -- attracts "
            "too little attention and is *under-reacted to*. So momentum profits, they showed, "
            "should live in the **gradual** names, not the jumpy ones.\n\n"
            "They built an **information-discreteness (ID)** measure from the sign-consistency of "
            "daily returns and sorted momentum by it. We replicate the interaction on a large-cap "
            "S&P 500 survivor basket -- naming the survivorship bias up front, because a gradual "
            "slide into delisting is itself a low-ID name.\n\n"
            "> **This is the plain-language layer.** Want the ID formula, the double-sort, the "
            "HAC *t* and the placebo? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does the gradual (low-ID) book carry momentum? | **No, not here.** The LOW-ID WML "
            f"earns **{R['low_mean']:.2f}%/yr**, HAC *t* = **{R['low_t']:.2f}** -- statistically "
            "zero on this 38-name survivor basket. |\n"
            "| Does gradual beat jumpy? | **In sign, yes.** LOW-ID WML "
            f"({R['low_mean']:.2f}%/yr) beats HIGH-ID WML ({R['high_mean']:.2f}%/yr) by "
            f"**+{R['fip_mean']:.2f}%/yr** -- but the interaction's HAC *t* is only "
            f"**+{R['fip_t']:.2f}** (< 2). |\n"
            "| Is it tradable? | **No.** Net of 5 bps/leg + 50 bps/yr borrow the gradual book is "
            f"**{R['low_net_mean']:.2f}%/yr**. |\n"
            "| Does the placebo agree? | **Yes.** A seed-robust label-shuffle gives "
            f"**p = {R['placebo_p']:.2f}** -- the real mean sits *below* its own null. |\n\n"
            "> The frog-in-the-pan effect is real and elegant in the broad-universe literature. "
            "On 38 large-cap survivors the *direction* survives but nothing clears the bar -- the "
            "expected fate of a pointed academic factor on a thin survivor basket."
        ),

        # ---- BEAT 1 -- THE CLAIM ---------------------------------------------
        md(
            "## 1 -- The claim\n\n"
            "> *\"Investors under-react more to continuous information than to discrete "
            "information. A series of gradual, frequent, small same-signed price changes attracts "
            "limited attention; the market is slow to fully incorporate it, so momentum is "
            "stronger among low-information-discreteness stocks.\"*\n\n"
            "-- Da, Gurun & Warachka (2014), *Review of Financial Studies*\n\n"
            "Their proxy is beautifully simple. Over the formation window, count the share of "
            "positive (%pos) and negative (%neg) daily returns, and take\n\n"
            "$$ \\text{ID} = \\text{sign}(\\text{PRET}) \\times (\\%\\text{neg} - \\%\\text{pos}) $$\n\n"
            "where PRET is the total formation return. A stock that drifted **up smoothly** has "
            "%pos > %neg and PRET > 0, so ID < 0 -- **low, continuous** (the frog in the pan). A "
            "stock that **jumped up** on a few days amid many small down days has ID > 0 -- "
            "**high, discrete**. Low ID = gradual."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 -- So what?\n\n"
            "If true, FIP is a *refinement* of momentum: it tells you *which* winners and losers "
            "to trust. Plain momentum on large-caps has gone flat for a decade (this desk's "
            "[507](../../507-cross-sectional-momentum/) and [24 Stampede](../../24-stampede/)). "
            "If the surviving juice all sits in the gradual names, conditioning on ID would "
            "resurrect a tradable book. That is the prize -- and the thing we test."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW -----------------------------------------
        md(
            "## 3 -- How would we even know?\n\n"
            "Three disciplines keep us honest:\n\n"
            "1. **One forward execution lag.** The momentum signal *and* the ID measure are "
            "computed from prices up to month-end *t*; we enter one trading day later and earn "
            "month *t+1*. No same-bar fill, no look-ahead.\n"
            "2. **A label-shuffle placebo.** Shuffle which stock the momentum signal points at "
            "inside the low-ID half -- if the gradual edge is real it survives; if it is "
            "data-mined, the real mean melts into the shuffled null.\n"
            "3. **Name the survivorship bias on the SIGNAL axis.** A *gradual* decline into "
            "delisting is precisely a low-ID loser. Those names are absent from a survivor basket, "
            "so the FIP slice is the **more** inflated of the two -- not less."
        ),

        # ---- BEAT 4 -- THE TEARDOWN ------------------------------------------
        md(
            "## 4 -- The teardown\n\n"
            "**First, does the FIP engine work when the effect is planted?** We build a synthetic "
            "world where the drift is delivered *smoothly* to the low-ID names and in *jumps* to "
            "the high-ID names, and check the engine recovers a bigger low-ID WML."
        ),
        code(
            "ctrl = st.synthetic_control(strengths=(0.0, 0.20, 0.40, 0.60), frac=FRAC, seed=510)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "x = np.arange(len(ctrl)); w = 0.38\n"
            "ax.bar(x - w/2, ctrl['low_id_mean_ann']*100, w, color=GREEN, label='LOW-ID (gradual)')\n"
            "ax.bar(x + w/2, ctrl['high_id_mean_ann']*100, w, color=RED, label='HIGH-ID (jumpy)')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{s:.2f}' for s in ctrl['mom_strength']])\n"
            "ax.set_xlabel('planted momentum strength'); ax.set_ylabel('WML mean (%/yr)')\n"
            "ax.set_title('Synthetic control: low-ID WML > high-ID WML when the effect is planted')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(ctrl.round(3).to_string(index=False))"
        ),
        md(
            "The engine is faithful: at planted strength 0 both slices score ~0, and as the drift "
            "grows the **low-ID WML pulls ahead of the high-ID WML** -- exactly the FIP "
            "interaction. So whatever we find on the real tape reflects **the market**, not a "
            "broken detector."
        ),
        md("**Now the honest test on the real yfinance panel.**"),
        code(
            "if HAVE_REAL:\n"
            "    lo_m, lo_t = s_lo['mean']*100, s_lo['tstat']\n"
            "    hi_m, hi_t = s_hi['mean']*100, s_hi['tstat']\n"
            "    base_m = s_base['mean']*100\n"
            "    fip_m, fip_t = s_fip['mean']*100, s_fip['tstat']\n"
            "else:\n"
            "    lo_m, lo_t = R['low_mean'], R['low_t']\n"
            "    hi_m, hi_t = R['high_mean'], R['high_t']\n"
            "    base_m = R['base_mean']; fip_m, fip_t = R['fip_mean'], R['fip_t']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.5))\n"
            "labels = ['Plain\\nmomentum', 'LOW-ID\\n(gradual)', 'HIGH-ID\\n(jumpy)']\n"
            "vals = [base_m, lo_m, hi_m]\n"
            "cols = [GREY, AMBER, RED]\n"
            "ax.bar(labels, vals, color=cols, width=0.55)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('WML mean annual return (%/yr, gross)')\n"
            "ax.set_title(f'Real tape: gradual beats jumpy by {fip_m:+.1f}%/yr, but t={fip_t:+.2f}')\n"
            "for i, v in enumerate(vals):\n"
            "    ax.text(i, v + (0.2 if v >= 0 else -0.6), f'{v:+.1f}%', ha='center', fontsize=9)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'LOW-ID WML: {lo_m:+.2f}%/yr (t {lo_t:+.2f}) | HIGH-ID: {hi_m:+.2f}%/yr (t {hi_t:+.2f})')\n"
            "print(f'FIP interaction (low - high): {fip_m:+.2f}%/yr (t {fip_t:+.2f})')"
        ),
        md(
            f"The gradual (LOW-ID) book that should *carry* the premium earns "
            f"**{R['low_mean']:.2f}%/yr** at HAC *t* = **{R['low_t']:.2f}** -- statistically "
            f"zero, and no better than plain momentum (**+{R['base_mean']:.2f}%/yr**). The jumpy "
            f"book is the more negative, so the **interaction is positive** "
            f"(**+{R['fip_mean']:.2f}%/yr**) -- gradual *does* beat jumpy, as the theory says. "
            f"But the interaction's HAC *t* is only **+{R['fip_t']:.2f}**: the sign is right, the "
            "significance is not."
        ),

        # ---- BEAT 5 -- THE VERDICT -------------------------------------------
        md(
            "## 5 -- The verdict\n\n"
            f"- **Signal -- NONE.** The LOW-ID WML earns {R['low_mean']:.2f}%/yr (*t* "
            f"{R['low_t']:.2f}) and the FIP interaction +{R['fip_mean']:.2f}%/yr (*t* "
            f"+{R['fip_t']:.2f}) -- neither clears |t| >= 2. A seed-robust placebo (p = "
            f"{R['placebo_p']:.2f}) puts the real mean below its own null. NONE.\n"
            f"- **Tradability -- MIRAGE.** Net of costs + borrow the gradual book is "
            f"{R['low_net_mean']:.2f}%/yr at {R['turnover']:.0f}%/mo turnover. Nothing to trade.\n"
            "- **Does gradual beat jumpy? -- MIXED.** The sign is on the theory's side and the "
            "synthetic control proves the engine would catch a real gap -- but the magnitude is "
            "statistically indistinguishable from zero here."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 -- Could you actually trade it?\n\n"
            "No. Three obstacles, all visible above:\n\n"
            "1. **The base book is dead.** Large-cap momentum has paid ~nothing for a decade; "
            "conditioning a flat signal on ID cannot conjure a premium that is not there.\n"
            "2. **Costs bury it.** 35%/mo two-sided turnover on a ~6-name leg, plus short borrow, "
            "turns a flat gross book into a negative net one.\n"
            "3. **Survivorship cuts the wrong way.** The gradual losers we would short -- firms "
            "drifting quietly toward delisting -- are exactly the names a survivor basket omits, "
            "so even the flat result is an upper bound."
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 -- Going further\n\n"
            "- **Broaden the universe.** The original FIP result is on the full CRSP cross-section, "
            "where small-caps and delisted names live. A Russell 3000 pull with delisting returns "
            "would be the fair test.\n"
            "- **Pair it with [507 Cross-Sectional-Momentum](../../507-cross-sectional-momentum/)**: "
            "the unconditioned base book this study refines.\n"
            "- **Pair it with [509 Intermediate-Momentum](../../509-intermediate-momentum/)**: "
            "conditions momentum on *when* past returns arrived rather than *how continuously*.\n\n"
            "*Think the frog-in-the-pan premium survives on a delisting-complete universe net of "
            "costs? Fork this, add the delisted names, and show the low-ID WML at t > 2 net. "
            "That is the bar.*"
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
            "# Frog-In-The-Pan -- a quantitative teardown\n"
            "### yfinance panel * 12-1 momentum x information-discreteness * HAC inference * placebo\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Does gradual beat jumpy%3F: Mixed](https://img.shields.io/badge/Does%20gradual%20beat%20jumpy%3F-Mixed-8b949e?style=flat-square)\n\n"
            "The quantitative companion to the [notebook for the curious](01_for_the_curious.ipynb) "
            "-- same seven beats, every claim carrying its standard error. We test "
            "Da-Gurun-Warachka (2014): build the information-discreteness measure, double-sort "
            "12-1 momentum by it, and race the low-ID WML against the high-ID WML.\n\n"
            "> **Not investment advice.** Real data: yfinance daily adjusted-close, 38 S&P 500 "
            f"large-caps, {R['date_start']}--{R['date_end']}, as-of 2026-06-26, fp `{R['fp']}`. "
            "Methods in [`docs/references.md`](../docs/references.md), reproducible numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **Survivorship bias is named on the SIGNAL axis:** a gradual decline into delisting "
            "is a low-ID loser, so the FIP slice is the *more* inflated of the two. Upper bounds."
        ),
        code(BOOT),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | LOW-ID WML **{R['low_mean']:.2f}%/yr** (HAC *t* "
            f"**{R['low_t']:.2f}**); FIP interaction **+{R['fip_mean']:.2f}%/yr** (HAC *t* "
            f"**+{R['fip_t']:.2f}**). Placebo p = **{R['placebo_p']:.2f}**. |\n"
            f"| **Tradability** | `MIRAGE` | Net **{R['low_net_mean']:.2f}%/yr** "
            f"(*t* {R['low_net_t']:.2f}) after 5 bps/leg + 50 bps/yr borrow, "
            f"{R['turnover']:.0f}%/mo turnover. |\n"
            "| **Does gradual beat jumpy?** | `MIXED` | Sign right (low > high ID), magnitude "
            "not significant; synthetic control confirms the engine would catch a real gap. |\n\n"
            "> The frog-in-the-pan effect is well documented on the broad CRSP cross-section. On "
            "38 large-cap survivors with a 12-month-3-quartile thin sort, the interaction is "
            "directionally right but statistically marginal -- consistent with a dead large-cap "
            "base book and a post-publication, survivorship-cut sample."
        ),

        # ---- BEAT 1 -- THE CLAIM ---------------------------------------------
        md(
            "## 1 -- The claim, steelmanned\n\n"
            "Let $\\beta^{12-1}_{i,t}$ be the trailing 12-1 momentum of stock $i$ and "
            "$\\text{ID}_{i,t} = \\text{sign}(\\text{PRET}_{i,t}) \\cdot (\\%\\text{neg} - "
            "\\%\\text{pos})$ over the same formation window. Da-Gurun-Warachka (2014) assert:\n\n"
            "- **H1 (interaction).** The WML momentum spread inside the LOW-ID half exceeds the "
            "WML spread inside the HIGH-ID half: $\\text{WML}^{\\text{lowID}} > "
            "\\text{WML}^{\\text{highID}}$, with the difference (the FIP premium) > 0 and "
            "significant.\n"
            "- **H2 (concentration).** Most of total momentum lives in the low-ID names; high-ID "
            "momentum is weak or absent.\n"
            "- **H3 (tradable).** The low-ID WML survives costs.\n\n"
            "We find the **sign** of H1 (low > high ID by +4.55%/yr) but not the significance "
            "(*t* +1.01); we **reject** H2 here (the low-ID book is itself ~0, *t* -0.09 -- the "
            "interaction comes from high-ID being *more negative*, not low-ID being positive); "
            "and we **reject** H3 (negative net of costs)."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 -- So what? -- the economic stakes\n\n"
            "FIP is one of the more cited momentum refinements of the 2010s, baked into "
            "attention-based factor models and several systematic shops' momentum overlays. If "
            "the interaction has decayed -- or never applied cleanly to large-caps -- then "
            "ID-conditioned momentum products are selling a story the large-cap tape no longer "
            "supports. That is the claim this teardown adjudicates."
        ),

        # ---- BEAT 3 -- PROTOCOL ----------------------------------------------
        md(
            "## 3 -- The protocol\n\n"
            "- **Momentum.** 12-1 trailing return (12 months, skip the most recent month).\n"
            "- **ID.** `sign(PRET) * (%neg - %pos)` over the same formation window, daily returns.\n"
            "- **Double-sort.** Split the cross-section at median ID; inside the chosen half, long "
            "the top 30% by momentum, short the bottom 30%, equal-weight, dollar-neutral.\n"
            "- **Execution.** Form on month-end *t*; enter one trading day later; hold month *t+1* "
            "(one forward lag, no same-bar fill).\n"
            "- **Inference.** Newey-West HAC *t* on the monthly WML and on the low-minus-high FIP "
            "interaction.\n"
            "- **Null.** Label-shuffle placebo inside the low-ID half, seed-robust over 5 seeds.\n"
            "- **Costs.** 5 bps/leg one-way x turnover x NAV; 50 bps/yr borrow on the short leg.\n"
            "- **Universe caveat.** 38 large-cap S&P 500 survivors; bias named on the signal axis."
        ),

        # ---- BEAT 4 -- TEARDOWN ----------------------------------------------
        md("## 4 -- The teardown"),
        md(
            "### 4a -- Positive control: the FIP engine is a faithful detector\n\n"
            "Sweep the planted momentum strength on the synthetic panel (smooth drift -> low-ID "
            "names, lumpy drift -> high-ID names). The low-ID WML should exceed the high-ID WML, "
            "and the gap should grow with strength."
        ),
        code(
            "ctrl = st.synthetic_control(strengths=(0.0, 0.20, 0.40, 0.60), frac=FRAC, seed=510)\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))\n"
            "x = np.arange(len(ctrl)); w = 0.38\n"
            "axes[0].bar(x - w/2, ctrl['low_id_mean_ann']*100, w, color=GREEN, label='LOW-ID')\n"
            "axes[0].bar(x + w/2, ctrl['high_id_mean_ann']*100, w, color=RED, label='HIGH-ID')\n"
            "axes[0].axhline(0, c='k', lw=1)\n"
            "axes[0].set_xticks(x); axes[0].set_xticklabels([f'{s:.2f}' for s in ctrl['mom_strength']])\n"
            "axes[0].set_xlabel('planted strength'); axes[0].set_ylabel('WML mean (%/yr)')\n"
            "axes[0].set_title('Low-ID vs high-ID WML by planted strength'); axes[0].legend()\n"
            "axes[1].plot(ctrl['mom_strength'], ctrl['fip_gap_ann']*100, 'o-', c=AMBER, lw=2)\n"
            "axes[1].axhline(0, c='k', lw=1)\n"
            "axes[1].set_xlabel('planted strength'); axes[1].set_ylabel('FIP gap (low - high, %/yr)')\n"
            "axes[1].set_title('FIP interaction grows monotonically with the planted effect')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(ctrl.round(3).to_string(index=False))"
        ),
        md(
            "The engine scores ~0 at the null and recovers a positive, monotone FIP gap when the "
            "effect is planted. A faithful detector -- so the real-tape result is the market's "
            "verdict, not the method's."
        ),
        md(
            "### 4b -- The information-discreteness distribution (real panel)\n\n"
            "What does ID look like on the latest cross-section? We expect a spread around 0, "
            "with the smoothest trenders at the negative (low-ID) tail."
        ),
        code(
            "if HAVE_REAL:\n"
            "    idm = st.information_discreteness(prices, mp)\n"
            "    last_id = idm.iloc[-1].dropna()\n"
            "    fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "    ax.hist(last_id, bins=20, color=AMBER, edgecolor='white', alpha=0.85)\n"
            "    ax.axvline(last_id.median(), c=RED, lw=2, ls='--', label=f'median={last_id.median():+.3f}')\n"
            "    ax.axvline(0, c=GREY, lw=1.5, ls=':', label='ID=0')\n"
            "    ax.set_xlabel('information discreteness  (low = gradual, high = jumpy)')\n"
            "    ax.set_ylabel('count')\n"
            "    ax.set_title('ID distribution, latest cross-section')\n"
            "    ax.legend(); plt.tight_layout(); plt.show()\n"
            "    print(f'ID median: {last_id.median():+.3f} | range: {last_id.min():+.3f} .. {last_id.max():+.3f}')\n"
            "else:\n"
            "    print('ID distribution requires the real cache.')"
        ),
        md("### 4c -- Real panel: the low-ID vs high-ID WML race"),
        code(
            "if HAVE_REAL:\n"
            "    lo_m, lo_t = s_lo['mean']*100, s_lo['tstat']\n"
            "    hi_m, hi_t = s_hi['mean']*100, s_hi['tstat']\n"
            "    base_m, base_t = s_base['mean']*100, s_base['tstat']\n"
            "    fip_m, fip_t = s_fip['mean']*100, s_fip['tstat']\n"
            "    lo_y = lo.copy(); lo_y['year'] = lo_y.index.year\n"
            "    annual = lo_y.groupby('year')['wml_gross'].apply(lambda x: float((1+x).prod()-1))\n"
            "else:\n"
            "    lo_m, lo_t = R['low_mean'], R['low_t']\n"
            "    hi_m, hi_t = R['high_mean'], R['high_t']\n"
            "    base_m, base_t = R['base_mean'], R['base_t']\n"
            "    fip_m, fip_t = R['fip_mean'], R['fip_t']\n"
            "    annual = pd.Series(R['annual']) / 100\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n"
            "labels = ['Plain\\nmom', 'LOW-ID\\n(gradual)', 'HIGH-ID\\n(jumpy)']\n"
            "vals = [base_m, lo_m, hi_m]; ts = [base_t, lo_t, hi_t]\n"
            "axes[0].bar(labels, vals, color=[GREY, AMBER, RED], width=0.55)\n"
            "axes[0].axhline(0, c='k', lw=1)\n"
            "for i, (v, t) in enumerate(zip(vals, ts)):\n"
            "    axes[0].text(i, v + (0.2 if v >= 0 else -0.7), f'{v:+.1f}%\\nt={t:+.2f}', ha='center', fontsize=8)\n"
            "axes[0].set_ylabel('WML mean (%/yr, gross)')\n"
            "axes[0].set_title(f'FIP gap (low - high) = {fip_m:+.1f}%/yr, t={fip_t:+.2f}')\n"
            "col_yr = [GREEN if v > 0 else RED for v in annual.values]\n"
            "axes[1].bar(annual.index, annual.values*100, color=col_yr)\n"
            "axes[1].axhline(0, c='k', lw=1)\n"
            "axes[1].set_xlabel('year'); axes[1].set_ylabel('LOW-ID WML (%/yr)')\n"
            "axes[1].set_title('Year-by-year LOW-ID WML: a coin-flip')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'LOW-ID: {lo_m:+.2f}%/yr (t {lo_t:+.2f}) | HIGH-ID: {hi_m:+.2f}%/yr (t {hi_t:+.2f}) | '\n"
            "      f'FIP: {fip_m:+.2f}%/yr (t {fip_t:+.2f})')"
        ),
        md(
            f"> The LOW-ID book that should carry the premium earns **{R['low_mean']:.2f}%/yr** at "
            f"HAC *t* = **{R['low_t']:.2f}** -- zero. The positive interaction "
            f"(**+{R['fip_mean']:.2f}%/yr**) comes from the high-ID book being *more* negative, "
            "not from the gradual book being positive. H2 (momentum concentrated in low-ID) is "
            "rejected on this tape."
        ),
        md("### 4d -- The label-shuffle placebo (seed-robust)"),
        code(
            "if HAVE_REAL:\n"
            "    seeds = [1, 7, 42, 510, 2024]\n"
            "    ps = []\n"
            "    for sd in seeds:\n"
            "        pl = st.placebo_pvalue(mp, prices, id_side='low', frac=FRAC, n_perm=400, seed=sd)\n"
            "        ps.append(pl['p_value'])\n"
            "    pl_main = st.placebo_pvalue(mp, prices, id_side='low', frac=FRAC, n_perm=1000, seed=510)\n"
            "    real_mean = pl_main['real_mean_ann']*100\n"
            "    fig, ax = plt.subplots(figsize=(9.0, 4.0))\n"
            "    ax.bar([str(s) for s in seeds], ps, color=GREY)\n"
            "    ax.axhline(0.05, c=RED, ls='--', lw=1.5, label='p=0.05 bar')\n"
            "    ax.set_ylim(0, 1); ax.set_xlabel('RNG seed'); ax.set_ylabel('placebo p-value')\n"
            "    ax.set_title(f'Label-shuffle placebo p (low-ID WML): real mean {real_mean:+.2f}%/yr'); ax.legend()\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print(f'p across seeds: {[round(p,3) for p in ps]}')\n"
            "    print(f'Main p (1000 perms): {pl_main[\"p_value\"]:.3f} | placebo mean {pl_main[\"placebo_mean_ann\"]*100:+.2f}%/yr')\n"
            "else:\n"
            "    print('Placebo seeds (frozen):', R['placebo_seeds'], '| p =', R['placebo_p'])"
        ),
        md(
            f"> The placebo p is **{R['placebo_p']:.2f}**, stable across five seeds "
            f"({min(R['placebo_seeds'].values()):.2f}--{max(R['placebo_seeds'].values()):.2f}). "
            "The real low-ID mean sits *below* the centre of its own shuffled null. There is no "
            "FIP edge here to shuffle away."
        ),
        md("### 4e -- Costs: the gradual book is negative net"),
        code(
            "if HAVE_REAL:\n"
            "    lo_net = st.long_short(mp, prices=prices, id_side='low', frac=FRAC,\n"
            "                           cost_bps=COST_BPS, borrow_ann_bps=BORROW_BPS)\n"
            "    s_g = st.summary(lo['wml_gross']); s_n = st.summary(lo_net['wml_net'])\n"
            "    g_m, n_m = s_g['mean']*100, s_n['mean']*100\n"
            "    g_t, n_t = s_g['tstat'], s_n['tstat']\n"
            "    to = lo['turnover'].mean()*100\n"
            "else:\n"
            "    g_m, n_m = R['low_mean'], R['low_net_mean']\n"
            "    g_t, n_t = R['low_t'], R['low_net_t']; to = R['turnover']\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.2))\n"
            "ax.bar(['gross', 'net of costs\\n+ borrow'], [g_m, n_m], color=[AMBER, RED], width=0.5)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "for i, (v, t) in enumerate(zip([g_m, n_m], [g_t, n_t])):\n"
            "    ax.text(i, v + (0.1 if v >= 0 else -0.25), f'{v:+.2f}%\\nt={t:+.2f}', ha='center', fontsize=9)\n"
            "ax.set_ylabel('LOW-ID WML mean (%/yr)')\n"
            "ax.set_title(f'Costs bury a flat book (turnover {to:.0f}%/mo)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'LOW-ID WML gross {g_m:+.2f}%/yr (t {g_t:+.2f}) -> net {n_m:+.2f}%/yr (t {n_t:+.2f})')"
        ),

        # ---- BEAT 5 -- VERDICT -----------------------------------------------
        md(
            "## 5 -- The verdict\n\n"
            f"- **Signal `NONE`** -- LOW-ID WML {R['low_mean']:.2f}%/yr (HAC *t* {R['low_t']:.2f}), "
            f"FIP interaction +{R['fip_mean']:.2f}%/yr (HAC *t* +{R['fip_t']:.2f}); neither clears "
            f"|t| >= 2, and the seed-robust placebo (p = {R['placebo_p']:.2f}) puts the real mean "
            "below its own null.\n"
            f"- **Tradability `MIRAGE`** -- net {R['low_net_mean']:.2f}%/yr (*t* {R['low_net_t']:.2f}) "
            f"after costs + borrow at {R['turnover']:.0f}%/mo turnover. Nothing to monetise.\n"
            "- **Does gradual beat jumpy? `MIXED`** -- the sign is on the theory's side and the "
            "synthetic control proves the detector works, but the magnitude is statistically "
            "indistinguishable from zero on 38 large-cap survivors."
        ),

        # ---- BEAT 6 -- TRADABILITY -------------------------------------------
        md(
            "## 6 -- Could you trade it?\n\n"
            "No, on three counts:\n\n"
            "1. **Dead base book.** Plain large-cap momentum pays ~0 (+0.18%/yr here); "
            "conditioning a flat signal cannot manufacture a premium.\n"
            "2. **Thin double-sort.** 38 names split at median ID then sorted into 30% legs leaves "
            "~6 names per leg -- noisy, high-turnover, expensive.\n"
            "3. **Survivorship cuts the FIP slice hardest.** Gradual losers drifting into delisting "
            "are precisely the low-ID short candidates a survivor basket omits."
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 -- Going further\n\n"
            "- **Delisting-complete universe.** The original FIP result is on broad CRSP with "
            "delisting returns. Feed [`strategy.long_short`](../frog_in_the_pan/strategy.py) a "
            "delisting-complete panel to lift the survivorship bias on the short leg.\n"
            "- **[507 Cross-Sectional-Momentum](../../507-cross-sectional-momentum/)** -- the "
            "unconditioned base WML this study refines.\n"
            "- **[509 Intermediate-Momentum](../../509-intermediate-momentum/)** -- conditions on "
            "the *timing* of past returns; FIP conditions on their *continuity*.\n"
            "- **[237 Residual-Momentum](../../237-residual-momentum/)** -- strips factor exposure "
            "instead of conditioning on path shape.\n\n"
            "*Think the frog-in-the-pan premium survives net of costs on a delisting-complete "
            "universe? Fork this, add the delisted names, and show the low-ID WML at t > 2 net. "
            "That is the bar.*"
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
