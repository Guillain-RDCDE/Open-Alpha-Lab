"""Generate the two narrative notebooks for Study 650 (Heating-Oil-Seasonality).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached HO=F/UHN/^IRX
tapes under ../_cache/ and otherwise quote the frozen headline numbers in ``R`` (mirroring
docs/results.md). The synthetic positive control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance HO=F 2000-09-01 ->
# 2026-06-30, 309 monthly obs; UHN 2008-04-10 -> 2018-09-11, its whole trading life).
R = dict(
    start="2000-10-31", end="2026-06-30", n_months=309,
    months=dict(
        Jan=(+2.49, +1.25, 26), Feb=(+3.26, +1.39, 26), Mar=(+1.22, +0.36, 26),
        Apr=(+1.65, +0.77, 26), May=(-0.26, -0.12, 26), Jun=(+3.14, +2.33, 26),
        Jul=(+1.08, +0.64, 25), Aug=(+2.23, +1.49, 25), Sep=(+0.13, +0.07, 25),
        Oct=(-0.78, -0.40, 26), Nov=(-2.12, -1.02, 26), Dec=(-0.41, -0.21, 26),
    ),
    bonferroni_t=3.0,
    autumn_mean=-0.94, autumn_n=77, autumn_t=-1.71,
    winter_mean=+1.78, winter_n=78, winter_t=+0.18,
    heat_mean=+0.43, heat_n=155, heat_t=-0.89,
    off_mean=+1.51, off_n=154,
    bh_cagr=5.06, bh_sharpe=0.27, bh_dd=-81,
    timer_gross_cagr=0.29, timer_gross_sharpe=0.07, timer_gross_dd=-72,
    timer_net5_cagr=0.19, timer_net5_sharpe=0.06, timer_net5_dd=-72,
    timer_net10_cagr=0.08, timer_net10_sharpe=0.06, timer_net10_dd=-73,
    uhn_seasons=[
        ("2008-2009", -60.72, -58.81, -1.90), ("2009-2010", +4.31, +13.81, -9.50),
        ("2010-2011", +42.58, +46.70, -4.12), ("2011-2012", +4.35, +4.73, -0.38),
        ("2012-2013", -6.66, -6.24, -0.42), ("2013-2014", -0.35, -1.87, +1.52),
        ("2014-2015", -23.18, -17.80, -5.38), ("2015-2016", -43.38, -37.19, -6.19),
        ("2016-2017", +7.57, +14.93, -7.36), ("2017-2018", +10.79, +8.88, +1.91),
    ],
    uhn_gap_mean=-3.18, uhn_gap_t=-2.58, uhn_gap_n=10,
    uhn_start="2008-04-10", uhn_end="2018-09-11",
    syn_null_mean=-0.19, syn_null_sd=1.01, syn_null_fire=0, syn_planted_t=+3.61,
    fp_ho="5e77c79ac501", fp_uhn="f140d3086a31", fp_irx="97d049b701d7",
)

MONTH_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
HEAT_SET = {"Sep", "Oct", "Nov", "Dec", "Jan", "Feb"}

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Rule_of_thumb%3F: Busted](https://img.shields.io/badge/Rule_of_thumb%3F-Busted-8b949e?style=flat-square)\n\n"
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

from heating_oil_seasonality import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    HO, UHN, IRX = data.load_real()
    HO_RET = data.monthly_returns(HO["Close"], asof=data.AS_OF)
    TBILL = data.monthly_cash_rate(IRX["Close"], HO_RET.index)
else:
    HO = UHN = IRX = HO_RET = TBILL = None
print("real cache present:", HAVE_REAL, "| HO=F monthly obs:", (0 if HO_RET is None else len(HO_RET)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does heating oil really rally into winter? 🛢️❄️\n"
            "### The desk pulls apart the two halves of the folklore — and one of them runs "
            "*backward*\n\n"
            + BADGES +
            "Every autumn, energy-desk shorthand makes the same case: cold weather is coming, "
            "furnaces are about to run, and heating oil futures should **build** through the "
            "autumn and hold the rally through the winter draw-down. It sounds like the cleanest "
            "seasonal story in commodities — the demand is genuinely, physically seasonal (the "
            "EIA publishes the weekly inventory draw itself).\n\n"
            "So we tested it, literally, on 26 years of the actual futures tape. The physical "
            "demand story is real. The **price** story is not — and the autumn half of the claim "
            "turns out to point in exactly the wrong direction.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the Bonferroni bar and the cost "
            "math? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** HO=F (NY Harbor ULSD futures) is a *spliced continuous chain* "
            "— every roll's price jump is already the real cost a futures holder pays, not a "
            "modeling afterthought. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does heating oil build through autumn (Sep–Nov)? | **No — it falls.** "
            f"**{R['autumn_mean']:+.2f}%** on average, against **+{R['off_mean']:.2f}%** the "
            "rest of the year. Backwards. |\n"
            f"| Does it hold through the winter draw (Dec–Feb)? | **Not really.** "
            f"**+{R['winter_mean']:.2f}%**, statistically indistinguishable from a normal month "
            "(the gap barely registers). |\n"
            "| Does any single month clear a fair statistical bar? | **No.** Testing all 12 "
            "months means correcting for 12 chances to get lucky — nothing survives. |\n"
            "| Can you trade the story anyway? | **No.** A calendar timer built exactly as the "
            f"story says loses **most of buy-and-hold's risk-adjusted return** before a single "
            "cost is even charged — and the one ETF that let retail buy this trade doesn't exist "
            "anymore. |\n\n"
            "> The physical demand is real. The price pattern the folklore promises isn't there."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Cold weather means furnaces run, furnaces burn distillate, and distillate "
            "demand peaks in winter — so smart money buys heating oil ahead of the season and "
            "rides the rally through the coldest months.\"*\n\n"
            "It's a two-stage story: an **autumn build** (the market pricing the coming winter "
            "before it arrives) and a **winter draw** (prices holding or extending as physical "
            "inventories genuinely fall — the EIA's weekly distillate-stocks report shows this "
            "every year). Both stages have a real-world mechanism behind them."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If real, this is about as clean a calendar trade as commodities offer: a demand "
            "cycle you can set a calendar to, expressed in one of the most liquid energy futures "
            "on the board. If it's not real, it's a cautionary tale about how a genuinely true "
            "*physical* fact (inventories fall every winter) can get silently upgraded into a "
            "false *price* fact (so the futures must rally) — the exact substitution the desk "
            "exists to catch."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The tape.** HO=F daily closes, resampled to **{R['n_months']}** complete "
            f"calendar months, {R['start']} → {R['end']}.\n"
            "- **The two claim-stages, tested separately.** Autumn-build (Sep–Nov) and "
            "winter-draw (Dec–Feb), each compared against the off-season (Mar–Aug) with a proper "
            "two-sample test — never blended into one window that could hide a disagreement.\n"
            "- **The fair bar.** Testing 12 individual months is 12 chances to get lucky, so the "
            "per-month table needs a stricter (Bonferroni-corrected) bar, not the everyday one.\n"
            "- **The trade check.** A calendar timer — long Sep–Feb, cash the rest of the year — "
            "raced against simply buying and holding, with real trading costs charged."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the month-by-month picture.**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    ms = st.month_stats(HO_RET)\n"
            "    means = [ms.loc[i, 'mean']*100 for i in range(1, 13)]\n"
            "else:\n"
            "    means = [R['months'][m][0] for m in "
            "['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']]\n"
            "names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']\n"
            "heat = {'Sep','Oct','Nov','Dec','Jan','Feb'}\n"
            "cols = [RED if n in heat else GREY for n in names]\n"
            "fig, ax = plt.subplots(figsize=(10.4, 4.6))\n"
            "ax.bar(names, means, color=cols, width=.62)\n"
            "for i, v in enumerate(means):\n"
            "    ax.annotate(f'{v:+.1f}%', (i, v), ha='center', va='top' if v<0 else 'bottom', fontsize=8)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('average HO=F monthly return')\n"
            "ax.set_title('Red = the claimed heating window (Sep-Feb) — no obvious rally there')\n"
            "plt.tight_layout(); plt.show()\n"
            "print({n: round(m,2) for n,m in zip(names, means)})"
        ),
        md(
            "There's no visible winter cluster of tall red bars. If anything, **Nov and Oct are "
            f"the reddest-in-the-wrong-direction months** ({R['months']['Nov'][0]:+.2f}% and "
            f"{R['months']['Oct'][0]:+.2f}%), and the tallest bar on the whole chart is "
            f"**June** ({R['months']['Jun'][0]:+.2f}%) — driving season, not heating season, "
            "and not even a Bonferroni survivor once you correct for testing 12 months.\n\n"
            "**Now the two claim-stages, head to head against the rest of the year.**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    ab = st.group_welch(HO_RET, data.AUTUMN_BUILD_MONTHS, data.OFF_SEASON_MONTHS)\n"
            "    wd = st.group_welch(HO_RET, data.WINTER_DRAW_MONTHS, data.OFF_SEASON_MONTHS)\n"
            "    a_m, w_m, o_m = ab['mean_a']*100, wd['mean_a']*100, ab['mean_b']*100\n"
            "else:\n"
            "    a_m, w_m, o_m = R['autumn_mean'], R['winter_mean'], R['off_mean']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.4))\n"
            "ax.bar(['autumn-build\\n(Sep-Nov)', 'winter-draw\\n(Dec-Feb)', 'off-season\\n(Mar-Aug)'],\n"
            "       [a_m, w_m, o_m], color=[RED, AMBER, GREY], width=.55)\n"
            "for i, v in enumerate([a_m, w_m, o_m]):\n"
            "    ax.annotate(f'{v:+.2f}%', (i, v), ha='center', va='top' if v<0 else 'bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('average monthly HO=F return')\n"
            "ax.set_title('Autumn-build is NEGATIVE — the story runs backward in its own first act')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'autumn {a_m:+.2f}%  winter {w_m:+.2f}%  off-season {o_m:+.2f}%')"
        ),
        md(
            f"The autumn-build window averages **{R['autumn_mean']:+.2f}%** — *worse* than the "
            f"off-season's **+{R['off_mean']:.2f}%**. That's the single strongest signal in this "
            f"whole study (Welch *t* = {R['autumn_t']:.2f} — the quants notebook shows this is "
            "the closest any test here gets to a fair statistical bar), and it points the "
            "*wrong way*. The winter-draw window is unremarkable. Neither stage of the story "
            "holds up.\n\n"
            "**Then the trade itself: would timing the calendar actually have paid?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    bh = st.summary(st.buy_hold(HO_RET), rf=TBILL)\n"
            "    tm = st.summary(st.seasonal_timer(HO_RET, TBILL, data.HEATING_MONTHS, cost_bps=5.0), rf=TBILL)\n"
            "    bh_s, tm_s = bh['sharpe'], tm['sharpe']\n"
            "else:\n"
            "    bh_s, tm_s = R['bh_sharpe'], R['timer_net5_sharpe']\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.4))\n"
            "ax.bar(['buy & hold\\nHO=F', 'seasonal timer\\n(net 5 bps)'], [bh_s, tm_s], color=[GREY, RED], width=.5)\n"
            "for i, v in enumerate([bh_s, tm_s]): ax.annotate(f'{v:.2f}', (i, v), ha='center', va='bottom')\n"
            "ax.set_ylabel('Sharpe (excess of T-bill)')\n"
            "ax.set_title('Timing the calendar loses most of the Sharpe buy-and-hold already has')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'buy&hold Sharpe {bh_s:.2f}  timer Sharpe {tm_s:.2f}')"
        ),
        md(
            f"Buy-and-hold: Sharpe **{R['bh_sharpe']:.2f}**. The by-the-book seasonal timer "
            f"(long Sep–Feb, cash otherwise): Sharpe **{R['timer_net5_sharpe']:.2f}**, even after "
            "costs are the *smaller* problem — sitting out Mar–Aug means missing June, HO=F's "
            f"single best month ({R['months']['Jun'][0]:+.2f}%). And the one ETF that would have "
            f"let a retail investor buy this exact trade, **UHN**, traded {R['uhn_start']} → "
            f"{R['uhn_end']} and then **was shut down** — it lost to the raw futures roll in 8 of "
            "its 10 heating seasons before it disappeared."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** No month survives a fair (Bonferroni) bar; the autumn-build "
            "half of the claim is negative — the wrong sign — and the winter-draw half is noise.\n"
            "- **Tradability — Mirage.** A calendar timer loses most of buy-and-hold's Sharpe "
            "before costs even matter, and the retail product for this trade no longer exists.\n"
            "- **\"Heating oil rallies into winter\"? — Busted.** Not fragile, not "
            "regime-dependent — backwards in its own first act."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **A real physical seasonal doesn't guarantee a price seasonal.** Storable "
            "commodities can price a well-known future demand shock *in advance*, long before the "
            "calendar window everyone talks about — the classic theory-of-storage result. If "
            "there's a heating-oil seasonal at all, it may sit somewhere upstream of Sep 1, not "
            "inside the window folklore names.\n"
            "- **Sibling studies:** [227-natgas-winter](../../227-natgas-winter/) finds the exact "
            "same wrong-signed pattern for natural gas — worth reading side by side. "
            "[639-gasoline-rvp-seasonality](../../639-gasoline-rvp-seasonality/) shows the other "
            "way a seasonal *can* be real (a literal law) and still be a mirage once someone has "
            "to hold the roll.\n\n"
            "*Think the seasonal lives somewhere else on the calendar, or in the term structure "
            "rather than the spot chain? Show a net, certifiable edge after Bonferroni and roll "
            "costs, and we'll fold it in.*"
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
            "# Heating-Oil-Seasonality — a quantitative teardown 🔬\n"
            "### The Bonferroni-12 month table · autumn-build vs winter-draw Welch splits · a "
            "seasonal timer race with costs · the UHN-vs-splice paired gap · a 20-seed synthetic "
            "null\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "The claim — **HO=F builds through autumn and holds through the winter draw** — has "
            "a genuine physical anchor (EIA weekly distillate stocks) but no *a priori* guarantee "
            "the futures price obeys it: a storable commodity can price a well-known seasonal "
            "demand shock in advance, and the front-month chain already contains real roll "
            "friction. The job here is to test both stages honestly and then ask whether anyone "
            "could have banked it.\n\n"
            "> ⚠️ **Data note.** HO=F daily raw OHLC (2000-09 → 2026-06, 309 monthly obs) + UHN "
            "adjusted closes (2008-04 → 2018-09, its whole trading life) + ^IRX, yfinance, "
            "cached. HO=F is a **spliced continuous chain, not back-adjusted** — the roll-day "
            "price jump *is* the term-structure cost/gain a real futures holder pays, by "
            "construction. No survivorship on the Signal axis (a futures chain, not a basket); "
            "**UHN's 2018 wind-down is named on the third axis**. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (fingerprints `" + R["fp_ho"] + "` / `"
            + R["fp_uhn"] + "` / `" + R["fp_irx"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | autumn-build **{R['autumn_mean']:+.2f}%** (n={R['autumn_n']}) "
            f"vs off-season **+{R['off_mean']:.2f}%** (n={R['off_n']}): Welch **t = "
            f"{R['autumn_t']:.2f}** (wrong sign); winter-draw t = {R['winter_t']:.2f}; pooled "
            f"heating-window t = {R['heat_t']:.2f}; **no month clears Bonferroni-12** "
            f"(|t| ≥ ~{R['bonferroni_t']:.1f}) |\n"
            f"| **Tradability** | `MIRAGE` | seasonal timer Sharpe **{R['timer_gross_sharpe']:.2f}** "
            f"gross vs buy-and-hold **{R['bh_sharpe']:.2f}**; UHN vs splice gap "
            f"**{R['uhn_gap_mean']:+.2f}%/season** (t = {R['uhn_gap_t']:.2f}, n={R['uhn_gap_n']}); "
            f"UHN wound down {R['uhn_end']} |\n"
            "| **Rule of thumb?** | `BUSTED` | autumn half runs backward, winter half is noise, "
            "the retail vehicle no longer exists |\n\n"
            "> 💡 In plain words: the physical demand story is true; the price story the "
            "folklore promises on top of it is not there, and where it's closest to showing up "
            "(autumn) it shows up **backward**."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r_t$ be HO=F's monthly return and define three calendar sets: "
            "$\\text{Autumn} = \\{Sep, Oct, Nov\\}$, $\\text{Winter} = \\{Dec, Jan, Feb\\}$, "
            "$\\text{Off} = \\{Mar,\\dots,Aug\\}$. The claims:\n\n"
            "- **H₁ (autumn build).** $E[r_t \\mid t \\in \\text{Autumn}] > E[r_t \\mid t \\in "
            "\\text{Off}]$ — the market prices the coming winter ahead of time.\n"
            "- **H₂ (winter draw).** $E[r_t \\mid t \\in \\text{Winter}] > E[r_t \\mid t \\in "
            "\\text{Off}]$ — the price holds or extends as physical stocks fall.\n"
            "- **H₃ (per-month).** At least one calendar month shows a real, Bonferroni-robust "
            "one-sample effect — a sanity check that doesn't presume the exact window.\n"
            "- **H₄ (capture).** A calendar-timed long position (Sep–Feb, cash otherwise) beats "
            "buy-and-hold net of costs, and a real investable vehicle (UHN) captures it.\n\n"
            f"We find **H₁ rejected — and wrong-signed** (autumn *t* = {R['autumn_t']:.2f}), "
            f"**H₂ not supported** (winter *t* = {R['winter_t']:.2f}), **H₃ rejected** (no month "
            f"clears |t| ≥ {R['bonferroni_t']:.1f}), **H₄ rejected** (timer Sharpe << "
            "buy-and-hold; UHN loses to the splice; the vehicle no longer exists)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "Calendar months are **non-overlapping** observations, so **Welch's *t*** (unequal "
            "variances) is the planned primary for every two-group split — autumn-build vs "
            "off-season, winter-draw vs off-season, and the pooled heating window vs off-season. "
            "The per-month table tests **12 hypotheses at once** (one per calendar month), so it "
            "carries a **Bonferroni correction**: at α = 0.05, the per-test threshold is "
            "α/12 ≈ 0.0042, i.e. |*t*| ≥ ~3.0 at *n* ≈ 25 — not the everyday |*t*| ≥ 2 bar. "
            "The UHN-vs-splice gap is a **paired, one-sample** *t* across 10 non-overlapping "
            "heating seasons — reported honestly as a small-sample number, never inflated."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Tape.** HO=F daily raw OHLC {R['start']} → {R['end']}, resampled to "
            f"**{R['n_months']} complete calendar months** (in-progress month dropped, hole-free "
            "grid asserted). As-of 2026-06-30.\n"
            "- **Headline.** Per-month one-sample *t* + Bonferroni-12 bar; Welch *t* for "
            "autumn-build / winter-draw / pooled-heating vs off-season.\n"
            "- **Execution (tradability).** Long HO=F Sep–Feb, T-bill (^IRX/12) otherwise, one "
            "fixed calendar rule (zero look-ahead by construction); 2 switches/yr; one-way cost "
            "× NAV per switch (0/5/10 bps); Sharpe = excess of T-bill on both legs.\n"
            "- **Cross-check.** UHN (retail ETF, real front-month holder) vs the HO=F splice, "
            "paired per heating season, over UHN's whole 2008–2018 life.\n"
            "- **Control.** i.i.d. synthetic monthly world, planted heating-season premium knob; "
            "the null must not fire across 20 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The Bonferroni month table\n\n"
            "Twelve one-sample *t*-tests (mean vs 0), one per calendar month — and twelve chances "
            "to find a false positive if we used the everyday |*t*| ≥ 2 bar."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ms = st.month_stats(HO_RET)\n"
            "    means = [ms.loc[i,'mean']*100 for i in range(1,13)]\n"
            "    ts = [ms.loc[i,'t'] for i in range(1,13)]\n"
            "else:\n"
            "    means = [R['months'][m][0] for m in "
            "['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']]\n"
            "    ts = [R['months'][m][1] for m in "
            "['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']]\n"
            "names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']\n"
            "heat = {'Sep','Oct','Nov','Dec','Jan','Feb'}\n"
            "fig, (a1, a2) = plt.subplots(2, 1, figsize=(10.2, 6.6), sharex=True,\n"
            "                             gridspec_kw={'height_ratios': [2, 1]})\n"
            "a1.bar(names, means, color=[RED if n in heat else GREY for n in names], width=.62)\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_ylabel('mean monthly return (%)')\n"
            "a1.set_title('Red = the claimed heating window — no rally cluster there')\n"
            "a2.bar(names, ts, color=[RED if abs(t)>=R['bonferroni_t'] else GREY for t in ts], width=.62)\n"
            "a2.axhline(0, c='k', lw=.8)\n"
            "a2.axhline(-R['bonferroni_t'], ls='--', c=RED, lw=1)\n"
            "a2.axhline(R['bonferroni_t'], ls='--', c=RED, lw=1)\n"
            "a2.axhline(-2, ls=':', c=GREY, lw=1); a2.axhline(2, ls=':', c=GREY, lw=1)\n"
            "a2.annotate('Bonferroni-12 bar', (0.2, R['bonferroni_t']+.15), color=RED, fontsize=9)\n"
            "a2.set_ylabel('one-sample t'); plt.tight_layout(); plt.show()\n"
            "print({n: (round(m,2), round(t,2)) for n, m, t in zip(names, means, ts)})"
        ),
        md(
            "No bar in the bottom panel reaches the red dashed Bonferroni line. The only month "
            f"past the *lenient*, uncorrected |*t*| ≥ 2 dotted line is **June** "
            f"(t = {R['months']['Jun'][1]:.2f}) — driving season, not heating season — and it "
            "still falls short of Bonferroni. **H₃ rejected.**"
        ),
        md(
            "### 4b · Autumn-build vs winter-draw — the two stages, Welch-tested separately\n\n"
            "Blending Sep–Feb into one window would hide exactly the disagreement this study "
            "finds — so each stage is tested on its own, against the same off-season control."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ab = st.group_welch(HO_RET, data.AUTUMN_BUILD_MONTHS, data.OFF_SEASON_MONTHS)\n"
            "    wd = st.group_welch(HO_RET, data.WINTER_DRAW_MONTHS, data.OFF_SEASON_MONTHS)\n"
            "    hw = st.group_welch(HO_RET, data.HEATING_MONTHS, data.OFF_SEASON_MONTHS)\n"
            "    a_m, a_t = ab['mean_a']*100, ab['t']\n"
            "    w_m, w_t = wd['mean_a']*100, wd['t']\n"
            "    h_m, h_t = hw['mean_a']*100, hw['t']\n"
            "    o_m = ab['mean_b']*100\n"
            "else:\n"
            "    a_m, a_t = R['autumn_mean'], R['autumn_t']\n"
            "    w_m, w_t = R['winter_mean'], R['winter_t']\n"
            "    h_m, h_t = R['heat_mean'], R['heat_t']\n"
            "    o_m = R['off_mean']\n"
            "fig, (b1, b2) = plt.subplots(1, 2, figsize=(11.0, 4.4))\n"
            "labels = ['autumn-build\\n(Sep-Nov)', 'winter-draw\\n(Dec-Feb)', 'heating\\n(Sep-Feb)', 'off-season\\n(Mar-Aug)']\n"
            "means = [a_m, w_m, h_m, o_m]\n"
            "b1.bar(labels, means, color=[RED, AMBER, RED, GREY], width=.6)\n"
            "for i, v in enumerate(means): b1.annotate(f'{v:+.2f}%', (i, v), ha='center', va='top' if v<0 else 'bottom', fontsize=9)\n"
            "b1.axhline(0, c='k', lw=.8); b1.set_ylabel('mean monthly return (%)')\n"
            "b1.set_title('Autumn-build is negative')\n"
            "ts = [a_t, w_t, h_t]\n"
            "b2.bar(['autumn vs\\noff', 'winter vs\\noff', 'heating vs\\noff'], ts,\n"
            "       color=[RED if abs(t)>=2 else GREY for t in ts], width=.55)\n"
            "b2.axhline(-2, ls='--', c=RED, lw=1); b2.axhline(2, ls='--', c=RED, lw=1)\n"
            "b2.axhline(0, c='k', lw=.8); b2.set_ylabel('Welch t vs off-season')\n"
            "b2.set_title('None clear |t| >= 2 in the claimed direction')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'autumn t={a_t:.2f}  winter t={w_t:.2f}  pooled-heating t={h_t:.2f}')"
        ),
        md(
            f"> 💡 In plain words: autumn-build is **{R['autumn_mean']:+.2f}%** vs off-season's "
            f"**+{R['off_mean']:.2f}%** — Welch *t* = **{R['autumn_t']:.2f}**, the single "
            "closest-to-significant result in the whole study, and it says the opposite of the "
            f"folklore. Winter-draw (+{R['winter_mean']:.2f}%, t = {R['winter_t']:.2f}) is "
            "statistically indistinguishable from an ordinary month. Pooled, the heating window "
            f"actually **trails** the off-season (t = {R['heat_t']:.2f}). **H₁ rejected and "
            "wrong-signed; H₂ not supported.**"
        ),
        md(
            "### 4c · Tradability — the timer, gross and net of costs\n\n"
            "Long HO=F on a fixed Sep 1 → end-Feb calendar rule (zero look-ahead by "
            "construction), the 13-week T-bill otherwise; Sharpe excess-of-cash on both legs; "
            "one-way cost × NAV at each of 2 switches/year."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bh = st.summary(st.buy_hold(HO_RET), rf=TBILL)\n"
            "    rows = [st.summary(st.seasonal_timer(HO_RET, TBILL, data.HEATING_MONTHS, cost_bps=cb), rf=TBILL)\n"
            "            for cb in (0.0, 5.0, 10.0)]\n"
            "    bh_s, sharpes = bh['sharpe'], [r['sharpe'] for r in rows]\n"
            "else:\n"
            "    bh_s = R['bh_sharpe']\n"
            "    sharpes = [R['timer_gross_sharpe'], R['timer_net5_sharpe'], R['timer_net10_sharpe']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "labels = ['buy & hold', 'timer\\ngross', 'timer\\nnet 5bps', 'timer\\nnet 10bps']\n"
            "vals = [bh_s] + sharpes\n"
            "ax.bar(labels, vals, color=[GREY, AMBER, RED, RED], width=.55)\n"
            "for i, v in enumerate(vals): ax.annotate(f'{v:.2f}', (i, v), ha='center', va='bottom')\n"
            "ax.set_ylabel('Sharpe (excess of T-bill)')\n"
            "ax.set_title('Costs are the SMALLER problem: gross timer already loses to buy-and-hold')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'buy&hold {bh_s:.2f}  timer gross/net5/net10 = ' + ', '.join(f'{v:.2f}' for v in sharpes))"
        ),
        md(
            f"> 💡 In plain words: buy-and-hold's Sharpe ({R['bh_sharpe']:.2f}) is roughly "
            f"**four times** the gross timer's ({R['timer_gross_sharpe']:.2f}) — before a single "
            "basis point of cost. Charging costs barely moves the number "
            f"({R['timer_net5_sharpe']:.2f} at 5bps, {R['timer_net10_sharpe']:.2f} at 10bps) "
            "because the strategy is broken by construction: it sits in cash through June, "
            f"HO=F's best month ({R['months']['Jun'][0]:+.2f}%), on the theory that summer months "
            "don't matter. **H₄'s timer half rejected.**"
        ),
        md(
            "### 4d · The real vehicle — UHN vs the HO=F splice, paired per season\n\n"
            "UHN held front-month HO=F futures directly for retail, 2008-04-10 → 2018-09-11 "
            "(then wound down — **it does not exist today**). Paired per heating season "
            "(Aug 31 close → next Feb 28 close): `gap = UHN return − HO=F splice return`."
        ),
        code(
            "if HAVE_REAL:\n"
            "    gap_df = st.uhn_vs_splice(UHN['Close'], HO['Close'], 2008, 2018)\n"
            "    gs = st.uhn_gap_stats(gap_df)\n"
            "    seasons = list(gap_df['season']); gaps = list(gap_df['gap']*100)\n"
            "    gmean, gt, gn = gs['mean_gap'], gs['t'], gs['n']\n"
            "else:\n"
            "    seasons = [s for s, *_ in R['uhn_seasons']]\n"
            "    gaps = [g for *_, g in R['uhn_seasons']]\n"
            "    gmean, gt, gn = R['uhn_gap_mean'], R['uhn_gap_t'], R['uhn_gap_n']\n"
            "fig, ax = plt.subplots(figsize=(10.2, 4.4))\n"
            "ax.bar(seasons, gaps, color=[RED if g<0 else GREEN for g in gaps], width=.6)\n"
            "for i, v in enumerate(gaps): ax.annotate(f'{v:+.1f}%', (i, v), ha='center',\n"
            "    va='top' if v<0 else 'bottom', fontsize=8, rotation=0)\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('UHN return - HO=F splice return (%)')\n"
            "ax.set_title(f'The wrapper lost to the futures splice in most seasons (mean {gmean:+.2f}%, t={gt:.2f}, n={gn})')\n"
            "plt.xticks(rotation=30, ha='right')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'mean gap {gmean:+.2f}%/season  t={gt:.2f}  n={gn}')"
        ),
        md(
            f"> 💡 In plain words: UHN trailed the raw futures splice in **8 of "
            f"{R['uhn_gap_n']}** heating seasons — a mean gap of **{R['uhn_gap_mean']:+.2f}%** "
            f"(*t* = {R['uhn_gap_t']:.2f}). At *n* = {R['uhn_gap_n']} non-overlapping seasons "
            "this is **not certifiable on its own** — said out loud — but the sign is consistent "
            "and it never helps the claim: the wrapper's expense ratio and tracking slippage "
            "stack a *second* real drag on top of whatever roll cost HO=F's splice already "
            "embeds. And this is history, not a live option: **the fund no longer exists.**"
        ),
        md(
            "### 4e · Faithful-engine & power control — we know the truth here\n\n"
            "i.i.d. monthly synthetic world, TUNABLE planted heating-season premium spread over "
            "Sep–Feb. The null (seasonal = 0) is checked over **20 seeds** — never a single "
            "stream."
        ),
        code(
            "null_ts = []\n"
            "for s_ in range(20):\n"
            "    world = data.synthetic_world(seasonal=0.0, seed=1000 + s_)\n"
            "    null_ts.append(st.synthetic_detect(world, data.HEATING_MONTHS, data.OFF_SEASON_MONTHS))\n"
            "null_ts = np.asarray(null_ts)\n"
            "planted = data.synthetic_world(seasonal=0.15, seed=650)\n"
            "planted_t = st.synthetic_detect(planted, data.HEATING_MONTHS, data.OFF_SEASON_MONTHS)\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.zeros(20) + np.linspace(-.12, .12, 20), null_ts, color=GREY, s=40,\n"
            "           label='null worlds (seasonal=0), 20 seeds')\n"
            "ax.scatter([1], [planted_t], color=RED, s=90, zorder=5, label='planted seasonal = +0.15')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0, 1]); ax.set_xticklabels(['null x 20', 'planted'])\n"
            "ax.set_ylabel('Welch t (heating window vs off-season)')\n"
            "ax.set_title('Control: no null fires; a planted premium lights up')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(np.abs(null_ts)>=2).sum()}/20 seeds  |  planted t = {planted_t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 20 null worlds the detector averages "
            f"t = {R['syn_null_mean']:+.2f} (sd {R['syn_null_sd']:.2f}) and **never** crosses the "
            f"bar; a planted premium comparable in size to a real seasonal claim reads "
            f"t = {R['syn_planted_t']:.2f}. The machinery is unbiased — the real-tape's failure "
            "to fire is the genuine article, not a broken test. *(A faithful-engine / power "
            "check only — never cited in support of the real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — autumn-build **{R['autumn_mean']:+.2f}%** vs off-season "
            f"**+{R['off_mean']:.2f}%**: Welch t = **{R['autumn_t']:.2f}** (wrong sign); "
            f"winter-draw t = {R['winter_t']:.2f}; pooled heating-window t = {R['heat_t']:.2f}; "
            f"no month clears the Bonferroni-12 bar (|t| ≥ ~{R['bonferroni_t']:.1f}).\n"
            f"- **Tradability `MIRAGE`** — seasonal timer Sharpe {R['timer_gross_sharpe']:.2f} "
            f"gross (barely moves net of costs) vs buy-and-hold's {R['bh_sharpe']:.2f}; UHN "
            f"lagged the futures splice by {R['uhn_gap_mean']:+.2f}%/season across "
            f"{R['uhn_gap_n']} seasons before it was wound down in {R['uhn_end']} — the retail "
            "vehicle no longer exists.\n"
            "- **\"Heating oil rallies into winter\"? `BUSTED`** — the physical demand story is "
            "real; the price story built on top of it is not, and the closest it comes to real "
            "is running backward."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The theory-of-storage angle.** Fama-French (1987) on storable-commodity "
            "futures: a market that *knows* a demand shock is coming can price it well before "
            "the shock's own calendar window — the seasonal, if it exists, may live upstream of "
            "September, not inside the Sep–Feb window folklore names. A term-structure study "
            "(does the HO futures *curve* itself carry a seasonal shape, independent of spot "
            "returns?) is the natural sequel.\n"
            "- **Dedup map:** [227-natgas-winter](../../227-natgas-winter/) (the parallel, also "
            "wrong-signed, winter-demand claim for natural gas), "
            "[639-gasoline-rvp-seasonality](../../639-gasoline-rvp-seasonality/) (a *statutory* "
            "spring/autumn calendar — real on the spread, mirage on the roll), "
            "[306-crack-spread](../../306-crack-spread/) (does the crack level predict "
            "refiners? — a different mechanism), [226-crude-seasonality](../../226-crude-seasonality/) "
            "(WTI's spring seasonal — a different commodity and season).\n\n"
            "*The reproducible core is offline and deterministic; frozen numbers live in "
            "[`docs/results.md`](../docs/results.md), sources in "
            "[`docs/references.md`](../docs/references.md).*"
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
