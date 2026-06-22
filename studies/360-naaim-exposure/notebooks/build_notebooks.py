"""Generate the two narrative notebooks for Study 360 (NAAIM-Exposure).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic: the real-tape cells read the cached weekly
NAAIM Number (``_cache/naaim_weekly.csv``) joined to cached SPY total-return closes and
otherwise quote the frozen headline numbers in ``R`` (mirroring docs/results.md). The
synthetic positive control runs anywhere with no network.
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (NAAIM weekly Exposure
# Index + SPY total return, 2006-07-05 -> 2026-06-10, 1,040 weeks; as-of 2026-06-22).
R = dict(
    window="2006-07-05 -> 2026-06-10", weeks=1040, fp="b887999dd648",
    naaim_lo=-3.6, naaim_hi=120.6, naaim_mean=67.3,
    uncond_yr=12.0, uncond_t=3.42,
    low_yr=16.2, low_t=1.92, low_n=347,
    mid_yr=11.3, mid_t=2.15, mid_n=346,
    high_yr=8.6, high_t=2.49, high_n=347,
    gap_yr=7.6,
    ls_yr=2.52, ls_t=0.81,
    beta_sd=-0.040, beta_t=-0.40, r2=0.0003,
    ov_yr=8.68, ov_sr=0.57, ov_t=2.72, ov_inmkt=67,
    bh_yr=12.04, bh_sr=0.73,
    sub=[("2006-2012", 6.3, 0.34, 1.2, 0.17, 339),
         ("2013-2019", 22.5, 2.28, 12.2, 2.57, 364),
         ("2020-2026", 22.0, 1.37, 12.0, 1.54, 337)],
    syn_null_t=-0.37, syn_edge=0.004, syn_edge_t=-6.12,
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Pros_smarter%3F: Not_supported](https://img.shields.io/badge/Pros_smarter%3F-Not_supported-8b949e?style=flat-square)\n\n"
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from naaim_exposure import data, strategy as st

HAVE_REAL = data.have_real()
PANEL = data.build_real_panel() if True else None   # always builds (weekly cache or real fallback)
print("real NAAIM weekly cache present:", HAVE_REAL, "| weeks in panel:", len(PANEL))
"""

# The frozen headline dict is embedded into the first code cell so every downstream cell
# can quote it whether or not the weekly cache is present.
BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# When the pros are all-in, should you sell? 📉\n"
            "### The NAAIM manager-exposure index as a contrarian tell, in plain English\n\n"
            + BADGES +
            "Every Wednesday, a group of professional money managers (NAAIM) reports how much "
            "of their clients' money is actually **in the stock market** right now -- from "
            "**0%** (everything in cash) to **200%** (leveraged double-long). The folklore: this "
            "is the *smart money*, so when they're **all-in** the easy gains are over (**sell**), "
            "and when they've **bailed to cash** the bottom is near (**buy**). Do the opposite of "
            "the pros and win.\n\n"
            "It's a great story, and the **direction** is genuinely right -- weeks *after* the pros "
            "panic really do beat weeks *after* they pile in. But the effect is small, unreliable, "
            "and as a trading rule it **loses to just staying invested**. This notebook shows why.\n\n"
            "> 📓 **Plain-language layer.** Want the HAC *t*-stats, the predictive regression and "
            "the cost-charged overlay? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice** -- research and education. Every chart below is drawn by "
            "the code beside it; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Do weeks after the pros bail to cash beat weeks after they go all-in? | **Yes -- "
            f"in direction.** Cash-regime weeks earn **+{R['low_yr']:.0f}%/yr** next week vs "
            f"**+{R['high_yr']:.0f}%/yr** for all-in weeks -- a **{R['gap_yr']:.0f} pp/yr** gap, "
            "the contrarian way. |\n"
            "| So fade the pros for profit? | **No.** The gap isn't statistically reliable (it "
            "fails the significance bar), and *every* regime -- even all-in -- still earns a solid "
            "**positive** return. \"All-in\" is a smaller buy, not a sell. |\n"
            "| Does a contrarian timing rule beat buy-and-hold? | **No -- it loses.** Sitting out "
            f"the all-in weeks nets **+{R['ov_yr']:.1f}%/yr** vs **+{R['bh_yr']:.1f}%/yr** for "
            "just staying invested. The pros go all-in *during* the best part of bull markets. |\n"
            "| Are the pros smarter than retail here? | **No.** This professional gauge gives the "
            "**same** weak, untradeable tilt as the individual-investor survey "
            "([Study 257](../../257-aaii-sentiment/)). |\n\n"
            "> The direction is real. The edge is too small to bank, and fading the pros means "
            "skipping the rallies they were right to ride."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"The NAAIM Exposure Index tells you what professional active managers are actually "
            "doing with their money -- not what they say, what they **do**. When they're maxed out "
            "(near 100-200%), the market is over-loved and due to fall: sell. When they've fled to "
            "cash (near 0), everyone who was going to sell already has: buy. It's the smart money, "
            "so fading its extremes beats fading a retail poll.\"*\n\n"
            "The index has been published **free, weekly, since July 2006**. It's the mean equity "
            "exposure across NAAIM member firms on a **0-200%** scale. We take it at its strongest: "
            "real positioning by people who trade for a living."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If true, this would be a free, public, weekly market-timing signal -- and it would say "
            "something deep: that the people *closest* to the market are systematically wrong at the "
            "extremes, and that you can profit just by doing the opposite. That's a strong claim "
            "about how markets work. Two things have to hold for it to pay: (1) the exposure reading "
            "this week has to actually *predict* next week's return, and (2) trading on it has to "
            "beat the dead-simple alternative of buying and holding. We can measure both directly."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We line up **{R['weeks']:,} weeks** of the real NAAIM number ({R['window']}) against "
            "the **total return** of SPY (the S&P 500 ETF, dividends reinvested). For each week we "
            "know the managers' exposure *now* and the market's return over the *following* week -- "
            "everything the rule sees, plus the answer.\n\n"
            "1. **Sort by the pros' positioning.** Split weeks into low / middle / high exposure and "
            "compare next week's return. Contrarian = low beats high.\n"
            "2. **Trade it honestly.** Build a rule that sits out the all-in weeks and stays long "
            "otherwise, charge it realistic costs, and race it against buy-and-hold.\n"
            "3. **What would make us say \"mirage\"?** If the gap isn't statistically reliable, or "
            "the rule loses to buy-and-hold once it pays for the upside it skips."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown -- let's actually look\n\n"
            "**First: which way does it lean?** Next-week SPY return, split by how exposed the pros "
            "were the week before."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rs = st.regime_summary(PANEL)\n"
            "    yrs = [rs.loc[k,'mean_ann']*100 for k in ('low','mid','high')]\n"
            "else:\n"
            "    yrs = [R['low_yr'], R['mid_yr'], R['high_yr']]\n"
            "labels = ['LOW\\n(pros in cash)', 'MID', 'HIGH\\n(pros all-in)']\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.4))\n"
            "ax.bar(labels, yrs, color=[GREEN, GREY, RED], width=.6)\n"
            "ax.axhline(R['uncond_yr'], ls='--', c='k', label=f\"unconditional ({R['uncond_yr']:.0f}%/yr)\")\n"
            "for i,v in enumerate(yrs): ax.annotate(f'{v:+.0f}%',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('next-week SPY return (annualised)'); ax.set_ylim(0, max(yrs)*1.25)\n"
            "ax.set_title('Direction is contrarian: cash-regime beats all-in -- but all-in is still positive')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('low / mid / high  %/yr:', [f'{v:+.1f}' for v in yrs])"
        ),
        md(
            f"The lean is the contrarian one: after the pros bail to cash, the next week runs "
            f"**+{R['low_yr']:.0f}%/yr**; after they're all-in, **+{R['high_yr']:.0f}%/yr** -- a "
            f"**{R['gap_yr']:.0f} pp/yr** gap. **But look at the all-in bar: it's still firmly "
            "positive.** \"The pros are maxed out\" has never meant \"the market falls\" -- it's "
            "meant \"the market rises a bit less.\" That's already fatal to a *sell* signal."
        ),
        md(
            "**Now the catch: can you trade it?** Build the contrarian rule -- stay long, but step "
            "aside whenever the pros are all-in -- charge it costs, and race it against simply "
            "buying and holding SPY."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ov = st.timing_overlay(PANEL, one_way_bps=5.0)\n"
            "    eq_ov = (1+ov['net']).cumprod(); eq_bh = (1+ov['bh']).cumprod()\n"
            "    ov_yr = st.summarize(ov['net'])['mean']*5200; bh_yr = st.summarize(ov['bh'])['mean']*5200\n"
            "    fig, ax = plt.subplots(figsize=(9.2, 4.6))\n"
            "    ax.plot(eq_bh.index, eq_bh.values, c=GREEN, lw=2, label=f'buy & hold ({bh_yr:+.1f}%/yr)')\n"
            "    ax.plot(eq_ov.index, eq_ov.values, c=RED, lw=2, label=f'fade-the-pros overlay ({ov_yr:+.1f}%/yr)')\n"
            "    ax.set_yscale('log'); ax.set_ylabel('$1 grows to (log)')\n"
            "else:\n"
            "    fig, ax = plt.subplots(figsize=(9.2, 4.6)); ov_yr, bh_yr = R['ov_yr'], R['bh_yr']\n"
            "    ax.bar(['fade-the-pros\\noverlay','buy & hold'],[ov_yr,bh_yr],color=[RED,GREEN],width=.5)\n"
            "    for i,v in enumerate([ov_yr,bh_yr]): ax.annotate(f'{v:+.1f}%/yr',(i,v),ha='center',va='bottom')\n"
            "    ax.set_ylabel('annualised return')\n"
            "ax.set_title('Fading the pros LOSES to just staying invested'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'overlay {ov_yr:+.1f}%/yr  vs  buy-and-hold {bh_yr:+.1f}%/yr')"
        ),
        md(
            f"There it is. The contrarian overlay nets **+{R['ov_yr']:.1f}%/yr**; buy-and-hold makes "
            f"**+{R['bh_yr']:.1f}%/yr**. Stepping aside when the pros are all-in means stepping aside "
            "during the strongest stretches of bull markets -- exactly the weeks the pros were right "
            "to ride. **The direction was real; the trade still loses.**"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal -- Weak.** The contrarian direction is real (cash +{R['low_yr']:.0f}%/yr vs "
            f"all-in +{R['high_yr']:.0f}%/yr) and shows up in every era, but it's not statistically "
            "reliable and explains almost none of next-week's moves.\n"
            f"- **Tradability -- Mirage.** The overlay nets **+{R['ov_yr']:.1f}%/yr** vs "
            f"**+{R['bh_yr']:.1f}%/yr** for buy-and-hold. Fading the pros skips the rallies.\n"
            "- **Pros smarter than retail? -- Not supported.** Same weak, untradeable tilt as the "
            "individual-investor survey. Knowing what the pros *do* is no more bankable than knowing "
            "what retail *thinks*."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? -- why the pros go all-in\n\n"
            "Here's the trap, in one picture. Plot the managers' exposure on top of the market. "
            "Notice **when** they're all-in: not at random tops, but right in the middle of the "
            "uptrends -- because they're (sensibly) trend-following. Fade that, and you're shorting "
            "the trend."
        ),
        code(
            "if HAVE_REAL:\n"
            "    p = PANEL\n"
            "    spy = data.fetch_spy()\n"
            "    lvl = spy.reindex(p.index, method='ffill')\n"
            "    fig, ax1 = plt.subplots(figsize=(9.6, 4.6))\n"
            "    ax1.plot(lvl.index, lvl.values, c=GREEN, lw=1.4, label='SPY (total return)')\n"
            "    ax1.set_ylabel('SPY', color=GREEN); ax1.set_yscale('log')\n"
            "    ax2 = ax1.twinx()\n"
            "    ax2.plot(p.index, p['naaim'].values, c=GREY, lw=.8, alpha=.7)\n"
            "    ax2.axhline(p['naaim'].quantile(2/3), ls='--', c=RED, lw=1, label='all-in threshold')\n"
            "    ax2.set_ylabel('NAAIM exposure %', color=GREY); ax2.set_ylim(-20, 140)\n"
            "    ax1.set_title('The pros go all-in DURING uptrends -- fading them shorts the trend')\n"
            "    ax1.legend(loc='upper left'); ax2.legend(loc='lower right')\n"
            "else:\n"
            "    fig, ax = plt.subplots(figsize=(9.6,4.6))\n"
            "    ax.text(.5,.5,'(real tape cache absent -- see docs/results.md)',ha='center')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('NAAIM range on the tape:', f\"{R['naaim_lo']:.0f}..{R['naaim_hi']:.0f}\", '(mean', f\"{R['naaim_mean']:.0f})\")"
        ),
        md(
            "> The pros' exposure rises *with* the market and falls *with* it. \"Be greedy when the "
            "pros are fearful\" sounds bold, but mechanically it's \"buy after a crash, sell after a "
            "rally\" -- a weak, pro-cyclical bet that occasionally nails a bottom (2009, 2020) and "
            "otherwise just costs you the upside."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The retail twin.** [Study 257 -- AAII](../../257-aaii-sentiment/): the same "
            "contrarian template on the *individual-investor* survey -- same WEAK / MIRAGE verdict. "
            "The crowd-vs-pros head-to-head is the punchline: neither is bankable.\n"
            "- **The options crowd & leverage.** [Study 261 -- put/call](../../261-put-call-ratio/) "
            "and [Study 260 -- margin-debt](../../260-margin-debt/) -- two more sentiment gauges, "
            "same fate.\n"
            "- **Try harder yourself.** Z-score the exposure, demand a multi-week extreme, or only "
            "fade *both* an all-in print *and* an overbought tape. The package is offline and "
            "deterministic -- fork [`naaim_exposure/`](../naaim_exposure/) and see if any variant "
            "clears the bar (we couldn't).\n\n"
            "*Think a cleaner NAAIM rule beats buy-and-hold after costs? Build it on this exact tape "
            "and show the *t*-stat -- then check it isn't just riding a couple of ex-post-obvious "
            "rebounds.*"
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
            "# NAAIM-Exposure as a contrarian timing signal -- a quantitative teardown 🔬\n"
            "### Regime sort + HAC *t* · predictive regression (slope, *t*, R²) · "
            "cost-charged overlay vs buy-and-hold · a deterministic synthetic control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The "
            "NAAIM Exposure Index is **active managers' reported equity exposure** (0-200%), weekly "
            "since 2006. We test the contrarian claim -- high exposure predicts weak forward returns, "
            "low exposure strong ones -- on the real SPY total-return tape with autocorrelation-robust "
            "inference, then ask whether any of it survives costs against buy-and-hold.\n\n"
            "> ⚠️ **Not investment advice.** Real data: NAAIM weekly Number (free since-inception "
            "spreadsheet, naaim.org) joined to SPY total return (yfinance, `auto_adjust=True`), "
            f"{R['weeks']:,} weeks ({R['window']}); one-week execution lag. Offline core + synthetic "
            "control are deterministic. Methods in [`docs/references.md`](../docs/references.md), "
            "numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | Contrarian *direction* holds (low-exposure +{R['low_yr']:.1f}%/yr "
            f"vs high +{R['high_yr']:.1f}%/yr, gap +{R['gap_yr']:.1f} pp/yr) but the long-short HAC "
            f"**t = +{R['ls_t']:.2f}** and regression slope **t = {R['beta_t']:.2f}** (R² ≈ "
            f"{R['r2']*100:.2f}%) are far below 2. |\n"
            f"| **Tradability** | `MIRAGE` | Long/flat contrarian overlay nets **+{R['ov_yr']:.2f}%/yr** "
            f"(SR {R['ov_sr']:.2f}) vs buy-and-hold **+{R['bh_yr']:.2f}%/yr** (SR {R['bh_sr']:.2f}); "
            "sitting out the all-in regime skips the bull-market core. |\n"
            "| **Pros smarter than retail?** | `NOT SUPPORTED` | Same weak/untradeable tilt as the "
            "AAII retail survey ([257](../../257-aaii-sentiment/)). |\n\n"
            "> 💡 In plain words: the pros' positioning leans the contrarian way, but it's a weak, "
            "pro-cyclical lean -- statistically it's a coin-flip's distance from noise, and as a "
            "trade it loses to staying invested because managers go all-in *during* the rallies."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $e_t$ be the NAAIM exposure observed at weekly survey $t$ and $r_{t+1}$ the SPY "
            "total return over the following week. The contrarian hypotheses:\n\n"
            "- **H₁ (signal).** $\\mathbb{E}[r_{t+1}\\mid e_t\\text{ high}] < \\mathbb{E}[r_{t+1}\\mid "
            "e_t\\text{ low}]$ -- a *negative* predictive slope, $\\beta < 0$ with HAC $|t|\\ge 2$.\n"
            "- **H₂ (tradable).** A long/flat (or long/short) rule keyed on $e_t$ beats buy-and-hold "
            "net of one-way costs.\n"
            "- **H₃ (smart money).** Because $e_t$ is *professional* positioning, the contrarian edge "
            "exceeds the equivalent retail-sentiment gauge.\n\n"
            "We find **H₁ weakly accepted in sign only** (slope negative but $|t|<1$, R² ≈ 0.03%), "
            "**H₂ rejected** (the overlay loses to buy-and-hold), and **H₃ rejected** (no better than "
            "AAII). The folklore is directionally true and practically empty."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? -- what rides on each answer\n\n"
            "A free, public, weekly timing signal from the people closest to the market would be a "
            "genuine crack in efficiency -- and a deep statement that proximity breeds error at the "
            "extremes. But three structural priors push hard the other way: (i) **prediction "
            "efficiency** -- a gauge thousands watch shouldn't hide a stable edge; (ii) the "
            "**favourite-of-trend** problem -- manager exposure is largely trend-following (Frazzini-"
            "Lamont *Dumb Money*), so \"fade the all-in pros\" ≈ \"fade recent strength,\" a "
            "pro-cyclical bet; (iii) the **dividend tax of timing** -- any rule that sits in cash "
            "forgoes total return, so the honest benchmark is dividend-reinvested buy-and-hold, which "
            "is a high bar. The test must charge all three."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know -- the protocol\n\n"
            f"- **Real tape.** NAAIM weekly Number ({R['window']}, {R['weeks']:,} weeks; fingerprint "
            f"`{R['fp']}`) joined to SPY total-return closes on each survey date; $r_{{t+1}}$ is the "
            "SPY return to the next survey date (one-week lag, applied once).\n"
            "- **Regime sort.** Next-week return by *prior*-exposure tercile; HAC *t* per regime and "
            "of the low−high long-short.\n"
            "- **Predictive regression.** $r_{t+1}$ on the standardised prior exposure; Newey-West "
            "HAC *t* on the slope and the R². Contrarian ⇒ $\\beta<0$, $|t|\\ge2$.\n"
            "- **Overlay.** Long when exposure is in the bottom tercile, flat (sit out) in the top, "
            "long otherwise; 5 bps one-way × NAV per state change; raced against buy-and-hold on the "
            "same total-return index.\n"
            "- **Positive control.** A deterministic AR(1) exposure tape with a *planted* contrarian "
            "loading (`data.synthetic_weekly`): the harness must recover $\\beta\\ll0$ at "
            f"edge={R['syn_edge']} and read ~0 on the null. (Machinery proof, not market evidence.)\n"
            "- **What says \"mirage.\"** A sub-2 HAC *t* on the real tape, or an overlay that loses "
            "to buy-and-hold."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Positive control -- the engine finds a contrarian edge when one is planted\n\n"
            "Before trusting a null result, prove the harness *can* detect a contrarian signal. On a "
            "deterministic synthetic tape we recover the planted loading and read ~zero on the null:"
        ),
        code(
            "de,_ = data.synthetic_weekly(edge=R['syn_edge'], seed=360)\n"
            "de = de.assign(ret=de['ret'].shift(-1)).dropna()\n"
            "dn,_ = data.synthetic_weekly(edge=0.0, seed=360)\n"
            "dn = dn.assign(ret=dn['ret'].shift(-1)).dropna()\n"
            "te = st.predictive_regression(de)['tstat']; tn = st.predictive_regression(dn)['tstat']\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.2))\n"
            "ax.bar(['null\\n(edge=0)', f'planted\\n(edge={R[\"syn_edge\"]})'], [tn, te], color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(-2, ls='--', c=RED, label='significance bar (t=-2)')\n"
            "for i,v in enumerate([tn,te]): ax.annotate(f'{v:+.2f}',(i,v),ha='center',va='top' if v<0 else 'bottom')\n"
            "ax.set_ylabel('regression slope HAC t'); ax.set_title('Machinery proof: detects a planted contrarian edge, ~0 on the null')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null t={tn:+.2f} (~0)   planted t={te:+.2f} (<< -2)')"
        ),
        md(
            f"> 💡 In plain words: with a real contrarian edge in the data the slope *t* is "
            f"**{R['syn_edge_t']:.1f}**; with none it's **{R['syn_null_t']:.2f}** (≈0). So when the "
            "real tape comes back flat, that's a property of the *market*, not a broken test."
        ),
        md(
            "### 4b · The signal on the real tape -- right direction, no significance\n\n"
            "Next-week SPY return by prior-exposure regime, with HAC *t*; then the predictive "
            "regression. The contrarian prediction is low > high and a negative slope."
        ),
        code(
            "rs = st.regime_summary(PANEL)\n"
            "reg = st.predictive_regression(PANEL)\n"
            "yrs = [rs.loc[k,'mean_ann']*100 for k in ('low','mid','high')]\n"
            "ts  = [rs.loc[k,'tstat'] for k in ('low','mid','high')]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "bars = ax.bar(['low','mid','high'], yrs, color=[GREEN, GREY, RED], width=.6)\n"
            "ax.axhline(R['uncond_yr'], ls='--', c='k', label=f\"unconditional {R['uncond_yr']:.0f}%/yr\")\n"
            "for b,v,t in zip(bars,yrs,ts): ax.annotate(f'{v:+.1f}%/yr\\nt={t:+.2f}',(b.get_x()+b.get_width()/2,v),ha='center',va='bottom',fontsize=9)\n"
            "ax.set_ylabel('next-week SPY return (annualised)'); ax.set_ylim(0, max(yrs)*1.3)\n"
            "ax.set_title('Contrarian in direction; every regime still positive'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"low={yrs[0]:+.1f}%/yr  high={yrs[2]:+.1f}%/yr  gap={yrs[0]-yrs[2]:+.1f} pp/yr\")\n"
            "print(f\"predictive slope: {reg['beta']*100:+.3f}%/sd  HAC t={reg['tstat']:+.2f}  R2={reg['r2']*100:.2f}%  n={reg['n']}\")\n"
            "ls = st.summarize(st.regime_spread(PANEL)); print(f\"long-short: {ls['mean']*5200:+.2f}%/yr  HAC t={ls['tstat']:+.2f}\")"
        ),
        md(
            f"> 💡 In plain words: the gap is the contrarian one (+{R['gap_yr']:.1f} pp/yr), but the "
            f"long-short HAC **t = +{R['ls_t']:.2f}** and the regression slope **t = {R['beta_t']:.2f}** "
            f"(R² ≈ {R['r2']*100:.2f}%) are nowhere near the *t* ≥ 2 bar. Exposure explains "
            "essentially **none** of next-week variance. H₁ holds in sign, fails in strength → "
            "**Signal WEAK**, exactly as the literature (Fisher-Statman; Brown-Cliff) would predict."
        ),
        md(
            "### 4c · Tradability -- the overlay loses to buy-and-hold\n\n"
            "Net-of-cost equity curve of the contrarian overlay (sit out the all-in regime) against "
            "dividend-reinvested buy-and-hold on the same SPY total-return index."
        ),
        code(
            "ov = st.timing_overlay(PANEL, one_way_bps=5.0)\n"
            "s_net = st.summarize(ov['net']); s_bh = st.summarize(ov['bh'])\n"
            "eq_ov = (1+ov['net']).cumprod(); eq_bh = (1+ov['bh']).cumprod()\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.6))\n"
            "ax.plot(eq_bh.index, eq_bh.values, c=GREEN, lw=2,\n"
            "        label=f\"buy & hold {s_bh['mean']*5200:+.1f}%/yr (SR {s_bh['sharpe']*np.sqrt(52):.2f})\")\n"
            "ax.plot(eq_ov.index, eq_ov.values, c=RED, lw=2,\n"
            "        label=f\"contrarian overlay {s_net['mean']*5200:+.1f}%/yr (SR {s_net['sharpe']*np.sqrt(52):.2f})\")\n"
            "ax.set_yscale('log'); ax.set_ylabel('$1 grows to (log)')\n"
            "ax.set_title('Net of costs, fading the pros LOSES to staying invested'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"overlay NET {s_net['mean']*5200:+.2f}%/yr  SR={s_net['sharpe']*np.sqrt(52):+.2f}  in-market {(ov['pos']>0).mean()*100:.0f}% of weeks\")\n"
            "print(f\"buy & hold  {s_bh['mean']*5200:+.2f}%/yr  SR={s_bh['sharpe']*np.sqrt(52):+.2f}\")"
        ),
        md(
            f"> 💡 In plain words: the overlay nets **+{R['ov_yr']:.1f}%/yr** (SR {R['ov_sr']:.2f}) vs "
            f"buy-and-hold **+{R['bh_yr']:.1f}%/yr** (SR {R['bh_sr']:.2f}). It is in-market "
            f"{R['ov_inmkt']}% of weeks and still loses, because the weeks it sits out -- the all-in "
            "regime -- are the bull-market core. Even the *high-sentiment* weeks earn a positive "
            "return; there is nothing to harvest by avoiding them. **H₂ rejected → Tradability MIRAGE.**"
        ),
        md(
            "### 4d · Stability -- direction everywhere, significance nowhere robust\n\n"
            "Split into three eras: the low > high ordering should persist (a real direction) even as "
            "the *t*-stats wander (a weak magnitude)."
        ),
        code(
            "rows = []\n"
            "for lab,a,b in [('2006-2012','2006','2012'),('2013-2019','2013','2019'),('2020-2026','2020','2026')]:\n"
            "    sub = PANEL[a:b]; r = st.regime_summary(sub)\n"
            "    rows.append((lab, r.loc['low','mean_ann']*100, r.loc['low','tstat'],\n"
            "                 r.loc['high','mean_ann']*100, r.loc['high','tstat']))\n"
            "labs = [x[0] for x in rows]; x = np.arange(len(labs))\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.bar(x-.2, [x[1] for x in rows], .4, color=GREEN, label='low (cash) regime')\n"
            "ax.bar(x+.2, [x[3] for x in rows], .4, color=RED, label='high (all-in) regime')\n"
            "ax.set_xticks(x); ax.set_xticklabels(labs); ax.set_ylabel('next-week return (%/yr)')\n"
            "ax.set_title('Low > High in every era (real direction) -- but t-stats wander'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for lab,ly,lt,hy,ht in rows: print(f'{lab}: low={ly:+6.1f}%/yr (t={lt:+.2f})  high={hy:+6.1f}%/yr (t={ht:+.2f})')"
        ),
        md(
            "> 💡 In plain words: low beats high in **all three** eras -- the contrarian *direction* "
            "is robust. But the low-regime *t* clears 2 only in 2013-2019; the rest of the time it's "
            "indistinguishable from the unconditional drift. A stable sign with an unstable, sub-2 "
            "magnitude is the textbook profile of a **WEAK** signal, not a real one."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** -- contrarian direction is real and stable (low +{R['low_yr']:.1f}%/yr "
            f"vs high +{R['high_yr']:.1f}%/yr in every era) but long-short HAC **t = +{R['ls_t']:.2f}**, "
            f"slope **t = {R['beta_t']:.2f}**, R² ≈ {R['r2']*100:.2f}%. Literature supports the sign, "
            "this tape can't certify the magnitude; current-vintage series ⇒ upper bound.\n"
            f"- **Tradability `MIRAGE`** -- overlay **+{R['ov_yr']:.1f}%/yr** vs buy-and-hold "
            f"**+{R['bh_yr']:.1f}%/yr**, net of 5 bps. Sitting out the all-in regime skips the "
            "bull-market core; the edge never pays for the missed upside.\n"
            "- **Pros smarter than retail? `NOT SUPPORTED`** -- identical weak/untradeable profile to "
            "the AAII individual-investor survey. Professional *positioning* is no more bankable than "
            "retail *opinion*."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? -- why it fails, structurally\n\n"
            "The failure isn't bad luck; it's mechanical. Manager exposure is **trend-following**, so "
            "the all-in regime *coincides* with strong markets. Bucket forward returns finely by "
            "exposure and the relationship is flat-to-mildly-negative with no exploitable cliff:"
        ),
        code(
            "p = PANEL.copy()\n"
            "qs = np.quantile(p['naaim'], np.linspace(0,1,6))\n"
            "p['bucket'] = np.clip(np.digitize(p['naaim'], qs[1:-1]), 0, 4)\n"
            "centers = [p.loc[p['bucket']==k,'naaim'].mean() for k in range(5)]\n"
            "fwd = [p.loc[p['bucket']==k,'ret'].mean()*5200 for k in range(5)]\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.plot(centers, fwd, 'o-', c=GREY, lw=2)\n"
            "ax.axhline(R['uncond_yr'], ls='--', c='k', label=f\"unconditional {R['uncond_yr']:.0f}%/yr\")\n"
            "ax.set_xlabel('NAAIM exposure quintile (avg %)'); ax.set_ylabel('next-week return (%/yr)')\n"
            "ax.set_title('Forward return slides down only gently with exposure -- no tradable cliff')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('exposure quintile centers:', [f'{c:.0f}' for c in centers])\n"
            "print('next-week %/yr        :', [f'{v:+.0f}' for v in fwd])"
        ),
        md(
            "> 💡 In plain words: there's no exposure level above which returns turn *negative* -- the "
            "slope is gentle and never crosses zero. A contrarian needs a regime where fading pays "
            "*more than it costs in forgone upside*, and on this tape that regime doesn't exist. The "
            "only sizing that would have 'worked' (heavy in 2009/2020 cash extremes) is visible only "
            "with hindsight."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The crowd-vs-pros bench.** Line this up against [257 -- AAII](../../257-aaii-sentiment/) "
            "(retail), [261 -- put/call](../../261-put-call-ratio/) (options) and "
            "[260 -- margin-debt](../../260-margin-debt/) (leverage): four sentiment gauges, four WEAK "
            "/ MIRAGE verdicts. The interesting result is the *non-difference* -- pros don't beat "
            "retail as a contrarian tell.\n"
            "- **Combine, don't isolate.** Test whether NAAIM *adds* to a momentum/trend baseline "
            "(orthogonalise exposure against trailing returns first), rather than as a standalone "
            "timer -- the trend-following confound is the prime suspect.\n"
            "- **Vintage.** Rebuild on a point-in-time archive of the weekly print (if obtainable) to "
            "see how much of even this weak edge is revision-driven.\n\n"
            "*The reproducible core is offline and deterministic; the real tape is the free, public "
            "NAAIM weekly spreadsheet joined to SPY total return. Methods and sources: "
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
