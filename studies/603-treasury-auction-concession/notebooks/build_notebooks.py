"""Generate the two narrative notebooks for Study 603 (Treasury Auction Concession).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached auction
records + yield tape + TLT under ../_cache/ and otherwise quote the frozen headline numbers
in ``R`` (mirroring docs/results.md). The synthetic positive control runs anywhere with no
network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (FiscalData auctions 1979->
# 2026 + yfinance ^TNX/^TYX/TLT/^IRX, as-of 2026-06-30).
R = dict(
    as_of="2026-06-30", fp_yields="388e6555b83b", fp_tlt="7d6a23aca7db",
    n_auctions=825, n_10y=496, n_30y=329, first="1979-10-31", last="2026-06-11",
    # per tenor: (n_on_tape, pre_cum_bps, post_cum_bps, base_bps, welch_pre,
    #             pre_bps_day, t_pre, post_bps_day, t_post)
    ten={"10Y": (492, 1.55, -1.20, 0.00, 2.19, 0.324, 2.15, -0.283, -1.99),
         "30Y": (329, 2.18, -0.62, -0.49, 3.36, 0.548, 3.27, -0.080, -0.52)},
    # era rows: (tenor, era, pre_bps_day, t_pre, post_bps_day, t_post, n)
    era=[("10Y", "1979-2007", 0.171, 0.53, -0.483, -1.77, 165),
         ("10Y", "2008-2019", 0.207, 0.97, -0.233, -1.06, 210),
         ("10Y", "2020-2026", 0.949, 3.39, 0.137, 0.44, 116),
         ("30Y", "1979-2007", 0.385, 0.80, -0.241, -0.63, 78),
         ("30Y", "2008-2019", 0.804, 3.86, -0.203, -0.94, 160),
         ("30Y", "2020-2026", 0.432, 1.41, 0.453, 1.52, 91)],
    # size split: (big_pre, n_big, small_pre, n_small, welch_pre, welch_post)
    size={"10Y": (1.55, 308, 1.88, 180, -0.28, -1.20),
          "30Y": (2.25, 244, 0.90, 77, 0.92, -1.47)},
    # TLT tradability
    tlt=dict(bh_excess=3.01, alpha_base=-0.80, alpha_extra=5.19, t_alpha=2.26,
             trips=338, held=38.5, per_trip=30.0,
             cost=[(2.0, 4.25, 3.69, 2.11), (5.0, 4.25, 2.84, 1.63)],
             sub=[("2002-2019", 6.09, 5.54, 2.96, 236),
                  ("2020-2026", -0.68, -1.32, -0.33, 102)]),
    # synthetic: (planted bps, pre/day, t_pre, post/day, t_post, net_ann, t_net)
    syn=[(0.0, -0.051, -0.29, -0.173, -1.04, 0.21, 0.14),
         (4.0, 0.749, 4.25, -0.973, -5.82, 8.35, 5.67)],
)

BADGES = (
    "![Signal: Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square)\n"
    "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
    "![Bigger auction bigger concession?: Busted](https://img.shields.io/badge/Bigger_auction_bigger_concession%3F-Busted-8b949e?style=flat-square)\n\n"
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

from treasury_auction_concession import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    AU, YLDS, TLTF = data.load_real()
else:
    AU = YLDS = TLTF = None
print("real cache present:", HAVE_REAL,
      "| auctions:", (0 if AU is None else len(AU)),
      "| yield tape days:", (0 if YLDS is None else len(YLDS)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    t10, t30 = R["ten"]["10Y"], R["ten"]["30Y"]
    cells = [
        md(
            "# Does the bond market really \"make room\" before big Treasury auctions? 🏛️\n"
            "### The auction concession — a rates-desk legend put on 47 years of tape, in plain English\n\n"
            + BADGES +
            "Every month the U.S. government sells tens of billions of dollars of 10-year notes and "
            "30-year bonds at **auction**, on a calendar published weeks in advance. The oldest story on "
            "any rates desk says the market **cheapens into** the auction — yields drift *up* for a few "
            "days because the dealers who must absorb all that paper demand a discount (a "
            "**concession**) — and then **richens after**, as the supply is digested and prices bounce "
            "back. A little V drawn on a thousand whiteboards.\n\n"
            "We put the whole legend on tape: **every single 10Y and 30Y auction since 1979** — 825 of "
            "them, straight from the Treasury's official records — against daily yields.\n\n"
            "> 📓 **Plain-language layer.** Want the HAC regressions, the size split and the cost math? "
            "See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Do yields really drift up **into** big auctions? | **Yes — really.** About "
            f"**+{t10[1]:.1f} bps** into a 10Y auction and **+{t30[1]:.1f} bps** into a 30Y, built over "
            "the auction week. Statistically solid on both, and *biggest* exactly when the government "
            "was issuing the most (after 2020). |\n"
            "| Do they snap back **after**, like the legend says? | **Not reliably.** The bounce-back is "
            "small, misses the significance bar on both tenors, and has actually **flipped sign since "
            "2020** — lately yields keep drifting *up* after auctions too. Half the legend fails. |\n"
            "| Could you have traded the bounce? | **Once.** Buying the long-bond ETF (TLT) after each "
            "auction beat the couch-potato version handily from 2002–2019 — then the trade **died "
            "around 2020**. |\n"
            "| Do **bigger** auctions get **bigger** concessions? | **No.** That's the story's own "
            "logic — more supply, more discount — and the tape says it just isn't there. |\n\n"
            "> The concession is **real**; the *snap-back* and the *size logic* are folklore."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Treasury supply is perfectly predictable — the auction calendar is public. Dealers "
            "still need to be paid to warehouse it. So the market cheapens into every big 10Y/30Y "
            "auction and richens once the paper is placed.\"*\n\n"
            "This isn't just trader talk — academics formalised it. Lou, Yan & Zhang (2013, *Review of "
            "Financial Studies*) documented a **V-shaped price pattern** around Treasury auctions: "
            "yields rise into the auction and fall back after, even though the event is anticipated to "
            "the day. The puzzle: in a perfectly efficient market, *anticipated* supply shouldn't move "
            "prices at all."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the concession is real, it means even the deepest market on Earth — U.S. Treasuries — "
            "pays a **storage fee** to intermediaries around predictable events: capital doesn't move "
            "fast enough to absorb $40bn of duration without a price cut. That matters for anyone "
            "timing bond purchases (buy on auction day, not the Friday before), for the Treasury's own "
            "issuance costs, and as a clean lab test of the *slow-moving capital* idea.\n\n"
            "And if the V's second leg is real, there's a **tradable bounce**: buy the long bond at the "
            "auction close, sell a week later. We test both legs separately — legends usually hide one "
            "weak leg behind one strong one."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **Every auction, officially.** All **{R['n_auctions']}** 10Y-note and 30Y-bond auctions "
            f"{R['first']} → {R['last']} from the Treasury's own FiscalData/TreasuryDirect records — "
            "including reopenings (a reopened 10Y is the same supply event). No survivorship possible: "
            "it's the complete government record.\n"
            "- **Daily yields.** The 10Y (^TNX) and 30Y (^TYX) constant-maturity yield indices.\n"
            "- **The windows.** *Pre* = the 5 trading days ending at the auction-day close (results hit "
            "at 1 pm, so that close knows the outcome). *Post* = the 5 days after.\n"
            "- **The test.** Are pre-auction days different from ordinary days, once you account for the "
            "fact that market moves cluster? (That's the HAC regression in the quants notebook.)"
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**The V itself.** Average path of the 10Y and 30Y yield around their own auctions — day 0 "
            "is the auction-day close."
        ),
        code(
            "k = 7\n"
            "if HAVE_REAL:\n"
            "    p10 = st.car_profile(YLDS['y10'], AU.loc[AU['tenor']=='10Y','auction_date'], k=k)\n"
            "    p30 = st.car_profile(YLDS['y30'], AU.loc[AU['tenor']=='30Y','auction_date'], k=k)\n"
            "else:  # sketch from the frozen numbers\n"
            "    ramp = np.r_[np.linspace(0, 1, k+1), np.linspace(1, 0, k)[1:]]\n"
            "    p10 = pd.Series(ramp*R['ten']['10Y'][1], index=range(-k, k+1))\n"
            "    p30 = pd.Series(ramp*R['ten']['30Y'][1], index=range(-k, k+1))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.6))\n"
            "ax.plot(p10.index, p10.values, marker='o', color=GREEN, label='10Y around its auctions')\n"
            "ax.plot(p30.index, p30.values, marker='s', color=AMBER, label='30Y around its auctions')\n"
            "ax.axvline(0, color=RED, ls='--', lw=1.2, label='auction day (1pm results)')\n"
            "ax.axhline(0, color=GREY, lw=.8)\n"
            "ax.set_xlabel('trading days around the auction'); ax.set_ylabel('avg cumulative yield change (bps)')\n"
            "ax.set_title('Yields drift UP into the auction... and barely come back')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'10Y: pre {p10[0]-p10[-5]:+.2f} bps  post {p10[5]-p10[0]:+.2f} bps   '\n"
            "      f'30Y: pre {p30[0]-p30[-5]:+.2f} bps  post {p30[5]-p30[0]:+.2f} bps')"
        ),
        md(
            f"The **left half of the V is there**: on average the 10Y cheapens **+{t10[1]:.1f} bps** and "
            f"the 30Y **+{t30[1]:.1f} bps** over the week into the auction (vs roughly flat ordinary "
            f"weeks). The **right half is anaemic**: the bounce-back is **{t10[2]:.1f}** / "
            f"**{t30[2]:.1f} bps** — smaller than the build-up, and (the quants notebook shows) not "
            "statistically trustworthy. A tilted checkmark, not a V.\n\n"
            "> 🔬 *For the quants:* the pre-leg clears the bar with HAC *t* = "
            f"**{t10[6]:.2f}** (10Y) and **{t30[6]:.2f}** (30Y); the post-leg misses at "
            f"*t* = {t10[8]:.2f} and {t30[8]:.2f}."
        ),
        md(
            "**When did the concession live?** Split the 47 years into three regimes: before the GFC, "
            "the QE decade, and the post-2020 deficit flood."
        ),
        code(
            "eras = [r for r in R['era']]\n"
            "if HAVE_REAL:\n"
            "    eras = []\n"
            "    for tenor, col in [('10Y','y10'),('30Y','y30')]:\n"
            "        for e in st.era_split(YLDS[col], AU.loc[AU['tenor']==tenor,'auction_date']):\n"
            "            eras.append((tenor, e['era'], e['pre_bps_day'], e['t_pre'], e['post_bps_day'], e['t_post'], e['n_auctions']))\n"
            "labels = [f\"{t}\\n{e}\" for t,e,*_ in eras]\n"
            "tvals = [r[3] for r in eras]\n"
            "colors = [GREEN if v >= 2 else GREY for v in tvals]\n"
            "fig, ax = plt.subplots(figsize=(9.6, 4.6))\n"
            "ax.bar(labels, tvals, color=colors, width=.6)\n"
            "ax.axhline(2, ls='--', c=RED, label='significance bar (t=2)')\n"
            "for i,v in enumerate(tvals): ax.annotate(f'{v:.1f}',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('strength of the pre-auction cheapening (HAC t)')\n"
            "ax.set_title('The concession is a modern phenomenon - biggest when issuance flooded')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('pre-leg HAC t by era:', [(l.replace(chr(10),' '), round(v,2)) for l,v in zip(labels, tvals)])"
        ),
        md(
            "Before 2008 the concession barely exists (*t* ≈ 0.5–0.8). It turns decisive in the **QE "
            "decade for the 30Y** (*t* = 3.9) and in the **post-2020 deficit flood for the 10Y** "
            "(*t* = 3.4 — nearly **5 bps** of cheapening per auction week). More supply pressure over "
            "time, more concession — the story's *era* logic holds.\n\n"
            "But here's the twist: its *size* logic doesn't. Within any era, **bigger-than-usual "
            "auctions get no bigger concession than smaller ones** (10Y even points the wrong way). "
            "The market clears its throat before *every* auction, big or small — a calendar habit, "
            "not a dosage response."
        ),
        md(
            "**Could you trade the bounce?** Buy TLT (the 20+ year Treasury ETF) at each auction close, "
            "sell 5 days later, pay costs, park in T-bills otherwise."
        ),
        code(
            "if HAVE_REAL:\n"
            "    full = st.tlt_post_auction(TLTF, AU['auction_date'], cost_bps=2.0)\n"
            "    t1 = TLTF[TLTF.index <= '2019-12-31']; d1 = AU.loc[AU['auction_date'] <= '2019-12-31','auction_date']\n"
            "    t2 = TLTF[TLTF.index >= '2020-01-01']; d2 = AU.loc[AU['auction_date'] >= '2020-01-01','auction_date']\n"
            "    sub1 = st.tlt_post_auction(t1, d1, cost_bps=2.0)\n"
            "    sub2 = st.tlt_post_auction(t2, d2, cost_bps=2.0)\n"
            "    vals = [full['net_ann_pct'], sub1['net_ann_pct'], sub2['net_ann_pct']]\n"
            "else:\n"
            "    vals = [R['tlt']['cost'][0][2], R['tlt']['sub'][0][2], R['tlt']['sub'][1][2]]\n"
            "labels = ['full sample\\n2002-2026', 'golden years\\n2002-2019', 'since the flood\\n2020-2026']\n"
            "colors = [AMBER, GREEN, RED]\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.5))\n"
            "ax.bar(labels, vals, color=colors, width=.55)\n"
            "ax.axhline(0, color=GREY, lw=.8)\n"
            "for i,v in enumerate(vals): ax.annotate(f'{v:+.1f}%/yr',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "ax.set_ylabel('net return over T-bills (%/yr, 2 bps/leg)')\n"
            "ax.set_title('The post-auction bounce trade: great for 17 years, then it died')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('net %/yr (2bps/leg):', [round(v,2) for v in vals])"
        ),
        md(
            f"From 2002–2019 the bounce trade netted **+{R['tlt']['sub'][0][2]:.1f}%/yr over bills** "
            "while being in the market only ~2 weeks a month — genuinely good. Since 2020 it nets "
            f"**{R['tlt']['sub'][1][2]:+.1f}%/yr**: the bounce is gone, exactly when the concession "
            "itself got biggest. Supply now cheapens the market into the auction *and it stays "
            "cheap* — nothing left to harvest on the other side."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Mixed.** The cheapen-into leg is **real**: +{t10[1]:.1f} bps (10Y) and "
            f"+{t30[1]:.1f} bps (30Y) per auction week, statistically solid, strongest when issuance "
            "flooded. The richen-after leg **misses the bar** and has flipped sign since 2020. Half a "
            "legend confirmed.\n"
            "- **Tradability — Fragile.** The post-auction TLT bounce beat bills convincingly for 17 "
            "years, then died in 2020. What remains (buy on auction day rather than the week before) "
            "is a *timing courtesy*, not a strategy.\n"
            "- **\"Bigger auction, bigger concession\"? — Busted.** No dose-response anywhere on the "
            "tape. The concession follows the calendar, not the size of the print."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **If you buy Treasuries anyway**, the practical takeaway survives: scheduled purchases "
            "land ~1.5–3 bps cheaper on 10Y/30Y **auction days** than in the days before. Free, tiny, "
            "real.\n"
            "- **Why did the bounce die?** Post-2020, supply is relentless and dealer balance sheets "
            "are capped — the concession stopped mean-reverting. Watch whether buybacks or dealer-"
            "regulation changes revive it.\n"
            "- **Different plumbing, different studies:** the cash-futures basis trade "
            "([382](../../382-treasury-basis-trade/README.md)), repo stress spikes "
            "([383](../../383-sofr-repo-stress/README.md)) and curve roll-down "
            "([380](../../380-curve-roll-down/README.md)) are separate Treasury-market legends on this "
            "bench.\n\n"
            "*Think the concession should scale with auction size and we measured it wrong? The size "
            "split is one function call — bring a detrending you like better and show a Welch t "
            "above 2.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "01_for_the_curious.ipynb")


# ===========================================================================
# 02 — FOR THE QUANTS
# ===========================================================================
def build_quants():
    t10, t30 = R["ten"]["10Y"], R["ten"]["30Y"]
    s10, s30 = R["size"]["10Y"], R["size"]["30Y"]
    T = R["tlt"]
    cells = [
        md(
            "# The Treasury Auction Concession — a quantitative teardown 🔬\n"
            "### 825 official auctions (1979→2026) · HAC event-dummy regression · era + size-detrended "
            "splits · a carry-clean TLT alpha test · costs × round trips · a planted-V synthetic "
            "control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The claim "
            "(Lou-Yan-Zhang 2013; every rates desk since forever): yields **cheapen into** 10Y/30Y "
            "auctions and **richen after** — dealers demand a concession to warehouse anticipated "
            "supply. We test the two legs separately, then the size dose-response, then whether the "
            "richening was ever tradable net of costs.\n\n"
            "> ⚠️ **Data note.** Auction records are the **complete official record** (FiscalData "
            "`auctions_query`, the TreasuryDirect feed; reopenings included) — no survivorship. Yields "
            "are constant-maturity indices (^TNX/^TYX); the tradable leg is TLT **total-return** with "
            "^IRX as cash. Offline core + synthetic control are deterministic. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (as-of " + R["as_of"] + ", fingerprints `"
            + R["fp_yields"] + "` / `" + R["fp_tlt"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),
        code(
            "if HAVE_REAL:\n"
            "    from quantlab import repro\n"
            "    print(repro.data_stamp('^TNX/^TYX daily yields (%)', YLDS, asof=data.AS_OF))\n"
            "    print(repro.data_stamp('TLT total-return close + ^IRX', TLTF, asof=data.AS_OF))\n"
            "    print('auctions:', len(AU), dict(AU['tenor'].value_counts()))\n"
            "else:\n"
            "    print('cache missing — quoting docs/results.md (fingerprints', R['fp_yields'], '/', R['fp_tlt'], ')')"
        ),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `MIXED` | *Cheapen-into leg REAL*: 10Y **+{t10[1]:.2f} bps**/window, HAC "
            f"**t = {t10[6]:+.2f}**; 30Y **+{t30[1]:.2f} bps**, HAC **t = {t30[6]:+.2f}**. *Richen-"
            f"after leg WEAK*: 10Y t = {t10[8]:+.2f}, 30Y t = {t30[8]:+.2f} — under the bar, and "
            "sign-flipped post-2020. |\n"
            f"| **Tradability** | `FRAGILE` | Post-auction TLT round trip: net **+{T['cost'][0][2]:.2f}%/yr** "
            f"at 2 bps/leg (HAC t = {T['cost'][0][3]:+.2f}), carry-clean alpha t = {T['t_alpha']:+.2f} — "
            f"but **+{T['sub'][0][2]:.2f}%/yr (t = {T['sub'][0][3]:.2f})** in 2002-2019 vs "
            f"**{T['sub'][1][2]:+.2f}%/yr** since 2020, and t = {T['cost'][1][3]:.2f} at 5 bps/leg. |\n"
            f"| **Size dose-response?** | `BUSTED` | Big vs small (detrended): Welch t = "
            f"**{s10[4]:+.2f}** (10Y, wrong sign) and **{s30[4]:+.2f}** (30Y, n.s.). |\n\n"
            "> 💡 In plain words: the market really does back up into auction week — but the fabled "
            "snap-back doesn't certify, the trade that harvested it died in 2020, and bigger prints "
            "don't get bigger discounts."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $y_t$ be the constant-maturity yield (bps) and $A$ an auction day of the same tenor. "
            "Define the windows\n\n"
            "$$\\Delta y^{pre}_A = y_A - y_{A-5},\\qquad \\Delta y^{post}_A = y_{A+5} - y_A .$$\n\n"
            "- **H₁ (concession).** $E[\\Delta y^{pre}] > 0$ — dealers demand a discount to warehouse "
            "anticipated supply (Lou-Yan-Zhang's V, left leg).\n"
            "- **H₂ (richening).** $E[\\Delta y^{post}] < 0$ — the discount decays once the paper is "
            "placed (right leg), and it is harvestable via a duration ETF net of costs.\n"
            "- **H₃ (dose-response).** Bigger auctions (relative to their own era) ⇒ bigger "
            "concessions — the mechanism's own scaling prediction.\n\n"
            "We find **H₁ supported** on both tenors (HAC t = 2.15 / 3.27), **H₂ not certified** in "
            "yield space (t = −1.99 / −0.52; the TLT return-space version cleared until 2020, then "
            "died), **H₃ rejected** (Welch t = −0.28 / +0.92)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — why the test design matters\n\n"
            "Three traps make naive versions of this study lie:\n\n"
            "1. **Overlapping windows.** 10Y and 30Y auctions land in the same refunding week; "
            "5-day windows overlap and daily yield changes autocorrelate. So the **primary** test is a "
            "daily dummy regression $\\Delta y_t = \\alpha + \\beta_{pre} D^{pre}_t + \\beta_{post} "
            "D^{post}_t + \\varepsilon_t$ with **Newey-West** (8 lags ≈ one window width) — not a "
            "per-event t that pretends windows are i.i.d. The per-auction Welch vs ordinary weeks is "
            "reported as *supporting only* (rolling baseline windows overlap massively).\n"
            "2. **Secular size growth.** Offerings grew ~20× since 1979 — a raw big/small split just "
            "compares eras. Size is detrended by the trailing median of the prior 8 same-tenor "
            "auctions.\n"
            "3. **Carry masquerading as alpha.** TLT earns a positive unconditional term premium, so a "
            "positive round-trip mean could be beta. The alpha test regresses daily TLT **excess** "
            "returns on $D^{post}$ — the coefficient is the *extra* return of a post-auction day over "
            "an ordinary day."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Events.** All **{R['n_auctions']}** 10Y/30Y auctions {R['first']} → {R['last']} "
            "(FiscalData `auctions_query`, `original_security_term` ∈ {10-Year, 30-Year}; reopenings "
            f"kept — same supply event). **{t10[0]}** (10Y) and **{t30[0]}** (30Y) land on the yield "
            "tape with full ±5-day windows.\n"
            "- **Windows.** Pre = 5 trading days ending **on the auction-day close** (1 pm results are "
            "inside it); post = the 5 after. Auction dates snapped to the last trading day ≤ date.\n"
            "- **Primary inference.** HAC/Newey-West t on the dummy regression, per tenor.\n"
            "- **Splits.** Eras declared up front (pre-2008 / 2008–2019 / 2020+); size detrended, "
            "median split, Welch.\n"
            "- **Tradability.** Long TLT day A+1…A+5 (entry decided at close A — **one execution "
            "lag**, the calendar is public weeks ahead); merged overlapping windows; 2 and 5 bps "
            "one-way × NAV per leg; cash earns ^IRX; excess-vs-excess.\n"
            "- **Positive control.** Deterministic random-walk world, synthetic auctions every 21 "
            "days, planted 4-bps-in / 4-bps-out V vs a zero-edge null."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The two legs, on the full tape\n\n"
            "The average cumulative yield path around auctions, and the HAC dummy regression that "
            "judges each leg."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for tenor, col in [('10Y','y10'),('30Y','y30')]:\n"
            "        s = st.summarize_tenor(YLDS[col], AU, tenor)\n"
            "        r = s['reg']\n"
            "        rows.append((tenor, s['n_auctions'], s['pre_mean_bps'], s['post_mean_bps'],\n"
            "                     r['t_pre'], r['t_post']))\n"
            "else:\n"
            "    rows = [(k, v[0], v[1], v[2], v[6], v[8]) for k, v in R['ten'].items()]\n"
            "x = np.arange(len(rows)); w = 0.35\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.2, 4.4))\n"
            "a1.bar(x-w/2, [r[2] for r in rows], w, color=AMBER, label='pre-window (cheapening)')\n"
            "a1.bar(x+w/2, [r[3] for r in rows], w, color=GREEN, label='post-window (richening)')\n"
            "a1.set_xticks(x); a1.set_xticklabels([r[0] for r in rows]); a1.axhline(0, c=GREY, lw=.8)\n"
            "a1.set_ylabel('cumulative yield change (bps)'); a1.set_title('Window means'); a1.legend()\n"
            "a2.bar(x-w/2, [r[4] for r in rows], w, color=AMBER, label='HAC t (pre)')\n"
            "a2.bar(x+w/2, [r[5] for r in rows], w, color=GREEN, label='HAC t (post)')\n"
            "a2.axhline(2, ls='--', c=RED); a2.axhline(-2, ls='--', c=RED, label='|t|=2 bar')\n"
            "a2.set_xticks(x); a2.set_xticklabels([r[0] for r in rows]); a2.axhline(0, c=GREY, lw=.8)\n"
            "a2.set_ylabel('Newey-West t'); a2.set_title('...and their significance'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for r in rows: print(f'{r[0]}: n={r[1]}  pre {r[2]:+.2f} bps (t={r[4]:+.2f})  post {r[3]:+.2f} bps (t={r[5]:+.2f})')"
        ),
        md(
            f"> 💡 In plain words: the **left leg clears** on both tenors (10Y t = **{t10[6]:+.2f}**, "
            f"30Y t = **{t30[6]:+.2f}**) — a genuine ~1.5–3 bps concession per auction week. The "
            f"**right leg doesn't**: 10Y t = {t10[8]:+.2f} sits *just* under the bar and 30Y "
            f"t = {t30[8]:+.2f} is nowhere. `MIXED`, spelled out: **Real on the "
            "cheapen-into · Weak on the richen-after.**"
        ),
        md(
            "### 4b · Era split — supply pressure moved the effect around\n\n"
            "Eras declared up front (GFC/QE start 2008, pandemic deficits 2020) — descriptive cuts, "
            "not snooped breakpoints."
        ),
        code(
            "if HAVE_REAL:\n"
            "    eras = []\n"
            "    for tenor, col in [('10Y','y10'),('30Y','y30')]:\n"
            "        for e in st.era_split(YLDS[col], AU.loc[AU['tenor']==tenor,'auction_date']):\n"
            "            eras.append((tenor, e['era'], e['pre_bps_day'], e['t_pre'], e['post_bps_day'], e['t_post'], e['n_auctions']))\n"
            "else:\n"
            "    eras = R['era']\n"
            "labels = [f'{t} {e}' for t,e,*_ in eras]\n"
            "x = np.arange(len(eras)); w=.35\n"
            "fig, ax = plt.subplots(figsize=(10.6, 4.6))\n"
            "ax.bar(x-w/2, [r[3] for r in eras], w, color=AMBER, label='HAC t (pre leg)')\n"
            "ax.bar(x+w/2, [r[5] for r in eras], w, color=GREEN, label='HAC t (post leg)')\n"
            "ax.axhline(2, ls='--', c=RED); ax.axhline(-2, ls='--', c=RED, label='|t|=2 bar')\n"
            "ax.axhline(0, c=GREY, lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha='right')\n"
            "ax.set_ylabel('Newey-West t'); ax.legend()\n"
            "ax.set_title('Concession: post-GFC and strongest in the 2020+ flood. Richening: gone after 2020.')\n"
            "plt.tight_layout(); plt.show()\n"
            "for r in eras: print(f'{r[0]} {r[1]}: pre {r[2]:+.3f} bps/day (t={r[3]:+.2f})  post {r[4]:+.3f} (t={r[5]:+.2f})  n={r[6]}')"
        ),
        md(
            "> 💡 In plain words: before 2008 there was barely any concession (t ≈ 0.5–0.8). It became "
            "decisive for the **30Y in the QE decade** (t = **3.86**) and for the **10Y in the 2020+ "
            "supply flood** (t = **3.39**, ~4.7 bps/week). Meanwhile the **post-leg flipped positive "
            "after 2020** on both tenors — yields now keep drifting up *after* auctions too. The "
            "concession grew; the mean-reversion died."
        ),
        md(
            "### 4c · The size dose-response (third axis)\n\n"
            "If the mechanism is dealer risk-bearing capacity, more supply should buy more concession. "
            "Size = offering ÷ trailing median of the prior 8 same-tenor auctions; median split; "
            "Welch t on per-auction pre-window changes."
        ),
        code(
            "if HAVE_REAL:\n"
            "    S = {t: st.size_split(AU, YLDS[c], t) for t, c in [('10Y','y10'),('30Y','y30')]}\n"
            "    rows = [(t, S[t]['pre_big_bps'], S[t]['n_big'], S[t]['pre_small_bps'], S[t]['n_small'], S[t]['welch_t_pre']) for t in S]\n"
            "else:\n"
            "    rows = [(t, *R['size'][t][:5]) for t in R['size']]\n"
            "x = np.arange(len(rows)); w=.35\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.4))\n"
            "ax.bar(x-w/2, [r[1] for r in rows], w, color=RED, label='BIG auctions (pre-window bps)')\n"
            "ax.bar(x+w/2, [r[3] for r in rows], w, color=GREY, label='small auctions')\n"
            "ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows])\n"
            "for i,r in enumerate(rows): ax.annotate(f'Welch t={r[5]:+.2f}', (i, max(r[1],r[3])+.15), ha='center')\n"
            "ax.set_ylabel('pre-window cheapening (bps)')\n"
            "ax.set_title('No dose-response: big prints do NOT get bigger concessions')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for r in rows: print(f'{r[0]}: big {r[1]:+.2f} bps (n={r[2]})  small {r[3]:+.2f} bps (n={r[4]})  Welch t={r[5]:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: the 10Y points the **wrong way** (big {s10[0]:+.2f} vs small "
            f"{s10[2]:+.2f} bps, Welch t = **{s10[4]:+.2f}**) and the 30Y points the right way but "
            f"nowhere near the bar (t = **{s30[4]:+.2f}**). The concession is a *calendar* habit — the "
            "market clears its throat before every auction, big or small. **BUSTED.**"
        ),
        md(
            "### 4d · Tradability — the post-auction TLT round trip\n\n"
            "Entry at the auction-day close (one lag), hold 5 days, merged windows, excess of bills, "
            "costs per leg. Plus the carry-clean alpha regression (is a post-auction day *abnormally* "
            "good, beyond TLT's unconditional carry?)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    al = st.tlt_post_alpha(TLTF, AU['auction_date'])\n"
            "    trs = {cb: st.tlt_post_auction(TLTF, AU['auction_date'], cost_bps=cb) for cb in (2.0, 5.0)}\n"
            "    t1 = TLTF[TLTF.index <= '2019-12-31']; d1 = AU.loc[AU['auction_date'] <= '2019-12-31','auction_date']\n"
            "    t2 = TLTF[TLTF.index >= '2020-01-01']; d2 = AU.loc[AU['auction_date'] >= '2020-01-01','auction_date']\n"
            "    subs = [('2002-2019', st.tlt_post_auction(t1, d1, cost_bps=2.0)),\n"
            "            ('2020-2026', st.tlt_post_auction(t2, d2, cost_bps=2.0))]\n"
            "    bars = [('full @2bp', trs[2.0]['net_ann_pct'], trs[2.0]['t_net_held']),\n"
            "            ('full @5bp', trs[5.0]['net_ann_pct'], trs[5.0]['t_net_held']),\n"
            "            ('2002-2019 @2bp', subs[0][1]['net_ann_pct'], subs[0][1]['t_net_held']),\n"
            "            ('2020-2026 @2bp', subs[1][1]['net_ann_pct'], subs[1][1]['t_net_held'])]\n"
            "    alpha_line = (al['base_bps_day'], al['post_extra_bps_day'], al['t_post_extra'], al['bh_excess_ann_pct'])\n"
            "else:\n"
            "    bars = [('full @2bp', R['tlt']['cost'][0][2], R['tlt']['cost'][0][3]),\n"
            "            ('full @5bp', R['tlt']['cost'][1][2], R['tlt']['cost'][1][3]),\n"
            "            ('2002-2019 @2bp', R['tlt']['sub'][0][2], R['tlt']['sub'][0][3]),\n"
            "            ('2020-2026 @2bp', R['tlt']['sub'][1][2], R['tlt']['sub'][1][3])]\n"
            "    alpha_line = (R['tlt']['alpha_base'], R['tlt']['alpha_extra'], R['tlt']['t_alpha'], R['tlt']['bh_excess'])\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.5))\n"
            "colors = [GREEN if b[2] >= 2 else (RED if b[1] < 0 else AMBER) for b in bars]\n"
            "ax.bar([b[0] for b in bars], [b[1] for b in bars], color=colors, width=.55)\n"
            "ax.axhline(0, c=GREY, lw=.8)\n"
            "for i,b in enumerate(bars): ax.annotate(f'{b[1]:+.1f}%\\nt={b[2]:+.2f}', (i, b[1]), ha='center', va='bottom' if b[1]>=0 else 'top')\n"
            "ax.set_ylabel('net excess return (%/yr)')\n"
            "ax.set_title('Post-auction TLT round trips: real at tight costs, dead since 2020')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'alpha reg: ordinary day {alpha_line[0]:+.2f} bps/day, post-auction extra {alpha_line[1]:+.2f} bps/day (t={alpha_line[2]:+.2f}); TLT B&H excess {alpha_line[3]:+.2f}%/yr')\n"
            "for b in bars: print(f'{b[0]}: net {b[1]:+.2f}%/yr  HAC t={b[2]:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: on the full tape the trade clears at tight costs (net "
            f"**+{T['cost'][0][2]:.2f}%/yr**, t = **{T['cost'][0][3]:+.2f}**; and the alpha regression "
            f"says post-auction days really were **+{T['alpha_extra']:.1f} bps/day** better than "
            f"ordinary days, t = **{T['t_alpha']:+.2f}** — not just carry, since TLT's unconditional "
            f"excess is only {T['bh_excess']:.1f}%/yr at 100% exposure vs this at 38.5%). But it is a "
            f"**2002–2019 phenomenon** (t = {T['sub'][0][3]:.2f}) that has been **negative since "
            f"2020** ({T['sub'][1][2]:+.1f}%/yr), and 5 bps/leg already drops it under the bar "
            f"(t = {T['cost'][1][3]:.2f}). Real once — **FRAGILE** now."
        ),
        md(
            "### 4e · Faithful-engine & power control — we know the truth here\n\n"
            "A deterministic random-walk yield world with synthetic auctions every 21 trading days. "
            "Null: no planted effect — the detector must stay quiet. Plant: 4 bps of cheapening in, "
            "4 bps of richening out — it must light up, and the fund trade must bank it."
        ),
        code(
            "res = []\n"
            "for c, r_, label in [(0.0, 0.0, 'null (0/0)'), (4.0, 4.0, 'planted 4/4 bps')]:\n"
            "    au_s, y_s, tlt_s = data.synthetic_world(concession_bps=c, richen_bps=r_, seed=603)\n"
            "    reg = st.dummy_regression(y_s['y10'], au_s['auction_date'])\n"
            "    tr = st.tlt_post_auction(tlt_s, au_s['auction_date'], cost_bps=2.0)\n"
            "    res.append((label, reg['t_pre'], reg['t_post'], tr['net_ann_pct'], tr['t_net_held']))\n"
            "x = np.arange(len(res)); w=.28\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.4))\n"
            "ax.bar(x-w, [r[1] for r in res], w, color=AMBER, label='HAC t (pre leg)')\n"
            "ax.bar(x,   [r[2] for r in res], w, color=GREEN, label='HAC t (post leg)')\n"
            "ax.bar(x+w, [r[4] for r in res], w, color=GREY, label='fund trade t (net)')\n"
            "ax.axhline(2, ls='--', c=RED); ax.axhline(-2, ls='--', c=RED, label='|t|=2 bar')\n"
            "ax.axhline(0, c=GREY, lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([r[0] for r in res])\n"
            "ax.set_ylabel('t-statistic'); ax.set_title('Null stays quiet; the planted V lights up everywhere')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for r in res: print(f'{r[0]}: t_pre={r[1]:+.2f}  t_post={r[2]:+.2f}  fund net {r[3]:+.2f}%/yr (t={r[4]:+.2f})')"
        ),
        md(
            f"> 💡 In plain words: with nothing planted the machinery reads t = "
            f"**{R['syn'][0][2]:+.2f} / {R['syn'][0][4]:+.2f}** (quiet); with a 4-bps V it reads "
            f"**{R['syn'][1][2]:+.2f} / {R['syn'][1][4]:+.2f}** and the fund banks "
            f"**+{R['syn'][1][5]:.2f}%/yr** (t = {R['syn'][1][6]:+.2f}). The real-tape t-stats are "
            "measurements, not construction artefacts. *(A faithful-engine / power check only — never "
            "cited in support of the real-tape stamps.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `MIXED`** — *Real on the cheapen-into leg*: 10Y **+{t10[1]:.2f} bps**/window "
            f"(HAC t = **{t10[6]:+.2f}**), 30Y **+{t30[1]:.2f} bps** (t = **{t30[6]:+.2f}**), strongest "
            "exactly where the supply story says (10Y 2020+ t = 3.39; 30Y 2008-2019 t = 3.86). *Weak "
            f"on the richen-after leg*: 10Y t = {t10[8]:+.2f}, 30Y t = {t30[8]:+.2f}, sign-flipped "
            "post-2020. No survivorship (official complete record, constant-maturity indices).\n"
            f"- **Tradability `FRAGILE`** — post-auction TLT: net +{T['cost'][0][2]:.2f}%/yr at "
            f"2 bps/leg (t = {T['cost'][0][3]:+.2f}), alpha t = {T['t_alpha']:+.2f}; but 2002-2019 "
            f"t = {T['sub'][0][3]:.2f} vs 2020+ net {T['sub'][1][2]:+.2f}%/yr, and t = "
            f"{T['cost'][1][3]:.2f} at 5 bps/leg. Real, then decayed. Not INVESTABLE.\n"
            f"- **Size dose-response `BUSTED`** — Welch t = {s10[4]:+.2f} (10Y, wrong sign) / "
            f"{s30[4]:+.2f} (30Y, n.s.). The concession follows the calendar, not the print size."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **Intraday would sharpen both legs.** Lou-Yan-Zhang find part of the reversal happens "
            "within days of the auction close; daily closes blur it. A tick-level replication (BrokerTec "
            "/ futures) is the natural upgrade — the daily tape is the honest retail-visible version.\n"
            "- **The 2020 regime break is the live question.** Concession up, mean-reversion gone: "
            "consistent with dealer balance-sheet caps binding *permanently* rather than episodically. "
            "Treasury buybacks (2024→) are a natural experiment to watch.\n"
            "- **Different plumbing, different studies:** "
            "[382-treasury-basis-trade](../../382-treasury-basis-trade/README.md) (cash-futures basis), "
            "[383-sofr-repo-stress](../../383-sofr-repo-stress/README.md) (repo spikes), "
            "[380-curve-roll-down](../../380-curve-roll-down/README.md) (static roll-down).\n\n"
            "*The reproducible core is offline and deterministic; the signal is the public auction "
            "calendar, the myth-check is the size dose-response. Methods and sources: "
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
