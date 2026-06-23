"""Generate the two narrative notebooks for Study 378 (ETF NAV Premium/Discount).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached ETF prices
under ../_cache/ (the NAV-proxy basis) and otherwise quote the frozen headline numbers in
``R`` (mirroring docs/results.md). The synthetic positive control runs anywhere with no
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance ETF prices, NAV proxy
# from a hedge basket per ETF; HYG/EMB/MUB + legs, 2012-01-03 -> 2026-06-18, 3,636 days, 14.5y).
R = dict(
    start="2012-01-03", end="2026-06-18", days=3636, years=14.5,
    targets=["HYG", "EMB", "MUB"],
    # pooled hedged harvest after a discount (z<-1, lag1): (h, n, harvest%, win%, HAC t)
    pool=[(5, 1664, 0.0525, 56, 2.03), (10, 1663, 0.0496, 54, 1.24),
          (20, 1663, 0.0543, 53, 0.92)],
    # per-ETF, 10-day hold: name -> (n, depth%, harvest%, win%, HAC t, placebo p)
    per={"HYG": (504, -0.27, -0.0119, 45, -0.58, 0.978),
         "EMB": (597, -1.47, 0.0696, 54, 0.67, 0.028),
         "MUB": (562, -0.36, 0.0835, 64, 2.40, 0.000)},
    # costs vs pooled 5-day gross (0.0525%): (bp_per_leg, cost%, net%)
    costs=[(1.0, 0.0400, 0.0125), (2.0, 0.0800, -0.0275), (3.0, 0.1200, -0.0675)],
    # robustness z sweep, pooled 5-day: (z, n, gross%, win%, t)
    robust=[(-0.5, 3047, 0.0441, 56, 2.83), (-1.0, 1664, 0.0525, 56, 2.03),
            (-1.5, 846, 0.0654, 56, 1.53), (-2.0, 403, 0.1338, 58, 1.74)],
    # stress clustering: name -> (share_2020%, worst%, worst_date)
    stress={"HYG": (24, -1.04, "2016-05-09"), "EMB": (22, -9.31, "2020-03-18"),
            "MUB": (25, -4.43, "2020-03-18")},
    # synthetic control: (edge_bp, detected, harvest%, t, p)
    syn=[(0.0, 576, -0.0642, -1.17, 0.989), (5.0, 576, 0.1702, 3.10, 0.001)],
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Free_discount%3F: Busted](https://img.shields.io/badge/Free_discount%3F-Busted-8b949e?style=flat-square)\n\n"
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

from etf_nav_premium import data, strategy as st

HAVE_REAL = data.have_real()
FRAMES = data.load_real() if HAVE_REAL else None
print("real ETF-price cache present:", HAVE_REAL,
      "| target ETFs:", (list(FRAMES) if FRAMES else None))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# The ETF \"on sale\" trade — does buying below NAV pay? ⚖️\n"
            "### When a fund's price dips under the value of what it owns, is that free money?\n\n"
            + BADGES +
            "Here's a tidy-sounding idea. An ETF is just a basket of bonds (or stocks). Each day the "
            "fund tells you the **NAV** — the value of everything inside. Sometimes the ETF's *price* "
            "dips **below** that NAV: the wrapper is cheaper than its contents. Lore says that's a "
            "**discount on sale** — the gap *has* to close, so buy the discount and pocket it when it "
            "snaps back. The bond-ETF crash of **March 2020** (HYG, EMB, MUB all printing big discounts) "
            "is Exhibit A.\n\n"
            "It sounds like arbitrage. But two things have to be true for it to *pay*: the gap has to "
            "close **in your pocket** (not just because the whole bond market bounced), and the snap-back "
            "has to be **bigger than the spread** you cross getting in and out. This notebook checks "
            "both — and the answer is that the discount is real arithmetic, not free money.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the hedge, the cost sweep and the "
            "stress-clustering test? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** The official intraday NAV isn't free on yfinance, so we "
            "**build a transparent proxy**: a fair value for each ETF from a sister fund + a rates leg. "
            "We call it a proxy throughout — but a *tighter*, real NAV would only make the gap **smaller** "
            "and the verdict stronger. Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| After a discount, does the gap shrink? | **Partly, yes.** Buy when the ETF is cheap to "
            "fair value and on average the gap closes a little over the next week. |\n"
            "| So you earn the discount back? | **Barely.** Once you strip out the bond market's own "
            f"move, the *harvest* is about **{R['pool'][0][2]:.0f}/100ths of a percent** "
            "(~5 basis points) — and it fades to nothing within two weeks. |\n"
            "| Does it beat trading costs? | **No.** The bid-ask you cross — on the ETF *and* the hedge "
            "— is the same size as the whole harvest. At a routine 2-bp spread the trade is already "
            "**net negative**. |\n"
            "| Then why does everyone remember it working? | **March 2020.** The biggest discounts all "
            "cluster in that one crisis — when the machinery that closes the gap **broke** and you "
            "*couldn't* trade at NAV anyway. The vivid case is the un-tradable one. |\n\n"
            "> The discount is real — but it's mostly the basket moving, what reverts is pennies, costs "
            "eat it, and the giant gaps show up exactly when you can't use them. \"On sale\" is a story."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"An ETF that trades below its NAV is cheaper than the things it holds. That discount "
            "is a free gap — the price and the NAV must converge — so buy the discount, wait, and "
            "collect the difference. Bond ETFs in March 2020 traded at huge discounts and then "
            "recovered: textbook.\"*\n\n"
            "The picture is seductive because the discount is a **real, published number** — every ETF "
            "reports its premium/discount daily, and during stress bond ETFs *did* trade percentage "
            "points below NAV. The leap is from *\"the gap exists and is reported\"* to *\"the gap is "
            "free money you can harvest.\"* We'll rebuild the discount and put that leap under a "
            "microscope, on the funds where the gap is biggest: **HYG** (high yield), **EMB** (emerging "
            "markets), **MUB** (munis)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If \"buy the discount\" really paid, it would be a rare thing: a **structural** edge, "
            "baked into how ETFs are wrapped, available to anyone with a brokerage account — not a "
            "fragile pattern but a mechanical one. That's why it's tempting.\n\n"
            "But notice the trap hiding in the word *discount*. When an ETF is 1% below NAV, that gap "
            "can close two ways: the **ETF rises** to meet NAV (you win) — or the **NAV falls** to meet "
            "the ETF (you don't). For bond ETFs the NAV is partly **stale** (many bonds don't trade "
            "every day), so a \"discount\" often means the ETF has *already* spotted a markdown the NAV "
            "hasn't caught up to. Buying that isn't buying a bargain — it can be buying **ahead of a "
            "drop**. The only honest test is: net of the whole bond market moving, and net of costs, "
            "does the buyer actually come out ahead?"
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We rebuild the discount on a **transparent NAV proxy**. For each ETF we build a fair "
            "value from a sister fund + a rates leg, and define the **basis** = how far the ETF has "
            "drifted from that fair value (positive = rich/**premium**, negative = cheap/**discount**). "
            f"Over **{R['years']:.0f} years** ({R['start']} → {R['end']}) we then:\n\n"
            "1. **Mark the discounts.** Flag every day the ETF is unusually cheap (basis well below its "
            "own recent average).\n"
            "2. **Measure the *hedged* harvest.** Enter the next day, hold a week or two, and bank the "
            "ETF's move **minus** the fair-value basket's move — so we measure the *gap closing*, not "
            "*the bond market rallying*.\n"
            "3. **Charge the spread, and look at the crises.** Net the harvest against a realistic "
            "bid-ask on both legs, and check *when* the biggest discounts actually happen."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, here's a discount.** The premium/discount basis for one ETF — drifting rich and "
            "cheap around fair value. The deep dive in early 2020 is the famous bond-ETF dislocation."
        ),
        code(
            "if HAVE_REAL:\n"
            "    f = FRAMES['EMB']; b = f['basis']*100\n"
            "    fig, ax = plt.subplots(figsize=(9.4, 4.2))\n"
            "    ax.plot(b.index, b.values, c=GREY, lw=.9)\n"
            "    ax.fill_between(b.index, b.values, 0, where=(b.values<0), color=RED, alpha=.5, label='discount (cheap)')\n"
            "    ax.fill_between(b.index, b.values, 0, where=(b.values>0), color=GREEN, alpha=.4, label='premium (rich)')\n"
            "    ax.axhline(0, c='k', lw=.8)\n"
            "    ax.set_title('EMB price vs NAV proxy — the prem/disc basis (note March 2020)')\n"
            "    ax.set_ylabel('premium(+) / discount(-)  %'); ax.legend()\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('deepest EMB discount:', f'{b.min():.2f}%', 'on', b.idxmin().date())\n"
            "else:\n"
            "    print('no cache — see docs/results.md; EMB worst discount', R['stress']['EMB'][1], '% on', R['stress']['EMB'][2])"
        ),
        md(
            f"Vivid — and look *where* the worst gap is: **{R['stress']['EMB'][1]:.1f}% on "
            f"{R['stress']['EMB'][2]}**, the COVID liquidity crunch. Hold that thought. Now the real "
            "question: after a discount, what does the buyer actually take home?"
        ),
        md(
            "**The harvest vs the bid-ask.** For each hold length, the *hedged* discount earn-back "
            "(pooled across the three ETFs) next to a realistic round-trip cost. The harvest is the "
            "green bar; the dashed line is what you pay to trade."
        ),
        code(
            "hs = [h for h,*_ in R['pool']]\n"
            "if HAVE_REAL:\n"
            "    harv = [st.pooled_harvest(FRAMES, h)['harvest_mean']*100 for h in hs]\n"
            "else:\n"
            "    harv = [p[2] for p in R['pool']]\n"
            "cost2 = R['costs'][1][1]   # 2bp/leg round-trip cost, %\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "x = np.arange(len(hs))\n"
            "ax.bar(x, harv, .5, color=GREEN, label='discount harvested (hedged)')\n"
            "ax.axhline(cost2, ls='--', c=RED, label=f'round-trip cost @2bp/leg ({cost2:.2f}%)')\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}-day hold' for h in hs])\n"
            "ax.set_ylabel('return (%)'); ax.set_title('The harvest (~0.05%) is smaller than the spread you cross')\n"
            "for i,v in enumerate(harv): ax.annotate(f'{v:.3f}%',(i,v),ha='center',va='bottom')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('harvest by horizon (%):', {f'{h}d': round(v,4) for h,v in zip(hs,harv)}, '| 2bp round-trip:', f'{cost2:.2f}%')"
        ),
        md(
            f"There's the whole story in one chart. The discount earn-back is about "
            f"**{R['pool'][0][2]:.3f}%** — five hundredths of a percent — and the cost of getting in and "
            f"out is **{R['costs'][1][1]:.2f}%**, *bigger than the harvest*. You are crossing a spread to "
            "collect less than the spread. And it gets worse: most of the apparent 'discount' was just "
            "the bond market dropping (which we hedged out), so the raw, unhedged 'snap-back' is mostly "
            "borrowed from beta you were carrying anyway."
        ),
        md(
            "**Does it even work the same way on each ETF?** A real structural edge should show up "
            "everywhere. Here's the hedged earn-back per fund — watch one of them go the *wrong* way."
        ),
        code(
            "names = R['targets']\n"
            "if HAVE_REAL:\n"
            "    vals = []\n"
            "    for tg in names:\n"
            "        ev = st.detect_discounts(FRAMES[tg]); s = st.summarize(FRAMES[tg], ev, 10)\n"
            "        vals.append(s['harvest_mean']*100)\n"
            "else:\n"
            "    vals = [R['per'][tg][2] for tg in names]\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "cols = [GREEN if v>0 else RED for v in vals]\n"
            "ax.bar(names, vals, color=cols, width=.55)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(vals): ax.annotate(f'{v:+.3f}%',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "ax.set_ylabel('hedged earn-back, 10-day (%)')\n"
            "ax.set_title('Buying the discount: helps for MUB, nothing for EMB, LOSES for HYG')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('per-ETF 10d harvest (%):', {tg: round(v,4) for tg,v in zip(names, vals)})"
        ),
        md(
            f"That's the tell. **HYG's discount earn-back is *negative*** "
            f"(**{R['per']['HYG'][2]:+.3f}%**): buying high-yield when it looks cheap, hedged, **loses** "
            "money. EMB is a coin-flip. Only **MUB** carries the pooled number. A 'free discount' that "
            "*reverses sign* across the funds it's meant to describe isn't a structural law — it's three "
            "noisy series, one of which happened to wobble up."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — Weak.** The discount *does* partly close — but the hedged harvest is ~5 bp, "
            "fades within two weeks, and **flips negative on HYG**. Real as a wobble, weak as an edge.\n"
            "- **Tradability — Mirage.** The harvest is **smaller than the bid-ask** on two legs "
            "(net-negative at a normal 2-bp spread), and the biggest discounts hit in **stress**, when "
            "the gap-closing machinery is broken.\n"
            "- **\"Free discount\"? — Busted.** Below-NAV isn't 'on sale': most of the gap is the basket "
            "moving, what reverts is pennies, costs erase it, and the headline crises are exactly the "
            "moments you *can't* trade it."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — the crisis trap\n\n"
            "Forget the basis points for a second. The deeper problem is *when* the big discounts "
            "happen. If they were sprinkled evenly through calm markets you could at least try to "
            "harvest them. They aren't. Here's where each ETF's **deepest** discounts land on the "
            "calendar."
        ),
        code(
            "names = R['targets']\n"
            "if HAVE_REAL:\n"
            "    shares = []\n"
            "    for tg in names:\n"
            "        b = FRAMES[tg]['basis']; deep = b[b < b.quantile(0.05)]\n"
            "        shares.append((deep.index.year==2020).mean()*100)\n"
            "else:\n"
            "    shares = [R['stress'][tg][0] for tg in names]\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "ax.bar(names, shares, color=RED, width=.55, label='share of deepest discounts in 2020')\n"
            "ax.axhline(7, ls='--', c=GREY, label='if spread evenly (~7%)')\n"
            "for i,v in enumerate(shares): ax.annotate(f'{v:.0f}%',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('% of deepest-5% discount days in 2020'); ax.set_ylim(0, 35)\n"
            "ax.set_title('A quarter of the biggest discounts hit in ONE crisis year'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('share of deepest discounts in 2020 (%):', {tg: round(s) for tg,s in zip(names, shares)}, '| uniform ~7%')"
        ),
        md(
            f"Around **a quarter** of every ETF's worst discounts fall in **2020** — roughly **3× more** "
            "than if they were spread evenly. That's the cruel joke of the trade: the discount is "
            "tiny and un-harvestable when markets are calm, and *enormous* exactly when the "
            "creation/redemption arbitrage has frozen, the underlying bonds aren't trading, and the "
            "'NAV' you're comparing to is itself stale. The romantic version — 'wait for a giant "
            "discount and back up the truck' — is the version you **can't execute**."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Use a real NAV.** Swap our hedge-basket proxy for a fund's published daily NAV (from "
            "the issuer or a paid feed) and re-run — the gap will be *tighter* and the harvest "
            "*smaller*, because the true creation/redemption arb is faster than our proxy.\n"
            "- **Try equity ETFs.** SPY/IVV trade within a basis point of NAV all day — there's nothing "
            "to harvest. The discount story only *looks* alive on illiquid underlyings (HY, EM, munis), "
            "which is exactly where costs and stale NAVs bite hardest.\n"
            "- **Separate stale-NAV from mispricing.** Build the test around *next-day NAV* to see how "
            "much of the 'discount' is the NAV catching down to a price the ETF already found.\n\n"
            "*Think the below-NAV trade pays after costs? Hedge out the basket, charge both legs the "
            "spread, and show a harvest that survives **outside** March 2020 — then we'll talk.*"
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
            "# ETF NAV premium/discount — a quantitative teardown 🔬\n"
            "### NAV-proxy basis construction · hedged-harvest with a HAC *t* + placebo null · the "
            "per-ETF sign reversal · a cost / threshold / stress battery · a faithful-engine control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The claim "
            "*\"buy below NAV, earn the discount back\"* fuses two things we separate: a **basis that "
            "mean-reverts** (true, by stationarity) and a **harvestable, net-of-cost return** (the only "
            "thing that pays). We measure the *hedged* residual — the ETF's move minus its fair-value "
            "basket — so the number is the **gap closing**, not the bond market rallying. The decisive "
            "object is its **magnitude vs the round-trip**, not its sign.\n\n"
            "> ⚠️ **Data + proxy note.** Official intraday iNAV / end-of-day NAV isn't on yfinance; we "
            "build a transparent **NAV proxy** — a per-ETF hedge-basket fair value (sister credit ETF + "
            "rate leg), with the **basis** = detrended cumulative return residual (stationary, like a "
            "real prem/disc). A tighter true NAV only *shrinks* the gap. Real data: yfinance daily "
            "adjusted closes, HYG/EMB/MUB + legs, 2012→2026. Offline core + synthetic control are "
            "deterministic. Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | pooled hedged harvest **+{R['pool'][0][2]:.4f}%** / 5-day, "
            f"HAC **t = {R['pool'][0][4]:.2f}** — but **t = {R['pool'][1][4]:.2f}** by 10 days, and "
            f"**HYG t = {R['per']['HYG'][4]:.2f}** (sign reversal across ETFs). |\n"
            f"| **Tradability** | `MIRAGE` | gross ~5 bp < round-trip: net **{R['costs'][1][2]:+.4f}%** "
            f"at {R['costs'][1][0]:.0f} bp/leg. Deepest discounts ~{R['stress']['MUB'][0]}% in 2020 "
            "(uniform ~7%) — un-harvestable in stress. |\n"
            f"| **Free discount?** | `BUSTED` | the *t* **fades as the discount deepens** "
            f"({R['robust'][2][4]:.2f} at z<−1.5) and is carried by one ETF (MUB). A fair-value wobble, "
            "not a structural arb. |\n\n"
            "> 💡 In plain words: the basis reverts because it's *built* to be stationary; the question "
            "is whether the buyer keeps anything. Strip the beta and charge the spread and ~5 bp becomes "
            "≤ 0 — and the giant discounts are the ones you can't touch."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $p_t = \\log P^{\\text{ETF}}_t$ and let $\\hat\\nu_t$ be a NAV-proxy fair value (a "
            "hedge basket of a sister ETF + a rate leg). Define the **basis** "
            "$b_t = (p_t - \\hat\\nu_t)$ detrended by a rolling mean — stationary, so $\\mathbb{E}[\\Delta "
            "b \\mid b<0] > 0$ *by construction*. A **discount** fires when the rolling z-score "
            "$z_t = (b_t-\\bar b)/s_b < -1$.\n\n"
            "- **H₁ (the discount is harvestable).** The **hedged** residual return after a discount, "
            "$\\sum_{i=1}^{h} \\varepsilon_{t+i}$, is positive with $t \\ge 2$ — a *net-of-beta* edge.\n"
            "- **H₂ (it's deployable).** That harvest exceeds the round-trip half-spread on the ETF "
            "**and** the hedge leg, at scale, including in stress.\n"
            "- **H₃ (\"free discount\").** The harvest is a structural property of below-NAV, stable "
            "across ETFs and *stronger* in deep discounts.\n\n"
            "We find **H₁ marginal** (pooled t≈2 at 5d, <2 beyond, sign-flips on HYG), **H₂ rejected** "
            "(gross < cost; stress-clustered), **H₃ rejected** (t *fades* with depth; one-ETF effect). "
            "The basis reverts; the investor doesn't get paid for it."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The whole teardown is one estimator: the mean **hedged** harvest after a discount, judged "
            "by a HAC standard error (overlapping holds ⇒ autocorrelation).\n\n"
            "$$\\widehat{H}_h = \\frac{1}{k}\\sum_{j=1}^{k}\\sum_{i=1}^{h}\\varepsilon_{\\tau_j+i},"
            "\\qquad t = \\frac{\\widehat{H}_h}{\\widehat{\\mathrm{se}}_{\\text{NW}}(\\widehat{H}_h)}.$$\n\n"
            "Three traps a naive test falls into: (1) using the **raw** ETF return — that's mostly the "
            "bond market, so we hedge to the residual $\\varepsilon$; (2) using a **plain** t on "
            "overlapping holds — autocorrelation inflates it, so we use Newey-West; (3) reading a "
            "stationary basis's reversion as *profit* — it's not, unless it clears the **round-trip**. "
            "The cost line, not the point estimate, is where the verdict is decided."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **NAV proxy.** Per ETF, OLS-hedge daily log-returns on the legs (sister ETF + rates); "
            "residual $\\varepsilon_t$ = ETF − hedge; basis $b_t$ = detrended cumulative $\\varepsilon$ "
            f"(rolling-60d). Explicit **proxy** for official NAV. {R['start']}→{R['end']}, "
            f"{R['days']:,} days.\n"
            "- **Discount detector.** Rolling-252d z of $b_t$; fire when $z<-1$ (sweep $-0.5…-2$).\n"
            "- **Harvest.** $\\sum_{i=1}^h \\varepsilon_{t+i}$, entered **1 day after** the signal (no "
            "look-ahead), $h\\in\\{5,10,20\\}$; events overrunning the tape dropped.\n"
            "- **Inference.** Newey-West (Bartlett, 10-lag) HAC t; plus a **placebo** null — 20,000 "
            "draws of $k$ random valid entry dates, $p=\\Pr[\\text{random mean}\\ge\\text{discount mean}]$.\n"
            "- **Costs.** One-way half-spread × (ETF + hedge) legs, swept 1–3 bp.\n"
            "- **Stress check.** Share of each ETF's deepest-5% discounts falling in 2020.\n"
            "- **Positive control.** AR(1) basis with a **planted per-day earn-back** on discount days: "
            "the inference must recover a large edge **and** must NOT manufacture one at zero edge."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The point estimates — positive, tiny, decaying\n\n"
            "Pooled hedged harvest by horizon with its $\\pm$ HAC standard error, and the Newey-West "
            "*t* annotated. Positive everywhere, but the *t* only clears 2 at 5 days and decays after."
        ),
        code(
            "hs = [h for h,*_ in R['pool']]\n"
            "if HAVE_REAL:\n"
            "    hv = []; ses = []; ts = []\n"
            "    for h in hs:\n"
            "        xs = []\n"
            "        for f in FRAMES.values():\n"
            "            ev = st.detect_discounts(f); xs.append(st.hedged_harvest(f, ev, h))\n"
            "        x = np.concatenate(xs); hv.append(x.mean()*100)\n"
            "        ses.append(x.std(ddof=1)/np.sqrt(len(x))*100); ts.append(st.hac_t(x))\n"
            "else:\n"
            "    hv = [p[2] for p in R['pool']]; ts = [p[4] for p in R['pool']]; ses=[0.026,0.04,0.059]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x, hv, yerr=ses, capsize=5, color=GREEN, width=.5, label='hedged harvest (±SE)')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,(v,t) in enumerate(zip(hv,ts)): ax.annotate(f't={t:.2f}',(i,v),ha='center',va='bottom')\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}-day' for h in hs])\n"
            "ax.set_ylabel('mean hedged harvest (%)')\n"
            "ax.set_title('~5 bp at 5d, HAC t fades below 2 by 10d'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('pooled harvest %:', {f'{h}d': round(v,4) for h,v in zip(hs,hv)}, '| HAC t:', {f'{h}d': round(t,2) for h,t in zip(hs,ts)})"
        ),
        md(
            f"> 💡 In plain words: at 5 days the harvest is **+{R['pool'][0][2]:.4f}%** at "
            f"**t = {R['pool'][0][4]:.2f}** (just clears 2); by 10 days **t = {R['pool'][1][4]:.2f}**, by "
            f"20 days **t = {R['pool'][2][4]:.2f}**. The mean barely moves but the standard error grows — "
            "a classic *not-an-edge* signature: a positive number that can't keep its significance."
        ),
        md(
            "### 4b · The decisive split — it reverses sign across ETFs\n\n"
            "A structural \"free discount\" must exist in every ETF. Here's the hedged 10-day earn-back "
            "per fund with its HAC *t* — and one of them is **negative**."
        ),
        code(
            "names = R['targets']\n"
            "if HAVE_REAL:\n"
            "    vals = []; ts = []\n"
            "    for tg in names:\n"
            "        ev = st.detect_discounts(FRAMES[tg]); s = st.summarize(FRAMES[tg], ev, 10)\n"
            "        vals.append(s['harvest_mean']*100); ts.append(s['t'])\n"
            "else:\n"
            "    vals = [R['per'][tg][2] for tg in names]; ts = [R['per'][tg][4] for tg in names]\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "cols = [GREEN if v>0 else RED for v in vals]\n"
            "ax.bar(names, vals, color=cols, width=.55)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,(v,t) in enumerate(zip(vals,ts)): ax.annotate(f'{v:+.3f}%\\nt={t:.2f}',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "ax.set_ylabel('hedged 10-day harvest (%)')\n"
            "ax.set_title('HYG t=-0.58 · EMB t=0.67 · MUB t=2.40 — not one structural edge')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('per-ETF (harvest%, t):', {tg:(round(v,4),round(t,2)) for tg,v,t in zip(names,vals,ts)})"
        ),
        md(
            f"> 💡 In plain words: **HYG t = {R['per']['HYG'][4]:.2f}** (buying HY's discount *loses*, "
            f"hedged), **EMB t = {R['per']['EMB'][4]:.2f}** (noise), **MUB t = {R['per']['MUB'][4]:.2f}** "
            "(the only one with a pulse). The pooled t≈2 is **one ETF wearing a crowd**. A law of "
            "ETF microstructure wouldn't pick favourites — H₃ rejected."
        ),
        md(
            "### 4c · Costs + the depth paradox — neither rescues it\n\n"
            "Left: the pooled 5-day harvest vs the round-trip half-spread (ETF + hedge). Right: the HAC "
            "*t* as we tighten the discount threshold — it should *strengthen* in the deep tail if the "
            "effect were structural; it **fades**."
        ),
        code(
            "if HAVE_REAL:\n"
            "    g5 = st.pooled_harvest(FRAMES, 5)['harvest_mean']*100\n"
            "    rob = []\n"
            "    for zt in (-0.5,-1.0,-1.5,-2.0):\n"
            "        p = st.pooled_harvest(FRAMES, 5, z_thr=zt); rob.append((zt, p['n'], p['t']))\n"
            "else:\n"
            "    g5 = R['pool'][0][2]; rob = [(r[0], r[1], r[4]) for r in R['robust']]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.2))\n"
            "cbs = [c[0] for c in R['costs']]; costv = [c[1] for c in R['costs']]\n"
            "a1.bar([f'{c:.0f}bp' for c in cbs], costv, color=RED, width=.5, label='round-trip cost')\n"
            "a1.axhline(g5, ls='--', c=GREEN, label=f'gross harvest {g5:.3f}%')\n"
            "a1.set_title('Gross harvest < round-trip at >=2bp/leg'); a1.set_xlabel('half-spread per leg'); a1.set_ylabel('%'); a1.legend()\n"
            "zs = [r[0] for r in rob]; tt = [r[2] for r in rob]; ns = [r[1] for r in rob]\n"
            "a2.bar([f'{z:.1f}' for z in zs], tt, color=AMBER, width=.55)\n"
            "a2.axhline(2, ls='--', c=RED, label='t=2')\n"
            "for i,(t,k) in enumerate(zip(tt,ns)): a2.annotate(f'n={k}',(i,t),ha='center',va='bottom')\n"
            "a2.set_title('t FADES as the discount deepens'); a2.set_xlabel('z threshold'); a2.set_ylabel('HAC t'); a2.set_ylim(0,3.1); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('robustness (z, n, t):', [(r[0], r[1], round(r[2],2)) for r in rob])"
        ),
        md(
            "> 💡 In plain words: a 2-bp/leg round-trip (**0.08%**) already exceeds the 5-bp gross. And "
            "the *t* is **largest at the loosest threshold** (z<−0.5, where 'discount' means almost "
            f"anything) and **smallest in the deep tail** ({R['robust'][2][4]:.2f} at z<−1.5) — the exact "
            "opposite of a real dislocation edge, which should bite hardest when the gap is biggest. "
            "This is broad, shallow mean-reversion, not discount-harvesting."
        ),
        md(
            "### 4d · The stress trap & the faithful-engine control\n\n"
            "Left: where the deepest discounts live (calendar). Right: the positive control — an AR(1) "
            "basis with a planted per-day earn-back: zero edge must stay below t=2, a large edge must "
            "light up."
        ),
        code(
            "names = R['targets']\n"
            "if HAVE_REAL:\n"
            "    shares = []\n"
            "    for tg in names:\n"
            "        b = FRAMES[tg]['basis']; deep = b[b < b.quantile(0.05)]\n"
            "        shares.append((deep.index.year==2020).mean()*100)\n"
            "else:\n"
            "    shares = [R['stress'][tg][0] for tg in names]\n"
            "res = []\n"
            "for edge in (0.0, 5.0):\n"
            "    syn = data.synthetic_basis(edge_bps=edge, seed=378)\n"
            "    ev = st.detect_discounts(syn); s = st.summarize(syn, ev, 10)\n"
            "    res.append((edge, s['n'], s['harvest_mean']*100, s['t'], s['p_placebo']))\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.2))\n"
            "a1.bar(names, shares, color=RED, width=.55); a1.axhline(7, ls='--', c=GREY, label='uniform ~7%')\n"
            "for i,v in enumerate(shares): a1.annotate(f'{v:.0f}%',(i,v),ha='center',va='bottom')\n"
            "a1.set_title('Deepest discounts cluster in 2020'); a1.set_ylabel('% in 2020'); a1.set_ylim(0,35); a1.legend()\n"
            "labels = [f'edge\\n{e:.0f}bp/day' for e,*_ in res]; tv = [r[3] for r in res]\n"
            "a2.bar(labels, tv, color=[GREY, GREEN], width=.5); a2.axhline(2, ls='--', c=RED, label='t=2')\n"
            "for i,t in enumerate(tv): a2.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom' if t>=0 else 'top')\n"
            "a2.set_title('Control: 0 edge stays <2, large edge lights up'); a2.set_ylabel('HAC t (10d)'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,k,c,t,p in res: print(f'planted {e:.0f}bp/day: detected={k} harvest={c:+.4f}% t={t:.2f} p={p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: ~{R['stress']['MUB'][0]}% of the worst discounts hit in 2020 — when "
            "the creation/redemption arb froze and the underlying bonds stopped trading, so the "
            "'discount' was partly **stale NAV**, not a bargain you could lift. And the control is clean: "
            f"zero planted edge → **t = {R['syn'][0][3]:.2f}** (no false positive), a 5 bp/day edge → "
            f"**t = {R['syn'][1][3]:.2f}**, p = {R['syn'][1][4]:.3f}. The engine is honest — so the real "
            "tape's marginal, cost-dominated, stress-clustered t is exactly what *no harvestable edge* "
            "looks like."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — pooled hedged harvest **+{R['pool'][0][2]:.4f}%/5d** at HAC "
            f"**t = {R['pool'][0][4]:.2f}**, but **t = {R['pool'][1][4]:.2f}** by 10d, **HYG t = "
            f"{R['per']['HYG'][4]:.2f}** (sign flip), and the *t* **fades with depth**. A stationary "
            "basis's reversion + a marginal, non-robust point estimate ⇒ WEAK, not REAL (and the NAV is "
            "a **proxy**).\n"
            f"- **Tradability `MIRAGE`** — gross ~5 bp is **below** a {R['costs'][1][0]:.0f}-bp/leg "
            f"round-trip (net **{R['costs'][1][2]:+.4f}%**); deepest discounts ~{R['stress']['MUB'][0]}% "
            "in 2020, un-harvestable when the arb breaks. No deployable edge.\n"
            "- **Free discount? `BUSTED`** — below-NAV isn't 'on sale': most of the gap is the basket, "
            "what reverts is pennies, costs erase it, the effect is one-ETF and *weakens* in the deep "
            "tail, and the giant gaps are stale-NAV stress artefacts you can't trade."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the break-even spread\n\n"
            "The operational truth in one picture: how tight would your **all-in round-trip** have to be "
            "for the discount harvest to clear zero? Below the break-even you'd need a spread tighter "
            "than these bond ETFs offer even in calm markets — and far tighter than they offer in the "
            "stress where the big discounts live."
        ),
        code(
            "if HAVE_REAL:\n"
            "    g5 = st.pooled_harvest(FRAMES, 5)['harvest_mean']*100\n"
            "else:\n"
            "    g5 = R['pool'][0][2]\n"
            "legs = 2\n"
            "spreads = np.linspace(0, 4, 100)            # half-spread per leg, bp\n"
            "net = g5 - 2*legs*spreads/100              # round-trip on both legs, %\n"
            "breakeven = g5/(2*legs)*100                # half-spread/leg that zeroes it, bp\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(spreads, net, c=AMBER, lw=2, label='net harvest (5-day)')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.axvline(breakeven, c=RED, ls='--', label=f'break-even ~{breakeven:.1f} bp/leg')\n"
            "ax.axvspan(1, 3, color=GREY, alpha=.2, label='typical HYG/EMB/MUB half-spread (calm)')\n"
            "ax.set_xlabel('half-spread per leg (bp)'); ax.set_ylabel('net harvest (%)')\n"
            "ax.set_title('Break-even spread is below what these ETFs actually quote'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross 5d harvest {g5:.4f}% -> break-even half-spread ~{breakeven:.1f} bp/leg (typical calm spread already 1-3 bp; stress 10-30 bp+)')"
        ),
        md(
            "> 💡 In plain words: the break-even half-spread is on the order of a **single basis "
            "point per leg** — and HYG/EMB/MUB routinely quote *wider* than that even on quiet days, "
            "blowing out to tens of bps in the exact stress where the discounts get large. There is no "
            "spread regime in which this clears: the harvest **is** the spread. Even granting the "
            "reversion is real, the wrapper's own friction eats 100% of it — **the discount is a feature "
            "of the microstructure, not a payout from it.**"
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Real NAV, faster arb.** Replace the hedge-basket proxy with issuer daily NAV (or "
            "intraday iNAV from a paid feed); the basis tightens and the harvest shrinks — the true "
            "creation/redemption arbitrage is faster than our proxy, so our number is an *upper* bound.\n"
            "- **Stale-NAV decomposition.** Regress next-day NAV change on today's discount to quantify "
            "how much of the 'gap' is the NAV catching down to a price the ETF already discovered "
            "(Madhavan-Sobczyk; Pan-Zeng).\n"
            "- **Equity vs credit.** Re-run on SPY/IVV (arb tight, nothing to harvest) vs HY/EM/munis "
            "(arb loose, costs high) — the 'discount edge' lives only where it can't be banked.\n\n"
            "*The reproducible core is offline and deterministic; the NAV input is an explicit proxy. "
            "Methods and sources: [`docs/references.md`](../docs/references.md); frozen numbers: "
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
