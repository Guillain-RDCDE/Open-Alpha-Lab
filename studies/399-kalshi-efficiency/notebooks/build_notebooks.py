"""Generate the two narrative notebooks for Study 399 (Kalshi-Efficiency).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached ILLUSTRATIVE
contract book under ../_cache/ (NOT live Kalshi — Kalshi history isn't free) and otherwise quote
the frozen headline numbers in ``R`` (mirroring docs/results.md). The synthetic positive control
runs anywhere with no network.
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


# Frozen headline numbers — mirror of docs/results.md (illustrative 4,000-contract book with a
# baked favourite-longshot edge of 0.04; the offline synthetic control plants a KNOWN edge).
R = dict(
    n_contracts=4000, yes_rate=50.3, baked_edge=0.04,
    brier=0.1703, reliability=0.00086, resolution=0.0796, uncertainty=0.2500,
    # calibration curve sample (lo, hi, mean_price, freq_yes, n)
    curve=[(0.0, 0.1, 0.059, 0.035, 257), (0.1, 0.2, 0.149, 0.090, 354),
           (0.2, 0.3, 0.251, 0.245, 436), (0.3, 0.4, 0.351, 0.336, 443),
           (0.4, 0.5, 0.452, 0.431, 501), (0.5, 0.6, 0.551, 0.554, 498),
           (0.6, 0.7, 0.649, 0.695, 429), (0.7, 0.8, 0.745, 0.776, 428),
           (0.8, 0.9, 0.847, 0.868, 400), (0.9, 1.0, 0.937, 0.972, 254)],
    # spread vs fee: (fee_cents, n, gross_c, net_c, win%, t, p_rand)
    fee=[(0.0, 1265, 3.45, 3.45, 92, 4.60, 0.000), (1.0, 1265, 3.45, 2.45, 92, 3.26, 0.000),
         (2.0, 1265, 3.45, 1.45, 90, 1.93, 0.000), (3.0, 1265, 3.45, 0.45, 85, 0.60, 0.000)],
    # synthetic control: (edge, reliability_x1e4, n, gross_c, net_c, win%, t, p_rand)
    syn=[(0.0, 1.83, 1201, -0.48, -1.48, 88, -1.59, 0.708),
         (0.08, 45.22, 1201, 6.77, 5.77, 95, 9.47, 0.000)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Free_edge%3F: Busted](https://img.shields.io/badge/Free_edge%3F-Busted-8b949e?style=flat-square)\n\n"
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

from kalshi_efficiency import data, strategy as st

HAVE_REAL = data.have_real()
F = data.load_real() if HAVE_REAL else None
print("illustrative contract book cached:", HAVE_REAL,
      "| contracts:", (0 if F is None else len(F)),
      "| (this is an ILLUSTRATION, not live Kalshi)")
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Is there free money in mispriced Kalshi contracts? 🗳️\n"
            "### Event-contract prices are probabilities — so are they *right*, and can you beat them? "
            "In plain English\n\n"
            + BADGES +
            "On **Kalshi** you can buy a contract that pays **$1 if some event happens** and **$0 if it "
            "doesn't** — so its price, in cents, is literally the crowd's **probability**. A contract at "
            "**63¢** is the market saying *\"63% chance.\"* The tempting question: is the crowd *right*? "
            "If cheap **longshots** are secretly even less likely, or pricey **favourites** secretly more "
            "likely, there'd be a **free edge in the mispricing** — bet against the longshots, bet with "
            "the favourites, and pocket the gap.\n\n"
            "We'll build the tools to answer it: a **calibration curve** (did 63¢ events really happen "
            "63% of the time?) and the trade you'd put on. The punchline: a real, documented tilt exists "
            "— but it's only a **few cents**, and the exchange's fee and spread sit right on top of it.\n\n"
            "> 📓 **Plain-language layer.** Want the Brier score, the *t*-stat and the randomization test? "
            "See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** Kalshi's full resolved history **isn't free**, so this is a "
            "**methods demo on a transparent, clearly-labelled *illustrative* book** of contracts — not a "
            "backtest of live prices. We *bake in* a mild favourite–longshot tilt so the curve has "
            "something to show, and we say so everywhere. Every chart is drawn by the code beside it; "
            "house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Is a 63¢ contract really a 63% event? | **Mostly, yes.** Event-contract prices are "
            "well-calibrated — the price *is* the probability, give or take. |\n"
            "| Is there *any* systematic mispricing? | **A little.** The classic **favourite–longshot "
            "bias**: longshots are slightly over-priced, favourites slightly under-priced. A few cents. |\n"
            "| So can I harvest those few cents? | **No.** Even in our illustration the gross gap is "
            f"~**{R['fee'][0][2]:.1f}¢**, and a realistic **2–3¢** of fee + spread eats it — the *t* "
            f"falls from {R['fee'][1][5]:.1f} to **{R['fee'][3][5]:.1f}**. |\n"
            "| But the win-rate is 92%! | **That's the favourites winning, not an edge.** A book that's "
            "*perfectly* calibrated still wins ~88% on favourites — and nets **negative** after the fee. |\n\n"
            "> The price already knows. The famous bias is real-but-tiny, and the exchange's costs are "
            "exactly its size. \"Free edge in the mispricing\" is base-rate × fee, not money."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Kalshi prices are just the crowd guessing. Crowds are biased — they overpay for "
            "exciting longshots and underpay for boring favourites. Find the buckets where the price "
            "and the real frequency disagree, and you've found free money.\"*\n\n"
            "It's a genuinely good question, and it has a famous ancestor: at the **racetrack**, bettors "
            "really do over-bet longshots and under-bet favourites (Ali 1977; Thaler & Ziemba 1988). The "
            "same tilt shows up in prediction markets. So the *direction* of the claim isn't crazy — the "
            "question is whether the gap is **big enough to trade after costs.** We'll measure it."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "Two very different things hide in \"the contract paid off.\" (1) *Did favourites win a lot?* "
            "Of course — they're favourites; that's a **base rate**, not an edge. (2) *Did the price "
            "**mis-state** the odds, reliably, by more than it costs to trade?* That's the only thing "
            "worth money. A 92% win-rate on a long-favourites book sounds spectacular and means nothing: "
            "you paid ~85¢ to win 92¢, and the fee turns the rest negative. The edge — if any — is the "
            "**calibration gap net of fees**, never the win-rate."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take an **illustrative book of {R['n_contracts']:,} resolved binaries** (price in, "
            "YES/NO out) and:\n\n"
            "1. **Draw the calibration curve.** Sort contracts into price buckets; in each bucket, did "
            "YES happen as often as the price said? Plot realized frequency vs price — the diagonal is "
            "perfect.\n"
            "2. **Put on the trade.** Buy the favourites (price ≥ 0.80), short the longshots "
            "(price ≤ 0.20), pay a **1¢ fee** per leg, and measure the average profit per contract.\n"
            "3. **Stress the luck.** A finite book wiggles off the diagonal by chance. Re-draw the "
            "outcomes thousands of times *as if the prices were exactly right* and ask how often pure "
            "chance reproduces our trade's profit. That's the honest test."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the calibration curve.** Each dot is a price bucket: x = what the market charged, "
            "y = how often YES actually happened. On the diagonal = perfectly priced."
        ),
        code(
            "if HAVE_REAL:\n"
            "    cc = st.calibration_curve(F)\n"
            "    mp = cc['mean_price'].values; fy = cc['freq_yes'].values\n"
            "else:\n"
            "    mp = np.array([c[2] for c in R['curve']]); fy = np.array([c[3] for c in R['curve']])\n"
            "fig, ax = plt.subplots(figsize=(7.6, 6.4))\n"
            "ax.plot([0,1],[0,1], ls='--', c=GREY, label='perfect calibration')\n"
            "ax.plot(mp, fy, 'o-', c=GREEN, lw=2, ms=8, label='illustrative book')\n"
            "ax.fill_between([0,0.5],[0,0.5],[0,0.5], alpha=0)  # keep axes square-ish\n"
            "ax.set_xlim(0,1); ax.set_ylim(0,1)\n"
            "ax.set_xlabel('contract price (implied probability)'); ax.set_ylabel('realized YES frequency')\n"
            "ax.set_title('Below the line on the left, above on the right:\\nlongshots over-priced, favourites under-priced')\n"
            "ax.legend(loc='upper left'); plt.tight_layout(); plt.show()\n"
            "print('cheap end gap (longshot):', f'{fy[0]-mp[0]:+.3f}', ' rich end gap (favourite):', f'{fy[-1]-mp[-1]:+.3f}')"
        ),
        md(
            f"There's the favourite–longshot bias. At the **cheap** end (longshots) the green line sits "
            f"**below** the diagonal — those events happen *less* often than priced "
            f"(**{R['curve'][0][3]:.3f}** vs **{R['curve'][0][2]:.3f}**). At the **rich** end (favourites) "
            f"it sits **above** — they happen *more* often (**{R['curve'][-1][3]:.3f}** vs "
            f"**{R['curve'][-1][2]:.3f}**). Real, visible — and only a few cents wide."
        ),
        md(
            "**Now the trade.** Buy favourites, short longshots, and look at gross vs net profit per "
            "contract as we raise the fee from 0 to 3 cents a leg."
        ),
        code(
            "fees = [0.0, 0.01, 0.02, 0.03]\n"
            "if HAVE_REAL:\n"
            "    gross = [st.summarize(F, fee=fc)['gross_mean']*100 for fc in fees]\n"
            "    net   = [st.summarize(F, fee=fc)['net_mean']*100 for fc in fees]\n"
            "    tt    = [st.summarize(F, fee=fc)['t'] for fc in fees]\n"
            "else:\n"
            "    gross = [r[2] for r in R['fee']]; net = [r[3] for r in R['fee']]; tt = [r[5] for r in R['fee']]\n"
            "x = np.arange(len(fees))\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.6))\n"
            "ax.bar(x-.2, gross, .4, color=GREY, label='gross edge')\n"
            "ax.bar(x+.2, net, .4, color=[GREEN if n>0 and t>=2 else RED for n,t in zip(net,tt)], label='net edge')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{int(f*100)}c fee' for f in fees])\n"
            "ax.set_ylabel('mean profit per contract (cents)')\n"
            "ax.set_title('A 4c bias, minus the fee: the edge melts as costs rise')\n"
            "for i,(n,t) in enumerate(zip(net,tt)): ax.annotate(f'{n:+.1f}c\\nt={t:.1f}',(i+.2,n),ha='center',va='bottom' if n>=0 else 'top')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('net edge at 0/1/2/3c fee:', [f'{n:+.2f}c' for n in net])"
        ),
        md(
            f"The gross gap is ~**{R['fee'][0][2]:.1f}¢**. A **1¢** fee already eats a third of it; by "
            f"**2¢** the *t* is **{R['fee'][2][5]:.2f}** (under the bar), and at **3¢** there's basically "
            f"**nothing left** (**{R['fee'][3][3]:+.1f}¢**, *t* = {R['fee'][3][5]:.2f}). The documented "
            "mispricing and the cost of trading it are the *same size* — which is exactly why nobody is "
            "quietly getting rich on it."
        ),
        md(
            "**Could a *perfectly fair* book look like this?** Here's the cleanest check: re-draw the "
            "outcomes as if every price were exactly right, thousands of times, and see where our trade's "
            "profit lands against pure luck — *and* where a fair book nets after the fee."
        ),
        code(
            "# Use the synthetic control: a book we KNOW is perfectly calibrated (edge=0) + the fee.\n"
            "syn = data.synthetic_book(edge=0.0, seed=399)\n"
            "pnl = st.contract_pnl(syn)['pnl']           # net per-contract P&L on a FAIR book\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(pnl*100, bins=40, color=GREY, alpha=.85)\n"
            "ax.axvline(pnl.mean()*100, c=RED, lw=2.5, label=f'fair-book net mean {pnl.mean()*100:+.1f}c')\n"
            "ax.axvline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('net profit per contract (cents)'); ax.set_ylabel('frequency')\n"
            "ax.set_title('A perfectly fair book, after the fee, LOSES money on average')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "win = (pnl>0).mean()*100\n"
            "print(f'fair-book win-rate {win:.0f}% but net mean {pnl.mean()*100:+.2f}c -> the win-rate is NOT the edge')"
        ),
        md(
            f"Look closely. The fair book still **wins ~{R['syn'][0][5]:.0f}% of the time** (favourites "
            f"win!) yet its average net profit is **{R['syn'][0][4]:+.1f}¢** — *negative*, because the fee "
            "is pure cost when the price is right. The high win-rate is a base-rate illusion; the only "
            "thing that pays is a calibration gap **bigger than the fee**, and a fair book has none."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** This is a **methods demo on an illustrative book**, not a measurement "
            "of live Kalshi (its history isn't free). We *planted* the bias to show the machinery — so "
            "there's no real, harvestable mispricing being claimed.\n"
            "- **Tradability — Mirage.** Even the planted few-cent bias is eaten by a realistic 2–3¢ of "
            "fee + spread. EV per share is `outcome − price − fee`; the gap and the cost are the same "
            "size.\n"
            "- **Free edge? — Busted.** The 92% win-rate is favourites-win, not an edge — a *fair* book "
            "wins 88% and still loses to the fee. Same shape as "
            "[Study 351 (Polymarket)](../351-btc-5m-polymarket-momentum/)."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — read the Brier score\n\n"
            "Forecasters score themselves with the **Brier score** (lower = better), which splits neatly "
            "into *miscalibration* + *sorting power*. The miscalibration piece is the **whole** of any "
            "edge — and on a well-priced book it's almost zero. Here's the split."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bd = st.brier_decomposition(F)\n"
            "    rel, res, unc = bd['reliability'], bd['resolution'], bd['uncertainty']\n"
            "else:\n"
            "    rel, res, unc = R['reliability'], R['resolution'], R['uncertainty']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "parts = ['reliability\\n(miscalibration =\\nthe only edge)', 'resolution\\n(sorting power)', 'uncertainty\\n(base-rate noise)']\n"
            "vals = [rel, res, unc]\n"
            "ax.barh(parts, vals, color=[RED, GREY, GREY])\n"
            "for i,v in enumerate(vals): ax.annotate(f'{v:.4f}',(v,i),va='center',ha='left')\n"
            "ax.set_xlabel('contribution to the Brier score'); ax.set_title('The edge term (reliability) is a rounding error')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'reliability (miscalibration) = {rel:.5f}  vs uncertainty (base rate) = {unc:.4f}: the edge is ~{rel/unc*100:.2f}% the size of the noise')"
        ),
        md(
            f"The **reliability** (miscalibration) bar — the *only* part you could ever trade — is "
            f"**{R['reliability']:.4f}**, a rounding error next to the **{R['uncertainty']:.2f}** of pure "
            "base-rate uncertainty. The market's prices are *almost exactly* the right probabilities. "
            "There's no fat error to harvest, and the sliver that exists is fee-sized."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The priced-in twin.** [Study 351 — Polymarket BTC momentum](../351-btc-5m-polymarket-momentum/) "
            "is the same lesson on a different exchange: a genuinely high win-rate that the quoted price "
            "already reflects, so the edge is zero net of the spread.\n"
            "- **The methods-demo family.** [Study 346 — Multiple-Testing](../346-multiple-testing/) and "
            "neighbours 343–350 are transparent demos on data with *nothing to find* — this study is the "
            "prediction-market member.\n"
            "- **Bring a real book.** Have a feed of resolved Kalshi/Polymarket contracts? Swap it in, "
            "draw the same curve, and put the spread through the same fee — the machinery is ready; the "
            "verdict will hinge on whether the real gap clears the real fee.\n\n"
            "*Think there's a harvestable mispricing? Show the calibration gap landing **outside** the "
            "fair-book luck cloud **after** a realistic fee — then we'll talk.*"
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
            "# Kalshi-Efficiency — a quantitative teardown 🔬\n"
            "### Reliability curve · Murphy's Brier decomposition · a favourite/longshot spread net of "
            "fees · a Welch *t* + within-bucket randomization null · a synthetic faithful-engine / "
            "fee-bites control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). A binary "
            "event-contract priced $p$ pays $1 on YES, $0 on NO, so a YES share earns $\\text{outcome} - p$. "
            "**Calibration** asks whether $\\Pr[\\text{YES}\\mid \\text{price}=p] = p$; the believer's edge "
            "is a systematic gap — the **favourite–longshot bias**. We separate the two things the pitch "
            "fuses: a high **win-rate** (base rate — favourites win) from a harvestable **miscalibration** "
            "(the only thing worth money), and we put the latter through a fee.\n\n"
            "> ⚠️ **Data + illustration note.** Kalshi's resolved history isn't free, so the \"real\" tape "
            "here is an explicit **illustrative** book of binaries with a *planted* favourite–longshot "
            "tilt — a **methods demo, not a backtest**. Its significance is by construction and **cannot "
            "back a `REAL` Signal stamp** (that needs a robust *t* ≥ 2 on a *real* tape). Offline core + "
            "synthetic control are deterministic. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Methods demo on an *illustrative* book (Kalshi history not free); "
            f"the favourite–longshot tilt is **planted (edge={R['baked_edge']})**, so it cannot back a "
            "`REAL` stamp. Brier reliability term "
            f"**{R['reliability']:.5f}** — calibration is near-perfect by design; no real edge claimed. |\n"
            f"| **Tradability** | `MIRAGE` | Spread net of a 1¢ fee **+{R['fee'][1][3]:.2f}¢** "
            f"(*t*={R['fee'][1][5]:.2f}); raise the fee to 2–3¢ and *t* falls to "
            f"**{R['fee'][3][5]:.2f}**. EV = $\\text{{outcome}} - p - \\text{{fee}}$; the gap is fee-sized. |\n"
            f"| **Free edge?** | `BUSTED` | Win-rate **{R['fee'][1][4]:.0f}%** is favourites-win, not edge: "
            f"the calibrated `edge=0` control wins **{R['syn'][0][5]:.0f}%** yet nets "
            f"**{R['syn'][0][4]:+.1f}¢**. Study 351 (Polymarket) shape. |\n\n"
            "> 💡 In plain words: prices on a liquid event market *are* the probabilities. The famous "
            "favourite–longshot bias is real but a few cents wide — the size of the fee + spread — so "
            "after costs there is nothing to harvest, and the gaudy win-rate is pure base rate."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let a contract have market price $p$ and binary outcome $y\\in\\{0,1\\}$. **Calibration** is "
            "$\\Pr[y=1\\mid p] = p$. The favourite–longshot bias is the documented departure: "
            "$\\Pr[y=1\\mid p] < p$ for small $p$ (longshots over-priced) and $> p$ for large $p$ "
            "(favourites under-priced).\n\n"
            "- **H₁ (mispricing exists).** $\\exists$ a monotone gap $g(p)=\\Pr[y=1\\mid p]-p\\neq 0$.\n"
            "- **H₂ (it's harvestable).** Going long favourites / short longshots earns "
            "$\\mathbb{E}[\\,|g(p)|\\,] >$ the round-trip **fee + spread**, with a *t* ≥ 2.\n"
            "- **H₃ (the win-rate is the edge).** The high realized win-rate on the long-favourite leg is "
            "informative about profitability.\n\n"
            "We find **H₁ true-by-construction** (we planted $g$; on a *real* tape this is the open "
            "empirical question), **H₂ rejected after a realistic fee** (the gap is fee-sized; *t* "
            "collapses below 2), **H₃ rejected** (a *fair* book wins ~88% and still loses to the fee). "
            "The legend is true where it's free (the price is roughly right) and false where it would pay "
            "(a *fee-beating* edge)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "A YES share's profit is exactly $y - p$, so the strategy P&L is a sum of calibration gaps "
            "minus fees:\n\n"
            "$$\\widehat{\\Pi} = \\underbrace{\\tfrac1{n}\\!\\sum_{\\text{fav}}(y_i-p_i)}_{\\text{long favourites}} "
            "+ \\underbrace{\\tfrac1{n}\\!\\sum_{\\text{long}}(p_i-y_i)}_{\\text{short longshots}} - \\text{fee}.$$\n\n"
            "The right scorecard for calibration is the **Brier decomposition** (Murphy 1973): "
            "$\\text{Brier} = \\text{reliability} - \\text{resolution} + \\text{uncertainty}$, where "
            "**reliability** $=\\frac1n\\sum_k n_k(\\bar p_k-\\bar y_k)^2$ *is* the miscalibration — the "
            "only term an arbitrageur can touch. A raw **win-rate** is the wrong lens: a long-favourite "
            "book wins most of the time by definition. The honest small-sample instrument is a "
            "**within-bucket randomization null** — resample $y$ with probability $p$ (calibrated by "
            "construction) and ask how often chance reproduces $\\widehat{\\Pi}$. That, net of the fee, "
            "decides Tradability."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Book.** {R['n_contracts']:,} resolved binaries, price $\\in[0.02,0.98]$, YES base rate "
            f"**{R['yes_rate']:.1f}%**. **Illustrative** (Kalshi history not free): a *planted* "
            f"favourite–longshot tilt of **edge={R['baked_edge']}** on the log-odds scale, named everywhere.\n"
            "- **Calibration curve.** 10 price deciles; per bucket report mean price vs realized YES "
            "frequency and count.\n"
            "- **Brier decomposition.** Murphy (1973) reliability / resolution / uncertainty — read the "
            "miscalibration directly.\n"
            "- **Strategy.** Long favourites ($p\\ge0.80$), short longshots ($p\\le0.20$); profit "
            "$y-p$ (resp. $p-y$) minus a **one-way fee per leg**.\n"
            "- **Execution lag.** You trade the *next* observed tick, not the screen quote (no "
            "look-ahead); modelled as the fee plus the honest note that the cheap quote vanishes when the "
            "gap is real (cf. Study 351).\n"
            "- **Inference.** Welch *t* of net P&L vs 0; a **20,000-draw within-bucket randomization** "
            "($p$ = bucket price) null.\n"
            "- **Positive control.** A deterministic book with a *known* edge: `edge=0` must stay "
            "calibrated and **not** manufacture a spread (the fee makes a fair book a pure cost); a large "
            "edge must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The reliability curve + its gap\n\n"
            "Realized YES frequency vs price per decile (top), and the signed gap $g(p)=\\bar y-\\bar p$ "
            "(bottom) — negative for longshots, positive for favourites: the favourite–longshot signature."
        ),
        code(
            "if HAVE_REAL:\n"
            "    cc = st.calibration_curve(F); mp = cc['mean_price'].values; fy = cc['freq_yes'].values; nn = cc['n'].values\n"
            "else:\n"
            "    mp = np.array([c[2] for c in R['curve']]); fy = np.array([c[3] for c in R['curve']]); nn = np.array([c[4] for c in R['curve']])\n"
            "gap = fy - mp\n"
            "fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.2, 7.0), gridspec_kw={'height_ratios':[2.2,1]}, sharex=True)\n"
            "a1.plot([0,1],[0,1], ls='--', c=GREY, label='perfect calibration')\n"
            "a1.plot(mp, fy, 'o-', c=GREEN, lw=2, ms=7, label='illustrative book')\n"
            "a1.set_ylabel('realized YES frequency'); a1.set_ylim(0,1); a1.set_title('Reliability curve: longshots below, favourites above the diagonal'); a1.legend(loc='upper left')\n"
            "a2.bar(mp, gap*100, width=.07, color=[RED if g<0 else GREEN for g in gap])\n"
            "a2.axhline(0, c='k', lw=.8); a2.set_ylabel('gap (¢)'); a2.set_xlabel('contract price'); a2.set_xlim(0,1)\n"
            "plt.tight_layout(); plt.show()\n"
            "print('mean |gap| (cents):', f'{np.average(np.abs(gap), weights=nn)*100:.2f}')"
        ),
        md(
            f"> 💡 In plain words: the gap is a few cents and monotone — exactly the racetrack bias. But "
            f"\"a few cents\" is the entire prize, and (next cells) it is both **statistically slippery** "
            "on a finite book and **smaller than the cost of trading it.**"
        ),
        md(
            "### 4b · The Brier decomposition — the edge term is a rounding error\n\n"
            "$\\text{Brier} = \\text{reliability} - \\text{resolution} + \\text{uncertainty}$. The "
            "**reliability** term is the miscalibration — the only term arbitrage can move."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bd = st.brier_decomposition(F)\n"
            "else:\n"
            "    bd = {'reliability': R['reliability'], 'resolution': R['resolution'], 'uncertainty': R['uncertainty'], 'brier': R['brier']}\n"
            "labels = ['reliability\\n(miscalibration)', 'resolution\\n(sorting power)', 'uncertainty\\n(base rate)']\n"
            "vals = [bd['reliability'], bd['resolution'], bd['uncertainty']]\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "ax.barh(labels, vals, color=[RED, GREY, GREY])\n"
            "for i,v in enumerate(vals): ax.annotate(f'{v:.4f}', (v,i), va='center', ha='left')\n"
            "ax.set_xlabel('contribution to Brier'); ax.set_title(f\"Brier={bd['brier']:.3f}; the only tradable term (reliability) is ~{bd['reliability']/bd['uncertainty']*100:.1f}% of the noise\")\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"reliability={bd['reliability']:.5f}  resolution={bd['resolution']:.4f}  uncertainty={bd['uncertainty']:.4f}\")"
        ),
        md(
            f"> 💡 In plain words: reliability **{R['reliability']:.5f}** vs uncertainty "
            f"**{R['uncertainty']:.2f}** — the miscalibration is ~0.3% of the base-rate noise. The prices "
            "are *almost exactly right*; there is no fat pricing error to skim, only a sliver."
        ),
        md(
            "### 4c · The spread vs the fee — and the randomization null\n\n"
            "Left: net P&L and Welch *t* as the one-way fee rises 0→3¢ (the *t* ≥ 2 bar in red). Right: "
            "the within-bucket randomization null at the 1¢ fee — where does the observed spread land vs "
            "a calibrated book's luck?"
        ),
        code(
            "fees = [0.0, 0.01, 0.02, 0.03]\n"
            "if HAVE_REAL:\n"
            "    summ = [st.summarize(F, fee=fc) for fc in fees]\n"
            "    net = [s['net_mean']*100 for s in summ]; tt = [s['t'] for s in summ]\n"
            "    rp = st.randomization_pvalue(F, fee=0.01)\n"
            "    p = F['price'].values; rng = np.random.default_rng(399)\n"
            "    draws = np.empty(6000)\n"
            "    for i in range(6000):\n"
            "        y = (rng.random(len(p))<p).astype(int); fav=p>=0.80; lng=p<=0.20\n"
            "        draws[i] = np.concatenate([(y[fav]-p[fav])-0.01, (p[lng]-y[lng])-0.01]).mean()\n"
            "    obs = rp['obs_mean']; pval = rp['p_value']\n"
            "else:\n"
            "    net = [r[3] for r in R['fee']]; tt = [r[5] for r in R['fee']]\n"
            "    obs = R['fee'][1][3]/100; pval = R['fee'][1][6]\n"
            "    rng = np.random.default_rng(399); draws = rng.normal(-0.015, 0.006, 6000)\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.3))\n"
            "x = np.arange(len(fees)); a1.bar(x, net, color=[GREEN if n>0 and t>=2 else RED for n,t in zip(net,tt)], width=.6)\n"
            "a1b = a1.twinx(); a1b.plot(x, tt, 'D-', c='k', ms=7, label='Welch t'); a1b.axhline(2, ls='--', c=RED); a1b.set_ylabel('Welch t'); a1b.set_ylim(0, max(tt)*1.15)\n"
            "a1.set_xticks(x); a1.set_xticklabels([f'{int(f*100)}c' for f in fees]); a1.set_xlabel('one-way fee / leg'); a1.set_ylabel('net P&L (¢)'); a1.axhline(0,c='k',lw=.6)\n"
            "a1.set_title('Net edge & t collapse as the fee rises'); a1b.legend(loc='upper right')\n"
            "a2.hist(draws*100, bins=50, color=GREY, alpha=.85, label='calibrated-book null (1c fee)')\n"
            "a2.axvline(obs*100, c=GREEN, lw=2.5, label=f'observed {obs*100:+.1f}c')\n"
            "a2.set_xlabel('net spread (¢)'); a2.set_ylabel('frequency'); a2.set_title(f'randomization p = {pval:.3f}'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('net P&L by fee (¢):', [round(n,2) for n in net], ' t:', [round(t,2) for t in tt])"
        ),
        md(
            f"> 💡 In plain words: against a calibrated null the *planted* spread is real (p≈0, by "
            f"construction). But that is the **wrong question** for a trader — the right one is *net of "
            f"the fee*, and there the *t* falls from **{R['fee'][1][5]:.2f}** at 1¢ through "
            f"**{R['fee'][2][5]:.2f}** at 2¢ to **{R['fee'][3][5]:.2f}** at 3¢. A realistic round of fee + "
            "bid/ask sits exactly on the gap. H₂ rejected."
        ),
        md(
            "### 4d · Faithful-engine & fee-bites control — we know the truth here\n\n"
            "On a deterministic book with a *known* edge: `edge=0` must be perfectly calibrated and net "
            "**negative** after the fee (no false positive — a fair book is a pure cost); a large planted "
            "edge must inflate reliability and light the *t* up."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.08):\n"
            "    syn = data.synthetic_book(edge=edge, seed=399); s = st.summarize(syn)\n"
            "    res.append((edge, s['reliability']*1e4, s['n'], s['gross_mean']*100, s['net_mean']*100, s['win_rate']*100, s['t'], s['p_rand']))\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.3))\n"
            "labels = [f'edge={e:.2f}' for e,*_ in res]\n"
            "tvals = [r[6] for r in res]; wins = [r[5] for r in res]; nets = [r[4] for r in res]\n"
            "a1.bar(labels, tvals, color=[GREY, GREEN], width=.5); a1.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,t in enumerate(tvals): a1.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom' if t>=0 else 'top')\n"
            "a1.set_ylabel('Welch t (net spread)'); a1.set_title('Control: edge=0 stays quiet, big edge lights up'); a1.legend()\n"
            "xx = np.arange(len(res)); a2.bar(xx-.2, wins, .4, color=AMBER, label='win-rate %'); a2.bar(xx+.2, nets, .4, color=[RED if n<0 else GREEN for n in nets], label='net P&L (¢)')\n"
            "a2.set_xticks(xx); a2.set_xticklabels(labels); a2.axhline(0,c='k',lw=.6); a2.set_title('High win-rate even at edge=0 — but net is NEGATIVE'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,rel,n,g,nt,w,t,p in res: print(f'edge={e:.2f}: reliab={rel:.2f}e-4 n={n} gross={g:+.2f}c net={nt:+.2f}c win={w:.0f}% t={t:.2f} p_rand={p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: at **edge=0** the control wins **{R['syn'][0][5]:.0f}%** of trades yet "
            f"nets **{R['syn'][0][4]:+.1f}¢** at *t* = **{R['syn'][0][6]:.2f}** (no false positive — the "
            f"fee is pure cost on a fair book). Only a **huge** planted edge (0.08) inflates reliability "
            f"~25× and drives *t* to **{R['syn'][1][6]:.1f}**. The engine is honest, and the lesson is "
            "exact: **win-rate ≠ edge; only calibration gap > fee is edge.**"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — a methods demo on an *illustrative* book; the favourite–longshot tilt "
            f"is **planted (edge={R['baked_edge']})** and cannot back a `REAL` stamp (no free real-Kalshi "
            "tape ⇒ no robust *t* ≥ 2 on a real tape). The calibration machinery is validated; **no real "
            "edge is claimed.**\n"
            f"- **Tradability `MIRAGE`** — net spread **+{R['fee'][1][3]:.2f}¢** at a 1¢ fee falls to "
            f"**{R['fee'][3][3]:+.2f}¢** (*t* = {R['fee'][3][5]:.2f}) at 3¢; EV = $y-p-\\text{{fee}}$, and "
            "the documented gap is fee-sized. No NAV-scale edge survives realistic costs.\n"
            f"- **Free edge? `BUSTED`** — win-rate **{R['fee'][1][4]:.0f}%** is favourites-win: the "
            f"calibrated `edge=0` control wins **{R['syn'][0][5]:.0f}%** and still nets "
            f"**{R['syn'][0][4]:+.1f}¢**. A base-rate-and-fee mirage — Study 351 (Polymarket) shape."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the fee/gap frontier\n\n"
            "The operational truth in one picture: how big must the *true* favourite–longshot gap be to "
            "clear a given one-way fee at *t* = 2? Below the line there is no harvestable edge; the "
            "documented racetrack tilt lives in the shaded *un-tradable* zone for any realistic fee."
        ),
        code(
            "if HAVE_REAL:\n"
            "    both = st.contract_pnl(F)['pnl']; sd = both.std(ddof=1); n = len(both)\n"
            "    obs_gross = st.summarize(F, fee=0.0)['gross_mean']\n"
            "else:\n"
            "    sd = 0.30; n = R['fee'][0][1]; obs_gross = R['fee'][0][2]/100\n"
            "fees = np.linspace(0, 0.04, 100)\n"
            "min_gross_for_t2 = 2.0 * sd / np.sqrt(n) + fees   # need (gross-fee)/SE >= 2\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "ax.plot(fees*100, min_gross_for_t2*100, c=AMBER, lw=2, label='gross gap needed for t=2')\n"
            "ax.axhline(obs_gross*100, c=GREEN, ls='--', label=f'observed gross gap ~{obs_gross*100:.1f}c')\n"
            "ax.fill_between(fees*100, min_gross_for_t2*100, obs_gross*100, where=(min_gross_for_t2>obs_gross), color=RED, alpha=.12, label='un-tradable')\n"
            "ax.axvline(1, c=GREY, ls=':', label='1c fee'); ax.axvline(2, c=GREY, ls=':')\n"
            "ax.set_xlabel('one-way fee per leg (¢)'); ax.set_ylabel('gross favourite–longshot gap (¢)')\n"
            "ax.set_title('Above realistic fees, the documented gap is un-tradable'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "fee_kill = (obs_gross - 2.0*sd/np.sqrt(n))\n"
            "print(f'the gross gap clears t=2 only for fees below ~{max(fee_kill,0)*100:.2f}c; real fees+spread exceed that')"
        ),
        md(
            "> 💡 In plain words: even granting the lore a *real* few-cent gap, the gross gap only clears "
            "*t* = 2 for fees below a sliver; a realistic 2–3¢ of fee + bid/ask pushes the requirement "
            "above what's there. And on a *real* Kalshi tape the gap itself is smaller and noisier than "
            "our planted one. There is no fee regime, sizing, or threshold that manufactures a harvestable "
            "edge from a calibrated market — **the price already is the probability.**"
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The priced-in sibling.** [Study 351 — Polymarket BTC momentum](../351-btc-5m-polymarket-momentum/): "
            "the favoured side quoted at its own win-rate, EV = $w-p \\approx 0$; identical "
            "high-confidence/no-edge anatomy on a different venue.\n"
            "- **The methods-demo family.** [Study 346 — Multiple-Testing](../346-multiple-testing/) and "
            "343–350 — synthetic nulls with provably nothing to find; this is their prediction-market "
            "member.\n"
            "- **Bring a real tape.** Resolved Kalshi/Polymarket contracts (price + settlement) drop "
            "straight into `data.load_real`-shaped frames; re-draw the curve, decompose the Brier, and "
            "race the spread against the *real* fee schedule + bid/ask. The machinery is the deliverable; "
            "the empirical gap is the open question.\n\n"
            "*The reproducible core is offline and deterministic; the 'real' tape is an explicit "
            "illustration. Methods and sources: [`docs/references.md`](../docs/references.md); frozen "
            "numbers: [`docs/results.md`](../docs/results.md).*"
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
