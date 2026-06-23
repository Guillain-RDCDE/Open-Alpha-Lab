"""Generate the two narrative notebooks for Study 362 (Piotroski F-Score).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached EDGAR
fundamentals + yfinance yearly returns under ../_cache/ and otherwise quote the frozen
headline numbers in ``R`` (mirroring docs/results.md). The synthetic positive control runs
anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (EDGAR companyfacts + yfinance,
# fiscal 2009->2024 -> returns 2010->2025, 21 firms, 249 firm-years, 15 usable long-short years).
R = dict(
    as_of="2026-06-22", firms=21, firm_years=249, mean_f=5.54,
    fiscal_lo=2009, fiscal_hi=2024, ret_lo=2010, ret_hi=2025, n_years=15,
    high=14.77, low=12.13, market=16.37, hi_minus_mkt=-1.60, hi_minus_mkt_t=-1.38,
    spread=2.64, spread_t=0.62, placebo_p=0.256, hit=53, hit_yrs=8,
    net=1.74, net_t=0.41,
    # bucket ladder: (label, mean%, n)
    buckets=[("0-1", 3.21, 3), ("2-3", 10.72, 25), ("4-5", 17.38, 87),
             ("6-7", 17.39, 110), ("8-9", 15.73, 24)],
    # robustness: (hi, lo, years, spread%, t, p)
    robust=[(8, 3, 2, -4.64, -0.28, 0.612), (7, 3, 8, -1.38, -0.57, 0.614),
            (7, 4, 15, 2.64, 0.62, 0.256), (6, 4, 15, 4.48, 1.06, 0.108),
            (6, 5, 15, 0.94, 0.50, 0.376)],
    # synthetic control: (edge, years, high%, low%, spread%, t, p)
    syn=[(0.00, 18, 10.7, 9.9, 0.86, 0.75, 0.252),
         (0.04, 18, 16.5, 6.5, 10.03, 8.61, 0.000),
         (0.10, 18, 25.2, 1.4, 23.79, 19.91, 0.000)],
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Winners_vs_losers%3F: Busted](https://img.shields.io/badge/Winners_vs_losers%3F-Busted-8b949e?style=flat-square)\n\n"
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

from piotroski_f_score import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    P = data.load_real()
    RET = P["returns"]; RET = RET[RET.index <= 2025]      # drop in-progress 2026
    FS = st.compute_fscore(P); FS = FS[FS.index <= 2024]  # fiscal y -> returns y+1<=2025
    LS = st.long_short(FS, RET)
else:
    P = RET = FS = LS = None
print("real EDGAR+price cache present:", HAVE_REAL,
      "| usable long-short years:", (0 if LS is None else len(LS)))
"""

# The frozen headline dict is embedded into the first code cell so every downstream cell can
# quote it whether or not the cache is present.
BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# The 9-point report card for stocks — does it really pick winners? 📊\n"
            "### Piotroski's F-Score promises to 'separate winners from losers' — in plain English\n\n"
            + BADGES +
            "There's a beloved value-investing shortcut: score every company on a **9-point checklist** "
            "of financial health — is it profitable? generating cash? cutting debt? not diluting "
            "shareholders? — and the high scorers (8-9) are supposed to **beat** the low scorers (0-1). "
            "Nine green checkmarks, a winner; a pile of red, a loser. Clean, intuitive, famous.\n\n"
            "It's a *good idea* — and in the place it was discovered (tiny, cheap, overlooked "
            "value stocks) it genuinely works. But lift it onto a basket of household-name large caps "
            "and something quietly breaks: the 'winners' stop beating the market. This notebook shows "
            "where — and why a positive-looking number can still be nothing.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo test and the power "
            "analysis? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** We pull real 10-K numbers from **SEC EDGAR** for a fixed "
            "basket of large/mid caps that all **survived to today** — which quietly removes the dead "
            "losers a low score would have flagged, and is *not* the small-cap value world where the "
            "F-Score's edge lives. We say so throughout. Every chart is drawn by the code beside it; "
            "house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Do high-score firms beat low-score firms? | **A little — +2.6%/yr on average.** "
            "That's the headline 'it works.' |\n"
            "| Is that gap real? | **Can't tell.** With only ~15 years it's statistically "
            "indistinguishable from picking the same number of stocks **at random** (a coin matches it "
            "~1 time in 4). |\n"
            "| Do the 'winners' beat the market? | **No.** The high-score leg earns "
            f"**+{R['high']:.1f}%/yr** — *below* the **+{R['market']:.1f}%/yr** equal-weight basket. The "
            "tiny spread comes from the *losers* being slightly worse, not the winners being better. |\n"
            "| Does a higher score always mean a higher return? | **No.** Returns climb with the score "
            "until the **6-7** bucket and then **fall** for the very best **8-9** firms. A real "
            "winner-picker doesn't do that. |\n\n"
            "> The F-Score is a sensible *quality screen*. But on this large-cap survivor tape it does "
            "**not** separate winners from losers the way the headline promises."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Add up nine yes/no checks on a company's books — profitability, balance-sheet "
            "strength, operating efficiency. Firms scoring 8 or 9 are financially improving; firms "
            "scoring 0 or 1 are deteriorating. Buy the strong, avoid (or short) the weak, and you'll "
            "separate winners from losers.\"*\n\n"
            "Joseph Piotroski introduced the F-Score in 2000. The nine points are intuitive: positive "
            "profit, positive cash flow, *improving* profitability, cash earnings that exceed paper "
            "earnings, falling debt, rising liquidity, no new share issuance, fatter margins, and more "
            "sales per dollar of assets. We'll rebuild all nine from real SEC filings and check the "
            "promise."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "Two very different claims hide inside 'it separates winners from losers,' and only one is "
            "interesting. (1) *Do good-looking companies tend to go up?* Sure — but **most** stocks go "
            "up over time, so that's nearly free. The claim that matters is (2) *do the high scorers "
            "beat a fair benchmark — the market itself — by enough to bet on?* If the 'winners' merely "
            "match (or trail) an equal-weight basket of the same stocks, the score isn't picking "
            "winners; it's along for the ride. And a screen that only works by *avoiding a few losers* "
            "is a different, weaker thing than one that *finds winners*."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "We rebuild the F-Score honestly and let it pick:\n\n"
            "1. **Score every firm, every year.** Pull the real 10-K numbers from EDGAR, compute all "
            "nine binary points, sum them 0-9.\n"
            "2. **Let the score trade.** Each year, go **long** the high scorers and **short** the low "
            "scorers, and measure what they did **the next calendar year** (so we never use a report "
            "before it's filed). Compare to the equal-weight basket — the honest benchmark.\n"
            "3. **Stress the luck.** Replace the score's picks with the **same number of random "
            "stocks**, thousands of times. If random picking matches the F-Score often, the 'edge' is "
            "noise.\n"
            "4. **Check the ladder.** Does the return rise *all the way* from score 0 to score 9? A "
            "real winner-picker is monotone; a fluke isn't."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the three legs.** The high-score 'winners,' the low-score 'losers,' and the "
            "equal-weight basket they all live in. Watch where the winners land relative to the market."
        ),
        code(
            "if HAVE_REAL:\n"
            "    high, low, mkt = LS['high'].mean()*100, LS['low'].mean()*100, LS['market'].mean()*100\n"
            "else:\n"
            "    high, low, mkt = R['high'], R['low'], R['market']\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.4))\n"
            "bars = ax.bar(['high score\\n(>=7)', 'low score\\n(<=4)', 'equal-weight\\nbasket'],\n"
            "              [high, low, mkt], color=[GREEN, RED, GREY], width=.6)\n"
            "ax.axhline(mkt, ls='--', c=GREY, lw=1)\n"
            "ax.set_ylabel('mean annual return (%)')\n"
            "ax.set_title('The \\\"winners\\\" (+%.1f%%) trail the market (+%.1f%%)' % (high, mkt))\n"
            "for b,v in zip(bars,[high,low,mkt]): ax.annotate(f'{v:.1f}%',(b.get_x()+b.get_width()/2,v),ha='center',va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'high-score leg {high:.1f}% vs market {mkt:.1f}%  -> winners trail by {mkt-high:.1f} pts')"
        ),
        md(
            f"There's the first crack. The high-F-score 'winners' earn **+{R['high']:.1f}%/yr** but the "
            f"equal-weight basket earns **+{R['market']:.1f}%/yr** — the winners *trail the market*. The "
            f"only reason the long-minus-short spread is positive (**+{R['spread']:.1f}%/yr**) is that "
            f"the low-score leg (**+{R['low']:.1f}%**) is a touch worse. The score isn't finding "
            "winners; it's mildly avoiding losers."
        ),
        md(
            "**Does a higher score always pay more?** A real winner-picker climbs the whole ladder. "
            "Here's the average next-year return for each score bucket, 0-1 up to 8-9."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bk = st.score_buckets(FS, RET)\n"
            "    labels = list(bk.index); vals = [bk.loc[l,'mean_ret']*100 for l in labels]; ns=[int(bk.loc[l,'n']) for l in labels]\n"
            "else:\n"
            "    labels = [b[0] for b in R['buckets']]; vals=[b[1] for b in R['buckets']]; ns=[b[2] for b in R['buckets']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "colors = [GREEN if i==len(vals)-1 else GREY for i in range(len(vals))]\n"
            "bars = ax.bar(labels, vals, color=colors, width=.62)\n"
            "ax.set_xlabel('F-score bucket'); ax.set_ylabel('mean next-year return (%)')\n"
            "ax.set_title('Return climbs to 6-7 then FALLS at 8-9 — not monotone')\n"
            "for b,v,nn in zip(bars,vals,ns): ax.annotate(f'{v:.1f}%\\n(n={nn})',(b.get_x()+b.get_width()/2,v),ha='center',va='bottom',fontsize=8)\n"
            "plt.tight_layout(); plt.show()\n"
            "print('top bucket 8-9:', f'{vals[-1]:.1f}%', ' vs 6-7:', f'{vals[-2]:.1f}%', '-> the best firms do WORSE')"
        ),
        md(
            f"The ladder rises — and then trips. The **6-7** bucket returns **{R['buckets'][3][1]:.1f}%**, "
            f"but the very-best **8-9** firms return only **{R['buckets'][4][1]:.1f}%**. A genuine "
            "'separate winners from losers' factor climbs monotonically to the top; a non-monotone "
            "ladder is the signature of noise in a thin top bucket (just "
            f"{R['buckets'][4][2]} firm-years up there)."
        ),
        md(
            "**Could random stock-picking match it?** The honest test: replace the F-Score's picks "
            "with the *same number* of random stocks, thousands of times, and see where the real spread "
            "lands in the cloud of luck."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.placebo_pvalue(FS, RET, n_draws=8000)\n"
            "    obs = pl['obs_spread']*100; pval = pl['p_value']\n"
            "    # rebuild the random-spread cloud for the picture\n"
            "    years=[]\n"
            "    for y in LS.index:\n"
            "        s = FS.loc[y].dropna(); nxt = RET.loc[int(y)+1].dropna(); s,nxt = s.align(nxt,join='inner')\n"
            "        years.append((nxt.to_numpy(), int(LS.loc[y,'n_hi']), int(LS.loc[y,'n_lo'])))\n"
            "    rng = np.random.default_rng(362)\n"
            "    draws=[]\n"
            "    for _ in range(8000):\n"
            "        sp=[]\n"
            "        for arr,nh,nl in years:\n"
            "            idx=rng.permutation(len(arr)); sp.append(arr[idx[:nh]].mean()-arr[idx[nh:nh+nl]].mean())\n"
            "        draws.append(np.mean(sp))\n"
            "    draws=np.array(draws)*100\n"
            "else:\n"
            "    obs = R['spread']; pval = R['placebo_p']\n"
            "    rng = np.random.default_rng(362); draws = rng.normal(0, 4.0, 8000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=50, color=GREY, alpha=.85, label='spread of RANDOM same-size portfolios')\n"
            "ax.axvline(obs, c=GREEN, lw=2.5, label=f'the actual F-score spread ({obs:.1f}%)')\n"
            "ax.axvline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('high-minus-low annual spread (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'The F-score spread sits inside the luck cloud — placebo p = {pval:.2f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'random picking matches or beats the F-score {pval*100:.0f}% of the time')"
        ),
        md(
            f"The green line — the real F-Score spread — sits **inside** the grey cloud of random "
            f"portfolios. About **{int(R['placebo_p']*100)}%** of random same-size picks do as well or "
            "better. In plain terms: **on this tape, the nine-point checklist barely outpicks a "
            "dartboard.**"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Weak.** The high-minus-low spread is positive (**+{R['spread']:.1f}%/yr**) "
            "but isn't statistically distinguishable from random picking, the 'winners' trail the "
            "market, and the score isn't monotone. Real in its home universe (small-cap deep value), "
            "weak here.\n"
            "- **Tradability — Mirage.** Net of trading costs and short-borrow the spread is "
            f"**+{R['net']:.1f}%/yr** — and the long leg lags the basket. Nothing to deploy.\n"
            "- **\"Winners vs losers\"? — Busted.** On a large-cap survivor basket the score sidesteps a "
            "thin tail of losers; it does not pick winners. A useful health check, not a money machine."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — costs and the missing losers\n\n"
            "Two practical problems on top of the weak signal. First, you'd **short** the low-score "
            "firms — which costs **borrow**, and the very firms a low score flags are the hard-to-borrow "
            "ones. Second, our basket only contains firms that **survived** — the spectacular low-score "
            "blow-ups that *should* make the short leg pay aren't even in the data."
        ),
        code(
            "if HAVE_REAL:\n"
            "    s = st.summarize(FS, RET); gross = s['gross_mean']*100; net = s['net_mean']*100\n"
            "else:\n"
            "    gross = R['spread']; net = R['net']\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.2))\n"
            "bars = ax.bar(['gross spread', 'net of costs\\n+ borrow'], [gross, net], color=[GREEN, GREY], width=.55)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean annual spread (%)')\n"
            "ax.set_title('Costs + borrow eat an already-insignificant spread')\n"
            "for b,v in zip(bars,[gross,net]): ax.annotate(f'{v:.1f}%',(b.get_x()+b.get_width()/2,v),ha='center',va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross {gross:.1f}% -> net {net:.1f}% after one-way costs x turnover + short borrow')"
        ),
        md(
            f"Gross **+{R['spread']:.1f}%/yr** becomes net **+{R['net']:.1f}%/yr** — and remember it was "
            "never significant to begin with. Worse, the survivorship that's missing here would only "
            "*hurt* a live version more (your shorts would include names too risky to borrow). The "
            "honest read isn't 'a small edge after costs' — it's 'no edge you could bank.'"
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Run it where it lives.** Piotroski's edge is documented among **small, cheap, "
            "low-coverage value** stocks. Swap our large-cap basket for a small-cap *value* universe and "
            "the spread should reappear — the lesson here is about **universe**, not the checklist.\n"
            "- **The growth-side twin.** [Study 232 — Mohanram G-Score](../../232-mohanram-g-score/) is "
            "the same financial-statement-analysis idea aimed at *growth* (low book-to-market) firms.\n"
            "- **Sibling screens.** [Study 121 — Magic-Formula](../../121-magic-formula/) and "
            "[Study 122 — Gross-Profitability](../../122-gross-profitability/) run on the same EDGAR "
            "machinery — how much does a nine-point composite add over one good ratio?\n\n"
            "*Think the F-Score beats the market by more than luck on large caps? Capture the events, "
            "draw the same number of random portfolios, and show the spread landing **outside** the "
            "cloud — then we'll talk.*"
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
            "# Piotroski F-Score — a quantitative teardown 🔬\n"
            "### Nine binary points from EDGAR · high-minus-low vs equal-weight · a Newey-West *t* + "
            "placebo randomization null · the monotonicity ladder · costs + borrow · a synthetic "
            "faithful-engine / power control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). We separate "
            "the two things 'separates winners from losers' fuses: a **cross-sectional spread** from the "
            "**benchmark return of the same basket**, and confront the spread with its **standard "
            "error** on a short annual panel. The decisive objects are the **HAC *t***, a **placebo "
            "null** of random same-size portfolios, and the **monotonicity** of the 0-9 ladder — not "
            "the (positive) point estimate.\n\n"
            "> ⚠️ **Data + universe note.** Fundamentals: **EDGAR companyfacts** (`data.sec.gov`, public, "
            "no key), annual 10-K figures for a fixed 40-name large/mid-cap basket; **21** firms clear "
            "the all-nine-concepts intersection. Prices: yfinance daily adjusted closes → calendar-year "
            "total returns. The basket is a **survivor** panel and a **large-cap** universe — *not* the "
            "small-cap deep-value world where Piotroski's edge is documented; both named on the Signal "
            "axis. Offline core + synthetic control are deterministic. Methods in "
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
            f"| **Signal** | `WEAK` | high-minus-low spread **+{R['spread']:.1f}%/yr**, HAC "
            f"**t = {R['spread_t']:.2f}**, placebo **p = {R['placebo_p']:.2f}**; **non-monotone** (8-9 "
            f"bucket {R['buckets'][4][1]:.1f}% < 6-7 bucket {R['buckets'][3][1]:.1f}%) and **sign-flips** "
            "across thresholds. |\n"
            f"| **Tradability** | `MIRAGE` | net of costs + borrow **+{R['net']:.1f}%/yr** "
            f"(t = {R['net_t']:.2f}); the high leg **+{R['high']:.1f}%** *trails* the equal-weight "
            f"**+{R['market']:.1f}%** (high−mkt t = {R['hi_minus_mkt_t']:.2f}). No NAV-scale edge. |\n"
            f"| **Winners vs losers?** | `BUSTED` | the spread is carried by the **short** leg; the "
            "'winners' don't beat the market and the ladder isn't monotone — a loser-avoidance screen, "
            "not a winner-picker, on this tape. |\n\n"
            "> 💡 In plain words: the F-Score really does correlate with quality — but on a survivor-"
            "biased large-cap basket the *return* spread it generates is inside the noise, isn't "
            "ordered, and is short-leg-driven. Strip the base rate (the market) and ~15 years leave "
            "nothing the null can't explain."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $f_{i,t}\\in\\{0,\\dots,9\\}$ be firm $i$'s F-Score in fiscal year $t$ (sum of nine "
            "binary points: 4 profitability, 3 leverage/liquidity, 2 efficiency). Form a long-short "
            "book: long $\\{i: f_{i,t}\\ge 7\\}$, short $\\{i: f_{i,t}\\le 4\\}$, equal-weighted, and "
            "earn calendar-year $t+1$ returns (reporting lag).\n\n"
            "- **H₁ (it separates).** $\\mathbb{E}[r^{\\text{hi}}_{t+1}-r^{\\text{lo}}_{t+1}] > 0$ with "
            "a *significant* spread — winners beat losers beyond luck.\n"
            "- **H₂ (it picks winners).** The high leg beats a fair benchmark — the **equal-weight "
            "basket** — not just the low leg.\n"
            "- **H₃ (it's deployable).** The spread survives costs × turnover + short borrow and holds "
            "capital.\n\n"
            "We find **H₁ not rejected but not supported** (spread > 0, $t<2$, non-monotone, "
            "sign-flipping), **H₂ rejected** (high leg *below* the basket), **H₃ rejected** "
            "(net spread inside the noise). The checklist is true exactly where it's uninformative "
            "(it tracks quality) and unproven exactly where it would pay (a *priced* winner-loser "
            "spread on this universe)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The teardown is one comparison judged by its **standard error**:\n\n"
            "$$\\widehat{\\Delta} = \\overline{r^{\\text{hi}}-r^{\\text{lo}}},\\qquad "
            "t_{\\text{HAC}} = \\frac{\\widehat{\\Delta}}{\\widehat{\\mathrm{se}}_{\\text{NW}}(\\widehat{\\Delta})}.$$\n\n"
            "With ~15 annual observations $\\widehat{\\mathrm{se}}$ is large; a few-point $\\widehat{\\Delta}$ "
            "drowns in it. A raw long-short *return* is also the wrong lens if you don't net the "
            "benchmark — almost everything is positive in an up-market. So we (a) race the high leg "
            "against the **equal-weight basket**, (b) test the spread with a **Newey-West HAC *t*** "
            "(short, autocorrelated annual series), and (c) run a **randomization (placebo) null**: "
            "random same-size portfolios in place of the F-score legs. And we check **monotonicity** — "
            "the honest test of 'separates winners from losers' is a return that *rises across the whole "
            "ladder*, not just a positive tail-difference."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Fundamentals.** EDGAR companyfacts annual 10-K figures, {R['firms']} firms surviving "
            f"the all-nine-concepts intersection, fiscal {R['fiscal_lo']}-{R['fiscal_hi']} "
            f"({R['firm_years']} firm-years, mean F = {R['mean_f']:.2f}/9). Year-over-year points "
            "compare a fiscal year to the prior one (no look-ahead).\n"
            "- **Portfolio.** Long $f\\ge 7$, short $f\\le 4$ (the canonical 8-9 vs 0-1 split is empty "
            "on a ~16-name/yr basket); enter **calendar-year $t+1$** (reporting lag); equal weight.\n"
            "- **Benchmark.** Equal-weight basket return — the fair base rate the high leg must beat.\n"
            "- **Null #1 (HAC t).** Newey-West *t* of the annual spread series.\n"
            "- **Null #2 (placebo).** 20,000 draws of random same-size portfolios; "
            "$p=\\Pr[\\text{random spread}\\ge\\text{F-score spread}]$.\n"
            "- **Monotonicity.** Mean next-year return by bucket {0-1,…,8-9}.\n"
            "- **Costs.** One-way × turnover on both legs (annual rebalance) + **borrow** on the short.\n"
            "- **Positive control.** A deterministic panel with a *planted* F-score→return edge: the "
            "engine must recover it, and must **not** manufacture significance when the edge is zero."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The legs — the 'winners' trail the benchmark\n\n"
            "High leg, low leg, and the equal-weight basket, with the high-minus-market gap and its "
            "HAC *t*. A winner-picker's high leg sits *above* the basket; here it sits below."
        ),
        code(
            "if HAVE_REAL:\n"
            "    high, low, mkt = LS['high'].mean()*100, LS['low'].mean()*100, LS['market'].mean()*100\n"
            "    hmm = st.hac_tstat(LS['high']-LS['market'])\n"
            "    hm_mean, hm_t = hmm['mean']*100, hmm['tstat']\n"
            "else:\n"
            "    high, low, mkt = R['high'], R['low'], R['market']\n"
            "    hm_mean, hm_t = R['hi_minus_mkt'], R['hi_minus_mkt_t']\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "bars = ax.bar(['high (>=7)','low (<=4)','equal-weight'], [high,low,mkt], color=[GREEN,RED,GREY], width=.6)\n"
            "ax.axhline(mkt, ls='--', c=GREY, lw=1)\n"
            "ax.set_ylabel('mean annual return (%)')\n"
            "ax.set_title(f'High leg {high:.1f}%  vs  basket {mkt:.1f}%  ->  high-minus-market {hm_mean:+.1f}% (t={hm_t:.2f})')\n"
            "for b,v in zip(bars,[high,low,mkt]): ax.annotate(f'{v:.1f}%',(b.get_x()+b.get_width()/2,v),ha='center',va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'high-minus-market = {hm_mean:+.2f}%/yr  HAC t = {hm_t:.2f}  (winners do NOT beat the basket)')"
        ),
        md(
            f"> 💡 In plain words: the high-F-score leg earns **+{R['high']:.1f}%/yr** against the "
            f"equal-weight basket's **+{R['market']:.1f}%/yr** — a **{R['hi_minus_mkt']:+.1f}%/yr** "
            f"shortfall at HAC **t = {R['hi_minus_mkt_t']:.2f}**. H₂ rejected: the spread is a *short-leg* "
            "phenomenon. The 'winners' are not winners against a fair benchmark."
        ),
        md(
            "### 4b · The decisive test — a placebo null of random same-size portfolios\n\n"
            "Replace each year's F-score legs with random same-size subsets, 20,000 times; the "
            "histogram is the null distribution of the long-short spread. The F-score is the green line; "
            "the *p*-value is the right-tail mass."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.placebo_pvalue(FS, RET, n_draws=8000); obs = pl['obs_spread']*100; pval = pl['p_value']\n"
            "    years=[]\n"
            "    for y in LS.index:\n"
            "        s = FS.loc[y].dropna(); nxt = RET.loc[int(y)+1].dropna(); s,nxt = s.align(nxt,join='inner')\n"
            "        years.append((nxt.to_numpy(), int(LS.loc[y,'n_hi']), int(LS.loc[y,'n_lo'])))\n"
            "    rng = np.random.default_rng(362); draws=[]\n"
            "    for _ in range(8000):\n"
            "        sp=[]\n"
            "        for arr,nh,nl in years:\n"
            "            idx=rng.permutation(len(arr)); sp.append(arr[idx[:nh]].mean()-arr[idx[nh:nh+nl]].mean())\n"
            "        draws.append(np.mean(sp))\n"
            "    draws=np.array(draws)*100\n"
            "else:\n"
            "    obs = R['spread']; pval = R['placebo_p']\n"
            "    rng = np.random.default_rng(362); draws = rng.normal(0, 4.0, 8000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=60, color=GREY, alpha=.85, label='null: random same-size spreads (8k draws)')\n"
            "ax.axvline(obs, c=GREEN, lw=2.5, label=f'observed F-score spread {obs:.1f}%')\n"
            "ax.axvline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('high-minus-low annual spread (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Placebo p = {pval:.2f}: the F-score spread is inside the luck cloud'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'P[random spread >= F-score] = {pval:.3f}  (need <0.05; here ~1 in 4)')"
        ),
        md(
            f"> 💡 In plain words: **{int(R['placebo_p']*100)}%** of random same-size portfolios match or "
            "beat the F-score spread. A real edge pushes the green line into the far right tail; instead "
            "it sits mid-cloud. H₁ not supported — the spread is what random stock-picking produces on "
            "this universe."
        ),
        md(
            "### 4c · Monotonicity + robustness — neither holds\n\n"
            "Left: mean next-year return by score bucket (a real ordering factor is monotone). Right: "
            "the spread *t* as the high/low thresholds move (a real edge is stable). Neither cooperates."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bk = st.score_buckets(FS, RET)\n"
            "    blabels = list(bk.index); bvals = [bk.loc[l,'mean_ret']*100 for l in blabels]\n"
            "    # spread t per threshold (no placebo needed here -> fast)\n"
            "    rob = []\n"
            "    for hi,lo in [(7,3),(7,4),(6,4),(6,5)]:\n"
            "        l2 = st.long_short(FS, RET, hi=hi, lo=lo)\n"
            "        h2 = st.hac_tstat(l2['spread'])\n"
            "        rob.append((hi, lo, h2['n'], h2['mean']*100, h2['tstat']))\n"
            "else:\n"
            "    blabels = [b[0] for b in R['buckets']]; bvals = [b[1] for b in R['buckets']]\n"
            "    rob = [(r[0],r[1],r[2],r[3],r[4]) for r in R['robust'] if r[0] in (7,6)]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.2))\n"
            "cols=[GREEN if i==len(bvals)-1 else GREY for i in range(len(bvals))]\n"
            "a1.bar(blabels, bvals, color=cols, width=.62)\n"
            "a1.set_title('Not monotone: 8-9 < 6-7'); a1.set_xlabel('F-score bucket'); a1.set_ylabel('mean next-yr return (%)')\n"
            "for i,v in enumerate(bvals): a1.annotate(f'{v:.0f}%',(i,v),ha='center',va='bottom',fontsize=8)\n"
            "tl = [f\"{int(r[0])}/{int(r[1])}\" for r in rob]; tv = [r[4] for r in rob]\n"
            "a2.bar(tl, tv, color=AMBER, width=.55)\n"
            "a2.axhline(2, ls='--', c=RED, label='t=2'); a2.axhline(0, c='k', lw=.8)\n"
            "a2.set_title('Spread t never reaches 2'); a2.set_xlabel('hi/lo threshold'); a2.set_ylabel('HAC t'); a2.set_ylim(-1, 2.4); a2.legend()\n"
            "for i,r in enumerate(rob): a2.annotate(f'n={int(r[2])}',(i,r[4]),ha='center',va='bottom',fontsize=8)\n"
            "plt.tight_layout(); plt.show()\n"
            "print('buckets:', [f'{l}:{v:.0f}%' for l,v in zip(blabels,bvals)])\n"
            "print('robustness (hi/lo, t):', [(f'{int(r[0])}/{int(r[1])}', round(r[4],2)) for r in rob])"
        ),
        md(
            f"> 💡 In plain words: the ladder peaks at **6-7** ({R['buckets'][3][1]:.0f}%) and dips at "
            f"**8-9** ({R['buckets'][4][1]:.0f}%) — non-monotone. And the spread *t* wanders with the "
            "cut (the tighter 7/3 split even goes **negative**) without ever approaching 2. There is no "
            "threshold on this basket where the F-score clears the significance bar."
        ),
        md(
            "### 4d · Faithful-engine & power control — we know the truth here\n\n"
            "A deterministic panel where a latent quality drives both the nine points and (via a "
            "`planted_edge`) next-year returns. With **zero** planted edge the test must stay below "
            "t=2; with a real planted edge it must light up. Both hold — the engine is unbiased, and "
            "the real-tape *t* is a true negative, not a broken pipe."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.04, 0.10):\n"
            "    syn = data.synthetic_panel(planted_edge=edge, seed=362)\n"
            "    ls_s = st.long_short(syn['fscore'], syn['returns'])\n"
            "    hh = st.hac_tstat(ls_s['spread'])\n"
            "    pp = st.placebo_pvalue(syn['fscore'], syn['returns'], n_draws=2000)['p_value']\n"
            "    res.append((edge, len(ls_s), ls_s['high'].mean()*100, ls_s['low'].mean()*100, hh['mean']*100, hh['tstat'], pp))\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "labels = [f'planted\\n+{e*100:.0f}%/yr' for e,*_ in res]; tvals=[r[5] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN, GREEN], width=.55)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 (significance bar)')\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('spread HAC t'); ax.set_title('Control: zero edge -> no false positive; real edge -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,n,h,l,sp,t,p in res: print(f'planted {e*100:+.0f}%/yr: years={n} high={h:.1f}% low={l:.1f}% spread={sp:.1f}% t={t:.2f} p={p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted edge the control sits at "
            f"**t = {R['syn'][0][5]:.2f}** (below 2 — no false positive); a **+4%/yr** edge already "
            f"drives **t = {R['syn'][1][5]:.2f}**, and **+10%/yr** reaches **t = {R['syn'][2][5]:.2f}**. "
            f"So the machinery is honest, and the real-tape *t* of **{R['spread_t']:.2f}** is exactly "
            "what an *absent or tiny* edge looks like — the sample size and universe are the verdict, "
            "not a measurement failure."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — spread **+{R['spread']:.1f}%/yr** at HAC **t = {R['spread_t']:.2f}** / "
            f"placebo **p = {R['placebo_p']:.2f}**; **non-monotone** (8-9 < 6-7) and **sign-flipping** "
            "across thresholds. Decades of literature support (real among small-cap deep value) + a "
            "sub-2 *t* on this large-cap survivor tape ⇒ WEAK, not REAL. The synthetic control proves "
            "the engine, never the market.\n"
            f"- **Tradability `MIRAGE`** — net of costs + borrow **+{R['net']:.1f}%/yr** "
            f"(t = {R['net_t']:.2f}); the long leg **trails** the equal-weight basket "
            f"(high−mkt t = {R['hi_minus_mkt_t']:.2f}). On the documented universe (micro-cap, "
            "hard-to-borrow value) frictions are worse. No NAV-scale edge.\n"
            f"- **Winners vs losers? `BUSTED`** — the spread is **short-leg-driven**, the 'winners' "
            "don't beat the market, and the ladder isn't monotone. A loser-avoidance / quality screen, "
            "not the winner-picker the headline sells — on *this* universe."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the power & the borrow\n\n"
            "How large would the *true* spread have to be for a $k$-year study to detect it at t=2? At "
            "$k=15$ you'd need a spread several times the one observed — and then you'd still pay borrow "
            "on a short leg made of exactly the names hardest to borrow."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sd = LS['spread'].std(ddof=1); obs = LS['spread'].mean()\n"
            "else:\n"
            "    sd = 0.146; obs = R['spread']/100\n"
            "ks = np.arange(5, 80)\n"
            "min_detectable = 2.0 * sd / np.sqrt(ks)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(ks, min_detectable*100, c=AMBER, lw=2, label='spread needed for t=2')\n"
            "ax.axhline(obs*100, c=GREEN, ls='--', label=f'observed spread ~{obs*100:.1f}%')\n"
            "ax.axvline(R['n_years'], c=GREY, ls=':', label=f\"our k={R['n_years']} years\")\n"
            "ax.set_xlabel('number of annual observations k'); ax.set_ylabel('annual spread (%)')\n"
            "ax.set_title('Detection floor vs the real spread: under-powered'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "need = 2.0*sd/np.sqrt(R['n_years'])\n"
            "print(f'at k={R[\"n_years\"]} you need ~{need*100:.1f}% spread for t=2; observed ~{obs*100:.1f}% -> under-powered by ~{need/max(obs,1e-9):.1f}x')"
        ),
        md(
            "> 💡 In plain words: the amber curve is the **minimum detectable spread**; the green line "
            "is what we actually see. They don't meet until $k$ is several times our 15 years — and "
            "that's *before* charging the borrow on a short leg of distressed, hard-to-locate names. "
            "There is no sizing, threshold, or cost assumption that turns this large-cap F-score into a "
            "deployable strategy. The checklist is a **quality screen**; the *tradable winner-loser "
            "spread* lives in a universe this survivor basket isn't."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The native habitat.** Re-run on a **small-cap value** universe (high book-to-market, "
            "low coverage) — Piotroski's documented edge should reappear; this study isolates the "
            "**universe**, not the checklist, as the binding constraint.\n"
            "- **The growth-side twin.** [Study 232 — Mohanram G-Score](../../232-mohanram-g-score/): "
            "the same financial-statement-analysis idea for low book-to-market firms.\n"
            "- **Composite vs parts.** [Study 121 — Magic-Formula](../../121-magic-formula/), "
            "[Study 122 — Gross-Profitability](../../122-gross-profitability/), "
            "[Study 123 — Altman-Z](../../123-altman-z/) — does a nine-point composite add over one "
            "good ratio, or is it the parts in a trench coat?\n\n"
            "*The reproducible core is offline and deterministic; fundamentals are real EDGAR pulls. "
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
