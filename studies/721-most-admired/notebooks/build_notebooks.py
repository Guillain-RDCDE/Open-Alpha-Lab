"""Generate the two narrative notebooks for Study 721 (Most-Admired).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached admired prices
under ../_cache/ (15 All-Stars + a spurned proxy + a large-cap pool + SPY) and otherwise quote
the frozen headline numbers in ``R`` (mirroring docs/results.md). The synthetic positive
control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance month-end, 15 admired
# All-Stars + spurned proxy + large-cap pool + SPY; 2004-01-31 -> 2026-06-30, as-of 2026-06-30;
# fingerprint 80c5c891a901).
R = dict(
    asof="2026-06-30", fp="80c5c891a901",
    n_admired=15, n_spurned=6, n_delisted=7,
    # variant: n, start, end, excess_ann%, hac_t, alpha_ann%, alpha_t, beta, r2, sharpe
    naive=dict(n=269, start="2004-02", end="2026-06", excess=9.89, hac_t=4.94,
               alpha=8.27, alpha_t=4.08, beta=1.14, r2=0.78, sharpe=1.09),
    lagged=dict(n=221, start="2008-02", end="2026-06", excess=7.05, hac_t=2.53, lags=4,
                alpha=5.63, alpha_t=2.00, beta=1.12, r2=0.70, sharpe=0.61),
    # placebo: (obs%, random_mean%, p)
    placebo=dict(lagged=(7.05, 3.23, 0.019), naive=(9.89, 3.33, 0.000), n_full=59),
    # robustness: (roster, excess%, hac_t, alpha%, alpha_t, beta)
    robust=[("all 15", 7.05, 2.53, 5.63, 2.00, 1.12),
            ("drop NVDA", 6.30, 2.25, 4.97, 1.76, 1.11),
            ("drop NVDA & AAPL", 3.70, 1.90, 1.82, 0.88, 1.12)],
    # long/short admired - spurned proxy
    ls=dict(ann=8.45, hac_t=1.74, n=221),
    # costs on the lagged book
    cost=dict(gross=7.05, net=7.03, turnover=20, bps=10, drag=0.02),
    # synthetic: (edge%, excess%, hac_t, alpha%, alpha_t)
    syn=[(0.0, 0.16, 0.17, 0.15, 0.16), (6.0, 6.16, 6.64, 6.15, 6.60)],
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Premium%3F: Misattributed](https://img.shields.io/badge/Premium%3F-Misattributed-8b949e?style=flat-square)\n\n"
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

from most_admired import data, strategy as st
try:
    from quantlab import repro
    ASOF = "2026-06-30"
except Exception:
    repro = None; ASOF = None

HAVE_REAL = data.have_real()
if HAVE_REAL:
    B = data.load_real()
    if repro is not None:
        B["prices"] = repro.as_of(B["prices"], ASOF)
else:
    B = None
print("real admired-tape cache present:", HAVE_REAL,
      "| months:", (0 if B is None else len(B["prices"])))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# The most admired companies — does buying the list beat the market? ⭐\n"
            "### 'Great company' vs 'great stock' — and the survey that keeps confusing the two\n\n"
            + BADGES +
            "Every year *Fortune* polls thousands of executives and analysts and publishes the "
            "**World's Most Admired Companies** — the best-run firms on earth, by acclamation. Apple "
            "has topped the overall list **every single year from 2008 to 2024**. The intuition is "
            "irresistible: if these are the greatest companies, surely they're the greatest *stocks* — "
            "just buy the list and let quality compound.\n\n"
            "This notebook asks whether that's true, or whether it's one of the oldest traps in "
            "investing: a **great company is not the same thing as a great stock**. We build the admired "
            "basket, race it against the market, and then do the one test that decides everything — "
            "**take away the hindsight** and see what's left.\n\n"
            "> 📓 **Plain-language layer.** Want the HAC *t*-stats, the market-model alpha and the "
            "placebo? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** Today's admired list is dominated by companies that are on "
            "it *because they already won* — Apple, Microsoft, Nvidia. Testing 'the winners won' proves "
            "nothing. So we do it two ways: the naive way (own today's list back in time — cheating) and "
            "the honest way (own a company only *after* Fortune first crowns it). Every chart is drawn by "
            "the code beside it; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Did the admired mega-caps beat the market? | **Yes.** Even the honest, no-cheating book "
            f"beat `SPY` by **+{R['lagged']['excess']:.1f}%/yr**. |\n"
            "| So the 'admiration premium' is real? | **Barely, and misleadingly.** It clears the "
            f"significance bar (*t* = {R['lagged']['hac_t']:.2f}) — but drop just **two names (Apple & "
            f"Nvidia)** and the edge all but vanishes (**+{R['robust'][2][3]:.1f}%/yr**, not significant). |\n"
            "| Is buying admired companies smart? | **Half of it is a free lunch anyone gets.** A "
            f"*random* basket of famous large caps beat `SPY` by **+{R['placebo']['lagged'][1]:.1f}%/yr** "
            "too — nothing to do with admiration. |\n"
            "| Could you have traded it? | **Not really.** The version that looks amazing needs a "
            "**time machine** (owning 2026's winners in 2004). Once you can only act on what Fortune has "
            "*already* announced, the edge is a coin-flippy tilt you were paid beta for. |\n\n"
            "> The admired companies did win — but the survey crowns yesterday's winners, so 'the list "
            "beats the market' mostly means 'Apple and Nvidia went up.' That's a story about the past, "
            "not a machine for the future."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Fortune's Most Admired Companies are the best-run businesses in the world — top "
            "management, top products, deep moats. Great companies make great stocks. Just buy the "
            "list and hold quality; it compounds and beats the index.\"*\n\n"
            "It has real academic backing: **Antunovich, Laster & Mishra (2000)** found the most-admired "
            "firms *out-performed* the least-admired and the market over 1983–98. But there's an equal "
            "and opposite school — **Statman, Fisher & Anginer (2008)** — arguing the *opposite*: a firm "
            "is admired only *after* it's already loved and expensive, so the **spurned** stocks are the "
            "ones that go on to win. Two respectable camps, opposite predictions. Only the tape can "
            "settle it."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If admiration really predicted returns, it'd be the easiest edge in the world — no models, "
            "no data feeds, just read a magazine once a year and buy the cover stars. It would also say "
            "something deep about markets: that a **reputation survey** carries information price hasn't "
            "already absorbed. But there's a famous catch, and it's the whole ballgame here: a company "
            "earns its spot on the list by having *already* done brilliantly. So we have to be ruthless "
            "about **timing** — you can only buy a company *after* it's crowned — and about **hindsight** "
            "— today's list is a roster of past champions. Miss either and you're just admiring the "
            "winners after the race."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We hardcode a **transparent table of {R['n_admired']} All-Stars** — the mega-caps Fortune "
            "has crowned year after year (Apple, Amazon, Microsoft, Berkshire, Disney, Alphabet, "
            "Starbucks, Nike, Costco, JPMorgan, and more) — and:\n\n"
            "1. **Build the basket.** Equal-weight the admired names; measure how much it beats the "
            "market (`SPY`) each month.\n"
            "2. **Two ways.** The **naive** book owns today's list all the way back (a time machine); "
            "the **honest** book owns a name only *after* Fortune first crowns it.\n"
            "3. **Kill the hindsight.** Remove the two biggest winners (Apple, Nvidia) and see if a "
            "'premium' survives. Draw *random* baskets of famous large caps and see if they beat the "
            "market just as much.\n\n"
            "And we say it loudly: the list is the winners that **stayed** winners — the companies that "
            "were admired and then *fell off* (or went bankrupt) aren't in it. The deck is stacked "
            "**for** the story."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the honest book vs the market.** Own each admired company only *after* it's "
            "crowned, equal-weight, and compare growth of \\$1 against `SPY`."
        ),
        code(
            "if HAVE_REAL:\n"
            "    book = st.admired_book(B['prices'], B['admired'], entry=B['entry'], lagged=True)\n"
            "    import pandas as pd\n"
            "    m = st.monthly_returns(B['prices'])['SPY']\n"
            "    d = pd.concat([book.rename('admired'), m.rename('SPY')], axis=1).dropna()\n"
            "    g_ad = (1+d['admired']).cumprod(); g_mk = (1+d['SPY']).cumprod()\n"
            "    idx = g_ad.index\n"
            "else:\n"
            "    import pandas as pd\n"
            "    idx = pd.date_range('2008-02-28', periods=R['lagged']['n'], freq='ME')\n"
            "    rng=np.random.default_rng(721)\n"
            "    ad=rng.normal((R['lagged']['excess']/100+0.09)/12,0.05,len(idx)); mk=rng.normal(0.09/12,0.043,len(idx))\n"
            "    g_ad=pd.Series((1+ad).cumprod(),index=idx); g_mk=pd.Series((1+mk).cumprod(),index=idx)\n"
            "fig, ax = plt.subplots(figsize=(9.6, 4.7))\n"
            "ax.plot(idx, g_ad, c=GREEN, lw=2, label='admired book (honest, lagged)')\n"
            "ax.plot(idx, g_mk, c=GREY, lw=2, label='the market (SPY)')\n"
            "ax.set_yscale('log'); ax.set_ylabel('growth of $1 (log)')\n"
            "ax.set_title(f\"Even the no-cheating admired book beat the market (+{R['lagged']['excess']:.1f}%/yr)\")\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"admired ended {g_ad.iloc[-1]:.1f}x vs market {g_mk.iloc[-1]:.1f}x\")"
        ),
        md(
            f"So it *did* win — the honest admired book beat `SPY` by **+{R['lagged']['excess']:.1f}%/yr**. "
            "Case closed? Not even close. The question is *why* it won — and whether you could have known "
            "which companies to buy. Watch what happens when we pull the hindsight out."
        ),
        md(
            "**The tell: it's two names.** Rebuild the honest book, then remove the single biggest "
            "winner (Nvidia), then the two biggest (Apple + Nvidia), and watch the 'premium' evaporate."
        ),
        code(
            "labels=[r[0] for r in R['robust']]; alphas=[r[3] for r in R['robust']]; ats=[r[4] for r in R['robust']]\n"
            "if HAVE_REAL:\n"
            "    labels, alphas, ats = [], [], []\n"
            "    no_nv=[r for r in B['admired'] if r[0]!='NVDA']\n"
            "    for nm, adm in [('all 15', B['admired']), ('drop NVDA', no_nv), ('drop NVDA & AAPL', [r for r in no_nv if r[0]!='AAPL'])]:\n"
            "        bk=st.admired_book(B['prices'], adm, entry=B['entry'], lagged=True)\n"
            "        mm=st.market_model_alpha(bk, B['prices'])\n"
            "        labels.append(nm); alphas.append(mm['alpha_ann']*100); ats.append(mm['alpha_t'])\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "cols=[GREEN if t>=2 else RED for t in ats]\n"
            "bars=ax.bar(labels, alphas, color=cols, width=.55)\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('admiration alpha (%/yr, beta-adjusted)')\n"
            "ax.set_title('Take away Apple & Nvidia and the premium is gone')\n"
            "for b,a,t in zip(bars, alphas, ats): ax.annotate(f'{a:+.1f}%/yr\\nt={t:.2f}',(b.get_x()+b.get_width()/2,a),ha='center',va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('alpha /yr:', [f'{a:+.1f}% (t={t:.2f})' for a,t in zip(alphas,ats)])"
        ),
        md(
            f"There it is. With all 15 names the beta-adjusted 'admiration alpha' is "
            f"**+{R['robust'][0][3]:.1f}%/yr** (green, *t* = {R['robust'][0][4]:.2f}). Drop **Nvidia** and "
            f"it's already not significant. Drop **Apple too** and it's **+{R['robust'][2][3]:.1f}%/yr at "
            f"*t* = {R['robust'][2][4]:.2f}** — a rounding error. The 'premium' isn't a property of being "
            "admired; it's two extraordinary companies, which the survey noticed *after* the fact."
        ),
        md(
            "**And half of what's left isn't special either.** Draw *random* baskets of 15 famous large "
            "caps thousands of times and see how often they beat the market by as much as the admired "
            "book does."
        ),
        code(
            "obs, rmean, pval = R['placebo']['lagged']\n"
            "if HAVE_REAL:\n"
            "    book = st.admired_book(B['prices'], B['admired'], entry=B['entry'], lagged=True)\n"
            "    ex = st.excess_over_market(book, B['prices'])\n"
            "    pv = st.placebo_pvalue(B['prices'], data.POOL, k=R['n_admired'], observed_ann=ex.mean()*12, start='2008-02-01', n_draws=3000)\n"
            "    obs, rmean, pval = ex.mean()*12*100, pv['placebo_mean_ann']*100, pv['p_value']\n"
            "    # rebuild a small cloud for the picture\n"
            "    import pandas as pd\n"
            "    rets=st.monthly_returns(B['prices']); avail=[t for t in data.POOL if t in rets.columns and t!='SPY']\n"
            "    sub=rets[avail]; sub=sub[sub.index>=pd.Timestamp('2008-02-01')]; mk=rets['SPY'].reindex(sub.index)\n"
            "    full=[t for t in avail if sub[t].notna().all()]; A=sub[full].to_numpy(); mv=mk.to_numpy()\n"
            "    rng=np.random.default_rng(721); cloud=np.array([ (A[:,rng.choice(len(full),R['n_admired'],replace=False)].mean(1)-mv).mean()*12*100 for _ in range(3000)])\n"
            "else:\n"
            "    rng=np.random.default_rng(721); cloud=rng.normal(rmean, 2.5, 3000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(cloud, bins=45, color=GREY, alpha=.85, label='random 15 large caps vs SPY')\n"
            "ax.axvline(rmean, c=AMBER, ls='--', lw=2, label=f'random average {rmean:+.1f}%/yr')\n"
            "ax.axvline(obs, c=GREEN, lw=2.5, label=f'the admired book {obs:+.1f}%/yr')\n"
            "ax.set_xlabel('beat the market by (%/yr)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'A RANDOM large-cap basket already beats SPY by ~{rmean:+.0f}%/yr'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'random large-cap basket: {rmean:+.1f}%/yr | admired: {obs:+.1f}%/yr | p={pval:.3f}')"
        ),
        md(
            f"The amber line is the punchline: **just picking 15 famous large caps at random** beat the "
            f"cap-weighted `SPY` by **+{R['placebo']['lagged'][1]:.1f}%/yr** — that's the plain "
            "equal-weight-large-cap tilt, free to anyone, no admiration required. It's **half** the "
            f"admired book's +{R['placebo']['lagged'][0]:.1f}%/yr. The green line is a bit further right "
            f"(*p* = {R['placebo']['lagged'][2]:.2f}) — but remember, even the random pool is only "
            "*survivors*, so this comparison is generous to the story."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Weak.** The honest book beats the market at *t* = {R['lagged']['hac_t']:.2f}, "
            "but it's **two names** (drop Apple + Nvidia → nothing) plus a generic large-cap tilt a "
            "*random* basket also gets. Significant on paper, fragile to the tiniest scrutiny.\n"
            "- **Tradability — Mirage.** The jaw-dropping version (*t* ≈ 5) needs a **time machine**. "
            "The honest version is a levered, tech-tilted basket — beta you were already paid for — and "
            "costs are trivial, so there's no friction alibi. Nothing to harvest going forward.\n"
            "- **Premium, or beta + hindsight? — Misattributed.** The out-performance is Apple and "
            "Nvidia plus an equal-weight tilt, wearing an 'admiration' label. And the opposite "
            "(contrarian) story fails too — the admired *beat* the spurned. The survey tracks the past; "
            "it doesn't forecast."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — the time-machine tax\n\n"
            "Line the two books up side by side: the **naive** one (own today's list back to 2004 — "
            "cheating) and the **honest** one (own a name only after it's crowned). The gap between them "
            "*is* the hindsight."
        ),
        code(
            "nv=R['naive']; lg=R['lagged']\n"
            "if HAVE_REAL:\n"
            "    sn=st.summarize(B, lagged=False); sl=st.summarize(B, lagged=True)\n"
            "    nv=dict(excess=sn['excess_ann']*100, hac_t=sn['hac_t']); lg=dict(excess=sl['excess_ann']*100, hac_t=sl['hac_t'])\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "bars=ax.bar(['naive\\n(time machine)','honest\\n(lagged)'], [nv['excess'], lg['excess']], color=[GREY, GREEN], width=.5)\n"
            "ax.set_ylabel('beat the market by (%/yr)')\n"
            "ax.set_title('The \"time-machine tax\": ~3%/yr of the edge is pure hindsight')\n"
            "for b,v,t in zip(bars,[nv['excess'],lg['excess']],[nv['hac_t'],lg['hac_t']]): ax.annotate(f'{v:+.1f}%/yr\\nt={t:.2f}',(b.get_x()+b.get_width()/2,v),ha='center',va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"naive {nv['excess']:+.1f}%/yr (t={nv['hac_t']:.2f}) -> honest {lg['excess']:+.1f}%/yr (t={lg['hac_t']:.2f})\")"
        ),
        md(
            f"The naive book beats the market by **+{R['naive']['excess']:.1f}%/yr** (*t* = "
            f"{R['naive']['hac_t']:.2f}) — but that number is unearnable, because it assumes you knew in "
            "2004 who'd be admired in 2026. Strip the hindsight and you're at "
            f"**+{R['lagged']['excess']:.1f}%/yr**, and — as we saw — that's two stocks and a tilt. "
            "There's no press-release-to-portfolio machine here; there's a magazine that's very good at "
            "recognising winners once they've won."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The sibling label study.** [Study 389 — Name-Change-Effect](../389-name-change-effect/): "
            "does *rebranding* toward the hot theme pay? Same family (a label, not a fundamental), same "
            "survivorship pathology.\n"
            "- **The corporate-event cousin.** [Study 391 — CEO-Turnover](../391-ceo-turnover/): a "
            "market-model event study on a hardcoded, cited announcement table.\n"
            "- **Add the fallen.** Our roster is the winners that *stayed* admired. Reconstruct the "
            "*actual* year-by-year list (including firms that later dropped off or went bankrupt) from a "
            "survivorship-free feed, and the premium likely shrinks further — the honest test the public "
            "data can't quite reach.\n\n"
            "*Think there's a real admiration edge? Rebuild the list year by year with no look-ahead, "
            "neutralise the market beta and the size tilt, and show an alpha that survives dropping your "
            "two best names — then we'll talk.*"
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
            "# Most-Admired — a quantitative teardown 🔬\n"
            "### Equal-weight admired book vs SPY · a Newey–West (HAC) *t* + market-model alpha · "
            "leave-two-out robustness · a random-large-cap placebo · the survivor-biased long/short · "
            "a synthetic power control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). We treat the "
            "folklore as a **characteristic-sort hypothesis** — an equal-weight book of Fortune's "
            "Most Admired All-Stars earns a positive **abnormal** (beta-adjusted) monthly return — and "
            "confront it with **look-ahead**, **survivorship**, and **factor beta**. The decisive "
            "objects are a HAC *t* on the monthly excess-over-SPY return, the market-model alpha, and a "
            "leave-two-out that isolates how much of the 'premium' is two hindsight winners.\n\n"
            "> ⚠️ **Data + selection note.** The admired table is hardcoded and cited "
            f"({R['n_admired']} perennial All-Stars); the priced tape is **look-ahead + survivor-"
            "selected** — the roster is the winners that *stayed* admired, biasing the book *up* (named "
            "on the Signal axis). The **LAGGED** variant removes the *timing* look-ahead (own a name only "
            "after it's crowned); the roster bias remains. Real data: yfinance month-end adjusted "
            f"closes, 2004→2026, as-of {R['asof']}, fingerprint `{R['fp']}`. Offline core + synthetic "
            "control are deterministic. Methods in [`docs/references.md`](../docs/references.md), numbers "
            "in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | lagged book excess **+{R['lagged']['excess']:.2f}%/yr** (HAC "
            f"**t = {R['lagged']['hac_t']:.2f}**), market-model alpha **+{R['lagged']['alpha']:.2f}%/yr** "
            f"(**t = {R['lagged']['alpha_t']:.2f}**) — but drop Apple+Nvidia → alpha "
            f"**+{R['robust'][2][3]:.1f}%/yr (t = {R['robust'][2][4]:.2f})**, and a random large-cap book "
            f"earns **+{R['placebo']['lagged'][1]:.1f}%/yr**. Significant-raw, fragile-to-selection. |\n"
            f"| **Tradability** | `MIRAGE` | the harvestable-looking book is naive/look-ahead "
            f"(**t = {R['naive']['hac_t']:.2f}**); the honest book is beta **{R['lagged']['beta']:.2f}** "
            f"and costs **{R['cost']['drag']:.2f}%/yr** — the mirage is hindsight, not frictions. |\n"
            f"| **Premium, or beta + hindsight?** | `MISATTRIBUTED` | out-performance = levered "
            "large-cap/tech tilt in Apple & Nvidia; the contrarian reversal fails too (admired − "
            f"spurned = **+{R['ls']['ann']:.1f}%/yr**, *t* = {R['ls']['hac_t']:.2f}). |\n\n"
            "> 💡 In plain words: the admired basket beat the market, but the beat is two stocks, a "
            "size-within-large-cap tilt, and a roster hand-picked from the survivors — not a reward for "
            "*admiration* you could have banked in advance."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r^{\\text{adm}}_t$ be the equal-weight admired book's month-$t$ return and $r^{m}_t$ "
            "the market (`SPY`). Define the monthly **excess** $x_t = r^{\\text{adm}}_t - r^{m}_t$ and the "
            "**market-model** regression $r^{\\text{adm}}_t = \\alpha + \\beta r^{m}_t + \\varepsilon_t$.\n\n"
            "- **H₁ (premium).** $\\mathbb{E}[x_t] > 0$ and $\\alpha > 0$ — admired firms earn an "
            "abnormal return beyond beta (Antunovich–Laster–Mishra 2000).\n"
            "- **H₂ (reversal).** admired − spurned $< 0$ — the *least*-admired out-earn (Statman–Fisher–"
            "Anginer 2008).\n"
            "- **H₃ (deployable).** the alpha survives a **publication lag**, a **random-large-cap "
            "placebo**, **dropping the two best names**, and costs.\n\n"
            "We find **H₁ holds raw but fragile** ($\\alpha$ at $t=2.0$, gone without Apple+Nvidia), "
            "**H₂ rejected** (admired *beat* spurned), **H₃ rejected** (the deployable version is "
            "look-ahead; the honest one is beta + two names). The label marks *past* returns, not future "
            "ones."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The teardown is a mean-and-alpha test on an autocorrelated monthly series, decided by its "
            "**HAC standard error**, its **beta**, and its **selection**:\n\n"
            "$$t_{\\text{HAC}} = \\frac{\\bar x}{\\widehat{\\mathrm{se}}_{\\text{NW}}(\\bar x)},\\qquad "
            "\\widehat S = \\hat\\gamma_0 + 2\\sum_{l=1}^{L}\\Big(1-\\tfrac{l}{L+1}\\Big)\\hat\\gamma_l.$$\n\n"
            "Newey–West widens the naive SE for the mild autocorrelation and fat tails of monthly equity "
            "returns (an iid *t* over-rejects). But the binding issues aren't the SE — they're that "
            "(a) the book carries **$\\beta>1$**, so raw excess overstates alpha; (b) the roster is "
            "**conditioned on staying admired**, biasing $\\bar x$ **up**; and (c) an equal-weight "
            "large-cap book beats the **cap-weighted** index by a generic tilt. The honest instruments "
            "are therefore the **market-model $\\alpha$**, a **leave-two-out**, and a **random-large-cap "
            "placebo** — not the point estimate."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Admired table.** {R['n_admired']} perennial All-Stars, hardcoded & cited, each with an "
            "approximate first-crown year; **NAIVE** owns the current list from 2004 (look-ahead), "
            "**LAGGED** owns a name only from Feb of its first-crown year (no timing look-ahead).\n"
            "- **Excess & alpha.** Monthly $x_t = r^{\\text{adm}}_t - r^{m}_t$; market-model "
            "$\\alpha,\\beta$ with a **HAC (Newey–West)** *t*, Bartlett kernel, "
            f"$L={R['lagged']['lags']}$ (auto).\n"
            "- **Robustness.** Leave-one-out (drop NVDA) and leave-two-out (drop NVDA & AAPL) — does the "
            "alpha survive without the extreme winners?\n"
            "- **Placebo.** Random equal-weight 15-name books from a broad large-cap pool; "
            "$p = \\Pr[|\\text{random excess}| \\ge |\\text{admired excess}|]$.\n"
            "- **Long/short.** admired − a **survivor** spurned proxy (survivorship-biased *up* on the "
            "short leg, i.e. against the premium).\n"
            "- **Costs.** Annual-rebalance turnover × one-way bps × NAV.\n"
            "- **Positive control.** Deterministic books with a **planted** annual premium `edge`: the "
            "HAC inference must recover a large edge **and** must NOT manufacture significance at "
            "`edge = 0`.\n\n"
            "**What would make us say REAL:** a beta-adjusted alpha with HAC $t \\ge 2$ that **survives "
            "the lag, the placebo, and dropping the two best names**. It does not."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Excess & alpha — naive vs honest\n\n"
            "The two books' monthly excess-over-SPY, annualised, with the HAC *t* and the beta-adjusted "
            "market-model alpha. The naive book is look-ahead; the lagged book is the honest one."
        ),
        code(
            "rows=[('NAIVE (look-ahead)', R['naive']), ('LAGGED (honest)', R['lagged'])]\n"
            "if HAVE_REAL:\n"
            "    rows=[]\n"
            "    for tag,lag in [('NAIVE (look-ahead)',False),('LAGGED (honest)',True)]:\n"
            "        s=st.summarize(B, lagged=lag)\n"
            "        rows.append((tag, dict(excess=s['excess_ann']*100, hac_t=s['hac_t'], alpha=s['alpha_ann']*100, alpha_t=s['alpha_t'], beta=s['beta'])))\n"
            "fig,(a1,a2)=plt.subplots(1,2,figsize=(11,4.3))\n"
            "names=[r[0].split()[0] for r in rows]; xx=np.arange(len(names))\n"
            "ex=[r[1]['excess'] for r in rows]; al=[r[1]['alpha'] for r in rows]\n"
            "a1.bar(xx-.2, ex, .4, color=GREY, label='raw excess'); a1.bar(xx+.2, al, .4, color=GREEN, label='alpha (beta-adj)')\n"
            "a1.set_xticks(xx); a1.set_xticklabels(names); a1.set_ylabel('%/yr'); a1.axhline(0,c='k',lw=.8)\n"
            "a1.set_title('Excess vs beta-adjusted alpha'); a1.legend()\n"
            "ats=[r[1]['alpha_t'] for r in rows]; hts=[r[1]['hac_t'] for r in rows]\n"
            "a2.bar(xx-.2, hts, .4, color=AMBER, label='excess HAC t'); a2.bar(xx+.2, ats, .4, color=GREEN, label='alpha HAC t')\n"
            "a2.axhline(2, ls='--', c=RED, label='t=2'); a2.set_xticks(xx); a2.set_xticklabels(names)\n"
            "a2.set_ylabel('Newey–West t'); a2.set_title('Honest book sits right on t=2'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for tag,d in rows: print(f\"{tag:20s} excess {d['excess']:+5.2f}%/yr (t={d['hac_t']:.2f})  alpha {d['alpha']:+5.2f}%/yr (t={d['alpha_t']:.2f})  beta {d['beta']:.2f}\")"
        ),
        md(
            f"> 💡 In plain words: the naive book is **+{R['naive']['excess']:.1f}%/yr** at HAC "
            f"*t* = {R['naive']['hac_t']:.2f} — but unearnable (look-ahead). The honest book is "
            f"**+{R['lagged']['excess']:.1f}%/yr**, alpha **+{R['lagged']['alpha']:.1f}%/yr** at "
            f"*t* = {R['lagged']['alpha_t']:.2f} — *exactly* on the line, on a book with beta "
            f"**{R['lagged']['beta']:.2f}**. A marginal alpha on a levered basket is not a robust REAL."
        ),
        md(
            "### 4b · The decisive test — leave-two-out\n\n"
            "Drop the extreme winners from the honest book and re-fit the market-model alpha. A genuine "
            "*admiration* effect should be a property of the whole roster, not of one or two names."
        ),
        code(
            "rob=R['robust']\n"
            "if HAVE_REAL:\n"
            "    rob=[]; no_nv=[r for r in B['admired'] if r[0]!='NVDA']\n"
            "    for nm,adm in [('all 15',B['admired']),('drop NVDA',no_nv),('drop NVDA & AAPL',[r for r in no_nv if r[0]!='AAPL'])]:\n"
            "        bk=st.admired_book(B['prices'],adm,entry=B['entry'],lagged=True); ex=st.excess_over_market(bk,B['prices'])\n"
            "        nw=st.newey_west_t(ex.to_numpy()); mm=st.market_model_alpha(bk,B['prices'])\n"
            "        rob.append((nm, nw['ann']*100, nw['t'], mm['alpha_ann']*100, mm['alpha_t'], mm['beta']))\n"
            "labs=[r[0] for r in rob]; al=[r[3] for r in rob]; at=[r[4] for r in rob]; xx=np.arange(len(labs))\n"
            "fig, ax = plt.subplots(figsize=(9.4,4.4))\n"
            "cols=[GREEN if t>=2 else RED for t in at]\n"
            "bars=ax.bar(xx, al, color=cols, width=.55)\n"
            "ax.axhline(0,c='k',lw=.8); ax.set_xticks(xx); ax.set_xticklabels(labs)\n"
            "ax.set_ylabel('market-model alpha (%/yr)'); ax.set_title('The premium is two names, not a roster')\n"
            "for b,a,t in zip(bars,al,at): ax.annotate(f'{a:+.1f}%\\nt={t:.2f}',(b.get_x()+b.get_width()/2,a),ha='center',va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "for nm,exx,ext,a,t,bta in rob: print(f'{nm:18s} excess {exx:+5.2f}%/yr(t={ext:.2f})  alpha {a:+5.2f}%/yr(t={t:.2f})')"
        ),
        md(
            f"> 💡 In plain words: alpha goes **+{R['robust'][0][3]:.1f}% (t={R['robust'][0][4]:.2f}) → "
            f"+{R['robust'][1][3]:.1f}% (t={R['robust'][1][4]:.2f}) → +{R['robust'][2][3]:.1f}% "
            f"(t={R['robust'][2][4]:.2f})** as we drop Nvidia, then Apple. A single 2023 arrival "
            "(Nvidia) already knocks it below significance. The 'admiration premium' is not diffuse "
            "across admired firms — it is concentrated in the two the survey crowned *after* they'd "
            "already run."
        ),
        md(
            "### 4c · Placebo + long/short — a free tilt, and no reversal\n\n"
            "**Left:** random equal-weight 15-name large-cap books vs the admired book's excess. "
            "**Right:** the contrarian long/short (admired − a survivor spurned proxy) against $t=2$."
        ),
        code(
            "obs,rmean,pval = R['placebo']['lagged']; ls=R['ls']\n"
            "if HAVE_REAL:\n"
            "    import pandas as pd\n"
            "    book=st.admired_book(B['prices'],B['admired'],entry=B['entry'],lagged=True); ex=st.excess_over_market(book,B['prices'])\n"
            "    pv=st.placebo_pvalue(B['prices'],data.POOL,k=R['n_admired'],observed_ann=ex.mean()*12,start='2008-02-01',n_draws=3000)\n"
            "    obs,rmean,pval = ex.mean()*12*100, pv['placebo_mean_ann']*100, pv['p_value']\n"
            "    rets=st.monthly_returns(B['prices']); avail=[t for t in data.POOL if t in rets.columns and t!='SPY']\n"
            "    sub=rets[avail]; sub=sub[sub.index>=pd.Timestamp('2008-02-01')]; mk=rets['SPY'].reindex(sub.index)\n"
            "    full=[t for t in avail if sub[t].notna().all()]; A=sub[full].to_numpy(); mv=mk.to_numpy()\n"
            "    rng=np.random.default_rng(721); cloud=np.array([(A[:,rng.choice(len(full),R['n_admired'],replace=False)].mean(1)-mv).mean()*12*100 for _ in range(3000)])\n"
            "    lsr=st.long_short(B['prices'],B['admired'],B['spurned'],entry=B['entry'],lagged=True); nw=st.newey_west_t(lsr.to_numpy())\n"
            "    ls=dict(ann=nw['ann']*100, hac_t=nw['t'])\n"
            "else:\n"
            "    rng=np.random.default_rng(721); cloud=rng.normal(rmean,2.5,3000); ls=dict(ann=R['ls']['ann'], hac_t=R['ls']['hac_t'])\n"
            "fig,(a1,a2)=plt.subplots(1,2,figsize=(11,4.3))\n"
            "a1.hist(cloud,bins=45,color=GREY,alpha=.85,label='random 15 large caps'); a1.axvline(rmean,c=AMBER,ls='--',lw=2,label=f'random avg {rmean:+.1f}%/yr')\n"
            "a1.axvline(obs,c=GREEN,lw=2.5,label=f'admired {obs:+.1f}%/yr'); a1.set_xlabel('excess over SPY (%/yr)'); a1.set_ylabel('freq')\n"
            "a1.set_title(f'Placebo p={pval:.3f} (half the edge is a free tilt)'); a1.legend()\n"
            "a2.bar(['admired − spurned'], [ls['ann']], color=AMBER, width=.4)\n"
            "a2.axhline(0,c='k',lw=.8); a2.set_ylabel('%/yr'); a2.set_title(f\"Reversal fails: admired BEAT spurned (t={ls['hac_t']:.2f})\")\n"
            "a2.annotate(f\"{ls['ann']:+.1f}%/yr\\nt={ls['hac_t']:.2f} (n.s.)\",(0,ls['ann']),ha='center',va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'placebo: random {rmean:+.1f}%/yr vs admired {obs:+.1f}%/yr (p={pval:.3f}) | long/short {ls[\"ann\"]:+.1f}%/yr t={ls[\"hac_t\"]:.2f}')"
        ),
        md(
            f"> 💡 In plain words: a *random* large-cap basket already earns "
            f"**+{R['placebo']['lagged'][1]:.1f}%/yr** over cap-weighted `SPY` — the equal-weight tilt, "
            f"half the admired book's edge and free to anyone. And the contrarian story is dead too: "
            f"admired *beat* spurned by **+{R['ls']['ann']:.1f}%/yr** (*t* = {R['ls']['hac_t']:.2f}, not "
            "significant, and the spurned leg is survivor-biased *up*, which only helps the reversal — "
            "and it still fails)."
        ),
        md(
            "### 4d · Faithful-engine & power control — we know the truth here\n\n"
            "Deterministic month-end books with a **planted** annual premium `edge`: with **`edge=0`** "
            "the HAC inference must stay flat; with **+6%/yr** it must light up on both the excess-*t* "
            "and the alpha-*t*. Both hold — the engine is unbiased, so the real-tape result is about the "
            "data, not the harness."
        ),
        code(
            "res=[]\n"
            "for edge in (0.0, 0.06):\n"
            "    syn=data.synthetic_admired(n_names=15, edge_ann=edge, seed=721); s=st.summarize(syn, lagged=False)\n"
            "    res.append((edge*100, s['excess_ann']*100, s['hac_t'], s['alpha_ann']*100, s['alpha_t']))\n"
            "labels=[f'planted\\n{e:.0f}%/yr' for e,*_ in res]; xx=np.arange(len(labels))\n"
            "ht=[r[2] for r in res]; at=[r[4] for r in res]\n"
            "fig, ax = plt.subplots(figsize=(8.8,4.3))\n"
            "ax.bar(xx-.2, ht, .4, color=AMBER, label='excess HAC t'); ax.bar(xx+.2, at, .4, color=GREEN, label='alpha HAC t')\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2'); ax.set_xticks(xx); ax.set_xticklabels(labels)\n"
            "ax.set_ylabel('Newey–West t'); ax.set_title('Control: flat at edge=0, lights up at +6%/yr'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,ex,ht_,a,at_ in res: print(f'planted {e:+.0f}%/yr: excess {ex:+5.2f}%/yr(t={ht_:.2f})  alpha {a:+5.2f}%/yr(t={at_:.2f})')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted edge the control's HAC *t* is "
            f"**{R['syn'][0][2]:.2f}** (no false positive); a **+6%/yr** plant reaches "
            f"**{R['syn'][1][2]:.2f}**. So the machinery is honest — and the real lagged alpha *t* of "
            f"**{R['lagged']['alpha_t']:.2f}**, which evaporates when two names leave, is a genuine but "
            "**fragile, hindsight-loaded** reading, not a robust effect."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — lagged excess **+{R['lagged']['excess']:.2f}%/yr** (HAC "
            f"**t = {R['lagged']['hac_t']:.2f}**), alpha **+{R['lagged']['alpha']:.2f}%/yr** "
            f"(**t = {R['lagged']['alpha_t']:.2f}**) clears the bar raw — but it is "
            "**significant-raw, fragile-to-selection**: drop Apple + Nvidia → alpha "
            f"**+{R['robust'][2][3]:.1f}%/yr (t = {R['robust'][2][4]:.2f})**, ~half is the generic "
            f"large-cap tilt (placebo **+{R['placebo']['lagged'][1]:.1f}%/yr**), and the roster is "
            "survivorship-selected. Literature split (Antunovich premium vs Statman reversal) ⇒ **WEAK**, "
            "not REAL. Look-ahead + survivorship named on this axis.\n"
            f"- **Tradability `MIRAGE`** — the harvestable-looking book (naive, HAC "
            f"**t = {R['naive']['hac_t']:.2f}**) is **look-ahead**; the honest book is beta "
            f"**{R['lagged']['beta']:.2f}** with a marginal alpha that halves without one stock, and "
            f"costs are **{R['cost']['drag']:.2f}%/yr** — the mirage is hindsight, not frictions.\n"
            f"- **Premium, or beta + hindsight? `MISATTRIBUTED`** — the out-performance is a levered "
            "equal-weight large-cap/tech tilt concentrated in Apple & Nvidia, on a roster and pool that "
            f"both survived to 2026; the reversal fails too (admired − spurned **+{R['ls']['ann']:.1f}%/yr**). "
            "The label tracks past returns; it doesn't predict future ones."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the look-ahead decomposition\n\n"
            "The operational truth in one stack: start from the naive book's excess and peel off, in "
            "order, the pieces you couldn't actually keep — the **timing** look-ahead (naive→lagged), "
            "the **generic large-cap tilt** (the placebo mean), and the **two-name concentration** — "
            "until you reach the honest, diversified, prospectively-earnable residual."
        ),
        code(
            "naive_ex = R['naive']['excess']; lagged_ex = R['lagged']['excess']\n"
            "tilt = R['placebo']['lagged'][1]; two_name = R['robust'][0][3] - R['robust'][2][3]\n"
            "if HAVE_REAL:\n"
            "    sn=st.summarize(B,lagged=False); sl=st.summarize(B,lagged=True)\n"
            "    naive_ex=sn['excess_ann']*100; lagged_ex=sl['excess_ann']*100\n"
            "steps=['naive\\nexcess','− timing\\nlook-ahead','− generic\\nlarge-cap tilt','− two-name\\nconcentration']\n"
            "vals=[naive_ex, naive_ex-lagged_ex, tilt, two_name]\n"
            "residual = naive_ex - (naive_ex-lagged_ex) - tilt - two_name\n"
            "fig, ax = plt.subplots(figsize=(9.6,4.5))\n"
            "cum=naive_ex; xs=range(len(steps))\n"
            "ax.bar(0, naive_ex, color=GREY, width=.6)\n"
            "base=naive_ex\n"
            "for i,(lab,v) in enumerate(zip(steps[1:],vals[1:]),start=1):\n"
            "    ax.bar(i, -v, bottom=base, color=RED, width=.6); base-=v\n"
            "ax.bar(len(steps), base, color=GREEN, width=.6)\n"
            "ax.set_xticks(list(xs)+[len(steps)]); ax.set_xticklabels(steps+['honest\\nresidual'])\n"
            "ax.axhline(0,c='k',lw=.8); ax.set_ylabel('excess over SPY (%/yr)')\n"
            "ax.set_title('Peeling the look-ahead and the tilt off the \"admiration premium\"')\n"
            "ax.annotate(f'{base:+.1f}%/yr',(len(steps),base),ha='center',va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'naive {naive_ex:+.1f}%/yr -> honest residual {base:+.1f}%/yr after removing timing, tilt, and 2-name concentration')"
        ),
        md(
            "> 💡 In plain words: start at the eye-popping naive **+9.9%/yr**, remove the **timing "
            "look-ahead** (you can't own tomorrow's list), remove the **generic large-cap tilt** (free "
            "to anyone), remove the **two-name concentration** (Apple + Nvidia) — and the prospectively-"
            "earnable 'admiration' residual is a whisper, on a book you're levered long (beta 1.12) to "
            "buy. There is no sizing or cost assumption that rescues an edge from a survey that recognises "
            "winners *after* they win."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The sibling label study.** [Study 389 — Name-Change-Effect](../389-name-change-effect/): "
            "does *rebranding* toward the hot theme pay? Same family (a label, not a fundamental).\n"
            "- **The corporate-event cousin.** [Study 391 — CEO-Turnover](../391-ceo-turnover/): "
            "market-model CARs on a hardcoded, cited announcement table.\n"
            "- **The survivorship-free rebuild.** Reconstruct Fortune's *actual* year-by-year list "
            "(including names that later dropped off or went bankrupt) from a CRSP/Norgate feed, form the "
            "portfolio at each publication, and re-estimate the alpha with no roster survivorship — the "
            "honest test public data can't quite reach. Antunovich's positive result was 1983–98; this "
            "tape's fragility suggests it does not extend.\n\n"
            "*The reproducible core is offline and deterministic; the admired table is hardcoded and the "
            "priced tape is look-ahead + survivor-selected (named). Methods and sources: "
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
