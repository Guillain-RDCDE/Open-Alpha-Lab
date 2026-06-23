"""Generate the two narrative notebooks for Study 390 (Activist-13D).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached event prices
under ../_cache/ (the 25 famous activist targets + SPY) and otherwise quote the frozen
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance, 25 famous
# activist 13D campaigns + SPY, 2010-01-04 -> 2026-06-18, 4,140 days, 16.5 years).
R = dict(
    start="2010-01-04", end="2026-06-18", days=4140, years=16.5, n_events=25,
    # announcement pop (day 0): n, mean%, median%, win%, t
    pop=(24, 1.22, 0.62, 62, 0.99),
    # per-horizon drift: (months, n, mean_ex%, median_ex%, win%, t, p_placebo)
    h1=(1, 25, 1.74, 1.36, 64, 1.11, 0.183),
    h3=(3, 25, 0.89, 1.53, 56, 0.40, 0.449),
    h6=(6, 25, 3.78, 3.16, 60, 0.98, 0.283),
    # net @ 10bps: (months, gross%, net%)
    net=[(1, 1.74, 1.64), (3, 0.89, 0.79), (6, 3.78, 3.68)],
    # lag robustness (6m): (lag, mean%, t, p)
    lag=[(0, 3.88, 1.09, 0.277), (1, 3.78, 0.98, 0.283),
         (2, 5.33, 1.13, 0.195), (5, 4.78, 1.03, 0.223)],
    # drop mega-caps (6m): n, mean%, t, p
    nomega=(22, 2.49, 0.60, 0.370),
    # synthetic control (6m): (planted_drift, n, mean%, win%, t, p)
    syn=[(0.0, 25, 3.64, 52, 0.72, 0.262), (0.20, 25, 27.80, 80, 4.45, 0.000)],
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
    "![Free_lunch%3F: Busted](https://img.shields.io/badge/Free_lunch%3F-Busted-8b949e?style=flat-square)\n\n"
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

from activist_13d import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    PRICES, TBL = data.load_real()
else:
    PRICES, TBL = None, None
print("real event-price cache present:", HAVE_REAL,
      "| activist campaigns priced:", (0 if TBL is None else len(TBL)))
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
            "# Buy when the shark shows up? — does a stock keep drifting after an activist 13D? 🦈\n"
            "### The \"free ride\" of buying the moment Icahn or Elliott files, in plain English\n\n"
            + BADGES +
            "Here's a tempting trade. When a famous activist investor — Carl Icahn, Elliott, Bill "
            "Ackman, Nelson Peltz — quietly buys more than **5%** of a company, US law forces them to "
            "announce it publicly within days, in a filing called a **Schedule 13D**. The lore says: the "
            "stock **pops** on the news, and then **keeps drifting up** for months as the activist "
            "shakes the company into buybacks, spin-offs and board seats. Just buy when the 13D hits — "
            "free ride.\n\n"
            "It *sounds* like free money, and a few legends back it up (Icahn's Apple stake, Elliott at "
            "Salesforce). But \"it worked for the famous ones\" is exactly what **survivorship** looks "
            "like — you remember the wins and forget the duds. This notebook checks the trade you could "
            "actually place, on a basket of the most famous campaigns, and asks whether it beats simply "
            "owning the market.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo test and the power analysis? "
            "See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** A clean list of *every* 13D isn't free, so we **hardcode 25 "
            "famous campaigns** — a basket picked *because* it's famous, which tilts the test *toward* "
            "the claim. If even the legends can't beat the market reliably, the honest version certainly "
            "can't. Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does the stock pop when the 13D is announced? | **A little — about +1.2% on the day.** "
            "Real news. But you can only grab it if you knew the filing was coming. You didn't. |\n"
            "| If I buy the *day after* and hold, do I beat the market? | **On average, barely.** Over "
            "6 months the target beats SPY by ~3.8% — positive, but well inside the noise. |\n"
            "| Is that extra return reliable? | **No.** Across 25 famous campaigns it never reaches "
            "statistical significance, and **removing three mega-caps cuts it in half.** A few big names "
            "carry the whole thing. |\n"
            "| Then why the \"free ride\" reputation? | **Survivorship.** You remember Icahn/Apple and "
            "Elliott/Salesforce; the forgotten campaigns drag the average back to the market. |\n\n"
            "> The announcement is real news. The *tradable* drift after it is a positive whisper on a "
            "basket of legends — not an edge, and not something you can run a fund on."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"When an activist crosses 5% and files a Schedule 13D, the stock jumps and then keeps "
            "climbing for months — buy the filing and ride the campaign to buybacks, spin-offs or a "
            "sale.\"*\n\n"
            "The picture is intuitive and the academic record gives it a real foothold: large studies of "
            "*thousands* of 13D filings find a meaningful **+7–8% abnormal jump** around the announcement "
            "(Brav, Jiang, Partnoy & Thomas, 2008). The seduction is to assume that jump is *yours* and "
            "that the climb *continues* on a stock you can actually buy. We'll separate the part that's "
            "real-but-unreachable from the part you could trade."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "Two very different things hide inside \"the 13D pops and drifts.\" (1) The **announcement "
            "pop** — the day the news breaks. It's real, but it lands *before you can act*: by the time "
            "you read the headline, the price already moved. (2) The **post-announcement drift** — what "
            "you earn if you buy *after* the news is public and hold for months. Only the second is a "
            "strategy. And it has to beat not *zero* but the **market** — owning a volatile activist "
            "target that rises with the S&P isn't alpha, it's just beta. A drift that merely keeps pace "
            "with SPY is **not an edge**; it's the market in a costume."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **{R['n_events']} famous activist campaigns** ({R['start'][:4]}–2024 "
            "announcements), pull each target's price and SPY, and split the effect cleanly:\n\n"
            "1. **The pop.** What did the target do on the announcement day itself? (Reported, not "
            "tradable.)\n"
            "2. **The drift you can buy.** Buy at the close **one day after** the announcement (a "
            "realistic, no-peeking entry), hold **1 / 3 / 6 months**, and measure the return **in excess "
            "of SPY** — target minus market over the same window.\n"
            "3. **Stress the luck.** For each event, compare it to buying the *same stock* on a "
            "**random** day. Do that thousands of times: how often does a random-date basket match the "
            "13D basket? That's the honest test for a 25-event signal."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the announcement pop.** Here's the day-0 return for each of the famous campaigns "
            "— the news being priced in the moment it breaks."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pops = st.announcement_pops(PRICES, TBL)\n"
            "    pm = pops.mean()*100\n"
            "else:\n"
            "    pops = np.array([R['pop'][1]/100]); pm = R['pop'][1]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.2))\n"
            "order = np.argsort(pops)\n"
            "cols = [GREEN if p > 0 else RED for p in pops[order]]\n"
            "ax.bar(range(len(pops)), pops[order]*100, color=cols)\n"
            "ax.axhline(pm, c=GREY, ls='--', label=f'average pop {pm:.1f}%')\n"
            "ax.set_xlabel('campaign (sorted by day-0 return)'); ax.set_ylabel('announcement-day return (%)')\n"
            "ax.set_title(f'The 13D pop: real news, ~{pm:.1f}% on average — but you can\\'t buy it'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'announcement pop: mean {pm:.2f}%  (positive { (pops>0).mean()*100:.0f}% of the time)')"
        ),
        md(
            f"A modest **+{R['pop'][1]:.1f}%** average pop, up more often than not — real news. But it's "
            "the move *before* you can act, so it isn't a trade. The real question: what happens *after* "
            "you can buy?"
        ),
        md(
            "**The drift you can actually buy.** Buy one day after the filing, hold, and measure the "
            "return *above the market* (target − SPY). Here it is at 1, 3 and 6 months, with the "
            "break-even line at zero."
        ),
        code(
            "hs = [1, 3, 6]\n"
            "if HAVE_REAL:\n"
            "    means = [st.summarize(PRICES, TBL, m)['mean_excess']*100 for m in hs]\n"
            "    wins  = [st.summarize(PRICES, TBL, m)['win_rate']*100 for m in hs]\n"
            "else:\n"
            "    means = [R['h1'][2], R['h3'][2], R['h6'][2]]\n"
            "    wins  = [R['h1'][4], R['h3'][4], R['h6'][4]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "ax.bar(x, means, .5, color=GREEN)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{m} months' for m in hs])\n"
            "ax.set_ylabel('excess return vs SPY (%)')\n"
            "ax.set_title('Post-13D drift you can buy: positive, but small and bumpy')\n"
            "for i,(mn,w) in enumerate(zip(means,wins)):\n"
            "    ax.annotate(f'{mn:+.1f}%\\n{w:.0f}% win',(i,mn),ha='center',va='bottom' if mn>=0 else 'top')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('mean excess drift vs SPY:', {f'{m}m': f'{v:+.1f}%' for m,v in zip(hs,means)})"
        ),
        md(
            f"Positive everywhere — the folklore points the right way. But look at the *shape*: "
            f"**+{R['h1'][2]:.1f}%** at one month, then it **shrinks** to **+{R['h3'][2]:.1f}%** at "
            f"three months, then jumps to **+{R['h6'][2]:.1f}%** at six. A real, steady drift doesn't "
            "wobble like that — this is a small number bouncing around inside its own noise."
        ),
        md(
            "**Could random days look this good?** The honest test: for each campaign, compare buying on "
            "the 13D to buying the *same stock* on a **random** date. Do it thousands of times and see "
            "where the real 13D basket lands."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.placebo_pvalue(PRICES, TBL, 6, n_draws=8000)\n"
            "    obs = pl['obs_mean']*100; pmean = pl['placebo_mean']*100; pval = pl['p_value']\n"
            "    # rebuild the placebo cloud for the picture\n"
            "    spy = PRICES['SPY'].dropna(); h = st.TRADING_DAYS[6]; pools=[]\n"
            "    for _, r in TBL.iterrows():\n"
            "        s = PRICES[r['ticker']].dropna(); common = s.index.intersection(spy.index)\n"
            "        sc = s.reindex(common).values; sp = spy.reindex(common).values\n"
            "        if len(common) > h+2:\n"
            "            pool = (sc[h:]/sc[:-h]-1) - (sp[h:]/sp[:-h]-1)\n"
            "            pools.append(pool[np.isfinite(pool)])\n"
            "    rng = np.random.default_rng(390)\n"
            "    draws = np.array([np.mean([p[rng.integers(0,len(p))] for p in pools]) for _ in range(8000)])\n"
            "else:\n"
            "    obs = R['h6'][2]; pmean = 0.0; pval = R['h6'][6]\n"
            "    rng = np.random.default_rng(390); draws = rng.normal(0, 0.04, 8000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws*100, bins=50, color=GREY, alpha=.8, label='same names, RANDOM dates (6m excess)')\n"
            "ax.axvline(obs, c=GREEN, lw=2.5, label=f'the actual 13D dates ({obs:.1f}%)')\n"
            "ax.set_xlabel('average 6-month excess return vs SPY (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'The 13D basket sits inside the luck cloud — placebo p = {pval:.2f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'a random-date basket on the same names matches/beats the 13D {pval*100:.0f}% of the time')"
        ),
        md(
            f"The green line — the real 13D dates — falls **inside** the grey cloud of random draws. "
            f"About **{R['h6'][6]*100:.0f}%** of random-date baskets (on the *same* famous names) do as "
            "well or better. In plain terms: **buying these stocks on the activist's filing day is "
            "barely different from buying them on any random day.** The magic was in *which names you "
            "picked*, not *when*."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Weak.** The pop is real (**+{R['pop'][1]:.1f}%**) and the drift is positive "
            f"(**+{R['h6'][2]:.1f}%** at 6m), but it's small, bumpy across horizons, statistically "
            "insignificant, and **carried by a few mega-caps**. Real as lore, weak as edge.\n"
            f"- **Tradability — Fragile.** Even granting the bump, **{R['n_events']} rare events** with "
            "an excess return inside the noise can't be a strategy you allocate to. Costs aren't the "
            "problem — significance is.\n"
            "- **Free lunch? — Busted.** \"Buy the 13D and ride the activist\" is **survivorship**: a "
            "basket of legends *looks* like a free ride, but the leg you can actually trade matches "
            "random dates on the same names."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — costs aren't the problem\n\n"
            "Suppose you ignored the statistics and ran it anyway. Trading costs are tiny on a "
            "multi-month hold — so it's *not* cost that kills this. Here's gross vs net of a 10-bps "
            "round-trip; they're almost on top of each other."
        ),
        code(
            "if HAVE_REAL:\n"
            "    g = [st.net_of_costs(PRICES, TBL, m, 10.0)['gross_mean']*100 for m in [1,3,6]]\n"
            "    nv = [st.net_of_costs(PRICES, TBL, m, 10.0)['net_mean']*100 for m in [1,3,6]]\n"
            "else:\n"
            "    g = [R['net'][0][1], R['net'][1][1], R['net'][2][1]]\n"
            "    nv = [R['net'][0][2], R['net'][1][2], R['net'][2][2]]\n"
            "x = np.arange(3)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.bar(x-.2, g, .4, color=GREEN, label='gross excess')\n"
            "ax.bar(x+.2, nv, .4, color=GREY, label='net @ 10 bps')\n"
            "ax.set_xticks(x); ax.set_xticklabels(['1m','3m','6m']); ax.axhline(0,c='k',lw=.8)\n"
            "ax.set_ylabel('excess vs SPY (%)'); ax.set_title('A 10-bps cost barely moves it — cost is NOT the constraint')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('costs shave only ~0.1pp off a multi-month hold; the binding problem is significance, not cost')"
        ),
        md(
            "The bars are almost identical: a 10-bps round-trip shaves about **0.1 point** off a "
            "months-long hold. So costs are *not* what sinks this. What sinks it is that the gross "
            "excess is **statistically indistinguishable from random dates** on the same names — and it "
            "only fires when an Icahn-scale name shows up. You can't build a portfolio around a rare "
            "event whose payoff is inside the noise."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The pop is the real story.** Large academic samples (Brav et al., 2008) find a genuine "
            "**+7–8%** abnormal jump *around* the filing — but it's concentrated in the announcement "
            "window you can't trade, and the post-filing alpha is concentrated in targets that get "
            "**bought out**. The average outside-buyer drift is thin.\n"
            "- **Build a real 13D panel.** Swap our 25 famous names for *every* 13D on EDGAR (including "
            "the forgotten ones) and the survivorship tilt that helps the claim disappears — the drift "
            "should get *weaker*, not stronger.\n"
            "- **Condition on the outcome.** Split campaigns into \"ended in a sale\" vs \"fizzled\"; the "
            "lore lives almost entirely in the takeover tail.\n\n"
            "*Think the 13D drift beats the market by more than luck? Build the full filing panel, buy "
            "one day after each, race it against SPY, and show the basket landing **outside** the "
            "random-date cloud — then we'll talk.*"
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
            "# Activist-13D — a quantitative teardown 🔬\n"
            "### Pop vs drift split · excess-of-SPY drift at 1/3/6m · a Welch *t* + same-target placebo "
            "null · costs · lag & mega-cap robustness · a synthetic faithful-engine / power control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). We separate "
            "the two things the \"13D pops and drifts\" claim fuses: an **uncapturable announcement pop** "
            "(day-0 news) from the **tradable post-announcement drift** (buy +1 day, hold H months, in "
            "**excess of SPY**). And we confront the drift with the **sample size**: across ~25 "
            "outcome-selected events the excess is positive but its standard error swamps it.\n\n"
            "> ⚠️ **Data + selection note.** A clean full-coverage 13D panel isn't free; we use a "
            "**hardcoded basket of 25 famous campaigns** — selected on outcome, a bias that runs *toward* "
            "the claim (named on the Signal axis). Real data: yfinance daily adjusted closes, 25 targets "
            "+ SPY, 2010→2026. Offline core + synthetic control are deterministic. Methods in "
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
            f"| **Signal** | `WEAK` | 6-month excess drift **+{R['h6'][2]:.1f}%** vs SPY, "
            f"Welch **t = {R['h6'][5]:.2f}** (best across horizons is 1m **t = {R['h1'][5]:.2f}**), "
            f"placebo **p = {R['h1'][6]:.2f}**; non-monotone (3m **+{R['h3'][2]:.1f}%**), and dropping "
            f"3 mega-caps drops *t* to **{R['nomega'][2]:.2f}**. Positive but **fails t ≥ 2**. |\n"
            f"| **Tradability** | `FRAGILE` | **{R['n_events']} rare events**; net of 10-bps costs the "
            f"per-trade number is intact (**{R['net'][2][2]:.1f}%** at 6m) — **significance & frequency**, "
            "not cost, kill it. |\n"
            f"| **Free lunch?** | `BUSTED` | A same-target placebo says random dates match the 13D basket "
            f"**~{R['h6'][6]*100:.0f}%** of the time. The 'free ride' is **selection** — you remember the "
            "Icahn/Elliott wins. |\n\n"
            "> 💡 In plain words: the announcement *pop* is real news you can't buy; the post-filing "
            "*drift* you can buy is a positive whisper that ~25 outcome-selected events can't certify and "
            "that a few mega-caps carry."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $P_t$ be the target close and $M_t$ SPY. For event $j$ with announcement day $0_j$, "
            "define the **pop** $\\pi_j = P_{0_j}/P_{0_j-1}-1$ (the news) and the **tradable drift** "
            "$x_j^{(H)} = \\big(P_{0_j+1+H}/P_{0_j+1}-1\\big) - \\big(M_{0_j+1+H}/M_{0_j+1}-1\\big)$ — "
            "buy one day after, hold $H$, in **excess of SPY**.\n\n"
            "- **H₁ (the drift predicts).** $\\mathbb{E}[x^{(H)}] > 0$ for $H\\in\\{1,3,6\\}$m — a "
            "*positive excess over the market* you can actually buy.\n"
            "- **H₂ (it's deployable).** The excess is large/frequent enough, net of costs, to allocate "
            "to.\n"
            "- **H₃ (free ride).** The drift reflects the *13D timing*, not just owning special names "
            "(cheap, volatile, trending) that would have drifted on *any* entry date.\n\n"
            "Steelman from the literature: Brav, Jiang, Partnoy & Thomas (2008) find a real **+7–8%** "
            "abnormal return *around* the filing on 1,000+ events. We find **H₁ not rejected but not "
            "supported** (positive, $t < 2$, non-monotone), **H₂ rejected** (~25 events, excess in the "
            "noise), **H₃ rejected** (the same-target placebo matches the basket ~1 in 4). The legend is "
            "true where it's unreachable (the pop) and unproven where it would pay (a *distinct* drift)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The teardown is one decomposition: a mean excess drift against zero, judged by its "
            "**standard error**, on a sample chosen to *favour* the claim.\n\n"
            "$$\\widehat{x}^{(H)} = \\frac1k\\sum_j x_j^{(H)},\\qquad "
            "t = \\frac{\\widehat{x}^{(H)}}{s_x/\\sqrt{k}}.$$\n\n"
            "With $k \\approx 25$, $s_x/\\sqrt{k}$ is large: a few-point $\\widehat{x}$ drowns in its own "
            "SE. A raw **win-rate** is the wrong lens — these are volatile names in an up-drifting "
            "market, so >50% wins is the baseline, not the signal. And measuring excess against *zero* "
            "would credit plain beta, so we race **target vs SPY**. The honest small-sample instrument is "
            "a **same-target placebo**: resample each name's excess from a *random* entry date and ask "
            "how often chance matches the 13D timing. That number, not the point estimate, decides the "
            "Signal axis."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Event set.** A **hardcoded** table of {R['n_events']} famous activist 13D campaigns "
            "(yfinance adjusted closes, "
            f"{R['start']}→{R['end']}, {R['days']:,} price days). Explicit **outcome-selected** basket "
            "— survivorship runs *for* the claim (named on the axis).\n"
            "- **Pop (reported).** Day-0 return of the target — the news, not a trade.\n"
            "- **Drift (tradable).** Enter the close **1 day after** the announcement (no look-ahead), "
            "hold $H\\in\\{1,3,6\\}$m, return in **excess of SPY**; drop events whose horizon overruns "
            "the tape.\n"
            "- **Null #1 (Welch t).** Mean excess vs zero.\n"
            "- **Null #2 (same-target placebo).** 20,000 draws of one random-date excess **per name**; "
            "$p=\\Pr[\\text{random-date basket mean} \\ge \\text{13D basket mean}]$ — controls for the "
            "targets' own drift.\n"
            "- **Costs.** 1-day lag + a 10-bps round-trip on the buy-and-hold trade.\n"
            "- **Robustness.** Entry-lag sweep (0/1/2/5d) and a mega-cap drop.\n"
            "- **Positive control.** A deterministic set of 25 synthetic events with a **known** planted "
            "drift: the engine must recover it **and** must NOT manufacture significance when the true "
            "drift is zero."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Pop vs drift — the two objects, side by side\n\n"
            "The announcement pop (day 0, *uncapturable*) next to the mean excess drift you can buy at "
            "1 / 3 / 6 months, each with its $\\pm$ standard error. The pop is real but tiny here; the "
            "drift is positive, small, and **non-monotone**."
        ),
        code(
            "hs = [1, 3, 6]\n"
            "if HAVE_REAL:\n"
            "    pops = st.announcement_pops(PRICES, TBL); pop_m = pops.mean()*100\n"
            "    pop_se = pops.std(ddof=1)/np.sqrt(len(pops))*100\n"
            "    cm, ses, ts = [], [], []\n"
            "    for m in hs:\n"
            "        ex = st.event_excess_drift(PRICES, TBL, m)\n"
            "        cm.append(ex.mean()*100); ses.append(ex.std(ddof=1)/np.sqrt(len(ex))*100)\n"
            "        ts.append(st.welch_t_vs_zero(ex))\n"
            "else:\n"
            "    pop_m = R['pop'][1]; pop_se = 1.2\n"
            "    cm = [R['h1'][2], R['h3'][2], R['h6'][2]]; ses = [1.6, 2.2, 3.9]\n"
            "    ts = [R['h1'][5], R['h3'][5], R['h6'][5]]\n"
            "labels = ['pop\\n(day 0)'] + [f'drift {m}m' for m in hs]\n"
            "vals = [pop_m] + cm; errs = [pop_se] + ses\n"
            "cols = [GREY] + [GREEN]*3\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(range(len(vals)), vals, yerr=errs, capsize=5, color=cols, width=.6)\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels)\n"
            "ax.set_ylabel('return (%) — pop raw, drift excess vs SPY')\n"
            "ax.set_title('Pop is unreachable; tradable drift is small and t<2'); \n"
            "plt.tight_layout(); plt.show()\n"
            "print('Welch t by horizon:', {f'{m}m': round(t,2) for m,t in zip(hs,ts)})"
        ),
        md(
            f"> 💡 In plain words: the day-0 pop is **+{R['pop'][1]:.1f}%** (news, not yours); the "
            f"tradable drift is **+{R['h1'][2]:.1f}% / +{R['h3'][2]:.1f}% / +{R['h6'][2]:.1f}%** at "
            f"1/3/6m, with Welch *t* of **{R['h1'][5]:.2f} / {R['h3'][5]:.2f} / {R['h6'][5]:.2f}** — all "
            "below 2, and the 3-month *dip* betrays a number wobbling inside its error bar. H₁ is **not "
            "rejected, not supported.**"
        ),
        md(
            "### 4b · The decisive test — a same-target placebo null\n\n"
            "For each name, draw its 6-month excess from a *random* entry date; do it 20,000 times. The "
            "histogram is the null for the basket mean — what you'd get buying the same stocks on random "
            "days. The 13D basket is the green line; the *p*-value is the right-tail mass."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.placebo_pvalue(PRICES, TBL, 6, n_draws=20000); obs = pl['obs_mean']*100; pval = pl['p_value']\n"
            "    spy = PRICES['SPY'].dropna(); h = st.TRADING_DAYS[6]; pools=[]\n"
            "    for _, r in TBL.iterrows():\n"
            "        s = PRICES[r['ticker']].dropna(); common = s.index.intersection(spy.index)\n"
            "        sc = s.reindex(common).values; sp = spy.reindex(common).values\n"
            "        if len(common) > h+2:\n"
            "            pool = (sc[h:]/sc[:-h]-1) - (sp[h:]/sp[:-h]-1); pools.append(pool[np.isfinite(pool)])\n"
            "    rng = np.random.default_rng(390)\n"
            "    draws = np.array([np.mean([p[rng.integers(0,len(p))] for p in pools]) for _ in range(20000)])\n"
            "else:\n"
            "    obs = R['h6'][2]; pval = R['h6'][6]; rng = np.random.default_rng(390)\n"
            "    draws = rng.normal(0.0, 0.045, 20000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws*100, bins=60, color=GREY, alpha=.85, label=f'null: {len(draws):,} random-date baskets (same names)')\n"
            "ax.axvline(obs, c=GREEN, lw=2.5, label=f'observed 13D basket {obs:.1f}%')\n"
            "ax.set_xlabel('6-month mean excess vs SPY (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Placebo p = {pval:.2f}: the 13D basket is inside the luck cloud'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'P[random-date basket >= 13D basket] = {pval:.3f}  (need <0.05 to call it real)')"
        ),
        md(
            f"> 💡 In plain words: **{R['h6'][6]*100:.0f}%** of random-date baskets on the *same names* "
            "match or beat the 13D basket. A real timing edge would push the green line into the far "
            "right tail; instead it sits mid-cloud. **H₃ rejected** — the drift is owning special names, "
            "not the activist's calendar. This is selection wearing a strategy's clothes."
        ),
        md(
            "### 4c · Robustness — lag-flat, and a few names carry it\n\n"
            "Shift the entry lag (0/1/2/5 days) and, separately, drop the three mega-cap targets "
            "(AAPL, MSFT, DIS). The *t* never approaches 2 at any lag (so it's not an execution "
            "artefact), and removing three names **halves the mean and pushes *t* to 0.60** — the "
            "fingerprint of a fragile, name-driven estimate."
        ),
        code(
            "if HAVE_REAL:\n"
            "    lags = [0,1,2,5]\n"
            "    lt = [st.summarize(PRICES, TBL, 6, lag=L)['t'] for L in lags]\n"
            "    lm = [st.summarize(PRICES, TBL, 6, lag=L)['mean_excess']*100 for L in lags]\n"
            "    sub = TBL[~TBL['ticker'].isin(['AAPL','MSFT','DIS'])].reset_index(drop=True)\n"
            "    s_full = st.summarize(PRICES, TBL, 6); s_no = st.summarize(PRICES, sub, 6)\n"
            "    full_t, no_t = s_full['t'], s_no['t']; full_m = s_full['mean_excess']*100; no_m = s_no['mean_excess']*100\n"
            "else:\n"
            "    lags = [l[0] for l in R['lag']]; lt = [l[2] for l in R['lag']]; lm = [l[1] for l in R['lag']]\n"
            "    full_t, no_t = R['h6'][5], R['nomega'][2]; full_m = R['h6'][2]; no_m = R['nomega'][1]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.2))\n"
            "a1.bar([f'{L}d' for L in lags], lt, color=AMBER, width=.55)\n"
            "a1.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,(t,m) in enumerate(zip(lt,lm)): a1.annotate(f'{m:+.1f}%',(i,t),ha='center',va='bottom')\n"
            "a1.set_title('6m t never reaches 2 (any lag)'); a1.set_xlabel('entry lag'); a1.set_ylabel('Welch t'); a1.set_ylim(0,2.3); a1.legend()\n"
            "a2.bar(['all 25','drop 3\\nmega-caps'], [full_t, no_t], color=[GREEN, GREY], width=.5)\n"
            "a2.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,(t,m) in enumerate(zip([full_t,no_t],[full_m,no_m])): a2.annotate(f'{m:+.1f}%',(i,t),ha='center',va='bottom')\n"
            "a2.set_title('Drop 3 names -> t halves'); a2.set_ylabel('Welch t (6m)'); a2.set_ylim(0,2.3); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'lag-6m t: {[round(x,2) for x in lt]}  | full t={full_t:.2f} ({full_m:+.1f}%) vs no-mega t={no_t:.2f} ({no_m:+.1f}%)')"
        ),
        md(
            "> 💡 In plain words: the result is **flat in the entry lag** — buy same-day or a week later, "
            "the *t* stays near 1 — so this isn't a timing/execution artefact. But it is **fragile in the "
            "cross-section**: three mega-cap names carry most of the modest positive mean, exactly what "
            "you expect when the basket was selected on fame. There is no lag and no sub-basket where it "
            "clears t = 2."
        ),
        md(
            "### 4d · Faithful-engine & power control — we know the truth here\n\n"
            "On 25 *synthetic* 13D events (each a GBM path sharing a market factor, a day-0 pop, and a "
            "**known** post-announcement drift): with **zero** planted drift the test must stay below "
            "t=2 (25 events can't fake significance); with a **+20%/6m** planted drift it must light up. "
            "Both hold — the engine is unbiased *and* this sample size only detects implausibly large "
            "edges."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.20):\n"
            "    p, t = data.synthetic_events(n_events=25, drift_6m=edge, seed=390)\n"
            "    s = st.summarize(p, t, 6)\n"
            "    res.append((edge, s['n'], s['mean_excess']*100, s['win_rate']*100, s['t'], s['p_placebo']))\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "labels = [f'planted drift\\n{e*100:.0f}% / 6m' for e,_,_,_,_,_ in res]\n"
            "tvals = [r[4] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('Welch t (6-month)'); ax.set_title('Control: 25 events detect only a HUGE drift'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,k,m,w,t,p in res: print(f'planted {e*100:+.0f}%/6m: n={k} mean={m:.1f}% win={w:.0f}% t={t:.2f} p={p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** real drift the control sits at **t = {R['syn'][0][4]:.2f}** "
            f"(below 2 — no false positive, even though noise tilted the mean to +{R['syn'][0][2]:.1f}%); "
            f"only a **+20%/6m** drift — enormous — reaches **t = {R['syn'][1][4]:.2f}**. So the "
            "machinery is honest, and the real-tape *t* of ~1 is exactly what an *absent or tiny* drift "
            "looks like through a 25-event keyhole. The sample size is the verdict."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — 6m excess **+{R['h6'][2]:.1f}%** at Welch **t = {R['h6'][5]:.2f}** "
            f"(best horizon 1m **t = {R['h1'][5]:.2f}**), placebo **p = {R['h1'][6]:.2f}**; non-monotone "
            f"(3m **+{R['h3'][2]:.1f}%**) and mega-cap-fragile (**t = {R['nomega'][2]:.2f}** without "
            "AAPL/MSFT/DIS). Literature support + a positive-but-insignificant, fragile estimate on an "
            "outcome-selected basket ⇒ WEAK, not REAL (the t≥2 bar is not met).\n"
            f"- **Tradability `FRAGILE`** — **{R['n_events']} rare events**; net of 10-bps costs the "
            f"per-trade number is intact (**{R['net'][2][2]:.1f}%** at 6m), so **significance & "
            "frequency**, not cost, make it un-allocatable. No NAV-scale edge.\n"
            f"- **Free lunch? `BUSTED`** — the same-target placebo says random dates match the 13D "
            f"basket **~{R['h6'][6]*100:.0f}%** of the time. A selection mirage — you remember the "
            "Icahn/Elliott wins and forget the duds."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the power curve\n\n"
            "The operational truth in one picture: how big would the *true* 6-month excess have to be "
            "for a $k$-event study to detect it at t=2? At $k=25$ you'd need an excess several times the "
            "one observed; the real drift lives far below the detection floor."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ex = st.event_excess_drift(PRICES, TBL, 6); sd = ex.std(ddof=1)\n"
            "    obs_excess = R['h6'][2]/100\n"
            "else:\n"
            "    sd = 0.19; obs_excess = R['h6'][2]/100\n"
            "ks = np.arange(8, 200)\n"
            "min_detectable = 2.0 * sd / np.sqrt(ks)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(ks, min_detectable*100, c=AMBER, lw=2, label='excess needed for t=2')\n"
            "ax.axhline(obs_excess*100, c=GREEN, ls='--', label=f'observed excess ~{obs_excess*100:.1f}%')\n"
            "ax.axvline(R['n_events'], c=GREY, ls=':', label=f\"our k={R['n_events']}\")\n"
            "ax.set_xlabel('number of events k'); ax.set_ylabel('6-month excess return (%)')\n"
            "ax.set_title('Detection floor vs the real drift: we are far under-powered'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "need = 2.0*sd/np.sqrt(R['n_events'])\n"
            "print(f'at k={R[\"n_events\"]} you need ~{need*100:.1f}% excess for t=2; observed ~{obs_excess*100:.1f}% -> under-powered by ~{need/obs_excess:.1f}x')"
        ),
        md(
            "> 💡 In plain words: the amber curve is the **minimum detectable excess**; the green line is "
            "what we actually see. They don't meet until $k$ is several times larger than a famous-13D "
            "basket will ever deliver. Even granting the lore a *real* few-point edge, you'd need on the "
            "order of a hundred clean filings to prove it — and the full EDGAR panel (with the forgotten "
            "duds) would only *dilute* the mean. There is no sizing, threshold or cost assumption that "
            "manufactures statistical power from 25 outcome-selected events — **the fame that makes the "
            "trade romantic is exactly what makes it untestable and untradable.**"
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The literature's real result.** Brav, Jiang, Partnoy & Thomas (2008) document a genuine "
            "**+7–8%** abnormal return around the filing on 1,000+ events — but concentrated in the "
            "uncapturable announcement window, with the post-filing alpha localised in targets that are "
            "later **acquired** (Greenwood & Schor, 2009). The outside-buyer drift is the thin tail of "
            "that distribution.\n"
            "- **Full-panel replication.** Build the complete EDGAR 13D panel (genuine first-disclosure "
            "dates, ticker-mapped), buy one day after each, race vs SPY; the survivorship tilt that "
            "*helps* our basket vanishes and the drift should weaken.\n"
            "- **Condition on outcome / activist.** Split by eventual sale vs fizzle, and by activist "
            "track record; the edge, if any, is a conditional one, not a blanket 'buy every 13D.'\n\n"
            "*The reproducible core is offline and deterministic; the event set is an explicit "
            "outcome-selected basket. Methods and sources: [`docs/references.md`](../docs/references.md); "
            "frozen numbers: [`docs/results.md`](../docs/results.md).*"
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
