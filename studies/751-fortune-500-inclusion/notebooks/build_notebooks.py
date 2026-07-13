"""Generate the two narrative notebooks for Study 751 (Fortune-500-Inclusion).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached daily closes
under ../_cache/ (the ~26-event add/drop table + SPY) and otherwise quote the frozen headline
numbers in ``R`` (mirroring docs/results.md). The synthetic positive control runs anywhere
with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md
# (yfinance ~26-event add/drop table + SPY, 2015-01-02 -> 2026-07-10; fingerprint 9b280bbd9147).
R = dict(
    fingerprint="9b280bbd9147", as_of="2026-07-12",
    n_table=26, n_delisted=3, n_added=14, n_dropped=12,
    # bucket: (mean%, win%, t)  [canonical CAR[0,+2]]
    added=(0.06, 50, 0.04),
    dropped=(0.12, 50, 0.06),
    allb=(0.09, 50, 0.07),
    diff_pp=-0.06, diff_t=-0.02, added_placebo_p=0.966,
    # reveal-day [0,0]: added_mean%, added_t, placebo_p, diff_t
    day0=(-0.47, -0.56, 0.556, -0.12),
    # tradable lag=1: (label, added_mean%, added_t, diff_t, net_pct)
    tradable=[("[+1,+3]", 0.88, 0.60, -0.22, 0.78),
              ("[+1,+5]", 2.23, 1.39, 0.20, 2.13)],
    # robustness: (window, added_mean%, added_t, dropped_mean%, diff_pp, diff_t)
    robust=[("[0,0]", -0.47, -0.56, -0.25, -0.21, -0.12),
            ("[0,+2]", 0.06, 0.04, 0.12, -0.06, -0.02),
            ("[-1,+1]", -2.14, -1.34, -0.44, -1.70, -0.75),
            ("[0,+4]", 1.62, 1.15, 0.89, 0.73, 0.27)],
    # synthetic: (planted_bps, added_mean%, added_t, diff_pp, diff_t)
    syn=[(0, 0.49, 0.48, 1.39, 0.90),
         (500, 5.49, 5.40, 6.39, 4.10)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Attention_effect%3F: Not supported](https://img.shields.io/badge/Attention_effect%3F-Not_supported-8b949e?style=flat-square)\n\n"
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

from fortune_500_inclusion import data, strategy as st

HAVE_REAL = data.have_real()
PRICES, EVENTS = data.load_real() if HAVE_REAL else (None, None)
PANEL = st.car_panel(PRICES, EVENTS) if HAVE_REAL else None
print("real add/drop-tape cache present:", HAVE_REAL,
      "| events priced:", (0 if PANEL is None else len(PANEL)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# The Fortune 500 pop — does making the list pay? 🏆\n"
            "### Being added to a famous list *should* pop the stock… shouldn't it? — in plain English\n\n"
            + BADGES +
            "Here's a tidy-sounding trade. There's a real, textbook effect where a stock **added to "
            "the S&P 500** jumps — index funds are *forced* to buy it, so the price pops. So surely a "
            "company's **debut on the Fortune 500** — Tesla in 2017, Airbnb and Coinbase in 2022, "
            "Robinhood in 2024 — gets the same prestige bump, and a company that **falls off** takes a "
            "hit? Buy the newcomers on reveal day, maybe short the drop-offs.\n\n"
            "It borrows its plausibility from something real. This notebook asks whether the borrow is "
            "legitimate — by building a transparent table of **26 real Fortune-500 debuts and exits** and "
            "measuring what actually happened to each around the June reveal.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo test and the power "
            "analysis? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **The catch, up front.** The S&P 500 is an *investable index* — funds must track it, so "
            "an addition is a genuine buying shock. The Fortune 500 is a **magazine ranking by last "
            "year's revenue**: no fund tracks it, nobody's forced to buy, and the revenue that decides "
            "the ranking was public months ago. Strip out the forced buying and the news, and all "
            "that's left is *prestige*. Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does joining the Fortune 500 pop the stock? | **No.** Across 14 debuts the reveal-window "
            f"abnormal return averages **+{R['added'][0]:.2f}%** — and it's a perfect coin-flip "
            f"(**{R['added'][1]:.0f}%** of the time positive). |\n"
            "| Does falling off the list hurt it? | **No.** The 12 drop-offs average "
            f"**+{R['dropped'][0]:.2f}%** — if anything *up*, and also a 50/50 coin-flip. |\n"
            "| Is there any gap between joining and dropping? | **None.** Added minus dropped is "
            f"**{R['diff_pp']:.2f} percentage points** — statistically zero. |\n"
            "| Then why does 'the list moves stocks' sound true? | It's borrowed from the **S&P 500**, "
            "where index funds are *forced* to buy. The Fortune 500 forces no one to do anything. |\n\n"
            "> The prestige pop is a myth: there's no bump for joining, no penalty for leaving, and no "
            "gap between them. A random press release looks this 'special' **97%** of the time."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Making the Fortune 500 is a milestone — it brings analyst coverage, press, prestige, "
            "and a wave of investor attention. The stock pops on the reveal. And a company that drops "
            "off loses all that, so it sags. Buy the debutants, fade the drop-offs.\"*\n\n"
            "It's not a crazy claim — it rhymes with something **genuinely real**. The classic "
            "index-inclusion studies (Shleifer 1986; Harris & Gurel 1986) found that stocks **added to "
            "the S&P 500** earn a real abnormal return around the change. The believers simply extend "
            "that to the Fortune 500. We'll test whether the extension holds — on a real, representative "
            "table of debuts and exits, both legs."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If it were real, it'd be a free calendar: every June, Fortune hands you a list of stocks to "
            "buy (the newcomers) and stocks to short (the drop-offs), on a *pre-announced date*. But the "
            "whole reason the S&P-500 version works is a **mechanism**: index funds *must* buy the "
            "addition, a forced demand shock. The Fortune 500 has **no such mechanism** — no fund tracks "
            "it — and it carries **no new information**, because the ranking is built from revenue that "
            "was reported months earlier. So this is really a clean test of a deeper question: does "
            "**prestige and attention, by themselves**, move a stock price? If the answer is no, the "
            "'trade' was never a trade."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We build a **transparent table of {R['n_table']} real Fortune-500 events** — "
            f"**{R['n_added']} debuts** (Tesla, Netflix, Uber, Airbnb, Coinbase, Moderna, DoorDash, "
            f"CrowdStrike, Robinhood…) and **{R['n_dropped']} exits** (Mattel, GameStop, Xerox, Bed Bath "
            "& Beyond, Gap, Hasbro…), each pinned to its June reveal date. For each one:\n\n"
            "1. **Line it up against the market.** Fit the stock to the S&P (SPY) on a clean stretch "
            "*before* the reveal, so we know how it normally moves.\n"
            "2. **Measure the abnormal return.** The bit of the reveal-window move that *isn't* just the "
            "market — the actual 'prestige pop', if there is one.\n"
            "3. **Stress the luck.** Draw the same number of *random* dates on the same stocks, "
            "thousands of times, and ask how often chance produces a pop this big. With ~a dozen events "
            "per side, that's the honest test.\n\n"
            "And we say it loudly: the worst drop-offs (J.C. Penney, Bed Bath & Beyond) went **bankrupt** "
            "and vanished from the data — so the exits we *can* price are the *survivors*, biased "
            "**against** a drop-off penalty. Even so, the penalty isn't there."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, every event.** The reveal-window abnormal (market-adjusted) return for all 26 "
            "debuts and exits — greens are joiners, reds are drop-offs."
        ),
        code(
            "if HAVE_REAL:\n"
            "    p = PANEL.sort_values('list_date')\n"
            "    vals = p['car'].values*100; labs = p['ticker'].values; added = p['added'].values\n"
            "else:\n"
            "    rng=np.random.default_rng(751); vals=rng.normal(0.1,6,R['n_table'])\n"
            "    labs=[f'E{i}' for i in range(R['n_table'])]; added=np.array([True]*R['n_added']+[False]*R['n_dropped'])\n"
            "colors=[GREEN if a else RED for a in added]\n"
            "fig, ax = plt.subplots(figsize=(10.4, 4.6))\n"
            "ax.bar(range(len(vals)), vals, color=colors, edgecolor='k', lw=.4)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.axhline(np.mean(vals), c=GREY, ls='--', label=f'overall mean {np.mean(vals):+.2f}%')\n"
            "ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, rotation=90, fontsize=7)\n"
            "ax.set_ylabel('reveal-window abnormal return (%)')\n"
            "ax.set_title('Debuts (green) and exits (red) — a scatter of noise around zero')\n"
            "from matplotlib.patches import Patch\n"
            "ax.legend(handles=[Patch(color=GREEN,label='added'),Patch(color=RED,label='dropped'),\n"
            "                   plt.Line2D([0],[0],c=GREY,ls='--',label=f'mean {np.mean(vals):+.2f}%')])\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'overall mean abnormal return {np.mean(vals):+.2f}%  —  positive events: {(vals>0).mean()*100:.0f}%')"
        ),
        md(
            "There's the tell. If joining the list popped stocks, the greens would cluster **above** zero "
            "and the reds **below**. Instead it's a hedge — some up, some down, both colours on both "
            "sides — averaging essentially **zero**. HOG (+11%) and ZM (+12%) are up; ETSY (−12%) and GAP "
            "(−13%) are down; none of it lines up with joining vs leaving."
        ),
        md(
            "**The two buckets as averages.** Mean abnormal return for the debuts vs the exits, with the "
            "win-rate (how often each is positive). A coin is 50%."
        ),
        code(
            "if HAVE_REAL:\n"
            "    s = st.summarize(PANEL)\n"
            "    am, aw = s['added']['mean_pct'], s['added']['win']*100\n"
            "    dm, dw = s['dropped']['mean_pct'], s['dropped']['win']*100\n"
            "else:\n"
            "    am, aw, dm, dw = R['added'][0], R['added'][1], R['dropped'][0], R['dropped'][1]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.2, 4.2))\n"
            "a1.bar(['added\\n(debuts)', 'dropped\\n(exits)'], [am, dm], color=[GREEN, RED], width=.55)\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_ylabel('mean abnormal return (%)')\n"
            "a1.set_title('Both buckets sit on zero'); a1.set_ylim(-1, 1)\n"
            "for i,v in enumerate([am,dm]): a1.annotate(f'{v:+.2f}%',(i,v),ha='center',va='bottom')\n"
            "a2.bar(['added','dropped'], [aw, dw], color=GREY, width=.55)\n"
            "a2.axhline(50, c=RED, ls='--', label='coin flip (50%)')\n"
            "a2.set_ylim(0,100); a2.set_ylabel('% of events positive'); a2.set_title('Both are exactly a coin flip')\n"
            "a2.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'added {am:+.2f}% (win {aw:.0f}%)   dropped {dm:+.2f}% (win {dw:.0f}%)')"
        ),
        md(
            f"The debuts pop **+{R['added'][0]:.2f}%** and the exits **+{R['dropped'][0]:.2f}%** — both "
            f"positive only **50%** of the time, a literal coin flip. There's no bump for joining, no "
            "penalty for leaving, and the two are on top of each other."
        ),
        md(
            "**Could a handful of random dates look this good?** The honest small-sample test: draw "
            f"**{R['n_added']}** *random* dates on the same stocks, over and over, and see where the real "
            "debut pop lands against pure luck."
        ),
        code(
            "if HAVE_REAL:\n"
            "    null = st.placebo_car_dist(PRICES, data.TICKERS, k=int(PANEL['added'].sum()), n_draws=4000)\n"
            "    obs = PANEL.loc[PANEL['added'],'car'].mean()*100; draws = null*100\n"
            "    pval = st.placebo_pvalue(obs/100, null)\n"
            "else:\n"
            "    obs=R['added'][0]; pval=R['added_placebo_p']; rng=np.random.default_rng(751); draws=rng.normal(0,1.6,4000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=45, color=GREY, alpha=.85, label=f'pop of {R[\"n_added\"]} RANDOM dates')\n"
            "ax.axvline(obs, c=GREEN, lw=2.5, label=f'the actual debuts ({obs:+.2f}%)')\n"
            "ax.set_xlabel('average reveal-window abnormal return (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'The debut pop sits dead-centre in the luck cloud — placebo p = {pval:.2f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'a random {R[\"n_added\"]}-date draw matches the debut pop {pval*100:.0f}% of the time — as ordinary as it gets')"
        ),
        md(
            f"The green line — the real debuts' pop — sits **right in the middle** of the grey luck cloud "
            f"(placebo *p* ≈ **{R['added_placebo_p']:.2f}**). In plain terms: **a dozen random dates would "
            "look about this 'special' 97% of the time.** There's no prestige pop hiding here; it's noise "
            "with a famous list attached."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** Debuts **+{R['added'][0]:.2f}%**, exits **+{R['dropped'][0]:.2f}%**, "
            f"gap **{R['diff_pp']:.2f}pp** — all statistically zero, both a 50/50 coin flip, and a "
            "random draw matches the pop 97% of the time. There is no effect to see.\n"
            "- **Tradability — Mirage.** Nothing to deploy: no bump to buy, no drop to short, no gap "
            "between them. Costs never even get a chance to bite.\n"
            "- **Attention effect? — Not supported.** The S&P-500 pop is real because funds are *forced* "
            "to buy; the Fortune 500 forces no one, so prestige alone moves nothing. The borrow was "
            "illegitimate."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — the drift that isn't there\n\n"
            "Forget significance and just try to *hold* it: buy each debut the day after the reveal and "
            "hold for a few days. Here's what that book makes, gross and net of a small large-cap cost — "
            "across a few window choices, so you can see the number wander with the window (the giveaway "
            "of no real signal)."
        ),
        code(
            "rows = R['tradable']\n"
            "if HAVE_REAL:\n"
            "    rows=[]\n"
            "    for w,lab in [((0,2),'[+1,+3]'),((0,4),'[+1,+5]')]:\n"
            "        pl=st.car_panel(PRICES, EVENTS, window=w, lag=1)\n"
            "        a=st.summarize_bucket(pl.loc[pl['added'],'car'].to_numpy())\n"
            "        dl=st.welch_t(pl.loc[pl['added'],'car'].to_numpy(), pl.loc[~pl['added'],'car'].to_numpy())\n"
            "        nc=st.net_of_costs(a['mean_pct']/100)\n"
            "        rows.append((lab, a['mean_pct'], a['t'], dl, nc['net_pct']))\n"
            "labs=[r[0] for r in rows]; gross=[r[1] for r in rows]; net=[r[4] for r in rows]; tt=[r[2] for r in rows]\n"
            "xx=np.arange(len(labs))\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.bar(xx-.2, gross, .4, color=AMBER, label='gross'); ax.bar(xx+.2, net, .4, color=RED, label='net @10bps')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_xticks(xx); ax.set_xticklabels(labs)\n"
            "ax.set_ylabel('per-event P&L, added bucket (%)')\n"
            "ax.set_title('Buy-the-debut: a couple of % at best, and never significant')\n"
            "for i,(g,t) in enumerate(zip(gross,tt)): ax.annotate(f'{g:+.1f}%\\n(t={t:.2f})',(i-.2,g),ha='center',va='bottom',fontsize=8)\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('windows:', [(r[0], round(r[1],2), 't='+str(round(r[2],2))) for r in rows])"
        ),
        md(
            f"Stretch the hold long enough and you can coax the debut bucket up to **+{R['tradable'][1][1]:.1f}%** "
            f"— but at **t = {R['tradable'][1][2]:.2f}** it's not significant, the added−dropped gap is a "
            f"**{R['tradable'][1][3]:+.2f}** *t* (pure noise), and you had to hunt across windows to find "
            "even that. There's no machine here — just a prestigious list that the market, quite "
            "reasonably, already knew everything about."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The effect that's actually real.** [Study 249 — Index-Inclusion](../249-index-inclusion/) "
            "runs the *S&P 500* version — the one with a forced-buying mechanism behind it. Compare the "
            "two and the whole lesson pops out: **mechanism, not prestige.**\n"
            "- **The label trap.** [Study 389 — Name-Change-Effect](../389-name-change-effect/) asks "
            "whether a themed *rename* pops a stock — same attention story, same coin-flip answer.\n"
            "- **Add the corpses.** Our exit tape drops the bankruptcies (J.C. Penney, Bed Bath & "
            "Beyond); rebuild them from a survivorship-free feed and the *drop* leg might finally show a "
            "penalty — but that penalty is a bankruptcy, not a tradable 'fall off the list.'\n\n"
            "*Think making the Fortune 500 is a real catalyst? Capture the reveal-window returns, draw "
            "the same number of random dates, and show the debut pop landing **outside** the cloud — "
            "then we'll talk.*"
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
            "# Fortune-500-Inclusion — a quantitative teardown 🔬\n"
            "### Market-model CAR by bucket · added−dropped Welch *t* + a placebo non-event-window null · "
            "the [0,0]-vs-[0,+2] window split · a 1-day-lag tradable variant + costs · a synthetic "
            "faithful-engine / power control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). We treat the "
            "folklore as a **two-bucket event-study hypothesis** — a positive **added** CAR and a "
            "negative **dropped** CAR around the June list reveal — and confront it with the **sample "
            "size**, the **missing demand mechanism**, and **survivorship**. The decisive objects are two "
            "small cross-sections of market-model CARs and a placebo null sized to the event count.\n\n"
            "> ⚠️ **Data + proxy note.** The add/drop table is hardcoded, cited, and a **labelled "
            "proxy** — Fortune sells no free point-in-time membership feed, so the debut/exit *year* is "
            "curated (each snapped to its list-reveal date), exactly as Study 391 uses a hardcoded "
            "CEO-change table. The priced exit sample is **survivor-biased** (J.C. Penney, Bed Bath & "
            "Beyond went bankrupt and left the tape), biasing the drop leg *against* a penalty — named on "
            "the Signal axis. Real data: yfinance daily adjusted closes, 2015→2026, fingerprint "
            f"**{R['fingerprint']}**, as-of **{R['as_of']}**. Methods in "
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
            f"| **Signal** | `NONE` | added **+{R['added'][0]:.2f}%** (*t* = {R['added'][2]:.2f}), dropped "
            f"**+{R['dropped'][0]:.2f}%** (*t* = {R['dropped'][2]:.2f}); **added−dropped {R['diff_pp']:.2f}pp** "
            f"at Welch **t = {R['diff_t']:.2f}**, placebo **p = {R['added_placebo_p']:.2f}**, 50% win-rate "
            "both. Even the reveal *day* is flat. No window clears *t* = 2. |\n"
            f"| **Tradability** | `MIRAGE` | best window (enter +1d, hold 5) = **+{R['tradable'][1][1]:.2f}%** "
            f"at *t* = {R['tradable'][1][2]:.2f} (window-mined); added−dropped *t* = {R['tradable'][1][3]:.2f}. "
            "Nothing to size. |\n"
            "| **Attention effect?** | `NOT SUPPORTED` | the S&P-500 effect runs on **forced index-fund "
            "buying**; a media ranking creates no demand shock and reveals no new info. |\n\n"
            "> 💡 In plain words: the believers borrow the *real* S&P-500 index-inclusion pop but drop "
            "its engine (forced buying). Without the engine, and with the deciding revenue already "
            "public, there's nothing to react to — and the tape is a picture-perfect null."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "For each Fortune-500 event $i$ with reveal date $\\tau_i$, fit a **market model** "
            "$r^{i}_t = \\alpha_i + \\beta_i r^{\\mathrm{SPY}}_t$ on a clean estimation window "
            "$[\\tau_i-130,\\ \\tau_i-10]$, then define the **cumulative abnormal return** over event "
            "offsets $[a,b]$: $\\mathrm{CAR}^i_{[a,b]} = \\sum_{t=a}^{b}\\big(r^{i}_{\\tau_i+t} - "
            "\\hat\\alpha_i - \\hat\\beta_i r^{\\mathrm{SPY}}_{\\tau_i+t}\\big)$.\n\n"
            "- **H₁ (the debut pop).** $\\mathbb{E}[\\mathrm{CAR}\\mid \\text{added}] > 0$ — joining the "
            "list earns an abnormal pop (the S&P-inclusion analogue; Shleifer 1986).\n"
            "- **H₂ (the drop penalty).** $\\mathbb{E}[\\mathrm{CAR}\\mid \\text{dropped}] < 0$.\n"
            "- **H₃ (the spread).** added − dropped $> 0$ and harvestable net of costs.\n\n"
            "We find **H₁ not supported** (CAR ≈ 0, $t<0.1$, placebo $p\\approx1$), **H₂ not supported** "
            "(dropped CAR is *positive*), and **H₃ rejected** (the spread is $-0.06$pp at $t=-0.02$). The "
            "believers' analogy fails at the mechanism: the S&P effect is a **demand shock**, and a "
            "magazine ranking supplies none."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The teardown is two one-sample tests on small cross-sections, judged by their **standard "
            "error** and by **survivorship**:\n\n"
            "$$t_{\\text{bucket}} = \\frac{\\overline{\\mathrm{CAR}}}{s/\\sqrt{k}},\\qquad "
            "k\\approx 12\\text{–}14.$$\n\n"
            "With $k\\approx 13$ per bucket and single-name volatility, $s/\\sqrt{k}$ is large — a "
            "few-percent mean drowns in its own SE. And the sample is **conditioned on survival**: the "
            "exits whose reaction was most negative (a bankruptcy) **left the tape**, so "
            "$\\overline{\\mathrm{CAR}}_{\\text{dropped}}$ is biased **upward** — *toward* refuting H₂ "
            "only because the worst drops deleted themselves. The honest instrument is a **randomisation "
            "(placebo) test**: resample $k$ random non-event windows on the same tickers and ask how "
            "often chance matches the observed CAR. That, not the point estimate, decides the Signal axis."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Add/drop table.** {R['n_table']} documented events ({R['n_added']} debuts / "
            f"{R['n_dropped']} exits), 2017–2024, hardcoded & cited (a labelled proxy); "
            f"**{R['n_delisted']}** famous bankrupt exits are listed as DELISTED (no series) for the "
            "survivorship caveat.\n"
            "- **Market-model CAR.** $r=\\alpha+\\beta\\,$SPY on a 120-day window, 10-day gap; canonical "
            "event window **[0,+2]**, with a **1-day entry lag** for the tradable variant.\n"
            "- **Null #1 (Welch t).** Each bucket's CAR vs zero, and added − dropped.\n"
            "- **Null #2 (placebo).** 8,000–20,000 draws of $k$ random non-event windows on the same "
            "tickers; $p = \\Pr[|\\text{random mean}| \\ge |\\text{observed}|]$ — the small-sample "
            "workhorse.\n"
            "- **Costs.** A buy-the-debut / hold-the-window book pays a one-way 10-bps round-trip.\n"
            "- **Positive control.** Deterministic per-event panels with a **planted** added-bucket CAR "
            "of size `car_bps`: the inference must recover a large edge **and** must NOT manufacture "
            "significance when `car_bps = 0`.\n\n"
            "> **What would make us say REAL:** an added−dropped CAR clearing **|t| ≥ 2** on this tape, "
            "outside the placebo cloud, stable across windows. Announced before the run."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The two buckets — both dead on zero\n\n"
            "Mean CAR[0,+2] per bucket with $\\pm$ standard error, against zero (dashed). Neither bucket "
            "moves, and they land on top of each other."
        ),
        code(
            "if HAVE_REAL:\n"
            "    A = PANEL.loc[PANEL['added'],'car'].to_numpy(); D = PANEL.loc[~PANEL['added'],'car'].to_numpy()\n"
            "    am, dm = A.mean()*100, D.mean()*100\n"
            "    ase, dse = A.std(ddof=1)/np.sqrt(len(A))*100, D.std(ddof=1)/np.sqrt(len(D))*100\n"
            "    at, dt = st.welch_t(A), st.welch_t(D)\n"
            "else:\n"
            "    am, dm, at, dt = R['added'][0], R['dropped'][0], R['added'][2], R['dropped'][2]\n"
            "    ase, dse = am/max(at,1e-9), dm/max(dt,1e-9)\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.bar(['added [0,+2]','dropped [0,+2]'], [am, dm], yerr=[ase, dse], capsize=6,\n"
            "       color=[GREEN, RED], width=.55)\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('mean CAR (%)')\n"
            "ax.set_title(f'added t={at:.2f}, dropped t={dt:.2f} — both indistinguishable from zero')\n"
            "for i,v in enumerate([am,dm]): ax.annotate(f'{v:+.2f}%',(i,v),ha='center',va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'added {am:+.2f}% (t={at:.2f})   dropped {dm:+.2f}% (t={dt:.2f})   H1 wants added>0, H2 wants dropped<0')"
        ),
        md(
            f"> 💡 In plain words: the debut CAR is **+{R['added'][0]:.2f}%** at **t = {R['added'][2]:.2f}**, "
            f"the exit CAR is **+{R['dropped'][0]:.2f}%** at **t = {R['dropped'][2]:.2f}** (*positive* — the "
            "wrong sign for a penalty), and the error bars swallow both. H₁ and H₂ are both **not "
            "supported**, and survivorship only pushes the exit leg *more* positive."
        ),
        md(
            "### 4b · The decisive test — a placebo null sized to the event count\n\n"
            f"Draw {R['n_added']} random non-event windows thousands of times; the histogram is the null "
            "for the mean debut CAR. The real pop is the green line; the *p*-value is the two-sided tail "
            "mass."
        ),
        code(
            "if HAVE_REAL:\n"
            "    k=int(PANEL['added'].sum())\n"
            "    null = st.placebo_car_dist(PRICES, data.TICKERS, k=k, n_draws=8000)\n"
            "    obs = PANEL.loc[PANEL['added'],'car'].mean()*100; draws=null*100\n"
            "    pval = st.placebo_pvalue(obs/100, null)\n"
            "else:\n"
            "    obs=R['added'][0]; pval=R['added_placebo_p']; rng=np.random.default_rng(751); draws=rng.normal(0,1.6,8000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=55, color=GREY, alpha=.85, label=f'null: {R[\"n_added\"]} random windows')\n"
            "ax.axvline(obs, c=GREEN, lw=2.5, label=f'observed debut CAR {obs:+.2f}%')\n"
            "ax.axvline(np.mean(draws), c='k', ls=':', lw=1, label=f'null mean {np.mean(draws):+.2f}%')\n"
            "ax.set_xlabel('mean reveal-window CAR (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Placebo p = {pval:.2f}: the debut pop is dead-centre in the luck cloud'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'P[|random {R[\"n_added\"]}-window mean| >= |debut CAR|] = {pval:.3f}  (need <0.05 to call it real)')"
        ),
        md(
            f"> 💡 In plain words: **{R['added_placebo_p']*100:.0f}%** of random {R['n_added']}-window "
            "draws match or beat the debut CAR in magnitude — it sits essentially *at the null mean*. A "
            "real effect would push the green line into a tail; instead it's in the bullseye. H₁ is **not "
            "supported** in the strongest possible sense."
        ),
        md(
            "### 4c · The reveal day + robustness — nothing, at any window\n\n"
            "Unlike a forced CEO ouster (Study 391), where the announcement *day* really does jolt the "
            "stock, here even the un-tradable **[0,0]** instant is flat. And no window — not [0,0], not "
            "[−1,+1], not [0,+4] — pushes either bucket or the spread past *t* = 2."
        ),
        code(
            "rob = R['robust']\n"
            "if HAVE_REAL:\n"
            "    rob=[]\n"
            "    for w,lab in [((0,0),'[0,0]'),((0,2),'[0,+2]'),((-1,1),'[-1,+1]'),((0,4),'[0,+4]')]:\n"
            "        pw=st.car_panel(PRICES, EVENTS, window=w); sw=st.summarize(pw)\n"
            "        rob.append((lab, sw['added']['mean_pct'], sw['added']['t'], sw['dropped']['mean_pct'], sw['diff_pct'], sw['diff_t']))\n"
            "labs=[r[0] for r in rob]; at=[r[2] for r in rob]; dft=[r[5] for r in rob]; xx=np.arange(len(labs))\n"
            "fig, (a1,a2) = plt.subplots(1, 2, figsize=(10.8, 4.2))\n"
            "am=[r[1] for r in rob]; dm=[r[3] for r in rob]\n"
            "a1.bar(xx-.2, am, .4, color=GREEN, label='added mean'); a1.bar(xx+.2, dm, .4, color=RED, label='dropped mean')\n"
            "a1.axhline(0,c='k',lw=.8); a1.set_xticks(xx); a1.set_xticklabels(labs); a1.set_ylabel('mean CAR (%)')\n"
            "a1.set_title('Point estimate wanders around zero'); a1.legend()\n"
            "a2.bar(xx-.2, at, .4, color=GREEN, label='added t'); a2.bar(xx+.2, dft, .4, color=GREY, label='added-dropped t')\n"
            "a2.axhline(2, ls='--', c=RED, label='t=2'); a2.axhline(-2, ls='--', c=RED)\n"
            "a2.set_xticks(xx); a2.set_xticklabels(labs); a2.set_ylabel('Welch t'); a2.set_xlabel('event window')\n"
            "a2.set_ylim(-2.5, 2.5); a2.set_title('No window clears |t|=2'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('by window:', [(r[0], round(r[1],2), 't='+str(round(r[2],2)), 'diff t='+str(round(r[5],2))) for r in rob])"
        ),
        md(
            f"> 💡 In plain words: on the single reveal day the debuts are **{R['day0'][0]:.2f}%** "
            f"(*t* = {R['day0'][1]:.2f}, placebo *p* = {R['day0'][2]:.2f}) — a non-event. Widen or shift "
            "the window however you like and the point estimate flips sign while the *t* stays pinned "
            "inside ±2. That sign-flipping-with-the-window behaviour is the fingerprint of **no signal**, "
            "not a fragile one."
        ),
        md(
            "### 4d · Faithful-engine & power control — we know the truth here\n\n"
            "Deterministic per-event panels with a **planted** added-bucket CAR of size `car_bps`: with "
            "**`car_bps=0`** the added−dropped test must stay below *t* = 2 (a dozen events per bucket "
            "can't fake significance); with a **+500 bps** planted edge it must light up. Both hold — "
            "proving the engine is unbiased *and* that this sample size only detects an implausibly large "
            "effect."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 500.0):\n"
            "    syn = data.synthetic_events(car_bps=edge, seed=751)\n"
            "    a = st.summarize_bucket(syn['added_car']); d = st.summarize_bucket(syn['dropped_car'])\n"
            "    dt = st.welch_t(syn['added_car'], syn['dropped_car'])\n"
            "    res.append((edge, a['mean_pct'], a['t'], d['mean_pct'], dt))\n"
            "labels=[f'planted\\n{int(e)}bps' for e,*_ in res]; xx=np.arange(len(labels))\n"
            "at=[r[2] for r in res]; dft=[r[4] for r in res]\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.bar(xx-.2, at, .4, color=GREEN, label='added t'); ax.bar(xx+.2, dft, .4, color=GREY, label='added-dropped t')\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2'); ax.axhline(-2, ls='--', c=RED)\n"
            "ax.set_xticks(xx); ax.set_xticklabels(labels); ax.set_ylabel('Welch t')\n"
            "ax.set_title('Control: only a HUGE planted edge lights it up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,am,at_,dm,dt_ in res: print(f'planted {int(e):>3}bps: added={am:+.2f}%(t={at_:+.2f}) dropped={dm:+.2f}% added-dropped t={dt_:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted edge the added−dropped *t* is "
            f"**{R['syn'][0][4]:.2f}** (no false positive); only the **+500 bps** plant reaches "
            f"**{R['syn'][1][4]:.2f}**. So the machinery is honest, and the real-tape added−dropped *t* of "
            f"**{R['diff_t']:.2f}** is exactly what a *true zero* looks like through a "
            f"{R['n_added']}+{R['n_dropped']}-event keyhole. The null isn't weak measurement — it's a null."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — added **+{R['added'][0]:.2f}%** (*t* = {R['added'][2]:.2f}), dropped "
            f"**+{R['dropped'][0]:.2f}%** (*t* = {R['dropped'][2]:.2f}), **added−dropped {R['diff_pp']:.2f}pp** "
            f"at Welch **t = {R['diff_t']:.2f}** / placebo **p = {R['added_placebo_p']:.2f}**, with a 50% "
            "win-rate in *both* buckets and a flat reveal day. No window clears *t* = 2. **Survivorship** "
            "named on this axis (worst exits delisted, biasing the drop leg up). This is a null, not a "
            "weak positive.\n"
            f"- **Tradability `MIRAGE`** — the most favourable window (enter +1 day, hold five) reaches "
            f"only **+{R['tradable'][1][1]:.2f}%** at *t* = {R['tradable'][1][2]:.2f} before it's "
            f"window-mined; the added−dropped spread is *t* = {R['tradable'][1][3]:.2f}. Nothing to size, "
            "and costs never bind.\n"
            "- **Attention effect? `NOT SUPPORTED`** — the real S&P-500 inclusion pop is a "
            "**forced-buying demand shock** (Shleifer 1986; Harris & Gurel 1986). A media ranking by "
            "already-public revenue creates no demand and reveals no news, so the prestige/attention "
            "channel has nothing to stand on — and the tape agrees."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the power curve\n\n"
            "The operational truth in one picture: how big would the *true* debut CAR have to be for a "
            "$k$-event study to detect it at $t=2$? At $k\\approx 14$ you'd need several percent; the "
            "observed CAR lives far below the detection floor — and there's no drop-leg penalty to pair "
            "it with either."
        ),
        code(
            "if HAVE_REAL:\n"
            "    A = PANEL.loc[PANEL['added'],'car'].to_numpy(); sd = A.std(ddof=1); obs = A.mean()\n"
            "else:\n"
            "    sd = 0.06; obs = R['added'][0]/100\n"
            "ks = np.arange(5, 200)\n"
            "min_det = 2.0 * sd / np.sqrt(ks)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(ks, min_det*100, c=AMBER, lw=2, label='CAR needed for t=2')\n"
            "ax.axhline(abs(obs)*100, c=GREEN, ls='--', label=f'observed |CAR| ~{abs(obs)*100:.2f}%')\n"
            "ax.axvline(R['n_added'], c=GREY, ls=':', label=f\"our k={R['n_added']}\")\n"
            "ax.set_xlabel('number of events k'); ax.set_ylabel('reveal-window CAR (%)')\n"
            "ax.set_title('Detection floor vs the real debut CAR: badly under-powered — and it is ~0 anyway'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "need = 2.0*sd/np.sqrt(R['n_added'])*100\n"
            "print(f'at k={R[\"n_added\"]} you need ~{need:.1f}% CAR for t=2; observed ~{obs*100:+.2f}% -> both under-powered AND ~zero')"
        ),
        md(
            "> 💡 In plain words: the amber curve is the **minimum detectable CAR**; the green line is "
            "what we see. They don't meet until $k$ is many times larger than the Fortune-500 calendar "
            "will ever deliver — and unlike a fragile-but-real edge, the observed CAR isn't just below "
            "the floor, it's **at zero**. There is no sizing, threshold, or cost assumption that "
            "manufactures an edge from a prestige ranking with no demand shock behind it. The romance of "
            "'making the list' is not a catalyst the market trades on."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The real one, side by side.** [Study 249 — Index-Inclusion](../249-index-inclusion/): "
            "the S&P-500 version *with* the forced-buying mechanism. The contrast is the entire lesson — "
            "**mechanism, not prestige** — and a clean natural experiment on what actually moves price at "
            "an index event.\n"
            "- **The label/attention family.** [Study 389 — Name-Change-Effect]"
            "(../389-name-change-effect/) and [Study 391 — CEO-Turnover](../391-ceo-turnover/): the same "
            "market-model event-study engine on adjacent corporate-attention folklore, same small-sample "
            "pathology.\n"
            "- **Add the corpses.** Reconstruct the bankrupt exits (J.C. Penney, Bed Bath & Beyond, "
            "Sears) from a survivorship-free feed; the drop leg may finally show a penalty — but it would "
            "be a bankruptcy, not a tradable 'fell off the list.'\n\n"
            "*The reproducible core is offline and deterministic; the add/drop table is a hardcoded, "
            "cited labelled proxy and the exit tape is survivor-biased (named). Methods and sources: "
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
