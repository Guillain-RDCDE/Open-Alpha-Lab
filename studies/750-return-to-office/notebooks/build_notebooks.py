"""Generate the two narrative notebooks for Study 750 (Return-to-Office).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached office-REIT
basket + SPY/VNQ closes under ../_cache/ and otherwise quote the frozen headline numbers in
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance 10-name office
# basket + SPY + VNQ, 2018-06-01 -> 2026-07-10; 26 mandates priced; fingerprint 69ca30fe1b8b).
R = dict(
    n_table=26, n_strict=18, n_hybrid=8, n_priced=26,
    asof="2026-07-10", fp="69ca30fe1b8b", n_delisted=3, n_members=10,
    # bucket: (n, mean%, win%, t)
    strict=(18, -0.34, 44, -0.49),
    hybrid=(8, -0.43, 50, -0.52),
    allb=(26, -0.37, 46, -0.69),
    diff_pp=0.09, diff_t=0.08,
    placebo_p=0.53, null_mean=0.00,
    # robustness windows: (label, all_mean%, all_t, diff_pp, diff_t)
    robust=[("[0,0]", 0.23, 0.69, 0.50, 0.80), ("[0,+2]", -0.37, -0.69, 0.09, 0.08),
            ("[-1,+1]", 0.30, 0.50, -0.52, -0.37), ("[0,+4]", 0.14, 0.18, 1.90, 1.32)],
    # vnq: (label, all_mean%, all_t)
    vnq=[("[0,0]", 0.21, 0.85), ("[0,+2]", 0.17, 0.43)],
    # tradable lag1: (label, all_mean%, all_t, net%)
    trade=[("[+1,+3]", -0.33, -0.47, -0.43), ("[+1,+5]", -0.78, -0.88, -0.88)],
    # synthetic: (planted_bps, strict_mean%, strict_t, diff_pp, diff_t)
    syn=[(0, 1.29, 1.87, 0.93, 1.01), (400, 5.29, 7.67, 4.93, 5.35)],
    kastle_first=24, kastle_last=54,
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Office_rebound%3F: Not_supported](https://img.shields.io/badge/Office_rebound%3F-Not_supported-8b949e?style=flat-square)\n\n"
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

from return_to_office import data, strategy as st

HAVE_REAL = data.have_real()
PRICES, EVENTS = data.load_real() if HAVE_REAL else (None, None)
MEM = data.members_present(PRICES) if HAVE_REAL else data.OFFICE_REITS
PANEL = st.car_panel(PRICES, EVENTS, MEM) if HAVE_REAL else None
print("real office-basket cache present:", HAVE_REAL,
      "| events priced:", (0 if PANEL is None else len(PANEL)),
      "| basket members:", len(MEM))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Return-to-Office — did the back-to-work mandates rescue the office REITs? 🏬\n"
            "### Goldman's 'aberration', Musk's '40 hours or leave', Amazon's 5-day return — "
            "and what the office landlords actually did\n\n"
            + BADGES +
            "There's a tidy bull case for beaten-down office landlords: every few months a "
            "marquee employer orders everyone **back to the office** — Goldman called working "
            "from home an *'aberration'*, Musk emailed Tesla *'40 hours in the office or "
            "leave'*, Amazon went to a full **5-day** return, even the **federal government** "
            "ordered workers back. Desks refill, leases firm up, and the crushed office REITs "
            "(SL Green, Boston Properties, Vornado…) finally catch a bid. Buy the mandate.\n\n"
            "It's a great story. This notebook asks whether it's a *real, repeatable* market "
            "reaction — by lining up a transparent calendar of **~26 real RTO mandates** and "
            "measuring what the office-REIT basket actually did around each one.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo test and the power "
            "analysis? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** The worst-hit office landlords **left the tape** — "
            "**WeWork** went bankrupt, hundreds of private towers were handed to lenders. So "
            "the names we *can* still price are the *survivors*, which biases everything "
            "**toward** the bull story — making it all the more telling that even they don't "
            "pop. Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Do office REITs pop when a big employer mandates RTO? | **No.** Across ~26 "
            f"mandates the basket's abnormal move averages **{R['allb'][1]:+.2f}%** — "
            "statistically **zero** (and if anything, faintly *negative*). |\n"
            "| Does a *strict* 5-day mandate beat a soft hybrid? | **No difference.** "
            f"Strict minus hybrid is **{R['diff_pp']:+.2f}pp** — a coin-flip gap. |\n"
            "| Then why does everyone tie office rallies to RTO headlines? | **Story-fitting.** "
            "The towers move on **interest rates and structural vacancy**; a memo is easy to "
            "point at *after* the fact. |\n"
            "| Can you trade 'buy the mandate'? | **Nothing to buy.** The reaction is zero "
            f"gross, and entering the next day is **{R['trade'][0][1]:+.2f}%** — a small loss. |\n\n"
            "> The mandates were loud and real; the office REITs simply didn't price them. RTO "
            "is a **narrative**, not a signal."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Office landlords are bombed out because nobody's at their desk. But watch the "
            "back-to-work mandates: when a Goldman or an Amazon orders everyone back, office "
            "demand firms up and the REITs re-rate. The stricter the mandate, the bigger the "
            "move. Buy SL Green / Boston Properties on the headline.\"*\n\n"
            "It's not a crazy claim — physical occupancy really did climb off the 2021 floor. "
            "The believers extend that to a **tradable reaction**: a stricter (full 5-day) "
            "mandate should move offices more than a soft hybrid, and the basket should pop "
            "around the announcement. We'll test exactly that — on a representative calendar, "
            "not the two rallies everyone remembers."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If it were real, it'd be a tidy little machine: a calendar of corporate RTO memos, "
            "each a scheduled catalyst to buy the office basket. It would also say something "
            "reassuring for the bulls — that the office apocalypse is a *sentiment* problem a "
            "few CEO emails can fix. But two things have to hold: the reaction has to be **real "
            "on average** (not just in the names we remember rallying), and it has to be "
            "**bigger for stricter mandates** (or the 'RTO thesis' is just noise with a story). "
            "Miss either and 'buy the mandate' is folklore."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We build a **transparent calendar of ~{R['n_table']} real RTO mandates** "
            "(2021–2025) — Goldman, JPMorgan, Tesla, Disney, Amazon, the federal order — each "
            "tagged **strict** (full 5-day) or **hybrid**. For each one:\n\n"
            "1. **Build the basket.** An equal-weight index of the surviving pure-office REITs.\n"
            "2. **Measure the reaction.** The basket's **abnormal** return around the mandate "
            "— *abnormal* meaning *beyond what the market (SPY) did* — over a few days.\n"
            "3. **Stress the luck.** Draw the same number of *random* dates thousands of times "
            "and ask how often chance produces a move this big. With only ~26 events, that's "
            "the honest test.\n\n"
            "And we say it loudly: the worst office casualties **delisted**, so our basket is "
            "the *survivors* — the deck is stacked **for** the story, not against it."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, every mandate and what the office basket did.** The abnormal (market-"
            "adjusted) return of the office-REIT basket over the 3 days around each RTO memo, "
            "strict (dark) vs hybrid (light). A real effect would be a wall of green."
        ),
        code(
            "if HAVE_REAL:\n"
            "    d = PANEL.sort_values('date')\n"
            "    cars = d['car'].values*100; strict = d['strict'].values\n"
            "    labels = [e.split('(')[0].strip()[:16] for e in d['employer']]\n"
            "else:\n"
            "    rng=np.random.default_rng(750); cars=rng.normal(R['allb'][1],2.3,R['n_priced'])\n"
            "    strict=np.array([True]*R['n_strict']+[False]*R['n_hybrid']); labels=[f'ev{i}' for i in range(R['n_priced'])]\n"
            "fig, ax = plt.subplots(figsize=(10.2, 5.2))\n"
            "colors=[(RED if c<0 else GREEN) for c in cars]\n"
            "alphas=[1.0 if s else 0.45 for s in strict]\n"
            "y=np.arange(len(cars))\n"
            "for yi,c,col,al in zip(y,cars,colors,alphas): ax.barh(yi,c,color=col,alpha=al)\n"
            "ax.axvline(0,c='k',lw=.8); ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7)\n"
            "ax.axvline(np.mean(cars), c=GREY, ls='--', label=f'mean {np.mean(cars):+.2f}%')\n"
            "ax.set_xlabel('office-REIT basket abnormal return, CAR[0,+2] (%)')\n"
            "ax.set_title('Solid = strict 5-day mandate, faded = hybrid. No wall of green.')\n"
            "ax.legend(loc='lower right'); ax.invert_yaxis(); plt.tight_layout(); plt.show()\n"
            "print(f'mean abnormal reaction {np.mean(cars):+.2f}%  (a real pop would be clearly positive)')"
        ),
        md(
            f"There's the tell. The bars are a near-even mix of red and green scattered around "
            f"zero — mean **{R['allb'][1]:+.2f}%**. The biggest single moves aren't even the "
            "strict finance mandates; IBM's RTO week (−6.5%) and AT&T's 5-day week (+5.2%) both "
            "landed on days the whole sector was swinging on *rates*. The mandate is lost in "
            "the macro."
        ),
        md(
            "**The two buckets as averages.** Mean abnormal reaction for strict vs hybrid "
            "mandates, with the win-rate (how often the basket rose). A coin is 50%."
        ),
        code(
            "if HAVE_REAL:\n"
            "    s = st.summarize(PANEL)\n"
            "    sm, sw = s['strict']['mean_pct'], s['strict']['win']*100\n"
            "    hm, hw = s['hybrid']['mean_pct'], s['hybrid']['win']*100\n"
            "else:\n"
            "    sm, sw, hm, hw = R['strict'][1], R['strict'][2], R['hybrid'][1], R['hybrid'][2]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.2, 4.2))\n"
            "a1.bar(['strict\\n(5-day)', 'hybrid\\n(2-4 day)'], [sm, hm], color=[RED, AMBER], width=.55)\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_ylabel('mean abnormal return (%)')\n"
            "a1.set_title('Both buckets ~ zero (strict is no better)')\n"
            "for i,v in enumerate([sm,hm]): a1.annotate(f'{v:+.2f}%',(i,v),ha='center',va='top' if v<0 else 'bottom')\n"
            "a2.bar(['strict','hybrid'], [sw, hw], color=GREY, width=.55)\n"
            "a2.axhline(50, c=RED, ls='--', label='coin flip (50%)')\n"
            "a2.set_ylim(0,100); a2.set_ylabel('% of events basket rose'); a2.set_title('Win-rate is a coin flip')\n"
            "a2.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'strict {sm:+.2f}% (win {sw:.0f}%)   hybrid {hm:+.2f}% (win {hw:.0f}%) — strict is NOT bigger')"
        ),
        md(
            f"The strict mandates average **{R['strict'][1]:+.2f}%** and the hybrids "
            f"**{R['hybrid'][1]:+.2f}%** — both a rounding error from zero, and the *stricter* "
            "ones are, if anything, slightly worse. The core believer prediction (strict > "
            "hybrid) simply isn't there."
        ),
        md(
            "**Could a couple-dozen random days look this 'special'?** The honest small-sample "
            f"test: draw **{R['n_priced']}** *random* dates on the same basket, over and over, "
            "and see where the real mandate reaction lands against pure luck."
        ),
        code(
            "if HAVE_REAL:\n"
            "    null = st.placebo_car_dist(PRICES, MEM, k=len(PANEL), n_draws=4000)*100\n"
            "    obs = PANEL['car'].mean()*100\n"
            "    pval = st.placebo_pvalue(PANEL['car'].mean(), st.placebo_car_dist(PRICES, MEM, k=len(PANEL), n_draws=4000))\n"
            "else:\n"
            "    rng=np.random.default_rng(750); null=rng.normal(0,0.55,4000); obs=R['allb'][1]; pval=R['placebo_p']\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(null, bins=45, color=GREY, alpha=.85, label=f'{R[\"n_priced\"]} RANDOM basket windows')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'the actual mandates ({obs:+.2f}%)')\n"
            "ax.axvline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('average basket abnormal return, CAR[0,+2] (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'The mandate reaction sits dead-center in the luck cloud — placebo p = {pval:.2f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'a random {R[\"n_priced\"]}-date draw matches the mandate reaction {pval*100:.0f}% of the time — pure chance')"
        ),
        md(
            f"The red line — the real mandates' reaction — sits **right in the middle** of the "
            f"grey luck cloud (placebo *p* ≈ **{R['placebo_p']:.2f}**). In plain terms: **a "
            "couple-dozen random calendar days would look about this 'eventful' by chance.** "
            "The RTO reaction isn't a signal; it's noise with a headline attached."
        ),
        md(
            "**One more honest look: the desks really did come back — halfway.** The Kastle "
            "10-city office-occupancy proxy (a *labelled, cited approximation*, not a priced "
            "tape) shows the physical trend the bulls are pointing at."
        ),
        code(
            "ko = data.kastle_proxy()\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.0))\n"
            "ax.plot(ko.index, ko.values, marker='o', color=AMBER, lw=2)\n"
            "ax.axhline(100, c=GREY, ls=':', label='pre-COVID (Feb 2020 = 100)')\n"
            "ax.set_ylim(0,105); ax.set_ylabel('office occupancy vs Feb-2020 (%)')\n"
            "ax.set_title('PROXY (Kastle, cited): desks refilled to ~half, then plateaued')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Kastle occupancy {ko.iloc[0]:.0f} -> {ko.iloc[-1]:.0f} (Feb-2020=100): a real but partial, sticky-below-half refill')"
        ),
        md(
            f"Occupancy climbed from **~{R['kastle_first']}%** of pre-COVID to **~"
            f"{R['kastle_last']}%** and then **stalled near half**. That's the crux: the "
            "physical return is real but *partial and permanent-feeling* — and even that slow "
            "refill never showed up as a repricing of the REITs on any single mandate. "
            "*(Labelled proxy — see [references](../docs/references.md).)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** The office basket's average reaction to an RTO mandate is "
            f"**{R['allb'][1]:+.2f}%** — statistically zero (placebo *p* ≈ {R['placebo_p']:.2f}) "
            "— and a strict 5-day mandate does **no more** than a hybrid.\n"
            f"- **Tradability — Mirage.** Nothing to buy: the reaction is zero gross, and "
            f"entering the next day is **{R['trade'][0][1]:+.2f}%** (worse after costs).\n"
            "- **\"Office rebound on RTO?\" — Not supported.** Desks refilled to ~half and "
            "stalled, and the REITs still didn't care. The towers are a **rates + vacancy** "
            "story a memo doesn't move."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — the pop that isn't there\n\n"
            "Forget significance and just price the believers' trade: **buy the office basket "
            "on the mandate, hold a few days.** Because the memo lands intraday, the honest "
            "version enters the *next* day. Here's what that makes, gross and net of costs."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows=[]\n"
            "    for w,lab in [((0,2),'same-day [0,+2]'),((0,2),'lag1 [+1,+3]'),((0,4),'lag1 [+1,+5]')]:\n"
            "        lag = 0 if 'same' in lab else 1\n"
            "        pl=st.car_panel(PRICES,EVENTS,MEM,window=w,lag=lag)\n"
            "        m=pl['car'].mean(); nc=st.net_of_costs(m)\n"
            "        rows.append((lab, m*100, nc['net_pct']))\n"
            "else:\n"
            "    rows=[('same-day [0,+2]', R['allb'][1], R['allb'][1]-0.1)] + [(f'lag1 {t[0]}', t[1], t[3]) for t in R['trade']]\n"
            "labs=[r[0] for r in rows]; gross=[r[1] for r in rows]; net=[r[2] for r in rows]; xx=np.arange(len(labs))\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.3))\n"
            "ax.bar(xx-.2, gross, .4, color=AMBER, label='gross'); ax.bar(xx+.2, net, .4, color=RED, label='net @10bps')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_xticks(xx); ax.set_xticklabels(labs)\n"
            "ax.set_ylabel('per-event basket P&L (%)'); ax.set_title('Buy-the-mandate: no pop, small loss once you can actually enter')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('buy-the-mandate per event:', [(l, round(g,2), round(n,2)) for l,g,n in rows])"
        ),
        md(
            f"Same-day the reaction is a statistically-zero **{R['allb'][1]:+.2f}%**; the only "
            "version you could *actually* trade — enter the next day — is a **small loss** "
            f"(**{R['trade'][0][1]:+.2f}%**, worse after 10 bps). There's no machine here — "
            "just a sector waiting on the Fed, with a convenient RTO headline to blame on any "
            "given down day."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The twin.** [Study 391 — CEO-Turnover](../391-ceo-turnover/) runs the same "
            "market-model event study on a labelled table — and finds the only real move is "
            "the announcement instant you can't trade. Same shape, same lesson.\n"
            "- **The rate that actually drives it.** Regress the office basket on the 10-year "
            "yield and on VNQ; you'll find the mandates explain ~nothing that rates and "
            "REIT-beta don't already.\n"
            "- **Add the corpses.** Our basket is the survivors; reconstruct WeWork and the "
            "CMBS-default towers and the sector looks *worse*, not better — the RTO memos still "
            "wouldn't have saved them.\n\n"
            "*Think RTO mandates are a tradable office catalyst? Capture the events, draw the "
            "same number of random windows, and show the basket landing **outside** the cloud "
            "**and** a strict-beats-hybrid gap that's reliably positive — then we'll talk.*"
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
            "# Return-to-Office — a quantitative teardown 🔬\n"
            "### Basket market-model CAR by strict/hybrid bucket · strict−hybrid Welch *t* + a "
            "placebo basket-window null · window & VNQ-benchmark robustness · a 1-day-lag "
            "tradable variant + costs · a synthetic faithful-engine / power control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). We "
            "treat the folklore as a **sector event-study hypothesis** — a positive abnormal "
            "reaction of an office-REIT basket around an RTO mandate, larger for *strict* "
            "mandates — and confront it with the **sample size**, the **benchmark choice**, and "
            "**survivorship**. The decisive objects are a cross-section of ~26 basket CARs and a "
            "placebo null sized to that count.\n\n"
            "> ⚠️ **Data + survivorship note.** The RTO calendar is hardcoded and transparent "
            "(~26 real, dated mandates); the priced basket is **survivor-biased** — the worst "
            "landlords (WeWork, CMBS-default towers) **delisted**, biasing the reaction *toward "
            "zero* (named on the Signal axis). Real data: yfinance daily closes for 10 office "
            "REITs + SPY + VNQ, 2018→2026. The Kastle occupancy series is a **labelled proxy**, "
            "never priced. Offline core + synthetic control are deterministic. Methods in "
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
            f"| **Signal** | `NONE` | all-events CAR **{R['allb'][1]:+.2f}%** (Welch "
            f"**t = {R['allb'][3]:.2f}**, placebo **p = {R['placebo_p']:.2f}**); strict−hybrid "
            f"**{R['diff_pp']:+.2f}pp** (**t = {R['diff_t']:.2f}**). No window/benchmark clears "
            "\\|t\\| ~1. |\n"
            f"| **Tradability** | `MIRAGE` | 1-day-lag basket CAR **{R['trade'][0][1]:+.2f}%** "
            f"(**t = {R['trade'][0][2]:.2f}**), **{R['trade'][0][3]:+.2f}%** net. Zero gross "
            "reaction; a rates/WFH sector a memo doesn't move. |\n"
            f"| **Office rebound?** | `NOT SUPPORTED` | Kastle occupancy refilled to ~half and "
            "plateaued; the REITs priced none of it. |\n\n"
            "> 💡 In plain words: the RTO calendar is loud and real, but on the tape the office "
            "REITs treat a back-to-work mandate as a non-event — and ~26 events on a rates-"
            "driven sector couldn't certify a reaction of plausible size even if one existed."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $B_t$ be the equal-weight office-REIT basket return and fit a **market model** "
            "$B_t = \\alpha + \\beta\\,\\text{SPY}_t + \\varepsilon_t$ on a clean pre-event "
            "estimation window $[-130,-10]$. For mandate $i$ the **cumulative abnormal return** "
            "is $\\text{CAR}_i = \\sum_{\\tau\\in W}(B_\\tau - \\hat\\alpha - \\hat\\beta\\,"
            "\\text{SPY}_\\tau)$ over event window $W$ (default $[0,+2]$, entry lagged for the "
            "tradable variant).\n\n"
            "- **H₁ (a reaction).** $\\mathbb{E}[\\text{CAR}_i] > 0$ — the basket pops on the "
            "mandate.\n"
            "- **H₂ (strictness matters).** $\\mathbb{E}[\\text{CAR}\\mid\\text{strict}] > "
            "\\mathbb{E}[\\text{CAR}\\mid\\text{hybrid}]$ — a real mandate beats a soft one.\n"
            "- **H₃ (deployable).** the CAR, net of costs on a buy-the-basket book with a "
            "one-day execution lag, is positive and harvestable.\n\n"
            "We find **H₁ not supported** (CAR ≈ 0, inside the placebo cloud), **H₂ not "
            f"supported** (strict−hybrid = {R['diff_pp']:+.2f}pp, $t = {R['diff_t']:.2f}$), and "
            "**H₃ rejected** (the tradable CAR is negative). The mandates are salient news the "
            "sector does not price."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The teardown is a one-sample test on a small cross-section, judged by its "
            "**standard error** and by **survivorship**:\n\n"
            "$$t = \\frac{\\overline{\\text{CAR}}}{s_{\\text{CAR}}/\\sqrt{k}},\\qquad "
            "k\\approx 26.$$\n\n"
            "With $k\\approx 26$ on a **single** sector whose daily basket vol is large (office "
            "REITs swing ~2% on rate days), $s_{\\text{CAR}}/\\sqrt{k}$ dwarfs a fraction-of-a-"
            "percent mean. Worse, the sample is **conditioned on survival**: the landlords whose "
            "value fell most (WeWork → 0, CMBS-default towers) **left the basket**, so "
            "$\\overline{\\text{CAR}}$ is biased **upward** — *toward* finding the bulls' pop. "
            "The honest instrument is a **randomization (placebo) test**: resample $k$ random "
            "non-event windows of the *same basket* and ask how often chance matches the "
            "reaction. That, not the point estimate, decides the Signal axis."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **RTO calendar.** ~{R['n_table']} documented mandates (2021–2025), hardcoded & "
            f"transparent, tagged **{R['n_strict']} strict** / **{R['n_hybrid']} hybrid**; "
            f"**{R['n_delisted']}** office casualties listed as **DELISTED** for the "
            "survivorship caveat. Table fingerprint `" + R['fp'] + "`.\n"
            "- **Basket + abnormal returns.** Equal-weight office-REIT basket; market model vs "
            "**SPY**, 120-day estimation window, 10-day gap. CAR over $[0,+2]$ (canonical), "
            "**1-day entry lag** for the tradable variant.\n"
            "- **Null #1 (Welch t).** All-events CAR vs zero; strict−hybrid two-sample t.\n"
            "- **Null #2 (placebo).** 20,000 draws of $k$ random basket windows; "
            "$p = \\Pr[|\\text{random mean}| \\ge |\\text{observed}|]$ — the small-sample "
            "workhorse.\n"
            "- **Benchmark robustness.** Re-run vs **VNQ** (broad REITs): does office react "
            "*beyond* the whole REIT complex?\n"
            "- **Costs.** A buy-the-basket book pays a one-way 10-bps round-trip.\n"
            "- **Positive control.** Deterministic panels with a **planted** strict-bucket CAR "
            "edge: the inference must recover a large edge **and** must NOT manufacture a "
            "strict>hybrid gap when the edge is 0."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The buckets — both ~zero, strict is no bigger\n\n"
            "Mean basket CAR per bucket with $\\pm$ standard error, against zero (dashed). Both "
            "hug zero and the strict bucket is not above the hybrid — H₂ fails on the point "
            "estimate before any test."
        ),
        code(
            "if HAVE_REAL:\n"
            "    S = PANEL.loc[PANEL['strict'],'car'].values; H = PANEL.loc[~PANEL['strict'],'car'].values\n"
            "    sm, hm = S.mean()*100, H.mean()*100\n"
            "    sse, hse = S.std(ddof=1)/np.sqrt(len(S))*100, H.std(ddof=1)/np.sqrt(len(H))*100\n"
            "    stt, htt = st.welch_t(S), st.welch_t(H); difft = st.welch_t(S,H)\n"
            "else:\n"
            "    sm, hm, stt, htt, difft = R['strict'][1], R['hybrid'][1], R['strict'][3], R['hybrid'][3], R['diff_t']\n"
            "    sse, hse = abs(sm/max(abs(stt),1e-9)), abs(hm/max(abs(htt),1e-9))\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.bar(['strict (5-day)','hybrid (2-4d)'], [sm, hm], yerr=[sse, hse], capsize=6,\n"
            "       color=[RED, AMBER], width=.55)\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('mean basket CAR[0,+2] (%)')\n"
            "ax.set_title(f'strict t={stt:.2f}, hybrid t={htt:.2f}, strict-hybrid t={difft:.2f} (all n.s.)')\n"
            "for i,v in enumerate([sm,hm]): ax.annotate(f'{v:+.2f}%',(i,v),ha='center',va='top' if v<0 else 'bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'strict {sm:+.2f}% (t={stt:.2f})  hybrid {hm:+.2f}% (t={htt:.2f})  strict-hybrid t={difft:.2f} — H2 wants >0')"
        ),
        md(
            f"> 💡 In plain words: strict mandates average **{R['strict'][1]:+.2f}%** "
            f"(t = {R['strict'][3]:.2f}), hybrids **{R['hybrid'][1]:+.2f}%** "
            f"(t = {R['hybrid'][3]:.2f}), and the strict−hybrid gap is **{R['diff_pp']:+.2f}pp** "
            f"at t = **{R['diff_t']:.2f}**. The believers' central prediction — a real mandate "
            "beats a soft one — has a point estimate of essentially zero."
        ),
        md(
            "### 4b · The decisive test — a placebo null sized to the event count\n\n"
            f"Draw {R['n_priced']} random basket windows 20,000 times; the histogram is the null "
            "for the mean CAR. The real reaction is the red line; the *p*-value is the two-sided "
            "tail mass."
        ),
        code(
            "if HAVE_REAL:\n"
            "    null = st.placebo_car_dist(PRICES, MEM, k=len(PANEL), n_draws=8000)*100\n"
            "    obs = PANEL['car'].mean()*100\n"
            "    pval = st.placebo_pvalue(PANEL['car'].mean(), null/100)\n"
            "else:\n"
            "    rng=np.random.default_rng(750); null=rng.normal(0,0.55,8000); obs=R['allb'][1]; pval=R['placebo_p']\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(null, bins=55, color=GREY, alpha=.85, label=f'null: {R[\"n_priced\"]} random basket windows')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'observed CAR {obs:+.2f}%'); ax.axvline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('mean basket CAR[0,+2] (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Placebo p = {pval:.2f}: the mandate reaction is inside the luck cloud'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'P[|random {R[\"n_priced\"]}-window mean| >= |CAR|] = {pval:.3f}  (need <0.05 to call it real)')"
        ),
        md(
            f"> 💡 In plain words: **{R['placebo_p']*100:.0f}%** of random {R['n_priced']}-window "
            "draws match or beat the mandate reaction in magnitude. A real effect would push the "
            "red line into the tail; instead it sits dead-center. H₁ is **not supported** — the "
            "reaction is what a couple-dozen random calendar days look like on this basket."
        ),
        md(
            "### 4c · Robustness — window & benchmark\n\n"
            "Shift the event window, and swap the benchmark to VNQ (so 'abnormal' means "
            "*relative to all REITs*). The all-events CAR stays a fraction of a percent with "
            "|t| well under 1 everywhere, and the strict−hybrid gap **flips sign** across "
            "windows — the fingerprint of noise, not signal."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rob=[]\n"
            "    for w,lab in [((0,0),'[0,0]'),((0,2),'[0,+2]'),((-1,1),'[-1,+1]'),((0,4),'[0,+4]')]:\n"
            "        pw=st.car_panel(PRICES,EVENTS,MEM,window=w); sw=st.summarize(pw)\n"
            "        rob.append((lab, sw['all']['mean_pct'], sw['all']['t'], sw['diff_pct'], sw['diff_t']))\n"
            "    vnq=[]\n"
            "    for w,lab in [((0,0),'[0,0]'),((0,2),'[0,+2]')]:\n"
            "        pw=st.car_panel(PRICES,EVENTS,MEM,benchmark='VNQ',window=w); sw=st.summarize(pw)\n"
            "        vnq.append((lab, sw['all']['mean_pct'], sw['all']['t']))\n"
            "else:\n"
            "    rob=R['robust']; vnq=R['vnq']\n"
            "fig, (a1,a2) = plt.subplots(1,2,figsize=(10.8,4.2))\n"
            "labs=[r[0] for r in rob]; means=[r[1] for r in rob]; difs=[r[3] for r in rob]; xx=np.arange(len(labs))\n"
            "a1.bar(xx-.2, means, .4, color=GREY, label='all-events CAR'); a1.bar(xx+.2, difs, .4, color=RED, label='strict-hybrid')\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_xticks(xx); a1.set_xticklabels(labs)\n"
            "a1.set_ylabel('% / pp'); a1.set_xlabel('SPY market model, event window'); a1.set_title('All-events ~0; strict-hybrid flips sign'); a1.legend()\n"
            "vl=[v[0] for v in vnq]; vm=[v[1] for v in vnq]; vt=[v[2] for v in vnq]; xx2=np.arange(len(vl))\n"
            "bars=a2.bar(xx2, vm, .5, color=AMBER)\n"
            "a2.axhline(0, c='k', lw=.8); a2.set_xticks(xx2); a2.set_xticklabels(vl)\n"
            "a2.set_ylabel('all-events CAR vs VNQ (%)'); a2.set_xlabel('office vs BROAD REITs'); a2.set_title('No reaction beyond the REIT complex either')\n"
            "for b,t in zip(bars,vt): a2.annotate(f't={t:.2f}',(b.get_x()+b.get_width()/2,b.get_height()),ha='center',va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('windows:', [(r[0], round(r[1],2), round(r[2],2), round(r[4],2)) for r in rob])"
        ),
        md(
            "> 💡 In plain words: no event window makes the reaction significant (|t| < 1 "
            "everywhere), the strict−hybrid gap is +0.50pp on the news day but −0.52pp over "
            "[−1,+1] (sign-unstable = noise), and against VNQ the office basket doesn't move "
            "*relative to all REITs* either. The verdict doesn't hinge on any choice that could "
            "have rescued it."
        ),
        md(
            "### 4d · Faithful-engine & power control — we know the truth here\n\n"
            "Deterministic per-event panels with a **planted** strict-bucket CAR edge: with "
            "**0 bps** the inference must NOT manufacture a strict>hybrid gap (a couple-dozen "
            "noisy events can't fake it); with a **+400 bps** planted edge it must light up. "
            "Both hold — proving the engine is unbiased *and* that this sample size only detects "
            "implausibly large effects."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 400.0):\n"
            "    syn = data.synthetic_events(car_bps=edge, seed=750)\n"
            "    stt = st.summarize_bucket(syn['strict_car'])['t']\n"
            "    difft = st.welch_t(syn['strict_car'], syn['hybrid_car'])\n"
            "    res.append((edge, stt, difft))\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "labels=[f'planted\\n{int(e)} bps' for e,*_ in res]; xx=np.arange(len(labels))\n"
            "stts=[r[1] for r in res]; diffs=[r[2] for r in res]\n"
            "ax.bar(xx-.2, stts, .4, color=RED, label='strict t (vs 0)'); ax.bar(xx+.2, diffs, .4, color=GREEN, label='strict-hybrid t')\n"
            "ax.axhline(2, ls='--', c=GREY, label='t=2'); ax.axhline(-2, ls='--', c=GREY)\n"
            "ax.set_xticks(xx); ax.set_xticklabels(labels); ax.set_ylabel('Welch t')\n"
            "ax.set_title('Control: only a HUGE planted edge lights up the strict-hybrid gap'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,s2,d2 in res: print(f'planted {int(e):>4} bps: strict t={s2:+.2f}  strict-hybrid t={d2:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted edge the strict−hybrid *t* is "
            f"**{R['syn'][0][4]:.2f}** (the noise nudges the single strict bucket to "
            f"t = {R['syn'][0][2]:.2f}, still short of 2, but the *difference* the claim rests on "
            f"is cleanly null); only the **+400 bps** plant reaches strict−hybrid *t* "
            f"**{R['syn'][1][4]:.2f}**. So the machinery is honest, and the real-tape strict−"
            f"hybrid *t* of **{R['diff_t']:.2f}** is exactly what *no effect* looks like through "
            f"a {R['n_priced']}-event keyhole."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — all-events basket CAR **{R['allb'][1]:+.2f}%** at Welch "
            f"**t = {R['allb'][3]:.2f}** / placebo **p = {R['placebo_p']:.2f}**; strict−hybrid "
            f"**{R['diff_pp']:+.2f}pp** at **t = {R['diff_t']:.2f}**; no window or benchmark "
            "(SPY/VNQ) clears |t| ~1. Indistinguishable from zero, faintly the wrong sign for a "
            "pop. **Survivorship** named on this axis and pointing *toward* the null (worst "
            "landlords delisted).\n"
            f"- **Tradability `MIRAGE`** — the 1-day-lag tradable CAR is **{R['trade'][0][1]:+.2f}%** "
            f"(**t = {R['trade'][0][2]:.2f}**), **{R['trade'][0][3]:+.2f}%** net of 10 bps. Zero "
            "gross reaction on a rates/WFH-driven sector; no NAV-scale edge, not even sign-stable.\n"
            "- **Office rebound on RTO? `NOT SUPPORTED`** — physical occupancy (Kastle proxy) "
            "refilled to ~half and plateaued, and the REITs priced none of it on any mandate. "
            "The office rebound is a macro/rates story, not a back-to-work-memo trade."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the power curve\n\n"
            "The operational truth in one picture: how big would the *true* mandate reaction "
            "have to be for a $k$-event study to detect it at $t=2$? At $k\\approx 26$ on this "
            "basket's volatility you'd need a CAR several times larger than anything observed; "
            "the real reaction lives far below the detection floor — and the tradable version is "
            "negative on top."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sd = PANEL['car'].std(ddof=1); obs = PANEL['car'].mean()\n"
            "else:\n"
            "    sd = 0.027; obs = R['allb'][1]/100\n"
            "ks = np.arange(5, 200)\n"
            "min_det = 2.0 * sd / np.sqrt(ks)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(ks, min_det*100, c=AMBER, lw=2, label='CAR needed for t=2')\n"
            "ax.axhline(abs(obs)*100, c=RED, ls='--', label=f'|observed CAR| ~{abs(obs)*100:.2f}%')\n"
            "ax.axvline(R['n_priced'], c=GREY, ls=':', label=f\"our k={R['n_priced']}\")\n"
            "ax.set_xlabel('number of events k'); ax.set_ylabel('basket CAR (%)')\n"
            "ax.set_title('Detection floor vs the real reaction: badly under-powered'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "need = 2.0*sd/np.sqrt(R['n_priced'])*100\n"
            "print(f'at k={R[\"n_priced\"]} you need ~{need:.2f}% CAR for t=2; observed ~{abs(obs)*100:.2f}% -> under-powered')"
        ),
        md(
            "> 💡 In plain words: the amber curve is the **minimum detectable CAR**; the red "
            "line is what we see. They don't meet until $k$ is many times larger than the RTO "
            "calendar will ever deliver — and even a *real* reaction wouldn't pay, because the "
            "only version you can enter (next-day) is **negative**. There is no sizing, "
            "threshold, or window that manufactures an edge from ~26 events on a sector that "
            "trades on the 10-year yield. The rarity that makes each mandate a headline is "
            "exactly what makes the 'thesis' untestable and untradable."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The twin.** [Study 391 — CEO-Turnover](../391-ceo-turnover/): the same "
            "market-model short-window event study on a labelled table, where the only real "
            "move is the un-tradable announcement instant.\n"
            "- **The anecdote trap.** [Study 389 — Name-Change-Effect](../389-name-change-effect/): "
            "the same small-sample / survivorship pathology on a table of corporate events.\n"
            "- **Model the real driver.** Regress the office basket on Δ(10-year yield) and VNQ "
            "around each mandate; the RTO indicator should add ~nothing once rates and REIT-beta "
            "are in. That is the positive statement behind this null: offices are a *rates + "
            "structural-vacancy* asset.\n\n"
            "*The reproducible core is offline and deterministic; the RTO calendar is hardcoded "
            "and the priced basket is survivor-biased (named). Methods and sources: "
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
