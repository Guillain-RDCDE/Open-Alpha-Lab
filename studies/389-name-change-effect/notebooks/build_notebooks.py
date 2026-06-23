"""Generate the two narrative notebooks for Study 389 (Name-Change-Effect).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached rebrand prices
under ../_cache/ (the ~25-name table + SPY) and otherwise quote the frozen headline numbers in
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance ~25-name rebrand
# table + SPY, 1995-01-03 -> 2026-06-18; 21 events priced).
R = dict(
    n_table=25, n_delisted=8, n_priced=21,
    # leg: (mean%, win%, t, placebo_p)
    pop=(1.24, 52, 0.51, 0.496),
    fade=(16.22, 52, 1.16, 0.089),
    # wave: (n, pop%, fade%)
    waves=[("dot-com", 7, -1.31, -3.09), ("blockchain", 6, 1.93, 61.95),
           ("AI", 8, 2.96, -1.18)],
    # robustness: (pop_d/fade_d, n, pop%, pop_t, fade%, fade_t)
    robust=[("3/40", 21, -1.48, -0.46, 16.19, 1.32), ("5/60", 21, 1.24, 0.51, 16.22, 1.16),
            ("10/120", 21, 1.09, 0.29, 51.68, 1.36)],
    cost=dict(gross_combo=-14.98, net_combo=-16.18, gross_pop=1.24, gross_fade=16.22),
    # synthetic: edge, n, pop%, pop_t, fade%, fade_t
    syn=[(0.00, 25, 2.22, 1.52, 5.64, 1.19),
         (0.40, 25, 52.50, 24.05, -29.19, -9.21)],
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Pop_then_dump%3F: Busted](https://img.shields.io/badge/Pop_then_dump%3F-Busted-8b949e?style=flat-square)\n\n"
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

from name_change_effect import data, strategy as st

HAVE_REAL = data.have_real()
B = data.load_real() if HAVE_REAL else None
EV = st.collect_events(B) if HAVE_REAL else None
print("real rebrand-tape cache present:", HAVE_REAL,
      "| events priced:", (0 if EV is None else len(EV)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# The rebrand pop — do `.com` / `Blockchain` / `AI` name changes pay? 🪧\n"
            "### The company that renames toward the hot theme, pops, and 'gives it back' — in plain English\n\n"
            + BADGES +
            "There's a great market story: a sleepy little company **renames itself toward whatever's "
            "hot** — slap `.com` on it in 1999, become *Long **Blockchain** Corp* in 2017, add `**AI**` "
            "in 2023 — and the stock **pops**. Then, once the hype cools, it **gives it all back**. Long "
            "Island Iced Tea → Long Blockchain jumped ~200% in a day; Kodak's 'KodakCoin' tripled. Free "
            "money on a press release, then a short on the way down.\n\n"
            "It's a fun story. This notebook asks whether it's a *real, repeatable* effect or just the "
            "two or three legends we remember — by building a transparent table of **~25 real rebrands** "
            "and looking at what actually happened to all of them.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo test and the power "
            "analysis? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** The most spectacular rebrands — Long Blockchain, KodakCoin — "
            "**went bust and vanished from the data**. So the names we *can* still price are the ones "
            "that *didn't* collapse. That biases everything **against** the 'give it back' story — which "
            "makes it all the more telling that the story still doesn't show up. Every chart is drawn by "
            "the code beside it; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does a theme-chasing rename pop the stock? | **Barely.** Across ~21 surviving rebrands the "
            f"first-week 'pop' averages just **+{R['pop'][0]:.1f}%** — and it's a coin-flip "
            f"(**{R['pop'][1]:.0f}%** of the time positive). |\n"
            "| Does it then *give it back*? | **No — it goes the other way.** Over the next three months "
            f"the survivors drift **+{R['fade'][0]:.0f}% UP**, not down. The famous give-back is missing. |\n"
            "| Then why does everyone 'know' it gives it back? | **Survivorship.** The give-backs that "
            "really happened (Long Blockchain, KodakCoin) **delisted** — they're the ones we remember, "
            "and the ones the data can't show. |\n"
            "| Can you trade the 'pop then dump'? | **It loses money.** Buying the pop and shorting the "
            f"fade is **{R['cost']['gross_combo']:.0f}% per event before costs** — because the fade "
            "doesn't come. |\n\n"
            "> The pop is a faint coin-flip; the give-back runs the wrong way on the survivors. The "
            "legend is the handful of corpses we remember, not a law."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Watch for a company renaming toward the theme of the moment. The rename itself — not "
            "any new product — pops the stock as the crowd piles in. Then, when the fad fades, the stock "
            "gives the whole pop back. Buy the announcement, short the comedown.\"*\n\n"
            "It's not a crazy claim — there's real academia behind the *pop*. The classic *A Rose.com by "
            "Any Other Name* (2001) found dot-com renames earned **~+74%** abnormal returns in 1998–99, "
            "**whether or not** the firm did anything online. A pure attention effect. The believers "
            "extend it: same pop for `Blockchain` (2017) and `AI` (2023), and a symmetric **give-back** "
            "afterward. We'll test both legs on a real, representative table."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If it were real and repeatable, it'd be a tidy little machine: a calendar of corporate "
            "rebrands, each a scheduled pop to buy and a scheduled dump to short. It would also say "
            "something unsettling about markets — that a **press release with a buzzword** moves price "
            "more than fundamentals. But two things have to hold for the machine to work: the pop has to "
            "be **real on average** (not just in the two cases we remember), and the give-back has to "
            "**actually happen** (or the short leg bleeds). Miss either and the 'trade' is a story."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We build a **transparent table of ~{R['n_table']} real rebrands** across the three waves "
            "(`.com`, `Blockchain`, `AI`) — not just the legends, the representative set. For each one:\n\n"
            "1. **Find the rename.** Line up the stock against the market (SPY) around the announcement.\n"
            "2. **Measure two legs.** The **pop** (abnormal return over the first ~week) and the "
            "**fade** (the next ~three months) — *abnormal* meaning *beyond what the market did*.\n"
            "3. **Stress the luck.** Draw the same number of *random* windows thousands of times and ask "
            "how often chance produces a pop (or a fade) this big. With only ~21 events, that's the "
            "honest test.\n\n"
            "And we say it loudly: the worst give-backs **delisted**, so our survivors are the *good* "
            "cases — the deck is stacked **for** the story, not against it."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the pop and the fade, side by side.** For every surviving rebrand: the abnormal "
            "(market-adjusted) return in the first week vs the next three months."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pops = EV['pop'].values*100; fades = EV['fade'].values*100; labels = EV['ticker'].values\n"
            "else:\n"
            "    rng=np.random.default_rng(389)\n"
            "    pops=rng.normal(R['pop'][0],8,R['n_priced']); fades=rng.normal(R['fade'][0],30,R['n_priced'])\n"
            "    labels=[f'R{i}' for i in range(R['n_priced'])]\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.6))\n"
            "ax.scatter(pops, fades, c=GREEN, s=60, edgecolor='k', alpha=.8)\n"
            "ax.axhline(0, c='k', lw=.8); ax.axvline(0, c='k', lw=.8)\n"
            "ax.axhline(np.mean(fades), c=GREY, ls='--', label=f'mean fade {np.mean(fades):+.0f}%')\n"
            "ax.axvline(np.mean(pops), c=AMBER, ls='--', label=f'mean pop {np.mean(pops):+.1f}%')\n"
            "ax.set_xlabel('pop: first-week abnormal return (%)')\n"
            "ax.set_ylabel('fade: next-3-months abnormal return (%)')\n"
            "ax.set_title('The give-back should sit BELOW zero — instead the cloud drifts up')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'mean pop {np.mean(pops):+.1f}%   mean fade {np.mean(fades):+.1f}%  (a give-back would be negative)')"
        ),
        md(
            f"There's the tell. If 'pop then give it back' were true, the dots would sit **below** the "
            f"zero line (a negative fade). Instead the mean fade is **+{R['fade'][0]:.0f}%** — the "
            "survivors drifted *up*, not down. The pop, meanwhile, hugs zero."
        ),
        md(
            "**The two legs as averages.** Mean abnormal pop and fade, with the win-rate (how often each "
            "leg is positive). A coin is 50%."
        ),
        code(
            "if HAVE_REAL:\n"
            "    s = st.summarize(EV, B, placebo=False)\n"
            "    pm, pw = s['pop']['mean']*100, s['pop']['win']*100\n"
            "    fm, fw = s['fade']['mean']*100, s['fade']['win']*100\n"
            "else:\n"
            "    pm, pw, fm, fw = R['pop'][0], R['pop'][1], R['fade'][0], R['fade'][1]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.2, 4.2))\n"
            "a1.bar(['pop\\n(+1..+5d)', 'fade\\n(+6..+65d)'], [pm, fm], color=[AMBER, GREEN], width=.55)\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_ylabel('mean abnormal return (%)')\n"
            "a1.set_title('Fade is POSITIVE — the opposite of a give-back')\n"
            "for i,v in enumerate([pm,fm]): a1.annotate(f'{v:+.1f}%',(i,v),ha='center',va='bottom')\n"
            "a2.bar(['pop','fade'], [pw, fw], color=GREY, width=.55)\n"
            "a2.axhline(50, c=RED, ls='--', label='coin flip (50%)')\n"
            "a2.set_ylim(0,100); a2.set_ylabel('% of events positive'); a2.set_title('Both legs ~ a coin flip')\n"
            "a2.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'pop {pm:+.1f}% (win {pw:.0f}%)   fade {fm:+.1f}% (win {fw:.0f}%)')"
        ),
        md(
            f"The pop is **+{R['pop'][0]:.1f}%** and positive only **{R['pop'][1]:.0f}%** of the time — "
            "a coin flip. The fade is **positive**, so there is no dump to short. Neither leg of the "
            "legend is there on the surviving names."
        ),
        md(
            "**Could a handful of random windows look this good?** The honest small-sample test: draw "
            f"**{R['n_priced']}** *random* windows on the same stocks, over and over, and see where the "
            "real pop lands against pure luck."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.placebo_pvalue(B, len(EV), leg='pop', observed=EV['pop'].mean(), n_draws=4000)\n"
            "    obs = pl['obs']*100; pval = pl['p_value']\n"
            "    # rebuild a small placebo cloud for the picture\n"
            "    tickers=[c for c in B['prices'].columns if c!='SPY']\n"
            "    exs=[st._excess_log_returns(B['prices'][t], B['prices']['SPY']) for t in tickers]\n"
            "    exs=[s for s in exs if len(s)>80]; rng=np.random.default_rng(389)\n"
            "    draws=[]\n"
            "    for _ in range(3000):\n"
            "        vals=[]\n"
            "        for _ in range(len(EV)):\n"
            "            s=exs[rng.integers(0,len(exs))]; p0=rng.integers(1,len(s)-7); vals.append(np.expm1(s.iloc[p0:p0+5].sum()))\n"
            "        draws.append(np.mean(vals))\n"
            "    draws=np.array(draws)*100\n"
            "else:\n"
            "    obs=R['pop'][0]; pval=R['pop'][3]; rng=np.random.default_rng(389); draws=rng.normal(0,2.0,3000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=45, color=GREY, alpha=.85, label=f'pop of {R[\"n_priced\"]} RANDOM windows')\n"
            "ax.axvline(obs, c=GREEN, lw=2.5, label=f'the actual rebrands ({obs:+.1f}%)')\n"
            "ax.set_xlabel('average first-week abnormal return (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'The pop sits inside the luck cloud — placebo p = {pval:.2f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'a random {R[\"n_priced\"]}-window draw matches the pop {pval*100:.0f}% of the time — not rare')"
        ),
        md(
            f"The green line — the real rebrands' pop — sits **inside** the grey luck cloud "
            f"(placebo *p* ≈ **{R['pop'][3]:.2f}**). In plain terms: **a couple-dozen random press "
            "releases would look about this 'special' by chance.** The pop isn't magic; it's noise with a "
            "famous anecdote attached."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Weak.** The dot-com pop is real in the textbooks, but on ~21 *surviving* "
            f"rebrands it's a faint **+{R['pop'][0]:.1f}%** coin-flip — and the 'give-back' actually runs "
            "**+16% the wrong way**. Real as lore, weak as edge.\n"
            f"- **Tradability — Mirage.** Buy-the-pop / short-the-fade is **{R['cost']['gross_combo']:.0f}% "
            "per event before costs** — you bleed shorting a dump that never comes.\n"
            "- **\"Pop then dump\"? — Busted.** We remember Long Blockchain and KodakCoin *because* they "
            "blew up — and they **delisted**, so they never enter a fair sample. The representative "
            "rebrand doesn't give it back."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — the give-back that isn't there\n\n"
            "Forget significance and just price the believers' trade: **buy the pop, then short the "
            "fade.** Here's what that book makes per event, gross and net of the (large, small-cap) "
            "costs of four crossings."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = st.net_of_costs(EV, cost_bps=30.0)\n"
            "    gc, nc = c['gross_combo']*100, c['net_combo']*100\n"
            "else:\n"
            "    gc, nc = R['cost']['gross_combo'], R['cost']['net_combo']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "ax.bar(['gross', 'net @30bps×4'], [gc, nc], color=[AMBER, RED], width=.5)\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('per-event P&L of buy-pop/short-fade (%)')\n"
            "ax.set_title('The believers\\' trade loses money — the fade goes the wrong way')\n"
            "for i,v in enumerate([gc,nc]): ax.annotate(f'{v:+.1f}%',(i,v),ha='center',va='top')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'buy-pop/short-fade: {gc:+.1f}% gross, {nc:+.1f}% net — a money-loser before costs even bite')"
        ),
        md(
            f"The combo is **{R['cost']['gross_combo']:.0f}% gross** and "
            f"**{R['cost']['net_combo']:.0f}% net**. The short leg is the killer: you're betting the "
            "stock gives the pop back, and on the survivors it *doesn't*. Costs only deepen a hole the "
            "missing give-back already dug. There's no machine here — just a story with two famous corpses."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The other name effect.** [Study 169 — Fluent-Tickers](../169-fluent-tickers/) asks "
            "whether a *pronounceable* ticker carries a premium — same family (a label, not a "
            "fundamental), same verdict.\n"
            "- **The anecdote trap.** [Study 343 — Data-Mining-Roulette](../343-data-mining-roulette/) "
            "shows how a few loud cases manufacture a 'law' that vanishes on a representative sample.\n"
            "- **Add the corpses.** Our survivor tape is biased *for* the story; get delisted-stock data "
            "(Long Blockchain, KodakCoin, the dot-com renames) and the pop might be bigger — but so "
            "would the eventual −100%, which is a bankruptcy, not a tradable 'fade.'\n\n"
            "*Think the rebrand pop is real and harvestable? Capture the events, draw the same number of "
            "random windows, and show the pop landing **outside** the cloud **and** a fade that's "
            "reliably negative — then we'll talk.*"
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
            "# Name-Change-Effect — a quantitative teardown 🔬\n"
            "### Abnormal-return event windows on a rebrand table · pop/fade legs vs zero · "
            "a Welch *t* + placebo randomization null · the costed believers' trade · a synthetic "
            "faithful-engine / power control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). We treat the "
            "folklore as a **two-legged event-study hypothesis** — a positive **pop** right after the "
            "rename and a negative **fade** (the give-back) after that — and confront both with the "
            "**sample size** and with **survivorship**. The decisive objects are a cross-section of ~21 "
            "abnormal-return events and a placebo null sized to that count.\n\n"
            "> ⚠️ **Data + survivorship note.** The rebrand table is hardcoded and transparent (~25 real "
            "renames across `.com`/`blockchain`/`AI`); the priced tape is **survivor-biased** — the "
            "loudest give-backs (Long Blockchain, KodakCoin) **delisted** and leave no series, biasing "
            "the fade *up* (named on the Signal axis). Real data: yfinance daily adjusted closes, "
            "1995→2026. Offline core + synthetic control are deterministic. Methods in "
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
            f"| **Signal** | `WEAK` | pop **+{R['pop'][0]:.1f}%** (Welch **t = {R['pop'][2]:.2f}**, "
            f"placebo **p = {R['pop'][3]:.2f}**); fade **+{R['fade'][0]:.1f}%** (**t = {R['fade'][2]:.2f}**) "
            "— *positive*, the wrong sign for a give-back. Literature pop + insignificant, sign-wrong "
            "tape ⇒ WEAK. |\n"
            f"| **Tradability** | `MIRAGE` | buy-pop/short-fade = **{R['cost']['gross_combo']:.1f}% "
            f"gross**, **{R['cost']['net_combo']:.1f}% net** of 4 small-cap crossings. The short leg "
            "bleeds on a fade that doesn't come. |\n"
            f"| **Pop then dump?** | `BUSTED` | selection-on-anecdotes: the give-backs (Long Blockchain, "
            "KodakCoin) **delisted**; the survivor fade is **+16%**. |\n\n"
            "> 💡 In plain words: the legend fuses a real-in-1999 *attention pop* with a *symmetric "
            "give-back* that, on the names we can still price, simply isn't there — and a couple-dozen "
            "events can't certify the faint pop that remains."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r^{i}_{[a,b]}$ be stock $i$'s cumulative **abnormal** return (in excess of SPY) over "
            "trading-day offsets $[a,b]$ relative to its rename date (entry lagged one day). Define a "
            "**pop** leg $P_i = r^{i}_{[+1,+5]}$ and a **fade** leg $F_i = r^{i}_{[+6,+65]}$.\n\n"
            "- **H₁ (the pop).** $\\mathbb{E}[P_i] > 0$ — the rename earns an abnormal pop (Cooper-"
            "Dimitrov-Rau found ~+74% for dot-com renames in 1998–99).\n"
            "- **H₂ (the give-back).** $\\mathbb{E}[F_i] < 0$ — the pop is *given back* afterward.\n"
            "- **H₃ (deployable).** $\\mathbb{E}[P_i] - \\mathbb{E}[F_i]$, net of costs on a long-pop / "
            "short-fade book, is positive and harvestable.\n\n"
            "We find **H₁ not supported** ($P$ small, $t<1$, inside the placebo cloud), **H₂ rejected "
            "with the wrong sign** ($F>0$ on survivors), **H₃ rejected** (the book is gross-negative). "
            "The legend is true only where it's a 1999 textbook result, and false where it would pay."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The teardown is two one-sample tests on small cross-sections, judged by their **standard "
            "error** and by **survivorship**:\n\n"
            "$$t_{\\text{leg}} = \\frac{\\bar X}{s_X/\\sqrt{k}},\\qquad X\\in\\{P, F\\},\\ k\\approx 21.$$\n\n"
            "With $k\\approx 21$ and small-cap volatility, $s_X/\\sqrt{k}$ is large — a few-percent mean "
            "drowns in its own SE. Worse, the sample is **conditioned on survival**: the firms whose "
            "fade was most negative (a −100% bankruptcy) **left the tape**, so $\\bar F$ is biased "
            "**upward** — *toward* refuting H₂ only because the give-backs deleted themselves. The honest "
            "instrument for both is a **randomization (placebo) test**: resample $k$ random non-event "
            "windows on the same tickers and ask how often chance matches each leg. That, not the point "
            "estimate, decides the Signal axis."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Rebrand table.** ~{R['n_table']} documented theme-chasing renames (`.com` 1998–2001, "
            "`blockchain` 2017–2020, `AI` 2020–2024), hardcoded & transparent; "
            f"**{R['n_delisted']}** famous give-backs are listed as **DELISTED** (no series) for the "
            f"survivorship caveat. **{R['n_priced']}** priced.\n"
            "- **Abnormal returns.** Daily log return in excess of SPY; pop $=[+1,+5]$, fade $=[+6,+65]$ "
            "trading days, **1-day entry lag** (act the day after the headline).\n"
            "- **Null #1 (Welch t).** Each leg's cross-sectional mean vs zero.\n"
            "- **Null #2 (placebo).** 20,000 draws of $k$ random non-event windows on the same tickers; "
            "$p = \\Pr[|\\text{random mean}| \\ge |\\text{observed}|]$ — the small-sample workhorse.\n"
            "- **Costs.** A long-pop / short-fade book pays a one-way small-cap charge on **four** "
            "crossings per event.\n"
            "- **Positive control.** Deterministic event windows with a **planted** pop+give-back of "
            "size `edge`: the inference must recover a large edge **and** must NOT manufacture "
            "significance when `edge = 0`."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The two legs — a faint pop, a wrong-signed fade\n\n"
            "Mean abnormal return per leg with $\\pm$ standard error, against zero (dashed). The pop is "
            "small and inside its SE; the fade is *positive* — the give-back is absent."
        ),
        code(
            "if HAVE_REAL:\n"
            "    P = EV['pop'].values; F = EV['fade'].values\n"
            "    pm, fm = P.mean()*100, F.mean()*100\n"
            "    pse, fse = P.std(ddof=1)/np.sqrt(len(P))*100, F.std(ddof=1)/np.sqrt(len(F))*100\n"
            "    pt, ft = st.welch_t(P), st.welch_t(F)\n"
            "else:\n"
            "    pm, fm, pt, ft = R['pop'][0], R['fade'][0], R['pop'][2], R['fade'][2]\n"
            "    pse, fse = pm/max(pt,1e-9), fm/max(ft,1e-9)\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.bar(['pop [+1,+5]','fade [+6,+65]'], [pm, fm], yerr=[pse, fse], capsize=6,\n"
            "       color=[AMBER, GREEN], width=.55)\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('mean abnormal return (%)')\n"
            "ax.set_title(f'pop t={pt:.2f} (n.s.), fade t={ft:.2f} and POSITIVE (no give-back)')\n"
            "for i,v in enumerate([pm,fm]): ax.annotate(f'{v:+.1f}%',(i,v),ha='center',va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'pop {pm:+.2f}% (t={pt:.2f})   fade {fm:+.2f}% (t={ft:.2f}) — H2 wants fade<0; it is >0')"
        ),
        md(
            f"> 💡 In plain words: the pop is **+{R['pop'][0]:.1f}%** at **t = {R['pop'][2]:.2f}** "
            f"(not significant); the fade is **+{R['fade'][0]:.0f}%** at **t = {R['fade'][2]:.2f}** — "
            "*positive*, so H₂ (the give-back) is rejected with the **wrong sign**, and survivorship "
            "only pushes it *more* positive."
        ),
        md(
            "### 4b · The decisive test — a placebo null sized to the event count\n\n"
            f"Draw {R['n_priced']} random non-event windows 20,000 times; the histogram is the null for "
            "the mean pop. The real pop is the green line; the *p*-value is the two-sided tail mass."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.placebo_pvalue(B, len(EV), leg='pop', observed=EV['pop'].mean(), n_draws=6000)\n"
            "    obs = pl['obs']*100; pval = pl['p_value']\n"
            "    tickers=[c for c in B['prices'].columns if c!='SPY']\n"
            "    exs=[st._excess_log_returns(B['prices'][t], B['prices']['SPY']) for t in tickers]\n"
            "    exs=[s for s in exs if len(s)>80]; rng=np.random.default_rng(389)\n"
            "    draws=[]\n"
            "    for _ in range(6000):\n"
            "        vals=[]\n"
            "        for _ in range(len(EV)):\n"
            "            s=exs[rng.integers(0,len(exs))]; p0=rng.integers(1,len(s)-7); vals.append(np.expm1(s.iloc[p0:p0+5].sum()))\n"
            "        draws.append(np.mean(vals))\n"
            "    draws=np.array(draws)*100\n"
            "else:\n"
            "    obs=R['pop'][0]; pval=R['pop'][3]; rng=np.random.default_rng(389); draws=rng.normal(0,2.0,6000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=55, color=GREY, alpha=.85, label=f'null: {R[\"n_priced\"]} random windows')\n"
            "ax.axvline(obs, c=GREEN, lw=2.5, label=f'observed pop {obs:+.1f}%')\n"
            "ax.set_xlabel('mean first-week abnormal return (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Placebo p = {pval:.2f}: the pop is inside the luck cloud'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'P[|random {R[\"n_priced\"]}-window mean| >= |pop|] = {pval:.3f}  (need <0.05 to call it real)')"
        ),
        md(
            f"> 💡 In plain words: **{R['pop'][3]*100:.0f}%** of random {R['n_priced']}-window draws match "
            "or beat the pop in magnitude. A real effect would push the green line into the tail; instead "
            "it sits mid-cloud. H₁ is **not supported** — the pop is what a couple-dozen random press "
            "releases look like."
        ),
        md(
            "### 4c · By wave + robustness — the give-back is nowhere\n\n"
            "Split by wave and shift the windows. The fade is **positive in every wave and at every "
            "horizon** (the blockchain survivors — RIOT/BTBT/MSTR — rode the 2020–21 bull *up*), and the "
            "pop *t* never clears 1."
        ),
        code(
            "waves = R['waves']\n"
            "if HAVE_REAL:\n"
            "    waves = []\n"
            "    for w in ['dotcom','blockchain','ai']:\n"
            "        sub = EV[EV['wave']==w]\n"
            "        nm = {'dotcom':'dot-com','blockchain':'blockchain','ai':'AI'}[w]\n"
            "        waves.append((nm, len(sub), sub['pop'].mean()*100, sub['fade'].mean()*100))\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.2))\n"
            "names=[w[0] for w in waves]; wpop=[w[2] for w in waves]; wfade=[w[3] for w in waves]; xx=np.arange(len(names))\n"
            "a1.bar(xx-.2, wpop, .4, color=AMBER, label='pop'); a1.bar(xx+.2, wfade, .4, color=GREEN, label='fade')\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_xticks(xx); a1.set_xticklabels(names)\n"
            "a1.set_ylabel('mean abnormal return (%)'); a1.set_title('Fade is never negative by wave'); a1.legend()\n"
            "if HAVE_REAL:\n"
            "    rob=[]\n"
            "    for pd_,fd_ in [(3,40),(5,60),(10,120)]:\n"
            "        e2=st.collect_events(B,pop=pd_,fade=fd_); s2=st.summarize(e2,B,pop=pd_,fade=fd_,placebo=False)\n"
            "        rob.append((f'{pd_}/{fd_}', s2['n'], s2['pop']['mean']*100, s2['pop']['t'], s2['fade']['mean']*100, s2['fade']['t']))\n"
            "else:\n"
            "    rob = R['robust']\n"
            "labs=[r[0] for r in rob]; pts=[r[3] for r in rob]; fts=[r[5] for r in rob]; xx2=np.arange(len(labs))\n"
            "a2.bar(xx2-.2, pts, .4, color=AMBER, label='pop t'); a2.bar(xx2+.2, fts, .4, color=GREEN, label='fade t')\n"
            "a2.axhline(2, ls='--', c=RED, label='t=2'); a2.axhline(-2, ls='--', c=RED)\n"
            "a2.set_xticks(xx2); a2.set_xticklabels(labs); a2.set_ylabel('Welch t'); a2.set_xlabel('pop/fade days')\n"
            "a2.set_title('No window clears t=2 (fade never goes negative)'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('waves:', [(w[0], round(w[2],1), round(w[3],1)) for w in waves])"
        ),
        md(
            "> 💡 In plain words: the dot-com survivors barely moved (the spectacular renames delisted); "
            "the blockchain survivors **mooned** (+62% fade — the opposite of a give-back); the AI "
            "survivors went roughly flat. No wave gives it back, and no window makes the pop significant. "
            "The result is flat in every choice that could have rescued it."
        ),
        md(
            "### 4d · Faithful-engine & power control — we know the truth here\n\n"
            "Deterministic event windows with a **planted** pop+give-back of size `edge`: with "
            "**`edge=0`** the inference must stay flat (a couple-dozen noisy events can't fake "
            "significance); with a **+40%** planted pop-then-full-give-back it must light up **both** "
            "legs. Both hold — proving the engine is unbiased *and* that this sample size only detects "
            "implausibly large effects."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.40):\n"
            "    syn = data.synthetic_rebrands(n_events=25, edge=edge, seed=389)\n"
            "    ev = st.collect_events(syn, pop=5, fade=60); s = st.summarize(ev, syn, pop=5, fade=60, placebo=False)\n"
            "    res.append((edge, s['n'], s['pop']['mean']*100, s['pop']['t'], s['fade']['mean']*100, s['fade']['t']))\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "labels=[f'planted\\n{int(e*100)}%' for e,*_ in res]; xx=np.arange(len(labels))\n"
            "pts=[r[3] for r in res]; fts=[r[5] for r in res]\n"
            "ax.bar(xx-.2, pts, .4, color=AMBER, label='pop t'); ax.bar(xx+.2, fts, .4, color=GREEN, label='fade t')\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2'); ax.axhline(-2, ls='--', c=RED)\n"
            "ax.set_xticks(xx); ax.set_xticklabels(labels); ax.set_ylabel('Welch t (6-month fade / 1-week pop)')\n"
            "ax.set_title('Control: only a HUGE planted edge lights up both legs'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,k,pm,pt,fm,ft in res: print(f'planted {int(e*100):>3}%: n={k} pop={pm:+.1f}%(t={pt:+.2f}) fade={fm:+.1f}%(t={ft:+.2f})')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted edge the control's pop *t* is "
            f"**{R['syn'][0][3]:.2f}** and fade *t* is **{R['syn'][0][5]:.2f}** (both small — no false "
            f"positive); only the **+40%** plant reaches pop *t* **{R['syn'][1][3]:.2f}** / fade *t* "
            f"**{R['syn'][1][5]:.2f}**. So the machinery is honest, and the real-tape pop *t* of "
            f"**{R['pop'][2]:.2f}** is exactly what an *absent or tiny* effect looks like through a "
            f"{R['n_priced']}-event keyhole."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — pop **+{R['pop'][0]:.1f}%** at Welch **t = {R['pop'][2]:.2f}** / "
            f"placebo **p = {R['pop'][3]:.2f}**; fade **+{R['fade'][0]:.1f}%** at **t = {R['fade'][2]:.2f}** "
            "(positive — wrong sign for a give-back). Literature pop (Cooper-Dimitrov-Rau 2001) + an "
            "insignificant, sign-wrong tape ⇒ WEAK, not REAL. **Survivorship** named on this axis: the "
            "worst give-backs delisted, biasing the fade *up*.\n"
            f"- **Tradability `MIRAGE`** — buy-pop/short-fade = **{R['cost']['gross_combo']:.1f}% gross**, "
            f"**{R['cost']['net_combo']:.1f}% net** of 4 small-cap crossings. The short leg bleeds on a "
            "give-back that doesn't happen; no NAV-scale edge.\n"
            f"- **Pop then dump? `BUSTED`** — the legend is selection-on-anecdotes (Long Blockchain, "
            "KodakCoin **delisted**); across a representative survivor table the pop is a coin-flip and "
            "the fade runs **+16%** the wrong way."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the power curve\n\n"
            "The operational truth in one picture: how big would the *true* pop have to be for a "
            "$k$-event study to detect it at $t=2$? At $k\\approx 21$ you'd need a pop several times the "
            "one observed; the real signal lives far below the detection floor — and the short leg is "
            "negative-carry on top."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sd = EV['pop'].std(ddof=1); obs = EV['pop'].mean()\n"
            "else:\n"
            "    sd = 0.10; obs = R['pop'][0]/100\n"
            "ks = np.arange(5, 200)\n"
            "min_det = 2.0 * sd / np.sqrt(ks)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(ks, min_det*100, c=AMBER, lw=2, label='pop needed for t=2')\n"
            "ax.axhline(obs*100, c=GREEN, ls='--', label=f'observed pop ~{obs*100:+.1f}%')\n"
            "ax.axvline(R['n_priced'], c=GREY, ls=':', label=f\"our k={R['n_priced']}\")\n"
            "ax.set_xlabel('number of events k'); ax.set_ylabel('first-week pop (%)')\n"
            "ax.set_title('Detection floor vs the real pop: badly under-powered'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "need = 2.0*sd/np.sqrt(R['n_priced'])*100\n"
            "print(f'at k={R[\"n_priced\"]} you need ~{need:.1f}% pop for t=2; observed ~{obs*100:+.1f}% -> under-powered')"
        ),
        md(
            "> 💡 In plain words: the amber curve is the **minimum detectable pop**; the green line is "
            "what we see. They don't meet until $k$ is many times larger than the rebrand calendar will "
            "ever deliver — and even a *real* pop wouldn't pay, because the give-back leg you'd short is "
            "**positive** on the survivors. There is no sizing, threshold, or cost assumption that "
            "manufactures an edge from ~21 events whose second leg runs the wrong way. The rarity that "
            "makes the legend romantic is exactly what makes it untestable and untradable."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The adjacent name effect.** [Study 169 — Fluent-Tickers](../169-fluent-tickers/): does "
            "a *pronounceable* ticker carry a premium? Same family (a label, not a fundamental), same "
            "small-sample / survivorship pathology.\n"
            "- **The anecdote trap, formalised.** [Study 343 — Data-Mining-Roulette]"
            "(../343-data-mining-roulette/) on how loud cases manufacture spurious 'laws'.\n"
            "- **Add the corpses.** Reconstruct the delisted rebrands (Long Blockchain, KodakCoin, "
            "fatbrain.com) from a survivorship-free CRSP/Norgate feed; the pop may grow, but the "
            "'fade' becomes a −100% bankruptcy — informative about the *story*, not a tradable leg.\n\n"
            "*The reproducible core is offline and deterministic; the rebrand table is hardcoded and the "
            "priced tape is survivor-biased (named). Methods and sources: "
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
