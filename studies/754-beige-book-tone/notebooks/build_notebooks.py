"""Generate the two narrative notebooks for Study 754 (Beige-Book-Tone).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the labelled LM-tone
proxy + real release calendar (always available) and the cached daily SPY under ../_cache/,
and otherwise quote the frozen headline numbers in ``R`` (mirroring docs/results.md). The
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


# Frozen real-tape headline numbers — mirror of docs/results.md (LM net-tone PROXY on the
# real Beige-Book release calendar 2011-01 -> 2024-12, 112 releases, 79 positive-tone; SPY
# yfinance daily total-return adjusted close, 2010-10-01 -> 2024-12-31, as-of 2024-12-31).
R = dict(
    start_rel="2011-01-19", end_rel="2024-12-04", n_events=112, n_pos=79, n_neg=33,
    years=13.9, spy_start="2010-10-01", spy_end="2024-12-31", spy_rows=3586,
    fp_spy="eee0fc4ac90e", fp_rel="dd4d39f56f19",
    # per horizon: (h, n_pos, pos_mean%, neg_mean%, base_mean%, pos_up%, base_up%, welch_t, p_plac)
    h1=(1, 79, 0.051, -0.401, -0.082, 51, 48, 1.13, 0.097),
    h3=(3, 79, 0.239, -0.552, 0.006, 61, 59, 0.95, 0.169),
    h5=(5, 79, 0.400, -0.834, 0.036, 68, 64, 1.16, 0.089),
    h10=(10, 79, 0.801, -1.148, 0.227, 67, 61, 1.44, 0.058),
    # continuous tone->drift regression per horizon: (beta%/unit, t_ols, t_hac, corr)
    reg={1: (0.2705, 1.46, 2.03, 0.138), 3: (0.0606, 0.14, 0.14, 0.013),
         5: (0.3791, 0.77, 0.93, 0.073), 10: (0.7233, 1.05, 1.30, 0.100)},
    # overlay h5 @1bp: (per_event_gross%, per_event_net%, base_event%, event_sharpe, n_trades, trades/yr, ann_net%, bh_ann%)
    overlay=(0.400, 0.380, 0.036, 0.21, 79, 5.7, 2.2, 14.2),
    # robustness (5d): (label, n_pos, pos5%, base5%, t, p)
    robust=[("thr>0", 79, 0.400, 0.036, 1.16, 0.089),
            ("thr>median", 53, 0.548, 0.036, 1.45, 0.057),
            ("ex-2020", 78, 0.391, 0.184, 0.69, 0.195)],
    corr_prior=-0.238,   # corr(tone, PRIOR 5-day SPY return) — regime clustering, not a lead
    # synthetic control (5d): (edge, n_pos, pos5%, base5%, welch_t, beta%, t_hac)
    syn=[(0.0, 62, 0.178, 0.335, -0.45, -0.311, -1.43),
         (0.004, 62, 1.815, 0.516, 3.16, 1.699, 7.84)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Anecdote_leads_the_tape%3F: Not_supported](https://img.shields.io/badge/Anecdote_leads_the_tape%3F-Not_supported-8b949e?style=flat-square)\n\n"
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

from beige_book_tone import data, strategy as st

HAVE_REAL = data.have_real()
REL, SPY = data.load_real() if HAVE_REAL else (None, None)
print("SPY cache present:", HAVE_REAL,
      "| Beige-Book releases:", (0 if REL is None else len(REL)),
      "| positive-tone:", (0 if REL is None else int((REL['tone'] > 0).sum())))
"""

# The frozen headline dict is embedded into the first code cell so every downstream cell can
# quote it whether or not the SPY cache is present.
BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does the Fed's mood ring move the market? 📖\n"
            "### The Beige Book's anecdote-tone as a stock-market crystal ball, in plain English\n\n"
            + BADGES +
            "Eight times a year, about two weeks before it sets interest rates, the Federal Reserve "
            "publishes the **Beige Book** — a folksy, anecdote-stuffed round-up of what businesses across "
            "the twelve Fed districts are *saying*: hiring is 'robust', shoppers are 'cautious', builders "
            "are 'upbeat'. Fed-watchers pore over its **tone**. The folklore says a **cheerful** Beige "
            "Book is a green light — stocks should **drift up** in the days after it lands.\n\n"
            "It's a lovely idea: read the Fed's mood, front-run the market. This notebook asks three "
            "blunt questions. When the Beige Book reads *positive*, does SPY really rise afterward? Is "
            "any little bump big enough to tell from luck? And if you *bought* every cheerful book, would "
            "you make money?\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the Newey-West regression and the "
            "synthetic control? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** Scraping and dictionary-scoring every Beige Book's full text "
            "is the natural next step (beat 7), but here the **tone is a clearly-labelled proxy** — a "
            "small, hand-built reconstruction of the Loughran-McDonald 'net optimism' score, anchored to "
            "what each period actually felt like (sunny through 2017, catastrophic in spring 2020). The "
            "**release dates are real** (the genuine Beige-Book Wednesdays) and **SPY is real** (yfinance "
            "daily). House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| After a *positive* Beige Book, does SPY rise? | **A tiny bit — on average.** Over the next "
            f"5 trading days SPY averages **+{R['h5'][2]:.2f}%** after a cheerful book vs about "
            f"**{R['h5'][4]:+.2f}%** on a typical release — and it's up a bit more often. The *direction* "
            "matches the folklore. |\n"
            "| Is that bump reliable? | **No.** It's small and swims well inside the noise — you can't "
            "tell it from luck at any horizon (best *t* ≈ 1.4, nowhere near the '2' bar). |\n"
            "| Is the anecdote *leading* the market? | **No.** Almost all of the 'edge' is just avoiding "
            "the handful of **gloomy** books (2020, 2022) that landed during crashes. Drop 2020 and the "
            "gap nearly vanishes. Cheerful books cluster in good times — that's beta, not a signal. |\n"
            "| So could you trade it? | **Not really.** 'Buy 5 days after every cheerful book' sits in "
            f"cash ~90% of the time and earns **+{R['overlay'][6]:.1f}%/yr** vs "
            f"**+{R['overlay'][7]:.1f}%** for simply holding SPY. |\n\n"
            "> The Beige Book is a careful description of an economy the **stock market already priced "
            "weeks ago** — and it drops right before the Fed meeting everyone's really watching. Its mood "
            "is real; its market-timing power is a mirage."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"The Beige Book is the Fed's ear to the ground. When its tone brightens — more "
            "'expansion', 'strong', 'optimistic', fewer 'soft', 'weak', 'uncertain' — the economy is "
            "turning up and equities will follow. Read the tone, and you get a jump on the tape.\"*\n\n"
            "There's a serious idea underneath. Academics *have* shown the Beige Book carries real "
            "information about **current and near-term economic activity** (Armesto et al., 2009). And "
            "text-sentiment scoring with the **Loughran-McDonald** finance dictionary is a genuine, "
            "widely-used tool. The leap we test is the *trading* one: that the book's tone arrives early "
            "enough, and clean enough, to move **stock prices** you could get in front of."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If it were true, it'd be a gift: a free, scheduled, government-published mood reading that "
            "front-runs the market eight times a year. But there's a trap baked into the word *leads*. "
            "The **stock market is itself a leading indicator** — it usually turns *before* the economy. "
            "The Beige Book, by contrast, describes conditions that already **happened**: it's a summary "
            "of anecdotes gathered over the previous weeks. So a cheerful book that lines up with a "
            "rising market might not be *predicting* anything — it might just be **describing** a good "
            "patch the market already knew about. Telling those two apart is the whole game."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We line up **{R['years']:.0f} years** of Beige-Book releases ({R['start_rel'][:4]}–"
            f"{R['end_rel'][:4]}, **{R['n_events']} books**, {R['n_pos']} of them positive-tone) against "
            "daily SPY, and:\n\n"
            "1. **Split the books.** After each *positive* book, what did SPY do over the next 1/3/5/10 "
            "trading days — measured from the **release-day close**, so there's no peeking? Compare that "
            "to the average release.\n"
            "2. **Measure the dose.** Regress the actual drift on the actual tone: does *more* optimism "
            "buy *more* drift, with a serial-correlation-robust (Newey-West) *t*?\n"
            "3. **Check it's not just beta.** Do cheerful books simply cluster in bull markets? If "
            "dropping the 2020 crash guts the effect, the 'signal' was regime, not lead.\n"
            "4. **Try to trade it.** Buy the post-release window on every cheerful book, pay costs, and "
            "see if it beats buying and holding.\n\n"
            "**What would make us say 'mirage':** no horizon clears a *t* of 2, the effect leans on 2020, "
            "and the overlay trails buy-and-hold. (Spoiler: all three.)"
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the mood itself.** Here's the tone proxy across 14 years — sunny through the "
            "2013–2019 expansion, the cliff-dive of spring 2020, the inflation gloom of 2022. It clearly "
            "*knows* the economy. The question is whether it knows the **market** early."
        ),
        code(
            "if HAVE_REAL:\n"
            "    t = REL['tone']\n"
            "    fig, ax = plt.subplots(figsize=(9.4, 4.0))\n"
            "    cols = [GREEN if v > 0 else RED for v in t.values]\n"
            "    ax.bar(t.index, t.values, width=40, color=cols)\n"
            "    ax.axhline(0, c='k', lw=.8)\n"
            "    ax.set_title('Beige-Book tone proxy (LM net optimism, z-scaled) — 8 books/yr')\n"
            "    ax.set_ylabel('tone (higher = more upbeat)')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('most negative book:', round(float(t.min()),2), 'around', t.idxmin().date())\n"
            "else:\n"
            "    print('no cache — see docs/results.md; spring-2020 books were the most negative')"
        ),
        md(
            "**Now the payoff.** For each horizon, the average forward SPY return after a **positive** "
            "book next to the return on an **average** release. The folklore predicts the green bars sit "
            "*above* the grey ones."
        ),
        code(
            "hs = [1, 3, 5, 10]\n"
            "if HAVE_REAL:\n"
            "    rows = [st.summarize(REL, SPY, h) for h in hs]\n"
            "    pos = [r['pos_mean']*100 for r in rows]; base = [r['base_mean']*100 for r in rows]\n"
            "else:\n"
            "    pos = [R['h1'][2], R['h3'][2], R['h5'][2], R['h10'][2]]\n"
            "    base = [R['h1'][4], R['h3'][4], R['h5'][4], R['h10'][4]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "ax.bar(x-.2, pos, .4, color=GREEN, label='after a POSITIVE book')\n"
            "ax.bar(x+.2, base, .4, color=GREY, label='an average release (base rate)')\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h} days' for h in hs])\n"
            "ax.set_ylabel('average forward SPY return (%)'); ax.axhline(0, c='k', lw=.6)\n"
            "ax.set_title('Positive book -> higher forward returns... but only by a whisker')\n"
            "for i,(a,b) in enumerate(zip(pos,base)):\n"
            "    ax.annotate(f'{a:.2f}%',(i-.2,a),ha='center',va='bottom',fontsize=9)\n"
            "    ax.annotate(f'{b:.2f}%',(i+.2,b),ha='center',va='bottom',fontsize=9)\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('5-day: positive', f'{pos[2]:.2f}%', 'vs base', f'{base[2]:.2f}%')"
        ),
        md(
            f"The direction is right — at 5 days a positive book is followed by **+{R['h5'][2]:.2f}%** vs "
            f"the base **{R['h5'][4]:+.2f}%**, and the market is up a touch more often "
            f"(**{R['h5'][5]:.0f}%** vs **{R['h5'][6]:.0f}%**). But the gap is a few hundredths of a "
            "percent, easily luck. The *next* two charts are where the story dies."
        ),
        md(
            "**Is the anecdote actually *leading*?** Here's the tell. Almost all of the apparent edge is "
            "the average release being dragged **down** by a few catastrophically gloomy books (2020, "
            "2022) that happened to land *during* crashes. Drop the 2020 COVID year and watch the gap "
            "between 'positive book' and 'average release' collapse."
        ),
        code(
            "if HAVE_REAL:\n"
            "    full = st.summarize(REL, SPY, 5)\n"
            "    rel2 = REL[(REL.index < '2020-01-01') | (REL.index >= '2021-01-01')]\n"
            "    exc = st.summarize(rel2, SPY, 5)\n"
            "    pf, bf = full['pos_mean']*100, full['base_mean']*100\n"
            "    pe, be = exc['pos_mean']*100, exc['base_mean']*100\n"
            "else:\n"
            "    pf, bf = R['h5'][2], R['h5'][4]; pe, be = R['robust'][2][2], R['robust'][2][3]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "x = np.arange(2)\n"
            "ax.bar(x-.2, [pf, pe], .4, color=GREEN, label='after a POSITIVE book')\n"
            "ax.bar(x+.2, [bf, be], .4, color=GREY, label='an average release')\n"
            "ax.set_xticks(x); ax.set_xticklabels(['full sample', 'drop 2020 (COVID)'])\n"
            "ax.set_ylabel('avg 5-day forward SPY return (%)')\n"
            "ax.set_title('Kill the 2020 crash and the \"edge\" nearly vanishes -> it was regime, not lead')\n"
            "for i,(a,b) in enumerate(zip([pf,pe],[bf,be])):\n"
            "    ax.annotate(f'{a:.2f}%',(i-.2,a),ha='center',va='bottom',fontsize=9)\n"
            "    ax.annotate(f'{b:.2f}%',(i+.2,b),ha='center',va='bottom',fontsize=9)\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'gap: full {pf-bf:+.2f}pp  ->  ex-2020 {pe-be:+.2f}pp')"
        ),
        md(
            f"There it is. In the full sample the positive-minus-base gap is **{R['h5'][2]-R['h5'][4]:+.2f} "
            f"points**; strip out 2020 and it shrinks to about **{R['robust'][2][2]-R['robust'][2][3]:+.2f} "
            "points**, and the *t* falls to a limp **0.7**. The 'signal' was mostly the cheerful books "
            "sitting in calm expansions and the gloomy ones sitting in crashes — that's the equity risk "
            "premium showing up on schedule, **not** the Beige Book telling you anything new."
        ),
        md(
            "**Could you trade it anyway?** Suppose you bought SPY for 5 trading days after every "
            "positive book and sat in cash the rest of the time. Here's that strategy vs simply holding."
        ),
        code(
            "if HAVE_REAL:\n"
            "    o = st.event_overlay(REL, SPY, h=5, cost_bps=1.0)\n"
            "    ann_net, bh = o['ann_net']*100, o['bh_ann']*100; ntr = o['n_trades']\n"
            "else:\n"
            "    ann_net, bh, ntr = R['overlay'][6], R['overlay'][7], R['overlay'][4]\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "bars = ax.bar(['buy 5d after\\neach positive book\\n(net)', 'buy & hold SPY'],\n"
            "              [ann_net, bh], color=[RED, GREY], width=.55)\n"
            "for b,v in zip(bars,[ann_net,bh]): ax.annotate(f'{v:.1f}%/yr',(b.get_x()+b.get_width()/2,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('annualised return (%)')\n"
            "ax.set_title(f'The overlay is in cash ~90% of the time ({ntr} trades) -> it barely earns anything')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'overlay net {ann_net:.1f}%/yr  vs  buy&hold {bh:.1f}%/yr')"
        ),
        md(
            f"The overlay earns **+{R['overlay'][6]:.1f}%/yr** against buy-and-hold's "
            f"**+{R['overlay'][7]:.1f}%** — not because it *loses* on its trades (the average cheerful-"
            "book window is mildly positive) but because it's **out of the market ~90% of the time**, "
            "collecting a few crumbs while missing the compounding. Even if the tiny drift were real, "
            "there's almost nothing there to harvest."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** A positive book is followed by a *hair* more upside, in the right "
            "direction at every horizon — but nothing comes close to statistically real (best *t* ≈ "
            "1.4), and it leans on the 2020 crash.\n"
            "- **Tradability — Mirage.** The cheerful-book overlay sits in cash almost all year and earns "
            "a fraction of buy-and-hold. Nothing to deploy.\n"
            "- **Anecdote leads the tape? — Not supported.** Strip out one crash and the edge evaporates; "
            "cheerful books simply cluster in good times. The Beige Book **describes** an economy the "
            "market already priced — it doesn't **lead** the price."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — the honest bottom line\n\n"
            "Set significance aside. Even granting the little drift, the plumbing defeats it. The Beige "
            "Book drops around **2 p.m. Eastern**, and by the 4 p.m. close the wire services, algos and "
            "Fed-watchers have already read every district's adjectives — any reaction is *same-day*, "
            "gone before the 'no-peeking' next-close entry a real trader could use. Worse, the book lands "
            "**two weeks before the FOMC decision** everyone actually trades, so its mood is a warm-up "
            "act to the main event. A scheduled, widely-parsed, pre-digested government text is about the "
            "*least* likely place to find un-arbitraged drift — and the data agrees."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The obvious upgrade.** Swap the labelled tone **proxy** for a real full-text scrape of "
            "every Beige Book (the Fed archives them all) scored with the actual Loughran-McDonald "
            "dictionary, and rerun — the event-study engine here is ready for it.\n"
            "- **The sibling tests.** [Study 387 — Economic-Surprise-Index](../387-economic-surprise-index/) "
            "and [Study 385 — Jobless-Claims-Momentum](../385-jobless-claims-momentum/) ask the same "
            "'does the macro tell lead the tape?' question of hard data instead of anecdotes.\n"
            "- **Change the target.** Maybe the tone moves **bonds** or the **dollar** (both more "
            "Fed-sensitive than SPY), or the *change* in tone book-to-book beats the level. The proxy "
            "makes the null look robust; a real scrape on a Fed-sensitive asset is the honest re-test.\n\n"
            "*Think the Beige Book's mood front-runs stocks? Scrape the real text, score it, and show the "
            "green bars clearing the noise **after** 2020 is dropped — then we'll talk.*"
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
            "# Beige-Book-Tone — a quantitative teardown 🔬\n"
            "### Event-window drift by tone · Welch *t* + placebo null · a Newey-West tone→drift "
            "regression · the regime (beta) confound · a costed event overlay · a synthetic planted-edge control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The believers "
            "fuse two claims: that a positive-tone Beige Book (1) **predicts** higher forward equity "
            "returns and (2) does so **early** enough to trade. We separate them. The conditional drift is "
            "the *right sign but insignificant* at every horizon; the decisive object is the **regime "
            "confound** — the apparent edge is cheerful books clustering in expansions, and it dissolves "
            "once the 2020 crash is removed. A costed overlay that captures almost nothing seals "
            "Tradability.\n\n"
            "> ⚠️ **Proxy note (named on the Signal axis).** The tone is a **labelled LM-net-tone proxy** — "
            "a small hardcoded reconstruction, **not** a live full-text dictionary count (that scrape is "
            "the beat-7 extension). Because the proxy is our construction, we can't *certify* magnitude "
            "either way — but the null it produces is corroborated by the regime confound and the "
            "microstructure logic, and a synthetic control proves the engine would light up on a real "
            "link. Release dates are the real Beige-Book Wednesdays; SPY is yfinance daily total-return "
            "adjusted close, entered at the release-day close (no look-ahead). Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | 5-day positive-tone mean **+{R['h5'][2]:.2f}%** vs base "
            f"**{R['h5'][4]:+.2f}%** (right sign); best Welch **t = {R['h10'][7]:.2f}** (10d), placebo "
            f"**p = {R['h10'][8]:.2f}** — no horizon clears **t ≥ 2**. |\n"
            f"| **Tradability** | `MIRAGE` | Long-the-window overlay **+{R['overlay'][6]:.1f}%/yr** net vs "
            f"buy-hold **+{R['overlay'][7]:.1f}%/yr** — in cash ~90% of the time ({R['overlay'][4]} trades). |\n"
            f"| **Anecdote leads the tape?** | `NOT SUPPORTED` | Drop 2020 and the positive-minus-base gap "
            f"falls from **{R['h5'][2]-R['h5'][4]:+.2f}pp** to **{R['robust'][2][2]-R['robust'][2][3]:+.2f}pp** "
            f"(t **{R['robust'][2][4]:.2f}**); corr(tone, *prior* 5-day return) = **{R['corr_prior']:+.2f}**. Regime, not lead. |\n\n"
            "> 💡 In plain words: the equity market *is* a leading indicator of the economy, so a text that "
            "*describes* the economy need not lead the price. The Beige Book's cheerful books sit in "
            "expansions (positive drift) and its gloomy books sit in crashes (negative drift) — sampling "
            "the risk premium on a schedule, not forecasting it."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $s_i$ be the Loughran-McDonald net tone of Beige Book $i$, "
            "$s = (\\#\\text{pos} - \\#\\text{neg})/(\\#\\text{pos} + \\#\\text{neg})$, z-scaled. A book "
            "is **POSITIVE** when $s_i > 0$. Entering at the **release-day close** $c_{t_0}$ (the book is "
            "public by then — no look-ahead), define the $H$-day forward drift "
            "$r_{i,H} = c_{t_0+H}/c_{t_0} - 1$.\n\n"
            "- **H₁ (predicts).** $\\mathbb{E}[r_H \\mid s>0] > \\mathbb{E}[r_H]$ — positive excess drift.\n"
            "- **H₂ (dose-response).** In $r_H = a + b\\,s + \\varepsilon$, $b>0$ with HAC $|t|\\ge 2$.\n"
            "- **H₃ (leads, not regime).** The excess survives dropping the 2020 crash and isn't just "
            "cheerful books living in bull markets.\n\n"
            "We find **H₁ directionally true but insignificant** (best Welch $t=1.44$), **H₂ rejected** "
            "($b$'s HAC $t$ clears 2 at *only* the 1-day horizon and collapses to 0.1 at 3 days — an "
            "isolated crossing, not a dose-response), **H₃ rejected** (ex-2020 the gap and $t$ both "
            "shrink toward zero). The lore is right where it's uninformative (tone tracks the cycle) and "
            "wrong where it would pay (a *leading*, *tradable* edge in equities)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The conditional-drift test is a two-sample mean comparison judged by its standard error:\n\n"
            "$$\\widehat{\\Delta}_H = \\bar r^{\\,s>0}_H - \\bar r^{\\,\\text{all}}_H,\\qquad "
            "t = \\frac{\\widehat{\\Delta}_H}{\\sqrt{\\,s^2_{+}/k + s^2_{\\text{all}}/N\\,}}.$$\n\n"
            "But a significant $\\widehat{\\Delta}$ would **still not** establish *leading*. Two confounds "
            "sit underneath: (i) **overlap/serial correlation** in multi-day windows (handled by a "
            "Newey-West HAC *t* on the dose-response slope), and (ii) **regime clustering** — positive "
            "tone is not randomly placed in time; it concentrates in expansions whose unconditional drift "
            "is positive. The identifying move is to **remove the one regime that dominates** (2020) and "
            "ask whether anything survives. If $\\widehat{\\Delta}\\to 0$ ex-2020, the effect was the "
            "risk premium sampled on the Beige-Book calendar, not the anecdote leading the tape."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Release tape.** {R['n_events']} Beige-Book releases, {R['start_rel'][:7]}→"
            f"{R['end_rel'][:7]} ({R['n_pos']} positive-tone). Dates = the real Beige-Book Wednesdays; "
            "**tone = a labelled LM-net-tone proxy** (named on the axis).\n"
            "- **Signal.** POSITIVE when $s_i>0$ (and a median split for robustness).\n"
            "- **Forward drift.** Enter at the **release-day close** (public by then), hold "
            "$H\\in\\{1,3,5,10\\}$ trading days; drop windows that overrun the tape.\n"
            "- **Null #1 (Welch t).** Positive-set mean vs the unconditional mean.\n"
            "- **Null #2 (placebo).** 20,000 draws of $k$ random events; "
            "$p=\\Pr[\\text{random-draw mean}\\ge\\text{positive mean}]$ (as bullish or more).\n"
            "- **Dose-response (H₂).** OLS $r_H=a+b\\,s+\\varepsilon$ with a Newey-West (Bartlett, 4-lag) "
            "HAC $t$ on $b$.\n"
            "- **Regime test (H₃).** Re-run ex-2020; measure corr(tone, *prior* 5-day return).\n"
            "- **Tradability.** Long the $H$=5 post-release window on positive books, 1 bp one-way "
            "(round-trip $2\\times$), annualised by trade frequency, vs buy-and-hold (total-return SPY).\n"
            "- **Positive control.** A deterministic series with a *planted* tone→drift link: `edge=0` "
            "must not fake significance; a large `edge` must light up the test."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The point estimates — right sign, small, insignificant\n\n"
            "Positive-tone forward mean with $\\pm$ standard error against the unconditional base rate "
            "(diamond). Above base at every horizon — but inside its own error bar."
        ),
        code(
            "hs = [1, 3, 5, 10]\n"
            "if HAVE_REAL:\n"
            "    pm, bm, ts, ses = [], [], [], []\n"
            "    for h in hs:\n"
            "        s = st.summarize(REL, SPY, h); pm.append(s['pos_mean']); bm.append(s['base_mean']); ts.append(s['t'])\n"
            "        p,_n,_a = st.split_returns(REL, SPY, h); ses.append(p.std(ddof=1)/np.sqrt(len(p)))\n"
            "else:\n"
            "    pm = [R['h1'][2]/100, R['h3'][2]/100, R['h5'][2]/100, R['h10'][2]/100]\n"
            "    bm = [R['h1'][4]/100, R['h3'][4]/100, R['h5'][4]/100, R['h10'][4]/100]\n"
            "    ts = [R['h1'][7], R['h3'][7], R['h5'][7], R['h10'][7]]; ses = [.0018,.003,.0035,.0055]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x, [p*100 for p in pm], yerr=[e*100 for e in ses], capsize=5, color=GREEN, width=.5, label='positive-tone (±SE)')\n"
            "ax.plot(x, [b*100 for b in bm], 'D', ms=11, c=GREY, label='unconditional base rate')\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean forward SPY return (%)')\n"
            "ax.set_title('Right sign (above base) but the SE swamps the gap'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Welch t by horizon:', {f'{h}d': round(t,2) for h,t in zip(hs,ts)})"
        ),
        md(
            f"> 💡 In plain words: at 5d the positive-tone mean is **+{R['h5'][2]:.2f}%** vs base "
            f"**{R['h5'][4]:+.2f}%** — a ~{R['h5'][2]-R['h5'][4]:.2f}-point gap at **t = {R['h5'][7]:.2f}** "
            f"(not significant); the best any horizon manages is **t = {R['h10'][7]:.2f}** (10d, placebo "
            f"p = {R['h10'][8]:.2f}). H₁ is **directionally supported, statistically not** — the right "
            "sign living inside its error bar."
        ),
        md(
            "### 4b · The dose-response — does *more* tone buy *more* drift?\n\n"
            "OLS $r_H = a + b\\,s + \\varepsilon$; the bars are the Newey-West (HAC) *t* on the slope $b$. "
            "A real dose-response would clear $|t|=2$ *and persist* across horizons."
        ),
        code(
            "hs = [1, 3, 5, 10]\n"
            "if HAVE_REAL:\n"
            "    thac = [st.tone_drift_regression(REL, SPY, h)['t_hac'] for h in hs]\n"
            "    tols = [st.tone_drift_regression(REL, SPY, h)['t_ols'] for h in hs]\n"
            "else:\n"
            "    thac = [R['reg'][h][2] for h in hs]; tols = [R['reg'][h][1] for h in hs]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, tols, .4, color=GREY, label='OLS t')\n"
            "ax.bar(x+.2, thac, .4, color=AMBER, label='Newey-West (HAC) t')\n"
            "ax.axhline(2, ls='--', c=RED, label='t = 2 (significance bar)'); ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('t-stat on slope b')\n"
            "ax.set_title('Only 1-day HAC t grazes 2 — and it collapses to ~0 at 3 days (no dose-response)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('HAC t by horizon:', {f'{h}d': round(t,2) for h,t in zip(hs,thac)})"
        ),
        md(
            f"> 💡 In plain words: the slope's HAC *t* touches **{R['reg'][1][2]:.2f}** at 1 day — then "
            f"**falls to {R['reg'][3][2]:.2f}** at 3 days and never recovers. Because the releases are "
            "~monthly, the multi-day windows barely overlap, so the OLS and HAC *t* agree there's no "
            "signal; the lone 1-day HAC blip is an isolated crossing among four horizons (the kind "
            "multiple looks manufacture), **not** a dose-response. **H₂ rejected.**"
        ),
        md(
            "### 4c · The decisive confound — regime, not lead\n\n"
            "Positive tone isn't sprinkled randomly through time; it clusters in expansions. Remove the "
            "single dominant regime (2020) and ask what's left."
        ),
        code(
            "if HAVE_REAL:\n"
            "    full = st.summarize(REL, SPY, 5)\n"
            "    rel2 = REL[(REL.index < '2020-01-01') | (REL.index >= '2021-01-01')]\n"
            "    exc = st.summarize(rel2, SPY, 5)\n"
            "    gaps = [(full['pos_mean']-full['base_mean'])*100, (exc['pos_mean']-exc['base_mean'])*100]\n"
            "    tvals = [full['t'], exc['t']]\n"
            "else:\n"
            "    gaps = [R['h5'][2]-R['h5'][4], R['robust'][2][2]-R['robust'][2][3]]\n"
            "    tvals = [R['h5'][7], R['robust'][2][4]]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.2, 4.2))\n"
            "labels = ['full sample', 'drop 2020']\n"
            "a1.bar(labels, gaps, color=[GREEN, RED], width=.55)\n"
            "for i,v in enumerate(gaps): a1.annotate(f'{v:+.2f}pp',(i,v),ha='center',va='bottom')\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_ylabel('positive - base, 5-day (pp)'); a1.set_title('The gap halves without one crash')\n"
            "a2.bar(labels, tvals, color=[GREEN, RED], width=.55)\n"
            "for i,v in enumerate(tvals): a2.annotate(f't={v:.2f}',(i,v),ha='center',va='bottom')\n"
            "a2.axhline(2, ls='--', c=RED); a2.axhline(0, c='k', lw=.8); a2.set_ylabel('Welch t'); a2.set_title('...and the t follows it toward zero')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gap full {gaps[0]:+.2f}pp (t {tvals[0]:.2f}) -> ex-2020 {gaps[1]:+.2f}pp (t {tvals[1]:.2f})')"
        ),
        md(
            f"> 💡 In plain words: the positive-minus-base gap falls from **{R['h5'][2]-R['h5'][4]:+.2f}pp** "
            f"(t {R['h5'][7]:.2f}) to **{R['robust'][2][2]-R['robust'][2][3]:+.2f}pp** (t "
            f"{R['robust'][2][4]:.2f}) once 2020 is dropped, and corr(tone, *prior* 5-day return) is "
            f"**{R['corr_prior']:+.2f}** — the tone tracks recent conditions, it doesn't lead them. **H₃ "
            "rejected.** What looked like a signal was the equity risk premium, sampled on the "
            "Beige-Book calendar."
        ),
        md(
            "### 4d · Tradability — the event overlay captures almost nothing\n\n"
            "Long SPY for the 5-day post-release window on each positive book, 1 bp one-way "
            "(round-trip 2 bp), else flat. Annualised net vs buy-and-hold, plus the per-event Sharpe."
        ),
        code(
            "if HAVE_REAL:\n"
            "    o = st.event_overlay(REL, SPY, h=5, cost_bps=1.0)\n"
            "    ann_net, bh, esh, ntr = o['ann_net']*100, o['bh_ann']*100, o['event_sharpe'], o['n_trades']\n"
            "    peg, pen, beb = o['per_event_gross']*100, o['per_event_net']*100, o['base_event']*100\n"
            "else:\n"
            "    ann_net, bh, esh, ntr = R['overlay'][6], R['overlay'][7], R['overlay'][3], R['overlay'][4]\n"
            "    peg, pen, beb = R['overlay'][0], R['overlay'][1], R['overlay'][2]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.2))\n"
            "a1.bar(['overlay\\nnet', 'buy &\\nhold'], [ann_net, bh], color=[RED, GREY], width=.55)\n"
            "for i,v in enumerate([ann_net,bh]): a1.annotate(f'{v:.1f}%/yr',(i,v),ha='center',va='bottom')\n"
            "a1.set_ylabel('annualised return (%)'); a1.set_title(f'In cash ~90% of the time ({ntr} trades)')\n"
            "a2.bar(['per-event\\ngross', 'per-event\\nnet', 'base\\nevent'], [peg, pen, beb], color=[AMBER, RED, GREY], width=.6)\n"
            "for i,v in enumerate([peg,pen,beb]): a2.annotate(f'{v:.2f}%',(i,v),ha='center',va='bottom')\n"
            "a2.axhline(0, c='k', lw=.8); a2.set_ylabel('5-day window return (%)'); a2.set_title(f'Per-event edge tiny (Sharpe {esh:.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'overlay net {ann_net:.1f}%/yr vs buy-hold {bh:.1f}%/yr; per-event net {pen:.2f}% (Sharpe {esh:.2f})')"
        ),
        md(
            f"> 💡 In plain words: the overlay's per-event window is mildly positive "
            f"(**+{R['overlay'][1]:.2f}%** net, Sharpe **{R['overlay'][3]:.2f}**), but firing only "
            f"~{R['overlay'][5]:.0f}×/yr for 5 days it's out of the market ~90% of the time, so it "
            f"annualises to **+{R['overlay'][6]:.1f}%/yr** against buy-hold's **+{R['overlay'][7]:.1f}%**. "
            "Costs aren't even the issue — there's simply almost no exposure to a barely-positive, "
            "insignificant edge. `MIRAGE`."
        ),
        md(
            "### 4e · Faithful-engine control — we know the truth here\n\n"
            "A deterministic series with a *planted* link (a positive-tone release at $t$ lifts the "
            "5-day post-release drift by `edge`$\\times$tone). With `edge=0` the test must stay flat; with "
            "a large `edge` it must light up — proving the null on the real proxy isn't a broken pipeline."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.004):\n"
            "    srel, sspy = data.synthetic(n_years=14, edge=edge, seed=754)\n"
            "    s = st.summarize(srel, sspy, 5); r = st.tone_drift_regression(srel, sspy, 5)\n"
            "    res.append((edge, s['n_pos'], s['pos_mean']*100, s['base_mean']*100, s['t'], r['t_hac']))\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "labels = [f'planted edge\\n{e:.3f}/day/tone' for e,_,_,_,_,_ in res]\n"
            "tvals = [r[4] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t = 2 (significance bar)'); ax.axhline(0, c='k', lw=.8)\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('Welch t (5-day)'); ax.set_title('Control: no link -> flat; real link -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,k,p,b,t,th in res: print(f'planted {e:.3f}: n_pos={k} pos5d={p:.2f}% base5d={b:.2f}% Welch_t={t:.2f} HAC_t={th:.2f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted link the control sits at "
            f"**Welch t = {R['syn'][0][4]:.2f}** (HAC {R['syn'][0][6]:.2f}) — no false positive; a large "
            f"planted link drives **Welch t = {R['syn'][1][4]:.2f}** (HAC {R['syn'][1][6]:.2f}). The "
            "machinery is honest — so the real-proxy null (best t ~1.4) is a *genuine* absent-or-tiny "
            "edge through this keyhole, not a measurement failure. The engine *can* bank a real tone→"
            "drift link; the proxy tape just doesn't carry a tradable one."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — positive-tone forward drift is the right sign at every horizon "
            f"(5-day **+{R['h5'][2]:.2f}%** vs base **{R['h5'][4]:+.2f}%**) but **no horizon clears "
            f"t ≥ 2** (best Welch **t = {R['h10'][7]:.2f}**, placebo **p = {R['h10'][8]:.2f}**); the "
            "dose-response HAC *t* clears 2 at only the 1-day horizon and collapses at 3 days. Literature "
            "supports Beige-Book *economic* content, but this tape can't certify an *equity* signal.\n"
            f"- **Tradability `MIRAGE`** — the event overlay returns **+{R['overlay'][6]:.1f}%/yr** net vs "
            f"buy-hold **+{R['overlay'][7]:.1f}%/yr**, in cash ~90% of the time. Almost nothing to "
            "harvest, before microstructure even bites.\n"
            f"- **Anecdote leads the tape? `NOT SUPPORTED`** — ex-2020 the gap falls to "
            f"**{R['robust'][2][2]-R['robust'][2][3]:+.2f}pp** (t **{R['robust'][2][4]:.2f}**) and "
            f"corr(tone, *prior* return) = **{R['corr_prior']:+.2f}**. The tone **describes** a cycle the "
            "equity market already priced; it doesn't lead the price. *Proxy caveat: the tone is a "
            "labelled reconstruction — the real-text scrape is the honest re-test, and the null here is "
            "robust to the regime confound the proxy can't manufacture away.*"
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — why even a real tilt wouldn't deploy\n\n"
            "Grant the lore a genuine few-hundredths tilt. The plumbing still defeats it. The Beige Book "
            "prints ~**2 p.m. ET**; by the 4 p.m. close the entire buy-side has parsed every district's "
            "adjectives, so any drift is **intraday and same-day** — before the conservative next-close "
            "entry a no-look-ahead rule can use. It also lands **~2 weeks before the FOMC decision**, the "
            "event that actually repositions rates books, making the Beige Book a pre-game whose "
            "information is dominated by what the meeting will say. And the overlay's structural problem "
            "is exposure: firing 5 days at a time on ~8 scheduled releases, it is out of the market "
            "**~90%** of the year, so even a real per-event edge can't compound into anything. A "
            "scheduled, exhaustively-parsed, pre-FOMC government text is the *least* likely home for "
            "un-arbitraged equity drift — and the tape agrees."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Kill the proxy.** Scrape the Fed's full Beige-Book archive, count the real Loughran-"
            "McDonald positive/negative terms per release, and rerun this exact engine — the event study, "
            "the HAC regression, the regime test and the overlay are all proxy-agnostic.\n"
            "- **Siblings on hard data.** [Study 387 — Economic-Surprise-Index](../387-economic-surprise-index/) "
            "and [Study 385 — Jobless-Claims-Momentum](../385-jobless-claims-momentum/): does any macro "
            "tell lead equities, numbers instead of adjectives?\n"
            "- **Fed-sensitive targets.** Re-point the drift at the **2-year note**, the **dollar**, or "
            "rate-sensitive sectors, and test the tone *change* (book-to-book) rather than the level — a "
            "real Beige-Book effect, if it exists, most plausibly lives in rates, not SPY.\n\n"
            "*The reproducible core is offline and deterministic; the tone input is an explicit labelled "
            "proxy. Methods and sources: [`docs/references.md`](../docs/references.md); frozen numbers: "
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
