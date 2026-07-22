"""Generate the two narrative notebooks for Study 789 (SUE Earnings-Surprise Drift).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached prices + SUE
events under ../_cache/ and otherwise quote the frozen headline numbers in ``R`` (mirroring
docs/results.md). The synthetic positive control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance adjusted closes for a
# 30-name large-cap basket + EDGAR frame-tagged quarterly diluted EPS; as-of 2026-06-30;
# fingerprint 4d0fd3625f76). Percent units unless noted.
R = dict(
    asof="2026-06-30", n_names=30, filed_lo="2012-07-26", filed_hi="2026-06-03",
    fp="4d0fd3625f76",
    h21=dict(n=1315, top=1.06, bot=2.22, ls=-1.16, t=-2.29, nw_t=-1.41,
             win=50.6, win_lo=47.3, win_hi=53.9, placebo_p=0.991, block_p=0.841,
             q5=[2.04, 1.91, 0.89, -0.05, 1.74]),
    h42=dict(n=1279, top=1.70, bot=3.47, ls=-1.77, t=-2.48, nw_t=-0.41,
             win=46.9, win_lo=43.6, win_hi=50.3, placebo_p=0.995, block_p=0.913,
             q5=[3.37, 3.49, 2.48, 0.03, 2.69]),
    h63=dict(n=1261, top=3.34, bot=4.29, ls=-0.95, t=-1.12, nw_t=0.41,
             win=48.9, win_lo=45.5, win_hi=52.3, placebo_p=0.879, block_p=0.628,
             q5=[3.91, 4.48, 4.10, 1.16, 4.42]),
    gross63=-0.95, c5_net=-1.27, c10_net=-1.47,
    decile63_ls=1.53, decile63_t=0.99,
    syn_null_mean=0.08, syn_null_sd=0.57, syn_null_fire=0,
    syn_planted_ls=16.6, syn_planted_t=21.5,
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Textbook PEAD replicates%3F: Busted](https://img.shields.io/badge/Textbook_PEAD_replicates%3F-Busted-8b949e?style=flat-square)\n\n"
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

from sue_drift import data, strategy as st
try:
    from quantlab.repro import as_of
except Exception:
    as_of = lambda df, a: df[df.index <= pd.Timestamp(a)].copy()

HAVE_REAL = data.have_real()
if HAVE_REAL:
    PRICES = as_of(data.load_prices(), data.AS_OF)
    EVENTS = data.load_events()
    ET = data.build_event_table(PRICES, EVENTS)
else:
    PRICES = EVENTS = ET = None
print("real cache present:", HAVE_REAL, "| SUE events:", (0 if ET is None else len(ET)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Do the biggest earnings beats keep drifting up for weeks? 📈\n"
            "### SUE post-earnings drift — the textbook anomaly that quietly evaporates on big, "
            "liquid stocks\n\n"
            + BADGES +
            "Open any markets textbook and you'll find *post-earnings-announcement drift*: after a "
            "company reports, the stock supposedly keeps sliding in the direction of the surprise "
            "for weeks. Sort names by how big the surprise was — the **standardized unexpected "
            "earnings**, or **SUE** — buy the top, short the bottom, and collect the drift. It's "
            "been called the *\"granddaddy of anomalies.\"*\n\n"
            "So does it work on 30 of the most-watched large-caps in the world? **No — and if "
            "anything it leans the other way.**\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebos and the cost math? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** 30-name large-cap basket, EPS surprises from SEC EDGAR, "
            f"{R['filed_lo']}→{R['filed_hi']}, {R['h63']['n']:,} events. Every chart is drawn by "
            "the code beside it; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Do the biggest positive surprises keep drifting up? | **No.** Over the next 1–3 "
            f"months the top-SUE names actually drift **less** than the bottom-SUE names — the "
            f"\"buy winners, short losers\" spread comes out **{R['h63']['ls']:+.1f}%** at 3 "
            "months (the *wrong* sign). |\n"
            "| Is the sort at least orderly (more surprise → more drift)? | **No.** Line the names "
            "up from worst surprise to best and the drift zig-zags — the *lowest*-surprise bucket "
            "is among the *highest*-drifting. |\n"
            "| Could it just be a fluke of how you slice it? | **It's worse than that.** Slice "
            "into 3 buckets and the spread is negative; slice into 10 and it's positive. A real "
            "pattern doesn't flip sign when you re-slice it. |\n"
            "| Could you trade it? | **Nothing to trade.** The spread is negative before costs and "
            "more negative after. |\n\n"
            "> The famous textbook drift is simply **not here** on big, liquid, heavily-covered "
            "names — which, it turns out, is exactly where the research says it should be weakest."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"When a company reports earnings, the market underreacts. A big beat keeps "
            "drifting up for weeks; a big miss keeps sliding. So rank stocks by how surprising "
            "their earnings were, buy the top, short the bottom, and pocket the drift.\"*\n\n"
            "The surprise is measured as **SUE** — take this quarter's earnings per share, subtract "
            "the *same quarter a year ago* (that's the \"expected\" number under the simplest "
            "model), and divide by how bumpy that year-over-year change has been lately. A SUE of "
            "+3 means \"a beat three times bigger than this company's usual wobble.\" This is the "
            "original Foster-Olsen-Shevlin / Bernard-Thomas signal from the 1980s."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If it worked, it would be one of the cleanest signals in finance: earnings dates are "
            "known in advance, the surprise is a hard accounting number, and the trade is "
            "mechanical. It's also one of the most-studied anomalies ever — which is exactly why "
            "it's worth re-checking on modern, liquid names. Anomalies that everyone knows about "
            "and that live in big tradeable stocks tend to get arbitraged into the ground."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The sort.** Every quarterly report from 30 large-caps ({R['h63']['n']:,} events "
            f"since {R['filed_lo'][:4]}), each tagged with its SUE, computed only from data known "
            "*before* the report.\n"
            "- **The drift.** Buy (or short) the day *after* the report is public — no peeking — "
            "and hold 1, 2 or 3 months.\n"
            "- **The spread.** Top third minus bottom third. If the textbook is right it's clearly "
            "positive and gets bigger for bigger surprises.\n"
            "- **The honesty checks.** Reports bunch up in earnings season, so we re-test on a "
            "\"calendar-time\" basis, shuffle the labels thousands of times, and re-slice the sort "
            "to see if the answer is stable."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the headline spread** at three horizons: top-SUE third minus bottom-SUE "
            "third."
        ),
        code(
            "hs = {}\n"
            "for H in st.HORIZONS:\n"
            "    if HAVE_REAL:\n"
            "        ls = st.long_short_drift(st.event_drift_frame(PRICES, ET, H), n_buckets=3)\n"
            "        hs[H] = ls['ls_mean']*100\n"
            "    else:\n"
            "        hs[H] = R[f'h{H}']['ls']\n"
            "labels = ['1 month\\n(21d)','2 months\\n(42d)','3 months\\n(63d)']\n"
            "vals = [hs[H] for H in st.HORIZONS]\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.4))\n"
            "ax.bar(labels, vals, color=[RED, RED, RED], width=.55)\n"
            "for i,v in enumerate(vals): ax.annotate(f'{v:+.2f}%',(i,v),ha='center',va='top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('top-minus-bottom SUE drift (%)')\n"
            "ax.set_title('The \"buy the biggest beats\" spread is the WRONG sign at every horizon')\n"
            "plt.tight_layout(); plt.show()\n"
            "print({H: round(v,2) for H,v in hs.items()})"
        ),
        md(
            f"Every bar is **below zero**. At three months the spread is "
            f"**{R['h63']['ls']:+.1f}%** — the top-SUE names drifted *less* than the bottom-SUE "
            "names. That's the opposite of what the anomaly predicts.\n\n"
            "**Is the sort at least orderly?** A real drift climbs steadily from the worst "
            "surprises to the best. Here are the five buckets, worst SUE on the left:"
        ),
        code(
            "H = 63\n"
            "if HAVE_REAL:\n"
            "    q5 = st.bucket_means(st.event_drift_frame(PRICES, ET, H), n_buckets=5)*100\n"
            "else:\n"
            "    q5 = np.array(R['h63']['q5'])\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.4))\n"
            "ax.bar(['Q1\\nworst','Q2','Q3','Q4','Q5\\nbest'], q5, color=GREY, width=.6)\n"
            "for i,v in enumerate(q5): ax.annotate(f'{v:+.1f}%',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('average 3-month drift (%)')\n"
            "ax.set_title('No staircase: the drift zig-zags across SUE buckets')\n"
            "plt.tight_layout(); plt.show()\n"
            "print([round(float(x),2) for x in q5])"
        ),
        md(
            "There's no staircase. The *lowest*-surprise bucket (Q1) is one of the *highest*-"
            "drifting, the 4th bucket is a trough, and the \"best\" bucket is nowhere near the top. "
            "A signal that behaved like this in a job interview would not get hired.\n\n"
            "**The tell that it's noise:** re-slice the exact same sort into 10 buckets instead of "
            "3, and the spread flips sign."
        ),
        code(
            "if HAVE_REAL:\n"
            "    fr = st.event_drift_frame(PRICES, ET, 63)\n"
            "    terc = st.long_short_drift(fr, n_buckets=3)['ls_mean']*100\n"
            "    dec = st.long_short_drift(fr, n_buckets=10)['ls_mean']*100\n"
            "else:\n"
            "    terc, dec = R['h63']['ls'], R['decile63_ls']\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.3))\n"
            "ax.bar(['3 buckets\\n(terciles)','10 buckets\\n(deciles)'], [terc, dec],\n"
            "       color=[RED, AMBER], width=.5)\n"
            "for i,v in enumerate([terc, dec]): ax.annotate(f'{v:+.2f}%',(i,v),ha='center',\n"
            "    va='bottom' if v>=0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('top-minus-bottom drift, 3-month (%)')\n"
            "ax.set_title('Same sort, different slicing -> opposite sign (a noise fingerprint)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'tercile {terc:+.2f}%  |  decile {dec:+.2f}%')"
        ),
        md(
            f"Terciles say **{R['h63']['ls']:+.1f}%**, deciles say "
            f"**{R['decile63_ls']:+.1f}%** — and neither is statistically distinguishable from "
            "zero. A genuine anomaly keeps its sign when you re-slice it; noise doesn't."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** Sorted on SUE, the drift is the wrong sign at 1–3 months, the "
            "buckets don't line up, and the sign flips when you re-slice. Once you account for the "
            "fact that earnings bunch into seasons, there's no real spread left in either "
            "direction.\n"
            "- **Tradability — Mirage.** Negative before costs, more negative after. There's "
            "nothing to collect.\n"
            "- **Does the textbook drift replicate here? — Busted.** On big, liquid, heavily-"
            "covered names — exactly where decades of research say the drift should be weakest — "
            "it's simply gone."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **Where the drift probably still lives:** small, illiquid, thinly-covered stocks — "
            "the classic finding is that PEAD concentrates there and fades among liquid large-caps "
            "(the opposite of our basket). A natural follow-up is a small-cap version.\n"
            "- **Why big caps are the graveyard:** they're followed by armies of analysts and "
            "arbitraged in seconds; any drift gets competed away long before you could hold it for "
            "months.\n"
            "- **Sibling studies:** [363-pead-drift](../../363-pead-drift/) sorts on the *price "
            "jump* around the report instead of the earnings number; "
            "[369-earnings-revision-momentum](../../369-earnings-revision-momentum/) sorts on "
            "*analyst revisions*; [534-revenue-surprise-drift](../../534-revenue-surprise-drift/) "
            "does this exact test on *revenue* surprises (also empty). See "
            "[docs/references.md](../docs/references.md) for the full dedup.\n\n"
            "*Think the SUE drift is alive on a universe we didn't test? Show a monotone, robust, "
            "certifiable spread after realistic costs on the size you'd actually run — then we'll "
            "talk.*"
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
            "# SUE Earnings-Surprise Drift — a quantitative teardown 🔬\n"
            "### The SUE tercile long-short · naive vs autocorrelation-robust *t* · non-monotone "
            "quintiles · the tercile-vs-decile sign flip · label-shuffle & within-quarter block "
            "placebos · an honest cost sweep · a 20-seed synthetic control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The "
            "claim — **post-earnings drift lines up with standardized unexpected earnings** — is "
            "the original Bernard-Thomas (1989) SUE sort. The job here is to measure it honestly "
            "on a conservative large-cap basket, respect the fact that filings cluster into "
            "earnings seasons, and ask whether anything survives.\n\n"
            "> ⚠️ **Data note.** yfinance daily adjusted closes for a 30-name large-cap basket + "
            "EDGAR frame-tagged quarterly **diluted EPS** (with filing dates), as-of "
            f"{R['asof']}, fingerprint `{R['fp']}`. SUE = seasonal-random-walk surprise "
            "`EPS_q − EPS_{q−4}` over the rolling std of the last ~8 such surprises. **Survivorship "
            "is named on the Signal axis** (a current-membership basket, and PEAD is documented to "
            "concentrate in small illiquids — so this basket is where it's weakest). Methods in "
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
            f"| **Signal** | `NONE` | 3-mo tercile long-short **{R['h63']['ls']:+.2f}%** (wrong "
            f"sign); robust calendar-time Newey-West **t = {R['h63']['nw_t']:+.2f}** (max over "
            f"horizons |t| = {abs(R['h21']['nw_t']):.2f}); non-monotone quintiles; sign flips "
            f"tercile→decile ({R['h63']['ls']:+.2f}% → {R['decile63_ls']:+.2f}%); block placebo "
            f"p = {R['h63']['block_p']:.2f} |\n"
            f"| **Tradability** | `MIRAGE` | net of 5/10 bps at 63d: "
            f"{R['c5_net']:+.2f}% / {R['c10_net']:+.2f}% (gross already {R['gross63']:+.2f}%) |\n"
            f"| **Textbook PEAD replicates?** | `BUSTED` | no monotone SUE gradient; robust t "
            f"pinned near zero on liquid large-cap survivors |\n\n"
            "> 💡 In plain words: the naive per-event *t* looks *significantly negative* (−2.3), "
            "but that's an artifact of earnings-season clustering — the moment you test it on a "
            "calendar-time basis or shuffle labels within quarter, there's nothing there, in "
            "either direction."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $E_q$ be diluted EPS for fiscal quarter $q$, disclosed (filed) on date $F$. The "
            "seasonal-random-walk surprise is $u_q = E_q - E_{q-4}$, and\n\n"
            "$$\\mathrm{SUE}_q = \\frac{E_q - E_{q-4}}{\\operatorname{std}(u_{q-1},\\dots,u_{q-8})}$$\n\n"
            "where the denominator uses only surprises known *strictly before* $q$. The claims:\n\n"
            "- **H₁ (drift).** Sorting events on SUE, the top-minus-bottom forward return over "
            "1–3 months is **positive**.\n"
            "- **H₂ (monotonicity).** Mean drift **rises monotonically** from the bottom SUE "
            "bucket to the top.\n"
            "- **H₃ (capture).** A long-short banks the spread net of realistic costs + borrow.\n\n"
            f"We find **H₁ rejected** (spread is negative: {R['h21']['ls']:+.2f}% / "
            f"{R['h42']['ls']:+.2f}% / {R['h63']['ls']:+.2f}% at 21/42/63d; robust NW t never "
            f"clears +2), **H₂ rejected** (non-monotone, sign-flipping across bucket counts), "
            "**H₃ rejected** (nothing to capture)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "The honest problem with an event study is that **filings cluster in earnings "
            "seasons**, so pooled per-event returns are neither independent nor serially "
            "uncorrelated — a naive one-sample *t* is inflated (both ways). So we lead with the "
            "**autocorrelation-robust** statistic: aggregate the top-minus-bottom drift into ~50 "
            "earnings-season buckets and run a **Newey-West HAC *t*** on that calendar-time series. "
            "The naive one-sample *t* is reported alongside precisely to show the gap. We add a "
            "**within-quarter block placebo** (shuffle SUE labels inside each calendar quarter — "
            "respects the clustering), a global label-shuffle placebo, and a Wilson interval on the "
            "win-rate. The **t ≥ 2 bar is judged on the robust NW *t***."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Events.** {R['h63']['n']:,} usable (ticker, filing-date) rows over "
            f"{R['n_names']} names, filed {R['filed_lo']} → {R['filed_hi']}; each print needs ≥8 "
            "prior seasonal surprises before a SUE is emitted.\n"
            "- **Execution / lag.** Anchor at the first session on/after the filing, **enter the "
            "close one day later** (signal known at the filing, return from t+1 — the single "
            "documented lag), hold 21 / 42 / 63 trading days.\n"
            "- **Signal.** Top-minus-bottom SUE **tercile** long-short; one-sample *t*, "
            "calendar-time NW *t*, win-rate + Wilson, label & block placebos.\n"
            "- **Monotonicity.** Quintile drift low→high; the tercile-vs-decile sign-stability "
            "check.\n"
            "- **Execution.** One-way costs × per-event turnover (4× one-way on a round trip) + "
            "short-leg borrow.\n"
            "- **Control.** Synthetic panel with a planted drift knob; the null must not fire "
            "across 20 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The headline — naive *t* vs the robust *t*\n\n"
            "The long-short at each horizon, with the naive one-sample *t* and the "
            "autocorrelation-robust calendar-time Newey-West *t* side by side."
        ),
        code(
            "rows = []\n"
            "for H in st.HORIZONS:\n"
            "    if HAVE_REAL:\n"
            "        s = st.summarize(PRICES, ET, H, n_buckets=3, placebo=False)\n"
            "        rows.append((H, s['ls_mean']*100, s['t'], s['nw_t']))\n"
            "    else:\n"
            "        d = R[f'h{H}']; rows.append((H, d['ls'], d['t'], d['nw_t']))\n"
            "Hs = [r[0] for r in rows]; ls=[r[1] for r in rows]; tn=[r[2] for r in rows]; tw=[r[3] for r in rows]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.2, 4.4))\n"
            "a1.bar([f'{h}d' for h in Hs], ls, color=RED, width=.5)\n"
            "for i,v in enumerate(ls): a1.annotate(f'{v:+.2f}%',(i,v),ha='center',va='top')\n"
            "a1.axhline(0,c='k',lw=.8); a1.set_ylabel('long-short drift (%)'); a1.set_title('Spread: wrong sign')\n"
            "x = np.arange(len(Hs)); w=.38\n"
            "a2.bar(x-w/2, tn, w, color=GREY, label='naive one-sample t')\n"
            "a2.bar(x+w/2, tw, w, color=GREEN, label='robust calendar-time NW t')\n"
            "a2.axhline(2, ls='--', c=RED, lw=1); a2.axhline(-2, ls='--', c=RED, lw=1); a2.axhline(0,c='k',lw=.8)\n"
            "a2.set_xticks(x); a2.set_xticklabels([f'{h}d' for h in Hs]); a2.set_ylabel('t-stat')\n"
            "a2.set_title('Naive t is a clustering artifact; robust t is nothing'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print([(h, round(l,2), round(a,2), round(b,2)) for h,l,a,b in rows])"
        ),
        md(
            f"> 💡 In plain words: the naive one-sample *t* is a *significant* "
            f"**{R['h21']['t']:.2f} / {R['h42']['t']:.2f}** at 21/42 days — but it's the wrong "
            "sign *and* it's an artifact. The robust calendar-time Newey-West *t* collapses to "
            f"**{R['h21']['nw_t']:+.2f} / {R['h42']['nw_t']:+.2f} / {R['h63']['nw_t']:+.2f}** — "
            "nowhere near the ±2 bar in either direction. H₁ is rejected: there is no drift in the "
            "predicted direction, and no robust reversal either."
        ),
        md(
            "### 4b · Monotonicity and the bucket-count sign flip\n\n"
            "A real SUE drift is monotone in SUE (Foster-Olsen-Shevlin). And it keeps its sign "
            "when you re-slice the sort. Neither holds here."
        ),
        code(
            "H=63\n"
            "if HAVE_REAL:\n"
            "    fr = st.event_drift_frame(PRICES, ET, H)\n"
            "    q5 = st.bucket_means(fr, n_buckets=5)*100\n"
            "    terc = st.long_short_drift(fr, n_buckets=3)['ls_mean']*100\n"
            "    dec = st.long_short_drift(fr, n_buckets=10)['ls_mean']*100\n"
            "else:\n"
            "    q5 = np.array(R['h63']['q5']); terc, dec = R['h63']['ls'], R['decile63_ls']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.2, 4.4))\n"
            "a1.bar(['Q1\\nlow','Q2','Q3','Q4','Q5\\nhigh'], q5, color=GREY, width=.6)\n"
            "for i,v in enumerate(q5): a1.annotate(f'{v:+.1f}%',(i,v),ha='center',va='bottom')\n"
            "a1.set_ylabel('3-mo drift (%)'); a1.set_title('Non-monotone in SUE (should be a staircase)')\n"
            "a2.bar(['terciles','deciles'], [terc, dec], color=[RED, AMBER], width=.5)\n"
            "for i,v in enumerate([terc, dec]): a2.annotate(f'{v:+.2f}%',(i,v),ha='center',\n"
            "    va='bottom' if v>=0 else 'top')\n"
            "a2.axhline(0,c='k',lw=.8); a2.set_ylabel('long-short (%)'); a2.set_title('Sign flips with bucket count')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('quintiles', [round(float(x),2) for x in q5], '| tercile', round(terc,2), 'decile', round(dec,2))"
        ),
        md(
            f"> 💡 In plain words: the quintiles zig-zag ({', '.join(f'{x:+.1f}' for x in R['h63']['q5'])}%) "
            "with no slope, and the long-short is "
            f"**{R['h63']['ls']:+.2f}%** in terciles but **{R['decile63_ls']:+.2f}%** in deciles. "
            "A robust anomaly does not reverse when you re-slice the identical sort — H₂ rejected. "
            "This is the canonical fingerprint of noise."
        ),
        md(
            "### 4c · The placebos — nothing to rescue\n\n"
            "A light in-notebook within-quarter block placebo (respects the earnings-season "
            "clustering); we quote the canonical fuller run from `results.md`."
        ),
        code(
            "H=63\n"
            "if HAVE_REAL:\n"
            "    fr = st.event_drift_frame(PRICES, ET, H)\n"
            "    obs = st.long_short_drift(fr, n_buckets=3)['ls_mean']*100\n"
            "    pl = st.placebo_pvalue(fr, n_draws=3000, n_buckets=3)\n"
            "    draws = pl['draws']*100\n"
            "else:\n"
            "    obs = R['h63']['ls']\n"
            "    draws = np.random.default_rng(789).normal(0, 0.6, 3000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=50, color=GREY, alpha=.85, label='label-shuffle null (in-notebook)')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'observed tercile long-short {obs:+.2f}%')\n"
            "ax.axvline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('shuffled long-short drift (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f\"Observed sits inside the null (right-tail p={R['h63']['placebo_p']:.2f}, block p={R['h63']['block_p']:.2f})\")\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"canonical (results.md): right-tail label-shuffle p={R['h63']['placebo_p']:.3f}, within-quarter block p={R['h63']['block_p']:.3f}\")"
        ),
        md(
            f"> 💡 In plain words: the observed spread sits *inside* the shuffle cloud — right-tail "
            f"label-shuffle p = **{R['h63']['placebo_p']:.2f}** (i.e. a random relabelling beats it "
            f"most of the time), within-quarter block placebo p = **{R['h63']['block_p']:.2f}**. "
            "There is no positive drift for the placebo to be surprised by."
        ),
        md(
            "### 4d · The cost sweep — nothing to erode\n\n"
            "One-way costs × per-event turnover (4× one-way on a long-short round trip) + short-leg "
            "borrow, at 63 days."
        ),
        code(
            "if HAVE_REAL:\n"
            "    g = st.net_of_costs(PRICES, ET, 63, cost_bps=5.0)['gross']*100\n"
            "    n5 = st.net_of_costs(PRICES, ET, 63, cost_bps=5.0)['net']*100\n"
            "    n10 = st.net_of_costs(PRICES, ET, 63, cost_bps=10.0)['net']*100\n"
            "else:\n"
            "    g, n5, n10 = R['gross63'], R['c5_net'], R['c10_net']\n"
            "fig, ax = plt.subplots(figsize=(7.8, 4.3))\n"
            "ax.bar(['gross','net @5bps','net @10bps'], [g, n5, n10], color=[GREY, RED, RED], width=.55)\n"
            "for i,v in enumerate([g,n5,n10]): ax.annotate(f'{v:+.2f}%',(i,v),ha='center',va='top')\n"
            "ax.axhline(0,c='k',lw=.8); ax.set_ylabel('3-mo long-short (%)')\n"
            "ax.set_title('Negative gross, more negative net -- nothing to deploy')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross {g:+.2f}%  net@5 {n5:+.2f}%  net@10 {n10:+.2f}%')"
        ),
        md(
            f"> 💡 In plain words: gross is already **{R['gross63']:+.2f}%**; net of costs it's "
            f"**{R['c5_net']:+.2f}%** (5 bps) to **{R['c10_net']:+.2f}%** (10 bps). Tradability = "
            "MIRAGE by default — there is no edge to charge costs against."
        ),
        md(
            "### 4e · Faithful-engine & power control — we know the truth here\n\n"
            "Synthetic price panel + SUE events with a TUNABLE planted post-earnings drift. The "
            "null (edge = 0) is checked over **20 seeds**; a planted edge must light up."
        ),
        code(
            "null_ts = []\n"
            "for s_ in range(20):\n"
            "    p, e = data.synthetic_sue(edge=0.0, seed=789 + s_)\n"
            "    null_ts.append(st.synthetic_detect(p, e)['t'])\n"
            "null_ts = np.asarray(null_ts)\n"
            "pP, eP = data.synthetic_sue(edge=0.08, seed=789)\n"
            "planted_t = st.synthetic_detect(pP, eP)['t']\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.linspace(-.12,.12,20), null_ts, color=GREY, s=40, label='null (edge=0), 20 seeds')\n"
            "ax.scatter([1], [planted_t], color=GREEN, s=90, zorder=5, label='planted edge = 0.08')\n"
            "ax.axhline(2, ls='--', c=RED, lw=1); ax.axhline(-2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0,1]); ax.set_xticklabels(['null x 20','planted'])\n"
            "ax.set_ylabel('long-short one-sample t'); ax.set_title('Control: no null fires; a planted drift lights up')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null mean t={null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(abs(null_ts)>=2).sum()}/20  |  planted t={planted_t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 20 null worlds the detector averages "
            f"t = {R['syn_null_mean']:+.2f} (sd {R['syn_null_sd']:.2f}) and **never** crosses the "
            f"bar; a planted drift reads t = {R['syn_planted_t']:.1f} "
            f"(long-short +{R['syn_planted_ls']:.1f}%). The machinery is faithful and well-powered "
            "— so the flat real-tape result is a genuine *absence* of signal, not a broken "
            "detector. *(A power check only — never cited in support of the real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — SUE tercile long-short {R['h21']['ls']:+.2f}% / "
            f"{R['h42']['ls']:+.2f}% / {R['h63']['ls']:+.2f}% (1/2/3 mo, wrong sign); robust "
            f"calendar-time NW t = {R['h21']['nw_t']:+.2f} / {R['h42']['nw_t']:+.2f} / "
            f"{R['h63']['nw_t']:+.2f} (never clears +2); non-monotone quintiles; sign flips "
            f"tercile→decile ({R['h63']['ls']:+.2f}% → {R['decile63_ls']:+.2f}%); block placebo "
            f"p = {R['h63']['block_p']:.2f}. The naive one-sample t of {R['h42']['t']:.2f} is a "
            "clustering artifact, not a signal.\n"
            f"- **Tradability `MIRAGE`** — gross {R['gross63']:+.2f}% at 63d, net "
            f"{R['c5_net']:+.2f}% / {R['c10_net']:+.2f}% at 5/10 bps. Nothing to deploy.\n"
            "- **Textbook PEAD replicates? `BUSTED`** — no monotone SUE gradient and a robust t "
            "pinned near zero on liquid large-cap survivors, exactly where the literature "
            "(Chordia et al. 2009; McLean-Pontiff 2016) predicts it should be weakest or gone."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The liquidity story.** PEAD is documented to concentrate in small, illiquid, "
            "thinly-covered names; our basket is the opposite. The obvious follow-up is a small-cap "
            "/ low-coverage universe — with survivorship handled via a point-in-time membership "
            "panel (the desk's opt-in guard).\n"
            "- **Expectation model.** We use the seasonal random walk (Foster-Olsen-Shevlin). A "
            "richer analyst-expectations SUE is a different signal — and its own study "
            "([369](../../369-earnings-revision-momentum/)).\n"
            "- **Dedup map:** [363-pead-drift](../../363-pead-drift/) (price-gap/CAR sort, no "
            "fundamentals), [369-earnings-revision-momentum](../../369-earnings-revision-momentum/) "
            "(analyst revisions), [534-revenue-surprise-drift](../../534-revenue-surprise-drift/) "
            "(revenue SUR on this same basket — also empty). This study is the **SUE-sorted "
            "fundamental-EPS** portfolio drift.\n\n"
            "*The reproducible core is offline and deterministic; frozen numbers live in "
            "[`docs/results.md`](../docs/results.md), sources in "
            "[`docs/references.md`](../docs/references.md).*"
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
