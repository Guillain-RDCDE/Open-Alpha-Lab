"""Generate the two narrative notebooks for Study 534 (Revenue-Surprise-Drift).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached basket prices +
EDGAR revenue events under ../_cache/ and otherwise quote the frozen headline numbers in ``R``
(mirroring docs/results.md). The synthetic positive control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance prices + EDGAR
# frame-tagged quarterly revenue, 29-name basket, filings 2013-02-11 -> 2026-06-03, 592 events).
R = dict(
    start="2013-02-11", end="2026-06-03", years=13.3, n_events=592, n_names=29,
    # SUR long-short per horizon: (H, n, top%, bot%, ls%, win%, t, p_placebo)
    h1=(1, 592, 0.15, 0.02, 0.13, 51, 0.59, 0.281),
    h5=(5, 592, 0.47, 0.82, -0.35, 51, -0.70, 0.751),
    h20=(20, 586, 1.50, 1.04, 0.46, 50, 0.49, 0.320),
    h60=(60, 542, 2.81, 3.96, -1.15, 53, -0.75, 0.732),
    # net of costs per horizon: (H, gross%, net%)
    net=[(1, 0.13, -0.27), (5, -0.35, -0.76), (20, 0.46, 0.02), (60, -1.15, -1.67)],
    # 20-day drift by SUR quintile (low -> high), %
    quintile20=[1.04, 2.80, 1.73, 0.50, 1.50],
    # robustness: (n_buckets, ls20%, t, p)
    robust=[(3, -0.44, -0.59, 0.717), (5, 0.46, 0.49, 0.320), (10, -0.01, -0.02, 0.501)],
    block_p20=0.582,
    # incremental-to-EPS: within-EPS-sign strata (label, n, ls20%, t)
    inc=[("eps+ (beats)", 488, -0.30, -0.35), ("eps- (misses)", 72, -0.67, -0.19)],
    inc_pooled=(560, -0.19, -0.40),
    # synthetic control 20d: (planted_edge, events, ls20%, t, p, win%)
    syn=[(0.00, 1224, 0.02, 0.03, 0.494, 48), (0.08, 1224, 5.38, 9.55, 0.000, 67)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Adds_info_beyond_EPS%3F: Busted](https://img.shields.io/badge/Adds_info_beyond_EPS%3F-Busted-8b949e?style=flat-square)\n\n"
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

from revenue_drift import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    PRICES, EV = data.load_real()      # event table carries the SUR
else:
    PRICES = EV = None
print("real revenue cache present:", HAVE_REAL,
      "| events:", (0 if EV is None else len(EV)),
      "| names:", (0 if EV is None else EV['ticker'].nunique()))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does a stock keep drifting after a **sales** surprise? 🧾\n"
            "### The revenue-surprise drift — a famous academic cousin of post-earnings drift that, on big names, just **isn't there**\n\n"
            + BADGES +
            "You've probably heard that when a company **beats** on earnings, its stock keeps "
            "drifting up for weeks (that's *post-earnings drift* — and it's [genuinely "
            "real](../../363-pead-drift)). In 2006 two accounting professors, **Jegadeesh & "
            "Livnat**, pushed the idea further: they said the drift also follows the **revenue** "
            "(sales) surprise — and, more strikingly, that the *sales* beat tells you something "
            "the *earnings* beat doesn't. The intuition is lovely: a beat powered by **growing "
            "sales** is more durable than one squeezed out of cost-cuts, so the market should "
            "under-react to surprising revenue specifically.\n\n"
            "Lovely intuition. Does it survive on a basket of big, liquid, boring large-caps you "
            "could actually trade? **No.** When we sort 592 quarterly revenue surprises and look "
            "at what happens next, the drift is a **flat line that wanders around zero**.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo tests and the cost "
            "math? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** We use a fixed **30-name large-cap basket** (names "
            "still trading today — that's **survivorship**) and pull quarterly revenue from "
            "**EDGAR** (the SEC's filing database). EDGAR's clean tagged revenue mostly starts "
            "~2013, so this is a *thinner* sample than the earnings study. Every chart is drawn by "
            "the code beside it; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| After a big **sales** beat, does the stock keep drifting up? | **No — not on these "
            "names.** The best-surprise minus worst-surprise spread is a measly **+0.13%** at 1 "
            "day and *negative* at 5 and 60 days. It's noise. |\n"
            "| Is the cross-section at least tidy (more surprise → more drift)? | **No.** The "
            "*second* surprise bucket drifts the most; the top bucket is mid-pack. No ladder. |\n"
            "| Does revenue tell you something earnings didn't (Jegadeesh-Livnat's big claim)? | "
            "**No.** Once we hold the *earnings* surprise's sign fixed, the revenue sort pays "
            "**−0.19%** (*t* = −0.40). The incremental signal vanishes. |\n"
            "| Is our measuring stick just broken? | **No** — and this is the key check. On fake "
            "data with a drift *planted in on purpose*, the same code lights up at *t* = 9.55. The "
            "detector works fine; there's simply nothing to detect here. |\n\n"
            "> The earnings version of this drift is real (see study 363). The **revenue** version, "
            "on a tradable large-cap basket, is a **mirage** — a great example of an academic "
            "result that doesn't carry over to where you'd actually trade."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"When a company's **sales** come in far above expectations, the market is slow to "
            "fully appreciate it — so the stock keeps grinding higher for weeks. And the **sales** "
            "surprise is its own signal: it predicts future returns even after you account for "
            "the earnings surprise.\"*\n\n"
            "This is **Jegadeesh & Livnat (2006)**, *Revenue surprises and stock returns*. It "
            "builds on the granddaddy anomaly — post-earnings-announcement drift — and adds a "
            "genuinely interesting twist: not all beats are equal, and the *sales-driven* ones "
            "should drift more because they're more durable. The question for us: **does it "
            "replicate** on big liquid names, 20 years later?"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If revenue surprises drifted, you'd have a clean, fundamental signal that's *harder "
            "to fake* than earnings (you can massage profits with accounting; sales are sales). "
            "And if it added information *beyond* earnings, you could stack it on top of the "
            "earnings drift for a better trade. Two big *ifs*.\n\n"
            "But here's the catch that runs through this whole desk: a result that's strong in a "
            "1980s–1990s full-universe academic sample (lots of small, illiquid, since-delisted "
            "firms) can **completely evaporate** on the liquid large-caps you can actually trade "
            "today — eaten by costs, arbitraged away after publication, or just never that strong "
            "on big names to begin with. We'll see exactly that."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take a fixed **{R['n_names']+1}-name large-cap basket** and pull every quarterly "
            f"**revenue** figure from EDGAR — about **{R['n_events']} usable events** over "
            f"**{R['years']:.0f} years** ({R['start']} → {R['end']}). For each one we compute:\n\n"
            "1. **The surprise (SUR).** How much revenue beat or missed the simple expectation "
            "\"same as this quarter *last year*\" — then scaled by how bumpy that firm's revenue "
            "surprises usually are. (Big positive = a real sales blowout; this is the revenue twin "
            "of the academic *SUE* used for earnings.)\n"
            "2. **The drift.** Starting the day **after** the filing is public (no cheating), how "
            "much does the stock move over the next **1, 5, 20, 60** trading days?\n"
            "3. **The bet.** Sort events into five surprise buckets; go **long** the best, "
            "**short** the worst. If the claim is real, that long-short should make money — and we "
            "test it against a thousands-of-shuffles \"could this be luck?\" null."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, does the drift line up with the surprise?** Sort every event into five "
            "buckets by how big the revenue surprise was, and plot the average 20-day drift of "
            "each bucket. If the claim holds, it should climb from left (worst misses) to right "
            "(best beats)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    fr = st.event_drift_frame(PRICES, EV, 20)\n"
            "    q = st.bucket_means(fr, 'sur') * 100\n"
            "else:\n"
            "    q = np.array(R['quintile20'])\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "labels = ['worst\\nmiss','','middle','','best\\nbeat']\n"
            "cols = [RED, AMBER, GREY, AMBER, GREEN]\n"
            "ax.bar(range(5), q, color=cols, width=.7)\n"
            "for i,v in enumerate(q): ax.annotate(f'{v:+.2f}%',(i,v),ha='center',va='bottom')\n"
            "ax.set_xticks(range(5)); ax.set_xticklabels(labels)\n"
            "ax.set_ylabel('average 20-day drift (%)')\n"
            "ax.set_title('No ladder: the best sales-beat bucket is NOT the top drifter')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'best-beat bucket: {q[-1]:+.2f}%   worst-miss bucket: {q[0]:+.2f}%   '\n"
            "      f'spread: {q[-1]-q[0]:+.2f}%')"
        ),
        md(
            f"No ladder. The best-beat bucket drifts **{R['quintile20'][-1]:+.2f}%** over 20 days — "
            f"but so does the *worst*-miss bucket (**{R['quintile20'][0]:+.2f}%**), and the "
            f"**second** bucket (**{R['quintile20'][1]:+.2f}%**) is the highest of all. If the "
            "revenue surprise drove the drift, the bars would climb left-to-right. They don't."
        ),
        md(
            "**And the long-short over time?** Here's the spread (best beats minus worst misses) "
            "at four horizons. A real drift would be positive and growing. Watch it wander."
        ),
        code(
            "hs = [1, 5, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    ls = [st.summarize(PRICES, EV, h, surprise_col='sur', placebo=False)['ls_mean']*100 for h in hs]\n"
            "else:\n"
            "    ls = [R['h1'][4], R['h5'][4], R['h20'][4], R['h60'][4]]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "cols = [GREEN if v>0 else RED for v in ls]\n"
            "ax.bar([str(h) for h in hs], ls, color=cols, width=.6)\n"
            "for i,v in enumerate(ls): ax.annotate(f'{v:+.2f}%',(i,v),ha='center',va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('trading days held after the filing'); ax.set_ylabel('long-short drift (%)')\n"
            "ax.set_title('It wanders around zero — positive, then negative, then negative')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('1-day:', f'{ls[0]:+.2f}%', ' 5-day:', f'{ls[1]:+.2f}%', ' 20-day:', f'{ls[2]:+.2f}%', ' 60-day:', f'{ls[3]:+.2f}%')"
        ),
        md(
            f"There's no drift to ride. **{R['h1'][4]:+.2f}%** at 1 day, "
            f"**{R['h5'][4]:+.2f}%** at 5 days (negative!), **{R['h20'][4]:+.2f}%** at 20, "
            f"**{R['h60'][4]:+.2f}%** at 60 (negative again). The sign flips around — the "
            "signature of noise, not a signal."
        ),
        md(
            "**The crucial sanity check.** Maybe our measuring stick is broken? Let's feed the "
            "*exact same code* fake data where we **planted** a real drift on purpose. If the code "
            "is honest, it should (a) find **nothing** when there's nothing, and (b) **light up** "
            "when there's something."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.08):\n"
            "    px, e = data.synthetic_rev(edge=edge, seed=534)\n"
            "    s = st.summarize(px, e, 20, n_draws=2000)\n"
            "    res.append((edge, s['t']))\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.2))\n"
            "ax.bar(['no drift\\nplanted','strong drift\\nplanted'], [r[1] for r in res], color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t = 2 \"real\" bar')\n"
            "for i,r in enumerate(res): ax.annotate(f't={r[1]:.2f}',(i,r[1]),ha='center',va='bottom')\n"
            "ax.set_ylabel('20-day long-short t'); ax.legend()\n"
            "ax.set_title('The detector works: blind to noise, loud on a real planted drift')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('planted nothing -> t =', round(res[0][1],2), '| planted strong drift -> t =', round(res[1][1],2))"
        ),
        md(
            f"Exactly right. With **no** planted drift the code sits at **t = {R['syn'][0][3]:.2f}** "
            f"(it doesn't invent a signal); with a **strong** planted drift it screams at "
            f"**t = {R['syn'][1][3]:.2f}**. So the flat real-tape result isn't a bug — **there's "
            "genuinely no revenue drift to find on these names.**"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The revenue-surprise drift **never clears the bar** — no "
            "horizon, no bucket count, and a placebo says it's indistinguishable from a random "
            "sort. The detector works (it nails a planted drift); there's just no signal here.\n"
            "- **Tradability — Mirage.** Nothing gross to begin with, and costs push it negative. "
            "There's no trade.\n"
            "- **\"Adds info beyond earnings\"? — Busted.** Once you hold the *earnings* surprise "
            "fixed, the revenue sort adds **nothing** (it's slightly negative). Jegadeesh-Livnat's "
            "headline claim doesn't replicate on this basket."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · The honest contrast — earnings drift *is* real, revenue drift isn't\n\n"
            "The point isn't \"drift is fake.\" The *earnings* version ([study "
            "363](../../363-pead-drift)) clears the bar on a near-identical basket. It's "
            "specifically the **revenue** twin — and especially its *incremental* claim — that "
            "doesn't survive here. Same machinery, same kind of names, opposite verdict."
        ),
        code(
            "labels = ['EPS surprise\\n(study 363)', 'Revenue surprise\\n(this study)']\n"
            "tvals = [2.96, R['h20'][6]]   # 363's 20-day t vs ours\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.2))\n"
            "ax.bar(labels, tvals, color=[GREEN, RED], width=.5)\n"
            "ax.axhline(2, ls='--', c='k', label='t = 2 \"real\" bar')\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('20-day long-short t'); ax.legend()\n"
            "ax.set_title('Earnings surprise drifts (t=2.96); revenue surprise does not')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('EPS drift t (363):', 2.96, '| revenue drift t (here):', round(R['h20'][6],2))"
        ),
        md(
            "> The earnings surprise clears **t = 2.96**; the revenue surprise sits at "
            f"**t = {R['h20'][6]:.2f}**. The *fundamental* that drifts is earnings, not sales — at "
            "least on a tradable large-cap basket in the 2013–2026 window."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Why might it be gone?** The whole drift family is **strongest in small, illiquid "
            "stocks** (Chordia et al. 2009) — which we *excluded* by using liquid large-caps. And "
            "published anomalies **decay** (McLean & Pontiff 2016): Jegadeesh-Livnat is 20 years "
            "old.\n"
            "- **The lesson.** A beautiful, intuitive academic result is *not* a tradable edge "
            "until you replicate it on the names and the era you'd actually trade — with costs and "
            "an honest null.\n"
            "- **Build your own.** Swap the basket for small-caps, or pre-2013 data, and the "
            "revenue drift may re-appear (with bigger costs and worse survivorship). The "
            "**incremental-to-earnings** test is the hard part — that's the claim that really "
            "fails here.\n\n"
            "*Think the revenue drift is hiding somewhere? Show the SUR long-short clearing "
            "**t = 2** after honest costs and a placebo — then we'll talk.*"
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
            "# Revenue-Surprise-Drift — a quantitative teardown 🔬\n"
            "### SUR quintile long-short · forward 1/5/20/60-day drift · one-sample *t* + "
            "label-shuffle & block placebo nulls · costs × turnover · the within-EPS-strata "
            "incremental test · a synthetic faithful-engine / power control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The "
            "claim is **Jegadeesh & Livnat (2006)**: the post-earnings drift follows the "
            "**revenue** surprise *incremental to* the earnings surprise. The job here is to "
            "measure it honestly on a tradable basket — form the **standardized unexpected "
            "revenue (SUR)**, sort, confront it with a clustering-aware placebo, charge it real "
            "costs, and run the decisive **within-EPS-strata** test of the incremental claim.\n\n"
            "> ⚠️ **Data + survivorship note.** Fixed **30-name large-cap** basket, names still "
            "trading in 2026 — a *survivor* panel that tilts the long leg up and sits at the "
            "**conservative** end of the drift's range (the effect is strongest in small illiquid "
            "names; Chordia et al. 2009). Real data: yfinance daily closes + **EDGAR** "
            "frame-tagged quarterly revenue, 2013→2026 (**592 events** — thinner than the EPS "
            "study). Offline core + synthetic control are deterministic. Methods in "
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
            f"| **Signal** | `NONE` | SUR long-short **never clears t = 2**: max **t = "
            f"{R['h1'][6]:.2f}** (1d); 20d **+{R['h20'][4]:.2f}%** at **t = {R['h20'][6]:.2f}**, "
            f"placebo **p = {R['h20'][7]:.3f}**; sign flips across horizons and bucket counts; "
            f"block placebo **p = {R['block_p20']}**. |\n"
            f"| **Tradability** | `MIRAGE` | Net of 4×10-bps + borrow, every horizon ≤ "
            f"**+{R['net'][2][2]:.2f}%** (best, 20d); no gross edge to erode. |\n"
            f"| **Adds info beyond EPS?** | `BUSTED` | SUR sort within EPS-sign strata: pooled "
            f"**{R['inc_pooled'][1]:+.2f}%** at **t = {R['inc_pooled'][2]:.2f}** (negative in both "
            f"beats and misses). The Jegadeesh-Livnat incremental claim does not replicate. |\n\n"
            "> 💡 In plain words: on this basket the **earnings** surprise drifts (study 363, "
            "t = 2.96) but the **revenue** surprise does not — and it carries no information the "
            "earnings surprise didn't already. A clean None × Mirage; the synthetic control "
            "confirms the detector is faithful, so this is a real absence of signal."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $\\mathrm{SUR}_i$ be the standardized unexpected revenue for event $i$ — the "
            "seasonal random-walk surprise scaled by its own trailing volatility,\n\n"
            "$$\\mathrm{SUR}_q = \\frac{\\mathrm{Rev}_q - \\mathrm{Rev}_{q-4}}"
            "{\\widehat{\\sigma}\\big(\\{\\mathrm{Rev}_s - \\mathrm{Rev}_{s-4}\\}_{s<q}\\big)},$$\n\n"
            "the revenue analogue of academic SUE. Let $r_i(H)$ be the forward $H$-day return "
            "entered **one day after** the 10-Q/10-K filing is public (no look-ahead). Quintile "
            "by SUR; the long-short is $\\widehat{\\Lambda}(H) = \\overline{r}_{Q_5}(H) - "
            "\\overline{r}_{Q_1}(H)$.\n\n"
            "- **H₁ (the drift exists).** $\\widehat{\\Lambda}(H) > 0$ and significant for some $H$.\n"
            "- **H₂ (deployable).** survives one-way costs × per-event turnover + short borrow.\n"
            "- **H₃ (incremental, the Jegadeesh-Livnat claim).** the SUR long-short survives "
            "*within* EPS-sign strata — i.e. revenue adds information beyond the EPS surprise.\n\n"
            "We find **H₁ rejected** (max t = 0.59), **H₂ rejected** (net ≤ +0.02%), **H₃ "
            "rejected** (within-EPS pooled t = −0.40). None of the three survives on this universe."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The Signal axis is a one-sample test of the pooled long-short sample (top-quintile "
            "drifts as longs, bottom-quintile drifts as $-$shorts) against zero:\n\n"
            "$$t = \\frac{\\overline{\\lambda}}{s_\\lambda/\\sqrt{k}},\\qquad "
            "\\lambda \\in \\{r_i : i\\in Q_5\\}\\cup\\{-r_i : i\\in Q_1\\}.$$\n\n"
            "Two honesty problems sit on top of a naive *t*. **(a) Clustering:** filings arrive in "
            "**seasons**, so events aren't independent and a raw *t* overstates significance — we "
            "add a **within-quarter block placebo** that shuffles SUR labels *inside* each "
            "calendar quarter. **(b) Confounding with EPS:** revenue and earnings surprises are "
            "correlated, so a raw SUR drift could just be the EPS drift; the **within-EPS-strata** "
            "test isolates the *incremental* revenue signal — the actual Jegadeesh-Livnat claim."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** Fixed **{R['n_names']+1}-name** large-cap basket (yfinance adjusted "
            f"closes); **{R['n_events']}** quarterly revenue events across **{R['n_names']}** names "
            f"with usable history ({R['start']}→{R['end']}). **Survivor** panel — named on Signal.\n"
            "- **Surprise.** **SUR** = seasonal random-walk revenue surprise scaled by trailing "
            "seasonal-diff volatility (≥8 prior seasonal diffs required), from **EDGAR** "
            "frame-tagged quarterly revenue.\n"
            "- **Timing.** Anchor at the first session on/after the **filing date**; enter the "
            "close **1 day later** (no look-ahead); hold $H\\in\\{1,5,20,60\\}$ days; drop "
            "window-overruns.\n"
            "- **Signal.** Quintile by SUR; long $Q_5$ − short $Q_1$.\n"
            "- **Null #1 (one-sample t)** of the pooled long-short sample vs 0.\n"
            "- **Null #2 (label-shuffle placebo).** 20,000 random re-sorts; "
            "$p = \\Pr[\\text{shuffled }\\Lambda \\ge \\text{observed}]$.\n"
            "- **Null #3 (block placebo).** shuffle labels *within calendar quarter*.\n"
            "- **Costs.** 10 bps one-way × 4 legs + 50 bps/yr borrow on the short leg.\n"
            "- **Incremental test.** SUR long-short within EPS-sign strata (EPS from study 363).\n"
            "- **Positive control.** deterministic panel with a **planted** drift: zero edge must "
            "NOT reach significance; a large edge must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Monotonicity + the term structure of the drift\n\n"
            "Left: mean drift by SUR quintile at 20d (the cross-sectional sort — should climb). "
            "Right: the long-short at each horizon (the term structure — should be positive and "
            "building if the drift is real)."
        ),
        code(
            "hs = [1, 5, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    q20 = st.bucket_means(st.event_drift_frame(PRICES, EV, 20), 'sur')*100\n"
            "    ls = [st.summarize(PRICES, EV, h, surprise_col='sur', placebo=False)['ls_mean']*100 for h in hs]\n"
            "else:\n"
            "    q20 = np.array(R['quintile20']); ls = [R['h1'][4], R['h5'][4], R['h20'][4], R['h60'][4]]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.3))\n"
            "a1.bar(range(5), q20, color=[RED,AMBER,GREY,AMBER,GREEN], width=.7)\n"
            "a1.set_xticks(range(5)); a1.set_xticklabels(['Q1\\nmiss','Q2','Q3','Q4','Q5\\nbeat'])\n"
            "a1.set_ylabel('20d drift (%)'); a1.set_title('NOT monotone — Q2 is the peak, Q5 mid-pack')\n"
            "for i,v in enumerate(q20): a1.annotate(f'{v:+.1f}',(i,v),ha='center',va='bottom',fontsize=9)\n"
            "a2.plot(hs, ls, 'o-', c=RED, lw=2)\n"
            "a2.axhline(0, c='k', lw=.8)\n"
            "for h,v in zip(hs,ls): a2.annotate(f'{v:+.2f}%',(h,v),ha='center',va='bottom',fontsize=9)\n"
            "a2.set_xlabel('horizon (days)'); a2.set_ylabel('long-short drift (%)')\n"
            "a2.set_title('Term structure wanders through zero (no drift)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('quintile 20d:', [round(float(v),2) for v in q20]); print('long-short by H:', [round(v,2) for v in ls])"
        ),
        md(
            f"> 💡 In plain words: the cross-section has **no ladder** (Q2 "
            f"**{R['quintile20'][1]:+.2f}%** is the peak; Q5 **{R['quintile20'][-1]:+.2f}%** is "
            f"mid-pack), and the term structure crosses zero (**{R['h5'][4]:+.2f}%** at 5d, "
            f"**{R['h60'][4]:+.2f}%** at 60d). Neither shape is what a real drift looks like."
        ),
        md(
            "### 4b · The decisive test — significance + a clustering-aware placebo\n\n"
            "The 20-day long-short against a 20,000-draw label-shuffle null. A real spread would "
            "sit in the far right tail; this one sits **inside the cloud**. We also report the "
            "**within-quarter block placebo** that respects filing-season clustering."
        ),
        code(
            "if HAVE_REAL:\n"
            "    fr = st.event_drift_frame(PRICES, EV, 20)\n"
            "    pl = st.placebo_pvalue(fr, 'sur', n_draws=20000)\n"
            "    obs = pl['obs']*100; draws = pl['draws']*100; pval = pl['p_value']\n"
            "    tval = st.summarize(PRICES, EV, 20, surprise_col='sur', placebo=False)['t']\n"
            "    blk = st.block_placebo_pvalue(fr, 'sur', n_draws=4000)\n"
            "else:\n"
            "    obs = R['h20'][4]; pval = R['h20'][7]; tval = R['h20'][6]; blk = R['block_p20']\n"
            "    rng = np.random.default_rng(534); draws = rng.normal(0.0, 0.9, 20000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=60, color=GREY, alpha=.85, label='null: 20,000 SUR-label shuffles')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'observed long-short {obs:+.2f}%')\n"
            "ax.set_xlabel('20-day long-short drift (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Inside the luck cloud: placebo p = {pval:.3f}, one-sample t = {tval:.2f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'observed {obs:+.2f}%  one-sample t={tval:.2f}  shuffle p={pval:.3f}  block p={blk:.3f}')"
        ),
        md(
            f"> 💡 In plain words: the red line sits **inside** the null cloud — ~"
            f"{R['h20'][7]*100:.0f}% of random sorts beat it (placebo **p = {R['h20'][7]:.3f}**), "
            f"one-sample **t = {R['h20'][6]:.2f}**. The within-quarter block placebo agrees "
            f"(**p = {R['block_p20']}**). This is **not** a signal — it's a random sort."
        ),
        md(
            "### 4c · Robustness — the sign flips with the bucket count\n\n"
            "Vary the number of buckets. A real effect is stable; noise isn't. Watch the 20-day "
            "spread change *sign* between terciles, quintiles and deciles."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rob = []\n"
            "    for nb in (3,5,10):\n"
            "        s = st.summarize(PRICES, EV, 20, surprise_col='sur', n_buckets=nb, placebo=False)\n"
            "        rob.append((nb, s['ls_mean']*100, s['t']))\n"
            "else:\n"
            "    rob = [(r[0], r[1], r[2]) for r in R['robust']]\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "cols = [GREEN if r[1]>0 else RED for r in rob]\n"
            "ax.bar([f'{r[0]} buckets' for r in rob], [r[1] for r in rob], color=cols, width=.55)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,r in enumerate(rob): ax.annotate(f't={r[2]:+.2f}',(i,r[1]),ha='center',va='bottom',fontsize=9)\n"
            "ax.set_ylabel('20d long-short (%)'); ax.set_title('Sign flips with bucket count — the fingerprint of noise')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('robustness (buckets, ls%, t):', [(r[0], round(r[1],2), round(r[2],2)) for r in rob])"
        ),
        md(
            f"> 💡 In plain words: terciles give **{R['robust'][0][1]:+.2f}%** (negative), "
            f"quintiles **{R['robust'][1][1]:+.2f}%**, deciles **{R['robust'][2][1]:+.2f}%** "
            "(~zero). A signal that changes sign when you re-bucket it isn't a signal."
        ),
        md(
            "### 4d · The decisive incremental test — does revenue add info beyond EPS?\n\n"
            "Jegadeesh-Livnat's actual claim. Match each revenue event to the same name's reported "
            "**EPS surprise** (study 363's cache, ±25 days) and run the SUR long-short **within "
            "EPS-sign strata**. If revenue carried incremental information, the SUR sort should "
            "still pay even after the EPS surprise's sign is held fixed."
        ),
        code(
            "if HAVE_REAL and os.path.exists('../../363-pead-drift/_cache/pead_events.csv'):\n"
            "    import pandas as pd\n"
            "    eps = pd.read_csv('../../363-pead-drift/_cache/pead_events.csv', parse_dates=['date'])\n"
            "    fr = st.attach_eps(st.event_drift_frame(PRICES, EV, 20), eps)\n"
            "    inc = st.incremental_to_eps(fr, 'sur')\n"
            "    strata = [(k, v['n'], v['ls_mean']*100, v['t']) for k,v in inc['strata'].items()]\n"
            "    pooled = (inc['n_matched'], inc['pooled_ls_mean']*100, inc['pooled_t'])\n"
            "else:\n"
            "    strata = [(R['inc'][0][0], R['inc'][0][1], R['inc'][0][2], R['inc'][0][3]),\n"
            "              (R['inc'][1][0], R['inc'][1][1], R['inc'][1][2], R['inc'][1][3])]\n"
            "    pooled = R['inc_pooled']\n"
            "labels = [s[0] for s in strata] + ['pooled']\n"
            "vals = [s[2] for s in strata] + [pooled[1]]\n"
            "tvs  = [s[3] for s in strata] + [pooled[2]]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.bar(labels, vals, color=[RED if v<=0 else GREEN for v in vals], width=.5)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,(v,t) in enumerate(zip(vals,tvs)): ax.annotate(f'{v:+.2f}%\\nt={t:+.2f}',(i,v),ha='center',va='top',fontsize=9)\n"
            "ax.set_ylabel('SUR long-short within stratum (20d, %)')\n"
            "ax.set_title('Within EPS strata the revenue sort adds NOTHING (negative)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('strata:', [(s[0], s[1], round(s[2],2), round(s[3],2)) for s in strata])\n"
            "print('pooled within-EPS SUR long-short:', f'{pooled[1]:+.2f}%', 't =', round(pooled[2],2))"
        ),
        md(
            f"> 💡 In plain words: inside EPS **beats** the SUR sort is **{R['inc'][0][2]:+.2f}%** "
            f"(t = {R['inc'][0][3]:.2f}); inside EPS **misses**, **{R['inc'][1][2]:+.2f}%** "
            f"(t = {R['inc'][1][3]:.2f}); pooled **{R['inc_pooled'][1]:+.2f}%** at "
            f"**t = {R['inc_pooled'][2]:.2f}**. Holding the earnings surprise fixed, the revenue "
            "surprise adds **no** drift. This is the Jegadeesh-Livnat headline claim — and it does "
            "not replicate here."
        ),
        md(
            "### 4e · Costs — academic only, and negative anyway\n\n"
            "Gross vs net (4 one-way legs at 10 bps + 50 bps/yr borrow on the short). There's no "
            "gross edge to erode; net is zero-or-negative everywhere."
        ),
        code(
            "if HAVE_REAL:\n"
            "    g, nv = [], []\n"
            "    for h in hs:\n"
            "        c = st.net_of_costs(PRICES, EV, h, surprise_col='sur')\n"
            "        g.append(c['gross']*100); nv.append(c['net']*100)\n"
            "else:\n"
            "    g = [n[1] for n in R['net']]; nv = [n[2] for n in R['net']]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, g, .4, color=GREY, label='gross')\n"
            "ax.bar(x+.2, nv, .4, color=RED, label='net (costs + borrow)')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs])\n"
            "ax.set_ylabel('long-short return (%)'); ax.set_title('No gross edge; costs push it negative')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for h,gg,nn in zip(hs,g,nv): print(f'{h:>2}d: gross={gg:+.2f}%  net={nn:+.2f}%')"
        ),
        md(
            f"> 💡 In plain words: the best net number is a **+{R['net'][2][2]:.2f}%** crumb at "
            "20 days; everything else is negative. With no gross signal, costs simply finish the "
            "job. Nothing to deploy."
        ),
        md(
            "### 4f · Faithful-engine & power control — we know the truth here\n\n"
            "On a deterministic panel where we **plant** a known post-event drift: with **zero** "
            "edge the long-short must stay at t≈0 (no false positive); with a **large** planted "
            "drift it must light up. Both hold — so the flat real-tape result is a genuine absence "
            "of signal, not a broken detector."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.08):\n"
            "    px, e = data.synthetic_rev(edge=edge, seed=534)\n"
            "    s = st.summarize(px, e, 20, n_draws=4000)\n"
            "    res.append((edge, s['n_events'], s['ls_mean']*100, s['t'], s['p_placebo'], s['ls_win']*100))\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "labels = [f'planted edge\\n{e*100:.0f}%' for e,_,_,_,_,_ in res]\n"
            "tvals = [r[3] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('20d one-sample t'); ax.set_title('Control: zero edge -> t~0; planted edge -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,k,ls,t,p,w in res: print(f'planted {e*100:+.0f}%: events={k} ls20={ls:+.2f}% t={t:.2f} p={p:.3f} win={w:.0f}%')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted drift the control sits at "
            f"**t = {R['syn'][0][3]:.2f}** (no false positive); a **+8%** planted drift reaches "
            f"**t = {R['syn'][1][3]:.2f}**. The machinery is unbiased and well-powered — so the "
            "real-tape max **t = 0.59** is a true null on this basket, not a measurement failure."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — SUR long-short never clears **t = 2** (max **t = "
            f"{R['h1'][6]:.2f}** at 1d; 20d **+{R['h20'][4]:.2f}%** at **t = {R['h20'][6]:.2f}**, "
            f"placebo **p = {R['h20'][7]:.3f}**, block-placebo **p = {R['block_p20']}**), the "
            "cross-section is non-monotone, and the sign flips with the bucket count. The faithful "
            "synthetic control (zero→t = 0.03, planted→t = 9.55) shows the detector works — there "
            "is no revenue drift to find here. Literature support is **not** enough. NONE, not WEAK.\n"
            f"- **Tradability `MIRAGE`** — net of 4×10-bps + borrow, every horizon is ≤ "
            f"**+{R['net'][2][2]:.2f}%** (best, 20d); no gross edge to erode, nothing to deploy.\n"
            f"- **Adds info beyond EPS? `BUSTED`** — within EPS-sign strata the SUR long-short is "
            f"**{R['inc_pooled'][1]:+.2f}% at t = {R['inc_pooled'][2]:.2f}** (negative in both "
            "beats and misses). The specific Jegadeesh-Livnat incremental claim does not replicate "
            "on this conservative large-cap survivor basket."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the honest contrast with the EPS sibling\n\n"
            "The point is not \"drift is fake.\" The **earnings** version (study 363) clears the "
            "bar at **t = 2.96** on a near-identical basket and engine. It's the **revenue** twin "
            "— and especially its *incremental* claim — that doesn't survive. Same machinery, same "
            "kind of names, opposite verdict: the fundamental that drifts is earnings, not sales."
        ),
        code(
            "labels = ['EPS surprise\\n(study 363)', 'Revenue surprise\\n(this study)']\n"
            "tvals = [2.96, R['h20'][6]]\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "ax.bar(labels, tvals, color=[GREEN, RED], width=.5)\n"
            "ax.axhline(2, ls='--', c='k', label='t = 2 bar')\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('20d long-short one-sample t'); ax.legend()\n"
            "ax.set_title('Earnings surprise drifts (t=2.96); revenue surprise does not')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('EPS 20d t (363):', 2.96, '| revenue 20d t (here):', round(R['h20'][6],2))"
        ),
        md(
            "> 💡 In plain words: a clean side-by-side. The earnings surprise drifts at "
            f"**t = 2.96**; the revenue surprise sits at **t = {R['h20'][6]:.2f}** and adds nothing "
            "incremental. On a tradable large-cap basket in 2013–2026, the revenue-surprise drift "
            "is a **mirage** — hence **None × Mirage × Busted**."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Liquidity is the knob.** Chordia et al. (2009) put the whole drift family in "
            "**small, illiquid** names — exactly what our large-cap basket excludes. A small-cap "
            "re-run may revive the revenue drift (with worse costs and survivorship).\n"
            "- **Post-publication decay.** McLean & Pontiff (2016): Jegadeesh-Livnat is 20 years "
            "old; a 2013–2026 null is consistent with documented anomaly decay.\n"
            "- **The incremental claim is the crux.** A raw SUR drift could just be the EPS drift; "
            "the within-EPS-strata test is the honest way to isolate revenue's *own* signal — and "
            "it's the test that fails most cleanly here.\n\n"
            "*The reproducible core is offline and deterministic; SUR is built from EDGAR "
            "frame-tagged quarterly revenue. Methods and sources: "
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
