"""Generate the two narrative notebooks for Study 536 (Anomaly-Decay-Post-Publication).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells read the cached
basket parquet under ../_cache/ if present and otherwise quote the frozen headline numbers
in ``R`` (mirroring docs/results.md), so the notebook re-runs for any reader.
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


# Frozen real-tape + synthetic headline numbers — mirror of docs/results.md (as-of 2026-05).
R = dict(
    window="1980-02 → 2026-05", n_months=556, n_names=40, fp="d7fbd2229df0",
    median_decay=0.31, mean_decay=0.37, mp_benchmark=0.50,
    # per-anomaly: (pub_year, pre_ann%, pre_t, post_ann%, post_t, ratio, decay%, placebo_p,
    #               net_post_ann%, net_post_t)
    mom=dict(pub=1993, pre_ann=14.79, pre_t=3.14, post_ann=3.66, post_t=1.11,
             ratio=0.25, decay=75, plac=0.039, net_post_ann=2.69, net_post_t=0.82),
    rev1=dict(pub=1990, pre_ann=9.16, pre_t=1.93, post_ann=0.18, post_t=0.07,
              ratio=0.02, decay=98, plac=0.054, net_post_ann=-1.92, net_post_t=-0.73, turnover=0.67),
    rev3=dict(pub=1985, pre_ann=7.16, pre_t=0.80, post_ann=2.67, post_t=1.09,
              ratio=0.37, decay=63, plac=0.343, net_post_ann=1.85, net_post_t=0.75, pre_n=24),
    lowvol=dict(pub=2006, pre_ann=-11.38, pre_t=-2.79, post_ann=-9.60, post_t=-2.50,
                ratio=0.84, decay=16, plac=0.630, net_post_ann=-10.35, net_post_t=-2.70),
    # synthetic control (20-seed averaged)
    syn=[dict(decay=0.0, exp=1.00, rec=0.984, pre_t=16.31, post_t=15.87),
         dict(decay=0.5, exp=0.50, rec=0.495, pre_t=16.29, post_t=8.01),
         dict(decay=1.0, exp=0.00, rec=0.004, pre_t=16.26, post_t=0.08)],
)


BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from anomaly_decay import data, strategy as st

def _have_cache():
    return os.path.exists(data._cache_path(data.DEFAULT_CACHE))

HAVE_REAL = _have_cache()

def real_panel():
    return data.load_real(fetch=False)

print("real basket cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Anomaly decay — do edges die once everyone knows about them? 📉\n"
            "### Split four textbook anomalies at their publication year and watch the premium shrink\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Decay%3F: Confirmed](https://img.shields.io/badge/Decay%3F-Confirmed-8b949e?style=flat-square)\n\n"
            "In 2016 McLean & Pontiff replicated 97 published return predictors and found a startling "
            "regularity: once an anomaly is in print, its average return **roughly halves**. The story is "
            "intuitive — publicise a free lunch and arbitrageurs eat it. This notebook makes the reckoning "
            "concrete: we rebuild four classic anomalies (momentum, low-vol, short- and long-term "
            "reversal), cut each series at the year its defining paper appeared, and compare the edge "
            "before vs after.\n\n"
            "> 📓 **This is the plain-language layer.** Want the *t*-stats, the placebo null and the "
            "planted-decay control? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool: every chart below is drawn by "
            "the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Do anomalies really weaken after publication? | **Yes, clearly.** Across our four "
            f"classics the median post/pre ratio is **{R['median_decay']:.2f}** — the edge shrinks by "
            "roughly two-thirds. |\n"
            "| What's the cleanest example? | **12-1 momentum.** Pre-1993 it earned "
            f"**+{R['mom']['pre_ann']:.1f}%/yr** (a strong *t* = {R['mom']['pre_t']:.2f}); after 1993, "
            f"**+{R['mom']['post_ann']:.1f}%/yr** at *t* = {R['mom']['post_t']:.2f} — below the bar. |\n"
            "| Is there anything left to trade? | **No.** Net of trading costs, **no** post-publication "
            "leg clears *t* = 2; the fast, high-turnover one (1-month reversal) goes negative. |\n"
            "| So what's the lesson? | **Defensive.** A backtest that ends before the paper was famous "
            "is not the edge you'll get going forward. |\n\n"
            "> The headline isn't a trade — it's a *warning*. The premium in a published anomaly is "
            "partly the premium for it being *un*-published."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "Every quant blog cites the same canon: momentum, value, low-vol, reversal. Each comes with a "
            "famous backtest and a *t*-stat that looks bulletproof. The implicit promise is that these are "
            "*permanent* features of markets — laws, not fashions.\n\n"
            "McLean & Pontiff's (2016) finding punctures that: the very act of publishing an anomaly — "
            "putting it in a journal, then a textbook, then an ETF — invites the trading that arbitrages it "
            "away. The strong version of the claim we test: *the edge you measure in the original sample is "
            "systematically larger than the edge you'll earn afterwards.*"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "This is the difference between a backtest and a forecast. If anomalies decay after they're "
            "discovered, then every published edge is quoting you its *best* years — the in-sample, "
            "pre-crowding ones — and the number you'll actually live with is smaller, sometimes much "
            "smaller. Fund the headline *t*-stat and you're buying the past tense. The fix is to ask, of "
            "any anomaly: *how much of this backtest happened before everyone knew?*"
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "We run the same experiment on a known answer, then on the real tape:\n\n"
            "1. **A fake anomaly we control.** Plant an edge that *halves* at a chosen 'publication' month. "
            "A faithful split engine should recover a post/pre ratio near **0.5** — and if we plant a "
            "*full* decay, the post-period leg should test as pure noise.\n"
            "2. **The real tape.** Four classic anomalies on a 40-name large-cap basket since 1980, each "
            "split at its real publication year. Watch the before-vs-after edge.\n\n"
            "If the machinery is honest, the planted half-decay comes back as ~0.5, the planted full-decay "
            "comes back as ~0, and the real anomalies show the McLean-Pontiff shrink."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the cleanest real example: 12-1 momentum, split at its 1993 publication.** Same "
            "strategy, two eras."
        ),
        code(
            "if HAVE_REAL:\n"
            "    panel = real_panel()\n"
            "    r = st.run_anomaly(panel, 'mom12_1', 1993)\n"
            "    pre_ann = r['gross']['pre']['ann']*100; post_ann = r['gross']['post']['ann']*100\n"
            "    pre_t = r['gross']['pre']['t']; post_t = r['gross']['post']['t']\n"
            "    ratio = r['gross']['decay_ratio']; plac = r['placebo']['p']\n"
            "else:\n"
            f"    pre_ann, post_ann = {R['mom']['pre_ann']}, {R['mom']['post_ann']}\n"
            f"    pre_t, post_t = {R['mom']['pre_t']}, {R['mom']['post_t']}\n"
            f"    ratio, plac = {R['mom']['ratio']}, {R['mom']['plac']}\n"
            "    print('OFFLINE — frozen numbers from docs/results.md')\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.4))\n"
            "bars = ax.bar(['Before 1993\\n(pre-publication)', 'After 1993\\n(post-publication)'],\n"
            "              [pre_ann, post_ann], color=[GREEN, RED], width=.55)\n"
            "ax.set_ylabel('12-1 momentum long-short (annualised %)')\n"
            "ax.set_title(f'Momentum more than halves after publication (ratio {ratio:.2f})')\n"
            "for b, t in zip(bars, [pre_t, post_t]):\n"
            "    ax.annotate(f'{b.get_height():+.1f}%\\nt={t:.2f}', (b.get_x()+b.get_width()/2, b.get_height()),\n"
            "                ha='center', va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'pre {pre_ann:+.1f}%/yr (t={pre_t:.2f}) -> post {post_ann:+.1f}%/yr (t={post_t:.2f}); '\n"
            "      f'ratio {ratio:.2f}, placebo p={plac:.3f}')"
        ),
        md(
            f"Pre-1993 momentum earned **+{R['mom']['pre_ann']:.1f}%/yr** at a convincing "
            f"*t* = {R['mom']['pre_t']:.2f}. Post-1993: **+{R['mom']['post_ann']:.1f}%/yr** at "
            f"*t* = {R['mom']['post_t']:.2f} — **below the t = 2 bar**. A label-shuffle test says a drop "
            f"this big happens by chance only **{R['mom']['plac']*100:.0f}%** of the time. The anomaly "
            "didn't die, but it more than halved — exactly the prediction."
        ),
        md(
            "**Now all four anomalies at once: the post/pre ratio.** A ratio of 1.0 means no decay; 0.5 "
            "means the edge halved; below 0 means it flipped sign."
        ),
        code(
            "anoms = ['mom12_1', 'st_reversal', 'lt_reversal', 'lowvol']\n"
            "labels = ['12-1\\nmomentum', '1-month\\nreversal', '3-year\\nreversal', 'low\\nvolatility']\n"
            "if HAVE_REAL:\n"
            "    ratios = [st.run_anomaly(panel, a, data.ANOMALIES[a]['pub_year'])['gross']['decay_ratio']\n"
            "              for a in anoms]\n"
            "else:\n"
            f"    ratios = [{R['mom']['ratio']}, {R['rev1']['ratio']}, {R['rev3']['ratio']}, {R['lowvol']['ratio']}]\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "cols = [GREEN if rr < 0.6 else AMBER for rr in ratios]\n"
            "ax.bar(labels, ratios, color=cols, width=.6)\n"
            f"ax.axhline({R['mp_benchmark']}, ls='--', c=GREY, label='McLean-Pontiff ~0.50 benchmark')\n"
            "ax.axhline(1.0, ls=':', c=GREY, label='no decay (ratio = 1)')\n"
            "ax.set_ylabel('post / pre return ratio'); ax.legend()\n"
            "ax.set_title('Most classic anomalies shrink after their paper appears')\n"
            "for i, rr in enumerate(ratios):\n"
            "    ax.annotate(f'{rr:.2f}', (i, rr), ha='center', va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('decay ratios:', dict(zip(['mom','rev1','rev3','lowvol'], [round(x,2) for x in ratios])))"
        ),
        md(
            f"Three of four show heavy decay (median ratio **{R['median_decay']:.2f}**). The exception is "
            "**low-vol**, and for an honest reason: on a *survivor* basket of mega-caps the high-vol names "
            "(the NVDA-type winners) are exactly the ones that survived and ran, so 'long low-vol' simply "
            "*loses* in both halves — there is no positive edge to decay. We keep it in the picture rather "
            "than quietly dropping the result that doesn't fit."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** A method demo, not a trade. The momentum decay is real, but the takeaway "
            "is that the post-publication leg *falls below t = 2* — the edge erodes. No new signal claimed.\n"
            "- **Tradability — Mirage.** Net of costs, no positive post-publication leg clears t = 2, and "
            "the fast one goes negative. The part you'd want to trade is the part that decayed.\n"
            "- **Do anomalies decay after publication? — Confirmed.** Median post/pre ratio "
            f"{R['median_decay']:.2f}; momentum −{R['mom']['decay']}%, reversal −{R['rev1']['decay']}%. The "
            "~50% half-life shows up even on a tiny survivor basket."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "No — and that's the lesson. The pre-publication edge is gone by definition (it's *pre*-"
            "publication), and the post-publication remnant doesn't survive trading costs: momentum's "
            f"+{R['mom']['post_ann']:.1f}%/yr becomes +{R['mom']['net_post_ann']:.1f}%/yr net "
            f"(*t* = {R['mom']['net_post_t']:.2f}), and 1-month reversal — which churns "
            f"{R['rev1']['turnover']:.0%} of the book a month — turns **negative**. The real value of this "
            "study is defensive: before you trust a famous backtest, ask how much of it predates the fame."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The survivorship angle.** [Study 345 — Survivorship-Bias](../../345-survivorship-bias/) "
            "is the distortion we lean into here: a survivor basket inflates the *pre*-publication half "
            "most.\n"
            "- **The in-sample angle.** [Study 346 — Multiple-Testing](../../346-multiple-testing/) shows "
            "how a *t*-stat inflates *within* one sample; this study shows how it deflates *out* of it.\n"
            "- **Swap the anomalies.** Fork this, add your favourite published factor and its publication "
            "year, and see whether its post/pre ratio joins the ~0.5 crowd.\n\n"
            "*Think your favourite anomaly is permanent? Split it at the year its paper came out and look "
            "at the second half — that's the half you actually get to trade.*"
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
            "# Anomaly-Decay-Post-Publication — a quantitative teardown 🔬\n"
            "### Tercile long-shorts · pre/post split at publication · one-sample *t* per leg · "
            "label-shuffle placebo · costs × turnover · planted-decay control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Decay%3F: Confirmed](https://img.shields.io/badge/Decay%3F-Confirmed-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same seven "
            "beats, every claim now carrying its standard error.* We rebuild four classic anomalies as "
            "dollar-neutral tercile long-shorts, split each at its publication year, and test each era's "
            "mean against zero — with a permutation null on the pre/post boundary and a planted-decay "
            "synthetic control proving the split engine is faithful.\n\n"
            f"> ⚠️ **Not investment advice.** Real data: a 40-name large-cap **survivor** basket, monthly "
            f"total return {R['window']} (as-of 2026-05, fingerprint `{R['fp']}`); the offline core runs "
            "on a deterministic synthetic panel. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Methodology demo. 12-1 momentum: pre-1993 ann "
            f"+{R['mom']['pre_ann']:.1f}% (*t*={R['mom']['pre_t']:.2f}) → post +{R['mom']['post_ann']:.1f}% "
            f"(*t*={R['mom']['post_t']:.2f}), **below t=2**. No new edge claimed. |\n"
            f"| **Tradability** | `MIRAGE` | Net of costs every positive post leg is sub-*t*=2 "
            f"(momentum +{R['mom']['net_post_ann']:.1f}%/yr, *t*={R['mom']['net_post_t']:.2f}); 1-month "
            "reversal goes negative. |\n"
            f"| **Decay?** | `CONFIRMED` | Median post/pre ratio **{R['median_decay']:.2f}**; momentum "
            f"−{R['mom']['decay']}% (placebo *p*={R['mom']['plac']:.3f}), reversal −{R['rev1']['decay']}%. |\n\n"
            "> 💡 In plain words: McLean & Pontiff (2016) replicated 97 predictors and found post-"
            "publication returns ~58% lower than in-sample. We reproduce the *mechanism* on four classics: "
            "split the series at the paper's year, and the second half is systematically the weaker one."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let an anomaly's monthly long-short return be $r_t$ with population mean $\\mu_{\\text{pre}}$ "
            "before the publication month $\\tau$ and $\\mu_{\\text{post}}$ after. Define the **decay ratio** "
            "$\\rho = \\mu_{\\text{post}} / \\mu_{\\text{pre}}$.\n\n"
            "- **H₀ (permanence).** $\\rho = 1$: the edge is a structural feature, unchanged by being "
            "published.\n"
            "- **H₁ (McLean-Pontiff decay).** $\\rho \\approx 0.5$: publication invites arbitrage that "
            "halves the premium.\n"
            "- **H₂ (full arbitrage).** $\\rho \\to 0$: the post-period leg is statistically "
            "indistinguishable from zero.\n\n"
            "We estimate $\\hat\\mu_{\\text{pre}}, \\hat\\mu_{\\text{post}}$ with a one-sample *t* on each "
            "leg, test the *drop* with a label-shuffle permutation null, and confirm the engine can recover "
            f"a *known* $\\rho$ on synthetic data. Real result: median $\\hat\\rho = {R['median_decay']:.2f}$ "
            "— between H₁ and H₂, decisively against H₀."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "If $\\rho < 1$ then every published backtest over-states the deliverable edge, and the "
            "over-statement is *predictable*: it scales with how much of the sample predates publication. "
            "The desk's contribution is to make this operational on a small, reproducible basket — not to "
            "re-estimate McLean-Pontiff's 97-anomaly average, but to *watch the split happen* on the four "
            "anomalies a retail quant is most likely to trust, with the survivorship caveat stated plainly."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Universe.** 40-name large-cap survivor basket, monthly total return, 1980–2026.\n"
            "- **Signals (price-only).** 12-1 momentum (skip last month); low-vol (−12m std); 1-month "
            "reversal (−last return); 3-year reversal (−months t-36..t-13).\n"
            "- **Long-short.** Dollar-neutral top-tercile minus bottom-tercile, equal-weight, rebuilt "
            "monthly. One execution lag: signal at month *t* earns month *t+1*'s return.\n"
            "- **Split.** Cut at each anomaly's real publication year; one-sample *t* on each leg; "
            "decay ratio = post/pre.\n"
            "- **Placebo.** 5,000-draw permutation of the pre/post labels — the share of random boundaries "
            "with a drop at least as large as the real one.\n"
            "- **Costs.** One-way 10 bps/leg × turnover + 50 bps/yr borrow on the short leg.\n"
            "- **Positive control.** A planted-decay panel (20-seed averaged) the split engine must recover "
            "— else a dead pipeline proves nothing."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The synthetic control — the split engine is faithful\n\n"
            "Plant an edge that decays by a known fraction at a 'publication' month and check the engine "
            "recovers the ratio (averaged over 20 seeds — no single lucky RNG draw). A planted half-decay "
            "must come back ~0.5; a planted full-decay must leave a post leg that tests as noise."
        ),
        code(
            "rows = []\n"
            "for d in (0.0, 0.5, 1.0):\n"
            "    sc = st.synthetic_control(post_decay=d, n_seeds=20)\n"
            "    rows.append((d, sc['expected_ratio'], sc['recovered_ratio_mean'],\n"
            "                 sc['pre_t_mean'], sc['post_t_mean']))\n"
            "tab = pd.DataFrame(rows, columns=['planted_decay','expected_ratio','recovered_ratio',\n"
            "                                  'pre_t','post_t']).set_index('planted_decay')\n"
            "print(tab.round(3).to_string())\n"
            "fig, ax = plt.subplots(figsize=(7.4, 4.2))\n"
            "ax.plot(tab.index, tab['expected_ratio'], 'o--', c=GREY, label='planted (expected)')\n"
            "ax.plot(tab.index, tab['recovered_ratio'], 'o-', c=GREEN, lw=2, label='recovered')\n"
            "ax.set_xlabel('planted post-publication decay'); ax.set_ylabel('post/pre ratio')\n"
            "ax.set_title('Engine recovers the planted decay to 2 decimals'); ax.legend()\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: planted half-decay → recovered **{R['syn'][1]['rec']:.3f}** (post "
            f"*t*={R['syn'][1]['post_t']:.1f}); planted full-decay → recovered **{R['syn'][2]['rec']:.3f}** "
            f"with a dead post leg (*t*={R['syn'][2]['post_t']:.2f}). The split detector is faithful — it "
            "cannot manufacture a decay that wasn't planted, nor hide one that was. **This is a power check "
            "only; it never supports a real-tape stamp.**"
        ),
        md(
            "### 4b · The real tape — per-anomaly pre/post split\n\n"
            "Each anomaly's long-short, cut at its publication year, with the one-sample *t* on each leg "
            "and the label-shuffle placebo *p* on the drop."
        ),
        code(
            "anoms = ['mom12_1', 'st_reversal', 'lt_reversal', 'lowvol']\n"
            "if HAVE_REAL:\n"
            "    panel = real_panel()\n"
            "    rows = []\n"
            "    for a in anoms:\n"
            "        r = st.run_anomaly(panel, a, data.ANOMALIES[a]['pub_year'])\n"
            "        g = r['gross']\n"
            "        rows.append((a, data.ANOMALIES[a]['pub_year'], g['pre']['ann']*100, g['pre']['t'],\n"
            "                     g['post']['ann']*100, g['post']['t'], g['decay_ratio'], r['placebo']['p']))\n"
            "    tab = pd.DataFrame(rows, columns=['anomaly','pub','pre_ann%','pre_t','post_ann%','post_t',\n"
            "                                      'ratio','placebo_p']).set_index('anomaly')\n"
            "    print(tab.round(3).to_string())\n"
            "else:\n"
            "    print('OFFLINE — frozen numbers from docs/results.md')\n"
            f"    print('mom: pre {R['mom']['pre_ann']}% t{R['mom']['pre_t']} -> post {R['mom']['post_ann']}% t{R['mom']['post_t']}, ratio {R['mom']['ratio']}, p{R['mom']['plac']}')\n"
            f"    print('rev1: pre {R['rev1']['pre_ann']}% t{R['rev1']['pre_t']} -> post {R['rev1']['post_ann']}% t{R['rev1']['post_t']}, ratio {R['rev1']['ratio']}, p{R['rev1']['plac']}')"
        ),
        md(
            f"> 💡 In plain words: momentum (pre *t*={R['mom']['pre_t']:.2f} → post {R['mom']['post_t']:.2f}) "
            f"and short-term reversal (pre {R['rev1']['pre_t']:.2f} → post {R['rev1']['post_t']:.2f}) both "
            "collapse below the *t*=2 bar after publication. Long-term reversal's pre leg has only "
            f"~{R['rev3']['pre_n']} months (basket starts ~1980, paper is 1985) so its estimate is noisy. "
            "Low-vol is *negative* throughout on this survivor basket — no positive edge to decay."
        ),
        md(
            "### 4c · The decay-ratio picture and the McLean-Pontiff benchmark"
        ),
        code(
            "labels = ['12-1\\nmomentum', '1-month\\nreversal', '3-year\\nreversal', 'low\\nvolatility']\n"
            "if HAVE_REAL:\n"
            "    ratios = [st.run_anomaly(panel, a, data.ANOMALIES[a]['pub_year'])['gross']['decay_ratio']\n"
            "              for a in anoms]\n"
            "    med = float(np.median([x for x in ratios if np.isfinite(x)]))\n"
            "else:\n"
            f"    ratios = [{R['mom']['ratio']}, {R['rev1']['ratio']}, {R['rev3']['ratio']}, {R['lowvol']['ratio']}]\n"
            f"    med = {R['median_decay']}\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "ax.bar(labels, ratios, color=[GREEN if r < 0.6 else AMBER for r in ratios], width=.6)\n"
            f"ax.axhline({R['mp_benchmark']}, ls='--', c=GREY, label='McLean-Pontiff ~0.50')\n"
            "ax.axhline(1.0, ls=':', c=GREY, label='no decay')\n"
            "ax.axhline(med, c=RED, lw=1.5, label=f'median {med:.2f}')\n"
            "ax.set_ylabel('post / pre ratio'); ax.legend()\n"
            "ax.set_title('Decay ratios vs the McLean-Pontiff ~0.50 half-life')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('median decay ratio:', round(med, 2))"
        ),
        md(
            f"> 💡 In plain words: the median ratio **{R['median_decay']:.2f}** is *below* the "
            f"~0.50 McLean-Pontiff average — more decay, not less. On a survivor basket the pre-publication "
            "half is the most over-stated (it's the survivors' best, earliest years), so the drop looks "
            "even sharper than in their full 97-anomaly cross-section."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — a methodology demo; the momentum decay is real but the post leg falls "
            f"below *t*=2 ({R['mom']['post_t']:.2f}). No new edge claimed; survivorship inflates the pre half.\n"
            f"- **Tradability `MIRAGE`** — net of costs no positive post leg clears *t*=2 "
            f"(momentum {R['mom']['net_post_ann']:+.1f}%/yr at *t*={R['mom']['net_post_t']:.2f}); the "
            "high-turnover reversal goes negative.\n"
            f"- **Decay? `CONFIRMED`** — median post/pre ratio {R['median_decay']:.2f}; momentum "
            f"−{R['mom']['decay']}% (placebo *p*={R['mom']['plac']:.3f}), short-term reversal "
            f"−{R['rev1']['decay']}%. The half-life is real even on a tiny survivor basket."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — costs and the absent edge\n\n"
            "The pre-publication edge is unrecoverable (it's in the past); the post-publication remnant is "
            "tested net of one-way 10 bps/leg × turnover + 50 bps/yr borrow below."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for a in anoms:\n"
            "        r = st.run_anomaly(panel, a, data.ANOMALIES[a]['pub_year'])\n"
            "        rows.append((a, r['gross']['post']['ann']*100, r['net']['post']['ann']*100,\n"
            "                     r['net']['post']['t']))\n"
            "    tab = pd.DataFrame(rows, columns=['anomaly','post_gross_ann%','post_net_ann%',\n"
            "                                      'post_net_t']).set_index('anomaly')\n"
            "    print(tab.round(2).to_string())\n"
            "else:\n"
            "    print('OFFLINE — frozen:')\n"
            f"    print('mom post net {R['mom']['net_post_ann']}%/yr t{R['mom']['net_post_t']}')\n"
            f"    print('rev1 post net {R['rev1']['net_post_ann']}%/yr t{R['rev1']['net_post_t']}')\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.0))\n"
            "if HAVE_REAL:\n"
            "    ax.bar(tab.index, tab['post_net_ann%'], color=[GREEN if v>0 else RED for v in tab['post_net_ann%']], width=.6)\n"
            "    ax.set_ylabel('post-publication NET annualised %')\n"
            "    ax.axhline(0, c=GREY)\n"
            "    ax.set_title('Net of costs, no post-publication leg is worth trading')\n"
            "    plt.tight_layout(); plt.show()"
        ),
        md(
            "> 💡 In plain words: the value here is *defensive*. Before funding anyone's famous backtest, "
            "ask how much of the sample predates the paper — and net the remnant for the costs you'll "
            "actually pay. The definition of a **mirage**: the edge that was there in the brochure and "
            "gone by the time you could read about it."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The survivorship distortion.** [Study 345 — Survivorship-Bias](../../345-survivorship-bias/) "
            "— the bias that inflates the pre-publication half most here.\n"
            "- **The in-sample mirror.** [Study 346 — Multiple-Testing](../../346-multiple-testing/) — how "
            "*t*-stats inflate *within* a sample; 536 shows how they deflate *out* of it.\n"
            "- **The same engine on an event anomaly.** [Study 363 — PEAD-Drift](../../363-pead-drift/) — "
            "the same tercile / one-sample-*t* / label-shuffle machinery, same survivor caveat.\n"
            "- **Extend it.** Add Benjamini-style out-of-sample decay across McLean-Pontiff's full 97, or "
            "swap in a point-in-time index-membership basket to strip the survivorship tilt and see whether "
            "the decay ratio holds.\n\n"
            "*A published edge quotes you its best years. Split it at the paper's date and the second half "
            "is the only one you get to keep.*"
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
