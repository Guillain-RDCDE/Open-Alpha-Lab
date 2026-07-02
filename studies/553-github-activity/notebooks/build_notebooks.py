"""Generate the two narrative notebooks for Study 553 (GitHub-Activity).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). This study is
**synthetic-only** (no point-in-time GitHub->ticker tape exists on a free stack), so *every* cell is
deterministic and offline — there is no real-tape banner anywhere. The dict ``R`` mirrors the
computed headline numbers in docs/results.md.
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


# Frozen headline numbers — mirror of docs/results.md (synthetic panel, ic=0.03, seed 553,
# 30 firms x 40 quarters, panel fp 33d0732cb08f; as-of 2026-06-30).
R = dict(
    fp_panel="33d0732cb08f", n_firms=30, n_periods=40, ic_plant=0.03,
    mean_ic=0.045, ic_t=1.64, hit=0.60, placebo_p=0.122,
    ls_ann=14.1, ls_t=1.51, net_ann=12.3, cost_bps=10.0, borrow_bps=100.0,
    null_ic_t=0.71,
    # robustness: (sample, split, mean_ic, ic_t, ls_ann%, ls_t)
    rob=[("full", "deciles", 0.045, 1.64, 11.3, 0.71),
         ("full", "quintiles", 0.045, 1.64, 14.1, 1.51),
         ("full", "terciles", 0.045, 1.64, 15.1, 2.25),
         ("first half", "quintiles", 0.030, 0.89, 5.5, 0.45),
         ("second half", "quintiles", 0.059, 1.37, 22.7, 1.59)],
    # synthetic control: (ic, mean IC-t over 25 seeds)
    ctrl=[(0.0, 0.22), (0.02, 0.82), (0.03, 1.14), (0.05, 1.77), (0.10, 3.38)],
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

from github_activity import data, strategy as st

# Synthetic-only: the point-in-time GitHub->ticker tape does not exist on a free stack, so the
# whole study runs on this deterministic panel. ic = 0.03 plants a *realistic* alt-data effect.
IC_PLANT = 0.03
PANEL, TRUTH = data.synthetic_panel(ic=IC_PLANT, seed=553)
print("synthetic panel:", PANEL.shape[0], "rows |",
      PANEL.firm.nunique(), "firms x", PANEL.period.nunique(), "quarters |",
      "planted ic =", IC_PLANT, "| fp", data.fingerprint(PANEL))
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# GitHub Activity as a Trading Signal — read the factory floor of software? 🐙\n"
            "### Do surges in a tech firm's open-source commits foreshadow its stock?\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n\n"
            "Here's a tempting alt-data idea. Every big tech company ships code in the open — you can "
            "*watch* their engineers commit, merge pull-requests and rack up stars on GitHub, live. "
            "If a company's open-source output suddenly **accelerates**, maybe real product-shipping "
            "is accelerating too, and the stock market hasn't caught on yet. So: score each firm by "
            "its **GitHub velocity**, and bet that the fast-shipping ones out-earn the slow ones. "
            "Read the factory floor before the earnings call does.\n\n"
            "It's a *plausible* idea — so we test it. But there's a catch we'll hit almost "
            "immediately: **you can't actually get honest data**, which caps how far this can ever "
            "go.\n\n"
            "> 📓 **This is the plain-language layer.** Want the IC *t*-stats, the placebo test and "
            "the cost maths? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice, and synthetic-only.** No free feed lets you rebuild the "
            "GitHub velocity a trader *would have seen* at each past date, so we test the idea on a "
            "deterministic simulated world where we *plant* the effect ourselves. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ----
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Is there *any* signal? | **A little, and the right way.** Firms with faster GitHub "
            f"velocity did earn a bit more next quarter — the correlation is **+{R['mean_ic']:.3f}**. |\n"
            f"| Is it strong enough to trust? | **No.** Its *t*-score is **{R['ic_t']}** — below the "
            "**2** we require — and a shuffle test can't rule out luck (*p* = "
            f"{R['placebo_p']:.3f}). |\n"
            "| Is it stable? | **Not really.** The sign holds, but the strength flickers depending on "
            "which half of the history and which cut you use. |\n"
            "| Could you trade it? | **No — you can't even get the data.** GitHub only shows you "
            "*today's* snapshot; repos get renamed, archived or made private, so the past you'd "
            "backtest on is unrecoverable and biased. |\n\n"
            "> So: a plausible, right-signed idea that never clears the bar — and, worse, can *never* "
            "be proven on real data because the real data doesn't exist in usable form."
        ),

        # ---- BEAT 1 — THE CLAIM ----
        md(
            "## 1 · The claim\n\n"
            "> *\"A tech firm's public open-source velocity — how fast it's committing, merging and "
            "gathering stars — is a live read on its innovation, and surges foreshadow the stock.\"*\n\n"
            "This is the alt-data playbook (satellite car-counts, credit-card panels, web traffic) "
            "pointed at a company's *own* engineering output. There's a real academic prior behind "
            "it: firms that invest more in innovation (R&D) have historically earned higher "
            "subsequent returns because the market under-weights intangibles. GitHub velocity is the "
            "software-native, *real-time* version of that idea."
        ),

        # ---- BEAT 2 — SO WHAT ----
        md(
            "## 2 · So what?\n\n"
            "If it worked, you'd have a fundamentals nowcast that updates *daily* and is *free to "
            "watch* — weeks or months ahead of the quarterly report. That's the whole appeal of "
            "alt-data: see the truth before the filing. The desk has already looked at the *audited* "
            "innovation proxy ([400 Patent-Intensity](../../400-patent-intensity/), R&D / revenue); "
            "here we go after the **live telemetry** version."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ----
        md(
            "## 3 · How would we even know?\n\n"
            "1. **Score velocity.** For each firm each quarter: blend commit / pull-request / star "
            "run-rate into one number, ranked across the field (higher = shipping faster than "
            "peers).\n"
            "2. **Look forward.** Line each firm's velocity up against its *next* quarter's return.\n"
            "3. **Measure the link.** The 'information coefficient' (IC) is just the rank correlation "
            "between velocity and next-quarter return. Average it across quarters; if it's reliably "
            "positive, the nowcast works.\n\n"
            "**What would make us say 'mirage'?** An IC that doesn't clear a *t* of 2, a shuffle test "
            "that can't rule out luck, or a sign that flips around across sub-samples.\n\n"
            "*The data problem:* we can't rebuild the *past* velocity a trader would've seen (GitHub "
            "hides deleted/renamed/private history), so we test the machinery on a **simulated** world "
            "where we control how strong the true signal is — and set it to a *realistic, modest* "
            "level."
        ),

        # ---- BEAT 4 — THE TEARDOWN ----
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**Quarter by quarter, does higher velocity line up with higher next-quarter returns?** "
            "Each dot is one quarter's correlation (IC)."
        ),
        code(
            "ics = st.period_ics(PANEL)\n"
            "summ = st.ic_summary(PANEL)\n"
            "fig, ax = plt.subplots(figsize=(9, 4.3))\n"
            "cols = [GREEN if v > 0 else RED for v in ics.values]\n"
            "ax.bar(range(len(ics)), ics.values, color=cols, width=.8)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.axhline(summ['mean_ic'], c=AMBER, lw=2, ls='--', label=f\"mean IC {summ['mean_ic']:+.3f}\")\n"
            "ax.set_xlabel('quarter'); ax.set_ylabel('IC (velocity vs next-qtr return)')\n"
            "ax.set_title(f\"Right sign, but weak: mean IC {summ['mean_ic']:+.3f} (t {summ['t']:.2f})\")\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"mean IC {summ['mean_ic']:+.3f} | t {summ['t']:.2f} | positive in \"\n"
            "      f\"{summ['hit']*100:.0f}% of quarters\")"
        ),
        md(
            f"So the link is **there and pointing the right way** (positive in ~{int(R['hit']*100)}% "
            f"of quarters, mean IC **+{R['mean_ic']:.3f}**) — but it's *small*, and small signals are "
            f"easy to fake with luck. The *t*-score is **{R['ic_t']}**, under our bar of 2."
        ),
        md(
            "**Is it just luck?** Shuffle the velocity labels against the returns thousands of times "
            "and see how often pure noise beats what we saw."
        ),
        code(
            "p = st.placebo_pvalue(PANEL, n_perm=1500, seed=553)\n"
            "obs = abs(st.ic_summary(PANEL)['mean_ic'])\n"
            "print(f'observed |mean IC| = {obs:.3f}  ->  placebo p = {p:.3f}')\n"
            "print('Not significant (p > 0.05): noise reproduces this too often.' if p > 0.05\n"
            "      else 'Significant.')"
        ),
        md(
            f"> The placebo *p* is about **{R['placebo_p']:.2f}** — noise reproduces a link this big "
            "more than 1 time in 10. We *cannot* rule out luck on this sample."
        ),

        # ---- BEAT 5 — VERDICT ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Weak.** Right sign, but IC-*t* **{R['ic_t']}** (below 2) and placebo *p* "
            f"**{R['placebo_p']:.2f}**; and it can *never* be 'Real' because there's no honest real "
            "tape to test it on.\n"
            "- **Tradability — Mirage.** You can't reconstruct past GitHub velocity, so you can't "
            "actually build or backtest the signal — before you even get to the short-borrow cost."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ----
        md(
            "## 6 · Could you actually trade it?\n\n"
            "No — and for an unusual reason. Normally a signal dies at *costs*. This one dies "
            "*earlier*: **you can't get the data.** GitHub's public feeds show you the *current* "
            "state of the world. Repos get renamed, archived, deleted, or moved private; commit "
            "histories can be rewritten; the API is rate-limited. So the velocity a trader would have "
            "*seen* on some Tuesday in 2019 is largely unrecoverable — and any backtest built on "
            "today's snapshot is quietly contaminated by survivorship and look-ahead. Even if the "
            "nowcast were real, you couldn't honestly measure it, let alone the short-borrow you'd "
            "pay to bet against the laggards."
        ),

        # ---- BEAT 7 — GOING FURTHER ----
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The audited cousin.** [400 Patent-Intensity](../../400-patent-intensity/) uses "
            "*reported R&D / revenue* — data you *can* get point-in-time — and still only reaches a "
            "'Weak' innovation premium.\n"
            "- **Other firm-telemetry alt-data.** [392 Glassdoor-Sentiment](../../392-glassdoor-sentiment/) "
            "and [528 Labor-Hiring-Rate](../../528-labor-hiring-rate/) hit the same *point-in-time* "
            "wall.\n\n"
            "*Think the nowcast is real? The honest test needs a **point-in-time GitHub archive** "
            "(e.g. the GH Archive on BigQuery, snapshotted daily) mapped to tickers with no "
            "survivorship holes. Fork this, build that tape, and show the IC clearing t = 2 out of "
            "sample.*"
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
            "# GitHub Activity as an Alt-Data Nowcast — a quantitative teardown 🔬\n"
            "### Per-period Spearman IC · label-shuffle placebo · decile long-short + borrow · tail/sub-sample robustness · seed-robust synthetic control\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We treat public GitHub "
            "velocity as a cross-sectional predictor and run the standard nowcast teardown: the "
            "information coefficient, its *t*, a placebo null, a long-short book, robustness, and a "
            "planted-effect control.\n\n"
            "> ⚠️ **Not investment advice — and synthetic-only.** No free, point-in-time, "
            "survivorship-clean GitHub→ticker tape exists (renames, archives, private moves, rate "
            f"limits), so *every* cell here is a deterministic simulated world (seed 553, panel fp "
            f"`{R['fp_panel']}`) with the believers' effect planted at a **realistic** `ic = "
            f"{R['ic_plant']}`. A synthetic-only study can never earn `REAL` — the ceiling is "
            "`WEAK`. Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | Mean per-period Spearman IC **+{R['mean_ic']:.3f}** at *t* "
            f"**{R['ic_t']}** (right sign, below the ≥ 2 bar), placebo *p* **{R['placebo_p']:.3f}**; "
            "significance flickers across cuts; **no real tape possible** ⇒ capped at `WEAK`. |\n"
            f"| **Tradability** | `MIRAGE` | The point-in-time signal is un-reconstructable on a free "
            f"stack; on top, the long-short shorts the laggards (borrow). Gross **+{R['ls_ann']}%/yr** "
            f"(*t* {R['ls_t']}) → net **+{R['net_ann']}%/yr**, never clearing significance. |\n\n"
            "> 💡 In plain words: the engine works (the synthetic control proves it), the planted "
            "effect is recovered with the right sign — but at a realistic alt-data strength it doesn't "
            "clear *t* = 2, and the real data needed to certify it doesn't exist."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $v_{i,t}$ be firm $i$'s cross-sectionally z-scored GitHub velocity at rebalance $t$ "
            "and $r_{i,t+1}$ its forward return.\n\n"
            "- **H₁ (nowcast).** The per-period rank IC $\\rho_t = \\text{corr}(\\text{rank } v_{\\cdot,t}, "
            "\\text{rank } r_{\\cdot,t+1})$ has $\\overline{\\rho} > 0$ at *t* ≥ 2.\n"
            "- **H₂ (tradable).** A long-top / short-bottom velocity book earns a positive spread at "
            "*t* ≥ 2, net of costs and the short-leg borrow.\n"
            "- **H₃ (robust).** The IC's sign/strength is stable across tail fractions and "
            "sub-samples.\n"
            "- **H₄ (data).** The signal can be reconstructed **point-in-time** on a free stack.\n\n"
            "We find **H₁ not cleared** (right sign, *t* 1.64), **H₂ not cleared** (spread *t* 1.51), "
            "**H₃ fragile** (sign stable, significance flickers), and **H₄ false** (no point-in-time "
            "GitHub→ticker tape) — which alone caps the Signal at `WEAK`."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The economic prior is real: innovation intensity has predicted returns "
            "(Chan-Lakonishok-Sougiannis 2001; Hirshleifer-Hsu-Li 2013), and GitHub velocity is a "
            "*live, output-side* proxy for it. Two forces cap it here: (a) a **realistic alt-data IC "
            "is small** (~0.03–0.05), so certifying it needs a long, clean tape; and (b) that tape "
            "**cannot be built** — GitHub feeds are current snapshots, so past velocity is "
            "survivorship-/look-ahead-contaminated. This is the same point-in-time wall as "
            "[392 Glassdoor](../../392-glassdoor-sentiment/) and [528 Hiring](../../528-labor-hiring-rate/)."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Signal.** Cross-sectional z-scored velocity $v_{i,t}$ each quarter.\n"
            "- **IC.** Per-period Spearman $\\rho_t$ of velocity vs *forward* return; headline = "
            "$\\overline{\\rho}$ and its *t* = $\\overline{\\rho} / (s_\\rho / \\sqrt{T})$.\n"
            "- **Placebo.** Shuffle velocity vs returns *within each period* (preserves the "
            "cross-sectional structure, breaks the link); *p* = tail fraction of |mean IC|.\n"
            "- **Book.** Long top-$f$ / short bottom-$f$, equal weight, rebalanced each period; "
            "*t* on the spread.\n"
            "- **Frictions.** One-way cost per leg per rebalance + annual borrow on the short leg.\n"
            "- **Robustness.** $f \\in \\{0.1, 0.2, 1/3\\}$ × {full, first half, second half}.\n"
            "- **Positive control.** Plant $ic$ of increasing strength; average the IC-*t* over 25 "
            "seeds (house rule) — must be flat at the null and clear the bar as $ic$ grows.\n\n"
            "Timing: velocity is a trailing z-score known at the rebalance close, scored against the "
            "*forward* return — one documented execution lag, no look-ahead."
        ),

        # ---- BEAT 4 ----
        md("## 4 · The teardown"),
        md(
            "### 4a · The information coefficient — H₁\n\n"
            "Per-period Spearman IC of velocity vs forward return, with the mean and its *t*."
        ),
        code(
            "ics = st.period_ics(PANEL); summ = st.ic_summary(PANEL)\n"
            "p = st.placebo_pvalue(PANEL, n_perm=1500, seed=553)\n"
            "fig, ax = plt.subplots(figsize=(9, 4.3))\n"
            "cols = [GREEN if v > 0 else RED for v in ics.values]\n"
            "ax.bar(range(len(ics)), ics.values, color=cols, width=.8)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.axhline(summ['mean_ic'], c=AMBER, lw=2, ls='--', label=f\"mean IC {summ['mean_ic']:+.3f}\")\n"
            "ax.set_xlabel('quarter'); ax.set_ylabel('Spearman IC')\n"
            "ax.set_title(f\"mean IC {summ['mean_ic']:+.3f} (t {summ['t']:.2f}) | placebo p {p:.3f}\")\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"mean IC {summ['mean_ic']:+.3f} | t {summ['t']:.2f} | \"\n"
            "      f\"positive {summ['hit']*100:.0f}% of quarters | placebo p {p:.3f}\")"
        ),
        md(
            f"> 💡 In plain words: H₁ **not cleared.** The IC is the right sign (+{R['mean_ic']:.3f}) "
            f"but *t* = {R['ic_t']} < 2 and the placebo *p* = {R['placebo_p']:.3f} can't rule out "
            "luck. A genuine-but-tiny alt-data signal."
        ),
        md(
            "### 4b · The long-short book — H₂ (with costs + borrow)\n\n"
            "Long top-quintile / short bottom-quintile velocity, then net of costs and the short-leg "
            "borrow."
        ),
        code(
            "bk = st.long_short_book(PANEL, frac=0.2)\n"
            "net = st.net_spread(bk, cost_bps=10.0, borrow_ann_bps=100.0)\n"
            "gross, netv, t = net['gross_ann']*100, net['net_ann']*100, bk['t']\n"
            "fig, ax = plt.subplots(figsize=(7, 4.2))\n"
            "ax.bar(['gross', 'net\\n(costs+borrow)'], [gross, netv], color=[AMBER, GREY], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('long-short spread %/yr')\n"
            "ax.set_title(f'Spread {gross:+.1f}%/yr gross (t {t:+.2f}) -> {netv:+.1f}%/yr net')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross {gross:+.1f}%/yr (t {t:+.2f}) -> net {netv:+.1f}%/yr '\n"
            "      f'(10 bps/leg/qtr + 100 bps/yr borrow)')"
        ),
        md(
            f"> 💡 In plain words: H₂ **not cleared.** The spread is positive (+{R['ls_ann']}%/yr) but "
            f"at *t* {R['ls_t']} it doesn't clear 2; costs shave it to +{R['net_ann']}%/yr. Positive, "
            "uncertified."
        ),
        md(
            "### 4c · Robustness — H₃ (does it hold across cuts?)\n\n"
            "IC and long-short across tail fractions and sub-samples."
        ),
        code(
            "rows = []\n"
            "periods = sorted(PANEL['period'].unique()); mid = periods[len(periods)//2]\n"
            "subs = {'full': PANEL, 'first half': PANEL[PANEL.period < mid],\n"
            "        'second half': PANEL[PANEL.period >= mid]}\n"
            "for sname, sub in subs.items():\n"
            "    ic = st.ic_summary(sub)\n"
            "    for frac, fl in [(0.1,'deciles'), (0.2,'quintiles'), (1/3,'terciles')]:\n"
            "        b = st.long_short_book(sub, frac=frac)\n"
            "        rows.append((sname, fl, ic['mean_ic'], ic['t'], b['mean_ann']*100, b['t']))\n"
            "tab = pd.DataFrame(rows, columns=['sample','split','mean_ic','ic_t','ls_ann%','ls_t'])\n"
            "print(tab.round(3).to_string(index=False))\n"
            "fig, ax = plt.subplots(figsize=(9, 4.3))\n"
            "q = tab[tab.split=='quintiles']\n"
            "cols = [GREEN if t>=2 else (AMBER if t>0 else RED) for t in q['ls_t']]\n"
            "ax.bar(q['sample'], q['ls_t'], color=cols, width=.5)\n"
            "ax.axhline(2, ls='--', c=GREY, label='t = 2 bar'); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('long-short t (quintiles)'); ax.legend()\n"
            "ax.set_title('Sign stable, significance flickers by sub-sample')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            "> 💡 In plain words: H₃ **fragile.** The IC is positive in every cut, but its *t* only "
            "reaches 2 at the barely-sorted tercile split, and each half on its own is insignificant. "
            "Sign-stable, significance-unstable — the hallmark of `WEAK`."
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — H₁ not cleared (IC +{R['mean_ic']:.3f}, *t* {R['ic_t']}, placebo "
            f"*p* {R['placebo_p']:.3f}); H₃ fragile; and H₄ false (no real tape) caps it at `WEAK`.\n"
            f"- **Tradability `MIRAGE`** — H₄ false: the point-in-time signal is un-reconstructable; "
            f"H₂ not cleared (gross +{R['ls_ann']}%/yr *t* {R['ls_t']} → net +{R['net_ann']}%/yr)."
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you trade it? — the data wall\n\n"
            "This signal dies *before* costs, at the data. GitHub's public feeds are **current "
            "snapshots**: renamed / archived / deleted / privatised repos vanish, histories are "
            "rewritable, the API is rate-limited. So the velocity known at each *past* date is "
            "un-reconstructable, and any free backtest is survivorship-/look-ahead-contaminated. The "
            "short-borrow on the laggard leg is a footnote by comparison. `MIRAGE`."
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further — the synthetic positive control\n\n"
            "Is the engine a faithful detector, or does it always print noise? Plant a nowcast of "
            "increasing strength `ic` and watch the mean IC-*t* rise from ≈ 0 at the null past 2 — "
            "averaged over 25 seeds so no single lucky seed can fake it."
        ),
        code(
            "ics_plant = [0.0, 0.02, 0.03, 0.05, 0.08, 0.10]\n"
            "ts = [st.synthetic_mean_t(data, ic=a, n_seeds=25) for a in ics_plant]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(ics_plant, ts, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1, label='t = 2 bar')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.axvline(0.03, ls=':', c=AMBER, lw=1.5, label='headline ic = 0.03')\n"
            "ax.set_xlabel('planted ic (true velocity->return correlation)')\n"
            "ax.set_ylabel('mean IC-t (25 seeds)')\n"
            "ax.set_title('Faithful detector: flat at null, clears t=2 for a strong nowcast')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for a, t in zip(ics_plant, ts): print(f'ic {a:.2f} -> mean IC-t {t:+.2f}')"
        ),
        md(
            f"The mean IC-*t* is ≈ 0 at the null (**{R['null_ic_t']}**) and clears 2 only for a "
            "*strong* nowcast (`ic ≥ 0.10`). At the **realistic** headline `ic = 0.03` it sits at "
            f"~1.1 over 25 seeds — exactly the `WEAK` zone. So the engine is faithful: the headline is "
            "a statement about a *realistic-strength alt-data signal on a finite, un-certifiable "
            "tape*, not a broken detector. For the audited-innovation cousin, see "
            "[400 Patent-Intensity](../../400-patent-intensity/)."
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
