"""Generate the two narrative notebooks for Study 567 (Uncertainty-Word-Count).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). This study is
**synthetic-only by design** — there is no free real feed of scored filings joined to forward vol —
so every cell runs on the deterministic, seeded synthetic panel, offline; there is no real-tape
banner and no fabricated tape. The dict ``R`` mirrors the frozen headline numbers in
docs/results.md so the prose matches the recomputed values.
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


# Frozen synthetic headline numbers — mirror of docs/results.md (as-of 2026-06-30; seed 567;
# uncert_beta 0.020, ret_beta -0.020; 480 firm-events; panel fp fa07e8ea11c3).
R = dict(
    fp_panel="fa07e8ea11c3", fp_knob="dc53b20dee4a",
    n=480, n_firms=40, n_periods=12, k=144,
    naive_slope=13.00, naive_t=13.41,
    ctrl_slope=6.13, ctrl_t=3.30, trail_slope=0.254,
    placebo_p=0.0005,
    ret_slope=-9.58, ret_t=-6.92,
    plain_mean=2.92, hedgy_mean=1.13, spread=1.79, ls_t=2.46,
    gross_ann=7.15, net_ann=5.85, cost_bps=5.0, borrow_bps=50.0,
    # robustness sub-panels: (label, controlled uncert slope, slope-t)
    win=[("P01-P03", -1.72, -0.46), ("P04-P06", 11.10, 2.89),
         ("P07-P09", 4.39, 1.20), ("P10-P12", 9.95, 2.73)],
    # synthetic control: (uncert_beta, mean controlled slope-t over 25 seeds)
    ctrl_ladder=[(0.0, -0.26), (0.010, 1.94), (0.020, 4.13), (0.030, 6.28), (0.045, 9.41)],
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

from uncertainty_word_count import data, strategy as st

# This study is SYNTHETIC-ONLY: no free feed of scored filings joined to forward vol exists, so
# fetch_panel() is a documented stub returning an empty frame. Every figure below runs on the
# deterministic, seeded synthetic panel -- offline, reproducible, and NEVER labelled as a real tape.
REAL = data.fetch_panel(fetch=False)
HAVE_REAL = not REAL.empty
print("real tape present:", HAVE_REAL, "(synthetic-only study -- expected False)")

PANEL, TRUTH = data.synthetic_panel()          # seed 567, planted uncert_beta=0.020
print("synthetic panel:", len(PANEL), "firm-events,", PANEL['ticker'].nunique(),
      "firms x", PANEL['period'].nunique(), "periods | fp", data.fingerprint(PANEL))
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Uncertainty Words — do hedgy filings mean a volatility spike? 🤔\n"
            "### Counting how often bosses say 'uncertain', 'maybe', 'could' — and what happens next\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n\n"
            "When a company files its annual report or holds an earnings call, someone can *count the "
            "hedging words* — 'uncertain', 'maybe', 'could', 'risk', 'approximately', 'depend'. A "
            "famous 2011 study (Loughran & McDonald) built the finance dictionary for exactly this and "
            "found that filings stuffed with uncertainty words are followed by **more volatility** — "
            "and, more weakly, **lower returns**. The idea: hedgy language flags that even management "
            "isn't sure, and that ambiguity shows up as a jumpier stock.\n\n"
            "Sounds clean. But there's a trap, and this notebook is really about the trap.\n\n"
            "> 📓 **This is the plain-language layer.** Want the *t*-stats, the placebo test and the "
            "cost maths? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice**, and note: **there is no free data feed** of scored filings "
            "joined to future volatility, so everything below runs on a *simulated* world where we "
            "*plant* the effect on purpose — to show how the machinery behaves and where it can fool "
            "you. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ----
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Do hedgy filings *look* like they predict volatility? | **Yes, hugely** — the raw link "
            f"is enormous (a *t*-stat of **{R['naive_t']}**). |\n"
            "| Is that link real? | **Mostly not.** The firms that write hedgy filings are the firms "
            f"that were *already jumpy*. Once you account for that, the link shrinks from *t* "
            f"**{R['naive_t']}** to *t* **{R['ctrl_t']}**. Two-thirds of it was an illusion. |\n"
            "| So is there *any* real edge? | **In this simulation, a small one** — but we can't check "
            "it on real data, because no free feed of scored filings + future volatility exists. |\n"
            "| Could you trade it? | **Barely.** The signal only comes ~4×/year and its strength "
            "swings a lot from quarter to quarter. |\n\n"
            "> The lesson isn't 'uncertainty words are useless' — it's that the *obvious* version of "
            "the signal is mostly a **confound**: uncertain companies were already volatile. You have "
            "to subtract that out before you believe the words."
        ),

        # ---- BEAT 1 — THE CLAIM ----
        md(
            "## 1 · The claim\n\n"
            "> *\"The more uncertainty / hedging words a firm uses, the higher its future volatility "
            "and the lower its future returns.\"* — Loughran & McDonald (2011); Campbell et al. (2014); "
            "Kravet & Muslu (2013).\n\n"
            "The intuition: language leaks information. If management litters a filing with 'may', "
            "'could', 'risk', 'uncertain', they're signalling — maybe unconsciously — that the future "
            "is foggy. Foggy futures resolve as *volatility*. And nervous investors might demand a "
            "discount, dragging returns down. It's one of the founding results of finance text mining."
        ),

        # ---- BEAT 2 — SO WHAT ----
        md(
            "## 2 · So what?\n\n"
            "If it's real and clean, you could read a 10-K, count the hedges, and *forecast the stock's "
            "risk* before the market fully prices it — useful for options, position sizing, and a "
            "long-plain / short-hedgy trade. The desk has looked at the *sentiment* side of text before "
            "([566 earnings-call-tone](../../566-earnings-call-tone/) scores upbeat-vs-guarded tone "
            "against **drift**); here we go after a *different* dictionary — the **uncertainty** list — "
            "and a *different* outcome — **volatility**."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ----
        md(
            "## 3 · How would we even know?\n\n"
            "1. **Score each filing.** Count the share of words that are hedges (the uncertainty "
            "density).\n"
            "2. **Wait.** Measure the stock's volatility over the *next* window (never peeking — the "
            "score is known at the filing, the volatility comes after).\n"
            "3. **Relate them.** Do hedgier filings come before jumpier stocks?\n\n"
            "**The catch — and the whole point.** Companies that are *already* volatile tend to write "
            "hedgier filings (they have more to hedge about), and volatility is *sticky* — a jumpy "
            "stock stays jumpy. So hedgy-filing → future-vol could be almost entirely 'jumpy stocks "
            "stay jumpy', with the words adding nothing. We *must* subtract out how jumpy each firm "
            "already was before we credit the words.\n\n"
            "> Because no free feed of scored filings exists, we build a **simulated** world where we "
            "control the truth: we plant a real (but modest) words→vol link *on top of* the "
            "'jumpy-stays-jumpy' stickiness, then see whether our test can tell them apart."
        ),

        # ---- BEAT 4 — THE TEARDOWN ----
        md(
            "## 4 · The teardown — the confound in one picture\n\n"
            "Here's the trap, drawn. The **naive** number ignores how jumpy each firm already was; the "
            "**honest** number subtracts it out first."
        ),
        code(
            "naive = st.uncertainty_vol_regression(PANEL, control_trail_vol=False)\n"
            "ctrl  = st.uncertainty_vol_regression(PANEL, control_trail_vol=True)\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.4))\n"
            "ax.bar(['NAIVE\\n(ignore prior vol)', 'HONEST\\n(control prior vol)'],\n"
            "       [naive['uncert_t'], ctrl['uncert_t']], color=[RED, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1, label='t = 2 bar')\n"
            "ax.set_ylabel('uncertainty -> future-vol strength (t-stat)')\n"
            "ax.set_title(f\"Most of the 'signal' is a mirage: t {naive['uncert_t']:.1f} -> {ctrl['uncert_t']:.1f}\")\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"naive t {naive['uncert_t']:.2f}  ->  honest (controlled) t {ctrl['uncert_t']:.2f}\")"
        ),
        md(
            f"There it is. The raw link between hedgy filings and future volatility is a monster (*t* "
            f"**{R['naive_t']}**). But once we subtract out **how volatile each firm already was**, "
            f"it collapses to *t* **{R['ctrl_t']}**. About two-thirds of the apparent signal was just "
            "*jumpy stocks stay jumpy* — nothing to do with the words. The surviving piece (the "
            "planted textual edge) is real *in this simulation*, but it's a fraction of what the naive "
            "number screamed."
        ),
        md(
            "**Does the surviving edge hold up quarter to quarter?** Split the periods into four "
            "chunks and re-measure the honest link in each:"
        ),
        code(
            "rows = st.robustness_windows(PANEL, 4)\n"
            "labs = [r[0] for r in rows]; ts = [r[2] for r in rows]\n"
            "fig, ax = plt.subplots(figsize=(8.3, 4.3))\n"
            "cols = [GREEN if t >= 2 else (AMBER if t > 0 else RED) for t in ts]\n"
            "ax.bar(labs, ts, color=cols, width=.55)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('honest words->vol strength (t)')\n"
            "ax.set_title('Even with the effect PLANTED, the per-quarter strength wanders')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('per-window t:', [round(t,2) for t in ts])"
        ),
        md(
            "So even in a world where we *know* the effect is there, any single quarter's worth of "
            "filings is noisy — the strength swings from below zero to well past the bar. A real, "
            "once-a-quarter text signal would be at least this fragile. That's why the tradability is "
            "**Fragile**, not solid."
        ),

        # ---- BEAT 5 — VERDICT ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Weak.** The literature backs the claim, and our simulated honest link "
            f"clears the bar (*t* {R['ctrl_t']}) — but **there is no free real data** to certify it, "
            "so it can't be `REAL`. And most of the *raw* signal is a confound.\n"
            "- **Tradability — Fragile.** A ~4×/year signal whose strength swings quarter to quarter, "
            "unverified on any real tape."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ----
        md(
            "## 6 · Could you actually trade it?\n\n"
            "The tradable version is *long the plain-spoken firms, short the hedgy ones* (betting the "
            "hedgy ones underperform). In our simulation that book makes money — but it's a thin, "
            "quarterly signal, the hedgy names you'd short are the ones that are expensive to borrow, "
            "and (most important) **none of it is checked against real filings**. Treat it as a lesson "
            "about confounds, not a strategy."
        ),

        # ---- BEAT 7 — GOING FURTHER ----
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The sentiment cousin.** [566 earnings-call-tone](../../566-earnings-call-tone/) scores "
            "*upbeat vs guarded* tone against post-call drift — a different dictionary, a different "
            "outcome.\n"
            "- **The real test needs real filings.** Parse EDGAR full-text against the LM Uncertainty "
            "list, join to forward realised vol, and re-run the *controlled* regression — that's the "
            "only way to move this off `WEAK`.\n\n"
            "*Have a scored-filing panel? Drop it into `_cache/uncertainty_panel.parquet` in the study "
            "schema and the same engine will read it — no code changes.*"
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
            "# Uncertainty-Word-Count — a quantitative teardown 🔬\n"
            "### LM uncertainty density · naive vs trailing-vol-controlled forward-vol slope · label-shuffle placebo · return-drag · long-short with costs & borrow · four-window wander · seed-robust synthetic control\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We score a firm's LM "
            "**Uncertainty** density and test whether it forecasts **forward realised volatility** "
            "(primary) and **forward return** (secondary), with the central honesty check being the "
            "confound by trailing volatility.\n\n"
            "> ⚠️ **Not investment advice, and synthetic-only by design.** There is no free feed of "
            "scored filings joined to forward vol, so a `REAL` stamp is out of reach (capped at "
            f"`WEAK`) and every cell runs on the deterministic seeded panel (seed 567, panel fp "
            f"`{R['fp_panel']}`) — never a real-tape banner. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | No free real tape (→ capped at `WEAK`). Synthetic: naive "
            f"forward-vol slope-*t* **{R['naive_t']}** collapses to **{R['ctrl_t']}** once trailing "
            f"vol is controlled (placebo *p* {R['placebo_p']}). |\n"
            f"| **Tradability** | `FRAGILE` | Long-plain/short-hedgy spread **+{R['spread']}%/event** "
            f"(*t* {R['ls_t']}), **+{R['net_ann']}%/yr** net — but per-window slope-*t* wanders "
            "(−0.46 → +2.89 → +1.20 → +2.73), ~4 events/yr, unverified live. |\n\n"
            "> 💡 In plain words: the engine works (the synthetic control proves it), the confound is "
            "the danger (two-thirds of the naive signal is vol persistence), and with no real tape the "
            "edge can only be *demonstrated*, not *certified*."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $U_i$ be firm-event $i$'s uncertainty-word density, $V^{f}_i$ its forward realised "
            "vol, $V^{t}_i$ its trailing realised vol, and $r^{f}_i$ its forward return.\n\n"
            "- **H₁ (vol, naive).** slope of $V^f$ on $U$ is $> 0$ at *t* ≥ 2.\n"
            "- **H₁′ (vol, honest).** slope of $V^f$ on $U$ **controlling for $V^t$** is $> 0$ at *t* "
            "≥ 2 — the *residual* words, not vol persistence. **This is the real hypothesis.**\n"
            "- **H₂ (return drag).** slope of $r^f$ on $U$ is $< 0$.\n"
            "- **H₃ (robustness).** the sign/strength of H₁′ is stable across sub-panels.\n"
            "- **H₄ (real tape).** any of the above holds on **real** scored filings.\n\n"
            "We find **H₁ trivially true but misleading**, **H₁′ true in-sample but much weaker**, "
            "**H₂ true in-sample**, **H₃ shaky** (per-window *t* wanders), and **H₄ untestable** — no "
            "free feed. Hence `WEAK`."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The content is the **confound**. Realised vol is one of the most autocorrelated series in "
            "finance, and hedgy language is *itself* a function of how volatile a firm has been. So the "
            "naive $V^f \\sim U$ slope is a textbook omitted-variable trap: it loads the entire "
            "vol-persistence channel onto the words. The Frisch-Waugh move — partial $V^t$ out first — "
            "is the difference between a spurious blockbuster and a modest real edge. This is the same "
            "discipline the tone study ([566](../../566-earnings-call-tone/)) applies to the *numeric "
            "surprise*; here the confounder is *prior volatility*."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Score.** $U$ = LM uncertainty-word density per filing (here: generated, not parsed — "
            "synthetic-only).\n"
            "- **Vol regression.** OLS of $V^f$ on $U$ (naive) and on $[U, V^t]$ (honest); the $U$ "
            "coefficient's *t* in the *controlled* fit is the headline.\n"
            "- **Placebo.** Shuffle $U$ against $(V^f, V^t)$ 2000× and refit the controlled slope — "
            "read the tail.\n"
            "- **Return drag.** OLS of $r^f$ on $[U, V^t]$; a negative $U$ slope is the drag.\n"
            "- **Long-short.** Tercile sort on $U$: long plain / short hedgy on $r^f$, with costs + "
            "borrow.\n"
            "- **Robustness.** Four contiguous sub-panels; read the controlled slope-*t* in each.\n"
            "- **Positive control.** A deterministic synthetic world, effect planted (`uncert_beta > "
            "0`) vs null, averaged over 25 seeds (house rule).\n\n"
            "Timing: $U$ is known *at* the filing; $V^f$/$r^f$ are measured strictly *after*; the "
            "position enters the next session. No future data enters the score — one documented "
            "execution lag, applied identically in the sweep and the control."
        ),

        # ---- BEAT 4 ----
        md("## 4 · The teardown"),
        md(
            "### 4a · The vol regression — H₁ vs H₁′ (the confound)\n\n"
            "The naive slope, then the trailing-vol-controlled slope, with the placebo null on the "
            "controlled one."
        ),
        code(
            "naive = st.uncertainty_vol_regression(PANEL, control_trail_vol=False)\n"
            "ctrl  = st.uncertainty_vol_regression(PANEL, control_trail_vol=True)\n"
            "p = st.placebo_pvalue(PANEL, n_perm=2000, seed=567)\n"
            "fig, ax = plt.subplots(figsize=(8, 4.3))\n"
            "ax.bar(['naive slope', 'controlled slope'], [naive['uncert_slope'], ctrl['uncert_slope']],\n"
            "       color=[RED, GREEN], width=.5)\n"
            "ax.set_ylabel('uncertainty -> forward-vol slope')\n"
            "ax.set_title(f\"naive t {naive['uncert_t']:.2f}  ->  controlled t {ctrl['uncert_t']:.2f}  (placebo p {p:.4f})\")\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"NAIVE slope {naive['uncert_slope']:.2f} (t {naive['uncert_t']:.2f})\")\n"
            "print(f\"CTRL  slope {ctrl['uncert_slope']:.2f} (t {ctrl['uncert_t']:.2f}), trail-vol slope {ctrl['trail_slope']:.3f}\")\n"
            "print(f\"placebo p (controlled) {p:.4f}\")"
        ),
        md(
            f"> 💡 In plain words: **H₁ is trivially true and useless** — the naive slope-*t* is "
            f"**{R['naive_t']}**, but it's carrying the whole vol-persistence channel (trailing-vol "
            f"slope +{R['trail_slope']}). **H₁′ survives but shrinks** — the controlled slope-*t* is "
            f"**{R['ctrl_t']}**, and the placebo *p* = {R['placebo_p']} confirms the residual isn't "
            "manufactured by the control. Two-thirds of the raw signal was the confound."
        ),
        md(
            "### 4b · The return drag — H₂\n\n"
            "Regress forward return on uncertainty (controlling for trailing vol). A *negative* slope "
            "is the 'lower returns' half."
        ),
        code(
            "reg = st.uncertainty_return_regression(PANEL, control_trail_vol=True)\n"
            "u = PANEL['uncert'].to_numpy(); y = PANEL['fwd_ret'].to_numpy()*100\n"
            "fig, ax = plt.subplots(figsize=(8, 4.5))\n"
            "ax.scatter(u, y, s=14, color=GREY, alpha=.5)\n"
            "b, a = np.polyfit(u, y, 1)\n"
            "xs = np.linspace(u.min(), u.max(), 50); ax.plot(xs, b*xs+a, c=RED, lw=2)\n"
            "ax.set_xlabel('uncertainty-word density'); ax.set_ylabel('forward return %')\n"
            "ax.set_title(f\"return drag: slope-t {reg['uncert_t']:.2f} (negative = lower returns)\")\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"return slope {reg['uncert_slope']:.2f}/unit, slope-t {reg['uncert_t']:.2f}\")"
        ),
        md(
            f"> 💡 In plain words: **H₂ holds in-sample** — the controlled return slope is "
            f"**{R['ret_slope']}** per density unit (*t* **{R['ret_t']}**): hedgier text goes with "
            "*lower* forward returns, by construction in this planted world."
        ),
        md(
            "### 4c · Robustness — H₃ (does the strength hold?)\n\n"
            "The controlled slope-*t* across four contiguous sub-panels."
        ),
        code(
            "rows = st.robustness_windows(PANEL, 4)\n"
            "tab = pd.DataFrame(rows, columns=['window','ctrl_slope','ctrl_t']).set_index('window')\n"
            "fig, ax = plt.subplots(figsize=(8.3, 4.3))\n"
            "cols = [GREEN if t>=2 else (AMBER if t>0 else RED) for t in tab['ctrl_t']]\n"
            "ax.bar(range(len(tab)), tab['ctrl_t'], color=cols, width=.55)\n"
            "ax.set_xticks(range(len(tab))); ax.set_xticklabels(tab.index)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('controlled uncert slope-t'); ax.set_title('Per-window strength wanders even with the effect planted')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(tab.round(2).to_string())"
        ),
        md(
            "> 💡 In plain words: **H₃ is shaky.** Even planted, the per-window slope-*t* runs −0.46, "
            "+2.89, +1.20, +2.73 — below the bar in two of four 120-event chunks. The pooled 480-event "
            "*t* is solid; a single quarter is not. A real quarterly text signal would be at least this "
            "fragile → `FRAGILE`."
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — no free real tape (→ capped at `WEAK`); synthetically, H₁′ clears "
            f"the bar (controlled *t* {R['ctrl_t']}, placebo {R['placebo_p']}) but H₁ is mostly "
            "confound.\n"
            f"- **Tradability `FRAGILE`** — long-short spread +{R['spread']}%/event (*t* {R['ls_t']}), "
            f"net +{R['net_ann']}%/yr, but sign-strength wanders across windows and it's ~4 events/yr, "
            "unverified live."
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you trade it? — the long-short, costs and borrow\n\n"
            "Long the plain tercile, short the hedgy tercile on forward return; annualise (~4 "
            "filings/yr) and net costs + a hedgy-leg borrow."
        ),
        code(
            "ls = st.uncertainty_long_short(PANEL, 0.3)\n"
            "gross_ann = ls['spread']*4*100\n"
            "net_ann = st.net_spread(ls, turns_per_year=4.0, cost_bps=5.0, borrow_ann_bps=50.0)*100\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.3))\n"
            "ax.bar(['plain\\ntercile','hedgy\\ntercile'], [ls['plain_mean']*100, ls['hedgy_mean']*100],\n"
            "       color=[GREEN, RED], width=.45)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('mean forward return %')\n"
            "ax.set_title(f\"spread {ls['spread']*100:+.2f}%/event (t {ls['t']:.2f}) -> net {net_ann:+.1f}%/yr\")\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"plain {ls['plain_mean']*100:+.2f}% | hedgy {ls['hedgy_mean']*100:+.2f}% | spread {ls['spread']*100:+.2f}% (t {ls['t']:.2f})\")\n"
            "print(f\"gross {gross_ann:+.2f}%/yr -> net {net_ann:+.2f}%/yr (5 bps/leg x 4 turns + 50 bps borrow)\")"
        ),
        md(
            f"> 💡 In plain words: the book earns **+{R['spread']}%/event** (*t* {R['ls_t']}), "
            f"**+{R['net_ann']}%/yr** net — profitable *in the simulation*. But it's thin (quarterly), "
            "sign-unstable, and — the point — **not verified on real filings**. `FRAGILE`."
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further — the synthetic positive control\n\n"
            "Is the engine a faithful detector, or does it always print a signal? Plant an "
            "uncertainty→vol link of increasing strength (`uncert_beta`) and watch the controlled "
            "slope-*t* rise from ≈ 0 at the null — averaged over 25 seeds so no single lucky seed can "
            "fake it."
        ),
        code(
            "betas = [0.0, 0.005, 0.010, 0.015, 0.020, 0.030, 0.045]\n"
            "ts = [st.synthetic_mean_t(data, uncert_beta=b, n_seeds=25) for b in betas]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(betas, ts, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1, label='t = 2 bar')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted uncert_beta (stronger words->vol link)')\n"
            "ax.set_ylabel('mean controlled slope-t (25 seeds)')\n"
            "ax.set_title('Flat at the null, monotone past t=2 as the effect grows')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for b, t in zip(betas, ts): print(f'uncert_beta {b:+.3f} -> mean controlled slope-t {t:+.2f}')"
        ),
        md(
            f"The controlled slope-*t* is ≈ 0 at the null (**{R['ctrl_ladder'][0][1]}**) and climbs "
            "monotonically past +2 as the planted link grows — so the engine is a faithful detector "
            "that *does not* print a false positive at the null. That's exactly why the real-tape gap "
            "matters: the machinery is ready, but with **no free feed of scored filings** the "
            "uncertainty-word signal stays `WEAK` on this desk. For the sentiment cousin, see "
            "[566 earnings-call-tone](../../566-earnings-call-tone/)."
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
