"""Generate the two narrative notebooks for Study 581 (Term-Premium).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic figures run
anywhere, offline and deterministic; the real-tape cells use the cached tape under ../_cache/ if
present and otherwise quote the frozen headline numbers in ``R`` (mirroring docs/results.md), so the
notebook re-runs for any reader offline.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-30; tape
# 2002-07-31 -> 2026-06-26; 6,009 days; tape fp 83b8f0823f11, tp-series fp c0a43303a561).
R = dict(
    fp_tape="83b8f0823f11", fp_tp="c0a43303a561",
    n=6009, start="2002-07-31", end="2026-06-26",
    # headline 21-day quintile sort (percent)
    q1=0.17, q5=0.43, spread=0.26, hac_t=0.43, placebo_p=0.719, n_used=5925, n_q=1182,
    # horizon sweep: (label, spread%, HAC t)
    horiz=[("5d", 0.00, 0.02), ("21d", 0.26, 0.43), ("63d", 1.22, 0.73), ("126d", 2.37, 0.70)],
    # sub-period sweep: (label, spread%, HAC t)
    sub=[("2002-2009", 2.22, 2.09), ("2010-2016", 0.30, 0.34),
         ("2017-2021", -0.85, -0.82), ("2022-2026", 0.66, 0.43)],
    # timing overlay
    timer_sharpe=0.291, bh_sharpe=0.247, timer_sharpe5=0.261,
    switches_yr=9.6, inv_frac=0.429, ov_spread_bps=-0.32, ov_spread_t=-0.39, cost_bps=2.0,
    # synthetic control (25 seeds, 21d): (tp_signal, mean spread%, mean HAC t)
    ctrl=[(0.000, 0.11, 0.20), (0.002, 2.58, 4.76), (0.004, 5.06, 8.88),
          (0.008, 10.01, 15.08), (0.012, 14.97, 18.77)],
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

from term_premium import data, strategy as st

def load_real():
    \"\"\"Cache-first real daily tape (empty frame offline).\"\"\"
    try:
        return data.fetch_daily(fetch=False)
    except FileNotFoundError:
        return pd.DataFrame()

DF = load_real()
HAVE_REAL = not DF.empty
print("real term-premium tape present:", HAVE_REAL,
      "" if not HAVE_REAL else f"({len(DF)} days, {DF.index[0].date()} -> {DF.index[-1].date()})")
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# The Term Premium — does 'extra yield for holding long bonds' tell you *when* to own them? 🏦\n"
            "### Stripping a long-bond yield down to its risk-reward, in plain English\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Term--premium_times_duration%3F: Mixed](https://img.shields.io/badge/Term--premium_times_duration%3F-Mixed-8b949e?style=flat-square)\n\n"
            "A 10-year Treasury yield is really two things added together. One part is just *where "
            "everyone thinks short-term interest rates are heading* over the next decade. The other "
            "part is a bonus — the **term premium** — extra yield you demand for the *risk* of locking "
            "your money up for ten years instead of rolling short bills. Economists (the famous "
            "Adrian-Crump-Moench, or 'ACM', model) estimate that bonus and publish it.\n\n"
            "The idea we test: when that bonus is **fat**, long bonds are generously paid for their "
            "risk, so you should *own duration* (buy the long-bond ETF, TLT); when it's **thin**, step "
            "aside. In short — can the term premium tell you *when* to hold long bonds?\n\n"
            "> 📓 **This is the plain-language layer.** Want the *t*-stats, the placebo test and the "
            "cost maths? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool: every chart is drawn by the "
            "code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ----
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Do fat-premium days precede higher long-bond returns? | **Yes — but only just.** Over "
            f"the next month, the fattest-premium days earned **+{R['q5']}%** vs **+{R['q1']}%** for the "
            f"thinnest — the *right* direction, a **+{R['spread']}%** gap. |\n"
            f"| Is that gap solid? | **No.** A shuffle test says a gap that size shows up "
            f"**{int(R['placebo_p']*100)}%** of the time by pure chance. The signal is real-*ish* but "
            "far too faint to bank. |\n"
            f"| Does it always work? | **No — it depends on the decade.** It worked in 2002-2009, "
            "faded in 2010-2016, and went *backwards* in 2017-2021. |\n"
            "| Could you trade it? | **No.** A timer that owns bonds on fat-premium days barely ties "
            "just-buying-and-holding, while flipping in and out ~10 times a year. |\n\n"
            "> The term premium points the right way — the theory isn't wrong — but this cheap "
            "estimate of it is a whisper, not a shout, and the whisper changes its mind by decade."
        ),

        # ---- BEAT 1 — THE CLAIM ----
        md(
            "## 1 · The claim\n\n"
            "> *\"The term premium — the compensation for holding long bonds — varies over time, and "
            "when it's high, long bonds subsequently earn more.\"* — Fama & Bliss (1987); "
            "Cochrane & Piazzesi (2005); Adrian, Crump & Moench (2013).\n\n"
            "The intuition: a long bond's yield = (expected average short rate) + (term premium). The "
            "first part is just a forecast of Fed policy — no free lunch there. The *second* part is a "
            "risk reward, and decades of research say it **moves around** and is **predictable**. When "
            "it's fat, you're being paid well to bear duration risk; that's when to own it."
        ),

        # ---- BEAT 2 — SO WHAT ----
        md(
            "## 2 · So what?\n\n"
            "If a term-premium estimate really timed long bonds, you'd have a clean rates signal: own "
            "TLT when the premium is fat, hold cash when it's thin — a bond-market market-timer. It "
            "would also validate the whole ACM/Cochrane-Piazzesi machinery on live retail data. The "
            "desk already looked at the raw curve *slope* ([132 Yield-Curve-Steepener](../../132-yield-curve-steepener/)); "
            "here we go one step deeper and strip out the *expectations* part to isolate the **premium** "
            "itself."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ----
        md(
            "## 3 · How would we even know?\n\n"
            "1. **Estimate the premium.** We can't reach the official ACM series with a free data feed, "
            "so we build the cheapest honest stand-in: the 10-year yield **minus** a slow moving "
            "average of the 3-month bill (a proxy for 'expected short rates'). What's left is the "
            "premium.\n"
            "2. **Rank it.** Each day, how fat is today's premium versus its own last year? (0 = "
            "thinnest, 1 = fattest.)\n"
            "3. **Sort forward returns.** Split days into five buckets by premium and compare how TLT "
            "did over the *next month*. The claim says the fattest bucket wins.\n\n"
            "*Timing:* the premium at today's close forms the signal; we act at *tomorrow's* close (a "
            "one-day lag) — no peeking at the future."
        ),

        # ---- BEAT 4 — THE TEARDOWN ----
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**Did the fattest-premium days beat the thinnest over the next month?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    qs = st.quintile_spread(DF, horizon=21)\n"
            "    q1, q5, spr, t = qs['q1']*100, qs['q5']*100, qs['spread']*100, qs['t']\n"
            "    banner = 'REAL tape (TLT + 10y yield + 3m bill, 2002-2026)'\n"
            "else:\n"
            f"    q1, q5, spr, t = {R['q1']}, {R['q5']}, {R['spread']}, {R['hac_t']}\n"
            "    banner = 'frozen real numbers (offline)'\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.4))\n"
            "ax.bar(['THIN premium\\n(Q1)','FAT premium\\n(Q5)'], [q1, q5], color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('forward 21-day TLT return %')\n"
            "ax.set_title(f'Fat premium did earn more (+{q5:.2f}% vs +{q1:.2f}%) — but the gap is tiny')\n"
            "fig.suptitle(banner, fontsize=9, color=GREY); plt.tight_layout(); plt.show()\n"
            "print(f'Thin +{q1:.2f}%  vs  Fat +{q5:.2f}%  ->  spread {spr:+.2f}% (HAC t {t:+.2f})')"
        ),
        md(
            f"So the direction is **right**: the fat-premium days earned **+{R['q5']}%** over the next "
            f"month vs **+{R['q1']}%** for the thin-premium days, a **+{R['spread']}%** edge. But look at "
            f"the size — a quarter of a percent, with a *t*-stat of just **+{R['hac_t']}** (you want ≥ 2). "
            "That's a whisper."
        ),
        md(
            "**Does the whisper hold across the decades?** Run the same sort within four separate "
            "sub-periods:"
        ),
        code(
            "sub = " + repr([(s[0], s[1]) for s in R["sub"]]) + "\n"
            "if HAVE_REAL:\n"
            "    edges = [('2002-2009','2002-07-01','2009-12-31'),('2010-2016','2010-01-01','2016-12-31'),\n"
            "             ('2017-2021','2017-01-01','2021-12-31'),('2022-2026','2022-01-01','2026-06-26')]\n"
            "    tab = st.subperiod_sweep(DF, edges, horizon=21)\n"
            "    labs = list(tab.index); spr = list(tab['spread']*100)\n"
            "else:\n"
            "    labs = [s[0] for s in sub]; spr = [s[1] for s in sub]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "cols = [GREEN if s>0 else RED for s in spr]\n"
            "ax.bar(labs, spr, color=cols, width=.55)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('fat - thin spread %')\n"
            "ax.set_title('It worked early (green), faded, then went BACKWARDS in 2017-21 (red)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('spreads by period:', [round(s,2) for s in spr])"
        ),
        md(
            "There's the catch. The premium timed bonds nicely in **2002-2009** (that's the only window "
            "that clears the bar), faded to nothing in **2010-2016**, and then *inverted* in "
            "**2017-2021** — the zero-rates era, when the premium barely moved and carried little "
            "information. A signal that changes its mind by decade isn't something you can bet the "
            "house on."
        ),

        # ---- BEAT 5 — VERDICT ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Weak.** Right direction (+{R['spread']}% fat-minus-thin) but a faint HAC *t* "
            f"of +{R['hac_t']}, and the sign flips across decades. The theory is solid; this cheap "
            "estimate on this tape can't certify it.\n"
            "- **Tradability — Mirage.** A timer on the signal barely ties buy-and-hold while churning "
            "~10 times a year (next notebook).\n"
            "- **Term-premium times duration? — Mixed.** Yes in 2002-09, no after."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ----
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Barely, and not worth it. A timer that owns TLT only on fat-premium days sits in cash "
            "more than half the time — so it looks *smoother* (a slightly better Sharpe) but doesn't "
            "actually earn more; on the honest measure (average return) it *loses* to plain "
            "buy-and-hold, after flipping in and out ~10 times a year. The next notebook does the cost "
            "maths."
        ),

        # ---- BEAT 7 — GOING FURTHER ----
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The raw-slope cousin.** [132 Yield-Curve-Steepener](../../132-yield-curve-steepener/) "
            "uses the plain 10y−3m slope; this study strips out the expectations part to isolate the "
            "*premium*.\n"
            "- **The real ACM series.** Re-run this with the actual NY-Fed ACM term premium (a CSV, not "
            "a free ticker) and a full cycle — that's the proper test.\n\n"
            "*Think the premium really times bonds and we just used too crude an estimate? Fork this, "
            "drop in the ACM series, and show the fat-minus-thin spread clearing t = 2 out of sample.*"
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
            "# The Term Premium — a quantitative teardown 🔬\n"
            "### ACM-style premium proxy · quintile sort · HAC *t* · block-shuffle placebo · horizon & sub-period sweeps · timing-overlay costs · synthetic control\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Term--premium_times_duration%3F: Mixed](https://img.shields.io/badge/Term--premium_times_duration%3F-Mixed-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We build an ACM-style "
            "term-premium proxy (10y yield minus an EWMA of the short rate), rank it out-of-sample, and "
            "test whether the fattest-premium days precede higher forward TLT returns.\n\n"
            "> ⚠️ **Not investment advice.** Real data: yfinance daily TLT + `^TNX` + `^IRX`, "
            f"{R['n']:,} days (2002-07-31 → 2026-06-26, tape fp `{R['fp_tape']}`); the offline core and "
            "tests run on a deterministic synthetic world. Methods in "
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
            f"| **Signal** | `WEAK` | Q5−Q1 forward-TLT spread **+{R['spread']}%** (21d), HAC *t* "
            f"**+{R['hac_t']}**, placebo *p* {R['placebo_p']} — right sign, sub-2 *t*; sign flips across "
            "sub-periods (real only 2002-09, *t* +2.09). ACM-*proxy* ⇒ `REAL` unreachable in principle. |\n"
            f"| **Tradability** | `MIRAGE` | Timer Sharpe **{R['timer_sharpe']}** vs buy-and-hold "
            f"**{R['bh_sharpe']}** — a cash-drag artefact; mean-return spread **{R['ov_spread_bps']} "
            f"bps/day** (*t* {R['ov_spread_t']}) at {R['switches_yr']} switches/yr. |\n"
            f"| **Term-premium times duration?** | `MIXED` | Present 2002-09 (*t* +2.09); faded 2010-16; "
            "inverted 2017-21 (*t* −0.82). |\n\n"
            "> 💡 In plain words: the engine works (the synthetic control proves it), but on this "
            "ACM-proxy tape the timing edge is right-signed, faint, and regime-dependent."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $\\text{TP}_t$ be the term-premium proxy and $r_{t\\to t+h}$ the forward TLT return.\n\n"
            "- **H₁ (the sort).** $\\mathbb{E}[r \\mid \\text{TP high (Q5)}] - "
            "\\mathbb{E}[r \\mid \\text{TP low (Q1)}] > 0$ at HAC *t* ≥ 2.\n"
            "- **H₂ (robustness).** The sign of the Q5−Q1 spread is stable across horizons and "
            "sub-periods.\n"
            "- **H₃ (tradable).** A premium-conditioned timing overlay beats buy-and-hold TLT net of "
            "costs, on mean return (not just Sharpe).\n\n"
            "We find **H₁ directionally right but not significant** (spread positive, HAC *t* "
            f"+{R['hac_t']}), **H₂ rejected** (sign flips across sub-periods), and **H₃ rejected** (the "
            "timer's Sharpe edge is a cash-drag artefact; its mean-return spread is negative)."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The content is the **decomposition**: a long yield is expected-short-rates + term premium. "
            "The raw curve slope ([132](../../132-yield-curve-steepener/)) mixes the two; subtracting an "
            "EWMA-expectations component isolates the *premium* — the risk-compensation piece that "
            "Fama-Bliss and Cochrane-Piazzesi show is time-varying and predictable. If it times "
            "duration, you have a clean rates signal; if not, the ACM machinery doesn't survive a "
            "no-key proxy on live data."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Signal.** $\\text{TP} = y_{10} - \\text{EWMA}_{252}(\\text{short})$, ranked "
            "out-of-sample over a trailing 252-day window into $\\text{rank}\\in[0,1]$ (higher = "
            "fatter), lagged one day.\n"
            "- **Sort.** Quintiles of the lagged rank; compare Q5 (fattest) vs Q1 (thinnest) forward "
            "TLT return.\n"
            "- **Inference.** HAC (Newey-West) *t* on the day-level Q5−Q1 difference (overlapping "
            "forward returns demand it); a **block-shuffle placebo** null (circular rotation, 21-day "
            "blocks).\n"
            "- **Robustness.** Horizons 5/21/63/126d; four sub-periods.\n"
            "- **Frictions.** A timing overlay (own TLT when rank > 0.5, else cash), one-way cost per "
            "switch × NAV; net Sharpe vs buy-and-hold and the mean-return spread.\n"
            "- **Positive control.** A deterministic synthetic world with a planted timing edge "
            "(`tp_signal > 0`) and a null — averaged over 25 seeds (house rule).\n\n"
            "Timing: the premium at close *t* forms the signal; the position enters at close *t+1*; the "
            "forward return runs from *t+1*. That one-bar lag is the single documented execution "
            "convention — the EWMA and the rolling rank are causal, so no future data enters the signal."
        ),

        # ---- BEAT 4 ----
        md("## 4 · The teardown"),
        md(
            "### 4a · The sort — H₁\n\n"
            "Q5 vs Q1 forward TLT return (21-day), with the HAC *t* and the block-shuffle placebo."
        ),
        code(
            "if HAVE_REAL:\n"
            "    qs = st.quintile_spread(DF, horizon=21)\n"
            "    p = st.placebo_pvalue(DF, horizon=21, n_perm=1000, seed=581)\n"
            "    q1, q5, spr, t = qs['q1']*100, qs['q5']*100, qs['spread']*100, qs['t']\n"
            "else:\n"
            f"    q1, q5, spr, t, p = {R['q1']}, {R['q5']}, {R['spread']}, {R['hac_t']}, {R['placebo_p']}\n"
            "fig, ax = plt.subplots(figsize=(8, 4.3))\n"
            "ax.bar(['Q1 (thin)','Q5 (fat)'], [q1, q5], color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('forward 21-day TLT return %')\n"
            "ax.set_title(f'Q5 - Q1 = {spr:+.2f}%  (HAC t {t:+.2f}, placebo p {p:.3f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Q1 +{q1:.2f}% | Q5 +{q5:.2f}% | spread {spr:+.2f}% | HAC t {t:+.2f} | placebo p {p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: H₁ **directionally right, not significant.** The spread is "
            f"**+{R['spread']}%** (HAC *t* +{R['hac_t']}), and the placebo *p* = {R['placebo_p']} says a "
            "spread this size is well inside the null — the fat premium points the right way but the "
            "edge is buried in noise."
        ),
        md(
            "### 4b · Horizon sweep — does the sign hold as you extend the hold?\n\n"
            "The same Q5−Q1 spread over 5, 21, 63 and 126-day forward horizons."
        ),
        code(
            "horiz = " + repr(R["horiz"]) + "\n"
            "if HAVE_REAL:\n"
            "    tab = st.horizon_sweep(DF, horizons=(5,21,63,126))\n"
            "    labs = list(tab.index); spr = list(tab['spread']*100); ts = list(tab['t'])\n"
            "else:\n"
            "    labs = [h[0] for h in horiz]; spr = [h[1] for h in horiz]; ts = [h[2] for h in horiz]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.bar(labs, spr, color=GREEN, width=.55)\n"
            "for x,(s,tt) in enumerate(zip(spr, ts)):\n"
            "    ax.annotate(f't={tt:+.2f}', (x, s), ha='center', va='bottom', fontsize=9, color=GREY)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('Q5 - Q1 spread %')\n"
            "ax.set_title('Spread grows with horizon, but the HAC t never clears 2')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('horizon spreads %:', [round(s,2) for s in spr], '| HAC t:', [round(x,2) for x in ts])"
        ),
        md(
            "> 💡 In plain words: the spread rises with horizon (a fat premium drifts into higher "
            "long-horizon returns) but the HAC correction deflates the *t* — the longer-horizon 'edge' "
            "is mostly overlap-induced persistence, not fresh information. Right sign, no significance."
        ),
        md(
            "### 4c · Sub-period sweep — H₂ (is the sign stable?)\n\n"
            "The Q5−Q1 spread within four sub-periods."
        ),
        code(
            "sub = " + repr(R["sub"]) + "\n"
            "if HAVE_REAL:\n"
            "    edges = [('2002-2009','2002-07-01','2009-12-31'),('2010-2016','2010-01-01','2016-12-31'),\n"
            "             ('2017-2021','2017-01-01','2021-12-31'),('2022-2026','2022-01-01','2026-06-26')]\n"
            "    tab = st.subperiod_sweep(DF, edges, horizon=21)\n"
            "    labs = list(tab.index); spr = list(tab['spread']*100); ts = list(tab['t'])\n"
            "else:\n"
            "    labs = [s[0] for s in sub]; spr = [s[1] for s in sub]; ts = [s[2] for s in sub]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "cols = [GREEN if s>0 else RED for s in spr]\n"
            "ax.bar(range(len(labs)), spr, color=cols, width=.55)\n"
            "ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, rotation=10)\n"
            "for x,(s,tt) in enumerate(zip(spr, ts)):\n"
            "    ax.annotate(f't={tt:+.2f}', (x, s), ha='center', va='bottom' if s>=0 else 'top', fontsize=9, color=GREY)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('Q5 - Q1 spread %')\n"
            "ax.set_title('Sign flips: present 2002-09 (t +2.09), inverted 2017-21 (t -0.82)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('sub-period spreads %:', [round(s,2) for s in spr], '| HAC t:', [round(x,2) for x in ts])"
        ),
        md(
            "> 💡 In plain words: H₂ **rejected.** The edge is significant *only* in 2002-2009 (*t* "
            "+2.09), fades in 2010-2016, and **inverts** in the ZIRP 2017-2021 era (*t* −0.82). A "
            "sign that depends on the regime is `MIXED`, not a signal."
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — H₁ right-signed but sub-2 (spread +{R['spread']}%, HAC *t* "
            f"+{R['hac_t']}, placebo *p* {R['placebo_p']}); H₂ rejected (sign flips). ACM-proxy ⇒ "
            "`REAL` is out of reach in principle.\n"
            f"- **Tradability `MIRAGE`** — timer Sharpe {R['timer_sharpe']} vs buy-and-hold "
            f"{R['bh_sharpe']} is a cash-drag artefact; mean-return spread {R['ov_spread_bps']} bps/day.\n"
            "- **Term-premium times duration? `MIXED`** — present in 1 of 4 windows, gone or backwards "
            "in the rest."
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you trade it? — the timing-overlay costs\n\n"
            "Own TLT when the premium rank > 0.5, else cash; one-way cost per switch. Compare Sharpe "
            "**and** mean return to buy-and-hold TLT."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ov = st.timing_overlay(DF, cost_bps=2.0)\n"
            "    ts, bs, sw = ov['timer_sharpe'], ov['bh_sharpe'], ov['switches_per_yr']\n"
            "    sp, spt = ov['spread_bps_day'], ov['spread_t']\n"
            "else:\n"
            f"    ts, bs, sw = {R['timer_sharpe']}, {R['bh_sharpe']}, {R['switches_yr']}\n"
            f"    sp, spt = {R['ov_spread_bps']}, {R['ov_spread_t']}\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))\n"
            "a1.bar(['timer\\n(net)','buy & hold'], [ts, bs], color=[AMBER, GREY], width=.5)\n"
            "a1.set_ylabel('annualised Sharpe'); a1.set_title(f'Sharpe: {ts:.3f} vs {bs:.3f} (cash-drag)')\n"
            "a1.axhline(0, c='k', lw=1)\n"
            "a2.bar(['timer - buy&hold'], [sp], color=(RED if sp<0 else GREEN), width=.4)\n"
            "a2.axhline(0, c='k', lw=1); a2.set_ylabel('mean-return spread (bps/day)')\n"
            "a2.set_title(f'Mean return: {sp:+.2f} bps/day (t {spt:+.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'timer Sharpe {ts:.3f} | buy&hold {bs:.3f} | switches/yr {sw:.1f} | mean spread {sp:+.2f} bps/d (t {spt:+.2f})')"
        ),
        md(
            "> 💡 In plain words: the timer's *Sharpe* edge is an illusion — it sits in cash 57% of the "
            "time, trimming volatility more than return. On the honest metric (mean return) it "
            f"**loses** {abs(R['ov_spread_bps'])} bps/day (*t* {R['ov_spread_t']}) while churning "
            f"{R['switches_yr']} round-trips a year. `MIRAGE`."
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further — the synthetic positive control\n\n"
            "Is the engine a faithful detector, or does it always print ~0? Plant a real timing edge "
            "(`tp_signal > 0`) of increasing strength and watch the Q5−Q1 HAC *t* rise — averaged over "
            "25 seeds so no single lucky seed can fake it."
        ),
        code(
            "signals = [0.0, 0.002, 0.004, 0.008, 0.012]\n"
            "res = [st.synthetic_mean_t(data, tp_signal=a, n_seeds=25, horizon=21) for a in signals]\n"
            "ts = [r['mean_t'] for r in res]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(signals, ts, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1, label='t = +2 bar')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted tp_signal (more positive = stronger timing edge)')\n"
            "ax.set_ylabel('mean Q5-Q1 HAC t (25 seeds)')\n"
            "ax.set_title('The harness banks a planted edge — flat at the null, past +2 as it grows')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for a, r in zip(signals, res): print(f'tp_signal {a:+.3f} -> mean spread {r[\"mean_spread\"]*100:+.2f}%, mean HAC t {r[\"mean_t\"]:+.2f}')"
        ),
        md(
            "The mean HAC *t* is ≈ 0 at the null (no false positive) and climbs far past +2 as the "
            "planted edge grows — so the engine is a faithful detector. The real-tape result is "
            "therefore a statement about **this ACM-proxy tape**: the term-premium timing edge is "
            "right-signed but faint and regime-dependent here. For the raw-slope cousin, see "
            "[132 Yield-Curve-Steepener](../../132-yield-curve-steepener/)."
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
