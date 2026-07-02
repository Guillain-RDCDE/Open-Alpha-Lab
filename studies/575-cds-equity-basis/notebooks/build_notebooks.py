"""Generate the two narrative notebooks for Study 575 (CDS-Equity-Basis).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). This study is
**synthetic-only** — there is no free single-name CDS tape — so every code cell runs on the
deterministic, offline synthetic world; ``HAVE_REAL`` is always False and no cell is ever drawn
under a real-tape banner. The frozen headline numbers in ``R`` mirror docs/results.md exactly and
are recomputed live from the seeded synthetic panel, so the numbers in the prose and the numbers in
the figures agree.
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


# Frozen synthetic-control headline numbers — mirror of docs/results.md (as-of 2026-06-30;
# seed 575, planted convergence_beta = -0.90; 60 names x 95 months; panel fp 884a4f7a09c4).
R = dict(
    fp_panel="884a4f7a09c4", fp_null="4b6557ac1c53",
    n_names=60, n_months=95, n_obs=5700, beta=-0.90,
    slope=-0.00461, slope_t=-5.79, corr=-0.082,
    ls_mean_m=1.18, ls_t=4.78, placebo_p=0.0005,
    gross_m=1.18, net_m=0.65, net_ann=7.8, cost_bps=10.0, borrow_bps=150.0,
    # robustness thirds: (label, slope_t)
    thirds=[("early third", -3.12), ("mid third", -3.45), ("late third", -3.36)],
    # seed-robust control: (beta, mean pooled slope-t, mean LS-t) over 25 seeds
    ctrl=[(0.0, -0.23, 0.19), (-0.30, -2.95, 2.60), (-0.60, -5.64, 4.98), (-0.90, -8.27, 7.31)],
    # null single-seed sanity (why we average): a lucky seed can print ~+-1.9
    null_ls_t=-1.92,
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

from cds_equity_basis import data, strategy as st

BETA = -0.90   # the planted convergence effect for the headline synthetic panel

# There is NO free single-name CDS tape -> this study is synthetic-only.
REAL = data.fetch_panel(fetch=False)
HAVE_REAL = not REAL.empty        # always False, by design
PANEL, TRUTH = data.synthetic_panel(convergence_beta=BETA, seed=575)   # deterministic, offline
print("real CDS tape present:", HAVE_REAL, "(synthetic-only study — no free CDS feed)")
print(f"synthetic panel: {PANEL['name'].nunique()} names x {PANEL['month'].nunique()} months "
      f"= {len(PANEL)} name-months, fp {data.fingerprint(PANEL)}")
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# The CDS-Equity Basis — when a stock and its credit disagree 🔀\n"
            "### Two markets, one company, two different verdicts on how risky it is\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n\n"
            "Every public company is watched by **two** markets at once. Its **stock** says one thing "
            "about how likely it is to get into trouble. And a quieter, professionals-only market — "
            "**credit-default swaps (CDS)**, basically insurance against the company defaulting — says "
            "another. Usually they roughly agree. But sometimes they *don't*: the insurance market gets "
            "nervous while the stock stays calm, or vice-versa. That disagreement is the **CDS-equity "
            "basis**.\n\n"
            "The folklore trade: when the two markets disagree, one of them is *wrong*, and it will "
            "eventually catch up to the other — so the gap should **predict** which way the stock "
            "moves next. When the insurers are more worried than the stockholders, bet the stock falls.\n\n"
            "> ⚠️ **A big honest caveat up front.** Clean single-name CDS data isn't free — it lives on "
            "licensed, professionals-only feeds (Bloomberg, Markit). So we can't point this test at "
            "real companies. Instead we build a **synthetic world** where we *know* the effect is "
            "there, and check that our detector finds it. That means this study can *never* earn the "
            "top 'REAL' stamp — the honest ceiling is `WEAK`.\n\n"
            "> 📓 **This is the plain-language layer.** Want the *t*-stats, the clustered regression and "
            "the cost maths? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ----
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does the gap between credit and equity predict the stock? | **Yes — in principle.** In a "
            "world where we plant the effect, our detector catches it cleanly (the credit-worried names' "
            "stocks do fall). The academic literature agrees the effect is *real*. |\n"
            "| So is it a money machine? | **No.** The effect is *faint* — the gap explains almost none "
            "of any single stock's move — and in liquid names professional arbitrageurs have already "
            "closed most of it. |\n"
            "| Could a normal investor trade it? | **No.** You'd need the CDS data (licensed, expensive) "
            "*and* you'd be shorting exactly the distressed names that are painful to borrow. Even in our "
            "frictionless test, costs eat half the profit. |\n\n"
            "> The signal is real enough to earn `WEAK` — but it lives behind a paywall you can't reach, "
            "on an effect too faint and too arbitraged to bank. `WEAK` × `MIRAGE`."
        ),

        # ---- BEAT 1 — THE CLAIM ----
        md(
            "## 1 · The claim\n\n"
            "> *\"When a company's CDS and its stock disagree about default risk, the gap predicts which "
            "way the stock moves next — the two markets converge.\"*\n\n"
            "The intuition: a company's stock and its credit are two bets on the *same* underlying — "
            "whether the firm thrives or fails. If the credit-insurance market suddenly demands a much "
            "higher premium (it smells trouble) while the stock hasn't budged, one of them is slow. "
            "Credit traders are often first to react to bad news (they live and die on default risk), so "
            "the folklore says the **stock will catch down** to the worried credit. The gap between the "
            "two — the *basis* — is the signal."
        ),

        # ---- BEAT 2 — SO WHAT ----
        md(
            "## 2 · So what?\n\n"
            "If true, it's a genuine cross-asset edge: watch the quiet, informed credit market to trade "
            "the loud, crowded stock market. It's also a clean test of *market efficiency* — if two "
            "prices for the same risk can drift apart and only slowly reconverge, arbitrage has limits. "
            "The desk has looked at *equity-only* distress signals "
            "([540 Distress-Risk-Anomaly](../../540-distress-risk-anomaly/)); here we add the **credit "
            "market's** opinion as a second lens and trade the *disagreement*."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ----
        md(
            "## 3 · How would we even know?\n\n"
            "1. **Measure the gap.** For each company each month: the CDS spread (what credit insurance "
            "costs) minus an *equity-implied* credit spread (what the stock price + its jumpiness say "
            "the credit should cost, via a textbook Merton model). That difference is the **basis**.\n"
            "2. **Rank.** Each month, line up all the names from most credit-*complacent* (stock calm, "
            "credit calm) to most credit-*worried* (credit nervous vs the stock).\n"
            "3. **Check next month.** Did the worried names' stocks fall and the complacent ones' rise? "
            "That's the convergence.\n\n"
            "**The catch we already flagged:** we can't get real CDS data for free, so steps 1-3 run on "
            "a *synthetic* world we built. That world has the effect baked in at a known strength, so it "
            "tests our **detector**, not the real market.\n\n"
            "*Timing:* the gap is measured at month-end (public then); we act at that close and see what "
            "the stock does over the *next* month. The gap never peeks at the return it's predicting."
        ),

        # ---- BEAT 4 — THE TEARDOWN ----
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First: what does the disagreement even look like?** Here's one synthetic company's CDS "
            "(what credit says) vs its equity-implied spread (what the stock says) over time — the gap "
            "between the lines is the basis."
        ),
        code(
            "one = PANEL[PANEL['name']=='N007'].sort_values('month')\n"
            "fig, ax = plt.subplots(figsize=(9, 4.4))\n"
            "ax.plot(one['month'], one['cds_bp'], c=RED, lw=1.8, label='CDS spread (credit says)')\n"
            "ax.plot(one['month'], one['eq_impl_bp'], c=GREEN, lw=1.8, label='equity-implied (stock says)')\n"
            "ax.fill_between(one['month'], one['cds_bp'], one['eq_impl_bp'], color=GREY, alpha=.25, label='the basis (gap)')\n"
            "ax.set_xlabel('month'); ax.set_ylabel('credit spread (bp)')\n"
            "ax.set_title('One company, two markets: the gap between the lines is the CDS-equity basis')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('When RED is above GREEN, credit is more worried than the stock (wide basis).')"
        ),
        md(
            "The two lines track each other loosely but keep drifting apart and back together — that "
            "drift *is* the basis. When the red (credit) line jumps above the green (stock) line, the "
            "insurers are more worried than the shareholders. The folklore says: watch the stock catch "
            "down."
        ),
        md(
            "**Now the actual test.** Rank every name each month by its basis, put them in five buckets, "
            "and look at each bucket's *next*-month stock return. If the folklore holds, the "
            "credit-complacent bucket (low basis) should beat the credit-worried one (high basis)."
        ),
        code(
            "df = st.add_basis_z(PANEL)\n"
            "df['bucket'] = df.groupby('month')['basis_bp'].transform(\n"
            "    lambda s: pd.qcut(s.rank(method='first'), 5, labels=['1 complacent','2','3','4','5 worried']))\n"
            "by = df.groupby('bucket', observed=True)['forward_ret'].mean()*100\n"
            "fig, ax = plt.subplots(figsize=(8, 4.4))\n"
            "cols = [GREEN, '#7bbf6a', GREY, '#d98a6a', RED]\n"
            "ax.bar(range(5), by.values, color=cols, width=.6)\n"
            "ax.set_xticks(range(5)); ax.set_xticklabels(by.index, rotation=10)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('next-month stock return %')\n"
            "ax.set_title('Credit-complacent names (green) beat credit-worried ones (red) — the convergence')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('bucket next-month returns %:', by.round(3).to_dict())"
        ),
        md(
            f"There's the pattern, cleanly: the *complacent* end (credit not worried) earns the most "
            f"next month, the *worried* end the least — a downward staircase. Trading it (long "
            f"complacent, short worried) makes **+{R['ls_mean_m']}%/month** in this world, and a shuffle "
            f"test says that's not luck (*p* = {R['placebo_p']}). **So the detector works.**"
        ),
        md(
            "**But is our detector just *always* finding a pattern, even in pure noise?** The crucial "
            "check: turn the effect **off** (set the planted strength to zero) and average over many "
            "random worlds. A good detector should go silent."
        ),
        code(
            "betas = [b for b,_,_ in " + repr(R["ctrl"]) + "]\n"
            "ls_ts = [st.synthetic_mean_ls_t(data, convergence_beta=b, n_seeds=25) for b in betas]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot([-b for b in betas], ls_ts, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1, label='t = 2 bar')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted effect strength (0 = off)'); ax.set_ylabel('long-short t (25-seed mean)')\n"
            "ax.set_title('Detector is silent when the effect is OFF, loud when it is ON')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for b, t in zip(betas, ls_ts): print(f'strength {-b:.2f} -> long-short t {t:+.2f}')"
        ),
        md(
            "At zero strength the detector prints *t* ≈ 0 (silent); crank the effect up and it climbs "
            "well past the *t* = 2 bar. It's honest. That matters because it tells us the `WEAK` here "
            "is about **data we can't get**, not a detector that cries wolf."
        ),

        # ---- BEAT 5 — VERDICT ----
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — Weak.** The convergence effect is real in the literature and our engine catches "
            "it faithfully — but there's **no free CDS data** to certify it on real companies, so it "
            "can't earn `REAL`.\n"
            "- **Tradability — Mirage.** The data is licensed and expensive, the true effect is faint and "
            "mostly arbitraged, and the short leg is the costly-to-borrow distressed tail."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ----
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Not as a normal investor. Step one is *getting single-name CDS quotes*, which means a "
            "Bloomberg or Markit licence — thousands of dollars a month, professionals only. Even with "
            "the data, the effect is faint, and the trade means **shorting** the most distressed names, "
            "which are exactly the ones that are painful and expensive to borrow. On our frictionless "
            f"synthetic tape, costs already cut the profit from **+{R['gross_m']}%** to "
            f"**+{R['net_m']}%** a month. In the real, illiquid version it would be worse."
        ),

        # ---- BEAT 7 — GOING FURTHER ----
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The equity-only cousin.** [540 Distress-Risk-Anomaly](../../540-distress-risk-anomaly/) "
            "asks whether distressed *stocks* underperform, using only equity data.\n"
            "- **A different basis.** [382 Treasury-Basis-Trade](../../382-treasury-basis-trade/) is a "
            "*rates* basis (bond cash-vs-futures carry) — same word, different market.\n\n"
            "*Have a licensed CDS feed? Fork this, drop a real single-name CDS + equity panel into "
            "`data.fetch_panel`, and see whether the basis clears t = 2 on a real tape — the engine is "
            "ready the day the data is.*"
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
            "# The CDS-Equity Basis — a quantitative teardown 🔬\n"
            "### Per-month z-scored basis · month-clustered pooled slope · decile long-short · label-shuffle placebo · sub-sample robustness · costs & borrow · seed-robust synthetic control\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We build the CDS-equity basis, "
            "regress forward equity returns on it (clustering by month), and sort a decile long-short — "
            "on a **synthetic** tape, because single-name CDS has no free feed.\n\n"
            "> ⚠️ **Synthetic-only, by construction.** There is no free single-name CDS data (licensed "
            "OTC: Markit/IHS, Bloomberg), so `HAVE_REAL` is always False and there is no real tape to "
            "certify the effect — the honest ceiling is `WEAK`. All numbers below are from the "
            f"deterministic seeded synthetic world (panel fp `{R['fp_panel']}`, as-of 2026-06-30). "
            "Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | Pooled slope-*t* (clustered by month) **{R['slope_t']}**; decile "
            f"long-short *t* **+{R['ls_t']}**, placebo *p* **{R['placebo_p']}**; null 25-seed mean "
            f"slope-*t* **{R['ctrl'][0][1]}** (flat). Real, faithful engine — but **no real CDS tape**, "
            "so capped below `REAL`. |\n"
            f"| **Tradability** | `MIRAGE` | Licensed inputs, faint effect (corr **{R['corr']}**), "
            f"costly-to-borrow short leg. Gross **+{R['gross_m']}%/mo** → net **+{R['net_m']}%/mo** "
            f"({R['cost_bps']:.0f} bps/leg + {R['borrow_bps']:.0f} bps borrow). |\n\n"
            "> 💡 In plain words: the machinery works and the effect is documented, but it lives behind a "
            "data paywall and is too faint/arbitraged to bank — `WEAK` × `MIRAGE`."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $B_{i,t}$ be name $i$'s CDS-equity basis at month $t$ (CDS spread minus a Merton "
            "equity-implied spread, bp) and $r_{i,t+1}$ its forward equity return.\n\n"
            "- **H₁ (convergence, pooled).** slope of $r_{t+1}$ on $z(B_t) < 0$ at $|t| \\ge 2$ "
            "(a wider basis predicts a lower forward equity return).\n"
            "- **H₂ (long-short).** long-low-basis / short-high-basis mean $> 0$ at $t \\ge 2$.\n"
            "- **H₃ (robustness).** the sign of H₁ is stable across sub-samples.\n"
            "- **H₄ (net).** H₂ survives costs and the short-leg borrow.\n"
            "- **H₅ (certifiable on a real tape).** the effect clears $t \\ge 2$ on a **real** CDS "
            "panel.\n\n"
            "On the synthetic tape we **confirm** H₁-H₄ (the engine is faithful) but **cannot test** "
            "H₅ — there is no free real CDS feed. H₅ is what a `REAL` stamp requires, so the study is "
            "capped at `WEAK`."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The content is the **why it's only WEAK**. Two forces cap it: (a) **data** — clean "
            "single-name CDS is licensed OTC (Markit/IHS, Bloomberg CDSW), unreachable on a no-key retail "
            "stack, and a structural equity-implied spread also needs point-in-time liabilities; and (b) "
            "**faintness** — Kapadia & Pu (2012) show the convergence is *real but slow and limited*, "
            "concentrated in frictional names where arbitrage can't fully close it. The corr here "
            f"(**{R['corr']}**) is deliberately faithful to that faintness: a genuine but tiny signal. "
            "Compare the equity-only distress lens of "
            "[540 Distress-Risk-Anomaly](../../540-distress-risk-anomaly/)."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Signal.** $z(B_{i,t})$ — the basis, z-scored *within each month* (dollar-neutral, "
            "removes the common credit level).\n"
            "- **Pooled regression.** $r_{i,t+1} = \\alpha + \\beta\\, z(B_{i,t}) + \\varepsilon$ across "
            "all name-months, SE **clustered by month** (Liang-Zeger) so within-month cross-sectional "
            "correlation can't inflate the *t*. $\\beta < 0$ = convergence.\n"
            "- **Long-short.** monthly long-lowest-quintile / short-highest-quintile, IID *t* on the "
            "monthly spreads, a **month-label-shuffle placebo** (2000 perms).\n"
            "- **Robustness.** re-estimate the pooled slope-*t* on early/mid/late thirds.\n"
            "- **Frictions.** round-trip cost per leg each month + a monthly slice of an annual borrow on "
            "the short.\n"
            "- **Positive control.** a deterministic synthetic world with a planted "
            "`convergence_beta < 0` and a null — averaged over 25 seeds (house rule).\n\n"
            "Timing: the basis is measured at month-*t* close (public then); entry is at that close and "
            "the return is realised over month *t+1* — the single one-month execution lag. No look-ahead."
        ),

        # ---- BEAT 4 ----
        md("## 4 · The teardown"),
        md(
            "### 4a · The pooled slope — H₁\n\n"
            "Regress forward equity return on the z-scored basis across all name-months, clustering the "
            "standard error by month. A *negative* slope is the convergence claim."
        ),
        code(
            "reg = st.pooled_regression(PANEL)\n"
            "df = st.add_basis_z(PANEL)\n"
            "# binned scatter for legibility (5700 points -> 25 basis-z bins)\n"
            "bins = pd.qcut(df['basis_z'], 25, duplicates='drop')\n"
            "binned = df.groupby(bins, observed=True).agg(x=('basis_z','mean'), y=('forward_ret','mean'))\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.6))\n"
            "ax.scatter(binned['x'], binned['y']*100, s=34, color=GREY)\n"
            "xs = np.linspace(df['basis_z'].min(), df['basis_z'].max(), 50)\n"
            "b0 = df['forward_ret'].mean() - reg['slope']*df['basis_z'].mean()\n"
            "ax.plot(xs, (reg['slope']*xs + b0)*100, c=RED, lw=2)\n"
            "ax.set_xlabel('basis z-score (higher = credit more worried)'); ax.set_ylabel('forward equity return %')\n"
            "ax.set_title(f\"Slope t {reg['slope_t']:+.2f} (clustered by month) — NEGATIVE = convergence\")\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"slope {reg['slope']:+.5f}, clustered slope-t {reg['slope_t']:+.2f}, \"\n"
            "      f\"corr(basis, fwd) {reg['corr']:+.3f}, n {reg['n']}\")"
        ),
        md(
            f"> 💡 In plain words: H₁ **confirmed** — the slope is negative (*t* **{R['slope_t']}**, "
            f"clustered), so a wider basis predicts a *lower* forward equity return. Note the corr is "
            f"only **{R['corr']}**: a real, faithful, but *faint* signal — most of any stock's move is "
            "idiosyncratic noise the basis can't see."
        ),
        md(
            "### 4b · The decile long-short — H₂ & the placebo\n\n"
            "Long the lowest-basis quintile, short the highest, each month. The convergence claim "
            "predicts a *positive* spread; the placebo shuffles the basis labels within each month."
        ),
        code(
            "ls = st.decile_long_short(PANEL, frac=0.2)\n"
            "p = st.placebo_pvalue(PANEL, n_perm=2000, seed=575, frac=0.2)\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))\n"
            "cum = (1 + ls['monthly']).cumprod()\n"
            "a1.plot(cum.index, cum.values, c=GREEN, lw=1.8)\n"
            "a1.axhline(1, c='k', lw=1); a1.set_xlabel('month'); a1.set_ylabel('growth of 1 (gross)')\n"
            "a1.set_title(f\"Long-short: mean {ls['mean']*100:+.2f}%/mo, IID t {ls['t']:+.2f}\")\n"
            "a2.hist(ls['monthly']*100, bins=20, color=GREY, alpha=.8)\n"
            "a2.axvline(ls['mean']*100, c=RED, lw=2, label=f\"mean {ls['mean']*100:+.2f}%\")\n"
            "a2.axvline(0, c='k', lw=1); a2.set_xlabel('monthly long-short return %'); a2.legend()\n"
            "a2.set_title(f'placebo p = {p:.4f}')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"long-short mean {ls['mean']*100:+.3f}%/mo, IID t {ls['t']:+.2f}, \"\n"
            "      f\"months {ls['n']}, placebo p {p:.4f}\")"
        ),
        md(
            f"> 💡 In plain words: H₂ **confirmed** — the long-short earns **+{R['ls_mean_m']}%/month** "
            f"(*t* **+{R['ls_t']}**), and the placebo *p* = **{R['placebo_p']}** places it far in the "
            "tail. The signal is genuinely there — *in this world where we planted it*."
        ),
        md(
            "### 4c · Robustness — H₃ (does the sign hold?)\n\n"
            "Re-estimate the pooled slope-*t* on the early, mid and late thirds of the sample."
        ),
        code(
            "rows = st.robustness_thirds(PANEL)\n"
            "tab = pd.DataFrame(rows).set_index('window')\n"
            "fig, ax = plt.subplots(figsize=(8, 4.2))\n"
            "ax.bar(range(len(tab)), tab['slope_t'], color=RED, width=.5)\n"
            "ax.set_xticks(range(len(tab))); ax.set_xticklabels(tab.index)\n"
            "ax.axhline(-2, ls='--', c=GREY, lw=1, label='t = -2 bar')\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('pooled slope-t (clustered)')\n"
            "ax.set_title('Convergence sign is stable across sub-samples'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(tab[['slope_t','n']].round(2).to_string())"
        ),
        md(
            "> 💡 In plain words: H₃ **confirmed** — the slope-*t* stays negative and past −2 in every "
            "third (−3.1 / −3.4 / −3.4). A faithful engine keeps the sign it was given."
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — H₁-H₃ confirmed on the synthetic tape (slope-*t* {R['slope_t']}, "
            f"long-short *t* +{R['ls_t']}, placebo *p* {R['placebo_p']}, null flat), and the literature "
            "backs the effect — but **H₅ (a real tape) can't be tested**: no free CDS feed. `REAL` "
            "requires a robust *t* on a real tape, so the ceiling is `WEAK`.\n"
            f"- **Tradability `MIRAGE`** — licensed inputs, faint effect (corr {R['corr']}), "
            f"costly-to-borrow short leg; gross +{R['gross_m']}%/mo → net +{R['net_m']}%/mo.\n"
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you trade it? — costs and the borrow\n\n"
            "Monthly rebalancing turns both legs over every month; the short leg (credit-worried names) "
            "is the expensive-to-borrow distressed tail. Charge it honestly."
        ),
        code(
            "ls = st.decile_long_short(PANEL, frac=0.2)\n"
            "net = st.net_long_short(ls, cost_bps=10.0, borrow_ann_bps=150.0)\n"
            "fig, ax = plt.subplots(figsize=(7.2, 4.2))\n"
            "vals = [net['gross_m']*100, net['net_m']*100]\n"
            "ax.bar(['gross', 'net\\n(costs+borrow)'], vals, color=[GREEN, GREY], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('mean long-short return %/mo')\n"
            "ax.set_title(f\"Costs halve it: {vals[0]:+.2f}%/mo gross -> {vals[1]:+.2f}%/mo net\")\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"gross {net['gross_m']*100:+.2f}%/mo -> net {net['net_m']*100:+.2f}%/mo \"\n"
            "      f\"({net['net_ann']*100:+.1f}%/yr; 10 bps/leg + 150 bps borrow)\")"
        ),
        md(
            "> 💡 In plain words: even on a *frictionless synthetic tape*, costs + borrow eat roughly "
            "half the gross spread. On the real, illiquid CDS/short-borrow reality — plus the licensed "
            "data bill — there is nothing left. `MIRAGE`."
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further — the seed-robust synthetic positive control\n\n"
            "Is the engine a faithful detector, or does it always print significance? Plant a convergence "
            "effect (`convergence_beta < 0`) of increasing strength and watch *both* the pooled slope-*t* "
            "and the long-short *t* respond — averaged over 25 seeds so no single lucky seed can fake it "
            "(a single seed at the null can print *t* ≈ ±1.9 — exactly why we average)."
        ),
        code(
            "betas = [0.0, -0.2, -0.4, -0.6, -0.8, -1.0, -1.2]\n"
            "slope_ts = [st.synthetic_mean_t(data, convergence_beta=b, n_seeds=25) for b in betas]\n"
            "ls_ts = [st.synthetic_mean_ls_t(data, convergence_beta=b, n_seeds=25) for b in betas]\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.4))\n"
            "x = [-b for b in betas]\n"
            "ax.plot(x, slope_ts, 'o-', c=RED, lw=2, label='pooled slope-t')\n"
            "ax.plot(x, ls_ts, 's-', c=GREEN, lw=2, label='long-short t')\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1); ax.axhline(-2, ls='--', c=GREY, lw=1)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted effect strength |convergence_beta|'); ax.set_ylabel('mean t (25 seeds)')\n"
            "ax.set_title('Flat at the null, monotone past |t|=2 as the planted effect grows')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for b, s, l in zip(betas, slope_ts, ls_ts):\n"
            "    print(f'beta {b:+.2f} -> pooled slope-t {s:+.2f} | long-short t {l:+.2f}')"
        ),
        md(
            "Both statistics sit at ≈ 0 at the null and move monotonically past ±2 as the planted effect "
            "grows — a faithful detector. So the `WEAK` ceiling here is a statement about **data "
            "availability**, not a broken engine: hand this code a licensed single-name CDS + equity "
            "panel and it is ready to certify (or reject) the basis on a real tape. For the equity-only "
            "distress lens, see [540 Distress-Risk-Anomaly](../../540-distress-risk-anomaly/); for a "
            "*rates* basis, [382 Treasury-Basis-Trade](../../382-treasury-basis-trade/)."
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
