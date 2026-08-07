"""Generate the two narrative notebooks for Study 837 (Look-Ahead Standardization).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

This is a synthetic-only method demo: every figure runs offline and deterministic on the seed-837
worlds — there is NO real-tape cell (real free data can never certify "zero edge"), so the study is
capped at NONE. The heavy 20-seed headline numbers are quoted from the frozen ``R`` dict below
(mirror of docs/results.md); the notebooks live-run only the fast single-seed panels (a few seconds),
so they execute in well under two minutes.
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


# Frozen headline numbers — mirror of docs/results.md (as-of 2026-06-30; sim config fp 5f6dcb4c991c;
# N=60, T=1000, horizon=10, min_periods=60, 20 seeds, base_seed 837). Recomputed live for the fast
# single-seed panels in the notebooks; the 20-seed aggregates are quoted from here.
R = dict(
    fp="5f6dcb4c991c", n_names=60, n_days=1000, horizon=10, min_periods=60, n_seeds=20,
    # stationary null (no leak): full/exp IC, t, Sharpe, sig seeds
    stat_full_ic=-0.0004, stat_exp_ic=0.0002, stat_full_sharpe=0.37, stat_exp_sharpe=0.37,
    stat_full_sig=3, stat_exp_sig=2,
    # non-stationary null (THE TRAP)
    ns_full_ic=-0.1383, ns_full_t=-12.12, ns_full_sharpe=15.84, ns_full_sig=20,
    ns_exp_ic=-0.0001, ns_exp_t=-0.01, ns_exp_sharpe=1.04, ns_exp_sig=0,
    ic_gap=0.1286, sharpe_gap=14.80,
    # planted real edge (control)
    pl_full_ic=0.0922, pl_full_t=22.79, pl_exp_ic=0.0910, pl_exp_t=21.64,
    pl_full_sig=20, pl_exp_sig=20,
    # timer (non-stationary null, full signal)
    timer_gross_bps=144.6, timer_net1_bps=142.5, timer_net5_bps=134.5, timer_net5_sharpe=13.53,
    # horizon sweep (10 seeds): (H, full_ic, exp_ic, full_sharpe)
    hsweep=[(1, -0.0344, -0.0013, 4.10), (5, -0.0934, 0.0001, 11.20), (10, -0.1378, -0.0007, 16.05),
            (20, -0.1995, -0.0029, 21.95), (40, -0.2826, -0.0012, 28.16)],
    # length sweep (10 seeds): (T, full_ic, exp_ic, full_sharpe)
    lsweep=[(250, -0.2749, -0.0040, 28.63), (500, -0.1937, -0.0017, 21.65),
            (1000, -0.1378, -0.0007, 16.05), (2000, -0.0985, -0.0005, 12.08)],
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

from lookahead_standardization import data, strategy as st

# Everything runs on deterministic, offline seed-837 synthetic worlds. Synthetic-only method demo:
# there is no real-tape cell by design (real free data can never certify "zero edge").
Xns, Rns = data.null_nonstationary(seed=data.BASE_SEED)   # the trap (random-walk feature)
Xst, Rst = data.null_stationary(seed=data.BASE_SEED)      # the contrast (stationary feature)
Xpl, Rpl = data.planted_edge(seed=data.BASE_SEED)         # the control (a real edge)
print("worlds:", Xns.shape, "| non-stationary null, stationary null, planted edge",
      "| sim fp", data.config_fingerprint(n_seeds=20))
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Look-Ahead Standardization — how *one line of preprocessing* fakes a great backtest 🔮\n"
            "### Why `z = (x - x.mean()) / x.std()` over the whole sample quietly cheats — in plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Full-sample standardisation leaks?: Confirmed](https://img.shields.io/badge/Full--sample_standardisation_leaks%3F-Confirmed-8b949e?style=flat-square)\n\n"
            "Before you feed a feature to a model, you almost always **standardise** it — subtract its "
            "average, divide by how much it wobbles, so everything is on a comparable scale. Totally "
            "routine. But *when* you compute that average and wobble matters enormously. Do it over the "
            "**whole history at once** — the usual `(x - x.mean()) / x.std()` — and you have just used "
            "**future** data to rescale the past. Your backtest now knows things it couldn't have "
            "known.\n\n"
            "We show what that does on a world we **built to contain no edge at all**: a purely random, "
            "unpredictable feature. The right way to standardise (using only the past) correctly finds "
            "*nothing*. The full-sample way conjures a strategy with a **Sharpe of 16** out of thin "
            "air.\n\n"
            "> 📓 **This is the plain-language layer.** Want the Information Coefficient, the "
            "Newey-West *t*, and the random-walk maths? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool: every chart is drawn by the "
            "code beside it, on deterministic synthetic worlds. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ----
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Can standardising a feature *the usual way* fake a backtest? | **Yes — enormously.** "
            f"On a feature that is pure random noise, full-sample z-scoring manufactures a strategy "
            f"with a Sharpe of **{R['ns_full_sharpe']}** — on **{R['ns_full_sig']}/{R['n_seeds']}** "
            "independent tries. |\n"
            f"| Does the *honest* way (use only the past) find it? | **No — it reads zero.** Same data, "
            f"point-in-time standardisation: Sharpe **{R['ns_exp_sharpe']}**, significant on "
            f"**{R['ns_exp_sig']}/{R['n_seeds']}** tries. Nothing there. |\n"
            "| So is the honest method just blind to everything? | **No.** Plant a *real* edge and the "
            f"honest method lights right up (we check this at the end). It's a fair detector — it just "
            "refuses to be fooled. |\n\n"
            "> The gorgeous backtest is an illusion baked in by *how the numbers were prepared*, not by "
            "any real signal. The fix costs nothing: standardise using only data you'd have had at the "
            "time."
        ),

        # ---- BEAT 1 — THE CLAIM ----
        md(
            "## 1 · The claim\n\n"
            "> *\"Standardising your features is harmless bookkeeping.\"*\n\n"
            "It isn't — not if you do it over the whole sample. To standardise, you need the feature's "
            "**mean** and **spread**. If you compute those from the entire history, you've peeked at "
            "the future: the value on day 10 gets rescaled using data from day 900. That future "
            "knowledge leaks into your signal, and a backtest can't tell the difference between a real "
            "edge and a leaked one."
        ),

        # ---- BEAT 2 — SO WHAT ----
        md(
            "## 2 · So what?\n\n"
            "Because this is the single most common accidental cheat in a quant pipeline. It hides "
            "inside 'preprocessing', so it dodges the suspicion a mis-dated signal would attract — "
            "nobody audits a `.mean()`. The desk has shown you can fake a track record by **searching** "
            "many strategies ([344 Backtest-Overfitting](../../344-backtest-overfitting/)) or by "
            "**reshaping the returns** ([590 Sharpe-Hacking](../../590-sharpe-hacking/)); this one needs "
            "*no search and no trickery* — just one innocent line run at the wrong moment."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ----
        md(
            "## 3 · How would we even know?\n\n"
            "We **build** a world with no possible edge. The feature is a *random walk* — think of a "
            "wandering price with no memory. What we try to predict is where it drifts next, which for "
            "a random walk is **genuinely unforecastable** from anything you know today. So an honest "
            "signal *must* score zero.\n\n"
            "Then we standardise the feature two ways and see which one 'finds' a (fake) edge:\n\n"
            "1. the **full-sample** way — mean & spread over the whole history (peeks at the future), "
            "and\n"
            "2. the **expanding** way — mean & spread over *only the past so far* (what you'd really "
            "have had).\n\n"
            "If the full-sample way 'works' and the expanding way doesn't, the 'edge' was leaked."
        ),

        # ---- BEAT 4 — THE TEARDOWN ----
        md(
            "## 4 · The teardown — let's actually try it\n\n"
            "Here's a single random-walk feature. Watch what full-sample centring does: it subtracts "
            "the average of the *whole* path, so early points land below zero and late points above — "
            "the standardised value secretly encodes **where in time you are**."
        ),
        code(
            "t = np.arange(len(Xns))\n"
            "x = Xns[:, 0]\n"
            "z_full = st.full_standardize(Xns)[:, 0]\n"
            "z_exp  = st.expanding_standardize(Xns, 60)[:, 0]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))\n"
            "a1.plot(t, x, c=GREY, lw=1.2); a1.axhline(x.mean(), ls='--', c=RED, lw=1.5,\n"
            "        label='full-sample mean (uses the FUTURE)')\n"
            "a1.set_title('A random-walk feature and its full-sample mean'); a1.set_xlabel('day'); a1.legend()\n"
            "a2.plot(t, z_full, c=RED, lw=1.2, label='full-sample z (peeks ahead)')\n"
            "a2.plot(t, z_exp, c=GREEN, lw=1.2, label='expanding z (past only)')\n"
            "a2.axhline(0, c='k', lw=.8)\n"
            "a2.set_title('full-sample z drifts from - to + across time; expanding z does not')\n"
            "a2.set_xlabel('day'); a2.legend(); plt.tight_layout(); plt.show()"
        ),
        md(
            "See how the red (full-sample) line sweeps from negative to positive across the sample? "
            "That sweep is the leak: a random walk drifts back toward its own average over time, so "
            "'below the full-sample mean' really means 'earlier, and about to rise'. The signal has "
            "*memorised the future*. The green (expanding) line has no such sweep — it only ever knew "
            "the past.\n\n"
            "Now score both as a real backtest would: rank the 60 names each day by the standardised "
            "feature, and see how well that ranking lines up with the next return (that's the "
            "**Information Coefficient**)."
        ),
        code(
            "ic_full = np.nanmean(st.cross_sectional_ic(st.full_standardize(Xns), Rns))\n"
            "ic_exp  = np.nanmean(st.cross_sectional_ic(st.expanding_standardize(Xns, 60), Rns))\n"
            "sh_full = st.book_stats(st.long_short_spread(st.full_standardize(Xns), Rns))['sharpe_abs']\n"
            "sh_exp  = st.book_stats(st.long_short_spread(st.expanding_standardize(Xns, 60), Rns))['sharpe_abs']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.2))\n"
            "a1.bar(['full-sample\\n(peeks)', 'expanding\\n(honest)'], [abs(ic_full), abs(ic_exp)],\n"
            "       color=[RED, GREEN], width=.55); a1.set_ylabel('|Information Coefficient|')\n"
            "a1.set_title('Fake predictive power from noise')\n"
            "a2.bar(['full-sample\\n(peeks)', 'expanding\\n(honest)'], [sh_full, sh_exp],\n"
            "       color=[RED, GREEN], width=.55); a2.set_ylabel('backtest Sharpe')\n"
            "a2.set_title('...and a gorgeous fake Sharpe')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'full-sample: |IC| {abs(ic_full):.3f}, Sharpe {sh_full:.1f}')\n"
            "print(f'expanding  : |IC| {abs(ic_exp):.3f}, Sharpe {sh_exp:.1f}')"
        ),
        md(
            f"There it is. On **pure noise**, the full-sample z-score posts a Sharpe around "
            f"**{R['ns_full_sharpe']:.0f}** and a real-looking IC; the honest expanding z-score posts "
            "**nothing**. And this isn't a lucky seed — across "
            f"**{R['n_seeds']} independent worlds** the full-sample leak is 'significant' on "
            f"**{R['ns_full_sig']}/{R['n_seeds']}** and the honest method on "
            f"**{R['ns_exp_sig']}/{R['n_seeds']}**."
        ),

        # ---- BEAT 5 — VERDICT ----
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The 'edge' is a leak; on data built to have none, only the "
            "future-peeking method finds anything.\n"
            "- **Tradability — Mirage.** You can't run it: it needs the full-sample average, which "
            "isn't known until *after* the last trade. Standardise honestly and it vanishes.\n"
            "- **Does full-sample standardisation leak? — Confirmed.** Spectacularly, on a "
            "non-stationary feature."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ----
        md(
            "## 6 · Could you actually trade it?\n\n"
            "No — and not for the usual 'costs eat it' reason. This book is **unimplementable**: to "
            "compute today's signal it uses the feature's average over the *whole* sample, including "
            "days that haven't happened yet. In a live account you simply don't have that number. The "
            "moment you switch to the honest, past-only standardisation you *could* run live, the "
            "Sharpe collapses from ~16 to ~1 (i.e. zero). The beautiful equity curve was never real."
        ),

        # ---- BEAT 7 — GOING FURTHER ----
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Is the honest method just numb?** No. In the quants notebook we plant a *genuine* edge "
            "and the expanding method finds it loud and clear — it's a fair detector, not a blind "
            "one.\n"
            "- **The generic cousin.** [347 Look-Ahead Bias](../../347-look-ahead-bias/) mis-*times* a "
            "signal; this study keeps the timing right but leaks through the **standardisation** step.\n"
            "- **The fix is one word: expanding.** Compute every mean, std, min, max, quantile on a "
            "past-only window — the same one you could run live.\n\n"
            "*Think your own feature pipeline is clean? Fork this, drop in your standardiser, and check "
            "whether it ever touches the test set.*"
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
            "# Look-Ahead Standardization — a quantitative teardown 🔬\n"
            "### full-sample vs expanding z-score · cross-sectional rank IC + Newey-West *t* · the random-walk mean-reversion mechanism · horizon/length scaling laws · costed timer · planted-edge control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Full-sample standardisation leaks?: Confirmed](https://img.shields.io/badge/Full--sample_standardisation_leaks%3F-Confirmed-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We standardise a feature panel "
            "two ways — **full-sample** (leaky) and **expanding / point-in-time** (honest) — and measure "
            "the damage with a cross-sectional Information Coefficient and a Newey-West *t*.\n\n"
            "> ⚠️ **Not investment advice.** A synthetic-only method demo: the worlds are built to have "
            f"no point-in-time edge (sim fp `{R['fp']}`), so real free data can never certify them and "
            "the study is capped at `NONE`. Methods in [`docs/references.md`](../docs/references.md), "
            "numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Non-stationary null: **full-sample** IC {R['ns_full_ic']} "
            f"(NW *t* {R['ns_full_t']}), fake Sharpe {R['ns_full_sharpe']}, sig {R['ns_full_sig']}/"
            f"{R['n_seeds']}; **expanding** IC {R['ns_exp_ic']} (*t* {R['ns_exp_t']}), Sharpe "
            f"{R['ns_exp_sharpe']}, sig {R['ns_exp_sig']}/{R['n_seeds']}. Synthetic-only (no real "
            "tape). |\n"
            f"| **Tradability** | `MIRAGE` | The signal needs the full-sample mean/std → unimplementable; "
            f"re-standardise point-in-time → Sharpe {R['ns_full_sharpe']}→{R['ns_exp_sharpe']}. |\n"
            f"| **Full-sample standardisation leaks?** | `CONFIRMED` | Leak grows with horizon "
            f"(IC {R['hsweep'][0][1]}→{R['hsweep'][-1][1]} for H 1→40), dilutes with length "
            f"({R['lsweep'][0][1]}→{R['lsweep'][-1][1]} for T 250→2000), ~0 on a stationary feature "
            f"({R['stat_full_ic']}); expanding recovers a planted edge (*t* {R['pl_exp_t']}). |\n\n"
            "> 💡 In plain words: full-sample centring of a non-stationary feature encodes the future; "
            "the point-in-time method sees through it *and* still banks a real edge."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $x_{i,t}$ be a feature and $r_{i,t}$ its forward return. A z-score needs a location and "
            "scale $(\\hat\\mu_{i}, \\hat\\sigma_{i})$:\n\n"
            "- **Leaky (full-sample):** $z^{F}_{i,t} = (x_{i,t} - \\bar x_i)/s_i$ with "
            "$\\bar x_i, s_i$ computed over $t = 0..T$ — **including the future**.\n"
            "- **Honest (expanding):** $z^{E}_{i,t} = (x_{i,t} - \\hat\\mu_{i,\\le t})/\\hat\\sigma_{i,\\le t}$, "
            "using only rows $0..t$.\n\n"
            "**H₁ (the leak).** For a **non-stationary** $x$ (a random walk), $\\bar x_i$ is a "
            "future-dependent quantity the expanding mean never converges to, so $z^{F}$ encodes the "
            "future and manufactures IC even when $r$ is unforecastable. **H₂ (the contrast).** For a "
            "**stationary** $x$, $\\bar x_i \\to$ a constant, so $z^{F}$ is ≈ a per-name affine rescale "
            "and leaks ≈ 0. **H₃ (unbiasedness).** $z^{E}$ recovers a *genuinely* predictive feature. "
            "We confirm all three."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what? — the mechanism\n\n"
            "A driftless random walk mean-reverts toward its **own sample mean**: "
            "$\\mathbb{E}[x_{i,t+h} - x_{i,t}\\,|\\,x_{i,t}-\\bar x_i] \\propto -(x_{i,t}-\\bar x_i)$ "
            "*because $\\bar x_i$ was computed from the whole path*. So $z^{F}_{i,t} \\propto "
            "x_{i,t}-\\bar x_i$ is **negatively** correlated with the forward change — a spurious "
            "'mean-reversion' edge that is 100% look-ahead (hence the negative IC). The expanding mean "
            "$\\hat\\mu_{i,\\le t}$ lags the level and carries no such information, so $z^{E}$ is "
            "uncorrelated with the (iid) forward increment. This is a *preprocessing* leak — the signal "
            "is correctly **timed**; only its scaling statistic peeked — which is exactly what makes it "
            "distinct from [347](../../347-look-ahead-bias/)'s mis-alignment."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Worlds.** (a) *non-stationary null*: random-walk feature, forward return = its own "
            "10-day change + noise (iid increments ⇒ unforecastable from the past); (b) *stationary "
            "null*: AR(1) feature (φ=0.9), iid returns; (c) *planted edge*: stationary feature, "
            "$r_{t+1}=0.10\\,x_t+\\varepsilon$ (real, point-in-time).\n"
            "- **Two standardisations.** `full_standardize` (future-inclusive) vs "
            "`expanding_standardize` (past-only, 60-day burn-in).\n"
            "- **Score.** Daily cross-sectional **rank IC** (Spearman) of signal vs forward return; a "
            "**Newey-West** (HAC, 10-lag) *t* on the daily IC series (robust to the strong "
            "autocorrelation the random walk induces); a long-short top/bottom-20% book and its "
            "annualised Sharpe.\n"
            "- **Robustness.** 20-seed aggregation (house rule); a forward-horizon sweep; a "
            "sample-length sweep.\n"
            "- **Costs.** 2 sides × one-way × NAV per rebalance + 50 bps/yr borrow.\n\n"
            "Execution honesty: the honest signal uses data through the close of $t$ only; the leaky one "
            "is flagged as using $t{+}1..T$."
        ),

        # ---- BEAT 4 ----
        md("## 4 · The teardown"),
        md(
            "### 4a · The daily IC series — full-sample leaks, expanding is flat (non-stationary null)\n\n"
            "The per-day cross-sectional IC, cumulated, for both standardisations on the trap world."
        ),
        code(
            "ic_f = st.cross_sectional_ic(st.full_standardize(Xns), Rns)\n"
            "ic_e = st.cross_sectional_ic(st.expanding_standardize(Xns, 60), Rns)\n"
            "tf, te = st.newey_west_t(ic_f), st.newey_west_t(ic_e)\n"
            "print(f'full-sample : mean IC {np.nanmean(ic_f):+.4f}  NW t {tf:+.2f}')\n"
            "print(f'expanding   : mean IC {np.nanmean(ic_e):+.4f}  NW t {te:+.2f}')\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.4))\n"
            "ax.plot(np.nancumsum(ic_f), c=RED, lw=1.8, label=f'full-sample (mean IC {np.nanmean(ic_f):+.3f}, t {tf:+.1f})')\n"
            "ax.plot(np.nancumsum(ic_e), c=GREEN, lw=1.8, label=f'expanding (mean IC {np.nanmean(ic_e):+.3f}, t {te:+.1f})')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('day'); ax.set_ylabel('cumulative daily IC')\n"
            "ax.set_title('Full-sample z-score leaks a steady IC from noise; expanding stays flat')\n"
            "ax.legend(); plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: **H₁ confirmed.** The full-sample IC marches steadily "
            f"({R['ns_full_ic']}, NW *t* {R['ns_full_t']}) on a panel with nothing to find; the "
            f"expanding IC is a flat {R['ns_exp_ic']} (*t* {R['ns_exp_t']}). The sign is negative — the "
            "manufactured edge is spurious reversion toward the peeked-at mean."
        ),
        md(
            "### 4b · The contrast — a stationary feature barely leaks (H₂)\n\n"
            "Run the *same* full-sample z-score on a **stationary** feature. Now its full-sample mean is "
            "a stable estimate of a constant, so standardisation is ≈ a per-name affine rescale — and "
            "the leak nearly vanishes. This pins the pitfall to **non-stationarity**."
        ),
        code(
            "worlds = {'non-stationary\\n(random walk)': (Xns, Rns), 'stationary\\n(AR(1))': (Xst, Rst)}\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3)); x = np.arange(len(worlds)); w = 0.38\n"
            "fic = [abs(np.nanmean(st.cross_sectional_ic(st.full_standardize(X), Rr))) for X, Rr in worlds.values()]\n"
            "eic = [abs(np.nanmean(st.cross_sectional_ic(st.expanding_standardize(X, 60), Rr))) for X, Rr in worlds.values()]\n"
            "ax.bar(x - w/2, fic, w, color=RED, label='full-sample (leaky)')\n"
            "ax.bar(x + w/2, eic, w, color=GREEN, label='expanding (honest)')\n"
            "ax.set_xticks(x); ax.set_xticklabels(list(worlds)); ax.set_ylabel('|mean IC|')\n"
            "ax.set_title('The leak needs non-stationarity: a stationary feature barely leaks')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for name, f, e in zip(worlds, fic, eic): print(f'{name.splitlines()[0]:16s} full |IC| {f:.4f} | exp |IC| {e:.4f}')"
        ),
        md(
            f"> 💡 In plain words: **H₂ confirmed.** On the stationary feature the full-sample |IC| is "
            f"~{abs(R['stat_full_ic']):.4f} — indistinguishable from the expanding "
            f"~{abs(R['stat_exp_ic']):.4f}. Full-sample standardisation is not *always* poison; it is "
            "poison specifically for **non-stationary** (trending / random-walk / integrated) "
            "features — exactly the ones you're most tempted to normalise."
        ),
        md(
            "### 4c · The scaling laws — the finite-sample fingerprint (H₁)\n\n"
            "A real effect doesn't care about your forward horizon or sample length; a **look-ahead "
            "artefact** does. The leak **grows** with the forward horizon (more overlap with the future "
            "the full mean saw) and **dilutes** with the sample length (each future point is a smaller "
            "slice of $\\bar x_i$). Quoted from the frozen 10-seed sweeps."
        ),
        code(
            "hs = np.array([[h, fi, ei, sh] for h, fi, ei, sh in "
            + repr(R['hsweep']) + "])\n"
            "ls = np.array([[T, fi, ei, sh] for T, fi, ei, sh in "
            + repr(R['lsweep']) + "])\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "a1.plot(hs[:,0], np.abs(hs[:,1]), 'o-', c=RED, lw=2, label='full-sample')\n"
            "a1.plot(hs[:,0], np.abs(hs[:,2]), 'o-', c=GREEN, lw=2, label='expanding')\n"
            "a1.set_xlabel('forward horizon (days)'); a1.set_ylabel('|mean IC|')\n"
            "a1.set_title('Leak GROWS with horizon'); a1.legend()\n"
            "a2.plot(ls[:,0], np.abs(ls[:,1]), 'o-', c=RED, lw=2, label='full-sample')\n"
            "a2.plot(ls[:,0], np.abs(ls[:,2]), 'o-', c=GREEN, lw=2, label='expanding')\n"
            "a2.set_xlabel('sample length T (days)'); a2.set_ylabel('|mean IC|')\n"
            "a2.set_title('Leak DILUTES with sample length'); a2.legend()\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            "> 💡 In plain words: both curves are the signature of a **finite-sample look-ahead**, not "
            "an edge. The expanding (green) line hugs zero throughout; only the full-sample (red) line "
            "moves, and it moves exactly the way a leak should."
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — on a provably-unforecastable null, full-sample z-scoring posts IC "
            f"{R['ns_full_ic']} (*t* {R['ns_full_t']}) / Sharpe {R['ns_full_sharpe']} on "
            f"{R['ns_full_sig']}/{R['n_seeds']} seeds; expanding posts ~0. No real signal; synthetic-only "
            "⇒ never `REAL`.\n"
            f"- **Tradability `MIRAGE`** — the signal is built from full-sample stats (unknown until "
            "after the last trade); point-in-time it collapses to Sharpe ~1.\n"
            "- **Full-sample standardisation leaks? `CONFIRMED`** — severely, with the exact "
            "finite-sample scaling of a look-ahead."
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you trade it? — the costed timer\n\n"
            "For completeness, charge the fake book to trade. The honest caveat: a Sharpe-16 look-ahead "
            "**shrugs off** friction — which is why the Mirage here is about **implementability**, not "
            "costs."
        ),
        code(
            "sp = st.long_short_spread(st.full_standardize(Xns), Rns, frac=0.2)\n"
            "for cb in (0.0, 1.0, 5.0):\n"
            "    tm = st.timer_stats(sp, cost_bps=cb, borrow_bps_yr=50.0)\n"
            "    print(f'cost {cb:>4.1f} bps/side: gross {tm[\"gross_bps\"]:+.1f} -> net {tm[\"net_bps\"]:+.1f} bps/day '\n"
            "          f'(net Sharpe {tm[\"sharpe_net\"]:.2f})')\n"
            "print()\n"
            "print('The REAL refutation is not costs but implementability:')\n"
            "print(f'  full-sample (needs the future) Sharpe {st.book_stats(sp)[\"sharpe_abs\"]:.1f}')\n"
            "spe = st.long_short_spread(st.expanding_standardize(Xns, 60), Rns, frac=0.2)\n"
            "print(f'  expanding  (runnable live)     Sharpe {st.book_stats(spe)[\"sharpe_abs\"]:.1f}')"
        ),
        md(
            f"> 💡 In plain words: even at 5 bps/side the fake book still 'earns' (net Sharpe "
            f"~{R['timer_net5_sharpe']:.0f}) — costs don't save you. What saves you is refusing to use "
            "data you won't have: the *runnable* (expanding) version is Sharpe ~1, i.e. **nothing**. "
            "`MIRAGE`."
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further — the planted-edge positive control\n\n"
            "Is the expanding method just numb to everything? Plant a **genuine, point-in-time edge** "
            "($r_{t+1}=0.10\\,x_t+\\varepsilon$ on a stationary feature) and check that the honest "
            "method *recovers* it — proof its silence on the nulls is meaningful (averaged over "
            f"{R['n_seeds']} seeds, the house rule)."
        ),
        code(
            "ic_e = st.cross_sectional_ic(st.expanding_standardize(Xpl, 60), Rpl)\n"
            "ic_f = st.cross_sectional_ic(st.full_standardize(Xpl), Rpl)\n"
            "print(f'planted edge -> expanding IC {np.nanmean(ic_e):+.4f} (NW t {st.newey_west_t(ic_e):+.2f})')\n"
            "print(f'planted edge -> full-sample IC {np.nanmean(ic_f):+.4f} (NW t {st.newey_west_t(ic_f):+.2f})')\n"
            "labels = ['non-stationary\\nNULL', 'planted\\nEDGE']\n"
            "exp_ics = [abs(np.nanmean(st.cross_sectional_ic(st.expanding_standardize(Xns,60), Rns))),\n"
            "           abs(np.nanmean(ic_e))]\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.3))\n"
            "ax.bar(labels, exp_ics, color=[GREY, GREEN], width=.55)\n"
            "ax.set_ylabel('expanding |mean IC|')\n"
            "ax.set_title('The HONEST method: silent on the null, alive on a real edge (unbiased)')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"The expanding method reads ~0 on the null but **+{R['pl_exp_ic']}** (NW *t* "
            f"{R['pl_exp_t']}, significant on {R['pl_exp_sig']}/{R['n_seeds']} seeds) on the planted "
            "edge — a fair, unbiased detector. That's what makes its silence on the leak worlds "
            "*evidence*: the full-sample 'edge' really is nothing but look-ahead. For the **generic** "
            "mis-timing leak see [347 Look-Ahead Bias](../../347-look-ahead-bias/); for faking a Sharpe "
            "by **searching**, [344 Backtest-Overfitting](../../344-backtest-overfitting/); by "
            "**reshaping returns**, [590 Sharpe-Hacking](../../590-sharpe-hacking/)."
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
