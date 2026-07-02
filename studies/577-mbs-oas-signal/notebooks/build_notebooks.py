"""Generate the two narrative notebooks for Study 577 (MBS-OAS-Signal).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). This is a **synthetic-only**
study — there is no free agency-MBS OAS tape (ICE BofA / Bloomberg are licensed; FRED has mortgage
*yields*, not OAS). So every code cell runs on the deterministic synthetic world, offline, and the
numbers in ``R`` mirror docs/results.md. ``HAVE_REAL`` gates any (never-present) real-tape cell; the
synthetic cells never sit under a real-tape banner.
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


# Frozen synthetic headline numbers — mirror of docs/results.md (as-of 2026-06-30; seed 577;
# planted lead_beta = -0.9; 779 weekly rows; panel fp 5a74734302c4). All reproducible offline.
R = dict(
    fp_panel="5a74734302c4", n_weeks=779, start="2011-01-07", end="2025-12-05",
    lead_beta=-0.9,
    slope=-0.88, slope_t=-11.41, corr=-0.38, r2=0.143, placebo=0.0005, n=779,
    null_slope=0.020, null_slope_t=0.26, null_placebo=0.80,
    bh_cagr=8.8, timed_gross_cagr=16.7, timed_net_cagr=16.2,
    bh_sharpe=0.59, timed_gross_sharpe=1.10, timed_net_sharpe=1.07,
    weeks_in_cash=68, switches=131, cost_bps=5.0, widen_z=1.0,
    # robustness: (label, slope, slope_t)
    sweep=[("weekly change (headline)", -0.880, -11.41),
           ("level z-score", -0.024, -0.29),
           ("4-week change", -0.352, -4.25)],
    # synthetic control: (lead_beta, mean slope-t over 25 seeds)
    ctrl=[(0.0, 0.04), (-0.3, -3.74), (-0.6, -7.53), (-0.9, -11.32), (-1.5, -18.90)],
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

from mbs_oas_signal import data, strategy as st

# There is NO free MBS-OAS tape (ICE BofA / Bloomberg are licensed; FRED has yields, not OAS).
# fetch_series(fetch=False) returns an empty frame -> HAVE_REAL is always False here.
REAL = data.fetch_series(fetch=False)
HAVE_REAL = not REAL.empty
print("real MBS-OAS tape present:", HAVE_REAL, "(synthetic-only study — no free OAS series exists)")

# The deterministic synthetic world with the lead PLANTED (lead_beta = -0.9), seed 577.
FRAME, TRUTH = data.synthetic_series()
print(f"synthetic world: {len(FRAME)} weekly rows, {FRAME.index[0].date()} -> {FRAME.index[-1].date()}, "
      f"planted lead_beta={TRUTH.lead_beta}")
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# The Mortgage-Spread Warning — does MBS OAS see trouble coming? 🏠\n"
            "### Do mortgage bonds' extra yield warn of trouble in stocks and credit, in plain English\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n\n"
            "Mortgage bonds (MBS) pay a little *extra* yield over safe government bonds. Strip out the "
            "quirk that homeowners can refinance early, and what's left is the **option-adjusted "
            "spread** (OAS) — the pure 'extra compensation for risk' in mortgage bonds. The folklore: "
            "when that spread **widens**, mortgage investors are getting nervous, and that nervousness "
            "shows up in *stocks and corporate bonds* soon after. So a widening OAS is supposed to be "
            "an early **warning light**.\n\n"
            "There's a catch we have to be honest about up front: **you can't get this number for "
            "free.** The clean OAS series live behind expensive data licences (ICE BofA, Bloomberg); "
            "the free public feeds only give you mortgage *rates*, not the option-adjusted spread. So "
            "we can't test the real thing. Instead we build a *make-believe* world where the warning "
            "light either works or doesn't, and check that our measuring stick can tell the "
            "difference.\n\n"
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
            "| Could a widening mortgage spread warn of trouble? | **In principle, yes.** In a world "
            "where we *plant* the effect, a widening this week reliably precedes weaker returns next "
            f"week — our measuring stick catches it loud and clear (*t* {R['slope_t']}). |\n"
            "| Did we test it on real data? | **No — we couldn't.** There is no free OAS series to "
            "test. The real number is licensed; the free feeds give mortgage rates, not the spread. |\n"
            "| So is it a real signal? | **Unproven — Weak.** The story is coherent and the machinery "
            "works, but with no real tape we can't earn a strong verdict. |\n"
            "| Could you trade it? | **No — Mirage.** Even the raw ingredient (the OAS feed) is "
            "unreachable for a normal investor, arrives late, and the mortgage index is costly to "
            "trade. |\n\n"
            "> The honest wall here isn't the *idea* — it's the *data*. A signal you can neither test "
            "for free nor trade cheaply is a signal on paper only."
        ),

        # ---- BEAT 1 — THE CLAIM ----
        md(
            "## 1 · The claim\n\n"
            "> *\"When the option-adjusted spread on mortgage bonds widens, risk-off is coming — sell "
            "stocks and credit.\"*\n\n"
            "Why it's plausible: mortgage bonds are held by specialised, leveraged investors. When "
            "those investors' balance sheets get squeezed, they demand more yield to hold mortgage "
            "risk — the OAS widens — and the *same* squeeze soon hits every other risky asset. So the "
            "mortgage market, being sensitive and specialised, is supposed to *flinch first*."
        ),

        # ---- BEAT 2 — SO WHAT ----
        md(
            "## 2 · So what?\n\n"
            "If the warning light were real *and* usable, you'd have a cheap macro early-warning: "
            "watch one spread, and de-risk before the crowd. The desk has looked at the *corporate* "
            "version of this idea ([115 Credit-Spreads](../../115-credit-spreads/)); here we go at the "
            "**mortgage** version specifically. The twist is that the mortgage version runs straight "
            "into a data wall the corporate one doesn't."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ----
        md(
            "## 3 · How would we even know?\n\n"
            "1. **Measure the signal.** Each week, how much did the mortgage spread *widen* (compared "
            "to its usual wobble)? A big widening is a big warning.\n"
            "2. **Look forward one week.** Did the risk asset do *worse* the following week? The claim "
            "says a widening → a weaker next week.\n"
            "3. **Build a switch.** De-risk to cash the week after a big widening, and see if that "
            "dodges losses.\n\n"
            "Catch: we can only run this on a **synthetic** mortgage-spread world, because the real "
            "spread isn't free. So we make our world two ways — one where the warning light genuinely "
            "works, one where it's pure noise — and check the measuring stick tells them apart.\n\n"
            "*Timing:* the signal each week uses only what's known by that week's close; the return "
            "it's judged against is the **next** week's. One clean step — no peeking ahead."
        ),

        # ---- BEAT 4 — THE TEARDOWN ----
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**Here's a synthetic mortgage-spread path — quiet most of the time, with a few stress "
            "spikes — and the risk asset it's meant to lead.**"
        ),
        code(
            "fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 5.6), sharex=True)\n"
            "ax1.plot(FRAME.index, FRAME['oas'], color=RED, lw=1.2)\n"
            "ax1.set_ylabel('MBS OAS (bps)'); ax1.set_title('Synthetic mortgage spread — quiet, then stress spikes')\n"
            "cum = (1 + FRAME['ret'].fillna(0)/100).cumprod()\n"
            "ax2.plot(FRAME.index, cum, color=GREEN, lw=1.2)\n"
            "ax2.set_ylabel('risk asset (growth of 1)'); ax2.set_title('The risk asset it is meant to lead')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'OAS ranges {FRAME.oas.min():.0f} - {FRAME.oas.max():.0f} bps over {len(FRAME)} weeks')"
        ),
        md(
            "The spread mean-reverts around a calm level and jumps in the odd stress week — just like "
            "the real thing does in 2008, 2020 or 2022. Now the key question: **when the spread "
            "widens, does the risk asset do worse next week?**"
        ),
        code(
            "reg = st.predictive_regression(FRAME)\n"
            "x = FRAME['d_oas_z']; y = FRAME['fwd_ret']\n"
            "fig, ax = plt.subplots(figsize=(8, 4.6))\n"
            "ax.scatter(x, y, s=14, color=GREY, alpha=.6)\n"
            "xs = np.linspace(x.min(), x.max(), 50)\n"
            "ax.plot(xs, reg['slope']*xs + (y.mean() - reg['slope']*x.mean()), c=RED, lw=2)\n"
            "ax.axhline(0, c='k', lw=.8); ax.axvline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('OAS widening this week (standardised)'); ax.set_ylabel('risk-asset return NEXT week %')\n"
            "ax.set_title(f'Widening -> weaker next week: slope {reg[\"slope\"]:+.2f} (t {reg[\"slope_t\"]:+.1f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'slope {reg[\"slope\"]:+.2f} %pt/sd, t {reg[\"slope_t\"]:+.1f}, corr {reg[\"corr\"]:+.2f}')"
        ),
        md(
            f"The line slopes **down**: a bigger widening this week goes with a *weaker* return next "
            f"week (slope {R['slope']}, *t* {R['slope_t']}). That's the warning light working — **in "
            "the world where we planted it.** The whole point of planting it is to prove our measuring "
            "stick isn't fooling us; next we check it stays silent when there's nothing there."
        ),
        md(
            "**Does the switch — go to cash after a widening — actually dodge losses?**"
        ),
        code(
            "ov = st.timing_overlay(FRAME, widen_z=1.0, cost_bps=5.0)\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.3))\n"
            "ax.bar(['buy & hold','risk-off switch\\n(net)'], [ov['bh_sharpe'], ov['timed_net_sharpe']],\n"
            "       color=[GREY, GREEN], width=.5)\n"
            "ax.set_ylabel('Sharpe ratio'); ax.set_title('The risk-off switch improves risk-adjusted return (planted world)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"buy&hold Sharpe {ov['bh_sharpe']:.2f} -> switch (net) {ov['timed_net_sharpe']:.2f}; \"\n"
            "      f\"{ov['weeks_in_cash']} weeks in cash, {ov['switches']} switches\")"
        ),
        md(
            f"Stepping to cash after a widening lifts the Sharpe from **{R['bh_sharpe']}** to "
            f"**{R['timed_net_sharpe']} net of costs** — the switch dodges the bad weeks. Again: this "
            "is the world where the light *works*. That's what makes the real verdict about **data**, "
            "not about the idea."
        ),

        # ---- BEAT 5 — VERDICT ----
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — Weak.** The story holds together and our engine proves the effect *would* be "
            "detectable — but with **no free OAS tape** to test, we can't earn a strong verdict.\n"
            "- **Tradability — Mirage.** The raw ingredient is licensed, arrives late, and the "
            "mortgage index is costly to trade. A signal you can't cheaply *get* isn't tradable."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ----
        md(
            "## 6 · Could you actually trade it?\n\n"
            "No. Three walls, all about *access*: (1) the OAS number is a paid vendor product — a "
            "retail investor never sees it live; (2) it's published with a lag, so by the time you'd "
            "act, the move is old; (3) trading the mortgage index itself is not cheap or clean for a "
            "small account. The idea might be fine; the plumbing isn't there."
        ),

        # ---- BEAT 7 — GOING FURTHER ----
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The corporate cousin.** [115 Credit-Spreads](../../115-credit-spreads/) tests the "
            "*corporate* HY/IG spread as a state variable — same family, reachable data.\n"
            "- **Other canaries.** [131 Utilities-Canary](../../131-utilities-canary/) and "
            "[111 VIX-Term-Structure](../../111-vix-term-structure/) are the 'one market warns the "
            "rest' idea on assets you *can* get for free.\n\n"
            "*Have a licensed MBS-OAS series? Drop a weekly parquet into `_cache/mbs_oas.parquet`, "
            "flip `HAVE_REAL`, and re-run — the engine is ready for a real tape.*"
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
            "# The MBS-OAS Risk-Off Lead — a quantitative teardown 🔬\n"
            "### OAS-change signal · predictive OLS + *t* · label-shuffle placebo · risk-off timing overlay · change-vs-level robustness · seed-robust synthetic control\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We test whether the "
            "standardised weekly change in agency-MBS OAS **leads** risk-asset returns with a negative "
            "sign (the risk-off folklore).\n\n"
            "> ⚠️ **Not investment advice.** This is a **synthetic-only** study: no free agency-MBS OAS "
            "series exists (ICE BofA / Bloomberg are licensed; FRED has mortgage *yields*, not OAS). "
            f"Every number runs on the deterministic synthetic world (seed 577, planted `lead_beta = "
            f"{R['lead_beta']}`, panel fp `{R['fp_panel']}`). Methods in "
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
            f"| **Signal** | `WEAK` | Planted world: forward-return-on-OAS-change slope **{R['slope']} "
            f"%pt/sd** (*t* **{R['slope_t']}**, placebo *p* **{R['placebo']}**), flat at the null (*t* "
            f"**+{R['null_slope_t']}**, *p* **{R['null_placebo']}**). But **no real OAS tape** ⇒ no *t* "
            "≥ 2 on a genuine tape ⇒ capped below `REAL`. |\n"
            f"| **Tradability** | `MIRAGE` | OAS is a licensed feed, published with a lag; the MBS "
            f"index is costly to trade. Overlay lifts Sharpe **{R['bh_sharpe']} → "
            f"{R['timed_net_sharpe']} net** *on the planted world* only. |\n\n"
            "> 💡 In plain words: the engine works (the synthetic control proves it), the folklore is "
            "coherent — but the data to certify *or* trade it is behind a licence wall."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $\\Delta_t$ be the standardised weekly change in MBS OAS (a widening is positive) and "
            "$r_{t+1}$ the risk-asset return over the following week.\n\n"
            "- **H₁ (the lead).** slope of $r_{t+1}$ on $\\Delta_t$ is $< 0$ at $|t| \\ge 2$ (widening "
            "→ weaker next week).\n"
            "- **H₂ (not noise).** the slope-$t$ sits in the tail of a label-shuffle null.\n"
            "- **H₃ (tradable).** a de-risk-on-widening overlay beats buy-and-hold *net of costs*.\n"
            "- **H₄ (robust).** the sign holds across signal definitions (change vs level vs 4-week).\n"
            "- **H₀ (data).** a `REAL` stamp needs a robust $t \\ge 2$ on a **genuine** OAS tape.\n\n"
            "On the planted synthetic world **H₁–H₄ all pass**; **H₀ fails by construction** — there "
            "is no free OAS tape — so the study is capped at `WEAK`."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The content is the **why the wall is the data, not the idea**. The MBS market is held by "
            "specialised leveraged investors (Gabaix-Krishnamurthy-Vigneron 2007), so its spread is a "
            "sensitive intermediary-capital gauge (He-Kelly-Manela 2017) that *could* lead risk "
            "assets — the same channel as the excess bond premium leading activity "
            "(Gilchrist-Zakrajšek 2012). But the **option-adjusted** spread requires a prepayment "
            "model and a licensed index; there is no free equivalent. So the interesting question "
            "collapses into an access question — mapped next onto the machinery."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Signal.** $\\Delta_t = $ standardised weekly change in OAS (known at week-$t$ close).\n"
            "- **Predictive OLS.** $r_{t+1} = \\alpha + \\beta \\Delta_t + \\varepsilon$; the *sign* "
            "of $\\beta$ is the claim, reported with its *t* and in-sample $R^2$.\n"
            "- **Placebo.** shuffle $\\Delta$ against $r_{t+1}$ (2000 perms); read the |slope-$t$| tail.\n"
            "- **Overlay.** de-risk to cash the week *after* $\\Delta_t > 1$ sd; costs 5 bps/switch.\n"
            "- **Robustness.** re-estimate $\\beta$ for the level-$z$ and the 4-week change.\n"
            "- **Positive control.** plant $\\text{lead\\_beta} < 0$ of increasing strength; average "
            "the slope-$t$ over 25 seeds (house rule) — flat at the null, monotone as it grows.\n\n"
            "Timing: $\\Delta_t$ uses only week-$t$-and-earlier data; it is judged on $r_{t+1}$. One "
            "explicit execution lag — the overlay never acts on a return it couldn't have known."
        ),

        # ---- BEAT 4 ----
        md("## 4 · The teardown"),
        md(
            "### 4a · The predictive relation — H₁ & H₂\n\n"
            "Regress next-week return on this week's standardised OAS change; a *negative* slope is the "
            "lead. Then the label-shuffle placebo."
        ),
        code(
            "reg = st.predictive_regression(FRAME)\n"
            "p = st.placebo_pvalue(FRAME, n_perm=2000, seed=577)\n"
            "x = FRAME['d_oas_z']; y = FRAME['fwd_ret']\n"
            "fig, ax = plt.subplots(figsize=(8, 4.6))\n"
            "ax.scatter(x, y, s=14, color=GREY, alpha=.55)\n"
            "xs = np.linspace(x.min(), x.max(), 50)\n"
            "ax.plot(xs, reg['slope']*xs + (y.mean() - reg['slope']*x.mean()), c=RED, lw=2)\n"
            "ax.axhline(0, c='k', lw=.8); ax.axvline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('standardised weekly OAS change $\\\\Delta_t$'); ax.set_ylabel('forward return $r_{t+1}$ %')\n"
            "ax.set_title(f'slope {reg[\"slope\"]:+.2f} %pt/sd (t {reg[\"slope_t\"]:+.1f}), R2 {reg[\"r2\"]:.3f}, placebo p {p:.4f}')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'slope {reg[\"slope\"]:+.3f}, t {reg[\"slope_t\"]:+.2f}, corr {reg[\"corr\"]:+.2f}, R2 {reg[\"r2\"]:.3f}, placebo p {p:.4f}')"
        ),
        md(
            f"> 💡 In plain words: H₁ & H₂ **pass on the planted world.** Slope **{R['slope']} %pt/sd** "
            f"(*t* {R['slope_t']}), placebo *p* = {R['placebo']} — the widening genuinely leads a "
            "weaker next week, and it's nowhere near the shuffled null. The engine is faithful."
        ),
        md(
            "### 4b · The null — no false positive\n\n"
            "Switch the lead off (`lead_beta = 0`) and re-run: the slope-*t* must collapse and the "
            "placebo *p* must go boring."
        ),
        code(
            "null_frame, _ = data.synthetic_series(lead_beta=0.0)\n"
            "nreg = st.predictive_regression(null_frame)\n"
            "np_ = st.placebo_pvalue(null_frame, n_perm=2000, seed=577)\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.2))\n"
            "ax.bar(['planted\\n(lead_beta -0.9)','null\\n(lead_beta 0)'], [reg['slope_t'], nreg['slope_t']],\n"
            "       color=[RED, GREY], width=.5)\n"
            "ax.axhline(-2, ls='--', c=AMBER, lw=1, label='t = -2'); ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('predictive slope-t'); ax.legend()\n"
            "ax.set_title('Faithful detector: strong when planted, ~0 at the null')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'null slope {nreg[\"slope\"]:+.3f}, slope-t {nreg[\"slope_t\"]:+.2f}, placebo p {np_:.2f}')"
        ),
        md(
            f"> 💡 In plain words: at the null the slope-*t* is **+{R['null_slope_t']}** and the placebo "
            f"*p* is a boring **{R['null_placebo']}** — no hallucinated signal. The engine only fires "
            "when there's something to fire on."
        ),
        md(
            "### 4c · The risk-off overlay — H₃ (net of costs)\n\n"
            "De-risk to cash the week after a >1 sd widening; charge 5 bps per switch."
        ),
        code(
            "ov = st.timing_overlay(FRAME, widen_z=1.0, cost_bps=5.0)\n"
            "fig, ax = plt.subplots(figsize=(8, 4.3))\n"
            "labs = ['buy & hold','timed gross','timed net']\n"
            "shp = [ov['bh_sharpe'], ov['timed_gross_sharpe'], ov['timed_net_sharpe']]\n"
            "cag = [ov['bh_cagr']*100, ov['timed_gross_cagr']*100, ov['timed_net_cagr']*100]\n"
            "ax.bar(labs, shp, color=[GREY, GREEN, GREEN], width=.5, alpha=.85)\n"
            "ax.set_ylabel('Sharpe'); ax.set_title(f'Overlay Sharpe {ov[\"bh_sharpe\"]:.2f} -> {ov[\"timed_net_sharpe\"]:.2f} net; CAGR {cag[0]:.1f}% -> {cag[2]:.1f}%')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"CAGR: BH {cag[0]:.1f}% | gross {cag[1]:.1f}% | net {cag[2]:.1f}%\")\n"
            "print(f\"Sharpe: BH {shp[0]:.2f} | gross {shp[1]:.2f} | net {shp[2]:.2f}; \"\n"
            "      f\"{ov['weeks_in_cash']} wks cash, {ov['switches']} switches\")"
        ),
        md(
            f"> 💡 In plain words: H₃ **passes on the planted world** — Sharpe {R['bh_sharpe']} → "
            f"{R['timed_net_sharpe']} net, CAGR {R['bh_cagr']}% → {R['timed_net_cagr']}%, and costs "
            f"barely dent it ({R['switches']} switches). But this is the world where the lead is real."
        ),
        md(
            "### 4d · Robustness — H₄ (the change leads, the level doesn't)\n\n"
            "Re-estimate the slope for three signal definitions."
        ),
        code(
            "rows = st.robustness_sweep(FRAME)\n"
            "tab = pd.DataFrame(rows, columns=['signal','slope','slope_t']).set_index('signal')\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.2))\n"
            "cols = [RED if t < -2 else GREY for t in tab['slope_t']]\n"
            "ax.barh(range(len(tab)), tab['slope_t'], color=cols)\n"
            "ax.set_yticks(range(len(tab))); ax.set_yticklabels(tab.index)\n"
            "ax.axvline(-2, ls='--', c=AMBER, lw=1); ax.axvline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('predictive slope-t'); ax.set_title('The CHANGE leads (t << -2); the LEVEL is flat')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(tab.round(3).to_string())"
        ),
        md(
            "> 💡 In plain words: H₄ **passes with a caveat.** The weekly *change* leads strongly "
            f"(*t* {R['sweep'][0][2]}) and the 4-week change too (*t* {R['sweep'][2][2]}), but the "
            f"*level* is flat (*t* {R['sweep'][1][2]}) — it's the widening *move*, not a wide *level*, "
            "that carries the signal. An honest nuance the generator faithfully preserves."
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — H₁–H₄ pass on the planted world (slope {R['slope']}, *t* "
            f"{R['slope_t']}, placebo *p* {R['placebo']}), but **H₀ fails**: no free OAS tape, so no "
            "robust *t* ≥ 2 on a genuine tape. Capped below `REAL`.\n"
            "- **Tradability `MIRAGE`** — the OAS feed is licensed and lagged; the MBS index is costly "
            "to trade. The overlay's edge lives only in the planted world.\n\n"
            "> The honest wall is **data availability**, stated on the SIGNAL axis (as in the desk's "
            "lego-returns / whisky-cask / sneaker-resale synthetic-only studies)."
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you trade it? — the access wall\n\n"
            "Three access frictions, none of which a working signal can overcome: the OAS number is a "
            "**licensed** vendor index (no retail feed), it is **published with a lag** (so acting on "
            "it is acting late), and the **MBS index itself is costly** for a small book to trade. "
            "The overlay's net Sharpe above is a property of the *synthetic* world, not a live "
            "opportunity — which is precisely the `MIRAGE`."
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further — the seed-robust synthetic positive control\n\n"
            "Is the engine a faithful detector, or does it always print a negative *t*? Plant a lead "
            "of increasing strength and watch the slope-*t* fall monotonically — averaged over 25 "
            "seeds so no single lucky seed can fake it."
        ),
        code(
            "betas = [0.0, -0.3, -0.6, -0.9, -1.5]\n"
            "ts = [st.synthetic_mean_t(data, lead_beta=b, n_seeds=25) for b in betas]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(betas, ts, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(-2, ls='--', c=AMBER, lw=1, label='t = -2 bar (lead)')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('planted lead_beta (more negative = stronger lead)')\n"
            "ax.set_ylabel('mean predictive slope-t (25 seeds)')\n"
            "ax.set_title('Flat at the null, monotonically past -2 as the planted lead grows')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for b, t in zip(betas, ts): print(f'lead_beta {b:+.2f} -> mean slope-t {t:+.2f}')"
        ),
        md(
            "The slope-*t* is ≈ 0 at the null and falls past −2 as the planted lead grows — a faithful "
            "detector. So the honest verdict is not about the code: it is that **there is no free OAS "
            "tape to test and no reachable OAS feed to trade**. `WEAK` (coherent + faithful engine, no "
            "real tape) × `MIRAGE` (unreachable input). For the reachable corporate cousin, see "
            "[115 Credit-Spreads](../../115-credit-spreads/)."
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
