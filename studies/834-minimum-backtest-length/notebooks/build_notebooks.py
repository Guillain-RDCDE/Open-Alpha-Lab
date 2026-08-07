"""Generate the two narrative notebooks for Study 834 (Minimum Backtest Length / MinTRL).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. This is a synthetic-only method demo — there is NO
real-tape cell (real free data can never certify "true Sharpe = 0"), so the study is capped at NONE.
The heavy headline numbers are quoted from the frozen ``R`` dict (mirror of docs/results.md); the
live cells run only the fast, analytic MinTRL curve and a modest 2,000-path Monte-Carlo, so
execution is quick and network-free.
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


# Frozen headline numbers — mirror of docs/results.md (as-of 2026-06-30; fingerprint c648fb3ad2b5;
# seed 834, daily freq 252, conf 0.95, ann-vol 0.15, 4,000 sims). Live cells recompute the fast bits.
R = dict(
    fp="c648fb3ad2b5", conf=0.95, Z=1.6449, freq=252, ann_vol=0.15,
    # MinTRL headline (Gaussian daily 95%): (SR, MinTRL_yr, rule_of_thumb_yr)
    mtl=[(2.0, 0.69, 0.68), (1.0, 2.71, 2.71), (0.5, 10.83, 10.82), (0.25, 43.30, 43.29)],
    # skew/kurt monthly SR=1: (label, skew, kurt, MinTRL_yr, x_vs_gaussian)
    skewm=[("Gaussian", 0.0, 3.0, 2.90, 1.00),
           ("skew -1, kurt 4.5", -1.0, 4.5, 3.77, 1.30),
           ("skew -2, kurt 9 (fat left tail)", -2.0, 9.0, 4.80, 1.65)],
    # short-backtest luck (true Sharpe 0)
    luck_1yr=15.6, luck_2yr=8.4, best_1yr=3.40, best_2yr=2.45, disp_sd_2yr=0.717,
    # calibration on the null: (years, reject_pct)
    cal=[(1.0, 5.60), (2.0, 5.65), (5.0, 5.27)],
    # positive control SR=1
    mtl1=2.71, pow1=10.82,
    power=[(1.0, 25.3), (2.71, 50.0), (5.0, 70.8), (10.82, 95.0)],
)


BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from min_backtest_length import data, strategy as st
Z = float(stats.norm.ppf(0.95))
print("synthetic-only method demo | daily freq %d, conf 0.95 (Z=%.4f), ann-vol %.2f, seed %d | fp %s"
      % (data.TRADING_DAYS, Z, data.ANN_VOL, data.BASE_SEED, data.config_fingerprint()))
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Minimum Backtest Length — how long must a track record be to trust a Sharpe? 📅\n"
            "### Why a five-year backtest usually *can't* prove your strategy works, in plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Short_backtests_fool_you%3F: Confirmed](https://img.shields.io/badge/Short_backtests_fool_you%3F-Confirmed-8b949e?style=flat-square)\n\n"
            "Every strategy pitch leads with a **Sharpe ratio** — return per unit of risk — measured on "
            "a backtest. Almost none tells you the one thing that makes it interpretable: **how long the "
            "backtest is.** A Sharpe measured over a short history is a *noisy* number, and below a "
            "certain length any Sharpe, however gorgeous, is statistically indistinguishable from a coin "
            "flip.\n\n"
            "Bailey, Borwein & López de Prado (2014) turned that into a hard rule: the **Minimum Track "
            "Record Length (MinTRL)** — the shortest history over which a Sharpe of a given size clears a "
            "95% significance bar. We'll see it explode as the Sharpe falls, watch a *worthless* "
            "strategy post a brilliant backtest by pure luck, and confirm that even a *genuine* edge "
            "hides in a short window.\n\n"
            "> 📓 **This is the plain-language layer.** Want the MinTRL/PSR formulas, the skew/kurtosis "
            "correction and the Monte-Carlo? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool: every chart is drawn by the "
            "code beside it, on a deterministic **synthetic** world (no real data — the only way to fix "
            "the truth). House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ----
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| How long to trust a **Sharpe-1** backtest? | **~{R['mtl1']} years** (95% confidence, "
            "Gaussian daily). |\n"
            f"| A **Sharpe-0.5** idea (typical of real strategies)? | **~{R['mtl'][2][1]:.0f} years.** "
            "Halve the Sharpe, *quadruple* the wait. |\n"
            f"| A **Sharpe-0.25** idea? | **~{R['mtl'][3][1]:.0f} years** — longer than most careers. |\n"
            f"| Can a *worthless* strategy fake a great backtest? | **Yes.** Over 2 years, "
            f"**{R['luck_2yr']}%** of zero-edge strategies post a Sharpe ≥ 1.0 by luck. |\n"
            f"| Do fat tails / crash risk make it worse? | **Yes** — a negatively-skewed, fat-tailed "
            f"track needs **×{R['skewm'][2][4]}** longer. |\n\n"
            "> The rule of thumb: you need about **(Z / Sharpe)² years** of history — and if you have "
            "less, the number on the slide is luck, not skill."
        ),

        # ---- BEAT 1 — THE CLAIM ----
        md(
            "## 1 · The claim\n\n"
            "> *\"Given a target Sharpe ratio, there is a minimum length of track record below which you "
            "cannot tell a real edge from luck.\"*\n\n"
            "A Sharpe ratio is your average return divided by how much it wobbles. Measure it over a "
            "handful of years and the estimate itself wobbles — a lot. The shorter the record, the wider "
            "the error bar, until the bar is so wide it comfortably includes **zero** (no skill at all)."
        ),

        # ---- BEAT 2 — SO WHAT ----
        md(
            "## 2 · So what?\n\n"
            "Because allocators wire billions on the strength of a backtest. If a mediocre, no-edge "
            "strategy can show a Sharpe of 1 over two years *by chance*, then the number is nearly "
            "worthless without knowing how long — and how the returns are shaped. The desk has shown you "
            "can fake a Sharpe by **searching** many rules "
            "([344 Backtest-Overfitting](../../344-backtest-overfitting/)); here the trap is simpler and "
            "even more common — a **single honest strategy** with a track record that is just **too "
            "short**."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ----
        md(
            "## 3 · How would we even know?\n\n"
            "We can't ask a real strategy 'do you truly have edge?' — but in a **built** world we know "
            "the answer. So we make two: a **worthless** tape (true Sharpe exactly 0) and a **skilful** "
            "one (true Sharpe 1). Then we ask how long a backtest has to be before the maths can tell "
            "them apart. The MinTRL formula gives the threshold; a Monte-Carlo of thousands of backtests "
            "checks it holds."
        ),

        # ---- BEAT 4 — THE TEARDOWN ----
        md(
            "## 4 · The teardown — the length you need explodes as the Sharpe falls\n\n"
            "The MinTRL is analytic and instant to compute. Plot it against the target Sharpe (95% "
            "confidence, Gaussian daily returns):"
        ),
        code(
            "sr = np.linspace(0.2, 2.5, 200)\n"
            "mtl = st.min_trl_curve(sr, freq=data.TRADING_DAYS, conf=0.95)\n"
            "fig, ax = plt.subplots(figsize=(9, 4.8))\n"
            "ax.plot(sr, mtl, c=RED, lw=2.5, label='MinTRL (exact)')\n"
            "ax.plot(sr, (Z/sr)**2, '--', c=GREY, lw=1.5, label='rule of thumb (Z/SR)^2')\n"
            "for s, yr, _ in [(2.0, 0.69, 0), (1.0, 2.71, 0), (0.5, 10.83, 0), (0.25, 43.30, 0)]:\n"
            "    ax.scatter([s], [yr], color=RED, zorder=5)\n"
            "    ax.annotate(f'SR {s}: {yr:.1f} yr', (s, yr), textcoords='offset points',\n"
            "                xytext=(8, 6), fontsize=9)\n"
            "ax.set_xlabel('target annualised Sharpe'); ax.set_ylabel('years of track record needed')\n"
            "ax.set_ylim(0, 50); ax.set_title('Minimum backtest length to clear a 95% bar')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for s in (2.0, 1.0, 0.5, 0.25):\n"
            "    print(f'Sharpe {s:>4}:  need {st.min_trl_years(s, conf=0.95):6.2f} years')"
        ),
        md(
            f"There it is. A **Sharpe-1** strategy needs **{R['mtl1']} years**; a **Sharpe-0.5** one "
            f"(the honest reality for most systematic strategies) needs **{R['mtl'][2][1]:.0f} years**; "
            f"a **Sharpe-0.25** one needs **{R['mtl'][3][1]:.0f} years**. The curve is a *quadratic* — "
            "halving the Sharpe quadruples the wait — because the noise in a Sharpe estimate only shrinks "
            "with the square root of the length."
        ),
        md(
            "**Now the punchline: a worthless strategy can look brilliant in a short window.** We draw "
            "4,000 two-year backtests of a tape with *zero* real edge and count how many post a Sharpe "
            "≥ 1.0 by luck:"
        ),
        code(
            "lp2 = st.luck_prob(data, threshold_sr=1.0, n_years=2.0, n_sims=4000, seed=834)\n"
            "lp1 = st.luck_prob(data, threshold_sr=1.0, n_years=1.0, n_sims=4000, seed=834)\n"
            "sim0 = st.simulate(data, sr_ann_true=0.0, n_years=2.0, n_sims=4000, seed=834)\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.4))\n"
            "ax.hist(sim0['sr_obs'], bins=50, color=GREY, alpha=.7)\n"
            "ax.axvline(0, c='k', lw=1, label='TRUE Sharpe = 0 (no edge)')\n"
            "ax.axvline(1.0, c=RED, lw=2, ls='--', label='a Sharpe-1 winner (by luck)')\n"
            "ax.set_xlabel('observed Sharpe of a 2-year backtest'); ax.set_ylabel('count')\n"
            "ax.set_title('A worthless strategy, 4000 short backtests: luck alone reaches Sharpe 1+')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"over 1 year : {lp1['frac']*100:5.1f}% of WORTHLESS backtests post Sharpe >= 1.0 \"\n"
            "      f\"(luckiest {lp1['best_sr']:.2f})\")\n"
            "print(f\"over 2 years: {lp2['frac']*100:5.1f}% of WORTHLESS backtests post Sharpe >= 1.0 \"\n"
            "      f\"(luckiest {lp2['best_sr']:.2f})\")\n"
            "print(f\"observed-Sharpe scatter at 2yr: sd {sim0['sd_obs_sr']:.2f} around 0\")"
        ),
        md(
            f"Over two years, **{R['luck_2yr']}%** of these zero-skill strategies clear a Sharpe of 1.0 "
            f"— and the luckiest posts **{R['best_2yr']}**. The observed Sharpe of a *worthless* two-year "
            f"backtest scatters with a standard deviation of **{R['disp_sd_2yr']}** around zero. If you "
            "run enough short backtests, a gorgeous one is *guaranteed*, with no skill involved."
        ),

        # ---- BEAT 5 — VERDICT ----
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** On a world built with zero edge there is nothing real to find; the "
            "only pretty backtest is luck.\n"
            "- **Tradability — Mirage.** A backtest shorter than its MinTRL is a coin flip — nothing to "
            "harvest.\n"
            "- **Do short backtests fool you? — Confirmed.** The length you need grows as `(Z/SR)²` "
            "(43 years at Sharpe 0.25), and a worthless strategy fakes a Sharpe-1 backtest 8% of the "
            "time over two years."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ----
        md(
            "## 6 · Could you actually 'trade' it?\n\n"
            "There is nothing here to trade — that *is* the point. Before you allocate to any backtest, "
            "compute its MinTRL: if the history is shorter than `(Z/Sharpe)²` years, the Sharpe on the "
            "slide cannot be told apart from luck, and no amount of execution skill changes that. The "
            "honest move is to demand a longer record, a lower expectation, or both."
        ),

        # ---- BEAT 7 — GOING FURTHER ----
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The searching cousin.** [344 Backtest-Overfitting](../../344-backtest-overfitting/) and "
            "[833 Deflated-Sharpe](../../833-deflated-sharpe/) fake a Sharpe by *trying many strategies* "
            "and keeping the luckiest; this study shows even **one** honest strategy needs a long enough "
            "record.\n"
            "- **The fat-tail twist (quants notebook).** Negative skew and fat tails — the profile of "
            "anything that sells crash insurance — make the required length *longer* still.\n\n"
            "*Have a backtest you love? Fork this, plug in its Sharpe and length, and check whether it "
            "even clears its own MinTRL before you believe it.*"
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
            "# Minimum Backtest Length — a quantitative teardown 🔬\n"
            "### MinTRL & the Probabilistic Sharpe Ratio · the skew/kurtosis correction · a calibrated null Monte-Carlo · the positive-control power curve\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Short_backtests_fool_you%3F: Confirmed](https://img.shields.io/badge/Short_backtests_fool_you%3F-Confirmed-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same seven "
            "beats, every claim now carrying its standard error.* We implement the Bailey–López de Prado "
            "**MinTRL** and **Probabilistic Sharpe Ratio**, add the skew/kurtosis correction, and stress "
            "them with a 4,000-path Monte-Carlo on worlds of known truth.\n\n"
            "> ⚠️ **Not investment advice.** A synthetic-only method demo: the worlds are *built* to fix "
            f"their truth (config fp `{R['fp']}`), so real free data can never certify them and the study "
            "is capped at `NONE`. Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Built-zero-edge world; 4,000 two-year backtests, "
            f"**{R['luck_2yr']}%** post Sharpe ≥ 1.0 by luck (best {R['best_2yr']}). Synthetic-only — no "
            "real tape. |\n"
            f"| **Tradability** | `MIRAGE` | A record shorter than MinTRL is a coin flip; MinTRL(0.5) = "
            f"**{R['mtl'][2][1]:.1f} yr**, MinTRL(0.25) = **{R['mtl'][3][1]:.0f} yr** — most strategies "
            "sit inside it. |\n"
            f"| **Do short backtests fail?** | `CONFIRMED` | MinTRL ≈ (Z/SR)²; null PSR test calibrated "
            f"at **{R['cal'][1][1]:.1f}%**; a true Sharpe-1 is only **{R['power'][1][1]:.0f}%** detectable "
            f"at its {R['mtl1']}-yr MinTRL. |\n\n"
            "> 💡 In plain words: the required length is quadratic in 1/Sharpe, the correction is honest "
            "(calibrated on the null), and short records are underpowered on *genuine* edge too."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $\\widehat{SR}$ be the per-observation Sharpe over $n$ observations. Its estimation "
            "variance (Lo 2002) is $\\widehat{\\sigma}^2_{SR} = \\frac{1}{n-1}\\left(1 - \\gamma_3 SR + "
            "\\frac{\\gamma_4 - 1}{4} SR^2\\right)$, where $\\gamma_3,\\gamma_4$ are the skewness and "
            "kurtosis of the returns.\n\n"
            "- **PSR** (probability the true Sharpe beats a benchmark $SR^*$): "
            "$\\widehat{PSR}(SR^*) = \\Phi\\!\\left(\\dfrac{(\\widehat{SR} - SR^*)\\sqrt{n-1}}"
            "{\\sqrt{1 - \\gamma_3 \\widehat{SR} + \\frac{\\gamma_4-1}{4}\\widehat{SR}^2}}\\right)$.\n"
            "- **MinTRL**: solve $\\widehat{PSR} = $ conf for $n$: "
            "$\\text{MinTRL} = 1 + \\left(1 - \\gamma_3 SR + \\frac{\\gamma_4-1}{4}SR^2\\right)"
            "\\left(\\dfrac{Z_{\\text{conf}}}{SR - SR^*}\\right)^2$ observations.\n"
            "- **Rule of thumb** (Gaussian, $SR^*=0$): $\\text{MinTRL (years)} \\approx (Z/SR_{ann})^2$.\n\n"
            "We **confirm** the rule of thumb, **confirm** the skew/kurtosis inflation, and **confirm** "
            "MinTRL is exactly the length at which PSR crosses the confidence level."
        ),
        code(
            "print('MinTRL is exactly the n that makes PSR = conf:')\n"
            "for s in (0.5, 1.0, 2.0):\n"
            "    n = st.min_trl_years(s, conf=0.95)\n"
            "    psr = st.probabilistic_sharpe_ratio(s, n, sr_star_ann=0.0)\n"
            "    print(f'  SR {s}: MinTRL {n:6.2f} yr  ->  PSR at that length = {psr:.4f}')"
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what? — the mechanism\n\n"
            "The Sharpe estimate's standard error shrinks only as $1/\\sqrt{n}$, so to *halve* the error "
            "bar you need *four times* the history — and to distinguish a Sharpe $SR$ from zero at "
            "$Z_{\\text{conf}}$ sigma you need $SR\\sqrt{n_{years}} \\ge Z$, i.e. $n_{years} \\ge "
            "(Z/SR)^2$. This is the same Bailey–López de Prado family as the **Deflated Sharpe Ratio** in "
            "[344](../../344-backtest-overfitting/) / [833](../../833-deflated-sharpe/) (there: a haircut "
            "for the *number of trials*; here: the *length* one honest strategy needs)."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Worlds of known truth.** A deterministic generator (`synthetic_returns` / "
            "`synthetic_panel`) makes tapes with an exact annualised Sharpe and an exact return shape "
            "(Gaussian, or a standardised negated-gamma with *closed-form* negative skew and excess "
            "kurtosis).\n"
            "- **Closed forms.** `min_trl_years`, `probabilistic_sharpe_ratio`, `min_trl_curve`, and "
            "`min_trl_for_power` implement the formulas above.\n"
            "- **Monte-Carlo.** `simulate` runs 4,000 backtests of a world at once (vectorised), "
            "measuring each backtest's *own* Sharpe and moments and its PSR — the null rejection rate is "
            "the false-positive rate; a genuine world's is the detection power.\n"
            "- **Inference.** Wilson bands on every rejection rate; the shared Newey-West / Welch / "
            "one-sample *t* toolkit is retained.\n\n"
            "Everything is deterministic (seed 834) and offline."
        ),

        # ---- BEAT 4 ----
        md("## 4 · The teardown"),
        md(
            "### 4a · Negative skew and fat tails lengthen the requirement\n\n"
            "The moment factor $1 - \\gamma_3 SR + \\frac{\\gamma_4-1}{4}SR^2$ bites hardest at the "
            "**monthly** reporting frequency (largest per-observation Sharpe) — the realistic hedge-fund "
            "setting. A left-skewed, fat-tailed track (the profile of a crash-insurance seller) needs a "
            "longer record."
        ),
        code(
            "rows = []\n"
            "g = st.min_trl_years(1.0, freq=data.MONTHS, skew=0.0, kurt=3.0)\n"
            "for lab, sk, ku in [('Gaussian', 0.0, 3.0), ('skew -1, kurt 4.5', -1.0, 4.5),\n"
            "                    ('skew -2, kurt 9', -2.0, 9.0)]:\n"
            "    m = st.min_trl_years(1.0, freq=data.MONTHS, skew=sk, kurt=ku)\n"
            "    rows.append((lab, m, m/g))\n"
            "labels = [r[0] for r in rows]; vals = [r[1] for r in rows]\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.2))\n"
            "ax.bar(labels, vals, color=[GREY, AMBER, RED], width=.55)\n"
            "ax.set_ylabel('MinTRL (years)'); ax.set_title('Fat left tails lengthen the required track (monthly, SR=1)')\n"
            "for i, (lab, m, x) in enumerate(rows):\n"
            "    ax.annotate(f'{m:.2f} yr\\n(x{x:.2f})', (i, m), ha='center', va='bottom', fontsize=9)\n"
            "plt.tight_layout(); plt.show()\n"
            "for lab, m, x in rows:\n"
            "    print(f'{lab:<20s} MinTRL {m:5.2f} yr  (x{x:.2f} vs Gaussian)')"
        ),
        md(
            f"> 💡 In plain words: at monthly frequency a Gaussian Sharpe-1 needs "
            f"**{R['skewm'][0][3]} yr**; add a fat left tail (skew −2, kurt 9) and it climbs to "
            f"**{R['skewm'][2][3]} yr** — a **×{R['skewm'][2][4]}** penalty. The naive `(Z/SR)²` rule "
            "*understates* the requirement for exactly the strategies most prone to blow up."
        ),
        md(
            "### 4b · Calibration — the PSR test is unbiased on the null\n\n"
            "A correction is only trustworthy if it fires at its *nominal* rate on a world with no edge. "
            "Run 4,000 null backtests (true Sharpe 0) at several lengths and count PSR(0) ≥ 0.95:"
        ),
        code(
            "rows = []\n"
            "for ny in (1.0, 2.0, 5.0):\n"
            "    s = st.simulate(data, sr_ann_true=0.0, n_years=ny, n_sims=4000, conf=0.95, seed=834)\n"
            "    rows.append((ny, s['reject_frac']*100, s['reject_lo']*100, s['reject_hi']*100))\n"
            "fig, ax = plt.subplots(figsize=(8, 4.2))\n"
            "yrs = [r[0] for r in rows]; fr = [r[1] for r in rows]\n"
            "err = [[r[1]-r[2] for r in rows], [r[3]-r[1] for r in rows]]\n"
            "ax.errorbar(yrs, fr, yerr=err, fmt='o', color=GREY, ecolor=GREY, capsize=6, ms=9, lw=2)\n"
            "ax.axhline(5.0, c=RED, ls='--', lw=1.5, label='nominal 5% false-positive rate')\n"
            "ax.set_xlabel('track length (years)'); ax.set_ylabel('PSR(0)>=0.95 fires (%)')\n"
            "ax.set_ylim(0, 10); ax.set_title('The null test is calibrated at ~5% at every length'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for ny, f, lo, hi in rows:\n"
            "    print(f'{ny:>4.1f}yr: fires {f:5.2f}%  [Wilson {lo:.2f}-{hi:.2f}%]  (nominal 5.00%)')"
        ),
        md(
            f"> 💡 In plain words: the PSR test fires on **~{R['cal'][1][1]:.1f}%** of null backtests at "
            "every length — its nominal 5%, inside the Wilson band. It does not manufacture false edges "
            "(over-reject) nor bury real ones (under-reject). *(A faithful-engine check — never cited to "
            "support a real-tape stamp; there is no real tape here.)*"
        ),
        md(
            "### 4c · The positive control — genuine edge is only confirmable past the length\n\n"
            "Is the machinery just numb? Plant a **real** annualised Sharpe of 1.0 and measure how often "
            "it is *detected* (PSR ≥ 0.95) as the track grows. MinTRL(SR=1) is where an "
            "observed-equals-target Sharpe becomes significant (~50% power); reliably detecting it needs "
            "the longer 95%-power length."
        ),
        code(
            "yrs = [1.0, 2.71, 5.0, 10.82]\n"
            "pc = st.power_curve(data, sr_ann_true=1.0, year_grid=yrs, n_sims=4000, conf=0.95, seed=834)\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.4))\n"
            "err = [pc['reject_frac']-pc['reject_lo'], pc['reject_hi']-pc['reject_frac']]\n"
            "ax.errorbar(pc['years'], pc['reject_frac']*100, yerr=np.array(err)*100, fmt='o-',\n"
            "            color=GREEN, ecolor=GREY, capsize=5, ms=8, lw=2, label='detection rate (true SR=1)')\n"
            "ax.axvline(pc['min_trl'], c=AMBER, ls='--', lw=1.5, label=f\"MinTRL {pc['min_trl']:.1f}yr (~50%)\")\n"
            "ax.axvline(pc['min_trl_power'], c=RED, ls='--', lw=1.5, label=f\"95%-power {pc['min_trl_power']:.1f}yr\")\n"
            "ax.axhline(95, c=GREY, ls=':', lw=1)\n"
            "ax.set_xlabel('track length (years)'); ax.set_ylabel('detected (%)')\n"
            "ax.set_title('A GENUINE Sharpe-1 strategy: a coin flip at MinTRL, reliable only ~10.8yr'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for ny, f in zip(pc['years'], pc['reject_frac']):\n"
            "    print(f'{ny:>5.1f}yr: detected {f*100:5.1f}%')"
        ),
        md(
            f"> 💡 In plain words: a *real* Sharpe-1 strategy is detected only **{R['power'][0][1]:.0f}%** "
            f"of the time at 1 year, **{R['power'][1][1]:.0f}%** at its {R['mtl1']}-yr MinTRL, and reaches "
            f"**{R['power'][3][1]:.0f}%** only at the **{R['pow1']:.1f}-yr** power-length. Short records "
            "are underpowered in *both* directions — they cry wolf on the worthless and stay silent on "
            "the skilful."
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            "- **Signal `NONE`** — a synthetic world built with zero edge; the only pretty backtest is "
            "luck. No real tape, so never `REAL`.\n"
            "- **Tradability `MIRAGE`** — a record shorter than its MinTRL is a coin flip; nothing to "
            "harvest.\n"
            "- **Do short backtests fail? `CONFIRMED`** — MinTRL grows as `(Z/SR)²`, is lengthened by "
            "fat tails, the null test is calibrated at 5%, and a true Sharpe-1 is a coin flip at its "
            "MinTRL."
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you trade it?\n\n"
            "Nothing to trade — by construction. The operational takeaway is a *filter*: before "
            "believing any backtest, compute `MinTRL = (Z/SR)²` (inflated for skew/kurtosis), and if the "
            "history is shorter, treat the Sharpe as unproven. Costs never enter because there is no "
            "edge to cost."
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further — the desk's other backtest traps\n\n"
            "- **[344 Backtest-Overfitting](../../344-backtest-overfitting/)** and "
            "**[833 Deflated-Sharpe](../../833-deflated-sharpe/)** — the *multiple-trials* haircut "
            "(searching inflates a Sharpe; MinTRL fixes the trial count at one and charges for *length*).\n"
            "- **[345 Survivorship](../../345-survivorship/)** — a *data-construction* bias, orthogonal "
            "to the statistical length question here.\n\n"
            "*Together they answer three separate questions about a suspiciously good backtest: did you "
            "search too much (344/833), was the data rigged (345), and — this study — was it simply too "
            "short?*"
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
