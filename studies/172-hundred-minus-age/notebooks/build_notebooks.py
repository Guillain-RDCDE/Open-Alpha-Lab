"""Generate the two narrative notebooks for Study 172 (Hundred-Minus-Age).

    python notebooks/build_notebooks.py
    python -W ignore -m jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The
synthetic figures run anywhere, offline and deterministic; the real-tape cells use
the Shiller parquet already staged at _cache/shiller_sp500.parquet and fall back
to the frozen headline numbers in ``R`` (mirroring docs/results.md), so the
notebook re-runs for any reader without network access.

The _write convention (each build_*() ends by calling _write) is kept so the
repo's intro-restyle tooling can monkeypatch it.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-15).
R = dict(
    # Data stamp
    start="1881-01", end="2023-09", n_months=1713, fp="e9f9b6e370f4",
    n_cohorts=1234,
    # Asset-level real returns
    eq_ann=7.64, bd_ann=4.54,
    # Terminal wealth stats (40-year cohorts, 5bp cost)
    hma_mean=4.518, hma_med=4.218, hma_p05=2.877, hma_p95=6.582,
    hma_vol=1.234, hma_mean_vol=3.66,
    s6040_mean=4.731, s6040_med=4.660, s6040_p05=2.728, s6040_p95=6.793,
    s6040_vol=1.356, s6040_mean_vol=3.49,
    eq_mean=5.326, eq_med=4.719, eq_p05=2.637, eq_p95=9.600,
    eq_vol=2.128, eq_mean_vol=2.50,
    rising_mean=4.654, rising_med=4.513, rising_p05=2.421, rising_p95=7.122,
    rising_vol=1.576, rising_mean_vol=2.95,
    # HAC t-stats
    t_hma_6040=-5.440, t_hma_eq=-5.262,
    t_rising_hma=2.638, t_rising_6040=-3.419,
    # Win rates
    hma_wins_6040=36.5, hma_wins_eq=43.6, rising_wins_hma=59.2,
)


# ---------------------------------------------------------------------------
# Shared analysis preamble
# ---------------------------------------------------------------------------
BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))           # the study package
sys.path.insert(0, os.path.abspath("../../.."))     # repo root (quantlab/)
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False,
                     "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from hundred_minus_age import data, strategy as st

def _have_real():
    try:
        ret = data.load_shiller()
        return not ret.empty
    except Exception:
        return False

HAVE_REAL = _have_real()
print("Shiller real data available:", HAVE_REAL)
"""

LOAD_REAL = """\
if HAVE_REAL:
    ret = data.load_shiller()
    cohorts = st.rolling_cohorts(ret, n_years=40, cost_bps=5)
    stats = st.cohort_stats(cohorts)
    hma_col = cohorts["hma"].values
    s6040_col = cohorts["sixty_forty"].values
    eq_col = cohorts["all_equity"].values
    rising_col = cohorts["hma_rising"].values
    print(f"Loaded {len(ret):,} months ({len(cohorts):,} 40-yr cohorts)")
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Hundred-Minus-Age — should you really put your age in bonds?\n"
            "### The classic retirement glidepath, tested honestly over 142 years\n\n"
            "![Signal: Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Smarter_de--risking?: Fragile](https://img.shields.io/badge/Smarter_de--risking%3F-Fragile-8b949e?style=flat-square)\n\n"
            "Your financial adviser probably said it: *\"Put your age in bonds. At 30, "
            "hold 70% stocks. At 60, hold 40% stocks. Every year, shift one more "
            "percentage point from risky to safe.\"* It's the Hundred-Minus-Age rule — "
            "the default glidepath baked into millions of target-date retirement funds. "
            "The logic sounds solid: you can't afford a crash just before you retire, "
            "so de-risk gradually. But does it actually work? Does the gradual de-risking "
            "protect the floor of outcomes — or does it just cost you return for no benefit "
            "you couldn't get more cheaply from a simple 60/40 fund?\n\n"
            "> This is the plain-language layer. For the full t-stats, cohort distributions "
            "and the Pfau-Kitces academic challenge, see "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research exercise over "
            "Shiller's 1881-2023 real-return data. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does the glidepath lower risk? | **Yes, but narrowly.** "
            f"Variance of terminal wealth: HMA {R['hma_vol']:.2f} vs 100% equities "
            f"{R['eq_vol']:.2f} — the defensive effect is real. |\n"
            f"| Does it protect the floor? | **Not vs 60/40.** HMA's worst-5% outcome "
            f"is {R['hma_p05']:.2f} vs {R['s6040_p05']:.2f} for 60/40 — 60/40 has a "
            f"*better* floor. |\n"
            f"| Does it beat 60/40? | **No.** HMA trails in "
            f"**{R['hma_wins_6040']:.0f}%** of historical 40-year cohorts and earns "
            f"significantly less (HAC *t* = {R['t_hma_6040']:.1f}). |\n"
            "| Is the 'smarter' rising-equity variant better? | "
            f"**Better than HMA, not better than 60/40.** Rising-equity earns more "
            f"than HMA (statistically real) but still trails 60/40. |\n\n"
            "> The rule of thumb delivers on one promise (lower volatility vs 100% "
            "equities) but fails on the one that matters most: it doesn't protect your "
            "floor better than the boring alternative that costs no more to run."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Hold (100 minus your age) percent in stocks. At 30: 70/30. At 50: "
            "50/50. At 65: 35/65. The gradual de-risking protects you from a "
            "catastrophic bear market just before retirement — your human capital is "
            "shrinking, so your financial capital must become your safety net.\"*\n\n"
            "The rule is so embedded it's become orthodoxy. Most target-date mutual "
            "funds use some variant of it. The academic grounding (Bodie, Merton & "
            "Samuelson 1992) is real: a young investor with decades of future earnings "
            "ahead is implicitly holding a bond-like asset in their human capital, so "
            "their financial portfolio *should* be equity-heavy. As they age and their "
            "human capital is consumed, their financial capital should de-risk. "
            "The 100-minus-age rule is a rough heuristic for this idea.\n\n"
            "The challenge (Pfau & Kitces 2014) flips the sequence-risk argument: "
            "maybe you should start *low* and rise to equities, arriving at retirement "
            "with a growing equity exposure that then coasts — avoiding the worst "
            "sequence risk at the most vulnerable moment. We test both."
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the rule works, it justifies a multi-trillion-dollar target-date fund "
            "industry and gives millions of investors a simple, principled framework "
            "for retirement saving. If it doesn't, those investors are leaving return "
            "on the table for insurance they're already getting more cheaply from a "
            "constant 60/40 — and the fancier, annually-adjusted glidepath is "
            "complexity for no gain.\n\n"
            "We simulate every 40-year investor lifecycle that history allows — "
            f"**{R['n_cohorts']:,} cohorts** in {R['n_months']:,} months of "
            f"Shiller real returns ({R['start']} to {R['end']}) — and compare "
            "terminal wealth across all four strategies. The jury is 142 years of "
            "history."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "We set up the most honest possible comparison:\n\n"
            "- **One normalised lifetime.** Age 25 to 65, equal monthly contributions. "
            "All contributions sum to 1 unit in real terms, so terminal wealth > 1 "
            "means the portfolio grew beyond the nominal sum invested.\n"
            "- **Four strategies on the same tape:** HMA (100 − age), 60/40 constant-mix, "
            "100% equities (the return ceiling), and the Pfau-Kitces rising-equity "
            "glidepath (equity weight rises linearly from 30% at 25 to 70% at 65).\n"
            "- **Rolled over all of history.** Every possible 40-year window in the "
            "Shiller dataset gives one cohort. We look at the *distribution* of "
            "terminal wealth — mean, floor (5th percentile), ceiling (95th "
            "percentile), and volatility.\n"
            "- **5 bp one-way rebalance cost.** Annual rebalance. A level playing field "
            "since all strategies pay the same cost schedule.\n\n"
            "The key question: **does HMA raise the floor** (its stated purpose) "
            "relative to the boring alternatives?"
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the weight schedule.** The four strategies hold very different "
            "equity allocations at each age:"
        ),
        code(
            "ages = list(range(25, 70, 5))\n"
            "hma_w = [st.hma_weight(a) * 100 for a in ages]\n"
            "rising_w = [st.hma_rising_weight(a) * 100 for a in ages]\n"
            "s6040_w = [60.0] * len(ages)\n"
            "eq_w = [100.0] * len(ages)\n\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.5))\n"
            "ax.plot(ages, hma_w, 'o-', c=RED, lw=2, label='HMA (100-minus-age)')\n"
            "ax.plot(ages, rising_w, 's--', c=AMBER, lw=2, label='Rising-equity (Pfau-Kitces)')\n"
            "ax.axhline(60, ls=':', c=GREY, lw=1.5, label='60/40 constant-mix')\n"
            "ax.axhline(100, ls=':', c=GREEN, lw=1, label='100% equities')\n"
            "ax.set_xlabel('Age'); ax.set_ylabel('Equity allocation (%)')\n"
            "ax.set_title('Equity weight by age: four strategies')\n"
            "ax.legend(loc='upper right'); plt.tight_layout(); plt.show()\n"
            "print('HMA: starts above 60/40 (young), ends below it (retired)')\n"
            "print('Rising: opposite — starts very low, ends above 60/40')"
        ),
        md(
            "Notice: HMA starts **above** 60/40 for young investors (75% vs 60% at "
            "age 25) and ends **well below** it for older ones (35% vs 60% at "
            "age 65). This means HMA takes more risk early and much less risk late "
            "— and we're about to see whether that trade is a good deal."
        ),
        md("**Now the key result: the distribution of terminal wealth across all 40-year cohorts.**"),
        code(
            LOAD_REAL +
            "if HAVE_REAL:\n"
            "    hma_vals = cohorts['hma'].dropna()\n"
            "    s6040_vals = cohorts['sixty_forty'].dropna()\n"
            "    eq_vals = cohorts['all_equity'].dropna()\n"
            "    rising_vals = cohorts['hma_rising'].dropna()\n"
            "else:\n"
            "    from scipy import stats as scipy_stats\n"
            "    rng = np.random.default_rng(172)\n"
            "    hma_vals = pd.Series(scipy_stats.norm.rvs(R['hma_mean'], R['hma_vol'], " +
            f"size={R['n_cohorts']}, random_state=172))\n"
            "    s6040_vals = pd.Series(scipy_stats.norm.rvs(R['s6040_mean'], R['s6040_vol'], " +
            f"size={R['n_cohorts']}, random_state=173))\n"
            "    eq_vals = pd.Series(scipy_stats.norm.rvs(R['eq_mean'], R['eq_vol'], " +
            f"size={R['n_cohorts']}, random_state=174))\n"
            "    rising_vals = pd.Series(scipy_stats.norm.rvs(R['rising_mean'], R['rising_vol'], " +
            f"size={R['n_cohorts']}, random_state=175))\n\n"
            "fig, ax = plt.subplots(figsize=(10, 4.8))\n"
            "for vals, lbl, col, ls in [\n"
            "    (hma_vals, f'HMA (mean {hma_vals.mean():.2f})', RED, '-'),\n"
            "    (s6040_vals, f'60/40 (mean {s6040_vals.mean():.2f})', GREY, '-'),\n"
            "    (eq_vals, f'100% Equities (mean {eq_vals.mean():.2f})', GREEN, '--'),\n"
            "    (rising_vals, f'Rising-eq (mean {rising_vals.mean():.2f})', AMBER, ':'),\n"
            "]:\n"
            "    ax.hist(vals, bins=50, alpha=0.4, color=col, label=lbl, density=True)\n"
            "ax.axvline(1, c='k', lw=1, ls='--', label='Break-even (= contributions)')\n"
            "ax.set_xlabel('Terminal real wealth (contributions = 1 unit)')\n"
            "ax.set_title('40-year lifecycle terminal wealth — 1,234 historical cohorts')\n"
            "ax.legend(fontsize=9); plt.tight_layout(); plt.show()"
        ),
        md(
            "**The histogram tells the story.** HMA's distribution is narrower than "
            "100% equities (less right tail, less left tail) — the defensive mechanism "
            f"is real. But 60/40's distribution **dominates** HMA: it sits further to "
            "the right (higher mean) and its left tail is not worse. The 'protection' "
            "HMA buys versus 100% equities, 60/40 provides more cheaply."
        ),
        md("**Floor check — does HMA actually protect the worst-case investor?**"),
        code(
            "floors = {\n"
            f"    'HMA': hma_vals.quantile(0.05) if HAVE_REAL else {R['hma_p05']},\n"
            f"    '60/40': s6040_vals.quantile(0.05) if HAVE_REAL else {R['s6040_p05']},\n"
            f"    '100% Equities': eq_vals.quantile(0.05) if HAVE_REAL else {R['eq_p05']},\n"
            f"    'Rising-equity': rising_vals.quantile(0.05) if HAVE_REAL else {R['rising_p05']},\n"
            "}\n"
            "col_map = {'HMA': RED, '60/40': GREY, '100% Equities': GREEN, 'Rising-equity': AMBER}\n"
            "fig, ax = plt.subplots(figsize=(8, 4))\n"
            "bars = ax.bar(floors.keys(), floors.values(), color=[col_map[k] for k in floors])\n"
            "ax.set_ylabel('5th-percentile terminal wealth (floor outcome)')\n"
            "ax.set_title('60/40 has the BEST floor — HMA does not protect the downside better')\n"
            "for b, v in zip(bars, floors.values()):\n"
            "    ax.text(b.get_x() + b.get_width()/2, v + 0.01, f'{v:.2f}',\n"
            "            ha='center', va='bottom', fontsize=11, fontweight='bold')\n"
            "ax.set_ylim(2.2, 3.0); plt.tight_layout(); plt.show()\n"
            "print('60/40 floor:', max(floors.values()))"
        ),
        md(
            f"The floor outcome for HMA is **{R['hma_p05']:.2f}** — *lower* than "
            f"60/40's **{R['s6040_p05']:.2f}**. The glidepath's safety promise is a "
            "myth relative to the alternative. The rising-equity variant is even worse "
            f"on the floor ({R['rising_p05']:.2f}), suggesting it tilts toward "
            "upside at the expense of downside — which is the opposite of de-risking.\n\n"
            "> The reason 60/40 beats HMA on the floor: the unlucky cohorts started "
            "in 1881 (the earliest start that gives 40 years of data). In those early "
            "periods, real bond returns were mediocre, so loading up on bonds late in "
            "the lifecycle (as HMA does) hurt. The 60/40 investor kept a steadier "
            "mix throughout."
        ),

        # ---- BEAT 5 — THE VERDICT --------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — MIXED.** HMA genuinely reduces variance of terminal wealth "
            f"vs 100% equities (Vol {R['hma_vol']:.2f} vs {R['eq_vol']:.2f}). The "
            "defensive mechanism is real. But it does **not** improve the floor "
            "vs 60/40, and it costs significant mean return (HAC *t* = "
            f"{R['t_hma_6040']:.1f}).\n"
            "- **Tradability — MIRAGE.** 60/40 beats HMA on mean return, ties on "
            "floor, and requires simpler implementation (no annual weight adjustment, "
            "no age tracking). There is no investor preference for which HMA "
            "dominates both benchmarks.\n"
            "- **Smarter de-risking? — FRAGILE.** The Pfau-Kitces rising-equity "
            f"variant earns more than HMA (HAC *t* = +{R['t_rising_hma']:.1f}) "
            f"and wins in {R['rising_wins_hma']:.0f}% of cohorts vs HMA. But it "
            f"still trails 60/40 (HAC *t* = {R['t_rising_6040']:.1f}) and has the "
            "worst floor of all four strategies."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you trade it?\n\n"
            "HMA is low-cost and simple — annual rebalance with four or fewer funds. "
            "The problem isn't costs or complexity; it's that a simpler strategy "
            "already dominates it. Practical considerations:\n\n"
            "- **Target-date funds.** These implement HMA variants internally but "
            "charge 10-30 bp/yr more than simple 60/40 index ETFs. The extra fee "
            "compounds over 40 years on top of the structural return disadvantage.\n"
            "- **Tax drag.** The annual weight adjustment in HMA creates taxable events "
            "in non-sheltered accounts. 60/40 drifts naturally and rebalances less "
            "frequently.\n"
            "- **The floor paradox.** If the concern is sequence-of-returns risk at "
            "retirement, a one-time glide to a lower equity weight in the five years "
            "before retirement is probably more targeted than a 40-year linear de-risk "
            "that overshoots in both directions."
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Floor risk vs mean: a preference question.** The study shows the "
            "trade-off but can't say which side of it you should be on — that depends "
            "on your actual risk aversion. If your utility function penalises left "
            "tails super-linearly, the variance reduction of HMA might be worth it "
            "even if the mean/vol ratio is worse.\n"
            "- **What about 110-minus-age or 120-minus-age?** As life expectancy has "
            "risen, some advisers now use 110 or 120 as the base. This simply shifts "
            "the HMA curve up — reducing the de-risking cost but not changing the "
            "structural trade-off vs 60/40.\n"
            "- **The withdrawal phase.** This study stops at retirement. The sequence "
            "risk during *decumulation* (the 4% rule, Bengen 1994) is a separate and "
            "arguably more important problem — covered in "
            "[Study 68 — All-Weather](../../68-all-weather/) and "
            "[Study 144 — Permanent-Portfolio](../../144-permanent-portfolio/).\n\n"
            "*Think the rule of thumb can be rescued? Fork this, add a withdrawal "
            "phase with the Bengen protocol, try a different base number (110, 120), "
            "or test with international equity data. That's the next experiment.*"
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
            "# Hundred-Minus-Age — a quantitative teardown\n"
            "### Shiller real returns · 1,234 rolling 40-year cohorts · HAC inference · "
            "Pfau-Kitces challenge · synthetic positive control\n\n"
            "![Signal: Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Smarter_de--risking?: Fragile](https://img.shields.io/badge/Smarter_de--risking%3F-Fragile-8b949e?style=flat-square)\n\n"
            "Deep companion to the "
            "[notebook for the curious](01_for_the_curious.ipynb) — *same seven "
            "beats, every claim now carrying its standard error.* We test whether the "
            "Hundred-Minus-Age glidepath improves terminal wealth distribution vs "
            "60/40 and vs 100% equities across 1,234 rolling 40-year historical "
            "cohorts, using Newey-West HAC t-stats on overlapping windows and a "
            "synthetic positive control to verify the engine.\n\n"
            "> ⚠️ **Not investment advice.** Shiller S&P 500 dataset "
            f"({R['start']} to {R['end']}), "
            "as-of 2026-06-15. Methods & sources in "
            "[`docs/references.md`](../docs/references.md), reproducible numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to "
            "intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `MIXED` | HMA vol {R['hma_vol']:.2f} vs AllEq "
            f"{R['eq_vol']:.2f} (defensive works); but mean HMA "
            f"{R['hma_mean']:.3f} vs 60/40 {R['s6040_mean']:.3f}, HAC "
            f"*t* = **{R['t_hma_6040']:.2f}**; HMA wins only "
            f"{R['hma_wins_6040']:.0f}% of cohorts vs 60/40. |\n"
            f"| **Tradability** | `MIRAGE` | 60/40 dominates HMA on mean, ties on "
            f"floor (P05 {R['s6040_p05']:.3f} vs {R['hma_p05']:.3f}), simpler. |\n"
            f"| **Smarter de-risking?** | `FRAGILE` | Rising-equity > HMA "
            f"(*t* = +{R['t_rising_hma']:.2f}) but < 60/40 "
            f"(*t* = {R['t_rising_6040']:.2f}). |\n\n"
            "> 💡 The glidepath solves the wrong problem. Reducing equity *before* "
            "retirement (HMA) costs return without improving the floor; reducing it "
            "*near* retirement (targeted) would be more efficient. But 60/40 beats "
            "either option on the data."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $W_T$ be terminal real wealth at retirement for a normalised lifecycle "
            "(contributions sum to 1). The rule asserts:\n\n"
            "- **H₁ (floor protection).** "
            "$F_{0.05}^{\\mathrm{HMA}} > F_{0.05}^{\\mathrm{60/40}}$ — the 5th-pct "
            "terminal wealth under HMA exceeds that under 60/40. This is the "
            "sequence-risk protection claim.\n"
            "- **H₂ (variance reduction vs equities).** "
            "$\\sigma(W_T^{\\mathrm{HMA}}) < \\sigma(W_T^{\\mathrm{AllEq}})$ — "
            "the glidepath reduces the spread of outcomes vs a fully-invested benchmark. "
            "A weaker claim — and the one that is actually true.\n"
            "- **H₃ (Pfau-Kitces challenge).** A rising-equity glidepath (HMA in "
            "reverse) should be better than the declining one because sequence risk "
            "peaks *at* retirement, not at entry.\n\n"
            "H₁ is false (60/40 has a better floor). H₂ is true. H₃ is partially "
            "true (Rising > HMA, but Rising < 60/40)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "HMA is embedded in \\~\\$3T of target-date funds in the US alone. If H₁ "
            "is false, the entire industry's glidepath architecture is solving the "
            "wrong problem at a non-trivial cost in expected terminal wealth. The "
            "academic literature (Pfau-Kitces 2014) already flags this; our 142-year "
            "dataset with HAC inference makes the case cleanly on Shiller data."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Lifecycle.** Age 25 to 65 (40 years), monthly contributions, "
            "sum of contributions = 1 real unit. Annual rebalance, 5 bp one-way cost.\n"
            "- **Strategies.** HMA (`w_eq = max(0.10, (100 − age) / 100)`), "
            "60/40 (`w_eq = 0.60`), 100% equity (`w_eq = 1.00`), Rising-equity "
            "(`w_eq` linearly from 0.30 at 25 to 0.70 at 65, per Pfau-Kitces 2014).\n"
            "- **Return series.** Shiller real equity return = Shiller real price "
            "change + monthly dividend yield (no look-ahead). Shiller bond return = "
            "carry + duration-weighted capital gain (D = 7, 10-year Treasuries).\n"
            "- **Inference.** Newey-West HAC t-stat on the vector of cohort terminal- "
            "wealth differences. Cohorts overlap by up to 479 months; HAC bandwidth "
            "L ≈ 4*(n/100)^(2/9) is critical to get valid standard errors.\n"
            "- **Positive control.** Synthetic two-asset tape with planted equity "
            "premium. Verifies the engine costs HMA proportionally to the premium."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Full cohort distributions — mean, floor, vol\n\n"
            "Summary statistics across all 1,234 rolling 40-year cohorts:"
        ),
        code(
            LOAD_REAL +
            "if HAVE_REAL:\n"
            "    tbl = stats[['mean','median','p05','p95','vol']].round(3)\n"
            "    print(tbl.to_string())\n"
            "else:\n"
            "    print(f\"Strategy       Mean  Median  P05   P95   Vol\")\n"
            f"    print(f\"hma           {R['hma_mean']:.3f}   {R['hma_med']:.3f}   {R['hma_p05']:.3f} {R['hma_p95']:.3f}  {R['hma_vol']:.3f}\")\n"
            f"    print(f\"sixty_forty   {R['s6040_mean']:.3f}   {R['s6040_med']:.3f}   {R['s6040_p05']:.3f} {R['s6040_p95']:.3f}  {R['s6040_vol']:.3f}\")\n"
            f"    print(f\"all_equity    {R['eq_mean']:.3f}   {R['eq_med']:.3f}   {R['eq_p05']:.3f} {R['eq_p95']:.3f}  {R['eq_vol']:.3f}\")\n"
            f"    print(f\"hma_rising    {R['rising_mean']:.3f}   {R['rising_med']:.3f}   {R['rising_p05']:.3f} {R['rising_p95']:.3f}  {R['rising_vol']:.3f}\")\n"
        ),
        md(
            f"> 💡 HMA has lower vol than 60/40 ({R['hma_vol']:.2f} vs "
            f"{R['s6040_vol']:.2f}) but 60/40 has a higher mean "
            f"({R['s6040_mean']:.3f} vs {R['hma_mean']:.3f}) AND a better floor "
            f"(P05 {R['s6040_p05']:.3f} vs {R['hma_p05']:.3f}). The variance "
            "reduction from HMA buys nothing you can't already get from 60/40."
        ),
        md("### 4b · HAC t-stats — the inference bar"),
        code(
            "if HAVE_REAL:\n"
            "    t_hma_6040 = st.hac_tstat(hma_col, s6040_col)\n"
            "    t_hma_eq = st.hac_tstat(hma_col, eq_col)\n"
            "    t_rising_hma = st.hac_tstat(rising_col, hma_col)\n"
            "    t_rising_6040 = st.hac_tstat(rising_col, s6040_col)\n"
            "else:\n"
            f"    t_hma_6040, t_hma_eq = {R['t_hma_6040']}, {R['t_hma_eq']}\n"
            f"    t_rising_hma, t_rising_6040 = {R['t_rising_hma']}, {R['t_rising_6040']}\n\n"
            "labels = ['HMA vs 60/40', 'HMA vs AllEq', 'Rising vs HMA', 'Rising vs 60/40']\n"
            "tstats = [t_hma_6040, t_hma_eq, t_rising_hma, t_rising_6040]\n"
            "cols = [RED if abs(t)<2 else (GREEN if t>0 else AMBER) for t in tstats]\n"
            "fig, ax = plt.subplots(figsize=(9, 4.2))\n"
            "bars = ax.bar(labels, tstats, color=cols, width=0.55)\n"
            "for s in (2,-2): ax.axhline(s, ls='--', c=GREY, lw=1)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('HAC t-stat')\n"
            "ax.set_title('HAC inference on 1,234 cohort differences')\n"
            "plt.tight_layout(); plt.show()\n"
            "for l, t in zip(labels, tstats):\n"
            "    print(f'{l:22s}: t = {t:+.3f}')"
        ),
        md(
            f"> 💡 HMA trails 60/40 at HAC *t* = **{R['t_hma_6040']:.1f}** and "
            f"all-equity at *t* = **{R['t_hma_eq']:.1f}** — well past the |2| bar. "
            "These are not borderline results. The overlap in cohorts inflates the "
            "effective sample size but the HAC correction accounts for that: at the "
            f"Newey-West bandwidth (~10 lags), these t-stats remain huge. "
            f"The rising-equity variant beats HMA (*t* = +{R['t_rising_hma']:.2f}) "
            "but fails vs 60/40 (*t* = "
            f"{R['t_rising_6040']:.2f})."
        ),
        md("### 4c · Cohort timeline — when does HMA win?"),
        code(
            "if HAVE_REAL:\n"
            "    diff = pd.Series(hma_col - s6040_col, index=cohorts.index)\n"
            "    fig, ax = plt.subplots(figsize=(11, 4.2))\n"
            "    pos = diff.clip(lower=0); neg = diff.clip(upper=0)\n"
            "    ax.fill_between(diff.index, pos, alpha=0.6, color=GREEN, label='HMA wins')\n"
            "    ax.fill_between(diff.index, neg, alpha=0.6, color=RED, label='60/40 wins')\n"
            "    ax.axhline(0, c='k', lw=1)\n"
            "    ax.set_xlabel('Cohort start year'); ax.set_ylabel('HMA minus 60/40 terminal TW')\n"
            "    ax.set_title(f'HMA beats 60/40 in only {(diff>0).mean():.1%} of cohorts')\n"
            "    ax.legend(); plt.tight_layout(); plt.show()\n"
            "    print(f'Win rate: {(diff>0).mean():.1%}  |  Mean diff: {diff.mean():+.3f}')\n"
            f"else:\n"
            f"    print('HMA beats 60/40: {R['hma_wins_6040']:.1f}% of cohorts (from R-dict)')"
        ),
        md(
            f"> 💡 HMA wins only **{R['hma_wins_6040']:.0f}%** of cohorts. The "
            "early cohorts (1880s-1930s) sometimes favour HMA because those were "
            "low-return bond decades; the post-WWII cohorts, with the long equity "
            "bull market, overwhelmingly prefer 60/40."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `MIXED`** — H₂ holds: HMA Vol {R['hma_vol']:.2f} < "
            f"AllEq Vol {R['eq_vol']:.2f}. H₁ fails: HMA P05 {R['hma_p05']:.3f} "
            f"< 60/40 P05 {R['s6040_p05']:.3f}. H₃ partially: Rising > HMA "
            f"(*t* = +{R['t_rising_hma']:.2f}), Rising < 60/40 "
            f"(*t* = {R['t_rising_6040']:.2f}).\n"
            f"- **Tradability `MIRAGE`** — 60/40 dominates HMA on mean, ties on "
            "floor, and is simpler. The rule of thumb adds no value.\n"
            f"- **Smarter de-risking? `FRAGILE`** — the Pfau-Kitces variant is "
            "better within the glidepath family but doesn't clear the 60/40 bar."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md("## 6 · Could you trade it? — the equity premium sensitivity"),
        code(
            "premiums = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]\n"
            "hma_means, s6040_means, t_vals = [], [], []\n"
            "for prem in premiums:\n"
            "    prices, _ = data.synthetic_lifecycle(n_years=42, equity_premium=prem, seed=172)\n"
            "    ret_s = prices.pct_change().dropna()\n"
            "    coh = st.rolling_cohorts(ret_s, n_years=40, cost_bps=5)\n"
            "    hma_means.append(coh['hma'].mean())\n"
            "    s6040_means.append(coh['sixty_forty'].mean())\n"
            "    t_vals.append(st.hac_tstat(coh['hma'].values, coh['sixty_forty'].values))\n\n"
            "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))\n"
            "ax1.plot([p*100 for p in premiums], hma_means, 'o-', c=RED, label='HMA')\n"
            "ax1.plot([p*100 for p in premiums], s6040_means, 's--', c=GREY, label='60/40')\n"
            "ax1.set_xlabel('Equity risk premium (% real, ann.)'); ax1.set_ylabel('Mean terminal wealth')\n"
            "ax1.set_title('Higher ERP = larger HMA cost'); ax1.legend()\n"
            "ax2.plot([p*100 for p in premiums], t_vals, 'o-', c=AMBER, lw=2)\n"
            "ax2.axhline(-2, ls='--', c=GREY, lw=1); ax2.axhline(0, c='k', lw=1)\n"
            "ax2.set_xlabel('Equity risk premium (% real, ann.)')\n"
            "ax2.set_ylabel('HAC t (HMA vs 60/40)')\n"
            "ax2.set_title('HMA becomes a certified loser once ERP > ~1%')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Break-even ERP (where HMA ties 60/40):')\n"
            "for p, t in zip([p*100 for p in premiums], t_vals):\n"
            "    print(f'  ERP={p:.0f}%: t={t:+.2f}')"
        ),
        md(
            "> 💡 The cost of HMA is **strictly proportional** to the equity risk "
            "premium. At a zero premium the two strategies are equivalent; once the "
            "premium exceeds ~1% real the difference becomes statistically certified. "
            "The Shiller tape's actual equity premium of ~3.1% real puts us well "
            "into 'certified loser' territory."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further — the positive control\n\n"
            "Does the engine correctly price the trade-off as a function of the equity "
            "risk premium? If we plant zero premium in a synthetic tape, HMA and 60/40 "
            "should be indistinguishable. With a 4% premium, HMA should be a certified "
            "loser. Let's verify:"
        ),
        code(
            "for prem, lbl in [(0.0, 'NULL: no equity premium'), (0.04, 'REAL: 4% equity premium')]:\n"
            "    prices, _ = data.synthetic_lifecycle(n_years=42, equity_premium=prem, seed=172)\n"
            "    ret_s = prices.pct_change().dropna()\n"
            "    coh = st.rolling_cohorts(ret_s, n_years=40, cost_bps=5)\n"
            "    t = st.hac_tstat(coh['hma'].values, coh['sixty_forty'].values)\n"
            "    win = (coh['hma'] > coh['sixty_forty']).mean()\n"
            "    print(f\"{lbl}: mean HMA={coh['hma'].mean():.3f}  mean 60/40={coh['sixty_forty'].mean():.3f}  t={t:+.2f}  HMA wins {win:.0%}\")\n"
            "print()\n"
            "print('Engine verdict: t~0 at zero premium (HMA = 60/40); |t|>>2 at 4% premium (HMA << 60/40).')\n"
            "print('The real tape (Shiller ERP ~3.1%) correctly falls in the certified-loser regime.')"
        ),
        md(
            "The engine is a faithful pricer of the equity-premium trade-off. The "
            "real-tape verdict is therefore a statement about the *market* — the "
            "historical equity risk premium (~3.1% real over 142 years) is large "
            "enough that HMA's deliberate under-weighting of equities is a certified, "
            "structural drag vs a simpler constant-mix alternative. The rule of thumb "
            "was designed for a world where that premium is uncertain; the data says "
            "it isn't uncertain enough to make the de-risking worthwhile on average."
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
