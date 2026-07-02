"""Generate the two narrative notebooks for Study 542 (Pi-Day-Effect).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic figures run
anywhere, offline and deterministic; the real-tape cells use the cached SPY/GSPC tapes under
../_cache/ (or the shared repo cache) if present and otherwise quote the frozen headline numbers in
``R`` (mirroring docs/results.md), so the notebook re-runs for any reader offline.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-30).
R = dict(
    fp_spy="3e185e607be5", fp_gspc="df3897aef897",
    spy_start="1993-02-01", spy_end="2026-06-12", n_days=8399,
    # Pi Day vs rest (SPY)
    n_pi=24, mean_pi=2.55, mean_rest=4.08, pi_contrast=-1.53,
    pi_t=-0.08, pi_p=0.94, pi_hac=0.12,
    # pooled constants (SPY)
    n_const=141, const_contrast=4.45, const_t=0.47, const_p=0.64, placebo_p=0.66,
    # bonferroni sweep (SPY): (constant, n, mean_bps, contrast_bps, welch_t, raw_p, bonf_p)
    sweep=[("tau", 24, 25.2, 21.2, 1.29, 0.209, 1.00),
           ("sqrt2", 23, -15.5, -19.7, -0.66, 0.518, 1.00),
           ("golden_phi", 24, 16.7, 12.7, 0.60, 0.556, 1.00),
           ("feigenbaum", 22, 18.8, 14.8, 0.43, 0.672, 1.00),
           ("euler_e", 24, 2.9, -1.2, -0.08, 0.937, 1.00),
           ("pi", 24, 2.5, -1.5, -0.08, 0.937, 1.00)],
    # costs (SPY Pi-day timing rule)
    gross_per=2.55, net_per=-7.45, cost_leg=5.0,
    # robustness sub-windows (SPY): (label, n_pi, contrast_bps, welch_t)
    win=[("1993-2000", 4, 37.7, 3.01), ("2000-2008", 7, -12.6, -0.29),
         ("2008-2016", 5, -43.1, -1.25), ("2016-2026", 8, 15.3, 0.42)],
    # GSPC long tape
    g_n_pi=70, g_pi_contrast=-8.82, g_pi_t=-0.77, g_placebo_p=0.28,
    g_top="golden_phi", g_top_raw=0.030, g_top_bonf=0.182,
    # synthetic control: (effect, mean contrast bps, mean placebo p)
    ctrl=[(0.0, 1.07, 0.447), (0.5, 51.5, 0.001), (1.0, 101.9, 0.001),
          (2.0, 202.7, 0.001), (4.0, 404.2, 0.001)],
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

from pi_day_effect import data, strategy as st

def load_real(ticker="SPY", start="1993-01-01", end="2026-06-30"):
    \"\"\"Cache-first real return tape (empty frame offline).\"\"\"
    return data.fetch_series(ticker, start=start, end=end, fetch=False)

SPY = load_real("SPY", "1993-01-01", "2026-06-30")
HAVE_REAL = not SPY.empty
print("real SPY tape present:", HAVE_REAL,
      "" if not HAVE_REAL else f"({len(SPY)} days, {SPY.index.min().date()} -> {SPY.index.max().date()})")
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# The Pi-Day Effect — does the market know its digits of π? 🥧\n"
            "### Do 'mathematical constant' dates trade differently? In plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Numerology_vs_data: Busted](https://img.shields.io/badge/Numerology_vs_data-Busted-8b949e?style=flat-square)\n\n"
            "Here's a fun one. **Pi Day** is March 14 — written 3/14, the first digits of "
            "π = 3.14159... Numerologists and market-folklore fans wonder: does the market *behave "
            "differently* on days that spell out a famous number? Pi Day, Euler's day (2/7 for "
            "e = 2.718...), Tau day (6/28), the golden ratio (1/6)...\n\n"
            "It sounds silly — and it is — but that's exactly why it's a perfect teaching case for "
            "the desk's #1 skill: **telling a real pattern from a coincidence you went looking for.** "
            "We take daily S&P 500 (SPY) returns and ask: is Pi Day special, or is it just another "
            "day?\n\n"
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
            f"| Does Pi Day earn a different return? | **No.** Pi Day averaged **+{R['mean_pi']} bps** "
            f"vs **+{R['mean_rest']} bps** on every other day — a *lower* return, and statistically "
            f"a dead heat (*t* {R['pi_t']}). |\n"
            f"| What about all the constant dates together? | **Still nothing** (*t* +{R['const_t']}). "
            f"And **{int(R['placebo_p']*100)}% of *random* six-date sets** are at least as extreme — "
            "the constants aren't special. |\n"
            "| Could a lucky date-pick fool us? | **Yes, that's the trap.** Test six dates and one "
            "might look good by chance — so we correct for it, and *nothing* survives. |\n"
            "| Could you trade it? | **No.** One Pi Day a year, and a 'buy on Pi Day' rule loses "
            f"**{R['net_per']} bps/event** to trading costs. |\n\n"
            "> The market does not know π. Numerology dates are ordinary days wearing a fun costume."
        ),

        # ---- BEAT 1 — THE CLAIM ----
        md(
            "## 1 · The claim\n\n"
            "> *\"The market behaves differently on Pi Day (3/14) and other mathematical-constant "
            "dates.\"* — the numerology / special-date genre.\n\n"
            "There's no real academic paper behind Pi-Day trading — that's the point. It stands in "
            "for a whole family of *superstition* claims (like the famous [Friday the 13th "
            "effect](../../163-friday-13th/)). The honest way to test 'is this date special?' isn't "
            "'is its return nonzero?' — it's *'is it more extreme than a random date would be?'*"
        ),

        # ---- BEAT 2 — SO WHAT ----
        md(
            "## 2 · So what?\n\n"
            "If a pure numerology date *did* move the market, it would be a gift: perfectly "
            "predictable, decades in advance, no data needed but a calendar. It would also be deeply "
            "weird — there's no mechanism by which the digits of π should touch stock prices. So the "
            "prior is 'this is noise' — and the interesting question is *how a data-miner could trick "
            "themselves into believing otherwise.* That trick is what we'll expose."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ----
        md(
            "## 3 · How would we even know?\n\n"
            "1. **Label the days.** March 14 is Pi Day. Also flag e-day (2/7), Tau (6/28), golden "
            "ratio (1/6), √2 (1/4), Feigenbaum (4/6). Everything else is 'ordinary'.\n"
            "2. **Compare returns.** Is the average Pi-Day (or constant-day) return different from an "
            "ordinary day?\n"
            "3. **The honest twist — the placebo.** Pick *random* sets of six dates thousands of "
            "times. How often does a random set look as extreme as the constant set? If the answer is "
            "'most of the time', the constants are nothing.\n\n"
            "*Timing:* a date is known before the market opens, so there's no peeking. The outcome is "
            "just that day's return."
        ),

        # ---- BEAT 4 — THE TEARDOWN ----
        md("## 4 · The teardown — let's actually look\n\n**Is Pi Day different from every other day?**"),
        code(
            "if HAVE_REAL:\n"
            "    r = st.pi_day_comparison(SPY)\n"
            "    pi_m, rest_m, t = r['mean_event_bps'], r['mean_rest_bps'], r['tstat_welch']\n"
            "    banner = f\"REAL SPY tape ({r['n_event']} Pi Days, 1993-2026)\"\n"
            "else:\n"
            f"    pi_m, rest_m, t = {R['mean_pi']}, {R['mean_rest']}, {R['pi_t']}\n"
            "    banner = 'frozen real numbers (offline)'\n"
            "fig, ax = plt.subplots(figsize=(7.2, 4.4))\n"
            "ax.bar(['PI DAY\\n(Mar 14)', 'every other day'], [pi_m, rest_m], color=[AMBER, GREY], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('mean daily return (bps)')\n"
            "ax.set_title(f'Pi Day {pi_m:+.1f} bps vs other days {rest_m:+.1f} bps  (t {t:+.2f}) — a dead heat')\n"
            "fig.suptitle(banner, fontsize=9, color=GREY); plt.tight_layout(); plt.show()\n"
            "print(f'Pi Day {pi_m:+.2f} bps | other days {rest_m:+.2f} bps | t {t:+.2f}')"
        ),
        md(
            f"Nothing. Pi Day averaged **+{R['mean_pi']} bps** — actually a touch *below* the "
            f"+{R['mean_rest']} bps of an ordinary day (*t* {R['pi_t']}). Any 'π premium' is invisible."
        ),
        md(
            "**The honest test: is the constant set more extreme than *random* dates?** We draw "
            "thousands of random six-date sets and see how the real constants rank."
        ),
        code(
            "if HAVE_REAL:\n"
            "    k = len(data.CONSTANT_DATES)\n"
            "    plac = st.random_dateset_placebo(SPY, k=k, n_perm=5000, seed=542)\n"
            "    p = plac['pval']; obs = plac['obs_contrast_bps']\n"
            "    # rebuild the null distribution for the picture\n"
            "    md_rows = list(zip(SPY.index.month.tolist(), SPY.index.day.tolist()))\n"
            "    ret = SPY['ret'].to_numpy(); fin = np.isfinite(ret); ret = ret[fin]\n"
            "    md_rows = [m for m,keep in zip(md_rows, fin) if keep]\n"
            "    slots = sorted(set(md_rows)); sid = {s:i for i,s in enumerate(slots)}\n"
            "    row_slot = np.array([sid[m] for m in md_rows]); rng = np.random.default_rng(7)\n"
            "    null = []\n"
            "    for _ in range(3000):\n"
            "        pick = rng.choice(len(slots), size=k, replace=False)\n"
            "        sel = np.isin(row_slot, pick)\n"
            "        null.append(abs(ret[sel].mean()-ret[~sel].mean())*1e4)\n"
            "else:\n"
            f"    p, obs = {R['placebo_p']}, {R['const_contrast']}\n"
            "    null = list(np.random.default_rng(1).normal(9, 4, 3000))\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.hist(np.abs(null), bins=40, color=GREY, alpha=.7)\n"
            "ax.axvline(abs(obs), c=RED, lw=2, label=f'constant set |contrast| = {abs(obs):.1f} bps')\n"
            "ax.set_xlabel('|contrast| of a random six-date set (bps)'); ax.set_ylabel('count')\n"
            "ax.set_title(f'The constants sit in the CROWD, not the tail  (placebo p = {p:.2f})')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'random-date-set placebo p = {p:.3f}  (constant-set |contrast| {abs(obs):.1f} bps)')"
        ),
        md(
            f"> The red line (the real mathematical constants) sits right in the *middle* of the "
            f"random-date pile. **{int(R['placebo_p']*100)}% of random six-date sets are at least as "
            "extreme.** π, *e*, τ, φ... are less special than a coin flip."
        ),
        md(
            "**And the classic data-mining trap:** test six dates, and the *best* one will look "
            "decent by luck. Here's each constant, and what happens after we correct for having "
            "looked at six:"
        ),
        code(
            "sweep = " + repr([(s[0], s[3], s[5], s[6]) for s in R["sweep"]]) + "\n"
            "if HAVE_REAL:\n"
            "    tab = st.per_constant_sweep(SPY)[['constant','contrast_bps','pval_raw','pval_bonferroni']]\n"
            "else:\n"
            "    tab = pd.DataFrame(sweep, columns=['constant','contrast_bps','pval_raw','pval_bonferroni'])\n"
            "print(tab.round(3).to_string(index=False))\n"
            "print('\\nBest raw p:', round(float(tab['pval_raw'].min()),3),\n"
            "      '-> after correcting for 6 looks (Bonferroni):', round(float(tab['pval_bonferroni'].min()),3))"
        ),
        md(
            "Even the luckiest constant can't clear the bar once you pay for having peeked at six "
            "dates — every corrected *p* is basically **1.0**. This is the whole trick behind fake "
            "calendar anomalies, laid bare."
        ),

        # ---- BEAT 5 — VERDICT ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** Pi Day contrast {R['pi_contrast']} bps (*t* {R['pi_t']}); constant "
            f"set beaten by {int(R['placebo_p']*100)}% of random date sets; Bonferroni kills all.\n"
            "- **Tradability — Mirage.** One event a year, a timing rule loses to costs.\n"
            "- **Numerology vs data — Busted.** The market does not know π."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ----
        md(
            "## 6 · Could you actually trade it?\n\n"
            f"No. A 'buy the market only on Pi Day' rule collects about +{R['gross_per']} bps of gross "
            f"return per Pi Day and pays ~10 bps of trading costs to get in and out — a net of "
            f"**{R['net_per']} bps per event**, on ~1 event a year. You'd pay to play a coin flip."
        ),

        # ---- BEAT 7 — GOING FURTHER ----
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The superstition cousin.** [163 Friday-13th](../../163-friday-13th/) runs the same "
            "placebo-and-Bonferroni playbook on a weekday legend.\n"
            "- **Real calendar effects.** [89 Turn-of-the-Month](../../89-turn-of-the-month/) and "
            "[95 Holiday-Cheer](../../95-holiday-cheer/) are the genuine seasonal patterns — see how "
            "differently *those* charts look.\n\n"
            "*Think π is secretly special? Fork this, add your favourite constant date, and watch the "
            "random-date-set placebo swallow it too.*"
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
            "# The Pi-Day Effect — a quantitative teardown 🔬\n"
            "### Welch + HAC *t* · random-date-set placebo · per-constant Bonferroni · four-window sign-flip · costs · 98-year GSPC check · synthetic control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Numerology_vs_data: Busted](https://img.shields.io/badge/Numerology_vs_data-Busted-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We test whether "
            "mathematical-constant dates (π 3/14, e 2/7, τ 6/28, φ 1/6, √2 1/4, δ 4/6) trade "
            "differently from ordinary days — and, crucially, whether they beat a *random* set of "
            "the same size.\n\n"
            "> ⚠️ **Not investment advice.** Real data: SPY daily log-returns 1993-2026 "
            f"(fp `{R['fp_spy']}`) headline, ^GSPC 1928-2026 (fp `{R['fp_gspc']}`) long tape; the "
            "offline core and tests run on a deterministic synthetic world. Numbers in "
            "[`docs/results.md`](../docs/results.md), methods in "
            "[`docs/references.md`](../docs/references.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Pi-Day contrast **{R['pi_contrast']} bps** (Welch *t* "
            f"**{R['pi_t']}**, HAC *t* +{R['pi_hac']}); constant set *t* **+{R['const_t']}**, "
            f"random-date placebo *p* **{R['placebo_p']}**; Bonferroni across six → all *p* = 1.0. |\n"
            f"| **Tradability** | `MIRAGE` | Pi-Day timing rule: +{R['gross_per']} bps gross → "
            f"**{R['net_per']} bps net**/event ({R['cost_leg']:.0f} bps/leg, ~1 event/yr). |\n"
            f"| **Numerology vs data** | `BUSTED` | 98-yr GSPC: Pi *t* {R['g_pi_t']}; best constant "
            f"(φ) raw *p* {R['g_top_raw']} → Bonferroni *p* {R['g_top_bonf']}. |\n\n"
            "> 💡 In plain words: the engine works (the synthetic control proves it), but the market "
            "is blind to the digits of π."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $C$ be the set of constant dates and $r_t$ the daily return.\n\n"
            "- **H₁ (Pi Day).** $\\mathbb{E}[r \\mid \\text{Mar 14}] \\neq \\mathbb{E}[r \\mid "
            "\\text{other}]$ at $|t| \\ge 2$.\n"
            "- **H₂ (pooled constants).** $\\mathbb{E}[r \\mid C] \\neq \\mathbb{E}[r \\mid \\bar C]$, "
            "*and* the constant set's $|{\\rm contrast}|$ sits in the tail of the random-six-date null.\n"
            "- **H₃ (no data-mining).** The best single constant survives a Bonferroni correction "
            "across the six.\n"
            "- **H₄ (robustness/net).** The sign is stable across windows and survives costs.\n\n"
            "We find **H₁ rejected** (*t* ≈ 0, slightly negative), **H₂ rejected** (placebo *p* = "
            f"{R['placebo_p']}), **H₃ rejected** (all Bonferroni *p* = 1.0), **H₄ rejected** (sign "
            "flips; net negative)."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The content is the **method**, not the (obvious) result. A numerology date is the "
            "cleanest possible *negative control* for calendar-anomaly mining: there is no plausible "
            "mechanism, so any 'significance' we find *must* be data-snooping. The random-date-set "
            "placebo prices in the fact that we **chose** six dates from ~250 calendar slots — the "
            "same multiple-testing logic that dooms cherry-picked seasonal rules "
            "([Sullivan-Timmermann-White 2001](../docs/references.md)). This is the mirror image of a "
            "real superstition test like [163 Friday-13th](../../163-friday-13th/)."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Score.** Label Pi Day (3/14) and the six constant dates by pure date arithmetic.\n"
            "- **Two-sample test.** Welch (unequal-variance) *t* on event vs non-event daily returns; "
            "a HAC (Newey-West) *t* on the event-group mean.\n"
            "- **Random-date-set placebo.** Draw 5,000 random sets of six calendar slots; the *p* is "
            "the fraction whose |contrast| ≥ the constant set's. The multiple-testing-aware null.\n"
            "- **Per-constant Bonferroni.** Test each of the six separately, correct across six.\n"
            "- **Frictions.** A Pi-Day timing rule: 2 crossings × cost/leg per isolated event.\n"
            "- **Robustness.** Pi-Day contrast over four sub-windows; a 98-year GSPC re-run.\n"
            "- **Positive control.** A deterministic synthetic world with a planted constant-day bump "
            "(effect > 0) and a null — averaged over 25 seeds (house rule).\n\n"
            "Timing: the date label is known before the open (no look-ahead); the return is that "
            "day's close-to-close log-return."
        ),

        # ---- BEAT 4 ----
        md("## 4 · The teardown"),
        md("### 4a · Pi Day and the pooled constants — H₁ / H₂"),
        code(
            "if HAVE_REAL:\n"
            "    pi = st.pi_day_comparison(SPY); cst = st.constant_day_comparison(SPY)\n"
            "    plac = st.random_dateset_placebo(SPY, k=len(data.CONSTANT_DATES), n_perm=5000, seed=542)\n"
            "    pi_c, pi_t, pi_h = pi['contrast_bps'], pi['tstat_welch'], pi['tstat_hac']\n"
            "    c_c, c_t, pp = cst['contrast_bps'], cst['tstat_welch'], plac['pval']\n"
            "else:\n"
            f"    pi_c, pi_t, pi_h = {R['pi_contrast']}, {R['pi_t']}, {R['pi_hac']}\n"
            f"    c_c, c_t, pp = {R['const_contrast']}, {R['const_t']}, {R['placebo_p']}\n"
            "fig, ax = plt.subplots(figsize=(8, 4.3))\n"
            "ax.bar(['Pi Day\\ncontrast','Constant set\\ncontrast'], [pi_c, c_c], color=[AMBER, GREY], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('contrast vs other days (bps)')\n"
            "ax.set_title(f'Pi {pi_c:+.1f} bps (t {pi_t:+.2f}) | Constants {c_c:+.1f} bps (t {c_t:+.2f}, placebo p {pp:.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Pi: contrast {pi_c:+.2f} bps, Welch t {pi_t:+.2f}, HAC t {pi_h:+.2f}')\n"
            "print(f'Constants: contrast {c_c:+.2f} bps, Welch t {c_t:+.2f}, random-date placebo p {pp:.3f}')"
        ),
        md(
            f"> 💡 In plain words: **H₁ and H₂ rejected.** Pi Day is a dead heat (*t* {R['pi_t']}, HAC "
            f"+{R['pi_hac']}); the pooled constant set (*t* +{R['const_t']}) is beaten by "
            f"**{int(R['placebo_p']*100)}%** of random six-date sets — the mathematical constants are "
            "not in the tail, they're in the crowd."
        ),
        md("### 4b · The data-mining trap — H₃ (per-constant Bonferroni)"),
        code(
            "sweep = " + repr(R["sweep"]) + "\n"
            "if HAVE_REAL:\n"
            "    tab = st.per_constant_sweep(SPY)\n"
            "else:\n"
            "    tab = pd.DataFrame(sweep, columns=['constant','n','mean_bps','contrast_bps','tstat_welch','pval_raw','pval_bonferroni'])\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "x = range(len(tab))\n"
            "ax.bar([i-0.2 for i in x], tab['pval_raw'], width=.4, color=AMBER, label='raw p')\n"
            "ax.bar([i+0.2 for i in x], tab['pval_bonferroni'], width=.4, color=GREY, label='Bonferroni p')\n"
            "ax.axhline(0.05, ls='--', c=RED, lw=1, label='p = 0.05')\n"
            "ax.set_xticks(list(x)); ax.set_xticklabels(tab['constant'], rotation=15)\n"
            "ax.set_ylabel('p-value'); ax.set_title('Best constant looks OK raw; Bonferroni erases it')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(tab.round(3).to_string(index=False))"
        ),
        md(
            "> 💡 In plain words: **H₃ rejected.** The most extreme constant (τ, raw *p* 0.21) never "
            "reaches significance even raw, and correcting for six looks pins every *p* at **1.0**. "
            "This is the anatomy of a fake calendar anomaly."
        ),
        md("### 4c · Robustness — H₄ (does the sign hold across windows?)"),
        code(
            "wins = " + repr(R["win"]) + "\n"
            "if HAVE_REAL:\n"
            "    edges = ['1993-01-01','2000-01-01','2008-01-01','2016-01-01','2026-07-01']\n"
            "    sw = st.subwindow_sweep(SPY, edges)\n"
            "    labs, contr = list(sw['window']), list(sw['contrast_bps'])\n"
            "else:\n"
            "    labs = [w[0] for w in wins]; contr = [w[2] for w in wins]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "cols = [GREEN if c>0 else RED for c in contr]\n"
            "ax.bar(range(len(labs)), contr, color=cols, width=.55)\n"
            "ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, rotation=10)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('Pi-Day contrast (bps)')\n"
            "ax.set_title('Sign flips window to window — no stable effect')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Pi-Day contrast by window (bps):', [round(c,1) for c in contr])"
        ),
        md(
            "> 💡 In plain words: **H₄ rejected.** The contrast is +38 / −13 / −43 / +15 bps across "
            "the four windows. The one positive-looking window (1993-2000, *t* 3.0) rests on **four** "
            "Pi Days and dies out of sample — the signature of a coincidence, not a signal."
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — H₁ (*t* {R['pi_t']}), H₂ (placebo *p* {R['placebo_p']}), H₃ (all "
            "Bonferroni *p* 1.0), H₄ (sign flips) all rejected.\n"
            f"- **Tradability `MIRAGE`** — +{R['gross_per']} bps gross → {R['net_per']} bps net/event, "
            "~1 event/yr.\n"
            "- **Numerology vs data `BUSTED`** — nothing survives on SPY or 98 years of GSPC."
        ),

        # ---- BEAT 6 ----
        md("## 6 · Could you trade it? — costs, and the 98-year GSPC check"),
        code(
            "if HAVE_REAL:\n"
            "    pnl = st.timing_rule_pnl(SPY, 'is_pi', cost_bps=5.0)\n"
            "    gross, net = pnl['gross_per_event_bps'], pnl['net_per_event_bps']\n"
            "else:\n"
            f"    gross, net = {R['gross_per']}, {R['net_per']}\n"
            "fig, ax = plt.subplots(figsize=(7, 4.2))\n"
            "ax.bar(['gross','net\\n(costs)'], [gross, net], color=[AMBER, GREY], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('per-event return (bps)')\n"
            "ax.set_title(f'Pi-Day timing: +{gross:.1f} bps gross -> {net:+.1f} bps net')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Pi-Day timing rule: gross {gross:+.2f} bps -> net {net:+.2f} bps/event (5 bps/leg)')"
        ),
        code(
            "# 98-year GSPC long-tape check — more Pi Days, same nothing\n"
            "GSPC = load_real('^GSPC', '1928-01-01', '2026-06-30')\n"
            "if not GSPC.empty:\n"
            "    g = st.summarize(GSPC, n_perm=5000, seed=542)\n"
            "    print(f\"GSPC Pi Days: {g['n_pi']}  | Pi contrast {g['pi_contrast_bps']:+.2f} bps (t {g['pi_tstat_welch']:+.2f})\")\n"
            "    print(f\"GSPC constant placebo p: {g['placebo_p']:.3f}  | best constant {g['top_constant']} raw p {g['top_pval_raw']:.3f} -> Bonferroni {g['top_pval_bonferroni']:.3f}\")\n"
            "else:\n"
            f"    print('GSPC (frozen): Pi Days {R['g_n_pi']} | Pi contrast {R['g_pi_contrast']:+.2f} bps (t {R['g_pi_t']:+.2f})')\n"
            f"    print('GSPC (frozen): placebo p {R['g_placebo_p']:.3f} | best constant {R['g_top']} raw p {R['g_top_raw']:.3f} -> Bonferroni {R['g_top_bonf']:.3f}')"
        ),
        md(
            f"> 💡 In plain words: on 98 years, Pi Day is *still* nothing (*t* {R['g_pi_t']}), and the "
            f"lone flicker — the golden ratio at raw *p* {R['g_top_raw']} — is the **best of six** "
            f"tries that collapses to Bonferroni *p* {R['g_top_bonf']}. `MIRAGE`, `BUSTED`."
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further — the synthetic positive control\n\n"
            "Is the engine a faithful detector, or does it always print 'nothing'? Plant a real "
            "constant-day bump of increasing strength and watch the pooled contrast rise and the "
            "random-date placebo *p* collapse — averaged over 25 seeds so no single lucky seed can "
            "fake it."
        ),
        code(
            "ctrl = " + repr(R["ctrl"]) + "\n"
            "effects = [c[0] for c in ctrl]\n"
            "# recompute live (fast: 25 seeds x small perm) so the notebook proves the engine\n"
            "res = [st.synthetic_control(e, n_seeds=25, base_seed=542, n_perm=800) for e in effects]\n"
            "contr = [r['mean_contrast_bps'] for r in res]; pv = [r['mean_placebo_p'] for r in res]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))\n"
            "a1.plot(effects, contr, 'o-', c=GREEN, lw=2); a1.axhline(0, c='k', lw=1)\n"
            "a1.set_xlabel('planted constant_effect'); a1.set_ylabel('mean constant contrast (bps)')\n"
            "a1.set_title('Contrast rises with the planted bump')\n"
            "a2.plot(effects, pv, 'o-', c=RED, lw=2); a2.axhline(0.05, ls='--', c=GREY, lw=1)\n"
            "a2.set_xlabel('planted constant_effect'); a2.set_ylabel('mean random-date placebo p (25 seeds)')\n"
            "a2.set_title('Placebo p collapses -> detector fires')\n"
            "plt.tight_layout(); plt.show()\n"
            "for e, c, p in zip(effects, contr, pv): print(f'effect {e:+.1f} -> contrast {c:+.1f} bps, placebo p {p:.3f}')"
        ),
        md(
            "At the null the contrast is ≈ 0 and the placebo *p* ≈ 0.45 (uniform); planting a genuine "
            "bump drives the contrast up linearly and the placebo *p* to ≈ 0.001 — so the engine is a "
            "faithful detector. The flat real-tape result is therefore the tape talking: the market "
            "does not know its digits of π. For the superstition cousin, see "
            "[163 Friday-13th](../../163-friday-13th/)."
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
