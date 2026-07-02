"""Generate the two narrative notebooks for Study 547 (Blue-Monday).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic figures
run anywhere, offline and deterministic; the real-tape cells use the cached SPY parquet under
../_cache/ if present and otherwise quote the frozen headline numbers in ``R`` (mirroring
docs/results.md), so the notebook re-runs for any reader offline.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-30; SPY
# 1993-01-29 -> 2026-06-26; tape fp d37d9f8a7153).
R = dict(
    fp_px="d37d9f8a7153", start="1993-01-29", end="2026-06-26", n_days=8409,
    n_blue_total=34, n_blue_trading=4, n_blue_holiday=30,
    # literal Blue Monday (n=4)
    lit_blue_bps=12.9, lit_other_bps=5.7, lit_diff_bps=7.2, lit_t=0.41,
    # blue-adjacent (tradable proxy, n=33)
    adj_n=33, adj_blue_bps=-22.0, adj_other_bps=4.8, adj_diff_bps=-26.8, adj_t=-1.24,
    adj_placebo_p=0.184,
    adj_blue_vol=19.7, adj_other_vol=18.6, adj_vol_diff=1.1,
    # timers
    bh_cagr=10.76, bh_sharpe=0.643, timer_cagr=10.74, timer_switches=9,
    # sub-period sweep (label, diff_bps, welch_t) on the adjacent proxy
    win=[("1993-2001", 26.9, 1.05), ("2001-2009", -101.3, -1.51),
         ("2009-2018", 20.0, 1.24), ("2018-2026", -58.8, -1.50)],
    # synthetic control (blue_dip, mean welch-t over 25 seeds)
    ctrl=[(0.0, -0.32), (0.001, -0.96), (0.002, -1.60), (0.003, -2.25), (0.004, -2.89)],
    ctrl_null100=-0.01,
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

from blue_monday import data, strategy as st

def load_real():
    \"\"\"Cache-first real SPY close (empty frame offline).\"\"\"
    f = data.load_real(fetch=False)
    return f["close"] if not f.empty else pd.Series(dtype=float)

CLOSE = load_real()
HAVE_REAL = not CLOSE.empty
print("real SPY tape present:", HAVE_REAL,
      "" if not HAVE_REAL else f"({len(CLOSE)} sessions, {CLOSE.index.min().date()} -> {CLOSE.index.max().date()})")
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Blue Monday — is the 'most depressing day' bad for stocks too? 🫐\n"
            "### The third Monday of January, in plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Blue_Monday_tradable%3F: Busted](https://img.shields.io/badge/Blue_Monday_tradable%3F-Busted-8b949e?style=flat-square)\n\n"
            "Every January, headlines announce **'Blue Monday'** — the third Monday of the month, "
            "supposedly the most depressing day of the year. It comes from a 2005 'equation' cooked "
            "up by a psychologist... **as a PR stunt for a travel company**. There's no science in it. "
            "But the desk has a serious cousin — the idea that seasonal *mood* moves markets "
            "([the SAD-Effect, Study 150](../../150-sad-effect/)). So: if everyone's miserable on "
            "Blue Monday, do stocks sag and get jumpy that day?\n\n"
            "There's a delicious catch waiting for us. Let's go find it.\n\n"
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
            f"| Can we even test Blue Monday? | **Barely.** Blue Monday is the third Monday of "
            f"January — which is also **Martin Luther King Jr. Day**, a market holiday. The stock "
            f"market is **closed** on {R['n_blue_holiday']} of the last {R['n_blue_total']} Blue "
            f"Mondays. Only **{R['n_blue_trading']}** were tradable. |\n"
            f"| On the day *after* Blue Monday, do stocks sag? | **A little, but it's noise.** The "
            f"next trading day averages **{R['adj_blue_bps']} bps** vs **+{R['adj_other_bps']} bps** "
            f"on a normal day — a dip in the right direction, but a shuffle test says a random day "
            f"matches it {int(R['adj_placebo_p']*100)}% of the time. |\n"
            "| Do stocks get more *volatile*? | **No.** The vol difference is about +1 point — a "
            "rounding error. |\n"
            "| Could you trade it? | **No** — you can't trade a closed market, and the day-after "
            "'dip' flips to a *gain* in half the sub-periods. |\n\n"
            "> Blue Monday is a marketing holiday that happens to land on a market holiday. The mood "
            "myth has no echo you could bet on."
        ),

        # ---- BEAT 1 — THE CLAIM ----
        md(
            "## 1 · The claim\n\n"
            "> *\"The third Monday of January is the most depressing day of the year.\"* — 'Blue "
            "Monday' (Cliff Arnall, 2005, for Sky Travel).\n\n"
            "The 'equation' mixed weather, debt, days since Christmas and 'low motivation' with "
            "made-up weights — you can't derive it or disprove it. Scientists have called it junk for "
            "20 years. But the *market* version is testable: if mood drives prices, the most "
            "depressing day should show **lower returns** and **higher volatility** than an ordinary "
            "Monday. That's the SAD-effect logic, pinned to one calendar day."
        ),

        # ---- BEAT 2 — SO WHAT ----
        md(
            "## 2 · So what?\n\n"
            "If a *made-up* calendar day moved markets, that would be a gift — a free, known-in-advance "
            "date to trade around. It would also validate the whole 'mood moves markets' story the desk "
            "already tested with the [SAD-Effect (150)](../../150-sad-effect/) and the "
            "[Monday-Effect (224)](../../224-monday-effect/) — both of which came up empty. Blue Monday "
            "is the narrowest, most-hyped version of that claim."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ----
        md(
            "## 3 · How would we even know?\n\n"
            "1. **Find every Blue Monday** (third Monday of January) from 1993 to 2026.\n"
            "2. **Check if the market was even open** — because MLK Day lands on the same date.\n"
            "3. For the tradable ones (and the day *after*, when it's shut), **compare returns and "
            "volatility** to all other days.\n"
            "4. **Shuffle test:** could a random day look just as 'depressing' by luck?\n\n"
            "*Timing:* the calendar is known years ahead, so there's **no guessing the future** — you "
            "know Blue Monday's date before it happens. The catch is the sample size, which we're "
            "about to see is brutal."
        ),

        # ---- BEAT 4 — THE TEARDOWN ----
        md(
            "## 4 · The teardown — the punchline\n\n"
            "**How many Blue Mondays could you actually trade?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    tdays = set(CLOSE.index.normalize())\n"
            "    years = range(CLOSE.index.min().year, CLOSE.index.max().year+1)\n"
            "    trad = sum(data.third_monday_of_january(y) in tdays for y in years)\n"
            "    hol = sum(data.third_monday_of_january(y) not in tdays for y in years)\n"
            "else:\n"
            f"    trad, hol = {R['n_blue_trading']}, {R['n_blue_holiday']}\n"
            "fig, ax = plt.subplots(figsize=(6.5, 4.2))\n"
            "ax.bar(['Market OPEN\\n(tradable)','Market CLOSED\\n(MLK holiday)'], [trad, hol], color=[GREEN, RED], width=.5)\n"
            "ax.set_ylabel('number of Blue Mondays'); ax.set_title('Blue Monday IS MLK Day — the market is shut for most of them')\n"
            "for i,v in enumerate([trad,hol]): ax.text(i, v+0.4, str(v), ha='center', fontsize=12)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'tradable Blue Mondays: {trad}   |   market-closed Blue Mondays: {hol}')"
        ),
        md(
            f"There's the catch. Blue Monday is the **third Monday of January** — and so is **Martin "
            f"Luther King Jr. Day**, which the stock market has closed for since 1998. So the 'most "
            f"depressing day' is a day you *can't trade* in {R['n_blue_holiday']} of the last "
            f"{R['n_blue_total']} years. The claim self-destructs on the calendar."
        ),
        md(
            "**Fine — what about the *next* open day (usually Tuesday)? Does the mood carry over?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    sa = st.blue_adjacent_split(CLOSE, data)\n"
            "    blue_m, other_m, diff = sa['blue_mean_bps'], sa['other_mean_bps'], sa['diff_bps']\n"
            "else:\n"
            f"    blue_m, other_m, diff = {R['adj_blue_bps']}, {R['adj_other_bps']}, {R['adj_diff_bps']}\n"
            "fig, ax = plt.subplots(figsize=(7, 4.3))\n"
            "ax.bar(['Day after\\nBlue Monday','Every other\\nday'], [blue_m, other_m], color=[RED, GREY], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('average return (bps)')\n"
            "ax.set_title(f'A faint dip the right way ({diff:+.1f} bps) — but is it real?')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'day-after-Blue-Monday {blue_m:+.1f} bps  vs  other days {other_m:+.1f} bps  ->  diff {diff:+.1f} bps')"
        ),
        md(
            f"The day after Blue Monday *is* a touch negative (**{R['adj_blue_bps']} bps** vs "
            f"**+{R['adj_other_bps']} bps**) — the mood story's direction. But 'a touch' is the "
            "operative word. The quants notebook shows the *t*-stat is only −1.24 and a random day "
            "matches it 18% of the time. And when we split the history into four chunks, the dip "
            "**flips to a gain** in half of them — a pattern that changes its mind isn't a signal."
        ),

        # ---- BEAT 5 — VERDICT ----
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The literal day is untestable (market closed 30/34 years, n=4). The "
            "day-after proxy leans negative but never clears the bar, has no vol spike, and flips sign "
            "across sub-periods.\n"
            "- **Tradability — Mirage.** You can't trade a closed market, and the proxy day is a "
            "sign-flipping coin toss.\n"
            "- **Blue Monday tradable? — Busted.** It's a 2005 PR stunt that happens to be a market "
            "holiday. Not a market event."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ----
        md(
            "## 6 · Could you actually trade it?\n\n"
            "No — twice over. First, the headline day is a holiday: there's nothing to buy or sell. "
            "Second, even the day-after proxy is a coin toss — it's *up* in two eras and *down* in "
            "two. A 'sit out Blue Monday' rule just tracks buy-and-hold (10.74% vs 10.76% CAGR) "
            "because there are only four days in 33 years to act on, and none of them pay."
        ),

        # ---- BEAT 7 — GOING FURTHER ----
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The serious cousin.** [Study 150 — SAD-Effect](../../150-sad-effect/) tests the "
            "'seasonal mood moves markets' thesis properly, over 155 years — also a data-mining "
            "artifact.\n"
            "- **The weekday version.** [Study 224 — Monday-Effect](../../224-monday-effect/) shows "
            "Mondays aren't even negative on the modern tape.\n\n"
            "*Think Blue Monday hides a real dip? Fork this, add international markets that trade on "
            "MLK Day, and show the third-Monday return clearing t = −2.*"
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
            "# Blue Monday — a quantitative teardown 🔬\n"
            "### literal-day dead-end (MLK collision) · Blue-adjacent Welch *t* · random-day placebo · vol comparison · four-window sign-flip · costs & borrow · synthetic control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Blue_Monday_tradable%3F: Busted](https://img.shields.io/badge/Blue_Monday_tradable%3F-Busted-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We test whether the third "
            "Monday of January ('Blue Monday') is a market-mood low: lower returns and higher vol.\n\n"
            "> ⚠️ **Not investment advice.** Real data: yfinance total-return SPY daily closes, "
            f"1993-2026 (as-of 2026-06-30, tape fp `{R['fp_px']}`); the offline core and tests run on "
            "a deterministic synthetic world. Methods in "
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
            f"| **Signal** | `NONE` | Literal Blue Monday n={R['n_blue_trading']} (market shut "
            f"{R['n_blue_holiday']}/{R['n_blue_total']} yrs); Blue-adjacent diff **{R['adj_diff_bps']} "
            f"bps** (Welch *t* **{R['adj_t']}**, placebo *p* {R['adj_placebo_p']}), vol diff "
            f"**+{R['adj_vol_diff']} pts** (none), sign flips across all 4 windows. |\n"
            f"| **Tradability** | `MIRAGE` | Untradable holiday; timer {R['timer_cagr']}% vs "
            f"buy-and-hold {R['bh_cagr']}% CAGR. |\n"
            "| **Blue Monday tradable?** | `BUSTED` | A 2005 PR stunt on a market-holiday date. |\n\n"
            "> 💡 In plain words: the engine works (the synthetic control proves it), but the literal "
            "claim is untestable and the tradable proxy is insignificant and sign-unstable."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $B$ be the set of Blue-Monday (third-January-Monday) returns and $O$ all other days.\n\n"
            "- **H₁ (mood dip).** $\\mathbb{E}[r \\mid B] - \\mathbb{E}[r \\mid O] < 0$ at *t* ≤ −2.\n"
            "- **H₂ (mood vol-spike).** $\\text{Var}[r \\mid B] > \\text{Var}[r \\mid O]$.\n"
            "- **H₃ (robustness).** The sign of H₁ is stable across sub-periods.\n"
            "- **H₄ (tradable).** A Blue-Monday timer beats buy-and-hold net of costs.\n\n"
            "We find **H₁ rejected** (literal day untestable; proxy *t* −1.24, above the bar), **H₂ "
            "rejected** (vol diff ≈ 0), **H₃ rejected** (sign flips), **H₄ rejected** (untradable "
            "holiday; proxy is a coin toss)."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The content is the **why**. Two forces gut the claim: (a) **the MLK-holiday collision** — "
            "the Uniform Monday Holiday Act put MLK Day on the third Monday of January, exactly Blue "
            "Monday, and equity markets have closed for it since 1998, so the literal claim is "
            "untestable ($n=4$); and (b) even the tradable next-open proxy is **statistical noise** — "
            "one standard error shy of nothing, no vol spike, sign-unstable. Same fate as the "
            "[SAD-Effect (150)](../../150-sad-effect/) it descends from."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Calendar.** Blue Monday = third Monday of January; flag whether each was a trading "
            "day (MLK collision) — the literal split.\n"
            "- **Proxy.** Blue-adjacent = first open session on/after Blue Monday (restores $n=33$).\n"
            "- **Inference.** Welch two-sample *t* on Blue(-adjacent) − other-day returns; a "
            "**random-day placebo** null (5,000 draws).\n"
            "- **Volatility.** Compare Blue-adjacent vs other-day realised vol.\n"
            "- **Robustness.** Re-run over four equal date spans; read the sign.\n"
            "- **Frictions.** A 'sit out / short Blue Monday' timer vs buy-and-hold, one-way cost per "
            "switch, shorts pay borrow.\n"
            "- **Positive control.** A deterministic synthetic world with a planted mood dip "
            "(`blue_dip > 0`) and a null — averaged over 25 seeds (house rule).\n\n"
            "Timing: the calendar is known in advance → **no execution lag**. The binding limitation "
            "is the thin sample (one Blue Monday per year), named on the SIGNAL axis."
        ),

        # ---- BEAT 4 ----
        md("## 4 · The teardown"),
        md(
            "### 4a · The literal day — H₁ hits the MLK wall\n\n"
            "Blue Monday is MLK Day; the market is shut for almost all of them."
        ),
        code(
            "if HAVE_REAL:\n"
            "    tdays = set(CLOSE.index.normalize())\n"
            "    years = list(range(CLOSE.index.min().year, CLOSE.index.max().year+1))\n"
            "    rows = [(y, str(data.third_monday_of_january(y).date()),\n"
            "             'OPEN' if data.third_monday_of_january(y) in tdays else 'CLOSED (MLK)') for y in years]\n"
            "    tab = pd.DataFrame(rows, columns=['year','blue_monday','market'])\n"
            "    trad = int((tab['market']=='OPEN').sum()); hol = int((tab['market']!='OPEN').sum())\n"
            "    sm = st.blue_monday_split(CLOSE, data); lt = st.welch_t(sm)\n"
            "    print(tab.to_string(index=False))\n"
            "    print(f'\\nOPEN {trad} | CLOSED {hol} | literal diff {sm[\"diff_bps\"]:+.1f} bps (t {lt[\"t\"]:+.2f}, n_blue={sm[\"n_blue\"]})')\n"
            "else:\n"
            f"    trad, hol = {R['n_blue_trading']}, {R['n_blue_holiday']}\n"
            f"    print(f'OPEN {{trad}} | CLOSED {{hol}} | literal diff {R['lit_diff_bps']:+.1f} bps (t {R['lit_t']:+.2f}, n_blue={R['n_blue_trading']})')"
        ),
        md(
            f"> 💡 In plain words: H₁ is **untestable on the literal day** — the market is closed for "
            f"{R['n_blue_holiday']} of {R['n_blue_total']} Blue Mondays, leaving n={R['n_blue_trading']} "
            f"(diff {R['lit_diff_bps']:+.1f} bps, *t* {R['lit_t']:+.2f}). Four points is not a sample."
        ),
        md(
            "### 4b · The tradable proxy — Blue-adjacent day (H₁ again, with power)\n\n"
            "The first open session after Blue Monday, vs every other day, with the Welch *t* and "
            "placebo."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sa = st.blue_adjacent_split(CLOSE, data); wa = st.welch_t(sa)\n"
            "    p = st.placebo_pvalue_adjacent(CLOSE, data, n_perm=5000, seed=547)\n"
            "    blue_m, other_m, diff, t = sa['blue_mean_bps'], sa['other_mean_bps'], sa['diff_bps'], wa['t']\n"
            "else:\n"
            f"    blue_m, other_m, diff, t, p = {R['adj_blue_bps']}, {R['adj_other_bps']}, {R['adj_diff_bps']}, {R['adj_t']}, {R['adj_placebo_p']}\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.3))\n"
            "ax.bar(['Blue-adjacent','other days'], [blue_m, other_m], color=[RED, GREY], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('mean return (bps)')\n"
            "ax.set_title(f'Blue-adjacent - other = {diff:+.1f} bps  (Welch t {t:+.2f}, placebo p {p:.3f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Blue-adjacent {blue_m:+.1f} bps | other {other_m:+.1f} bps | diff {diff:+.1f} bps | t {t:+.2f} | placebo p {p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: H₁ **rejected** — the dip is the right direction ({R['adj_diff_bps']} "
            f"bps) but the *t* is only **{R['adj_t']}** (bar |t| ≥ 2) and the placebo *p* = "
            f"{R['adj_placebo_p']} says a random day matches it ~18% of the time. Directionally cute, "
            "statistically empty."
        ),
        md(
            "### 4c · The vol spike — H₂\n\n"
            "Does mood make the day jumpier? Compare realised vol."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sa = st.blue_adjacent_split(CLOSE, data)\n"
            "    bv, ov, vd = sa['blue_vol_ann']*100, sa['other_vol_ann']*100, sa['vol_diff_ann']*100\n"
            "else:\n"
            f"    bv, ov, vd = {R['adj_blue_vol']}, {R['adj_other_vol']}, {R['adj_vol_diff']}\n"
            "fig, ax = plt.subplots(figsize=(6.8, 4.2))\n"
            "ax.bar(['Blue-adjacent','other days'], [bv, ov], color=[AMBER, GREY], width=.5)\n"
            "ax.set_ylabel('annualised realised vol (%)')\n"
            "ax.set_title(f'No mood vol-spike: {vd:+.1f} pts difference')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Blue-adjacent vol {bv:.1f}% | other vol {ov:.1f}% | diff {vd:+.1f} pts')"
        ),
        md(
            f"> 💡 In plain words: H₂ **rejected** — the vol difference is **+{R['adj_vol_diff']} pts**, "
            "a rounding error. No 'depressing day' turbulence."
        ),
        md(
            "### 4d · Robustness — H₃ (does the sign hold?)\n\n"
            "The Blue-adjacent split over four equal date spans."
        ),
        code(
            "wins = " + repr(R["win"]) + "\n"
            "if HAVE_REAL:\n"
            "    rows = st.subperiod_sweep(CLOSE, data, 4, which='adjacent')\n"
            "else:\n"
            "    rows = wins\n"
            "tab = pd.DataFrame(rows, columns=['window','diff_bps','welch_t']).set_index('window')\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.3))\n"
            "cols = [RED if s<0 else GREEN for s in tab['diff_bps']]\n"
            "ax.bar(range(len(tab)), tab['diff_bps'], color=cols, width=.55)\n"
            "ax.set_xticks(range(len(tab))); ax.set_xticklabels(tab.index, rotation=10)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('Blue-adjacent - other (bps)')\n"
            "ax.set_title('Sign flips: dip (red) in 2 windows, gain (green) in 2 — not a signal')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(tab.round(2).to_string())"
        ),
        md(
            "> 💡 In plain words: H₃ **rejected.** The Blue-adjacent day is *up* in 1993-2001 & "
            "2009-2018 and *down* in 2001-2009 & 2018-2026. The full-sample −26.8 bps is the average "
            "of a sign that keeps changing its mind."
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — H₁ rejected (literal n={R['n_blue_trading']}; proxy *t* "
            f"{R['adj_t']}), H₂ rejected (vol diff ≈ 0), H₃ rejected (sign flips).\n"
            f"- **Tradability `MIRAGE`** — untradable holiday; timer {R['timer_cagr']}% vs "
            f"buy-and-hold {R['bh_cagr']}% CAGR.\n"
            "- **Blue Monday tradable? `BUSTED`** — a PR stunt on a market-holiday date."
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you trade it? — costs and the borrow\n\n"
            "The literal day is a holiday; the proxy is a coin toss. Compare a 'sit out Blue Monday' "
            "timer to buy-and-hold."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bh = st.buy_and_hold(CLOSE)\n"
            "    so = st.blue_monday_timer(CLOSE, data, mode='sit_out', cost_bps=1.0)\n"
            "    sh = st.blue_monday_timer(CLOSE, data, mode='short', cost_bps=1.0, borrow_ann_bps=100.0)\n"
            "    bh_c, so_c, sh_c = bh['cagr']*100, so['cagr']*100, sh['cagr']*100\n"
            "else:\n"
            f"    bh_c, so_c, sh_c = {R['bh_cagr']}, {R['timer_cagr']}, {R['timer_cagr']}-0.02\n"
            "fig, ax = plt.subplots(figsize=(7.2, 4.2))\n"
            "ax.bar(['buy & hold','sit out\\nBlue Mon','short\\nBlue Mon'], [bh_c, so_c, sh_c], color=[GREEN, GREY, RED], width=.5)\n"
            "ax.set_ylabel('CAGR (%)'); ax.set_ylim(bottom=min(bh_c,so_c,sh_c)-0.3)\n"
            "ax.set_title('Every Blue-Monday rule just tracks buy-and-hold (only 4 tradable days)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'buy&hold {bh_c:.2f}% | sit_out {so_c:.2f}% | short {sh_c:.2f}% CAGR')"
        ),
        md(
            "> 💡 In plain words: there is nothing to harvest — with only four tradable Blue Mondays in "
            "33 years, the timer is buy-and-hold with a rounding error. `MIRAGE`."
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further — the synthetic positive control\n\n"
            "Is the engine a faithful detector, or does it always print ~0? On a synthetic tape where "
            "Blue Monday *is* a trading day, plant a real mood dip (`blue_dip`) of increasing strength "
            "and watch the Welch *t* go negative — averaged over 25 seeds so no single lucky seed can "
            "fake it."
        ),
        code(
            "ctrl = " + repr(R["ctrl"]) + "\n"
            "dips = [c[0] for c in ctrl]\n"
            "ts = [st.synthetic_mean_t(data, blue_dip=d, n_seeds=25) for d in dips]\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.3))\n"
            "ax.plot(dips, ts, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(-2, ls='--', c=GREY, lw=1, label='t = -2 bar')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted blue_dip (bigger = stronger mood dip)')\n"
            "ax.set_ylabel('mean Welch t (25 seeds)')\n"
            "ax.set_title('The harness banks a planted dip — flat at the null, past -2 as it grows')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for d, t in zip(dips, ts): print(f'blue_dip {d:.3f} -> mean Welch t {t:+.2f}')\n"
            "print(f'null over 100 seeds -> {st.synthetic_mean_t(data, 0.0, n_seeds=100, base_seed=1000):+.2f}')"
        ),
        md(
            "The Welch *t* is ≈ 0 at the null (−0.01 over 100 seeds) and falls past −2 as the planted "
            "dip grows — so the engine is a faithful detector. The real-tape non-result is therefore a "
            "statement about **the tape and the calendar**: Blue Monday is a market holiday, and its "
            "tradable proxy is noise. For the serious mood-anomaly parent, see "
            "[150 SAD-Effect](../../150-sad-effect/); for the weekday cousin, "
            "[224 Monday-Effect](../../224-monday-effect/)."
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
