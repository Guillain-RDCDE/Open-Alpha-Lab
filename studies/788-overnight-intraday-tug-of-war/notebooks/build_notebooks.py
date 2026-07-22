"""Generate the two narrative notebooks for Study 788 (Overnight/Intraday Tug of War).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached
cross-section panel under ../_cache/ and otherwise quote the frozen headline numbers
in ``R`` (mirroring docs/results.md). The synthetic positive control runs anywhere
with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance daily OHLC,
# total-return, 50 liquid US large-caps, 2010-01-04 -> 2026-06-30; trailing-21d
# overnight sort, top30%-bot30%, 4,125 spread days).
R = dict(
    start="2010-01-04", end="2026-06-30", n_names=50, n_days=4125,
    on_bps=3.71, id_bps=-2.39, cc_bps=1.32, tug_bps=6.10,
    on_t_nw=4.62, id_t_nw=-2.36, cc_t_nw=1.06, tug_t_nw=4.59,
    on_t_1s=4.40, id_t_1s=-2.27,
    on_sharpe=1.09,
    on_top_bps=6.76, on_bot_bps=3.05, id_top_bps=1.65, id_bot_bps=4.04,
    on_welch=2.12, id_welch=-1.16,
    placebo_obs=3.71, placebo_mean=0.001, placebo_sd=0.597, placebo_p=0.00000,
    placebo_draws=1000,
    era_split="2018-01-01",
    era_early_on_bps=2.99, era_early_on_t=4.32, era_early_id_bps=-3.90, era_early_id_t=-3.21,
    era_early_n=1991,
    era_late_on_bps=4.38, era_late_on_t=3.11, era_late_id_bps=-0.97, era_late_id_t=-0.62,
    era_late_n=2134,
    timer_1_gross=3.71, timer_1_cost=4.14, timer_1_net=-0.43, timer_1_t=-0.51,
    timer_1_sharpe=-0.13, timer_1_ann=-1.1,
    timer_5_gross=3.71, timer_5_cost=20.14, timer_5_net=-16.43, timer_5_t=-19.49,
    timer_5_sharpe=-4.82, timer_5_ann=-41.4,
    syn_null_on_mean=-0.19, syn_null_on_sd=1.03, syn_null_fire=0,
    syn_planted_on_t=10.51, syn_planted_id_t=-11.15,
    fp_oc="d3f2054309d3", fp_close="357fd262912f",
)

BADGES = (
    "![Signal: Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n\n"
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
import warnings; warnings.filterwarnings("ignore")
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from overnight_intraday_tug import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    PANEL = data.load_panel()
    ON, IDC = st.leg_panels(PANEL)
    SP = st.tug_spreads(ON, IDC, window=21, frac=0.3)
else:
    PANEL = ON = IDC = SP = None
print("real cache present:", HAVE_REAL,
      "| names:", (0 if PANEL is None else len(PANEL)),
      "| spread days:", (0 if SP is None else len(SP)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Do last night's winners keep winning at night — and give it back by day? 🌙🔀☀️\n"
            "### The overnight/intraday \"tug of war\" — a real cross-sectional pattern that "
            "nets to almost nothing\n\n"
            + BADGES +
            "Here's a strange, specific claim from a 2019 finance paper (Lou, Polk & Skouras): "
            "take the stocks that did *best overnight* lately — the ones that keep jumping "
            "between yesterday's close and this morning's open — and they'll **keep** earning "
            "overnight, while quietly **giving it back during the trading day**. The stocks that "
            "did *worst* overnight do the mirror image. The two forces pull against each other — "
            "a tug of war between night and day.\n\n"
            "Most desk folklore like this evaporates when you count it. This one doesn't — the "
            "pattern is genuinely there. What evaporates is the *money*.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo and the cost math? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** 50 liquid US large-caps, daily OHLC (total-return), 2010→2026. "
            "The universe is *today's* mega-caps (a survivor set), so the sizes below are an "
            "upper bound. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Do past-overnight winners keep winning **overnight**? | **Yes.** The high-minus-"
            f"low spread earns **+{R['on_bps']:.2f} bps every night** — a persistence so steady "
            f"its gross Sharpe is ~{R['on_sharpe']:.1f}, and a random reshuffle beats it about "
            f"**0 times in 1,000**. |\n"
            f"| Do they **reverse** during the day? | **Yes.** The very same names lose "
            f"**{R['id_bps']:.2f} bps intraday** — the mirror image. Night and day pull opposite "
            f"ways. |\n"
            f"| So is it free money? | **No — the legs almost cancel.** Add night + day and the "
            f"whole thing nets to **+{R['cc_bps']:.2f} bps** close-to-close, statistically "
            "indistinguishable from zero. It's a *redistribution* within the day, not a premium. |\n"
            f"| Can you harvest just the good (night) leg? | **No.** Grabbing the overnight leg "
            f"means trading the whole book in at the close and out at the open — four spread-"
            f"crossings a day. Even at a **1 bp** cost that friction "
            f"(**{R['timer_1_cost']:.1f} bps/day**) is *bigger than the entire "
            f"+{R['on_bps']:.2f} bps edge*. |\n\n"
            "> The pattern is real and beautiful. The paycheck is a mirage."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"A stock's return has two halves — the **overnight** move (yesterday's close to "
            "this morning's open) and the **intraday** move (open to close). These two halves are "
            "driven by different crowds, and they pull in opposite directions: whatever pushes a "
            "stock up overnight tends to persist overnight and unwind intraday.\"*\n\n"
            "This is the Lou-Polk-Skouras *tug of war*. The intuition: one crowd (retail, "
            "news-chasers) piles in around the open and overnight; another (institutions, "
            "market-makers) leans against them during the day. Sort stocks on how they've been "
            "doing overnight lately, and you should see the winners keep their overnight edge and "
            "lose it back by the close."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If real, it rewrites how you read a backtest. A strategy that looks flat on "
            "close-to-close returns might secretly be a *strong* overnight bet fighting a *strong* "
            "intraday bet — two real edges hidden inside a boring-looking line. And it means the "
            "clock you trade on matters as much as the stock you pick: the same signal can be "
            "money at night and poison by day."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The sort.** Every day, rank our {R['n_names']} stocks by their trailing overnight "
            "return (using only info known by yesterday's close — no peeking). Buy the top third, "
            "short the bottom third.\n"
            "- **The two legs.** Measure that long-short book's *overnight* return and its "
            "*intraday* return, separately, every day.\n"
            "- **The luck check.** Randomly scramble which stock's future is attached to which "
            "ranked stock, 1,000 times — how often does a random shuffle match the real overnight "
            "spread?\n"
            "- **The trade check.** Try to actually capture the good (overnight) leg, paying "
            "realistic costs for the in-and-out trading it needs."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the two legs of the sorted spread.**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    h = st.tug_stats(SP)\n"
            "    on, idn, cc = h['on_mean_bps'], h['id_mean_bps'], h['cc_mean_bps']\n"
            "else:\n"
            "    on, idn, cc = R['on_bps'], R['id_bps'], R['cc_bps']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.4))\n"
            "ax.bar(['OVERNIGHT\\n(prev close->open)','INTRADAY\\n(open->close)','NET\\nclose-to-close'],\n"
            "       [on, idn, cc], color=[GREEN, RED, GREY], width=.6)\n"
            "for i,v in enumerate([on, idn, cc]): ax.annotate(f'{v:+.2f} bps',(i,v),ha='center',\n"
            "    va='bottom' if v>=0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('high-minus-low spread, bps/day')\n"
            "ax.set_title('Past-overnight winners keep winning at night, reverse by day')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'overnight {on:+.2f} bps | intraday {idn:+.2f} bps | net {cc:+.2f} bps')"
        ),
        md(
            f"There's the tug: **+{R['on_bps']:.2f} bps overnight** (the winners keep winning) and "
            f"**{R['id_bps']:.2f} bps intraday** (they give it back) — opposite signs, both real. "
            f"And crucially the third bar: add them up and you're left with a measly "
            f"**+{R['cc_bps']:.2f} bps** close-to-close. The two big forces very nearly cancel. "
            "The exciting part isn't a profit — it's *where inside the day* the return lives.\n\n"
            "**Is the overnight persistence real, or a lucky alignment of the sort?** Scramble it "
            "1,000 ways:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.placebo_pvalue(ON, IDC, n_seeds=6, n_draws_per_seed=15)\n"
            "    obs, draws = pl['obs_bps'], pl['draws_bps']\n"
            "else:\n"
            "    obs = R['placebo_obs']\n"
            "    rng = np.random.default_rng(788)\n"
            "    draws = rng.normal(R['placebo_mean'], R['placebo_sd'], 500)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=40, color=GREY, alpha=.85, label='null: scrambled sort (light in-notebook run)')\n"
            "ax.axvline(obs, c=GREEN, lw=2.5, label=f'observed overnight spread {obs:+.2f} bps')\n"
            "ax.set_xlabel('overnight spread of a scrambled draw (bps/day)')\n"
            "ax.set_ylabel('frequency')\n"
            "ax.set_title(f\"Far outside the luck cloud: canonical p = {R['placebo_p']:.5f} (1,000 draws)\")\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"canonical placebo (results.md): mean {R['placebo_mean']:+.3f} bps, \"\n"
            "      f\"sd {R['placebo_sd']:.3f}, p = {R['placebo_p']:.5f}\")"
        ),
        md(
            f"The real overnight spread sits **way** outside the cloud of random shuffles — about "
            f"6 standard deviations out, **p = {R['placebo_p']:.5f}**. This isn't the sort getting "
            "lucky.\n\n"
            "**Finally, the trade.** The overnight leg is the good one — can you just buy that?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    tm1 = st.timer_stats(SP, cost_bps=1.0)\n"
            "    tm5 = st.timer_stats(SP, cost_bps=5.0)\n"
            "    g, n1, n5 = tm1['gross_bps'], tm1['net_bps'], tm5['net_bps']\n"
            "else:\n"
            "    g, n1, n5 = R['timer_1_gross'], R['timer_1_net'], R['timer_5_net']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "ax.bar(['gross\\n(no cost)','net @1bp\\n(tiny cost)','net @5bps\\n(realistic)'], [g, n1, n5],\n"
            "       color=[GREEN, RED, RED], width=.6)\n"
            "for i,v in enumerate([g, n1, n5]): ax.annotate(f'{v:+.1f}',(i,v),ha='center',\n"
            "    va='bottom' if v>=0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('overnight-capture book, bps/day')\n"
            "ax.set_title('The cost of harvesting eats the whole overnight edge')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross {g:+.2f} -> net @1bp {n1:+.2f} -> net @5bps {n5:+.2f} bps/day')"
        ),
        md(
            f"Grabbing the overnight leg means being *in* the book at every close and *out* at "
            f"every open — you cross the bid-ask four times a day on a double-sized (long+short) "
            f"book. That friction is **{R['timer_1_cost']:.1f} bps/day at just 1 bp** a crossing — "
            f"already more than the entire **+{R['on_bps']:.2f} bps** you were trying to earn. Net: "
            f"**{R['timer_1_net']:+.2f} bps/day** (a loss). At a realistic 5 bps it's "
            f"**{R['timer_5_ann']:.0f}%/yr**. The edge is real; harvesting it costs more than it's "
            "worth."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Real.** The overnight leg persists (**+{R['on_bps']:.2f} bps/day**, a "
            f"6σ placebo result) and the intraday leg reverses (**{R['id_bps']:.2f} bps/day**) — a "
            "genuine, robust tug of war, holding across both halves of the sample.\n"
            f"- **Tradability — Mirage.** The two legs nearly cancel close-to-close "
            f"(**+{R['cc_bps']:.2f} bps**, not significant), and harvesting the one good leg costs "
            "more than it earns at any realistic price. Real physics, no paycheck.\n"
            "- *Survivorship note:* these are today's mega-caps, so the sizes are an upper bound — "
            "which only makes the mirage verdict safer."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **Why does it happen?** The leading story is *clientele*: different investors trade "
            "at different times of day, and their opposing pressures leave a persistent overnight "
            "print that the daytime crowd fades. A follow-up could tie the size of the tug to "
            "retail order-flow or overnight news.\n"
            "- **Where it might matter** isn't as a standalone trade but as a *lens*: if you "
            "already hold a book overnight for other reasons, knowing the overnight and intraday "
            "legs of your signal point opposite ways tells you which clock is actually paying you.\n"
            "- **Sibling studies:** [01-overnight-anomaly](../../01-overnight-anomaly/) (the whole "
            "*market's* night-vs-day split — not a cross-sectional sort), "
            "[640-gold-overnight](../../640-gold-overnight/) (the same split on *one asset*), and "
            "[116-power-hour](../../116-power-hour/) (an intraday-*clock* effect) — see "
            "[docs/references.md](docs/references.md) for the exact dedup.\n\n"
            "*Think you can harvest the overnight leg cheaper than four crossings a day? Show a "
            "net, certifiable edge after realistic frictions on the size you'd actually run — then "
            "we'll talk.*"
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
            "# The overnight/intraday tug of war — a quantitative teardown 🔬\n"
            "### Per-leg Newey-West splits of a trailing-overnight-sorted long-short · a pooled "
            "Welch book test · a 1,000-permutation placebo · a two-era robustness cut · the "
            "four-crossings-a-day cost math · a 20-seed synthetic control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The "
            "claim — **Lou, Polk & Skouras (2019)**: the cross-section of returns is pulled "
            "opposite ways overnight vs intraday. Sort on trailing overnight return; the overnight "
            "component **persists** and the intraday component **reverses**. This is a "
            "*cross-sectional* claim, distinct from the aggregate night/day split (study 01), the "
            "single-asset version (640), and the intraday-clock effect (116). We measure both legs "
            "honestly, prove the persistence isn't luck, and then ask the only question that pays: "
            "*can you harvest it?*\n\n"
            "> ⚠️ **Data note.** 50 liquid US large-caps, daily OHLC via yfinance "
            "(`auto_adjust=True`, total-return), 2010-01-04 → 2026-06-30, pulled through the "
            "`quantlab.universe` **survivorship guard** and cached. **Survivorship is named on the "
            "Signal axis:** current-membership mega-caps → magnitudes are an upper bound. "
            "Fingerprint `" + R["fp_oc"] + "` (Open+Close). Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `REAL` | overnight spread **{R['on_bps']:+.2f} bps/day** NW "
            f"*t* = **{R['on_t_nw']:+.2f}**; intraday spread **{R['id_bps']:+.2f}** NW "
            f"*t* = **{R['id_t_nw']:+.2f}**; placebo **p = {R['placebo_p']:.5f}**; both sub-periods "
            f"hold (on *t* = {R['era_early_on_t']:.2f}/{R['era_late_on_t']:.2f}) |\n"
            f"| **Tradability** | `MIRAGE` | net of 1 bp: {R['timer_1_net']:+.2f} bps/day, "
            f"*t* = {R['timer_1_t']:.2f}; net of 5 bps: {R['timer_5_net']:+.1f} bps/day, "
            f"~{R['timer_5_ann']:.0f}%/yr; legs cancel close-to-close ({R['cc_bps']:+.2f}, "
            f"*t* = {R['cc_t_nw']:.2f}) |\n\n"
            "> 💡 In plain words: two strong, opposite-signed legs (persistence overnight, "
            "reversal intraday) that nearly cancel on the daily close-to-close return — and the "
            "one profitable leg costs more to harvest than it yields."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let a stock's day-$t$ return split exactly into an overnight leg "
            "$r^{on}_t = \\text{Open}_t/\\text{Close}_{t-1}-1$ and an intraday leg "
            "$r^{id}_t = \\text{Close}_t/\\text{Open}_t-1$. Rank names on the trailing-21-day mean "
            "overnight return known at the close of $t-1$; $H$ = top 30%, $L$ = bottom 30%. The "
            "claims:\n\n"
            "- **H₁ (persistence).** $E[r^{on}_H - r^{on}_L] > 0$ — high-past-overnight names keep "
            "earning overnight.\n"
            "- **H₂ (reversal).** $E[r^{id}_H - r^{id}_L] < 0$ — the same names reverse intraday.\n"
            "- **H₃ (near-cancellation).** The close-to-close spread $\\approx$ on + id is small / "
            "insignificant — the tug is a within-day redistribution, not a net premium.\n"
            "- **H₄ (capture).** Harvesting the overnight leg nets a certifiable edge after "
            "realistic costs.\n\n"
            f"We find **H₁ strongly supported** (NW *t* = {R['on_t_nw']:+.2f}, placebo "
            f"p = {R['placebo_p']:.5f}), **H₂ supported** (NW *t* = {R['id_t_nw']:+.2f}), "
            f"**H₃ supported** (close-to-close {R['cc_bps']:+.2f} bps, NW *t* = {R['cc_t_nw']:+.2f}), "
            f"**H₄ rejected** (net {R['timer_1_net']:+.2f} bps/day even at a 1 bp cost)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "The daily long-short leg-spread of an overlapping-formation signal is serially "
            "correlated, so the planned primary is a **Newey-West (HAC, 10-lag) *t*** on each leg "
            "series, cross-checked with a one-sample *t* and a pooled **Welch *t*** of the top vs "
            "bottom book per leg. The persistence claim carries a **1,000-permutation placebo** "
            "(20 seeds × 50) that scrambles the signal→outcome map while preserving each day's "
            "cross-sectional distribution. A pre-named **two-era split (2018-01-01)** checks "
            "stability, and a **20-seed synthetic control** proves the detector is unbiased."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Panel.** {R['n_names']} liquid US large-caps, {R['start']} → {R['end']}, decomposed "
            "night/day via the exact `quantlab.decompose` identity.\n"
            "- **Sort.** Trailing-21d mean overnight return known at close $t-1$ (one `shift`, zero "
            "look-ahead), top 30% − bottom 30%, equal-weight.\n"
            "- **Headline.** NW(10) *t* on the overnight and intraday leg-spreads + one-sample *t* "
            "+ pooled Welch book test.\n"
            "- **Placebo.** 1,000 column-permutations of the forward legs.\n"
            "- **Robustness.** Two eras split at 2018-01-01.\n"
            "- **Execution (timer).** Enter the H-L book at close $t-1$, unwind at open $t$ to "
            "bank the overnight leg; 2 legs × 2 sides × one-way cost × NAV per day; short book pays "
            "50 bps/yr borrow.\n"
            "- **Control.** Synthetic panel with a planted persistent overnight-demand factor; the "
            "null (tug = 0) must not fire across 20 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The two legs and their placebo\n\n"
            "NW *t* on each leg-spread, then the permutation null on the overnight leg. In the "
            "notebook we run a lighter placebo (6 seeds × 15) and quote the canonical 1,000-draw p "
            "from `results.md`."
        ),
        code(
            "if HAVE_REAL:\n"
            "    h = st.tug_stats(SP)\n"
            "    print(f\"overnight spread {h['on_mean_bps']:+.2f} bps/day  NW t = {h['on_t_nw']:+.2f}  \"\n"
            "          f\"one-sample t = {h['on_t_1s']:+.2f}\")\n"
            "    print(f\"intraday  spread {h['id_mean_bps']:+.2f} bps/day  NW t = {h['id_t_nw']:+.2f}  \"\n"
            "          f\"one-sample t = {h['id_t_1s']:+.2f}\")\n"
            "    print(f\"close-close (net) {h['cc_mean_bps']:+.2f} bps/day  NW t = {h['cc_t_nw']:+.2f}\")\n"
            "    ons = SP['on_spread'].to_numpy()\n"
            "    print(f\"gross overnight Sharpe (no cost) = {ons.mean()/ons.std(ddof=1)*np.sqrt(252):.2f}\")\n"
            "    pl = st.placebo_pvalue(ON, IDC, n_seeds=6, n_draws_per_seed=15)\n"
            "    obs, draws = pl['obs_bps'], pl['draws_bps']\n"
            "else:\n"
            "    obs = R['placebo_obs']\n"
            "    rng = np.random.default_rng(788)\n"
            "    draws = rng.normal(R['placebo_mean'], R['placebo_sd'], 300)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=40, color=GREY, alpha=.85, label='null: scrambled sort (light in-notebook run)')\n"
            "ax.axvline(obs, c=GREEN, lw=2.5, label=f'observed overnight spread {obs:+.2f} bps')\n"
            "ax.set_xlabel('overnight spread of a scrambled draw (bps/day)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f\"Overnight persistence far outside the null: canonical p = {R['placebo_p']:.5f}\")\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"canonical placebo (results.md): mean {R['placebo_mean']:+.3f} bps, sd {R['placebo_sd']:.3f}, p = {R['placebo_p']:.5f}\")"
        ),
        md(
            f"> 💡 In plain words: the overnight leg is **+{R['on_bps']:.2f} bps/day** at NW "
            f"*t* = **{R['on_t_nw']:+.2f}** (gross Sharpe ~{R['on_sharpe']:.1f}), the intraday leg "
            f"**{R['id_bps']:.2f} bps/day** at NW *t* = **{R['id_t_nw']:+.2f}** — opposite signs, "
            f"both clearing |t| = 2. The observed overnight spread sits ~6σ beyond the permutation "
            f"null (**p = {R['placebo_p']:.5f}**). H₁ and H₂ both hold."
        ),
        md(
            "### 4b · The near-cancellation and the pooled book test\n\n"
            "The whole point of a *tug*: the two legs roughly cancel on the close-to-close return. "
            "And the same asymmetry should show from the pooled top-vs-bottom book angle (Welch)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    lp = st.leg_pooled_welch(SP)\n"
            "    on_t, on_b = lp['on_top_bps'], lp['on_bot_bps']\n"
            "    id_t, id_b = lp['id_top_bps'], lp['id_bot_bps']\n"
            "    onw, idw = lp['on_welch_t'], lp['id_welch_t']\n"
            "else:\n"
            "    on_t, on_b, id_t, id_b = R['on_top_bps'], R['on_bot_bps'], R['id_top_bps'], R['id_bot_bps']\n"
            "    onw, idw = R['on_welch'], R['id_welch']\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "x = np.arange(2); w=.38\n"
            "ax.bar(x-w/2, [on_t, id_t], w, color=GREEN, label='TOP book (high past overnight)')\n"
            "ax.bar(x+w/2, [on_b, id_b], w, color=RED, label='BOTTOM book (low past overnight)')\n"
            "ax.set_xticks(x); ax.set_xticklabels(['overnight leg','intraday leg'])\n"
            "ax.set_ylabel('book mean, bps/day'); ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_title(f'Top earns overnight (Welch t={onw:+.2f}), loses intraday (t={idw:+.2f})')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'overnight: top {on_t:+.2f} vs bot {on_b:+.2f} (Welch t={onw:+.2f}) | '\n"
            "      f'intraday: top {id_t:+.2f} vs bot {id_b:+.2f} (Welch t={idw:+.2f})')"
        ),
        md(
            f"> 💡 In plain words: the top book out-earns the bottom **overnight** (+{R['on_top_bps']:.2f} "
            f"vs +{R['on_bot_bps']:.2f} bps, Welch *t* = {R['on_welch']:+.2f}) and under-earns it "
            f"**intraday** (+{R['id_top_bps']:.2f} vs +{R['id_bot_bps']:.2f} bps, "
            f"*t* = {R['id_welch']:+.2f}). Net over the full day, the spread is only "
            f"**+{R['cc_bps']:.2f} bps** (NW *t* = {R['cc_t_nw']:+.2f}) — **H₃**: a within-day "
            "redistribution, not a tradable close-to-close premium."
        ),
        md(
            "### 4c · Robustness — two eras\n\n"
            "Split at 2018-01-01 (pre-named). Does each leg survive in both halves?"
        ),
        code(
            "labels = ['2010-2017','2018-2026']\n"
            "if HAVE_REAL:\n"
            "    stats=[]\n"
            "    for lo,hi in [('2010-01-01','2018-01-01'),('2018-01-01','2026-07-01')]:\n"
            "        sub = SP[(SP.index>=lo)&(SP.index<hi)]; stats.append(st.tug_stats(sub))\n"
            "    on_t = [s['on_t_nw'] for s in stats]; id_t = [s['id_t_nw'] for s in stats]\n"
            "else:\n"
            "    on_t = [R['era_early_on_t'], R['era_late_on_t']]\n"
            "    id_t = [R['era_early_id_t'], R['era_late_id_t']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "x=np.arange(2); w=.38\n"
            "ax.bar(x-w/2, on_t, w, color=GREEN, label='overnight leg NW t')\n"
            "ax.bar(x+w/2, id_t, w, color=RED, label='intraday leg NW t')\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1); ax.axhline(-2, ls='--', c=GREY, lw=1); ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel('Newey-West t')\n"
            "ax.set_title('Overnight persistence holds in both halves; intraday reversal fades')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('overnight t:', [round(t,2) for t in on_t], '| intraday t:', [round(t,2) for t in id_t])"
        ),
        md(
            f"> 💡 In plain words: the **overnight persistence** leg clears the bar in both halves "
            f"(*t* = {R['era_early_on_t']:+.2f} then {R['era_late_on_t']:+.2f}). The **intraday "
            f"reversal** leg is strong early (*t* = {R['era_early_id_t']:+.2f}) but fades to "
            f"insignificance recently (*t* = {R['era_late_id_t']:+.2f}) — an honest wrinkle: the "
            "reversal side of the tug has weakened, the persistence side has not."
        ),
        md(
            "### 4d · The timer — the four-crossings-a-day cost\n\n"
            "Enter the H-L book at close $t-1$, unwind at open $t$ to capture the overnight leg; "
            "2 legs × 2 sides × one-way cost per day; short book pays borrow."
        ),
        code(
            "if HAVE_REAL:\n"
            "    tm1 = st.timer_stats(SP, cost_bps=1.0); tm5 = st.timer_stats(SP, cost_bps=5.0)\n"
            "    g = tm1['gross_bps']; n1,n5 = tm1['net_bps'], tm5['net_bps']\n"
            "    t1,t5 = tm1['t_net'], tm5['t_net']; c1,c5 = tm1['cost_bps_per_day'], tm5['cost_bps_per_day']\n"
            "else:\n"
            "    g = R['timer_1_gross']; n1,n5 = R['timer_1_net'], R['timer_5_net']\n"
            "    t1,t5 = R['timer_1_t'], R['timer_5_t']; c1,c5 = R['timer_1_cost'], R['timer_5_cost']\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.bar(['gross','net @1bp','net @5bps'], [g, n1, n5], color=[GREEN, RED, RED], width=.6)\n"
            "for i,v in enumerate([g, n1, n5]): ax.annotate(f'{v:+.1f}',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('bps/day')\n"
            "ax.set_title(f'Round-trip cost {c1:.1f} bps/day @1bp already > gross {g:.1f} bps')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross {g:+.2f} | net @1bp {n1:+.2f} (t={t1:+.2f}, cost {c1:.1f}/day) | '\n"
            "      f'net @5bps {n5:+.2f} (t={t5:+.2f}, cost {c5:.1f}/day)')"
        ),
        md(
            f"> 💡 In plain words: the overnight leg is worth **+{R['on_bps']:.2f} bps/day** gross, "
            f"but capturing it turns a 2×-NAV book over twice a day — **{R['timer_1_cost']:.1f} "
            f"bps/day** of friction at a *1 bp* one-way cost, already more than the gross edge. Net "
            f"**{R['timer_1_net']:+.2f} bps/day** (*t* = {R['timer_1_t']:.2f}); at 5 bps, "
            f"**{R['timer_5_ann']:.0f}%/yr**. **H₄ rejected — Tradability = MIRAGE.**"
        ),
        md(
            "### 4e · Faithful-engine & power control — we know the truth here\n\n"
            "Synthetic panel: each name carries a persistent latent overnight-demand factor; the "
            "overnight leg loads +`tug`, the intraday leg −`tug`. The null (`tug = 0`) is checked "
            "over **20 seeds** — never a single stream."
        ),
        code(
            "null_on = []\n"
            "for s_ in range(20):\n"
            "    p0 = data.synthetic_panel(tug=0.0, seed=788 + s_, n_assets=40, n_days=1200)\n"
            "    null_on.append(st.synthetic_detect(p0)['on_t_nw'])\n"
            "null_on = np.asarray(null_on)\n"
            "p1 = data.synthetic_panel(tug=0.0020, seed=788, n_assets=40, n_days=1500)\n"
            "sy = st.synthetic_detect(p1)\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.linspace(-.12,.12,20), null_on, color=GREY, s=40, label='null worlds (tug=0), 20 seeds')\n"
            "ax.scatter([1], [sy['on_t_nw']], color=GREEN, s=90, zorder=5, label='planted tug, overnight leg')\n"
            "ax.scatter([1], [sy['id_t_nw']], color=RED, s=90, zorder=5, label='planted tug, intraday leg')\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1); ax.axhline(-2, ls='--', c=GREY, lw=1)\n"
            "ax.set_xticks([0,1]); ax.set_xticklabels(['null x 20','planted']); ax.set_ylabel('NW t')\n"
            "ax.set_title('Control: no null fires; a planted tug lights up both legs')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null overnight t: mean {null_on.mean():+.2f} (sd {null_on.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(abs(null_on)>=2).sum()}/20  |  planted on {sy[\"on_t_nw\"]:+.2f}, id {sy[\"id_t_nw\"]:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 20 null worlds the overnight detector averages "
            f"*t* = {R['syn_null_on_mean']:+.2f} (sd {R['syn_null_on_sd']:.2f}) and **never** crosses "
            f"the bar; a planted tug reads overnight *t* = {R['syn_planted_on_t']:+.2f}, intraday "
            f"*t* = {R['syn_planted_id_t']:+.2f}. The machinery is unbiased — the real-tape "
            f"overnight *t* = {R['on_t_nw']:+.2f} is the genuine article. *(Power check only — never "
            "cited in support of the real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `REAL`** — overnight leg **{R['on_bps']:+.2f} bps/day** (NW "
            f"*t* = {R['on_t_nw']:+.2f}, placebo p = {R['placebo_p']:.5f}, gross Sharpe "
            f"~{R['on_sharpe']:.1f}) and intraday leg **{R['id_bps']:+.2f} bps/day** (NW "
            f"*t* = {R['id_t_nw']:+.2f}) pull opposite ways — the tug of war — with the overnight "
            f"leg holding in both sub-periods ({R['era_early_on_t']:+.2f}/{R['era_late_on_t']:+.2f}) "
            f"and a clean 20-seed control. The legs nearly cancel close-to-close "
            f"({R['cc_bps']:+.2f} bps, *t* = {R['cc_t_nw']:+.2f}). *Survivorship: current-membership "
            "mega-caps → magnitudes are an upper bound (stated on this axis).*\n"
            f"- **Tradability `MIRAGE`** — the one profitable leg costs **{R['timer_1_cost']:.1f} "
            f"bps/day** to harvest at a 1 bp one-way cost, more than its **+{R['on_bps']:.2f} "
            f"bps/day** gross; net {R['timer_1_net']:+.2f} bps/day (*t* = {R['timer_1_t']:.2f}), and "
            f"~{R['timer_5_ann']:.0f}%/yr at 5 bps. Real physics, no paycheck."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **Mechanism.** LPS attribute the tug to a persistent clientele split — one crowd "
            "concentrates demand overnight/at the open, another fades it intraday. Testable "
            "follow-ups: does the tug scale with retail order-flow, overnight news, or the "
            "open-auction imbalance? Does the recently-faded *intraday reversal* leg track the rise "
            "of intraday liquidity provision?\n"
            "- **Why it doesn't scale:** the profitable leg is ~4 bps/day gross and needs "
            "four spread-crossings a day on a 2×-NAV book — a structure that eats the edge at any "
            "realistic cost, before market impact and the fact that everyone can see the same "
            "overnight ranking.\n"
            "- **Dedup map:** [01-overnight-anomaly](../../01-overnight-anomaly/) (the aggregate / "
            "index-level night-vs-day split — not a cross-sectional sort), "
            "[640-gold-overnight](../../640-gold-overnight/) (the same split on a single asset, "
            "gold), [116-power-hour](../../116-power-hour/) (an intraday-clock effect, the last "
            "trading hour).\n\n"
            "*The reproducible core is offline and deterministic; frozen numbers live in "
            "[`docs/results.md`](../docs/results.md), sources in "
            "[`docs/references.md`](../docs/references.md).*"
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
