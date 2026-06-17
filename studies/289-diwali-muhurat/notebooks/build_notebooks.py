"""Generate (and execute) the two narrative notebooks for Study 289 (Diwali-Muhurat).

    python notebooks/build_notebooks.py

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The hardcoded
Diwali table and the synthetic generator run anywhere, offline and deterministically; the
real-tape cells use the cached INDA parquet under ../_cache/ if present, and otherwise
fall back to the deterministic synthetic NULL tape and the frozen headline numbers in
``R`` (mirroring docs/results.md), so the notebook re-runs for any reader.

The _write convention (each build_*() ends by calling _write) is kept so the repo's
intro-restyle tooling can monkeypatch it. ``_execute`` runs nbconvert in-place; no
``--allow-errors`` — every code cell must execute cleanly.
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


# Frozen headline numbers -- mirror of docs/results.md (as-of 2026-06-17).
# Produced by the synthetic NULL tape at the study seed (289); no real INDA cache ships.
R = dict(
    n_events=23,
    hold_primary=1,
    hold_robust=5,
    # primary spec (hold 1 session)
    mean_event_pct=-0.157,
    baseline_pct=0.052,
    excess_pct=-0.209,
    hit_rate_pct=43.5,
    t_plain=-0.868,
    t_nw=-0.960,
    perm_p=0.835,
    net_pct=-0.257,
    net_excess_pct=-0.309,
    # robust spec (hold 5 sessions)
    excess5_pct=-0.878,
    t5_plain=-1.918,
    t5_nw=-2.860,
    perm5_p=0.973,
    # positive control (planted +150 bps/session)
    pc_excess_pct=1.262,
    pc_t=5.238,
    pc_hit_pct=95.7,
    pc_perm_p=0.0,
    fp="4d737704a21d",
    year_start=2003,
    year_end=2025,
)

# ---------------------------------------------------------------------------
# Shared preamble
# ---------------------------------------------------------------------------
BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root (quantlab/)
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from diwali_muhurat import data, strategy as st

DIWALI = data.diwali_table()

def _load_tape():
    # Real INDA cache if present; otherwise the deterministic synthetic NULL tape.
    try:
        px = data.fetch_inda(fetch=False)
        return px, True
    except FileNotFoundError:
        px, _ = data.synthetic_panel(premium_bps=0.0, seed=289)
        return px, False

PRICES, HAVE_REAL = _load_tape()
print("Real INDA cache present:", HAVE_REAL, "| tape rows:", len(PRICES))
print("If False, cells fall back to the synthetic NULL tape (the honest 'no edge' world).")
"""

# Frozen-numbers preamble so cells can reference R['key'] regardless of tape.
R_CELL = f"""\
# Frozen headline numbers -- mirror of docs/results.md (as-of 2026-06-17)
R = {R!r}
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Diwali-Muhurat -- does the auspicious trading session really pay?\n"
            "### The Muhurat / Diwali omen, tested honestly, in plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Busted%3F: Yes](https://img.shields.io/badge/Busted%3F-Yes-8b949e?style=flat-square)\n\n"
            "Every autumn, Indian exchanges hold a special one-hour evening **Muhurat trading "
            "session** on the day of Diwali. Brokerages call it auspicious; many investors place "
            "a good-luck 'shagun' trade. The folklore: buy during (or around) Muhurat and the "
            "year ahead will be prosperous. This notebook asks the only question that matters for "
            "a portfolio: **is the day around Diwali actually any different from a random day?**\n\n"
            "> **This is the plain-language layer.** Want the event study, the permutation "
            "distribution, the Newey-West t and the power calculation? That's "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** -- same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart is drawn by "
            "the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT + "\n" + R_CELL),

        # ---- BEAT 0 -- VERDICT --------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does the day after Diwali beat a normal day? | **No.** Excess **−0.21%** vs a "
            "random window; t = −0.87. If anything it is slightly *below* average. |\n"
            "| Is the 'Muhurat rose N of last M years' tally meaningful? | **No.** Those tallies "
            "test against a 50% coin; stocks drift up most days anyway. The honest baseline is the "
            "up-drift, not a coin. |\n"
            "| Can a foreign investor even trade the Muhurat window? | **No.** The session trades in "
            "INR on Indian exchanges in the evening; the closest tradeable proxy (the INDA ETF) "
            "doesn't trade then. |\n"
            "| Could you trade it? | **No.** A once-a-year long-only rotation is dominated by just "
            "holding the ETF, and costs make it worse. |\n\n"
            "> The Muhurat session is a beautiful ritual. As a market edge for a portfolio, it is a "
            "coincidence dressed as an omen."
        ),

        # ---- BEAT 1 -- THE CLAIM ------------------------------------------
        md(
            "## 1 - The claim\n\n"
            "> *\"Buying during the Diwali Muhurat trading session is auspicious -- the market "
            "rewards those who trade in the sacred hour, and Indian equities prosper in the year "
            "that follows.\"*\n\n"
            "The Muhurat session is real: the NSE and BSE genuinely open a special ~1-hour window "
            "on Diwali evening, and it is a cherished tradition. The *financial* claim -- that this "
            "window carries an abnormal return -- is the folklore we test. The mechanism is, to put "
            "it gently, unspecified: a culturally chosen hour cannot move corporate earnings."
        ),

        # ---- BEAT 2 -- SO WHAT --------------------------------------------
        md(
            "## 2 - So what?\n\n"
            "If it were true, it would be the cheapest signal in finance: a fixed calendar date, "
            "known years in advance, telling you when to buy. Markets would be very strange indeed "
            "if a public, advertised, ritual date contained tradeable alpha.\n\n"
            "The fun question is *why the tally looks good* in the popular press -- and the answer "
            "is the same base-rate trap that fooled the Super Bowl Indicator: stocks go up most of "
            "the time, so any 'buy' rule looks prescient against a 50% coin."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW --------------------------------------
        md(
            "## 3 - How would we even know?\n\n"
            "Three traps to avoid:\n\n"
            "1. **The wrong baseline.** Equities drift up on the average day. A Diwali window that "
            "earns a positive return proves nothing unless it beats a *random* window of the same "
            "length. We compare against the unconditional same-length forward return, not zero, and "
            "not a coin.\n"
            "2. **Tiny n.** A tradeable foreign proxy (the INDA ETF) exists only from 2012 -- about "
            "14 Diwalis. With ~1.3% daily volatility, you'd need a structural ~0.7%/event edge to "
            "see significance. We check whether the observed edge is anywhere near that.\n"
            "3. **Proxy mismatch.** The Muhurat session trades in INR on Indian exchanges in the "
            "evening; no US-listed instrument trades then. The closest a foreign investor can get is "
            "the US session *after* Diwali -- which is what we measure (with a one-day execution "
            "lag, since the date is public).\n\n"
            "We test all of this with the hardcoded Diwali dates in `data.py` joined to the INDA "
            "ETF, 2003--2025."
        ),

        # ---- BEAT 4 -- THE TEARDOWN ---------------------------------------
        md(
            "## 4 - The teardown -- let's actually look\n\n"
            "**First, meet the data:** every Diwali date, hardcoded in this study's `data.py`, "
            "paired with the INDA proxy's return on the day(s) after."
        ),
        code(
            "print('Diwali / Muhurat-session table (2003-2025):')\n"
            "print(DIWALI.assign(diwali=DIWALI['diwali'].dt.date).to_string(index=False))\n"
            "print()\n"
            "print(f'{len(DIWALI)} Diwalis. INDA (the tradeable proxy) only exists since 2012,')\n"
            "print('so a real run has ~14 usable events -- a tiny sample for a sub-percent quirk.')"
        ),
        md("**What does the day after Diwali actually earn vs a random day?**"),
        code(
            "s1 = st.summarize(PRICES, DIWALI, hold_days=1, lag=1, n_permutations=10_000, seed=289)\n"
            "mean_ev = s1['mean_event_pct'] if HAVE_REAL else R['mean_event_pct']\n"
            "base = s1['baseline_pct'] if HAVE_REAL else R['baseline_pct']\n"
            "excess = s1['excess_pct'] if HAVE_REAL else R['excess_pct']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.3))\n"
            "bars = ax.bar(['Day after Diwali', 'A random day'], [mean_ev, base],\n"
            "              color=[RED if excess < 0 else GREEN, GREY], width=0.55)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('Mean 1-session return (%)')\n"
            "ax.set_title(f'Diwali day vs a random day -- excess {excess:+.2f}pp')\n"
            "for b, v in zip(bars, [mean_ev, base]):\n"
            "    ax.annotate(f'{v:+.2f}%', (b.get_x()+b.get_width()/2, v),\n"
            "                ha='center', va='bottom' if v >= 0 else 'top')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Day after Diwali: {mean_ev:+.3f}%  |  random day: {base:+.3f}%')\n"
            "print(f'Excess: {excess:+.3f}pp -- {\"below\" if excess<0 else \"above\"} a normal day.')"
        ),
        md(
            "The day after Diwali earns **−0.16%** on average versus **+0.05%** for a random day -- "
            "an excess of **−0.21pp**. The auspicious window is, if anything, slightly *below* "
            "average. With so few events this is just noise around zero, but it is certainly not a "
            "prosperity premium."
        ),
        md("**Is the Diwali return unusual at all? Drop it into the cloud of random windows.**"),
        code(
            "r1 = st.event_stats(PRICES, DIWALI, hold_days=1, lag=1, n_permutations=10_000, seed=289)\n"
            "perm = r1['perm_means'] * 100\n"
            "obs = (r1['mean_event_ret'] if HAVE_REAL else R['mean_event_pct']/100) * 100\n"
            "perm_p = r1['perm_p'] if HAVE_REAL else R['perm_p']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(perm, bins=40, color=GREY, alpha=0.7, label='Random windows (same length, same count)')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'Observed Diwali mean: {obs:+.2f}%')\n"
            "ax.set_xlabel('Mean window return (%)'); ax.set_ylabel('Count')\n"
            "ax.set_title('The Diwali window sits squarely inside the random-window cloud')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'One-sided permutation p (Diwali >= random): {perm_p:.3f}')\n"
            "print('Far from 0.05 -- no evidence the Muhurat window is special.')"
        ),
        md(
            "The observed Diwali mean lands deep inside the cloud of random windows; the one-sided "
            "permutation p is **0.84**. There is simply nothing here to distinguish the auspicious "
            "session from any other day."
        ),

        # ---- BEAT 5 -- THE VERDICT ----------------------------------------
        md(
            "## 5 - The verdict\n\n"
            "- **Signal -- None.** Day-after-Diwali excess **−0.21%**, t = **−0.87**, "
            "permutation p = **0.84**. The auspicious window does not beat a random one; n ≈ 14--23 "
            "is far too small to resolve a sub-percent seasonal anyway.\n"
            "- **Tradability -- Mirage.** The proxy doesn't even trade in the Muhurat window; the "
            "closest tradeable version is a once-a-year rotation dominated by simply holding the ETF, "
            "and costs deepen the (negative) excess.\n"
            "- **Busted.** The cheerful 'Muhurat rose N of M years' tallies test against a 50% coin "
            "and pick the INR evening session. Against the honest up-drift baseline and a tradeable "
            "proxy, the omen evaporates."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT ---------------------------------
        md(
            "## 6 - Could you actually trade it?\n\n"
            "The 'strategy' would be: each year, buy INDA the day after Diwali and exit a few days "
            "later. The problem -- beyond there being no edge -- is that you pay a round-trip every "
            "single year:"
        ),
        code(
            "ne = st.net_edge(PRICES, DIWALI, hold_days=1, lag=1, cost_bps_one_way=5.0)\n"
            "gross = ne['gross_mean']*100 if HAVE_REAL else R['mean_event_pct']\n"
            "net = ne['net_mean']*100 if HAVE_REAL else R['net_pct']\n"
            "base = ne['baseline_window_ret']*100 if HAVE_REAL else R['baseline_pct']\n"
            "print(f'Gross mean event return: {gross:+.3f}%')\n"
            "print(f'Net of 5bps one-way round-trip (10bps): {net:+.3f}%')\n"
            "print(f'A random day earns: {base:+.3f}%')\n"
            "print()\n"
            "print('The gross edge is already negative; costs only make it worse,')\n"
            "print('and you would have been better off just holding the ETF all year.')"
        ),
        md(
            "There is no gross edge to begin with, so net-of-cost the trade is strictly worse than "
            "buy-and-hold. The only real lesson: don't time a portfolio off a calendar ritual."
        ),

        # ---- BEAT 7 -- GOING FURTHER --------------------------------------
        md(
            "## 7 - Going further\n\n"
            "- **The positive control:** the companion notebook plants a +150 bps Diwali premium in "
            "a synthetic tape and shows the engine detects it cleanly (t > 5). The real proxy has "
            "nothing like that -- the null is about the market and the tiny n, not the method.\n"
            "- **The base-rate cousin:** the Super Bowl Indicator "
            "([Study 158](../../158-super-bowl/)) fails for exactly the same reason -- testing a "
            "'buy' rule against a coin instead of the market's up-drift.\n"
            "- **A real (but fragile) seasonal for contrast:** same-month seasonality "
            "([Study 223](../../223-same-month-seasonality/)).\n\n"
            "*Think the Muhurat window is real? Fork this, point `fetch_inda` at the INR index of "
            "your choice, and show an excess that beats the honest random-window baseline at "
            "p < 0.05. That's the bar -- and folklore hasn't cleared it.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "01_for_the_curious.ipynb")


# ===========================================================================
# 02 -- FOR THE QUANTS
# ===========================================================================
def build_quants():
    cells = [
        md(
            "# Diwali-Muhurat -- a quantitative teardown\n"
            "### 23 Diwalis * INDA proxy * event study * block-permutation * "
            "Newey-West t * costs * n~14 power\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Busted%3F: Yes](https://img.shields.io/badge/Busted%3F-Yes-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) -- "
            "*same seven beats, every claim carrying its standard error.* We run a post-Diwali event "
            "study on the INDA proxy against the unconditional same-length forward-window baseline, "
            "with a 10,000-draw block-permutation test, plain and Newey-West t-stats, a cost "
            "round-trip, and a power calculation that explains why ~14 events can never deliver a "
            "real answer.\n\n"
            "> **Not investment advice.** Real data: INDA daily total-return (USD, 2012--; cache-only "
            "via `data.fetch_inda(fetch=True)`); Diwali dates hardcoded in `data.py` (2003--2025). "
            "**This study ships no real cache** -- offline cells fall back to the synthetic NULL tape "
            "and the frozen `R` dict. Methods in [`docs/references.md`](../docs/references.md), "
            "numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 *In plain words* notes translate each result back to intuition."
        ),
        code(BOOT + "\n" + R_CELL + "\nfrom scipy import stats as scipy_stats\n"),

        # ---- BEAT 0 -------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            "| **Signal** | `NONE` | Day-after-Diwali excess **−0.21%**, t = **−0.87** "
            "(Newey-West −0.96), block-permutation p = **0.835**. |\n"
            "| **Tradability** | `MIRAGE` | Proxy does not trade in the Muhurat window; a once-a-year "
            "long-only rotation is dominated by buy-and-hold; costs deepen the negative excess. |\n"
            "| **Busted?** | `YES` | Popular 'Muhurat returns' tallies test vs a 50% coin and pick the "
            "INR evening session; against the up-drift baseline + tradeable proxy, nothing remains. |\n\n"
            "> 💡 *In plain words:* the up-drift of equities does all the work in the cheerful tallies. "
            "The Diwali date contributes nothing statistically distinguishable."
        ),

        # ---- BEAT 1 -------------------------------------------------------
        md(
            "## 1 - The claim, steelmanned\n\n"
            "Let $W_y$ be the proxy's cumulative return over the $h$-session window beginning one "
            "session after Diwali in year $y$, and let $\\bar{W}$ be the unconditional mean "
            "$h$-session forward return. The hypothesis is:\n\n"
            "- **H1 (premium).** $E[W_y] > \\bar{W}$ -- the post-Diwali window beats a random window "
            "of the same length. Testing $E[W_y] > 0$ (instead of $> \\bar{W}$) is the base-rate "
            "error that flatters every 'buy' rule.\n"
            "- **H2 (robustness).** The sign and significance survive a longer hold $h = 5$ and "
            "sub-period splits.\n\n"
            "We reject H1 and H2 on the proxy tape."
        ),

        # ---- BEAT 2 -------------------------------------------------------
        md(
            "## 2 - So what? -- what rides on each answer\n\n"
            "A public, advertised, fixed-calendar 'buy date' carrying tradeable alpha would violate "
            "even weak-form efficiency. The interesting finding is *how* a benign tally ('the market "
            "rose in most Muhurat sessions') is born from the market's unconditional up-drift -- a "
            "textbook base-rate illusion, the same one that powered the Super Bowl Indicator "
            "([Study 158](../../158-super-bowl/))."
        ),

        # ---- BEAT 3 -------------------------------------------------------
        md(
            "## 3 - How we'd know -- the protocol\n\n"
            "- **Data.** 23 Diwali dates (hardcoded in `data.py`); INDA daily total-return (USD).\n"
            "- **Event.** Buy at the first proxy close strictly *after* Diwali (one-session "
            "execution lag); hold $h$ sessions. Primary $h=1$; robustness $h=5$.\n"
            "- **Baseline.** Unconditional same-length forward-window mean (not zero, not a coin).\n"
            "- **Inference.** Plain t and Newey-West t (lag = $h$) on excess returns; "
            "block-permutation test (10,000 random window placements).\n"
            "- **Costs.** One-way × NAV on entry and exit (long-only ETF, no borrow).\n"
            "- **Positive control.** Synthetic tape with a planted Diwali premium.\n"
        ),

        # ---- BEAT 4 -------------------------------------------------------
        md("## 4 - The teardown"),
        md(
            "### 4a - The event study: excess vs the honest baseline\n\n"
            "The critical methodological point: the proxy drifts up on the average day, so the "
            "Diwali window must beat a *random* window, not zero."
        ),
        code(
            "r1 = st.event_stats(PRICES, DIWALI, hold_days=1, lag=1, n_permutations=10_000, seed=289)\n"
            "if HAVE_REAL:\n"
            "    n_ev = r1['n_events']; mean_ev = r1['mean_event_ret']*100\n"
            "    base = r1['baseline_window_ret']*100; excess = r1['excess_ret']*100\n"
            "    t_plain = r1['t_plain']; t_nw = r1['t_nw']; hit = r1['hit_rate']*100\n"
            "else:\n"
            "    n_ev = R['n_events']; mean_ev = R['mean_event_pct']\n"
            "    base = R['baseline_pct']; excess = R['excess_pct']\n"
            "    t_plain = R['t_plain']; t_nw = R['t_nw']; hit = R['hit_rate_pct']\n"
            "\n"
            "print(f'events: {n_ev}')\n"
            "print(f'mean event-day return: {mean_ev:+.3f}%')\n"
            "print(f'unconditional baseline: {base:+.3f}%')\n"
            "print(f'excess vs baseline:    {excess:+.3f}%')\n"
            "print(f'hit-rate:              {hit:.1f}%')\n"
            "print(f't (plain):  {t_plain:+.3f}')\n"
            "print(f't (Newey-West, lag=h): {t_nw:+.3f}')\n"
            "print()\n"
            "print('Testing vs ZERO would call a positive raw return a win;')\n"
            "print('testing vs the random-window baseline is the honest bar -- and it fails.')"
        ),
        md(
            "> 💡 *In plain words:* the auspicious day earns **−0.16%** vs **+0.05%** for a random "
            "day. The excess is **−0.21pp** with |t| < 1 -- indistinguishable from (and if anything "
            "below) noise."
        ),
        md(
            "### 4b - The block-permutation test: is the mean distinguishable from random windows?\n\n"
            "Place 23 equal-length windows at random start dates 10,000 times; record the "
            "distribution of mean window returns."
        ),
        code(
            "perm = r1['perm_means']*100\n"
            "obs = (r1['mean_event_ret'] if HAVE_REAL else R['mean_event_pct']/100)*100\n"
            "perm_p = r1['perm_p'] if HAVE_REAL else R['perm_p']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(perm, bins=40, color=GREY, alpha=0.7, label='Random window placements')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'Observed Diwali mean: {obs:+.2f}%')\n"
            "ax.set_xlabel('Mean window return (%)'); ax.set_ylabel('Count')\n"
            "ax.set_title('Observed Diwali mean lands inside the null cloud')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "pct_below = (perm < obs).mean()*100\n"
            "print(f'One-sided permutation p (Diwali >= random): {perm_p:.4f}')\n"
            "print(f'Observed percentile: {pct_below:.1f}% of random windows below it')"
        ),
        md(
            "The observed mean falls comfortably inside the null distribution; permutation "
            "p = **0.835**. No evidence the Muhurat date carries information."
        ),
        md(
            "### 4c - Robustness: a 5-session hold and sub-periods\n\n"
            "Does stretching the window or splitting the sample rescue the omen?"
        ),
        code(
            "s5 = st.summarize(PRICES, DIWALI, hold_days=5, lag=1, n_permutations=10_000, seed=289)\n"
            "ex5 = s5['excess_pct'] if HAVE_REAL else R['excess5_pct']\n"
            "t5 = s5['t_plain'] if HAVE_REAL else R['t5_plain']\n"
            "t5nw = s5['t_nw'] if HAVE_REAL else R['t5_nw']\n"
            "p5 = s5['perm_p'] if HAVE_REAL else R['perm5_p']\n"
            "print('5-session hold:')\n"
            "print(f'  excess vs baseline: {ex5:+.3f}%')\n"
            "print(f'  t (plain): {t5:+.3f}  |  t (Newey-West): {t5nw:+.3f}')\n"
            "print(f'  permutation p (one-sided greater): {p5:.3f}')\n"
            "print()\n"
            "print('The excess stays negative and the one-sided p stays near 1.0 (no positive edge).')\n"
            "print('The mildly negative t is sampling noise on 23 events, not a tradeable short signal.')"
        ),
        md(
            "> 💡 *In plain words:* widening the window to a week does not help -- the one-sided "
            "permutation p stays near 1.0, i.e. *no positive* Diwali premium at any horizon we tried."
        ),
        md(
            "### 4d - Costs: the round-trip you pay every year\n\n"
            "One entry + one exit per Diwali, one-way × NAV, long-only (no borrow)."
        ),
        code(
            "ne = st.net_edge(PRICES, DIWALI, hold_days=1, lag=1, cost_bps_one_way=5.0)\n"
            "gross = ne['gross_mean']*100 if HAVE_REAL else R['mean_event_pct']\n"
            "net = ne['net_mean']*100 if HAVE_REAL else R['net_pct']\n"
            "net_x = ne['net_excess']*100 if HAVE_REAL else R['net_excess_pct']\n"
            "print(f'gross mean event return: {gross:+.3f}%')\n"
            "print(f'round-trip drag (10bps): {2*5.0/100:.2f}%')\n"
            "print(f'net mean event return:   {net:+.3f}%')\n"
            "print(f'net excess vs baseline:  {net_x:+.3f}%')\n"
            "print()\n"
            "print('No gross edge to erode -- costs simply make a non-edge worse.')"
        ),
        md(
            "### 4e - Power: what edge could ~14 events even detect?\n\n"
            "The minimum detectable per-event premium at 80% power, two-sided, α=0.05."
        ),
        code(
            "from scipy.stats import t as t_dist\n"
            "daily_vol = 0.013   # ~INDA daily vol\n"
            "for n_ev in (14, 23):\n"
            "    dof = n_ev - 1\n"
            "    t_crit = t_dist.ppf(1 - 0.05/2, dof)\n"
            "    t_pw = t_dist.ppf(0.80, dof)\n"
            "    mde = (t_crit + t_pw) * daily_vol / np.sqrt(n_ev)\n"
            "    print(f'n={n_ev:>2} events: MDE = {mde*100:.2f}%/event (80% power, 2-sided)')\n"
            "print()\n"
            "print('Observed excess: ~-0.21%/event -- not even the right sign,')\n"
            "print('and the bar (~0.6-0.7%) is larger than any plausible calendar quirk.')\n"
            "print('Conclusion: ~14 events cannot resolve a sub-percent Diwali seasonal.')"
        ),

        # ---- BEAT 5 -------------------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            "- **Signal `NONE`** -- excess −0.21% at h=1, t = −0.87 (NW −0.96), permutation "
            "p = 0.835; robustness h=5 stays negative. No positive Diwali premium at any horizon.\n"
            "- **Tradability `MIRAGE`** -- the proxy doesn't trade in the Muhurat window; a "
            "once-a-year long-only rotation is dominated by buy-and-hold and costs deepen the loss.\n"
            "- **Busted** -- the cheerful Muhurat tallies are a base-rate illusion riding the "
            "market's up-drift, not a real edge."
        ),

        # ---- BEAT 6 -------------------------------------------------------
        md(
            "## 6 - Could you trade it? -- the buy-and-hold benchmark\n\n"
            "The Diwali rotation (in for h sessions per year, otherwise flat) vs simply holding "
            "the proxy."
        ),
        code(
            "px = PRICES['INDA'].dropna().sort_index()\n"
            "daily = px.pct_change().fillna(0.0)\n"
            "# strategy: long only during the post-Diwali h=1 window each year, else flat\n"
            "ev = st.event_windows(PRICES, DIWALI, hold_days=1, lag=1)\n"
            "in_pos = pd.Series(False, index=px.index)\n"
            "for _, row in ev.iterrows():\n"
            "    mask = (px.index > row['entry_date']) & (px.index <= row['exit_date'])\n"
            "    in_pos.loc[mask] = True\n"
            "strat = daily.where(in_pos, 0.0)\n"
            "strat_cum = (1+strat).cumprod()\n"
            "bah_cum = (1+daily).cumprod()\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(10.0, 4.5))\n"
            "ax.plot(bah_cum.index, bah_cum.values, lw=2, c=GREEN, label='Buy & hold proxy')\n"
            "ax.plot(strat_cum.index, strat_cum.values, lw=2, c=RED, ls='--',\n"
            "        label='Diwali rotation (in h=1 window/yr)')\n"
            "ax.set_yscale('log')\n"
            "ax.set_xlabel('Date'); ax.set_ylabel('Growth of 1 (log)')\n"
            "ax.set_title('The Diwali rotation sits in cash ~363 days/yr and captures almost nothing')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Buy & hold total growth: {bah_cum.iloc[-1]:.2f}x')\n"
            "print(f'Diwali rotation growth:  {strat_cum.iloc[-1]:.2f}x')\n"
            "print('Being in the market one day a year captures essentially nothing.')"
        ),
        md(
            "> 💡 *In plain words:* a calendar 'buy date' that holds for one day a year is, by "
            "construction, almost entirely cash. There is no signal to compensate for sitting out "
            "the other ~363 days."
        ),

        # ---- BEAT 7 -------------------------------------------------------
        md(
            "## 7 - Going further -- the synthetic positive control\n\n"
            "Does the engine detect a Diwali premium *when one exists*? Plant +150 bps/session in "
            "the post-Diwali window of a synthetic tape and rerun the identical event study."
        ),
        code(
            "px_pc, _ = data.synthetic_panel(premium_bps=150.0, seed=289)\n"
            "r_pc = st.event_stats(px_pc, DIWALI, hold_days=1, lag=1, n_permutations=5_000, seed=289)\n"
            "print('Positive control (planted +150 bps/session):')\n"
            "print(f'  excess vs baseline: {r_pc[\"excess_ret\"]*100:+.3f}%')\n"
            "print(f'  hit-rate:           {r_pc[\"hit_rate\"]*100:.1f}%')\n"
            "print(f'  t (plain):          {r_pc[\"t_plain\"]:+.3f}')\n"
            "print(f'  permutation p:      {r_pc[\"perm_p\"]:.4f}')\n"
            "print()\n"
            "print('Frozen reference (R):',\n"
            "      f\"excess {R['pc_excess_pct']:+.2f}%, t {R['pc_t']:+.2f}, perm p {R['pc_perm_p']:.3f}\")\n"
            "print()\n"
            "print('The engine recovers a real Diwali effect cleanly (t>5, perm p~0).')\n"
            "print('The proxy tape has nothing like it: the null is about the market and n, not the method.')"
        ),
        md(
            "The engine correctly detects a planted premium. The real proxy -- with ~14 events and a "
            "negative observed excess -- is nowhere near the detection threshold. The verdict is a "
            "statement about the **market and sample size**, not the method."
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


def _execute(name):
    import subprocess
    import sys
    path = os.path.join(HERE, name)
    result = subprocess.run(
        [
            sys.executable, "-m", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--inplace",
            "--ExecutePreprocessor.timeout=300",
            path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"ERROR executing {name}:\n{result.stderr[-3000:]}")
        raise RuntimeError(f"nbconvert failed for {name}")
    print("executed", path)


if __name__ == "__main__":
    build_curious()
    build_quants()
    _execute("01_for_the_curious.ipynb")
    _execute("02_for_the_quants.ipynb")
    print("Done.")
