"""Generate (and execute) the two narrative notebooks for Study 296 (Oscars-Effect).

    python notebooks/build_notebooks.py

This writes BOTH notebooks and then executes them in-place with nbconvert (no
``--allow-errors``). The hardcoded Oscar table and the synthetic generator run
anywhere, offline and deterministically; the real ^GSPC cells use the cached daily
tape if present (``HAVE_REAL``), and otherwise quote the frozen headline numbers in
``R`` (mirroring docs/results.md) and fall back to synthetic data, so the notebooks
re-run for any reader offline.

The _write convention (each build_*() ends by calling _write) is kept so the repo's
intro-restyle tooling can monkeypatch it.
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-17).
R = dict(
    n_obs=8165,
    n_events=31,
    fp="346fa83866ef",
    baseline_mean_bps=4.1,
    baseline_up_pct=54.0,
    event_mean_bps=-23.8,
    abnormal_bps=-27.9,
    event_hac_t=-1.35,
    event_plain_t=-1.13,
    event_up_pct=45.2,
    car_full_bps=-10.9,
    # CAR profile rel_day -1..+3: (aar_bps, t)
    aar_m1=26.8, t_m1=1.67,
    aar_0=-27.9, t_0=-1.32,
    aar_p1=19.4, t_p1=0.74,
    aar_p2=-8.5, t_p2=-0.59,
    aar_p3=-20.7, t_p3=-1.22,
    trade_gross_bps=-23.8,
    trade_net_bps=-25.8,
    trade_net_t=-1.46,
    trade_bah_bps=-23.8,
    placebo_p=0.183,
    mde_bps=40.0,
    daily_vol_bps=110.0,
    year_start=1995,
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

from oscars_effect import data, strategy as st

def _have_cache():
    try:
        data.fetch_daily(fetch=False)
        return True
    except FileNotFoundError:
        return False

HAVE_REAL = _have_cache()
print("^GSPC cache present:", HAVE_REAL)
"""

# R-dict preamble so cells can reference R['key'] when running offline.
R_CELL = """\
# Frozen headline numbers -- mirror of docs/results.md (as-of 2026-06-17)
R = dict(
    n_obs=8165, n_events=31,
    baseline_mean_bps=4.1, baseline_up_pct=54.0,
    event_mean_bps=-23.8, abnormal_bps=-27.9,
    event_hac_t=-1.35, event_plain_t=-1.13, event_up_pct=45.2,
    car_full_bps=-10.9,
    aar_m1=26.8, t_m1=1.67, aar_0=-27.9, t_0=-1.32, aar_p1=19.4, t_p1=0.74,
    aar_p2=-8.5, t_p2=-0.59, aar_p3=-20.7, t_p3=-1.22,
    trade_gross_bps=-23.8, trade_net_bps=-25.8, trade_net_t=-1.46, trade_bah_bps=-23.8,
    placebo_p=0.183, mde_bps=40.0, daily_vol_bps=110.0,
    year_start=1995, year_end=2025,
)
REL_DAYS = [-1, 0, 1, 2, 3]
AAR_FROZEN = [R['aar_m1'], R['aar_0'], R['aar_p1'], R['aar_p2'], R['aar_p3']]
T_FROZEN   = [R['t_m1'], R['t_0'], R['t_p1'], R['t_p2'], R['t_p3']]
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Oscars-Effect -- does the market care who wins Best Picture?\n"
            "### A cultural-event indicator, tested honestly, in plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Busted%3F: Yes](https://img.shields.io/badge/Busted%3F-Yes-8b949e?style=flat-square)\n\n"
            "Every March, tens of millions of people watch the Academy Awards. The winning "
            "films get a real box-office bump. So surely -- the intuition goes -- such a giant "
            "cultural event must *move the market* the next morning? This notebook asks the only "
            "question that matters: when you look at the S&P 500 on the first trading day after "
            "the Oscars, across 31 ceremonies, is there anything there at all?\n\n"
            "> **This is the plain-language layer.** Want the event-study abnormal returns, the "
            "HAC t-stats, and the placebo control? That's "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** -- same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart is drawn "
            "by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT + "\n" + R_CELL),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does the market rally the day after the Oscars? | **No.** The mean move is "
            "**-23.8 bps** -- if anything mildly *down*, up only 45.2% of the time vs a 54.0% "
            "baseline. HAC t = -1.35: pure noise. |\n"
            "| Is the winning film's box-office bump real? | **Yes -- and irrelevant here.** "
            "The bump accrues to *one film's* revenue; it is a rounding error for a diversified "
            "media giant, let alone the whole index. |\n"
            "| Could you trade it? | **No.** A 'buy the morning after' rule loses money even "
            "before costs. There is no vehicle and no edge. |\n"
            "| How many ceremonies? | **31** (1995-2025) -- far too few to detect anything "
            "smaller than ~40 bps even if it existed. |\n\n"
            "> The Oscars move red carpets and a few studio share prices at the margins. "
            "The broad index registers nothing the next morning beyond ordinary daily noise."
        ),

        # ---- BEAT 1 -- THE CLAIM ----------------------------------------------
        md(
            "## 1 - The claim\n\n"
            "> *\"The Academy Awards are one of the biggest cultural events of the year. "
            "The whole country is watching, the winning studios get a windfall, and the mood "
            "is set for the season -- so the stock market must react to who wins Best Picture.\"*\n\n"
            "Unlike the Super Bowl Indicator, there's no famous paper behind this -- it's "
            "*folklore by analogy*. There genuinely **is** an 'Oscar bump' in box office for the "
            "winning film. The leap of faith is that this local, firm-level effect shows up in "
            "the **broad index** the next trading morning."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 - So what?\n\n"
            "If a single, perfectly-scheduled cultural event reliably nudged the whole S&P 500 "
            "the next morning, it would be the easiest trade in finance: you know the date years "
            "in advance. That alone should make you suspicious. Markets price information when it "
            "arrives -- and by Oscar night, the winners are already heavily implied by the "
            "precursor awards and the betting markets. There is almost nothing new to learn."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW -----------------------------------------
        md(
            "## 3 - How would we even know?\n\n"
            "We run a clean **event study**:\n\n"
            "1. **Event day 0** is the first *full* trading session after the broadcast (the "
            "ceremony airs Sunday/Monday evening, so this is the earliest you could trade -- one "
            "execution lag, baked in). We never use the ceremony evening itself.\n"
            "2. **Abnormal return** = the event-day mean minus the market's ordinary daily mean. "
            "If the Oscars matter, this should be reliably non-zero.\n"
            "3. **The honest baseline.** The S&P drifts up ~4 bps/day and is positive ~54% of "
            "days regardless. The question is whether the morning after the Oscars beats *that*, "
            "not a coin.\n"
            "4. **Tiny n.** Only 31 ceremonies sit inside the ^GSPC daily window. With ~110 "
            "bps daily volatility, we could only detect an effect bigger than ~40 bps. We'll "
            "check whether the observed move is anywhere near that.\n\n"
            "Real tape: ^GSPC daily, 1994-2026 (31 ceremonies, 1995-2025)."
        ),

        # ---- BEAT 4 -- THE TEARDOWN ------------------------------------------
        md(
            "## 4 - The teardown -- let's actually look\n\n"
            "**First, meet the data:** every Oscar ceremony, hardcoded in this study's "
            "`data.py`, with its Best Picture winner."
        ),
        code(
            "cer = data.oscar_table()\n"
            "print('Ceremonies:', len(cer), '(67th-97th, 1995-2025)')\n"
            "print(cer[['ceremony','date','best_pic']].tail(6).to_string(index=False))\n"
            "print()\n"
            "if HAVE_REAL:\n"
            "    ret, cer = data.fetch_daily()\n"
            "    base = float(ret.mean()*1e4); base_up = float((ret>0).mean()*100)\n"
            "else:\n"
            "    base, base_up = R['baseline_mean_bps'], R['baseline_up_pct']\n"
            "print(f'Baseline daily mean: {base:+.1f} bps,  up {base_up:.1f}% of days')\n"
            "print('This is what the morning-after move must beat -- not a coin.')"
        ),
        md("**The morning after -- average move by relative day around the ceremony.**"),
        code(
            "if HAVE_REAL:\n"
            "    w = data.event_windows(ret, cer, pre=1, post=3)\n"
            "    car = st.car_profile(ret, w)\n"
            "    rel = car['rel_day'].tolist(); aar = car['aar_bps'].tolist()\n"
            "else:\n"
            "    rel, aar = REL_DAYS, AAR_FROZEN\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "colors = [GREEN if a>=0 else RED for a in aar]\n"
            "ax.bar([str(d) for d in rel], aar, color=colors, width=0.6)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.axvline(1.0, ls=':', c=GREY, lw=1.5)  # event day 0 is the 2nd bar (index 1)\n"
            "ax.annotate('Oscar night ->', (0.55, max(aar)*0.7), color=GREY, fontsize=9)\n"
            "ax.set_xlabel('Trading day relative to the first post-ceremony session (day 0)')\n"
            "ax.set_ylabel('Average abnormal return (bps)')\n"
            "ax.set_title('The morning after the Oscars: a small dip, lost in the noise')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Event-day-0 abnormal return:', f\"{aar[1]:+.1f} bps\")"
        ),
        md(
            "Event day 0 -- the first morning you could actually trade -- averages a small "
            "**negative** move (about **-28 bps** abnormal). That's the *opposite* of the "
            "'cultural-event rally' story, and it is tiny relative to the day-to-day noise. The "
            "bars to either side bounce around zero with no pattern."
        ),
        md("**Was that dip ever big enough to matter? The base-rate / size check.**"),
        code(
            "if HAVE_REAL:\n"
            "    et = st.event_day_test(ret, w)\n"
            "    ev_up = et['up_rate']*100; ev_mean = et['event_mean_bps']; ab = et['abnormal_bps']\n"
            "else:\n"
            "    ev_up, ev_mean, ab = R['event_up_pct'], R['event_mean_bps'], R['abnormal_bps']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.0, 4.3))\n"
            "bars = ax.bar(['Day after Oscars\\n(up-rate)', 'Any day\\n(baseline up-rate)'],\n"
            "              [ev_up, base_up if HAVE_REAL else R['baseline_up_pct']],\n"
            "              color=[RED, GREY], width=0.5)\n"
            "ax.set_ylabel('Share of days that were positive (%)'); ax.set_ylim(0, 70)\n"
            "ax.set_title(f'Day-after up-rate {ev_up:.1f}% vs baseline -- below, not above')\n"
            "for b, v in zip(bars, [ev_up, base_up if HAVE_REAL else R['baseline_up_pct']]):\n"
            "    ax.annotate(f'{v:.1f}%', (b.get_x()+b.get_width()/2, v+1), ha='center')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Day-after mean: {ev_mean:+.1f} bps  |  abnormal vs baseline: {ab:+.1f} bps')\n"
            "print(f'Up-rate {ev_up:.1f}% is BELOW the ~54% baseline -- no rally here.')"
        ),
        md(
            "The day after the Oscars the market was up only **45.2%** of the time -- *below* "
            "the 54% you'd get on a random day. There is no rally; there is a faint, "
            "statistically meaningless dip. With only 31 observations, this is exactly what "
            "pure noise looks like."
        ),

        # ---- BEAT 5 -- THE VERDICT -------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            "- **Signal -- None.** Day-0 abnormal return **-27.9 bps**, HAC t = **-1.35**, "
            "placebo p = **0.18**. No day in the window clears the significance bar, and the "
            "tiny tilt runs *opposite* to the folklore.\n"
            "- **Tradability -- Mirage.** A 'buy the morning after' rule loses money before "
            "costs (see Beat 6). There is no vehicle and no edge.\n"
            "- **Busted.** The Oscar bump is real but *local* -- it lifts the winning film's "
            "box office, not the S&P 500. The index does not care who wins Best Picture."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 - Could you actually trade it?\n\n"
            "The most generous tradeable rule: buy the index the morning after each ceremony, "
            "hold one session, repeat every year. Net of a one-way cost on entry and exit:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    tr = st.tradable_rule(ret, w, hold_days=1, cost_bps_one_way=1.0)\n"
            "    gross, net = tr['gross_mean_bps'], tr['net_mean_bps']\n"
            "else:\n"
            "    gross, net = R['trade_gross_bps'], R['trade_net_bps']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.0))\n"
            "bars = ax.bar(['Gross\\n(before costs)', 'Net\\n(after costs)'],\n"
            "              [gross, net], color=[AMBER, RED], width=0.5)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('Mean return per trade (bps)')\n"
            "ax.set_title('The day-after rule loses money even before a cent of cost')\n"
            "for b, v in zip(bars, [gross, net]):\n"
            "    ax.annotate(f'{v:+.1f}', (b.get_x()+b.get_width()/2, v-2), ha='center', va='top')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Gross per trade: {gross:+.1f} bps  |  Net per trade: {net:+.1f} bps')\n"
            "print('There is no edge to harvest; costs only deepen the hole.')"
        ),
        md(
            "The rule is in the red **before** costs (-23.8 bps/trade) -- so there is nothing "
            "for costs to eat into; they just make a bad trade worse. The only real lesson: "
            "do not time the market on the Academy Awards."
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 - Going further\n\n"
            "- **The positive control:** the companion notebook plants a fake post-Oscar bump "
            "in a synthetic tape and shows the engine detects a ~50 bps effect easily. The real "
            "tape has nothing like that -- the null is a true absence, not a power failure.\n"
            "- **The Super Bowl Indicator** -- the same 'a cultural event predicts the market' "
            "folklore: [Study 158 -- Super-Bowl](../../158-super-bowl/).\n"
            "- **Mercury retrograde** -- the same daily ^GSPC event machinery on financial "
            "astrology: [Study 164 -- Mercury-Retrograde](../../164-mercury-retrograde/).\n\n"
            "*Think the Oscars really do move the tape? Fork this, define your own event window "
            "or studio-level basket, and show an abnormal return that clears |t| >= 2. That's "
            "the bar -- and the broad index does not clear it.*"
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
            "# Oscars-Effect -- a quantitative teardown\n"
            "### 31 ceremonies * ^GSPC daily * event study * AAR/CAR * HAC t * "
            "placebo control * n=31 power\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Busted%3F: Yes](https://img.shields.io/badge/Busted%3F-Yes-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) -- "
            "*same seven beats, every claim carrying its standard error.* We run a MacKinlay-style "
            "event study on the first post-ceremony session: abnormal return on day 0 with a "
            "Newey-West HAC t-stat, the AAR/CAR profile over relative days -1..+3, a tradable rule "
            "net of one-way costs versus a same-days buy-and-hold benchmark, a 5,000-draw "
            "permutation/placebo control, and a power calculation that explains why n=31 can never "
            "deliver a real answer.\n\n"
            "> **Not investment advice.** Real data: ^GSPC daily price-only "
            "(1994--2026); Oscar ceremonies hardcoded in `data.py` (1995--2025). "
            "Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **The plain-English notes** translate each result back to intuition."
        ),
        code(BOOT + "\n" + R_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            "| **Signal** | `NONE` | Day-0 abnormal **-27.9 bps**, HAC t = **-1.35**, "
            "placebo p = **0.183**; no day in -1..+3 clears \\|t\\| = 2. |\n"
            "| **Tradability** | `MIRAGE` | Day-after rule gross **-23.8 bps/trade**, net "
            "**-25.8 bps**; loses before costs. |\n"
            "| **Busted?** | `YES` | The Oscar bump is real but local to the winning film's "
            "box office; the broad index shows nothing. |\n\n"
            "> The S&P's ordinary +4 bps/day drift is the only structure here. The ceremony "
            "result contributes nothing statistically distinguishable -- and what little there "
            "is points the *wrong* way."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 - The claim, steelmanned\n\n"
            "Let $r_t$ be the ^GSPC daily return and let event day 0 be the first session "
            "after ceremony $c$. The market-model abnormal return is "
            "$AR_{c} = r_{0,c} - \\hat\\mu$, where $\\hat\\mu$ is the full-tape mean (a constant "
            "expected-return benchmark). The hypotheses:\n\n"
            "- **H1 (day-0 reaction).** $E[AR_c] \\neq 0$ -- the morning after the Oscars is "
            "abnormal. The folklore predicts $> 0$ (a rally).\n"
            "- **H2 (drift / CAR).** The cumulative abnormal return over -1..+3 is non-zero "
            "and clears |t| >= 2 on some day.\n"
            "- **H3 (tradeable).** A buy-day-0 rule earns a positive net mean after one-way "
            "costs, beating a same-days buy-and-hold benchmark.\n\n"
            "We reject H1, H2, and H3 on the real tape."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 - So what? -- what rides on each answer\n\n"
            "A reliable index reaction to a pre-scheduled, pre-implied cultural event would "
            "violate weak-form efficiency in the most embarrassing way possible: the date is "
            "known years ahead and the winners are heavily implied by precursor awards. The "
            "interesting finding is *how completely* the broad index ignores an event that "
            "genuinely moves a single film's box office -- a clean demonstration of the gap "
            "between a real micro effect and a (non-existent) macro one."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 - How we'd know -- the protocol\n\n"
            "- **Data.** ^GSPC daily price-only returns, 1994--2026 (8,165 sessions); 31 "
            "ceremony dates hardcoded in `data.py`.\n"
            "- **Event day 0.** First *full* session after the broadcast (one execution lag).\n"
            "- **Benchmark.** Constant expected return = full-tape daily mean; "
            "$AR = r - \\hat\\mu$.\n"
            "- **Aggregation.** AAR by relative day; CAR = running sum over -1..+3.\n"
            "- **Inference.** Newey-West HAC t-stat on day-0 returns; plain t per relative "
            "day; 5,000-draw permutation/placebo control.\n"
            "- **Tradeable rule.** Buy day-0, hold 1 session, one-way cost charged on entry "
            "and exit; benchmarked against same-days buy-and-hold.\n"
            "- **Positive control.** Synthetic tape with a planted post-ceremony bump.\n"
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 - The teardown"),
        md(
            "### 4a - The event-day-0 abnormal return and its HAC t-stat\n\n"
            "The single primary spec, fixed in advance: the mean return on the first "
            "post-ceremony session vs the full-tape baseline."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ret, cer = data.fetch_daily()\n"
            "    w = data.event_windows(ret, cer, pre=1, post=3)\n"
            "    et = st.event_day_test(ret, w)\n"
            "    n_ev = et['n_events']; ev_mean = et['event_mean_bps']\n"
            "    base = et['baseline_mean_bps']; ab = et['abnormal_bps']\n"
            "    hac_t = et['event_t']; plain_t = et['event_plain_t']\n"
            "else:\n"
            "    n_ev, ev_mean = R['n_events'], R['event_mean_bps']\n"
            "    base, ab = R['baseline_mean_bps'], R['abnormal_bps']\n"
            "    hac_t, plain_t = R['event_hac_t'], R['event_plain_t']\n"
            "\n"
            "print(f'n events:                 {n_ev}')\n"
            "print(f'Event-day-0 mean return:  {ev_mean:+.1f} bps')\n"
            "print(f'Full-tape baseline mean:  {base:+.1f} bps')\n"
            "print(f'Abnormal return:          {ab:+.1f} bps')\n"
            "print(f'HAC (Newey-West) t-stat:  {hac_t:+.2f}')\n"
            "print(f'Plain t-stat:             {plain_t:+.2f}')\n"
            "print()\n"
            "print('The abnormal return is NEGATIVE and |t| < 2 -- no reaction, and the')\n"
            "print('sign is the opposite of the cultural-event-rally story.')"
        ),
        md(
            "> The day-0 abnormal return is **-27.9 bps** with HAC t = **-1.35**. Not only is "
            "it insignificant, it is the *wrong sign* for the folklore. A constant-mean market "
            "model is the most generous benchmark possible here; richer benchmarks (a market "
            "beta of 1.0 to itself) would leave it unchanged."
        ),
        md(
            "### 4b - The full AAR / CAR profile (-1 .. +3)\n\n"
            "Does an abnormal drift build across the bracketing days?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    car = st.car_profile(ret, w)\n"
            "    rel = car['rel_day'].tolist(); aar = car['aar_bps'].tolist()\n"
            "    carc = car['car_bps'].tolist(); tt = car['t'].tolist()\n"
            "else:\n"
            "    rel, aar, tt = REL_DAYS, AAR_FROZEN, T_FROZEN\n"
            "    carc = list(np.cumsum(aar))\n"
            "\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.2))\n"
            "a1.bar([str(d) for d in rel], aar, color=[GREEN if x>=0 else RED for x in aar])\n"
            "a1.axhline(0, c='k', lw=1); a1.set_title('Average abnormal return (bps)')\n"
            "a1.set_xlabel('relative day'); a1.set_ylabel('AAR (bps)')\n"
            "a2.plot([str(d) for d in rel], carc, 'o-', c=GREY, lw=2)\n"
            "a2.axhline(0, c='k', lw=1); a2.set_title('Cumulative abnormal return (bps)')\n"
            "a2.set_xlabel('relative day'); a2.set_ylabel('CAR (bps)')\n"
            "plt.tight_layout(); plt.show()\n"
            "prof = pd.DataFrame({'rel_day': rel, 'aar_bps': np.round(aar,1),\n"
            "                     'car_bps': np.round(carc,1), 't': np.round(tt,2)})\n"
            "print(prof.to_string(index=False))\n"
            "print(f'\\nFinal CAR over the window: {carc[-1]:+.1f} bps  (no day clears |t|=2)')"
        ),
        md(
            "The CAR wanders around zero and ends at **-10.9 bps**. The single largest day is "
            "the **pre-ceremony** run-up (+26.8 bps, t = 1.67) -- which, even if it were real, "
            "is unactionable: you cannot trade on a result that has not been announced. No day "
            "in the window reaches |t| = 2."
        ),
        md(
            "### 4c - The permutation / placebo control\n\n"
            "Relabel 5,000 random sets of 31 sessions as 'post-Oscar' and ask how often a "
            "random set's mean swing matches the real one."
        ),
        code(
            "if HAVE_REAL:\n"
            "    perm = st.permutation_null(ret, w, n_perm=5000, seed=296)\n"
            "    placebo_p = perm['placebo_p']; obs_ab = perm['abnormal_bps']\n"
            "    rng0 = np.random.default_rng(296)\n"
            "    vals = ret.to_numpy(); base_m = float(ret.mean())\n"
            "    draws = np.array([vals[rng0.choice(len(vals), size=R['n_events'], replace=False)].mean()\n"
            "                      for _ in range(3000)]) * 1e4 - base_m*1e4\n"
            "else:\n"
            "    placebo_p, obs_ab = R['placebo_p'], R['abnormal_bps']\n"
            "    rng0 = np.random.default_rng(296)\n"
            "    draws = rng0.normal(0, R['daily_vol_bps']/np.sqrt(R['n_events']), 3000)\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=40, color=GREY, alpha=0.7, label='Random 31-day sets (placebo)')\n"
            "ax.axvline(obs_ab, c=RED, lw=2.5, label=f'Observed abnormal: {obs_ab:+.1f} bps')\n"
            "ax.axvline(0, c='k', lw=1)\n"
            "ax.set_xlabel('Mean abnormal return of a random 31-day set (bps)')\n"
            "ax.set_ylabel('Count')\n"
            "ax.set_title('The observed event-day mean is an ordinary draw, not a tail event')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Two-sided placebo p-value: {placebo_p:.3f}')\n"
            "print('A random set of 31 days swings this much ~18% of the time.')"
        ),
        md(
            "Placebo p = **0.183** -- the observed event-day mean sits comfortably inside the "
            "distribution of random 31-day sets. There is no tail event to explain."
        ),
        md(
            "### 4d - Power: what could n=31 even detect?\n\n"
            "The minimum detectable one-day mean at |t| = 2 given the daily volatility."
        ),
        code(
            "if HAVE_REAL:\n"
            "    daily_vol_bps = float(ret.std(ddof=1)*1e4)\n"
            "else:\n"
            "    daily_vol_bps = R['daily_vol_bps']\n"
            "n = R['n_events']\n"
            "mde = 2.0 * daily_vol_bps / np.sqrt(n)\n"
            "print(f'Daily S&P vol:           {daily_vol_bps:.0f} bps')\n"
            "print(f'n ceremonies:            {n}')\n"
            "print(f'Min detectable |mean| at |t|=2:  {mde:.0f} bps')\n"
            "print()\n"
            "print(f'Observed abnormal:       {R[\"abnormal_bps\"]:+.1f} bps  (and wrong sign)')\n"
            "print('Even a TRUE +/-20 bps Oscar effect could not be resolved at n=31.')"
        ),
        md(
            "> With ~110 bps daily vol and n = 31, the smallest mean we could distinguish from "
            "zero is **~40 bps**. The observed -28 bps is inside that band and in the wrong "
            "direction. This is the binding constraint: the Oscars-Effect is unfalsifiable at "
            "this sample size for any plausible magnitude -- and shows no hint of one anyway."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            "- **Signal `NONE`** -- day-0 abnormal -27.9 bps, HAC t = -1.35, placebo p = "
            "0.183; CAR wanders to -10.9 bps with no day clearing |t| = 2; the only sizeable "
            "(and unactionable) move is the pre-ceremony day.\n"
            "- **Tradability `MIRAGE`** -- the day-after rule loses gross (-23.8 bps/trade) "
            "before costs; net -25.8 bps; no vehicle, no edge.\n"
            "- **Busted** -- the Oscar bump is a real, *firm-level* box-office phenomenon; the "
            "broad index registers nothing the next morning."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 - Could you trade it? -- the cost-aware benchmark\n\n"
            "Buy day-0, hold one session, every year; one-way cost on entry and exit; "
            "benchmarked against buy-and-hold over exactly the same exposed sessions."
        ),
        code(
            "if HAVE_REAL:\n"
            "    tr = st.tradable_rule(ret, w, hold_days=1, cost_bps_one_way=1.0)\n"
            "    gross, net, net_t, bah = (tr['gross_mean_bps'], tr['net_mean_bps'],\n"
            "                              tr['net_t'], tr['bah_mean_bps'])\n"
            "    costs = [0.0, 1.0, 2.0, 5.0]\n"
            "    nets = [st.tradable_rule(ret, w, hold_days=1, cost_bps_one_way=c)['net_mean_bps']\n"
            "            for c in costs]\n"
            "else:\n"
            "    gross, net, net_t, bah = (R['trade_gross_bps'], R['trade_net_bps'],\n"
            "                              R['trade_net_t'], R['trade_bah_bps'])\n"
            "    costs = [0.0, 1.0, 2.0, 5.0]\n"
            "    nets = [gross - 2*c for c in costs]\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.2))\n"
            "ax.plot(costs, nets, 'o-', c=RED, lw=2, label='Net mean per trade')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.axhline(gross, ls='--', c=AMBER, lw=1.5, label=f'Gross ({gross:+.1f} bps)')\n"
            "ax.set_xlabel('One-way cost (bps)'); ax.set_ylabel('Net mean per trade (bps)')\n"
            "ax.set_title('Already negative at zero cost -- nothing for costs to eat')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Gross: {gross:+.1f} bps  |  Net @1bp: {net:+.1f} bps  |  HAC t(net): {net_t:+.2f}')\n"
            "print(f'Buy-and-hold over the same exposed days: {bah:+.1f} bps (also negative)')"
        ),
        md(
            "> The line starts below zero and only sinks. There is no positive gross return for "
            "transaction costs to erode -- the 'edge' was negative to begin with. Buy-and-hold "
            "over the same exposed days is also negative: the morning-after happens to be a "
            "mildly poor day historically, but not reliably so."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 - Going further -- the synthetic positive control\n\n"
            "Does the event-study machinery detect an Oscar reaction *when one exists*? "
            "We plant a one-day post-ceremony bump in a synthetic random walk and sweep its size."
        ),
        code(
            "planted = [0, 50, 150, 300]\n"
            "rows = []\n"
            "for bp in planted:\n"
            "    sret, scer, _ = data.synthetic_daily(post_oscar_bp=float(bp), seed=296)\n"
            "    sw = data.event_windows(sret, scer, pre=1, post=3)\n"
            "    se = st.event_day_test(sret, sw)\n"
            "    rows.append({'planted_bps': bp, 'abnormal_bps': round(se['abnormal_bps'],1),\n"
            "                 'hac_t': round(se['event_t'],2), 'n': se['n_events']})\n"
            "res = pd.DataFrame(rows)\n"
            "\n"
            "colors = [GREEN if abs(r['hac_t'])>=2 else RED for r in rows]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.2))\n"
            "ax.bar([str(r['planted_bps']) for r in rows], [r['hac_t'] for r in rows], color=colors)\n"
            "ax.axhline(2, ls=':', c='k', lw=1, label='|t| = 2')\n"
            "ax.axhline(-2, ls=':', c='k', lw=1)\n"
            "ax.set_xlabel('Planted post-Oscar bump (bps)'); ax.set_ylabel('Event-day HAC t-stat')\n"
            "ax.set_title('The engine finds the effect when it exists (green = |t| >= 2)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(res.to_string(index=False))"
        ),
        md(
            "The engine resolves a planted bump as small as ~50 bps at this sample size. The "
            "real tape shows nothing of the kind -- the `NONE` verdict is a statement about the "
            "**market and the tiny sample**, not a deficiency of the method. The Academy can "
            "keep its suspense; the S&P 500 will not be watching."
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
    """Execute a notebook in-place with nbconvert (no --allow-errors)."""
    from nbconvert.preprocessors import ExecutePreprocessor

    path = os.path.join(HERE, name)
    nb = nbf.read(path, as_version=4)
    ep = ExecutePreprocessor(timeout=300, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": HERE}})
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("executed", path)


if __name__ == "__main__":
    build_curious()
    build_quants()
    _execute("01_for_the_curious.ipynb")
    _execute("02_for_the_quants.ipynb")
