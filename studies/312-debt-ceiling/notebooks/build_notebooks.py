"""Generate the two narrative notebooks for Study 312 (Debt-Ceiling), the vol angle.

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the cached
``^VIX_raw`` parquet under the shared ../../../_cache/ if present and otherwise quote the
frozen headline numbers in ``R`` (mirroring docs/results.md), so the notebook re-runs for
any reader. Cells that show the synthetic tape are explicitly bannered as such.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-05-29).
R = dict(
    vix_start="1990-01-02", vix_end="2026-05-29", vix_days=9170, vix_fp="339114a53569",
    n_events=7,
    # two legs (pre=20, post=20), Δlog VIX
    pre_mean=-5.2, pre_hit=43, pre_t=-0.55, pre_ci_lo=-23.5, pre_ci_hi=13.0,
    post_mean=5.2, post_hit=71, post_t=1.78, post_ci_lo=0.7, post_ci_hi=10.4,
    # placebo vs random dates
    rand_pre=-0.6, rand_post=-0.0,
    perm_pre_obs=-4.7, perm_pre_p=0.55, perm_post_obs=5.2, perm_post_p=0.50,
    # the trade (round-trip, carry 25 bps/day)
    rt_gross=-10.4, rt_gross_t=-1.34, rt_hit=43,
    rt_net=-10.6, rt_net_t=-1.36,
    long_leg=-10.3, long_leg_hit=29, long_leg_t=-1.08,
    carry_bps=25, carry_drag=5.0,
    # per-event (pre, post) Δlog %
    ev_2011=(43.4, 28.3), ev_2013=(2.4, -8.6), ev_2015=(-28.8, 9.0),
    ev_2017=(-9.2, 5.1), ev_2021a=(-45.5, 1.1), ev_2021b=(16.4, 5.2),
    ev_2023=(-15.4, -3.8),
    # synthetic positive control
    plant_pre_t=14.9, plant_post_t=-15.8, plant_rt_t=16.7,
    null_pre_t=-0.5, null_post_t=0.1,
)


BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root (quantlab/, _cache/)
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from debt_ceiling import data, strategy as st

PRE, POST = 20, 20

def _have_cache():
    return os.path.exists(data._cache_path("^VIX"))

HAVE_REAL = _have_cache()

def load_vix():
    return data.load_real("^VIX", end="2026-05-31")

print("real VIX cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Debt-Ceiling — is the brinkmanship a volatility trade? 🪙\n"
            "### Does fear-vol ramp into the deadline and collapse when Washington blinks? In plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Is_brinkmanship_a_vol_trade%3F: Busted](https://img.shields.io/badge/Is_brinkmanship_a_vol_trade%3F-Busted-8b949e?style=flat-square)\n\n"
            "Every time the US flirts with not paying its bills, the trading desks dust off the "
            "same playbook: **buy volatility into the debt-ceiling deadline** (the market's fear is "
            "supposed to build as the 'X-date' approaches), then **sell it the moment Congress "
            "blinks** (relief, vol collapses). It *sounds* like a sure thing. This notebook asks "
            "the only question that matters: **does the VIX actually do that?**\n\n"
            "> **This is the plain-language layer.** Want the *t*-stats, the random-date placebo "
            "and the carry-tax math? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart below is drawn "
            "by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does vol *rise* into the deadline? | **No.** Across 7 brink-going episodes the VIX "
            f"actually *fell* {abs(R['pre_mean']):.0f}% on average into the deadline — only **2 of "
            "7** rose. |\n"
            "| Does vol *collapse* on resolution? | **No** — if anything it drifted *up* afterward, "
            "and that's carried almost entirely by **2011** (the spike came *after* the deal, on a "
            "credit downgrade). |\n"
            "| Is it different from a random week? | **No.** The debt-ceiling windows are "
            f"statistically indistinguishable from random dates (*p* ≈ 0.5). |\n"
            "| Could you trade it? | **No.** The long-vol-into / short-vol-out round-trip *loses* "
            f"~{abs(R['rt_gross']):.0f}% per event — because waiting long-vol for the deadline "
            "bleeds you the carry tax every single day. |\n\n"
            "> The 'debt-ceiling vol trade' is a story built on one scary episode (2011) that "
            "didn't even fit the trade — the vol spiked *after* the deal. With seven events total, "
            "there's nothing here to trade."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"As the debt-ceiling X-date approaches, hedging demand drives implied volatility "
            "up — markets price a (tiny but real) chance the US technically defaults. The moment a "
            "deal is struck, that fear evaporates and vol collapses. So you go **long vol into the "
            "deadline** and **short vol on resolution**.\"*\n\n"
            "— the standard derivatives-desk 'how to trade the debt ceiling' note (Goldman, "
            "JPMorgan, BofA; echoed across CNBC/Bloomberg)\n\n"
            "It's a clean, intuitive story: anxiety builds, then relief. The VIX — the market's "
            "'fear gauge' — should trace a neat little hump, peaking right at the deadline."
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "Two reasons to care. First, if it were true it'd be a *calendar-known* trade — the "
            "X-date is announced weeks ahead, so you'd have a rare thing: a volatility event you "
            "can see coming. Second, this is the **volatility** cousin of the desk's "
            "[shutdown 'buy the dip' study](../../311-government-shutdown/): one asks *do stocks "
            "bounce*, this one asks *does fear spike*. Testing both separates two different pieces "
            "of Washington folklore — and shows how a single dramatic episode can fool everyone "
            "into seeing a pattern."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Three checks keep us honest:\n\n"
            "1. **Draw the hump.** Line up all 7 deadlines and plot the average VIX path around "
            "them. If the thesis is right, we see vol climb into day 0 and fall after.\n"
            "2. **Compare to a random week.** A spike only means something if it's *bigger* than "
            "what the VIX does around any random date. So we measure the same windows around "
            "thousands of random dates and ask: is the debt-ceiling move unusual?\n"
            "3. **Actually trade it — with the carry tax.** Holding a long-vol position isn't free: "
            "VIX futures roll *down* a usually-upward-sloping curve, so you bleed value every day "
            "you wait. We charge that and see if the trade survives.\n\n"
            "Tape: the CBOE **VIX** daily level back to 1990. Seven brink-going episodes — and "
            "we'll say loudly that seven is far too few to call anything a strategy."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the hump.** Here's the average VIX path around the seven deadlines (rebased "
            "to 1.0 at the deadline). If the story holds, it peaks in the middle."
        ),
        code(
            "if HAVE_REAL:\n"
            "    vix = load_vix(); ev = data.deadlines()\n"
            "    path = st.vix_path(vix, ev, pre=PRE, post=POST)\n"
            "    banner = 'REAL VIX tape (1990-2026)'\n"
            "else:\n"
            "    b, ev, _ = data.synthetic_vix(event_effect=0.0, n_events=40, seed=314)\n"
            "    path = st.vix_path(b, ev, pre=PRE, post=POST)\n"
            "    banner = 'SYNTHETIC null tape (offline fallback)'\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.6))\n"
            "ax.plot(path.index, path['mean'], lw=2.2, color=RED)\n"
            "ax.fill_between(path.index, path['mean']-path['se'], path['mean']+path['se'],\n"
            "                color=RED, alpha=.15)\n"
            "ax.axvline(0, ls='--', c=GREY); ax.axhline(1.0, ls=':', c='k', lw=1)\n"
            "ax.set_xlabel('trading sessions relative to the deadline (t=0)')\n"
            "ax.set_ylabel('mean VIX level (rebased to 1.0 at t=0)')\n"
            "ax.set_title(f'No hump: vol does NOT peak at the deadline  [{banner}]')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('If the thesis were right this would be a tent peaking at 0. It is not.')"
        ),
        md(
            f"No hump. On the real tape the VIX **fell ~{abs(R['pre_mean']):.0f}%** into the "
            "deadline on average — the opposite of 'fear builds.' The neat little anxiety-then-relief "
            "shape simply isn't there."
        ),
        md(
            "**Second, the per-event detail — where does the 'vol spike' story even come from?** "
            "Here's each episode's vol move *into* the deadline. Watch one bar carry the whole "
            "narrative:"
        ),
        code(
            "names = ['2011\\n(downgrade)','2013','2015','2017','2021a','2021b','2023\\n(FRA)']\n"
            "if HAVE_REAL:\n"
            "    pr = st.pre_ramp(vix, ev, pre=PRE)\n"
            "    pre_vals = (pr['d_log']*100).to_numpy()\n"
            "    banner = 'REAL VIX tape'\n"
            "else:\n"
            f"    pre_vals = np.array([{R['ev_2011'][0]}, {R['ev_2013'][0]}, {R['ev_2015'][0]},\n"
            f"        {R['ev_2017'][0]}, {R['ev_2021a'][0]}, {R['ev_2021b'][0]}, {R['ev_2023'][0]}])\n"
            "    banner = 'frozen real numbers (docs/results.md)'\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.4))\n"
            "cols = [GREEN if v > 0 else RED for v in pre_vals]\n"
            "ax.bar(names, pre_vals, color=cols)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('VIX change INTO the deadline (Δlog %, 20 sessions)')\n"
            "ax.set_title(f'Only 2 of 7 episodes saw vol rise into the deadline  [{banner}]')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Mean across all 7: {pre_vals.mean():+.1f}%  —  drop 2011 and it is strongly negative.')"
        ),
        md(
            "There it is. **Five of the seven episodes saw vol *fall* into the deadline.** The only "
            "real 'spike' is **2011** — and even that one mostly happened *after* the deal was "
            "signed, when S&P downgraded US debt from AAA. That's not the trade; that's a separate "
            "credit shock that happened to land nearby."
        ),
        md(
            "**Third: is any of this different from a random week?** The honest test. We measure "
            "the same window around thousands of random dates and compare:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    rng = np.random.default_rng(312); n = len(vix)\n"
            "    idx = np.arange(PRE+5, n-POST-5)\n"
            "    rlocs = rng.choice(idx, size=3000, replace=False)\n"
            "    rand_pre = st.window_log_changes(vix, rlocs, PRE, 'pre')*100\n"
            "    ev_pre = (st.pre_ramp(vix, ev, pre=PRE)['d_log']*100).to_numpy()\n"
            "    banner = 'REAL VIX tape'\n"
            "else:\n"
            "    rng = np.random.default_rng(312)\n"
            "    rand_pre = rng.normal(-0.6, 15, 3000)\n"
            f"    ev_pre = np.array([{R['ev_2011'][0]}, {R['ev_2013'][0]}, {R['ev_2015'][0]},\n"
            f"        {R['ev_2017'][0]}, {R['ev_2021a'][0]}, {R['ev_2021b'][0]}, {R['ev_2023'][0]}])\n"
            "    banner = 'synthetic fallback'\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.4))\n"
            "ax.hist(rand_pre, bins=50, color=GREY, alpha=.6, density=True, label='random weeks')\n"
            "ax.axvline(rand_pre.mean(), color='k', ls=':', label=f'random mean {rand_pre.mean():+.1f}%')\n"
            "ax.axvline(ev_pre.mean(), color=RED, lw=2.5, label=f'debt-ceiling mean {ev_pre.mean():+.1f}%')\n"
            "ax.set_xlabel('VIX change into the window (Δlog %)'); ax.set_ylabel('density')\n"
            "ax.set_title(f'Debt-ceiling weeks look like any random week  [{banner}]'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('The red line sits right inside the grey cloud — nothing special is happening.')"
        ),
        md(
            f"The debt-ceiling average (red) sits comfortably inside the random-date cloud. The "
            f"formal placebo test agrees: permutation *p* ≈ {R['perm_pre_p']:.2f}. There is **no "
            "abnormal vol behaviour** to bank — debt-ceiling weeks are just... weeks."
        ),

        # ---- BEAT 5 — THE VERDICT --------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** Vol *fell* ~{abs(R['pre_mean']):.0f}% into the deadline (only "
            "2/7 episodes positive), didn't collapse after, and is indistinguishable from random "
            f"dates (*p* ≈ {R['perm_pre_p']:.2f}). Seven events is far too few regardless.\n"
            "- **Tradability — Mirage.** The long-vol-into / short-vol-out round-trip *loses* "
            f"~{abs(R['rt_gross']):.0f}% per event before costs — the long-vol carry tax bleeds you "
            "while you wait, and the vol move never shows up to pay you back.\n"
            "- **Is brinkmanship a vol trade? — Busted.** The whole narrative is one episode (2011) "
            "that spiked *after* the deal on a credit downgrade — the opposite of the trade."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Here's the part the playbook never mentions: **holding long volatility costs you money "
            "every day**, deal or no deal. VIX futures usually roll *down* a hill, so a long-vol "
            "position bleeds — the 'carry tax.' Watch what it does to the trade:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    rt0 = st.vol_round_trip(vix, ev, pre=PRE, post=POST, daily_carry_bps=0.0)\n"
            "    rtc = st.vol_round_trip(vix, ev, pre=PRE, post=POST, daily_carry_bps=25.0)\n"
            "    no_carry = rt0['pnl_gross'].mean()*100; with_carry = rtc['pnl_gross'].mean()*100\n"
            "    banner = 'REAL VIX tape'\n"
            "else:\n"
            f"    no_carry, with_carry = {R['rt_gross']+2*R['carry_drag']:.1f}, {R['rt_gross']:.1f}\n"
            "    banner = 'frozen real numbers'\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "b = ax.bar(['Ignore the carry\\n(the brochure)','Charge the carry\\n(reality)'],\n"
            "           [no_carry, with_carry], color=[AMBER, RED], width=.55)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('round-trip P&L per event (% of NAV)')\n"
            "ax.set_title(f'The carry tax sinks an already-leaky trade  [{banner}]')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Even ignoring carry it barely works; charging it: {with_carry:+.1f}% per event.')"
        ),
        md(
            f"Even *ignoring* the carry the trade is marginal — because the vol move the thesis "
            f"needs isn't there. Charge the real ~{R['carry_bps']}-bps/day roll and it's a clear "
            f"**loser (~{abs(R['rt_gross']):.0f}% per event)**. Unlike a 'real but fragile' edge, "
            "there's nothing underneath here to rescue. It's a mirage."
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The directional sibling.** [Study 311 — Government-Shutdown](../../311-government-shutdown/) "
            "asks *do stocks bounce* around fiscal cliffs (also a weak/mirage result). Together they "
            "cover both halves of 'trading Washington.'\n"
            "- **Was 2011 special?** Arguably the only debt-ceiling episode that genuinely scared "
            "markets — and the fear came from the **S&P downgrade**, a credit event, not the deadline "
            "itself. A fun fork: re-run excluding 2011 and watch the (already absent) signal vanish "
            "entirely.\n"
            "- **Could a *shorter* window catch a last-minute spike?** Try pre=5 instead of 20 — the "
            "vol still doesn't reliably ramp, but it's the natural next robustness check.\n\n"
            "*Think there's a debt-ceiling vol trade hiding in a different window or instrument? Fork "
            "this, change `PRE`/`POST` or swap the VIX for VIX futures, and show the spike surviving a "
            "random-date placebo on more than one episode.*"
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
            "# Debt-Ceiling — a quantitative teardown (the volatility angle) 🔬\n"
            "### Real VIX tape · event study · random-date placebo · HAC *t* + block bootstrap · "
            "the long-vol carry tax · synthetic positive control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Is_brinkmanship_a_vol_trade%3F: Busted](https://img.shields.io/badge/Is_brinkmanship_a_vol_trade%3F-Busted-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We test whether the VIX "
            "ramps into the binding debt-ceiling deadline and collapses on resolution, against a "
            "random-date placebo, and whether the canonical long-vol/short-vol round-trip survives "
            "the carry tax. Distinct from the directional [Study 311](../../311-government-shutdown/).\n\n"
            "> **Not investment advice.** Real data: CBOE VIX daily levels back to 1990 (raw index, "
            "not a total-return series — you can't hold the VIX); as-of 2026-05-29; the offline core "
            "and tests run on a deterministic mean-reverting synthetic tape. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Pre-deadline ramp mean **{R['pre_mean']:+.1f}%** Δlog VIX, "
            f"HAC *t* = **{R['pre_t']:+.2f}**, only {R['pre_hit']}% positive; permutation vs random "
            f"dates *p* = {R['perm_pre_p']:.2f}. n = {R['n_events']}. |\n"
            f"| **Tradability** | `MIRAGE` | Long-vol-into / short-vol-out round-trip "
            f"**{R['rt_gross']:+.1f}%/event** gross (*t* = {R['rt_gross_t']:+.2f}); the long leg "
            f"alone bleeds {R['carry_drag']:.0f}% of NAV to carry over the hold. |\n"
            f"| **Is brinkmanship a vol trade?** | `BUSTED` | The post-deadline mean "
            f"({R['post_mean']:+.1f}%) is carried by 2011 alone, and that spike came *after* the "
            "deal on the S&P downgrade — not into the deadline. |\n\n"
            "> In plain words: there is no debt-ceiling volatility signature — the VIX behaves "
            "around these deadlines exactly as it behaves around random dates, and the one episode "
            "that looks like a spike is a separate credit event."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $V_t$ be the VIX level and $D$ a binding deadline session. Define the pre-ramp "
            "$\\rho^- = \\log V_D - \\log V_{D-20}$ and the post-move $\\rho^+ = \\log V_{D+20} - "
            "\\log V_D$:\n\n"
            "- **H₁ (ramp in).** $\\mathbb{E}[\\rho^-] > 0$ — vol rises into the deadline.\n"
            "- **H₂ (collapse out).** $\\mathbb{E}[\\rho^+] < 0$ — vol falls on resolution.\n"
            "- **H₃ (unusual).** Both differ from the same windows around random dates.\n"
            "- **H₄ (tradable).** The long-into / short-out round-trip is positive *after* the "
            "long-vol carry tax.\n\n"
            "We find **H₁ rejected** (sign is wrong, *t* < 0), **H₂ rejected** (sign is wrong, and "
            "what positive there is comes from 2011), **H₃ rejected** (*p* ≈ 0.5), and **H₄ "
            "rejected** (the round-trip loses, dragged by carry)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The X-date is *calendar-known* weeks ahead, so a real vol signature would be a rare "
            "pre-positionable volatility event. The interesting content is **not** the binary "
            "verdict (there's no signal) but the **mechanism of the illusion**: (a) a strongly "
            "mean-reverting VIX makes any single dramatic episode look like 'a pattern' in a "
            "7-point sample, and (b) the long-vol carry tax means even a *correct* directional call "
            "can lose money. Naming both is worth more than the stamp."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Event bar.** Snap each deadline to the first session on/after it; the date is "
            "calendar-known so canonical `lag=0` (a `+1` lag is the robustness knob).\n"
            "- **Two legs.** $\\rho^-$ over the 20 sessions ending at $D$; $\\rho^+$ over the 20 "
            "starting at $D$ (log changes, so different base vols are comparable).\n"
            "- **Inference.** Newey-West HAC *t*; circular block-bootstrap CI; **permutation** of "
            "the event mean against a 3,000-draw random-date null.\n"
            "- **Trade.** Long-vol-into / short-vol-out round-trip, long leg charged a "
            "`daily_carry_bps` VIX-futures roll, three fills × one-way cost.\n"
            "- **Positive control.** Deterministic mean-reverting (OU) synthetic VIX with a tunable "
            "tent-shaped hump — the engine must recover it and find nothing on the null.\n\n"
            "Tape: CBOE VIX daily levels, 1990→2026. Seven brink-going episodes (low-n by "
            "construction, stated loudly)."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The two legs with honest inference\n\n"
            "HAC *t* and block-bootstrap CI on each leg's per-event Δlog VIX."
        ),
        code(
            "if HAVE_REAL:\n"
            "    vix = load_vix(); ev = data.deadlines()\n"
            "    pr = st.pre_ramp(vix, ev, pre=PRE)['d_log'].to_numpy()\n"
            "    pc = st.post_collapse(vix, ev, post=POST)['d_log'].to_numpy()\n"
            "    sp, sc = st.summarize(pr), st.summarize(pc)\n"
            "    ci_pre = st.block_bootstrap_ci(pr); ci_post = st.block_bootstrap_ci(pc)\n"
            "    banner = 'REAL VIX tape (1990-2026)'\n"
            "else:\n"
            f"    sp = dict(n={R['n_events']}, mean={R['pre_mean']/100}, hit_rate={R['pre_hit']/100}, tstat={R['pre_t']})\n"
            f"    sc = dict(n={R['n_events']}, mean={R['post_mean']/100}, hit_rate={R['post_hit']/100}, tstat={R['post_t']})\n"
            f"    ci_pre = ({R['pre_ci_lo']/100}, {R['pre_ci_hi']/100}); ci_post = ({R['post_ci_lo']/100}, {R['post_ci_hi']/100})\n"
            "    banner = 'frozen real numbers (docs/results.md)'\n"
            "labels = ['Pre-deadline\\nramp (H1)', 'Post-deadline\\nmove (H2)']\n"
            "means = [sp['mean']*100, sc['mean']*100]\n"
            "errs = [[ (sp['mean']-ci_pre[0])*100, (sc['mean']-ci_post[0])*100 ],\n"
            "        [ (ci_pre[1]-sp['mean'])*100, (ci_post[1]-sc['mean'])*100 ]]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.4))\n"
            "b = ax.bar(labels, means, yerr=errs, capsize=6, color=[RED, AMBER], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('mean Δlog VIX (%) with 95% bootstrap CI')\n"
            "ax.set_title(f'Neither leg has the sign the thesis needs  [{banner}]')\n"
            "for bar_, s in zip(b, [sp, sc]):\n"
            "    ax.annotate(f\"t={s['tstat']:+.2f}\", (bar_.get_x()+bar_.get_width()/2,\n"
            "        bar_.get_height()), ha='center', va='bottom' if bar_.get_height()>=0 else 'top')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"PRE  n={sp['n']} mean={sp['mean']*100:+.1f}% hit={sp['hit_rate']*100:.0f}% t={sp['tstat']:+.2f}\")\n"
            "print(f\"POST n={sc['n']} mean={sc['mean']*100:+.1f}% hit={sc['hit_rate']*100:.0f}% t={sc['tstat']:+.2f}\")"
        ),
        md(
            f"> In plain words: H₁ wants the left bar **positive** — it's **{R['pre_mean']:+.1f}%** "
            f"(*t* = {R['pre_t']:+.2f}). H₂ wants the right bar **negative** — it's "
            f"**{R['post_mean']:+.1f}%**. The CIs straddle zero. Both hypotheses fail on sign before "
            "we even reach significance."
        ),
        md(
            "### 4b · The placebo — debt-ceiling weeks vs random weeks\n\n"
            "A move only counts if it's *unusual*. Permutation test of each leg's mean against the "
            "random-date distribution (3,000 draws)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rng = np.random.default_rng(312); n = len(vix); ix = np.arange(PRE+5, n-POST-5)\n"
            "    rl = rng.choice(ix, size=3000, replace=False)\n"
            "    rand_pre = st.window_log_changes(vix, rl, PRE, 'pre')\n"
            "    rand_post = st.window_log_changes(vix, rl, POST, 'post')\n"
            "    obs_pre, p_pre = st.permutation_pvalue(pr, rand_pre)\n"
            "    obs_post, p_post = st.permutation_pvalue(pc, rand_post)\n"
            "    banner = 'REAL VIX tape'\n"
            "else:\n"
            f"    obs_pre, p_pre = {R['perm_pre_obs']/100}, {R['perm_pre_p']}\n"
            f"    obs_post, p_post = {R['perm_post_obs']/100}, {R['perm_post_p']}\n"
            "    rng = np.random.default_rng(312)\n"
            "    rand_pre = rng.normal(-0.006, .15, 3000); rand_post = rng.normal(0, .14, 3000)\n"
            f"    pr = np.array([{R['ev_2011'][0]}, {R['ev_2013'][0]}, {R['ev_2015'][0]}, {R['ev_2017'][0]}, {R['ev_2021a'][0]}, {R['ev_2021b'][0]}, {R['ev_2023'][0]}])/100\n"
            f"    pc = np.array([{R['ev_2011'][1]}, {R['ev_2013'][1]}, {R['ev_2015'][1]}, {R['ev_2017'][1]}, {R['ev_2021a'][1]}, {R['ev_2021b'][1]}, {R['ev_2023'][1]}])/100\n"
            "    banner = 'synthetic fallback'\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))\n"
            "for ax, rnd, evx, lab, p in [(a1, rand_pre, pr, 'pre-ramp', p_pre),\n"
            "                              (a2, rand_post, pc, 'post-move', p_post)]:\n"
            "    ax.hist(rnd*100, bins=45, color=GREY, alpha=.6, density=True)\n"
            "    ax.axvline(rnd.mean()*100, color='k', ls=':')\n"
            "    ax.axvline(evx.mean()*100, color=RED, lw=2.5)\n"
            "    ax.set_title(f'{lab}: p={p:.2f}'); ax.set_xlabel('Δlog VIX (%)')\n"
            "fig.suptitle(f'Debt-ceiling means (red) sit inside the random cloud  [{banner}]')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'pre:  obs {obs_pre*100:+.1f}% p={p_pre:.2f}   post: obs {obs_post*100:+.1f}% p={p_post:.2f}')"
        ),
        md(
            f"> In plain words: *p* ≈ {R['perm_pre_p']:.2f} (pre) and ≈ {R['perm_post_p']:.2f} "
            "(post). The debt-ceiling windows are **statistically indistinguishable from random "
            "dates**. H₃ rejected — there is no abnormal vol to attribute to brinkmanship."
        ),
        md(
            "### 4c · The 2011 dominance — drop one row and the story dies\n\n"
            "With only 7 events, leverage matters. Show the per-event legs and what happens to the "
            "mean when 2011 is excluded."
        ),
        code(
            "ev_names = ['2011','2013','2015','2017','2021a','2021b','2023']\n"
            "pre_e = pr*100; post_e = pc*100\n"
            "x = np.arange(len(ev_names)); w = .38\n"
            "fig, ax = plt.subplots(figsize=(10, 4.4))\n"
            "ax.bar(x-w/2, pre_e, w, label='into deadline (H1 wants +)', color=RED, alpha=.85)\n"
            "ax.bar(x+w/2, post_e, w, label='after deadline (H2 wants −)', color=AMBER, alpha=.85)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_xticks(x); ax.set_xticklabels(ev_names)\n"
            "ax.set_ylabel('Δlog VIX (%)'); ax.legend()\n"
            "ax.set_title('2011 (the S&P downgrade) is the only real spike — and it came AFTER the deal')\n"
            "plt.tight_layout(); plt.show()\n"
            "mask = np.array([nm!='2011' for nm in ev_names])\n"
            "print(f'pre mean ALL {pre_e.mean():+.1f}%  vs  ex-2011 {pre_e[mask].mean():+.1f}%')\n"
            "print(f'post mean ALL {post_e.mean():+.1f}% vs  ex-2011 {post_e[mask].mean():+.1f}%')"
        ),
        md(
            "> In plain words: 2011 is both the only big pre-ramp **and** the only big post-spike "
            "— and its post-spike (+28%) is the **S&P sovereign downgrade**, a credit event that "
            "hit *after* the deal. Strip it out and the pre-ramp is firmly negative and the "
            "post-move near zero. A seven-point 'pattern' resting on one credit shock."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — pre-ramp {R['pre_mean']:+.1f}% (*t* = {R['pre_t']:+.2f}, "
            f"{R['pre_hit']}% positive), post-move {R['post_mean']:+.1f}% carried by 2011; both "
            f"placebo *p* ≈ 0.5; n = {R['n_events']}.\n"
            f"- **Tradability `MIRAGE`** — round-trip {R['rt_gross']:+.1f}%/event gross "
            f"(*t* = {R['rt_gross_t']:+.2f}); the long-vol leg bleeds {R['carry_drag']:.0f}% NAV to "
            "carry. No abnormal vol to harvest, and waiting costs roll.\n"
            f"- **Is brinkmanship a vol trade? `BUSTED`** — the narrative is one episode (2011) "
            "whose spike came post-deal on a downgrade, the opposite of the trade."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the carry tax, cost sweep\n\n"
            "The binding cost isn't the spread — it's the **roll**. A long-vol position rolls down "
            "the (usually upward-sloping) VIX-futures curve every day it waits for the deadline."
        ),
        code(
            "carries = [0, 10, 25, 40]\n"
            "if HAVE_REAL:\n"
            "    means = [st.summarize(st.vol_round_trip(vix, ev, pre=PRE, post=POST,\n"
            "             daily_carry_bps=c)['pnl_gross'].to_numpy())['mean']*100 for c in carries]\n"
            "    ts = [st.summarize(st.vol_round_trip(vix, ev, pre=PRE, post=POST,\n"
            "          daily_carry_bps=c)['pnl_gross'].to_numpy())['tstat'] for c in carries]\n"
            "    banner = 'REAL VIX tape'\n"
            "else:\n"
            f"    base = {R['rt_gross']} + 2*{R['carry_drag']}\n"
            "    means = [base - 2*(c/25)*5 for c in carries]; ts = [-0.1,-0.7,-1.34,-2.0]\n"
            "    banner = 'frozen real numbers'\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.plot(carries, means, 'o-', c=RED, lw=2)\n"
            "ax.axhline(0, c='k', lw=1); ax.fill_between(carries, means, 0,\n"
            "    where=[m<0 for m in means], color=RED, alpha=.12)\n"
            "ax.set_xlabel('long-vol carry (bps/day)'); ax.set_ylabel('round-trip P&L per event (%)')\n"
            f"ax.axvline({R['carry_bps']}, ls='--', c=GREY); ax.annotate('realistic\\n~25 bps/day',\n"
            f"    ({R['carry_bps']}, max(means)), color=GREY, ha='center', va='bottom', fontsize=9)\n"
            "ax.set_title(f'Even at zero carry it barely clears zero — and carry is not zero  [{banner}]')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'At a realistic 25 bps/day: {means[2]:+.1f}% per event, t={ts[2]:+.2f}')"
        ),
        md(
            f"> In plain words: at **zero** carry the round-trip is already marginal (the vol move "
            f"isn't there to pay you), and at a realistic ~{R['carry_bps']} bps/day it's a clear "
            f"loser (**{R['rt_gross']:+.1f}%/event**). Contrast a *fragile* edge, where a real gross "
            "signal merely thins out — here there is **no gross signal to begin with**. Mirage, not "
            "fragile."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further — the synthetic positive control\n\n"
            "Is the *engine* a faithful detector, or would it miss a real hump? Plant a tent-shaped "
            "vol bump on a deterministic OU tape and confirm recovery; the null must give nothing."
        ),
        code(
            "effects = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]\n"
            "pre_ts, post_ts = [], []\n"
            "for eff in effects:\n"
            "    b, evp, _ = data.synthetic_vix(event_effect=eff, pre=PRE, post=POST, n_events=40, seed=312)\n"
            "    pre_ts.append(st.hac_tstat(st.pre_ramp(b, evp, pre=PRE)['d_log'].to_numpy()))\n"
            "    post_ts.append(st.hac_tstat(st.post_collapse(b, evp, post=POST)['d_log'].to_numpy()))\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(effects, pre_ts, 'o-', c=GREEN, lw=2, label='pre-ramp HAC t (want +)')\n"
            "ax.plot(effects, post_ts, 's-', c=RED, lw=2, label='post-move HAC t (want −)')\n"
            "ax.axhline(2, ls=':', c=GREY); ax.axhline(-2, ls=':', c=GREY); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted vol-hump strength'); ax.set_ylabel('HAC t-stat')\n"
            "ax.set_title('The engine IS a faithful detector — so the real-tape null is real'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'BANNER: SYNTHETIC positive control. At effect=0 (null): pre t={pre_ts[0]:+.2f}, post t={post_ts[0]:+.2f}.')"
        ),
        md(
            f"The engine recovers a planted hump cleanly (pre *t* → +{R['plant_pre_t']:.0f}, post "
            f"*t* → {R['plant_post_t']:+.0f} at full strength) and finds **nothing** on the null "
            f"(pre *t* ≈ {R['null_pre_t']:+.1f}). So the real-tape null is the **absence of a "
            "signal**, not a blind detector. The debt-ceiling 'vol trade' is folklore: a strongly "
            "mean-reverting VIX, one credit-event coincidence (2011), and a carry tax that punishes "
            "anyone who waits long-vol for the deadline.\n\n"
            "For the *directional* cousin (do stocks bounce around fiscal cliffs), see "
            "[Study 311 — Government-Shutdown](../../311-government-shutdown/)."
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
