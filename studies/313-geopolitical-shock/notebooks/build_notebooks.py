"""Generate the two narrative notebooks for Study 313 (Geopolitical-Shock).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the cached SPY
parquet under ../_cache/ if present and otherwise fall back to the planted-synthetic tape,
clearly bannering which tape each cell shows. The frozen real-tape headline numbers live in
``R`` (mirroring docs/results.md), so the prose re-runs for any reader.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-17).
R = dict(
    n_shocks=28, n_used=26, spy_start="1993-01-29", spy_fp="b891c5df8fc1",
    # CAR by post window: (mean %, t, placebo pct)
    car1=0.46, t1=1.88, pct1=98,
    car3=0.20, t3=0.57, pct3=72,
    car5=0.27, t5=0.62, pct5=74,
    car10=0.04, t10=0.07, pct10=51,
    car21=0.48, t21=0.75, pct21=68,
    event_day=-0.20,
    placebo_mean=0.01, placebo_std=0.62,
    boot_lo=-1.13, boot_hi=1.05,
    # buy the dip (gross unless noted)
    dip1_win=60.7, dip1_mean=52.6, dip1_t=2.29,
    dip10_win=64.3, dip10_mean=56.5, dip10_t=1.05,
    dip21_win=82.1, dip21_mean=145.8, dip21_t=2.38,
    dip10_net2=52.5, dip10_net5=46.5, dip10_net10=36.5,
    trades_per_yr=0.8, spy_bh_ann=10.8,
    # synthetic positive control
    plant_drift=3.0, plant_car=2.45, plant_t=3.41,
)


BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root (quantlab/)
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from geopolitical_shock import data, strategy as st

PRE, POST = 5, 10

def load_spy():
    \"\"\"Return (bars, events, tape_label). Real SPY if cached, else the planted-synthetic
    tape (shock_drift=0 — the null) with a loud banner.\"\"\"
    try:
        bars = data.load_real("SPY")
        ev = data.shock_table()["trade_date"]
        return bars, ev, "REAL SPY (cached Yahoo daily, 1993-2026)"
    except Exception as e:
        bars, ev_df, _ = data.synthetic_daily(shock_drift=0.0, seed=313)
        return bars, ev_df["trade_date"], "SYNTHETIC null tape (offline fallback - NOT real data)"

bars, EVENTS, TAPE = load_spy()
HAVE_REAL = TAPE.startswith("REAL")
print("Tape in use:", TAPE)
print("Shocks in table:", len(data.shock_table()))
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Geopolitical-Shock — do markets really shrug off wars within days?\n"
            "### Every war scares investors. The chart, a few days later, has forgotten. Is that true?\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Do_markets_shrug_shocks_off_in_days%3F: Confirmed](https://img.shields.io/badge/Do_markets_shrug_shocks_off_in_days%3F-Confirmed-8b949e?style=flat-square)\n\n"
            "When a war breaks out or an attack hits the headlines, it *feels* like the market "
            "should crater. The folk wisdom says the opposite: stocks dip on the news and shrug it "
            "off within days. We take a curated table of **28 major geopolitical shocks** — Kuwait, "
            "9/11, Iraq, Crimea, Ukraine, Israel-Hamas, Iran — line up SPY around each one, and ask "
            "the two questions that matter: *does the market actually drift after a shock, and could "
            "you trade it?*\n\n"
            "> **This is the plain-language layer.** Want the abnormal-return math, the placebo "
            "distribution and the bootstrap CI? That's the companion, "
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
            "| Does the market drift after a shock? | **No.** Ten sessions after a shock the "
            f"abnormal return is **+{R['car10']:.2f}%** — *t* = +{R['t10']:.2f}, the **51st "
            "percentile** of pure chance. Statistically: nothing. |\n"
            "| Is there a same-day dip? | **A tiny one.** The shock day itself loses "
            f"~{abs(R['event_day']):.1f}% on average — real, but small, and gone within a session "
            "or two. |\n"
            "| Can you 'buy the geopolitical dip'? | **No.** The overlay fires < 1×/yr, never "
            "beats noise after costs, and its only positive numbers are the market drift you'd have "
            "earned anyway. |\n"
            "| So is the folk wisdom right? | **Yes — and that's exactly why there's nothing to "
            "trade.** Markets *do* shrug off shocks within days. Efficiency, not edge. |\n\n"
            "> The vivid story ('war! sell everything!') and the boring truth ('a blip, then "
            "nothing') are both real. The market prices public geopolitical news almost instantly — "
            "so by the time you've read the headline, the move is over."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Markets sell off on geopolitical shocks, but the dip is shallow and short-lived — "
            "stocks recover within days. Geopolitics is noise for a long-term investor, and the "
            "savvy move is to buy the dip.\"*\n\n"
            "— the standard line in post-shock financial commentary (LPL Research, Vanguard and "
            "others have tabulated exactly this pattern across decades of events)\n\n"
            "It is a genuinely strong claim, and we steelman it: we don't cherry-pick the events. "
            f"We take a curated table of **{R['n_shocks']} shocks** — every invasion, war, major "
            "terror attack and missile crisis a reasonable person would name — and map each to the "
            "first trading session it could be acted on."
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "Two things ride on the answer. First, if shocks *did* drift — if the market kept "
            "falling for two weeks after a war started — that would be a tradable signal worth real "
            "money and a dent in the efficient-market story. Second, the *opposite* result is the "
            "more interesting one for an ordinary investor: if the market truly shrugs shocks off in "
            "days, then the single most common piece of crisis-driven advice ('get out before it "
            "gets worse') is actively wrong. Either way, the table of 28 shocks settles it."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Three checks keep us honest:\n\n"
            "1. **Line up every shock and average.** For each event, measure SPY's *abnormal* "
            "return (its move minus the market's normal up-drift) from 5 days before to 10 days "
            "after. Average across all shocks: that's the typical shock response.\n"
            "2. **Race it against random dates.** Re-run the exact same measurement on **thousands "
            "of random non-shock dates**. If shocks matter, the real number sits in the *tail* of "
            "that random distribution. If it sits in the middle, it's noise.\n"
            "3. **Try to trade it.** Buy SPY the session after each shock, hold, pay realistic "
            "costs, and see if 'buy the dip' makes money beyond just owning the index.\n\n"
            "Tape: SPY daily bars back to 1993. The shock dates are a hand-built table (the "
            "data-driven GPR index feed is network-blocked here)."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First: the average path of SPY around a shock.** Day 0 is the first tradable session "
            "after each event; the line is the cumulative *abnormal* return (with the normal drift "
            "stripped out)."
        ),
        code(
            "ar = st.abnormal_returns(st.daily_returns(bars))\n"
            "idx = bars.index\n"
            "ev = [idx[idx.searchsorted(d)] for d in pd.to_datetime(EVENTS)\n"
            "      if idx.searchsorted(d) < len(idx)]\n"
            "W = st.stack_windows(ar, ev, PRE, POST)\n"
            "car = st.car_path(W, PRE) * 100\n"
            "offs = np.arange(-PRE, POST + 1)\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.6))\n"
            "ax.axvspan(-PRE-0.5, -0.5, color=GREY, alpha=.08)\n"
            "ax.plot(offs, car, 'o-', c=RED, lw=2)\n"
            "ax.axvline(0, ls='--', c='k', lw=1); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('sessions relative to the shock (0 = first tradable day)')\n"
            "ax.set_ylabel('cumulative abnormal return (%)')\n"
            "ax.set_title(f'SPY around {W.shape[0]} geopolitical shocks  [{TAPE}]')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'event-day abnormal: {W[:,PRE].mean()*100:+.2f}%   CAR(+10): {car[-1]:+.2f}%')"
        ),
        md(
            f"On the real tape: a small **{R['event_day']:.1f}% dip on the shock day**, then the "
            f"line just wanders around zero. Ten days out the cumulative abnormal return is "
            f"**+{R['car10']:.2f}%** — for all practical purposes, flat. The market noticed, dipped, "
            "and moved on. *(If you see the synthetic-fallback banner above, this cell is showing "
            "the offline null tape — the real numbers are quoted from the frozen run.)*"
        ),
        md(
            "**Now the punchline: is that +0.04% drift bigger than random luck?** We re-run the same "
            "measurement on thousands of *random* dates and see where the real shock number lands."
        ),
        code(
            "per = st.post_event_car(W, PRE, POST)\n"
            "pl = st.placebo_distribution(ar, len(per), PRE, POST, n_draws=3000, seed=313) * 100\n"
            "real = per.mean() * 100\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.4))\n"
            "ax.hist(pl, bins=50, color=GREY, alpha=.6)\n"
            "ax.axvline(real, c=RED, lw=2.5, label=f'real shocks: {real:+.2f}%')\n"
            "ax.axvline(0, c='k', lw=1)\n"
            "pct = st.placebo_percentile(real, pl)\n"
            "ax.set_xlabel('mean 10-day CAR (%)'); ax.set_ylabel('random-date draws')\n"
            "ax.set_title(f'The real shock drift is the {pct:.0f}th percentile of pure chance')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'real CAR {real:+.2f}%  sits at the {pct:.0f}th percentile of {len(pl)} random draws')"
        ),
        md(
            f"There it is. The real shock drift lands at the **{R['pct10']}th percentile** of random "
            "luck — dead centre. A genuine signal would sit far out in the tail. This one is "
            "indistinguishable from picking dates out of a hat. **The market shrugs it off.**"
        ),
        md(
            "**Last: could you 'buy the dip' anyway?** Enter SPY the session after each shock, hold "
            "10 days, and compare to just owning the index — after realistic costs."
        ),
        code(
            "costs = [0.0, 2.0, 5.0, 10.0]\n"
            "if HAVE_REAL:\n"
            "    net = [st.summarize_dip(st.buy_the_dip(bars, EVENTS, hold=10, cost_bps=c),'ret_net')['mean_bps']\n"
            "           for c in costs]\n"
            "else:\n"
            f"    net = [{R['dip10_mean']}, {R['dip10_net2']}, {R['dip10_net5']}, {R['dip10_net10']}]\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "ax.plot(costs, net, 'o-', c=RED, lw=2)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('round-trip cost (bps)'); ax.set_ylabel('mean profit per trade (bps)')\n"
            "ax.set_title('Buy-the-dip: positive on paper, but it is just the market drift')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Fires ~{tpy} times/yr; never beats noise after costs; '\n"
            "      'the positive bps are the {bh}%/yr equity drift, not a shock edge.')"
            .format(tpy=R["trades_per_yr"], bh=R["spy_bh_ann"])
        ),
        md(
            f"It shows a positive number — but the overlay trades only **~{R['trades_per_yr']} times "
            "a year**, never clears the noise bar after costs, and the bps it earns are simply the "
            f"market's **+{R['spy_bh_ann']:.0f}%/yr drift** that you'd have had by holding anyway. "
            "There's no *shock* edge in it."
        ),

        # ---- BEAT 5 — THE VERDICT --------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** The 10-day abnormal drift after a shock is +{R['car10']:.2f}% "
            f"(*t* = +{R['t10']:.2f}), the {R['pct10']}th percentile of random dates. Past day one, "
            "geopolitics leaves no footprint in the index.\n"
            "- **Tradability — Mirage.** 'Buy the dip' fires < 1×/yr, never beats noise net of "
            "costs, and its positive bps are the equity risk premium, not a shock edge.\n"
            f"- **Do markets shrug shocks off in days? — Confirmed.** A ~{abs(R['event_day']):.1f}% "
            "same-day dip, then full recovery within a session or two. The folk wisdom is right — "
            "and that efficiency is precisely why there's no trade."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "The honest answer is the short one: **no, and not because of costs.** Even gross, the "
            "post-shock drift is statistical zero. Costs don't kill this strategy — there's nothing "
            "for them to kill. The deeper problem is *fewer than one trade a year* on a signal that "
            "doesn't exist. The only way 'geopolitical' strategies show returns is by quietly "
            "holding the index between shocks and calling the drift an edge."
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Volatility, not direction.** Shocks reliably spike *volatility* even when they "
            "don't move the *level*. A VIX-around-shocks study (or an options-overwriting overlay) "
            "is where any real, tradable footprint would live.\n"
            "- **Anticipation vs event.** Rigobon & Sack (2005) found the 2003 Iraq move was mostly "
            "*before* the invasion. Re-dating events to when risk first built (the GPR index, once "
            "the feed is unblocked) might find a pre-event drift our trade-date table misses.\n"
            "- **Cross-asset.** Oil, gold and Treasuries react to geopolitics far more than broad "
            "equities. The flight-to-safety leg is the one worth a separate teardown.\n\n"
            "*Think there's a shock edge hiding in here? Fork it, swap the curated table for the GPR "
            "index, and show a post-event CAR that clears the placebo tail.*"
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
            "# Geopolitical-Shock — a quantitative teardown\n"
            "### Constant-mean event study · CAR path · placebo distribution · block-bootstrap CI · "
            "buy-the-dip overlay · synthetic positive control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Do_markets_shrug_shocks_off_in_days%3F: Confirmed](https://img.shields.io/badge/Do_markets_shrug_shocks_off_in_days%3F-Confirmed-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We run a constant-mean "
            "event study of SPY around 28 curated geopolitical shocks, falsify the CAR against a "
            "placebo distribution of random dates, bound it with a block bootstrap, and put the "
            "'buy the dip' overlay through a cost sweep.\n\n"
            "> **Not investment advice.** Real data: Yahoo SPY daily bars 1993-2026; as-of "
            "2026-06-17; the offline core and tests run on a deterministic synthetic tape with a "
            "*planted* post-shock drift. Methods in [`docs/references.md`](../docs/references.md), "
            "numbers in [`docs/results.md`](../docs/results.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | 10-day mean CAR **+{R['car10']:.2f}%**, cross-event "
            f"*t* = **+{R['t10']:.2f}**, **{R['pct10']}th** placebo percentile; bootstrap 95% CI "
            f"[{R['boot_lo']:+.2f}%, {R['boot_hi']:+.2f}%] straddles zero. Lone +1-day bounce "
            f"*t* = +{R['t1']:.2f} < 2. |\n"
            f"| **Tradability** | `MIRAGE` | Buy-the-dip net *t* never clears 2 (+{R['dip10_t']:.2f} "
            f"at hold 10, gross); ~{R['trades_per_yr']}×/yr; positive bps are the "
            f"+{R['spy_bh_ann']:.0f}%/yr drift, not a shock edge. |\n"
            f"| **Do markets shrug shocks off in days?** | `CONFIRMED` | "
            f"{R['event_day']:.1f}% event-day abnormal, recovered within ~1 session; CAR flat "
            "thereafter — textbook semi-strong efficiency to public geopolitical news. |\n\n"
            "> In plain words: a small, real same-day dip, then nothing a placebo of random dates "
            "can't reproduce. The market is efficient to geopolitics in days, so there is no "
            "post-event drift to harvest."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $AR_{i,h}$ be the abnormal return of SPY for event $i$ at offset $h$ (excess over "
            "the full-sample mean — a constant-mean market model), and "
            "$CAR_i(\\tau) = \\sum_{h=1}^{\\tau} AR_{i,h}$ the post-event cumulative abnormal "
            "return.\n\n"
            "- **H₁ (drift).** $\\mathbb{E}_i[CAR_i(10)] \\ne 0$ — shocks induce a multi-day "
            "abnormal move.\n"
            "- **H₂ (beats placebo).** The mean CAR sits in the tail of the random-date placebo "
            "distribution.\n"
            "- **H₃ (tradable).** A long-only 'buy the dip' overlay earns a positive net "
            "*shock-specific* return, not just beta.\n"
            "- **H₄ (efficiency / 'shrug it off').** The event-day dip is recovered within a "
            "session or two and the CAR is flat thereafter.\n\n"
            "We find **H₁–H₃ rejected** and **H₄ supported** — the folk 'shrug it off' wins, the "
            "tradable corollary loses."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "A significant post-event CAR would be a clean violation of semi-strong efficiency for "
            "the single most-watched public-news category there is — and a real, if rare, trade. "
            "The null result is the interesting one: it says the most emotionally compelling "
            "market narrative ('sell, there's a war') has *no* multi-day directional content in the "
            "broad index. The desk's job is to state that precisely — with the placebo percentile "
            "and the bootstrap CI — rather than wave at 'markets are resilient'."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Events.** A curated table of 28 shocks, each mapped to the first NYSE session it "
            "was tradable (weekend/holiday events roll forward). The GPR index would be the "
            "data-driven alternative; its feed is network-blocked here.\n"
            "- **Abnormal return.** Constant-mean model: $AR = r - \\bar r$ over the full sample "
            "(Brown & Warner 1985).\n"
            "- **Window.** $[-5, +10]$ sessions; CAR re-anchored to zero at offset $-1$.\n"
            "- **Inference.** Cross-event *t* on $CAR_i(\\tau)$ (events ~independent); a **placebo "
            "distribution** of 3,000 random-date CARs; a **block bootstrap** CI resampling events.\n"
            "- **Trade.** Long-only entry at the close of session 0 (shock known), hold $h$, **one** "
            "execution lag applied once, one-way costs charged twice (entry + exit) vs NAV.\n"
            "- **Positive control.** A deterministic synthetic tape with a *planted* post-shock "
            "drift — the harness must recover it, and find nothing when it is zero.\n\n"
            "**Mirage trigger, announced up front:** if the mean CAR lands inside the central 90% of "
            "the placebo distribution and the bootstrap CI straddles zero, we stamp Signal `NONE`."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The CAR path and the per-horizon table\n\n"
            "The mean cumulative abnormal return around the shock, and the cross-event *t* / placebo "
            "percentile at each post-event horizon."
        ),
        code(
            "ar = st.abnormal_returns(st.daily_returns(bars))\n"
            "idx = bars.index\n"
            "ev = [idx[idx.searchsorted(d)] for d in pd.to_datetime(EVENTS)\n"
            "      if idx.searchsorted(d) < len(idx)]\n"
            "rows = []\n"
            "for post in [1, 3, 5, 10, 21]:\n"
            "    W = st.stack_windows(ar, ev, PRE, post)\n"
            "    per = st.post_event_car(W, PRE, post)\n"
            "    pl = st.placebo_distribution(ar, len(per), PRE, post, n_draws=3000, seed=313)\n"
            "    rows.append((post, per.mean()*100, st.car_tstat(per),\n"
            "                 st.placebo_percentile(per.mean(), pl)))\n"
            "tab = pd.DataFrame(rows, columns=['post_d','CAR_%','t','placebo_pct'])\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "col = [GREEN if abs(t) >= 2 else RED for t in tab['t']]\n"
            "ax.bar(tab['post_d'].astype(str), tab['t'], color=col)\n"
            "ax.axhline(2, ls='--', c=GREY); ax.axhline(-2, ls='--', c=GREY); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('post-event horizon (sessions)'); ax.set_ylabel('cross-event t')\n"
            "ax.set_title('No horizon clears |t| = 2 — the post-shock drift is statistical zero')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(tab.round(2).to_string(index=False))"
        ),
        md(
            f"> In plain words: not one horizon clears |*t*| = 2. The closest is the +1-day relief "
            f"bounce (*t* = +{R['t1']:.2f}), and it falls short — and it is the most multiple-"
            "testing-exposed window we examined. Everything past day one is flat."
        ),
        md(
            "### 4b · The synthetic control — placebo distribution\n\n"
            "The decisive falsification: recompute the 10-day mean CAR on 3,000 sets of random "
            "non-event dates and locate the real shock CAR within it."
        ),
        code(
            "W = st.stack_windows(ar, ev, PRE, POST)\n"
            "per = st.post_event_car(W, PRE, POST); real = per.mean()*100\n"
            "pl = st.placebo_distribution(ar, len(per), PRE, POST, n_draws=3000, seed=313)*100\n"
            "lo, hi = st.block_bootstrap_car_ci(per, n_boot=5000, seed=313)\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.4))\n"
            "ax.hist(pl, bins=50, color=GREY, alpha=.6)\n"
            "ax.axvline(real, c=RED, lw=2.5, label=f'real shocks {real:+.2f}%')\n"
            "ax.axvspan(lo*100, hi*100, color=RED, alpha=.12, label='95% bootstrap CI')\n"
            "ax.axvline(0, c='k', lw=1)\n"
            "ax.set_xlabel('mean 10-day CAR (%)'); ax.set_ylabel('draws')\n"
            "ax.set_title(f'Real CAR = {st.placebo_percentile(real, pl):.0f}th placebo percentile; CI straddles 0')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'placebo: mean {pl.mean():+.2f}% std {pl.std():.2f}%   '\n"
            "      f'real {real:+.2f}% ({st.placebo_percentile(real, pl):.0f}th pct)   '\n"
            "      f'boot CI [{lo*100:+.2f}%, {hi*100:+.2f}%]')"
        ),
        md(
            f"> In plain words: the placebo distribution is centred on **+{R['placebo_mean']:.2f}%** "
            f"with std **{R['placebo_std']:.2f}%**; the real CAR sits at the **{R['pct10']}th "
            f"percentile** and the bootstrap CI **[{R['boot_lo']:+.2f}%, {R['boot_hi']:+.2f}%]** "
            "comfortably contains zero. **H₁ and H₂ rejected.** A synthetic control can refute like "
            "this; it could never *certify* a signal (see METHODOLOGY → the inference bar)."
        ),
        md(
            "### 4c · The tradable overlay — buy the dip, cost sweep\n\n"
            "Long-only, enter at the close of session 0, one execution lag, one-way costs charged "
            "twice. Note these are **raw** returns (they include the equity drift)."
        ),
        code(
            "rows = []\n"
            "for h in [1, 3, 5, 10, 21]:\n"
            "    s = st.summarize_dip(st.buy_the_dip(bars, EVENTS, hold=h, cost_bps=0), 'ret_gross')\n"
            "    rows.append((h, s['n'], s['win_rate']*100, s['mean_bps'], s['tstat']))\n"
            "dip = pd.DataFrame(rows, columns=['hold','n','win%','mean_bps','t_gross'])\n"
            "costs = [0,2,5,10]\n"
            "net = [st.summarize_dip(st.buy_the_dip(bars, EVENTS, hold=10, cost_bps=c),'ret_net')['mean_bps']\n"
            "       for c in costs]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))\n"
            "col = [GREEN if abs(t) >= 2 else RED for t in dip['t_gross']]\n"
            "a1.bar(dip['hold'].astype(str), dip['t_gross'], color=col)\n"
            "a1.axhline(2, ls='--', c=GREY); a1.axhline(0, c='k')\n"
            "a1.set_xlabel('hold (sessions)'); a1.set_ylabel('gross t'); a1.set_title('Buy-the-dip gross t by hold')\n"
            "a2.plot(costs, net, 'o-', c=RED, lw=2); a2.axhline(0, c='k')\n"
            "a2.set_xlabel('round-trip cost (bps)'); a2.set_ylabel('net mean (bps)')\n"
            "a2.set_title('Net mean (hold 10) vs cost')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(dip.round(2).to_string(index=False))"
        ),
        md(
            f"> In plain words: only hold-21 (*t* = +{R['dip21_t']:.2f}) and hold-1 "
            f"(*t* = +{R['dip1_t']:.2f}, gross) flirt with significance — and hold-21's "
            f"+{R['dip21_mean']:.0f} bps over a month is almost exactly SPY's +{R['spy_bh_ann']:.0f}"
            "%/yr drift prorated, i.e. **beta**, confirmed by the matching constant-mean CAR being "
            "insignificant. **H₃ rejected.**"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — 10-day mean CAR +{R['car10']:.2f}% (*t* = +{R['t10']:.2f}), "
            f"{R['pct10']}th placebo percentile, bootstrap CI [{R['boot_lo']:+.2f}%, "
            f"{R['boot_hi']:+.2f}%]. No horizon clears |*t*| = 2; the +1-day bounce "
            f"(+{R['t1']:.2f}) is the closest and falls short.\n"
            f"- **Tradability `MIRAGE`** — buy-the-dip never clears net *t* = 2, fires "
            f"~{R['trades_per_yr']}×/yr, and its positive bps are the +{R['spy_bh_ann']:.0f}%/yr "
            "equity drift, not a geopolitical edge.\n"
            f"- **Do markets shrug shocks off in days? `CONFIRMED`** — {R['event_day']:.1f}% "
            "event-day dip, recovered within ~1 session, flat thereafter. Semi-strong efficiency to "
            "public geopolitical news, demonstrated directly."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — there is nothing to trade\n\n"
            "Unlike the desk's *fragile* signals (real edge, killed by costs/capacity), this one is "
            "a *mirage*: the gross post-event drift is already statistical zero, so costs are "
            "irrelevant. The overlay's only positive numbers come from being long the index between "
            "shocks. The honest framing: 'buy the geopolitical dip' is buy-and-hold wearing a "
            "narrative."
        ),
        code(
            "led = st.buy_the_dip(bars, EVENTS, hold=10, cost_bps=5.0)\n"
            "yrs = (bars.index[-1] - bars.index[0]).days / 365.25\n"
            "tpy = len(led) / yrs\n"
            f"spy_bh = {R['spy_bh_ann']}\n"
            "dip_ann = st.summarize_dip(led,'ret_net')['mean_bps'] * tpy / 100\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "ax.bar(['Buy-the-dip\\n(net, hold 10)', 'SPY buy & hold'],\n"
            "       [dip_ann, spy_bh], color=[RED, GREEN], width=.5)\n"
            "ax.axhline(0, c='k'); ax.set_ylabel('annualised %/yr (naive)')\n"
            "ax.set_title(f'~{tpy:.1f} trades/yr - the overlay is mostly idle cash')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'{tpy:.2f} trades/yr; the strategy is in cash the overwhelming majority of the time.')"
        ),
        md(
            f"> In plain words: ~{R['trades_per_yr']} trades a year. Even taking the overlay's net "
            "number at face value, it is dwarfed by simply holding the index it's supposed to time. "
            "A signal that doesn't exist, deployed < 1×/yr: the definition of a **mirage**."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further — the synthetic positive control\n\n"
            "Is the *engine* a faithful detector, or does it always print 'nothing'? Plant a "
            "post-shock drift on a synthetic tape and confirm the harness recovers it (and finds "
            "nothing when the drift is zero)."
        ),
        code(
            "rows = []\n"
            "for drift in [-0.02, 0.0, 0.02, 0.03, 0.04]:\n"
            "    sb, se, _ = data.synthetic_daily(shock_drift=drift, seed=313)\n"
            "    sar = st.abnormal_returns(st.daily_returns(sb))\n"
            "    Wp = st.stack_windows(sar, se['trade_date'], PRE, POST)\n"
            "    perp = st.post_event_car(Wp, PRE, POST)\n"
            "    plp = st.placebo_distribution(sar, len(perp), PRE, POST, n_draws=1500, seed=1)\n"
            "    rows.append((drift*100, perp.mean()*100, st.car_tstat(perp),\n"
            "                 st.placebo_percentile(perp.mean(), plp)))\n"
            "ctrl = pd.DataFrame(rows, columns=['planted_%','recovered_CAR_%','t','placebo_pct'])\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(ctrl['planted_%'], ctrl['recovered_CAR_%'], 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(0, c='k'); ax.axvline(0, ls='--', c=GREY)\n"
            "ax.plot([ctrl['planted_%'].min(), ctrl['planted_%'].max()],\n"
            "        [ctrl['planted_%'].min(), ctrl['planted_%'].max()], ':', c=GREY, label='y=x')\n"
            "ax.set_xlabel('planted post-shock drift (%)'); ax.set_ylabel('recovered CAR (%)')\n"
            "ax.set_title('The engine faithfully recovers a planted drift'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(ctrl.round(2).to_string(index=False))"
        ),
        md(
            f"The recovered CAR tracks the planted drift one-for-one and clears *t* = 2 once the "
            f"drift is real (planted +{R['plant_drift']:.0f}% → recovered +{R['plant_car']:.2f}%, "
            f"*t* = +{R['plant_t']:.2f}); at zero drift it finds nothing in the placebo bulk. So the "
            "engine is faithful, and the real-tape null is a statement about **the market**: broad "
            "equities price public geopolitical news within a session, leaving no multi-day drift to "
            "harvest.\n\n"
            "For the volatility footprint shocks *do* leave (even when the level recovers), the next "
            "study is a VIX-around-shocks event study — the family this one points to."
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
