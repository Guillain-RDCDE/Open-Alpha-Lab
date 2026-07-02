"""Generate the two narrative notebooks for Study 554 (Airline-Bookings).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). This is a synthetic-only
study — there is no real flight-bookings tape — so every code cell recomputes the headline numbers
from the deterministic generator (seed 554), fully offline. The dict ``R`` below mirrors
docs/results.md and is used only for the prose; the live cells reproduce it exactly.
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


# Headline numbers — mirror of docs/results.md (synthetic efficient-market world, seed 554,
# lead_beta=0, contemp_beta=0.9, 180 months 2011-01 -> 2025-12; frame fp fb00515e49fb; as-of 2026-06-30).
R = dict(
    fp="fb00515e49fb", n=180, span="2011-01 -> 2025-12",
    pred_slope=-0.0003, pred_t_ols=-0.07, pred_t_hac=-0.07, pred_corr=-0.005, placebo_p=0.943,
    contemp_slope=0.0553, contemp_t_ols=11.66, contemp_t_hac=12.61, contemp_corr=0.66,
    lf_net_ann=-0.5, lf_bh_ann=3.3, lf_gap=-3.8, lf_trades=71,
    ls_net_ann=-5.6, ls_gap=-8.9, ls_trades=143,
    cost_bps=10.0, borrow_bps=300.0,
    # robustness: (label, slope, t_hac)
    win=[("full", -0.000, -0.07), ("first half", 0.004, 0.70), ("second half", -0.006, -0.93),
         ("third 1", -0.002, -0.27), ("third 2", 0.000, 0.05), ("third 3", -0.004, -0.51)],
    # positive control: (lead_beta, mean HAC t over 25 seeds)
    ctrl=[(0.0, -0.35), (0.25, 2.75), (0.50, 5.77), (0.75, 8.63), (1.00, 11.28)],
    planted_t=7.73, planted_placebo_p=0.0005,
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

from airline_bookings import data, strategy as st

# Synthetic-only: no free flight-bookings index exists. The headline world is the realistic
# efficient-market case -- bookings co-move with the sector (contemp_beta=0.9) but carry NO
# forward information (lead_beta=0). Everything below recomputes from this deterministic frame.
FRAME, TRUTH = data.synthetic_series(lead_beta=0.0, contemp_beta=0.9, seed=554)
HAVE_REAL = not data.fetch_series(fetch=False).empty     # False by design (no real tape)
print("synthetic frame:", len(FRAME), "months", FRAME.index[0].date(), "->", FRAME.index[-1].date(),
      "| fp", data.fingerprint(FRAME), "| real tape present:", HAVE_REAL)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Do flight bookings *lead* airline stocks? ✈️\n"
            "### An alt-data 'leading indicator', in plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Already_in_the_price%3F: Confirmed](https://img.shields.io/badge/Already_in_the_price%3F-Confirmed-8b949e?style=flat-square)\n\n"
            "Here's a pitch you can buy today. A data vendor watches millions of flight bookings — "
            "ticket sales, search volume, card transactions — and packages it into a *bookings "
            "momentum* index. Their claim: 'we see the travel rebound **weeks before** the airlines "
            "report it, so our signal **leads** airline stocks — get in early.'\n\n"
            "It sounds unbeatable. If you *know* demand is surging before everyone else, surely you "
            "buy the airlines and wait for the pop? We test exactly that: does *this* month's booking "
            "momentum predict *next* month's airline return — the only thing you could actually "
            "trade on?\n\n"
            "> ⚠️ **A data note up front.** There is **no free, honest flight-bookings index** a "
            "retail reader can reach — the real feeds are paywalled and revised after the fact. So we "
            "build a *synthetic* world we fully control, where we can dial the 'lead' up or down and "
            "check whether our test can even see it. That means this study can never be stamped "
            "`REAL` — but it can still show you the **trap**.\n\n"
            "> 📓 **This is the plain-language layer.** Want the *t*-stats, the placebo test and the "
            "cost maths? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ----
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Do bookings and airline stocks move together? | **Yes, strongly** — in the *same* "
            f"month they track each other tightly (correlation **+{R['contemp_corr']}**). The chart "
            "looks amazing. |\n"
            f"| Does booking momentum *lead* next month's return? | **No.** The forward relationship "
            f"is **noise** (correlation **{R['pred_corr']}**). A shuffle test says the odds it's real "
            f"are *p* = **{R['placebo_p']}** — i.e. it's indistinguishable from luck. |\n"
            f"| Could you trade it? | **No.** Buying airlines when bookings are strong *loses* to "
            "just holding them. |\n"
            "| So what happened to the lead? | **It's already in the price.** By the time you can see "
            "the booking print, the market has already discounted the demand. |\n\n"
            "> The seductive part (the same-month co-move) is real. The tradable part (a *forward* "
            "edge) is gone. That gap is the whole story."
        ),

        # ---- BEAT 1 — THE CLAIM ----
        md(
            "## 1 · The claim\n\n"
            "> *\"Our flight-bookings index turns up before airlines report — so it leads airline "
            "stocks.\"* — the alt-data leading-indicator pitch.\n\n"
            "The logic is tempting: bookings are a real-time read on travel *demand*, and demand "
            "drives airline *earnings*, and earnings drive *stock prices*. So bookings → earnings → "
            "prices, with bookings first in line. If you sit at the front of that chain, you're early "
            "to every move. The hidden assumption: that the price waits for the earnings report. In "
            "a forward-looking market, it doesn't."
        ),

        # ---- BEAT 2 — SO WHAT ----
        md(
            "## 2 · So what?\n\n"
            "Alt-data is a multi-billion-dollar industry sold on exactly this promise — satellite "
            "images of parking lots, card-panel spend, web scrapes, and travel bookings, all pitched "
            "as *leading* indicators. If the promise held for bookings, an early buyer would harvest "
            "a clean edge. If it doesn't — if the signal is real but *already priced* — you've paid "
            "for a dashboard that tells you what the market already knows. The desk has looked at "
            "sentiment leads before ([257 AAII](../../257-aaii-sentiment/), "
            "[335 Buzz-ETF](../../335-buzz-sentiment-etf/)); this is the **travel-booking** instance."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ----
        md(
            "## 3 · How would we even know?\n\n"
            "1. **Build the signal.** Each month, a *booking momentum* number (is travel demand "
            "trending up or down?).\n"
            "2. **Two questions, not one.**\n"
            "   - *Same-month:* do bookings and the airline return move together **this** month? "
            "(This is the shiny chart the vendor shows you.)\n"
            "   - *Forward:* does **this** month's booking momentum predict **next** month's return? "
            "(This is the only thing you can trade.)\n"
            "3. **Try to trade it.** Buy the airline basket when bookings are strong; compare to just "
            "holding the basket, after costs.\n\n"
            "*Timing:* the signal at a month uses only data through that month's close; the return we "
            "test is the *next* month's. That one-month gap between a public signal and the traded "
            "return is our single, documented rule — no peeking at the future."
        ),

        # ---- BEAT 4 — THE TEARDOWN ----
        md(
            "## 4 · The teardown — the two charts that tell the story\n\n"
            "**First, the vendor's chart: bookings vs the airline return in the *same* month.**"
        ),
        code(
            "c = st.contemporaneous_regression(FRAME)\n"
            "p = st.predictive_regression(FRAME)\n"
            "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))\n"
            "ax1.scatter(FRAME['bookings_mom'], FRAME['ret']*100, s=18, color=GREEN, alpha=.7)\n"
            "ax1.set_title(f\"SAME month: bookings vs return (corr +{c['corr']:.2f}) -- looks great!\")\n"
            "ax1.set_xlabel('booking momentum'); ax1.set_ylabel('airline return %')\n"
            "ax2.scatter(FRAME['bookings_mom'], FRAME['fwd_ret']*100, s=18, color=RED, alpha=.7)\n"
            "ax2.set_title(f\"NEXT month: bookings vs forward return (corr {p['corr']:+.2f}) -- nothing\")\n"
            "ax2.set_xlabel('booking momentum this month'); ax2.set_ylabel('airline return NEXT month %')\n"
            "fig.suptitle('synthetic efficient-market world (offline, deterministic)', fontsize=9, color=GREY)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"same-month corr {c['corr']:+.2f} | next-month corr {p['corr']:+.2f}\")"
        ),
        md(
            f"Left panel: a gorgeous straight line (correlation **+{R['contemp_corr']}**). Right "
            f"panel: a shapeless cloud (correlation **{R['pred_corr']}**). Same signal, same stock — "
            "the only difference is *timing*. Bookings track the airline return **this** month, but "
            "tell you **nothing** about **next** month. The lead is an illusion of the same-month "
            "chart."
        ),
        md(
            "**Second: could you trade the signal?** Buy the airline basket when booking momentum is "
            "positive, else stand aside — and compare to just holding the basket."
        ),
        code(
            "lf = st.timing_rule(FRAME, mode='long_flat', cost_bps=10.0)\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.3))\n"
            "ax.bar(['time it\\non bookings\\n(net)', 'just hold\\nthe basket'],\n"
            "       [lf['net_ann']*100, lf['bh_ann']*100], color=[RED, GREEN], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('annualised return %')\n"
            "ax.set_title(f\"Timing on bookings nets {lf['net_ann']*100:+.1f}%/yr vs holding {lf['bh_ann']*100:+.1f}%/yr\")\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"long/flat net {lf['net_ann']*100:+.1f}%/yr | buy-and-hold {lf['bh_ann']*100:+.1f}%/yr | gap {(lf['net_ann']-lf['bh_ann'])*100:+.1f} pp\")"
        ),
        md(
            f"Trading the booking signal **loses to doing nothing**: about **{R['lf_net_ann']}%/yr** "
            f"net versus **+{R['lf_bh_ann']}%/yr** for just holding the basket — a "
            f"**{R['lf_gap']} pp/yr** hole. The signal has no forward information, so timing on it "
            "just adds trading costs to a coin flip."
        ),

        # ---- BEAT 5 — VERDICT ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** The *forward* relationship is noise (correlation {R['pred_corr']}, "
            f"shuffle *p* {R['placebo_p']}); it has no direction you can bet on. (And because there's "
            "no real booking tape, it could never be stamped higher than `WEAK` anyway.)\n"
            "- **Tradability — Mirage.** Timing on bookings loses to buy-and-hold before you even pay "
            "the short borrow.\n"
            "- **Already in the price? — Confirmed.** The huge *same-month* co-move with a dead "
            "*forward* signal is the fingerprint of a market that already discounted the demand."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ----
        md(
            "## 6 · Could you actually trade it?\n\n"
            "No — and it's worth seeing *why* it's not just 'the effect is small'. The effect is "
            "**there**, it's just in the wrong place: contemporaneous, not forward. You can't trade "
            "'this month's return' with 'this month's signal' — by the time the booking print is "
            "public, the price has moved. To profit you'd need the signal *before* everyone else, "
            "which is exactly what a widely-sold data feed can't give you."
        ),

        # ---- BEAT 7 — GOING FURTHER ----
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The engine really can see a lead** — when one exists. In the quant notebook we "
            "*plant* a genuine forward lead and watch the test light up, then dial it to zero and "
            "watch it go flat. That's how we know the 'no signal' here is the world, not a broken "
            "ruler.\n"
            "- **Other alt-data leads on the bench:** [257 AAII](../../257-aaii-sentiment/), "
            "[335 Buzz-ETF](../../335-buzz-sentiment-etf/), "
            "[392 Glassdoor](../../392-glassdoor-sentiment/).\n\n"
            "*Think a real booking feed would lead? Fork this, wire in a genuine point-in-time "
            "bookings index, and show the **forward** regression clearing t = 2 net of costs — that's "
            "the bar.*"
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
            "# Airline-Bookings — a quantitative teardown 🔬\n"
            "### predictive vs contemporaneous HAC-*t* · block-shuffle placebo · timing rule net of costs & borrow · six-window sweep · seed-robust synthetic control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Already_in_the_price%3F: Confirmed](https://img.shields.io/badge/Already_in_the_price%3F-Confirmed-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We test whether a flight-"
            "booking momentum signal **leads** airline-basket returns (a tradable forward edge) or "
            "merely **co-moves** with them (already-priced).\n\n"
            "> ⚠️ **Synthetic-only.** No free, point-in-time flight-bookings index is retail-reachable "
            "(the real card-panel / GDS feeds are paywalled), so a `REAL` stamp is impossible — the "
            f"ceiling is `WEAK`. The headline world is the efficient-market case (`lead_beta=0`, "
            f"`contemp_beta=0.9`, seed 554, 180 months, frame fp `{R['fp']}`, as-of 2026-06-30). "
            "Numbers mirror [`docs/results.md`](../docs/results.md); methods in "
            "[`docs/references.md`](../docs/references.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Predictive slope HAC *t* **{R['pred_t_hac']}** (corr "
            f"**{R['pred_corr']}**, placebo *p* **{R['placebo_p']}**); sign wanders across sub-samples. "
            "Synthetic-only ⇒ capped at `WEAK` regardless. |\n"
            f"| **Tradability** | `MIRAGE` | Long/flat nets **{R['lf_net_ann']}%/yr** vs buy-and-hold "
            f"**+{R['lf_bh_ann']}%/yr** ({R['lf_gap']} pp); long/short nets **{R['ls_net_ann']}%/yr** "
            f"after {R['borrow_bps']:.0f} bps borrow. |\n"
            f"| **Already in the price?** | `CONFIRMED` | *Contemporaneous* slope HAC *t* "
            f"**+{R['contemp_t_hac']}** (corr **+{R['contemp_corr']}**) vs a dead *forward* slope. |\n\n"
            "> 💡 In plain words: the engine works (the synthetic control proves it), but the forward "
            "signal is noise while the same-month co-move is enormous — the classic already-priced "
            "alt-data trap."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $b_t$ be booking momentum at month $t$ and $r_t$ the airline basket's return in "
            "month $t$.\n\n"
            "- **H₁ (the lead / tradable).** In $r_{t+1} = \\alpha + \\beta\\, b_t + \\varepsilon$, "
            "$\\beta > 0$ at HAC *t* ≥ 2 (this month's bookings forecast next month's return).\n"
            "- **H₂ (contemporaneous).** In $r_t = a + c\\, b_t + u$, $c > 0$ (they co-move). This is "
            "the *vendor's* chart — and it can be true while H₁ is false.\n"
            "- **H₃ (tradable net).** A sign-of-momentum timing rule beats buy-and-hold after costs "
            "and short borrow.\n"
            "- **H₄ (robustness).** The sign of H₁ is stable across sub-samples.\n\n"
            "We find **H₂ overwhelmingly true**, **H₁ rejected** (forward $\\beta$ is noise), "
            "**H₃ rejected** (timing loses to holding), and **H₄ rejected** (the forward *t* flips "
            "sign across windows). H₂-true / H₁-false *is* the already-priced signature."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The content is the **why an alt-data lead decays**. Even a signal that genuinely leads "
            "*fundamentals* need not lead *returns*: a forward-looking market (Fama 1970) prices the "
            "demand as soon as it is discernible, so a *public* booking print forecasts the earnings "
            "but not the price move. Grossman-Stiglitz (1980) sharpens it — a widely-sold feed erodes "
            "its own forward edge as it is priced in. This study is the clean instance: a booking "
            "signal with a huge contemporaneous loading and a zero forward one."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Signal.** Monthly booking momentum $b_t$ (a noisy read of a persistent latent "
            "travel-demand shock).\n"
            "- **Predictive regression (H₁).** OLS of $r_{t+1}$ on $b_t$, reporting a **Newey-West** "
            "HAC *t* (lag 3) alongside the OLS *t* — the desk's autocorrelation-robust bar.\n"
            "- **Contemporaneous regression (H₂).** OLS of $r_t$ on $b_t$.\n"
            "- **Placebo.** Circular block-shuffle (block 6) of $b$ against forward returns, 2000 "
            "perms; the fraction with |HAC *t*| ≥ observed.\n"
            "- **Trading rule (H₃).** Long/flat and long/short on $\\text{sign}(b_t)$, held one "
            "month, one-way 10 bps per rebalance, short leg pays 300 bps/yr borrow; vs buy-and-hold.\n"
            "- **Robustness (H₄).** Predictive HAC *t* over full / halves / thirds.\n"
            "- **Positive control.** A deterministic world with a planted lead (`lead_beta > 0`) and "
            "a null — averaged over 25 seeds (house rule).\n\n"
            "Timing: $b_t$ uses only data through month $t$'s close; the traded return is $r_{t+1}$. "
            "That one-month lag is the single documented execution convention (no look-ahead)."
        ),

        # ---- BEAT 4 ----
        md("## 4 · The teardown"),
        md(
            "### 4a · Predictive vs contemporaneous — H₁ vs H₂\n\n"
            "The same signal, regressed on *this*-month and *next*-month returns."
        ),
        code(
            "c = st.contemporaneous_regression(FRAME)\n"
            "p = st.predictive_regression(FRAME)\n"
            "tab = pd.DataFrame({\n"
            "  'slope': [c['slope'], p['slope']],\n"
            "  'OLS t': [c['t_ols'], p['t_ols']],\n"
            "  'HAC t': [c['t_hac'], p['t_hac']],\n"
            "  'corr':  [c['corr'], p['corr']],\n"
            "}, index=['contemporaneous (H2)', 'predictive (H1)'])\n"
            "print(tab.round(3).to_string())\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.2))\n"
            "ax.bar(['contemporaneous\\n(H2)', 'predictive\\n(H1)'], [c['t_hac'], p['t_hac']],\n"
            "       color=[GREEN, RED], width=.5)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1, label='t = 2 bar'); ax.axhline(-2, ls='--', c=GREY, lw=1)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('HAC t')\n"
            "ax.set_title(f\"Same-month HAC t +{c['t_hac']:.1f} vs forward HAC t {p['t_hac']:+.2f}\")\n"
            "ax.legend(); plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: **H₂ crushes it, H₁ is dead.** Same-month HAC *t* "
            f"**+{R['contemp_t_hac']}** (corr **+{R['contemp_corr']}**) — bookings and the airline "
            f"return are tightly linked *now*. Forward HAC *t* **{R['pred_t_hac']}** (corr "
            f"**{R['pred_corr']}**) — the booking print says nothing about next month. That split is "
            "the already-priced signature."
        ),
        md(
            "### 4b · The placebo — is the forward *t* even distinguishable from noise?\n\n"
            "Circular block-shuffle the signal against forward returns; where does the observed "
            "forward |HAC *t*| land?"
        ),
        code(
            "pp = st.placebo_pvalue(FRAME, n_perm=2000, block=6)\n"
            "print(f'predictive placebo p = {pp:.3f}  (fraction of shuffles with |HAC t| >= observed)')\n"
            "# and show the test HAS teeth: plant a real lead, re-run the placebo\n"
            "frame_pos, _ = data.synthetic_series(lead_beta=0.75, contemp_beta=0.9, seed=554)\n"
            "pp_pos = st.placebo_pvalue(frame_pos, n_perm=2000, block=6)\n"
            "t_pos = st.predictive_regression(frame_pos)['t_hac']\n"
            "print(f'planted lead (0.75): forward HAC t {t_pos:+.2f}, placebo p {pp_pos:.4f}')"
        ),
        md(
            f"> 💡 In plain words: the headline forward |*t*| sits at the **{int(R['placebo_p']*100)}th** "
            f"percentile of shuffled noise (*p* = **{R['placebo_p']}**) — squarely in the bulk. The "
            f"test is not blind: plant a real lead and it fires (*t* +{R['planted_t']}, placebo *p* "
            f"**{R['planted_placebo_p']}**). The headline world simply has no forward signal."
        ),
        md(
            "### 4c · The trading rule — H₃, gross and net\n\n"
            "Sign-of-momentum timing vs buy-and-hold, after costs and (for the short) borrow."
        ),
        code(
            "lf = st.timing_rule(FRAME, mode='long_flat', cost_bps=10.0)\n"
            "ls = st.timing_rule(FRAME, mode='long_short', cost_bps=10.0, borrow_ann_bps=300.0)\n"
            "labels = ['long/flat\\ngross','long/flat\\nnet','long/short\\nnet','buy &\\nhold']\n"
            "vals = [lf['gross_ann']*100, lf['net_ann']*100, ls['net_ann']*100, lf['bh_ann']*100]\n"
            "cols = [AMBER, RED, RED, GREEN]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.bar(labels, vals, color=cols, width=.6); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('annualised return %')\n"
            "ax.set_title('Timing on bookings loses to buy-and-hold, gross and net')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"long/flat  gross {lf['gross_ann']*100:+.1f}%  net {lf['net_ann']*100:+.1f}%  ({lf['n_trades']} trades)\")\n"
            "print(f\"long/short net {ls['net_ann']*100:+.1f}%  ({ls['n_trades']} trades, {300} bps borrow)\")\n"
            "print(f\"buy-and-hold   {lf['bh_ann']*100:+.1f}%/yr\")"
        ),
        md(
            f"> 💡 In plain words: **H₃ rejected.** Long/flat nets **{R['lf_net_ann']}%/yr** vs "
            f"buy-and-hold **+{R['lf_bh_ann']}%/yr** ({R['lf_gap']} pp); long/short bleeds "
            f"**{R['ls_net_ann']}%/yr** after borrow. Timing on a noise signal just pays costs."
        ),
        md(
            "### 4d · Robustness — H₄ (does the forward sign hold?)\n\n"
            "Predictive HAC *t* over the full sample, halves and thirds."
        ),
        code(
            "rob = st.robustness_windows(FRAME)\n"
            "print(rob.round(3).to_string())\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "cols = [GREEN if t>0 else RED for t in rob['t_hac']]\n"
            "ax.bar(range(len(rob)), rob['t_hac'], color=cols, width=.6)\n"
            "ax.set_xticks(range(len(rob))); ax.set_xticklabels(rob.index, rotation=15)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1); ax.axhline(-2, ls='--', c=GREY, lw=1)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('predictive HAC t')\n"
            "ax.set_title('Forward t wanders around zero, both signs, never near +/-2')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            "> 💡 In plain words: **H₄ rejected.** No sub-sample clears |*t*| = 2 and the sign flips "
            "between halves and thirds — a forward lead that is not present in any window."
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — H₁ rejected (forward HAC *t* {R['pred_t_hac']}, placebo *p* "
            f"{R['placebo_p']}); H₄ rejected (sign flips). Synthetic-only ⇒ `WEAK` ceiling anyway.\n"
            f"- **Tradability `MIRAGE`** — H₃ rejected; long/flat {R['lf_net_ann']}%/yr vs "
            f"buy-and-hold +{R['lf_bh_ann']}%/yr.\n"
            "- **Already in the price? `CONFIRMED`** — H₂ overwhelming (contemporaneous HAC *t* "
            f"+{R['contemp_t_hac']}) with a dead forward slope."
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you trade it? — the costs are almost a footnote\n\n"
            "There is no forward edge to charge costs against; the rule is a coin flip that then pays "
            "10 bps a rebalance and, short, 300 bps/yr borrow. The `MIRAGE` is decided *before* "
            "frictions — the frictions just deepen it (see 4c)."
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further — the synthetic positive control\n\n"
            "Is the engine a faithful detector, or would it print ~0 no matter what? Plant a genuine "
            "forward lead (`lead_beta > 0`) of increasing strength and watch the predictive HAC *t* "
            "climb — averaged over 25 seeds so no single lucky seed can fake it."
        ),
        code(
            "leads = [0.0, 0.25, 0.5, 0.75, 1.0]\n"
            "ts = [st.synthetic_mean_t(data, lead_beta=b, contemp_beta=0.9, n_seeds=25) for b in leads]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(leads, ts, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1, label='t = 2 bar')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted lead_beta (0 = null, higher = stronger forward lead)')\n"
            "ax.set_ylabel('mean predictive HAC t (25 seeds)')\n"
            "ax.set_title('The harness banks a planted lead -- flat at the null, past +2 as it grows')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for b, t in zip(leads, ts): print(f'lead_beta {b:+.2f} -> mean HAC t {t:+.2f}')"
        ),
        md(
            "The predictive HAC *t* is ≈ 0 at the null and climbs cleanly past +2 as the planted lead "
            "grows — so the engine is a faithful detector. The headline 'no forward edge' is therefore "
            "a statement about the **efficient-market world** (the realistic one), not a broken "
            "engine. For sibling alt-data leads see [257 AAII](../../257-aaii-sentiment/), "
            "[335 Buzz-ETF](../../335-buzz-sentiment-etf/) and "
            "[392 Glassdoor](../../392-glassdoor-sentiment/)."
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
