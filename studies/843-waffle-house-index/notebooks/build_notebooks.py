"""Generate the two narrative notebooks for Study 843 (Waffle House Index).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached
SPY / ALL / TRV / PGR / HD / LOW tapes under ../_cache/ and otherwise quote the frozen
headline numbers in ``R`` (mirroring docs/results.md). The synthetic positive control
runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance
# SPY/ALL/TRV/PGR/HD/LOW 2004-06-01 -> 2026-06-30; 16 hardcoded major US hurricane
# landfalls 2005-08-29 -> 2024-10-09; market-adjusted CAR over [+0..+20] sessions).
R = dict(
    n_events=16, cal_lo="2005-08-29", cal_hi="2024-10-09",
    ins_car=1.15, ins_t=1.48, ins_nw=1.63, ins_down=6, ins_down_pct=37.5, ins_wilson=(18.5, 61.4),
    reb_car=-0.28, reb_t=-0.25, reb_nw=-0.37, reb_up=8, reb_up_pct=50.0, reb_wilson=(28.0, 72.0),
    ls_spread=-1.43, ls_t=-0.88, ls_nw=-1.06, ls_ci=(-4.32, 1.62),
    plc_ins_mean=0.26, plc_ins_sd=1.03, plc_ins_p=0.385,
    plc_reb_mean=0.28, plc_reb_sd=1.31, plc_reb_p=0.662,
    plc_ls_mean=0.04, plc_ls_sd=1.67, plc_ls_p=0.379,
    # CAR path by offset [-10..+20], in percent (anchored so CAR(-10)=0)
    offsets=list(range(-10, 21)),
    ins_path=[0.0, 0.0, 0.27, 0.47, 0.83, 0.26, 0.19, 0.37, -0.36, 0.05, 0.0, 0.09, 0.02,
              0.23, 0.39, 0.22, 0.63, 0.64, 0.32, 0.65, 0.84, 1.3, 1.13, 1.55, 1.81, 2.27,
              1.4, 0.93, 1.28, 1.23, 1.2],
    reb_path=[0.0, -0.1, 0.77, 0.83, 0.96, 0.98, 1.11, 1.28, 1.57, 1.28, 1.71, 1.35, 1.32,
              1.65, 1.52, 1.34, 1.14, 1.26, 1.51, 2.06, 2.27, 1.46, 1.27, 1.35, 1.04, 0.78,
              0.81, 0.92, 1.41, 1.57, 1.0],
    ins_m5_t=-2.77,     # the one look-elsewhere daily-AR bar (offset -5)
    # robustness cuts
    tier3_n=7, tier3_ins=-0.28, tier3_ins_t=-0.36, tier3_reb=3.09, tier3_reb_t=2.02,
    tier3_ls=3.36, tier3_ls_t=1.64,
    pre15_n=5, pre15_ls=-0.03, pre15_t=-0.01, post15_n=11, post15_ls=-2.06, post15_t=-1.00,
    # timer: hold -> (gross_bps, net5_bps, t_net5, win_pct)
    timer={5: (-57.8, -78.8, -0.82, 38), 10: (-22.2, -44.2, -0.41, 44),
           20: (-153.3, -177.2, -1.22, 25)},
    # synthetic control
    syn_null_mean=-0.12, syn_null_sd=1.19, syn_null_fire=2,
    syn_planted_t=3.42, syn_planted_ins_t=-2.64, syn_planted_reb_t=1.81, syn_planted_spread=2.85,
    fp_spy="06c89e09c7e9", fp_all="3eba4d87a095", fp_trv="1d9d0f43fbd0",
    fp_pgr="9730d67197d9", fp_hd="60093d13105c", fp_low="73373520bb2e",
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Index%3F: Busted](https://img.shields.io/badge/Index%3F-Busted-8b949e?style=flat-square)\n\n"
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from waffle_index import data, strategy as st

EVENTS = data.disaster_table()
HAVE_REAL = data.have_real()
PRE, POST, LO, HI = 10, 20, 0, 20
if HAVE_REAL:
    CLOSES = data.load_real()
    SPY = CLOSES["SPY"]
    AR_INS = st.basket_ar(CLOSES, data.INSURERS, SPY)
    AR_REB = st.basket_ar(CLOSES, data.REBUILDERS, SPY)
else:
    CLOSES = SPY = AR_INS = AR_REB = None
print("real cache present:", HAVE_REAL, "| hurricanes in table:", len(EVENTS))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does the market read the Waffle House Index? 🧇🌀\n"
            "### FEMA gauges a storm by whether the always-open chain closes — but does "
            "Wall Street move the obvious stocks?\n\n"
            + BADGES +
            "There's a famous bit of disaster folklore: the **Waffle House Index**. The "
            "chain almost never closes, so FEMA uses it as a fast, on-the-ground severity "
            "gauge — *green* means full menu (mild), *yellow* means a limited menu "
            "(serious), *red* means the Waffle House actually shut its doors "
            "(catastrophic). It's real, and FEMA administrators have said so on the "
            "record.\n\n"
            "Flip it into a market question. If a storm is bad enough to close a Waffle "
            "House, it's bad enough to matter to the two most *obviously* exposed corners "
            "of the stock market:\n\n"
            "- **Property & casualty insurers** — Allstate, Travelers, Progressive — who "
            "write the checks. Their stocks should **dip**.\n"
            "- **Home-improvement / rebuild names** — Home Depot, Lowe's — who sell the "
            "plywood, generators and drywall. Their stocks should **rally**.\n\n"
            "That's the claim we test on **16 major US hurricanes since Katrina**. Do "
            "insurers drop? Do rebuilders pop? And could you have traded it?\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo and the paired "
            "long-short test? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Do insurers dip in the month after a big storm? | **No — the opposite, if "
            f"anything.** ALL/TRV/PGR drift **{R['ins_car']:+.2f}%** (market-adjusted) — "
            "*up*, not down, and statistically nothing. |\n"
            f"| Do rebuilders rally? | **No.** HD/LOW come in at **{R['reb_car']:+.2f}%** — "
            "flat, wrong sign for the claim. |\n"
            "| Does the folklore's long-short trade work? | **No — it loses.** Long "
            "rebuilders / short insurers is short the leg that drifts up and long the leg "
            "that goes nowhere; it loses money at every horizon we tried. |\n"
            "| Why? | **The market already knew the storm was coming.** A hurricane is "
            "*forecast* for days — unlike a plane crash or a bank failure, there's no "
            "surprise left to trade by landfall. |\n\n"
            "> A tidy, intuitive disaster-trade story — tested on the modern tape, and it "
            "doesn't show up."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"A major hurricane is a huge, tangible cash-flow event. Insurers are "
            "about to pay out billions — sell them. Home-improvement stores are about to "
            "sell a mountain of rebuild supplies — buy them. The Waffle House Index says "
            "the storm is real; the trade writes itself.\"*\n\n"
            "It's the most natural disaster trade a person can think of, and both legs "
            "have a genuine fundamental mechanism (real payouts, real reconstruction "
            "demand). The question isn't whether those things happen — they do — it's "
            "whether there's a **tradable abnormal move left** by the time the storm "
            "actually lands."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If real and tradable, it would be a clean, mechanical event edge: a "
            "calendar of storms, two baskets, a market-neutral book. But there's an "
            "obvious catch that makes the prior *skeptical*: hurricanes are **forecast "
            "days in advance**. An efficient market can price the expected insurer losses "
            "and the expected rebuild demand *before* landfall — leaving nothing for the "
            "post-landfall trader. So we ask three things: do insurers actually dip, do "
            "rebuilders actually rally, and does the long-short trade pay after costs?"
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The calendar.** **{R['n_events']}** major US hurricane landfalls "
            f"from {R['cal_lo']} to {R['cal_hi']} — Katrina, Rita, Wilma, Ike, Sandy, "
            "Matthew, Harvey, Irma, Florence, Michael, Laura, Ida, Ian, Idalia, Helene, "
            "Milton (public NHC/NOAA landfall dates).\n"
            "- **The comparison.** Each stock's move **minus the S&P's** same-day move "
            "(so we're measuring the *stock-specific* reaction, not \"the market went "
            "up\"), cumulated over the [−10..+20]-session window around landfall — wide "
            "on purpose, because the storm is forecast ahead.\n"
            "- **The luck check.** Draw 16 random days instead, 20,000 times — how often "
            "does a random calendar produce a move this big?\n"
            "- **The trade check.** Buy the rebuilders, short the insurers at the "
            "landfall close, hold a few weeks, pay costs, compare to zero."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the headline.** Where do the two baskets end up in the month after "
            "a major storm (market-adjusted, so the S&P's own move is stripped out)?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    ins = st.car_stats(AR_INS, EVENTS['date'], PRE, POST, LO, HI)\n"
            "    reb = st.car_stats(AR_REB, EVENTS['date'], PRE, POST, LO, HI)\n"
            "    ins_c, reb_c = ins['mean'] * 100, reb['mean'] * 100\n"
            "else:\n"
            "    ins_c, reb_c = R['ins_car'], R['reb_car']\n"
            "fig, ax = plt.subplots(figsize=(7.8, 4.4))\n"
            "bars = ax.bar(['insurers\\n(ALL/TRV/PGR)\\nclaim: DIP',\n"
            "               'rebuilders\\n(HD/LOW)\\nclaim: RALLY'],\n"
            "              [ins_c, reb_c], color=[RED, GREEN], width=.55)\n"
            "for b, v in zip(bars, [ins_c, reb_c]):\n"
            "    ax.annotate(f'{v:+.2f}%', (b.get_x() + b.get_width()/2, v), ha='center',\n"
            "                va='bottom' if v >= 0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean market-adjusted CAR, landfall → +20 sessions (%)')\n"
            "ax.set_title('Both baskets land on the WRONG side of the claim')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'insurers {ins_c:+.2f}% (claim: negative)  |  rebuilders {reb_c:+.2f}% (claim: positive)')"
        ),
        md(
            f"Insurers come in at **{R['ins_car']:+.2f}%** — *positive*, the wrong "
            f"direction (a hardening-premium / \"the uncertainty cleared\" read beats the "
            f"payout-panic read). Rebuilders are a flat **{R['reb_car']:+.2f}%**. Neither "
            "is anywhere near statistically meaningful. So much for the obvious trade.\n\n"
            "**Next, the whole path.** Maybe the action is *before* landfall (the market "
            "front-running the forecast) rather than after?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    cp_i = st.car_path_stats(AR_INS, EVENTS['date'], PRE, POST)\n"
            "    cp_r = st.car_path_stats(AR_REB, EVENTS['date'], PRE, POST)\n"
            "    ks = list(cp_i.index); ip = list(cp_i['car'] * 100); rp = list(cp_r['car'] * 100)\n"
            "else:\n"
            "    ks, ip, rp = R['offsets'], R['ins_path'], R['reb_path']\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.8))\n"
            "ax.plot(ks, ip, '-o', color=RED, ms=3, label='insurers (ALL/TRV/PGR)')\n"
            "ax.plot(ks, rp, '-o', color=GREEN, ms=3, label='rebuilders (HD/LOW)')\n"
            "ax.axvline(0, c='k', lw=.8, ls='--', label='landfall session')\n"
            "ax.axhline(0, c='k', lw=.6)\n"
            "ax.set_xlabel('trading days relative to landfall (0 = first tradable session)')\n"
            "ax.set_ylabel('mean cumulative market-adjusted return (%)')\n"
            "ax.set_title('No dip, no rally — two gently wandering lines, both above zero')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('insurer CAR at +20:', round(ip[-1], 2), '%  | rebuilder CAR at +20:', round(rp[-1], 2), '%')"
        ),
        md(
            "Neither line does what the story needs. There's no pre-landfall insurer "
            "*plunge* and no rebuilder *breakout* — just two noisy paths that both happen "
            "to drift slightly positive. The market has already digested the forecast by "
            "the time the storm arrives.\n\n"
            "**Finally, the trade.** Long rebuilders, short insurers, entered at the "
            "landfall close — does the market-neutral book make money?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    holds = [5, 10, 20]; nets = []\n"
            "    for h in holds:\n"
            "        lg = st.timer(CLOSES, EVENTS['date'], data.INSURERS, data.REBUILDERS,\n"
            "                      hold=h, cost_bps=5.0)\n"
            "        nets.append(st.summarize_trades(lg, 'ret_net')['mean_bps'])\n"
            "else:\n"
            "    holds = sorted(R['timer']); nets = [R['timer'][h][1] for h in holds]\n"
            "fig, ax = plt.subplots(figsize=(8.0, 4.4))\n"
            "b = ax.bar([f'{h}d' for h in holds], nets, color=RED, width=.5)\n"
            "for bb, v in zip(b, nets):\n"
            "    ax.annotate(f'{v:+.0f} bps', (bb.get_x()+bb.get_width()/2, v), ha='center', va='top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean net return per trade (bps, 5 bps costs)')\n"
            "ax.set_title('The long-short disaster trade loses at every horizon')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('net (bps):', dict(zip(holds, [round(v,1) for v in nets])))"
        ),
        md(
            "Every holding period loses money — because the book is short the leg "
            "(insurers) that quietly drifts *up* and long the leg (rebuilders) that goes "
            "nowhere. There's no dip to short and no rally to ride."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** Insurers **{R['ins_car']:+.2f}%**, rebuilders "
            f"**{R['reb_car']:+.2f}%** — both wrong-signed, neither significant, a random "
            "calendar reproduces them. The tidy dip/rally story isn't on the tape.\n"
            "- **Tradability — Mirage.** The long-short trade loses at every horizon, "
            "before you even get to costs.\n"
            "- **\"Does the market read the Waffle House Index?\" — Busted.** The storm is "
            "priced before it lands."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **This doesn't mean hurricanes are free for insurers.** A specific "
            "quarter's earnings absolutely take a hit — but that's *known and priced* by "
            "landfall, which is a different thing from a tradable post-event abnormal "
            "move in the stock.\n"
            "- **Where a real version might live:** a single hard-hit *regional* insurer "
            "(not a diversified national), or a *homebuilder* rather than a big-box "
            "retailer, or the reinsurance / cat-bond market where the pricing actually "
            "moves — narrower instruments than the blunt large-cap baskets here.\n"
            "- **Sibling studies:** [283-hurricane-season](../../283-hurricane-season/) "
            "(the seasonal calendar, not the event), "
            "[316-bank-failure](../../316-bank-failure/) and "
            "[707-plane-crash-effect](../../707-plane-crash-effect/) (the same "
            "event-study machinery on other disaster calendars).\n\n"
            "*Think a narrower instrument catches what a large-cap basket misses? Show it "
            "— on out-of-sample storms, after costs — then we'll talk.*"
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
            "# The Waffle House Index — a quantitative teardown 🔬\n"
            "### A market-adjusted event study · a paired rebuilders−insurers directional "
            "test · a 20-seed random-calendar placebo · the [−10..+20] anatomy and its "
            "look-elsewhere caveat · a costed long-short timer · a 20-seed synthetic null\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious]"
            "(01_for_the_curious.ipynb). The claim: a major US hurricane should *dip* "
            "listed P&C insurers (ALL/TRV/PGR, payout shock) and *rally* home-improvement "
            "names (HD/LOW, rebuild demand) — with the sceptic's prior that a **forecast** "
            "event is already priced by landfall. The job here is to measure it honestly "
            "on today's tradable tape, then ask the only question that pays: *is any of it "
            "real, and if so, tradable?*\n\n"
            "> ⚠️ **Data note.** SPY + ALL/TRV/PGR/HD/LOW total-return closes (2004→2026), "
            "yfinance, cached; **16 hardcoded major US hurricane landfalls** 2005→2024 "
            "(NHC/NOAA public dates). Survivorship named on the Signal axis (surviving "
            "large-caps, not the full P&C universe). Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (fingerprints `" + R["fp_spy"] +
            "` SPY / `" + R["fp_all"] + "` ALL / `" + R["fp_trv"] + "` TRV / `" +
            R["fp_pgr"] + "` PGR / `" + R["fp_hd"] + "` HD / `" + R["fp_low"] + "` LOW).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | insurers CAR **{R['ins_car']:+.2f}%** "
            f"(*t* = {R['ins_t']:+.2f}), rebuilders **{R['reb_car']:+.2f}%** "
            f"(*t* = {R['reb_t']:+.2f}) — both wrong sign; paired spread "
            f"**{R['ls_spread']:+.2f}%** (*t* = {R['ls_t']:+.2f}), placebo "
            f"*p* = {R['plc_ls_p']:.3f} |\n"
            f"| **Tradability** | `MIRAGE` | long-reb/short-ins loses every horizon "
            f"(5/10/20d), worst net **{R['timer'][20][1]:+.0f} bps** (*t* = "
            f"{R['timer'][20][2]:.2f}) |\n"
            f"| **Waffle House Index in prices?** | `BUSTED` | forecast event, priced "
            "pre-landfall; n=16 low power |\n\n"
            "> 💡 In plain words: two legs with genuine fundamental mechanisms, and the "
            "modern tradable test comes back empty — even *backwards* — on every axis."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r^i_t$ be name $i$'s daily return and $m_t$ SPY's; the market-adjusted "
            "abnormal return is $a^i_t = r^i_t - m_t$ (Brown & Warner 1985, market-adjusted "
            "model — an implicit $\\beta=1$, standard for short windows). For each "
            "landfall $\\tau_e$, take the equal-weight basket abnormal return "
            "$a^{ins}$, $a^{reb}$ and the per-event CAR "
            "$C^b_e = \\sum_{k=0}^{20} a^b_{\\tau_e + k}$. The claims:\n\n"
            "- **H₁ (insurer dip).** $E[C^{ins}_e] < 0$, systematic across events.\n"
            "- **H₂ (rebuilder rally).** $E[C^{reb}_e] > 0$.\n"
            "- **H₃ (directional spread).** $E[C^{reb}_e - C^{ins}_e] > 0$ — the paired, "
            "cleanest one-number test.\n"
            "- **H₄ (capture).** A long-reb/short-ins book, entered at the landfall close, "
            "beats zero net of costs.\n\n"
            "We find **H₁ rejected on sign** (+1.15%, *t* = +1.48), **H₂ rejected on "
            "sign** (−0.28%, *t* = −0.25), **H₃ rejected on sign** (−1.43%, *t* = −0.88), "
            "**H₄ not supported** (negative net at every horizon)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "Landfalls are **independent, far-apart calendar dates** (weeks to years "
            "apart), so the planned primary is a **one-sample t-test** across the 16 "
            "per-event CARs — the unit of analysis is already \"one number per event\", "
            "not a daily series, so no HAC correction is needed the way a daily-panel "
            "regression would require (a Newey-West *t* is reported anyway and "
            "auto-falls-back to a plain *t* below n=8). The hit rate carries a **Wilson** "
            "interval; the placebo draws 16 random non-disaster dates **20,000 times** "
            "(20 seeds × 1,000); the [−10..+20] anatomy is read as a **31-offset "
            "multiple-comparison** exercise; H₃ **pairs** each event's two baskets on the "
            "same window so any residual common shock cancels. The window is wide on the "
            "*pre*-landfall side because a hurricane is **forecast** — anticipation is "
            "part of the hypothesis, not a nuisance."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Calendar.** {R['n_events']} major US hurricane landfalls {R['cal_lo']} → "
            f"{R['cal_hi']}, hardcoded (NHC/NOAA). All 16 sit on the tape.\n"
            "- **Tape.** SPY + ALL/TRV/PGR/HD/LOW total-return closes, 2004 → 2026-06-30 "
            "(as-of, last complete month). All six trade the full window.\n"
            "- **Abnormal return.** Market-adjusted (name − SPY), equal-weight baskets.\n"
            "- **Headline.** Per-event CAR [+0..+20], one-sample + NW *t*, Wilson hit "
            "rate, 20-seed placebo — for insurers, rebuilders, and the paired spread.\n"
            "- **Anatomy.** CAR path [−10..+20], per-offset *t*, read with the "
            "multiple-comparison caveat.\n"
            "- **Robustness.** Catastrophic-only (loss-tier 3, n=7) and a pre-/post-2015 "
            "era split.\n"
            "- **Execution (timer).** Enter long-reb/short-ins at the landfall close "
            "(zero look-ahead — landfall is forecast and calendar-known), exit `hold` "
            "sessions later; one-way cost × NAV on all 4 legs + borrow on the short leg.\n"
            "- **Control.** Synthetic world, planted insurer-down/rebuilder-up drift; the "
            "null must not systematically fire across 20 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The headline CARs and the paired spread\n\n"
            "One-sample *t* on each basket's per-event CAR, plus the paired "
            "rebuilders−insurers spread (H₃) with an event-bootstrap CI."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ins = st.car_stats(AR_INS, EVENTS['date'], PRE, POST, LO, HI)\n"
            "    reb = st.car_stats(AR_REB, EVENTS['date'], PRE, POST, LO, HI)\n"
            "    ls = st.long_short_stats(AR_INS, AR_REB, EVENTS['date'], PRE, POST, LO, HI)\n"
            "    lo, hi = st.block_bootstrap_ci(ls['diff'])\n"
            "    ins_c, ins_t = ins['mean']*100, ins['t']\n"
            "    reb_c, reb_t = reb['mean']*100, reb['t']\n"
            "    ls_c, ls_t = ls['mean_diff']*100, ls['t']; ci = (lo*100, hi*100)\n"
            "else:\n"
            "    ins_c, ins_t = R['ins_car'], R['ins_t']\n"
            "    reb_c, reb_t = R['reb_car'], R['reb_t']\n"
            "    ls_c, ls_t = R['ls_spread'], R['ls_t']; ci = R['ls_ci']\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.5))\n"
            "labels = ['insurers\\n(claim <0)', 'rebuilders\\n(claim >0)', 'spread reb-ins\\n(claim >0)']\n"
            "vals = [ins_c, reb_c, ls_c]; cols = [RED, GREEN, AMBER]\n"
            "b = ax.bar(labels, vals, color=cols, width=.55)\n"
            "ax.errorbar(2, ls_c, yerr=[[ls_c-ci[0]], [ci[1]-ls_c]], fmt='none', ecolor='k', capsize=5)\n"
            "for bb, v, t in zip(b, vals, [ins_t, reb_t, ls_t]):\n"
            "    ax.annotate(f'{v:+.2f}%\\n(t={t:+.2f})', (bb.get_x()+bb.get_width()/2, v),\n"
            "                ha='center', va='bottom' if v>=0 else 'top', fontsize=9)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean per-event CAR [+0..+20] (%)')\n"
            "ax.set_title('Every headline number is wrong-signed for the claim')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'insurers {ins_c:+.2f}% (t={ins_t:+.2f}); rebuilders {reb_c:+.2f}% (t={reb_t:+.2f})')\n"
            "print(f'spread {ls_c:+.2f}% (t={ls_t:+.2f}), boot 95% CI [{ci[0]:+.2f}%, {ci[1]:+.2f}%]')"
        ),
        md(
            f"> 💡 In plain words: H₁ wants insurers **down** — they're "
            f"**{R['ins_car']:+.2f}%** (*t* = {R['ins_t']:+.2f}). H₂ wants rebuilders "
            f"**up** — they're **{R['reb_car']:+.2f}%** (*t* = {R['reb_t']:+.2f}). H₃ "
            f"wants the spread **positive** — it's **{R['ls_spread']:+.2f}%** "
            f"(*t* = {R['ls_t']:+.2f}), CI [{R['ls_ci'][0]:+.2f}%, {R['ls_ci'][1]:+.2f}%]. "
            "Three predictions, three wrong signs, none significant."
        ),
        md(
            "### 4b · The random-calendar placebo\n\n"
            "Draw 16 random non-disaster dates and recompute each CAR; repeat 20,000 "
            "times. If the observed number sits in the bulk, a random calendar of the "
            "same size produces it anyway. In the notebook we run a lighter placebo "
            "(4 seeds × 500) and quote the canonical 20,000-draw *p* from `results.md`."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ls = st.long_short_stats(AR_INS, AR_REB, EVENTS['date'], PRE, POST, LO, HI)\n"
            "    obs = ls['mean_diff']\n"
            "    dr_r = np.concatenate([st.placebo_distribution(AR_REB, ls['n'], PRE, POST, LO, HI,\n"
            "               n_draws=500, seed=1843+s) for s in range(4)])\n"
            "    dr_i = np.concatenate([st.placebo_distribution(AR_INS, ls['n'], PRE, POST, LO, HI,\n"
            "               n_draws=500, seed=843+s) for s in range(4)])\n"
            "    draws = dr_r - dr_i\n"
            "else:\n"
            "    obs = R['ls_spread'] / 100\n"
            "    rng = np.random.default_rng(843)\n"
            "    draws = rng.normal(R['plc_ls_mean']/100, R['plc_ls_sd']/100, 2000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws*100, bins=45, color=GREY, alpha=.85,\n"
            "        label='null: random 16-day calendars (light in-notebook run)')\n"
            "ax.axvline(obs*100, c=RED, lw=2.5, label=f'observed spread {obs*100:+.2f}%')\n"
            "ax.axvline(0, c='k', lw=.8, ls='--')\n"
            "ax.set_xlabel('mean rebuilders−insurers spread of a random 16-day calendar (%)')\n"
            "ax.set_ylabel('frequency')\n"
            "ax.set_title(f\"Squarely inside the luck cloud: canonical p = {R['plc_ls_p']:.3f} (20k draws)\")\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"canonical placebo (results.md): insurers p={R['plc_ins_p']:.3f}, \"\n"
            "      f\"rebuilders p={R['plc_reb_p']:.3f}, spread p={R['plc_ls_p']:.3f}\")"
        ),
        md(
            f"> 💡 In plain words: the observed spread **{R['ls_spread']:+.2f}%** sits "
            f"right inside the null cloud ({R['plc_ls_mean']:+.2f} ± {R['plc_ls_sd']:.2f}%); "
            f"canonical **p = {R['plc_ls_p']:.3f}**. The insurer and rebuilder CARs are "
            f"just as unremarkable (p = {R['plc_ins_p']:.3f} / {R['plc_reb_p']:.3f}). A "
            "random calendar of 16 days does all of this."
        ),
        md(
            "### 4c · Anatomy — the [−10..+20] path, and one look-elsewhere bar\n\n"
            "The full CAR path (anchored at −10). With **31 offsets** tested per basket, "
            "roughly 1–2 crossing |*t*| ≥ 2 by chance is *expected* — the one that does "
            "is the insurer **daily** abnormal return at −5 sessions (*t* ≈ −2.8), a faint "
            "pre-storm wobble with no mechanism to make it a headline."
        ),
        code(
            "if HAVE_REAL:\n"
            "    cp_i = st.car_path_stats(AR_INS, EVENTS['date'], PRE, POST)\n"
            "    cp_r = st.car_path_stats(AR_REB, EVENTS['date'], PRE, POST)\n"
            "    ks = list(cp_i.index); ip = list(cp_i['car']*100); rp = list(cp_r['car']*100)\n"
            "    it = list(cp_i['t'])\n"
            "else:\n"
            "    ks, ip, rp = R['offsets'], R['ins_path'], R['reb_path']\n"
            "    it = [0]*len(ks); it[5] = R['ins_m5_t']\n"
            "fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.6, 6.6), sharex=True,\n"
            "                             gridspec_kw={'height_ratios': [2, 1]})\n"
            "a1.plot(ks, ip, '-o', color=RED, ms=3, label='insurers')\n"
            "a1.plot(ks, rp, '-o', color=GREEN, ms=3, label='rebuilders')\n"
            "a1.axvline(0, c='k', lw=.8, ls='--'); a1.axhline(0, c='k', lw=.6)\n"
            "a1.set_ylabel('cumulative market-adj. return (%)')\n"
            "a1.set_title('Event anatomy: no dip, no rally — two wandering lines'); a1.legend()\n"
            "a2.bar([str(k) for k in ks], it,\n"
            "       color=[RED if abs(t) >= 2 else GREY for t in it], width=.7)\n"
            "a2.axhline(0, c='k', lw=.8); a2.axhline(-2, ls='--', c=RED, lw=1); a2.axhline(2, ls='--', c=RED, lw=1)\n"
            "a2.set_ylabel('insurer daily-AR t'); a2.set_xlabel('offset (sessions from landfall)')\n"
            "a2.set_xticks(range(0, len(ks), 3)); a2.set_xticklabels([ks[i] for i in range(0, len(ks), 3)])\n"
            "plt.tight_layout(); plt.show()\n"
            "print('insurer CAR +20:', round(ip[-1],2), '%  rebuilder CAR +20:', round(rp[-1],2), '%')"
        ),
        md(
            f"> 💡 In plain words: the only bar to cross the line is the insurer daily "
            f"abnormal return at **−5 sessions** (*t* = {R['ins_m5_t']:.2f}) — a whisper "
            "of a pre-storm dip as the forecast firms up, but 1-in-31 crossing by chance "
            "is exactly what you'd expect, it has no mechanism that would make it *the* "
            "result, and it's gone within days. No cumulative dip or rally survives to the "
            "+20 horizon."
        ),
        md(
            "### 4d · Robustness — the biggest storms, and two eras\n\n"
            "The one place the folklore flickers: sort to the **7 catastrophic (loss-tier "
            "3)** storms and rebuilders finally rally. Is it robust?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    big3 = EVENTS.loc[EVENTS['loss_tier'] == 3, 'date']\n"
            "    rb = st.car_stats(AR_REB, big3, PRE, POST, LO, HI)\n"
            "    lsb = st.long_short_stats(AR_INS, AR_REB, big3, PRE, POST, LO, HI)\n"
            "    pre = EVENTS.loc[EVENTS['date'] < '2015-01-01', 'date']\n"
            "    post = EVENTS.loc[EVENTS['date'] >= '2015-01-01', 'date']\n"
            "    lpre = st.long_short_stats(AR_INS, AR_REB, pre, PRE, POST, LO, HI)\n"
            "    lpost = st.long_short_stats(AR_INS, AR_REB, post, PRE, POST, LO, HI)\n"
            "    rows = [('catastrophic reb (n=%d)' % rb['n'], rb['mean']*100, rb['t']),\n"
            "            ('catastrophic spread (n=%d)' % lsb['n'], lsb['mean_diff']*100, lsb['t']),\n"
            "            ('spread pre-2015 (n=%d)' % lpre['n'], lpre['mean_diff']*100, lpre['t']),\n"
            "            ('spread 2015+ (n=%d)' % lpost['n'], lpost['mean_diff']*100, lpost['t'])]\n"
            "else:\n"
            "    rows = [(f\"catastrophic reb (n={R['tier3_n']})\", R['tier3_reb'], R['tier3_reb_t']),\n"
            "            (f\"catastrophic spread (n={R['tier3_n']})\", R['tier3_ls'], R['tier3_ls_t']),\n"
            "            (f\"spread pre-2015 (n={R['pre15_n']})\", R['pre15_ls'], R['pre15_t']),\n"
            "            (f\"spread 2015+ (n={R['post15_n']})\", R['post15_ls'], R['post15_t'])]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "labs = [r[0] for r in rows]; vals = [r[1] for r in rows]; ts = [r[2] for r in rows]\n"
            "cols = [GREEN if abs(t) >= 2 else GREY for t in ts]\n"
            "b = ax.barh(labs, vals, color=cols)\n"
            "for bb, v, t in zip(b, vals, ts):\n"
            "    ax.annotate(f'{v:+.2f}% (t={t:+.2f})', (v, bb.get_y()+bb.get_height()/2),\n"
            "                va='center', ha='left' if v>=0 else 'right', fontsize=9)\n"
            "ax.axvline(0, c='k', lw=.8); ax.set_xlabel('mean CAR / spread (%)')\n"
            "ax.set_title('One green bar (n=7), and it does not generalise'); plt.tight_layout(); plt.show()\n"
            "print('rows:', [(l, round(v,2), round(t,2)) for l, v, t in rows])"
        ),
        md(
            f"> 💡 In plain words: the **only** nominally significant number in the study "
            f"is the rebuilder rally in the 7 catastrophic storms — "
            f"**{R['tier3_reb']:+.2f}%** at *t* = {R['tier3_reb_t']:.2f}. But it's a "
            "*post-hoc subgroup* (storms sorted by realised loss), it clears the bar on a "
            f"hand-picked seven, it doesn't drag the paired spread past significance "
            f"(*t* = {R['tier3_ls_t']:.2f}), and the full-sample spread is wrong-signed and "
            f"driven by the recent era (2015+ spread {R['post15_ls']:+.2f}%). It fails the "
            "desk's \"robust **and** holds across sub-eras\" bar — a grain of truth for "
            "truly catastrophic storms, not a certifiable edge."
        ),
        md(
            "### 4e · The timer — an honest long-short cost sweep\n\n"
            "Long rebuilders / short insurers, dollar-neutral, entered at the landfall "
            "close (zero look-ahead — the storm is forecast and calendar-known), held `h` "
            "sessions; one-way cost × NAV on all 4 legs + 50 bps/yr borrow on the short "
            "insurer leg."
        ),
        code(
            "if HAVE_REAL:\n"
            "    holds = [5, 10, 20]; gross, net5, ts = [], [], []\n"
            "    for h in holds:\n"
            "        g = st.summarize_trades(st.timer(CLOSES, EVENTS['date'], data.INSURERS,\n"
            "              data.REBUILDERS, hold=h, cost_bps=0.0), 'ret_gross')\n"
            "        n5 = st.summarize_trades(st.timer(CLOSES, EVENTS['date'], data.INSURERS,\n"
            "              data.REBUILDERS, hold=h, cost_bps=5.0), 'ret_net')\n"
            "        gross.append(g['mean_bps']); net5.append(n5['mean_bps']); ts.append(n5['t'])\n"
            "else:\n"
            "    holds = sorted(R['timer'])\n"
            "    gross = [R['timer'][h][0] for h in holds]\n"
            "    net5 = [R['timer'][h][1] for h in holds]\n"
            "    ts = [R['timer'][h][2] for h in holds]\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.5))\n"
            "x = np.arange(len(holds)); w = 0.38\n"
            "ax.bar(x - w/2, gross, width=w, color=GREY, label='gross')\n"
            "ax.bar(x + w/2, net5, width=w, color=RED, label='net (5 bps)')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in holds])\n"
            "ax.set_ylabel('mean return per trade (bps)')\n"
            "ax.set_title('Long-reb/short-ins loses at every horizon, gross and net')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('gross (bps):', dict(zip(holds, [round(v,1) for v in gross])))\n"
            "print('net 5bps (bps):', dict(zip(holds, [round(v,1) for v in net5])))\n"
            "print('t (net):', dict(zip(holds, [round(t,2) for t in ts])))"
        ),
        md(
            f"> 💡 In plain words: at 5 bps costs the trade nets "
            f"**{R['timer'][5][1]:+.0f} / {R['timer'][10][1]:+.0f} / "
            f"{R['timer'][20][1]:+.0f} bps** at 5/10/20-day holds — negative everywhere "
            f"(worst *t* = {R['timer'][20][2]:.2f}), win rate ≤ {max(v[3] for v in R['timer'].values())}%. "
            "H₄ is not supported: the book is short the leg that drifts up and long the "
            "leg that goes nowhere. No edge to charge costs against."
        ),
        md(
            "### 4f · Faithful-engine & power control — we know the truth here\n\n"
            "Synthetic world: an SPY-like market + 5 single names + a TUNABLE planted "
            "market-adjusted drift (insurers −`edge`/day, rebuilders +`edge`/day) decaying "
            "over 15 sessions. The null (edge=0) is checked over **20 seeds**."
        ),
        code(
            "null_ts = []\n"
            "for s_ in range(20):\n"
            "    cl, ev = data.synthetic_world(edge=0.0, seed=843 + s_)\n"
            "    null_ts.append(st.synthetic_detect(cl, ev, data.INSURERS, data.REBUILDERS)['ls_t'])\n"
            "null_ts = np.asarray(null_ts)\n"
            "cl, ev = data.synthetic_world(edge=0.0015, seed=843)\n"
            "planted = st.synthetic_detect(cl, ev, data.INSURERS, data.REBUILDERS)\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.linspace(-.12, .12, 20), null_ts, color=GREY, s=40,\n"
            "           label='null worlds (edge=0), 20 seeds')\n"
            "ax.scatter([1], [planted['ls_t']], color=GREEN, s=90, zorder=5,\n"
            "           label='planted edge = +0.15%/day')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.axhline(0, c='k', lw=.6)\n"
            "ax.set_xticks([0, 1]); ax.set_xticklabels(['null x 20', 'planted'])\n"
            "ax.set_ylabel('directional (reb-ins) one-sample t')\n"
            "ax.set_title('Control: null centered at zero; a planted drift lights up')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(np.abs(null_ts) >= 2).sum()}/20 seeds  |  planted t = {planted[\"ls_t\"]:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 20 null worlds the detector averages "
            f"t = {R['syn_null_mean']:+.2f} (sd {R['syn_null_sd']:.2f}); the "
            f"{R['syn_null_fire']}/20 that cross the bar are the ~5% nominal false-positive "
            "rate at n=16, **not** a bias (it's centered at zero). A planted "
            f"+0.15%/day drift reads t = {R['syn_planted_t']:.2f} (insurers "
            f"t = {R['syn_planted_ins_t']:.2f}, rebuilders t = {R['syn_planted_reb_t']:.2f}). "
            "The machinery is unbiased — the real-tape nulls are genuine. *(A "
            "faithful-engine / power check only — never cited in support of the real-tape "
            "stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — insurers **{R['ins_car']:+.2f}%** (*t* = "
            f"{R['ins_t']:+.2f}), rebuilders **{R['reb_car']:+.2f}%** (*t* = "
            f"{R['reb_t']:+.2f}), paired spread **{R['ls_spread']:+.2f}%** (*t* = "
            f"{R['ls_t']:+.2f}) — three wrong signs, none significant, placebo *p* = "
            f"{R['plc_ls_p']:.3f}. The lone significant cut (tier-3 rebuilder rally, *t* = "
            f"{R['tier3_reb_t']:.2f}, n={R['tier3_n']}) is a post-hoc subgroup failing the "
            "sub-era bar.\n"
            f"- **Tradability `MIRAGE`** — long-reb/short-ins loses at every horizon "
            f"(5/10/20d), gross and net (worst {R['timer'][20][1]:+.0f} bps, *t* = "
            f"{R['timer'][20][2]:.2f}). No edge to charge costs against.\n"
            "- **\"Does the market read the Waffle House Index?\" `BUSTED`** — a "
            "*forecast* disaster is priced before landfall, the post-event reaction is "
            "flat-to-backwards in the obvious large-caps, and n=16 is low power. A real, "
            "smaller, more localized effect (a regional insurer, a homebuilder, the "
            "reinsurance/cat-bond market) is not ruled out at this size."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **Power is the honest limitation.** n=16 independent events is small for a "
            "modest hypothesized effect that anticipation has mostly eaten. A "
            "pre-registered extension (more storms, or a broader disaster register — "
            "wildfires, floods) with the same discipline is the natural next step.\n"
            "- **Narrower instruments** — a single hard-hit regional insurer, a "
            "homebuilder rather than a big-box retailer, or the reinsurance / cat-bond "
            "market where risk actually re-prices — might carry a signal a diversified "
            "large-cap basket cannot.\n"
            "- **Dedup map:** [283-hurricane-season](../../283-hurricane-season/) (the "
            "seasonal calendar, not the event), [316-bank-failure](../../316-bank-failure/) "
            "(same machinery, a *surprise* shock), "
            "[313-geopolitical-shock](../../313-geopolitical-shock/) (war/terror) and "
            "[707-plane-crash-effect](../../707-plane-crash-effect/) (the closest cousin — "
            "a disaster calendar and a sector extra-drop). Same event-study rails, "
            "different trigger and exposed sectors.\n\n"
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
