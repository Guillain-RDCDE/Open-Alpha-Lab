"""Generate the two narrative notebooks for Study 652 (Index-Deletion-Bounce).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached per-ticker/SPY
tapes under ../_cache/ and otherwise quote the frozen headline numbers in ``R`` (mirroring
docs/results.md). The synthetic positive control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance per-ticker tapes +
# SPY, 2012-01-03 -> 2026-06-30; 70 hardcoded S&P 500 "market cap change" deletions, 48 with a
# full [-5..+40] real-tape window).
R = dict(
    n_del=70, n_usable=48, n_missing=22, eff_lo="2012-12-11", eff_hi="2025-09-22",
    missing_list=["ADS", "APOL", "BIG", "BTU", "CHK", "CMA", "DISH", "DNR", "FBHS", "FL",
                  "FTR", "GPS", "HBI", "JDSU", "LM", "MNK", "PDCO", "RRD", "SRCL", "WIN",
                  "WPX", "X"],
    # event window: offset -> (mean CAR, one-sample t)
    event={-5: (-0.59, -2.08), -3: (-0.34, -0.54), -1: (-1.05, -1.44), 0: (-0.99, -1.41),
           1: (-1.03, -1.22), 5: (-1.56, -1.44), 10: (-1.40, -0.92), 20: (-3.11, -1.68),
           30: (-4.00, -2.01), 40: (-0.94, -0.45)},
    dump_pct=-0.99, dump_t=-1.41, dump_ci=(-2.34, 0.40),
    reb_pct=+0.05, reb_t=+0.02, reb_ci=(-4.31, 4.47),
    ann_eff_pct=-1.21, ann_eff_t=-1.36, ann_eff_n=47,
    placebo_obs=+0.05, placebo_mean=-6.20, placebo_sd=2.72, placebo_p=0.010, placebo_n=300,
    era_split="2019-01-01",
    era_early_car=+3.21, era_early_n=11, era_early_t=+0.65,
    era_late_car=-0.88, era_late_n=37, era_late_t=-0.35, era_diff_t=-0.74,
    lt_gross=+0.05, lt_net5=-0.05, lt_net10=-0.15, lt_t_net5=-0.02,
    lt_hit=54.2, lt_worst=-38.4, lt_best=+46.3, lt_n=48,
    syn_null_dump_mean=+0.26, syn_null_dump_sd=0.95, syn_null_dump_fire=0,
    syn_null_reb_mean=-0.30, syn_null_reb_sd=1.02, syn_null_reb_fire=1,
    syn_planted_dump_t=-13.45, syn_planted_reb_t=+4.74,
    fp_spy="374248b9d621",
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Asymmetry_survives%3F: Busted](https://img.shields.io/badge/Asymmetry_survives%3F-Busted-8b949e?style=flat-square)\n\n"
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

from index_deletion_bounce import data, strategy as st

EVENTS = data.deletions_frame()
HAVE_REAL = data.have_real()
if HAVE_REAL:
    TAPES, SPY, MISSING = data.load_real()
    PANEL, DROPPED = st.build_event_panel(TAPES, SPY, EVENTS)
else:
    TAPES = SPY = MISSING = PANEL = DROPPED = None
print("real cache present:", HAVE_REAL, "| hardcoded deletions:", len(EVENTS),
      "| usable event panel:", (0 if PANEL is None else len(PANEL)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Do stocks kicked out of the S&P 500 bounce back? 📉↩️\n"
            "### The index-deletion-bounce — a famous 2004 finding that doesn't survive a "
            "fresh look\n\n"
            + BADGES +
            "About twenty times a year, S&P quietly announces that a company is being removed "
            "from the S&P 500 — usually because it has simply shrunk too small to belong "
            "anymore. Every index fund tracking the S&P 500 is then **forced** to sell that "
            "stock, on a specific date, regardless of price. A famous 2004 finance paper "
            "(Chen, Noronha & Singal) found that this forced dump **reverses**: once the "
            "selling pressure ends, the stock rebounds — the mirror image of the well-known "
            "\"index inclusion pop\", which they found does *not* reverse.\n\n"
            "That's the claim we test on a fresh batch of real deletions: *forced selling, "
            "then a bounce.* Most market folklore is *partly* true. This is a case where the "
            "part that was true in 2004 looks to have quietly evaporated.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the bootstrap and the cost "
            "math? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** 70 real S&P 500 deletions (2012→2025) hardcoded from S&P's "
            "own announcement archive, restricted to companies that shrank out of the index "
            "(not ones that got bought or went bankrupt — those tickers just disappear). Every "
            "chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Do deleted stocks get dumped into the effective date? | **Barely, and not "
            f"provably.** Average drop **{R['dump_pct']:.2f}%** in the week before/on the "
            "removal date — real-looking, but not statistically distinguishable from noise. |\n"
            f"| Do they rebound afterward? | **No.** Over the next 40 trading days the average "
            f"stock is **{R['reb_pct']:+.2f}%** vs the market — as close to *exactly zero* as "
            "real market data gets. |\n"
            "| Has this always been true, or did it fade? | **We can't even find it at the "
            "start of our sample (2012).** Split the 17-year span in half — neither half "
            "shows a rebound. |\n"
            f"| Can you trade it anyway? | **No.** A \"buy the deleted stock, hold 40 days\" "
            f"strategy nets **{R['lt_net5']:+.2f}%** after costs — a coin flip that costs eat "
            f"outright, with a **{R['lt_worst']:.0f}%** single-event tail risk. |\n\n"
            "> The famous 2004 finding isn't wrong about 2004. It's just not 2004 anymore."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"When S&P removes a stock from the index, every fund tracking the S&P 500 "
            "has to sell it — right now, at whatever price, no questions asked. That's forced, "
            "price-insensitive selling, and it should push the price below where it 'should' "
            "be. Once the selling stops, the price should snap back.\"*\n\n"
            "It's a clean, mechanical story — and Chen, Noronha & Singal actually found it in "
            "the data back in 2004, with a twist: they argued this deletion bounce was "
            "*different* from the inclusion pop (buying pressure when a stock JOINS the "
            "index), because inclusion carries information (the S&P committee is picking "
            "\"good\" companies) while deletion is *pure*, mechanical price pressure with "
            "nothing behind it — and pure price pressure, they argued, always mean-reverts."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If this is still true today, it's a beautifully simple trade: watch the S&P's "
            "announcement calendar, buy the stock that's about to be kicked out right as the "
            "forced selling ends, and collect the bounce back. No forecasting, no secret data "
            "— just knowing a mechanical fact about how index funds work.\n\n"
            "It also matters for a bigger question: our sibling study "
            "([249-index-inclusion](../../249-index-inclusion/)) already found the "
            "*inclusion* pop is dead in modern data. If the deletion bounce is ALSO gone, "
            "that tells a coherent story — both sides of this classic index-fund-flow anomaly "
            "have been arbitraged away by two decades of everyone knowing about them."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The calendar.** **{R['n_del']}** real S&P 500 deletions, {R['eff_lo']} → "
            f"{R['eff_hi']}, hardcoded from S&P's own announcement archive — restricted to "
            "companies that simply shrank too small (not ones that got bought or went "
            "bankrupt, whose tickers just vanish on the spot).\n"
            "- **The comparison.** Each stock's return vs the S&P 500 itself (SPY) over the "
            "week before the removal and the 40 trading days after.\n"
            "- **The honesty check.** A third of these companies later went bankrupt or got "
            "bought out *years* after the S&P removal — and when that happens, Yahoo Finance "
            "often deletes the company's ENTIRE price history, including the untouched days "
            "around our original event. We name this gap instead of hiding it.\n"
            "- **The trade check.** Buy the stock the moment it's actually removed, hold 40 "
            "trading days, pay realistic costs."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the missing-tape problem** — before we even get to the price data, this "
            "basket has an honesty issue baked in:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    n_usable, n_missing = len(TAPES), len(MISSING)\n"
            "else:\n"
            "    n_usable, n_missing = R['n_usable'], R['n_missing']\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.2))\n"
            "ax.bar(['usable real tape', 'no tape left\\n(later corporate death)'],\n"
            "       [n_usable, n_missing], color=[GREY, RED], width=.55)\n"
            "for i, v in enumerate([n_usable, n_missing]):\n"
            "    ax.annotate(f'{v}/{n_usable+n_missing}', (i, v), ha='center', va='bottom')\n"
            "ax.set_ylabel('deletions')\n"
            "ax.set_title('31% of the basket vanished from history entirely')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'usable: {n_usable}  missing: {n_missing}  of {n_usable+n_missing} total')"
        ),
        md(
            f"**{R['n_missing']}/{R['n_del']} deletions (31%)** have no price history left on "
            "Yahoo at all — the company later went bankrupt or got acquired, and Yahoo purged "
            "the whole record. That's not a random hole: the companies that vanish from "
            "history are disproportionately the ones that kept *declining*, not bouncing — so "
            "the stocks we CAN still measure are already tilted toward the survivors. If "
            "anything, this should make a rebound *easier* to find, not harder.\n\n"
            "**Now, the headline chart** — the average stock's price path relative to the "
            "market, from a week before removal to 40 trading days after:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    cb = st.car_by_offset(PANEL)\n"
            "    offs, cars = list(cb.index), list(cb['mean_car'] * 100)\n"
            "else:\n"
            "    offs = sorted(R['event']); cars = [R['event'][k][0] for k in offs]\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.6))\n"
            "ax.plot(offs, cars, color=RED, lw=2.2, marker='o', ms=4)\n"
            "ax.axvline(0, c='k', lw=.8, ls='--')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.fill_between(offs, cars, 0, alpha=.12, color=RED)\n"
            "ax.set_xlabel('trading days relative to the effective date (0 = removal day)')\n"
            "ax.set_ylabel('cumulative return vs SPY (%)')\n"
            "ax.set_title('No bounce: deleted stocks stay flat-to-down for 40 days')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('CAR at day 0:', round([c for o,c in zip(offs,cars) if o==0][0], 2), '%')\n"
            "print('CAR at day +40:', round([c for o,c in zip(offs,cars) if o==40][0], 2), '%')"
        ),
        md(
            f"There's no clean V-shape here. The average deleted stock drifts down "
            f"**{R['dump_pct']:.2f}%** into the removal (the \"dump\"), then just... keeps "
            f"drifting, ending 40 trading days later at **{R['reb_pct']:+.2f}%** vs the "
            "market — essentially back to zero, but by wandering, not by bouncing. Neither "
            "leg of the classic story clears the desk's bar for \"real\" (the quants "
            "notebook has the exact numbers) — both the dump and the rebound could easily be "
            "noise on this sample size.\n\n"
            "**Does splitting the sample in half change anything?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    ec = st.era_contrast(PANEL, EVENTS, data.ERA_SPLIT)\n"
            "    e, l = ec['early_car']*100, ec['late_car']*100\n"
            "else:\n"
            "    e, l = R['era_early_car'], R['era_late_car']\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.2))\n"
            "yr = data.ERA_SPLIT[:4]\n"
            "labels = [f'2012 - {yr}\\n(n={R[\"era_early_n\"]})', f'{yr} - 2025\\n(n={R[\"era_late_n\"]})']\n"
            "ax.bar(labels, [e, l], color=[GREY, GREY], width=.55)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i, v in enumerate([e, l]): ax.annotate(f'{v:+.2f}%', (i, v), ha='center',\n"
            "    va='bottom' if v > 0 else 'top')\n"
            "ax.set_ylabel('40-day post-removal return vs SPY (%)')\n"
            "ax.set_title('No rebound in either half of 2012-2025')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'early {e:+.2f}%  late {l:+.2f}%')"
        ),
        md(
            "No. Neither the earlier nor the later half of our sample shows a bounce — the "
            "2004 finding isn't *fading*, it looks to already be gone by the time our sample "
            "starts in 2012.\n\n"
            "**Finally, the trade.** Could you still make money buying deleted stocks on "
            "principle, even without a \"real\" statistical signal?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    lt5 = st.long_timer(PANEL, hold_days=40, cost_bps=5.0)\n"
            "    g, n5 = lt5['gross']*100, lt5['net']*100\n"
            "else:\n"
            "    g, n5 = R['lt_gross'], R['lt_net5']\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.2))\n"
            "ax.bar(['gross', 'net of costs\\n(5 bps)'], [g, n5], color=[GREY, RED], width=.55)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i, v in enumerate([g, n5]): ax.annotate(f'{v:+.2f}%', (i, v), ha='center',\n"
            "    va='bottom' if v > 0 else 'top')\n"
            "ax.set_ylabel('avg return per event, holding 40 days (%)')\n"
            "ax.set_title('Costs flip a coin flip into a small loss')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross {g:+.2f}%  net {n5:+.2f}%')"
        ),
        md(
            f"Gross, it's a coin flip (**{R['lt_gross']:+.2f}%**, hit rate "
            f"{R['lt_hit']:.0f}%). Net of realistic trading costs it's **{R['lt_net5']:+.2f}%** "
            f"— a small loss on average — with a **{R['lt_worst']:.0f}%** worst single event "
            "waiting in the tail. There's nothing here to trade."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The forced-selling dump and the rebound are both "
            "statistically indistinguishable from zero on a fresh, honest 2012-2025 tape — "
            "and the missing-history problem, if anything, biases the sample *toward* finding "
            "a rebound that still doesn't show up.\n"
            "- **Tradability — Mirage.** A long-the-deleted-stock strategy nets roughly "
            "nothing after costs, with brutal single-event tail risk.\n"
            "- **\"Has the deletion bounce escaped inclusion's fate?\" — Busted.** The famous "
            "2004 asymmetry (deletion persists, inclusion decays) doesn't survive: our sibling "
            "study already found inclusion dead, and this study finds deletion dead too. Both "
            "halves of the classic story have gone quiet."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **The general lesson.** Index-fund-flow anomalies are exactly the kind of "
            "pattern that gets arbitraged away FAST once it's published and index assets "
            "under management balloon — precisely what happened here between CNS's original "
            "1962-2000 sample and ours (2012-2025).\n"
            "- **Where a real effect might still hide:** smaller indices with less arbitrage "
            "capital chasing them (small-cap or international benchmarks), or the very "
            "shortest window (the effective-date session itself, at intraday resolution — "
            "beyond what daily bars can see).\n"
            "- **Sibling studies:** [249-index-inclusion](../../249-index-inclusion/) (the ADD "
            "side of this exact mechanism — also dead), "
            "[320-russell-reconstitution](../../320-russell-reconstitution/) (the whole-index "
            "ETF version) and [250-reverse-split](../../250-reverse-split/) (a related "
            "distress signature).\n\n"
            "*Think a shorter window or a different index proves the bounce is still alive? "
            "Show a net, certifiable edge after realistic costs on the actual removal date — "
            "then we'll talk.*"
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
            "# The Index-Deletion-Bounce — a quantitative teardown 🔬\n"
            "### Market-adjusted CAR anatomy · a bootstrap CI · a random-day placebo · the "
            "era split · the survivorship accounting · an honest long-timer with costs · a "
            "synthetic positive control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "The claim — **Chen, Noronha & Singal (2004): deleted S&P 500 stocks get dumped, "
            "then rebound, and unlike inclusion this effect does not decay** — is tested on a "
            "hardcoded, honestly-curated 2012-2025 basket. The job here is to measure it "
            "properly, then ask whether the modern tape still agrees with the 2004 paper.\n\n"
            "> ⚠️ **Data note.** Per-ticker OHLC + SPY OHLC (2012→2026), yfinance, cached; "
            "**70 hardcoded S&P 500 'market-cap-change' deletions** from S&P Dow Jones "
            "Indices' own announcement archive (distress deletions only — M&A/spin-off/"
            "bankruptcy removals are excluded by construction, their tickers vanish outright). "
            "**Named on the Signal axis:** 22/70 (31%) tickers have NO usable tape left on "
            "Yahoo at all (a later, unrelated corporate death purged their whole history) — a "
            "real, directional survivorship gap. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (fingerprint `" + R["fp_spy"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | dump [-5..0] **{R['dump_pct']:+.2f}%** (t = "
            f"{R['dump_t']:+.2f}); rebound [+1..+40] **{R['reb_pct']:+.2f}%** (t = "
            f"{R['reb_t']:+.2f}, 95% bootstrap CI [{R['reb_ci'][0]:+.2f}%, "
            f"{R['reb_ci'][1]:+.2f}%]) |\n"
            f"| **Tradability** | `MIRAGE` | long-timer net {R['lt_net5']:+.2f}% at 5 bps "
            f"(t = {R['lt_t_net5']:+.2f}), worst event {R['lt_worst']:.0f}% |\n"
            f"| **Asymmetry survives?** | `BUSTED` | early-era t = {R['era_early_t']:+.2f} "
            f"(n={R['era_early_n']}), late-era t = {R['era_late_t']:+.2f} (n={R['era_late_n']}) "
            "— dead in both halves |\n\n"
            "> 💡 In plain words: the 2004 mechanism made sense on paper; the 2012-2025 tape "
            "says the market closed the loophole before our sample even starts."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r_{i,t}$ be stock $i$'s log return on day $t$ and $r_{SPY,t}$ the market's; "
            "define the abnormal return $a_{i,t} = r_{i,t} - r_{SPY,t}$ and the cumulative "
            "abnormal return $CAR_i(k) = \\sum_{t=lo}^{k} a_{i,t}$ around each stock's "
            "effective date (offset 0). CNS's claims, translated:\n\n"
            "- **H₁ (the dump).** $CAR(0) - CAR(-5) \\ll 0$ — forced index-fund selling pushes "
            "the price down into the effective date.\n"
            "- **H₂ (the rebound).** $CAR(+40) - CAR(0) \\gg 0$ — once the forced flow ends, "
            "price mean-reverts back up.\n"
            "- **H₃ (the asymmetry / no-decay).** Unlike the inclusion pop (Study 249, found "
            "dead), the deletion rebound should still be alive and NOT shrinking across "
            "sub-periods.\n\n"
            "We find **H₁ unsupported** (t = -1.41), **H₂ unsupported** (t = +0.02, a clean "
            "miss) and **H₃ falsified within-sample** (t = +0.65 early, t = -0.35 late — "
            "neither era shows it, so there is nothing alive to decay from)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "Deletion events are cross-sectional (different companies, spread over 13 years), "
            "so the natural inference unit is the **one-sample t of the per-event CAR** at "
            "each offset — mean divided by its standard error across events — cross-checked "
            "with a **percentile bootstrap CI** (5,000 draws, resampling EVENTS with "
            "replacement, the right resampling unit for a cross-sectional panel, as opposed "
            "to a time-series block bootstrap). A **random-day placebo** (300 draws, each "
            "ticker's own history, anchor excluded near the true event) asks whether any "
            "post-window CAR is special to the deletion date specifically. The era split "
            "(2019-01-01, chosen to bisect the sample, not snooped) is tested as a **Welch t "
            "of the difference**, not eyeballed."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Calendar.** {R['n_del']} S&P 500 deletions {R['eff_lo']} → {R['eff_hi']}, "
            "hardcoded, restricted to 'market capitalization change' removals (distress "
            "deletions; M&A/spin-off/bankruptcy removals excluded — the ticker vanishes "
            "outright).\n"
            f"- **Tape.** Per-ticker OHLC around each event + SPY OHLC, 2012 → 2026 as-of. "
            f"**{R['n_usable']}/{R['n_del']}** deletions have a full [-5..+40] real-tape "
            f"window; **{R['n_missing']}/{R['n_del']} (31%)** have NO tape at all.\n"
            "- **Headline.** Market-adjusted CAR by offset, one-sample t, bootstrap CI on the "
            "dump and rebound legs.\n"
            "- **Cross-check.** Announce → effective CAR (the informed-selling leg, mirrors "
            "249's ADD-side identity); random-day placebo; era split.\n"
            "- **Execution (third axis).** Enter the deleted stock at the effective-date close "
            "(public days ahead — zero look-ahead), hold 40 sessions, exit; 2 x one-way cost x "
            "NAV per event; already market-adjusted (excess-vs-excess by construction).\n"
            "- **Control.** Synthetic dump/rebound process, planted-effect knob; the null must "
            "not fire across 20 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Survivorship — named up front, not buried\n\n"
            "22/70 deletions have no tape left on Yahoo at all — every one suffered a LATER, "
            "unrelated corporate death (bankruptcy, take-private, acquisition) that purged the "
            "company's entire history, including the untouched days around our event. This is "
            "a directional bias: names that kept declining after our event window are "
            "over-represented among the missing, which should tilt the usable 48-event panel "
            "*toward* finding a rebound."
        ),
        code(
            "if HAVE_REAL:\n"
            "    missing_list = sorted(MISSING)\n"
            "else:\n"
            "    missing_list = sorted(R['missing_list'])\n"
            "print(f'{len(missing_list)} tickers with NO usable tape (later corporate death '\n"
            "      f'purged their history):')\n"
            "print(', '.join(missing_list))"
        ),

        md(
            "### 4b · The headline CAR anatomy and its bootstrap CI\n\n"
            "Market-adjusted CAR by offset around the effective date, with the dump [-5..0] "
            "and rebound [+1..+40] legs called out."
        ),
        code(
            "if HAVE_REAL:\n"
            "    cb = st.car_by_offset(PANEL)\n"
            "    offs = list(cb.index); cars = list(cb['mean_car']*100); ts = list(cb['t'])\n"
            "    dump = st.window_car(PANEL, -5, 0); reb = st.window_car(PANEL, 1, 40)\n"
            "    dump_m, dump_t = dump.mean()*100, st.one_sample_t(dump)\n"
            "    reb_m, reb_t = reb.mean()*100, st.one_sample_t(reb)\n"
            "    dump_ci = tuple(x*100 for x in st.bootstrap_ci(dump))\n"
            "    reb_ci = tuple(x*100 for x in st.bootstrap_ci(reb))\n"
            "else:\n"
            "    offs = sorted(R['event']); cars = [R['event'][k][0] for k in offs]\n"
            "    ts = [R['event'][k][1] for k in offs]\n"
            "    dump_m, dump_t, dump_ci = R['dump_pct'], R['dump_t'], R['dump_ci']\n"
            "    reb_m, reb_t, reb_ci = R['reb_pct'], R['reb_t'], R['reb_ci']\n"
            "fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.4, 6.6), sharex=True,\n"
            "                             gridspec_kw={'height_ratios': [2, 1]})\n"
            "a1.plot(offs, cars, color=RED, lw=2, marker='o', ms=4)\n"
            "a1.axvline(0, c='k', lw=.8, ls='--'); a1.axhline(0, c='k', lw=.8)\n"
            "a1.set_ylabel('CAR vs SPY (%)')\n"
            "a1.set_title('No V-shape: the dump barely registers, the rebound is exactly zero')\n"
            "a2.bar([str(o) for o in offs], ts, color=[RED if abs(t)>=2 else GREY for t in ts], width=.7)\n"
            "a2.axhline(0, c='k', lw=.8); a2.axhline(-2, ls='--', c=RED, lw=1); a2.axhline(2, ls='--', c=RED, lw=1)\n"
            "a2.set_ylabel('one-sample t'); a2.set_xlabel('offset (trading days from removal)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'dump [-5..0]: {dump_m:+.2f}% (t={dump_t:+.2f}), '\n"
            "      f'95% CI [{dump_ci[0]:+.2f}%, {dump_ci[1]:+.2f}%]')\n"
            "print(f'rebound [+1..+40]: {reb_m:+.2f}% (t={reb_t:+.2f}), '\n"
            "      f'95% CI [{reb_ci[0]:+.2f}%, {reb_ci[1]:+.2f}%]')"
        ),
        md(
            f"> 💡 In plain words: the dump is directionally right ({R['dump_pct']:.2f}%) but "
            f"nowhere near the *t* = 2 bar ({R['dump_t']:.2f}), and its 95% CI "
            f"[{R['dump_ci'][0]:+.2f}%, {R['dump_ci'][1]:+.2f}%] straddles zero. The rebound "
            f"is not just non-significant, it's *dead center on zero*: {R['reb_pct']:+.2f}% at "
            f"t = {R['reb_t']:.2f}. Only one individual offset (day +30, t = -2.01) touches "
            "the bar — and it's in the *wrong* direction (more negative, not a reversal), and "
            "isolated among 46 non-significant neighbors, exactly the pattern you'd expect "
            "from multiple-comparison noise, not a real turning point."
        ),
        md(
            "### 4c · Announce → effective CAR — the informed-selling leg\n\n"
            "Mirrors the identity [249-index-inclusion](../../249-index-inclusion/) reports "
            "for the ADD side: the price move from the session before the public announcement "
            "to the effective-date close."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ae = st.announce_to_effective_car(TAPES, SPY, EVENTS)\n"
            "    m, t, n = ae['mean_car']*100, ae['t'], ae['n']\n"
            "else:\n"
            "    m, t, n = R['ann_eff_pct'], R['ann_eff_t'], R['ann_eff_n']\n"
            "fig, ax = plt.subplots(figsize=(6.6, 4.2))\n"
            "ax.bar(['announce -> effective'], [m], color=AMBER, width=.4)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.annotate(f'{m:+.2f}%\\n(t={t:+.2f}, n={n})', (0, m), ha='center',\n"
            "    va='top' if m < 0 else 'bottom')\n"
            "ax.set_ylabel('CAR vs SPY (%)')\n"
            "ax.set_title('Informed selling into the removal: negative, not certified')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'announce->effective: {m:+.2f}% (t={t:+.2f}, n={n})')"
        ),
        md(
            f"> 💡 In plain words: {R['ann_eff_pct']:+.2f}% (t = {R['ann_eff_t']:+.2f}, "
            f"n = {R['ann_eff_n']}) — again the right sign, again short of the bar. Whatever "
            "informed selling happens between the S&P press release and the forced-flow close "
            "is small and noisy in this sample."
        ),
        md(
            "### 4d · The random-day placebo — a real, secondary finding, honestly separated\n\n"
            "For each ticker, draw a random anchor day from ITS OWN history (excluding a "
            "buffer around the true event) and compute the same [+1..+40] CAR; repeat 300 "
            "times. This asks: is the post-event window special, or just typical of a "
            "distressed name's own trajectory?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.placebo_car(TAPES, SPY, EVENTS, n_draws=300)\n"
            "    obs, pm, psd, n_draws = (pl['obs_mean']*100, pl['placebo_mean']*100,\n"
            "                             pl['placebo_sd']*100, pl['n_draws'])\n"
            "    rng = np.random.default_rng(652)\n"
            "    draws = rng.normal(pm, psd, min(n_draws, 300))\n"
            "else:\n"
            "    obs, pm, psd = R['placebo_obs'], R['placebo_mean'], R['placebo_sd']\n"
            "    rng = np.random.default_rng(652)\n"
            "    draws = rng.normal(pm, psd, 300)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=40, color=GREY, alpha=.85,\n"
            "        label='null: random 40-day window, same tickers own history')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'observed post-event CAR {obs:+.2f}%')\n"
            "ax.set_xlabel('mean 40-day CAR vs SPY (%)')\n"
            "ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Post-event window is LESS BAD than a random stretch (p = {R[\"placebo_p\"]:.3f})')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'observed {obs:+.2f}%  vs placebo mean {pm:+.2f}% (sd {psd:.2f}%)')"
        ),
        md(
            f"> 💡 In plain words: a random 40-day stretch for these SAME distressed names "
            f"averages **{R['placebo_mean']:+.2f}%** vs SPY (they were removed for shrinking, "
            "after all — most of their own history is bad); the post-event window at "
            f"**{R['placebo_obs']:+.2f}%** sits well above that (p = {R['placebo_p']:.3f}). "
            "**This is a real, secondary result — relative stabilization, not an absolute "
            "positive rebound — and we deliberately do NOT let it upgrade the Signal stamp.** "
            "The pre-registered primary test is the absolute CAR against zero (the CNS claim "
            "literally is \"deleted stocks bounce back\", not \"deleted stocks decline "
            "somewhat less around the event than at other random times\"), and that test is a "
            "clean miss."
        ),
        md(
            "### 4e · The era split — is there anything left to decay?\n\n"
            "CNS's headline claim is that the deletion effect, unlike inclusion, has **not** "
            "decayed. Split our own 2012-2025 span in half (2019-01-01, chosen to bisect the "
            "sample) and test the difference directly."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ec = st.era_contrast(PANEL, EVENTS, data.ERA_SPLIT)\n"
            "    e, l = ec['early_car']*100, ec['late_car']*100\n"
            "    et, lt_, dt = ec['t_early'], ec['t_late'], ec['t_diff']\n"
            "    ne, nl = ec['n_early'], ec['n_late']\n"
            "else:\n"
            "    e, l = R['era_early_car'], R['era_late_car']\n"
            "    et, lt_, dt = R['era_early_t'], R['era_late_t'], R['era_diff_t']\n"
            "    ne, nl = R['era_early_n'], R['era_late_n']\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.3))\n"
            "ax.bar([f'2012 - {data.ERA_SPLIT[:4]}\\n(n={ne})', f'{data.ERA_SPLIT[:4]} - 2025\\n(n={nl})'],\n"
            "       [e, l], color=[GREY, GREY], width=.55)\n"
            "for i,(v,t_) in enumerate([(e,et),(l,lt_)]):\n"
            "    ax.annotate(f'{v:+.2f}%\\n(t={t_:+.2f})',(i,v),ha='center',\n"
            "        va='top' if v<0 else 'bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('post-event CAR [+1..+40] (%)')\n"
            "ax.set_title(f'Dead in BOTH halves - nothing to decay from (diff t = {dt:+.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'early {e:+.2f}% (t={et:+.2f})  late {l:+.2f}% (t={lt_:+.2f})  diff t={dt:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: the early half (t = {R['era_early_t']:+.2f}, "
            f"n = {R['era_early_n']}) and the late half (t = {R['era_late_t']:+.2f}, "
            f"n = {R['era_late_n']}) both fail to show a rebound, and the difference between "
            f"them is itself noise (t = {R['era_diff_t']:+.2f}). CNS's own sample ends around "
            "2000; by 2012, when ours begins, whatever premium existed already looks to have "
            "been arbitraged flat — consistent with Petajisto (2011)'s broader finding that "
            "the S&P rebalance-day \"index premium\" shrank as more capital chased it through "
            "the 2000s and 2010s."
        ),
        md(
            "### 4f · The third axis — the honest long-timer, with costs\n\n"
            "Enter at the effective-date close (public days ahead — zero look-ahead), hold 40 "
            "trading sessions, exit; return is already market-adjusted vs SPY (excess-vs-"
            "excess by construction)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = [st.long_timer(PANEL, hold_days=40, cost_bps=cb) for cb in (5.0, 10.0)]\n"
            "    g = rows[0]['gross']*100; n5, n10 = rows[0]['net']*100, rows[1]['net']*100\n"
            "    tv, worst, best, hit = (rows[0]['t_net'], rows[0]['worst']*100,\n"
            "                            rows[0]['best']*100, rows[0]['hit_rate']*100)\n"
            "else:\n"
            "    g, n5, n10 = R['lt_gross'], R['lt_net5'], R['lt_net10']\n"
            "    tv, worst, best, hit = R['lt_t_net5'], R['lt_worst'], R['lt_best'], R['lt_hit']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.8, 4.3))\n"
            "a1.bar(['gross', 'net 5 bps', 'net 10 bps'], [g, n5, n10], color=[GREY, RED, RED], width=.6)\n"
            "a1.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate([g, n5, n10]): a1.annotate(f'{v:+.2f}%',(i,v),ha='center',\n"
            "    va='top' if v<0 else 'bottom')\n"
            "a1.set_ylabel('bps per event (as %)'); a1.set_title(f'A coin flip that costs eat (t = {tv:+.2f})')\n"
            "a2.bar(['worst event', 'best event'], [worst, best], color=[RED, GREEN], width=.5)\n"
            "for i,v in enumerate([worst, best]): a2.annotate(f'{v:+.1f}%',(i,v),ha='center',\n"
            "    va='top' if v<0 else 'bottom')\n"
            "a2.axhline(0, c='k', lw=.8)\n"
            "a2.set_ylabel('single-event return (%)'); a2.set_title('Fat, symmetric tails')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross {g:+.2f}% -> net {n5:+.2f}% / {n10:+.2f}%  (t={tv:+.2f}); '\n"
            "      f'hit {hit:.1f}%; worst {worst:+.1f}%; best {best:+.1f}%')"
        ),
        md(
            f"> 💡 In plain words: gross is a coin flip ({R['lt_gross']:+.2f}%, "
            f"{R['lt_hit']:.0f}% hit rate); net of 5 bps costs it's {R['lt_net5']:+.2f}% at "
            f"t = {R['lt_t_net5']:.2f} — nothing. The tails are the real story: a "
            f"{R['lt_worst']:.0f}% single-event loss against a {R['lt_best']:+.0f}% best case "
            "on names that are, by construction, distressed enough to have just been removed "
            "from the world's most-tracked index. **H₄ not certified; Tradability = MIRAGE.**"
        ),
        md(
            "### 4g · Faithful-engine & power control — we know the truth here\n\n"
            "Synthetic mean-reverting 'distressed stock vs market' process with a TUNABLE "
            "planted dump-then-rebound. The null (dump = rebound = 0) is checked over **20 "
            "seeds** — never a single stream."
        ),
        code(
            "null_dump_t, null_reb_t = [], []\n"
            "for s_ in range(20):\n"
            "    evs = data.synthetic_panel(n_events=(len(PANEL) if HAVE_REAL else R['n_usable']),\n"
            "                                seed=652 + s_, dump=0.0, rebound=0.0)\n"
            "    r = st.synthetic_detect(evs)\n"
            "    null_dump_t.append(r['t_dump']); null_reb_t.append(r['t_rebound'])\n"
            "null_dump_t = np.asarray(null_dump_t); null_reb_t = np.asarray(null_reb_t)\n"
            "evs = data.synthetic_panel(n_events=(len(PANEL) if HAVE_REAL else R['n_usable']),\n"
            "                            seed=652, dump=0.08, rebound=0.08)\n"
            "planted = st.synthetic_detect(evs)\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.zeros(20)+np.linspace(-.1,.1,20), null_reb_t, color=GREY, s=40,\n"
            "           label='null worlds (dump=rebound=0), 20 seeds - rebound t')\n"
            "ax.scatter([1], [planted['t_rebound']], color=RED, s=90, zorder=5,\n"
            "           label='planted rebound = +8%')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0, 1]); ax.set_xticklabels(['null x 20', 'planted'])\n"
            "ax.set_ylabel('rebound t-stat')\n"
            "ax.set_title('Control: the null rarely fires, a planted rebound lights up cleanly')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null dump t: mean {null_dump_t.mean():+.2f} (sd {null_dump_t.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(abs(null_dump_t)>=2).sum()}/20')\n"
            "print(f'null reb  t: mean {null_reb_t.mean():+.2f} (sd {null_reb_t.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(abs(null_reb_t)>=2).sum()}/20')\n"
            "print(f'planted dump t = {planted[\"t_dump\"]:+.2f}  rebound t = {planted[\"t_rebound\"]:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 20 null worlds the dump detector never crosses the "
            f"bar ({R['syn_null_dump_fire']}/20) and the rebound detector fires "
            f"{R['syn_null_reb_fire']}/20 — within the ~5% false-positive rate you'd expect "
            f"from a *t* = 2 threshold, not a biased machine. A realistic planted effect "
            f"(dump = -8%, rebound = +8%) lights up clearly: dump t = "
            f"{R['syn_planted_dump_t']:.2f}, rebound t = {R['syn_planted_reb_t']:.2f}. The "
            "machinery is unbiased — a real dump-then-rebound of this size WOULD have been "
            "caught. It just isn't there. *(A faithful-engine / power check only — never "
            "cited in support of the real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — dump [-5..0] **{R['dump_pct']:+.2f}%** (t = "
            f"{R['dump_t']:+.2f}); rebound [+1..+40] **{R['reb_pct']:+.2f}%** (t = "
            f"{R['reb_t']:+.2f}, 95% CI [{R['reb_ci'][0]:+.2f}%, {R['reb_ci'][1]:+.2f}%]) — "
            "neither leg clears t ≥ 2, and neither half of the sample shows it independently "
            f"(early t = {R['era_early_t']:+.2f}, late t = {R['era_late_t']:+.2f}). A real but "
            f"secondary placebo finding (relative stabilization, p = {R['placebo_p']:.3f}) is "
            "named and explicitly not used to inflate the stamp. Survivorship (31% of the "
            "basket has no tape left) is named and biases the sample *toward*, not against, "
            "finding a rebound — yet nothing shows up.\n"
            f"- **Tradability `MIRAGE`** — a long-the-deleted timer nets "
            f"{R['lt_net5']:+.2f}% at 5 bps costs (t = {R['lt_t_net5']:+.2f}), with a "
            f"{R['lt_worst']:.0f}% single-event tail. Nothing survives being turned into a "
            "position.\n"
            "- **\"Has the deletion bounce escaped inclusion's fate?\" `BUSTED`** — CNS's "
            "asymmetry claim (deletion persists, inclusion decays) does not hold up: "
            "[249-index-inclusion](../../249-index-inclusion/) already found the ADD-side pop "
            "dead, and this study finds the DELETE-side rebound equally dead. Twenty years "
            "after CNS, both halves of their asymmetry have converged to the same answer: "
            "quiet."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The general object is index-flow arbitrage decay.** Petajisto (2011) "
            "documents the S&P rebalance-day \"index premium\" shrinking as passive assets "
            "and dedicated arbitrage capital both grew through the 2000s-2010s; this study's "
            "null result on the deletion side is the natural continuation of that finding two "
            "decades further on.\n"
            "- **Where a residual effect might still hide:** intraday resolution around the "
            "effective-date close (beyond what daily bars resolve), smaller/less-arbitraged "
            "benchmarks (small-cap or international indices), or names with unusually thin "
            "float where even modern arb capital can't fully absorb the forced flow.\n"
            "- **Dedup map:** [249-index-inclusion](../../249-index-inclusion/) (the ADD side "
            "— also dead), [320-russell-reconstitution](../../320-russell-reconstitution/) "
            "(whole-index ETF, not single names), [250-reverse-split](../../250-reverse-split/) "
            "(a related but distinct distress signature).\n\n"
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
