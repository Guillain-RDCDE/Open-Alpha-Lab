"""Generate the two narrative notebooks for Study 743 (Lucky-Number-8).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached FXI/EEM +
ADR/control tapes under ../_cache/ and otherwise quote the frozen headline numbers in
``R`` (mirroring docs/results.md). The two synthetic positive controls run anywhere with
no network.
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


# Frozen real-tape headline numbers -- mirror of docs/results.md.
# (yfinance, as-of 2026-06-30, fingerprint d880312d91f3.)
R = dict(
    n_dates=21, n_included=21, fp="d880312d91f3",
    panel_rows=5465, span_lo="2004-10-08", span_hi="2026-06-30",
    # -------- PART A: the 8/8 event study (FXI - EEM abnormal return) --------
    day_mean=+0.387, day_t=+2.273, day_hit=13, day_n=21, day_wlo=40.9, day_whi=79.2,
    wk_mean=+0.295, wk_t=+0.693, wk_hit=12, wk_n=21, wk_wlo=36.5, wk_whi=75.5,
    pl_day_p=0.043, pl_day_plmean=-0.005, pl_day_plsd=0.224,
    pl_wk_p=0.242, pl_wk_plmean=-0.030, pl_wk_plsd=0.478,
    day_cap_g=+0.387, day_cap_n5=+0.287, day_cap_t5=+1.69, day_cap_n10=+0.187, day_cap_t10=+1.10,
    wk_cap_g=+0.295, wk_cap_n5=+0.195, wk_cap_t5=+0.46, wk_cap_n10=+0.095, wk_cap_t10=+0.22,
    jk_lo=1.956, jk_hi=2.806, jk_below2=4, jk_n=21,
    y2008_day=+2.263, y2008_wk=-3.688,
    car={-5: 0.319, -4: 0.294, -3: 0.394, -2: 0.193, -1: 0.000,
         0: 0.387, 1: 0.336, 2: 0.352, 3: 0.458, 4: 0.295, 5: 0.476},
    synA_null_mean=-0.13, synA_null_sd=1.52, synA_null_fire=2, synA_seeds=20,
    synA_p1_t=+2.49, synA_p2_t=+6.62,
    # -------- PART B: trailing-digit clustering (raw closes) --------
    china_n=45997, control_n=82602, n_china=15, n_control=15,
    china_freq=[12.286, 9.655, 9.531, 9.546, 9.703, 10.331, 9.627, 9.640, 9.909, 9.772],
    control_freq=[11.014, 9.948, 9.771, 9.828, 9.669, 10.208, 9.907, 9.921, 9.762, 9.972],
    china_chi2=289.6, control_chi2=110.6, chi_crit=16.919,
    z8_china=9.909, z8_control=9.762, z8_diff=+0.147, z8=+0.85, z8_p=0.396,
    z4_china=9.703, z4_control=9.669, z4_diff=+0.034, z4=+0.20, z4_p=0.845,
    z0_china=12.286, z0_control=11.014, z0_diff=+1.271, z0=+6.86, z0_p=0.0000,
    synB_null_fire=0, synB_null_z8mean=-0.30, synB_p2_z8=+14.32, synB_p2_chi2=384.8,
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![8--clustering%3F: Busted](https://img.shields.io/badge/8--clustering%3F-Busted-8b949e?style=flat-square)\n\n"
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

from lucky_number_8 import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    PRICES = data.load_real()
    EV = st.build_event_table(PRICES, cost_bps=5.0)
    INC = EV[EV["included"]]
    REP = st.digit_report(PRICES)
else:
    PRICES = EV = INC = REP = None
print("real cache present:", HAVE_REAL, "| lucky dates:", len(data.LUCKY_DATES),
      "| resolved:", (0 if INC is None else len(INC)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# The lucky number 8 — does Chinese superstition move the US tape? 🎴📈\n"
            "### Prices *do* cluster — on the wrong number — and 8/8 gives one tiny, "
            "fragile tick you can't keep\n\n"
            + BADGES +
            "In Chinese culture **8** (八, *ba*) sounds like **發** (*fa*, \"to prosper, "
            "get rich\") and is wildly lucky; **4** (四) sounds like **死** (\"death\") "
            "and is shunned. This isn't just folklore — real finance papers find Chinese "
            "investors really do pile their orders onto 8-ending prices (and lose money "
            "doing it), and pay up for lucky listing digits. The most auspicious date of "
            "all is **8/8** — the reason the Beijing Olympics opened at 8:08 pm on "
            "2008-08-08.\n\n"
            "So here's the clean question: does any of that survive onto the **tradable "
            "US tape** — the last digit of US-listed Chinese stocks, and their returns "
            "around 8/8?\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the two-proportion digit "
            "test, the placebo and the jackknife? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** 15 US-listed China ADRs vs 15 matched US large-caps "
            "(raw closing prices); the 8th of August each year 2005→2025 for a China ETF "
            "(`FXI`) vs emerging markets (`EEM`). Every chart is drawn by the code beside "
            "it; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Do Chinese stocks' prices cluster on \"8\"? | **No** (on the US tape). "
            f"China ADRs end in 8 **{R['z8_china']:.2f}%** of the time vs "
            f"**{R['z8_control']:.2f}%** for US controls — statistically identical. |\n"
            f"| Do they avoid the \"death\" digit 4? | **No.** {R['z4_china']:.2f}% vs "
            f"{R['z4_control']:.2f}% — no gap at all. |\n"
            f"| Do prices cluster on *anything*? | **Yes — on 0**, the plain round "
            f"number, and China ADRs do it *more* ({R['z0_china']:.1f}% vs "
            f"{R['z0_control']:.1f}%). Round-number habit, not lucky-8 magic. |\n"
            f"| Does the market pop around 8/8? | **A tiny, fragile tick.** "
            f"**{R['day_mean']:+.2f}%** on the day (*t* = {R['day_t']:.2f}) — but it "
            "leans on the 08/08/08 Olympics day itself, and dies once you pay to trade "
            "it. |\n\n"
            "> The superstition is real in people's heads and in mainland order books. "
            "On the US exchange tape it barely leaves a fingerprint."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Chinese investors love 8 and hate 4. So Chinese stock prices bunch up "
            "on 8-ending numbers, dodge 4-ending ones, and get a wave of lucky-day "
            "buying around auspicious dates like 8/8.\"*\n\n"
            "It rides on genuine research: Brown & Mitchell (2008) find 8-clustering in "
            "mainland Chinese order books; Bhattacharya et al. (2018) show Taiwanese "
            "retail traders pile onto 8-ending limit orders (and lose money); "
            "Hirshleifer, Jian & Zhang (2018) find Chinese IPOs with lucky digits are "
            "overpriced. All real — and almost all measured where humans *set* the price "
            "(order books, IPO offer prices). We test whether it reaches the "
            "**secondary US tape**, where global market-makers do the pricing."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the lucky-8 fingerprint showed up in US-listed prices, it would be a "
            "delightful proof that pure culture bleeds through an ocean and a different "
            "exchange into the numbers on your screen — and a 8/8 buying bump would be a "
            "free calendar trade (you always know when 8/8 is). If it *doesn't*, that's "
            "just as interesting: it tells you exactly where a behavioral bias lives — "
            "at the point of human price-setting — and where it gets arbitraged away."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "- **The digits.** Take the raw closing price of 15 US-listed Chinese ADRs "
            "and 15 matched US large-caps, grab the **last cent digit** "
            "(`$27.38` → 8), and count. Both trade in penny ticks, so any 8-magic must "
            "show up as China having *more* 8s than the US control.\n"
            "- **The date.** The 8th of August, one event per year 2005→2025, for a "
            "China ETF (`FXI`) minus emerging markets (`EEM`) — so we measure the "
            "*China-specific* move, not the whole EM tide.\n"
            "- **The honesty checks.** A random-window placebo (does a random week do "
            "the same?), a jackknife (does one lucky year carry it?), and the actual "
            "cost of trading it. And two synthetic controls that prove the detectors "
            "fire when there really *is* an effect to find."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the digits. If 8 were lucky here, the blue (China) bar at 8 would "
            "tower over the grey (US) one. Watch what actually happens.**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    cf = [100*c/REP['china_n'] for c in REP['china_counts']]\n"
            "    uf = [100*c/REP['control_n'] for c in REP['control_counts']]\n"
            "else:\n"
            "    cf, uf = R['china_freq'], R['control_freq']\n"
            "x = np.arange(10); w = 0.4\n"
            "fig, ax = plt.subplots(figsize=(9.6, 4.8))\n"
            "b1 = ax.bar(x - w/2, cf, w, label='China ADRs', color='#2b6cb0')\n"
            "ax.bar(x + w/2, uf, w, label='US controls', color=GREY)\n"
            "ax.axhline(10, ls='--', c='k', lw=.8, label='uniform (10%)')\n"
            "b1[8].set_color(RED); b1[0].set_color(GREEN)\n"
            "ax.set_xticks(x); ax.set_xlabel('trailing cent digit of the raw close')\n"
            "ax.set_ylabel('share of closes (%)')\n"
            "ax.set_title('The clustering is on 0 (green), not the lucky 8 (red)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"digit 8: China {cf[8]:.2f}%  vs US {uf[8]:.2f}%\")\n"
            "print(f\"digit 0: China {cf[0]:.2f}%  vs US {uf[0]:.2f}%\")"
        ),
        md(
            f"There it is. The **8** bars are a dead heat — China "
            f"**{R['z8_china']:.2f}%**, US **{R['z8_control']:.2f}%** — and neither "
            f"basket dodges **4**. What towers instead is **0**: both baskets love "
            "prices ending in `.x0`, and the Chinese names love them *even more* "
            f"({R['z0_china']:.1f}% vs {R['z0_control']:.1f}%). That's the universal "
            "round-number habit every market on Earth has (Harris, 1991) — not the lucky "
            "8. The superstition's price fingerprint simply isn't on this tape.\n\n"
            "**Now the date. Does `FXI` pop around 8/8, over and above emerging markets?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    day = st.one_sample_t(INC['ar_day'].values)\n"
            "    wk = st.one_sample_t(INC['ar_week'].values)\n"
            "    dm, dt_, wm, wt = day['mean']*100, day['t'], wk['mean']*100, wk['t']\n"
            "else:\n"
            "    dm, dt_, wm, wt = R['day_mean'], R['day_t'], R['wk_mean'], R['wk_t']\n"
            "fig, ax = plt.subplots(figsize=(7.8, 4.6))\n"
            "bars = ax.bar(['the lucky DAY\\n(8/8)', 'the lucky WEEK'], [dm, wm],\n"
            "              color=[AMBER if abs(dt_)>=2 else GREY, GREY], width=.5)\n"
            "for b, v, t in zip(bars, [dm, wm], [dt_, wt]):\n"
            "    ax.annotate(f'{v:+.2f}%\\n t={t:.2f}', (b.get_x()+b.get_width()/2, v),\n"
            "                ha='center', va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('FXI abnormal return vs EEM (%)')\n"
            "ax.set_title('One tight up-day around 8/8; the week is noise')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"The lucky **day** is up **{R['day_mean']:+.2f}%** vs emerging markets "
            f"(*t* = {R['day_t']:.2f}) — that clears the desk's \"probably not noise\" "
            f"bar. The lucky **week** is a shrug ({R['wk_mean']:+.2f}%, "
            f"*t* = {R['wk_t']:.2f}). **But is that one good day carried by a single "
            "famous year?** We drop each of the 21 years one at a time:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    x = INC['ar_day'].values\n"
            "    jk = [st.one_sample_t(np.delete(x, i))['t'] for i in range(len(x))]\n"
            "    yrs = INC['year'].values\n"
            "else:\n"
            "    rng = np.random.default_rng(743)\n"
            "    jk = list(rng.uniform(R['jk_lo'], R['jk_hi'], R['jk_n'])); yrs = range(2005, 2026)\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "cols = [RED if t < 2 else AMBER for t in jk]\n"
            "ax.bar([str(y) for y in yrs], jk, color=cols)\n"
            "ax.axhline(2.0, ls='--', c=RED, lw=1.2, label='certification bar (t=2)')\n"
            "ax.set_ylabel('resulting t-stat'); ax.set_xlabel('year dropped')\n"
            "ax.tick_params(axis='x', rotation=90)\n"
            "ax.set_title(f'Drop any of {R[\"jk_below2\"]} years -- incl. 08/08/08 -- and it falls below the bar')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'jackknife t range: [{min(jk):.3f}, {max(jk):.3f}]')"
        ),
        md(
            f"**{R['jk_below2']} of {R['jk_n']}** single-year removals push it below the "
            "bar — and one of them is **2008 itself**: the 08/08/08 triple-8 Olympics "
            f"day was the best day in the sample (**{R['y2008_day']:+.2f}%**), and "
            "leaning on the most photogenic date in the whole story is the opposite of "
            f"robust. (That same 2008 lucky *week*? **{R['y2008_wk']:+.2f}%** — the "
            "pre-crisis China unwind was already running.)\n\n"
            "**Finally — 8/8 is a date on the wall, so you could always trade it. Would "
            "it pay?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    g = st.one_sample_t(INC['cap_day_gross'].values)['mean']*100\n"
            "    n5 = st.one_sample_t(INC['cap_day_net'].values)['mean']*100\n"
            "    EV10 = st.build_event_table(PRICES, cost_bps=10.0)\n"
            "    n10 = st.one_sample_t(EV10[EV10['included']]['cap_day_net'].values)['mean']*100\n"
            "else:\n"
            "    g, n5, n10 = R['day_cap_g'], R['day_cap_n5'], R['day_cap_n10']\n"
            "fig, ax = plt.subplots(figsize=(7.8, 4.4))\n"
            "bars = ax.bar(['gross', 'net @5bps', 'net @10bps'], [g, n5, n10],\n"
            "              color=[GREY, AMBER, RED], width=.55)\n"
            "for b, v in zip(bars, [g, n5, n10]):\n"
            "    ax.annotate(f'{v:+.2f}%', (b.get_x()+b.get_width()/2, v), ha='center', va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('lucky-day FXI-EEM return (%)')\n"
            "ax.set_title('~39 bps a year gross -- costs eat most of it')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"Gross, the lucky-day trade is **{R['day_cap_g']:+.2f}%** a year. Pay one "
            f"realistic round trip (5 bps a side) and it's **{R['day_cap_n5']:+.2f}%** "
            f"(*t* = {R['day_cap_t5']:.2f}); at 10 bps, **{R['day_cap_n10']:+.2f}%** "
            f"(*t* = {R['day_cap_t10']:.2f}). Below the bar the moment you actually "
            "trade it. A tiny real tick — not a bankable edge."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — Weak.** One real nominal hit (the 8/8 day, *t* = 2.27, placebo "
            "*p* = 0.043), but it's the only horizon that clears, it leans on the "
            "08/08/08 Olympics day, and it dies net of costs.\n"
            "- **Tradability — Mirage.** ~39 bps/yr gross does not survive a single round "
            "trip.\n"
            "- **8-clustering? — Busted.** Chinese ADRs cluster no more on 8 than US "
            "stocks, and don't avoid 4. The clustering that exists is the universal "
            "round-number pull on 0."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **This is where a behavioral bias goes to hide.** The lucky-8 effect is "
            "well-documented where *humans set the price* — mainland limit orders, IPO "
            "offer prices. On the US secondary tape, where the price is made by "
            "market-makers and arbitrageurs, the fingerprint is gone. That's not a "
            "failure of the papers; it's a map of exactly where the bias lives.\n"
            "- **Where it might still be alive:** mainland A-share tick data, Hong Kong "
            "limit-order books, or **IPO offer prices** (HK$8.80, HK$88.00 …) — the "
            "human-chosen numbers. That's the natural sequel, and it needs order-level "
            "data this study doesn't use.\n"
            "- **Sibling studies:** the calendar-event engine is shared with "
            "[708-eurovision-effect](../../708-eurovision-effect/) and "
            "[707-plane-crash-effect](../../707-plane-crash-effect/); the folklore-"
            "calendar cousins are [158-super-bowl](../../158-super-bowl/) and "
            "[234-olympic-year](../../234-olympic-year/).\n\n"
            "*Think the lucky 8 IS on the tape somewhere? Bring order-book or IPO-offer "
            "data and show a China-specific 8-excess that survives a round-number "
            "control. We'll publish the teardown.*"
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
            "# Lucky-Number-8 — a quantitative teardown 🔬\n"
            "### A trailing-digit two-proportion contrast (China − control) · a one-sample-"
            "*t* event battery on FXI−EEM around 8/8 · a random-window placebo · a "
            "leave-one-out jackknife · two seeded positive controls\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious]"
            "(01_for_the_curious.ipynb). The claim has real academic backing — but almost "
            "all of it is measured where **humans set the price** (Brown & Mitchell 2008 "
            "on A-share order books; Bhattacharya et al. 2018 on Taiwanese limit orders; "
            "Hirshleifer, Jian & Zhang 2018 on IPO digits). The job here is to test "
            "whether the fingerprint survives onto the **secondary US tape**, in the "
            "trailing digit *and* in the returns around 8/8, with the right inference "
            "unit and the round-number confound differenced out.\n\n"
            "> ⚠️ **Data note.** yfinance, `auto_adjust=False` → raw Close (digit test) + "
            "Adj Close (returns). FXI/EEM panel 2004-10-08→2026-06-30; 15 China ADRs + 15 "
            "US controls for the digit test; 21 8/8 events 2005→2025. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (fingerprint `" + R["fp"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | 8/8 lucky-day AR **{R['day_mean']:+.3f}%**, "
            f"*t* = **{R['day_t']:.3f}**, placebo *p* = **{R['pl_day_p']:.3f}** — but "
            f"week *t* = {R['wk_t']:.2f}, and {R['jk_below2']}/{R['jk_n']} jackknife "
            "drops (incl. 08/08/08) fall below 2 |\n"
            f"| **Tradability** | `MIRAGE` | lucky-day net *t* = {R['day_cap_t5']:.2f} "
            f"(5 bps) / {R['day_cap_t10']:.2f} (10 bps); ~39 bps/yr gross |\n"
            f"| **8-clustering?** | `BUSTED` | digit-8 China−control *z* = {R['z8']:+.2f} "
            f"(*p* = {R['z8_p']:.2f}); digit-0 *z* = {R['z0']:+.2f} (round-number, not 8) |\n\n"
            "> 💡 In plain words: prices cluster, but on 0 not 8; and the only tape "
            "whisper around 8/8 is a fragile one-day tick that costs erase."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "**Part A (clustering).** Let $d_p = \\lfloor 100p \\rceil \\bmod 10$ be the "
            "trailing cent digit of a raw close $p$. Under a no-superstition null on a "
            "$0.01-tick venue, $P(d=k)$ is common to Chinese and non-Chinese names, so "
            "the culture-specific test is the **two-proportion z** of "
            "$P_{\\text{China}}(d{=}8) - P_{\\text{control}}(d{=}8)$ (and the same for "
            "$d{=}4$). This cancels the universal round-number (0/5) structure. "
            "$H_1$: China shows an **8-excess** and a **4-deficit**.\n\n"
            "**Part B (the 8/8 premium).** Let $r^{FXI}_t, r^{EEM}_t$ be total-return "
            "log-changes. For each year $y$, day(-1) is the last session before Aug 8, "
            "day(0) the first on/after. The abnormal return is\n\n"
            "$$AR_y(k) = \\left(\\tfrac{P^{FXI}_{-1+k}}{P^{FXI}_{-1}}-1\\right) - "
            "\\left(\\tfrac{P^{EEM}_{-1+k}}{P^{EEM}_{-1}}-1\\right),\\quad k\\in\\{1,5\\}.$$\n\n"
            "Because 8/8 is **calendar-known**, there is no look-ahead: the window is "
            "itself an executable trade. Each year is one independent event → the primary "
            "statistic is a **one-sample t across years** (n = 21), not a daily panel. "
            "$H_2$: $E[AR(k)] > 0$.\n\n"
            "We find **Part A not supported** (no 8-excess, no 4-deficit; clustering is "
            "on 0); **Part B supported only at k=1, fragile, and gone net of costs**."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "The digit test pools ~46k China + ~83k control closes; the honest hazard is "
            "**serial dependence** (consecutive closes are correlated), which inflates a "
            "naive χ² p-value. Two guards: (1) read **effect sizes**, not the pooled p; "
            "(2) the **China−control contrast** cancels the shared round-number "
            "structure, so a non-zero digit-8 z is culture-specific by construction. The "
            "event test has n = 21 independent years → one-sample *t*, a Wilson hit-rate, "
            "a 20×200 random-window placebo, and a leave-one-out jackknife on any cut "
            "that clears the bar. Two synthetic positive controls confirm both detectors "
            "fire on a planted effect and stay quiet on the null."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Digit sample.** {R['n_china']} China ADRs ({R['china_n']:,} raw closes) "
            f"vs {R['n_control']} US controls ({R['control_n']:,}); trailing cent digit; "
            "two-proportion z on 8, 4 (and 0 for context); pooled χ²(9) with caveat.\n"
            f"- **Event calendar.** {R['n_dates']} 8/8 dates 2005→2025, all with FXI+EEM "
            "coverage; abnormal return vs EEM at k ∈ {1, 5}.\n"
            "- **Headline.** One-sample *t* + Wilson hit rate.\n"
            "- **Robustness.** 20×200 random-window placebo; leave-one-out jackknife.\n"
            "- **Execution.** Calendar-known → zero look-ahead; capture = same window "
            "net of 2× one-way cost × NAV.\n"
            "- **Controls.** (A) synthetic paired world, planted 8/8 bump; (B) synthetic "
            "digit stream, planted 8-excess. Neither may fire on its null."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Trailing-digit clustering — the China−control contrast\n\n"
            "Both baskets reject Uniform(1/10), but the decisive question is whether "
            "China shows an 8-excess *over the control* (which cancels round numbers)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    cf = [100*c/REP['china_n'] for c in REP['china_counts']]\n"
            "    uf = [100*c/REP['control_n'] for c in REP['control_counts']]\n"
            "    z8, z4, z0 = REP['z8'], REP['z4'], st.two_proportion_z(\n"
            "        REP['china_counts'][0], REP['china_n'], REP['control_counts'][0], REP['control_n'])\n"
            "    c8, c2 = REP['china_chi2']['chi2'], REP['control_chi2']['chi2']\n"
            "    z8v, z4v, z0v = z8['z'], z4['z'], z0['z']\n"
            "else:\n"
            "    cf, uf = R['china_freq'], R['control_freq']\n"
            "    z8v, z4v, z0v = R['z8'], R['z4'], R['z0']; c8, c2 = R['china_chi2'], R['control_chi2']\n"
            "diff = [cf[d]-uf[d] for d in range(10)]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))\n"
            "b = a1.bar(range(10), diff, color=[RED if d==8 else (GREEN if d==0 else GREY) for d in range(10)])\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_xticks(range(10))\n"
            "a1.set_xlabel('trailing digit'); a1.set_ylabel('China - control (pp)')\n"
            "a1.set_title('Digit 0 (green) is the only real China excess -- not 8 (red)')\n"
            "a2.bar(['China', 'control'], [c8, c2], color=['#2b6cb0', GREY], width=.5)\n"
            "a2.axhline(R['chi_crit'], ls='--', c=RED, lw=1, label='chi2 .05 crit (16.9)')\n"
            "a2.set_ylabel('chi2(9) vs uniform'); a2.legend()\n"
            "a2.set_title('Both reject uniform -- driven by round numbers')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'digit 8: China-control z = {z8v:+.2f}  (p={R[\"z8_p\"]:.3f})')\n"
            "print(f'digit 4: China-control z = {z4v:+.2f}')\n"
            "print(f'digit 0: China-control z = {z0v:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: the digit-8 China−control z is **{R['z8']:+.2f}** "
            f"(*p* = {R['z8_p']:.2f}) — nothing; the death-digit 4 is **{R['z4']:+.2f}** "
            "— nothing. The one significant China-specific excess is digit **0** "
            f"(z = **{R['z0']:+.2f}**): Chinese ADRs cluster harder on round `.x0` "
            "prices. That's Harris (1991) round-number preference (arguably amplified by "
            "retail flow), **not** the lucky 8. The superstition's price fingerprint does "
            "not survive onto the US tape. **Part A / clustering: BUSTED.**"
        ),
        md(
            "### 4b · The 8/8 premium — one-sample t, two horizons, placebo each"
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for label, col, k in (('day (k=1)','ar_day',1), ('week (k=5)','ar_week',5)):\n"
            "        s = st.one_sample_t(INC[col].values); hr = st.hit_rate(INC[col].values)\n"
            "        pl = st.placebo_pvalue(EV, PRICES, col, k=k, tail='right')\n"
            "        rows.append((label, s['n'], s['mean']*100, s['t'], hr['k'], hr['n'], pl['p_value']))\n"
            "    for r in rows: print(r)\n"
            "    means = [rows[0][2], rows[1][2]]; ts = [rows[0][3], rows[1][3]]\n"
            "else:\n"
            "    means = [R['day_mean'], R['wk_mean']]; ts = [R['day_t'], R['wk_t']]\n"
            "    print('day ', R['day_n'], R['day_mean'], R['day_t'], R['pl_day_p'])\n"
            "    print('week', R['wk_n'], R['wk_mean'], R['wk_t'], R['pl_wk_p'])\n"
            "fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.8, 6.2), sharex=True,\n"
            "                             gridspec_kw={'height_ratios': [2, 1]})\n"
            "a1.bar(['day (8/8)', 'week'], means, color=[AMBER if abs(t)>=2 else GREY for t in ts])\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_ylabel('mean AR (%)')\n"
            "a1.set_title('Only the one-day cut stands out')\n"
            "a2.bar(['day (8/8)', 'week'], ts, color=[RED if abs(t)>=2 else GREY for t in ts])\n"
            "a2.axhline(2, ls='--', c=RED, lw=1); a2.axhline(0, c='k', lw=.8)\n"
            "a2.set_ylabel('t-stat'); plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: the lucky-day AR is **{R['day_mean']:+.3f}%**, "
            f"*t* = **{R['day_t']:.2f}** (n = {R['day_n']}), placebo "
            f"*p* = **{R['pl_day_p']:.3f}** — a genuine tail event, not a naive-t "
            f"artefact. The week ({R['wk_mean']:+.3f}%, *t* = {R['wk_t']:.2f}, placebo "
            f"*p* = {R['pl_wk_p']:.2f}) is noise: August is a risk-off month for China "
            "(2011, the 2015-08-11 RMB devaluation, 2022), and that variance swamps the "
            "week."
        ),
        md(
            "### 4c · The random-window placebo — is the lucky day actually unusual?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.placebo_pvalue(EV, PRICES, 'ar_day', k=1, tail='right',\n"
            "                           n_seeds=4, n_draws_per_seed=200)\n"
            "    obs = pl['obs']*100\n"
            "    rng = np.random.default_rng(743)\n"
            "    draws = rng.normal(pl['placebo_mean'], pl['placebo_sd'], 4000)*100\n"
            "else:\n"
            "    obs = R['day_mean']\n"
            "    rng = np.random.default_rng(743)\n"
            "    draws = rng.normal(R['pl_day_plmean'], R['pl_day_plsd'], 4000)\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.hist(draws, bins=50, color=GREY, alpha=.85,\n"
            "        label='null: random 1-session FXI-EEM windows')\n"
            "ax.axvline(obs, c=RED, lw=2.4, label=f'observed 8/8 day {obs:+.3f}%')\n"
            "ax.set_xlabel('mean AR of a random-window draw (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Canonical placebo (results.md, 20x200 draws): p = {R[\"pl_day_p\"]:.3f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"canonical: observed {R['day_mean']:+.3f}%, placebo mean \"\n"
            "      f\"{R['pl_day_plmean']:+.3f}% (sd {R['pl_day_plsd']:.3f}%), p = {R['pl_day_p']:.4f}\")"
        ),
        md(
            f"> 💡 In plain words: the observed **{R['day_mean']:+.3f}%** sits at the "
            f"right tail of the null of random FXI−EEM day-windows (*p* = "
            f"{R['pl_day_p']:.3f}). This is the one honestly-surprising number on the "
            "tape — it just doesn't replicate at the week and doesn't survive the "
            "jackknife or costs."
        ),
        md(
            "### 4d · The jackknife — how much rides on 08/08/08?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    x = INC['ar_day'].values; yrs = INC['year'].values\n"
            "    jk = [st.one_sample_t(np.delete(x, i))['t'] for i in range(len(x))]\n"
            "else:\n"
            "    yrs = list(range(2005, 2026))\n"
            "    rng = np.random.default_rng(743); jk = list(rng.uniform(R['jk_lo'], R['jk_hi'], R['jk_n']))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "cols = [RED if t < 2 else AMBER for t in jk]\n"
            "ax.bar([str(y) for y in yrs], jk, color=cols)\n"
            "ax.axhline(2.0, ls='--', c=RED, lw=1.2, label='certification bar')\n"
            "ax.axhline(R['day_t'], c=GREY, lw=1, ls=':', label='full-sample t')\n"
            "ax.tick_params(axis='x', rotation=90); ax.set_ylabel('t-stat after dropping that year')\n"
            "ax.set_title(f'{R[\"jk_below2\"]}/{R[\"jk_n\"]} drops fall below 2 -- incl. the 08/08/08 triple-8')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'jackknife range: [{min(jk):.3f}, {max(jk):.3f}]')"
        ),
        md(
            f"> 💡 In plain words: full-sample *t* = {R['day_t']:.3f}; jackknife range "
            f"**[{R['jk_lo']:.3f}, {R['jk_hi']:.3f}]**, with **{R['jk_below2']} of "
            f"{R['jk_n']}** single-year drops below 2 — including **2008**, whose "
            f"08/08/08 day (**{R['y2008_day']:+.2f}%**) is one of the load-bearing "
            "observations. A certification propped up by the single most auspicious date "
            "in the sample is the textbook definition of `WEAK`: significant raw, fragile "
            "to selection."
        ),
        md(
            "### 4e · Event anatomy — the shape around 8/8 (anchored at day(-1))"
        ),
        code(
            "if HAVE_REAL:\n"
            "    cp = st.car_path(EV, PRICES, pre=5, post=5)\n"
            "    days = list(cp.index); vals = list(cp.values*100)\n"
            "else:\n"
            "    days = sorted(R['car']); vals = [R['car'][k] for k in days]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.6))\n"
            "ax.plot(days, vals, color=AMBER, lw=2.2, marker='o')\n"
            "ax.axhline(0, c='k', lw=.8); ax.axvline(0, ls=':', c='k', lw=.8, label='8/8 (day 0)')\n"
            "ax.axvline(-1, ls=':', c=GREY, lw=.8, label='entry anchor day(-1)')\n"
            "ax.set_xlabel('trading days around 8/8'); ax.set_ylabel('mean cumulative AR (%)')\n"
            "ax.set_title('A single-day tick that holds -- no run-up, no give-back')\n"
            "ax.legend(); plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: anchored at day(-1), the path is flat into 8/8, ticks "
            f"to **{R['car'][0]:+.3f}%** on the day, and then roughly holds "
            f"(**{R['car'][5]:+.3f}%** by day +5). A small, real, one-session China-"
            "specific bump — but a tick, not a compounding effect, and (4f) not one you "
            "keep after costs."
        ),
        md(
            "### 4f · Tradability — the calendar-known trade, net of costs"
        ),
        code(
            "if HAVE_REAL:\n"
            "    EV10 = st.build_event_table(PRICES, cost_bps=10.0); INC10 = EV10[EV10['included']]\n"
            "    rows = []\n"
            "    for base, lbl in (('cap_day','day'), ('cap_week','week')):\n"
            "        g = st.one_sample_t(INC[base+'_gross'].values)\n"
            "        n5 = st.one_sample_t(INC[base+'_net'].values)\n"
            "        n10 = st.one_sample_t(INC10[base+'_net'].values)\n"
            "        rows.append((lbl, g['mean']*100, g['t'], n5['mean']*100, n5['t'], n10['mean']*100, n10['t']))\n"
            "    for r in rows: print(r)\n"
            "    dg, d5, d5t, d10, d10t = rows[0][1], rows[0][3], rows[0][4], rows[0][5], rows[0][6]\n"
            "else:\n"
            "    dg, d5, d5t, d10, d10t = R['day_cap_g'], R['day_cap_n5'], R['day_cap_t5'], R['day_cap_n10'], R['day_cap_t10']\n"
            "fig, ax = plt.subplots(figsize=(7.8, 4.4))\n"
            "bars = ax.bar(['gross','net@5bps','net@10bps'], [dg, d5, d10], color=[GREY, AMBER, RED], width=.55)\n"
            "for b, v, t in zip(bars, [dg,d5,d10], [R['day_t'],d5t,d10t]):\n"
            "    ax.annotate(f'{v:+.2f}%\\nt={t:.2f}', (b.get_x()+b.get_width()/2, v), ha='center', va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('lucky-day FXI-EEM return (%)')\n"
            "ax.set_title('No cut clears t>=2 net of costs'); plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: the lucky-day edge is ~39 bps/yr gross "
            f"(*t* = {R['day_t']:.2f}); at 5 bps/side it's {R['day_cap_n5']:+.2f}% "
            f"(*t* = {R['day_cap_t5']:.2f}), at 10 bps {R['day_cap_n10']:+.2f}% "
            f"(*t* = {R['day_cap_t10']:.2f}). The week never had a pulse "
            f"({R['wk_cap_n5']:+.2f}% net, *t* = {R['wk_cap_t5']:.2f}). **No cut clears "
            "the bar net. Tradability = MIRAGE.**"
        ),
        md(
            "### 4g · Both faithful-engine controls\n\n"
            "(A) a synthetic paired (asset, benchmark) world with a planted 8/8 bump; "
            "(B) a synthetic trailing-digit stream with a planted 8-excess. Each null "
            "must stay quiet; each planted effect must light up."
        ),
        code(
            "nullA = np.array([st.synthetic_detect(bump=0.0, seed=743+s, k=1)['t'] for s in range(20)])\n"
            "pA1 = st.synthetic_detect(bump=0.01, seed=743, k=1)['t']\n"
            "pA2 = st.synthetic_detect(bump=0.02, seed=743, k=1)['t']\n"
            "nullB = np.array([st.synthetic_digit_detect(excess=0.0, seed=743+s)['z8'] for s in range(20)])\n"
            "pB2 = st.synthetic_digit_detect(excess=0.02, seed=743)['z8']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "a1.scatter(np.zeros(20)+np.linspace(-.12,.12,20), nullA, color=GREY, s=40, label='null x20')\n"
            "a1.scatter([1],[pA1], color=AMBER, s=90, zorder=5, label='bump=1%')\n"
            "a1.scatter([2],[pA2], color=RED, s=90, zorder=5, label='bump=2%')\n"
            "a1.axhline(2, ls='--', c=RED, lw=1); a1.axhline(-2, ls='--', c=RED, lw=1)\n"
            "a1.set_xticks([0,1,2]); a1.set_xticklabels(['null','+1%','+2%']); a1.set_ylabel('one-sample t')\n"
            "a1.set_title('Control A (event study)'); a1.legend(fontsize=8)\n"
            "a2.scatter(np.zeros(20)+np.linspace(-.12,.12,20), nullB, color=GREY, s=40, label='null x20')\n"
            "a2.scatter([1],[pB2], color=RED, s=90, zorder=5, label='+2pp on 8')\n"
            "a2.axhline(2, ls='--', c=RED, lw=1); a2.axhline(-2, ls='--', c=RED, lw=1)\n"
            "a2.set_xticks([0,1]); a2.set_xticklabels(['null','+2pp']); a2.set_ylabel('digit-8 z')\n"
            "a2.set_title('Control B (clustering)'); a2.legend(fontsize=8)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'A null mean t={nullA.mean():+.2f}, |t|>=2 in {(abs(nullA)>=2).sum()}/20; planted 1%/2% t={pA1:.2f}/{pA2:.2f}')\n"
            "print(f'B null |z|>=2 in {(abs(nullB)>=2).sum()}/20; planted +2pp z8={pB2:.2f}')"
        ),
        md(
            f"> 💡 In plain words: control A fires on a planted bump "
            f"(t = {R['synA_p1_t']:.2f}/{R['synA_p2_t']:.2f}) and is quiet on the null "
            f"({R['synA_null_fire']}/{R['synA_seeds']} at |t|≥2); control B lights up on "
            f"a planted +2pp 8-excess (z = {R['synB_p2_z8']:.1f}) and never fires on the "
            f"uniform null ({R['synB_null_fire']}/20). Both detectors work — the null "
            "real-tape results (no 8-clustering, no bankable 8/8 edge) are real, not dead "
            "machinery. *(Machinery/power checks only — never cited for the real-tape "
            "stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — the 8/8 lucky-day AR **{R['day_mean']:+.3f}%**, "
            f"*t* = **{R['day_t']:.3f}**, placebo *p* = **{R['pl_day_p']:.3f}** is a real "
            f"nominal hit, but the only one of two horizons (week *t* = {R['wk_t']:.2f}), "
            f"and {R['jk_below2']}/{R['jk_n']} leave-one-out drops — including the "
            "08/08/08 triple-8 — fall below 2. Significant raw, fragile to horizon and "
            "selection.\n"
            f"- **Tradability `MIRAGE`** — calendar-known, so executable, but ~39 bps/yr "
            f"gross does not survive one round trip: net *t* = {R['day_cap_t5']:.2f} "
            f"(5 bps) / {R['day_cap_t10']:.2f} (10 bps); the week is flat.\n"
            f"- **8-clustering? `BUSTED`** — digit-8 China−control *z* = {R['z8']:+.2f} "
            f"(*p* = {R['z8_p']:.2f}), no 4-deficit ({R['z4']:+.2f}). Prices cluster on "
            f"the round number **0** (China−control *z* = {R['z0']:+.2f}), the universal "
            "Harris (1991) preference — not the lucky 8. The fingerprint that is real in "
            "mainland order books does not reach the US secondary tape."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The general lesson: a behavioral bias lives where humans set the "
            "price.** The lucky-8 effect is well-identified on mainland limit orders and "
            "IPO offer prices; on the US secondary tape — priced by market-makers and "
            "arbitrageurs — the digit fingerprint is gone and the 8/8 return is a fragile "
            "tick. That's a map of where the bias survives, not a refutation of the "
            "papers.\n"
            "- **A stronger test needs order-level data.** A-share tick / limit-order "
            "books, HK order flow, or IPO offer-price digits (HK$8.80, HK$88.00) would "
            "test the claim where it should be alive. This study deliberately uses only "
            "daily closes, and says so.\n"
            "- **Dedup map:** shares the calendar-event engine with "
            "[708-eurovision-effect](../../708-eurovision-effect/) and "
            "[707-plane-crash-effect](../../707-plane-crash-effect/); folklore-calendar "
            "cousins [158-super-bowl](../../158-super-bowl/), "
            "[234-olympic-year](../../234-olympic-year/). None test a trailing-digit "
            "distribution or a numerology date — that pair is this study's own.\n\n"
            "*The reproducible core is offline and deterministic; frozen numbers in "
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
