"""Generate the two narrative notebooks for Study 630 (Late Filers, Form NT).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached EDGAR NT panel +
yfinance prices under ../_cache/ and otherwise quote the frozen headline numbers in ``R``
(mirroring docs/results.md). The synthetic positive control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (EDGAR NT panel 2004-01 ->
# 2026-03, yfinance prices as-of 2026-06-30; examples/verify.py output).
R = dict(
    nt_total=102505, ntk=37502, ntq=65003, in_window=89497, mapped=17199, mapped_pct=19.2,
    events=1993, tickers=421, usable=1180, use_ntk=494, use_ntq=686, use_repeat=822,
    px_days=5807, px_names=422, px_start="2003-06-02", px_end="2026-06-30", as_of="2026-06-30",
    fp_events="82ef3cc79145", fp_spy="860d1267b8a2",
    reaction_bps=-33.0, t_reaction=-0.99,
    car_bps=-105.7, t_naive=-0.71,
    ct_bps_day=3.39, ct_hac_t=0.83, ct_ann=8.9, ct_days=5623, ct_avg_names=12.6,
    pseudo_bps=81.9, pseudo_n=714, welch_vs_pseudo=-0.77,
    placebo_bps=381.2, placebo_std=147.1, placebo_seeds=25,
    ntk_car=-205.1, ntk_t=-0.89, ntk_n=494, ntk_hac=0.52,
    ntq_car=-34.2, ntq_t=-0.17, ntq_n=686, ntq_hac=0.66,
    rep_car=58.0, rep_n=822, rep_hac=1.18, fst_car=-481.7, fst_n=358, fst_hac=-0.59,
    rep_minus_fst=539.7, welch_rep=1.85,
    robust=[(20, -146.4, -0.96), (40, -5.5, 0.06), (60, -105.7, 0.83)],  # (window, CAR bps, HAC t)
    costs=[(10.0, 3.0, 14.3), (10.0, 10.0, -152.4), (20.0, 3.0, -5.7), (20.0, 10.0, -172.4)],
    syn=[(0.0, 90.6, 0.94, 0.55), (-20.0, -1152.3, -11.93, -10.42)],  # (edge, CAR, naive t, HAC t)
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Second NT the kill signal?: Busted](https://img.shields.io/badge/Second_NT_the_kill_signal%3F-Busted-8b949e?style=flat-square)\n\n"
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

from late_filers import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    EVENTS, PRICES = data.load_real()
    CARS = st.build_event_cars(EVENTS, PRICES)
else:
    EVENTS = PRICES = CARS = None
print("real NT cache present:", HAVE_REAL,
      "| usable events:", (0 if CARS is None else len(CARS)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# The company that couldn't hand in its homework 📝\n"
            "### When a firm files an *NT 10-K* (\"we can't file our annual report on time\"), "
            "is that a sell signal?\n\n"
            + BADGES +
            "Every listed US company has a legal deadline to publish its annual report (the 10-K) "
            "and its quarterly reports (10-Qs). Miss the deadline and you must file a confession "
            "with the SEC — **Form 12b-25**, which shows up on EDGAR as **\"NT 10-K\"** or "
            "**\"NT 10-Q\"** (*NT* = notification of late filing). It buys a few extra days, and "
            "it tells the whole market: *something inside our accounting broke*.\n\n"
            "The folk rule says the NT is a **sell signal** — audits don't get delayed for good "
            "news. We pulled **every NT filing on EDGAR since 2004** (102,505 of them!), matched "
            "them to stock prices, and measured what happens in the **60 trading days after** the "
            "confession.\n\n"
            "> 📓 Want the *t*-stats, the calendar-time portfolio and the placebos? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **The one warning that changes everything.** We can only price companies that "
            "still have a ticker **today** — barely **1 NT filing in 5** does. The very worst late "
            "filers (frauds, bankruptcies) got **delisted** and vanish from the panel. So whatever "
            "we measure is the NT effect on the *survivors* — its mildest corner. Keep that in "
            "mind for every chart below."
        ),
        code(BOOT_CELL),
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does the stock keep falling after the NT? | On the *survivors* we can measure: "
            f"**not measurably**. About **{R['car_bps'] / 100:+.1f}%** vs the market over ~3 months "
            "— statistically indistinguishable from zero. |\n"
            "| So the folk rule is dead? | Not quite. The academic studies (with point-in-time "
            "data that includes the later-delisted names) DO find a real drop. Our survivors-only "
            "tape simply **can't certify it** — and it's biased toward finding nothing. |\n"
            "| Can you make money shorting NT filers? | **No.** Even granting the (uncertified) "
            "drift, borrow fees and spreads eat it; the names you'd most want to short are the "
            "hardest to borrow — or already delisted. |\n"
            "| Is the SECOND NT in a row the real kill signal? | **Busted.** Repeat offenders "
            "drift *milder* than first-timers. Chronic lateness gets priced as routine. |"
        ),
        md(
            "## What an NT filing looks like on the tape\n\n"
            "The chart tracks the average path of a late filer's stock **relative to the market "
            "(SPY)** from the filing day forward. Day 0 is the confession; anything before the "
            "next day's close is untradable (filings often land at 9 pm)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    # average cumulative market-adjusted path, day 0 .. +60 trading days\n"
            "    rets = PRICES.pct_change(); mkt = rets['SPY'].to_numpy()\n"
            "    paths = []\n"
            "    for ev in CARS.itertuples(index=False):\n"
            "        r = rets[ev.ticker].to_numpy()\n"
            "        d0 = ev.pos - st.LAG\n"
            "        seg = r[d0: d0 + 62] - mkt[d0: d0 + 62]\n"
            "        if len(seg) == 62 and np.isfinite(seg).sum() >= 50:\n"
            "            paths.append(np.nancumsum(np.where(np.isfinite(seg), seg, 0)))\n"
            "    avg = np.mean(paths, axis=0) * 100\n"
            "    fig, ax = plt.subplots()\n"
            "    ax.plot(range(62), avg, color=RED, lw=2)\n"
            "    ax.axhline(0, color='k', lw=.8)\n"
            "    ax.axvline(st.LAG, color=GREY, ls='--', lw=1)\n"
            "    ax.annotate('entry close (day +1)\\n= first tradable moment', xy=(st.LAG, avg[st.LAG]),\n"
            "                xytext=(14, avg[st.LAG] - .55), color=GREY,\n"
            "                arrowprops=dict(arrowstyle='->', color=GREY))\n"
            "    ax.set_xlabel('trading days since the NT filing')\n"
            "    ax.set_ylabel('avg cumulative return vs SPY (%)')\n"
            "    ax.set_title(f'{len(paths):,} NT filings by survivors, 2004-2026: a drizzle, not a cliff')\n"
            "    plt.show()\n"
            "else:\n"
            "    print('cache missing - the frozen numbers in R tell the story')"
        ),
        md(
            "> 💡 **How to read it.** There's a small dip, but on ~1,200 events it never separates "
            "from noise — these are wildly volatile small caps, and the average hides swings of "
            "±80% per name. No statistician would sign off on this line as a \"signal\".\n\n"
            "## Why we still don't call the folk rule *false*\n\n"
            "Three facts sit together:\n\n"
            "1. **Our panel is the survivors.** Only 19.2% of NT filings map to a stock that still "
            "trades today. Firms whose NT preceded fraud revelations or bankruptcy — the cases the "
            "folk rule is *about* — ended delisted and can't be priced here.\n"
            "2. **Published studies with point-in-time data** (Griffin 2003; Bartov, DeFond & "
            "Konchitchki; Duarte-Silva et al. 2013) find genuinely negative returns after late-filing "
            "announcements, continuing for weeks.\n"
            "3. **On random days, our surviving names drift UP** vs the market (that's what "
            "surviving looks like). Their NT windows sit ~5% *below* that personal baseline — the "
            "right direction, just not provable with this noise.\n\n"
            "So the honest reading is: *the literature says real; our survivors-only tape can't "
            "certify it* — the desk stamps that **Weak**, never **Real**.\n\n"
            "## Could you trade it anyway?\n\n"
            "Suppose you granted the drift. A short seller pays the spread **twice** and pays "
            "**borrow fees** for three months — and NT filers are small, distressed, hard-to-borrow "
            "names where borrow runs 10%+/yr when it exists at all. The quants' cost table says the "
            "net edge is ~zero in the friendliest case and **negative** everywhere realistic. Add "
            "that the fattest short payoffs (stocks that went to zero) sit in delisted names you "
            "couldn't borrow — the *trade* is a mirage even where the *information* is real.\n\n"
            "## The second confession\n\n"
            "Folklore says: forgive the first NT (audits hiccup), run from the second. The tape "
            f"says the opposite: repeat offenders drift {R['rep_car'] / 100:+.1f}% vs "
            f"{R['fst_car'] / 100:+.1f}% for first offenses — the chronic laggards are *milder*, "
            "because by the third late quarter everyone expects it. Whatever information an NT "
            "carries is in the **first** one.\n\n"
            "## The bottom line\n\n"
            "A late filing genuinely is bad news — the accounting literature shows it with data we "
            "can't reconstruct for free. But on the survivors a free tape can see, the sell signal "
            "is **not certifiable**, and as a *trade* it's a mirage: costs, borrow and delisting "
            "sit exactly where the edge should be.\n\n"
            "*Research & education, not investment advice. House style: "
            "[METHODOLOGY.md](../../../METHODOLOGY.md).*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata={"language_info": {"name": "python"}})
    nbf.write(nb, os.path.join(HERE, "01_for_the_curious.ipynb"))


# ===========================================================================
# 02 — FOR THE QUANTS
# ===========================================================================
def build_quants():
    cells = [
        md(
            "# Late Filers (Form NT) — the quant teardown 📝\n\n"
            + BADGES +
            "**Claim.** An NT 10-K / NT 10-Q filing (Rule 12b-25 notification of late filing) is a "
            "sell signal: the filer underperforms the market in the weeks after the filing.\n\n"
            "**Design.** One event per (CIK, filing date) of an exact-form NT from the EDGAR "
            "quarterly master index, 2004-01 → 2026-03; CIK→ticker via the SEC registry "
            "(currently-registered names only — the named **survivorship gate**); prices via "
            "yfinance (total-return); benchmark SPY. Short at the close of day +1 (exactly ONE "
            "execution lag), CAR = Σ daily (stock − SPY) over the next 60 trading days. Screens "
            "(stated, none silent): entry ≥ $1, no \\|daily move\\| > 100% in-window, ≥ 80% "
            "coverage. The **primary test is a calendar-time portfolio with a Newey-West HAC "
            "*t*** — NT events cluster around filing deadlines, so the naive cross-sectional *t* "
            "over-counts.\n\n"
            "All real numbers live in the frozen dict `R` (mirror of "
            "[docs/results.md](../docs/results.md)); cells recompute them live from the cache."
        ),
        code(BOOT_CELL),
        md(
            "## 1 — Panel\n\n"
            f"- EDGAR NT index 2002→2026: **{R['nt_total']:,}** filings "
            f"(**{R['ntk']:,}** NT 10-K / **{R['ntq']:,}** NT 10-Q).\n"
            f"- Event window 2004-01→2026-03: **{R['in_window']:,}** filings, of which only "
            f"**{R['mapped']:,} ({R['mapped_pct']:.1f}%)** map to a currently-registered ticker — "
            "the survivorship gate. Delisted offenders (the worst outcomes) are structurally "
            "excluded, so every drift number below is biased **toward zero**.\n"
            f"- Seeded caps (500 tickers / 2,000 events) → {R['events']:,} events on "
            f"{R['tickers']} tickers; **{R['usable']:,}** usable after the stated screens "
            f"({R['use_ntk']:,} NT 10-K / {R['use_ntq']:,} NT 10-Q; {R['use_repeat']:,} repeat "
            "offenders).\n"
            f"- Prices: {R['px_days']:,} days × {R['px_names']} names, {R['px_start']} → "
            f"{R['px_end']}, as-of **{R['as_of']}**. Fingerprints: events `{R['fp_events']}`, "
            f"SPY `{R['fp_spy']}`."
        ),
        code(
            "if HAVE_REAL:\n"
            "    cs = st.cross_sectional_stats(CARS)\n"
            "    ct = st.calendar_time_portfolio(CARS, PRICES)\n"
            "    print(f\"usable events {len(CARS):,} | mean CAR[+1,+60td] {cs['car_bps']:+.1f} bps \"\n"
            "          f\"(naive t {cs['t_naive']:+.2f}) | reaction {cs['reaction_bps']:+.1f} bps\")\n"
            "    print(f\"calendar-time: {ct['mean_bps_day']:+.2f} bps/day vs SPY, HAC t = {ct['hac_t']:+.2f} \"\n"
            "          f\"({ct['ann_pct']:+.1f}%/yr over {ct['n_days']:,} active days)\")\n"
            "else:\n"
            "    print('frozen:', R['car_bps'], 'bps, HAC t', R['ct_hac_t'])"
        ),
        md(
            "## 2 — The drift and the primary test\n\n"
            f"| quantity | value |\n|---|--:|\n"
            f"| announcement reaction (day 0 → entry close, NOT tradable) | "
            f"**{R['reaction_bps']:+.1f} bps** (naive t {R['t_reaction']:+.2f}) |\n"
            f"| mean CAR[+1, +60td] vs SPY | **{R['car_bps']:+.1f} bps** |\n"
            f"| naive cross-sectional *t* (over-counted; reported, not relied on) | "
            f"{R['t_naive']:+.2f} |\n"
            f"| **calendar-time portfolio (PRIMARY)** | **{R['ct_bps_day']:+.2f} bps/day**, "
            f"**HAC *t* = {R['ct_hac_t']:+.2f}** ({R['ct_ann']:+.1f}%/yr, "
            f"avg {R['ct_avg_names']:.1f} names/day) |\n\n"
            "> 💡 **In plain words** — the per-event average is a ~1% dip, but the honest test "
            "(one that doesn't double-count the heavily overlapping deadline-clustered windows) "
            "reads +0.83 — pure noise. The sign even flips between the two aggregations, which "
            "is exactly what \"nothing here\" looks like on a wide-tailed small-cap panel."
        ),
        md(
            "## 3 — Controls: same firms earlier + random dates\n\n"
            f"| control | mean CAR | test |\n|---|--:|--|\n"
            f"| real NT events (n={R['usable']:,}) | **{R['car_bps']:+.1f} bps** | — |\n"
            f"| same firms, 252 td earlier (n={R['pseudo_n']:,}) | {R['pseudo_bps']:+.1f} bps | "
            f"Welch *t* vs events = **{R['welch_vs_pseudo']:+.2f}** |\n"
            f"| random dates × {R['placebo_seeds']} seeds | {R['placebo_bps']:+.1f} bps "
            f"(σ across seeds {R['placebo_std']:.1f}) | — |\n\n"
            "> 💡 **In plain words** — the surviving names drift *up* on random days (survival "
            "bias in person). NT windows sit ~490 bps below that personal baseline — the right "
            "sign for the claim, but Welch *t* = −0.77 is noise. We report the direction; we "
            "don't grade on it."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pse = st.pseudo_events(EVENTS, PRICES)\n"
            "    pcars = st.build_event_cars(pse, PRICES)\n"
            "    wt, m_ev, m_ps = st.welch_t(CARS['car'].to_numpy(), pcars['car'].to_numpy())\n"
            "    fig, ax = plt.subplots()\n"
            "    bins = np.linspace(-1.5, 1.5, 49)\n"
            "    ax.hist(np.clip(pcars['car'], -1.5, 1.5), bins=bins, alpha=.55, color=GREY,\n"
            "            density=True, label=f'pseudo (1y earlier), mean {m_ps*1e4:+.0f} bps')\n"
            "    ax.hist(np.clip(CARS['car'], -1.5, 1.5), bins=bins, alpha=.55, color=RED,\n"
            "            density=True, label=f'NT events, mean {m_ev*1e4:+.0f} bps')\n"
            "    ax.axvline(0, color='k', lw=.8)\n"
            "    ax.set_xlabel('CAR[+1,+60td] vs SPY (clipped at +/-150%)'); ax.set_ylabel('density')\n"
            "    ax.legend(); ax.set_title(f'Event vs matched self-control (Welch t = {wt:+.2f})')\n"
            "    plt.show()\n"
            "else:\n"
            "    print('frozen: Welch t vs pseudo =', R['welch_vs_pseudo'])"
        ),
        md(
            "## 4 — Splits & robustness\n\n"
            f"| slice | CAR (bps) | n | cal-time HAC *t* |\n|---|--:|--:|--:|\n"
            f"| NT 10-K (annual miss) | {R['ntk_car']:+.1f} | {R['ntk_n']:,} | "
            f"{R['ntk_hac']:+.2f} |\n"
            f"| NT 10-Q (quarterly miss) | {R['ntq_car']:+.1f} | {R['ntq_n']:,} | "
            f"{R['ntq_hac']:+.2f} |\n\n"
            f"| window | CAR (bps) | cal-time HAC *t* |\n|---|--:|--:|\n"
            + "".join(f"| {w} td | {c:+.1f} | {t:+.2f} |\n" for w, c, t in R["robust"]) +
            "\n> 💡 **In plain words** — no slice and no horizon clears (or approaches) the "
            "*t* ≥ 2 bar. The 10-K miss looks heavier than the 10-Q miss in the point estimate, "
            "as the literature would predict, but neither certifies."
        ),
        md(
            "## 5 — Third axis: is the SECOND consecutive NT the kill signal?\n\n"
            f"| group | CAR (bps) | n | cal-time HAC *t* |\n|---|--:|--:|--:|\n"
            f"| repeat offenders (prior NT ≤ 15 months) | **{R['rep_car']:+.1f}** | "
            f"{R['rep_n']:,} | {R['rep_hac']:+.2f} |\n"
            f"| first offenses | **{R['fst_car']:+.1f}** | {R['fst_n']:,} | {R['fst_hac']:+.2f} |\n\n"
            f"Difference (repeat − first): **{R['rep_minus_fst']:+.1f} bps**, Welch *t* = "
            f"**{R['welch_rep']:+.2f}** — sub-2 AND the **wrong sign** for the folklore: chronic "
            "lateness is priced as routine, first confessions carry what little drift there is. "
            "**Busted.**"
        ),
        md(
            "## 6 — Tradability: the short pays twice, then pays the borrow\n\n"
            f"Gross short capture = {-R['car_bps']:+.1f} bps per event (60-td hold), *granting* "
            "the uncertified drift. Net after 2 × one-way cost + borrow held 60/252 of a year:\n\n"
            "| one-way cost | borrow %/yr | net (bps/event) |\n|--:|--:|--:|\n"
            + "".join(f"| {c:.0f} bps | {b:.0f}% | **{n:+.1f}** |\n" for c, b, n in R["costs"]) +
            "\n> 💡 **In plain words** — +14 bps per event in the friendliest case (large-cap "
            "costs on small-cap shorts — already a fiction) and negative everywhere realistic. "
            "Survivorship cuts both ways here: it understates the *signal* (delisted disasters "
            "missing) and overstates the *accessibility* (those same names had no borrow). "
            "**Mirage.**"
        ),
        md(
            "## 7 — Synthetic positive control (machinery proof only)\n\n"
            "Deterministic beta-1 panel with planted NT-style events: with **0** planted drift "
            "the machinery must stay silent; with **−20 bps/day** planted over the measured "
            "window it must light up.\n\n"
            "| planted drift | CAR (bps) | naive *t* | cal-time HAC *t* |\n|--:|--:|--:|--:|\n"
            + "".join(f"| {e:+.0f} bps/d | {c:+.1f} | {t:+.2f} | {h:+.2f} |\n"
                      for e, c, t, h in R["syn"]) +
            "\nThe real-tape non-result is not a power failure of the harness. *(Never cited in "
            "support of a Signal stamp.)*"
        ),
        code(
            "for edge in (0.0, -20.0):\n"
            "    ev, px = data.synthetic_world(edge_bps_day=edge, seed=630)\n"
            "    cars = st.build_event_cars(ev, px)\n"
            "    cs2 = st.cross_sectional_stats(cars)\n"
            "    ct2 = st.calendar_time_portfolio(cars, px)\n"
            "    print(f\"planted {edge:+6.1f} bps/d: CAR {cs2['car_bps']:+8.1f} bps  \"\n"
            "          f\"naive t {cs2['t_naive']:+5.2f}  cal-time HAC t {ct2['hac_t']:+6.2f}\")"
        ),
        md(
            "## Verdict\n\n"
            f"- **Signal — WEAK.** Literature (Griffin 2003; Bartov et al.; Duarte-Silva et al. "
            f"2013) says real; this tape can't certify it: calendar-time HAC *t* = "
            f"{R['ct_hac_t']:+.2f}, self-control Welch *t* = {R['welch_vs_pseudo']:+.2f}, no "
            "slice near 2. The panel is severely survivor-biased (19.2% ticker coverage; the "
            "delisting disasters are structurally excluded), biasing the drift toward zero — "
            "\"literature says real; this survivors-only tape alone can't certify it\".\n"
            "- **Tradability — MIRAGE.** Net short edge ≈ +14 bps/event at best, negative at "
            "realistic small-cap borrow; the names that would carry the edge are hard-to-borrow "
            "or delisted.\n"
            f"- **Second NT the kill signal? — BUSTED.** Repeat − first = "
            f"{R['rep_minus_fst']:+.1f} bps (Welch *t* = {R['welch_rep']:+.2f}): wrong sign for "
            "the folklore — the first confession carries whatever information exists.\n\n"
            "*Research & education, not investment advice.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata={"language_info": {"name": "python"}})
    nbf.write(nb, os.path.join(HERE, "02_for_the_quants.ipynb"))


if __name__ == "__main__":
    build_curious()
    build_quants()
    print("notebooks written")
