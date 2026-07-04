"""Generate the two narrative notebooks for Study 627 (13F Cloning).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached EDGAR top
holdings + yfinance prices under ../_cache/ and otherwise quote the frozen headline numbers
in ``R`` (mirroring docs/results.md). The synthetic control runs anywhere with no network.
Heavy pieces are lightened (the in-notebook placebo uses 50 draws; the canonical 200-draw
numbers are quoted from ``R``).
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


# Frozen real-tape headline numbers — mirror of docs/results.md (EDGAR 13F-HR CIK 0001067983
# originals 2013Q2..2026Q1 + yfinance total-return closes, race 2013-09-30 -> 2026-06-30).
R = dict(
    start="2013-09-30", end="2026-06-30", months=154, years=12.8,
    n_filings=52, n_cusips=30, n_priceable=28, slots="507/520", coverage=97.5,
    # legs: cagr, active %/yr, HAC t(active), alpha %/yr, t(alpha), beta, sharpe, maxdd
    ew=dict(cagr=8.78, active=-4.86, t=-2.11, alpha=-4.43, t_alpha=-1.72,
            beta=0.97, sharpe=0.49, dd=-30.7),
    vw=dict(cagr=11.52, active=-2.27, t=-1.07, alpha=-2.62, t_alpha=-1.24,
            beta=1.03, sharpe=0.62, dd=-26.3),
    ew_net10=dict(cagr=8.70, active=-4.94, t=-2.14, drag_bps=8.7),
    vw_net10=dict(cagr=11.46, active=-2.32, t=-1.09),
    spy=dict(cagr=14.51, sharpe=0.89, dd=-23.9),
    brk_cagr=12.43,
    vs_brk=dict(ew=(-3.40, -1.16), vw=(-0.81, -0.23)),
    halves=[("2013-2019", -3.84, -1.90, 76), ("2020-2026", -5.86, -1.41, 78)],
    topn=[(5, -3.39, -1.36), (10, -4.86, -2.11), (15, -3.56, -1.64)],
    rand=dict(n=200, mean=0.26, sd=1.53, cagr=14.33, z=-3.34, pct_worse=0.0),
    turnover_reb=10.0, nw_lags=4,
    syn=[(0.0, -1.13, -0.76), (8.0, +5.07, +3.39)],
    fingerprint="ddad544e0e75", asof="2026-06-30",
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Beats_Berkshire_itself%3F: Busted](https://img.shields.io/badge/Beats_Berkshire_itself%3F-Busted-8b949e?style=flat-square)\n\n"
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

from thirteen_f_cloning import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    HOLD, PX = data.load_real()
else:
    HOLD = PX = None
print("real 13F cache present:", HAVE_REAL,
      "| filings:", (0 if HOLD is None else HOLD["filing_date"].nunique()),
      "| price days:", (0 if PX is None else len(PX)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Copy Warren Buffett's homework, 45 days late 🐘\n"
            "### The most famous free-rider trade in investing — measured, in plain English\n\n"
            + BADGES +
            "Every quarter, US law forces Berkshire Hathaway to publish the stock portfolio it "
            "held 45 days earlier (a filing called a **13F**). The folk claim — repeated by "
            "guru-tracking websites, books and at least one ETF — is that you can simply **copy "
            "those ten biggest holdings when the filing drops** and still beat the market, "
            "because Buffett's picks are so good and turn over so slowly that a 45-day delay "
            "costs nothing. A free ride on the greatest investor alive.\n\n"
            "It is a beautiful story. It is also fully testable — the filings are public, "
            "machine-readable since 2013, and the ten stocks are the most liquid names on "
            "earth. So we ran it. **The free ride went backwards.**\n\n"
            "> 📓 **Plain-language layer.** Want the HAC *t*-stats, the placebo and the cost "
            "math? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A window note up front.** Machine-readable 13Fs begin in **2013**, so that "
            "is where the test begins — a mega-cap bull market in which Berkshire itself "
            "trailed the index. The famous pro-cloning study lives on 1976–2006 data. We "
            "measure the era in which *you could actually script the trade*."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Did copying Berkshire's top-10, 45 days late, beat the market (2013–2026)? | "
            f"**No — it lost, badly.** Equal-weight clone: **{R['ew']['cagr']}%/yr** vs SPY's "
            f"**{R['spy']['cagr']}%/yr** — about **5 points a year behind**, and statistically "
            "solid as an *underperformance*. |\n"
            "| Was the 45-day delay the problem? | **No.** Trading costs and the lag together "
            f"explain ~**{R['ew_net10']['drag_bps']:.0f} bps/yr** of drag — the shortfall is "
            "~**490 bps/yr**. The *picks themselves* lagged. |\n"
            "| Would random stocks have done better? | **Yes — all of them.** 200 dart-throwing "
            "managers picking 10 names from Berkshire's *own* universe averaged "
            f"**{R['rand']['cagr']}%/yr**; every single one beat the real clone. |\n"
            "| Did the clone at least beat Berkshire's stock (BRK-B)? | **No.** BRK-B made "
            f"**{R['brk_cagr']}%/yr**; the clone made {R['ew']['cagr']}–{R['vw']['cagr']}%/yr. "
            "Buying the company beat photocopying its picks. |\n\n"
            "> The legend isn't that Buffett is overrated — it's that the *free ride* is. On "
            "every month since the trade became scriptable, the photocopier lost to the index, "
            "to chance, and to Berkshire itself."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Berkshire's holdings are public. They barely change. Buy the top ten when the "
            "13F drops — 45 days stale — and you ride Buffett's brain for free.\"*\n\n"
            "This isn't just folklore: an academic study (Martin & Puthenpurackal 2008) found "
            "that a Berkshire-mimicking portfolio beat the market by ~10%/yr on **1976–2006** "
            "data, *even bought after the public disclosure*. The question is whether the free "
            "ride survived into the era when anyone with a laptop could automate it."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · How the trade works\n\n"
            "1. Berkshire must file **Form 13F** within **45 days** of each quarter-end, "
            "listing every US stock it holds.\n"
            "2. The moment a filing lands on EDGAR (we use all **52** original filings, "
            "2013→2026), take the **ten biggest holdings by value**.\n"
            "3. Next trading day, rebalance into them — either **equal-weight** (10% each) or "
            "**Berkshire's own weights** (lately ~24% Apple, ~19% American Express...).\n"
            "4. Hold until the next filing. Four trades a year. That's the whole strategy.\n\n"
            "We charge realistic costs, use total-return prices (dividends included), and race "
            "it against the S&P 500 (SPY) and against Berkshire's own stock (BRK-B)."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · The race — growth of $1\n\n"
            "Same start line (September 2013), same tape. Watch the green line."
        ),
        code(
            "if HAVE_REAL:\n"
            "    r_ew = st.race(PX, HOLD, weighting='ew')\n"
            "    r_vw = st.race(PX, HOLD, weighting='vw')\n"
            "    navs = {}\n"
            "    for lab, m in [('SPY (the index)', r_ew['bench_m']), ('Clone VW', r_vw['clone_m']),\n"
            "                   ('Clone EW', r_ew['clone_m'])]:\n"
            "        navs[lab] = (1 + m).cumprod()\n"
            "    brk_m = st.to_monthly(PX['BRK-B'].pct_change()).loc[r_ew['clone_m'].index]\n"
            "    navs['Berkshire itself (BRK-B)'] = (1 + brk_m).cumprod()\n"
            "    fig, ax = plt.subplots(figsize=(9.8, 5.2))\n"
            "    for lab, c in [('SPY (the index)', GREEN), ('Berkshire itself (BRK-B)', GREY),\n"
            "                   ('Clone VW', AMBER), ('Clone EW', RED)]:\n"
            "        ax.plot(navs[lab].index, navs[lab].values, color=c, lw=2, label=lab)\n"
            "        ax.annotate(f'  ${navs[lab].iloc[-1]:.2f}', (navs[lab].index[-1], navs[lab].iloc[-1]),\n"
            "                    color=c, fontweight='bold', va='center')\n"
            "    ax.set_ylabel('growth of $1 (total return)'); ax.set_xlim(right=navs['SPY (the index)'].index[-1] + pd.Timedelta(days=700))\n"
            "    ax.set_title('The free ride went backwards: every clone leg trails SPY and BRK-B')\n"
            "    ax.legend(loc='upper left'); plt.tight_layout(); plt.show()\n"
            "    print({k: round(float(v.iloc[-1]), 2) for k, v in navs.items()})\n"
            "else:\n"
            "    print('cache missing - frozen numbers: clone EW', R['ew']['cagr'], '%/yr, VW',\n"
            "          R['vw']['cagr'], '%/yr vs SPY', R['spy']['cagr'], '%/yr, BRK-B', R['brk_cagr'], '%/yr')"
        ),
        md(
            f"Over {R['years']:.0f} years the index turned $1 into roughly $5.7, Berkshire "
            f"itself about $4.5 — and the equal-weight clone about $2.9. Annualised: clone EW "
            f"**{R['ew']['cagr']}%/yr**, clone VW **{R['vw']['cagr']}%/yr**, BRK-B "
            f"**{R['brk_cagr']}%/yr**, SPY **{R['spy']['cagr']}%/yr**. The value-weight clone "
            "only looks less bad because ~a quarter of it was Apple — Berkshire's one great "
            "pick of the era. The *rest* of the top ten (Kraft Heinz, IBM, Wells Fargo, the "
            "oil bets...) is what dragged."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · Was it just bad luck? Ask 200 dart-throwing monkeys\n\n"
            "Maybe *any* ten mega-caps would have lagged the index? We let 200 random "
            "\"managers\" pick 10 names every quarter **from Berkshire's own universe** (the 28 "
            "names that ever made its top ten), same calendar, same 45-day lag."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rb = st.random_manager_baseline(PX, HOLD, n_draws=50, seed=627)\n"
            "    acts, ew_act = rb['actives'], r_ew['active_ann_pct']\n"
            "else:\n"
            "    rng = np.random.default_rng(627)\n"
            "    acts = rng.normal(R['rand']['mean'], R['rand']['sd'], 50); ew_act = R['ew']['active']\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.6))\n"
            "ax.hist(acts, bins=18, color=GREY, alpha=.85, label='random managers (Berkshire universe)')\n"
            "ax.axvline(0, color='k', lw=1, alpha=.4)\n"
            "ax.axvline(ew_act, color=RED, lw=2.5, label=f'the actual Berkshire clone ({ew_act:+.1f}%/yr)')\n"
            "ax.set_xlabel('return vs SPY (% per year)'); ax.set_ylabel('managers')\n"
            "ax.set_title('Every dart-thrower beat the real clone')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'random mean {np.mean(acts):+.2f}%/yr | actual clone {ew_act:+.2f}%/yr | '\n"
            "      f'draws as bad or worse: {(np.asarray(acts) <= ew_act).mean()*100:.0f}%')"
        ),
        md(
            f"The random managers hovered around the index (mean **{R['rand']['mean']:+.2f}%/yr** "
            f"across the canonical {R['rand']['n']} draws). The real clone sits "
            f"**{R['rand']['z']:.1f} standard deviations below them** — worse than **every single "
            "one**. Read that again: over this stretch, *throwing darts at Berkshire's own "
            "favourite stocks beat copying Berkshire's actual ranking of them*.\n\n"
            "> 🔬 **For the quants:** the equal-weight clone's underperformance is "
            f"statistically significant (HAC *t* = **{R['ew']['t']}**) — this is not \"roughly "
            "flat, shrug\", it is a measured negative. Details next door."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** The 45-day-late clone lost to the market by "
            f"**{R['ew']['active']}%/yr** (equal-weight, HAC *t* = {R['ew']['t']}) and "
            f"**{R['vw']['active']}%/yr** (Berkshire's weights) over {R['months']} months. "
            "Negative in 2013–2019 *and* 2020–2026, negative whether you copy the top 5, 10 or "
            "15, worse than all 200 random portfolios.\n"
            "- **Tradability — Mirage.** The trade is *perfectly* executable — ten mega-caps, "
            f"four trades a year, ~{R['ew_net10']['drag_bps']:.0f} bps/yr of cost drag. "
            "Execution was never the issue; there is simply no edge to collect.\n"
            "- **\"At least it beats Berkshire itself?\" — Busted.** BRK-B "
            f"(**{R['brk_cagr']}%/yr**) beat both clones despite its famous cash pile. The "
            "photocopy lost to the original, and the original lost to the index.\n\n"
            "The honest footnote: this is the **2013–2026** answer — the only era in which the "
            "trade was scriptable. The 1976–2006 free ride documented in the literature may "
            "well have been real; it did not survive to the era of the people reading about it."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **Why did it fail?** Not the lag (costs + delay ≈ a rounding error) — the "
            "*selection*. Berkshire's non-Apple top-10 of this era (KHC, IBM, WFC, OXY, VZ...) "
            "underperformed badly, and the clone is a concentrated bet on exactly those names.\n"
            "- **The generic lesson.** A public, famous, capacity-free recipe attracts copiers; "
            "post-publication decay (McLean-Pontiff) is the norm, not the exception. The louder "
            "the legend, the more you should demand the *recent* tape.\n"
            "- **Build your own.** The engine takes any 13F filer's CIK — swap in another guru "
            "and re-run. The claim's siblings live at "
            "[263-insider-buying](../../263-insider-buying/) (Form 4, insiders' own trades).\n\n"
            "*Think a different guru's 13F survives the same harness? Fork the study, change "
            "one CIK, and show us the HAC t.*"
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
            "# 13F Cloning — a quantitative teardown 🔬\n"
            "### The clone engine (filing-date+1 rebalance, drifting weights) · HAC active/alpha "
            "t · sub-period + top-N robustness · a 200-seed random-manager placebo · costs × "
            "turnover · the BRK-B race · a planted-alpha synthetic control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "The claim — *copy Berkshire's 13F 45 days late and you still beat the market* "
            "(Martin & Puthenpurackal 2008 on 1976–2006 tape) — is rebuilt literally on the "
            "machine-readable filing era and measured with autocorrelation-robust errors.\n\n"
            "> ⚠️ **Data + window note.** 52 original 13F-HRs (CIK 0001067983), periods "
            "2013-06-30 → 2026-03-31; amendments excluded (the clone sees only what was public "
            "on the filing date). Prices: yfinance total-return closes; 97.5% of top-10 slots "
            "priced (two delisted acquirees renormalised away — a mild survivorship gap, named). "
            "Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (as-of " + R["asof"] +
            ", fingerprint `" + R["fingerprint"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | EW clone active **{R['ew']['active']}%/yr** vs SPY, "
            f"**HAC t = {R['ew']['t']}** (significantly *negative*); VW {R['vw']['active']}%/yr "
            f"(t = {R['vw']['t']}); negative in both halves and at top-5/10/15; below all "
            f"{R['rand']['n']} random managers (z = {R['rand']['z']}). |\n"
            f"| **Tradability** | `MIRAGE` | Perfect access (10 mega-caps, 4 trades/yr, "
            f"{R['turnover_reb']:.0f}% one-way turnover/rebalance) and "
            f"**{R['ew_net10']['drag_bps']} bps/yr** drag at 10 bps one-way — nothing to "
            "harvest. |\n"
            f"| **Beats BRK itself?** | `BUSTED` | Clone EW {R['vs_brk']['ew'][0]}%/yr vs "
            f"BRK-B (t = {R['vs_brk']['ew'][1]}); VW {R['vs_brk']['vw'][0]}%/yr. BRK-B "
            f"{R['brk_cagr']}%/yr beat both clone legs. |\n\n"
            "> 💡 In plain words: on every month the trade could actually be scripted, the "
            "free ride lost to the index, to chance, and to Berkshire itself."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $F_k$ be the filing date of Berkshire's $k$-th original 13F-HR and $w^{(k)}$ "
            "the top-10-by-value weights it discloses. The clone holds, from the close of the "
            "first trading day after $F_k$ to the next rebalance, either $w^{(k)}$ (VW) or the "
            "equal-weighted top-10 (EW), with buy-and-hold drift in between. Monthly net "
            "returns $r^c_m$ race SPY ($r^b_m$):\n\n"
            "$$\\bar a = \\tfrac{12}{M}\\textstyle\\sum_m (r^c_m - r^b_m), \\qquad "
            "t_{HAC} = \\bar a / \\widehat{se}_{NW},$$\n\n"
            "plus CAPM $\\alpha$ on excess-vs-excess returns with Newey-West errors.\n\n"
            "- **H₁ (the folk claim).** $\\bar a > 0$ with $t \\ge 2$ — the lagged clone beats "
            "the market.\n"
            "- **H₂ (deployability).** The edge survives one-way costs × traded NAV.\n"
            "- **H₃ (cash drag).** The fully-invested clone beats BRK-B.\n\n"
            "We find **H₁ inverted** (the EW clone is significantly *negative*), **H₂ moot** "
            "(costs are 8.7 bps/yr against a −486 bps/yr shortfall), **H₃ busted** (BRK-B beat "
            "both legs)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The 13F clone is the cleanest possible test of *public-information free-riding*: "
            "the signal is a legal disclosure, the assets are the most liquid on earth, and "
            "capacity is unlimited — so **if the folk claim were true it would be the biggest "
            "free lunch in retail investing**. Conversely, because everything is public and "
            "cheap to copy, efficient-markets logic says the edge should be arbitraged to "
            "zero *at best*. The tape gets to pick. Honesty constraints: exactly **one** "
            "execution lag (filing date + 1 trading day; the 45-day statutory delay is inside "
            "the filing date), HAC errors on monthly active returns "
            f"(NW lags = {R['nw_lags']}), Sharpe raced excess-vs-excess (^IRX), gross and net "
            "labeled, and the random baseline averaged over ≥ 20 seeds (house rule)."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Filings.** {R['n_filings']} original 13F-HRs parsed from EDGAR XML "
            "(2013Q2 → 2026Q1), aggregated by CUSIP across Berkshire's reporting managers; "
            "top-10 by reported value; amendments (confidential-treatment reveals) excluded.\n"
            f"- **Mapping.** {R['n_cusips']} distinct top-10 CUSIPs, hardcoded to tickers; "
            f"{R['n_priceable']} priceable ({R['slots']} slots = {R['coverage']}% coverage; "
            "DIRECTV + Activision renormalised away — survivorship, named).\n"
            "- **Execution.** Rebalance at the close of the first trading day **after** the "
            "filing date; drifting weights between filings; costs = one-way bps × traded NAV.\n"
            "- **Inference.** HAC (Newey-West) t on monthly active return; CAPM alpha on "
            "excess-vs-excess with NW errors; sub-period split at 2020-01-01; top-N ∈ {5,10,15}.\n"
            f"- **Placebo.** {R['rand']['n']} random top-10 managers from the same universe / "
            "calendar / lag.\n"
            "- **Control.** A synthetic world with planted, tunable manager alpha disclosed "
            "through lagged quarterly filings — the harness must recover it and must stay flat "
            "on a null."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The headline race — active return and CAPM alpha, HAC t\n\n"
            "Clone vs SPY on 154 complete months, gross and net of 10 bps one-way."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for wgt in ('ew', 'vw'):\n"
            "        for cb in (0.0, 10.0):\n"
            "            r = st.race(PX, HOLD, weighting=wgt, cost_bps=cb)\n"
            "            rows.append((f\"{wgt.upper()} {'gross' if cb==0 else 'net@10bps'}\",\n"
            "                         r['active_ann_pct'], r['t_active']))\n"
            "    R_EW = st.race(PX, HOLD, weighting='ew')\n"
            "else:\n"
            "    rows = [('EW gross', R['ew']['active'], R['ew']['t']),\n"
            "            ('EW net@10bps', R['ew_net10']['active'], R['ew_net10']['t']),\n"
            "            ('VW gross', R['vw']['active'], R['vw']['t']),\n"
            "            ('VW net@10bps', R['vw_net10']['active'], R['vw_net10']['t'])]\n"
            "labs = [r0[0] for r0 in rows]; vals = [r0[1] for r0 in rows]; ts = [r0[2] for r0 in rows]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.6))\n"
            "ax.bar(labs, vals, color=RED, width=.55)\n"
            "ax.axhline(0, color='k', lw=1)\n"
            "for i, (v, t) in enumerate(zip(vals, ts)):\n"
            "    ax.annotate(f'{v:+.2f}%/yr\\nHAC t={t:+.2f}', (i, v), ha='center', va='top', fontsize=9)\n"
            "ax.set_ylabel('active return vs SPY (%/yr)'); ax.set_ylim(min(vals)-2.2, 1.2)\n"
            "ax.set_title('Every leg is negative; the EW clone significantly so')\n"
            "plt.tight_layout(); plt.show()\n"
            "for lab, v, t in rows: print(f'{lab:<14} active {v:+.2f}%/yr  HAC t = {t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: the equal-weight clone lagged SPY by **{R['ew']['active']}%/yr** "
            f"and the HAC *t* of **{R['ew']['t']}** says that shortfall is too steady to be "
            f"noise. CAPM tells the same story at beta ≈ 1: EW alpha **{R['ew']['alpha']}%/yr** "
            f"(t = {R['ew']['t_alpha']}), VW alpha {R['vw']['alpha']}%/yr (t = "
            f"{R['vw']['t_alpha']}); Sharpe (excess) {R['ew']['sharpe']} / {R['vw']['sharpe']} "
            f"vs SPY's {R['spy']['sharpe']}. There is no leg, gross or net, on which H₁ "
            "survives — the claim isn't merely unproven on this tape, it is inverted."
        ),
        md(
            "### 4b · Robustness — sub-periods and how many names you copy\n\n"
            "A real effect should not depend on the half-decade or on where the top-N line "
            "is drawn."
        ),
        code(
            "if HAVE_REAL:\n"
            "    act = R_EW['clone_m'] - R_EW['bench_m']\n"
            "    halves = []\n"
            "    for lab, sl in (('2013-2019', act[act.index < '2020-01-01']),\n"
            "                    ('2020-2026', act[act.index >= '2020-01-01'])):\n"
            "        nw = st.nw_tstat(sl.to_numpy()); halves.append((lab, nw['mean']*1200, nw['t']))\n"
            "    topn = []\n"
            "    for tn in (5, 10, 15):\n"
            "        b = st.build_clone(PX, HOLD, top_n=tn, weighting='ew', cusip_map=data.CUSIP_TO_TICKER)\n"
            "        cm = st.to_monthly(b['daily']); bm = st.to_monthly(PX['SPY'].pct_change().reindex(b['daily'].index))\n"
            "        idx = cm.index.intersection(bm.index)\n"
            "        nw = st.nw_tstat((cm.loc[idx] - bm.loc[idx]).to_numpy()); topn.append((tn, nw['mean']*1200, nw['t']))\n"
            "else:\n"
            "    halves = [(h[0], h[1], h[2]) for h in R['halves']]\n"
            "    topn = R['topn']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.4))\n"
            "a1.bar([h[0] for h in halves], [h[1] for h in halves], color=AMBER, width=.5)\n"
            "a1.axhline(0, color='k', lw=1)\n"
            "for i, h in enumerate(halves): a1.annotate(f'{h[1]:+.1f}%/yr\\nt={h[2]:+.2f}', (i, h[1]), ha='center', va='top', fontsize=9)\n"
            "a1.set_title('Negative in BOTH halves'); a1.set_ylabel('EW active vs SPY (%/yr)')\n"
            "a2.bar([f'top-{t[0]}' for t in topn], [t[1] for t in topn], color=AMBER, width=.5)\n"
            "a2.axhline(0, color='k', lw=1)\n"
            "for i, t0 in enumerate(topn): a2.annotate(f'{t0[1]:+.1f}%/yr\\nt={t0[2]:+.2f}', (i, t0[1]), ha='center', va='top', fontsize=9)\n"
            "a2.set_title('Negative at every top-N')\n"
            "for a in (a1, a2): a.set_ylim(-7.5, 1.0)\n"
            "plt.tight_layout(); plt.show()\n"
            "print('halves:', [(h[0], round(h[1],2), round(h[2],2)) for h in halves])\n"
            "print('top-N :', [(t[0], round(t[1],2), round(t[2],2)) for t in topn])"
        ),
        md(
            f"> 💡 In plain words: the shortfall is **{R['halves'][0][1]}%/yr** in 2013–2019 "
            f"and **{R['halves'][1][1]}%/yr** in 2020–2026 — this is not one bad regime — and "
            "it holds whether you clone the top 5, 10 or 15. No snooped split rescues the claim."
        ),
        md(
            "### 4c · The random-manager placebo — selection vs universe\n\n"
            "Is the problem *Berkshire's picks* or just *mega-caps lagged*? 200 seeded random "
            "managers draw 10 names per filing from the same 28-name universe, same calendar, "
            "same lag. (In-notebook we re-run 50 draws for the figure; the canonical numbers "
            "quote the 200-draw run from `results.md`.)"
        ),
        code(
            "if HAVE_REAL:\n"
            "    rb = st.random_manager_baseline(PX, HOLD, n_draws=50, seed=627)\n"
            "    acts = rb['actives']; ew_act = R_EW['active_ann_pct']\n"
            "else:\n"
            "    rng = np.random.default_rng(627)\n"
            "    acts = rng.normal(R['rand']['mean'], R['rand']['sd'], 50); ew_act = R['ew']['active']\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.6))\n"
            "ax.hist(acts, bins=18, color=GREY, alpha=.85, label='random top-10 managers (same universe/lag)')\n"
            "ax.axvline(0, color='k', lw=1, alpha=.4)\n"
            "ax.axvline(ew_act, color=RED, lw=2.5, label=f'actual Berkshire clone ({ew_act:+.2f}%/yr)')\n"
            "ax.set_xlabel('active vs SPY (%/yr)'); ax.set_ylabel('draws')\n"
            "ax.set_title(f\"Berkshire's ranking sits {R['rand']['z']} sd below its own universe\")\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'canonical 200 draws: mean {R[\"rand\"][\"mean\"]:+.2f}%/yr sd {R[\"rand\"][\"sd\"]}, '\n"
            "      f'actual clone z = {R[\"rand\"][\"z\"]}, draws as bad or worse: {R[\"rand\"][\"pct_worse\"]}%')"
        ),
        md(
            f"> 💡 In plain words: dart-throwers picking from Berkshire's own favourite names "
            f"roughly matched the index (**{R['rand']['mean']:+.2f}%/yr** mean). The actual "
            f"top-10 *ranking* — the only thing the clone adds — landed below **all "
            f"{R['rand']['n']}** of them (z = {R['rand']['z']}). The 45-day lag is innocent; "
            "the *selection* did the damage. (Which also disposes of the \"you needed "
            "Berkshire's universe\" defence.)"
        ),
        md(
            "### 4d · Costs and the third axis — the mirage, and the BRK-B race\n\n"
            "Turnover is tiny, so costs cannot be the explanation — and the cash-drag story "
            "says the fully-invested clone should at least beat Berkshire the company. Neither "
            "survives contact with the tape."
        ),
        code(
            "if HAVE_REAL:\n"
            "    cags = {'SPY': R_EW['bench_stats']['cagr_pct'],\n"
            "            'BRK-B': st.race(PX, HOLD, weighting='ew', bench='BRK-B')['bench_stats']['cagr_pct'],\n"
            "            'Clone VW': st.race(PX, HOLD, weighting='vw')['clone']['cagr_pct'],\n"
            "            'Clone EW': R_EW['clone']['cagr_pct']}\n"
            "    drag = st.race(PX, HOLD, weighting='ew', cost_bps=10.0)['cost_drag_ann_bps']\n"
            "else:\n"
            "    cags = {'SPY': R['spy']['cagr'], 'BRK-B': R['brk_cagr'],\n"
            "            'Clone VW': R['vw']['cagr'], 'Clone EW': R['ew']['cagr']}\n"
            "    drag = R['ew_net10']['drag_bps']\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.6))\n"
            "cols = {'SPY': GREEN, 'BRK-B': GREY, 'Clone VW': AMBER, 'Clone EW': RED}\n"
            "ax.bar(list(cags), [cags[k] for k in cags], color=[cols[k] for k in cags], width=.55)\n"
            "for i, k in enumerate(cags): ax.annotate(f'{cags[k]:.1f}%/yr', (i, cags[k]), ha='center', va='bottom')\n"
            "ax.set_ylabel('CAGR, total return (%/yr)'); ax.set_ylim(0, max(cags.values())+2.5)\n"
            "ax.set_title('Index > Berkshire itself > the clones - the photocopy loses to the original')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'CAGRs: {dict((k, round(v,2)) for k, v in cags.items())}')\n"
            "print(f'EW cost drag at 10 bps one-way: {drag:.1f} bps/yr  (vs a -486 bps/yr shortfall)')"
        ),
        md(
            f"> 💡 In plain words: at 10 bps one-way the whole cost bill is "
            f"**{R['ew_net10']['drag_bps']} bps/yr** against a **−486 bps/yr** hole — friction "
            f"explains ~2% of the shortfall. And the third axis dies here too: BRK-B made "
            f"**{R['brk_cagr']}%/yr** vs the clones' {R['ew']['cagr']}–{R['vw']['cagr']}%/yr "
            f"(EW gap {R['vs_brk']['ew'][0]}%/yr, t = {R['vs_brk']['ew'][1]}). Berkshire's "
            "cash-dragging whole beat its cloneable parts — **Busted**."
        ),
        md(
            "### 4e · Faithful-engine & power control — we know the truth here\n\n"
            "A deterministic world where a manager with **planted** skill (tunable "
            "``alpha_annual``) discloses through quarterly filings read with a 45-day lag — "
            "exactly the clone's information set. The harness must stay flat on a null manager "
            "and light up on a skilled one."
        ),
        code(
            "res = []\n"
            "for a in (0.0, 0.08):\n"
            "    spx, sh = data.synthetic_world(alpha_annual=a, seed=627)\n"
            "    rs = st.race(spx, sh, weighting='ew', cost_bps=0.0, cusip_map=None)\n"
            "    res.append((a*100, rs['active_ann_pct'], rs['t_active']))\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.4))\n"
            "labs = [f'planted alpha\\n{e:.0f}%/yr' for e, _, _ in res]\n"
            "ax.bar(labs, [r0[2] for r0 in res], color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t = 2 bar'); ax.axhline(0, color='k', lw=1)\n"
            "for i, r0 in enumerate(res): ax.annotate(f'active {r0[1]:+.2f}%/yr\\nt={r0[2]:+.2f}', (i, r0[2]), ha='center', va='bottom', fontsize=9)\n"
            "ax.set_ylabel('HAC t of clone active return')\n"
            "ax.set_title('Null stays flat; planted skill lights up through the same lagged pipe')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for e, v, t in res: print(f'planted {e:4.1f}%/yr: clone active {v:+.2f}%/yr  HAC t = {t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** skill the clone harness reads "
            f"**t = {R['syn'][0][2]}** (it cannot manufacture significance); with a planted "
            f"8%/yr it recovers **{R['syn'][1][1]:+.2f}%/yr at t = {R['syn'][1][2]}** — about "
            "2/3 of the alpha survives the 45-day lag + quarterly rotation. Two lessons: the "
            "machinery is faithful (so the real-tape negative is genuine), and a genuinely "
            "skilled manager's clone *would have shown up*. Berkshire's 2013–2026 top-10 "
            "didn't. *(Machinery proof only — never market evidence.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the claim is *beats the market*; the tape says the EW "
            f"clone lost **{R['ew']['active']}%/yr** at **HAC t = {R['ew']['t']}** "
            f"(significantly negative), VW {R['vw']['active']}%/yr (t = {R['vw']['t']}), "
            f"CAPM alphas negative (EW {R['ew']['alpha']}%/yr, t = {R['ew']['t_alpha']}), "
            "negative in both halves and at every top-N, below all 200 random managers from "
            "its own universe. Survivorship (97.5% slot coverage) and the post-2013-only "
            "window are named; the 1976–2006 literature era is out of scope.\n"
            f"- **Tradability `MIRAGE`** — flawless access, ~{R['turnover_reb']:.0f}% one-way "
            f"turnover per rebalance, {R['ew_net10']['drag_bps']} bps/yr drag at 10 bps — and "
            "nothing to collect. A perfectly executable way to lag the index.\n"
            f"- **Beats Berkshire itself? `BUSTED`** — BRK-B ({R['brk_cagr']}%/yr) beat both "
            f"clone legs (EW {R['vs_brk']['ew'][0]}%/yr, VW {R['vs_brk']['vw'][0]}%/yr) "
            "despite the cash pile the clone was supposed to shed."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The decay reading.** Martin-Puthenpurackal's +10%/yr mimicking alpha "
            "(1976–2006) vs our −4.9%/yr (2013–2026) is a textbook McLean-Pontiff arc: the "
            "edge was published in 2008, the filings went machine-readable in 2013, and the "
            "free ride has been negative ever since. We cannot rule on the early era — only "
            "note that no scriptable month of it remains.\n"
            "- **The one-CIK generalisation.** The engine is guru-agnostic: "
            "``data.fetch_13f`` takes any CIK. A cross-guru panel (does *any* famous 13F "
            "clone survive HAC scrutiny post-2013?) is the natural follow-up study.\n"
            "- **Sibling.** [263-insider-buying](../../263-insider-buying/) tests the Form 4 "
            "cousin — insiders' own trades on a 2-day clock, a different mechanism entirely.\n\n"
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
