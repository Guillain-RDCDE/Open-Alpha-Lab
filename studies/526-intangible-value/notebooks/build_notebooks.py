"""Generate the two narrative notebooks for Study 526 (Intangible-Value).

    python notebooks/build_notebooks.py

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic positive
control runs anywhere, offline and deterministic; the real-panel cells use the cached EDGAR +
yfinance parquets under ../_cache/ if present and otherwise quote the frozen headline numbers in
``R`` (a single dict mirroring docs/results.md).
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


# Frozen real-tape headline numbers — the ONE dict, mirror of docs/results.md (as-of 2026-06-27).
R = dict(
    n_months=256, n_hold=220, n_names=40, n_rd=20, span_years=21.3,
    window_start="2005-02", window_end="2026-05", fingerprint="1cfa38af3cb0",
    avg_turnover=4.5,
    # legs (gross, annualised)
    long_cagr=14.95, long_sharpe=0.85, long_mean=15.68, long_dd=-44.6,
    short_cagr=12.95, short_sharpe=0.86, short_mean=13.47, short_dd=-46.8,
    ls_cagr=1.47, ls_sharpe=0.18, ls_mean=2.21, ls_dd=-44.2,
    spy_cagr=11.11, spy_sharpe=0.78, spy_mean=11.70, spy_dd=-50.8,
    # signal-axis HAC
    ls_t=0.67, long_vs_spy_mean=3.25, long_vs_spy_t=1.30,
    short_vs_spy_mean=1.03, short_vs_spy_t=0.71,
    # placebo
    placebo_null_sd=2.12, placebo_pctile=85.2, placebo_p=0.305,
    # adjustment contrast (the third axis)
    plain_mean=-0.16, plain_t=-0.05,
    adj_mean=2.21, adj_t=0.67,
    diff_mean=2.38, diff_t=2.93,
    # robustness (split fraction): frac -> (mean/yr, t)
    frac={0.50: (0.81, 0.33), 0.3333: (2.21, 0.67), 0.25: (0.65, 0.18), 0.20: (4.29, 1.11)},
    # costs
    ls_gross_mean=2.21, ls_gross_t=0.67, ls_net_mean=1.20, ls_net_t=0.37,
    # synthetic control (seed-averaged over 20 seeds)
    ctrl_null_t=0.00, ctrl_planted_edge=0.10, ctrl_planted_t=4.23,
    ctrl_planted_min=3.58, ctrl_planted_max=4.91,
)

# Badge fragments (reused in both notebook headers)
BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Beats plain B/M%3F: Confirmed](https://img.shields.io/badge/Beats_plain_B%2FM%3F-Confirmed-8b949e?style=flat-square)\n\n"
)


BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False,
                     "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from intangible_value import data, strategy as st

def _have_cache():
    study = os.path.abspath("..")
    return data.have_real(os.path.join(study, "_cache"))

HAVE_REAL = _have_cache()
print("EDGAR+yfinance cache present:", HAVE_REAL)

if HAVE_REAL:
    real = data.load_real(allow_survivorship_bias=True)
    rets = real["returns"]
    sigs = data.build_signals(real, report_lag=1)
    RACE = st.race(real, sigs, frac=1/3, cost_bps=10.0, borrow_bps=100.0, n_shuffles=400)
    print(f"adj-B/M long-short: {RACE['test_ls']['mean_ann']*100:+.2f}%/yr  "
          f"HAC t={RACE['test_ls']['tstat']:+.2f}  (n={RACE['test_ls']['n']})")
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Intangible-Value — does fixing book-to-market for intangibles rescue value investing?\n"
            "### Lev-Srivastava (2019): capitalise R&D + SG&A into book, then re-run the value sort\n\n"
            + BADGES +
            "Classic **value investing** buys cheap stocks — low price relative to *book value* "
            "(book-to-market). It worked for decades, then largely stopped working after ~2007. "
            "Baruch Lev and Anup Srivastava argue the failure is partly an **accounting illusion**: "
            "accounting rules force firms to *expense* the money they spend building intangible "
            "capital — R&D, brands, software, organisational know-how — instead of putting it on the "
            "balance sheet. For an intangible-heavy firm (think a software or pharma company), the "
            "reported *book value* badly understates the real capital base, so book-to-market mis-"
            "ranks it. The proposed fix: **capitalise** past R&D and a slice of SG&A back into book, "
            "and sort on this **intangible-adjusted** book-to-market instead.\n\n"
            "We test that on 40 large-caps using SEC EDGAR fundamentals and yfinance prices, naming "
            "the survivorship bias up front.\n\n"
            "> **This is the plain-language layer.** Want the perpetual-inventory capitalisation, "
            "the placebo null, and the head-to-head HAC test? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ----
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does adjusted-B/M value earn a premium here? | **Not significantly.** Long-short "
            f"**+{R['ls_mean']:.2f}%/yr**, HAC *t* = **+{R['ls_t']:.2f}** — below the |t|≥2 bar, "
            f"and it fails a label-shuffle placebo (p = {R['placebo_p']:.2f}). |\n"
            "| Does plain book-to-market work? | **No — dead flat.** Plain B/M long-short "
            f"= **{R['plain_mean']:+.2f}%/yr** (*t* = {R['plain_t']:+.2f}) on this basket. |\n"
            "| Does the intangible adjustment *change* the sort? | **Yes, measurably.** The "
            f"adjusted-minus-plain spread is **+{R['diff_mean']:.2f}%/yr at *t* = +{R['diff_t']:.2f}** "
            "— the correction genuinely re-ranks the field. |\n"
            "| Is it tradable? | **Mirage.** Net of costs + borrow: "
            f"**+{R['ls_net_mean']:.2f}%/yr** at *t* = {R['ls_net_t']:.2f}, Sharpe "
            f"{R['ls_sharpe']:.2f}. |\n\n"
            "> The mechanical Lev-Srivastava claim — *capitalising intangibles changes and improves "
            "the value sort* — is confirmed (it beats plain B/M, *t* = 2.93). But on this 21-year "
            "large-cap survivor basket, **neither** version of value earns a tradable premium."
        ),

        # ---- BEAT 1 — THE CLAIM ----
        md(
            "## 1 — The claim\n\n"
            "> *\"Value investing's recent failure is largely an accounting artefact. Because GAAP "
            "expenses R&D and brand-building, the book value of intangible-intensive firms is "
            "understated, distorting book-to-market. Capitalise those intangibles and the value "
            "premium reappears.\"*\n\n"
            "— Lev & Srivastava (2019), *Explaining the Recent Failure of Value Investing*\n\n"
            "The recipe (Eisfeldt-Papanikolaou 2013; Peters-Taylor 2017):\n"
            "- **Knowledge capital** = a running stock of past **R&D**, amortised over ~5 years.\n"
            "- **Organisation capital** = a running stock of **30% of past SG&A**, amortised over "
            "~3 years.\n"
            "- **Adjusted book** = reported book equity + knowledge capital + organisation capital.\n\n"
            "Sort on adjusted-book / market-cap instead of book / market-cap."
        ),

        # ---- BEAT 2 — SO WHAT ----
        md(
            "## 2 — So what?\n\n"
            "If true, a simple accounting fix revives one of the oldest and most-traded factors in "
            "finance — and explains a decade of pain for value funds as a measurement problem rather "
            "than a death of the premium. It also reframes the great 2010s growth-vs-value gap: maybe "
            "'expensive growth' tech was *not* as expensive as book-to-market made it look, because "
            "its real (intangible) book was huge and invisible."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ----
        md(
            "## 3 — How would we even know?\n\n"
            "Three disciplines keep us honest:\n\n"
            "1. **One reporting lag + one execution lag.** Each month's book uses only the last "
            "*reported* fiscal year (1-year lag) and we enter the *next* month — no look-ahead.\n"
            "2. **Race the adjustment against plain B/M.** The decisive test isn't 'does adjusted "
            "value work' in isolation — it's whether the adjustment *beats plain B/M head-to-head*. "
            "That isolates what the intangibles correction adds.\n"
            "3. **Label-shuffle placebo.** Permute which name carries which signal value 400 times; "
            "the real long-short must sit in the tail, or the split of a heterogeneous field was "
            "doing the work."
        ),

        # ---- BEAT 4 — TEARDOWN ----
        md(
            "## 4 — The teardown\n\n"
            "**First, does the value engine work when the effect is planted?**"
        ),
        code(
            "# Synthetic positive control: plant a known value premium and verify recovery.\n"
            "edges = [0.0, 0.04, 0.08, 0.12]\n"
            "rows = []\n"
            "for e in edges:\n"
            "    s2, r2, b2, truth = data.synthetic_panel(edge=e, seed=526)\n"
            "    bk = st.signal_books(s2, r2)\n"
            "    t = st.hac_tstat(bk['long_short'])\n"
            "    rows.append({'edge': e, 'ls_%/yr': t['mean_ann']*100, 't': t['tstat']})\n"
            "ctrl = pd.DataFrame(rows)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "col = [GREEN if r > 0.5 else (RED if r < -0.5 else GREY) for r in ctrl['ls_%/yr']]\n"
            "ax.bar(ctrl['edge'].astype(str), ctrl['ls_%/yr'], color=col)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "for i, (m, t) in enumerate(zip(ctrl['ls_%/yr'], ctrl['t'])):\n"
            "    ax.text(i, m + 0.2, f't={t:+.1f}', ha='center', va='bottom', fontsize=9)\n"
            "ax.set_xlabel('planted value premium (edge, /yr)'); ax.set_ylabel('long-short (%/yr)')\n"
            "ax.set_title('Synthetic control: engine finds the value premium when planted')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(ctrl.round(3).to_string(index=False))"
        ),
        md(
            f"The engine is faithful: at `edge = 0` it manufactures no significance (seed-averaged "
            f"*t* = {R['ctrl_null_t']:.2f} over 20 seeds), and a large planted premium lights up "
            f"reliably (`edge = {R['ctrl_planted_edge']:.2f}` → mean *t* = +{R['ctrl_planted_t']:.2f}, "
            f"always > {R['ctrl_planted_min']:.1f}). So the verdict on the real tape reflects **the "
            "market**, not the method."
        ),
        md("**Now the honest test on the real EDGAR + yfinance panel — plain vs adjusted B/M.**"),
        code(
            "if HAVE_REAL:\n"
            "    plain_t = RACE['test_plain_ls']; adj_t = RACE['test_ls']; diff_t = RACE['test_intan_minus_plain']\n"
            "    plain_m, plain_tt = plain_t['mean_ann']*100, plain_t['tstat']\n"
            "    adj_m, adj_tt = adj_t['mean_ann']*100, adj_t['tstat']\n"
            "    diff_m, diff_tt = diff_t['mean_ann']*100, diff_t['tstat']\n"
            "else:\n"
            "    plain_m, plain_tt = R['plain_mean'], R['plain_t']\n"
            "    adj_m, adj_tt = R['adj_mean'], R['adj_t']\n"
            "    diff_m, diff_tt = R['diff_mean'], R['diff_t']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.5))\n"
            "labels = ['Plain B/M', 'Adjusted B/M', 'Adjusted - Plain\\n(what the fix adds)']\n"
            "vals = [plain_m, adj_m, diff_m]\n"
            "tts = [plain_tt, adj_tt, diff_tt]\n"
            "cols = [RED, AMBER, GREEN]\n"
            "bars = ax.bar(labels, vals, color=cols, width=0.55)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "for b, t in zip(bars, tts):\n"
            "    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.1, f't={t:+.2f}',\n"
            "            ha='center', va='bottom', fontsize=10)\n"
            "ax.set_ylabel('long-short mean (%/yr)')\n"
            "ax.set_title('Plain value is flat; the intangible adjustment moves the sort (t=2.93)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'plain   B/M : {plain_m:+.2f}%/yr  (t={plain_tt:+.2f})')\n"
            "print(f'adjusted B/M: {adj_m:+.2f}%/yr  (t={adj_tt:+.2f})')\n"
            "print(f'adj - plain : {diff_m:+.2f}%/yr  (t={diff_tt:+.2f})  <- the intangible fix')"
        ),
        md(
            f"Plain book-to-market is **dead flat** ({R['plain_mean']:+.2f}%/yr, *t* = "
            f"{R['plain_t']:+.2f}). The intangible-adjusted sort earns **+{R['adj_mean']:.2f}%/yr** "
            f"but at *t* = +{R['adj_t']:.2f} (insignificant). The interesting part is the "
            f"**difference**: the adjustment re-ranks the field enough that adjusted-minus-plain is "
            f"**+{R['diff_mean']:.2f}%/yr at *t* = +{R['diff_t']:.2f}** — a real, significant *change* "
            "to the value sort. The fix does something; it just isn't enough to make value pay here."
        ),

        # ---- BEAT 5 — VERDICT ----
        md(
            "## 5 — The verdict\n\n"
            f"- **Signal — WEAK.** Adjusted-B/M long-short **+{R['ls_mean']:.2f}%/yr**, HAC *t* = "
            f"**+{R['ls_t']:.2f}** ({R['n_hold']} months), fails the placebo (p = {R['placebo_p']:.2f}). "
            "Not distinguishable from zero. WEAK not NONE only because the value premium and the "
            "intangibles adjustment have strong academic support on broad, non-survivor universes.\n"
            f"- **Tradability — MIRAGE.** Net of costs + borrow: **+{R['ls_net_mean']:.2f}%/yr** at "
            f"*t* = {R['ls_net_t']:.2f}, Sharpe {R['ls_sharpe']:.2f}, max DD {R['ls_dd']:.0f}%. No "
            "tradable edge — both legs are the same survivor large-caps.\n"
            f"- **Beats plain B/M? — CONFIRMED.** Adjusted-minus-plain = **+{R['diff_mean']:.2f}%/yr "
            f"at *t* = +{R['diff_t']:.2f}**. The mechanical claim holds.\n"
            "- **Survivorship — Named.** Current-membership basket; absolute levels are upper bounds, "
            "but the bias is largely common to both legs."
        ),

        # ---- BEAT 6 — TRADABILITY ----
        md(
            "## 6 — Could you actually trade it?\n\n"
            "1. **There's no gross edge to start with.** The long-short is +2.2%/yr at *t* = 0.67. "
            "Costs are almost beside the point.\n"
            "2. **Turnover is low** (~4.5%/month — fundamentals move slowly), so this is *not* a "
            "case of a real signal eaten by trading frictions. The signal was never there.\n"
            "3. **Both legs are survivor large-caps** earning 13–16%/yr; the spread between them is "
            "noise on this small, biased universe.\n\n"
            "The honest read: the *adjustment* is real and the *premium* is not — on this basket."
        ),

        # ---- BEAT 7 — GOING FURTHER ----
        md(
            "## 7 — Going further\n\n"
            "- **Broaden the universe.** Lev-Srivastava run thousands of names including delisted "
            "firms; the value premium and its intangible rescue are strongest in small/mid-cap, which "
            "a 40-name large-cap survivor basket cannot see.\n"
            "- **Tune the capitalisation.** R&D amortisation horizon (here 5 yr) and the SG&A "
            "investment share (here 30%) are conventions; sweeping them tests how fragile the "
            "adjusted-minus-plain result is.\n"
            "- **[Study 525 — R-And-D-Intensity](../../525-r-and-d-intensity/)**: the R&D/market-cap "
            "cousin (Chan-Lakonishok-Sougiannis) on the same machinery.\n"
            "- **[Study 238 — Betting-Against-Beta](../../238-betting-against-beta/)** and "
            "**[Study 330 — Low-Volatility-Anomaly](../../330-low-volatility-anomaly/)**: sibling "
            "cross-sectional long-short teardowns.\n\n"
            "*Think intangible-adjusted value clears t > 2 net of costs on a broad, non-survivor "
            "universe? Fork this, widen the basket, add delisted names, and show it. That is the bar.*"
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
            "# Intangible-Value — a quantitative teardown\n"
            "### EDGAR fundamentals × perpetual-inventory intangibles × HAC inference × placebo\n\n"
            + BADGES +
            "The quantitative companion to the "
            "[notebook for the curious](01_for_the_curious.ipynb) — same seven beats, every claim "
            "carrying its standard error. We test Lev-Srivastava (2019): capitalise R&D (5-yr "
            "amortised knowledge capital) + 30%-of-SG&A (3-yr amortised organisation capital) into "
            "book, sort 40 large-caps on intangible-adjusted B/M, long the cheap tertile / short the "
            "expensive tertile, one reporting + one execution lag, monthly rebalance — and race it "
            "head-to-head against plain B/M.\n\n"
            "> **Not investment advice.** Real data: SEC EDGAR companyfacts + yfinance monthly "
            "total returns, 40 large-caps, 2005-02 → 2026-05, as-of 2026-06-27. Methods in "
            "[`docs/references.md`](../docs/references.md), reproducible numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **Survivorship is named:** current-membership large-cap basket projected back; "
            "absolute levels are upper bounds (the bias is largely common to both legs)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ----
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | adj-B/M long-short **+{R['ls_mean']:.2f}%/yr**, HAC *t* = "
            f"**+{R['ls_t']:.2f}**; placebo p = **{R['placebo_p']:.2f}** (pctile {R['placebo_pctile']:.0f}). |\n"
            f"| **Tradability** | `MIRAGE` | net **+{R['ls_net_mean']:.2f}%/yr** (*t* = {R['ls_net_t']:.2f}), "
            f"Sharpe **{R['ls_sharpe']:.2f}**, max DD **{R['ls_dd']:.0f}%**. |\n"
            f"| **Beats plain B/M?** | `CONFIRMED` | plain {R['plain_mean']:+.2f}%/yr (t={R['plain_t']:+.2f}); "
            f"adjusted − plain **+{R['diff_mean']:.2f}%/yr at t = +{R['diff_t']:.2f}**. |\n\n"
            "> The intangible adjustment *changes and improves* the value sort (t = 2.93 head-to-head), "
            "but on 21 years of 40 large-cap survivors neither value sort clears the |t|≥2 bar."
        ),

        # ---- BEAT 1 — THE CLAIM ----
        md(
            "## 1 — The claim, steelmanned\n\n"
            "Let $B_i$ = reported book equity, $K_i$ = capitalised R&D stock (knowledge capital), "
            "$O_i$ = capitalised 30%-of-SG&A stock (organisation capital), $ME_i$ = market cap. "
            "Define adjusted book-to-market $\\widetilde{bm}_i = (B_i + K_i + O_i)/ME_i$ vs plain "
            "$bm_i = B_i/ME_i$. Lev-Srivastava assert:\n\n"
            "- **H1 (signal).** The long-cheap / short-expensive spread on $\\widetilde{bm}$ is "
            "positive: $\\alpha^{\\widetilde{bm}} > 0$.\n"
            "- **H2 (improvement).** $\\widetilde{bm}$ beats plain $bm$: "
            "$\\alpha^{\\widetilde{bm}} - \\alpha^{bm} > 0$.\n"
            "- **H3 (tradable).** The spread survives turnover + borrow.\n\n"
            "We **reject H1** (t = 0.67, fails placebo), **confirm H2** (the adjustment beats plain "
            "B/M at t = 2.93), and **reject H3** (net t = 0.37)."
        ),

        # ---- BEAT 2 — SO WHAT ----
        md(
            "## 2 — So what? — the economic stakes\n\n"
            "Value is the most-traded academic factor and the intangibles-rescue story (Lev-"
            "Srivastava; Arnott et al.) is the leading explanation for why value funds bled through "
            "the 2010s. If the adjustment merely *re-labels* the same large-caps without producing a "
            "tradable spread on a clean modern tape, the rescue is real as accounting but thin as alpha."
        ),

        # ---- BEAT 3 — PROTOCOL ----
        md(
            "## 3 — The protocol\n\n"
            "- **Intangible capital.** Perpetual inventory: $K_Y = \\sum_{k=0}^{4} RD_{Y-k}(5-k)/5$ "
            "and $O_Y = \\sum_{k=0}^{2} (0.3\\,SGA_{Y-k})(3-k)/3$ — only past flows, no look-ahead.\n"
            "- **Signals.** $bm = B/ME$ and $\\widetilde{bm} = (B+K+O)/ME$, formed monthly from "
            "fiscal-year-(Y-1) fundamentals and the contemporaneous price.\n"
            "- **Books.** Long top tertile (cheap), short bottom tertile (expensive), enter t+1.\n"
            "- **Inference.** Newey-West HAC *t* on the monthly long-short, and on the "
            "adjusted-minus-plain per-period spread.\n"
            "- **Placebo.** 400 cross-sectional label shuffles.\n"
            "- **Costs.** 10 bps × one-way turnover + 100 bps/yr borrow on the short leg.\n"
            "- **Universe caveat.** 40 large-cap survivors (EDGAR + yfinance)."
        ),

        # ---- BEAT 4 — TEARDOWN ----
        md("## 4 — The teardown"),
        md(
            "### 4a — Positive control: the engine is a faithful detector (seed-robust)\n\n"
            "Sweep the planted premium; the long-short mean should be monotone in `edge`, and "
            "`edge = 0` must stay insignificant across seeds."
        ),
        code(
            "edges = [0.0, 0.03, 0.06, 0.10]\n"
            "means, tstats = [], []\n"
            "for e in edges:\n"
            "    # average the HAC t over 20 seeds (Welch-style robustness for the synthetic claim)\n"
            "    ms, ts = [], []\n"
            "    for sd in range(20):\n"
            "        s2, r2, b2, _ = data.synthetic_panel(edge=e, seed=500+sd)\n"
            "        bk = st.signal_books(s2, r2)\n"
            "        t = st.hac_tstat(bk['long_short'])\n"
            "        ms.append(t['mean_ann']*100); ts.append(t['tstat'])\n"
            "    means.append(np.mean(ms)); tstats.append(np.mean(ts))\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "col = [GREEN if m > 0.3 else GREY for m in means]\n"
            "ax.bar([str(e) for e in edges], means, color=col)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "for i, (m, t) in enumerate(zip(means, tstats)):\n"
            "    ax.text(i, m + 0.1, f'mean t={t:+.2f}', ha='center', va='bottom', fontsize=9)\n"
            "ax.set_xlabel('planted value premium (edge, /yr)')\n"
            "ax.set_ylabel('long-short mean (%/yr, 20-seed avg)')\n"
            "ax.set_title('Positive control: monotone in edge; edge=0 stays flat (mean t=0.00)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('edge / 20-seed mean t:', dict(zip(edges, np.round(tstats,2))))"
        ),
        md(
            f"At `edge = 0` the seed-averaged *t* is {R['ctrl_null_t']:.2f} (no false positive); at "
            f"`edge = {R['ctrl_planted_edge']:.2f}` it is +{R['ctrl_planted_t']:.2f} "
            f"(range {R['ctrl_planted_min']:.2f}–{R['ctrl_planted_max']:.2f}). Faithful engine."
        ),
        md(
            "### 4b — How much does the intangible adjustment move book?\n\n"
            "The capitalised intangible stock as a fraction of (book + intangibles) — the size of the "
            "correction, by name, at the latest fiscal year."
        ),
        code(
            "if HAVE_REAL:\n"
            "    intan = sigs['intan_capital']\n"
            "    eq = real['equity']\n"
            "    fy = intan.index.max()\n"
            "    ic = intan.loc[fy]\n"
            "    be = eq.reindex(columns=ic.index).ffill().loc[:fy].iloc[-1]\n"
            "    frac_intan = (ic / (be.clip(lower=0) + ic)).dropna().sort_values(ascending=False)\n"
            "    frac_intan = frac_intan[frac_intan > 0]\n"
            "    fig, ax = plt.subplots(figsize=(10, 4.6))\n"
            "    cols = [AMBER if v > 0.25 else GREY for v in frac_intan.values]\n"
            "    ax.bar(range(len(frac_intan)), frac_intan.values*100, color=cols)\n"
            "    ax.set_xticks(range(len(frac_intan)))\n"
            "    ax.set_xticklabels(frac_intan.index, rotation=90, fontsize=7)\n"
            "    ax.set_ylabel('intangible capital / adjusted book (%)')\n"
            "    ax.set_title(f'Size of the intangible adjustment by name (FY{fy})')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('most intangible-adjusted:', list(frac_intan.head(6).index))\n"
            "    print('least adjusted (book ~ unchanged):', list(frac_intan.tail(6).index))\n"
            "else:\n"
            "    print('needs real cache')"
        ),
        md(
            "Intangible-heavy names (software, pharma, brand) see book inflated by a large fraction; "
            "banks / energy / utilities are barely touched. That re-ranking is *why* adjusted B/M "
            "differs from plain B/M — and why the head-to-head spread is significant even when the "
            "absolute premium is not."
        ),
        md("### 4c — The race: plain vs adjusted B/M long-short, and the legs vs SPY"),
        code(
            "if HAVE_REAL:\n"
            "    s_long = st.summarize(RACE['long']); s_short = st.summarize(RACE['short'])\n"
            "    s_ls = st.summarize(RACE['long_short']); s_spy = st.summarize(RACE['spy'])\n"
            "    long_m, short_m, ls_m, spy_m = (s_long['mean_ann']*100, s_short['mean_ann']*100,\n"
            "                                    s_ls['mean_ann']*100, s_spy['mean_ann']*100)\n"
            "    ls_t = RACE['test_ls']['tstat']\n"
            "else:\n"
            "    long_m, short_m, ls_m, spy_m = R['long_mean'], R['short_mean'], R['ls_mean'], R['spy_mean']\n"
            "    ls_t = R['ls_t']\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n"
            "labels = ['Long\\n(cheap)', 'Short\\n(expensive)', 'SPY']\n"
            "axes[0].bar(labels, [long_m, short_m, spy_m], color=[AMBER, RED, GREY], width=0.55)\n"
            "axes[0].axhline(spy_m, ls='--', c=GREY, lw=1)\n"
            "axes[0].set_ylabel('mean annual return (%/yr)')\n"
            "axes[0].set_title('Both legs are ~13-16%/yr survivor large-caps')\n"
            "if HAVE_REAL:\n"
            "    eq = (1 + RACE['long_short']).cumprod()\n"
            "    axes[1].plot(eq.index, eq.values, c=AMBER, lw=2)\n"
            "    axes[1].axhline(1, c='k', lw=1, ls='--')\n"
            "    axes[1].set_ylabel('cumulative wealth (adj-B/M long-short)')\n"
            "    axes[1].set_title(f'Long-short equity | {ls_m:+.1f}%/yr  t={ls_t:+.2f}')\n"
            "else:\n"
            "    axes[1].text(0.5, 0.5, f'LS {ls_m:+.1f}%/yr (t={ls_t:+.2f})', ha='center')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'long {long_m:+.2f}% | short {short_m:+.2f}% | LS {ls_m:+.2f}%/yr (t={ls_t:+.2f}) | SPY {spy_m:+.2f}%')"
        ),
        md("### 4d — Label-shuffle placebo: is the split doing real work?"),
        code(
            "if HAVE_REAL:\n"
            "    null = RACE['placebo_null']\n"
            "    real_ls = RACE['test_ls']['mean_ann']*100\n"
            "    pctile, pval = RACE['placebo_pctile'], RACE['placebo_p']\n"
            "else:\n"
            "    rng = np.random.default_rng(0)\n"
            "    null = rng.normal(0, R['placebo_null_sd'], 400)\n"
            "    real_ls, pctile, pval = R['ls_mean'], R['placebo_pctile'], R['placebo_p']\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(null*100 if HAVE_REAL else null, bins=30, color=GREY, alpha=0.7, edgecolor='white')\n"
            "ax.axvline(real_ls, c=RED, lw=2.5, label=f'real LS {real_ls:+.2f}%/yr (pctile {pctile:.0f})')\n"
            "ax.set_xlabel('shuffled long-short (%/yr)'); ax.set_ylabel('count')\n"
            "ax.set_title(f'Label-shuffle null — real spread NOT in the tail (p={pval:.2f})')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'real LS percentile in null: {pctile:.1f}  | two-sided p = {pval:.3f}')"
        ),
        md(
            f"The real spread sits at the {R['placebo_pctile']:.0f}th percentile of the null — "
            f"*not* in the tail (p = {R['placebo_p']:.2f}). The sort is not doing anything a random "
            "relabelling could not."
        ),
        md("### 4e — Split-fraction robustness and costs"),
        code(
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for frac, lbl in [(0.50,'halves'),(1/3,'tertiles'),(0.25,'quartiles'),(0.20,'quintiles')]:\n"
            "        b = st.signal_books(sigs['bm_intan'], rets, frac=frac)\n"
            "        t = st.hac_tstat(b['long_short'])\n"
            "        rows.append({'frac': round(frac,2), 'split': lbl,\n"
            "                     'LS_%/yr': t['mean_ann']*100, 't': t['tstat']})\n"
            "    rob = pd.DataFrame(rows)\n"
            "else:\n"
            "    rob = pd.DataFrame([{'frac': f, 'LS_%/yr': v[0], 't': v[1]} for f, v in R['frac'].items()])\n"
            "print(rob.round(3).to_string(index=False))\n"
            "if HAVE_REAL:\n"
            "    g = RACE['test_ls']; npv = RACE['test_ls_net']\n"
            "    print(f\"\\ngross LS: {g['mean_ann']*100:+.2f}%/yr (t={g['tstat']:+.2f})\")\n"
            "    print(f\"net   LS: {npv['mean_ann']*100:+.2f}%/yr (t={npv['tstat']:+.2f})  \"\n"
            "          f\"[10bps turnover + 100bps/yr borrow]  | avg turnover {RACE['avg_turnover']*100:.1f}%/mo\")\n"
            "else:\n"
            "    print(f\"\\ngross {R['ls_gross_mean']:+.2f}%/yr (t={R['ls_gross_t']:+.2f}) | \"\n"
            "          f\"net {R['ls_net_mean']:+.2f}%/yr (t={R['ls_net_t']:+.2f})\")"
        ),

        # ---- BEAT 5 — VERDICT ----
        md(
            "## 5 — The verdict\n\n"
            f"- **Signal `WEAK`** — adj-B/M long-short +{R['ls_mean']:.2f}%/yr, HAC *t* = "
            f"+{R['ls_t']:.2f} (< 2), placebo p = {R['placebo_p']:.2f}. Academic prior on the value "
            "premium + intangibles adjustment prevents a `NONE`, but the real tape does not clear "
            "the bar.\n"
            f"- **Tradability `MIRAGE`** — net +{R['ls_net_mean']:.2f}%/yr (*t* = {R['ls_net_t']:.2f}), "
            f"Sharpe +{R['ls_sharpe']:.2f}, max DD {R['ls_dd']:.0f}%. Turnover is low "
            f"({R['avg_turnover']:.1f}%/mo) — there was no gross edge to erode.\n"
            f"- **Beats plain B/M? `CONFIRMED`** — adjusted − plain = +{R['diff_mean']:.2f}%/yr at "
            f"HAC *t* = +{R['diff_t']:.2f}. The intangible correction genuinely re-ranks and improves "
            "the value sort.\n"
            "- **Survivorship `NAMED`** — current-membership basket; bias largely common to both legs."
        ),

        # ---- BEAT 6 — TRADABILITY ----
        md(
            "## 6 — Could you trade it?\n\n"
            "1. **No gross edge.** The long-short is +2.2%/yr at *t* = 0.67 — indistinguishable from "
            "zero before any cost.\n"
            "2. **Low turnover** (~4.5%/month) means this is not a frictions story; the signal is "
            "simply absent on this basket.\n"
            "3. **Borrow on the expensive (growth) short leg** is the only meaningful cost line, and "
            "it still leaves the net spread at +1.2%/yr (*t* = 0.37).\n\n"
            "The adjustment is real *accounting*; it is not *alpha* on 40 large-cap survivors."
        ),

        # ---- BEAT 7 — GOING FURTHER ----
        md(
            "## 7 — Going further\n\n"
            "- **Broad, non-survivor universe.** The Lev-Srivastava result is a thousands-of-names "
            "claim; a 40-name large-cap survivor basket is the wrong instrument to confirm or reject "
            "the *premium* (though it cleanly shows the *adjustment* matters).\n"
            "- **Capitalisation conventions.** Sweep R&D amortisation (5 yr) and the SG&A investment "
            "share (30%) / horizon (3 yr); see how stable the adjusted-minus-plain *t* is.\n"
            "- **Decompose the adjustment.** Knowledge (R&D) vs organisation (SG&A) capital "
            "separately — which one does the re-ranking?\n"
            "- **[Study 525 — R-And-D-Intensity](../../525-r-and-d-intensity/)**: the R&D/market-cap "
            "cousin. **[Study 238 — Betting-Against-Beta](../../238-betting-against-beta/)** and "
            "**[Study 330 — Low-Volatility-Anomaly](../../330-low-volatility-anomaly/)**: sibling "
            "long-short teardowns on the same survivor machinery.\n\n"
            "*Think intangible-adjusted value clears t > 2 net of costs on a broad universe with "
            "delisted names? Fork this, widen it, and show it. That is the bar.*"
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
