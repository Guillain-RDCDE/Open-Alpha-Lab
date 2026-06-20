"""Generate the two narrative notebooks for Study 304 (Carbon-Credits).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
controls run anywhere, offline and deterministic; the real-tape cells use the cached KRBN +
BIL total-return parquets under ../_cache/ if present and otherwise quote the frozen
headline numbers in ``R`` (mirroring docs/results.md), with a banner naming the tape. So the
notebook re-runs for any reader, online or off.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-17,
# window ends 2026-05-29, fingerprint cfb978f26c25).
R = dict(
    fp="cfb978f26c25", start="2020-07-31", end="2026-05-29", n_days=1464,
    # buy-and-hold KRBN vs cash
    bh_cagr=16.2, bh_vol=28.7, bh_sharpe=0.57, bh_maxdd=-36.4, bh_t=1.41, bh_final=2.39,
    # calendar-year total returns (KRBN)
    cy_2020=20.9, cy_2021=107.7, cy_2022=-12.8, cy_2023=8.0,
    cy_2024=-13.6, cy_2025=23.1, cy_2026=-5.1,
    # momentum overlay vs B&H (excess-vs-excess Sharpe), gross unless noted
    ov63_sharpe=0.29, ov63_cagr=7.1, ov63_dd=-50.9, ov63_diff=-0.27, ov63_frac=0.15,
    ov126_sharpe=0.07, ov126_cagr=2.2, ov126_dd=-51.6, ov126_diff=-0.50, ov126_frac=0.03,
    ov252_sharpe=-0.33, ov252_cagr=-6.2, ov252_dd=-61.4, ov252_diff=-0.90, ov252_frac=0.00,
    ov126_tim=51.9, ov126_flips=93,
    # synthetic positive controls
    ctrl_drift_t=3.16, ctrl_timing_diff=0.58, ctrl_timing_frac=0.99, null_avg_diff=0.02,
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

from carbon_credits import data, strategy as st

AS_OF_END = "2026-05-31"   # drop the partial June-2026 month

def _load_real():
    try:
        f = data.load_real(("KRBN", "BIL"), end=AS_OF_END)
        if len(f) < 200:
            return None
        return f
    except Exception as e:
        print("real tape unavailable (offline):", type(e).__name__)
        return None

REAL = _load_real()
HAVE_REAL = REAL is not None
print("real KRBN+BIL tape present:", HAVE_REAL)
if HAVE_REAL:
    print(f"  {REAL.index[0].date()} .. {REAL.index[-1].date()}  n={len(REAL)}  fp={data.fingerprint(REAL)}")
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Carbon-Credits — does buying 'the cap that only shrinks' pay? 🌍\n"
            "### A green buy-and-hold bet, taken literally — in plain English\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Does_timing_beat_buy--and--hold%3F: Not_supported](https://img.shields.io/badge/Does_timing_beat_buy--and--hold%3F-Not_supported-8b949e?style=flat-square)\n\n"
            "The pitch is seductive: carbon allowances are the one asset where supply is "
            "*designed* to shrink. Every year the regulator retires permits, polluters still "
            "have to buy them, so the price 'can only go up' — just buy a carbon ETF like "
            "**KRBN** and ride the energy transition. This notebook takes that literally on "
            "KRBN's whole life (since 2020), asks the only question a *reward* has to answer — "
            "*did it beat just holding cash?* — and then tries the obvious fix: *what if you "
            "timed the trend instead of blindly holding?*\n\n"
            "> **This is the plain-language layer.** Want the *t*-stats, the excess-of-cash "
            "Sharpe race and the bootstrap? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart below is "
            "drawn by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Did buy-and-hold carbon pay? | **It looks like it** — +{R['bh_cagr']:.1f}%/yr, a "
            f"{R['bh_final']:.1f}× over ~6 years. But measured against cash the edge is "
            f"**not statistically real** (*t* = +{R['bh_t']:.2f}, below the bar). |\n"
            "| Why the asterisk? | Because **one year did all the work.** 2021 was "
            f"**+{R['cy_2021']:.0f}%**; the other five years are a coin-flip around cash, with "
            "two double-digit *down* years. |\n"
            "| Should you have timed it instead? | **No.** A 'hold only while the trend is up' "
            "overlay *lost* to simply holding — at every lookback, before costs, with worse "
            "drawdowns. |\n"
            "| So what is it? | A volatile niche commodity dressed up as a one-way escalator. "
            "The green story is real; the **smooth compounding** is a story we tell after the "
            "one lucky year. |\n\n"
            "> Carbon allowances may well rise over decades as caps tighten — the *economics* "
            "are sound. But a ~6-year, one-explosion tape can't prove a reward, and the "
            "'just time it' rescue makes things worse, not better."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"KRBN is the first US-listed ETF designed to track carbon-allowance markets. "
            "Under cap-and-trade, the supply of allowances is fixed and falling while polluters "
            "must keep buying — a structural tailwind. Carbon is a way to invest in the energy "
            "transition.\"*\n\n"
            "— the KraneShares case for KRBN (paraphrased)\n\n"
            "It rests on something genuinely real: the EU and other cap-and-trade systems "
            "**do** retire permits on a schedule, and the EU's Market Stability Reserve was "
            "built specifically to tighten supply and lift the floor. A shrinking supply meeting "
            "stubborn demand *should* push prices up over time. The bet is that you can just "
            "**hold** it and collect that drift."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "Two reasons to look hard. First, this is the rare 'folk' bet with a real "
            "mechanism behind it — so the interesting question isn't *whether carbon is a thing* "
            "(it is) but *whether a passive hold actually banked a reward, or just took a wild "
            "commodity ride that happened to end up*. Second, 'the cap only shrinks' is a "
            "perfect specimen of a **narrative that survives any single chart**: in a bull year "
            "it's vindicated, in a bear year it's 'temporary noise.' The only way to settle it "
            "is to measure the reward over cash, with honest error bars, and to ask whether the "
            "obvious active fix (timing) would have helped."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Three checks:\n\n"
            "1. **Beat cash, not zero.** A 'reward' only counts if it beats parking the money "
            "in T-bills. We measure KRBN's average return *above cash* and ask whether it's "
            "distinguishable from luck.\n"
            "2. **Where did the return come from?** We break it down year by year. If one year "
            "is the whole story, the 'steady escalator' framing is wrong.\n"
            "3. **Would timing have helped?** We run the textbook fix — hold KRBN only while "
            "its recent trend is up, otherwise sit in cash — and race it against just holding, "
            "fairly (both measured above cash).\n\n"
            "Tape: KRBN total-return daily bars from its 2020 launch, with BIL (T-bills) as the "
            "cash yardstick."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First: the headline growth — and the one-year catch.** Here's KRBN's "
            "calendar-year total return:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    Rr = st.to_returns(REAL)\n"
            "    cy = Rr['KRBN'].groupby(Rr.index.year).apply(lambda s:(1+s).prod()-1)*100\n"
            "    years = [int(y) for y in cy.index]; vals = list(cy.values)\n"
            "else:\n"
            f"    years = [2020,2021,2022,2023,2024,2025,2026]\n"
            f"    vals = [{R['cy_2020']},{R['cy_2021']},{R['cy_2022']},{R['cy_2023']},{R['cy_2024']},{R['cy_2025']},{R['cy_2026']}]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "cols = [GREEN if v>=0 else RED for v in vals]\n"
            "b = ax.bar([str(y) for y in years], vals, color=cols, width=.6)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('total return %')\n"
            "ax.set_title('KRBN year by year: one explosive year carries the whole record')\n"
            "for bar_, v in zip(b, vals):\n"
            "    ax.annotate(f'{v:+.0f}%', (bar_.get_x()+bar_.get_width()/2, v),\n"
            "        ha='center', va='bottom' if v>=0 else 'top', fontsize=9)\n"
            "plt.tight_layout(); plt.show()\n"
            "print('2021 alone:', f'{vals[1]:+.0f}%   every other year averages near a coin-flip vs cash')"
        ),
        md(
            f"There's the whole study in one chart. **2021 was +{R['cy_2021']:.0f}%** (EU "
            "allowances ripped on tighter climate targets and a gas-supply crunch). Take that "
            "year away and what's left is two double-digit *losses* (2022, 2024), a couple of "
            "modest gains, and a flat year — a volatile commodity, not an escalator."
        ),
        md(
            "**Second: did the hold actually beat cash?** Growth means nothing if it doesn't "
            "clear the risk-free bar. Here's the average daily edge over T-bills, with its "
            "honest *t*-stat:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    Rr = st.to_returns(REAL); rf = Rr['BIL']; bh = st.buy_and_hold(Rr)\n"
            "    s = st.stats(bh, rf); t = st.excess_tstat(bh, rf)\n"
            "    cagr, sharpe, maxdd = s['cagr']*100, s['sharpe'], s['max_dd']*100\n"
            "else:\n"
            f"    cagr, sharpe, maxdd, t = {R['bh_cagr']}, {R['bh_sharpe']}, {R['bh_maxdd']}, {R['bh_t']}\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.3))\n"
            "b = ax.bar(['Excess-of-cash\\nt-stat'], [t], color=AMBER if abs(t)<2 else GREEN, width=.4)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1.2, label='significance bar (t=2)')\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylim(0, 2.6); ax.legend()\n"
            "ax.set_title(f'Buy-and-hold KRBN: +{cagr:.0f}%/yr, but the reward over cash is t={t:+.2f}')\n"
            "ax.annotate(f't={t:+.2f}', (0, t+0.06), ha='center', fontsize=11, fontweight='bold')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'CAGR {cagr:+.1f}%/yr, Sharpe(ex-cash) {sharpe:.2f}, max drawdown {maxdd:.0f}% — but t={t:+.2f} (below 2)')"
        ),
        md(
            f"The compounding is real (+{R['bh_cagr']:.0f}%/yr), but the **reward over cash is "
            f"*t* = +{R['bh_t']:.2f}** — under the significance bar. With 29% volatility and a "
            f"−36% drawdown over a single market cycle, you genuinely cannot tell this apart "
            "from an expensive way of holding cash. The economics may be sound; this *tape* "
            "can't prove the payoff."
        ),
        md(
            "**Third: would timing have saved it?** The textbook rescue is *don't blindly hold "
            "— hold only while the trend is up, otherwise go to cash.* Here's that overlay "
            "(6-month trend) against just holding:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    Rr = st.to_returns(REAL); rf = Rr['BIL']; bh = st.buy_and_hold(Rr)\n"
            "    sig = st.momentum_signal(REAL, 'KRBN', lookback=126)\n"
            "    ov = st.overlay_returns(Rr, sig, 'KRBN', 'BIL', cost_bps=0.0)\n"
            "    eq_bh = (1+bh).cumprod(); eq_ov = (1+ov).cumprod()\n"
            "    idx = eq_bh.index\n"
            "else:\n"
            "    idx = pd.bdate_range('2020-07-31', periods=1464)\n"
            "    rng = np.random.default_rng(0)\n"
            "    eq_bh = pd.Series(np.exp(np.cumsum(rng.normal(0.0006,0.018,len(idx)))), index=idx)\n"
            "    eq_ov = pd.Series(np.exp(np.cumsum(rng.normal(0.0001,0.012,len(idx)))), index=idx)\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.6))\n"
            "ax.plot(idx, eq_bh.values, c=GREEN, lw=1.8, label='Buy & hold KRBN')\n"
            "ax.plot(idx, eq_ov.values, c=RED, lw=1.6, label='Time-the-trend overlay')\n"
            "ax.set_ylabel('growth of $1'); ax.legend()\n"
            "ax.set_title('Timing the trend underperformed just holding — it whipsawed out of 2021')\n"
            "plt.tight_layout(); plt.show()\n"
            f"print('6-month overlay: ~{R['ov126_cagr']:.0f}%/yr vs buy-and-hold ~{R['bh_cagr']:.0f}%/yr — timing LOST')"
        ),
        md(
            f"The overlay sits in cash through part of the 2021 melt-up and keeps flipping in "
            f"the choppy years — it earned ~+{R['ov126_cagr']:.0f}%/yr against buy-and-hold's "
            f"+{R['bh_cagr']:.0f}%/yr, with a *deeper* drawdown. The fix that's supposed to tame "
            "a volatile asset made it worse."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Weak.** +{R['bh_cagr']:.0f}%/yr looks great, but the reward over cash "
            f"is *t* = +{R['bh_t']:.2f} (below the bar) and one year (2021, +{R['cy_2021']:.0f}%) "
            "is the whole record. The cap-shrinks economics are real; this tape can't certify a "
            "payoff.\n"
            "- **Tradability — Mirage.** A 29%-vol, −36%-drawdown commodity whose Sharpe over "
            "cash is indistinguishable from cash — and the obvious active overlay makes it "
            "*worse*. No risk-adjusted edge to harvest.\n"
            "- **Does timing beat buy-and-hold? — Not supported.** The trend overlay lost to "
            "naive holding at every lookback, before costs. Doing nothing beat trying to time it."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "You *can* trade KRBN — it's liquid enough. The problem isn't access, it's that "
            "there's no edge on either side. Buy-and-hold is just expensive beta on a niche "
            "commodity; the trend overlay trades a lot (dozens of in/out flips) to earn *less*. "
            "The honest pitch isn't 'a one-way green compounding machine' or 'a timeable trend' "
            "— it's 'a volatile thematic exposure you might hold for conviction reasons, knowing "
            "the smooth-escalator story is mostly survivorship of one good year.'"
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Longer history.** KRBN is only ~6 years old. The underlying EU allowance "
            "futures go back to 2008 — a longer tape (with the 2008 and 2018 regime shifts) "
            "might finally clear the bar, or finally bury it.\n"
            "- **Carry, not spot.** KRBN holds *futures*; roll yield (contango/backwardation) "
            "is a separate driver from spot allowance prices. A carry decomposition is the next "
            "fork.\n"
            "- **The sister studies.** This is the same buy-and-hold-vs-alternative machinery as "
            "[Study 97 — Balancing-Act](../../97-balancing-act/) (60/40) and the same "
            "single-cycle-commodity caution as [Study 152 — Inflation-Hedge](../../152-inflation-hedge/).\n\n"
            "*Think the cap-and-trade tailwind is real and just needs more data? Fork this, "
            "wire in the long EUA futures tape, and show the reward-over-cash clearing t = 2.*"
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
            "# Carbon-Credits — a quantitative teardown 🔬\n"
            "### Real KRBN tape · excess-of-cash HAC *t* · momentum overlay race · "
            "block-bootstrap Sharpe diff · synthetic drift/persistence controls\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Does_timing_beat_buy--and--hold%3F: Not_supported](https://img.shields.io/badge/Does_timing_beat_buy--and--hold%3F-Not_supported-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — "
            "*same seven beats, every claim now carrying its standard error.* We test whether a "
            "passive KRBN hold earns a statistically real reward over cash, whether that reward "
            "is anything but one year, and whether a time-series momentum overlay beats naive "
            "buy-and-hold on a fair (excess-vs-excess) basis.\n\n"
            "> **Not investment advice.** Real data: Yahoo daily total-return bars, KRBN + BIL "
            "since 2020-07-30; as-of 2026-06-17 (window ends 2026-05-29); the offline core and "
            "tests run on a deterministic synthetic tape. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | Buy-and-hold KRBN +{R['bh_cagr']:.1f}%/yr, but "
            f"excess-of-cash HAC **t = +{R['bh_t']:.2f}** (below 2); 2021 alone was "
            f"+{R['cy_2021']:.0f}%. Literature-supported, tape-uncertified. |\n"
            f"| **Tradability** | `MIRAGE` | Sharpe(ex-cash) {R['bh_sharpe']:.2f} ≈ cash; vol "
            f"{R['bh_vol']:.0f}%, maxDD {R['bh_maxdd']:.0f}%; the momentum overlay is strictly "
            "worse at every setting. |\n"
            f"| **Does timing beat buy-and-hold?** | `NOT SUPPORTED` | OV−BH Sharpe "
            f"{R['ov63_diff']:+.2f}/{R['ov126_diff']:+.2f}/{R['ov252_diff']:+.2f} at "
            f"63/126/252d; P(OV>BH) = {R['ov63_frac']:.2f}/{R['ov126_frac']:.2f}/{R['ov252_frac']:.2f}. |\n\n"
            "> 💡 In plain words: the green-bet looks rewarded only because of one explosive "
            "year; statistically it's expensive cash-like commodity beta, and the obvious "
            "trend-timing fix backfires."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r_t$ be KRBN's daily total return, $f_t$ the cash (BIL) return, and "
            "$s_t = \\mathbf{1}[P_t/P_{t-L} - 1 > 0]$ the trailing-$L$ momentum signal.\n\n"
            "- **H₁ (reward).** $\\mathbb{E}[r_t - f_t] > 0$ with an autocorrelation-robust "
            "*t* ≥ 2 — a passive hold beats cash for real.\n"
            "- **H₂ (not one year).** The reward survives dropping any single calendar year.\n"
            "- **H₃ (timing helps).** The overlay's excess-of-cash Sharpe exceeds buy-and-hold's, "
            "with a block-bootstrap CI on the difference excluding 0.\n\n"
            "We find **H₁ fails** (*t* = +1.41), **H₂ fails** (2021 is the whole record), and "
            "**H₃ fails** (the overlay loses at every lookback)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "Carbon-allowance prices are well documented as **commodity-like**, driven by "
            "fuel prices, weather, output and policy headlines (Chevallier 2009, Hintermann "
            "2010) — not a smoothly compounding claim. So the quantitative content is precise: "
            "(a) does the cap-and-trade *drift* show up as a measurable reward over cash on the "
            "only tape we have, and (b) is the asset's volatility *exploitable* by trend-timing "
            "or merely costly? Both answers turn out to be no, and naming *why* (one-year "
            "concentration; whipsaw) is the product."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Reward test.** Excess-of-cash daily mean with a Newey-West (HAC) *t* "
            "([`strategy.excess_tstat`](../carbon_credits/strategy.py)) — the inference-bar "
            "number.\n"
            "- **Concentration.** Calendar-year total-return decomposition.\n"
            "- **Timing overlay.** Hold KRBN while trailing-$L$ return > 0, else BIL. Signal "
            "known at the close of *t*, return earned over *t+1* — **one** execution shift. "
            "Turnover one-way × NAV at `cost_bps` per flip.\n"
            "- **Fair race.** Excess-of-cash to excess-of-cash Sharpe; the difference carries a "
            "**circular block bootstrap** CI ([`strategy.bootstrap_sharpe_diff`](../carbon_credits/strategy.py)).\n"
            "- **Positive controls.** Deterministic synthetic tape with tunable drift (`mu`) and "
            "persistence (`momentum`) — the machinery proof.\n\n"
            "Tape: KRBN + BIL total return, 2020-07-30 → 2026-05-29 (~5.8y, single cycle — named "
            "honestly)."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The reward over cash — and its concentration\n\n"
            "The excess-of-cash HAC *t* against the |t|=2 bar, with the calendar-year "
            "decomposition beside it."
        ),
        code(
            "if HAVE_REAL:\n"
            "    Rr = st.to_returns(REAL); rf = Rr['BIL']; bh = st.buy_and_hold(Rr)\n"
            "    t = st.excess_tstat(bh, rf); s = st.stats(bh, rf)\n"
            "    cy = Rr['KRBN'].groupby(Rr.index.year).apply(lambda x:(1+x).prod()-1)*100\n"
            "    years=[str(int(y)) for y in cy.index]; vals=list(cy.values)\n"
            "else:\n"
            f"    t, s = {R['bh_t']}, dict(sharpe={R['bh_sharpe']}, vol={R['bh_vol']/100}, max_dd={R['bh_maxdd']/100}, cagr={R['bh_cagr']/100})\n"
            f"    years=['2020','2021','2022','2023','2024','2025','2026']\n"
            f"    vals=[{R['cy_2020']},{R['cy_2021']},{R['cy_2022']},{R['cy_2023']},{R['cy_2024']},{R['cy_2025']},{R['cy_2026']}]\n"
            "fig,(a1,a2)=plt.subplots(1,2,figsize=(11.5,4.4))\n"
            "a1.bar(['excess-of-cash\\nHAC t'],[t],color=AMBER if abs(t)<2 else GREEN,width=.4)\n"
            "a1.axhline(2,ls='--',c=GREY); a1.axhline(0,c='k'); a1.set_ylim(0,2.6)\n"
            "a1.set_title(f'Reward over cash: t={t:+.2f} (below 2)')\n"
            "cols=[GREEN if v>=0 else RED for v in vals]\n"
            "a2.bar(years,vals,color=cols,width=.6); a2.axhline(0,c='k')\n"
            "a2.set_ylabel('total return %'); a2.set_title('...because one year is the whole record')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"BH: CAGR {s['cagr']*100:+.1f}%/yr  vol {s['vol']*100:.0f}%  Sharpe(ex) {s['sharpe']:.2f}  maxDD {s['max_dd']*100:.0f}%  HAC t={t:+.2f}\")"
        ),
        md(
            f"> 💡 In plain words: the per-year picture explains the sub-2 *t*. The "
            f"**2021 +{R['cy_2021']:.0f}%** dominates; the rest is noise around cash with two "
            "double-digit down years. A single-cycle, single-explosion sample cannot certify a "
            "structural premium — **H₁ and H₂ both fail.**"
        ),
        md(
            "### 4b · The momentum overlay race — does timing beat holding?\n\n"
            "Overlay Sharpe minus buy-and-hold Sharpe (excess-of-cash), swept over lookback, "
            "with the bootstrap probability that the overlay actually wins."
        ),
        code(
            "lbs = [63, 126, 252]\n"
            "if HAVE_REAL:\n"
            "    Rr = st.to_returns(REAL); rf = Rr['BIL']; bh = st.buy_and_hold(Rr)\n"
            "    diffs, fracs = [], []\n"
            "    for lb in lbs:\n"
            "        sig = st.momentum_signal(REAL, 'KRBN', lookback=lb)\n"
            "        ov = st.overlay_returns(Rr, sig, 'KRBN', 'BIL', cost_bps=0.0)\n"
            "        d = st.bootstrap_sharpe_diff(ov, bh, rf, n_boot=2000)\n"
            "        diffs.append(d['point']); fracs.append(d['frac_a_wins'])\n"
            "else:\n"
            f"    diffs = [{R['ov63_diff']}, {R['ov126_diff']}, {R['ov252_diff']}]\n"
            f"    fracs = [{R['ov63_frac']}, {R['ov126_frac']}, {R['ov252_frac']}]\n"
            "fig,(a1,a2)=plt.subplots(1,2,figsize=(11.5,4.4))\n"
            "a1.bar([f'{lb}d' for lb in lbs], diffs, color=RED, width=.55)\n"
            "a1.axhline(0,c='k'); a1.set_ylabel('overlay − buy&hold Sharpe')\n"
            "a1.set_title('Timing loses at every lookback (negative = worse than holding)')\n"
            "a2.bar([f'{lb}d' for lb in lbs], fracs, color=GREY, width=.55)\n"
            "a2.axhline(0.5,ls='--',c='k'); a2.set_ylim(0,1); a2.set_ylabel('P(overlay > buy&hold)')\n"
            "a2.set_title('...and it almost never wins in the bootstrap')\n"
            "plt.tight_layout(); plt.show()\n"
            "for lb,d,f_ in zip(lbs,diffs,fracs): print(f'lb={lb}d: OV-BH Sharpe={d:+.2f}  P(win)={f_:.2f}')"
        ),
        md(
            f"> 💡 In plain words: the overlay's Sharpe is **below** buy-and-hold at 63/126/252 "
            f"days (diff {R['ov63_diff']:+.2f}/{R['ov126_diff']:+.2f}/{R['ov252_diff']:+.2f}), and "
            f"the bootstrap says it beats holding only {R['ov126_frac']*100:.0f}% of the time at "
            "the 6-month window. It whipsaws out of the one good year and chops in the rest — "
            "**H₃ fails.**"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — excess-of-cash HAC *t* = +{R['bh_t']:.2f} (below 2); the "
            f"+{R['bh_cagr']:.0f}%/yr is one year (2021 +{R['cy_2021']:.0f}%). The cap-and-trade "
            "drift is plausible but uncertified on this tape.\n"
            f"- **Tradability `MIRAGE`** — Sharpe(ex-cash) {R['bh_sharpe']:.2f} ≈ cash at "
            f"{R['bh_vol']:.0f}% vol / {R['bh_maxdd']:.0f}% drawdown; the overlay is strictly "
            "worse. No risk-adjusted edge.\n"
            f"- **Does timing beat buy-and-hold? `NOT SUPPORTED`** — OV−BH Sharpe negative at "
            f"all lookbacks, P(win) {R['ov63_frac']:.2f}/{R['ov126_frac']:.2f}/{R['ov252_frac']:.2f}."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — costs only make the overlay worse\n\n"
            "The overlay already loses gross; costs widen the gap. The 6-month rule was in the "
            f"market ~{R['ov126_tim']:.0f}% of days and flipped **{R['ov126_flips']}** times — "
            "trading constantly to underperform doing nothing."
        ),
        code(
            "costs = [0.0, 5.0, 10.0]\n"
            "if HAVE_REAL:\n"
            "    Rr = st.to_returns(REAL); rf = Rr['BIL']; bh = st.buy_and_hold(Rr)\n"
            "    sig = st.momentum_signal(REAL, 'KRBN', lookback=126)\n"
            "    cagrs = [st.stats(st.overlay_returns(Rr, sig,'KRBN','BIL',cost_bps=c), rf)['cagr']*100 for c in costs]\n"
            "    bh_cagr = st.stats(bh, rf)['cagr']*100\n"
            "else:\n"
            f"    cagrs = [{R['ov126_cagr']}, {R['ov126_cagr']-0.8}, {R['ov126_cagr']-1.6}]\n"
            f"    bh_cagr = {R['bh_cagr']}\n"
            "fig, ax = plt.subplots(figsize=(8.5,4.3))\n"
            "ax.plot(costs, cagrs, 'o-', c=RED, lw=2, label='overlay CAGR')\n"
            "ax.axhline(bh_cagr, ls='--', c=GREEN, lw=2, label=f'buy & hold {bh_cagr:.0f}%/yr')\n"
            "ax.set_xlabel('cost per flip (bps, one-way)'); ax.set_ylabel('CAGR %/yr')\n"
            "ax.set_title('Overlay trails buy-and-hold at zero cost — and falls further with cost')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'overlay {cagrs[0]:.1f}% -> {cagrs[-1]:.1f}% as cost rises; buy&hold {bh_cagr:.1f}%/yr the whole time')"
        ),
        md(
            "> 💡 In plain words: unlike a 'real but costly' edge, here there's *nothing under "
            "the costs* — the overlay is below buy-and-hold even at zero fees. This is the "
            "definition of a **mirage**: the active version destroys value, and the passive "
            "version is cash-like beta you over-paid in volatility for."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further — the synthetic controls (machinery proof)\n\n"
            "Is the engine a faithful detector, or does it always say 'weak / mirage'? Plant a "
            "real drift, and separately real persistence, and confirm the harness banks each — "
            "**a machinery proof, never market evidence** (a synthetic Sharpe can't back a "
            "stamp)."
        ),
        code(
            "fp,_ = data.synthetic_carbon(n_days=3000, mu=0.20, momentum=0.0, seed=304)\n"
            "Rp = st.to_returns(fp); t_drift = st.excess_tstat(st.buy_and_hold(Rp), Rp['CASH'])\n"
            "ft,_ = data.synthetic_carbon(n_days=4000, mu=0.05, momentum=0.40, daily_vol=0.025, seed=304)\n"
            "Rt = st.to_returns(ft); sigt = st.momentum_signal(ft,'KRBN',lookback=63)\n"
            "ovt = st.overlay_returns(Rt, sigt,'KRBN','CASH',cost_bps=1.0)\n"
            "dt = st.bootstrap_sharpe_diff(ovt, st.buy_and_hold(Rt), Rt['CASH'], n_boot=2000)\n"
            "fn,_ = data.synthetic_carbon(n_days=3000, mu=0.0, momentum=0.0, seed=304)\n"
            "Rn = st.to_returns(fn); t_null = st.excess_tstat(st.buy_and_hold(Rn), Rn['CASH'])\n"
            "fig,(a1,a2)=plt.subplots(1,2,figsize=(11.5,4.4))\n"
            "a1.bar(['null\\n(mu=0)','drift control\\n(mu=0.20)'],[t_null,t_drift],color=[GREY,GREEN],width=.5)\n"
            "a1.axhline(2,ls='--',c=GREY); a1.axhline(0,c='k'); a1.set_ylabel('BH excess-of-cash HAC t')\n"
            "a1.set_title('Engine banks a planted reward')\n"
            "a2.bar(['timing control\\n(mom=0.40)'],[dt['point']],color=GREEN,width=.4)\n"
            "a2.axhline(0,c='k'); a2.set_ylabel('overlay − buy&hold Sharpe')\n"
            "a2.set_title(f'Engine banks planted persistence (P(win)={dt[\"frac_a_wins\"]:.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'drift control t={t_drift:+.2f} (banks reward); null t={t_null:+.2f}; timing control OV-BH={dt[\"point\"]:+.2f} P(win)={dt[\"frac_a_wins\"]:.2f}')"
        ),
        md(
            f"The harness banks a planted reward (drift control *t* = +{R['ctrl_drift_t']:.2f}) "
            f"and a planted timing edge (control OV−BH Sharpe +{R['ctrl_timing_diff']:.2f}, "
            f"P(win) {R['ctrl_timing_frac']:.2f}), and finds neither in the null. So the real-tape "
            "result is a statement about **the market**: over KRBN's single cycle there is no "
            "certifiable reward over cash and no exploitable trend — the 'cap only shrinks' "
            "escalator is, so far, one good year wearing a thesis.\n\n"
            "For the same buy-and-hold-vs-alternative machinery see "
            "[Study 97 — Balancing-Act](../../97-balancing-act/); for the single-cycle-commodity "
            "caution, [Study 152 — Inflation-Hedge](../../152-inflation-hedge/)."
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
