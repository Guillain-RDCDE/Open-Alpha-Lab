"""Generate the two narrative notebooks for Study 594 (Leverage Rotation, TQQQ + 200SMA).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached
QQQ/TQQQ/^IRX tape under ../_cache/ and otherwise quote the frozen headline numbers in
``R`` (mirroring docs/results.md). The synthetic control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance QQQ/TQQQ/^IRX,
# signal live 1999-12-22 -> 2026-06-30, 26.5 yrs, net of 5 bps unless labeled).
R = dict(
    asof="2026-06-30", fingerprint="82b00d3939c8",
    start="1999-12-22", end="2026-06-30", years=26.5, n_days=6669,
    exposure_pct=73.7, switches=186, switches_yr=7.0,
    # sim-3x audit vs real TQQQ 2010-02-12 -> 2026-06-30
    val=dict(corr=0.9989, resid_pct=-0.04, rms_bps=18.6, nav_ratio=1.080,
             fee_pct=2.5, n_days=4119),
    # legs at 5 bps: (CAGR%, vol%, Sharpe, MaxDD%, final multiple)
    rot=(11.50, 50.9, 0.433, -95.14, 17.81),
    qqq=(9.00, 26.8, 0.384, -82.96, 9.77),
    sim3x=(-2.11, 80.4, 0.353, -99.98, 0.57),
    # HAC t on daily differences
    hac_rot_qqq=(4.67, 1.66), hac_rot_3x=(-2.50, -0.61),
    # random-timer baseline (40 seeds)
    rt=dict(avg_welch=0.05, sd_welch=0.37, seeds=40, real_sharpe=0.433,
            rand_sharpe=0.314, beat_pct=78, rand_cagr=-0.11),
    # above/below-SMA conditioning
    cond=dict(mean_above=5.08, mean_below=4.18, t_mean=0.14,
              vol_above=19.8, vol_below=40.4, ratio=2.05, t_var=13.79),
    # cost sweep: (bps, CAGR%, Sharpe, MaxDD%)
    costs=[(0, 11.89, 0.440, -95.1), (2, 11.73, 0.437, -95.1),
           (5, 11.50, 0.433, -95.1), (10, 11.10, 0.426, -95.2)],
    # SMA windows: (window, CAGR%, MaxDD%, switches, HAC t vs QQQ)
    windows=[(150, 10.24, -98.6, 220, 1.37), (200, 11.50, -95.1, 186, 1.66),
             (250, 14.38, -86.9, 146, 2.11)],
    # sub-periods: (label, rot%, rot dd%, qqq%, qqq dd%, x3%, x3 dd%, switches)
    subs=[("dot-com 2000-03 -> 2002-10", -91.8, -92.3, -82.7, -83.0, -100.0, -100.0, 21),
          ("GFC 2007-10 -> 2009-03", -54.6, -56.6, -52.3, -53.4, -94.4, -94.6, 15),
          ("COVID 2020-02-19 -> 04-30", -41.5, -55.3, -6.5, -28.6, -38.5, -69.7, 4),
          ("2022 bear", -42.9, -45.6, -32.6, -34.8, -79.0, -80.9, 9),
          ("2020 full year", 100.9, -55.3, 48.4, -28.6, 111.4, -69.7, 4)],
    # $10k at the 2000-03-24 peak: (trough$, MaxDD%, made-whole, yrs underwater, final$)
    cohort=dict(rotation=(507, -95.14, "2018-06-14", 18.2, 104982),
                qqq=(1729, -82.96, "2015-02-13", 14.9, 75338),
                tqqq=(2, -99.98, "NEVER", None, 3349)),
    # real-TQQQ cross-check 2010-02-12+: (CAGR%, Sharpe, MaxDD%)
    real2010=dict(rot=(33.30, 0.820, -55.9), qqq=(19.91, 0.913, -35.1),
                  tqqq=(44.12, 0.883, -81.7),
                  t_vs_qqq=2.52, t_vs_tqqq=-2.02, corr_rot=0.9992),
    # synthetic control: (world, avg Welch t, mean-channel t)
    syn=[("null (iid)", -0.18, -0.67), ("planted regimes", 3.65, 6.24)],
)

BADGES = (
    "![Signal: Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![2000 cohort survives?: Busted](https://img.shields.io/badge/2000_cohort_survives%3F-Busted-8b949e?style=flat-square)\n\n"
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

from leverage_rotation_200sma import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    PX = data.load_real()
    QQQ = data.qqq_returns(PX)
    RF = data.rf_daily(PX)
    SIM3 = data.sim3x_returns(QQQ, RF)
    TQQQ = data.tqqq_returns(PX)
    SIG = st.sma_signal(PX["QQQ"].dropna())
    CASH = RF.reindex(QQQ.index).ffill()
    ROT, NSW = st.rotation_returns(SIM3, CASH, SIG, cost_bps=5.0)
else:
    PX = QQQ = RF = SIM3 = TQQQ = SIG = CASH = ROT = None
    NSW = 0
print("real tape cached:", HAVE_REAL, "| rotation days:", (0 if ROT is None else len(ROT)),
      "| switches:", NSW)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"

NAV_HELPER = """\
def nav(r):
    return np.exp(np.log1p(r.dropna()).cumsum())

def dd(r):
    n = nav(r)
    return n / n.cummax() - 1.0
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# \"Hold TQQQ above the 200-day line\" — 3x upside without 3x crashes? 🎢\n"
            "### The Reddit-famous leverage rotation, finally tested through the crash its backtests never show\n\n"
            + BADGES +
            "There's a plan that gets reposted on r/LETFs every single week. It goes: *TQQQ* (a fund "
            "that moves **3x** the Nasdaq-100 every day) *makes you rich in bull markets but loses 99% "
            "in crashes. So just hold TQQQ while the market is above its 200-day average, and hide in "
            "cash below it. You keep the 3x rocket, you dodge the crater.* The backtest attached to the "
            "post is always spectacular — and it always starts in **2010**.\n\n"
            "TQQQ was born in February 2010. The worst thing that ever happened to the Nasdaq — the "
            "2000-02 dot-com collapse, **−83%** on the index itself — is simply *not in the data* those "
            "charts use. So we rebuilt a 3x fund from QQQ's own daily returns (it matches the real TQQQ "
            "day-by-day at correlation **0.999**), ran the exact rule from 1999, and let the missing "
            "decade vote.\n\n"
            "> 📓 **Plain-language layer.** Want the t-stats, the matched random timers and the cost "
            "math? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚙️ Every chart below is drawn by the code beside it from the cached market tape; house "
            "style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does the 200-day line *know* something? | **Half of it.** Below the line the market is genuinely "
            "**twice as violent** (that part is rock-solid). But the *direction* of the next day is the same on "
            "both sides — the line predicts **storms, not returns**. |\n"
            "| Did the rotation beat just buying QQQ? | On paper, +11.5%/yr vs +9.0%/yr over 26.5 years — but "
            "statistically that gap is **not distinguishable from luck**, and random timers with the same "
            "exposure do just as well on average. |\n"
            "| Did it dodge the crater? | **No.** Started at the 2000 top, the \"protected\" plan still lost "
            "**95%** of the money and needed **18 years** to break even. |\n"
            "| So why does everyone post it? | Because the backtests start in 2010 — right *after* the only "
            "regime that kills it. Since 2010 it truly beat QQQ… and *lagged* just holding TQQQ. |\n"
        ),

        md(
            "## The race everyone imagines — and the one that actually happened\n\n"
            "Three ways to spend 26.5 years, starting the day the 200-day signal first exists "
            "(December 1999, net of 5 bps per switch): the **rotation**, plain **QQQ**, and **3x forever**."
        ),
        code(NAV_HELPER + """\
if HAVE_REAL:
    idx = ROT.index
    fig, ax = plt.subplots()
    ax.plot(nav(ROT), color=AMBER, lw=1.6, label="Rotation (3x above 200SMA, T-bills below)")
    ax.plot(nav(QQQ.reindex(idx)), color=GREY, lw=1.6, label="QQQ buy & hold")
    ax.plot(nav(SIM3.reindex(idx)), color=RED, lw=1.2, label="3x forever (synthesised TQQQ)")
    ax.set_yscale("log")
    ax.set_title("$1 from Dec-1999, log scale — the rotation wins, but look at the road")
    ax.set_ylabel("growth of $1 (log)")
    ax.legend(loc="upper left", frameon=False)
    plt.show()
    print("final multiples: rotation x%.2f | QQQ x%.2f | 3x-forever x%.2f"
          % (R["rot"][4], R["qqq"][4], R["sim3x"][4]))
else:
    print("cache missing - canonical numbers:", R["rot"], R["qqq"], R["sim3x"])
"""),
        md(
            "The rotation *does* finish ahead: **$1 → $17.8** vs **$9.8** for QQQ, while 3x-forever "
            "ends *below* $1 (the [melting-ice](../../100-melting-ice/) grave). That's the poster. "
            "Now the road:"
        ),
        code("""\
if HAVE_REAL:
    fig, ax = plt.subplots()
    ax.fill_between(dd(ROT).index, dd(ROT) * 100, 0, color=AMBER, alpha=.55,
                    label="Rotation drawdown")
    ax.plot(dd(QQQ.reindex(ROT.index)) * 100, color=GREY, lw=1.2, label="QQQ drawdown")
    ax.set_title("How far below the peak you are — the 'protected' plan hit −95%")
    ax.set_ylabel("% below high-water mark")
    ax.legend(loc="lower right", frameon=False)
    plt.show()
    print("max drawdown: rotation %.2f%% | QQQ %.2f%% | 3x-forever %.2f%%"
          % (R["rot"][3], R["qqq"][3], R["sim3x"][3]))
"""),

        md(
            "## Failure mode #1 — the parabolic-top trap (2000)\n\n"
            "The rule exits when price closes below the 200-day average. But in 1999 QQQ went "
            "**vertical** — so by the March-2000 top, the 200-day average was **~35% below the price**. "
            "A 3x holder loses roughly **three times that** before the exit even fires. Then the 2000-02 "
            "chop whipsawed the rule **21 times**, each re-entry buying a bear rally, each exit selling "
            "the next leg down."
        ),
        code("""\
if HAVE_REAL:
    px = PX["QQQ"].dropna().loc["1999-06-01":"2001-06-30"]
    sma = PX["QQQ"].dropna().rolling(200).mean().loc[px.index]
    fig, ax = plt.subplots()
    ax.plot(px, color=GREY, lw=1.5, label="QQQ close")
    ax.plot(sma, color=AMBER, lw=1.8, label="200-day SMA (the exit line)")
    gap_day = px.idxmax()
    ax.annotate("the exit line is ~35% below the top", xy=(gap_day, sma.loc[gap_day]),
                xytext=(gap_day, px.max() * 0.55),
                arrowprops=dict(arrowstyle="->", color=RED), color=RED)
    ax.set_title("2000: a parabolic run-up leaves the 200-day line miles below")
    ax.legend(frameon=False)
    plt.show()
    lbl, r1, d1, r2, d2, r3, d3, sw = R["subs"][0]
    print("dot-com window: rotation %+.1f%% (dd %+.1f%%) vs QQQ %+.1f%% - with %d whipsaw switches"
          % (r1, d1, r2, sw))
"""),

        md(
            "## Failure mode #2 — the V-crash whipsaw (COVID 2020)\n\n"
            "The opposite kind of crash kills it too. COVID fell **too fast** for a 200-day line (you "
            "ride the 3x fund most of the way down) and recovered **too fast** (you're in cash while it "
            "V-bounces). Between Feb-19 and Apr-30 2020, plain QQQ was down just **−6.5%**; the "
            "\"protected\" rotation was down **−41.5%** — it *sold low and bought back high*."
        ),
        code("""\
if HAVE_REAL:
    win = slice("2020-01-01", "2020-12-31")
    fig, ax = plt.subplots()
    for r, c, lb in [(ROT.loc[win], AMBER, "Rotation"),
                     (QQQ.reindex(ROT.index).loc[win], GREY, "QQQ B&H"),
                     (SIM3.reindex(ROT.index).loc[win], RED, "3x forever")]:
        ax.plot(nav(r), color=c, lw=1.5, label=lb)
    ax.set_title("2020: the rotation sells the bottom and misses the V")
    ax.set_ylabel("growth of $1 in 2020")
    ax.legend(frameon=False)
    plt.show()
    lbl, r1, d1, r2, d2, r3, d3, sw = R["subs"][2]
    print("COVID crash window: rotation %+.1f%% | QQQ %+.1f%% | 3x %+.1f%%" % (r1, r2, r3))
"""),

        md(
            "## The one thing the line really does know\n\n"
            "So is the 200-day line useless? No — it knows something real, just not what the pitch "
            "says. **Below the line, the market is about twice as violent** (annualised volatility "
            "~40% vs ~20% above). That is a genuine, statistically overwhelming fact of the tape. "
            "But the *average next-day return* is the same on both sides. The line forecasts "
            "**weather, not direction** — which is why it can smooth the ride of a 1x portfolio "
            "([Faber's rule](../../110-faber-timing/)) yet can't manufacture a return edge for a 3x one."
        ),
        code("""\
if HAVE_REAL:
    c = st.sma_conditioning(QQQ, SIG)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    axes[0].bar(["above line", "below line"], [c["vol_above_pct"], c["vol_below_pct"]],
                color=[GREEN, RED])
    axes[0].set_title("next-day volatility (ann. %)\\n<- the REAL discovery")
    axes[1].bar(["above line", "below line"], [c["mean_above_bps"], c["mean_below_bps"]],
                color=[GREY, GREY])
    axes[1].set_title("next-day average return (bps)\\n<- no difference")
    plt.tight_layout(); plt.show()
    print("vol below/above = %.2fx | return difference: statistically nothing (t = %+.2f)"
          % (R["cond"]["ratio"], R["cond"]["t_mean"]))
"""),

        md(
            "## The 2000 cohort's ledger — the third axis\n\n"
            "The promise is *\"3x upside without 3x crashes\"*. Here is $10,000 placed at the "
            "worst possible moment (the 2000-03-24 peak) under the three plans:\n\n"
            "| plan | worst point | back to $10k | 2026 value |\n|---|--:|---|--:|\n"
            "| **Rotation** | **$507** (−95%) | **2018** — 18.2 years | **$104,982** |\n"
            "| QQQ buy & hold | $1,729 (−83%) | 2015 — 14.9 years | $75,338 |\n"
            "| 3x forever | $2 (−99.98%) | never | $3,349 |\n\n"
            "The rotation *eventually* wins the money race — **if** you are the investor who watches "
            "$10,000 become $507 and keeps following the spreadsheet for eighteen more years. A 95% "
            "loss **is** a 3x crash. The promise is **busted**; what survives is only the comparison "
            "with something even worse.\n\n"
            "> 🔬 **For the quants:** the return edge never certifies — HAC *t* = +1.66 vs QQQ over the "
            "full tape, Welch *t* = +0.05 vs exposure-matched random timers; the vol channel is the only "
            "real signal (*t* = +13.79). The 2010+ window shows *t* = +2.52 vs QQQ — and *t* = −2.02 vs "
            "just holding TQQQ. Full gauntlet in [02_for_the_quants.ipynb](02_for_the_quants.ipynb).\n\n"
            "## Verdict\n\n"
            "**Signal: MIXED** — the 200-day line is a real *volatility* switch, not a return picker. "
            "**Tradability: MIRAGE** — costs don't hurt it; the −95% path and the fast-crash whipsaws do. "
            "**\"The 2000 cohort survives\": BUSTED** — they got exactly the crash they were promised "
            "they wouldn't.\n\n"
            "*Research & education, not investment advice. Numbers: [docs/results.md](../docs/results.md), "
            "reproduced by `python examples/verify.py`.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata={"language_info": {"name": "python"}})
    return nb


# ===========================================================================
# 02 — FOR THE QUANTS
# ===========================================================================
def build_quants():
    cells = [
        md(
            "# Leverage Rotation (TQQQ + 200SMA) — the quant teardown 🎢\n\n"
            + BADGES +
            "**Claim.** Hold TQQQ (3x daily Nasdaq-100) while QQQ closes above its 200-day SMA; "
            "T-bills below (Gayed 2016, *Leverage for the Long Run*; r/LETFs folklore). Promise: "
            "*3x upside without 3x crashes.*\n\n"
            "**Design.** Signal on QQQ's close vs its own 200SMA; position effective the NEXT day "
            "(one execution lag, applied once). Instrument: real TQQQ where it exists (2010-02→), a "
            "synthesised 3x before that: `r₃ₓ = 3·r_QQQ − 2·rf − 2.5%/252` with rf = ^IRX, all-in fee "
            "calibrated once vs the real fund. Costs one-way × full NAV per switch (headline 5 bps). "
            "Cash leg accrues ^IRX; **Sharpe races are excess-vs-excess**. As-of "
            f"**{R['asof']}**, fingerprint `{R['fingerprint']}`; canonical numbers frozen in `R` "
            "(mirror of [docs/results.md](../docs/results.md))."
        ),
        code(BOOT_CELL),

        md(
            "## 1 — The instrument audit: synthesised 3x vs real TQQQ\n\n"
            "Everything pre-2010 rides on the synthesis, so it gets audited first, on the "
            "2010-2026 overlap.\n\n"
            "> 💡 **In plain words:** we rebuilt TQQQ from QQQ's daily moves and a financing bill. "
            "On the 16 years where the real fund exists, the copy tracks it almost perfectly — so "
            "using the copy for 1999-2010 is measurement, not imagination."
        ),
        code("""\
if HAVE_REAL:
    v = data.validate_sim3x(PX)
    print("common span %s -> %s (%d days)" % (v["start"], v["end"], v["n_days"]))
    print("daily-return corr    : %.4f" % v["corr"])
    print("residual drag        : %+.2f%%/yr after the %.1f%%/yr all-in fee"
          % (v["resid_drag_ann_pct"], data.SIM3X_FEE * 100))
    print("RMS tracking error   : %.1f bps/day" % v["rms_te_bps"])
    print("terminal-NAV ratio   : %.3f (real / sim)" % v["nav_ratio"])
    common = TQQQ.index.intersection(SIM3.index)
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.scatter(SIM3.loc[common] * 100, TQQQ.loc[common] * 100, s=4, alpha=.25, color=GREY)
    lim = 18
    ax.plot([-lim, lim], [-lim, lim], color=RED, lw=1)
    ax.set_xlabel("synthesised 3x, daily %"); ax.set_ylabel("real TQQQ, daily %")
    ax.set_title("corr %.4f — the pre-2010 extension is audited" % v["corr"])
    plt.show()
"""),

        md(
            "## 2 — Headline race (net of 5 bps, 1999-12-22 → 2026-06-30)\n\n"
            "> 💡 **In plain words:** over the full history the rotation earns ~2.5 pp/yr more than "
            "QQQ — with double the volatility and a deeper worst loss. The 3x-forever leg is the "
            "melting-ice grave."
        ),
        code("""\
if HAVE_REAL:
    idx = ROT.index
    rows = [("rotation", ROT), ("QQQ B&H", QQQ.reindex(idx)), ("3x B&H (sim)", SIM3.reindex(idx))]
    print("%-14s %8s %7s %8s %9s %8s" % ("leg", "CAGR%", "vol%", "Sharpe", "MaxDD%", "$1 -> "))
    for nm, r in rows:
        s = st.perf_stats(r, RF)
        print("%-14s %+8.2f %7.1f %+8.3f %+9.2f %8.2f" %
              (nm, s["cagr_pct"], s["vol_pct"], s["sharpe"], s["maxdd_pct"], s["final_mult"]))
    pos = st.positions(SIG, SIM3.index)
    print("exposure %.1f%% of days | %d switches (%.1f/yr)"
          % (pos.mean() * 100, NSW, NSW / (len(idx) / 252)))
"""),

        md(
            "## 3 — Is the timing statistically real? (HAC + matched random timers)\n\n"
            "Two tests, both on the real tape:\n"
            "1. **HAC (Newey-West) t** on the daily return difference rotation − benchmark.\n"
            "2. **Exposure-and-switch-matched random timers** — permute the rotation's own in/out "
            "run lengths (exposure fraction and switch count preserved exactly), rebuild the "
            "strategy per seed, average the Welch *t* over 40 seeds (single-seed baselines are "
            "banned desk-wide).\n\n"
            "> 💡 **In plain words:** if the rule has skill, it must beat (a) plain QQQ and (b) a "
            "monkey flipping the same number of switches with the same market time. It does neither "
            "at any certifiable level."
        ),
        code("""\
if HAVE_REAL:
    h1 = st.hac_t(ROT - QQQ.reindex(ROT.index))
    h2 = st.hac_t(ROT - SIM3.reindex(ROT.index))
    print("rotation - QQQ B&H : %+.2f bps/day  HAC t = %+.2f (lags %d, n %d)"
          % (h1["mean_bps"], h1["t"], h1["lags"], h1["n"]))
    print("rotation - 3x  B&H : %+.2f bps/day  HAC t = %+.2f" % (h2["mean_bps"], h2["t"]))
    pos = st.positions(SIG, SIM3.index)
    rt = st.random_timer_test(SIM3, CASH, pos, cost_bps=5.0, n_seeds=40)
    print("\\nrandom-timer baseline (40 seeds, exposure+switches matched):")
    print("  seed-averaged Welch t = %+.2f (sd %.2f)" % (rt["avg_welch_t"], rt["std_welch_t"]))
    print("  real Sharpe %+.3f vs random mean %+.3f | beats %.0f%% of seeds"
          % (rt["real_sharpe"], rt["rand_sharpe_mean"], rt["beat_frac"] * 100))
"""),

        md(
            "## 4 — Decomposition: WHY it half-works (mean channel vs vol channel)\n\n"
            "Condition next-day QQQ on the (lagged) SMA state.\n\n"
            "> 💡 **In plain words:** below the line the market is twice as stormy — that's real and "
            "enormous. But the average next-day move is the same on both sides. The SMA forecasts "
            "*variance*, not *drift*; a variance forecast helps a 1x portfolio's Sharpe but cannot "
            "certify a CAGR edge for a 3x rocket."
        ),
        code("""\
if HAVE_REAL:
    c = st.sma_conditioning(QQQ, SIG)
    print("mean channel: above %+.2f bps (n=%d) vs below %+.2f bps (n=%d)  Welch t = %+.2f"
          % (c["mean_above_bps"], c["n_above"], c["mean_below_bps"], c["n_below"], c["welch_t_mean"]))
    print("vol  channel: above %.1f%% vs below %.1f%% ann. (%.2fx)  Welch t (sq. ret) = %+.2f"
          % (c["vol_above_pct"], c["vol_below_pct"],
             c["vol_below_pct"] / c["vol_above_pct"], c["welch_t_var"]))
"""),

        md(
            "## 5 — Costs and the SMA-window sweep\n\n"
            "> 💡 **In plain words:** trading costs are a rounding error at 7 switches/yr — the "
            "strategy's problem was never the broker. And the *t* crosses 2 only at the post-hoc "
            "best window (250 days), the classic shape of parameter fishing."
        ),
        code("""\
if HAVE_REAL:
    print("cost sweep (one-way bps x NAV per switch):")
    for cb in (0.0, 2.0, 5.0, 10.0):
        r, _ = st.rotation_returns(SIM3, CASH, SIG, cost_bps=cb)
        s = st.perf_stats(r, RF)
        print("  %4.1f bps: CAGR %+6.2f%%  Sharpe %+.3f  MaxDD %+.1f%%"
              % (cb, s["cagr_pct"], s["sharpe"], s["maxdd_pct"]))
    print("\\nSMA-window sweep (net of 5 bps):")
    for w in (150, 200, 250):
        sg = st.sma_signal(PX["QQQ"].dropna(), window=w)
        r, ns = st.rotation_returns(SIM3, CASH, sg, cost_bps=5.0)
        r = r.reindex(ROT.index).dropna()
        s = st.perf_stats(r, RF)
        h = st.hac_t(r - QQQ.reindex(r.index))
        print("  SMA %d: CAGR %+6.2f%%  MaxDD %+.1f%%  switches %d  HAC t vs QQQ = %+.2f"
              % (w, s["cagr_pct"], s["maxdd_pct"], ns, h["t"]))
"""),

        md(
            "## 6 — Sub-periods: two failure modes on the tape\n\n"
            "> 💡 **In plain words:** slow bears (2022) are the rule's home turf. Parabolic tops "
            "(2000) leave the exit line miles below the price, and V-crashes (2020) whipsaw it — "
            "sell low, re-buy high."
        ),
        code("""\
if HAVE_REAL:
    pos = st.positions(SIG, SIM3.index)
    subs = st.SUBPERIODS + [("2020 full year", "2020-01-01", "2020-12-31")]
    print("%-24s %18s %18s %18s %s" % ("window", "rotation", "QQQ", "3x B&H", "switches"))
    for nm, a, b in subs:
        w1, w2, w3 = (st.window_stats(x, a, b) for x in (ROT, QQQ, SIM3))
        psub = pos.loc[a:b]
        nsw_sub = int((psub.values[1:] != psub.values[:-1]).sum())
        print("%-24s %+7.1f%% (dd %+5.1f%%) %+6.1f%% (dd %+5.1f%%) %+6.1f%% (dd %+5.1f%%)   %d"
              % (nm, w1["total_pct"], w1["maxdd_pct"], w2["total_pct"], w2["maxdd_pct"],
                 w3["total_pct"], w3["maxdd_pct"], nsw_sub))
"""),

        md(
            "## 7 — The third axis: the 2000 cohort's ledger\n\n"
            "$10,000 at the 2000-03-24 QQQ peak — the entry the 2010+ backtests cannot contain.\n\n"
            "> 💡 **In plain words:** the \"protected\" plan still fell to $507 and took 18 years to "
            "get back to even. It beats the alternative that dies ($2, never recovers) — but a −95% "
            "loss is precisely the 3x crash the plan promised away. **Busted.**"
        ),
        code("""\
if HAVE_REAL:
    co = st.cohort_2000(ROT, QQQ.reindex(ROT.index), SIM3.reindex(ROT.index))
    for nm, label in [("rotation", "rotation"), ("qqq_bh", "QQQ B&H"), ("tqqq_bh", "3x B&H")]:
        d = co[nm]
        rec = d["recovered_on"] or "NEVER"
        print("%-10s trough $%8.0f  maxDD %+7.2f%%  whole again %-11s final $%10.0f"
              % (label, d["trough_value"], d["maxdd_pct"], rec, d["final_value"]))
    fig, ax = plt.subplots()
    start = "2000-03-24"
    for r, c, lb in [(ROT.loc[start:], AMBER, "rotation"),
                     (QQQ.reindex(ROT.index).loc[start:], GREY, "QQQ B&H"),
                     (SIM3.reindex(ROT.index).loc[start:], RED, "3x forever")]:
        navs = 10_000 * np.exp(np.log1p(r.dropna()).cumsum())
        ax.plot(navs, color=c, lw=1.4, label=lb)
    ax.axhline(10_000, color="k", lw=.8, ls="--")
    ax.set_yscale("log"); ax.set_ylabel("$ (log)")
    ax.set_title("$10,000 at the 2000 top — the road each cohort actually walked")
    ax.legend(frameon=False)
    plt.show()
"""),

        md(
            "## 8 — The window the Reddit posts show (real TQQQ, 2010+)\n\n"
            "> 💡 **In plain words:** start the chart in 2010 and the rotation finally beats QQQ with "
            "a *t* above 2 — but that window begins right after the one regime that kills the rule "
            "(start-date selection, not certification). And on that same window it significantly "
            "**lagged** just holding TQQQ: the filter only cost money once the crash it guards "
            "against stopped happening."
        ),
        code("""\
if HAVE_REAL:
    rot_tq, nsw_tq = st.rotation_returns(TQQQ, CASH, SIG, cost_bps=5.0)
    print("corr(rotation on real TQQQ, rotation on sim 3x) = %.4f"
          % rot_tq.corr(ROT.reindex(rot_tq.index)))
    for nm, r in [("rotation (real TQQQ)", rot_tq),
                  ("QQQ B&H", QQQ.reindex(rot_tq.index)), ("TQQQ B&H", TQQQ)]:
        s = st.perf_stats(r, RF)
        print("%-22s CAGR %+7.2f%%  Sharpe %+.3f  MaxDD %+6.1f%%"
              % (nm, s["cagr_pct"], s["sharpe"], s["maxdd_pct"]))
    print("rotation - QQQ  (2010+): HAC t = %+.2f  <- bull-heavy sub-sample"
          % st.hac_t(rot_tq - QQQ.reindex(rot_tq.index))["t"])
    print("rotation - TQQQ (2010+): HAC t = %+.2f  <- lagged holding TQQQ outright"
          % st.hac_t(rot_tq - TQQQ.reindex(rot_tq.index))["t"])
"""),

        md(
            "## 9 — Synthetic control (machinery proof — never market evidence)\n\n"
            "A pipeline that can't bank a planted signal proves nothing by finding nothing. In an "
            "i.i.d. world (no trend structure) the rotation must NOT beat the matched random timer; "
            "with planted persistent bear regimes (−0.8%/day, 2x vol) it must light up.\n\n"
            "> 💡 **In plain words:** we test the detector on a world where the answer is known — "
            "both ways. It stays quiet when there is nothing and fires when there is something, so "
            "its silence on the real QQQ tape means something."
        ),
        code("""\
for eff, tag in [(0.0, "null (iid)     "), (0.008, "planted regimes")]:
    close = data.synthetic_world(effect=eff, n_days=7560)
    r = close.pct_change().dropna()
    zero = pd.Series(0.0, index=r.index)
    sg = st.sma_signal(close)
    p = st.positions(sg, r.index)
    res = st.random_timer_test(r, zero, p, cost_bps=5.0, n_seeds=40)
    cnd = st.sma_conditioning(r, sg)
    print("%s: avg Welch t = %+.2f (40 seeds) | real Sharpe %+.3f vs random %+.3f | "
          "mean-channel t = %+.2f"
          % (tag, res["avg_welch_t"], res["real_sharpe"], res["rand_sharpe_mean"],
             cnd["welch_t_mean"]))
print("\\n(the Welch t on squared returns is anti-conservative under heavy tails, which is")
print(" why the desk's real-tape vol-channel claim leans on t = +13.79 AND a 2.05x vol ratio)")
"""),

        md(
            "## Verdict\n\n"
            "- **Signal — MIXED.** Real on the volatility channel (below-SMA vol 2.05× above, Welch "
            "*t* = **+13.79**), none on the return channel (HAC *t* = **+1.66** vs QQQ; seed-averaged "
            "Welch *t* = **+0.05** vs matched random timers; mean channel *t* = +0.14).\n"
            "- **Tradability — MIRAGE.** Costs irrelevant (0→10 bps: 11.89%→11.10% CAGR); the path is "
            "the killer — **−95.14%** MaxDD, 18.2 years underwater, COVID whipsaw (−41.5% vs QQQ's "
            "−6.5%), and 2010+ the filter *lagged* raw TQQQ at *t* = **−2.02**. Regime-conditional "
            "leveraged beta.\n"
            "- **\"2000 cohort survives?\" — BUSTED.** Trough **$507 / $10,000**, whole again in "
            "2018. Only the comparison with the $2 grave survives.\n\n"
            "*Numbers: [docs/results.md](../docs/results.md) (as-of "
            f"{R['asof']}, fingerprint `{R['fingerprint']}`), reproduced by "
            "`python examples/verify.py`. Siblings: [110-faber-timing](../../110-faber-timing/), "
            "[100-melting-ice](../../100-melting-ice/). Research & education, not investment advice.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata={"language_info": {"name": "python"}})
    return nb


if __name__ == "__main__":
    for name, builder in [("01_for_the_curious.ipynb", build_curious),
                          ("02_for_the_quants.ipynb", build_quants)]:
        nb = builder()
        path = os.path.join(HERE, name)
        with open(path, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        print("wrote", path)
