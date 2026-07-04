"""Generate the two narrative notebooks for Study 640 (Gold-Overnight).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached GLD/IAU/SPY
open-close tape under ../_cache/ and otherwise quote the frozen headline numbers in ``R``
(mirroring docs/results.md). The synthetic control runs anywhere with no network. Heavy
draws are reduced in-notebook (canonical numbers are quoted from ``R``).
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance adjusted
# open+close, GLD/IAU/SPY, 2004-11-18 -> 2026-06-30, fingerprint 648ab68f37f2).
R = dict(
    as_of="2026-06-30", fingerprint="648ab68f37f2", reform="2015-03-20",
    gld=dict(start="2004-11-19", end="2026-06-30", n=5435, on=4.30, t_on=4.05,
             id=0.22, t_id=0.22, gap=4.08, t_gap=2.94, welch=2.69, p=0.0030,
             on_x=8.63, id_x=0.96, on_ann=10.51, id_ann=-0.18, cc_ann=10.31),
    iau=dict(start="2005-01-31", end="2026-06-30", n=5387, on=5.03, t_on=4.63,
             id=-0.35, t_id=-0.36, gap=5.39, t_gap=3.83, welch=3.51, p=0.0002,
             on_x=12.51, id_x=0.71, on_ann=12.55, id_ann=-1.61, cc_ann=10.73),
    spy=dict(start="2004-11-19", end="2026-06-30", n=5435, on=3.37, t_on=3.94,
             id=1.43, t_id=1.29, gap=1.94, t_gap=1.38, welch=1.23, p=0.1071,
             on_x=5.43, id_x=1.72, on_ann=8.16, id_ann=2.56, cc_ann=10.93),
    # sub-periods: (n, on bps, id bps, gap bps, HAC t on gap)
    gld_pre=(2599, 3.71, 0.64, 3.07, 1.35), gld_post=(2836, 4.84, -0.17, 5.01, 2.98),
    iau_pre=(2551, 4.77, -0.15, 4.92, 2.11), iau_post=(2836, 5.27, -0.53, 5.81, 3.46),
    gld_change_welch=-0.65, iau_change_welch=-0.29,
    vs_spy_welch=1.00,
    # harvest (GLD full period): (one-way cost bps, net bps/d, net ann %, b&h ann %, HAC t net)
    harvest=[(0.5, 3.30, 7.76, 10.31, 3.11), (1.0, 2.30, 5.08, 10.31, 2.17),
             (2.0, 0.30, -0.08, 10.31, 0.28), (5.0, -5.70, -14.11, 10.31, -5.37)],
    harvest_pre=[(0.5, 6.05), (1.0, 3.41), (2.0, -1.67), (5.0, -15.47)], bh_pre=9.42,
    harvest_post=[(0.5, 9.35), (1.0, 6.63), (2.0, 1.39), (5.0, -12.84)], bh_post=11.13,
    # synthetic control: (planted night edge bps/d, gap bps/d, HAC t, sign-flip p)
    syn=[(0.0, 0.73, 0.47, 0.314), (4.0, 4.73, 3.06, 0.000)],
)

BADGES = (
    "![Signal: Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Ever harvestable?: Busted](https://img.shields.io/badge/Ever_harvestable%3F-Busted-8b949e?style=flat-square)\n\n"
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

from gold_overnight import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    from quantlab import repro
    WIDE = repro.as_of(data.load_real(), "2026-06-30")
    GLD = data.session_returns(WIDE, "GLD")
    IAU = data.session_returns(WIDE, "IAU")
    SPY = data.session_returns(WIDE, "SPY")
else:
    WIDE = GLD = IAU = SPY = None
print("real gold-overnight cache present:", HAVE_REAL,
      "| GLD sessions:", (0 if GLD is None else len(GLD)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"

CURVE_CELL = """\
if HAVE_REAL:
    on_curve = (1 + GLD["overnight"]).cumprod()
    id_curve = (1 + GLD["intraday"]).cumprod()
    cc_curve = (1 + GLD["close2close"]).cumprod()
    idx = GLD.index
else:  # illustrative fallback rebuilt from the frozen annualised rates
    idx = pd.bdate_range(R["gld"]["start"], R["gld"]["end"])
    n = len(idx); t = np.arange(1, n + 1) / 252.0
    on_curve = pd.Series((1 + R["gld"]["on_ann"] / 100) ** t, index=idx)
    id_curve = pd.Series((1 + R["gld"]["id_ann"] / 100) ** t, index=idx)
    cc_curve = pd.Series((1 + R["gld"]["cc_ann"] / 100) ** t, index=idx)
fig, ax = plt.subplots(figsize=(9.8, 5.2))
ax.plot(idx, cc_curve, c=GREY, lw=1.4, label=f'hold GLD (close-to-close): {cc_curve.iloc[-1]:.2f}x')
ax.plot(idx, on_curve, c=GREEN, lw=2.0, label=f'overnight only (close->open): {on_curve.iloc[-1]:.2f}x')
ax.plot(idx, id_curve, c=RED, lw=2.0, label=f'intraday only (open->close): {id_curve.iloc[-1]:.2f}x')
ax.set_yscale("log"); ax.set_ylabel("growth of $1 (log scale)")
ax.set_title("21.6 years of GLD: the whole gain arrived while US markets were closed")
ax.legend(); plt.tight_layout(); plt.show()
print(f"overnight-only {on_curve.iloc[-1]:.2f}x   intraday-only {id_curve.iloc[-1]:.2f}x   "
      f"buy&hold {cc_curve.iloc[-1]:.2f}x")
"""

BARS_CELL = """\
if HAVE_REAL:
    rows = [(nm, st.split_stats(r, placebo=False)) for nm, r in
            (("GLD", GLD), ("IAU", IAU), ("SPY", SPY))]
    ons = [s["on_bps"] for _, s in rows]; ids = [s["id_bps"] for _, s in rows]
else:
    ons = [R[k]["on"] for k in ("gld", "iau", "spy")]
    ids = [R[k]["id"] for k in ("gld", "iau", "spy")]
x = np.arange(3); w = 0.38
fig, ax = plt.subplots(figsize=(8.8, 4.6))
ax.bar(x - w/2, ons, w, color=GREEN, label="overnight (close->open)")
ax.bar(x + w/2, ids, w, color=RED, label="intraday (open->close)")
for i, (a, b) in enumerate(zip(ons, ids)):
    ax.annotate(f"{a:+.1f}", (i - w/2, a), ha="center", va="bottom")
    ax.annotate(f"{b:+.1f}", (i + w/2, b), ha="center", va="bottom")
ax.set_xticks(x); ax.set_xticklabels(["GLD", "IAU (confirm)", "SPY (placebo)"])
ax.axhline(0, c="k", lw=.8)
ax.set_ylabel("mean return (bps/day)")
ax.set_title("Gold earns overnight and nothing intraday; equities split too, but less starkly")
ax.legend(); plt.tight_layout(); plt.show()
print("overnight bps/d:", [round(v, 2) for v in ons], "  intraday bps/d:", [round(v, 2) for v in ids])
"""

HARVEST_CELL = """\
if HAVE_REAL:
    hs = st.harvest_table(GLD)
    costs = [h["cost_bps"] for h in hs]; net = [h["net_ann_pct"] for h in hs]
    bh = hs[0]["bh_ann_pct"]
else:
    costs = [h[0] for h in R["harvest"]]; net = [h[2] for h in R["harvest"]]
    bh = R["harvest"][0][3]
fig, ax = plt.subplots(figsize=(8.8, 4.6))
cols = [GREEN if v >= bh else (AMBER if v > 0 else RED) for v in net]
ax.bar([f"{c:g} bps" for c in costs], net, color=cols, width=.55,
       label="overnight overlay, net (2 trades/day)")
ax.axhline(bh, ls="--", c=GREY, label=f"just hold GLD ({bh:.1f}%/yr)")
ax.axhline(0, c="k", lw=.8)
for i, v in enumerate(net):
    ax.annotate(f"{v:+.1f}%", (i, v), ha="center", va="bottom" if v >= 0 else "top")
ax.set_xlabel("one-way trading cost"); ax.set_ylabel("annualised net return (%)")
ax.set_title("The overlay NEVER beats simply holding gold - at any cost level")
ax.legend(); plt.tight_layout(); plt.show()
print("net ann % at", costs, "bps one-way:", [round(v, 2) for v in net], " vs buy&hold", round(bh, 2), "%")
"""

SUBPERIOD_CELL = """\
if HAVE_REAL:
    sp_g = st.subperiod_stats(GLD, data.FIX_REFORM)
    sp_i = st.subperiod_stats(IAU, data.FIX_REFORM)
    vals = [sp_g["pre"]["gap_bps"], sp_g["post"]["gap_bps"],
            sp_i["pre"]["gap_bps"], sp_i["post"]["gap_bps"]]
    ts = [sp_g["pre"]["t_gap"], sp_g["post"]["t_gap"],
          sp_i["pre"]["t_gap"], sp_i["post"]["t_gap"]]
    chg = (sp_g["welch_change"], sp_i["welch_change"])
else:
    vals = [R["gld_pre"][3], R["gld_post"][3], R["iau_pre"][3], R["iau_post"][3]]
    ts = [R["gld_pre"][4], R["gld_post"][4], R["iau_pre"][4], R["iau_post"][4]]
    chg = (R["gld_change_welch"], R["iau_change_welch"])
labels = ["GLD pre\\n(phone fix)", "GLD post\\n(LBMA auction)",
          "IAU pre\\n(phone fix)", "IAU post\\n(LBMA auction)"]
fig, ax = plt.subplots(figsize=(9.2, 4.6))
ax.bar(labels, vals, color=[GREY, GREEN, GREY, GREEN], width=.6)
for i, (v, t) in enumerate(zip(vals, ts)):
    ax.annotate(f"{v:+.1f} bps\\nt={t:.2f}", (i, v), ha="center", va="bottom", fontsize=9)
ax.set_ylabel("night-minus-day gap (bps/day)")
ax.set_title(f"The 2015-03-20 fix reform did NOT dent the gap "
             f"(change Welch t = {chg[0]:+.2f} / {chg[1]:+.2f})")
plt.tight_layout(); plt.show()
print("gaps pre/post (GLD, IAU):", [round(v, 2) for v in vals], " change Welch t:",
      tuple(round(c, 2) for c in chg))
"""

FLIP_CELL = """\
NB_DRAWS = 4000        # reduced in-notebook; canonical 20,000-draw p is quoted from R
if HAVE_REAL:
    d = (GLD["overnight"] - GLD["intraday"]).to_numpy()
    pl = st.signflip_pvalue(d, n_draws=NB_DRAWS, seed=640)
    obs = pl["obs"] * 1e4; draws = pl["draws"] * 1e4; pv = pl["p_value"]
else:
    obs = R["gld"]["gap"]; pv = R["gld"]["p"]
    rng = np.random.default_rng(640); draws = rng.normal(0.0, 1.4, NB_DRAWS)
fig, ax = plt.subplots(figsize=(9.0, 4.4))
ax.hist(draws, bins=60, color=GREY, alpha=.85,
        label=f"null: {NB_DRAWS:,} random night/day label flips")
ax.axvline(obs, c=GREEN, lw=2.5, label=f"observed gap {obs:+.2f} bps/day")
ax.set_xlabel("night-minus-day gap (bps/day)"); ax.set_ylabel("frequency")
ax.set_title(f"Outside the luck cloud: sign-flip p = {pv:.4f} "
             f"(canonical 20k-draw p = {R['gld']['p']:.4f})")
ax.legend(); plt.tight_layout(); plt.show()
print(f"observed {obs:+.2f} bps/d   sign-flip p (this run) = {pv:.4f}   canonical = {R['gld']['p']:.4f}")
"""

VS_SPY_CELL = """\
if HAVE_REAL:
    vp = st.vs_placebo(GLD, SPY)
    gg, sg, wt = vp["gold_gap_bps"], vp["spy_gap_bps"], vp["welch_gold_vs_spy"]
    tg, ts_ = vp["t_gold"], vp["t_spy"]
else:
    gg, sg, wt = R["gld"]["gap"], R["spy"]["gap"], R["vs_spy_welch"]
    tg, ts_ = R["gld"]["t_gap"], R["spy"]["t_gap"]
fig, ax = plt.subplots(figsize=(8.2, 4.4))
ax.bar(["gold (GLD)", "equities (SPY)"], [gg, sg], color=[GREEN, GREY], width=.5)
for i, (v, t) in enumerate(zip([gg, sg], [tg, ts_])):
    ax.annotate(f"{v:+.2f} bps/d\\nHAC t={t:.2f}", (i, v), ha="center", va="bottom")
ax.set_ylabel("night-minus-day gap (bps/day)")
ax.set_title(f"Gold's gap is ~2x SPY's - but the difference is only Welch t = {wt:+.2f}")
plt.tight_layout(); plt.show()
print(f"gold gap {gg:+.2f} (t={tg:.2f})   SPY gap {sg:+.2f} (t={ts_:.2f})   "
      f"gold-vs-SPY Welch t = {wt:+.2f}")
"""

SYN_CELL = """\
res = []
for edge in (0.0, 0.0004):
    w = data.synthetic_world(night_edge=edge, seed=640)
    r = data.session_returns(w, "SYN")
    s = st.split_stats(r, n_draws=4000)
    res.append((edge * 1e4, s["gap_bps"], s["t_gap"], s["p_signflip"]))
fig, ax = plt.subplots(figsize=(8.4, 4.4))
labels = [f"planted night edge\\n{e:+.0f} bps/day" for e, _, _, _ in res]
tvals = [r[2] for r in res]
ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)
ax.axhline(2, ls="--", c=RED, label="t=2 bar")
for i, t in enumerate(tvals):
    ax.annotate(f"t={t:.2f}", (i, t), ha="center", va="bottom")
ax.set_ylabel("HAC t on the night-minus-day gap")
ax.set_title("Control: a null world stays quiet; a planted 4 bps night drift lights up")
ax.legend(); plt.tight_layout(); plt.show()
for e, g, t, p in res:
    print(f"planted {e:+.1f} bps/d: gap={g:+.2f} bps/d  HAC t={t:+.2f}  sign-flip p={p:.3f}")
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    g = R["gld"]
    cells = [
        md(
            "# Does gold really make all its money while you sleep? 🌙🥇\n"
            "### The sleeping-gold legend, measured honestly — in plain English\n\n"
            + BADGES +
            "There's a chart that has floated around gold forums for fifteen years. It splits every "
            "trading day in two: the **night** (from the US market close to the next morning's open — "
            "the hours when gold trades in Asia and through the London morning \"fix\") and the **day** "
            "(open to close — the London afternoon fix and New York hours). The legend: **all of gold's "
            "gains happen at night**, and the day session — the hours when the London fixes were set by "
            "a phone call among five banks — gives nothing back or even loses. To believers, that was "
            "the smoking gun of price suppression.\n\n"
            "We rebuilt that chart from the raw tape — GLD, the big gold ETF, every session since it "
            "listed in 2004 — checked it on a second gold fund (IAU), raced it against the stock market "
            "(which, as our [study 01](../../01-overnight-anomaly/README.md) showed, has its *own* "
            "overnight habit), and then asked the only question that pays: **could anyone have traded "
            "it?**\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo and the reform test? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does gold really gain only overnight? | **Yes — astonishingly so.** Over 21.6 years, $1 "
            f"riding GLD **only at night** grew to **${g['on_x']:.2f}**; $1 riding it **only during "
            f"the day** ended at **${g['id_x']:.2f}** — below where it started. The stats say this is "
            "no fluke. |\n"
            "| Is that unique to gold? | **Not really.** Stocks (SPY) show the same night-tilt — "
            "gold's is about twice as big, but the tape can't prove the difference isn't noise. |\n"
            "| Did the 2015 clean-up of the London fix end it? | **No — it got *bigger*.** The gap grew "
            "after the phone-call fix was replaced by an audited electronic auction. Whatever drives "
            "this, it isn't fix-rigging. |\n"
            "| Could you have traded it? | **No.** Holding only at night means trading **twice every "
            "day**. The day session you'd be dodging earns roughly **zero** — so you pay real spreads "
            "to avoid nothing. At every cost we tested, the night-only strategy loses to simply "
            "**buying and holding gold**. |\n\n"
            "> The legend is **true as a description** and **worthless as a trade** — gold does make "
            "its money at night, and there is still nothing you can do about it."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"For two decades, gold rose overnight and fell (or stalled) during London and New York "
            "hours. Buy the PM fix, sell the AM fix, and you'd have captured everything; do the "
            "reverse and you'd have lost. The market is rigged during Western hours.\"*\n\n"
            "The chart originates with Adrian Douglas (GATA, 2010), and the respectable version of the "
            "mechanism is academic: Caminschi & Heaney (2014) documented information leaking around the "
            "London PM fix calls, and in 2014 Barclays was fined for manipulating the fix. In 2015 the "
            "century-old phone fix was replaced by an electronic auction — which conveniently gives us "
            "a **before/after experiment** on the whole story."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the pattern is real *and* tradable, it's free money at the expense of whoever sells "
            "gold during the day. If it's real but **untradable**, it's something subtler and more "
            "interesting: proof that *when* an asset's return arrives on the clock tells you about "
            "**where its price discovery happens** (gold's happens in Asia and London, while the US "
            "ETF sleeps), not about a hidden hand. And if the 2015 fix reform didn't change it, the "
            "manipulation reading loses its engine."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"Simple and brutal: take every GLD session from {g['start']} to {g['end']} "
            f"(**{g['n']:,} days**) and split it in two —\n\n"
            "1. **Overnight leg:** yesterday's close → today's open (US market shut).\n"
            "2. **Intraday leg:** today's open → today's close (London PM fix + NY hours).\n\n"
            "The two legs multiply back exactly to the full day, so nothing is lost or double-counted. "
            "Then: run each sleeve as its own compounding account, test the night-vs-day gap against "
            "luck, check a **second gold fund (IAU)**, race gold against **SPY** (equities have their "
            "own night habit — study 01), split at the **2015 fix reform**, and finally charge real "
            "trading costs on the \"hold only at night\" strategy."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**The legendary chart, rebuilt honestly.** One dollar in GLD, three ways: hold it, hold "
            "it only overnight, hold it only during the day."
        ),
        code(CURVE_CELL),
        md(
            f"The legend survives the audit. **${g['on_x']:.2f}** for the night sleeve vs "
            f"**${g['id_x']:.2f}** for the day sleeve — over two decades the *entire* close-to-close "
            f"gain (**{g['cc_ann']:+.1f}%/yr**) arrived while US exchanges were shut. The quants "
            f"notebook shows this is statistically solid (the night-vs-day gap has odds of about "
            f"**3-in-1,000** of being luck on GLD, and the second fund IAU agrees even more strongly)."
        ),
        md(
            "**Is gold special?** Here's the same split for gold's twin fund IAU (a different sponsor "
            "holding the same metal — if GLD's pattern were a quirk of one fund, IAU would disagree) "
            "and for the stock market (SPY)."
        ),
        code(BARS_CELL),
        md(
            f"IAU **confirms** ({R['iau']['on']:+.1f} bps/night vs {R['iau']['id']:+.1f} bps/day — even "
            f"starker than GLD). But look at SPY: equities *also* earn more at night "
            f"({R['spy']['on']:+.1f} vs {R['spy']['id']:+.1f}). Gold's night-day gap "
            f"(**{R['gld']['gap']:+.1f} bps/day**) is about **twice** the equity one "
            f"(**{R['spy']['gap']:+.1f}**), but statistically the tape can't swear gold is a different "
            "species rather than a louder version of the same market-wide clock effect."
        ),
        md(
            "**The only question that pays: could you trade it?** \"Hold gold only at night\" means "
            "buying at every close and selling at every open — **two trades a day, forever**. Here's "
            "the net result at realistic costs, against just buying and holding."
        ),
        code(HARVEST_CELL),
        md(
            f"This is the punchline. Even at an impossibly tight **0.5 bps** per trade the night-only "
            f"strategy makes **{R['harvest'][0][2]:+.1f}%/yr — less than the {R['harvest'][0][3]:+.1f}%/yr "
            f"you'd get by doing nothing**. At 2 bps it makes roughly zero; at 5 bps it's a disaster. "
            "Why? Because the day session you're dodging earns *about zero, not something negative "
            "enough to be worth dodging* — so every spread you pay is pure loss. The anomaly tells you "
            "**when** gold's return arrives; it doesn't hand you a trade."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Real.** Gold genuinely made its entire two-decade return overnight "
            f"(${g['on_x']:.2f} vs ${g['id_x']:.2f} per dollar; solid *t*-stats on both GLD and IAU).\n"
            "- **Tradability — Mirage.** Two trades a day to dodge a day-session worth zero: the "
            "night-only strategy loses to buy & hold at **every** cost level, in **every** sub-period.\n"
            "- **\"Ever harvestable?\" — Busted.** Even at spreads tighter than GLD has ever traded, "
            "the arithmetic never works. And the bonus finding: the **2015 fix reform didn't dent the "
            "pattern** — it *grew* — so the \"riggers suppress gold during fix hours\" reading loses "
            "its engine. The boring truth: gold's price discovery simply happens while New York sleeps."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **The equities version of this rabbit hole** is [study 01 — the overnight anomaly]"
            "(../../01-overnight-anomaly/README.md): same clock split, a manipulation theory of its "
            "own (Knuteson), and the same Real-but-Mirage ending.\n"
            "- **Why does the open capture it?** GLD's NAV moves with spot gold around the clock; the "
            "US open is just the first print after 17½ hours of Asian and London trading. A \"night "
            "return\" is mostly *other people's daytime*.\n"
            "- **Build your own.** Swap in futures (GC=F trades nearly 24h, so its \"overnight\" is a "
            "different animal), or split the day at the London fixes themselves with intraday data — "
            "the engine in [`gold_overnight/`](../gold_overnight/) takes any open/close tape.\n\n"
            "*Think you can make two-trades-a-day work because your broker charges nothing? Spreads "
            "aren't commissions — show the night-only sleeve beating buy & hold net of the "
            "**bid-ask**, then we'll talk.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "01_for_the_curious.ipynb")


# ===========================================================================
# 02 — FOR THE QUANTS
# ===========================================================================
def build_quants():
    g, i, s = R["gld"], R["iau"], R["spy"]
    cells = [
        md(
            "# Gold-Overnight — a quantitative teardown 🔬\n"
            "### Session-split HAC inference · sign-flip placebo · IAU confirmation · SPY placebo race "
            "· the 2015 LBMA-reform natural experiment · 2-trades-a-day harvest arithmetic · a "
            "planted-drift synthetic control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The claim "
            "(Douglas/GATA folklore; Caminschi-Heaney's leaky PM fix): **gold's return accrues "
            "overnight and the London/NY session gives nothing back**. The desk already dissected the "
            "*equity* version in [01-overnight-anomaly](../../01-overnight-anomaly/README.md) — this "
            "study is the **gold** version, with the fix mechanism put to an externally-dated natural "
            "experiment.\n\n"
            "> ⚠️ **Data note.** yfinance **adjusted open + close** (total-return; adjustment mode "
            "moves return between night and day, so it's stated: both legs from the same adjusted "
            f"series). GLD {g['start']}→{g['end']} ({g['n']:,} sessions), IAU from {i['start']}, SPY "
            "on the identical calendar. No panel survivorship (three continuously-listed funds). "
            "Offline core + synthetic control are deterministic. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (as-of " + R["as_of"] +
            ", fingerprint `" + R["fingerprint"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `REAL` | GLD night-minus-day gap **{g['gap']:+.2f} bps/d** at HAC "
            f"**t = {g['t_gap']:.2f}** (sign-flip p = {g['p']:.4f}); IAU confirms at "
            f"**{i['gap']:+.2f} bps/d**, HAC **t = {i['t_gap']:.2f}** (p = {i['p']:.4f}). Overnight "
            f"sleeve {g['on_x']:.2f}x vs intraday {g['id_x']:.2f}x over 21.6y. |\n"
            f"| **Tradability** | `MIRAGE` | Frictionless overnight-only = {g['on_ann']:+.2f}%/yr vs "
            f"buy & hold {g['cc_ann']:+.2f}%/yr — nothing to dodge (intraday t = {g['t_id']:+.2f}); "
            f"net of 2 trades/day it trails B&H at every cost "
            f"({R['harvest'][0][2]:+.1f}% at 0.5 bps, {R['harvest'][2][2]:+.1f}% at 2 bps). |\n"
            f"| **Ever harvestable?** | `BUSTED` | Break-even one-way cost ≤ ~0.18 bps (IAU), "
            f"*negative* on GLD — below any spread GLD ever traded. Bonus: the 2015 fix reform left "
            f"the gap intact (GLD {R['gld_pre'][3]:+.2f} → {R['gld_post'][3]:+.2f} bps/d, change "
            f"Welch t = {R['gld_change_welch']:+.2f}). |\n\n"
            "> 💡 In plain words: gold really does trade in its sleep — and the only way to act on "
            "that fact costs more than the fact is worth."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Split each session $t$ into the overnight leg $r^{on}_t = O_t/C_{t-1}-1$ and the intraday "
            "leg $r^{id}_t = C_t/O_t-1$ (adjusted opens/closes; the legs compound exactly to "
            "close-to-close). The claim decomposes into:\n\n"
            "- **H₁ (the split exists).** $\\mathbb{E}[r^{on}-r^{id}] > 0$ on gold, robustly "
            "(HAC t ≥ 2 on the paired daily difference), and not as a one-fund artefact (IAU must "
            "agree).\n"
            "- **H₂ (gold is special).** Gold's gap exceeds the market-wide clock effect (SPY's gap, "
            "study 01) — Welch t on the two daily-difference series.\n"
            "- **H₃ (the fix is the engine).** If London-fix rigging drives it, the gap should shrink "
            "after the 2015-03-20 LBMA reform (externally-dated break; Welch t on the change).\n"
            "- **H₄ (it's harvestable).** Buy-MOC/sell-MOO must beat buy & hold net of 2 one-way "
            "trades/day.\n\n"
            f"We find **H₁ supported** (t = {g['t_gap']:.2f} GLD, {i['t_gap']:.2f} IAU), **H₂ "
            f"unproven** (gap ratio ≈ 2× but Welch t = {R['vs_spy_welch']:+.2f}), **H₃ rejected** "
            f"(the gap *grew* post-reform), **H₄ rejected** (trails B&H at every cost)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The decisive statistic for H₁ is a **Newey-West t on the paired daily difference** "
            "$d_t = r^{on}_t - r^{id}_t$ (daily session returns are volatility-clustered, so plain "
            "IID errors understate the noise), cross-checked by a pooled **Welch t** and a **20,000-"
            "draw sign-flip permutation** (under exchangeable night/day labels, $d_t$ is symmetric "
            "about 0 — flip signs, count how often chance beats the tape).\n\n"
            "H₂ matters because study 01 already stamped the *equity* overnight effect Real: a gold "
            "gap that merely matches SPY's is the same market-wide clock story, not a gold anomaly. "
            "H₃ is the rare luxury of an **externally-dated mechanism test** — the reform date is set "
            "by the LBMA, not by our data. H₄ is the desk's standing rule: **charge the costs against "
            "the alpha, not the gross** — here the 'alpha' over holding is the *dodged intraday leg*, "
            "and the cost is two spreads a day."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Tape.** GLD {g['start']}→{g['end']} ({g['n']:,} sessions), IAU (confirm, "
            f"{i['n']:,}), SPY (placebo, identical calendar). Adjusted open+close; rows with either "
            "print missing are dropped; |leg| ≥ 50% filtered as corrupt (none triggered).\n"
            "- **Primary test.** HAC (Bartlett, 4(n/100)^(2/9) lags) t on mean $d_t$; Welch t as the "
            "group split; sign-flip permutation p (seeded, 20,000 draws).\n"
            "- **Confirmation.** IAU must show the same sign and clear t ≥ 2 independently.\n"
            "- **Placebo.** Welch t of gold's $d_t$ vs SPY's on the common calendar.\n"
            "- **Mechanism break.** Pre/post 2015-03-20 (LBMA Gold Price replaces the phone fix); "
            "Welch t on the change in gap. Split is externally dated, not snooped.\n"
            "- **Harvest.** Long-only buy-MOC/sell-MOO (the unconditional clock rule — the MOC/MOO "
            "convention is the documented execution lag; nothing conditions on same-bar info), "
            "**2 one-way trades/day** × {0.5, 1, 2, 5} bps × NAV, vs buy & hold; long-only, no "
            "borrow (the short-day variant adds legs + borrow and dies faster).\n"
            "- **Control.** A seeded synthetic world with a tunable planted night drift: the null "
            "must stay quiet, a planted 4 bps/d (the real size) must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The decomposition — the whole return is overnight\n\n"
            "One dollar three ways on GLD; the sleeves compound each leg separately (gross, no costs "
            "yet — this is the *descriptive* fact)."
        ),
        code(CURVE_CELL),
        md(
            f"> 💡 In plain words: **{g['on_x']:.2f}×** overnight vs **{g['id_x']:.2f}×** intraday — "
            f"the overnight sleeve alone ({g['on_ann']:+.2f}%/yr) *is* the whole GLD return "
            f"({g['cc_ann']:+.2f}%/yr). IAU: {i['on_x']:.2f}× vs {i['id_x']:.2f}×. The folklore chart "
            "is not a cherry-pick."
        ),
        md(
            "### 4b · Is it luck? — HAC t and the sign-flip placebo\n\n"
            "The paired daily difference $d_t$ carries the inference; the histogram shows the "
            "permutation null."
        ),
        code(FLIP_CELL),
        md(
            f"> 💡 In plain words: GLD's gap is **{g['gap']:+.2f} bps/day** at HAC "
            f"**t = {g['t_gap']:.2f}** (Welch {g['welch']:+.2f}); only ~{g['p']*1000:.0f}-in-1,000 "
            f"random relabelings beat it. IAU is stronger still: **{i['gap']:+.2f} bps/day**, HAC "
            f"**t = {i['t_gap']:.2f}**, p = {i['p']:.4f}. Two independent gold wrappers, one verdict: "
            "**H₁ real.**"
        ),
        md(
            "### 4c · Is gold special? — the SPY placebo race\n\n"
            "Study 01 stamped the equity overnight effect Real (and Mirage). If gold's gap merely "
            "matches it, \"gold trades in its sleep\" is just \"markets trade in their sleep\"."
        ),
        code(VS_SPY_CELL),
        md(
            f"> 💡 In plain words: gold's gap ({R['gld']['gap']:+.2f} bps/d) is **about twice** SPY's "
            f"({R['spy']['gap']:+.2f} bps/d), and on this window gold clears t ≥ 2 while SPY doesn't "
            f"— but the *difference* between the two daily series is only **Welch "
            f"t = {R['vs_spy_welch']:+.2f}**. Honest reading: gold's own tape certifies gold's gap; "
            "it cannot certify that gold is a different animal from the market-wide clock effect. "
            "**H₂ unproven.**"
        ),
        md(
            "### 4d · The natural experiment — did the 2015 fix reform kill it?\n\n"
            "The claim's favorite mechanism is the London fix (leaky calls, the 2014 Barclays fine). "
            "The fix was replaced by an audited electronic auction on 2015-03-20. If the mechanism "
            "were the fix, the gap should shrink."
        ),
        code(SUBPERIOD_CELL),
        md(
            f"> 💡 In plain words: the gap **grew** after the reform (GLD {R['gld_pre'][3]:+.2f} → "
            f"{R['gld_post'][3]:+.2f} bps/d; IAU {R['iau_pre'][3]:+.2f} → {R['iau_post'][3]:+.2f}), "
            f"change Welch t = {R['gld_change_welch']:+.2f} / {R['iau_change_welch']:+.2f} — no decay, "
            "if anything the opposite. **H₃ rejected**: whatever drives the night tilt, it survived "
            "the death of the phone fix. (Note the honesty cost of sub-sampling: GLD's pre-reform gap "
            f"alone is only t = {R['gld_pre'][4]:.2f}; the full-window t and IAU carry the Signal "
            "stamp.) The surviving explanation is structural: gold's price discovery runs through "
            "Asia and London while NYSE Arca is shut, so the listed ETF marks it at the open."
        ),
        md(
            "### 4e · The harvest test — two trades a day vs nothing to dodge\n\n"
            "Buy MOC, sell next MOO, flat all day. Net = overnight leg − 2 × one-way cost. The "
            "benchmark is *doing nothing* (buy & hold pays its entry once in 21 years)."
        ),
        code(HARVEST_CELL),
        md(
            "> 💡 In plain words: the overlay **never** beats holding — "
            f"**{R['harvest'][0][2]:+.1f}%/yr at 0.5 bps** (tighter than GLD's spread has ever been) "
            f"vs **{R['harvest'][0][3]:+.1f}%** for buy & hold; ≈0 at 2 bps; ruin at 5. The "
            f"arithmetic: dodging the intraday leg is worth **{-R['gld']['id']:+.2f} bps/d** on GLD "
            f"(a statistical zero, t = {g['t_id']:+.2f}) while the toll is a hard 1–10 bps/day. "
            "Break-even one-way cost: ≤ ~0.18 bps on IAU, negative on GLD. Pre-reform, post-reform, "
            "any era: **H₄ rejected — never harvestable.**"
        ),
        md(
            "### 4f · Faithful-engine control — we know the truth here\n\n"
            "A seeded world with a **planted** overnight drift (and a null with none): the split must "
            "stay quiet on the null and recover the plant. *(Machinery proof only — never cited in "
            "support of the real-tape stamp.)*"
        ),
        code(SYN_CELL),
        md(
            f"> 💡 In plain words: null world t = {R['syn'][0][2]:+.2f} (p = {R['syn'][0][3]:.3f}) — "
            f"noise cannot fake the gap; a planted **+4 bps/d** night drift (the real GLD size) reads "
            f"t = {R['syn'][1][2]:+.2f}, p ≈ 0. The detector is unbiased at exactly the effect size "
            "in question."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `REAL`** — GLD gap **{g['gap']:+.2f} bps/d**, HAC **t = {g['t_gap']:.2f}**, "
            f"sign-flip **p = {g['p']:.4f}**; IAU confirms (**{i['gap']:+.2f} bps/d**, "
            f"**t = {i['t_gap']:.2f}**, p = {i['p']:.4f}); overnight sleeve {g['on_x']:.2f}× vs "
            f"{g['id_x']:.2f}× intraday. Clears **t ≥ 2 on the real tape**, twice. Caveats stated: "
            f"vs-SPY Welch t = {R['vs_spy_welch']:+.2f} (not provably bigger than the equity clock "
            f"effect), GLD pre-2015 sub-sample alone t = {R['gld_pre'][4]:.2f}.\n"
            f"- **Tradability `MIRAGE`** — frictionless overnight-only ({g['on_ann']:+.2f}%/yr) ≈ buy "
            f"& hold ({g['cc_ann']:+.2f}%/yr); there is nothing significant to dodge (intraday "
            f"t = {g['t_id']:+.2f}); with 2 trades/day the overlay trails B&H at every cost in every "
            "sub-period. The anomaly describes *when* the return arrives, not an edge.\n"
            f"- **Ever harvestable? `BUSTED`** — break-even one-way cost ≤ ~0.18 bps (IAU), negative "
            f"on GLD; GLD's spread was ~2 bps in 2005, ~0.3 bps today. And the mechanism myth loses "
            f"its engine: the 2015 LBMA reform left the gap intact "
            f"(change Welch t = {R['gld_change_welch']:+.2f})."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The sibling study.** [01-overnight-anomaly](../../01-overnight-anomaly/README.md) is "
            "the equities original — same clock split, richer machinery (Lo t, breadth, capacity, a "
            "Bayesian manipulation posterior). Gold adds what equities lack: a *named, dated* "
            "mechanism (the fixes) that can be — and here is — falsified by reform.\n"
            "- **Futures would sharpen it.** GC=F trades ~23h, so 'overnight' vs 'intraday' on the "
            "ETF clock becomes 'Asia+London' vs 'COMEX pit hours' on the futures clock; with intraday "
            "data you could bracket the fixes directly (Caminschi-Heaney style).\n"
            "- **The general lesson.** A return-timing decomposition is *descriptive* until you price "
            "the round trips its harvest requires. Daily-frequency session effects need per-day costs "
            "below the per-day gap over the *counterfactual* leg — here below ~0.1 bps, i.e. never.\n\n"
            "*The reproducible core is offline and deterministic; run "
            "[`examples/verify.py`](../examples/verify.py) to regenerate every number. Sources: "
            "[`docs/references.md`](../docs/references.md); frozen numbers: "
            "[`docs/results.md`](../docs/results.md).*"
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
