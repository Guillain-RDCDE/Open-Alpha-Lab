"""Generate the two narrative notebooks for Study 831 (Gold Real-Yield Timing).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the desk beats. The synthetic control runs anywhere, offline and
deterministic; the real-tape cells use the cached tape under ../_cache/ if present and
otherwise quote the frozen headline numbers in ``R`` (mirroring docs/results.md), so the
notebook re-runs for any reader offline.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-30; tape
# 2004-11-19 -> 2026-06-29; 5,434 days; tape fp 4f27dc5f4b4f, ry-series fp ef386ee60d51).
R = dict(
    fp_tape="4f27dc5f4b4f", fp_ry="ef386ee60d51",
    n=5434, start="2004-11-19", end="2026-06-29",
    # contemporaneous inverse link (descriptive)
    link_corr=-0.26, link_beta=-0.77, link_t=-9.67,
    # headline 21-day quintile sort (percent)
    q1=1.16, q5=0.93, spread=-0.23, hac_t=-0.36, placebo_p=0.734, n_q=1033,
    # horizon sweep: (label, spread%, HAC t)
    horiz=[("5d", -0.10, -0.44), ("21d", -0.23, -0.36), ("63d", 1.24, 0.67), ("126d", 3.32, 1.29)],
    # lookback sweep: (label, spread%, HAC t)
    look=[("21d", 0.23, 0.31), ("63d", -0.23, -0.36), ("126d", 0.44, 0.52), ("252d", 0.44, 0.55)],
    # sub-period sweep: (label, spread%, HAC t)
    sub=[("2004-2009", -0.93, -0.41), ("2010-2015", 0.11, 0.09),
         ("2016-2020", 0.23, 0.17), ("2021-2026", 0.32, 0.16)],
    # timing overlay
    timer_sharpe=0.565, bh_sharpe=0.564, timer_sharpe5=0.523,
    switches_yr=16.7, inv_frac=0.479, ov_spread_bps=-1.40, ov_spread_t=-1.24, cost_bps=2.0,
    # gold buy-and-hold context
    gld_ann=9.8, gld_vol=18.2, gld_sharpe=0.537,
    # synthetic control (25 seeds, 21d): (edge, mean spread%, mean HAC t)
    ctrl=[(0.000, 0.03, 0.00), (0.005, 6.80, 6.56), (0.010, 13.59, 11.36),
          (0.020, 27.27, 16.36), (0.040, 55.43, 18.00)],
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

from gold_real_yield import data, strategy as st

def load_real():
    \"\"\"Cache-first real daily tape (empty frame offline).\"\"\"
    try:
        return data.load_series()
    except FileNotFoundError:
        return pd.DataFrame()

DF = load_real()
HAVE_REAL = not DF.empty
print("real gold/real-yield tape present:", HAVE_REAL,
      "" if not HAVE_REAL else f"({len(DF):,} days, {DF.index[0].date()} -> {DF.index[-1].date()})")
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Gold & Real Yields — does 'gold hates rising real rates' tell you *when* to own it? 🥇\n"
            "### Separating a true same-day fact from a false forecast, in plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Inverse_link: Confirmed_but_untradable](https://img.shields.io/badge/Inverse_link-Confirmed_but_untradable-8b949e?style=flat-square)\n\n"
            "Gold pays no interest. So the story goes: when the *real* interest rate on safe bonds (the "
            "yield after subtracting expected inflation) **falls**, the opportunity cost of parking money "
            "in shiny metal drops, and gold rises; when real rates **climb**, gold should sink. It is one "
            "of the most-repeated lines in macro.\n\n"
            "There are actually **two** claims hiding in that sentence, and they are worlds apart:\n\n"
            "1. **Same-day:** *do gold and real yields move in opposite directions on the same day?*\n"
            "2. **Forecast:** *if real yields have been falling lately, will gold rise from here?*\n\n"
            "The first can be true while the second is worthless — and that is exactly what we find.\n\n"
            "> 📓 **This is the plain-language layer.** Want the *t*-stats, the placebo test and the cost "
            "maths? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool: every chart is drawn by the "
            "code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ----
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Do gold and real yields move oppositely *on the same day*? | **Yes — strongly.** Their "
            f"daily correlation is **{R['link_corr']}** (rock-solid, *t* {R['link_t']}). The famous "
            "inverse fact is **real**. |\n"
            f"| Does a *falling* real-yield trend predict *higher* gold next month? | **No.** The "
            f"fastest-falling-yield days went on to earn **+{R['q5']}%** vs **+{R['q1']}%** for the "
            f"fastest-*rising* — a **{R['spread']}%** gap, i.e. faintly the **wrong** way, and "
            "statistically nothing. |\n"
            "| Could you trade it? | **No.** A timer that owns gold only when real yields are falling "
            f"just *ties* buy-and-hold — and only by hiding in cash half the time — while flipping in and "
            f"out ~{R['switches_yr']:.0f} times a year. |\n\n"
            "> The inverse link is a **same-day mirror**, not a **crystal ball**. Gold *co-moves* with "
            "real yields; it is not *timed* by their trend."
        ),

        # ---- BEAT 1 — THE CLAIM ----
        md(
            "## 1 · The claim\n\n"
            "> *\"Gold has no yield, so it is priced off the real interest rate: real rates down → gold "
            "up, real rates up → gold down.\"* — the practitioner staple, rooted in Barsky & Summers "
            "(1988) and sharpened by Erb & Harvey (2013).\n\n"
            "The intuition is clean. Holding gold means giving up the real return you *could* have earned "
            "on an inflation-protected bond. When that real return shrinks, gold looks better; when it "
            "grows, gold looks worse. The 2013 'taper tantrum' — real yields spiked, gold crashed — is "
            "the poster child."
        ),

        # ---- BEAT 2 — SO WHAT ----
        md(
            "## 2 · So what?\n\n"
            "If the real-yield *trend* actually timed gold, you'd have a clean macro switch: own gold "
            "when real rates are falling, step aside when they're rising. That's the difference between "
            "a *fact* (gold reacts to today's rate move) and an *edge* (you can act on yesterday's trend "
            "and profit). The desk has looked at gold's session split ([640](../../640-gold-overnight/)) "
            "and its calendar ([649](../../649-gold-seasonality/)); here we test the biggest "
            "*fundamental* story of all."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ----
        md(
            "## 3 · How would we even know?\n\n"
            "1. **Measure the real yield.** The official 10-year real yield isn't a free ticker, so we "
            "use a clean stand-in: the **TIPS ETF (TIP)**. Inflation-protected bonds rise exactly when "
            "real yields fall, so TIP's recent return *is* a real-yield-fall meter.\n"
            "2. **The same-day check.** Line up each day's gold move against that day's real-yield move. "
            "A strong negative match confirms the fact.\n"
            "3. **The forecast check.** Rank each day by how much real yields have *fallen* over the last "
            "quarter, split days into five buckets, and see how gold did over the *next month*. The "
            "claim says the biggest-fall bucket wins.\n\n"
            "*Timing:* the trend at today's close forms the signal; we act at *tomorrow's* close (a "
            "one-day lag) — no peeking."
        ),

        # ---- BEAT 4 — THE TEARDOWN ----
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the same-day fact.** Does gold really move opposite to real yields?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    il = st.inverse_link(DF)\n"
            "    corr, t = il['corr'], il['t']\n"
            "    g = np.log(DF['GLD_close']).diff(); dry = -np.log(DF['TIP_close']).diff()\n"
            "    d = pd.DataFrame({'g':g*100,'dry':dry*100}).dropna()\n"
            "    banner = 'REAL tape (GLD vs TIP-implied real yield, 2004-2026)'\n"
            "else:\n"
            f"    corr, t = {R['link_corr']}, {R['link_t']}\n"
            "    rng = np.random.default_rng(0); dry_ = rng.normal(0,0.05,4000)\n"
            "    d = pd.DataFrame({'dry':dry_*100, 'g':(-0.8*dry_+rng.normal(0,0.09,4000))*100})\n"
            "    banner = 'illustrative (offline)'\n"
            "fig, ax = plt.subplots(figsize=(7.2, 5.0))\n"
            "ax.scatter(d['dry'], d['g'], s=5, alpha=.25, color=GREY)\n"
            "b = np.polyfit(d['dry'], d['g'], 1)\n"
            "xs = np.linspace(d['dry'].quantile(.01), d['dry'].quantile(.99), 50)\n"
            "ax.plot(xs, np.polyval(b, xs), color=RED, lw=2)\n"
            "ax.set_xlabel('same-day real-yield change (proxy, %)'); ax.set_ylabel('same-day gold return (%)')\n"
            "ax.set_title(f'Gold vs real yields, same day: corr {corr:+.2f} (t {t:+.1f}) — inverse & strong')\n"
            "fig.suptitle(banner, fontsize=9, color=GREY); plt.tight_layout(); plt.show()\n"
            "print(f'same-day correlation {corr:+.2f}  (HAC t {t:+.1f}) -> the inverse link is REAL')"
        ),
        md(
            f"There's the famous fact, confirmed: a same-day correlation of **{R['link_corr']}** with a "
            f"*t*-stat of **{R['link_t']}** (anything past ±2 is significant; this is off the charts). "
            "Gold and real yields genuinely mirror each other. **But** — this is a *same-day* mirror. To "
            "profit you'd need to know today's yield move before it happens. So let's ask the question "
            "that could actually pay."
        ),
        md(
            "**Now the forecast.** If real yields have been *falling*, does gold rise over the next "
            "month? Split days into five buckets by the real-yield-fall trend and compare forward gold."
        ),
        code(
            "if HAVE_REAL:\n"
            "    qs = st.quintile_spread(DF, horizon=21, lookback=63)\n"
            "    q1, q5, spr, t = qs['q1']*100, qs['q5']*100, qs['spread']*100, qs['t']\n"
            "else:\n"
            f"    q1, q5, spr, t = {R['q1']}, {R['q5']}, {R['spread']}, {R['hac_t']}\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.4))\n"
            "cols = [GREY, RED if spr < 0 else GREEN]\n"
            "ax.bar(['yields RISING\\n(Q1)','yields FALLING\\n(Q5)'], [q1, q5], color=cols, width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('forward 21-day gold return %')\n"
            "ax.set_title(f'The claim says Q5 should WIN — it doesn\\'t (spread {spr:+.2f}%, t {t:+.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'yields rising -> +{q1:.2f}% | yields falling -> +{q5:.2f}%  ->  spread {spr:+.2f}% (HAC t {t:+.2f})')"
        ),
        md(
            f"The forecast **fails**. Days after real yields fell fastest earned **+{R['q5']}%** over the "
            f"next month — *less* than the **+{R['q1']}%** after yields *rose* fastest. The gap is "
            f"**{R['spread']}%**, pointing the wrong way, with a *t*-stat of **{R['hac_t']}** (you want "
            f"≥ 2) and a shuffle test that says a gap this size is pure chance **{int(R['placebo_p']*100)}%** "
            "of the time. The trend of real yields tells you nothing useful about *tomorrow's* gold."
        ),

        # ---- BEAT 5 — VERDICT ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** The real-yield *trend* doesn't forecast gold (spread {R['spread']}%, "
            f"wrong sign, *t* {R['hac_t']}, shuffle *p* {R['placebo_p']}).\n"
            "- **Tradability — Mirage.** A timer on the signal ties buy-and-hold only by cash-drag and "
            "loses on average return (next notebook).\n"
            "- **Inverse link — Confirmed but untradable.** The same-day correlation is real and strong "
            f"({R['link_corr']}, *t* {R['link_t']}) — but same-day is not a forecast."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ----
        md(
            "## 6 · Could you actually trade it?\n\n"
            "No. A timer that owns gold only when real yields are falling sits in cash about half the "
            "time — so it *looks* a touch smoother (a near-identical Sharpe) but doesn't actually earn "
            "more; on the honest measure (average return) it **loses** to plain buy-and-hold, after "
            f"flipping in and out ~{R['switches_yr']:.0f} times a year. The next notebook does the cost "
            "maths in full."
        ),

        # ---- BEAT 7 — GOING FURTHER ----
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Same asset, other angles.** [640 Gold-Overnight](../../640-gold-overnight/) (session "
            "split), [649 Gold-Seasonality](../../649-gold-seasonality/) (calendar), [580 "
            "Gold-Lease-Rate](../../580-gold-lease-rate/) (carry).\n"
            "- **The breakeven cousin.** [381 TIPS-Breakeven](../../381-tips-breakeven/) trades the "
            "inflation-compensation spread itself; here TIP/IEF only *build the real-yield gauge* for "
            "timing gold.\n"
            "- **The real DFII10 tape.** Re-run with the official 10-year TIPS real yield (a FRED CSV, "
            "not a free ticker) — but note the failure here isn't the proxy: the proxy *does* deliver "
            "the strong same-day link; it's the *forecast* that's missing.\n\n"
            "*Think the trend really times gold and we used too crude a gauge? Fork this, drop in "
            "DFII10, and show the falling-yield bucket beating the rising-yield bucket at t = 2 out of "
            "sample.*"
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
            "# Gold Real-Yield Timing — a quantitative teardown 🔬\n"
            "### TIP real-yield gauge · contemporaneous inverse-link HAC *t* · quintile sort · block-shuffle placebo · horizon/lookback/sub-period sweeps · timing-overlay costs · synthetic control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Inverse_link: Confirmed_but_untradable](https://img.shields.io/badge/Inverse_link-Confirmed_but_untradable-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "beats, every claim now carrying its standard error.* We build a TIP-gauge real-yield proxy "
            "(`ryfall = log(TIP_t) − log(TIP_{t−63})`), confirm the **contemporaneous** inverse link, "
            "then test whether the real-yield *trend* forecasts forward gold.\n\n"
            "> ⚠️ **Not investment advice.** Real data: yfinance daily GLD + TIP + IEF + `^TNX`, "
            f"{R['n']:,} days (2004-11-19 → 2026-06-29, tape fp `{R['fp_tape']}`); the offline core and "
            "tests run on a deterministic synthetic world. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Q5−Q1 forward-gold spread **{R['spread']}%** (21d), HAC *t* "
            f"**{R['hac_t']}**, placebo *p* {R['placebo_p']} — wrong-signed at the headline; right-signed "
            "only at 63-126d and still sub-2 (peak *t* +1.29); tiny/unstable across sub-periods. |\n"
            f"| **Tradability** | `MIRAGE` | Timer Sharpe **{R['timer_sharpe']}** vs buy-and-hold "
            f"**{R['bh_sharpe']}** — a cash-drag tie; mean-return spread **{R['ov_spread_bps']} bps/day** "
            f"(*t* {R['ov_spread_t']}) at {R['switches_yr']} switches/yr; below buy-and-hold at 5 bps. |\n"
            f"| **Inverse link** | `CONFIRMED (untradable)` | Contemporaneous corr **{R['link_corr']}**, "
            f"beta **{R['link_beta']}**, HAC *t* **{R['link_t']}** — real & strong, but same-day. |\n\n"
            "> 💡 In plain words: the engine works (the synthetic control proves it) and the inverse link "
            "is real — but on this tape the real-yield *trend* has no *forward* edge on gold."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $\\Delta y^r_t$ be the same-day real-yield change and $r_{t\\to t+h}$ the forward gold "
            "return; let $\\text{ryfall}_t$ be the trailing real-yield fall.\n\n"
            "- **H₀ (the link).** $\\text{corr}(r_t, \\Delta y^r_t) < 0$ — the *contemporaneous* inverse "
            "fact.\n"
            "- **H₁ (the forecast).** $\\mathbb{E}[r \\mid \\text{ryfall high (Q5)}] - "
            "\\mathbb{E}[r \\mid \\text{ryfall low (Q1)}] > 0$ at HAC *t* ≥ 2.\n"
            "- **H₂ (robust).** The Q5−Q1 sign is stable across horizons, lookbacks and sub-periods.\n"
            "- **H₃ (tradable).** A real-yield-conditioned timer beats buy-and-hold gold net of costs, "
            "on mean return.\n\n"
            "We find **H₀ strongly confirmed**, **H₁ rejected** (spread wrong-signed, HAC *t* "
            f"{R['hac_t']}), **H₂ rejected** (sign wanders, never significant), and **H₃ rejected** (the "
            "timer's Sharpe tie is a cash-drag artefact; mean-return spread negative)."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The whole content is the gap between **co-movement** and **prediction**. A long literature "
            "(Barsky-Summers 1988; Erb-Harvey 2013; Baur-McDermott 2010) documents gold's inverse "
            "*contemporaneous* sensitivity to real rates. The tradeable question is different: does the "
            "*lagged trend* forecast forward gold? If yes, you have a macro timing switch; if no, the "
            "inverse fact is a same-day statistical mirror with no actionable edge — the desk's recurring "
            "lesson."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Gauge.** $\\text{ryfall} = \\log \\text{TIP}_t - \\log \\text{TIP}_{t-63}$ (TIP rises ⇔ "
            "real yields fall), ranked out-of-sample over a trailing 252-day window into "
            "$\\text{rank}\\in[0,1]$, lagged one day. Cross-checked against $\\text{TNX} - "
            "\\text{breakeven}$ from TIP-vs-IEF.\n"
            "- **Link.** HAC *t* on the same-day OLS beta of gold on the real-yield change (descriptive).\n"
            "- **Sort.** Quintiles of the lagged rank; Q5 (fastest fall) vs Q1 (fastest rise) forward "
            "gold.\n"
            "- **Inference.** HAC (Newey-West) *t* on the day-level Q5−Q1 difference; a **block-shuffle "
            "placebo** (circular rotation, 21-day blocks).\n"
            "- **Robustness.** Horizons 5/21/63/126d; lookbacks 21/63/126/252d; four sub-periods.\n"
            "- **Frictions.** A timer (own GLD when rank > 0.5, else cash), one-way cost per switch × "
            "NAV; net Sharpe and mean-return spread vs buy-and-hold.\n"
            "- **Positive control.** A deterministic synthetic world with a planted *predictive* edge "
            "(`edge > 0`) *while keeping the contemporaneous link on* — averaged over 25 seeds.\n\n"
            "Timing: the trend at close *t* forms the signal; the position enters at close *t+1*. That "
            "one-bar lag is the single documented execution convention — the rolling window and rank are "
            "causal, so no future data enters the signal."
        ),

        # ---- BEAT 4 ----
        md("## 4 · The teardown"),
        md(
            "### 4a · The contemporaneous inverse link — H₀\n\n"
            "The same-day beta of gold on the (proxied) real-yield change, with a HAC *t*."
        ),
        code(
            "if HAVE_REAL:\n"
            "    il = st.inverse_link(DF)\n"
            "    corr, beta, t = il['corr'], il['beta'], il['t']\n"
            "else:\n"
            f"    corr, beta, t = {R['link_corr']}, {R['link_beta']}, {R['link_t']}\n"
            "print(f'corr {corr:+.3f} | OLS beta {beta:+.3f} | Newey-West t {t:+.2f} -> inverse link CONFIRMED')\n"
            "fig, ax = plt.subplots(figsize=(6.5, 3.0))\n"
            "ax.barh(['HAC t on beta'], [t], color=GREY)\n"
            "ax.axvline(-2, ls='--', c=RED, lw=1); ax.axvline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_title(f'Same-day gold~real-yield: t {t:+.1f} (|t|>2 = significant)')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: H₀ **confirmed** — corr {R['link_corr']}, beta {R['link_beta']}, HAC "
            f"*t* {R['link_t']}. Gold and real yields genuinely move oppositely *the same day*. This is "
            "the fact everyone quotes; the rest of the notebook shows it doesn't survive being turned "
            "into a *forecast*."
        ),
        md(
            "### 4b · The forecast sort — H₁\n\n"
            "Q5 (real yields falling fastest) vs Q1 (rising fastest) forward gold, 21-day, with the HAC "
            "*t* and the block-shuffle placebo."
        ),
        code(
            "if HAVE_REAL:\n"
            "    qs = st.quintile_spread(DF, horizon=21, lookback=63)\n"
            "    p = st.placebo_pvalue(DF, horizon=21, lookback=63, n_perm=1000, seed=831)\n"
            "    q1, q5, spr, t = qs['q1']*100, qs['q5']*100, qs['spread']*100, qs['t']\n"
            "else:\n"
            f"    q1, q5, spr, t, p = {R['q1']}, {R['q5']}, {R['spread']}, {R['hac_t']}, {R['placebo_p']}\n"
            "fig, ax = plt.subplots(figsize=(8, 4.3))\n"
            "ax.bar(['Q1 (yields rising)','Q5 (yields falling)'], [q1, q5], color=[GREY, RED if spr<0 else GREEN], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('forward 21-day gold return %')\n"
            "ax.set_title(f'Claim predicts Q5 > Q1 — data give Q5 - Q1 = {spr:+.2f}% (HAC t {t:+.2f}, placebo p {p:.3f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Q1 +{q1:.2f}% | Q5 +{q5:.2f}% | spread {spr:+.2f}% | HAC t {t:+.2f} | placebo p {p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: H₁ **rejected.** The spread is **{R['spread']}%** — the *wrong* sign — "
            f"at HAC *t* {R['hac_t']}, and the placebo *p* = {R['placebo_p']} says a |spread| this size is "
            "well inside the null. The famous inverse link does **not** carry into a forward edge."
        ),
        md(
            "### 4c · Horizon & lookback sweeps — H₂ (robustness)\n\n"
            "Does any hold-horizon or trend-window rescue the signal?"
        ),
        code(
            "horiz = " + repr(R["horiz"]) + "\n"
            "look = " + repr(R["look"]) + "\n"
            "if HAVE_REAL:\n"
            "    th = st.horizon_sweep(DF); tl = st.lookback_sweep(DF)\n"
            "    hl = list(th.index); hs = list(th['spread']*100); ht = list(th['t'])\n"
            "    ll = list(tl.index); ls_ = list(tl['spread']*100); lt = list(tl['t'])\n"
            "else:\n"
            "    hl=[x[0] for x in horiz]; hs=[x[1] for x in horiz]; ht=[x[2] for x in horiz]\n"
            "    ll=[x[0] for x in look]; ls_=[x[1] for x in look]; lt=[x[2] for x in look]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))\n"
            "a1.bar(hl, hs, color=[GREEN if s>0 else RED for s in hs], width=.55)\n"
            "for x,(s,tt) in enumerate(zip(hs,ht)): a1.annotate(f't={tt:+.2f}',(x,s),ha='center',va='bottom' if s>=0 else 'top',fontsize=8,color=GREY)\n"
            "a1.axhline(0,c='k',lw=1); a1.set_ylabel('Q5 - Q1 spread %'); a1.set_title('Horizon sweep')\n"
            "a2.bar(ll, ls_, color=[GREEN if s>0 else RED for s in ls_], width=.55)\n"
            "for x,(s,tt) in enumerate(zip(ls_,lt)): a2.annotate(f't={tt:+.2f}',(x,s),ha='center',va='bottom' if s>=0 else 'top',fontsize=8,color=GREY)\n"
            "a2.axhline(0,c='k',lw=1); a2.set_title('Lookback sweep')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('horizon t:', [round(x,2) for x in ht], '| lookback t:', [round(x,2) for x in lt])"
        ),
        md(
            "> 💡 In plain words: H₂ **rejected.** The spread is right-signed only at 63-126d — where "
            "overlapping windows inflate the raw size — and the HAC correction keeps *t* under 2 "
            "everywhere (peak +1.29). No lookback (month → year) lifts |*t*| above 0.55. There is no "
            "window where this works."
        ),
        md(
            "### 4d · Sub-period sweep — H₂ across regimes\n\n"
            "Four eras, including the 2021-2026 inflation scare where a real-yield rule should shine."
        ),
        code(
            "sub = " + repr(R["sub"]) + "\n"
            "if HAVE_REAL:\n"
            "    edges = [('2004-2009','2004-11-01','2009-12-31'),('2010-2015','2010-01-01','2015-12-31'),\n"
            "             ('2016-2020','2016-01-01','2020-12-31'),('2021-2026','2021-01-01','2026-06-30')]\n"
            "    tab = st.subperiod_sweep(DF, edges, horizon=21)\n"
            "    labs = list(tab.index); spr = list(tab['spread']*100); ts = list(tab['t'])\n"
            "else:\n"
            "    labs=[s[0] for s in sub]; spr=[s[1] for s in sub]; ts=[s[2] for s in sub]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.bar(range(len(labs)), spr, color=[GREEN if s>0 else RED for s in spr], width=.55)\n"
            "ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, rotation=10)\n"
            "for x,(s,tt) in enumerate(zip(spr,ts)): ax.annotate(f't={tt:+.2f}',(x,s),ha='center',va='bottom' if s>=0 else 'top',fontsize=9,color=GREY)\n"
            "ax.axhline(0,c='k',lw=1); ax.set_ylabel('Q5 - Q1 spread %')\n"
            "ax.set_title('Tiny and unstable — no window clears t = 2')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('sub-period spreads %:', [round(s,2) for s in spr], '| HAC t:', [round(x,2) for x in ts])"
        ),
        md(
            "> 💡 In plain words: even the inflation-scare 2021-2026 window — the one era a real-yield "
            f"rule *should* nail — musters only +{R['sub'][3][1]}% at *t* +{R['sub'][3][2]}. Nowhere does "
            "the trend forecast gold."
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — H₁ rejected (spread {R['spread']}%, wrong sign, HAC *t* {R['hac_t']}, "
            f"placebo *p* {R['placebo_p']}); H₂ rejected (unstable, never significant).\n"
            f"- **Tradability `MIRAGE`** — timer Sharpe {R['timer_sharpe']} vs buy-and-hold "
            f"{R['bh_sharpe']} is a cash-drag tie; mean-return spread {R['ov_spread_bps']} bps/day.\n"
            f"- **Inverse link `CONFIRMED (untradable)`** — H₀ holds hard (corr {R['link_corr']}, HAC *t* "
            f"{R['link_t']}) but it is same-day, not a forecast."
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you trade it? — the timing-overlay costs\n\n"
            "Own GLD when the real-yield-fall rank > 0.5, else cash; one-way cost per switch. Compare "
            "Sharpe **and** mean return to buy-and-hold gold."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ov = st.timing_overlay(DF, cost_bps=2.0); ov5 = st.timing_overlay(DF, cost_bps=5.0)\n"
            "    ts, bs, sw = ov['timer_sharpe'], ov['bh_sharpe'], ov['switches_per_yr']\n"
            "    sp, spt, ts5 = ov['spread_bps_day'], ov['spread_t'], ov5['timer_sharpe']\n"
            "else:\n"
            f"    ts, bs, sw = {R['timer_sharpe']}, {R['bh_sharpe']}, {R['switches_yr']}\n"
            f"    sp, spt, ts5 = {R['ov_spread_bps']}, {R['ov_spread_t']}, {R['timer_sharpe5']}\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))\n"
            "a1.bar(['timer\\n(2bps)','timer\\n(5bps)','buy & hold'], [ts, ts5, bs], color=[AMBER, RED, GREY], width=.6)\n"
            "a1.set_ylabel('annualised Sharpe'); a1.set_title(f'Sharpe tie then loss ({ts:.3f}/{ts5:.3f} vs {bs:.3f})')\n"
            "a1.axhline(0, c='k', lw=1)\n"
            "a2.bar(['timer - buy&hold'], [sp], color=(RED if sp<0 else GREEN), width=.4)\n"
            "a2.axhline(0, c='k', lw=1); a2.set_ylabel('mean-return spread (bps/day)')\n"
            "a2.set_title(f'Mean return: {sp:+.2f} bps/day (t {spt:+.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'timer Sharpe {ts:.3f} (5bps {ts5:.3f}) | buy&hold {bs:.3f} | {sw:.1f} switches/yr | mean spread {sp:+.2f} bps/d (t {spt:+.2f})')"
        ),
        md(
            "> 💡 In plain words: the timer's *Sharpe* tie is an illusion — it sits in cash ~52% of the "
            "time, trimming volatility, not adding return. On the honest metric (mean return) it "
            f"**loses** {abs(R['ov_spread_bps'])} bps/day (*t* {R['ov_spread_t']}) while churning "
            f"{R['switches_yr']} round-trips a year, and at 5 bps/switch its Sharpe ({R['timer_sharpe5']}) "
            f"falls *below* buy-and-hold ({R['bh_sharpe']}). `MIRAGE`."
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further — the synthetic positive control\n\n"
            "Is the engine a faithful detector, or does it always print ~0? Plant a real *predictive* "
            "edge (`edge > 0`) of increasing strength — **while keeping the same contemporaneous inverse "
            "link on** (so the null exactly matches the real world: link present, forecast absent) — and "
            "watch the Q5−Q1 HAC *t* rise, averaged over 25 seeds."
        ),
        code(
            "signals = [0.0, 0.005, 0.010, 0.020, 0.040]\n"
            "res = [st.synthetic_mean_t(data, edge=a, n_seeds=25, horizon=21, n_days=3000) for a in signals]\n"
            "ts = [r['mean_t'] for r in res]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(signals, ts, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1, label='t = +2 bar')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted predictive edge (0 = null: link on, forecast off)')\n"
            "ax.set_ylabel('mean Q5-Q1 HAC t (25 seeds)')\n"
            "ax.set_title('The harness banks a planted edge — flat at the null, past +2 as it grows')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for a, r in zip(signals, res): print(f'edge {a:.3f} -> mean spread {r[\"mean_spread\"]*100:+.2f}%, mean HAC t {r[\"mean_t\"]:+.2f}')"
        ),
        md(
            "The mean HAC *t* is ≈ 0 at the null (no false positive, even though the inverse *link* is "
            "present) and climbs far past +2 as the planted *forecast* edge grows — so the engine is a "
            "faithful detector. The real-tape null is therefore a statement about **this tape**: gold "
            "co-moves with real yields but is not timed by their trend. For gold from other angles, see "
            "[640 Gold-Overnight](../../640-gold-overnight/) and [649 Gold-Seasonality](../../649-gold-seasonality/)."
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
