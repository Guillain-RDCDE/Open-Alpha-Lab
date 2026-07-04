"""Generate the two narrative notebooks for Study 618 (GBTC Premium Cycle).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached GBTC + BTC-USD
closes under ../_cache/ and otherwise quote the frozen headline numbers in ``R`` (mirroring
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance GBTC + BTC-USD,
# 2015-05-11 -> 2026-06-30, 2,801 GBTC trading days, as-of 2026-06-30).
R = dict(
    start="2015-05-11", end="2026-06-30", n_days=2801, fingerprint="5ffd3440484c",
    # model anchors (pre-spin-off terms) vs Grayscale disclosures
    bps_split=0.0010076, bps_conv=0.00089442,
    # regimes: (label, start, end, n, mean%, hac_t, extreme%, extreme_date)
    regimes=[("premium era", "2015-05-11", "2021-02-22", 1457, 36.32, 8.97, 132.01, "2017-08-31"),
             ("discount era", "2021-02-23", "2024-01-10", 726, -24.49, -7.44, -48.80, "2022-12-13"),
             ("ETF era", "2024-01-11", "2026-06-30", 618, -0.03, -0.85, -4.11, "2026-04-07")],
    first_discount="2021-02-26", welch_regimes=80.23,
    etf_mean=-0.026, etf_sd=0.915, etf_absmean=0.638,
    # convergence: (label, days, prem_in%, prem_out%, total_logpts, bps_day, hac_t)
    conv_full=("full 2023", 257, -45.95, -1.52, 59.99, 23.34, 1.92),
    conv_trig=("ex-ante trigger", 143, -36.29, -1.52, 43.54, 30.45, 2.69),
    # costs on the triggered trade: (one-way bps, net simple %)
    costs=[(10, 49.63), (25, 48.74), (50, 47.26)],
    legs_pct=0.40, borrow_pct=2.84, years=0.57,
    # events: (date, label, move%, z)
    events=[("2023-06-15", "BlackRock files (after hours)", -0.73, -0.3),
            ("2023-06-16", "first close after the filing", 9.19, 3.5),
            ("2023-08-29", "Grayscale v. SEC ruling", 9.64, 3.7),
            ("2024-01-10", "SEC approves spot ETFs", 2.72, 1.0),
            ("2024-01-11", "first day as an ETF", 1.04, 0.4)],
    # lockup arb cohorts
    arb_early_n=63, arb_early_mean=26.49, arb_early_worst=-1.98, arb_early_pos=98,
    arb_late_n=10, arb_late_mean=-16.96, arb_late_worst=-23.03, arb_welch=17.76,
    # synthetic control: (label, plateau%, hac_t)
    syn=[("premium + drift", 41.05, 3.37), ("premium, NO drift", 41.05, 0.29),
         ("null (prem ~ 0)", 0.75, 0.29)],
)

BADGES = (
    "![Signal: Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square)\n"
    "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
    "![Who could harvest?: Mixed](https://img.shields.io/badge/Who_could_harvest%3F-Mixed-8b949e?style=flat-square)\n\n"
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

from gbtc_premium_cycle import data, strategy as st

HAVE_REAL = data.have_real()
DF = data.load_real(as_of=data.AS_OF) if HAVE_REAL else None
print("real GBTC/BTC cache present:", HAVE_REAL,
      "| rows:", (0 if DF is None else len(DF)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# The fund that traded 40 % above its own bitcoin — then 49 % below — then exactly at it 🎁\n"
            "### The GBTC premium cycle: one wrapper, three regimes, all mechanical — in plain English\n\n"
            + BADGES +
            "From 2015 to 2024, one fund — Grayscale's GBTC — held nothing but bitcoin, and *still* "
            "managed to be spectacularly mispriced in **both directions**. For six years the shares cost "
            "up to **2.3× the bitcoin inside them**. Then for three years you could buy the same bitcoin "
            "at **half price** — and it kept getting cheaper. Then, on one dated, scheduled day in "
            "January 2024, the gap **vanished** and never came back.\n\n"
            "No secret information, no genius. Every one of those three regimes was **mechanical** — "
            "built out of who was allowed to create shares, who was allowed to redeem them (nobody), and "
            "what happened when that changed.\n\n"
            "> 📓 **Plain-language layer.** Want the HAC *t*-stats, the calibration residuals and the "
            "cost math? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **One honest note up front.** The star trade of this story — buying the discount in "
            "2023 — was real, big, and *is now permanently closed*. This is a teardown of a finished "
            "trade, not a tip. Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Did GBTC really trade 40 % above its bitcoin? | **Yes.** From 2015 to early 2021 the "
            "premium averaged **+36 %**, peaking at **+132 %** in 2017. People paid double for wrapped "
            "bitcoin because a brokerage account couldn't hold the real thing. |\n"
            "| And then 45 % below? | **Yes — worse.** From Feb-2021 the same shares flipped to a "
            "discount that bottomed at **−48.8 %** in December 2022. The coins were trapped: the trust "
            "had **no redemption mechanism**. |\n"
            "| Did it really snap to zero when it became an ETF? | **Yes, to the basis point.** Since "
            "2024-01-11 the gap averages **−0.03 %**. ETF market-makers can create *and redeem* at NAV, "
            "so any gap gets arbitraged away within hours. |\n"
            "| Could *you* have made money on this? | **Only on one of the three legs.** The premium was "
            "harvestable only by accredited insiders (and it eventually bankrupted several of them). The "
            "**2023 discount convergence** was the public's leg: ~**+50 % net** in 7 months, hedged. "
            "Today: nothing left. |\n\n"
            "> One wrapper, three regimes, three different sets of winners. All of it mechanical."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"GBTC traded 40 % above its bitcoin, then 45 % below, then snapped to zero the day it "
            "became an ETF — one wrapper, three regimes, all mechanical.\"*\n\n"
            "The mechanics in one breath: GBTC was a **trust**. New shares could be created — at fair "
            "value — **only by accredited investors**, who then had to hold them for a 6-to-12-month "
            "lockup before selling to the public. And **nobody** could ever redeem shares back into "
            "bitcoin. One-way in, no way out.\n\n"
            "- While GBTC was the *only* bitcoin in a brokerage account → retail demand piled into a "
            "fixed public float → **premium**.\n"
            "- Once futures, Canadian ETFs and rivals arrived → demand left, but the coins couldn't → "
            "**discount** (no redemption = no floor).\n"
            "- The moment it converted to a real ETF (2024-01-11) → create **and redeem** at NAV → "
            "**zero**, forever."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "This is the cleanest live lesson in what actually pins a fund's price to its assets: **not "
            "fairness, not efficiency — the redemption arb**. Remove it and a fund holding the world's "
            "most liquid crypto asset can trade at *half* its value for years. Restore it and the gap "
            "dies in a day.\n\n"
            "It also matters because each regime paid a *different* person. If you can't say **who** "
            "gets to do the arbitrage, you don't understand the price. That's the third axis of this "
            "study: who could actually harvest each regime?"
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "GBTC publishes how much bitcoin backs each share — and it moves for known, boring reasons: "
            "a **2 %/yr fee** taken in bitcoin, a **91-for-1 split** in 2018, a fee cut at the ETF "
            "conversion, and a 2024 spin-off. So we can **rebuild the bitcoin-per-share for every day "
            "since 2015** from those public mechanics, multiply by the bitcoin price, and compare with "
            "what GBTC actually traded at.\n\n"
            "Best of all, the model **grades itself**: after the ETF conversion the arbitrage forces the "
            "gap to ~0. If our reconstruction were off by even 1 %, the post-2024 'premium' would sit at "
            "a visible constant offset. It sits at **−0.03 %**. The ruler is straight."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**The whole life of the wrapper.** GBTC's premium/discount to its own bitcoin, every "
            "trading day from its 2015 OTC debut through today."
        ),
        code(
            "if HAVE_REAL:\n"
            "    prem = DF['prem'] * 100\n"
            "    fig, ax = plt.subplots(figsize=(10.5, 5.2))\n"
            "    ax.plot(prem.index, prem.values, lw=1.0, color='#333333')\n"
            "    ax.axhline(0, c=GREY, lw=1)\n"
            "    ax.axvspan(pd.Timestamp('2015-05-11'), pd.Timestamp('2021-02-22'), color=GREEN, alpha=.10)\n"
            "    ax.axvspan(pd.Timestamp('2021-02-23'), pd.Timestamp('2024-01-10'), color=RED, alpha=.10)\n"
            "    ax.axvspan(pd.Timestamp('2024-01-11'), prem.index[-1], color=GREY, alpha=.12)\n"
            "    ax.annotate('PREMIUM era\\navg +36%', (pd.Timestamp('2017-06-01'), 95), color=GREEN, fontsize=11, ha='center')\n"
            "    ax.annotate('DISCOUNT era\\ntrough -49%', (pd.Timestamp('2022-06-01'), 45), color=RED, fontsize=11, ha='center')\n"
            "    ax.annotate('ETF era\\n= 0', (pd.Timestamp('2025-06-01'), 45), color=GREY, fontsize=11, ha='center')\n"
            "    ax.axvline(pd.Timestamp('2024-01-11'), c=GREY, ls='--', lw=1)\n"
            "    ax.set_ylabel('premium / discount to held bitcoin (%)')\n"
            "    ax.set_title('One wrapper, three regimes: GBTC vs the bitcoin inside it (2015-2026)')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print(f'peak {prem.max():+.1f}%  trough {prem.min():+.1f}%  today {prem.iloc[-1]:+.2f}%')\n"
            "else:\n"
            "    print('cache missing - see docs/results.md:',\n"
            "          {k: R[k] for k in ('regimes',)})"
        ),
        md(
            f"Three planets. The premium era averaged **+{R['regimes'][0][4]:.0f} %** (peak "
            f"**+{R['regimes'][0][6]:.0f} %** in August 2017 — people paid 2.3× for wrapped bitcoin). "
            f"The discount era averaged **{R['regimes'][1][4]:.0f} %**, bottoming at "
            f"**{R['regimes'][1][6]:.1f} %** the week after FTX died. And since the ETF stamp: "
            f"**{R['regimes'][2][4]:.2f} %**. Not *roughly* zero — zero to the basis point, because "
            "market-makers now arbitrage any gap the same hour it appears."
        ),
        md(
            "**The public's one leg: the 2023 convergence.** On 2023-06-15 BlackRock — whose ETF "
            "filings almost never fail — filed for a spot bitcoin ETF. From that moment the endgame had "
            "a date: either GBTC converts (gap → 0) or it loses to rivals. Buy GBTC at −36 %, short the "
            "same amount of bitcoin (so bitcoin's own swings cancel), wait for the stamp."
        ),
        code(
            "if HAVE_REAL:\n"
            "    win = DF.loc['2023-01-01':'2024-03-01', 'prem'] * 100\n"
            "    fig, ax = plt.subplots(figsize=(10.0, 4.8))\n"
            "    ax.plot(win.index, win.values, lw=1.6, color='#333333')\n"
            "    ax.axhline(0, c=GREY, lw=1)\n"
            "    for d, lab, c in [('2023-06-15', 'BlackRock files', AMBER),\n"
            "                      ('2023-08-29', 'Grayscale wins in court', GREEN),\n"
            "                      ('2024-01-11', 'ETF day: gap = 0', RED)]:\n"
            "        ax.axvline(pd.Timestamp(d), c=c, ls='--', lw=1.4)\n"
            "        ax.annotate(lab, (pd.Timestamp(d), win.min()+2), rotation=90, fontsize=9, color=c, va='bottom', ha='right')\n"
            "    ax.set_ylabel('discount to held bitcoin (%)')\n"
            "    ax.set_title('The dated trade: the discount walks to zero into the conversion (2023)')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print(f\"entry (2023-06-16 close): {DF['prem'].loc[:'2023-06-16'].iloc[-1]*100:+.1f}%   \"\n"
            "          f\"exit (2024-01-11 close): {DF['prem'].loc[:'2024-01-11'].iloc[-1]*100:+.1f}%\")\n"
            "else:\n"
            "    print('cache missing - triggered trade:', R['conv_trig'])"
        ),
        md(
            f"Entering at the close **one full day after** the filing (no crystal ball, and you forgo "
            f"the first-day jump) still captured **{R['conv_trig'][4]:+.1f} log-points** in "
            f"{R['conv_trig'][1]} trading days — about **+{R['costs'][0][1]:.0f} % net** of trading "
            f"costs and the cost of shorting bitcoin. The quants notebook shows this clears the desk's "
            f"statistical bar (HAC *t* = {R['conv_trig'][6]:.2f}) — it wasn't noise, it was a scheduled "
            "convergence with two 3.5-sigma news days along the way."
        ),
        md(
            "**And the premium era? Who was cashing that?** Not you. Only accredited investors could "
            "create shares at fair value — then dump them on the public at +36 % after a lockup. We "
            "price that assembly line, month by month:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    coh = st.lockup_arb_cohorts(DF)\n"
            "    fig, ax = plt.subplots(figsize=(10.0, 4.6))\n"
            "    colors = [GREEN if v > 0 else RED for v in coh['net_log_pct']]\n"
            "    ax.bar(coh.index, coh['net_log_pct'], width=22, color=colors)\n"
            "    ax.axhline(0, c=GREY, lw=1)\n"
            "    ax.set_ylabel('net P&L per created cohort (log-pts)')\n"
            "    ax.set_title('The accredited create-and-dump: free money for 5 years, then the widow-maker')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print(f\"green cohorts: {int((coh['net_log_pct']>0).sum())} / {len(coh)}\")\n"
            "else:\n"
            "    print('cache missing - cohorts:', R['arb_early_mean'], R['arb_late_mean'])"
        ),
        md(
            f"For five years, every monthly cohort of created shares exited its lockup into a fat "
            f"premium: **+{R['arb_early_mean']:.0f} log-points per round**, {R['arb_early_pos']} % "
            f"hit rate. Hedge funds ran it levered — it looked riskless. Then the premium flipped "
            f"**mid-lockup** and the same trade delivered **{R['arb_late_mean']:.0f}** to "
            f"**{R['arb_late_worst']:.0f}**: that's the mechanism that helped sink Three Arrows "
            "Capital and BlockFi. The public, meanwhile, was never allowed in — retail could only *pay* "
            "the premium the insiders were dumping."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Real.** All three regimes are on the tape exactly as the story says: "
            f"**+{R['regimes'][0][4]:.0f} %** average premium, **{R['regimes'][1][4]:.0f} %** average "
            f"discount (trough {R['regimes'][1][6]:.0f} %), then **{R['regimes'][2][4]:.2f} %** under "
            "the ETF arb. The 2023 convergence passes the honest statistical bar on its no-look-ahead "
            "version.\n"
            "- **Tradability — Fragile.** The one public leg paid ~+50 % net in 7 months with billions "
            "of capacity — and then the conversion **deleted the trade forever**. Real, harvested, "
            "extinct.\n"
            "- **Who could harvest? — Mixed.** Premium era: accredited only (until it blew them up). "
            "Discount era: anyone with a brokerage. ETF era: no one — the arbitrage became the product."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **The general law.** Any fund without redemption can drift arbitrarily far from its "
            "assets (closed-end funds have done it for a century). The GBTC cycle is that law run at "
            "crypto speed and size.\n"
            "- **Where it echoes today.** Wrapped-asset premiums didn't die — they moved (MSTR's "
            "bitcoin-per-share premium is the operating-company cousin: see "
            "[324-bitcoin-treasury](../../324-bitcoin-treasury/)).\n"
            "- **The meta-lesson.** When a mispricing has a *mechanical* cause, look for the *dated "
            "event* that removes the mechanism. That's when a gap is a trade rather than a curiosity.\n\n"
            "*Think you've found the next locked-up wrapper trading half off? First question, always: "
            "**what exactly forces it back — and when?***"
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
            "# The GBTC premium cycle — a quantitative teardown 🔬\n"
            "### A self-calibrating BTC-per-share reconstruction · regime HAC/Welch stats · the ex-ante "
            "2023 convergence trade with HAC *t* and costs · catalyst event-day z's · the lockup-arb "
            "cohort arithmetic · a planted-regime synthetic control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The claim "
            "— **one wrapper, three regimes, all mechanical** — is unusually testable: every regime "
            "boundary is a *dated public event*, and the ETF era provides a built-in calibration check "
            "on the whole reconstruction.\n\n"
            "> ⚠️ **Data note.** yfinance GBTC + BTC-USD daily closes, 2015-05-11 → 2026-06-30. "
            "BTC-per-share is *modeled* from public trust mechanics (0.1 BTC/share inception, 2 %/yr "
            "fee in BTC, 91:1 split, 1.5 %/yr after conversion, ×0.90 Mini-Trust spin-off — Yahoo bakes "
            "the spin-off into all prices, so the factor applies to the whole path). BTC-USD stamps "
            "00:00 UTC vs GBTC 16:00 ET: noise, not bias. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (fingerprint `" + R["fingerprint"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `REAL` | Premium era **+{R['regimes'][0][4]:.2f} %** (HAC "
            f"*t* = {R['regimes'][0][5]:+.2f}), discount era **{R['regimes'][1][4]:.2f} %** (HAC "
            f"*t* = {R['regimes'][1][5]:+.2f}), regime split Welch *t* = {R['welch_regimes']:.1f}; "
            f"ex-ante convergence **HAC *t* = {R['conv_trig'][6]:+.2f}** "
            f"({R['conv_trig'][4]:+.1f} log-pts); catalysts at z = +3.5/+3.7; ETF era "
            f"**{R['regimes'][2][4]:+.2f} %**. Full-2023 hindsight window reads t = "
            f"{R['conv_full'][6]:.2f} — reported as such. |\n"
            f"| **Tradability** | `FRAGILE` | The triggered trade netted **+{R['costs'][0][1]:.1f} %** "
            f"(10 bps legs + 5 %/yr borrow) in {R['years']:.2f} yrs with ETF-scale capacity — but the "
            "wrapper arb is structurally dead since 2024-01-11 (mean \\|prem\\| "
            f"{R['etf_absmean']:.2f} %). One-shot, closed. |\n"
            f"| **Who could harvest?** | `MIXED` | Premium era: accredited create-and-dump only "
            f"(+{R['arb_early_mean']:.1f} log-pts/cohort, {R['arb_early_pos']} % hit rate → then "
            f"{R['arb_late_mean']:.1f} mean / {R['arb_late_worst']:.1f} worst when the regime flipped "
            f"mid-lockup; Welch *t* = {R['arb_welch']:.1f}). Discount era: anyone. ETF era: no one. |\n\n"
            "> 💡 In plain words: the lifecycle is real and mechanical end to end; the only leg the "
            "public could ever trade paid ~+50 % once — and can never pay again."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $B_t$ be the modeled bitcoin-per-share and $P_t$ the GBTC close. The premium is\n\n"
            "$$\\pi_t = \\frac{P_t}{B_t \\cdot \\mathrm{BTC}_t} - 1,\\qquad\n"
            "B_t = 0.9\\cdot\\frac{0.1}{91}\\,e^{-f(t)}\\;$$\n\n"
            "with $f$ the cumulated sponsor fee (2 %/yr to 2024-01-11, 1.5 %/yr after) and 0.9 the "
            "spin-off factor Yahoo folds into every price. A long-GBTC / short-BTC book earns exactly "
            "$\\Delta\\log(1+\\pi_t)$ per day — the premium change, with bitcoin beta cancelled and the "
            "fee decay as the carry.\n\n"
            "- **H₁ (three regimes).** $\\pi$ is strongly positive 2015–2021, strongly negative "
            "2021–2024, ≈ 0 after — with *dated* boundaries.\n"
            "- **H₂ (the dated trade).** The 2023 discount → 0 convergence was statistically real and "
            "survivable after costs, on a no-look-ahead entry.\n"
            "- **H₃ (who harvests).** Premium era = accredited create-and-dump through a lockup; "
            "discount era = public; ETF era = nobody.\n\n"
            "We find **H₁ supported** (HAC t = +9.0 / −7.4, Welch t = 80, ETF era −0.03 %), **H₂ "
            "supported on the honest window** (HAC t = 2.69 net +49.6 %; the hindsight full-2023 "
            "window alone reads 1.92), **H₃ quantified** (cohort table)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — where the inference must live\n\n"
            "Two statistical traps sit in this study and we name both:\n\n"
            "1. **The premium level is near-unit-root inside a regime.** A naive t on the level would "
            "be fantasy; even our HAC(63) level t's are *supporting* evidence only. The decisive "
            "inference lives on the **drift** (daily hedged returns, HAC(10)) and on **event days** "
            "(z vs the regime's own daily sd).\n"
            "2. **Window snooping.** \"Buy the discount in January 2023\" is a hindsight window — we "
            "report it (t = 1.92) but grade the **ex-ante** version: entry at the close one trading "
            "day *after* a public, dated catalyst (BlackRock's S-1, 2023-06-15, filed after hours) — "
            "the study's single documented execution lag — exit at the conversion close.\n\n"
            "And one modeling trap: a reconstructed NAV could hide an error. Hence the built-in "
            "calibration: post-conversion the in-kind arb pins $\\pi$ to ≈ 0, so any BPS-model error "
            "appears as a constant ETF-era offset. Measured offset: **−0.026 %**."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Tape.** GBTC + BTC-USD daily closes (yfinance), {R['start']} → {R['end']} "
            f"({R['n_days']:,} GBTC trading days), cache-first, as-of {R['end']}, fingerprint "
            f"`{R['fingerprint']}`.\n"
            "- **BPS model.** Public mechanics only (no fitted parameters): inception ratio, fee "
            "accrual, split, conversion fee cut, spin-off factor. Anchors: 0.0010076 vs disclosed "
            "≈0.00101 (2018 split), 0.00089442 vs ≈0.00089 (conversion).\n"
            "- **Regimes.** Boundaries are dated public events (OTCQX debut, first sustained discount "
            "2021-02, conversion 2024-01-11) — not fitted breakpoints. HAC(63) on levels "
            "(supporting), Welch t across regimes.\n"
            "- **The trade.** Daily hedged return $\\Delta\\log(1+\\pi)$; HAC(10) t over the ex-ante "
            "window (entry lag = one trading day after the filing); costs 10/25/50 bps one-way × 4 "
            "legs; short-BTC borrow 5 %/yr.\n"
            "- **Events.** One-day hedged moves on the documented catalysts, z vs discount-era sd.\n"
            "- **Third axis.** Monthly creation cohorts through a 126-trading-day lockup (the "
            "*favourable* post-2019 six-month rule), net of fee drag, 3 × 25 bps legs, 5 %/yr borrow.\n"
            "- **Control.** A planted three-regime synthetic world + a null; the machinery must "
            "recover levels and light up only when a drift is planted."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The reconstruction and its built-in calibration\n\n"
            "First the ruler. The ETF era is the check: if the BPS model were wrong by x %, the "
            "post-2024 premium would sit at a constant x % offset instead of 0."
        ),
        code(
            "if HAVE_REAL:\n"
            "    etf = DF.loc[data.CONVERSION:, 'prem'] * 100\n"
            "    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.2))\n"
            "    a1.plot(etf.index, etf.values, lw=1.0, color='#333333')\n"
            "    a1.axhline(0, c=GREEN, lw=1.5)\n"
            "    a1.set_title(f'ETF era: mean {etf.mean():+.3f}%  (arb-pinned)')\n"
            "    a1.set_ylabel('reconstructed premium (%)')\n"
            "    a2.hist(etf.values, bins=45, color=GREY)\n"
            "    a2.axvline(etf.mean(), c=GREEN, lw=2, label=f'mean {etf.mean():+.3f}%')\n"
            "    a2.set_title(f'sd {etf.std(ddof=1):.3f}%   mean |prem| {etf.abs().mean():.3f}%')\n"
            "    a2.legend()\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print(f'calibration residual: mean {etf.mean():+.3f}%  sd {etf.std(ddof=1):.3f}%')\n"
            "else:\n"
            "    print('cache missing:', R['etf_mean'], R['etf_sd'], R['etf_absmean'])"
        ),
        md(
            f"> 💡 In plain words: after the conversion, market-makers force price = NAV. Our "
            f"reconstruction of that NAV lands **{R['etf_mean']:+.3f} %** away on average (sd "
            f"{R['etf_sd']:.3f} %, and the sd is mostly the UTC-vs-4pm-ET timestamp mismatch). A "
            "no-free-parameter model that ends up 3 bp from the arb-enforced truth also validates every "
            "*earlier* premium it produces."
        ),
        md(
            "### 4b · Three regimes on one tape\n\n"
            "Levels per documented regime — HAC(63) t's, with the persistence caveat stated: these are "
            "supporting statistics; the premium is near-unit-root inside a regime."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = st.regime_table(DF)\n"
            "    for r in rows:\n"
            "        print(f\"{r['regime']:<26s} {r['start']} -> {r['end']}  n={r['n']:>5d}  \"\n"
            "              f\"mean {r['mean_pct']:+7.2f}%  HAC t={r['t_hac']:+6.2f}  \"\n"
            "              f\"[{r['min_pct']:+7.2f}% .. {r['max_pct']:+7.2f}%]\")\n"
            "    regs = data.regimes(DF)\n"
            "    wt = st.welch_t(regs['premium era (2015-2021)']['prem'].to_numpy(),\n"
            "                    regs['discount era (2021-2024)']['prem'].to_numpy())\n"
            "    print(f'premium vs discount era: Welch t = {wt:+.2f}')\n"
            "    print('first sustained discount (5 closes):', st.first_sustained_discount(DF))\n"
            "else:\n"
            "    for r in R['regimes']: print(r)"
        ),
        md(
            f"> 💡 In plain words: **+{R['regimes'][0][4]:.1f} %** for six years, then "
            f"**{R['regimes'][1][4]:.1f} %** for three, then **{R['regimes'][2][4]:.2f} %**. The flip "
            f"date ({R['first_discount']}) is a one-way regime change, not a wobble — the two eras are "
            f"Welch *t* = {R['welch_regimes']:.0f} apart. No survivorship anywhere: this is one named "
            "vehicle followed cradle to grave."
        ),
        md(
            "### 4c · The dated trade — HAC t on the hedged drift\n\n"
            "Long GBTC / short BTC earns $\\Delta\\log(1+\\pi)$. Two windows: the hindsight full-2023 "
            "window (reported, not graded) and the ex-ante triggered window (graded)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    full = st.convergence_test(DF, '2023-01-03', '2024-01-11', lags=10)\n"
            "    trig = st.convergence_test(DF, '2023-06-16', '2024-01-11', lags=10)\n"
            "    for lab, ct in [('full 2023 (hindsight)', full), ('ex-ante trigger', trig)]:\n"
            "        print(f\"{lab:<22s} {ct['start']} -> {ct['end']}  n={ct['n_days']:>3d}  \"\n"
            "              f\"prem {ct['prem_entry_pct']:+.2f}% -> {ct['prem_exit_pct']:+.2f}%  \"\n"
            "              f\"total {ct['total_log_pct']:+.2f} log-pts  \"\n"
            "              f\"{ct['mean_daily_bps']:+.2f} bps/d  HAC(10) t = {ct['t_hac']:+.2f}\")\n"
            "    h = st.hedged_returns(DF)['2023-06-17':'2024-01-11']\n"
            "    fig, ax = plt.subplots(figsize=(9.6, 4.4))\n"
            "    ax.plot(h.index, h.cumsum().values * 100, lw=1.6, color=GREEN)\n"
            "    ax.axvline(pd.Timestamp('2023-08-29'), c=AMBER, ls='--', label='court ruling (z=+3.7)')\n"
            "    ax.set_ylabel('cumulative hedged log return (pp)')\n"
            "    ax.set_title('The triggered convergence trade, entry 2023-06-16 close -> conversion')\n"
            "    ax.legend(); plt.tight_layout(); plt.show()\n"
            "else:\n"
            "    print(R['conv_full']); print(R['conv_trig'])"
        ),
        md(
            f"> 💡 In plain words: entered with a clean one-day lag after a public filing, the trade "
            f"made **{R['conv_trig'][4]:+.1f} log-pts** in {R['conv_trig'][1]} days at "
            f"**HAC *t* = {R['conv_trig'][6]:+.2f}** — real by the desk's bar on the real tape. The "
            f"full-year hindsight window (t = {R['conv_full'][6]:.2f}) doesn't clear alone at daily "
            "HAC and we say so; the graded claim is the triggered version, which does."
        ),
        md("### 4d · Event days — the catalysts in standard deviations"),
        code(
            "if HAVE_REAL:\n"
            "    ev = st.event_days(DF, sd_window=('2021-02-23', '2024-01-10'))\n"
            "    for e in ev:\n"
            "        print(f\"{e['date']}  {e['move_pct']:+6.2f}%  z={e['z']:+5.1f}  {e['label']}\")\n"
            "    fig, ax = plt.subplots(figsize=(9.2, 4.2))\n"
            "    labs = [e['date'] for e in ev]; zs = [e['z'] for e in ev]\n"
            "    ax.bar(labs, zs, color=[GREEN if abs(z) >= 2 else GREY for z in zs], width=.55)\n"
            "    ax.axhline(2, ls='--', c=RED, label='|z| = 2')\n"
            "    ax.set_ylabel('one-day hedged move (z)'); ax.legend()\n"
            "    ax.set_title('The two 2023 catalysts are 3.5-3.7 sd days; the approval itself was priced')\n"
            "    plt.tight_layout(); plt.show()\n"
            "else:\n"
            "    for e in R['events']: print(e)"
        ),
        md(
            "> 💡 In plain words: the discount didn't melt smoothly — it **jumped** on the two days "
            "that changed the conversion's probability (the BlackRock filing's first close, "
            f"z = +3.5, and the court ruling, z = +3.7), and barely moved on approval day itself "
            "(z = +1.0): by then the market had done the math. Textbook event-driven convergence."
        ),
        md(
            "### 4e · Costs — does the triggered trade survive?\n\n"
            "Four legs of one-way costs (enter/exit × long GBTC + short BTC) plus 5 %/yr borrow on the "
            "short leg. The long leg's 2 %/yr sponsor fee is already inside the hedged return (NAV "
            "decay)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    for cb in (10.0, 25.0, 50.0):\n"
            "        c = st.convergence_costs(DF, '2023-06-16', '2024-01-11', cost_bps=cb, borrow_pa=5.0)\n"
            "        print(f\"{cb:>4.0f} bps/leg: gross {c['gross_log_pct']:+.2f} - legs \"\n"
            "              f\"{c['legs_cost_pct']:.2f}pp - borrow {c['borrow_cost_pct']:.2f}pp \"\n"
            "              f\"= net {c['net_log_pct']:+.2f} log-pts = {c['net_simple_pct']:+.2f}% simple\")\n"
            "else:\n"
            "    for cb, net in R['costs']: print(f'{cb} bps/leg -> net {net:+.2f}% simple')"
        ),
        md(
            f"> 💡 In plain words: **+{R['costs'][0][1]:.1f} %** net at 10 bps legs, still "
            f"**+{R['costs'][2][1]:.1f} %** at an absurd 50 bps — a 43-log-pt bounded convergence "
            "doesn't care about spreads. Capacity was ETF-scale (~$25B of GBTC at conversion). The "
            "fragility isn't costs — it's that the trade **can never recur**."
        ),
        md(
            "### 4f · Third axis — the create-and-dump, cohort by cohort\n\n"
            "Create at NAV (accredited only), hedge the BTC, exit at the market premium 126 trading "
            "days later (the favourable six-month lockup), net of fee drag + 3 × 25 bps legs + 5 %/yr "
            "borrow. Welch t between the pre-2020-09 cohorts and the cohorts that exited into the flip."
        ),
        code(
            "if HAVE_REAL:\n"
            "    coh = st.lockup_arb_cohorts(DF)\n"
            "    s = st.lockup_arb_summary(coh)\n"
            "    print(f\"early cohorts  (n={s['n_early']}): mean {s['early_mean_pct']:+.2f} log-pts  \"\n"
            "          f\"worst {s['early_worst_pct']:+.2f}  share>0 {s['early_share_pos']*100:.0f}%\")\n"
            "    print(f\"blow-up cohorts(n={s['n_late']}): mean {s['late_mean_pct']:+.2f} log-pts  \"\n"
            "          f\"worst {s['late_worst_pct']:+.2f}\")\n"
            "    print(f\"early vs late Welch t = {s['welch_t']:+.2f}\")\n"
            "else:\n"
            "    print(R['arb_early_mean'], R['arb_late_mean'], R['arb_welch'])"
        ),
        md(
            f"> 💡 In plain words: a **{R['arb_early_pos']} % hit-rate, "
            f"+{R['arb_early_mean']:.0f}-log-pt** semi-annual carry, gated to accredited creators — "
            "the public was the *exit liquidity*, not a participant. And because the exit was locked "
            f"126 days forward, the regime flip converted the carry into **{R['arb_late_worst']:.0f} "
            "log-pt** losses with no way out — the exact mechanism in the 3AC/BlockFi post-mortems. "
            "That asymmetry is why the third axis reads **MIXED**: each regime had a different (and "
            "shrinking) set of eligible harvesters."
        ),
        md(
            "### 4g · Faithful-engine control — planted regimes, planted drift\n\n"
            "A synthetic world with a +40 % plateau, a −45 % trough and a *tunable* convergence drift "
            "into a known conversion date, plus a null whose premium is AR(1) noise around zero."
        ),
        code(
            "for label, kw in [('premium + drift', dict(drift_on=True, premium_on=True)),\n"
            "                  ('premium, NO drift', dict(drift_on=False, premium_on=True)),\n"
            "                  ('null (prem ~ 0)', dict(drift_on=False, premium_on=False))]:\n"
            "    sw = data.synthetic_world(seed=618, **kw)\n"
            "    c0, c1 = sw.attrs['conv_window']\n"
            "    ct = st.convergence_test(sw, str(c0.date()), str(c1.date()), lags=10)\n"
            "    pr = sw.attrs['regime_slices']['premium']\n"
            "    lvl = sw.loc[pr[0]:pr[1], 'prem'].mean() * 100\n"
            "    print(f'{label:<20s} plateau {lvl:+7.2f}%   conv HAC t = {ct[\"t_hac\"]:+6.2f}   '\n"
            "          f'({ct[\"total_log_pct\"]:+7.2f} log-pts)')"
        ),
        md(
            f"> 💡 In plain words: the machinery reads a planted +40 % plateau as "
            f"**+{R['syn'][0][1]:.1f} %**, fires **t = {R['syn'][0][2]:.2f}** only when a convergence "
            f"drift is actually planted, and stays at **t = {R['syn'][1][2]:.2f}** when it isn't — on "
            "the null it invents nothing. *(A faithful-engine / power check only — never cited in "
            "support of the real-tape stamps.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `REAL`** — the three regimes are on the tape with robust statistics (levels "
            f"HAC t = {R['regimes'][0][5]:+.1f} / {R['regimes'][1][5]:+.1f}, split Welch "
            f"t = {R['welch_regimes']:.0f}); the dated 2023 convergence clears the bar on its ex-ante "
            f"version (**HAC t = {R['conv_trig'][6]:+.2f}**, {R['conv_trig'][4]:+.1f} log-pts, "
            "catalysts at z = +3.5/+3.7); the ETF era sits at "
            f"**{R['regimes'][2][4]:+.2f} %** exactly as the in-kind arb demands. Full-2023 alone reads "
            f"t = {R['conv_full'][6]:.2f} — reported, not hidden.\n"
            f"- **Tradability `FRAGILE`** — the public leg netted **+{R['costs'][0][1]:.1f} %** in "
            f"{R['years']:.2f} yrs at ETF-scale capacity and survives any realistic cost — but it was "
            "a one-shot convergence into a terminal event; the wrapper arb no longer exists. Not "
            "INVESTABLE, and not a MIRAGE either: it was real and harvested.\n"
            f"- **Who could harvest? `MIXED`** — accredited-only in the premium era "
            f"(+{R['arb_early_mean']:.0f} log-pts/cohort until the flip made it "
            f"{R['arb_late_worst']:.0f}), anyone in the discount era, no one since. The wrapper's "
            "mispricing was always a story about *access*, not insight."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **Closed-end-fund theory, crypto speed.** Lee-Shleifer-Thaler and Pontiff bound a "
            "fund's price-NAV gap by the cost of the arb; GBTC ran the boundary cases in one decade — "
            "one-way gated arb (premium), no arb (discount), free in-kind arb (zero).\n"
            "- **The sibling wrapper.** [324-bitcoin-treasury](../../324-bitcoin-treasury/) prices "
            "MSTR's equity premium on wrapped BTC — same coins, a wrapper whose premium rests on "
            "leverage and narrative instead of creation gates, and *no* dated event that forces it "
            "shut.\n"
            "- **Generalisation.** Every locked-up vehicle trading far from NAV should be asked this "
            "study's two questions: *what mechanically forces convergence, and who is allowed to trade "
            "it?* Without a dated answer to the first, a 50 % discount can stay a 50 % discount.\n\n"
            "*The reproducible core is offline and deterministic; methods and sources in "
            "[`docs/references.md`](../docs/references.md); frozen numbers in "
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
