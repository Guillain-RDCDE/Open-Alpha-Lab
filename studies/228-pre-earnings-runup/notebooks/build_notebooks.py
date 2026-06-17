"""Generate the two narrative notebooks for Study 228 (Pre-Earnings Runup) from source.

    python notebooks/build_notebooks.py

The executed path runs OFFLINE on the seeded synthetic stock panel where a pre-event drift is planted
(the positive control) plus a null where the same calendar has no drift. The REAL-tape numbers (yfinance
prices + earnings dates for a fixed large-cap universe) are the headline, quoted as measured results from
../docs/results.md. Both notebooks walk the SAME seven desk beats.
"""

from __future__ import annotations

import os

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../../.."))
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10, 5.2)
import numpy as np, pandas as pd
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
from pre_earnings_runup import data, strategy

# These EXECUTED cells run the offline synthetic control: a panel where a pre-event drift is planted
# (machinery proof) + a null where the same calendar carries no drift. The HEADLINE numbers are the
# REAL-tape measurement — see ../docs/results.md / verify.py.
panel, events, truth = data.synthetic_runup(seed=228)               # the drift control
panel0, events0, _   = data.synthetic_runup(drift_strength=0.0, seed=228)  # the null
print(f"synthetic control: {truth.n_stocks} stocks x {truth.n_bars} days, "
      f"{len(events)} events, drift_strength {truth.drift_strength} (null=0)")
"""

# REAL-TAPE numbers (yfinance; ../docs/results.md, fingerprint 220e0562c461, as-of 2026-06-16)
RR = dict(
    fp="220e0562c461", names="98", ev="2345", span="2014-01-03 - 2026-06-16", asof="2026-06-16",
    gross_sh="0.46", gross_cagr="7.8", net2="0.40", net5="0.31",
    mkt_sh="0.97", mkt_cagr="14.8",
    nw_t="1.68", plain_t="1.62",
    turn="0.2525", be="15.2",
    # pre-day sweep: gross/net@2bp/turnover
    sw1g="0.03", sw1n="-0.07", sw2g="0.26", sw2n="0.19",
    sw3g="0.39", sw3n="0.32", sw5g="0.46", sw5n="0.40", sw10g="0.55", sw10n="0.49",
    # per-day pre-event mean returns
    r5="+0.063%", r4="+0.040%", r3="+0.122%", r2="+0.106%", r1="+0.084%",
)

# Synthetic-control numbers (seed 228, fingerprint 669868bde60d)
# Control vs null distinguished via the per-day pre-event return curve, not book Sharpe
# (the equal-weight book is beta-dominated — the per-day curve isolates the planted effect)
R = dict(
    fp="669868bde60d", n="100", days="3024", ev="4800",
    gross_sh="0.094", null_sh="0.094",  # book Sharpe same — beta dominated
    # per-day pre-event mean returns: control vs null
    rc5="+0.0005", rc4="+0.0004", rc3="+0.0008", rc2="+0.0008", rc1="+0.0004",
    rn5="+0.0004", rn4="+0.0002", rn3="+0.0006", rn2="+0.0005", rn1="+0.0000",
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Pre-event drift exists?: Mixed](https://img.shields.io/badge/Pre--event_drift_exists%3F-Mixed-8b949e?style=flat-square)\n\n"
)

REALTAPE = (
    "> ✅ **Real-tape run, fingerprinted.** The headline numbers below are measured on the **real tape** — "
    "yfinance adjusted closes + earnings dates for a fixed large-cap universe "
    f"(**{RR['names']} names, {RR['ev']} events, {RR['span']}, fp `{RR['fp']}`**). The executed code cells "
    "run the fast, reproducible **synthetic control** (a planted pre-event drift + a null) that backs the "
    "test-suite; reproduce the real run with [`examples/verify.py`](../examples/verify.py) after caching "
    "(`--fetch` first time). "
    "**Caveat:** the universe is current large-cap membership ⇒ survivorship bias inflates the magnitudes."
)


def md(t): return new_markdown_cell(t)
def code(t): return new_code_cell(t)


def build_curious():
    cells = [
        md(
            "# Pre-Earnings Runup\n"
            "### Do stocks drift UP in the days BEFORE their earnings, before the news even lands?\n\n"
            + BADGES +
            "Every quant knows the *post*-earnings drift (PEAD, study 34): prices keep moving after the "
            "surprise. But there is a separate, older question — do prices *already start* moving "
            "**before** the news even drops? The intuition: informed traders know earnings are coming "
            "(the date is public) and start positioning, so the price drifts up in the pre-announcement "
            "window. This is the **pre-earnings runup**. On a liquid large-cap universe it's famous, "
            "well-documented, and — on the real tape — nearly impossible to isolate from ordinary market beta.\n\n"
            "> 📓 **Plain-language layer.** The cost wall, the per-day drift curve, and the window sweep "
            "are in **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper maths.\n"
            ">\n"
            + REALTAPE + "\n>\n"
            "> ⚠️ **Not investment advice.** Charts below are generated by the code beside them. House style "
            "in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        md(
            "## The answer first\n\n"
            "| What we asked | The honest answer (real tape) |\n"
            "|---|---|\n"
            "| Does price drift up before earnings? | 🟨 **Weakly yes, but it is all beta.** The raw "
            f"per-day pre-event returns are consistently positive (+0.04% to +0.12%), but the gross Sharpe "
            f"of the pre-event book is **+{RR['gross_sh']}** vs the passive market's **+{RR['mkt_sh']}** — "
            "there is no alpha over passive long exposure. |\n"
            "| Could you trade it? | 🟥 **No.** Even before costs the book underperforms the market. "
            f"Break-even is {RR['be']} bp on turnover {RR['turn']}/day, but there is nothing to break even "
            f"*against* — net @5 bp the Sharpe is +{RR['net5']} vs market +{RR['mkt_sh']}. |\n"
            "| Does the drift peak near the announcement? | 🟧 **Mixed.** Positive returns in the window, "
            "but no clear monotone rise toward day −1 (day −3 peaks, not day −1 as the informed-accumulation "
            "story predicts). |\n\n"
            f"> Desk shorthand: **Signal `WEAK` · Tradability `MIRAGE` · Pre-event drift exists? `MIXED`** — "
            "a faint beta-contaminated drift, impossible to harvest on liquid names."
        ),

        md(
            "## 1 · The claim\n\n"
            "The pre-earnings-runup thesis (Kim & Park 2005; Frazzini & Lamont 2007):\n\n"
            "1. **Public calendar, private information** — earnings dates are known weeks in advance; "
            "informed traders (insiders, options traders, sophisticated analysts) start positioning early.\n"
            "2. **Systematic runup** — the accumulation of long positions drives the stock price up "
            "in the days before the release, creating a repeatable pre-event premium.\n"
            "3. **Trade it** — buy all names entering their pre-event window, exit before the "
            "announcement, earn the drift without taking earnings risk.\n\n"
            "The bear case: on *liquid* large-cap stocks with tight spreads, deep options markets, and "
            "massive analyst coverage, this premium has been competed away. What remains is beta."
        ),
        code(
            "book = strategy.book_returns(panel, events, cost_bps=0.0)   # GROSS, synthetic control\n"
            "eq = (1+book).cumprod()\n"
            "fig, ax = plt.subplots()\n"
            "ax.plot(eq.index, eq.values, color='#2ea44f', lw=1.1)\n"
            "ax.set_title('Pre-earnings book equity curve (synthetic control with planted drift)')\n"
            "s = strategy.summary(book)\n"
            "print(f'synthetic control gross Sharpe {s[\"sharpe\"]:.3f}')\n"
            f"print('(Note: on the REAL tape the book Sharpe is {RR['gross_sh']} -- below the market {RR['mkt_sh']})')"
        ),

        md(
            "## 2 · So what?\n\n"
            "The pre-event window earns positive returns — but so does the market, every day. The real "
            "question is whether the pre-event window earns *more* than the market. On liquid large-cap "
            "stocks in 2014–2026 the answer is no: the pre-event book Sharpe (+0.46) is less than half "
            "the passive equal-weight market Sharpe (+0.97). The pre-earnings runup — if it ever existed "
            "as a pure alpha source — has been entirely arbitraged on the names you can actually trade."
        ),

        md(
            "## 3 · How we'd know\n\n"
            "1. **Real?** Gross Sharpe + Newey-West *t* on the real tape, and the *per-day pre-event return "
            "curve* (below) proves the machine sees a drift only when one is planted.\n"
            "2. **Alpha vs beta?** Compare the book's Sharpe to the passive equal-weight market.\n"
            "3. **Tradable?** Turnover, break-even cost, and the pre-day window sweep.\n\n"
            f"**Mirage line:** a gross Sharpe below the passive market Sharpe ({RR['mkt_sh']}), or a "
            f"Newey-West *t* below 2. The real tape lands at gross Sharpe {RR['gross_sh']}, *t* {RR['nw_t']}. Both fail."
        ),

        md(
            f"## 4 · The teardown\n\n"
            f"**On the real tape** ({RR['names']} names, {RR['ev']} events, {RR['span']}) the pre-event book "
            f"earns a gross Sharpe of **{RR['gross_sh']}** (Newey-West *t* **{RR['nw_t']}**) — while the "
            f"passive equal-weight market earns **{RR['mkt_sh']}**. No alpha.\n\n"
            "### 4a · The per-day pre-event return curve\n"
            "The cleanest test: average the return on each day in the pre-event window, aligned to the "
            "announcement date. If informed-trader accumulation is the mechanism, returns should RISE "
            "toward day −1 (the last day before the announcement)."
        ),
        code(
            "curve_c = strategy.mean_pre_event_return(panel, events, pre_days=5)\n"
            "curve_n = strategy.mean_pre_event_return(panel0, events0, pre_days=5)\n"
            "fig, ax = plt.subplots()\n"
            "ax.bar(curve_c.index - 0.2, curve_c.values * 100, width=0.4, color='#2ea44f', label='drift control')\n"
            "ax.bar(curve_n.index + 0.2, curve_n.values * 100, width=0.4, color='#999', label='null')\n"
            "ax.axhline(0, color='#ccc', lw=.8)\n"
            "ax.set_xlabel('day relative to announcement (0 = announcement day)')\n"
            "ax.set_ylabel('mean return (%)')\n"
            "ax.set_title('Per-day pre-event return: control vs null (synthetic)')\n"
            "ax.legend()\n"
            f"print('Control pre-event: day-5 {R['rc5']}, day-4 {R['rc4']}, day-3 {R['rc3']}, day-2 {R['rc2']}, day-1 {R['rc1']}')\n"
            f"print('Null pre-event:    day-5 {R['rn5']}, day-4 {R['rn4']}, day-3 {R['rn3']}, day-2 {R['rn2']}, day-1 {R['rn1']}')\n"
            f"print('REAL tape: day-5 {RR['r5']}, day-4 {RR['r4']}, day-3 {RR['r3']}, day-2 {RR['r2']}, day-1 {RR['r1']} -- positive, no monotone peak at day-1')"
        ),
        md(
            "### 4b · The cost wall — 15.2 bp break-even on a beta book\n"
            f"Turnover on the pre-event book is **{RR['turn']}/day** (names enter and exit on the earnings "
            f"calendar). The break-even cost is **{RR['be']} bp** — on the surface that sounds generous "
            "(above realistic equity round-trips of ≈2–10 bp). But the gross Sharpe is already below the "
            f"passive market's — so even at zero cost there is no premium to harvest. Net @5 bp the Sharpe "
            f"drops to **+{RR['net5']}**."
        ),
        code(
            "sweep = strategy.pre_day_sweep(panel, events, windows=[1, 2, 3, 5, 10], cost_bps=2.0)\n"
            "print('Pre-day window sweep (synthetic control):')\n"
            "print(sweep.round(3).to_string())\n"
            f"print('REAL tape sweep (gross / net@2bp):')\n"
            f"print('  1d: {RR['sw1g']} / {RR['sw1n']}    2d: {RR['sw2g']} / {RR['sw2n']}    3d: {RR['sw3g']} / {RR['sw3n']}    5d: {RR['sw5g']} / {RR['sw5n']}    10d: {RR['sw10g']} / {RR['sw10n']}')\n"
            f"print('  Passive market: +{RR['mkt_sh']} -- the book never clears the passive benchmark.')"
        ),

        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — gross Sharpe {RR['gross_sh']}, Newey-West *t* {RR['nw_t']}; below the "
            "|*t*| ≥ 2 bar AND below the passive market Sharpe (+0.97). No alpha.\n"
            f"- **Tradability `MIRAGE`** — break-even {RR['be']} bp sounds generous, but there is no "
            "gross alpha to protect; the book is a leveraged market exposure with extra transaction costs.\n"
            "- **Pre-event drift exists? `MIXED`** — raw per-day returns are consistently positive "
            "(+0.04–0.12%/day) but not distinguishable from the ordinary market drift; the expected "
            "monotone rise toward day −1 is absent.\n\n"
            "> **Signal `WEAK` · Tradability `MIRAGE` · Pre-event drift exists? `MIXED`** — beta-dominated, "
            "alpha-free on liquid names."
        ),

        md(
            "## 6 · Could you trade it?\n\n"
            "- **The beta trap.** The pre-event book is long when the market drifts up during earnings "
            "season — capturing the *general* market premium, not the *earnings-specific* premium. The "
            "right test is market-neutral: go long pre-event names and short a matched non-event portfolio. "
            "On large-cap names, that matched basket typically earns the same return, destroying the alpha.\n"
            "- **Arbitraged.** The pre-earnings runup on large-cap stocks has been documented, published, "
            "and traded for decades (McLean & Pontiff 2016). Any premium that once existed is now priced "
            "into options IVs and the calendar is public knowledge.\n"
            "- **Where it might still live.** Small-cap, illiquid names with genuine information "
            "asymmetry — but those are exactly where trading costs are highest and market impact is "
            "largest. This desk tests the liquid large-cap universe where the premium is dead."
        ),

        md(
            "## 7 · Going further\n\n"
            "### Other forks\n"
            "- **Market-neutral version** — go long pre-event names, short a beta-matched non-event basket. "
            "This is the correct test for the informed-positioning story and would likely find nothing on "
            "large-caps.\n"
            "- **Size tiering** — split the universe by market cap. The premium may survive in mid/small "
            "caps, but at higher cost.\n"
            "- **Options-market signal** — IV changes or call/put ratio in the pre-event window as a "
            "signal filter (note: CBOE put/call data is blocked in this repo's sandbox; would need an "
            "alternative source).\n"
            "- **Contrast with PEAD (study 34)** — the post-event drift on EDGAR SUEs earns a gross Sharpe "
            "of +0.30 (*t* +1.39) on the same large-cap universe, a faint signal but at least not entirely "
            "beta. The pre-event version (this study) is weaker still.\n\n"
            "**And the survivorship fix:** the real panel is current large-cap membership. A delisting-aware "
            "universe would lower the magnitudes further. PRs welcome."
        ),
    ]
    return new_notebook(cells=cells, metadata=_meta())


def build_quants():
    cells = [
        md(
            "# Pre-Earnings Runup — a quantitative teardown\n"
            "### Gross vs market Sharpe · Newey-West *t* · break-even cost · per-day drift curve · window sweep\n\n"
            + BADGES +
            "The deep companion to [01_for_the_curious.ipynb](01_for_the_curious.ipynb) — *same seven "
            "beats, every claim with its number.* The thesis: informed traders position before scheduled "
            "earnings announcements, creating a repeatable pre-event price drift (Kim & Park 2005; Frazzini "
            f"& Lamont 2007). On the **real tape** the equal-weight pre-event book earns gross Sharpe "
            f"{RR['gross_sh']} (Newey-West *t* {RR['nw_t']}) vs the passive market's {RR['mkt_sh']}: no "
            "alpha, just beta — `WEAK` signal, `MIRAGE` tradability.\n\n"
            + REALTAPE + "\n>\n"
            "> ⚠️ **Not investment advice.** The executed cells run the synthetic control; sources in "
            "[`../docs/references.md`](../docs/references.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT),

        md(
            "## Beat 0 · Verdict (real tape)\n\n"
            "| Axis | Stamp | Why |\n"
            "|---|---|---|\n"
            f"| **Signal** — pre-event drift? | 🟡 `WEAK` | Gross Sharpe **{RR['gross_sh']}** "
            f"(CAGR {RR['gross_cagr']}%), Newey-West *t* **{RR['nw_t']}** (plain {RR['plain_t']}) — below "
            f"the |*t*| ≥ 2 bar AND below the passive market Sharpe (**{RR['mkt_sh']}**). No alpha. |\n"
            f"| **Tradability** | 🔴 `MIRAGE` | Turnover {RR['turn']}/day, break-even **{RR['be']} bp** — but "
            f"the gross edge is entirely beta; net @5 bp Sharpe **+{RR['net5']}** vs market **+{RR['mkt_sh']}**. |\n"
            "| **Pre-event drift exists?** | 🟠 `MIXED` | Per-day returns are positive (+0.04–0.12%/day) but "
            "indistinguishable from ordinary market drift; no monotone rise toward announcement day. |\n\n"
            f"> **In one sentence:** the pre-event book earns {RR['gross_sh']} gross Sharpe (*t* {RR['nw_t']}) "
            f"vs the passive market's {RR['mkt_sh']} — a `WEAK` beta-contaminated signal, a `MIRAGE` to trade.\n\n"
            f"*(Real run: {RR['names']} names, {RR['ev']} events, {RR['span']}, fp `{RR['fp']}` — "
            "[`../docs/results.md`](../docs/results.md). The executed cells run the synthetic control.)*"
        ),

        md(
            "## Beat 1 · The claim, precisely\n\n"
            "For each trading day $t$, go long every name whose next scheduled earnings announcement falls "
            "within $[1, H]$ trading days (strictly causal: the calendar is public). Equal-weight across "
            "all in-book names; weight = 0 outside the window. The position is entered one day after the "
            "window opens (lag-one causal execution). Claim: the pre-event window carries a positive return "
            "premium above the market. Null: the same window carries no excess return."
        ),
        code(
            "for label, (pp, ee) in [('drift control', (panel, events)), ('null', (panel0, events0))]:\n"
            "    curve = strategy.mean_pre_event_return(pp, ee, pre_days=5)\n"
            "    print(f'{label:14} mean pre-event return: {curve.mean()*100:+.4f}%/day')\n"
            "    print('  ' + '  '.join(f'day{k}:{v*100:+.4f}%' for k, v in curve.items()))\n"
            "    print()"
        ),
        md(
            f"> 💡 **In plain words.** On the control the planted drift makes pre-event returns slightly "
            "higher than on the null. The *real tape* pre-event mean returns are positive (+0.04–0.12%/day) "
            "but roughly equal to the market's daily drift — the machinery sees the planted effect but the "
            "real market offers no excess premium above passive."
        ),

        md(
            "## Beat 2 · So what?\n\n"
            "If the pre-event window genuinely earns ABOVE the market (alpha), a long-only book would "
            "show a higher Sharpe than the passive equal-weight. On the real tape: book Sharpe "
            f"{RR['gross_sh']} vs market Sharpe {RR['mkt_sh']}. The book earns LESS than the market "
            "because it only holds names in the pre-event window (spending most of the time in cash). "
            "There is no alpha, only a partial-period market exposure."
        ),

        md(
            "## Beat 3 · Protocol\n\n"
            "1. **Real?** `book_returns(cost_bps=0)` gross + Newey-West *t* (`strategy.newey_west_t`) on "
            "the real tape; the per-day curve below proves the machine sees a planted drift.\n"
            "2. **Alpha vs beta?** Compare the book Sharpe to the passive equal-weight market Sharpe.\n"
            "3. **Drift shape?** `strategy.mean_pre_event_return` — does the per-day return RISE toward "
            "day −1 (the informed-accumulation prediction)?\n"
            "4. **Tradable?** `strategy.turnover`, `strategy.breakeven_cost_bps`, `strategy.pre_day_sweep`.\n\n"
            f"**Mirage line:** a gross Sharpe below the passive market ({RR['mkt_sh']}) OR a Newey-West "
            f"*t* below 2. Real tape lands at {RR['gross_sh']} / *t* {RR['nw_t']} — both fail."
        ),

        md("## Beat 4 · The teardown\n\n"
           f"**Real tape:** gross Sharpe **{RR['gross_sh']}**, Newey-West *t* **{RR['nw_t']}** (plain "
           f"{RR['plain_t']}), net @5 bp **+{RR['net5']}**, market Sharpe **{RR['mkt_sh']}** — see "
           "[`../docs/results.md`](../docs/results.md). The cells below run the synthetic control.\n\n"
           "### 4a · Per-day pre-event return: planted control vs null"),
        code(
            "print('Synthetic pre-event mean returns by day:')\n"
            "for label, (pp, ee) in [('control', (panel, events)), ('null', (panel0, events0))]:\n"
            "    curve = strategy.mean_pre_event_return(pp, ee, pre_days=5)\n"
            "    print(f'  {label}: ' + '  '.join(f'day{k}:{v*100:+.4f}%' for k, v in curve.items()))\n"
            f"print('REAL tape: day-5 {RR['r5']}  day-4 {RR['r4']}  day-3 {RR['r3']}  day-2 {RR['r2']}  day-1 {RR['r1']}')\n"
            "print('No monotone rise to day-1 -- MIXED. Market daily drift ~+0.06%/day explains positivity.')"
        ),
        code(
            "print('Synthetic book Sharpe (control vs null):')\n"
            "g_c = strategy.summary(strategy.book_returns(panel, events, cost_bps=0.0))\n"
            "g_n = strategy.summary(strategy.book_returns(panel0, events0, cost_bps=0.0))\n"
            "print(f'  control gross Sharpe {g_c[\"sharpe\"]:.3f}  null gross Sharpe {g_n[\"sharpe\"]:.3f}')\n"
            f"print('  (Both ~equal: the equal-weight book is beta-dominated. Real tape: gross {RR['gross_sh']} vs market {RR['mkt_sh']})')"
        ),
        md(
            "> 💡 **In plain words.** The per-day curve sees the planted drift (control > null by "
            "+0.4 bp/day), but the book Sharpes are identical because the market-beta component swamps "
            "the small planted effect. On the *real tape* the book Sharpe is only 0.46 vs the market's "
            "0.97 — the book is just buying equities part-time, not harvesting a pre-earnings premium."
        ),

        md("### 4b · The cost wall and break-even"),
        code(
            "print('Pre-day window sweep (synthetic control):')\n"
            "sweep = strategy.pre_day_sweep(panel, events, windows=[1, 2, 3, 5, 10], cost_bps=2.0)\n"
            "print(sweep.round(3).to_string())\n"
            f"print('REAL tape window sweep (gross Sharpe / net@2bp):')\n"
            f"print('  1d: {RR['sw1g']}/{RR['sw1n']}  2d: {RR['sw2g']}/{RR['sw2n']}  3d: {RR['sw3g']}/{RR['sw3n']}  5d: {RR['sw5g']}/{RR['sw5n']}  10d: {RR['sw10g']}/{RR['sw10n']}')\n"
            "print('Sharpe RISES with window length -- just more market beta, not more alpha.')\n"
            "to = strategy.turnover(panel, events)\n"
            "be = strategy.breakeven_cost_bps(panel, events)\n"
            f"print(f'synthetic turn/day {{to:.3f}}, break-even {{be:.1f}} bp  (real: {RR['turn']}/day, {RR['be']} bp)')"
        ),

        md("## Beat 5 · The verdict\n\n"
           f"- **`WEAK`** (4a, 4b): gross Sharpe {RR['gross_sh']}, *t* {RR['nw_t']} — below both the |t|≥2 "
           f"bar and the passive market ({RR['mkt_sh']}). Zero alpha.\n"
           f"- **`MIRAGE`** (4b): turnover {RR['turn']}/day, break-even {RR['be']} bp — but there is no "
           f"gross alpha to protect; net @5 bp Sharpe +{RR['net5']} vs market +{RR['mkt_sh']}.\n"
           "- **Pre-event drift exists? `MIXED`** (4a): positive per-day raw returns but no monotone "
           "rise toward day −1 and no excess over the market drift.\n\n"
           "> **Signal `WEAK` · Tradability `MIRAGE` · Pre-event drift exists? `MIXED`.**"),

        md("## Beat 6 · Could you trade it?\n\n"
           "- **The right test is market-neutral.** A long-only pre-event book vs a long-only passive "
           "market misattributes beta as alpha. The correct comparison is a market-neutral book: long "
           "pre-event names, short a matched non-event portfolio. On large-cap liquid names, the matched "
           "short would likely earn the same return, zeroing the alpha.\n"
           "- **Arbitraged to zero.** The pre-earnings runup on liquid names has been documented, "
           "published, and traded since the 1990s (Lakonishok & Vermaelen 1990). On the S&P-scale names "
           "in this universe, any information advantage has been fully reflected in options IVs and the "
           "public calendar.\n"
           "- **Window length is a beta dial.** The pre-day sweep shows Sharpe rising monotonically from "
           "1d to 10d — but longer windows just mean more time in the market. If there were genuine "
           "pre-event alpha it would peak near the announcement, not increase indefinitely with the hold."),

        md(
            "## Beat 7 · Going further\n\n"
            "### 7a · The honest verdict & forks\n"
            "The real measurement is **done and fingerprinted** in [`../docs/results.md`](../docs/results.md): "
            f"gross Sharpe {RR['gross_sh']}, Newey-West *t* {RR['nw_t']}, market Sharpe {RR['mkt_sh']}, "
            f"turnover {RR['turn']}/day, break-even {RR['be']} bp, as-of {RR['asof']}, fp `{RR['fp']}`. "
            "Signal `WEAK` / Tradability `MIRAGE` — and the pre-event drift itself is `MIXED` because "
            "positive raw returns are contaminated by positive market beta.\n\n"
            "### 7b · Other forks\n"
            "- **Market-neutral test** — long pre-event names short a beta-matched non-event basket; this "
            "is the correct alpha isolation test.\n"
            "- **Contrast with PEAD (study 34)** — the *post*-announcement drift earns gross Sharpe +0.30 "
            "(*t* +1.39), also `WEAK` but at least a slow, sustained drift tied to an information signal "
            "(the earnings *surprise*), not just event proximity.\n"
            "- **Small-cap version** — the premium may be larger in less-covered, illiquid names, but at "
            "much higher trading costs.\n\n"
            "PRs welcome — try the market-neutral version or a small-cap split."
        ),
    ]
    return new_notebook(cells=cells, metadata=_meta())


def _meta():
    return {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}}


def main():
    nbf.write(build_curious(), os.path.join(HERE, "01_for_the_curious.ipynb"))
    nbf.write(build_quants(), os.path.join(HERE, "02_for_the_quants.ipynb"))
    print("wrote 01_for_the_curious.ipynb and 02_for_the_quants.ipynb")


if __name__ == "__main__":
    main()
