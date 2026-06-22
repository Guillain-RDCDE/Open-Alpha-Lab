"""Generate the two narrative notebooks for Study 352 (Opening-Range Breakout).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. The real-tape cells read the cached yfinance
5-minute bars under ``../_cache/`` (folded into regular sessions) and otherwise quote the frozen
headline numbers in ``R`` (a mirror of docs/results.md). The synthetic positive control runs
anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance QQQ 5m, 60 regular
# sessions through 2026-06-18; as-of 2026-06-22). Reproduced by examples/verify.py.
R = dict(
    sym="QQQ", days=60, bars=78, bh_bps=15.5, up_days=62,
    # ORB vs null (gross), [orb_bps, null_bps, edge_bps, t0, tpair, hit, permp]
    orb5_matched=[-1.3, 10.1, -11.3, -0.10, -1.32, 48, 0.992],
    orb5_coin=[-1.3, -0.1, -1.2, -0.10, -0.80, 48, 0.552],
    orb5_drift=[-1.3, 8.1, -9.4, -0.10, -0.86, 48, 0.976],
    orb15_matched=[-2.5, 9.7, -12.2, -0.20, -1.78, 45, 0.996],
    orb15_coin=[-2.5, -0.1, -2.3, -0.20, -1.18, 45, 0.601],
    orb15_drift=[-2.5, 7.9, -10.3, -0.20, -0.90, 45, 0.986],
    # cost sweep ORB15 matched: cost -> (orb_net, edge, t0)
    cost=[(0, -2.5, -12.2, -0.20), (1, -4.5, -14.2, -0.36), (2, -6.5, -16.2, -0.52),
          (3, -8.5, -18.2, -0.68), (5, -12.5, -22.2, -0.99)],
    breakeven=-6.1,
    # synthetic control ORB15 matched: persistence -> (orb, null, edge, tpair, permp)
    syn=[(0.00, 4.1, 2.2, 2.0, 1.72, 0.088), (0.25, 8.3, 0.9, 7.4, 3.00, 0.000),
         (0.40, 11.8, -0.8, 12.6, 3.64, 0.000)],
    spy=[59, 1.5, -6.5, 0.18, -2.19],   # [trades, orb, edge, t0, tpair]
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Beats_a_coin%3F: Busted](https://img.shields.io/badge/Beats_a_coin%3F-Busted-8b949e?style=flat-square)\n\n"
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from opening_range_breakout import data, strategy as st

HAVE_REAL = data.have_real("QQQ")
DAYS = data.build_days("QQQ") if HAVE_REAL else None
print("real QQQ 5m cache present:", HAVE_REAL,
      "| sessions:", (0 if DAYS is None else len(DAYS)))
"""

# The frozen headline dict is embedded into the notebook's first code cell so every downstream
# cell can quote it whether or not the cache is present.
BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Opening-Range Breakout — does the first 5 minutes call the day? 📈\n"
            "### The day-trading-TikTok staple, in plain English\n\n"
            + BADGES +
            "Every day-trading feed sells the same move: *\"Mark the high and low of the first **5 "
            "minutes**. When price breaks **above** that range, go **long**. When it breaks **below**, "
            "go **short**. Hold to the close. The morning range tells you the day's direction.\"* It "
            "*feels* obviously right — the market 'chose a side', so ride it.\n\n"
            "There's even real research behind a careful version (Zarattini & Grewal, 2023). So we "
            "tested it honestly. The punchline: on a real intraday tape the breakout doesn't beat a "
            "**coin** — a random entry that holds the same amount of time actually does **better**. "
            "This notebook shows why.\n\n"
            "> 📓 **Plain-language layer.** Want the paired *t*-stats, the three random nulls, the cost "
            "sweep and the planted-signal control? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** Research & education. Every chart below is drawn by the "
            "code beside it; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does breaking the morning range predict the day? | **No.** Across 60 real days the "
            f"breakout trade made about **nothing** ({R['orb15_matched'][0]:+.1f} bps a day) — "
            "statistically a flat line. |\n"
            "| But my backtest shows it makes money! | **That's the market drifting up, not the "
            "breakout.** In the same window the index rose most days; *any* mostly-long rule looks "
            "good. |\n"
            "| So is it better than just… guessing? | **It's worse.** A **random** entry that holds "
            f"the same amount of time made **{R['orb15_matched'][1]:+.1f} bps** a day — it *beat* the "
            "breakout ~99% of the time. |\n"
            "| Do trading costs change the verdict? | **They only make it worse.** The rule is already "
            "behind a coin *before* costs, so there's no spread small enough to save it. |\n\n"
            "> The morning range feels like information. On a real tape, it isn't — you'd have done "
            "better throwing a dart at the clock."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"The first 5 minutes set the day's battle lines. Mark that high and low. The first "
            "**break above** the high → go long. The first **break below** the low → go short. Hold "
            "to the close and let the trend pay you.\"*\n\n"
            "The serious version is a real paper — Zarattini & Grewal (2023), *Can Day Trading Really "
            "Be Profitable?* — a 5-minute ORB on QQQ with leverage and stops. We steelman the bare "
            "idea: **the first break of the opening range carries the day's direction.** Either that's "
            "true (and breaking the range beats entering at random), or the apparent profit is just "
            "the market's drift in disguise."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If true, it would be a free daily edge anyone with a chart could harvest — and it would "
            "say something deep: that the opening auction *leaks* the day's direction. But there's a "
            "trap baked into how people test it. In a rising market, a rule that's **in the market "
            "most of the day** will print profits no matter *when* it enters. The only honest way to "
            "ask 'does the **range break** matter?' is to compare it to entering at a **random time** "
            "with the **same exposure**. If the breakout has real information, it must beat that coin."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **{R['days']} real trading days** of {R['sym']} in 5-minute bars (that's all the "
            "free intraday history Yahoo gives — a short, mostly-up sample, and we say so). For each "
            "day:\n\n"
            "1. **Run the rule.** Mark the first-5-minute (and first-15-minute) high/low; enter the "
            "bar *after* the first break, in the break's direction; hold to the close.\n"
            "2. **Run the coin.** On the *same* day, enter at a **random** minute and hold to the same "
            "close — same amount of time in the market. Do it thousands of times.\n"
            "3. **Compare.** If breaking the range is real, the rule beats the coin. If it doesn't, "
            "the 'edge' was the day's drift all along.\n\n"
            "**What would make us say 'mirage':** the breakout failing to beat a same-exposure random "
            "entry. (Spoiler: it loses to it.)"
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First: what does the breakout make vs a random entry of the same exposure?** Bars are "
            "average return per day, in basis points (1 bp = 0.01%)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    r15 = st.evaluate(DAYS, opening_range_min=15, cost_bps=0, null_mode='matched_dir', n_iter=1500)\n"
            "    orb, nul = r15['orb_mean_bps'], r15['null_mean_bps']\n"
            "else:\n"
            "    orb, nul = R['orb15_matched'][0], R['orb15_matched'][1]\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.4))\n"
            "ax.bar(['ORB breakout\\n(the rule)', 'random entry\\n(same exposure)'], [orb, nul],\n"
            "       color=[RED, GREEN], width=.55)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('avg return per day (bps)')\n"
            "ax.set_title('The coin wins: random entry beats the opening-range breakout')\n"
            "for i,v in enumerate([orb,nul]): ax.annotate(f'{v:+.1f}',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'ORB breakout: {orb:+.1f} bps/day   random same-exposure entry: {nul:+.1f} bps/day')"
        ),
        md(
            f"There it is. The breakout makes **{R['orb15_matched'][0]:+.1f} bps** a day — basically "
            f"zero — while a **random** entry holding the same hours makes **{R['orb15_matched'][1]:+.1f} "
            "bps**. The dart *beats the rule*. Whatever profit a naive backtest showed was the market "
            "drifting up underneath both of them; only the random entry actually pocketed that drift, "
            "because the breakout often enters late and on the wrong side."
        ),
        md(
            "**Is it just bad luck in this sample?** No — the random entry beats the breakout on "
            "**~99%** of resamples, and it holds at both the 5-minute and 15-minute range, on a "
            "second ticker (SPY), and across every cost level. Here's the breakout's profit as we add "
            "realistic trading costs — it only sinks."
        ),
        code(
            "costs = [c[0] for c in R['cost']]\n"
            "if HAVE_REAL:\n"
            "    nets = [st.evaluate(DAYS, 15, cost_bps=c, null_mode='matched_dir', n_iter=400)['orb_mean_bps'] for c in costs]\n"
            "else:\n"
            "    nets = [c[1] for c in R['cost']]\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "ax.plot(costs, nets, 'o-', c=RED, lw=2)\n"
            "ax.axhline(0, ls='--', c=GREY, label='break-even')\n"
            "ax.set_xlabel('trading cost per side (bps)'); ax.set_ylabel('ORB net return (bps/day)')\n"
            "ax.set_title('Costs only dig the hole deeper'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('ORB net by cost (bps/day):', [f'{n:+.1f}' for n in nets])"
        ),
        md(
            "The line starts below zero and walks downhill. Because the rule is already behind a coin "
            "*before* costs, there is no spread small enough to rescue it — the break-even cost is "
            f"**negative** (~{R['breakeven']:+.1f} bps). It was never a question of trimming fees."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** The breakout's daily return is a flat line ({R['orb15_matched'][0]:+.1f} "
            f"bps, t ≈ 0) and it **loses** to a same-exposure random entry by ~{abs(R['orb15_matched'][2]):.0f} "
            "bps/day. Breaking the morning range told us nothing the clock didn't.\n"
            "- **Tradability — Mirage.** No cost level makes it win — it's behind a coin gross. Net of "
            "any realistic spread, it's a small, steady loser.\n"
            "- **Beats a coin? — Busted.** The backtest profit was the market's drift; a random entry "
            "captured *more* of it. The 'edge' is an exposure illusion."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — proving the test isn't broken\n\n"
            "Fair worry: maybe our measuring stick just *can't* find an opening-range edge. So we built "
            "a fake market where the breakout **must** work — prices that genuinely keep going the way "
            "they broke — and checked the test finds it. Then a fake market that's pure coin-flips, "
            "where it must find **nothing**."
        ),
        code(
            "pers = [s[0] for s in R['syn']]\n"
            "if HAVE_REAL or True:  # synthetic control runs anywhere\n"
            "    edges = []\n"
            "    for p in pers:\n"
            "        S = data.synthetic_days(n_days=800, persistence=p, seed=352)\n"
            "        edges.append(st.evaluate(S, 15, cost_bps=0, null_mode='matched_dir', n_iter=800)['edge_bps'])\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "ax.bar([f'{p:.2f}' for p in pers], edges, color=[GREY, GREEN, GREEN], width=.55)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_xlabel('how strongly the fake market trends')\n"
            "ax.set_ylabel('ORB edge over random (bps/day)')\n"
            "ax.set_title('The test works: plant a real breakout edge and it finds it')\n"
            "for i,v in enumerate(edges): ax.annotate(f'{v:+.1f}',(i,v),ha='center',va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('planted trend ->', [f'{p:.2f}' for p in pers], '  edge found ->', [f'{e:+.1f}' for e in edges])"
        ),
        md(
            "When the fake market really does trend, the breakout beats the coin by a clear, growing "
            "margin — the test catches it. When the fake market is pure noise, the edge is ~0. So the "
            "**flat real-tape result is the market's verdict, not a broken ruler.** The opening range "
            "simply doesn't carry the day on real data."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The serious steelman.** Zarattini & Grewal (2023) report a profitable ORB over *many "
            "years* with **leverage and ATR stops**. Our 60-day, no-stop, no-leverage test can't "
            "disprove that — but it shows the *bare* rule adds nothing over a coin here. Fork it and "
            "add the stop/target: does the edge appear, or is it the leverage?\n"
            "- **Longer history.** Yahoo caps 5-minute data at ~60 days. Wire in a deeper intraday "
            "source and re-run across bull *and* bear regimes — the drift confound flips in a "
            "down-market.\n"
            "- **The right cousin.** [Study 351](../../351-btc-5m-polymarket-momentum/) found a *real* "
            "intraday momentum signal that was simply priced in. Here the signal itself is missing — "
            "a cleaner 'no'.\n\n"
            "*Think the 5-minute range has an edge a random entry can't match? Add the stop/target the "
            "pros use and show it beats the coin — the harness and the null are right here.*"
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
            "# Opening-Range Breakout — a quantitative teardown 🔬\n"
            "### ORB vs a same-exposure random null (paired *t* + permutation *p*) · three null "
            "modes · a one-way cost sweep · a planted-persistence positive control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The "
            "decisive object is **not** the ORB return — in a bull tape any net-long rule prints "
            "positive — but the **ORB-minus-random** spread at matched exposure. A directional rule "
            "must beat its own exposure drawn at random to earn `REAL`; this one doesn't.\n\n"
            "> ⚠️ **Not investment advice.** Real data: yfinance QQQ/SPY 5-minute bars, ~60 regular "
            "sessions to 2026-06-18 (Yahoo's 5m cap — a short, bull-only sample, named on every axis). "
            "The synthetic positive control is deterministic and offline. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | ORB15m daily return {R['orb15_matched'][0]:+.1f} bps "
            f"(t vs 0 = {R['orb15_matched'][3]:+.2f}); paired (ORB − matched-random) t = "
            f"**{R['orb15_matched'][4]:+.2f}**, permutation p = **{R['orb15_matched'][6]:.3f}** — the "
            "random entry beats ORB almost surely. SPY agrees. |\n"
            f"| **Tradability** | `MIRAGE` | Break-even one-way cost vs the null = "
            f"**{R['breakeven']:+.1f} bps** (negative): no spread makes ORB win — it's behind a coin "
            "gross. |\n"
            f"| **Beats a coin?** | `BUSTED` | Random same-exposure entry = **{R['orb15_matched'][1]:+.1f} "
            f"bps/day** vs ORB **{R['orb15_matched'][0]:+.1f}**. The backtest profit is drift, not the "
            "range. |\n\n"
            "> 💡 In plain words: strip the day's drift out with a matched-exposure random control and "
            "the opening-range break has *negative* marginal information on this tape. The engine would "
            "catch a real edge (positive control below); the market just doesn't offer one here."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let a session have bars $b = 0,\\dots,77$ (78 five-minute bars, 09:30–16:00 ET). The "
            "opening range over the first $R$ minutes is "
            "$[\\,\\underline{r},\\overline{r}\\,] = [\\min_{b<k} L_b,\\ \\max_{b<k} H_b\\,]$ with "
            "$k = R/5$ bars. The first break is the first $b\\ge k$ with $C_b>\\overline{r}$ (long) or "
            "$C_b<\\underline{r}$ (short); we enter at the **next** bar's open $O_{b+1}$ (one lag) and "
            "exit at the close $C_{77}$. Day return $x_d = s_d\\,(C_{77}/O_{b+1}-1)$, $s_d=\\pm1$.\n\n"
            "- **H₁ (the break carries direction).** $\\mathbb{E}[x_d^{\\text{ORB}} - x_d^{\\text{rand}}] > 0$ "
            "against a same-exposure random entry $x_d^{\\text{rand}}$.\n"
            "- **H₂ (it's tradable).** That spread survives one-way costs $c$: break-even $c^\\* > $ a "
            "realistic spread.\n\n"
            "We find **H₁ rejected** (the spread is *negative*) and therefore **H₂ vacuously rejected** "
            "($c^\\*<0$). The steelman (Zarattini–Grewal 2023, leveraged + ATR-stopped, multi-year) is "
            "out of this short sample's reach — but the *bare* rule fails its own null."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — why the null is the whole game\n\n"
            "In a sample with mean daily drift $\\mu>0$, **any** rule that is net-long a fraction "
            "$f$ of the session earns $\\approx f\\mu$ regardless of *when* it enters. So a positive "
            "ORB return is **not** evidence of a signal — it's evidence of exposure. The only "
            "identified quantity is the **difference** at matched exposure:\n\n"
            "$$\\Delta = \\mathbb{E}[x^{\\text{ORB}}_d - x^{\\text{rand}}_d],\\qquad "
            "x^{\\text{rand}}\\ \\text{holds the same hours, drawn uniformly.}$$\n\n"
            "Three nulls bracket the claim: **matched_dir** (same side, random time — isolates "
            "*timing*), **coin** (random side + time — isolates timing **and** sign), **drift** "
            "(always long — the pure exposure benchmark). A real breakout beats all three; an "
            "exposure illusion beats none."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Real tape.** yfinance {R['sym']} (and SPY) 5-minute bars, {R['days']} complete "
            "regular sessions to 2026-06-18 (Yahoo's ~60-day 5m cap; bull-only — named on every axis), "
            "folded into $(\\text{days}, 78, 4)$ OHLC arrays.\n"
            "- **Signal test.** Paired day-by-day $(x^{\\text{ORB}}-x^{\\text{rand}})$ t-stat, plus a "
            "permutation p of the ORB mean against the distribution of $n{=}2000$ random-entry means. "
            "(A directional accuracy *vs its own exposure*, not a raw return.)\n"
            "- **Pre-registered knobs.** $R\\in\\{5,15\\}$, hold-to-close, **no** stop/target tuning, "
            "one execution lag. No search over the headline.\n"
            "- **Costs.** One-way bps on NAV, twice per round trip; report the break-even one-way cost.\n"
            "- **Positive control.** A deterministic AR(1)-per-bar day generator plants a known "
            "intraday persistence: $\\Delta$ must be $>0$ and significant as persistence rises, and "
            "$\\approx 0$ on a random walk. Machinery proof, never market evidence.\n"
            "- **What says 'mirage':** $\\Delta \\le 0$ on the real tape. (It is.)"
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · ORB vs three same-exposure nulls — the spread is negative\n\n"
            "ORB daily mean (red) against each random null (green), for the 5- and 15-minute range. "
            "The honest controls are **matched_dir** and **drift**; under both, the random entry wins."
        ),
        code(
            "variants = [('ORB 5m','orb5'), ('ORB 15m','orb15')]\n"
            "modes = ['matched_dir','coin','drift']\n"
            "rows = {}\n"
            "for label, key in variants:\n"
            "    orb_min = 5 if '5m' in label else 15\n"
            "    for m in modes:\n"
            "        if HAVE_REAL:\n"
            "            r = st.evaluate(DAYS, opening_range_min=orb_min, cost_bps=0, null_mode=m, n_iter=1200)\n"
            "            rows[(label,m)] = (r['orb_mean_bps'], r['null_mean_bps'])\n"
            "        else:\n"
            "            v = R[f'{key}_{m}']; rows[(label,m)] = (v[0], v[1])\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.4))\n"
            "x = np.arange(len(modes)); w=.36\n"
            "ax.bar(x-w/2, [rows[('ORB 15m',m)][0] for m in modes], w, color=RED, label='ORB 15m return')\n"
            "ax.bar(x+w/2, [rows[('ORB 15m',m)][1] for m in modes], w, color=GREEN, label='random null')\n"
            "ax.axhline(0,c='k',lw=1); ax.set_xticks(x); ax.set_xticklabels(modes)\n"
            "ax.set_ylabel('avg return (bps/day)'); ax.set_title('ORB 15m vs its three nulls: the coin wins where it counts'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for m in modes: print(f\"15m {m:11s}: ORB {rows[('ORB 15m',m)][0]:+5.1f}  null {rows[('ORB 15m',m)][1]:+5.1f}  edge {rows[('ORB 15m',m)][0]-rows[('ORB 15m',m)][1]:+5.1f}\")"
        ),
        md(
            f"> 💡 In plain words: against **matched_dir** (same side, random time) ORB runs "
            f"{R['orb15_matched'][2]:+.1f} bps/day — a *random clock* beats the breakout by ~12 bps. "
            "Against **coin** the gap shrinks (both lose the sign edge), and against **drift** the "
            "always-long benchmark crushes ORB by ~10 bps — the rule's profit was the exposure, full "
            "stop. H₁ rejected: the break has **negative** marginal information here."
        ),
        md(
            "### 4b · Significance — paired t and the permutation null\n\n"
            "The paired (ORB − matched-random) t-stat and the ORB return's own t vs zero, for both "
            "range lengths and on SPY. Nothing clears 2; the paired t is *negative*."
        ),
        code(
            "labels, t0s, tps = [], [], []\n"
            "for orb_min, key in [(5,'orb5_matched'),(15,'orb15_matched')]:\n"
            "    if HAVE_REAL:\n"
            "        r = st.evaluate(DAYS, opening_range_min=orb_min, cost_bps=0, null_mode='matched_dir', n_iter=1500)\n"
            "        t0s.append(r['t_vs_zero']); tps.append(r['t_paired'])\n"
            "    else:\n"
            "        t0s.append(R[key][3]); tps.append(R[key][4])\n"
            "    labels.append(f'QQQ {orb_min}m')\n"
            "if HAVE_REAL:\n"
            "    rspy = st.evaluate(data.build_days('SPY'), 15, cost_bps=0, null_mode='matched_dir', n_iter=1500)\n"
            "    t0s.append(rspy['t_vs_zero']); tps.append(rspy['t_paired'])\n"
            "else:\n"
            "    t0s.append(R['spy'][3]); tps.append(R['spy'][4])\n"
            "labels.append('SPY 15m')\n"
            "x=np.arange(len(labels)); w=.36; fig,ax=plt.subplots(figsize=(9.0,4.3))\n"
            "ax.bar(x-w/2, t0s, w, color=GREY, label='ORB return, t vs 0')\n"
            "ax.bar(x+w/2, tps, w, color=RED, label='paired (ORB - random), t')\n"
            "ax.axhline(2, ls='--', c=GREEN, label='t = +2 (REAL bar)'); ax.axhline(-2, ls='--', c=GREEN)\n"
            "ax.axhline(0,c='k',lw=1); ax.set_xticks(x); ax.set_xticklabels(labels)\n"
            "ax.set_ylabel('t-statistic'); ax.set_title('No t clears 2; the paired t is negative'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('t vs 0 :', [f'{t:+.2f}' for t in t0s]); print('t paired:', [f'{t:+.2f}' for t in tps])"
        ),
        md(
            f"> 💡 In plain words: the ORB return is statistically a flat line (|t vs 0| ≤ 0.2), and the "
            f"*paired* test is **negative** ({R['orb15_matched'][4]:+.2f} at 15m, {R['spy'][4]:+.2f} on "
            f"SPY) — the random entry doesn't just match ORB, it beats it. Permutation p = "
            f"**{R['orb15_matched'][6]:.3f}**: the random mean exceeds the ORB mean in ~99.6% of "
            "resamples. There is no $R$, ticker, or sign convention here that rescues H₁."
        ),
        md(
            "### 4c · Costs — the break-even is negative\n\n"
            "ORB net daily return as one-way cost rises. Because the gross spread over the null is "
            "already negative, the break-even one-way cost is **below zero** — no real spread can help."
        ),
        code(
            "costs = [c[0] for c in R['cost']]\n"
            "if HAVE_REAL:\n"
            "    nets = [st.evaluate(DAYS, 15, cost_bps=c, null_mode='matched_dir', n_iter=500)['orb_mean_bps'] for c in costs]\n"
            "    be = st.breakeven_cost(DAYS, 15, null_mode='matched_dir')\n"
            "else:\n"
            "    nets = [c[1] for c in R['cost']]; be = R['breakeven']\n"
            "fig, ax = plt.subplots(figsize=(8.8,4.3))\n"
            "ax.plot(costs, nets, 'o-', c=RED, lw=2, label='ORB net return')\n"
            "ax.axhline(0, ls='--', c=GREY); ax.axvline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('one-way cost (bps)'); ax.set_ylabel('ORB net (bps/day)')\n"
            "ax.set_title(f'Break-even one-way cost vs the null = {be:+.1f} bps (negative)'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('net by cost:', list(zip(costs,[round(n,1) for n in nets])), ' break-even:', round(be,1),'bps')"
        ),
        md(
            "> 💡 In plain words: a negative break-even cost is the cleanest possible statement of "
            "'no edge' — you would need to be *paid* the spread for ORB to match a coin. Real "
            "QQQ/SPY spreads are ~0.5–1 bp each way; every one of them pushes the rule further "
            "underwater."
        ),
        md(
            "### 4d · Faithful-engine control — plant a breakout edge, the test finds it\n\n"
            "On a deterministic AR(1)-per-bar tape, persistence $\\rho$ makes a range break genuinely "
            "tend to continue. The measured ORB-minus-random edge must be $\\approx0$ at $\\rho=0$ and "
            "rise, significantly, with $\\rho$:"
        ),
        code(
            "pers = [s[0] for s in R['syn']]\n"
            "edges, tps_s = [], []\n"
            "for p in pers:\n"
            "    S = data.synthetic_days(n_days=800, persistence=p, seed=352)\n"
            "    r = st.evaluate(S, 15, cost_bps=0, null_mode='matched_dir', n_iter=1000)\n"
            "    edges.append(r['edge_bps']); tps_s.append(r['t_paired'])\n"
            "fig, ax = plt.subplots(figsize=(8.8,4.3))\n"
            "x=np.arange(len(pers))\n"
            "ax.bar(x, edges, .5, color=[GREY,GREEN,GREEN], label='ORB edge over random (bps)')\n"
            "ax2=ax.twinx(); ax2.plot(x, tps_s, 'D-', c=RED, label='paired t'); ax2.axhline(2, ls='--', c=RED, alpha=.5)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'rho={p:.2f}' for p in pers])\n"
            "ax.set_ylabel('edge (bps/day)'); ax2.set_ylabel('paired t', color=RED)\n"
            "ax.set_title('Positive control: the harness recovers a planted ORB edge'); ax.axhline(0,c='k',lw=1)\n"
            "ax.legend(loc='upper left'); ax2.legend(loc='lower right')\n"
            "plt.tight_layout(); plt.show()\n"
            "for p,e,t in zip(pers,edges,tps_s): print(f'rho={p:.2f}: edge {e:+6.1f} bps  paired t {t:+.2f}')"
        ),
        md(
            "> 💡 In plain words: plant a real opening-range edge and the test banks it — edge climbs "
            f"from ~0 at $\\rho{{=}}0$ to {R['syn'][2][3]:+.1f} bps at $\\rho{{=}}0.40$ with paired "
            f"t = {R['syn'][2][4]:+.2f} (p≈0). The harness can detect the effect it's accused of "
            "missing; the real tape simply has none. (This is a **machinery** proof — a synthetic "
            "Sharpe is never market evidence.)"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — ORB15m {R['orb15_matched'][0]:+.1f} bps (t vs 0 = "
            f"{R['orb15_matched'][3]:+.2f}); paired vs matched-random t = **{R['orb15_matched'][4]:+.2f}**, "
            f"permutation p = **{R['orb15_matched'][6]:.3f}**; SPY paired t = {R['spy'][4]:+.2f}. The "
            "break has negative marginal information; H₁ rejected.\n"
            f"- **Tradability `MIRAGE`** — break-even one-way cost = **{R['breakeven']:+.1f} bps** "
            "(negative). Behind a coin gross, a loser net of any spread.\n"
            "- **Beats a coin? `BUSTED`** — random same-exposure entry "
            f"{R['orb15_matched'][1]:+.1f} bps vs ORB {R['orb15_matched'][0]:+.1f}. The 'edge' was "
            "exposure to a bull drift, captured *better* by a dart than by the rule."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the honest scope\n\n"
            "The bare rule is a non-starter: negative gross spread, negative break-even cost. The only "
            "live question is whether the *engineered* version — leverage + ATR stop/target, the "
            "Zarattini–Grewal recipe — adds something our 60-day, no-stop test can't see. We can frame "
            "the bound: even granting ORB the random null's exposure return, the **stop/target** would "
            "have to manufacture the entire edge, since the *entry* contributes nothing positive here."
        ),
        code(
            "# What the rule would need: the gap ORB must close just to MATCH a coin, by range length\n"
            "gaps = {}\n"
            "for orb_min, key in [(5,'orb5_matched'),(15,'orb15_matched')]:\n"
            "    if HAVE_REAL:\n"
            "        r = st.evaluate(DAYS, orb_min, cost_bps=0, null_mode='matched_dir', n_iter=1200)\n"
            "        gaps[f'{orb_min}m'] = -r['edge_bps']\n"
            "    else:\n"
            "        gaps[f'{orb_min}m'] = -R[key][2]\n"
            "fig, ax = plt.subplots(figsize=(7.8,4.0))\n"
            "ax.bar(list(gaps), list(gaps.values()), color=RED, width=.5)\n"
            "ax.set_ylabel('bps/day a stop/target must ADD'); ax.axhline(0,c='k',lw=1)\n"
            "ax.set_title('Deficit ORB must overcome just to tie a coin')\n"
            "for i,(k,v) in enumerate(gaps.items()): ax.annotate(f'{v:+.1f}',(i,v),ha='center',va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('bps the overlay must add to break even vs random:', {k:round(v,1) for k,v in gaps.items()})"
        ),
        md(
            "> 💡 In plain words: a stop/target/leverage overlay would have to add ~10–12 bps/day "
            "*net of its own costs* just to drag the entry up to a coin — before any actual edge. "
            "That's a tall order for risk management alone, and it's exactly where the burden of proof "
            "sits for the engineered ORB. On the *bare* rule the answer to 'could you trade it?' is a "
            "clean **no**."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Re-run the steelman in full.** Add ATR stops, a target multiple, and leverage "
            "(Zarattini–Grewal 2023) and re-measure the paired spread — is the alpha the *range* or "
            "the *overlay*? The harness and null are parameterised for it.\n"
            "- **Regime split.** Yahoo's 60-day cap forces a bull-only window; a deeper intraday feed "
            "lets you test the drift confound in a down-market, where the always-long null inverts.\n"
            "- **Intraday-momentum done right.** Gao et al. (2018) first-30m→last-30m is a *different, "
            "real* object; contrast it with the ORB to see why one survives a null and the other "
            "doesn't.\n\n"
            "*The reproducible core is offline and deterministic; the real tape is yfinance public "
            "data. Methods and sources: [`docs/references.md`](../docs/references.md); frozen numbers: "
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
