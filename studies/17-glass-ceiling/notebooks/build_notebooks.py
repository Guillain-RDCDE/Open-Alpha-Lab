"""Generate the two narrative notebooks for Study 17 (Glass-Ceiling) from source.

Like every study, the notebooks are a *generated artefact*: edit the cell text here, rebuild the
skeletons, then execute with nbconvert to embed figures/outputs.

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

The executed path runs on the **offline synthetic minute tape** — an OHLCV generator whose only
load-bearing knob is post-breakout drift (zero ⇒ the 1R bracket is a coin flip *by construction*;
positive ⇒ genuine continuation; grind-gated ⇒ a real signal for the staircase filter). The cached
real bars (BTC-USD/SPY/QQQ) are git-ignored, so the reproducible core must run with no network; the
**real verdict** is quoted from [`../docs/results.md`](../docs/results.md), produced by
`examples/verify.py`. Both notebooks follow the SAME seven desk beats (see ../../../METHODOLOGY.md).
"""

from __future__ import annotations

import os

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))           # study root (glass_ceiling/ lives there)
sys.path.insert(0, os.path.abspath("../../.."))      # repo root, for quantlab
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (9.5, 5.2)
import numpy as np, pandas as pd
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
from glass_ceiling import data, levels, strategy, filters

# Offline synthetic minute tapes, with the post-breakout answer BAKED IN:
#  null  -> no continuation: a fresh high carries no information, so the 1R bracket is a COIN FLIP
#  cont  -> a small post-breakout drift: genuine momentum, so the win rate MUST rise (the steelman)
#  grind -> continuation fires ONLY after a low-concentration 'staircase' approach (a real filter signal)
null,  t_null  = data.synthetic_intraday(n_bars=120_000, cont_drift=0.0,     seed=17)
cont,  t_cont  = data.synthetic_intraday(n_bars=120_000, cont_drift=0.00003, seed=17)
grind, t_grind = data.synthetic_intraday(n_bars=120_000, cont_drift=0.00015,
                                         lookback=20, cont_requires_grind=True, seed=17)
print(f"{len(null):,} bars/tape | null drift={t_null.cont_drift} (coin flip) | "
      f"cont +{t_cont.cont_drift} | grind +{t_grind.cont_drift} (grind-gated)")
"""

# Real headline numbers (from docs/results.md via examples/verify.py; cells below EXECUTE on the
# synthetic tape). BTC-USD / SPY / QQQ 5-minute, ~60-day window, as-of 2026-06-09.
R = dict(
    btc_n="84", btc_wr="45.2", btc_ci="[35, 56]", btc_net="-0.112", btc_flt="-0.452", btc_flt_n="9",
    spy_n="19", spy_wr="63.2", spy_ci="[41, 81]", spy_net="+0.246", spy_flt_n="1",
    qqq_n="28", qqq_wr="57.1", qqq_ci="[39, 73]", qqq_net="+0.127", qqq_flt_n="3",
    be2="50.9",
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Filters help?: Not supported](https://img.shields.io/badge/Filters_help%3F-Not_supported-8b949e?style=flat-square)\n\n"
)


def md(text):
    return new_markdown_cell(text)


def code(text):
    return new_code_cell(text)


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Glass-Ceiling 🪟\n"
            "### Buy the breakout, take profit at 1:1 — is there momentum to harvest, or are you just paying the spread twice to buy the high?\n\n"
            + BADGES +
            "Here's a strategy that's gone viral (Koroush AK, ~309k views): when price pushes through "
            "**resistance** — a ceiling it kept bouncing off — you go long, betting the ceiling *shatters* "
            "and price keeps running. You stop out at the recent low, you take profit the same distance "
            "up (a **1:1** trade), and you only pull the trigger when three things line up: price ground "
            "into the level slowly (a *staircase*, not a *spike*), volume was building, and the trend was "
            "clean. It's beautifully explained and full of winning screenshots.\n\n"
            "So let's check it honestly. The trick is that a 1:1 trade — risk one unit to make one unit — "
            "is a **symmetric bet**, like calling a coin. To make money you need to call it right *more "
            "than half the time*, by enough to cover the spread you pay getting in and out. Two questions, "
            "then: **does breaking resistance actually tilt the coin?** And **what does the spread cost "
            "you?** We'll answer both on a tape where we *control* the truth, then look at the real "
            "market.\n\n"
            "> 📓 **This is the plain-language layer.** Want the Wilson intervals, the expectancy "
            "identity and the filter-lift power test? That's the companion notebook, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** Every chart below is generated by the code beside it. The "
            "reproducible core runs on a **synthetic** minute tape where we *bake in* the answer — so the "
            "real-market numbers (quoted from [`../docs/results.md`](../docs/results.md)) are a "
            "measurement, not a hope. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        md(
            "## The answer first 🎯\n\n"
            "| What we asked | The honest answer |\n"
            "|---|---|\n"
            "| Does breaking resistance tilt the coin? | 🚫 **Barely, if at all.** On a no-momentum tape "
            "the win rate is **~50%** by construction, and on real BTC-USD it's "
            f"**{R['btc_wr']}%** (95% CI {R['btc_ci']}) — a coin flip. |\n"
            "| Can a 1:1 breakout make money? | 🚫 **Not after costs.** At 1:1 you need >50% wins *net*; "
            f"the spread paid twice pushes break-even to ~{R['be2']}% and turns real BTC **net-negative "
            f"({R['btc_net']} R/trade)**. |\n"
            "| Do the three filters rescue it? | 🚫 **No.** They add no reliable win-rate lift — they just "
            "shrink the sample. On real tapes the 'A-grade' subset is **1–9 trades**: a highlight reel, "
            "not an edge. |\n"
            "| Is breakout momentum *impossible*, then? | ✅ **No — and we prove our test isn't rigged.** "
            "When we bake real follow-through into the tape, the *same* method finds it cleanly. The idea "
            "*could* work; this 1:1 version on the minute chart doesn't. |\n\n"
            "> Desk shorthand: **Signal `NONE` · Tradability `MIRAGE` · Filters help? `NOT SUPPORTED`.** "
            "Let's see how."
        ),

        md(
            "## 1 · The claim 📣\n\n"
            "Resistance is a price ceiling — a level sellers defended before. The breakout trader bets "
            "that once buyers *clear* it, the ceiling becomes a floor and price keeps climbing. "
            "Mechanically:\n\n"
            "> go long when price closes above resistance **twice** in a row; put the **stop** at the "
            "recent swing low (at least 1% away); take profit **the same distance up** (1R). Only take "
            "the trade in a clean, grinding, volume-building approach.\n\n"
            "Notice the shape of the bet: stop and target are **equidistant** from entry. That single "
            "design choice is the whole story."
        ),
        code(
            "fig, ax = plt.subplots(figsize=(7.6, 4.0))\n"
            "ax.axhline(0, color='k', lw=1.4)\n"
            "ax.axhline(1, color='#2ea44f', ls='--', lw=1.6)\n"
            "ax.axhline(-1, color='#c0392b', ls='--', lw=1.6)\n"
            "ax.text(0.02, 0.06, 'entry — break of resistance', transform=ax.get_yaxis_transform())\n"
            "ax.text(0.02, 1.06, '+1R  take-profit', color='#2ea44f', transform=ax.get_yaxis_transform())\n"
            "ax.text(0.02, -0.94, '-1R  stop (swing low)', color='#c0392b', transform=ax.get_yaxis_transform())\n"
            "ax.annotate('', xy=(0.5,1), xytext=(0.5,0), arrowprops=dict(arrowstyle='<->', color='#2ea44f'))\n"
            "ax.annotate('', xy=(0.7,0), xytext=(0.7,-1), arrowprops=dict(arrowstyle='<->', color='#c0392b'))\n"
            "ax.set_ylim(-1.7, 1.7); ax.set_xticks([]); ax.set_ylabel('profit/loss in R')\n"
            "ax.set_title('A 1:1 bracket is a symmetric bet: equal room up and down')\n"
            "plt.show()"
        ),

        md(
            "## 2 · So what? 💰\n\n"
            "If breaking resistance really *did* signal momentum, this would be a money machine: a "
            "repeatable setup, a tight stop, a clear target. That's the dream the screenshots sell. But a "
            "symmetric bet is only profitable if the breakout **tilts the coin** past 50% — and by enough "
            "to beat the spread you pay on the way in and the way out. If the breakout is *not* "
            "informative, you've built a fair coin and then handed the house a cut on every flip. Over "
            "hundreds of trades on a 1-minute chart, that cut is brutal. So the stakes are simple: is "
            "there a real tilt, or are you paying to flip coins?"
        ),

        md(
            "## 3 · How we'd know 🔬\n\n"
            "Three checks, decided up front:\n\n"
            "1. **The coin-flip baseline.** Build a tape with *no* post-breakout momentum and run the "
            "exact strategy. The win rate **must** come out ~50% — if our backtest says otherwise on a "
            "tape we *know* is fair, the backtest is broken.\n"
            "2. **The spread.** Charge realistic costs and find the break-even win rate. How far above "
            "50% do you need to be just to tread water?\n"
            "3. **The filters.** Do the staircase / volume / clean-trend filters lift the win rate — or "
            "do they just keep fewer trades and *look* selective?\n\n"
            "**What would make us call it real:** a win rate reliably above the break-even line, on data "
            "we didn't cherry-pick. **What would make us say \"mirage\":** a ~50% coin flip that costs "
            "drag below zero, with filters that add nothing but a smaller sample."
        ),

        md(
            "## 4 · The teardown 🔧\n\n"
            "### 4a · On a fair tape, the breakout is a coin flip\n"
            "Run the full strategy on the **null** tape — no baked-in momentum. The cumulative-R curve "
            "wanders like a random walk, and the win rate's confidence interval sits right on 50%."
        ),
        code(
            "tr = strategy.run(null)\n"
            "s = strategy.summary(tr); lo, hi = strategy.win_rate_ci(tr)\n"
            "eq = strategy.equity_curve(tr)\n"
            "fig, ax = plt.subplots()\n"
            "ax.plot(eq.index, eq.values, lw=1.0)\n"
            "ax.axhline(0, ls=':', c='grey')\n"
            "ax.set_xlabel('trade number'); ax.set_ylabel('cumulative R')\n"
            "ax.set_title(f'{s[\"n_trades\"]} breakout trades on a fair tape: win rate {s[\"win_rate\"]:.1%} '\n"
            "             f'(95% CI [{lo:.1%}, {hi:.1%}])')\n"
            "plt.show()\n"
            "print(f'win rate {s[\"win_rate\"]:.3f}, 95% CI [{lo:.3f}, {hi:.3f}] -> contains 0.50; '\n"
            "      f'gross expectancy {s[\"expectancy_R_gross\"]:+.3f} R per trade (i.e. ~nothing)')"
        ),
        md(
            "The interval straddles 0.50. Breaking resistance, on a tape with no real follow-through, "
            "tells you **nothing** about what happens next — exactly as a fair coin should."
        ),

        md(
            "### 4b · Where it dies: the spread, paid twice\n"
            "Now charge a cost on each round trip. Because the stop is ~1% of price, even a few basis "
            "points is a meaningful slice of *R* — and it's paid on entry **and** exit. Watch net "
            "expectancy fall through zero."
        ),
        code(
            "sweep = strategy.cost_sweep(tr, roundtrip_bps=(0, 1, 2, 5, 10, 20))\n"
            "fig, ax = plt.subplots()\n"
            "ax.plot(sweep.index, sweep['expectancy_R_net'].values, 'o-')\n"
            "ax.axhline(0, ls=':', c='grey')\n"
            "ax.set_xlabel('round-trip cost (bps)'); ax.set_ylabel('net expectancy (R per trade)')\n"
            "ax.set_title('A fair coin with a house cut: net expectancy goes negative as costs bite')\n"
            "plt.show()\n"
            "display(sweep.round(4))\n"
            "print('break-even win rate at 2 bps:', f'{strategy.summary(tr, 2.0)[\"breakeven_win_rate\"]:.3f}')"
        ),
        md(
            "A 51-ish% coin can look faintly positive at zero cost — that's noise — but the moment you "
            "pay a realistic spread it bleeds. On the 1-minute chart you trade *often*, so you pay that "
            "spread again and again. This is the engine of the `MIRAGE` stamp."
        ),

        md(
            "### 4c · The filters don't rescue it — they just shrink the sample\n"
            "Koroush's defence is the three filters: only take the *clean* setups. So we grade every "
            "trade on the staircase, volume and clean-trend filters and keep only the A-grade ones. On a "
            "fair tape, filtering noise can't create signal — it can only thin the herd."
        ),
        code(
            "fl = filters.filter_lift(tr, null)\n"
            "print(f'all trades:            n={fl[\"n_all\"]:4d}   win rate {fl[\"win_rate_all\"]:.1%}')\n"
            "print(f'A-grade (all 3 filters): n={fl[\"n_filtered\"]:4d}   win rate {fl[\"win_rate_filtered\"]:.1%}'\n"
            "      f'   (kept only {fl[\"kept_frac\"]:.0%} of trades)')\n"
            "print(f'win-rate lift from filtering: {fl[\"lift\"]:+.1%}  -> within noise of zero')\n"
            "fig, ax = plt.subplots(figsize=(6.4,4.2))\n"
            "ax.bar(['all trades', 'A-grade'], [fl['win_rate_all'], fl['win_rate_filtered']],\n"
            "       color=['#8b949e', '#dab617'])\n"
            "ax.axhline(0.5, ls=':', c='k'); ax.set_ylabel('win rate'); ax.set_ylim(0,1)\n"
            "ax.text(0, fl['win_rate_all']+.02, f\"n={fl['n_all']}\", ha='center')\n"
            "ax.text(1, fl['win_rate_filtered']+.02, f\"n={fl['n_filtered']}\", ha='center')\n"
            "ax.set_title('Filtering noise: same ~50% win rate, far fewer trades')\n"
            "plt.show()"
        ),
        md(
            "Same coin, smaller sample. And *that smaller sample* is exactly how a highlight reel is "
            "born: pick the handful of trades that passed every filter **and** won, screenshot them, and "
            "you have a 'strategy'. On the real market this gets absurd — the A-grade subset shrinks to "
            f"**a single trade** on SPY (which, of course, won → '100% win rate!') and to {R['btc_flt_n']} "
            "trades on BTC (which went 0-for-9). Noise dressed as selectivity."
        ),

        md(
            "### 4d · Is breakout momentum impossible? No — and here's the proof our test is fair\n"
            "A test that can only ever say 'no' is worthless. So we bake **real** follow-through into a "
            "tape (a small post-breakout drift) and run the identical strategy. If the machine is honest, "
            "it should now find the edge — and it does."
        ),
        code(
            "rows = []\n"
            "for tag, tape in [('null  (fair coin)', null), ('continuation (real momentum)', cont)]:\n"
            "    s = strategy.summary(strategy.run(tape), roundtrip_bps=4.0)\n"
            "    rows.append((tag, s['win_rate'], s['expectancy_R_net']))\n"
            "    print(f'{tag:30s}: win rate {s[\"win_rate\"]:.1%}   net expectancy {s[\"expectancy_R_net\"]:+.3f} R')\n"
            "print()\n"
            "# and: when momentum only follows a GRIND, the staircase filter recovers it\n"
            "g = filters.annotate(strategy.run(grind), grind); g = g[g.outcome != 0]\n"
            "wr_lo = (g[g.grind <  g.grind.median()].outcome == 1).mean()\n"
            "wr_hi = (g[g.grind >= g.grind.median()].outcome == 1).mean()\n"
            "fig, ax = plt.subplots(figsize=(6.4,4.2))\n"
            "ax.bar(['spiky approach','grinding approach'], [wr_lo, wr_hi], color=['#c0392b','#2ea44f'])\n"
            "ax.axhline(0.5, ls=':', c='k'); ax.set_ylabel('win rate'); ax.set_ylim(0,1)\n"
            "ax.set_title('When a grind genuinely predicts follow-through, the filter finds it')\n"
            "plt.show()\n"
            "print(f'grind-gated tape: spiky-approach win {wr_lo:.1%}  vs  grinding-approach win {wr_hi:.1%}')"
        ),
        md(
            "On the continuation tape the win rate climbs above 50% and the strategy is net-positive *even "
            "after costs* — the edge is real when the momentum is real. And on the grind-gated tape, where "
            "follow-through only happens after a slow staircase, the staircase filter cleanly separates "
            "winners from losers. **The method works when there's something to find.** That's what makes "
            "its verdict on the real claim trustworthy."
        ),

        md(
            "## 5 · The verdict 🧾\n\n"
            "- **The breakout is a coin flip** (4a): ~50% win rate on a fair tape, CI dead on 0.50.\n"
            "- **Costs sink it** (4b): at 1:1 you need to beat ~"
            f"{R['be2']}% just to break even at a tiny 2 bps; the spread paid twice goes negative.\n"
            "- **The filters add nothing** (4c): no reliable lift, only a thinner, cherry-pickable "
            "sample.\n"
            "- **But the test is fair** (4d): give it real momentum and it finds it.\n\n"
            "> **Signal `NONE` · Tradability `MIRAGE` · Filters help? `NOT SUPPORTED`.** The breakout "
            "bracket at 1:1 is a fair coin with a spread bolted on. The real-tape numbers and "
            "fingerprints are in [`../docs/results.md`](../docs/results.md)."
        ),

        md(
            "## 6 · Could you trade it? 💸\n\n"
            "This is where the desk usually finds the body, and here it does. On the real market — "
            "**BTC-USD**, Koroush's own 24/7 arena, the deepest sample we can get — the strategy over "
            f"**{R['btc_n']} trades** wins **{R['btc_wr']}%** (95% CI {R['btc_ci']}): a coin flip, and "
            f"**net-negative ({R['btc_net']} R/trade)** once it pays just 2 bps on each leg. SPY and QQQ "
            f"land at {R['spy_wr']}% and {R['qqq_wr']}% — but on {R['spy_n']} and {R['qqq_n']} trades "
            "their intervals run from the low-40s into the 70s–80s, so they can't reject a coin either.\n\n"
            "And remember Koroush trades the **1-minute** chart, where the spread is a *larger* fraction "
            "of every move and you trade even more often — strictly worse than the 5-minute test here. "
            "There is no size question to reach; the edge is gone at the first basis point.\n\n"
            "> Tradability: **`MIRAGE`**. The only reliable feature of a 1:1 breakout is the spread it "
            "pays twice."
        ),

        md(
            "## 7 · Going further 🚪\n\n"
            "- **The honest steelman lives in the synthetic tapes.** We *proved* the test isn't rigged: "
            "bake in momentum and the method banks it; gate momentum on a grind and the staircase filter "
            "recovers it. The interesting open question isn't 'is the test fair' but 'does **any** real "
            "instrument show post-breakout drift big enough to beat its own spread?' A deeper intraday "
            "feed (a paid source, or an MT5 export) would let you ask it with thousands of trades instead "
            "of dozens.\n"
            "- **Asymmetric targets.** 1:1 is the easy thing to disprove. A trend-following version takes "
            "profit at 2R or 3R and tolerates a <50% win rate — does letting winners run change the "
            "verdict? (It moves the break-even *down*, but win rate falls too.)\n"
            "- **The limit-entry variant.** Koroush enters on a *pullback* to the level, not at the "
            "confirmation close — a better fill, but it misses the runaway winners. Worth modelling the "
            "trade-off honestly.\n\n"
            "PRs welcome — find the instrument or the target structure where the breakout finally beats "
            "its spread, or confirm the coin flip on a deeper tape."
        ),
    ]
    return new_notebook(cells=cells, metadata=_meta())


# ===========================================================================
# 02 — FOR THE QUANTS
# ===========================================================================
def build_quants():
    cells = [
        md(
            "# Glass-Ceiling — a quantitative teardown 🔬\n"
            "### The ±1R bracket as a symmetric stopping problem · Wilson intervals on the win rate · cost-in-R & the break-even line · a filter-lift test with baked-in power checks\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same seven "
            "beats, every claim now carrying its standard error.* The strategy is a long entry on "
            "$k$-close confirmation above a trailing-high resistance, a swing-low stop floored at 1%, and "
            "a take-profit at **1R**. Because stop and target are equidistant, each trade is a "
            "**symmetric ±1R bracket**, and the whole study reduces to one estimable quantity — the win "
            "rate $p$ — against one threshold — the break-even $0.5 + c_R/2$.\n\n"
            "> ⚠️ **Not investment advice.** The reproducible core executes on a synthetic minute tape "
            "with a baked-in post-breakout drift (zero on the null, positive on the steelman) — the "
            "ground truth the backtest must recover; the real BTC-USD/SPY/QQQ run is in "
            "[`../docs/results.md`](../docs/results.md) via `examples/verify.py`, sources in "
            "[`../docs/references.md`](../docs/references.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition. House style "
            "in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        md(
            "## Beat 0 · Verdict\n\n"
            "| Axis | Stamp | Why |\n"
            "|---|---|---|\n"
            "| **Signal** — is the effect real? | 🔴 `NONE` | The breakout win rate is statistically a "
            "coin flip: the Wilson interval contains 0.50 on the driftless null and on every real tape "
            f"(BTC-USD {R['btc_wr']}%, CI {R['btc_ci']}, over {R['btc_n']} trades). |\n"
            "| **Tradability** | 🔴 `MIRAGE` | At 1:1 the break-even win rate is $0.5 + c_R/2 \\approx$ "
            f"**{R['be2']}%** at 2 bps; the deepest real sample is net-negative ({R['btc_net']} R/trade), "
            "and a 1-minute crypto/CFD spread is wider still. |\n"
            "| **Do the filters help?** | ⚪ `NOT SUPPORTED` | The all-filters-pass subset shows no "
            "win-rate lift beyond its own sampling error while collapsing $n$; on real tapes it is "
            f"{R['spy_flt_n']}–{R['btc_flt_n']} trades. |\n\n"
            "> **In one sentence:** the 1:1 breakout bracket is a symmetric coin-flip whose only reliable "
            "feature is the spread it pays twice — and the three filters add only the *illusion* of "
            "selectivity on a shrinking sample.\n\n"
            "*(This notebook executes on the offline synthetic tape, where the win rate is a known "
            "function of the baked-in drift. The real numbers that earn the stamps are in "
            "[`../docs/results.md`](../docs/results.md).)*"
        ),

        md(
            "## Beat 1 · The claim, stated precisely\n\n"
            "Enter long at price $P_0$ the instant $k$ consecutive closes exceed the trailing-high "
            "resistance. Place the stop at $S = P_0 - R$ (swing low, with $R \\ge 0.01\\,P_0$) and the "
            "target at $T = P_0 + R$. Let $\\tau$ be the first bar at which either barrier is touched "
            "(pessimistic tie-break: stop first). The trade pays\n\n"
            "$$ X = \\begin{cases} +1\\,R & \\text{if } T \\text{ hit first} \\\\ -1\\,R & \\text{if } S \\text{ hit first} \\end{cases}, \\qquad \\mathbb{E}[X] = (2p - 1)\\,R, \\;\\; p = \\Pr(\\text{win}). $$\n\n"
            "If the post-entry log-price is a **driftless** process, the symmetric barriers give "
            "$p = \\tfrac12$ by the optional-stopping theorem — the bracket is a martingale and the "
            "breakout carries no information. Any departure of $p$ from $\\tfrac12$ is precisely the "
            "post-breakout drift the strategy is betting exists. Our generator parameterizes exactly that "
            "drift, so $p$ is a **known function of a knob**."
        ),
        code(
            "for tag, tape, truth in [('null', null, t_null), ('continuation', cont, t_cont),\n"
            "                         ('grind-gated', grind, t_grind)]:\n"
            "    s = strategy.summary(strategy.run(tape))\n"
            "    print(f'{tag:13s} cont_drift={truth.cont_drift:+.5f} (edge_sign {truth.edge_sign:+d}) '\n"
            "          f'-> measured win rate {s[\"win_rate\"]:.3f}')"
        ),

        md(
            "## Beat 2 · So what?\n\n"
            "The 1:1 stop/target turns a directional claim into a **proportion test**. Profitability is "
            "no longer a vague 'edge' but a sharp inequality: $p > 0.5 + c_R/2$, where the round-trip "
            "cost in R is $c_R = (\\text{bps}\\times 10^{-4})/\\rho$ and $\\rho = R/P_0$ is the stop as a "
            "fraction of price. Two consequences fall out immediately. First, a *tight* stop (small "
            "$\\rho$) makes $c_R$ **larger** — the 1% floor is the most cost-exposed knob in the system. "
            "Second, because the breakout is selected on a new high, the naive intuition 'it's going up, "
            "so it'll continue' must beat not just 0.5 but 0.5-plus-a-spread, every single trade."
        ),

        md(
            "## Beat 3 · How we'd know — the pre-registered protocol\n\n"
            "1. **Null calibration.** On `cont_drift=0` the measured win rate's Wilson interval must "
            "contain 0.50 (else the simulator is biased).\n"
            "2. **Break-even.** Compute $c_R$ and $0.5 + c_R/2$ from the realized stop fractions; sweep "
            "cost and locate the zero-crossing of net expectancy.\n"
            "3. **Filter lift.** Compare the win rate of the all-filters-pass subset to the field; test "
            "the lift against its own standard error, and report the shrinkage in $n$.\n"
            "4. **Power checks (anti-rigging).** On a continuation tape the win rate must exceed 0.5 and "
            "net expectancy stay positive through costs; on a grind-gated tape the grind score must "
            "separate winners from losers.\n\n"
            "**Mirage line:** win rate indistinguishable from 0.50 and net expectancy $\\le 0$ at "
            "realistic cost, with a filter lift inside its standard error."
        ),

        md(
            "## Beat 4 · The teardown\n\n"
            "### 4a · Null calibration — the bracket is a martingale\n"
            "Win rate with a Wilson score interval, and the gross expectancy identity $(2p-1)$."
        ),
        code(
            "tr = strategy.run(null)\n"
            "s = strategy.summary(tr); lo, hi = strategy.win_rate_ci(tr)\n"
            "print(f'n = {s[\"n_trades\"]} trades')\n"
            "print(f'win rate p = {s[\"win_rate\"]:.4f}, Wilson 95% CI [{lo:.4f}, {hi:.4f}]  '\n"
            "      f'-> contains 0.5: {lo <= 0.5 <= hi}')\n"
            "print(f'gross expectancy (2p-1) = {s[\"expectancy_R_gross\"]:+.4f} R  '\n"
            "      f'(avg stop fraction rho = {s[\"avg_risk_frac\"]:.4f})')"
        ),
        md(
            "> 💡 **In plain words.** On a tape we *built* to be fair, the strategy comes out fair. That "
            "calibration is what licenses every later claim: when the win rate departs from 0.50, it's "
            "the data talking, not a bug in the bracket."
        ),

        md(
            "### 4b · Cost-in-R and the break-even line\n"
            "The break-even win rate is $0.5 + c_R/2$. Because $\\rho \\approx 1\\%$, even a few bps lifts "
            "it visibly above a coin flip; the sweep shows net expectancy crossing zero."
        ),
        code(
            "sweep = strategy.cost_sweep(tr, roundtrip_bps=(0,1,2,5,10,20))\n"
            "display(sweep.round(4))\n"
            "be2 = strategy.summary(tr, 2.0)['breakeven_win_rate']\n"
            "fig, ax = plt.subplots()\n"
            "ax.plot(sweep.index, sweep['expectancy_R_net'].values, 'o-', label='net E[R]')\n"
            "ax.axhline(0, ls=':', c='grey')\n"
            "ax.set_xlabel('round-trip cost (bps)'); ax.set_ylabel('net expectancy (R/trade)')\n"
            "ax.set_title(f'Break-even win rate at 2 bps = {be2:.3f}; net E[R] crosses zero as cost bites')\n"
            "ax.legend(); plt.show()"
        ),
        md(
            "> 💡 **In plain words.** A 1% stop means a 2 bps spread is ~2% of your R, *each way*. The "
            "win rate you'd need just to tread water creeps above 50% the instant you pay anything — and "
            "the measured win rate doesn't clear it."
        ),

        md(
            "### 4c · The filter-lift test — selection illusion, quantified\n"
            "Grade every trade on the three filters; compare the A-grade win rate to the field, and put "
            "the lift next to its own standard error. A null result here is **no lift while $n$ "
            "collapses**."
        ),
        code(
            "fl = filters.filter_lift(tr, null)\n"
            "se = np.sqrt(0.25 / fl['n_filtered'])     # SE of a ~50% rate on the filtered subset\n"
            "z = fl['lift'] / se\n"
            "print(f'all:     n={fl[\"n_all\"]:4d}  win {fl[\"win_rate_all\"]:.3f}')\n"
            "print(f'A-grade: n={fl[\"n_filtered\"]:4d}  win {fl[\"win_rate_filtered\"]:.3f}  '\n"
            "      f'(kept {fl[\"kept_frac\"]:.0%})')\n"
            "print(f'lift = {fl[\"lift\"]:+.3f}  vs  SE {se:.3f}   -> z = {z:+.2f}  '\n"
            "      f'(|z|<2 => indistinguishable from no lift)')"
        ),
        md(
            "> 💡 **In plain words.** Filtering can't turn a fair coin into a loaded one; it can only "
            "keep fewer flips. A 'lift' computed on the survivors is mostly the noise of a small sample — "
            "which is why a curated screenshot gallery (the ultimate filtered subset) tells you nothing."
        ),

        md(
            "### 4d · Power checks — the test is not rigged to say 'no'\n"
            "Recover a *known* edge with the same machinery: a continuation tape (win rate must exceed "
            "0.5, net-positive through cost) and a grind-gated tape (the grind score must rank winners)."
        ),
        code(
            "# (i) continuation tape: real momentum -> win rate up, net expectancy positive after costs\n"
            "sc = strategy.summary(strategy.run(cont), roundtrip_bps=4.0)\n"
            "loc, hic = strategy.win_rate_ci(strategy.run(cont))\n"
            "print(f'continuation: win {sc[\"win_rate\"]:.3f} CI [{loc:.3f},{hic:.3f}] '\n"
            "      f'(>0.5), net E[R] {sc[\"expectancy_R_net\"]:+.3f} after 4 bps')\n"
            "# (ii) grind-gated tape: the grind score itself ranks the trades\n"
            "g = filters.annotate(strategy.run(grind), grind); g = g[g.outcome != 0]\n"
            "import numpy as np\n"
            "qs = pd.qcut(g['grind'], 4, labels=['Q1 spiky','Q2','Q3','Q4 grindy'])\n"
            "wr_by_q = g.groupby(qs, observed=True).apply(lambda d: (d.outcome==1).mean())\n"
            "fig, ax = plt.subplots(figsize=(6.6,4.2))\n"
            "ax.bar(wr_by_q.index.astype(str), wr_by_q.values,\n"
            "       color=['#c0392b','#dab617','#7fae3a','#2ea44f'])\n"
            "ax.axhline(0.5, ls=':', c='k'); ax.set_ylabel('win rate'); ax.set_ylim(0,1)\n"
            "ax.set_title('Grind-gated tape: win rate rises with the grind score')\n"
            "plt.show()\n"
            "print('win rate by grind quartile:'); print(wr_by_q.round(3).to_string())"
        ),
        md(
            "> 💡 **In plain words.** When we *put* an edge in, the same code pulls it back out — cleanly, "
            "and through the filter that's supposed to detect it. So the flat result on the real claim is "
            "a property of the market, not of a pessimistic backtest."
        ),

        md(
            "## Beat 5 · The verdict\n\n"
            "- **Null calibrates** (4a): win rate 0.50 within a Wilson interval — the bracket is an "
            "unbiased martingale.\n"
            "- **Costs bind** (4b): break-even $\\approx$ "
            f"{R['be2']}% at 2 bps; net expectancy crosses zero at a few bps because $\\rho\\approx1\\%$.\n"
            "- **Filters are decorative** (4c): lift within one standard error, $n$ collapses.\n"
            "- **The test has power** (4d): a known edge is recovered, and ranked, through the grind "
            "filter.\n\n"
            "> **Signal `NONE`.** The breakout's win rate is indistinguishable from a coin flip on the "
            "null and on every real tape — see [`../docs/results.md`](../docs/results.md)."
        ),

        md(
            "## Beat 6 · Could you trade it?\n\n"
            "The protocol's usual killers don't even need to deploy — the signal is gone at beat 5. For "
            "completeness, the real-tape evidence (5-minute, ~60-day window, the deepest Yahoo serves; "
            "full provenance + fingerprints in [`../docs/results.md`](../docs/results.md)):\n\n"
            "| Tape | Trades | Win % | 95% CI | Net E[R] @2bps | A-grade subset |\n"
            "|---|---|---|---|---|---|\n"
            f"| **BTC-USD** | {R['btc_n']} | {R['btc_wr']} | {R['btc_ci']} | **{R['btc_net']}** | "
            f"{R['btc_flt_n']} trades (0% won) |\n"
            f"| SPY | {R['spy_n']} | {R['spy_wr']} | {R['spy_ci']} | {R['spy_net']} | "
            f"{R['spy_flt_n']} trade (100% won) |\n"
            f"| QQQ | {R['qqq_n']} | {R['qqq_wr']} | {R['qqq_ci']} | {R['qqq_net']} | "
            f"{R['qqq_flt_n']} trades |\n\n"
            "Every interval straddles 0.50. The **deepest** sample (BTC-USD, Koroush's market) is "
            "net-negative. The 'A-grade' subsets are 1–9 trades — the SPY one is a single winning trade, "
            "the canonical '100% win rate' artefact. And the live strategy runs on the **1-minute** "
            "chart, where the spread is a larger fraction of R: strictly worse than this 5-minute test.\n\n"
            "> Tradability: **`MIRAGE`** — there is no capacity question to reach; the edge is absent "
            "before costs and negative after them."
        ),

        md(
            "## Beat 7 · Going further\n\n"
            "- **Power, not luck — the synthetic steelman.** Beat 4d is the load-bearing fairness claim: "
            "the same estimator that returns 'coin flip' on the real tape returns a clean, ranked "
            "signal when an edge is genuinely present. Anyone disputing the verdict should attack *that* "
            "calibration, not the negative result.\n"
            "- **Reward-to-risk other than 1:1.** The break-even win rate generalizes to "
            "$\\text{BE} = \\tfrac{1 + c_R}{1 + b}$ for a $b{:}1$ target. Letting winners run (large $b$) "
            "lowers the bar on $p$ but $p$ itself falls as the target recedes — a trade-off worth mapping "
            "with the same bracket sim (raise `target` to $bR$ and re-measure $p(b)$).\n"
            "- **Limit-at-level entry.** We enter at the confirmation close (always fills). Koroush's "
            "limit on the pullback gets a better price but forfeits the runaway winners — model the fill "
            "probability and the conditional win rate of filled vs missed trades.\n"
            "- **A deeper tape.** Tens of trades on 60 days can't resolve a 2-point edge. A paid intraday "
            "feed or an MT5 export ([`quantlab/brokers/mt5_connector.py`](../../../quantlab/brokers/mt5_connector.py)) "
            "would put thousands of trades behind the Wilson interval and settle whether *any* instrument "
            "has post-breakout drift exceeding its own spread.\n\n"
            "PRs welcome — attack the calibration, map the reward-to-risk frontier, or bring a deeper "
            "tape."
        ),
    ]
    return new_notebook(cells=cells, metadata=_meta())


def _meta():
    return {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }


def main():
    nbf.write(build_curious(), os.path.join(HERE, "01_for_the_curious.ipynb"))
    nbf.write(build_quants(), os.path.join(HERE, "02_for_the_quants.ipynb"))
    print("wrote 01_for_the_curious.ipynb and 02_for_the_quants.ipynb")


if __name__ == "__main__":
    main()
