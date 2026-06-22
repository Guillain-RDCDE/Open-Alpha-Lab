"""Generate the two narrative notebooks for Study 353 (Smart-Money-Concepts: ICT order blocks
and fair-value gaps).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic: the real-tape cells read the cached SPY daily
OHLC under ../_cache/spy_daily.csv and otherwise quote the frozen headline numbers in ``R``
(mirroring docs/results.md). The synthetic random-walk control runs anywhere with no network.
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (SPY daily OHLC, yfinance,
# 1993-01-29 -> 2026-06-18, 8,404 bars; as-of 2026-06-18; fingerprint ed7e466d0523).
R = dict(
    bars=8404, start="1993-01-29", asof="2026-06-18", px_lo=43, px_hi=760, vol=0.0117,
    finger="ed7e466d0523",
    n_fvg=2337, fvg_bull=1423, fvg_bear=914, fvg_width=0.53,
    # fill-rate by horizon: (H, real, placebo, synth, z_real_minus_placebo)
    fill=[(5, 0.718, 0.740, 0.705, -1.71), (10, 0.791, 0.805, 0.779, -1.20),
          (20, 0.842, 0.864, 0.836, -2.15), (40, 0.887, 0.888, 0.882, -0.05)],
    ttf=1,
    n_ob=1440, ob_bull=787, ob_bear=653,
    ob_n=1403, ob_mean=-0.114, ob_t=-1.04,
    rnd_n=28060, rnd_mean=-0.027, rnd_t=-1.41,
    welch_diff=-0.087, welch_t=-0.78, welch_p=0.435,
    syn_ob_n=1896, syn_ob_t=-0.72,
    ob_net1_mean=-0.134, ob_net1_t=-1.23, ob_net5_mean=-0.214, ob_net5_t=-1.96,
    syn_fvg=2900, syn_fvg_fill=0.836, syn_n_ob=1933,
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Footprints%3F: Not_supported](https://img.shields.io/badge/Footprints%3F-Not_supported-8b949e?style=flat-square)\n\n"
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

from smart_money_concepts import data, strategy as st

HAVE_REAL = data.have_real()
DF = data.load_real() if HAVE_REAL else None
print("real SPY tape cache present:", HAVE_REAL,
      "| bars:", (0 if DF is None else len(DF)))
"""

# The frozen headline dict is embedded into the notebook's first code cell so every
# downstream cell can quote it whether or not the cache is present.
BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Smart Money Concepts — does \"smart money\" leave footprints on the chart? 🕵️\n"
            "### The huge ICT / \"SMC\" meme: order blocks and fair-value gaps, in plain English\n\n"
            + BADGES +
            "Scroll any trading corner of YouTube and you'll meet **ICT / Smart Money Concepts**: "
            "the idea that big institutions leave *footprints* on the chart, and if you learn to read "
            "two shapes — **order blocks** and **fair-value gaps** — you can trade alongside the "
            "whales. The killer demo is always the same: *\"see how price came back and filled that "
            "gap? see how it bounced right off that order block? that's smart money.\"*\n\n"
            "It's mesmerising because it's **half true**: the gaps are real, the bounces are real. "
            "The question this notebook asks is the one the videos never do — **does a coin-flip "
            "random chart do exactly the same thing?** If it does, the footprint isn't a footprint.\n\n"
            "> 📓 **Plain-language layer.** Want the two-proportion *z*, the Welch *t* and the "
            "horizon sweep? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** Every chart below is drawn by the code beside it; house "
            "style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Do \"fair-value gaps\" really get filled? | **Yes — about 84% of them**, within a "
            "month. Sounds like a magnet. |\n"
            "| So price respects the gap? | **No.** Drop a *random* same-size box near the price and "
            "it gets touched **just as often** (86%). A noise chart fills its gaps too. |\n"
            "| Do \"order blocks\" mark where price bounces? | **No.** Forward returns from an order "
            "block are a hair *negative* and statistically zero — no different from entering on a "
            "random day. |\n"
            "| So is there a footprint? | **Not one we can measure.** A coin-flip random walk leaves "
            "the same gaps, fills them at the same rate, and bounces the same way. The pattern is "
            "**geometry**, not institutions. |\n\n"
            "> The shapes are real. The story bolted on top — *\"this is where smart money acted\"* — "
            "is indistinguishable from chance."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Institutions can't hide. When they move size they leave **imbalances** — fair-value "
            "gaps — and **order blocks**. Price always comes back to rebalance the gap and to "
            "mitigate the order block. Mark those zones and you're trading with the smart money.\"*\n\n"
            "Two concrete, testable shapes:\n\n"
            "- **Fair-value gap (FVG).** A 3-candle pattern where candle 1 and candle 3 don't "
            "overlap, leaving an empty price band — a \"void\". The claim: price is drawn back to "
            "**fill** it.\n"
            "- **Order block (OB).** The last opposite-colour candle right before a big push. The "
            "claim: when price **returns** to that candle's zone, it **reverses** there — so buying "
            "(or selling) the revisit has an edge."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If true, this is a free, chart-only edge: no fundamentals, no data feeds, just two "
            "shapes anyone can draw. That's exactly why it sells. But there's a catch hiding in plain "
            "sight: **a purely random price wiggles too.** It gaps. It comes back. It bounces off "
            "levels. So a high \"fill-rate\" or a nice-looking bounce can't *by itself* prove an "
            "actor is there — you have to show the real chart does it **more** than a chart with no "
            "institutions in it at all. That comparison is the whole game."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **{R['bars']:,} days** of real SPY prices ({R['start']} → {R['asof']}) and, "
            "right beside it, a **fake chart**: a coin-flip random walk with the same wiggle size and "
            "**no institutions, no memory, nothing**. Then we run the exact same detector on both.\n\n"
            "1. **Fair-value gaps.** Find every 3-bar gap; see how often price comes back to fill it. "
            "Compare the real fill-rate to (a) a *random box* of the same size dropped at a random "
            "spot, and (b) the fake random-walk chart.\n"
            "2. **Order blocks.** Find every order block; measure the return *after* price revisits "
            "it. Compare to entering on a **random** day.\n\n"
            "**What would make us say \"footprint\"?** The real chart beating the random ones by a "
            "clear margin. **What would make us say \"mirage\"?** The random chart matching it. Spoiler: "
            "it matches it."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First: do gaps fill?** Yes — and impressively so. Here's the fill-rate of real SPY "
            "fair-value gaps as we give price more time, next to a random same-size box and the "
            "coin-flip chart."
        ),
        code(
            "if HAVE_REAL:\n"
            "    g = st.find_fvgs(DF, min_rel=0.001)\n"
            "    syn = data.synthetic_ohlc(n=len(DF), sigma=DF['close'].pct_change().std(), seed=353)\n"
            "    sg = st.find_fvgs(syn, min_rel=0.001)\n"
            "    Hs = [5, 10, 20, 40]\n"
            "    real = [st.fvg_fill_stats(DF, horizon=H, gaps=g)['fill_rate'] for H in Hs]\n"
            "    plac = [st.placebo_zone_fill(DF, g, horizon=H, seed=353)['fill_rate'] for H in Hs]\n"
            "    synf = [st.fvg_fill_stats(syn, horizon=H, gaps=sg)['fill_rate'] for H in Hs]\n"
            "else:\n"
            "    Hs   = [r[0] for r in R['fill']]\n"
            "    real = [r[1] for r in R['fill']]; plac = [r[2] for r in R['fill']]; synf = [r[3] for r in R['fill']]\n"
            "fig, ax = plt.subplots(figsize=(9, 4.6))\n"
            "ax.plot(Hs, [x*100 for x in real], 'o-', c=GREEN, lw=2, label='real SPY gaps')\n"
            "ax.plot(Hs, [x*100 for x in plac], 's--', c=GREY, lw=2, label='random same-size box')\n"
            "ax.plot(Hs, [x*100 for x in synf], '^:', c=RED, lw=2, label='coin-flip random walk')\n"
            "ax.set_xlabel('bars allowed to fill'); ax.set_ylabel('% of gaps filled')\n"
            "ax.set_title('Gaps fill ~84% — but so do random boxes and a noise chart'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('real   fill-rate:', [f'{x*100:.0f}%' for x in real])\n"
            "print('random box      :', [f'{x*100:.0f}%' for x in plac])\n"
            "print('coin-flip walk  :', [f'{x*100:.0f}%' for x in synf])"
        ),
        md(
            f"There it is. Real gaps fill **~{R['fill'][2][1]*100:.0f}%** of the time within 20 bars — "
            "a number a SMC video would flash as proof. But the **random box** fills "
            f"**~{R['fill'][2][2]*100:.0f}%**, and the **coin-flip chart** "
            f"**~{R['fill'][2][3]*100:.0f}%**. The real chart isn't ahead — if anything it's a "
            "whisker *behind*. An 84% fill-rate is just what happens when you put a small box near a "
            "price that wanders \\$10 a day: sooner or later the wander touches it. No magnet needed."
        ),
        md(
            "**Now the order blocks.** This is the stronger claim — not just \"price comes back\" but "
            "\"price *reverses* there\", which would be real money. We enter in the order block's "
            "direction the day after price revisits it, hold two weeks, and compare to entering on a "
            "**random** day."
        ),
        code(
            "if HAVE_REAL:\n"
            "    obs = st.find_order_blocks(DF, impulse=3, impulse_rel=0.015)\n"
            "    ob  = st.ob_forward_returns(DF, obs, horizon=10, lag=1)\n"
            "    rnd = st.random_entry_returns(DF, n_entries=ob['n']*20, horizon=10, lag=1, seed=353)\n"
            "    ob_m, ob_t, rnd_m = ob['mean']*100, ob['t'], rnd['mean']*100\n"
            "else:\n"
            "    ob_m, ob_t, rnd_m = R['ob_mean'], R['ob_t'], R['rnd_mean']\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.3))\n"
            "ax.bar(['order block\\nrevisit', 'random\\nday'], [ob_m, rnd_m], color=[GREEN, GREY], width=.55)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('avg forward 10-day return (%)')\n"
            "ax.set_title('\"Order block\" bounce vs a coin flip: no difference')\n"
            "for i,v in enumerate([ob_m, rnd_m]): ax.annotate(f'{v:+.2f}%',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'order-block forward return: {ob_m:+.3f}%  (t={ob_t:+.2f})')\n"
            "print(f'random-day forward return : {rnd_m:+.3f}%')"
        ),
        md(
            f"The order-block \"edge\" is **{R['ob_mean']:+.2f}%** over two weeks — *negative*, and "
            "statistically indistinguishable from zero. Entering on a **random day** does just as "
            "well (or as badly). The shape that's supposed to mark where the whales reverse marks "
            "**nothing**. (And we haven't even charged trading costs yet — those only make it worse.)"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** Gaps fill ~84%, but random boxes and a coin-flip chart fill just as "
            "often; order-block bounces are statistically zero and no better than a random day. "
            "Nothing beats the null.\n"
            "- **Tradability — Mirage.** With no edge gross, there's nothing left after the spread. "
            "The order-block book is already negative; costs deepen the hole.\n"
            "- **Footprints? — Not supported.** A random walk with **no institutions** leaves the "
            "same gaps and the same bounces. The footprint can't be told apart from chance."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — the coin-flip twin\n\n"
            "The cleanest way to see it: run the *identical* detector on a chart we **know** has no "
            "smart money in it — a driftless coin-flip random walk — and count its footprints."
        ),
        code(
            "syn = data.synthetic_ohlc(n=R['bars'], sigma=R['vol'], seed=353)\n"
            "sg  = st.find_fvgs(syn, min_rel=0.001)\n"
            "sob = st.find_order_blocks(syn, impulse=3, impulse_rel=0.015)\n"
            "sfill = st.fvg_fill_stats(syn, horizon=20, gaps=sg)['fill_rate']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "ax.bar(['fair-value\\ngaps found', 'order blocks\\nfound'], [len(sg), len(sob)],\n"
            "       color=GREY, width=.5)\n"
            "for i,v in enumerate([len(sg), len(sob)]): ax.annotate(f'{v:,}',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('count on a pure-noise chart')\n"
            "ax.set_title(f'A coin-flip chart: {len(sg):,} gaps (fill {sfill*100:.0f}%), {len(sob):,} order blocks')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'coin-flip chart: {len(sg):,} fair-value gaps (fill {sfill*100:.0f}% in 20 bars), {len(sob):,} order blocks')"
        ),
        md(
            f"A chart with literally **nothing** behind it produces **{R['syn_fvg']:,}** fair-value "
            f"gaps that fill **{R['syn_fvg_fill']*100:.0f}%** of the time and **{R['syn_n_ob']:,}** "
            "order blocks — about as many as the real SPY, doing about the same thing. That's the "
            "whole argument in one chart: **if noise leaves the same footprints, the footprints aren't "
            "evidence of anyone.** There's no edge here to trade, with or without costs."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Try to break our null.** Maybe a *stricter* gap definition, a specific session, or a "
            "\"displacement + FVG + OB confluence\" filter finally beats the random walk. Fork the "
            "detector and show one that does — measured against the matched random box, not against "
            "zero.\n"
            "- **The same shape elsewhere.** [Study 301 — Triple-RSI](../../301-triple-rsi/) and "
            "[Study 351 — Polymarket momentum](../../351-btc-5m-polymarket-momentum/): viral "
            "strategies whose headline statistic evaporates the moment you compare it to the right "
            "null.\n"
            "- **Why it *feels* real.** Harry Roberts (1959) showed chartists confidently read "
            "\"support\" and \"head-and-shoulders\" off random-walk charts. Our result is the modern "
            "version: the pattern-recognition is real, the institution behind it is imagined.\n\n"
            "*Think there's a footprint we missed? The detector and the coin-flip control are right "
            "here — beat the random walk and you've found something.*"
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
            "# Smart-Money-Concepts — a quantitative teardown 🔬\n"
            "### FVG fill-rate vs a placebo two-proportion z · OB forward returns vs random entry "
            "(Welch t) · a horizon sweep · costs on NAV · the random-walk positive control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). ICT / "
            "\"Smart Money Concepts\" makes two falsifiable claims about **footprints**: (a) "
            "fair-value gaps fill more/faster than random same-size zones, and (b) forward returns "
            "from order blocks beat random levels. We test both on real SPY daily OHLC against the "
            "only honest null — **a matched random walk**, which produces gaps and bounces with no "
            "institutions in it.\n\n"
            "> ⚠️ **Not investment advice.** Real data: SPY daily OHLC (yfinance), "
            f"{R['start']} → {R['asof']} ({R['bars']:,} bars). Offline core + synthetic control are "
            "deterministic (seed 353). Methods in [`docs/references.md`](../docs/references.md), "
            "numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | FVG fill-rate ≤ matched random zone at every horizon "
            f"(real−placebo z = **{R['fill'][2][4]:.2f}** at H=20, the wrong sign); OB forward "
            f"return **{R['ob_mean']:+.2f}%** (t={R['ob_t']:.2f}), Welch vs random entry "
            f"t = **{R['welch_t']:.2f}** (p={R['welch_p']:.2f}). |\n"
            f"| **Tradability** | `MIRAGE` | Signal absent gross; OB book "
            f"{R['ob_net5_mean']:+.2f}% net of 5 bp/leg (t={R['ob_net5_t']:.2f}). Nothing to "
            "scale. |\n"
            f"| **Footprints?** | `NOT SUPPORTED` | The same detector on a driftless random walk "
            f"finds **{R['syn_fvg']:,}** FVGs (fill {R['syn_fvg_fill']*100:.0f}%) and "
            f"**{R['syn_n_ob']:,}** OBs (edge t={R['syn_ob_t']:.2f}) — noise reproduces the "
            "pattern. |\n\n"
            "> 💡 In plain words: the shapes are real and common, but a memory-less random walk makes "
            "them just as often and they predict nothing — so they are geometry, not a footprint."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let a tape be candles $\\{(O_t, H_t, L_t, C_t)\\}$. ICT defines two structures:\n\n"
            "- **Fair-value gap** at bar $i$: *bullish* iff $L_i > H_{i-2}$ (void "
            "$(H_{i-2}, L_i)$ below price), *bearish* iff $H_i < L_{i-2}$. Claim **H₁**: "
            "$\\Pr[\\text{fill within }H \\mid \\text{FVG}] > \\Pr[\\text{fill} \\mid \\text{random same-width zone}]$.\n"
            "- **Order block** at bar $i$: the last down-candle before a $\\ge\\delta$ up-impulse "
            "(bullish), or mirror. Claim **H₂**: the signed forward return on a revisit, "
            "$r_H \\cdot \\mathrm{dir}$, has $\\mathbb{E}[r_H\\,\\mathrm{dir}] > 0$ and beats a "
            "random-entry control.\n\n"
            "We find **H₁ rejected** (fill-rate $\\le$ placebo at every $H$; real−placebo z $\\le 0$) "
            "and **H₂ rejected** (signed forward return $\\approx 0$, Welch t vs random entry "
            f"$={R['welch_t']:.2f}$). Both structures are reproduced in equal abundance on a "
            "driftless random walk — the null the claim must beat and does not."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The trap is taking a raw statistic as proof. A diffusing price has range that grows like "
            "$\\sqrt{t}$, so a fixed-width band placed near it is hit with high probability purely by "
            "**first passage** — Brownian barrier-hitting, no actor required. An 84% fill-rate is a "
            "first-passage probability, not a magnet. Likewise, if price is near a martingale the "
            "conditional forward return from *any* chart-defined level is ~0. So the only admissible "
            "evidence of a footprint is **excess over a matched random process**: the real fill-rate "
            "minus the placebo, and the OB return minus random entry. Everything below is that "
            "subtraction."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Real tape.** SPY daily OHLC (yfinance), {R['start']} → {R['asof']}, "
            f"{R['bars']:,} bars; as-of {R['asof']}, fingerprint `{R['finger']}`.\n"
            "- **FVG test.** Detect all 3-bar gaps ≥ 0.1% width; fill = a later bar's range overlaps "
            "the void. **Two-proportion z** of real fill-rate vs a *within-tape placebo* (same-width "
            "zone at a random anchor bar), across a horizon sweep $H\\in\\{5,10,20,40\\}$.\n"
            "- **OB test.** Last opposite candle before a ≥1.5% 3-bar impulse; enter signed in the OB "
            "direction the bar **after** the first revisit (**one** execution lag), hold 10 bars. "
            "**Welch t** of the OB book vs a random-entry control (same horizon, random signs).\n"
            "- **Positive control.** A deterministic driftless random-walk OHLC (seed 353, σ matched "
            "to SPY): the detector must fire on it (proving it isn't broken) and the real tape must "
            "**beat** it (it doesn't).\n"
            "- **Costs.** One-way bps × NAV on both legs of the round-trip — charged even though the "
            "gross signal is already absent.\n\n"
            "**Mirage trigger, pre-registered:** real fill-rate $\\le$ placebo, and OB Welch t $< 2$. "
            "Both fire."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Fair-value gaps fill no better than random — the horizon sweep\n\n"
            "Real fill-rate vs the placebo same-width zone and the random-walk synthetic, across "
            "horizons. The real−placebo two-proportion z is printed; it never clears 0 in the "
            "claim's direction."
        ),
        code(
            "if HAVE_REAL:\n"
            "    g = st.find_fvgs(DF, min_rel=0.001)\n"
            "    syn = data.synthetic_ohlc(n=len(DF), sigma=DF['close'].pct_change().std(), seed=353)\n"
            "    sg = st.find_fvgs(syn, min_rel=0.001)\n"
            "    Hs = [r[0] for r in R['fill']]; real=[]; plac=[]; synf=[]; zs=[]\n"
            "    for H in Hs:\n"
            "        r = st.fvg_fill_stats(DF, horizon=H, gaps=g)\n"
            "        p = st.placebo_zone_fill(DF, g, horizon=H, seed=353)\n"
            "        s = st.fvg_fill_stats(syn, horizon=H, gaps=sg)\n"
            "        real.append(r['fill_rate']); plac.append(p['fill_rate']); synf.append(s['fill_rate'])\n"
            "        zs.append(st.two_prop_z(round(r['fill_rate']*r['n']), r['n'], round(p['fill_rate']*p['n']), p['n']))\n"
            "else:\n"
            "    Hs=[r[0] for r in R['fill']]; real=[r[1] for r in R['fill']]\n"
            "    plac=[r[2] for r in R['fill']]; synf=[r[3] for r in R['fill']]; zs=[r[4] for r in R['fill']]\n"
            "x=np.arange(len(Hs)); fig,ax=plt.subplots(figsize=(9.2,4.5))\n"
            "ax.bar(x-.25,[v*100 for v in real],.25,color=GREEN,label='real FVG')\n"
            "ax.bar(x,    [v*100 for v in plac],.25,color=GREY, label='placebo zone')\n"
            "ax.bar(x+.25,[v*100 for v in synf],.25,color=RED,  label='random walk')\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'H={h}' for h in Hs]); ax.set_ylim(60,95)\n"
            "ax.set_ylabel('fill-rate (%)'); ax.set_title('FVG fill-rate ≤ matched random zone at every horizon'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for h,r,p,z in zip(Hs,real,plac,zs): print(f'H={h:2d}: real {r*100:.1f}%  placebo {p*100:.1f}%  z(real-plac)={z:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: at H=20 real gaps fill **{R['fill'][2][1]*100:.1f}%**, the placebo "
            f"**{R['fill'][2][2]*100:.1f}%** — the difference runs the **wrong way** (z = "
            f"{R['fill'][2][4]:.2f}). H₁ rejected at every horizon. The fill-rate is a first-passage "
            "probability of a wandering price, identical for a random box."
        ),
        md(
            "### 4b · Order-block forward returns vs random entry — the Welch t\n\n"
            "Signed forward 10-bar return entering on the first OB revisit (1-bar lag), against a "
            "random-entry control with the same horizon and a 50/50 long-short mix."
        ),
        code(
            "if HAVE_REAL:\n"
            "    obs = st.find_order_blocks(DF, impulse=3, impulse_rel=0.015)\n"
            "    ob  = st.ob_forward_returns(DF, obs, horizon=10, lag=1)\n"
            "    rnd = st.random_entry_returns(DF, n_entries=ob['n']*20, horizon=10, lag=1, seed=353)\n"
            "    w   = st.welch_t(ob['rets'], rnd['rets'])\n"
            "    ob_m, ob_t, ob_se = ob['mean']*100, ob['t'], ob['std']/np.sqrt(ob['n'])*100\n"
            "    rnd_m, rnd_se = rnd['mean']*100, rnd['std']/np.sqrt(rnd['n'])*100\n"
            "    wt, wp = w['t'], w['p']\n"
            "else:\n"
            "    ob_m, ob_t, ob_se = R['ob_mean'], R['ob_t'], 0.11\n"
            "    rnd_m, rnd_se = R['rnd_mean'], 0.02; wt, wp = R['welch_t'], R['welch_p']\n"
            "fig, ax = plt.subplots(figsize=(8.6,4.3))\n"
            "ax.bar(['OB revisit','random entry'], [ob_m, rnd_m], yerr=[ob_se, rnd_se],\n"
            "       color=[GREEN, GREY], width=.55, capsize=5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('mean signed forward 10-bar return (%)')\n"
            "ax.set_title(f'OB vs random entry: Welch t={wt:+.2f} (p={wp:.2f}) — no difference')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'OB book : mean={ob_m:+.3f}%  t(vs0)={ob_t:+.2f}')\n"
            "print(f'random  : mean={rnd_m:+.3f}%')\n"
            "print(f'Welch OB-random: t={wt:+.2f}  p={wp:.3f}')"
        ),
        md(
            f"> 💡 In plain words: the OB book returns **{R['ob_mean']:+.2f}%** (t={R['ob_t']:.2f} vs "
            f"0 — a hair negative) and is statistically the same as random entry (Welch "
            f"t={R['welch_t']:.2f}, p={R['welch_p']:.2f}). H₂ rejected. The free parameters "
            "(impulse size, horizon) had every chance to find an edge **raw** and didn't — the "
            "cleanest possible negative."
        ),
        md(
            "### 4c · Costs only deepen a hole that's already there\n\n"
            "Even granting a non-existent gross edge, charging the spread one-way on both legs of the "
            "round-trip makes the OB book strictly worse:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    rets = ob['rets']\n"
            "    bps = [0, 1, 2, 5]\n"
            "    means = [st.net_of_costs(rets, b)['mean']*100 for b in bps]\n"
            "    ts    = [st.net_of_costs(rets, b)['t'] for b in bps]\n"
            "else:\n"
            "    bps=[0,1,2,5]; means=[R['ob_mean'], R['ob_net1_mean'], -0.154, R['ob_net5_mean']]\n"
            "    ts=[R['ob_t'], R['ob_net1_t'], -1.41, R['ob_net5_t']]\n"
            "fig, ax = plt.subplots(figsize=(8.6,4.2))\n"
            "ax.plot(bps, means, 'o-', c=RED, lw=2)\n"
            "ax.axhline(0, ls='--', c=GREY); ax.set_xlabel('one-way cost per leg (bp)')\n"
            "ax.set_ylabel('net mean forward return (%)')\n"
            "ax.set_title('Net OB return: negative at zero cost, worse with costs')\n"
            "for b,m in zip(bps,means): ax.annotate(f'{m:+.2f}%',(b,m),ha='center',va='top')\n"
            "plt.tight_layout(); plt.show()\n"
            "for b,m,t in zip(bps,means,ts): print(f'{b}bp/leg: net mean={m:+.3f}%  t={t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: the book is already **{R['ob_mean']:+.2f}%** at zero cost; at 5 "
            f"bp/leg it's **{R['ob_net5_mean']:+.2f}%** (t={R['ob_net5_t']:.2f}). There is no cost "
            "level at which a negative gross edge becomes tradable. Tradability `MIRAGE` is "
            "over-determined."
        ),
        md(
            "### 4d · The positive control — the detector fires on pure noise\n\n"
            "Run the identical detector on a driftless random walk (seed 353, σ matched to SPY). It "
            "**must** find gaps and order blocks — and it finds about as many as the real tape, "
            "filling and \"bouncing\" the same way. That's the proof the structures are geometry."
        ),
        code(
            "syn = data.synthetic_ohlc(n=R['bars'], sigma=R['vol'], seed=353)\n"
            "sg  = st.find_fvgs(syn, min_rel=0.001)\n"
            "sob = st.find_order_blocks(syn, impulse=3, impulse_rel=0.015)\n"
            "sfill = st.fvg_fill_stats(syn, horizon=20, gaps=sg)['fill_rate']\n"
            "sret  = st.ob_forward_returns(syn, sob, horizon=10, lag=1)\n"
            "labels = ['FVGs', 'order blocks']\n"
            "real_counts = [R['n_fvg'], R['n_ob']]; syn_counts = [len(sg), len(sob)]\n"
            "x=np.arange(2); fig,ax=plt.subplots(figsize=(8.6,4.3))\n"
            "ax.bar(x-.2, real_counts, .4, color=GREEN, label='real SPY')\n"
            "ax.bar(x+.2, syn_counts, .4, color=RED, label='random walk')\n"
            "ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel('count')\n"
            "ax.set_title('Same footprints on a chart with no institutions'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'random walk: {len(sg):,} FVGs (fill {sfill*100:.1f}% @H20), {len(sob):,} OBs (fwd t={sret[\"t\"]:+.2f})')"
        ),
        md(
            f"> 💡 In plain words: noise yields **{R['syn_fvg']:,}** FVGs filling "
            f"**{R['syn_fvg_fill']*100:.0f}%** and **{R['syn_n_ob']:,}** OBs with a forward t of "
            f"**{R['syn_ob_t']:.2f}** — indistinguishable from SPY. A footprint a memory-less process "
            "leaves in equal measure carries no information about who was trading."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — FVG fill-rate ≤ placebo at every horizon (real−placebo z = "
            f"{R['fill'][2][4]:.2f} at H=20, wrong sign); OB forward return {R['ob_mean']:+.2f}% "
            f"(t={R['ob_t']:.2f}), Welch vs random entry t={R['welch_t']:.2f} (p={R['welch_p']:.2f}). "
            "No robust t ≥ 2 in the claim's direction anywhere.\n"
            f"- **Tradability `MIRAGE`** — gross signal absent; OB book {R['ob_net5_mean']:+.2f}% net "
            f"of 5 bp/leg (t={R['ob_net5_t']:.2f}). Nothing to scale.\n"
            f"- **Footprints? `NOT SUPPORTED`** — the synthetic control reproduces **{R['syn_fvg']:,}** "
            f"FVGs (fill {R['syn_fvg_fill']*100:.0f}%) and **{R['syn_n_ob']:,}** OBs (t={R['syn_ob_t']:.2f}) "
            "on pure noise. The patterns are random-walk geometry, not institutional footprints."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — there is no edge to size\n\n"
            "Beat 6 normally asks how much capital an edge holds. Here there is no edge: the FVG fill "
            "doesn't beat a random box, the OB return is statistically zero and indistinguishable "
            "from random entry, and both are reproduced by noise. The only honest sizing of a "
            "zero-edge, negative-after-costs rule is **zero**. The table below is the whole tradability "
            "case in one place."
        ),
        code(
            "rows = [\n"
            "    ('FVG fill vs placebo (H=20)', f\"z = {R['fill'][2][4]:+.2f}\", 'no excess fill'),\n"
            "    ('OB forward return (gross)',  f\"{R['ob_mean']:+.2f}%  (t={R['ob_t']:+.2f})\", 'zero / negative'),\n"
            "    ('OB vs random entry (Welch)', f\"t = {R['welch_t']:+.2f}  (p={R['welch_p']:.2f})\", 'no difference'),\n"
            "    ('OB net of 5 bp/leg',         f\"{R['ob_net5_mean']:+.2f}%  (t={R['ob_net5_t']:+.2f})\", 'worse'),\n"
            "    ('random-walk OB edge',        f\"t = {R['syn_ob_t']:+.2f}\", 'same as real'),\n"
            "]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 2.6)); ax.axis('off')\n"
            "tbl = ax.table(cellText=[[a,b,c] for a,b,c in rows],\n"
            "               colLabels=['test', 'statistic', 'reading'], loc='center', cellLoc='left')\n"
            "tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.5)\n"
            "ax.set_title('Tradability scorecard — every line says MIRAGE', pad=12)\n"
            "plt.tight_layout(); plt.show()\n"
            "for a,b,c in rows: print(f'{a:30s} {b:24s} -> {c}')"
        ),
        md(
            "> 💡 In plain words: there is no row that points to a harvestable edge. The signal is "
            "absent gross, negative net, and reproduced by a coin flip. Capacity is moot — you don't "
            "size a non-edge."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Beat the null, not zero.** Any rescue of SMC — stricter FVG width, "
            "displacement+OB confluence, a specific session, multi-timeframe alignment — must be "
            "scored as **excess over the matched random walk / placebo zone**, with a t on the "
            "*difference*. A raw fill-rate or raw bounce is not admissible evidence.\n"
            "- **First-passage framing.** The 84% fill-rate is a Brownian barrier-hitting probability "
            "for a fixed-width band near a diffusing price; model it in closed form and the "
            "\"magnet\" disappears analytically.\n"
            "- **Multiple-testing honesty.** The OB rule already has free parameters (impulse δ, "
            "horizon) and fails *raw* — a White Reality Check over the parameter grid would only "
            "lower the bar further.\n\n"
            "*The reproducible core is offline and deterministic (seed 353). Methods and sources: "
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
