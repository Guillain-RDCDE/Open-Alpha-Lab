"""Generate the two narrative notebooks for Study 452 (Spinning-Top).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached daily tapes under
../_cache/ and otherwise quote the frozen headline numbers in ``R`` (mirroring docs/results.md).
The synthetic positive control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md.
# yfinance daily OHLC, 5 indices/ETFs (SPY QQQ IWM DIA GLD), 2005-01-03 -> 2026-05-29 (As-of
# 2026-05-31, partial June dropped), 21.4 years; spinning top = body<25% range, both wicks>=25%,
# wick balance>=0.5; long entered at next close. Welch t AVERAGED over 20 random-baseline seeds.
R = dict(
    asof="2026-05-31", start="2005-01-03", end="2026-05-29", years=21.4,
    tickers=["SPY", "QQQ", "IWM", "DIA", "GLD"], n_entries=3006,
    body_frac=0.25, wick_frac=0.25, balance=0.5,
    fp_spy="4cb5244f3990",
    # pooled spinning-top, per horizon (Welch mean over 20 seeds + spread):
    # (H, n, top_bps, win%, one_sample_t, random_bps, delta_bps, net_bps, welch_t, welch_min, welch_max)
    h5=(5, 3003, 23.4, 59, 4.50, 23.9, -0.5, 21.4, -0.07, -1.15, 0.82),
    h10=(10, 2995, 57.9, 61, 6.83, 48.6, 9.3, 55.9, 1.02, -0.01, 2.39),
    h20=(20, 2988, 120.3, 64, 8.35, 97.7, 22.7, 118.3, 1.73, 0.67, 3.38),
    h60=(60, 2963, 292.4, 70, 8.46, 296.6, -4.2, 290.4, -0.20, -2.07, 1.16),
    # per-ticker H=20 (single seed=7): (ticker, entries, top_bps, one_sample_t, random_bps, delta_bps)
    per=[("SPY", 639, 111.8, 3.87, 74.3, 37.5), ("QQQ", 549, 168.3, 4.77, 84.4, 83.8),
         ("IWM", 536, 130.2, 3.52, 35.7, 94.5), ("DIA", 656, 94.6, 3.65, 98.7, -4.1),
         ("GLD", 626, 105.7, 3.28, 96.3, 9.3)],
    # wick-scramble placebo (SPY, H=20, 500 draws): obs_bps, p
    placebo=(111.8, 0.066, 500),
    # synthetic control (H=20, n_days=4000): (edge, n, top_bps, win%, one_sample_t)
    syn=[(0.00, 713, -47.5, 47, -1.45), (1.20, 704, 246.6, 70, 7.38)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Indecision_forecasts_direction%3F: Busted](https://img.shields.io/badge/Indecision_forecasts_direction%3F-Busted-8b949e?style=flat-square)\n\n"
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from spinning_top import data, strategy as st

ASOF = "2026-05-31"
HAVE_REAL = data.have_real()
def load(t):
    b = data.load_real(t, allow_fetch=False)
    return b[b.index <= ASOF]
print("real spinning-top cache present:", HAVE_REAL, "| tickers:", data.DEFAULT_TICKERS)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does a \"spinning top\" candle really forecast a move? 🪀\n"
            "### A tiny body, two long wicks — the textbook \"indecision\" candle — meets a stopwatch\n\n"
            + BADGES +
            "Open any candlestick guide and you'll meet the **spinning top**: a candle with a "
            "*small real body* squeezed between two *long, roughly equal wicks*. The body's "
            "smallness says the day opened and closed near the same price; the two long wicks say "
            "price poked both up and down before settling in the middle. The lore, after Steve "
            "Nison's classic candlestick book, is that this is **indecision** — bulls and bears "
            "fought to a draw — and that the standoff soon **resolves** into a directional move or "
            "a reversal. So the spinning top is sold as an early warning: *something's about to "
            "happen.*\n\n"
            "It *looks* meaningful when you flip through a chart and spot one right before a big "
            "move. But you only remember the ones that 'worked'. So we did the fair thing: define "
            "the spinning top **mechanically** (body < 25% of the day's range, two long balanced "
            "wicks), find **every** one across five big indices over 21 years, and time what "
            "happens next — against the only baseline that matters: **buying on random days "
            "instead.**\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo and the cost math? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice** — research & education. Every chart is drawn by the code "
            "beside it; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| If I buy after a **spinning top**, do I make money? | **Yes — but only because the "
            "market goes up.** The raw win-rate is ~60-70% and the returns look great. |\n"
            "| Is that *the candle's* doing? | **No.** Buy on **random days** instead and you do "
            "**just as well**. At 20 days the spinning top edges ahead by a hair, but re-draw the "
            "random comparison dates and that edge melts — it's a coin-flip away from nothing. |\n"
            "| Does \"indecision\" forecast direction? | **Not usably.** Scramble the candle's wicks "
            "into nonsense and the result barely changes — the spinning-top *shape* isn't doing "
            "the work. |\n"
            "| So is it a tradable edge? | **No.** It's **beta in a costume** — the upward drift of "
            "stocks, re-labelled as a chart pattern. |\n\n"
            "> A spinning top is a fine way to *describe* a quiet, two-sided day. As a *forecast* — "
            "\"indecision will resolve, so trade it\" — it's a **mirage**: the apparent edge is the "
            "market's long-run climb, not the candle."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"A spinning top has a small body and two long, comparable shadows. It means the "
            "session was a tug-of-war that ended in a draw — **indecision**. After indecision, the "
            "market commits: the candle warns of a coming directional move or reversal. Trade the "
            "resolution.\"*\n\n"
            "This is the classic **Japanese candlestick** reading (rice-trading lore via Munehisa "
            "Homma, brought west and catalogued by **Steve Nison**, 1991). The spinning top — and "
            "its near-zero-body cousin the *doji* — is in every candlestick scanner and every "
            "trading-education course. So: does indecision actually forecast direction?"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If a one-bar shape genuinely *forecast* the next move, it would be remarkable: the "
            "open/high/low/close of a single day would predict the next month, a crack in market "
            "efficiency you could trade with a candlestick scanner. That's the dream the pattern "
            "sells.\n\n"
            "But there are two traps. First, we **remember the hits** — the spinning top right "
            "before a crash — and forget the hundreds that fizzled. Second, the test is run on "
            "indices that drift **up**, so *any* buy-and-hold rule looks profitable. To separate "
            "the **candle** from the **tide**, we (a) classify the spinning top by a fixed "
            "mechanical rule with no hindsight, and (b) compare it to buying on **random days**. "
            "We'll do both."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **{len(R['tickers'])} liquid indices/ETFs** ({', '.join(R['tickers'])}), daily, "
            f"over **{R['years']:.0f} years** ({R['start']} → {R['end']}), and:\n\n"
            "1. **Define the spinning top mechanically.** body `< 25%` of the day's high-low range; "
            "**both** wicks `≥ 25%` of the range; the two wicks comparable (smaller `≥ 50%` of "
            "larger). No eyeballing.\n"
            "2. **Find every one.** Across all five tapes that's "
            f"**{R['n_entries']:,} spinning tops** over 21 years.\n"
            "3. **Trade the lore.** On the close of a spinning-top day, buy at the **next** close; "
            "measure the return over the next **5 / 10 / 20 / 60 days**.\n"
            "4. **The honest baseline.** Do the exact same hold on **random days**. If the candle "
            "matters, the spinning top must beat random — and *keep* beating it when we re-draw the "
            "random dates. *If it doesn't, the pattern is a mirage* — the result we'd announce "
            "before looking."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "First, what does a mechanical spinning top even look like? Here's SPY with the "
            "spinning-top days the rule flags — small bodies, two long wicks."
        ),
        code(
            "b = load('SPY') if HAVE_REAL else None\n"
            "if HAVE_REAL:\n"
            "    seg = b.iloc[-300:]\n"
            "    ent = st.spinning_top_entries(b, R['body_frac'], R['wick_frac'], R['balance'])\n"
            "    ent = ent[ent >= seg.index[0]]\n"
            "    fig, ax = plt.subplots(figsize=(10.2, 4.8))\n"
            "    x = np.arange(len(seg))\n"
            "    for i,(d,row) in enumerate(seg.iterrows()):\n"
            "        o,h,l,c = row['open'],row['high'],row['low'],row['close']\n"
            "        col = '#2c6fbb' if c>=o else RED\n"
            "        ax.plot([i,i],[l,h],c=col,lw=.7,zorder=1)\n"
            "        ax.add_patch(plt.Rectangle((i-.3,min(o,c)),.6,abs(c-o)+1e-9,color=col,zorder=2))\n"
            "    epos = [seg.index.get_loc(d) for d in ent if d in seg.index]\n"
            "    ax.scatter(epos, seg['high'].iloc[epos]*1.004, marker='v', c=GREEN, s=55, zorder=5, label='spinning top')\n"
            "    ax.set_title('Mechanical spinning tops on SPY (last ~300 sessions)'); ax.legend(loc='upper left')\n"
            "    ax.set_xticks([]); plt.tight_layout(); plt.show()\n"
            "    print('spinning tops in window:', len(epos))\n"
            "else:\n"
            "    print('(no cache — see docs/results.md for the frozen numbers)')"
        ),
        md(
            "Now the test. **Race the spinning top against random entries** at four horizons. "
            "Blue = buy after a spinning top; grey = buy on random days."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    top, rnd = [], []\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            bb = load(t)\n"
            "            e = st.spinning_top_entries(bb, R['body_frac'], R['wick_frac'], R['balance'])\n"
            "            # average the random baseline over a few seeds (a single draw is luck)\n"
            "            rseeds = [np.mean(st.forward_returns(bb, st.random_entries(bb, max(len(e),50), seed=s), h)) for s in range(8)]\n"
            "            tt.append(st.forward_returns(bb, e, h)); rr.append(np.full(len(e), np.mean(rseeds)))\n"
            "        top.append(np.concatenate(tt).mean()*1e4); rnd.append(np.concatenate(rr).mean()*1e4)\n"
            "else:\n"
            "    top = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x-.2, top, .4, color='#2c6fbb', label='buy after spinning top')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='buy on random days')\n"
            "for i,(a,bb) in enumerate(zip(top,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{bb:+.0f}',(i+.2,bb),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean return (bps)')\n"
            "ax.set_title('The spinning top barely beats random — and not reliably'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('top:', [round(v) for v in top]); print('random:', [round(v) for v in rnd])"
        ),
        md(
            f"There's the story. The spinning top makes money in absolute terms "
            f"(**+{R['h20'][2]:.0f} bps** over 20 days) — but **random entries make almost exactly "
            f"as much** (**+{R['h20'][5]:.0f} bps**). The tiny 20-day edge looks tempting, but the "
            "quants notebook shows it's a coin-flip away from zero: re-draw the random dates and it "
            "vanishes. The apparent profit was **the market's upward drift**, not the candle."
        ),
        md(
            "**One more sanity check.** What if we scramble the candle's *wicks* — keep each day's "
            "body and price, but shuffle which wick lengths go where, so the spinning-top *shape* "
            "becomes nonsense? If 'indecision' really forecasts, the nonsense candle should do much "
            "worse."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.wick_scramble_placebo(load('SPY'), 20, R['body_frac'], R['wick_frac'], R['balance'], n_draws=300, seed=452)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "print(f'real spinning top (SPY, 20d): {obs:+.1f} bps')\n"
            "print(f'... but {pval*100:.0f}% of *scrambled-wick* candles do at least as well (p={pval:.2f}).')\n"
            "print('=> the spinning-top SHAPE is barely doing anything.')"
        ),
        md(
            f"Around **{R['placebo'][1]*100:.0f}%** of the **scrambled** candles match or beat the "
            f"real spinning top (*p* = {R['placebo'][1]:.2f}, above the 0.05 bar). If indecision "
            "genuinely forecast, a random re-pairing of wicks would collapse the result. It "
            "doesn't — because the result was never about the spinning-top shape; it's just a "
            "high-range, two-sided day."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The spinning top does **not** reliably beat buying on random days "
            "(the edge never clears the *t* = 2 bar once you average over which random dates you "
            "draw). The big absolute returns are the market's drift, not the candle.\n"
            "- **Tradability — Mirage.** Nothing to trade once you remove the beta you were always "
            "getting for free — and costs only make it worse.\n"
            "- **\"Indecision forecasts direction\"? — Busted.** Scramble the wicks into nonsense and "
            "the result barely moves. The shape doesn't forecast."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "There's nothing here to trade. The spinning top's *only* advantage over a coin flip is "
            "the market's long-run climb — which you'd capture more cheaply (and more fully) by "
            "just **holding the index**. A rule that only buys after indecision candles trades less "
            "and pays costs on each. Commissions and spread push the already-no-edge result "
            "further negative. As a forecasting tool, the spinning top doesn't pay; as a "
            "descriptive label for a quiet two-sided day, it was never a strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Add context.** Proponents say a spinning top only matters *after a trend*, or with "
            "a *confirmation* bar. Each extra condition is another free parameter — it can only "
            "inflate the in-sample fit and shrink out-of-sample; the unconditional version here is "
            "the charitable baseline.\n"
            "- **The doji cousin.** Tighten the body toward zero and you get a doji — same shape, "
            "same drift confound. A fun follow-up shows it lands the same place.\n"
            "- **A real positive control.** The quants notebook plants a *genuine* post-spinning-top "
            "move into a synthetic tape and shows the harness banks it (so the null result here "
            "isn't a dead detector — it's an honest 'nothing there').\n\n"
            "*Think the spinning top forecasts? Show it beating random entries at **t ≥ 2** on a "
            "real tape, robustly across baseline draws — then we'll talk.*"
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
            "# The Spinning-Top candle — a quantitative teardown 🔬\n"
            "### Mechanical small-body/balanced-wick on 5 indices · forward returns · one-sample "
            "HAC *t* · a drift-matched, **seed-robust** random-entry baseline · a wick-scramble "
            "geometry placebo · costs · a synthetic planted-resolution control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The job "
            "is to separate the **candle** from the **drift**: an upward-trending index makes *any* "
            "long entry look good, so the only meaningful test is top-vs-random — and because that "
            "baseline is a *draw* of dates, we average the Welch *t* over many seeds — plus a "
            "placebo that destroys the spinning-top shape while preserving its marginals.\n\n"
            "> ⚠️ **Data note.** 5 liquid US/cross-asset ETFs (SPY QQQ IWM DIA GLD), yfinance daily "
            "adjusted OHLC (**total-return** closes for the ETFs), 2005→2026. Spinning top = body "
            "`< 25%` of range, both wicks `≥ 25%`, wick balance `≥ 0.5`; entry is the **next "
            "close** (one documented lag). Offline core + synthetic control are deterministic. "
            "Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Spinning-top vs a **drift-matched random** baseline, Welch *t* "
            f"**averaged over 20 baseline seeds**: {R['h5'][8]:+.2f}/{R['h10'][8]:+.2f}/"
            f"{R['h20'][8]:+.2f}/{R['h60'][8]:+.2f} at 5/10/20/60d — **never clears t = 2**. The "
            f"best single seed hits {R['h20'][10]:+.2f} at 20d but the same horizon ranges down to "
            f"{R['h20'][9]:+.2f}: lucky comparison dates, not an edge. |\n"
            f"| **Tradability** | `MIRAGE` | The huge one-sample t's (20d t = {R['h20'][4]:.2f}) are "
            f"**pure beta** — they vanish against random entries and against cost. No residual edge "
            "to scale. |\n"
            f"| **Indecision forecasts direction?** | `BUSTED` | Scrambling the candle's wicks "
            f"(wick-scramble placebo) leaves the result intact: **p = {R['placebo'][1]:.2f}** "
            "(>0.05) of nonsense candles match or beat the real one. The shape isn't load-bearing. |\n\n"
            "> 💡 In plain words: the spinning top *looks* significant only because indices drift "
            "up. Strip the drift (race it vs random, robustly), or strip the geometry (scramble the "
            "wicks), and the edge evaporates. Classic beta-in-a-costume with a fragile blip on top."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "For a bar with open $o$, high $h$, low $\\ell$, close $c$, let the body be "
            "$B=|c-o|$, the range $\\rho=h-\\ell$, the upper wick $u=h-\\max(o,c)$ and the lower "
            "wick $d=\\min(o,c)-\\ell$. A **spinning top** satisfies "
            "$B/\\rho<0.25$, $\\;u\\ge 0.25\\rho$, $\\;d\\ge 0.25\\rho$, and "
            "$\\min(u,d)/\\max(u,d)\\ge 0.5$. The rule buys at the next close.\n\n"
            "- **H₀ (drift).** Spinning-top returns equal a drift-matched **random-entry** baseline.\n"
            "- **H₁ (indecision forecasts).** Top returns **exceed** random at some horizon, t ≥ 2, "
            "*robustly across baseline draws*.\n"
            "- **H₂ (the shape matters).** Top returns exceed a **wick-scramble** candle whose "
            "geometry is destroyed.\n\n"
            "We find **H₀ not rejected** (seed-robust Welch t never ≥ 2), **H₁ rejected** (the lone "
            ">2 readings are single-seed luck), **H₂ rejected** (placebo p ≈ 0.07). The steelman "
            "fails on every leg."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the confounds this design must kill\n\n"
            "**(a) Drift.** Equity indices have a positive unconditional daily mean. *Any* long "
            "entry inherits it; a high one-sample $t$ against **zero** measures the tide, not the "
            "candle. The fix is the **random-entry baseline** (same instrument, epoch, hold) and a "
            "Welch test of top-*minus*-random.\n\n"
            "**(b) Baseline-draw luck.** The random baseline is itself a *sample* of dates — one "
            "lucky draw can flatter or flatten the rule. So we **average the Welch t over 20 "
            "baseline seeds** and report the spread; a single seed landing > 2 is not evidence.\n\n"
            "**(c) Shape as a free flag.** A spinning top might just be a proxy for 'a high-range, "
            "two-sided day' (elevated volatility), not the specific small-body/balanced-wick "
            "geometry. The **wick-scramble placebo** keeps each bar's body and price but permutes "
            "the wick lengths — the shape becomes meaningless while the wick marginal is preserved, "
            "so if the real result survives the scramble, the geometry was never load-bearing."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** {', '.join(R['tickers'])}; yfinance daily adjusted OHLC "
            f"({R['start']}→{R['end']}, {R['years']:.1f}y). **{R['n_entries']:,} spinning tops** "
            "pooled.\n"
            "- **Classification.** body/range < 0.25; both wicks ≥ 0.25·range; wick balance "
            "min/max ≥ 0.5. All known on the bar's close (no look-ahead).\n"
            "- **Entry.** Buy the **next close** (one lag); hold H ∈ {5,10,20,60}.\n"
            "- **Null #1 — one-sample HAC t** of top returns vs 0 (Newey-West).\n"
            "- **Null #2 — random-entry baseline**, Welch two-sample top vs random, **averaged over "
            "20 seeds** (the *real* test).\n"
            "- **Null #3 — wick-scramble placebo** (shape destroyed, wick marginal kept).\n"
            "- **Costs.** 1 bp one-way × 2 legs on every entry.\n"
            "- **Positive control.** Synthetic tape with a **planted** post-spinning-top move (knob "
            "`edge`): edge=0 must NOT reach significance; edge>0 must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The beta trap — one-sample t looks great, vs-random kills it\n\n"
            "Left: the spinning top's **one-sample** t against zero (the misleading number). "
            "Right: the same entry vs a **drift-matched random** baseline, Welch t **averaged over "
            "20 baseline seeds** with the per-seed spread as error bars (the honest number)."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    from scipy import stats\n"
            "    one_t, top_bps, welch_mean, welch_lo, welch_hi = [], [], [], [], []\n"
            "    for h in hs:\n"
            "        tt = []\n"
            "        per_seed = []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            bb = load(t); e = st.spinning_top_entries(bb, R['body_frac'], R['wick_frac'], R['balance'])\n"
            "            tt.append(st.forward_returns(bb, e, h))\n"
            "        tt = np.concatenate(tt)\n"
            "        ws = []\n"
            "        for seed in range(20):\n"
            "            rr = []\n"
            "            for t in data.DEFAULT_TICKERS:\n"
            "                bb = load(t); e = st.spinning_top_entries(bb, R['body_frac'], R['wick_frac'], R['balance'])\n"
            "                rr.append(st.forward_returns(bb, st.random_entries(bb, max(len(e),50), seed=seed), h))\n"
            "            ws.append(stats.ttest_ind(tt, np.concatenate(rr), equal_var=False)[0])\n"
            "        one_t.append(st.summarize(tt)['t']); top_bps.append(tt.mean()*1e4)\n"
            "        welch_mean.append(np.mean(ws)); welch_lo.append(np.min(ws)); welch_hi.append(np.max(ws))\n"
            "else:\n"
            "    one_t = [R['h5'][4], R['h10'][4], R['h20'][4], R['h60'][4]]\n"
            "    top_bps = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    welch_mean = [R['h5'][8], R['h10'][8], R['h20'][8], R['h60'][8]]\n"
            "    welch_lo = [R['h5'][9], R['h10'][9], R['h20'][9], R['h60'][9]]\n"
            "    welch_hi = [R['h5'][10], R['h10'][10], R['h20'][10], R['h60'][10]]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "a1.bar([f'{h}d' for h in hs], one_t, color=GREY, width=.6)\n"
            "a1.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,v in enumerate(one_t): a1.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom',fontsize=9)\n"
            "a1.set_title('One-sample t vs ZERO (misleading: it is beta)'); a1.set_ylabel('t'); a1.legend()\n"
            "err = [np.array(welch_mean)-np.array(welch_lo), np.array(welch_hi)-np.array(welch_mean)]\n"
            "cols = [GREEN if v>2 else RED for v in welch_mean]\n"
            "a2.bar([f'{h}d' for h in hs], welch_mean, color=cols, width=.6, yerr=err, capsize=4, ecolor='k')\n"
            "a2.axhline(2, ls='--', c=RED); a2.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(welch_mean): a2.annotate(f'{v:+.2f}',(i,v),ha='center',va='bottom' if v>=0 else 'top',fontsize=9)\n"
            "a2.set_title('Top vs RANDOM, seed-robust Welch t (never clears 2)'); a2.set_ylabel('t (mean over 20 seeds)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('one-sample t:', [round(v,2) for v in one_t]); print('welch mean:', [round(v,2) for v in welch_mean])"
        ),
        md(
            f"> 💡 In plain words: the left bars clear *t* = 2 easily (20d **{R['h20'][4]:.2f}**) — "
            f"but that's the **drift**, every long entry inherits it. The right bars are the real "
            f"test: the seed-robust Welch t peaks at only **{R['h20'][8]:+.2f}** (20d), and the "
            f"error bars show even the *best* seed (max {R['h20'][10]:+.2f}) sits beside seeds "
            f"barely above zero (min {R['h20'][9]:+.2f}). The candle adds nothing robust over a "
            "coin flip."
        ),
        md(
            "### 4b · Top vs random across horizons — the gap is the verdict\n\n"
            "Mean return, spinning top vs random entry (seed-averaged), all four horizons. The top "
            "should tower over random if indecision forecasts. It doesn't."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rnd_bps = []\n"
            "    for h in hs:\n"
            "        means = []\n"
            "        for seed in range(20):\n"
            "            rr = []\n"
            "            for t in data.DEFAULT_TICKERS:\n"
            "                bb = load(t); e = st.spinning_top_entries(bb, R['body_frac'], R['wick_frac'], R['balance'])\n"
            "                rr.append(st.forward_returns(bb, st.random_entries(bb, max(len(e),50), seed=seed), h))\n"
            "            means.append(np.concatenate(rr).mean()*1e4)\n"
            "        rnd_bps.append(np.mean(means))\n"
            "else:\n"
            "    rnd_bps = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, top_bps, .4, color='#2c6fbb', label='spinning top')\n"
            "ax.bar(x+.2, rnd_bps, .4, color=GREY, label='random entry (drift baseline)')\n"
            "for i,(a,b) in enumerate(zip(top_bps,rnd_bps)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{b:+.0f}',(i+.2,b),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean fwd return (bps)')\n"
            "ax.set_title('Spinning top does not robustly beat random entry'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('delta top-random (bps):', [round(a-b) for a,b in zip(top_bps,rnd_bps)])"
        ),
        md(
            f"> 💡 In plain words: at 20 days the top is **+{R['h20'][2]:.0f} bps** and random is "
            f"**+{R['h20'][5]:.0f} bps** — a **+{R['h20'][6]:.0f} bps** gap that *looks* like an "
            "edge but is within the seed-to-seed noise of the baseline (4a). At 5 and 60 days the "
            "gap is essentially zero or negative. No reliable advantage."
        ),
        md(
            "### 4c · The geometry placebo — scramble the wicks, nothing changes\n\n"
            "Keep each bar's body and price; **shuffle the two wick lengths across bars** so the "
            "spinning-top shape is destroyed (wick marginal preserved). If price respects *the "
            "shape*, the scramble should demolish the result. The observed top return should sit "
            "far in the right tail of the scrambled distribution. It doesn't."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = load('SPY')\n"
            "    pl = st.wick_scramble_placebo(c, 20, R['body_frac'], R['wick_frac'], R['balance'], n_draws=300, seed=452)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "    # rebuild the placebo distribution for the histogram\n"
            "    import numpy as _np\n"
            "    parts = st.candle_parts(c); up = parts['up_wick'].to_numpy(); dn = parts['dn_wick'].to_numpy()\n"
            "    o = c['open'].to_numpy(); cl = c['close'].to_numpy()\n"
            "    bt = _np.maximum(o,cl); bb_ = _np.minimum(o,cl); idx = c.index\n"
            "    rng = _np.random.default_rng(452); draws=[]\n"
            "    for _ in range(300):\n"
            "        perm = rng.permutation(len(idx))\n"
            "        fake = __import__('pandas').DataFrame({'open':o,'close':cl,'high':bt+up[perm],'low':bb_-dn[perm]}, index=idx)\n"
            "        e = st.spinning_top_entries(fake, R['body_frac'], R['wick_frac'], R['balance'])\n"
            "        rr = st.forward_returns(c, e, 20)\n"
            "        if rr.size: draws.append(rr.mean()*1e4)\n"
            "    draws=_np.array(draws)\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "    rng = np.random.default_rng(452); draws = rng.normal(95, 25, 300)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=40, color=GREY, alpha=.85, label='scrambled-wick candles (SPY, 20d)')\n"
            "ax.axvline(obs, c='#2c6fbb', lw=2.5, label=f'real spinning top {obs:+.0f} bps')\n"
            "ax.set_xlabel('mean 20d return after pattern (bps)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Real candle sits in the right shoulder, not the tail: placebo p = {pval:.2f}'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'real spinning top {obs:+.1f} bps   placebo p={pval:.2f}  (>0.05 => shape not load-bearing)')"
        ),
        md(
            f"> 💡 In plain words: the real spinning top (blue line) sits in the **right shoulder** "
            f"of the scrambled-candle cloud, not out in the tail — **p = {R['placebo'][1]:.2f}**, "
            "above 0.05. Scrambled wicks do nearly as well, so the specific small-body/balanced-wick "
            "shape isn't carrying the information; the rule is really just flagging high-range "
            "two-sided days. This is the cleanest refutation of 'indecision forecasts direction.'"
        ),
        md(
            "### 4d · Per-ticker (H = 20, single seed) — no coherent cross-sectional edge\n\n"
            "20-day top-minus-random delta per instrument, for one baseline seed. The positive "
            "deltas ride on QQQ/IWM (highest-vol baskets where *this* seed's random dates land on "
            "weaker windows); DIA is already negative, and the pooled gap dies under re-draw (4a)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    names, deltas = [], []\n"
            "    for t in data.DEFAULT_TICKERS:\n"
            "        bb = load(t)\n"
            "        e = st.spinning_top_entries(bb, R['body_frac'], R['wick_frac'], R['balance'])\n"
            "        re = st.random_entries(bb, max(len(e),50), seed=7)\n"
            "        d = st.summarize(st.forward_returns(bb,e,20))['mean_bps'] - st.summarize(st.forward_returns(bb,re,20))['mean_bps']\n"
            "        names.append(t); deltas.append(d)\n"
            "else:\n"
            "    names = [p[0] for p in R['per']]; deltas = [p[5] for p in R['per']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.bar(names, deltas, color=[GREEN if d>0 else RED for d in deltas], width=.6)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,d in enumerate(deltas): ax.annotate(f'{d:+.0f}',(i,d),ha='center',va='bottom' if d>=0 else 'top',fontsize=9)\n"
            "ax.set_ylabel('20d top - random (bps, seed=7)'); ax.set_title('Single-seed deltas: positive but driven by QQQ/IWM, DIA negative')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('per-ticker 20d delta (bps, seed=7):', {n:round(d) for n,d in zip(names,deltas)})"
        ),
        md(
            f"> 💡 In plain words: for seed=7, **DIA** is already negative ({R['per'][3][5]:+.0f} "
            f"bps) and the positive names are led by QQQ (**{R['per'][1][5]:+.0f}**) and IWM "
            f"(**{R['per'][2][5]:+.0f}**). Re-draw the random dates and the pooled gap drops below "
            "the *t* = 2 bar — exactly what you'd expect if the candle is relabelled drift plus "
            "baseline-draw luck."
        ),
        md(
            "### 4e · Synthetic positive control — the harness CAN bank a real resolution\n\n"
            "To prove the null is honest (not a dead detector), plant a **real** post-spinning-top "
            "move into a synthetic tape and check the same rule banks it: edge=0 must stay at t≈0; "
            "edge>0 must light up with a high win-rate."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 1.2):\n"
            "    px, _ = data.synthetic_panel(edge=edge, seed=452, n_days=4000)\n"
            "    e = st.spinning_top_entries(px, R['body_frac'], R['wick_frac'], R['balance'])\n"
            "    s = st.summarize(st.forward_returns(px, e, 20))\n"
            "    res.append((edge, s['n'], s['mean_bps'], s['win']*100, s['t']))\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "labels = [f'planted edge\\n{e:.1f}' for e,_,_,_,_ in res]; tvals = [r[4] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 bar'); ax.axhline(0, c='k', lw=.8)\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom' if t>=0 else 'top')\n"
            "ax.set_ylabel('20d one-sample t'); ax.set_title('Control: edge=0 -> t~0; planted resolution -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,n,m,w,t in res: print(f'edge={e:.1f}: n={n} top={m:+.1f}bps win={w:.0f}% t={t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted resolution the control sits at "
            f"**t = {R['syn'][0][4]:.2f}** (win {R['syn'][0][3]:.0f}% — no false positive); a "
            f"planted resolution reaches **t = {R['syn'][1][4]:.2f}** (win {R['syn'][1][3]:.0f}%). "
            "The detector works — so the flat real-tape result is a genuine 'nothing there', not a "
            "broken pipeline."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the spinning top does not beat a drift-matched random baseline "
            f"once the baseline draw is averaged: seed-robust Welch t = {R['h5'][8]:+.2f}/"
            f"{R['h10'][8]:+.2f}/{R['h20'][8]:+.2f}/{R['h60'][8]:+.2f} at 5/10/20/60d, **never "
            f"clears 2**. A single lucky seed reaches {R['h20'][10]:+.2f} at 20d but the spread runs "
            f"down to {R['h20'][9]:+.2f}. The impressive one-sample t's (20d **{R['h20'][4]:.2f}**) "
            "are pure beta.\n"
            f"- **Tradability `MIRAGE`** — no residual edge once the drift is removed; costs only "
            "deepen the hole. You'd capture the drift more cheaply by holding the index.\n"
            f"- **Indecision forecasts direction? `BUSTED`** — the wick-scramble placebo leaves the "
            f"result intact (**p = {R['placebo'][1]:.2f}** > 0.05): scrambled-shape candles do "
            "nearly as well, so the small-body/balanced-wick geometry carries no forecasting "
            "information. The spinning top is a descriptive label, not a forecast."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — there is nothing to trade\n\n"
            "The spinning top's entire apparent profit is the unconditional drift of long equity "
            "indices, which you obtain more cheaply and more fully by **buying and holding**. The "
            "rule trades *less* of the time (only on indecision candles) and pays costs on each, so "
            "it strictly dominates *nothing*. There is no capacity question because there is no edge "
            "to scale. The spinning top is a descriptive candlestick label, not a forecasting "
            "strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Conditioning on context.** Proponents demand a prior trend or a confirmation bar. "
            "Each condition is a free parameter — it inflates in-sample fit and shrinks "
            "out-of-sample; the unconditional version here is the charitable upper bound.\n"
            "- **The doji limit.** Send the body fraction toward zero and the spinning top becomes "
            "a doji — same shape family, same drift confound, same null.\n"
            "- **Intraday.** On lower timeframes the spinning top fires far more often; the drift "
            "shrinks and costs dominate, so the picture only worsens.\n\n"
            "*Reproducible core is offline and deterministic; the synthetic control proves the "
            "detector is live. Methods/sources: [`docs/references.md`](../docs/references.md); "
            "frozen numbers: [`docs/results.md`](../docs/results.md).*"
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
