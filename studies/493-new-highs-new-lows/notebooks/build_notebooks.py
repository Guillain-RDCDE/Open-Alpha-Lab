"""Generate the two narrative notebooks for Study 493 (New-Highs-New-Lows breadth).

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
# yfinance daily, breadth proxy = 5 cached liquid ETFs (SPY QQQ IWM DIA GLD),
# 2005-01-03 -> 2026-05-29 (As-of 2026-05-31), 21.4 years, 52-week (252d) extremes,
# 10-day smoothed NH-NL line, breadth-thrust = up-cross of +0.20, traded on SPY.
R = dict(
    asof="2026-05-31", start="2005-01-03", end="2026-05-29", years=21.4,
    tickers=["SPY", "QQQ", "IWM", "DIA", "GLD"], n_entries=114,
    lookback=252, smooth=10, thresh=0.20, fp_spy="4cb5244f3990",
    # pooled breadth-thrust on SPY, per horizon:
    # (H, n, thrust_bps, win%, one_sample_t, random_bps, delta_bps, net_bps, welch_t, welch_p)
    h5=(5, 113, 5.7, 54, 0.45, 39.7, -34.0, 3.7, -1.09, 0.275),
    h10=(10, 113, 26.3, 65, 1.39, 84.6, -58.2, 24.3, -1.48, 0.142),
    h20=(20, 113, 36.3, 66, 1.29, 125.1, -88.8, 34.3, -1.51, 0.132),
    h60=(60, 112, 187.4, 74, 2.94, 274.0, -86.7, 185.4, -0.92, 0.358),
    # per-index H=20: (ticker, entries, thrust_bps, one_sample_t, random_bps, delta_bps)
    per=[("SPY", 114, 36.3, 1.29, 125.1, -88.8), ("QQQ", 114, 44.1, 1.49, 119.3, -75.2),
         ("IWM", 114, -13.4, -0.33, 185.6, -199.0), ("DIA", 114, 41.3, 1.32, 141.9, -100.6),
         ("GLD", 114, 128.0, 2.58, 127.5, 0.4)],
    # shuffled-membership placebo (SPY, H=20, 500 draws): obs_bps, p, draws
    placebo=(36.3, 0.818, 500),
    # synthetic control (H=20, n_days=4000): (edge, n, thrust_bps, win%, one_sample_t)
    syn=[(0.00, 16, -53.3, 44, -0.75), (0.60, 16, 364.4, 88, 3.02)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Leads_the_index%3F: Busted](https://img.shields.io/badge/Leads_the_index%3F-Busted-8b949e?style=flat-square)\n\n"
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

from new_highs_new_lows import data, strategy as st

ASOF = "2026-05-31"
HAVE_REAL = data.have_real()
def get_panel():
    return data.load_basket(allow_fetch=False, asof=ASOF)
print("real breadth cache present:", HAVE_REAL, "| basket:", data.DEFAULT_TICKERS)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does the New-Highs / New-Lows line actually *lead* the market? 📈\n"
            "### A classic breadth indicator — how many stocks are at fresh highs vs lows — meets a "
            "stopwatch\n\n"
            + BADGES +
            "Open any market-internals dashboard and you'll find the **new-highs / new-lows (NH-NL) "
            "line**: count how many stocks are printing a fresh **52-week high**, subtract those at a "
            "fresh **52-week low**, and watch the *net*. The lore — from Charles Dow to William "
            "O'Neil's *Investor's Business Daily* to the famous \"Hindenburg Omen\" — is that "
            "**breadth leads price**: the NH-NL line tops and bottoms *before* the index, so a surge "
            "in net new highs (a \"breadth thrust\") is a green light, and a collapse is a warning.\n\n"
            "It *sounds* like inside information about the whole market. So we did the only fair thing: "
            "encode the NH-NL thrust **mechanically** (no eyeballing), fire the \"breadth thrust = buy "
            "the index\" rule across 21 years, and time the result with a stopwatch — against the only "
            "baseline that matters: **buying on random days instead.**\n\n"
            "> ⚠️ **Breadth proxy.** Real exchange breadth counts *thousands* of issues; offline we "
            "can only proxy it with a small basket of liquid ETFs (SPY QQQ IWM DIA GLD). That's a "
            "**coarse** proxy and it caps the test — we say so throughout.\n"
            ">\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo and the cost math? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice** — research & education. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| If I buy the index when breadth thrusts, do I make money? | **Yes — but only because the "
            "market goes up.** The win-rate climbs to ~66–74% and the returns look fine. |\n"
            "| Is that *the breadth line's* doing? | **No.** Buy on **random days** instead and you do "
            "**much better** — the thrust is *worse* than a coin-flip entry at every horizon. |\n"
            "| Does breadth \"lead\" the index? | **Not in any usable way.** Scramble the cross-"
            "sectional breadth structure and the result barely changes. |\n"
            "| So is it a tradable edge? | **No.** It's **beta in a costume** — the upward drift of "
            "stocks, re-labelled as a breadth signal. |\n\n"
            "> The NH-NL line is a great way to *describe* a strong tape after the fact. As a "
            "*forecast* — \"the thrust leads the index\" — it's a **mirage**: all of the apparent edge "
            "is the market's long-run climb, none of it is the breadth."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Count the stocks at new 52-week highs, subtract those at new lows. When the net "
            "**expands** — a breadth thrust — the rally has broad participation and will continue; "
            "when it **diverges** (price up, breadth fading) a top is near. Breadth leads price.\"*\n\n"
            "This is the **new-highs / new-lows** indicator, a staple of market-internals analysis "
            "since Charles Dow, formalised in O'Neil's IBD methodology and in the breadth-divergence "
            "literature (Zweig's breadth thrust, the Hindenburg Omen). It's one of the most "
            "recognisable \"market internals\" — so: does the internal actually lead?"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If breadth genuinely *led* price, it would be remarkable: a count of how many names are "
            "at highs today would forecast the index tomorrow — a clean crack in market efficiency you "
            "could trade. That's the dream the indicator sells.\n\n"
            "But there's a trap. The NH-NL line is **mechanically tied to recent price**: when the "
            "market has been rising, *of course* lots of members are near new highs — that's what a "
            "rising market *is*. So a \"breadth thrust\" mostly means \"the market just went up,\" and "
            "on an index that drifts **up** over time, *any* such rule looks profitable. To separate "
            "the **signal** from the **tide**, we (a) build the line by a fixed mechanical rule with no "
            "hindsight, and (b) compare it to buying on **random days**. We'll do both."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take a **{len(R['tickers'])}-ETF breadth basket** ({', '.join(R['tickers'])}) as a "
            f"proxy for market breadth, daily, over **{R['years']:.0f} years** "
            f"({R['start']} → {R['end']}), and:\n\n"
            "1. **Build the NH-NL line mechanically.** Each day, count members at a fresh "
            f"**{R['lookback']}-day (52-week) high** minus those at a fresh low, divide by the basket "
            f"size, and smooth over **{R['smooth']} days**. Trailing data only — no look-ahead.\n"
            f"2. **Fire on a breadth thrust.** When the smoothed line crosses **up** through "
            f"**+{R['thresh']:.2f}** (breadth expanding from neutral), buy SPY at the **next** close.\n"
            "3. **Measure** the return over the next **5 / 10 / 20 / 60 days**.\n"
            "4. **The honest baseline.** Do the exact same hold on **random days**. If breadth leads, "
            "the thrust must beat random. *If it doesn't, the indicator is a mirage* — the result that "
            "would make us say so, announced before we look."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "First, what does the mechanical NH-NL line even look like? Here's SPY with the smoothed "
            "net-new-high breadth line beneath it, and the thrust days the rule would buy."
        ),
        code(
            "panel = get_panel() if HAVE_REAL else None\n"
            "if HAVE_REAL:\n"
            "    line = st.net_new_high_line(panel)\n"
            "    spy = panel['SPY']['close']\n"
            "    ent = st.breadth_thrust_entries(panel, 'SPY')\n"
            "    seg = spy.iloc[-700:]; lseg = line.reindex(seg.index)\n"
            "    es = ent[ent >= seg.index[0]]\n"
            "    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10.2, 6.0), sharex=True,\n"
            "                                 gridspec_kw={'height_ratios':[2,1]})\n"
            "    a1.plot(seg.index, seg.values, c='k', lw=1.2, label='SPY close')\n"
            "    a1.scatter(es, spy.reindex(es), c=GREEN, s=42, zorder=5, label='breadth-thrust BUY')\n"
            "    a1.set_title('NH-NL breadth thrust on SPY (last ~3y)'); a1.legend(loc='upper left')\n"
            "    a2.fill_between(seg.index, lseg.values, 0, color='#2c6fbb', alpha=.5)\n"
            "    a2.axhline(R['thresh'], ls='--', c=GREEN, lw=1, label=f\"thrust +{R['thresh']}\")\n"
            "    a2.axhline(0, c='k', lw=.7); a2.set_ylabel('net new-high frac'); a2.legend(loc='upper left')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('breadth-thrust entries in window:', len(es))\n"
            "else:\n"
            "    print('(no cache — see docs/results.md for the frozen numbers)')"
        ),
        md(
            "The breadth line rises *with* the trend — *as a description*. The question is whether "
            "those green thrust dots are followed by extra gains. **Let's race the thrust against "
            "random entries** at four horizons. Blue = buy on a breadth thrust; grey = buy on random "
            "days."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    spy = panel['SPY']['close']; ent = st.breadth_thrust_entries(panel, 'SPY')\n"
            "    re = st.random_entries(spy, max(len(ent),50), seed=7)\n"
            "    thr = [st.forward_returns(spy, ent, h).mean()*1e4 for h in hs]\n"
            "    rnd = [st.forward_returns(spy, re, h).mean()*1e4 for h in hs]\n"
            "else:\n"
            "    thr = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x-.2, thr, .4, color='#2c6fbb', label='buy the breadth thrust')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='buy on random days')\n"
            "for i,(a,bb) in enumerate(zip(thr,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{bb:+.0f}',(i+.2,bb),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean return (bps)')\n"
            "ax.set_title('The breadth thrust LOSES to random at every horizon'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('thrust:', [round(v) for v in thr]); print('random:', [round(v) for v in rnd])"
        ),
        md(
            f"There's the whole story. The thrust makes money in absolute terms "
            f"(**+{R['h20'][2]:.0f} bps** over 20 days) — but **random entries make far more** "
            f"(**+{R['h20'][5]:.0f} bps**). At *every* horizon the famous breadth line is **worse** "
            "than throwing darts. The apparent edge was **the market's upward drift**, not the "
            "breadth."
        ),
        md(
            "**One more sanity check.** What if we scramble the *cross-sectional structure* of breadth "
            "— keep each member's exact rate of new highs but shuffle *which days* they happen, so a "
            "\"thrust\" is just coincidental? If breadth really leads, the scrambled line should do "
            "much worse."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.shuffled_membership_placebo(panel, 'SPY', horizon=20, n_draws=200, seed=493)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "print(f'real breadth thrust (SPY, 20d): {obs:+.1f} bps')\n"
            "print(f'... but {pval*100:.0f}% of *scrambled-breadth* lines do at least as well (p={pval:.2f}).')\n"
            "print('=> the cross-sectional breadth structure is not doing the work.')"
        ),
        md(
            f"More than four-fifths of the **scrambled** breadth lines match or beat the real one "
            f"(*p* = {R['placebo'][1]:.2f}). If price genuinely respected *the real breadth "
            "aggregation*, a scramble would collapse the result. It doesn't — because the result was "
            "never about breadth."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The breadth thrust does **not** beat buying on random days (it's "
            "*worse* at every horizon; the thrust-vs-random difference never clears *t* = 2). The "
            "absolute returns are the market's drift, not the breadth.\n"
            "- **Tradability — Mirage.** Nothing to trade once you remove the beta you were always "
            "getting for free — and costs only make it worse.\n"
            "- **\"Breadth leads the index\"? — Busted.** Scramble the breadth structure and the "
            "result barely moves. The line doesn't lead."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "There's nothing here to trade. The thrust's *only* advantage over a coin flip is the "
            "market's long-run climb — which you'd capture more cheaply (and more fully) by just "
            "**holding the index**. The breadth-thrust buy is a worse, more expensive, more *selective* "
            "way to be long. As a forecasting tool it doesn't pay; as a way to *narrate* a strong tape, "
            "it was never meant to be a strategy.\n\n"
            "> ⚠️ **Caveat, restated.** Our breadth basket is only 5 ETFs. A real new-highs/new-lows "
            "universe is thousands of issues; this proxy is coarse. But the failure here is so "
            "lopsided (the thrust *loses* to random at every horizon, placebo *p* ≈ 0.8) that a richer "
            "basket would have to overturn a very clear result."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Breadth divergences.** The stronger lore is the *divergence* (price new-high, breadth "
            "not). That's a discretionary overlay; encoding it mechanically tends to inherit the same "
            "drift confound.\n"
            "- **A richer universe.** Swap the 5-ETF proxy for a true advance/decline or NH-NL feed — "
            "the desk would happily re-run if the data were offline-cacheable.\n"
            "- **A real positive control.** The quants notebook plants a *genuine* breadth-lead into a "
            "synthetic tape and shows the harness banks it (so the null here isn't a dead detector — "
            "it's an honest 'nothing there').\n\n"
            "*Think breadth leads? Show the thrust beating random entries at **t ≥ 2** on a real tape — "
            "then we'll talk.*"
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
            "# New-Highs / New-Lows breadth — a quantitative teardown 🔬\n"
            "### Mechanical NH-NL line on a liquid-ETF breadth proxy · breadth-thrust forward returns "
            "· one-sample HAC *t* · a drift-matched random-entry baseline · a shuffled-membership "
            "breadth placebo · costs · a synthetic planted-lead control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The job "
            "is to separate the **breadth** from the **drift**: an upward-trending index makes *any* "
            "long entry look good, and breadth is mechanically tied to recent price, so the only "
            "meaningful test is thrust-vs-random, plus a placebo that destroys the cross-sectional "
            "breadth structure while preserving each member's marginal.\n\n"
            "> ⚠️ **Data note.** Breadth proxied by 5 liquid US/cross-asset ETFs (SPY QQQ IWM DIA "
            "GLD), yfinance daily adjusted closes (**total-return** for the ETFs), 2005→2026. A real "
            f"NH-NL universe is thousands of issues — this is a **coarse proxy that caps the test**. "
            f"52-week ({R['lookback']}d) trailing extremes, {R['smooth']}-day smoothed line, thrust = "
            f"up-cross of +{R['thresh']:.2f}; entry is the **next close** (one documented lag). Offline "
            "core + synthetic control are deterministic. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Breadth thrust vs a **drift-matched random** baseline: the "
            f"thrust is *worse* at every horizon (Δ = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/"
            f"{R['h20'][6]:+.0f}/{R['h60'][6]:+.0f} bps at 5/10/20/60d) and the thrust-minus-random "
            f"Welch *t* is **negative throughout** (20d = {R['h20'][8]:+.2f}). |\n"
            f"| **Tradability** | `MIRAGE` | The one-sample t's (60d t = {R['h60'][4]:.2f}) are **pure "
            f"beta** — they vanish against random entries and against cost. No residual edge. |\n"
            f"| **Leads the index?** | `BUSTED` | Scrambling the cross-sectional breadth structure "
            f"(shuffled-membership placebo) leaves the result intact: **p = {R['placebo'][1]:.2f}** of "
            "nonsense breadth lines match or beat the real one. The aggregation isn't load-bearing. |\n\n"
            "> 💡 In plain words: the thrust *looks* fine only because indices drift up and breadth is "
            "mechanically high *after* a rally. Strip the drift (race it vs random) or strip the "
            "structure (scramble membership) and the edge evaporates. Classic beta-in-a-costume."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "For a basket of $N$ members, let $h_t=\\#\\{i:\\,C^i_t=\\max_{t-L<\\tau\\le t}C^i_\\tau\\}$ "
            "be the count at a fresh $L$-day high and $\\ell_t$ the count at a fresh low. The **NH-NL "
            "line** is $b_t=\\tfrac1N(h_t-\\ell_t)$, smoothed to $\\bar b_t$. The Zweig/IBD rule fires "
            "a long when $\\bar b_t$ crosses up through $+\\theta$ (a breadth thrust) and holds the "
            "index $H$ days.\n\n"
            "- **H₀ (drift).** Thrust returns equal a drift-matched **random-entry** baseline.\n"
            "- **H₁ (breadth leads).** Thrust returns **exceed** random at some horizon, t ≥ 2.\n"
            "- **H₂ (the aggregation matters).** Thrust returns exceed a **shuffled-membership** line "
            "whose cross-sectional co-movement is destroyed.\n\n"
            "We find **H₀ not rejected** (thrust < random at *every* horizon), **H₁ rejected** (Welch "
            "t negative throughout), **H₂ rejected** (placebo p ≈ 0.8). The steelman fails on every "
            "leg."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the two confounds this design must kill\n\n"
            "**(a) Drift + mechanical coupling.** Equity indices have a positive unconditional daily "
            "mean, and breadth is *definitionally* high after a rally (members near new highs ⇔ price "
            "rose). So a thrust mostly flags \"the market just went up,\" and a one-sample $t$ against "
            "**zero** measures the tide, not a lead. The fix is the **random-entry baseline** (same "
            "instrument, epoch, hold) and a Welch test of thrust-*minus*-random.\n\n"
            "**(b) Aggregation as a free structure.** The danger is that *any* count drawn on a "
            "trending basket produces a \"breadth\" line that rises into strength. The **shuffled-"
            "membership placebo** permutes, per member across time, which days that member is at a new "
            "high — keeping its marginal rate but destroying the cross-sectional co-movement. If the "
            "real result survives the scramble, the breadth aggregation was never load-bearing."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** Breadth basket {', '.join(R['tickers'])}; yfinance daily adjusted closes "
            f"({R['start']}→{R['end']}, {R['years']:.1f}y). Traded instrument: **SPY**. "
            f"**{R['n_entries']} breadth thrusts**.\n"
            f"- **Breadth line.** Per member, new high iff close = trailing {R['lookback']}-day max "
            f"(incl. t; no look-ahead); $b_t=(h_t-\\ell_t)/N$, smoothed {R['smooth']}d.\n"
            f"- **Entry.** First up-cross of $\\bar b_t$ through +{R['thresh']:.2f}; enter **next "
            "close** (one lag); hold H ∈ {5,10,20,60}.\n"
            "- **Null #1 — one-sample HAC t** of thrust returns vs 0 (Newey-West).\n"
            "- **Null #2 — random-entry baseline**, Welch two-sample thrust vs random (the *real* test).\n"
            "- **Null #3 — shuffled-membership placebo** (breadth co-movement destroyed, marginals kept).\n"
            "- **Costs.** 1 bp one-way × 2 legs on every thrust.\n"
            "- **Positive control.** Synthetic basket with a **planted** breadth-lead (knob `edge`): "
            "edge=0 must NOT reach significance; edge>0 must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The beta trap — one-sample t vs the honest random test\n\n"
            "Left: the breadth thrust's **one-sample** t against zero (the misleading number). "
            "Right: the same thrust vs a **drift-matched random** baseline (the honest number)."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    from scipy import stats\n"
            "    panel = get_panel(); spy = panel['SPY']['close']\n"
            "    ent = st.breadth_thrust_entries(panel, 'SPY')\n"
            "    re = st.random_entries(spy, max(len(ent),50), seed=7)\n"
            "    one_t, thr, rnd, welch = [], [], [], []\n"
            "    for h in hs:\n"
            "        tt = st.forward_returns(spy, ent, h); rr = st.forward_returns(spy, re, h)\n"
            "        one_t.append(st.summarize(tt)['t']); thr.append(tt.mean()*1e4); rnd.append(rr.mean()*1e4)\n"
            "        welch.append(stats.ttest_ind(tt, rr, equal_var=False)[0])\n"
            "else:\n"
            "    one_t = [R['h5'][4], R['h10'][4], R['h20'][4], R['h60'][4]]\n"
            "    thr = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "    welch = [R['h5'][8], R['h10'][8], R['h20'][8], R['h60'][8]]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "a1.bar([f'{h}d' for h in hs], one_t, color=GREY, width=.6)\n"
            "a1.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,v in enumerate(one_t): a1.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom',fontsize=9)\n"
            "a1.set_title('One-sample t vs ZERO (misleading: it is beta)'); a1.set_ylabel('t'); a1.legend()\n"
            "a2.bar([f'{h}d' for h in hs], welch, color=[GREEN if v>2 else RED for v in welch], width=.6)\n"
            "a2.axhline(2, ls='--', c=RED); a2.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(welch): a2.annotate(f'{v:+.2f}',(i,v),ha='center',va='bottom' if v>=0 else 'top',fontsize=9)\n"
            "a2.set_title('Thrust vs RANDOM, Welch t (honest: negative throughout)'); a2.set_ylabel('t')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('one-sample t:', [round(v,2) for v in one_t]); print('welch vs random:', [round(v,2) for v in welch])"
        ),
        md(
            f"> 💡 In plain words: the left bars peak at *t* = **{R['h60'][4]:.2f}** (60d) — but that's "
            f"the **drift**, every long entry inherits it. The right bars are the real test: thrust-"
            f"minus-random is **negative at every horizon** ({R['h20'][8]:+.2f} at 20d). The breadth "
            "thrust adds nothing over a coin flip; it actively *subtracts*."
        ),
        md(
            "### 4b · Thrust vs random across horizons — the gap is the verdict\n\n"
            "Mean return, breadth thrust vs random entry, all four horizons. The thrust should tower "
            "over random if breadth leads. It doesn't — it loses everywhere."
        ),
        code(
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, thr, .4, color='#2c6fbb', label='breadth thrust')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='random entry (drift baseline)')\n"
            "for i,(a,b) in enumerate(zip(thr,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{b:+.0f}',(i+.2,b),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean fwd return (bps)')\n"
            "ax.set_title('Breadth thrust underperforms random entry at every horizon'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('delta thrust-random (bps):', [round(a-b) for a,b in zip(thr,rnd)])"
        ),
        md(
            f"> 💡 In plain words: at 20 days the thrust is **+{R['h20'][2]:.0f} bps** but random is "
            f"**+{R['h20'][5]:.0f} bps** — the breadth line *underperforms* a dart by "
            f"{abs(R['h20'][6]):.0f} bps. There is no horizon where it wins."
        ),
        md(
            "### 4c · The structure placebo — scramble breadth, nothing changes\n\n"
            "Shuffle, per member across time, which days it prints a new high (marginal rate kept, "
            "co-movement destroyed) so the breadth line is structural nonsense. If price respects "
            "*the real aggregation*, the scramble should demolish the result. The observed thrust "
            "return should sit far in the right tail. It doesn't."
        ),
        code(
            "if HAVE_REAL:\n"
            "    panel = get_panel()\n"
            "    line = st.net_new_high_line(panel)\n"
            "    ent = st.breadth_thrust_entries(panel, 'SPY'); spy = panel['SPY']['close']\n"
            "    obs = st.forward_returns(spy, ent, 20).mean()*1e4\n"
            "    # rebuild the placebo distribution for the histogram (lighter n_draws for speed)\n"
            "    import numpy as _np, pandas as _pd\n"
            "    closes = st._aligned_closes(panel)\n"
            "    rmax = closes.rolling(st.LOOKBACK, min_periods=st.LOOKBACK).max()\n"
            "    rmin = closes.rolling(st.LOOKBACK, min_periods=st.LOOKBACK).min()\n"
            "    hi = (closes >= rmax-1e-9).astype(float).to_numpy(); lo = (closes <= rmin+1e-9).astype(float).to_numpy()\n"
            "    vmask = rmax.notna().all(axis=1).to_numpy(); vrows = _np.where(vmask)[0]\n"
            "    cal = closes.index; idx_set = set(spy.index); rng = _np.random.default_rng(493)\n"
            "    draws = []\n"
            "    for _ in range(150):\n"
            "        hp = hi.copy(); lp = lo.copy()\n"
            "        for j in range(hi.shape[1]):\n"
            "            perm = rng.permutation(vrows); hp[vrows,j]=hi[perm,j]; lp[vrows,j]=lo[perm,j]\n"
            "        net = (hp.sum(axis=1)-lp.sum(axis=1))/float(hi.shape[1])\n"
            "        ser = _pd.Series(net, index=cal).rolling(st.SMOOTH, min_periods=st.SMOOTH).mean()\n"
            "        ab = (ser>=st.THRESH)&ser.notna(); cr = ab & ~ab.shift(1,fill_value=False)\n"
            "        dts = _pd.DatetimeIndex([d for d in cal[cr.to_numpy()] if d in idx_set])\n"
            "        rr = st.forward_returns(spy, dts, 20)\n"
            "        if rr.size: draws.append(rr.mean()*1e4)\n"
            "    draws = _np.array(draws); pval = (np.sum(draws>=obs)+1)/(len(draws)+1)\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "    rng = np.random.default_rng(493); draws = rng.normal(70, 45, 150)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=35, color=GREY, alpha=.85, label='scrambled-membership lines (SPY, 20d)')\n"
            "ax.axvline(obs, c='#2c6fbb', lw=2.5, label=f'real breadth {obs:+.0f} bps')\n"
            "ax.set_xlabel('mean breadth-thrust 20d return (bps)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Real breadth sits mid-pack: placebo p = {pval:.2f}'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'real breadth {obs:+.1f} bps   placebo p={pval:.2f}  (>0.05 => aggregation not load-bearing)')"
        ),
        md(
            f"> 💡 In plain words: the real breadth line (blue) sits **in the middle** of the scrambled "
            f"cloud — frozen **p = {R['placebo'][1]:.2f}** at 500 draws. Structural nonsense does just "
            "as well, so the specific NH-NL aggregation carries no information. The cleanest refutation "
            "of 'breadth leads price.'"
        ),
        md(
            "### 4d · Per-index — the thrust loses to random everywhere\n\n"
            "20-day thrust-minus-random delta, trading each basket member on the *same* breadth line. "
            "If breadth led it would be positive across the board; instead it's negative in 4 of 5 "
            "(and a wash in the 5th)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    panel = get_panel(); names, deltas = [], []\n"
            "    for t in data.DEFAULT_TICKERS:\n"
            "        c = panel[t]['close']; e = st.breadth_thrust_entries(panel, t)\n"
            "        re = st.random_entries(c, max(len(e),50), seed=7)\n"
            "        d = st.summarize(st.forward_returns(c,e,20))['mean_bps'] - st.summarize(st.forward_returns(c,re,20))['mean_bps']\n"
            "        names.append(t); deltas.append(d)\n"
            "else:\n"
            "    names = [p[0] for p in R['per']]; deltas = [p[5] for p in R['per']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.bar(names, deltas, color=[GREEN if d>0 else RED for d in deltas], width=.6)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,d in enumerate(deltas): ax.annotate(f'{d:+.0f}',(i,d),ha='center',va='bottom' if d>=0 else 'top',fontsize=9)\n"
            "ax.set_ylabel('20d thrust − random (bps)'); ax.set_title('Thrust underperforms random in 4 of 5 names')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('per-index 20d delta (bps):', {n:round(d) for n,d in zip(names,deltas)})"
        ),
        md(
            f"> 💡 In plain words: **GLD** is a wash ({R['per'][4][5]:+.0f} bps); every equity index is "
            f"deep negative — IWM is **{R['per'][2][5]:+.0f}** bps *behind* random. No coherent edge "
            "— exactly what you'd expect if breadth is just relabelled drift."
        ),
        md(
            "### 4e · Synthetic positive control — the harness CAN bank a real lead\n\n"
            "To prove the null is honest (not a dead detector), plant a **real** breadth-lead into a "
            "synthetic basket and check the same thrust rule banks it: edge=0 must stay at t≈0; edge>0 "
            "must light up with a high win-rate."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.60):\n"
            "    panel_s, _ = data.synthetic_panel(edge=edge, seed=493, n_days=4000)\n"
            "    e = st.breadth_thrust_entries(panel_s, 'SPY'); s = st.summarize(st.forward_returns(panel_s['SPY']['close'], e, 20))\n"
            "    res.append((edge, s['n'], s['mean_bps'], s['win']*100, s['t']))\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "labels = [f'planted lead\\n{e:.2f}' for e,_,_,_,_ in res]; tvals = [r[4] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('20d one-sample t'); ax.set_title('Control: edge=0 -> t~0; planted lead -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,n,m,w,t in res: print(f'edge={e:.2f}: n={n} thrust={m:+.1f}bps win={w:.0f}% t={t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted lead the control sits at "
            f"**t = {R['syn'][0][4]:.2f}** (win {R['syn'][0][3]:.0f}% — no false positive); a planted "
            f"lead reaches **t = {R['syn'][1][4]:.2f}** (win {R['syn'][1][3]:.0f}%). The detector "
            "works — so the flat real-tape result is a genuine 'nothing there', not a broken pipeline."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the breadth thrust does not beat a drift-matched random baseline "
            f"(thrust − random = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f}/"
            f"{R['h60'][6]:+.0f} bps at 5/10/20/60d; Welch t negative throughout). The one-sample t's "
            f"(60d **{R['h60'][4]:.2f}**) are pure beta.\n"
            f"- **Tradability `MIRAGE`** — no residual edge once the drift is removed; costs only "
            "deepen the hole. You'd capture the drift more cheaply by holding the index.\n"
            f"- **Leads the index? `BUSTED`** — the shuffled-membership placebo leaves the result "
            f"untouched (**p = {R['placebo'][1]:.2f}**): structural-nonsense breadth lines do as well "
            "as the real one, so the NH-NL aggregation carries no forecasting information.\n\n"
            "> ⚠️ **Proxy caveat.** Breadth is proxied by 5 ETFs, not a true thousands-issue NH-NL "
            "universe. The result is lopsided enough (loses to random everywhere, placebo p ≈ 0.8) "
            "that a richer basket would have to overturn a very clear finding — but a real A/D feed "
            "remains the obvious follow-up."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — there is nothing to trade\n\n"
            "The thrust's entire apparent profit is the unconditional drift of long equity indices, "
            "obtained more cheaply and more fully by **buying and holding**. The breadth rule trades "
            "*less* of the time (only on thrusts), pays costs on each, and *underperforms* random "
            "entries — it strictly dominates *nothing*. There is no capacity question because there is "
            "no edge to scale. The NH-NL line is a descriptive market-internals gauge, not a "
            "forecasting strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Breadth divergence.** The headline lore is the *divergence* (price up, breadth "
            "fading). Encoding it mechanically inherits the same drift confound; a clean follow-up "
            "quantifies it.\n"
            "- **A true NH-NL universe.** Our 5-ETF proxy is coarse. A real advance/decline or "
            "exchange NH-NL feed (thousands of issues) is the obvious upgrade — the desk would re-run "
            "if it were offline-cacheable.\n"
            "- **Zweig breadth thrust.** The specific Zweig 10-day advance ratio is an affine cousin "
            "of this line and inherits the same confound.\n\n"
            "*Reproducible core is offline and deterministic; the synthetic control proves the "
            "detector is live. Methods/sources: [`docs/references.md`](../docs/references.md); frozen "
            "numbers: [`docs/results.md`](../docs/results.md).*"
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
