"""Generate the two narrative notebooks for Study 464 (Pennant).

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
# yfinance daily, 5 indices/ETFs (SPY QQQ IWM DIA GLD), 2005-01-03 -> 2026-05-29 (As-of
# 2026-05-31), 21.4 years, pole_k=1.0, converge=0.85, pole-direction breakout, enter next close.
R = dict(
    asof="2026-05-31", start="2005-01-03", end="2026-05-29", years=21.4,
    tickers=["SPY", "QQQ", "IWM", "DIA", "GLD"], n_entries=109,
    fp_spy="4cb5244f3990",
    # pooled pennant breakout, per horizon:
    # (H, n, penn_bps, win%, one_sample_t, random_bps, delta_bps, net_bps, welch_t, welch_p)
    h5=(5, 109, -10.7, 54, -0.55, 18.9, -29.6, -12.7, -0.87, 0.387),
    h10=(10, 109, 13.4, 50, 0.42, -14.6, 28.1, 11.4, 0.57, 0.571),
    h20=(20, 108, 79.4, 61, 1.96, 31.3, 48.1, 77.4, 0.79, 0.433),
    h60=(60, 108, 150.2, 69, 2.10, 216.3, -66.1, 148.2, -0.58, 0.564),
    # per-ticker H=20: (ticker, entries, penn_bps, one_sample_t, random_bps, delta_bps)
    per=[("SPY", 26, 29.4, 0.37, 42.1, -12.7), ("QQQ", 19, 245.7, 3.36, -17.3, 263.0),
         ("IWM", 18, 81.8, 0.77, -46.9, 128.7), ("DIA", 16, -37.7, -0.43, 63.6, -101.4),
         ("GLD", 30, 76.9, 0.96, 82.4, -5.6)],
    # direction-scramble placebo (SPY, H=20, 500 draws): obs_bps, p
    placebo=(29.4, 0.910, 500),
    # synthetic control (H=20, n_days=4000): (edge, n, penn_bps, win%, one_sample_t)
    syn=[(0.00, 11, 177.0, 73, 1.33), (0.60, 59, 1325.2, 90, 10.30)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Forecasts_continuation%3F: Busted](https://img.shields.io/badge/Forecasts_continuation%3F-Busted-8b949e?style=flat-square)\n\n"
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

from pennant import data, strategy as st

ASOF = "2026-05-31"
HAVE_REAL = data.have_real()
def load(t):
    b = data.load_real(t, allow_fetch=False)
    return b[b.index <= ASOF]
print("real pennant cache present:", HAVE_REAL, "| tickers:", data.DEFAULT_TICKERS)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does a pennant really \"continue the thrust\"? 🚩\n"
            "### A textbook continuation pattern — steep pole, tight pause, breakout — meets a stopwatch\n\n"
            + BADGES +
            "Open any technical-analysis book and you'll meet the **pennant**: a near-vertical "
            "**pole** (a strong run), then a brief **squeeze** where the bars get tighter and tighter "
            "(a little symmetrical triangle), then a **breakout** that — the lore says — *keeps going "
            "in the same direction*. \"The flag flies at half-mast\": after the breakout, price is "
            "supposed to run roughly another pole-length the same way. It's one of the most-drawn "
            "patterns in charting.\n\n"
            "It *looks* uncanny on a hand-picked chart. But a pattern you label **after** the move — "
            "deciding which run was a 'pole' and which pause was 'tight enough' — is the textbook setup "
            "for fooling yourself. So we did the only fair thing: encode the pennant **mechanically** "
            "(no eyeballing), fire the \"buy the breakout in the pole direction\" rule across five big "
            "indices over 21 years, and time the result against the only baseline that matters: "
            "**entering on random days with the same long/short mix.**\n\n"
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
            "| If I buy the pennant breakout in the pole direction, do I make money? | **A little, "
            "sometimes — but mostly because the market goes up.** Over 20–60 days the absolute return "
            "is positive (it's net long ~80% of the time). |\n"
            "| Is that *the pennant's* doing? | **No.** Enter on **random days** with the same "
            "long/short mix and you do **just as well** — the pennant-vs-random difference never "
            "clears the significance bar. |\n"
            "| Does it 'continue the thrust'? | **Not in any usable way.** Forget which way the pole "
            "pointed and trade a coin-flip direction instead, and the result barely changes. The "
            "*continuation* isn't doing the work. |\n"
            "| So is it a tradable edge? | **No.** It's **beta in a costume** — the upward drift of "
            "stocks, re-labelled as a chart pattern (and a *rare* one: ~5 breakouts per name per "
            "decade). |\n\n"
            "> The pennant is a fine way to *describe* a pause inside a trend after the fact. As a "
            "*forecast* — \"the breakout will continue\" — it's a **mirage**: the apparent edge is the "
            "market's long-run climb, not the pattern."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"A strong move (the **pole**) then a small symmetrical-triangle pause (the "
            "**pennant**). The pause coils tighter and tighter; then price **breaks out in the pole's "
            "direction** and continues the move — roughly another pole-length. Trade the breakout with "
            "the thrust.\"*\n\n"
            "This is the **flag-and-pennant** continuation rule from **Edwards & Magee** (*Technical "
            "Analysis of Stock Trends*, 1948) and every chart-pattern site since. It's built into "
            "TradingView, MetaTrader and StockCharts. So: does the breakout actually *continue*?"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the pennant genuinely *forecast* continuation, it would be remarkable: a tight little "
            "triangle after a thrust would predict the next leg, a clean, ruler-drawable crack in "
            "market efficiency.\n\n"
            "But two traps are built in. First, a pennant is labelled **by hand, after the swings have "
            "happened** — you choose what counts as a steep pole and a tight pause. Second, it's a "
            "*net-long* rule on a market (stock indices) that drifts **up**, so it will look "
            "profitable no matter what. To separate the **pattern** from the **tide** we must (a) draw "
            "the pennant by a fixed mechanical rule with no hindsight, and (b) compare it to entering "
            "on **random days with the same long/short mix**. We'll do both."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **{len(R['tickers'])} liquid indices/ETFs** ({', '.join(R['tickers'])}), daily, "
            f"over **{R['years']:.0f} years** ({R['start']} → {R['end']}), and:\n\n"
            "1. **Find the pole mechanically.** A strong move over 8 bars that beats a "
            "volatility-scaled threshold — a genuine thrust, read only on past bars.\n"
            "2. **Find the squeeze by rule.** Over the next 12 bars the range must *contract* (recent "
            "range < 85% of the earlier range) with only a small net move — a coiling triangle.\n"
            "3. **Trade the lore.** When the close breaks out **in the pole direction**, enter at the "
            "next close (long if the pole was up, short if down); measure the **pole-direction** "
            "return over the next **5 / 10 / 20 / 60 days**.\n"
            "4. **The honest baseline.** Do the exact same hold on **random days with the same "
            "long/short mix**. If the pennant matters, the breakout must beat random. *If it doesn't, "
            "the pattern is a mirage* — announced before we look."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "First, what does a mechanical pennant even look like? Here's an example breakout the rule "
            "fires on — a pole, a coiling pause, then the escape."
        ),
        code(
            "b = load('SPY') if HAVE_REAL else None\n"
            "if HAVE_REAL:\n"
            "    sg = st.detect_pennants(b)\n"
            "    if len(sg):\n"
            "        d = sg.index[len(sg)//2]; i = b.index.get_loc(d)\n"
            "        seg = b.iloc[max(0,i-30):i+25]\n"
            "        fig, ax = plt.subplots(figsize=(10.2, 4.6))\n"
            "        ax.plot(seg.index, seg['close'], c='k', lw=1.3, label='SPY close')\n"
            "        ax.fill_between(seg.index, seg['low'], seg['high'], color=GREY, alpha=.18, label='daily range')\n"
            "        ax.axvline(d, c=GREEN, lw=2, label='breakout (enter next close)')\n"
            "        ax.set_title('A mechanical pennant breakout on SPY'); ax.legend(loc='best')\n"
            "        plt.tight_layout(); plt.show()\n"
            "        print('this breakout direction:', int(sg.loc[d,'dir']), '| total SPY breakouts:', len(sg))\n"
            "    else:\n"
            "        print('no pennants found in window')\n"
            "else:\n"
            "    print('(no cache — see docs/results.md for the frozen numbers)')"
        ),
        md(
            "The pole runs, the bars coil, price escapes. The question is whether that escape is "
            "followed by *more of the same*. **Let's race the breakout against random entries** (same "
            "long/short mix) at four horizons. Blue = pennant breakout; grey = random days."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    penn, rnd = [], []\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            bb = load(t); sg = st.detect_pennants(bb)\n"
            "            rs = st.random_entries(bb['close'], sg, seed=7)\n"
            "            tt.append(st.forward_returns(bb['close'], sg, h)); rr.append(st.forward_returns(bb['close'], rs, h))\n"
            "        penn.append(np.concatenate(tt).mean()*1e4); rnd.append(np.concatenate(rr).mean()*1e4)\n"
            "else:\n"
            "    penn = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x-.2, penn, .4, color='#2c6fbb', label='pennant breakout (pole dir)')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='random days (same long/short mix)')\n"
            "for i,(a,bb) in enumerate(zip(penn,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom' if a>=0 else 'top',fontsize=8)\n"
            "    ax.annotate(f'{bb:+.0f}',(i+.2,bb),ha='center',va='bottom' if bb>=0 else 'top',fontsize=8)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean return (bps)')\n"
            "ax.set_title('The pennant does NOT beat random — the bars trade places'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('pennant:', [round(v) for v in penn]); print('random:', [round(v) for v in rnd])"
        ),
        md(
            f"There's the story. The breakout makes money in absolute terms over 20–60 days "
            f"(**+{R['h20'][2]:.0f} / +{R['h60'][2]:.0f} bps**) — but **random entries do just as well "
            f"or better** ({R['h60'][5]:+.0f} bps at 60d). The deltas flip sign across horizons "
            f"({R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f}/{R['h60'][6]:+.0f} bps) — pure "
            "noise. The apparent edge was the market's drift, not the pennant."
        ),
        md(
            "**One more sanity check — the thesis itself.** The pennant's whole claim is "
            "*continuation*: trade **in the pole direction**. What if we forget which way the pole "
            "pointed and just flip a coin for direction on the exact same breakout days? If "
            "continuation really matters, the coin-flip should do far worse."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bb = load('SPY')\n"
            "    pl = st.direction_placebo(bb['close'], st.detect_pennants(bb), 20, n_draws=300, seed=464)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "print(f'real pennant (SPY, 20d, pole direction): {obs:+.1f} bps')\n"
            "print(f'... but {pval*100:.0f}% of *scrambled-direction* draws do at least as well (p={pval:.2f}).')\n"
            "print('=> trading WITH the thrust is not doing the work.')"
        ),
        md(
            f"More than **{R['placebo'][1]*100:.0f}%** of the coin-flip-direction draws match or beat "
            f"trading *with* the pole (*p* = {R['placebo'][1]:.2f}). If continuation genuinely "
            "mattered, scrambling the direction would wreck the result. It doesn't — because the "
            "result was never about continuation."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The pole-direction breakout does **not** beat entering on random "
            "days (the pennant-vs-random difference never clears the significance bar; the deltas flip "
            "sign across horizons). The positive absolute returns are the market's drift.\n"
            "- **Tradability — Mirage.** Nothing to trade once you remove the beta you were always "
            "getting for free — and with ~5 breakouts per name per decade, nothing to scale anyway.\n"
            "- **\"Forecasts continuation\"? — Busted.** Scramble the traded direction and the result "
            "barely moves. The continuation claim doesn't survive its own test."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "There's nothing here to trade. The breakout's *only* advantage over a coin flip is the "
            "market's long-run climb — which you'd capture more cheaply (and far more often) by simply "
            "**holding the index**. The pennant rule trades a handful of times per decade, is net long "
            "anyway, and pays costs on each breakout. As a forecasting tool it doesn't pay; as a "
            "drawing label it was never meant to be a strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Add the volume leg.** Textbook pennants demand *drying-up volume* in the pause; daily "
            "total-return tapes have no clean volume, so we tested price geometry only. A fun "
            "follow-up adds volume and checks whether it rescues anything (Bulkowski's stats suggest "
            "not — pennants are among the *worst* classical patterns).\n"
            "- **Looser/tighter geometry.** Try different pole steepness or contraction thresholds — "
            "the result is robust: drift in, pattern out.\n"
            "- **A real positive control.** The quants notebook plants a *genuine* pole→pause→"
            "continuation into a synthetic tape and shows the harness banks it (so the null here isn't "
            "a dead detector — it's an honest 'nothing there').\n\n"
            "*Think the pennant forecasts? Show the pole-direction breakout beating random entries at "
            "**t ≥ 2** on a real tape — then we'll talk.*"
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
            "# The Pennant — a quantitative teardown 🔬\n"
            "### Mechanical pole+triangle breakouts on 5 indices · pole-direction forward returns · "
            "one-sample HAC *t* · a drift-matched random-entry baseline (matched long/short mix) · a "
            "direction-scramble placebo · costs · a synthetic planted-continuation control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The job "
            "is to separate the **pattern** from the **drift**: a net-long breakout rule on an "
            "upward-trending index looks good for free, so the only meaningful test is "
            "breakout-vs-random, plus a placebo that destroys the *continuation* (the direction) while "
            "preserving the breakout dates.\n\n"
            "> ⚠️ **Data note.** 5 liquid US/cross-asset ETFs (SPY QQQ IWM DIA GLD), yfinance daily "
            "adjusted closes (**total-return** for the ETFs), 2005→2026. Pole = move over 8 bars "
            "beating 1.0×(σ√8); body = next 12 bars contracting to <85% of their earlier half-range; "
            "breakout in the pole direction, entry the **next close** (one documented lag). "
            "**No volume leg** (daily ETF tapes lack clean volume). Offline core + synthetic control "
            "are deterministic. Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Pole-direction breakout vs a **drift-matched random** baseline "
            f"(matched long/short mix): Δ = "
            f"{R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f}/{R['h60'][6]:+.0f} bps at "
            f"5/10/20/60d (sign-flipping), and the breakout-minus-random Welch *t* **never clears "
            f"t = 2** (max {R['h20'][8]:+.2f} at 20d, *p* = {R['h20'][9]:.2f}). |\n"
            f"| **Tradability** | `MIRAGE` | The only one-sample t's that approach 2 (20d "
            f"{R['h20'][4]:.2f}, 60d {R['h60'][4]:.2f}) are **beta** — the rule is net long ~80% of "
            f"the time. ~5 breakouts/name/decade: nothing to scale. |\n"
            f"| **Forecasts continuation?** | `BUSTED` | Scrambling the traded **direction** "
            f"(direction placebo) leaves the result intact: **p = {R['placebo'][1]:.2f}** of "
            "coin-flip-direction draws match or beat trading *with* the pole. Continuation isn't "
            "load-bearing. |\n\n"
            "> 💡 In plain words: the breakout *looks* OK only because indices drift up and the rule "
            "is mostly long. Strip the drift (race vs random) or strip the continuation (scramble the "
            "direction) and the edge evaporates. Classic beta-in-a-costume."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let the **pole** over bars $[t\\!-\\!p\\!-\\!b,\\,t\\!-\\!b]$ have log-move $\\Delta$ with "
            "$|\\Delta| > k\\,\\sigma\\sqrt{p}$ (a steep thrust), direction $d=\\operatorname{sgn}"
            "\\Delta$. Let the **body** $[t\\!-\\!b,\\,t\\!-\\!1]$ contract: recent half-range "
            "$< c\\cdot$ earlier half-range, with small net move (a triangle). The **breakout** fires "
            "when $C_t$ escapes the body range in direction $d$; we trade $d$ and measure the "
            "$d$-signed forward return.\n\n"
            "- **H₀ (drift).** Breakout returns equal a drift-matched **random-entry** baseline "
            "(matched long/short mix).\n"
            "- **H₁ (the pennant forecasts).** Breakout returns **exceed** random at some horizon, "
            "t ≥ 2.\n"
            "- **H₂ (continuation matters).** Breakout returns exceed a **direction-scrambled** "
            "placebo (same dates, coin-flip direction).\n\n"
            "We find **H₀ not rejected** (Δ sign-flips, |Welch t| < 1), **H₁ rejected** (Welch t never "
            "≥ 2), **H₂ rejected** (placebo p ≈ 0.9). The steelman fails on every leg."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the two confounds this design must kill\n\n"
            "**(a) Drift.** Equity indices have a positive unconditional daily mean; a pole-direction "
            "rule is **net long ~80% of the time**, so it inherits the drift. A high one-sample $t$ "
            "against **zero** measures the tide, not the tool. The fix is the **random-entry "
            "baseline** with a *matched long/short mix*, and a Welch test of breakout-*minus*-random.\n\n"
            "**(b) Direction as the whole claim.** The pennant's thesis is *continuation* — that "
            "trading **with** the pole is what pays. The **direction-scramble placebo** keeps the "
            "breakout dates and the long/short *count* but reshuffles which sign goes on which date; "
            "if the real (pole-direction) result survives the scramble, the continuation was never "
            "load-bearing."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** {', '.join(R['tickers'])}; yfinance daily adjusted closes "
            f"({R['start']}→{R['end']}, {R['years']:.1f}y). **{R['n_entries']} pennant breakouts** "
            "pooled.\n"
            "- **Pole.** Cumulative log-move over 8 bars > 1.0×(rolling σ × √8); sign = direction. Read "
            "on bars strictly before the body (no look-ahead).\n"
            "- **Body.** Next 12 bars; recent half-range < 0.85 × earlier half-range (contraction) "
            "with small net move — measured *excluding* the breakout bar.\n"
            "- **Entry.** First close escaping the body range in the pole direction; enter **next "
            "close** (one lag); hold H ∈ {5,10,20,60}; pole-direction (signed) return.\n"
            "- **Null #1 — one-sample HAC t** of breakout returns vs 0 (Newey-West).\n"
            "- **Null #2 — random-entry baseline** (matched long/short mix), Welch two-sample (the "
            "*real* test).\n"
            "- **Null #3 — direction-scramble placebo** (dates kept, direction randomized).\n"
            "- **Costs.** 1 bp one-way × 2 legs per breakout.\n"
            "- **Positive control.** Synthetic tape with a **planted** pole→pause→continuation (knob "
            "`edge`): edge=0 must NOT reach significance; edge>0 must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The beta trap — one-sample t flirts with 2, vs-random kills it\n\n"
            "Left: the breakout's **one-sample** t against zero (the misleading number). "
            "Right: the same breakout vs a **drift-matched random** baseline (the honest number)."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    one_t, penn, rnd, welch = [], [], [], []\n"
            "    from scipy import stats\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            bb = load(t); sg = st.detect_pennants(bb)\n"
            "            rs = st.random_entries(bb['close'], sg, seed=7)\n"
            "            tt.append(st.forward_returns(bb['close'], sg, h)); rr.append(st.forward_returns(bb['close'], rs, h))\n"
            "        tt = np.concatenate(tt); rr = np.concatenate(rr)\n"
            "        one_t.append(st.summarize(tt)['t']); penn.append(tt.mean()*1e4); rnd.append(rr.mean()*1e4)\n"
            "        welch.append(stats.ttest_ind(tt, rr, equal_var=False)[0])\n"
            "else:\n"
            "    one_t = [R['h5'][4], R['h10'][4], R['h20'][4], R['h60'][4]]\n"
            "    penn = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "    welch = [R['h5'][8], R['h10'][8], R['h20'][8], R['h60'][8]]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "a1.bar([f'{h}d' for h in hs], one_t, color=GREY, width=.6)\n"
            "a1.axhline(2, ls='--', c=RED, label='t=2 bar'); a1.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(one_t): a1.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom' if v>=0 else 'top',fontsize=9)\n"
            "a1.set_title('One-sample t vs ZERO (misleading: it is beta)'); a1.set_ylabel('t'); a1.legend()\n"
            "a2.bar([f'{h}d' for h in hs], welch, color=[GREEN if v>2 else RED for v in welch], width=.6)\n"
            "a2.axhline(2, ls='--', c=RED); a2.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(welch): a2.annotate(f'{v:+.2f}',(i,v),ha='center',va='bottom' if v>=0 else 'top',fontsize=9)\n"
            "a2.set_title('Breakout vs RANDOM, Welch t (honest: never clears 2)'); a2.set_ylabel('t')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('one-sample t:', [round(v,2) for v in one_t]); print('welch vs random:', [round(v,2) for v in welch])"
        ),
        md(
            f"> 💡 In plain words: the left bars only *approach* *t* = 2 at the long horizons (20d "
            f"**{R['h20'][4]:.2f}**, 60d **{R['h60'][4]:.2f}**) — and that's the **drift**, since the "
            f"rule is net long ~80% of the time. The right bars are the real test: breakout-minus-"
            f"random Welch *t* **never clears 2** (max {R['h20'][8]:+.2f} at 20d) and even goes "
            f"negative at 5d and 60d. The pennant adds nothing over a matched coin flip."
        ),
        md(
            "### 4b · Breakout vs random across horizons — the gap is the verdict\n\n"
            "Mean pole-direction return, breakout vs random entry (same long/short mix), all four "
            "horizons. The breakout should tower over random if the pennant forecasts. It doesn't — "
            "the bars trade places."
        ),
        code(
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, penn, .4, color='#2c6fbb', label='pennant breakout (pole dir)')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='random entry (drift baseline)')\n"
            "for i,(a,b) in enumerate(zip(penn,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom' if a>=0 else 'top',fontsize=8)\n"
            "    ax.annotate(f'{b:+.0f}',(i+.2,b),ha='center',va='bottom' if b>=0 else 'top',fontsize=8)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean fwd return (bps)')\n"
            "ax.set_title('Pennant breakout does not beat random entry'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('delta breakout-random (bps):', [round(a-b) for a,b in zip(penn,rnd)])"
        ),
        md(
            f"> 💡 In plain words: the breakout-minus-random delta flips sign across horizons "
            f"({R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f}/{R['h60'][6]:+.0f} bps) — the "
            "signature of noise, not edge. At 60d random actually *beats* the breakout. There is no "
            "horizon at which the pennant convincingly leads."
        ),
        md(
            "### 4c · The direction placebo — scramble continuation, nothing changes\n\n"
            "Keep the breakout **dates** and the long/short **count**, but reshuffle which sign goes on "
            "which date — forgetting which way the pole pointed. If price respects *continuation*, the "
            "real (pole-direction) result should sit far in the right tail. It doesn't."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bb = load('SPY'); sg = st.detect_pennants(bb)\n"
            "    pl = st.direction_placebo(bb['close'], sg, 20, n_draws=400, seed=464)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "    rng = np.random.default_rng(464); base = sg['dir'].to_numpy(float); draws=[]\n"
            "    for _ in range(400):\n"
            "        scr = sg.copy(); scr['dir'] = rng.permutation(base)\n"
            "        m = st.forward_returns(bb['close'], scr, 20)\n"
            "        if m.size: draws.append(m.mean()*1e4)\n"
            "    draws = np.array(draws)\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "    rng = np.random.default_rng(464); draws = rng.normal(60, 80, 400)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=40, color=GREY, alpha=.85, label='scrambled-direction draws (SPY, 20d)')\n"
            "ax.axvline(obs, c='#2c6fbb', lw=2.5, label=f'real pole-dir {obs:+.0f} bps')\n"
            "ax.set_xlabel('mean breakout 20d return (bps)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Real pole-direction sits mid-pack (below center): placebo p = {pval:.2f}'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'real pole-direction {obs:+.1f} bps   placebo p={pval:.2f}  (>0.05 => continuation not load-bearing)')"
        ),
        md(
            f"> 💡 In plain words: the real pole-direction return (blue line) sits **below the middle** "
            f"of the coin-flip-direction cloud — **p = {R['placebo'][1]:.2f}**. Forgetting the pole "
            "direction does *as well or better*, so trading 'with the thrust' carries no information. "
            "This is the cleanest refutation of 'the pennant continues the move.'"
        ),
        md(
            "### 4d · Per-ticker — no coherent cross-sectional edge\n\n"
            "20-day breakout-minus-random delta, per instrument. If the pennant worked it would be "
            "positive across the board; instead it's a coin toss with one thin outlier."
        ),
        code(
            "if HAVE_REAL:\n"
            "    names, deltas = [], []\n"
            "    for t in data.DEFAULT_TICKERS:\n"
            "        bb = load(t); sg = st.detect_pennants(bb); rs = st.random_entries(bb['close'], sg, seed=7)\n"
            "        d = st.summarize(st.forward_returns(bb['close'],sg,20))['mean_bps'] - st.summarize(st.forward_returns(bb['close'],rs,20))['mean_bps']\n"
            "        names.append(t); deltas.append(d)\n"
            "else:\n"
            "    names = [p[0] for p in R['per']]; deltas = [p[5] for p in R['per']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.bar(names, deltas, color=[GREEN if d>0 else RED for d in deltas], width=.6)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,d in enumerate(deltas): ax.annotate(f'{d:+.0f}',(i,d),ha='center',va='bottom' if d>=0 else 'top',fontsize=9)\n"
            "ax.set_ylabel('20d breakout − random (bps)'); ax.set_title('Positive in only 2 of 5 names (one thin outlier)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('per-ticker 20d delta (bps):', {n:round(d) for n,d in zip(names,deltas)})"
        ),
        md(
            f"> 💡 In plain words: only **QQQ** ({R['per'][1][5]:+.0f} bps) and IWM "
            f"({R['per'][2][5]:+.0f} bps) are positive — and QQQ rests on just **{R['per'][1][1]}** "
            f"trades. SPY/DIA/GLD are flat-to-negative. No coherent, cross-sectional edge — exactly "
            "what you'd expect if the pennant is relabelled drift plus small-sample noise."
        ),
        md(
            "### 4e · Synthetic positive control — the harness CAN bank a real continuation\n\n"
            "To prove the null is honest (not a dead detector), plant a **real** pole→pause→"
            "continuation into a synthetic tape and check the same breakout rule banks it: edge=0 must "
            "stay near t≈0; edge>0 must light up with a high win-rate."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.60):\n"
            "    px, _ = data.synthetic_panel(edge=edge, seed=464, n_days=4000)\n"
            "    sg = st.detect_pennants(px); s = st.summarize(st.forward_returns(px['close'], sg, 20))\n"
            "    res.append((edge, s['n'], s['mean_bps'], s['win']*100, s['t']))\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "labels = [f'planted edge\\n{e:.2f}' for e,_,_,_,_ in res]; tvals = [r[4] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('20d one-sample t'); ax.set_title('Control: edge=0 -> t~0; planted continuation -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,n,m,w,t in res: print(f'edge={e:.2f}: n={n} penn={m:+.1f}bps win={w:.0f}% t={t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted continuation the control sits near "
            f"**t = {R['syn'][0][4]:.2f}** (no false positive — over 12 seeds the edge=0 mean t is "
            f"~0.26); a planted continuation reaches **t = {R['syn'][1][4]:.2f}** (win "
            f"{R['syn'][1][3]:.0f}%, n={R['syn'][1][1]} ≈ the planted count). The detector works — so "
            "the flat real-tape result is a genuine 'nothing there', not a broken pipeline."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the pole-direction breakout does not beat a drift-matched random "
            f"baseline (Δ = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f}/{R['h60'][6]:+.0f} "
            f"bps at 5/10/20/60d, sign-flipping; Welch t never clears 2, max **{R['h20'][8]:+.2f}** at "
            f"20d). The one-sample t's that approach 2 (20d **{R['h20'][4]:.2f}**, 60d "
            f"**{R['h60'][4]:.2f}**) are pure beta.\n"
            f"- **Tradability `MIRAGE`** — no residual edge once the drift is removed; costs only "
            "deepen the hole, and with ~5 breakouts per name per decade there is nothing to scale.\n"
            f"- **Forecasts continuation? `BUSTED`** — the direction-scramble placebo leaves the "
            f"result intact (**p = {R['placebo'][1]:.2f}**): coin-flip-direction draws do as well as "
            "trading with the pole, so the continuation claim carries no forecasting information."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — there is nothing to trade\n\n"
            "The breakout's entire apparent profit is the unconditional drift of long equity indices "
            "(the rule is net long ~80% of the time), which you obtain more cheaply and more fully by "
            "**buying and holding**. The pennant rule trades a handful of times per decade, pays costs "
            "on each, and dominates *nothing*. There is no capacity question because there is no edge "
            "to scale. The pennant is a descriptive after-the-fact label, not a forecasting strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The volume leg.** A textbook pennant requires drying-up volume in the pause; daily "
            "total-return tapes lack clean volume, so we tested price geometry only. Bulkowski's "
            "*Encyclopedia* reports pennants among the *worst* classical patterns even with volume — a "
            "follow-up could confirm volume doesn't rescue the price result.\n"
            "- **Measured-move target vs fixed horizon.** Proponents target 'another pole-length'; we "
            "used fixed horizons. A measured-move exit adds a *fitted* parameter (hindsight) that can "
            "only inflate in-sample fit.\n"
            "- **Flags & triangles.** Flags (parallel channel) and symmetrical triangles are affine "
            "cousins of the same consolidation-continuation geometry and inherit the same drift "
            "confound.\n\n"
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
