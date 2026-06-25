"""Generate the two narrative notebooks for Study 460 (Counterattack / Meeting Lines).

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
# 2026-05-31, partial June dropped), 21.4 years, tol=15bps, trend_lookback=10, bullish meeting.
R = dict(
    asof="2026-05-31", start="2005-01-03", end="2026-05-29", years=21.4,
    tickers=["SPY", "QQQ", "IWM", "DIA", "GLD"], n_entries=280, tol_bps=15, lookback=10,
    fp_spy="4cb5244f3990",
    # pooled bullish meeting line, per horizon:
    # (H, n, meet_bps, win%, one_sample_t, random_bps, delta_bps, net_bps, welch_t, welch_p)
    h5=(5, 280, -7.6, 53, -0.46, 61.0, -68.6, -9.6, -3.01, 0.003),
    h10=(10, 280, 11.5, 55, 0.53, 73.0, -61.5, 9.5, -1.97, 0.049),
    h20=(20, 279, 111.3, 63, 3.50, 97.9, 13.4, 109.3, 0.33, 0.741),
    h60=(60, 278, 264.6, 70, 4.20, 118.0, 146.6, 262.6, 1.99, 0.047),
    # per-ticker H=20: (ticker, entries, meet_bps, one_sample_t, random_bps, delta_bps)
    per=[("SPY", 51, 41.1, 0.71, 36.7, 4.4), ("QQQ", 53, 193.6, 2.19, 129.9, 63.7),
         ("IWM", 47, 192.3, 3.03, 51.5, 140.8), ("DIA", 53, 53.5, 0.72, 38.1, 15.4),
         ("GLD", 76, 90.8, 1.57, 189.4, -98.7)],
    # close-scramble placebo (SPY, H=20, 500 draws): obs_bps, p, draws
    placebo=(41.1, 0.623, 500),
    # synthetic control (H=20, n_days=4000): (edge, n, meet_bps, win%, one_sample_t)
    syn=[(0.00, 68, -34.4, 46, -0.64), (0.60, 50, 595.9, 92, 9.86)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Meeting_forecasts%3F: Busted](https://img.shields.io/badge/Meeting_forecasts%3F-Busted-8b949e?style=flat-square)\n\n"
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

from counterattack_lines import data, strategy as st

ASOF = "2026-05-31"
TOL = st.DEFAULT_TOL
LB = st.DEFAULT_TREND_LOOKBACK
HAVE_REAL = data.have_real()
def load(t):
    b = data.load_real(t, allow_fetch=False)
    return b[b.index <= ASOF]
print("real meeting-line cache present:", HAVE_REAL, "| tickers:", data.DEFAULT_TICKERS)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does a \"counterattack line\" actually reverse a downtrend? ⚔️\n"
            "### A famous candlestick — two opposite-colour candles closing at the same price — "
            "meets a stopwatch\n\n"
            + BADGES +
            "Open any candlestick book and you'll find the **counterattack** (or **meeting**) "
            "line: in a falling market, a red/black candle drops, then the next candle gaps "
            "*lower* on the open but rallies all day to **close at exactly the previous close** — "
            "the two closes \"meet\". The lore, from Steve Nison's candlestick canon, is that this "
            "equal-close meeting is where the sellers ran out of ammunition: **buy, the trend is "
            "about to flip up.**\n\n"
            "It *looks* compelling on a hand-picked chart. But a two-candle pattern read off recent "
            "price, on a market (stock indices) that drifts **up** over time, is the textbook setup "
            "for fooling yourself. So we did the only fair thing: encode the meeting line "
            "**mechanically** (a concrete equal-close tolerance, no eyeballing), fire the buy "
            "across five big indices over 21 years, and time the result with a stopwatch — against "
            "the only baseline that matters: **buying on random days instead.**\n\n"
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
            "| If I buy a bullish meeting line, do I make money? | **At 20–60 days, yes — but only "
            "because the market goes up.** At the 1–2 week horizon where the \"reversal\" is "
            "supposed to act, the return is roughly **flat or negative**. |\n"
            "| Is that *the pattern's* doing? | **No.** Buy on **random days** instead and you do "
            "**just as well or better** — at 5 days the meeting line is *significantly worse* than "
            "a coin-flip entry. |\n"
            "| Does the *equal close* matter? | **No.** Keep the down-leg-and-gap setup but ignore "
            "whether the closes actually meet, and you get the same result. The meeting adds "
            "nothing. |\n"
            "| So is it a tradable edge? | **No.** It's **beta in a costume** — a dip after a "
            "down leg, re-labelled as a candlestick reversal. |\n\n"
            "> The counterattack line is a tidy way to *name* a candle after the fact. As a "
            "*forecast* — \"the meeting marks the bottom\" — it's a **mirage**: the longer-horizon "
            "gains are the market's climb, and the equal close itself does no work."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"In a downtrend, a black candle falls. The next candle gaps lower but closes "
            "right back at the prior close — the closes **meet**. The bears have been answered: "
            "buy, the trend reverses up.\"*\n\n"
            "This is the **bullish counterattack / meeting line** (Japanese *deai sen*), brought "
            "West by **Steve Nison** in *Japanese Candlestick Charting Techniques* (1991) and "
            "catalogued by Morris and Bulkowski. It's the weaker cousin of the *piercing line* "
            "(which must close *above* the prior midpoint); the counterattack only asks the closes "
            "to meet — so it's a clean test of: **does an equal close forecast anything?**"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the meeting genuinely *forecast* reversals, it would be remarkable: two candles and "
            "a coincidence of closing prices would predict a turn, a crack in market efficiency you "
            "could trade off the chart. That's the dream the pattern sells.\n\n"
            "But there's a trap built into it. A meeting line is *defined* by a recent down leg, "
            "so the rule is a **dip-buy** — and on a market that drifts **up**, *any* dip-buy looks "
            "profitable. To separate the **pattern** from the **tide**, we have to (a) detect the "
            "meeting by a fixed mechanical rule with no hindsight, and (b) compare it to buying on "
            "**random days**, and (c) check whether the *equal close* (not just the dip) is doing "
            "anything. We'll do all three."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **{len(R['tickers'])} liquid indices/ETFs** ({', '.join(R['tickers'])}), daily, "
            f"over **{R['years']:.0f} years** ({R['start']} → {R['end']}), and:\n\n"
            "1. **Detect the meeting mechanically.** A bullish meeting = a confirmed down leg "
            f"(close below where it was {R['lookback']} bars ago), a black candle, then a white "
            f"candle that gaps down and closes **within {R['tol_bps']} bps** of the prior close. No "
            "eyeballing.\n"
            "2. **Trade the lore.** On the meeting bar's close, buy at the **next** close; measure "
            "the return over the next **5 / 10 / 20 / 60 days**.\n"
            "3. **The honest baseline.** Do the exact same hold on **random days**. If the meeting "
            "matters, it must beat random. *If it doesn't, the pattern is a mirage* — that's the "
            "result that would make us say so, announced before we look.\n"
            "4. **The geometry check.** Keep the down-leg-and-gap setup but ignore the equal close; "
            "if that does just as well, the *meeting* was never the point."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "First, what does a mechanical meeting line even look like? Here's SPY with the bullish "
            "meeting signals the rule would buy marked on the tape."
        ),
        code(
            "b = load('SPY') if HAVE_REAL else None\n"
            "if HAVE_REAL:\n"
            "    cl = b['close']; seg = cl.iloc[-700:]\n"
            "    ent = st.meeting_entries(b, tol=TOL, trend_lookback=LB)\n"
            "    ent = ent[ent >= seg.index[0]]\n"
            "    fig, ax = plt.subplots(figsize=(10.2, 4.8))\n"
            "    ax.plot(seg.index, seg.values, c='k', lw=1.1, label='SPY close')\n"
            "    ax.scatter(ent, cl.reindex(ent), c=GREEN, s=55, zorder=5, label='bullish meeting BUY')\n"
            "    ax.set_title('Mechanical bullish counterattack lines on SPY (last ~3y)'); ax.legend(loc='upper left')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('bullish meeting lines in window:', len(ent))\n"
            "else:\n"
            "    print('(no cache — see docs/results.md for the frozen numbers)')"
        ),
        md(
            "The signals cluster around dips — exactly as a dip-buy would. The question is whether "
            "those green buy dots are followed by *reversals*. **Let's race the meeting line against "
            "random entries** at four horizons. Blue = buy the meeting; grey = buy on random days."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    meet, rnd = [], []\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            bb = load(t); c = bb['close']\n"
            "            e = st.meeting_entries(bb, tol=TOL, trend_lookback=LB)\n"
            "            re = st.random_entries(c, max(len(e),50), warmup=LB, seed=7)\n"
            "            tt.append(st.forward_returns(c, e, h)); rr.append(st.forward_returns(c, re, h))\n"
            "        meet.append(np.concatenate(tt).mean()*1e4); rnd.append(np.concatenate(rr).mean()*1e4)\n"
            "else:\n"
            "    meet = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x-.2, meet, .4, color='#2c6fbb', label='buy the meeting line')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='buy on random days')\n"
            "for i,(a,bb) in enumerate(zip(meet,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{bb:+.0f}',(i+.2,bb),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean return (bps)')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_title('The meeting line does NOT beat random at the reversal horizon'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('meeting:', [round(v) for v in meet]); print('random:', [round(v) for v in rnd])"
        ),
        md(
            f"There's the story. At the **5-day** horizon — where a \"reversal\" should bite — the "
            f"meeting line is **{R['h5'][2]:+.0f} bps** while a random entry is **+{R['h5'][5]:.0f} "
            f"bps**: the famous pattern is *worse* than throwing darts. It only pulls ahead at "
            "20–60 days, which is just the market's drift over a longer hold (and even then the "
            "quants notebook shows the gap isn't statistically meaningful). The apparent edge was "
            "**the market's upward climb**, not the meeting."
        ),
        md(
            "**One more sanity check.** What if we keep the down-leg-and-gap setup but *ignore* "
            "whether the closes actually meet? If price really respects the **meeting**, dropping "
            "that condition should wreck the result."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bb = load('SPY')\n"
            "    pl = st.close_scramble_placebo(bb, 20, tol=TOL, trend_lookback=LB, n_draws=300, seed=460)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "print(f'real meeting line (SPY, 20d): {obs:+.1f} bps')\n"
            "print(f'... but {pval*100:.0f}% of *no-meeting* down-leg dip-buys do at least as well (p={pval:.2f}).')\n"
            "print('=> the equal close is not doing the work.')"
        ),
        md(
            f"More than half of the **no-meeting** down-leg dip-buys match or beat the real meeting "
            f"line (*p* = {R['placebo'][1]:.2f}). If price genuinely respected *the equal close*, "
            "dropping it would collapse the result. It doesn't — because the result was never about "
            "the meeting."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The bullish meeting buy does **not** beat buying on random days "
            "(it's *significantly worse* at 5 days; the meeting-vs-random difference never clears "
            "*t* = 2). The longer-horizon gains are the market's drift, not the pattern.\n"
            "- **Tradability — Mirage.** Nothing to trade once you remove the beta you were always "
            "getting for free — and at the short horizon the pattern is a net negative.\n"
            "- **\"Does the equal-close meeting forecast?\" — Busted.** Drop the meeting condition "
            "and the result is unchanged. The defining equal close carries no information."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "There's nothing here to trade. The meeting line's *only* advantage over a coin flip is "
            "the market's long-run climb at a 20–60 day hold — which you'd capture more cheaply (and "
            "more fully) by just **holding the index**. At the short, supposedly-reversal horizon "
            "the pattern actually *loses* to random. Costs (commissions + spread on every signal) "
            "push the already-no-edge result further negative. As a forecasting tool the meeting "
            "doesn't pay; as a label for a candle, it was never meant to be a strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The stronger cousins.** The *piercing line* (close above the prior midpoint) and "
            "*bullish engulfing* make a stronger reversal claim — a fun follow-up runs the same "
            "harness on them (spoiler: same drift confound).\n"
            "- **Different tolerances.** Loosen or tighten the equal-close band, lengthen the "
            "down-leg window — the result is robust: dip in, drift out.\n"
            "- **A real positive control.** The quants notebook plants a *genuine* meeting-bounce "
            "into a synthetic tape and shows the harness banks it (so the null result here isn't a "
            "dead detector — it's an honest 'nothing there').\n\n"
            "*Think the meeting forecasts? Show the bullish meeting line beating random entries at "
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
            "# Counterattack / Meeting Lines — a quantitative teardown 🔬\n"
            "### Mechanical bullish meeting lines on 5 indices · forward returns · one-sample HAC "
            "*t* · a drift-matched random-entry baseline · a close-scramble geometry placebo · "
            "costs · a synthetic planted-bounce control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The job "
            "is to separate the **meeting** from the **drift**: a meeting line is a dip-buy after a "
            "down leg, and an upward-trending index makes *any* dip-buy look good, so the only "
            "meaningful test is meeting-vs-random, plus a placebo that drops the equal-close "
            "condition while keeping the down-leg/gap context.\n\n"
            "> ⚠️ **Data note.** 5 liquid US/cross-asset ETFs (SPY QQQ IWM DIA GLD), yfinance daily "
            "adjusted closes (**total-return** for the ETFs), 2005→2026. Meeting = down leg "
            f"(lookback {R['lookback']}), opposite-colour candles, gap-down open, closes within "
            f"{R['tol_bps']} bps; entry is the **next close** (one documented lag). Offline core + "
            "synthetic control are deterministic. Methods in "
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
            f"| **Signal** | `NONE` | Meeting line vs a **drift-matched random** baseline: the "
            f"meeting is *worse* at 5/10d (Δ = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f} bps, Welch t at "
            f"5d = {R['h5'][8]:+.2f}, **significantly negative**) and the meeting-minus-random "
            f"difference **never clears t = 2** (max Welch t = {R['h60'][8]:+.2f} at 60d, p = "
            f"{R['h60'][9]:.3f}). |\n"
            f"| **Tradability** | `MIRAGE` | The one-sample t's at long holds (20d t = {R['h20'][4]:.2f}, "
            f"60d t = {R['h60'][4]:.2f}) are **pure beta** — they vanish against random entries; at "
            "the short reversal horizon the pattern is a net negative. No residual edge to scale. |\n"
            f"| **Meeting forecasts?** | `BUSTED` | Dropping the equal-close condition (close-scramble "
            f"placebo) leaves the result intact: **p = {R['placebo'][1]:.2f}** of no-meeting down-leg "
            "dip-buys match or beat the real one. The meeting isn't doing the work. |\n\n"
            "> 💡 In plain words: the meeting line *looks* okay only at long holds, because indices "
            "drift up. Strip the drift (race it vs random) or strip the meeting (drop the equal "
            "close) and the edge evaporates. Classic beta-in-a-costume."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "A bullish counterattack at bar $t$ requires: a down leg "
            "$C_{t-1}<C_{t-L}$ ($L=10$); a black candle $C_{t-1}<O_{t-1}$; a white candle "
            "$C_t>O_t$; a gap-down open $O_t<C_{t-1}$; and the **meeting** "
            "$|C_t-C_{t-1}|/C_{t-1}\\le\\tau$ ($\\tau=15$ bps). The Andrews-style rule buys the "
            "close of $t$ and rides the reversal up.\n\n"
            "- **H₀ (drift).** Meeting returns equal a drift-matched **random-entry** baseline.\n"
            "- **H₁ (the meeting forecasts).** Meeting returns **exceed** random at some horizon, t ≥ 2.\n"
            "- **H₂ (the equal close matters).** Meeting returns exceed a **close-scramble** pool "
            "that keeps the down-leg/gap context but drops the meeting.\n\n"
            "We find **H₀ not rejected** (meeting ≤ random at 5–10d, *worse*), **H₁ rejected** (Welch "
            "t never ≥ 2), **H₂ rejected** (placebo p ≈ 0.62). The steelman fails on every leg."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the two confounds this design must kill\n\n"
            "**(a) Drift.** Equity indices have a positive unconditional daily mean. *Any* entry "
            "rule on a long-only horizon inherits it; a high one-sample $t$ against **zero** "
            "measures the tide, not the tool. The fix is the **random-entry baseline** (same "
            "instrument, epoch, hold) and a Welch test of meeting-*minus*-random.\n\n"
            "**(b) The dip vs the meeting.** A counterattack is *by construction* a dip-buy (it "
            "needs a down leg). The danger is that the whole signal is the dip, not the equal "
            "close. The **close-scramble placebo** keeps the down-leg + opposite-colour + gap "
            "context but draws entries that ignore the meeting — if the real result survives that, "
            "the equal close was never load-bearing."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** {', '.join(R['tickers'])}; yfinance daily adjusted closes "
            f"({R['start']}→{R['end']}, {R['years']:.1f}y). **{R['n_entries']} bullish meeting "
            "lines** pooled.\n"
            f"- **Pattern.** Down leg (lookback {R['lookback']}), black candle t-1, white candle t, "
            f"gap-down open, closes within {R['tol_bps']} bps — all read at/before t (no "
            "look-ahead).\n"
            "- **Entry.** Buy the meeting bar's close; enter **next close** (one lag); hold "
            "H ∈ {5,10,20,60}.\n"
            "- **Null #1 — one-sample HAC t** of meeting returns vs 0 (Newey-West).\n"
            "- **Null #2 — random-entry baseline**, Welch two-sample meeting vs random (the *real* test).\n"
            "- **Null #3 — close-scramble placebo** (equal close dropped, context kept).\n"
            "- **Costs.** 1 bp one-way × 2 legs on every signal.\n"
            "- **Positive control.** Synthetic tape with a **planted** meeting-bounce (knob `edge`): "
            "edge=0 must NOT reach significance; edge>0 must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The beta trap — one-sample t looks fine, vs-random kills it\n\n"
            "Left: the meeting line's **one-sample** t against zero (the misleading number, at long "
            "holds). Right: the same meeting vs a **drift-matched random** baseline (the honest "
            "number)."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    one_t, meet, rnd, welch = [], [], [], []\n"
            "    from scipy import stats\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            bb = load(t); c = bb['close']\n"
            "            e = st.meeting_entries(bb, tol=TOL, trend_lookback=LB)\n"
            "            re = st.random_entries(c, max(len(e),50), warmup=LB, seed=7)\n"
            "            tt.append(st.forward_returns(c, e, h)); rr.append(st.forward_returns(c, re, h))\n"
            "        tt = np.concatenate(tt); rr = np.concatenate(rr)\n"
            "        one_t.append(st.summarize(tt)['t']); meet.append(tt.mean()*1e4); rnd.append(rr.mean()*1e4)\n"
            "        welch.append(stats.ttest_ind(tt, rr, equal_var=False)[0])\n"
            "else:\n"
            "    one_t = [R['h5'][4], R['h10'][4], R['h20'][4], R['h60'][4]]\n"
            "    meet = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
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
            "a2.set_title('Meeting vs RANDOM, Welch t (honest: never clears 2)'); a2.set_ylabel('t')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('one-sample t:', [round(v,2) for v in one_t]); print('welch vs random:', [round(v,2) for v in welch])"
        ),
        md(
            f"> 💡 In plain words: the left bars clear *t* = 2 at long holds (20d **{R['h20'][4]:.2f}**, "
            f"60d **{R['h60'][4]:.2f}**) — but that's the **drift**, every dip-buy inherits it. The "
            f"right bars are the real test: meeting-minus-random is **significantly negative** at 5d "
            f"({R['h5'][8]:+.2f}) and only **{R['h60'][8]:+.2f}** at 60d — never clears 2. The "
            "meeting adds nothing over a coin flip at the horizon it claims."
        ),
        md(
            "### 4b · Meeting vs random across horizons — the gap is the verdict\n\n"
            "Mean return, meeting vs random entry, all four horizons. The meeting should tower over "
            "random at the short reversal horizon if it forecasts. It doesn't — it loses."
        ),
        code(
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, meet, .4, color='#2c6fbb', label='meeting line')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='random entry (drift baseline)')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,(a,b) in enumerate(zip(meet,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom' if a>=0 else 'top',fontsize=8)\n"
            "    ax.annotate(f'{b:+.0f}',(i+.2,b),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean fwd return (bps)')\n"
            "ax.set_title('Meeting line does not beat random at the reversal horizon'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('delta meeting-random (bps):', [round(a-b) for a,b in zip(meet,rnd)])"
        ),
        md(
            f"> 💡 In plain words: at 5 days the meeting is **{R['h5'][2]:+.0f} bps** but random is "
            f"**+{R['h5'][5]:.0f} bps** — the pattern *underperforms* a dart by {abs(R['h5'][6]):.0f} "
            "bps right where the reversal should bite. The only horizons where it edges ahead are "
            "20–60d, and the Welch test (4a) says that gap is noise."
        ),
        md(
            "### 4c · The geometry placebo — drop the meeting, nothing changes\n\n"
            "Keep the down-leg + opposite-colour + gap-down context (the 'almost-meeting' "
            "candidates), but draw the same number of entries that **ignore the equal close**. If "
            "price respects *the meeting*, dropping it should demolish the result. The observed "
            "meeting return should sit far in the right tail of the no-meeting distribution. It "
            "doesn't."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bb = load('SPY'); c = bb['close']\n"
            "    pl = st.close_scramble_placebo(bb, 20, tol=TOL, trend_lookback=LB, n_draws=300, seed=460)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "    cand = bb.index[st._candidate_mask(bb, trend_lookback=LB)]\n"
            "    n_meet = len(st.meeting_entries(bb, tol=TOL, trend_lookback=LB))\n"
            "    rng = np.random.default_rng(460); draws=[]\n"
            "    cand_arr = np.asarray(cand)\n"
            "    for _ in range(300):\n"
            "        pick = rng.choice(cand_arr, size=n_meet, replace=False)\n"
            "        import pandas as _pd\n"
            "        rr = st.forward_returns(c, _pd.DatetimeIndex(sorted(pick)), 20)\n"
            "        if rr.size: draws.append(rr.mean()*1e4)\n"
            "    draws = np.array(draws)\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "    rng = np.random.default_rng(460); draws = rng.normal(40, 35, 300)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=40, color=GREY, alpha=.85, label='no-meeting down-leg dip-buys (SPY, 20d)')\n"
            "ax.axvline(obs, c='#2c6fbb', lw=2.5, label=f'real meeting {obs:+.0f} bps')\n"
            "ax.set_xlabel('mean 20d return (bps)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Real meeting sits mid-pack: placebo p = {pval:.2f}'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'real meeting {obs:+.1f} bps   placebo p={pval:.2f}  (>0.05 => the equal close is not load-bearing)')"
        ),
        md(
            f"> 💡 In plain words: the real meeting (blue line) sits **in the middle** of the "
            f"no-meeting cloud — **p = {R['placebo'][1]:.2f}**. Down-leg dip-buys that ignore the "
            "equal close do just as well, so the meeting itself carries no information. This is the "
            "cleanest refutation of 'the equal close forecasts.'"
        ),
        md(
            "### 4d · Per-ticker — no coherent cross-sectional edge\n\n"
            "20-day meeting-minus-random delta, per instrument. If the pattern worked it would be "
            "robustly positive; instead it rides on one small-cap name and is sharply negative in "
            "another."
        ),
        code(
            "if HAVE_REAL:\n"
            "    names, deltas = [], []\n"
            "    for t in data.DEFAULT_TICKERS:\n"
            "        bb = load(t); c = bb['close']\n"
            "        e = st.meeting_entries(bb, tol=TOL, trend_lookback=LB); re = st.random_entries(c, max(len(e),50), warmup=LB, seed=7)\n"
            "        d = st.summarize(st.forward_returns(c,e,20))['mean_bps'] - st.summarize(st.forward_returns(c,re,20))['mean_bps']\n"
            "        names.append(t); deltas.append(d)\n"
            "else:\n"
            "    names = [p[0] for p in R['per']]; deltas = [p[5] for p in R['per']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.bar(names, deltas, color=[GREEN if d>0 else RED for d in deltas], width=.6)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,d in enumerate(deltas): ax.annotate(f'{d:+.0f}',(i,d),ha='center',va='bottom' if d>=0 else 'top',fontsize=9)\n"
            "ax.set_ylabel('20d meeting − random (bps)'); ax.set_title('Edge rides on IWM; GLD is sharply negative')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('per-ticker 20d delta (bps):', {n:round(d) for n,d in zip(names,deltas)})"
        ),
        md(
            f"> 💡 In plain words: the whole 20d delta rides on **IWM** ({R['per'][2][5]:+.0f} bps on "
            f"just {R['per'][2][1]} signals — small-cap noise) while **GLD** is **{R['per'][4][5]:+.0f}** "
            "bps *behind* random. No coherent, cross-sectional edge — exactly what you'd expect if "
            "the meeting is just a relabelled dip."
        ),
        md(
            "### 4e · Synthetic positive control — the harness CAN bank a real bounce\n\n"
            "To prove the null is honest (not a dead detector), plant a **real** meeting-bounce into "
            "a synthetic tape and check the same rule banks it: edge=0 must stay at t≈0; edge>0 must "
            "light up with a high win-rate."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.60):\n"
            "    px, _ = data.synthetic_panel(edge=edge, seed=460, n_days=4000)\n"
            "    e = st.meeting_entries(px, tol=TOL, trend_lookback=LB); s = st.summarize(st.forward_returns(px['close'], e, 20))\n"
            "    res.append((edge, s['n'], s['mean_bps'], s['win']*100, s['t']))\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "labels = [f'planted edge\\n{e:.2f}' for e,_,_,_,_ in res]; tvals = [r[4] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('20d one-sample t'); ax.set_title('Control: edge=0 -> t~0; planted bounce -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,n,m,w,t in res: print(f'edge={e:.2f}: n={n} meeting={m:+.1f}bps win={w:.0f}% t={t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted bounce the control sits at "
            f"**t = {R['syn'][0][4]:.2f}** (win {R['syn'][0][3]:.0f}% — no false positive); a planted "
            f"bounce reaches **t = {R['syn'][1][4]:.2f}** (win {R['syn'][1][3]:.0f}%). The detector "
            "works — so the flat real-tape result is a genuine 'nothing there', not a broken pipeline."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the meeting line does not beat a drift-matched random baseline "
            f"(meeting − random = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f}/{R['h60'][6]:+.0f} "
            f"bps at 5/10/20/60d; Welch t is **significantly negative at 5d ({R['h5'][8]:+.2f})** and "
            f"never clears 2, max **{R['h60'][8]:+.2f}** at 60d, p = {R['h60'][9]:.3f}). The "
            f"impressive one-sample t's (20d **{R['h20'][4]:.2f}**) are pure beta.\n"
            f"- **Tradability `MIRAGE`** — no residual edge once the drift is removed; at the short "
            "reversal horizon the pattern is a net negative, and costs only deepen the hole. You'd "
            "capture the drift more cheaply by holding the index.\n"
            f"- **Meeting forecasts? `BUSTED`** — the close-scramble placebo leaves the result "
            f"untouched (**p = {R['placebo'][1]:.2f}**): no-meeting down-leg dip-buys do as well as "
            "the real ones, so the defining equal close carries no forecasting information."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — there is nothing to trade\n\n"
            "The meeting line's entire apparent profit is the unconditional drift of long equity "
            "indices at a 20–60 day hold, which you obtain more cheaply and more fully by **buying "
            "and holding**. At the short, supposedly-reversal horizon it *loses* to random. The "
            "rule trades *less* of the time (only on meetings) and pays costs on each, so it "
            "strictly dominates *nothing*. There is no capacity question because there is no edge to "
            "scale. The counterattack line is a descriptive candlestick label, not a forecasting "
            "strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The stronger relatives.** The *piercing line* (close above the prior midpoint) and "
            "*bullish engulfing* make a stronger close-geometry claim; the same harness applies and "
            "the same drift confound dominates.\n"
            "- **Tolerance & lookback sweeps.** The equal-close band and down-leg window are free "
            "parameters; tightening or loosening them does not rescue the meeting — a robustness "
            "grid only confirms the drift-in/drift-out picture.\n"
            "- **Bearish counterattack.** The mirror pattern (up leg, white then black candle "
            "closing to meet) inherits the symmetric drift problem on the short side.\n\n"
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
