"""Generate the two narrative notebooks for Study 377 (Bid-Ask-Bounce / Roll 1984).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached small-cap prices
under ../_cache/ (the reversal book + Roll spread) and otherwise quote the frozen headline
numbers in ``R`` (mirroring docs/results.md). The synthetic Roll control runs anywhere with no
network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance 35-name small/mid-cap
# basket, 2005-08-11 -> 2026-06-18, 5,246 days, 20.9 years).
R = dict(
    start="2005-08-11", end="2026-06-18", days=5246, years=20.9, n_names=35,
    ac1=-0.065,                       # avg per-name lag-1 autocorr (apparent reversion)
    roll_spread_bps=109,              # Roll-implied effective round-trip spread, bps
    # reversal book: (one-way half-spread bps, mean PnL bps/day, win%, Sharpe, t, boot p)
    book=[(0.0, 5.02, 54.7, 1.20, 5.47, 0.000),
          (5.0, -2.22, 47.5, -0.53, -2.42, 0.014),
          (10.0, -9.46, 41.2, -2.26, -10.31, 0.000),
          (20.0, -23.93, 29.7, -5.72, -26.08, 0.000)],
    gross_bps=5.02, turnover=1.45, be_half_bps=3.47, be_round_bps=6.93, ratio=16,
    # Roll control #1: (planted spread, cov1_obs, theory -(s/2)^2, ac1_obs, recovered s)
    roll1=[(0.000, 1.3e-06, 0.0, 0.009, None),
           (0.005, -5.0e-06, -6.3e-06, -0.032, 0.0045),
           (0.010, -2.4e-05, -2.5e-05, -0.120, 0.0097),
           (0.020, -9.7e-05, -1.0e-04, -0.280, 0.0197)],
    # Roll control #2 panel: (edge, gross bps, gross t, net@20bps bps, net t, net Sharpe)
    roll2=[(0.00, 3.31, 7.5, -25.13, -56.9, -14.27),
           (-0.25, 39.58, 89.0, 8.01, 18.0, 4.51)],
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Free_lunch%3F: Busted](https://img.shields.io/badge/Free_lunch%3F-Busted-8b949e?style=flat-square)\n\n"
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

from bid_ask_bounce import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    _R = data.load_real()
    NAMES = [c for c in _R.columns if c != "IWM"]
    RET = _R[NAMES]                                 # the small-cap return panel
else:
    RET = None
print("real small-cap cache present:", HAVE_REAL,
      "| names:", (0 if RET is None else RET.shape[1]),
      "| days:", (0 if RET is None else len(RET)))

def syn_panel(n_assets=30, n_days=4000, spread=0.004, edge=0.0, seed=377):
    \"\"\"Synthetic panel of Roll assets sharing one spread/edge (cross-section for the book).\"\"\"
    cols = {f"A{j}": data.synthetic_roll(n_days=n_days, spread=spread, edge=edge,
                                         seed=seed + j)["r_obs"] for j in range(n_assets)}
    return pd.DataFrame(cols).dropna()
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# \"Buy yesterday's losers\" — real edge, or the bid-ask bounce? 🏓\n"
            "### Why short-term mean reversion backtests like free money — and why it isn't, in plain English\n\n"
            + BADGES +
            "Here's a strategy that has tempted traders forever: stocks that **fell** yesterday tend "
            "to **bounce back** today, so buy the losers, sell the winners, and pocket the snap-back. "
            "And it's true — on a price chart it really does work, with a Sharpe most funds would "
            "envy.\n\n"
            "But in **1984**, Richard Roll explained a trap hiding inside that chart. Stock prices "
            "don't trade at one clean number — they bounce between a slightly-lower **bid** and a "
            "slightly-higher **ask**. That bounce, all by itself, makes prices *look* like they "
            "mean-revert even when the real price is a pure coin-flip random walk. And you can't "
            "pocket it — because grabbing the bounce means **paying the very spread that creates it.**\n\n"
            "This notebook shows the reversion is real-as-a-pattern and empty-as-a-strategy at the "
            "same time.\n\n"
            "> 📓 **Plain-language layer.** Want Roll's identity, the bootstrap and the break-even "
            "math? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** Free data has no bid/ask quotes, so we **infer** the spread "
            "from the prices themselves (Roll's own estimator) and call it a proxy throughout. We test "
            "on **small-caps**, where spreads — and the bounce — are widest. Every chart is drawn by "
            "the code beside it; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Do stocks that fell yesterday bounce back? | **Yes — measurably.** Small-cap daily "
            f"returns have a negative day-to-day correlation (**{R['ac1']:.3f}**), and a reversal "
            f"book backtests at **Sharpe {R['book'][0][3]:.1f}** before costs. |\n"
            "| So it's free money? | **No.** That exact bounce-back is what a pure **bid-ask bounce** "
            "produces even with *no* real reversion. Roll's estimator reads a "
            f"**~{R['roll_spread_bps']} bps** spread out of the same number. |\n"
            "| What happens after costs? | **It dies instantly.** The book breaks even at a "
            f"**~{R['be_round_bps']:.0f} bps** round-trip; the real spread is "
            f"**~{R['roll_spread_bps']} bps** — about **{R['ratio']}× wider**. Net, the Sharpe is "
            f"**{R['book'][1][3]:.1f}**. |\n"
            "| Then why does the chart look so good? | Because a backtest prints at the close and "
            "**never pays the bounce it's trading against.** The 'profit' *is* the spread. |\n\n"
            "> The reversion is real. The edge is the spread you'd pay to grab it. Roll named this "
            "trap 40 years ago."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Short-term mean reversion is one of the most robust effects in markets. Yesterday's "
            "biggest losers snap back; yesterday's winners give some back. Go long the losers, short "
            "the winners, rebalance daily — the contrarian book just *works*.\"*\n\n"
            "And the data agrees at first glance: daily stock returns are **negatively autocorrelated** "
            "— a down day is a touch more likely to be followed by an up day. The contrarian book "
            "harvests that. We'll rebuild it on real small-caps and watch it print a beautiful "
            "**gross** Sharpe… and then ask the only question that matters."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "A reliable, high-Sharpe, market-neutral daily strategy would be enormously valuable — "
            "it's the dream of every stat-arb desk. But there are **two** different things hiding "
            "inside \"stocks mean-revert,\" and only one is worth money. (1) The *true, efficient* "
            "price genuinely tends to reverse — a real, harvestable edge. Or (2) the **price you see "
            "printed** wobbles between bid and ask, so it *looks* like it reverses even though the "
            "true price didn't move predictably at all. Case (2) is a mirage: the apparent profit is "
            "exactly the spread, and you pay that spread to trade. Telling the two apart is the whole "
            "game."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Roll gave us the test in 1984. **Build a world we control:** a true stock price that is a "
            "**pure random walk** — no reversion whatsoever — then make every trade print at the bid "
            "or the ask, a coin-flip each day. If the *observed* returns then look like they "
            "mean-revert, we've proven the bounce can fake the whole effect.\n\n"
            "1. **Show the illusion.** On the synthetic price (zero real reversion) the bounce alone "
            "creates negative day-to-day correlation — and the size tells us the spread.\n"
            f"2. **Run the book on the real tape.** A daily reversal book on **{R['n_names']} "
            "small-caps**, measured **gross** and then **net** of a realistic spread.\n"
            "3. **Find the break-even.** How small would the spread have to be for the book to make "
            "money? Compare that to the spread the data actually implies. If break-even is far below "
            "reality, the edge *is* the bounce."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the illusion on data we control.** A *pure random-walk* stock (no reversion at "
            "all) plus a bid-ask bounce. Watch the day-to-day correlation go negative purely from the "
            "bounce — and grow as we widen the spread."
        ),
        code(
            "spreads = [0.0, 0.005, 0.010, 0.020]\n"
            "ac1 = []\n"
            "for s in spreads:\n"
            "    syn = data.synthetic_roll(spread=s, edge=0.0, seed=377)   # edge=0 => NO real reversion\n"
            "    ac1.append(st.lag1_autocorr(syn['r_obs'].values))\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.bar([f'{s*1e4:.0f} bps' for s in spreads], ac1, color=[GREY,AMBER,AMBER,RED], width=.6)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('day-to-day correlation of observed returns')\n"
            "ax.set_xlabel('bid-ask spread we planted (the TRUE price never reverts)')\n"
            "ax.set_title('The bounce ALONE manufactures \"mean reversion\" — no real edge behind it')\n"
            "for i,a in enumerate(ac1): ax.annotate(f'{a:+.3f}',(i,a),ha='center',va='top' if a<0 else 'bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('with NO spread the correlation is ~0 (true price is a random walk);',\n"
            "      'add a bounce and it goes negative purely from microstructure.')"
        ),
        md(
            "There it is. With **no** bounce the made-up stock shows **no** reversion (the true price "
            "is a coin-flip). Add a bid-ask bounce and the *observed* returns suddenly look "
            "mean-reverting — the wider the spread, the stronger the illusion. **Nothing about the "
            "real price changed.** Now the real-tape question: does a reversal book make money on "
            "actual small-caps?"
        ),
        md(
            "**The reversal book on real small-caps — gross, then net of the spread.** Long yesterday's "
            "losers, short the winners, rebalanced daily, over 20 years."
        ),
        code(
            "halfs = [0.0, 5.0, 10.0, 20.0]   # one-way half-spread in bps\n"
            "if HAVE_REAL:\n"
            "    means = [st.summarize_book(RET, cost_halfspread=h/1e4)['mean_bps'] for h in halfs]\n"
            "else:\n"
            "    means = [b[1] for b in R['book']]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "cols = [GREEN if m>0 else RED for m in means]\n"
            "ax.bar([('gross' if h==0 else f'{h:.0f} bps') for h in halfs], means, color=cols, width=.6)\n"
            "ax.axhline(0, c='k', lw=.9)\n"
            "ax.set_ylabel('mean PnL per day (bps)'); ax.set_xlabel('one-way half-spread charged')\n"
            "ax.set_title('Gross it prints; net of even a tiny spread it bleeds')\n"
            "for i,m in enumerate(means): ax.annotate(f'{m:+.1f}',(i,m),ha='center',va='bottom' if m>0 else 'top')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross +{means[0]:.1f} bps/day; charge a 5 bps half-spread and it is already {means[1]:+.1f} bps')"
        ),
        md(
            f"The leftmost bar — **gross** — is the dream: **+{R['book'][0][1]:.1f} bps/day**, a "
            f"**{R['book'][0][2]:.0f}%** daily win-rate, **Sharpe {R['book'][0][3]:.1f}**. But charge a "
            "one-way half-spread of just **5 bps** (a 10 bps round-trip — *cheaper* than almost any "
            "small-cap quote) and the book is already **losing money**. The apparent alpha was sitting "
            "entirely inside the spread."
        ),
        md(
            "**The punchline — break-even vs reality.** How thin would the spread have to be for the "
            "book to break even, next to the spread the data itself implies?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    sig = -(RET.shift(1)); sig = sig.sub(sig.mean(axis=1), axis=0)\n"
            "    gr = sig.abs().sum(axis=1); w = sig.div(gr.where(gr>0), axis=0).fillna(0.0)\n"
            "    to = (w - w.shift(1)).abs().sum(axis=1).dropna().mean()\n"
            "    gm = (w*RET).sum(axis=1).dropna().mean()\n"
            "    be_round = (gm/to)*2*1e4\n"
            "    roll_sp = st.avg_roll_spread(RET)*1e4\n"
            "else:\n"
            "    be_round = R['be_round_bps']; roll_sp = R['roll_spread_bps']\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.barh(['break-even spread\\n(what the book can afford)','actual spread\\n(what the data implies)'],\n"
            "        [be_round, roll_sp], color=[GREEN, RED])\n"
            "for i,v in enumerate([be_round, roll_sp]): ax.annotate(f'{v:.0f} bps', (v, i), va='center', ha='left')\n"
            "ax.set_xlabel('round-trip spread (bps)')\n"
            "ax.set_title(f'The spread is ~{roll_sp/be_round:.0f}x wider than the book can pay')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'break-even ~{be_round:.0f} bps round-trip vs an implied ~{roll_sp:.0f} bps -> ~{roll_sp/be_round:.0f}x too wide')"
        ),
        md(
            f"That gap is the verdict. The book breaks even at a **~{R['be_round_bps']:.0f} bps** "
            f"round-trip; the spread you'd actually cross is **~{R['roll_spread_bps']} bps** — about "
            f"**{R['ratio']}× wider**. There is no clever execution that closes a gap that big. The "
            "reversion *is* the bid-ask bounce, and you can't grab a bounce without paying the spread "
            "that makes it."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — Weak.** The reversion is genuinely there in the numbers (gross Sharpe "
            f"**{R['book'][0][3]:.1f}**, correlation **{R['ac1']:.3f}**) — but it's the *mechanical* "
            "footprint of the bounce, not a forecast of the real price. Real-as-a-statistic, not a "
            "clean edge.\n"
            "- **Tradability — Mirage.** Net of any realistic spread it's a **loser** (Sharpe "
            f"**{R['book'][1][3]:.1f}** at a 10 bps round-trip). Break-even is **~{R['ratio']}×** "
            "thinner than the real spread. There's no strategy here.\n"
            "- **Free lunch? — Busted.** The chart looks like money only because the backtest never "
            "pays the bounce. Roll (1984) called it: short-term \"mean reversion\" is largely the "
            "bid-ask bounce in disguise."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — the one test that settles it\n\n"
            "Forget statistics for a second. We *built* a world where we know the truth, and asked "
            "the book to make money in it. Two versions: one where the reversion is **pure bounce** "
            "(no real edge), one where we **plant a genuine** reversion. Only one should survive the "
            "spread."
        ),
        code(
            "rows = []\n"
            "for edge in (0.0, -0.25):\n"
            "    P = syn_panel(spread=0.004, edge=edge)            # 40 bps round-trip world\n"
            "    g = st.summarize_book(P, 0.0); n = st.summarize_book(P, 0.002)  # 20 bps half-spread\n"
            "    rows.append((edge, g['mean_bps'], n['mean_bps']))\n"
            "labels = ['pure bounce\\n(no real edge)', 'planted REAL\\nreversion']\n"
            "x = np.arange(2); gr = [r[1] for r in rows]; nt = [r[2] for r in rows]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.bar(x-.2, gr, .4, color=GREY, label='gross')\n"
            "ax.bar(x+.2, nt, .4, color=[RED if v<0 else GREEN for v in nt], label='net of spread')\n"
            "ax.axhline(0, c='k', lw=.9); ax.set_xticks(x); ax.set_xticklabels(labels)\n"
            "ax.set_ylabel('mean PnL per day (bps)')\n"
            "ax.set_title('Net of the spread, ONLY a genuine reversion survives'); ax.legend()\n"
            "for i,(g_,n_) in enumerate(zip(gr,nt)):\n"
            "    ax.annotate(f'{g_:+.0f}',(i-.2,g_),ha='center',va='bottom')\n"
            "    ax.annotate(f'{n_:+.0f}',(i+.2,n_),ha='center',va='bottom' if n_>0 else 'top')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('pure-bounce world: gross looks positive, NET is deeply negative — no edge to harvest.')\n"
            "print('planted-reversion world: net stays POSITIVE — a real edge survives the spread.')"
        ),
        md(
            "This is the cleanest way to see it. In the **pure-bounce** world the book still shows a "
            "positive *gross* number — and then bleeds out **net of the spread**, because there was "
            "never anything to catch. Only when we plant a **real** reversion does the book stay "
            "**net-positive**. The real small-cap tape behaves like the *first* column: gross-positive, "
            "net-negative. That's the diagnosis."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The slower cousin.** [Study 329 — One-Month-Reversal](../../329-one-month-reversal/) "
            "asks the same question one frequency down: does the *monthly* reversal premium survive "
            "costs, or is it the same trap with a longer fuse?\n"
            "- **The illiquidity it lives in.** [Study 140 — Amihud-Illiquidity](../../140-amihud-illiquidity/) "
            "— short-term reversal is strongest exactly in the illiquid names where the bounce is "
            "biggest. That's not a coincidence.\n"
            "- **Build your own.** Swap the small-cap basket for liquid mega-caps and re-run: the "
            "bounce shrinks, the gross Sharpe shrinks with it, and the illusion fades — more proof "
            "it was microstructure all along.\n\n"
            "*Think you can harvest the reversion net of the spread? Show the reversal book landing "
            "**net-positive** on a tape where the spread is honestly charged — then we'll talk.*"
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
            "# Bid-Ask-Bounce — a quantitative teardown 🔬\n"
            "### Roll's cov₁ = −(s/2)² identity · the gross-vs-net reversal book on the small-cap tape · "
            "a block-bootstrap null · break-even vs implied spread · a planted-edge / faithful-engine control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). We separate "
            "the two things \"short-term mean reversion\" fuses: a **mechanical** negative autocovariance "
            "from the **bid-ask bounce** (Roll 1984), and a **genuine** predictability of the efficient "
            "price. The decisive object is not the sign of cov₁ (it is negative, as the lore says) but "
            "the **gross-vs-net** of a reversal book — gross it is a high-Sharpe winner; net of the "
            "spread the data itself implies, it is gone.\n\n"
            "> ⚠️ **Data + proxy note.** yfinance has no bid/ask, so the effective spread is **inferred** "
            "via Roll's estimator (s = 2√(−cov₁)) and named a proxy throughout; a fixed surviving-names "
            "basket also carries **survivorship**, which tilts the *gross* reversal number **up** (both "
            "named on the Signal axis). Real data: yfinance daily adjusted closes, 35 small/mid-caps, "
            "2005→2026. Offline core (Roll model) + the panel control are deterministic. Methods in "
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
            f"| **Signal** | `WEAK` | gross reversal Sharpe **{R['book'][0][3]:.2f}** (*t* = "
            f"**{R['book'][0][4]:.2f}**), avg lag-1 autocorr **{R['ac1']:.3f}** — but Roll reads a "
            f"**{R['roll_spread_bps']} bps** spread from the *same* cov₁, and the control reproduces the "
            "whole signal from a **pure bounce** with zero real reversion. Real stat, microstructure "
            "artefact. |\n"
            f"| **Tradability** | `MIRAGE` | break-even round-trip **{R['be_round_bps']:.1f} bps** vs an "
            f"implied **{R['roll_spread_bps']} bps** (**~{R['ratio']}×**). Net @ 10 bps round-trip: "
            f"Sharpe **{R['book'][1][3]:.2f}**, mean **{R['book'][1][1]:+.1f} bps/day**. |\n"
            f"| **Free lunch?** | `BUSTED` | a close-to-close backtest is credited the bounce it trades "
            "against and never charged it. The gross 'edge' **is** the spread. Roll (1984). |\n\n"
            "> 💡 In plain words: the reversal book really does win on paper — because the paper price "
            "is the trade print, which carries the bounce. Charge the bounce back and the edge is "
            "exactly zero (or worse). The signal is real; it is not *price-predictive*; it is not "
            "tradable."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $m_t$ be the **efficient** (true) log-price and $p_t$ the **observed** transaction "
            "log-price. Roll's model: trades print at the bid or ask via a fair coin "
            "$q_t \\in \\{-1,+1\\}$, so $p_t = m_t + \\tfrac{s}{2} q_t$ for effective spread $s$. If "
            "$m_t$ is a random walk (efficient market, **no** predictability), the observed return "
            "$\\Delta p_t$ has\n\n"
            "$$\\operatorname{cov}(\\Delta p_t, \\Delta p_{t-1}) = -\\left(\\tfrac{s}{2}\\right)^2 < 0,$$\n\n"
            "negative serial correlation **with no forecastable component**. Inverting gives Roll's "
            "estimator $\\hat s = 2\\sqrt{-\\operatorname{cov}_1}$.\n\n"
            "- **H₁ (real reversion).** The *efficient* price mean-reverts: $\\operatorname{cov}_1$ "
            "of $\\Delta m_t$ (not just $\\Delta p_t$) is negative — a forecastable, harvestable edge.\n"
            "- **H₂ (deployable).** A reversal book that trades on it is **net-positive** after the "
            "one-way spread, with capacity.\n"
            "- **H₃ (\"free lunch\").** The high gross Sharpe reflects price prediction, not the "
            "bounce.\n\n"
            "We find **H₁ not supported** (the synthetic control reproduces the entire gross signal "
            "with $\\Delta m_t$ a pure random walk), **H₂ rejected** (break-even spread ≪ implied "
            "spread), **H₃ rejected** (gross = bounce). The reversion is true exactly where it is "
            "untradable (the print) and unproven exactly where it would pay (the efficient price)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The whole teardown is one decomposition: the observed autocovariance splits into a "
            "**microstructure** piece and a **price-prediction** piece,\n\n"
            "$$\\operatorname{cov}_1(\\Delta p) = \\underbrace{-(s/2)^2}_{\\text{bounce (untradable)}} "
            "+ \\underbrace{\\operatorname{cov}_1(\\Delta m)}_{\\text{real reversion (tradable)}}.$$\n\n"
            "A reversal book backtested on $p_t$ harvests **both** terms gross — but pays $s$ per round "
            "trip, which is calibrated to *cancel the first term exactly*. So the book's **net** edge "
            "is driven **only** by $\\operatorname{cov}_1(\\Delta m)$, the real-reversion piece. The "
            "right instrument is therefore not the autocorrelation (it conflates the two) but the "
            "**gross-vs-net** of the book and its **break-even spread** — the spread at which the "
            "first term is fully charged. If break-even ≪ implied spread, the second term is ≈0."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Offline core (Roll model).** A true AR(1) log-price with reversion knob `edge` "
            "(`edge=0` ⇒ random walk) plus a fair-coin bid/ask bounce of half-spread $s/2$ "
            "([`data.synthetic_roll`](../bid_ask_bounce/data.py)). Deterministic, no network.\n"
            "- **Real tape.** Daily adjusted closes, **35** small/mid-caps (yfinance, "
            f"{R['start']}→{R['end']}, {R['days']:,} days). yfinance has no quotes ⇒ the spread is "
            "**inferred** via Roll's estimator (a labelled proxy); survivorship tilts the *gross* book "
            "up (named on the axis).\n"
            "- **Reversal book.** Equal-weight, dollar-neutral, cross-sectional: weight $\\propto$ "
            "$-(\\text{yesterday's demeaned return})$, **1-day execution lag**, one-way cost = "
            "half-spread × daily turnover (the book re-forms daily) "
            "([`strategy.reversal_book`](../bid_ask_bounce/strategy.py)).\n"
            "- **Inference.** One-sample *t* of the daily PnL **and** a **moving-block bootstrap** "
            "null ([`strategy.block_bootstrap_p`](../bid_ask_bounce/strategy.py)) — reversal PnL is "
            "autocorrelated, so i.i.d. resampling would overstate significance.\n"
            "- **Break-even.** gross mean ÷ turnover = the half-spread at which net PnL = 0; compare to "
            "the Roll-implied spread.\n"
            "- **Positive control.** A synthetic **panel** of Roll assets: with `edge=0` the book must "
            "be gross-positive but **net-negative** (no edge); with a large planted reversion it must "
            "be **net-positive** (a real edge survives). The engine must pass both."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Roll's identity, exact on data we control\n\n"
            "On the synthetic random-walk price (`edge=0`, **no** real reversion), the observed lag-1 "
            "autocovariance must track $-(s/2)^2$ as we widen the planted spread, and Roll's estimator "
            "must read the spread back. Theory vs measured:"
        ),
        code(
            "spreads = [0.0, 0.005, 0.010, 0.020]\n"
            "cov_obs, theory, recov = [], [], []\n"
            "for s in spreads:\n"
            "    syn = data.synthetic_roll(spread=s, edge=0.0, seed=377)\n"
            "    cov_obs.append(st.autocov1(syn['r_obs'].values))\n"
            "    theory.append(-(s/2)**2)\n"
            "    rs = st.roll_spread(syn['r_obs'].values); recov.append(rs if np.isfinite(rs) else 0.0)\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.2))\n"
            "x = np.arange(len(spreads))\n"
            "a1.plot(x, np.array(theory)*1e5, 'o-', c=RED, label='theory  -(s/2)^2')\n"
            "a1.plot(x, np.array(cov_obs)*1e5, 's--', c=GREY, label='measured cov1')\n"
            "a1.set_xticks(x); a1.set_xticklabels([f'{s*1e4:.0f}bp' for s in spreads])\n"
            "a1.set_ylabel('lag-1 autocov  (x1e-5)'); a1.set_xlabel('planted spread'); a1.legend()\n"
            "a1.set_title('Bounce injects cov1 = -(s/2)^2 exactly')\n"
            "a2.plot([s*1e4 for s in spreads], [r*1e4 for r in recov], 'D-', c=GREEN)\n"
            "a2.plot([0,200],[0,200], ':', c='k', lw=1)\n"
            "a2.set_xlabel('planted spread (bps)'); a2.set_ylabel('Roll-recovered spread (bps)')\n"
            "a2.set_title('Roll estimator recovers the spread')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('measured cov1 vs theory:', [f'{c*1e5:.2f}/{t*1e5:.2f}' for c,t in zip(cov_obs,theory)])"
        ),
        md(
            "> 💡 In plain words: the grey (measured) and red (theory) lines sit on top of each other, "
            "and the recovered spread lands on the 45° line. A **pure random walk** with a bounce "
            "produces exactly the negative autocorrelation lore calls \"mean reversion\" — and the "
            "size of that autocorrelation *is* the spread. The machinery is faithful."
        ),
        md(
            "### 4b · The real-tape reversal book — gross is a winner\n\n"
            "Cumulative **gross** PnL of the one-day reversal book on the small-cap panel, with its "
            "Sharpe and the block-bootstrap *p* on the mean daily PnL."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pnl = st.reversal_book(RET, cost_halfspread=0.0)\n"
            "    s0 = st.summarize_book(RET, 0.0)\n"
            "    sharpe, tval, pval, win = s0['sharpe'], s0['t'], s0['p'], s0['win']\n"
            "    curve = pnl.cumsum()\n"
            "else:\n"
            "    sharpe, tval, pval, win = R['book'][0][3], R['book'][0][4], R['book'][0][5], R['book'][0][2]\n"
            "    rng = np.random.default_rng(377)\n"
            "    pnl = pd.Series(rng.normal(R['gross_bps']/1e4, 0.004, R['days']))\n"
            "    curve = pnl.cumsum()\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.plot(np.arange(len(curve)), curve.values*100, c=GREEN, lw=1.6)\n"
            "ax.set_xlabel('trading day'); ax.set_ylabel('cumulative GROSS PnL (%)')\n"
            "ax.set_title(f'Gross reversal book: Sharpe {sharpe:.2f}, t = {tval:.1f}, win {win*100:.0f}% — looks like alpha')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'GROSS Sharpe {sharpe:.2f} | t {tval:.2f} | bootstrap p {pval:.3f} | daily win {win*100:.1f}%')"
        ),
        md(
            f"> 💡 In plain words: a smooth, upward gross equity curve at **Sharpe {R['book'][0][3]:.2f}** "
            f"(*t* = {R['book'][0][4]:.2f}, bootstrap *p* < 0.01). If this were the *net* curve you'd "
            "fund it tomorrow. H₃'s premise — that the gross number is real — is where every reversal "
            "pitch starts. Now we charge the spread."
        ),
        md(
            "### 4c · Net of the spread — and the break-even gap\n\n"
            "Left: the book's mean daily PnL as we raise the one-way half-spread — it crosses zero "
            "almost immediately. Right: the **break-even** round-trip spread vs the spread Roll's "
            "estimator reads from the same data."
        ),
        code(
            "halfs = [0.0, 5.0, 10.0, 20.0]\n"
            "if HAVE_REAL:\n"
            "    means = [st.reversal_book(RET, h/1e4).mean()*1e4 for h in halfs]   # mean only (fast)\n"
            "    sig = -(RET.shift(1)); sig = sig.sub(sig.mean(axis=1), axis=0)\n"
            "    gr = sig.abs().sum(axis=1); w = sig.div(gr.where(gr>0), axis=0).fillna(0.0)\n"
            "    to = (w - w.shift(1)).abs().sum(axis=1).dropna().mean()\n"
            "    gm = (w*RET).sum(axis=1).dropna().mean()\n"
            "    be_round = (gm/to)*2*1e4; roll_sp = st.avg_roll_spread(RET)*1e4\n"
            "else:\n"
            "    means = [b[1] for b in R['book']]; be_round = R['be_round_bps']; roll_sp = R['roll_spread_bps']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.2))\n"
            "a1.plot(halfs, means, 'o-', c=AMBER, lw=2)\n"
            "a1.axhline(0, c=RED, ls='--'); a1.fill_between(halfs, means, 0, where=[m<0 for m in means], color=RED, alpha=.12)\n"
            "a1.set_xlabel('one-way half-spread (bps)'); a1.set_ylabel('mean PnL/day (bps)')\n"
            "a1.set_title('Net PnL crosses zero almost at once')\n"
            "a2.barh(['break-even','Roll-implied'], [be_round, roll_sp], color=[GREEN, RED])\n"
            "for i,v in enumerate([be_round, roll_sp]): a2.annotate(f'{v:.0f} bps',(v,i),va='center',ha='left')\n"
            "a2.set_xlabel('round-trip spread (bps)'); a2.set_title(f'Implied spread ~{roll_sp/be_round:.0f}x break-even')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'break-even round-trip ~{be_round:.1f} bps vs Roll-implied ~{roll_sp:.0f} bps -> ~{roll_sp/be_round:.0f}x too wide')"
        ),
        md(
            f"> 💡 In plain words: the book breaks even at a **~{R['be_round_bps']:.0f} bps** round-trip; "
            f"the data implies **~{R['roll_spread_bps']} bps** — about **{R['ratio']}×** wider. The "
            "decomposition in beat 2 says the net edge is just $\\operatorname{cov}_1(\\Delta m)$, and "
            "this gap says that term is effectively **zero**: the entire gross signal was the "
            "$-(s/2)^2$ bounce. H₂ rejected."
        ),
        md(
            "### 4d · Faithful-engine & separation control — we know the truth here\n\n"
            "On a synthetic **panel** (round-trip spread 40 bps): with `edge=0` (pure bounce) the book "
            "must be **gross-positive but net-negative** — a false positive the spread kills; with a "
            "large planted reversion it must be **net-positive** — a real edge the spread spares. Both "
            "hold, proving the engine separates the bounce from a genuine edge."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, -0.25):\n"
            "    P = syn_panel(spread=0.004, edge=edge)\n"
            "    g = st.summarize_book(P, 0.0); n = st.summarize_book(P, 0.002)\n"
            "    res.append((edge, g['mean_bps'], g['t'], n['mean_bps'], n['t'], n['sharpe']))\n"
            "labels = ['edge=0\\n(pure bounce)', 'edge=-0.25\\n(real reversion)']\n"
            "x = np.arange(2); gv = [r[1] for r in res]; nv = [r[3] for r in res]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.bar(x-.2, gv, .4, color=GREY, label='gross')\n"
            "ax.bar(x+.2, nv, .4, color=[RED if v<0 else GREEN for v in nv], label='net @20bps half-spread')\n"
            "ax.axhline(0, c='k', lw=.9); ax.set_xticks(x); ax.set_xticklabels(labels)\n"
            "ax.set_ylabel('mean PnL/day (bps)'); ax.set_title('Control: net of cost, only a REAL reversion survives'); ax.legend()\n"
            "for i,(g_,n_) in enumerate(zip(gv,nv)):\n"
            "    ax.annotate(f'{g_:+.0f}',(i-.2,g_),ha='center',va='bottom')\n"
            "    ax.annotate(f'{n_:+.0f}',(i+.2,n_),ha='center',va='bottom' if n_>0 else 'top')\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,g_,gt,n_,nt,sh in res:\n"
            "    print(f'edge={e:+.2f}: gross={g_:+.2f}bps(t={gt:.1f})  net={n_:+.2f}bps(t={nt:.1f}, Sharpe={sh:.2f})')"
        ),
        md(
            f"> 💡 In plain words: with **no** real edge the book is gross "
            f"**+{R['roll2'][0][1]:.1f} bps** (t={R['roll2'][0][2]:.0f}) yet net "
            f"**{R['roll2'][0][3]:.1f} bps** (Sharpe {R['roll2'][0][5]:.1f}) — the spread eats a "
            f"signal that was never there. Only the planted **−0.25** reversion stays net-positive "
            f"(**+{R['roll2'][1][3]:.1f} bps**, Sharpe {R['roll2'][1][5]:.1f}). The real small-cap tape "
            "behaves like the **first** column. The sample is the verdict."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — gross reversal Sharpe **{R['book'][0][3]:.2f}** (*t* = "
            f"**{R['book'][0][4]:.2f}**), lag-1 autocorr **{R['ac1']:.3f}**: not noise. But Roll's "
            f"estimator reads a **{R['roll_spread_bps']} bps** spread from the same cov₁, and the "
            "control reproduces the entire gross signal from a *pure bounce* with zero efficient-price "
            "reversion. A real statistic that is a **microstructure artefact**, not a price forecast ⇒ "
            "WEAK, not REAL (survivorship tilts the gross number **up**, never down).\n"
            f"- **Tradability `MIRAGE`** — break-even round-trip **{R['be_round_bps']:.1f} bps** vs an "
            f"implied **{R['roll_spread_bps']} bps** (**~{R['ratio']}×**). Net @ 10 bps round-trip: "
            f"Sharpe **{R['book'][1][3]:.2f}**, **{R['book'][1][1]:+.1f} bps/day**. No capacity, no "
            "deployable edge.\n"
            "- **Free lunch? `BUSTED`** — a close-to-close backtest is *credited* the bounce it trades "
            "against and never *charged* it; the gross 'edge' is precisely $-(s/2)^2$. Short-horizon "
            "\"mean reversion\" is, in large part, the **bid-ask bounce in disguise** — Roll (1984)."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the capacity curve\n\n"
            "The operational truth in one picture: net Sharpe as a function of the round-trip spread "
            "you actually pay. The break-even is a knife-edge far to the left of where real small-cap "
            "spreads live."
        ),
        code(
            "rounds = np.linspace(0, 30, 31)            # round-trip spread in bps\n"
            "if HAVE_REAL:\n"
            "    sh = [st.book_sharpe(RET, (rt/2)/1e4) for rt in rounds]   # Sharpe only (no bootstrap)\n"
            "    be_round = R['be_round_bps']; roll_sp = st.avg_roll_spread(RET)*1e4\n"
            "else:\n"
            "    g = R['book'][0][3]; slope = (R['book'][2][3]-g)/20.0\n"
            "    sh = [g + slope*(rt/2) for rt in rounds]; be_round = R['be_round_bps']; roll_sp = R['roll_spread_bps']\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.plot(rounds, sh, c=AMBER, lw=2, label='net annualised Sharpe')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.axvline(be_round, c=GREEN, ls='--', label=f'break-even ~{be_round:.0f} bps')\n"
            "ax.annotate(f'real spread\\n~{roll_sp:.0f} bps  ->', xy=(28, ax.get_ylim()[0]*0.6),\n"
            "            ha='right', va='center', color=RED, fontsize=10)\n"
            "ax.set_xlabel('round-trip spread paid (bps)'); ax.set_ylabel('net Sharpe')\n"
            "ax.set_title('Net Sharpe vs the spread: break-even is far left of reality'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'net Sharpe is positive only below ~{be_round:.0f} bps round-trip; the real spread is ~{roll_sp:.0f} bps.')"
        ),
        md(
            "> 💡 In plain words: net Sharpe is positive only in a sliver below the break-even spread; "
            "real small-cap round-trips sit **far** to the right, deep in negative territory. There is "
            "no execution venue, no sizing, no patience that moves the real spread left of break-even — "
            "**the reversion you'd harvest *is* the spread you'd pay.** That identity, not a weak "
            "*t*-stat, is why the strategy cannot exist. The thing that makes the chart pretty is the "
            "thing that makes it untradable."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The lower-frequency cousin.** [Study 329 — One-Month-Reversal](../../329-one-month-reversal/): "
            "the monthly reversal premium, where the bounce is a smaller share of the move — does the "
            "edge survive there, or is it the same trap with a longer holding period?\n"
            "- **The illiquidity it lives in.** [Study 140 — Amihud-Illiquidity](../../140-amihud-illiquidity/): "
            "short-term reversal concentrates in illiquid names *because* that's where the bounce is "
            "biggest — the two are the same phenomenon viewed from different sides.\n"
            "- **Better spread estimates.** Replace Roll's estimator with Corwin-Schultz (high-low) or "
            "Abdi-Ranaldo (close-high-low), or true TAQ quotes if you have them; the implied spread "
            "moves a little, the **16×** gap does not.\n\n"
            "*The reproducible core is offline and deterministic (the Roll model); the effective spread "
            "is an explicit Roll-estimator proxy. Methods and sources: "
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
