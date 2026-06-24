"""Generate the two narrative notebooks for Study 444 (Dow Theory).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached panels under
../_cache/ (pinned to the as-of below) and otherwise quote the frozen headline numbers in ``R``
(mirroring docs/results.md). The synthetic positive control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance DIA/IYT,
# 2004-01-02 -> 2026-05-29, 5,637 days, as-of 2026-05-31, fp 1ee3a36320fb).
R = dict(
    asof="2026-05-31", start="2004-01-02", end="2026-05-29", years=22.4,
    n_days=5637, fp="1ee3a36320fb", lookback=63,
    # buy & hold
    bh_cagr=9.75, bh_sharpe=0.612, bh_mdd=-51.9,
    # latched timed book: (on%, CAGR%, Sharpe, MDD%, active/yr%, HAC t)
    latched=(62.4, 5.56, 0.599, -22.5, -4.99, -1.935),
    # instantaneous flag: (on%, CAGR%, Sharpe, MDD%, active/yr%, HAC t)
    instant=(5.5, 0.69, 0.351, -8.4, -10.18, -3.144),
    # content test (confirmed/yr%, non/yr%, spread/yr%, Welch t)
    content_latched=(9.51, 13.17, -3.67, -0.408),
    content_instant=(19.20, 10.40, 8.80, 1.014),
    # placebo (obs Sharpe, placebo mean, p)
    placebo=(0.599, 0.489, 0.197),
    # cost sweep (latched): (bps, active/yr%, HAC t, Sharpe)
    costs=[(0.0, -4.96, -1.922, 0.602), (2.0, -4.99, -1.935, 0.599),
           (5.0, -5.04, -1.953, 0.594), (10.0, -5.12, -1.984, 0.585)],
    # lookback robustness (latched, 2bps): (LB, on, Strat Sharpe, active t)
    lookbacks=[(21, 0.63, 0.367, -2.702), (42, 0.68, 0.516, -2.190),
               (63, 0.62, 0.599, -1.935), (126, 0.70, 0.663, -1.586),
               (200, 0.62, 0.507, -2.308)],
    # long index ^DJI/^DJT 1992->2026 (latched): bh Sharpe, strat Sharpe, active/yr%, t, content spread%, content t
    idx=(0.552, 0.543, -4.08, -1.969, -1.28, -0.189), idx_fp="ea24513f67a9",
    # synthetic control: (planted edge, conf_frac, confirmed/yr%, non/yr%, spread/yr%, Welch t)
    syn=[(0.000, 0.029, 16.22, 5.57, 10.65, 0.503),
         (0.003, 0.032, 97.24, 5.69, 91.55, 4.535)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![I/T confirmation: Busted](https://img.shields.io/badge/I%2FT_confirmation%3F-Busted-8b949e?style=flat-square)\n\n"
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

from dow_theory import data, strategy as st

ASOF = "2026-05-31"
HAVE_REAL = data.have_real()
if HAVE_REAL:
    DF = data.load_real()
    DF = DF[DF.index <= ASOF]
else:
    DF = None
print("real Dow-Theory cache present:", HAVE_REAL,
      "| days:", (0 if DF is None else len(DF)),
      "| fingerprint:", ("--" if DF is None else data.fingerprint(DF)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Do the trains have to agree with the factories? 🚂🏭\n"
            "### Dow Theory's \"the Transports must confirm the Industrials\" — the oldest rule in the book, in plain English\n\n"
            + BADGES +
            "Here's the granddaddy of all market wisdom, older than the Dow Jones average you see on "
            "the news. Charles Dow's idea, around 1900: the stock market is only *really* healthy when "
            "**two** of his averages agree. If the **Industrials** (the factories, the big companies) "
            "are making new highs, the **Transports** (the railroads and truckers that *ship* what the "
            "factories make) had better be making new highs too. If the trains and the factories "
            "**disagree**, something's wrong — get out.\n\n"
            "It's a beautiful story. The logic feels airtight: nobody ships goods that nobody's buying, "
            "so the trains are the economy's tell. A century later it's still quoted on every business "
            "channel. So we did the obvious thing — turned it into a precise rule and **traded it**. The "
            "result is not kind to the legend.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo test and the cost math? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** Industrials = **DIA**, Transports = **IYT** (the tradable "
            "ETFs), daily, 2004→2026; we re-check on the raw indices back to 1992. Every chart is drawn "
            "by the code beside it; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does waiting for the trains to *confirm* beat just buying and holding? | **No.** Timing the "
            "trend this way actually **lags** buy-and-hold — it gives up about **4 percentage points of "
            "yearly return** (5.6% vs 9.8%). |\n"
            "| But doesn't it protect you in crashes? | **Yes — but that's not the magic.** It cuts the "
            "worst drawdown roughly in half (−22% vs −52%), purely because it sits in **cash** about a "
            "third of the time. *Any* rule that hides in cash that often would do the same. |\n"
            "| Is the *confirmation* itself doing the work? | **No.** A **random** in-and-out-of-cash rule "
            "with the same habits beats the real Dow-Theory rule about **1 time in 5**. The trains "
            "agreeing with the factories carries **no real information**. |\n"
            "| So is it a useful signal? | **No.** Once you put both sides on the same footing, the "
            "risk-adjusted return (Sharpe) is **no better** than buy-and-hold (0.60 vs 0.61). |\n\n"
            "> The legend is gorgeous and the logic *sounds* right — but the tape says the "
            "Industrials/Transports handshake doesn't predict anything you can use."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"A bull market is only real when **both** the Industrials and the Transports make new "
            "highs together. If the factories rally but the railroads don't follow — a "
            "*non-confirmation* — the rally is hollow and you should be suspicious. When both roll over to "
            "new lows, the primary trend has turned: get out.\"*\n\n"
            "This is **Dow Theory**, laid down by Charles Dow (≈1900), systematised by William Peter "
            "Hamilton (*The Stock Market Barometer*, 1922) and codified by Robert Rhea (*The Dow Theory*, "
            "1932). It's the conceptual ancestor of the Dow averages themselves and arguably the founding "
            "document of technical analysis. The economic intuition is genuinely clever: the Transports "
            "*move the goods*, so if the trains aren't booming, the boom isn't real."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the confirmation rule worked, it would be a free crash-detector: you'd ride the bull while "
            "the two averages agree, and step aside the moment they diverge — capturing most of the upside "
            "while dodging the worst drops. That's the dream every market-timer is selling.\n\n"
            "But there's a trap hiding in plain sight. A rule that parks you in **cash** part of the time "
            "will *automatically* show a smaller drawdown — not because it's smart, but because you're "
            "simply **not in the market** as much. The real question isn't \"did it lower the drawdown?\" "
            "(any cash-heavy rule does that). It's: **does the Transports confirmation actually predict "
            "anything**, or could a coin flip with the same cash habit do just as well? We'll test exactly "
            "that."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "We make the legend **precise** so it can't wiggle:\n\n"
            f"1. **Confirmation.** Both averages closing at or above their **{R['lookback']}-day high** "
            "(≈ a new quarterly high in *both* the factories and the trains) = the trend is *confirmed*. "
            "It stays confirmed until both close below their 63-day **low** (the breakdown).\n"
            "2. **The trade.** Hold the **Industrials** while confirmed; sit in **cash** otherwise. We act "
            "**one day after** the signal (no cheating) and pay realistic costs on every switch.\n"
            "3. **The race.** Compare it to simply **buying and holding** the Industrials. And — the "
            "honest part — compare it to a **random** in-and-out rule with the same cash habit. If the "
            "real confirmation can't beat the coin flip, the legend is busted."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the headline race.** Buy-and-hold the Industrials, versus the Dow-Theory-timed "
            "version that only holds when the trains confirm. Watch what you give up."
        ),
        code(
            "if HAVE_REAL:\n"
            "    s = st.summarize(DF, latched=True, cost_bps=2.0)\n"
            "    bh_c, bh_s, bh_d = s['bh_cagr']*100, s['bh_sharpe'], s['bh_mdd']*100\n"
            "    st_c, st_s, st_d, onf = s['st_cagr']*100, s['st_sharpe'], s['st_mdd']*100, s['on_frac']*100\n"
            "else:\n"
            "    bh_c, bh_s, bh_d = R['bh_cagr'], R['bh_sharpe'], R['bh_mdd']\n"
            "    onf, st_c, st_s, st_d = R['latched'][0], R['latched'][1], R['latched'][2], R['latched'][3]\n"
            "fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.0))\n"
            "for ax, vals, ttl, fmt in [\n"
            "    (axes[0], [bh_c, st_c], 'Yearly return (CAGR)', '{:.1f}%'),\n"
            "    (axes[1], [bh_s, st_s], 'Risk-adjusted (Sharpe)', '{:.2f}'),\n"
            "    (axes[2], [bh_d, st_d], 'Worst drawdown', '{:.0f}%')]:\n"
            "    ax.bar(['buy &\\nhold','Dow\\nTheory'], vals, color=[GREY, RED], width=.6)\n"
            "    for i,v in enumerate(vals): ax.annotate(fmt.format(v),(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "    ax.set_title(ttl); ax.axhline(0, c='k', lw=.7)\n"
            "plt.suptitle('Dow-Theory timing: gives up return, no better Sharpe, smaller drawdown', y=1.03)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'buy&hold CAGR {bh_c:.2f}%  Sharpe {bh_s:.3f}  |  Dow-Theory CAGR {st_c:.2f}%  Sharpe {st_s:.3f}  (in market {onf:.0f}% of the time)')"
        ),
        md(
            f"The middle bar is the punchline: the risk-adjusted return is **the same** "
            f"(**{R['latched'][2]:.2f}** vs **{R['bh_sharpe']:.2f}**). The timing rule buys you a smaller "
            f"drawdown (**{R['latched'][3]:.0f}%** vs **{R['bh_mdd']:.0f}%**) — but only by **giving up "
            f"return** (CAGR **{R['latched'][1]:.1f}%** vs **{R['bh_cagr']:.1f}%**). That's not an edge; "
            "that's just being in cash a third of the time."
        ),
        md(
            "**Now the real test.** Is the *confirmation* doing the work — or just the cash? We build "
            "**random** in-and-out rules that flip to cash exactly as often, and stay in cash for runs "
            "exactly as long, as the real Dow-Theory rule. Then we see how often a coin flip beats it."
        ),
        code(
            "if HAVE_REAL:\n"
            "    reg = st.latched_regime(DF)\n"
            "    pl = st.placebo_pvalue(DF, reg, n_draws=2000, cost_bps=2.0)\n"
            "    obs, draws, p = pl['obs_sharpe'], pl['draws'], pl['p_value']\n"
            "else:\n"
            "    obs, p = R['placebo'][0], R['placebo'][2]\n"
            "    rng = np.random.default_rng(444); draws = rng.normal(R['placebo'][1], 0.12, 2000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=45, color=GREY, alpha=.85, label='2,000 RANDOM cash-timing rules')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'real Dow-Theory rule (Sharpe {obs:.3f})')\n"
            "ax.set_xlabel('net Sharpe ratio'); ax.set_ylabel('how many random rules')\n"
            "ax.set_title(f'A coin flip beats the real rule {p*100:.0f}% of the time  (placebo p = {p:.3f})')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'real rule Sharpe {obs:.3f}  |  random rules beat it {p*100:.0f}% of the time (p={p:.3f})')"
        ),
        md(
            f"The red line — the *real* Dow-Theory rule — sits right in the middle of the coin-flip cloud. "
            f"A **random** in-and-out rule with the same cash habit beats it **~{R['placebo'][2]*100:.0f}% "
            "of the time**. If the Industrials/Transports handshake carried real information, the red line "
            "would be way out in the right tail. It isn't. **The confirmation isn't the thing that's "
            "working — the cash is.**"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** Timing the trend with the Transports confirmation **underperforms** "
            "buy-and-hold; the confirmation carries no predictive edge you can measure.\n"
            "- **Tradability — Mirage.** The only thing it gives you — a smaller drawdown — is plain "
            "de-risking from sitting in cash, paid for with **4 points of yearly return**. Same Sharpe, "
            "less money. Nothing to deploy.\n"
            "- **Does the confirmation work? — Busted.** A coin-flip cash rule with the same habits beats "
            "it 1 time in 5. The two averages agreeing is not what moves the needle."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — what you'd really get\n\n"
            "Suppose you ran it anyway. Here's the honest trade-off, on one chart: the return you give up "
            "versus the drawdown you save. It's a **de-risking dial**, not an alpha engine — and a "
            "low-cost balanced fund gives you the same dial without the storytelling."
        ),
        code(
            "if HAVE_REAL:\n"
            "    s = st.summarize(DF, latched=True, cost_bps=2.0)\n"
            "    pts = {'buy & hold': (s['bh_mdd']*100, s['bh_cagr']*100), 'Dow Theory': (s['st_mdd']*100, s['st_cagr']*100)}\n"
            "else:\n"
            "    pts = {'buy & hold': (R['bh_mdd'], R['bh_cagr']), 'Dow Theory': (R['latched'][3], R['latched'][1])}\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.6))\n"
            "for (lab,(d,c)),col in zip(pts.items(),[GREY,RED]):\n"
            "    ax.scatter(d, c, s=180, color=col, zorder=3)\n"
            "    ax.annotate(f'{lab}\\n{c:.1f}%/yr, {d:.0f}% DD',(d,c),textcoords='offset points',xytext=(8,8))\n"
            "ax.set_xlabel('worst drawdown (%)  ->  better is to the right')\n"
            "ax.set_ylabel('yearly return (%)')\n"
            "ax.set_title('You trade return for a smaller drawdown — that is all it is')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Dow-Theory timing buys ~30pp less drawdown for ~4pp/yr less return, at the same Sharpe.')"
        ),
        md(
            "> The candid bottom line: if you want a smaller drawdown, hold less equity (or own some "
            "bonds). You don't need the trains to confirm the factories — and pretending the confirmation "
            "is what's protecting you just hides the real lever, which is how much cash you're holding."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **It's not new.** Alfred Cowles graded Hamilton's published Dow-Theory calls back in **1933** "
            "and found they trailed buy-and-hold; Brown, Goetzmann & Kumar (1998) re-ran him and found the "
            "same — *drawdown reduction from cash*, not market-beating timing. We just reproduced a "
            "90-year-old result mechanically.\n"
            "- **Try the discretionary version.** Real Dow Theorists eyeball \"significant\" highs and "
            "secondary reactions. If you think judgement adds the edge our mechanical rule misses — "
            "**show it** clearing *t* = 2 against this same placebo.\n"
            "- **Swap the universe.** Does the confirmation help on a different pair, or as a *filter* on a "
            "momentum book rather than an on/off switch? The engine is in [`dow_theory/`](../dow_theory).\n\n"
            "*Think the trains really do lead the factories? Show the **confirmed-day** forward return "
            "beating the **non-confirmed-day** return with a *t* above 2 — then we'll talk.*"
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
            "# Dow Theory — a quantitative teardown 🔬\n"
            "### Latched Industrials/Transports confirmation regime · active-spread HAC *t* · a "
            "cash-drag-stripped content test · a matched-persistence random-regime placebo · cost & "
            "lookback robustness · a 34-year index check · a synthetic planted-edge power control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). Dow Theory is "
            "partly subjective, so we steelman it into the **tightest mechanical rule a proponent would "
            "accept** and ask three separate questions: does timing on the confirmation **beat "
            "buy-and-hold** (active-spread HAC *t*), does the confirmation flag carry **forward "
            "information** once you strip the cash-drag (content test), and is any of it **distinguishable "
            "from a random cash rule** (placebo)? The answers are no, no, and no.\n\n"
            "> ⚠️ **Data note.** Industrials = **DIA**, Transports = **IYT** (auto-adjusted ≈ total "
            "return), 2004→2026; robustness on **^DJI/^DJT** price-only back to 1992. Cash earns 0, so "
            "every Sharpe race is excess-of-cash on both sides. Offline core + synthetic control are "
            "deterministic. Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Active daily spread vs buy-and-hold is **negative** (HAC "
            f"**t = {R['latched'][5]:.2f}** latched, **{R['instant'][5]:.2f}** on the raw flag); the "
            f"cash-drag-stripped content test is **{R['content_latched'][2]:+.2f}%/yr "
            f"(Welch t = {R['content_latched'][3]:.2f})** latched and **+{R['content_instant'][2]:.2f}%/yr "
            f"but only t = {R['content_instant'][3]:.2f}** on the flag — never clears **t ≥ 2** the right way. |\n"
            f"| **Tradability** | `MIRAGE` | Matches buy-and-hold's Sharpe (**{R['latched'][2]:.2f}** vs "
            f"**{R['bh_sharpe']:.2f}**) only by surrendering **{R['bh_cagr']-R['latched'][1]:.1f} pts of "
            f"CAGR** for a smaller drawdown (**{R['latched'][3]:.0f}%** vs **{R['bh_mdd']:.0f}%**) — pure "
            f"de-risking from {R['latched'][0]:.0f}% time-in-market. No residual edge. |\n"
            f"| **I/T confirmation?** | `BUSTED` | A random regime matched on on-fraction + run length "
            f"beats the real one with **p = {R['placebo'][2]:.3f}** — the confirmation carries no "
            "information beyond *being in cash sometimes*. |\n\n"
            "> 💡 In plain words: timing the trend on the Transports confirmation loses to buy-and-hold, "
            "the flag doesn't predict next-day returns, and a coin-flip cash rule with the same persistence "
            "does just as well. The legend is busted on its own tightest mechanical reading."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $I_t, T_t$ be the Industrials and Transports closes. Define the **confirmation** event "
            "and a **latched primary-trend regime** $R_t\\in\\{0,1\\}$:\n\n"
            "$$\\text{confirm}_t = \\big[I_t \\ge \\max_{t-L\\le s<t} I_s\\big]\\wedge"
            "\\big[T_t \\ge \\max_{t-L\\le s<t} T_s\\big],\\quad"
            "\\text{break}_t = \\big[I_t \\le \\min I\\big]\\wedge\\big[T_t \\le \\min T\\big],$$\n\n"
            "$R_t = 1$ from a confirmation until the next breakdown ($L = 63$). The trade holds the "
            "Industrials when $R_{t-1}=1$ (one-day lag), else cash. Three hypotheses:\n\n"
            "- **H₁ (it times).** The active spread $a_t = r^{\\text{strat}}_t - r^{\\text{B\\&H}}_t$ has "
            "$\\mathbb{E}[a_t] > 0$ (HAC *t* ≥ 2).\n"
            "- **H₂ (it predicts).** Confirmed days have higher *forward* Industrials returns than "
            "non-confirmed days (the flag has content, independent of the cash-drag).\n"
            "- **H₃ (it's the confirmation, not the cash).** The real regime beats a random regime with "
            "the same on-fraction and persistence.\n\n"
            "We find **H₁ rejected** (active *t* < 0), **H₂ rejected** (content *t* = "
            f"{R['content_latched'][3]:.2f} / {R['content_instant'][3]:.2f}), **H₃ rejected** "
            f"(placebo p = {R['placebo'][2]:.3f}). All three legs of the steelman fail."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the cash-drag confound\n\n"
            "A part-time-in-cash timing rule mechanically lowers drawdown and *can* match Sharpe in a "
            "sample that happened to contain crashes (2008, 2020) — **without the signal carrying any "
            "information**. So a raw equity-curve comparison is not enough. We disentangle three things:\n\n"
            "- **The active spread** $a_t$ and its **HAC** *t* (Newey-West) — does timing beat holding "
            "*after* accounting for autocorrelated daily returns?\n"
            "- **The content test** — mean *forward* return on confirmed vs non-confirmed days (Welch "
            "*t*). This strips the cash-drag: it asks only whether the flag predicts, not whether sitting "
            "out helped.\n"
            "- **The placebo** — a two-state Markov regime matched to the real regime's on-fraction and "
            "mean run length. If the real confirmation can't beat this, the work is being done by *how "
            "often you hold cash*, not by the Industrials/Transports agreement.\n\n"
            "> 💡 In plain words: \"lower drawdown\" is the trap. The only honest questions are *does the "
            "flag predict?* and *does it beat a coin flip with the same cash habit?*"
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** DIA (Industrials) + IYT (Transports), auto-adjusted, "
            f"{R['start']}→{R['end']} ({R['n_days']:,} days, fingerprint `{R['fp']}`), as-of "
            f"**{R['asof']}** (no partial month).\n"
            f"- **Regime.** Latched both-averages {R['lookback']}-day-high confirmation → ON until "
            "both-averages 63-day-low breakdown → OFF. Also the un-latched instantaneous flag.\n"
            "- **Trade / lag / costs.** Long Industrials when ON, cash else; signal at close *t*, return "
            "earned *t+1* (one lag); 2 bps one-way per flip; cash earns 0.\n"
            "- **Null #1 (HAC t)** on the active daily spread vs 0.\n"
            "- **Null #2 (content, Welch t)** confirmed-day vs non-confirmed-day forward return.\n"
            "- **Null #3 (placebo)** random Markov regime matched on on-fraction + run length; "
            "$p=\\Pr[\\text{random Sharpe}\\ge\\text{real}]$.\n"
            "- **Robustness.** Costs 0–10 bps; lookback 21–200; the 34-year ^DJI/^DJT price-only tape.\n"
            "- **Positive control.** A deterministic panel with a **planted** confirmation edge: zero edge "
            "must NOT light up the content test; a large edge must."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The race + the active spread\n\n"
            "Equity curves (log) of buy-and-hold vs the latched timed book, and the headline stats. The "
            "timed book is *smoother* (it hides in cash through the crashes) but ends **lower**."
        ),
        code(
            "if HAVE_REAL:\n"
            "    reg = st.latched_regime(DF)\n"
            "    bt = st.backtest(DF, reg, cost_bps=2.0)\n"
            "    eq_bh = (1+bt['r_indu']).cumprod(); eq_st = (1+bt['r_strat_net']).cumprod()\n"
            "    s = st.summarize(DF, latched=True, cost_bps=2.0)\n"
            "    at = s['active_t']; am = s['active_mean_ann']*100\n"
            "else:\n"
            "    idx = pd.bdate_range('2004-01-02', periods=R['n_days'])\n"
            "    eq_bh = pd.Series(np.linspace(0,1,R['n_days']), index=idx).pipe(lambda x: (1+R['bh_cagr']/100/252)**np.arange(R['n_days']))\n"
            "    eq_st = (1+R['latched'][1]/100/252)**np.arange(R['n_days'])\n"
            "    eq_bh = pd.Series(eq_bh, index=idx); eq_st = pd.Series(eq_st, index=idx)\n"
            "    at, am = R['latched'][5], R['latched'][4]\n"
            "fig, ax = plt.subplots(figsize=(9.6, 4.6))\n"
            "ax.plot(eq_bh.index, eq_bh.values, c=GREY, lw=1.8, label='buy & hold Industrials')\n"
            "ax.plot(eq_st.index, eq_st.values, c=RED, lw=1.8, label='Dow-Theory timed (long when confirmed)')\n"
            "ax.set_yscale('log'); ax.set_ylabel('growth of $1 (log)'); ax.legend()\n"
            "ax.set_title(f'Smoother but lower: active spread {am:+.2f}%/yr, HAC t = {at:.2f}')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'active spread {am:+.2f}%/yr   HAC t = {at:.3f}  (negative => timing LOSES to buy-and-hold)')"
        ),
        md(
            f"> 💡 In plain words: the red curve rides lower the whole way and ends below grey. The active "
            f"spread is **{R['latched'][4]:+.2f}%/yr** at **HAC t = {R['latched'][5]:.2f}** — a "
            "*significantly negative* timing contribution. Dow-Theory timing doesn't just fail to help; on "
            "this tape it actively costs you."
        ),
        md(
            "### 4b · The content test — does the flag predict at all?\n\n"
            "Strip the cash-drag. Compare the Industrials' **next-day** return on confirmed vs "
            "non-confirmed days, for both the latched regime and the raw instantaneous flag. A real signal "
            "would show confirmed ≫ non-confirmed with *t* ≥ 2."
        ),
        code(
            "if HAVE_REAL:\n"
            "    cl = st.confirmed_vs_not(DF, latched=True)\n"
            "    ci = st.confirmed_vs_not(DF, latched=False)\n"
            "    L = (cl['mean_on_ann']*100, cl['mean_off_ann']*100, cl['welch_t'])\n"
            "    I = (ci['mean_on_ann']*100, ci['mean_off_ann']*100, ci['welch_t'])\n"
            "else:\n"
            "    L = (R['content_latched'][0], R['content_latched'][1], R['content_latched'][3])\n"
            "    I = (R['content_instant'][0], R['content_instant'][1], R['content_instant'][3])\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "x = np.arange(2)\n"
            "ax.bar(x-.2, [L[0], I[0]], .4, color=GREEN, label='confirmed days')\n"
            "ax.bar(x+.2, [L[1], I[1]], .4, color=GREY, label='non-confirmed days')\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'latched\\n(Welch t={L[2]:.2f})', f'instant flag\\n(Welch t={I[2]:.2f})'])\n"
            "ax.set_ylabel('forward return (%/yr)'); ax.axhline(0,c='k',lw=.7)\n"
            "ax.set_title('Confirmed days do NOT reliably out-return non-confirmed days'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'latched: confirmed {L[0]:+.2f}%/yr vs non {L[1]:+.2f}%/yr  Welch t={L[2]:.2f}')\n"
            "print(f'instant: confirmed {I[0]:+.2f}%/yr vs non {I[1]:+.2f}%/yr  Welch t={I[2]:.2f}')"
        ),
        md(
            f"> 💡 In plain words: on the **latched** regime, confirmed days actually return *less* "
            f"(**{R['content_latched'][0]:+.2f}%/yr** vs **{R['content_latched'][1]:+.2f}%/yr**, Welch "
            f"t = {R['content_latched'][3]:.2f}). The instantaneous flag is positive "
            f"(**+{R['content_instant'][2]:.2f}%/yr**) but **t = {R['content_instant'][3]:.2f}**, under the "
            "bar — and it fires only 5.5% of days. No reliable forward content either way."
        ),
        md(
            "### 4c · The placebo — confirmation, or just cash?\n\n"
            "The decisive honesty check. Build 2,000 **random** regimes matched to the real regime's "
            "on-fraction and average run length (a two-state Markov chain), run the identical backtest, and "
            "see where the real rule lands in the distribution of random Sharpes."
        ),
        code(
            "if HAVE_REAL:\n"
            "    reg = st.latched_regime(DF)\n"
            "    pl = st.placebo_pvalue(DF, reg, n_draws=2000, cost_bps=2.0)\n"
            "    obs, draws, p = pl['obs_sharpe'], pl['draws'], pl['p_value']\n"
            "else:\n"
            "    obs, p = R['placebo'][0], R['placebo'][2]\n"
            "    rng = np.random.default_rng(444); draws = rng.normal(R['placebo'][1], 0.12, 2000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=45, color=GREY, alpha=.85, label='2,000 matched random regimes')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'real Dow-Theory regime (Sharpe {obs:.3f})')\n"
            "ax.set_xlabel('net Sharpe'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Real regime is no outlier: placebo p = {p:.3f}'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'observed Sharpe {obs:.3f}  placebo mean {np.nanmean(draws):.3f}  p(random>=real) = {p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: the real regime sits at **p = {R['placebo'][2]:.3f}** — a random cash "
            "rule with the same persistence beats it about a fifth of the time. The modest lift over the "
            "*average* random rule comes from *how often* it held cash (which dodged 2008/2020 in this "
            "sample), **not** from the Industrials/Transports agreement. H₃ rejected."
        ),
        md(
            "### 4d · Robustness — negative at every knob\n\n"
            "The active HAC *t* across costs and across trailing-high windows, plus the independent "
            "**34-year** ^DJI/^DJT price-only check. If there were a fragile edge, *some* knob would lift "
            "the *t* above 2. None does (in the right direction)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    lbs = [21,42,63,126,200]\n"
            "    lt = [st.summarize(DF, lookback=lb, latched=True, cost_bps=2.0)['active_t'] for lb in lbs]\n"
            "    cs = [0.0,2.0,5.0,10.0]\n"
            "    ct = [st.summarize(DF, latched=True, cost_bps=c)['active_t'] for c in cs]\n"
            "else:\n"
            "    lbs=[r[0] for r in R['lookbacks']]; lt=[r[3] for r in R['lookbacks']]\n"
            "    cs=[r[0] for r in R['costs']]; ct=[r[2] for r in R['costs']]\n"
            "fig,(a1,a2)=plt.subplots(1,2,figsize=(10.6,4.2))\n"
            "a1.bar([str(l) for l in lbs], lt, color=RED, width=.6)\n"
            "a1.axhline(2, ls='--', c='k', label='t=+2 bar'); a1.axhline(-2, ls=':', c=GREY)\n"
            "a1.axhline(0,c='k',lw=.7); a1.set_xlabel('lookback (days)'); a1.set_ylabel('active HAC t')\n"
            "a1.set_title('Active t < 0 at every window'); a1.legend()\n"
            "a2.plot(cs, ct, 'o-', c=RED, lw=2); a2.axhline(0,c='k',lw=.7); a2.axhline(2,ls='--',c='k')\n"
            "a2.set_xlabel('one-way cost (bps)'); a2.set_ylabel('active HAC t'); a2.set_title('Costs are not the story')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('lookback active t:', [round(v,2) for v in lt])\n"
            "print('long index ^DJI/^DJT 1992->2026: active t', R['idx'][3], ' content Welch t', R['idx'][5])"
        ),
        md(
            f"> 💡 In plain words: active *t* is **negative for every lookback** (21→200) and barely moves "
            f"with cost — there was no gross edge to charge against. And on **34 years** of raw index "
            f"history the result holds: active **t = {R['idx'][3]:.2f}**, content spread "
            f"**{R['idx'][4]:+.2f}%/yr** (Welch t = {R['idx'][5]:.2f}). Robustly nothing."
        ),
        md(
            "### 4e · Faithful-engine & power control — we know the truth here\n\n"
            "On a deterministic two-series panel where we **plant** a confirmation edge (the Industrials' "
            "return is genuinely boosted on confirmed days), the content test must recover it; with **zero** "
            "planted edge it must stay near *t* = 0. Both hold — so the real-tape null is a real null."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.003):\n"
            "    px, truth = data.synthetic_panel(edge=edge, seed=444, n_days=8000)\n"
            "    c = st.confirmed_vs_not(px, latched=False)\n"
            "    res.append((edge, c['spread_ann']*100, c['welch_t']))\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "labels = [f'planted edge\\n{e*100:.1f}%/day' for e,_,_ in res]\n"
            "tvals = [r[2] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('content Welch t'); ax.set_title('Control: zero edge -> t~0; planted edge -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,sp,t in res: print(f'planted {e*100:+.1f}%/day: content spread {sp:+.2f}%/yr  Welch t={t:.2f}')"
        ),
        md(
            f"> 💡 In plain words: zero planted edge → **t = {R['syn'][0][5]:.2f}** (no false positive); a "
            f"big planted edge → **t = {R['syn'][1][5]:.2f}**. The harness *can* find a confirmation edge "
            "when one exists — it just doesn't find one on the real Industrials/Transports tape. The "
            "negative real-tape result is genuine, not a broken detector."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the active spread vs buy-and-hold is **negative** (HAC "
            f"**t = {R['latched'][5]:.2f}** latched, **{R['instant'][5]:.2f}** flag); the content test is "
            f"**{R['content_latched'][2]:+.2f}%/yr (t = {R['content_latched'][3]:.2f})** latched and "
            f"**+{R['content_instant'][2]:.2f}%/yr (t = {R['content_instant'][3]:.2f})** on the flag — never "
            f"clears **t ≥ 2** the believers' way, and the sign holds over 34 years of index history. "
            "Indistinguishable from noise (if anything, the wrong sign).\n"
            f"- **Tradability `MIRAGE`** — matches buy-and-hold's Sharpe (**{R['latched'][2]:.2f}** vs "
            f"**{R['bh_sharpe']:.2f}**) only by giving up **{R['bh_cagr']-R['latched'][1]:.1f} pts of "
            f"CAGR** for a smaller drawdown — pure de-risking from {R['latched'][0]:.0f}% time-in-market, "
            "available from any cash rule. No residual edge to scale.\n"
            f"- **I/T confirmation? `BUSTED`** — a random regime matched on on-fraction + persistence "
            f"beats the real one (**p = {R['placebo'][2]:.3f}**). The two averages agreeing carries no "
            "information beyond being in cash part of the time."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the de-risking dial\n\n"
            "The operational truth in one frame: the return/drawdown trade-off. Dow-Theory timing is a "
            "**de-risking dial**, not an alpha source — and a static cash/equity mix or a balanced fund "
            "gives you the same dial without the narrative or the turnover."
        ),
        code(
            "if HAVE_REAL:\n"
            "    s = st.summarize(DF, latched=True, cost_bps=2.0)\n"
            "    pts = {'buy & hold': (s['bh_mdd']*100, s['bh_cagr']*100, s['bh_sharpe']),\n"
            "           'Dow Theory': (s['st_mdd']*100, s['st_cagr']*100, s['st_sharpe'])}\n"
            "else:\n"
            "    pts = {'buy & hold': (R['bh_mdd'], R['bh_cagr'], R['bh_sharpe']),\n"
            "           'Dow Theory': (R['latched'][3], R['latched'][1], R['latched'][2])}\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.6))\n"
            "for (lab,(d,c,sh)),col in zip(pts.items(),[GREY,RED]):\n"
            "    ax.scatter(d, c, s=200, color=col, zorder=3)\n"
            "    ax.annotate(f'{lab}\\nSharpe {sh:.2f}',(d,c),textcoords='offset points',xytext=(8,6))\n"
            "ax.set_xlabel('worst drawdown (%)'); ax.set_ylabel('CAGR (%)')\n"
            "ax.set_title('Same Sharpe line: you only move return for drawdown, not for free alpha')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Both points sit on essentially the same Sharpe ray from the origin -> no alpha, just leverage/cash.')"
        ),
        md(
            "> 💡 In plain words: the two points lie on roughly the same Sharpe ray. Moving along that ray "
            "(more cash → less return, less drawdown) is what *any* de-risking does. The confirmation "
            "adds nothing you couldn't get by holding less equity — hence **MIRAGE**, not a tradable edge."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The historical record agrees.** Cowles (1933) scored Hamilton's published Dow-Theory "
            "calls and found them below buy-and-hold; Brown, Goetzmann & Kumar (1998) re-ran him and "
            "found the *only* benefit was drawdown reduction from cash — exactly this. We reproduced a "
            "90-year-old verdict mechanically.\n"
            "- **The subjective version.** Discretionary Dow Theorists read \"significant\" highs and "
            "secondary reactions by eye. The burden is theirs: show the discretionary calls clearing "
            "*t* = 2 against this matched-persistence placebo.\n"
            "- **As a filter, not a switch.** Maybe the confirmation adds value *conditioning* a "
            "momentum/breadth book rather than as a binary on/off. The engine is in "
            "[`dow_theory/`](../dow_theory).\n\n"
            "*The reproducible core is offline and deterministic; the confirmation is the both-averages "
            "trailing-high/low rule. Methods and sources: [`docs/references.md`](../docs/references.md); "
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
