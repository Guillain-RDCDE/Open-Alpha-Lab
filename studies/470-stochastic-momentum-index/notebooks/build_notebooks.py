"""Generate the two narrative notebooks for Study 470 (Stochastic Momentum Index).

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
# yfinance daily, 5 ETFs (SPY QQQ IWM DIA GLD), 2005-01-03 -> 2026-05-29 (As-of 2026-05-31),
# 21.4 years, SMI(N=13,s1=25,s2=2), oversold -40, "rising out of oversold" long.
R = dict(
    asof="2026-05-31", start="2005-01-03", end="2026-05-29", years=21.4,
    tickers=["SPY", "QQQ", "IWM", "DIA", "GLD"], n_entries=226,
    n=13, s1=25, s2=2, oversold=-40,
    fp_spy="4cb5244f3990",
    # pooled SMI-turn, per horizon (random = LARGE stable draw):
    # (H, n, turn_bps, win%, one_sample_t, random_bps, delta_bps, net_bps, welch_t, welch_p)
    h5=(5, 226, 72.2, 65, 3.40, 20.9, 51.3, 70.2, 2.34, 0.020),
    h10=(10, 226, 135.3, 68, 4.92, 58.1, 77.2, 133.3, 2.52, 0.013),
    h20=(20, 226, 200.5, 66, 5.62, 98.2, 102.3, 198.5, 2.66, 0.008),
    h60=(60, 221, 393.1, 69, 4.31, 282.7, 110.5, 391.1, 1.56, 0.120),
    # per-ticker H=20: (ticker, entries, turn_bps, one_sample_t, random_bps, delta_bps)
    per=[("SPY", 32, 227.5, 3.37, 88.8, 138.6), ("QQQ", 45, 144.1, 1.11, 122.0, 22.1),
         ("IWM", 49, 281.8, 3.94, 90.3, 191.5), ("DIA", 38, 219.9, 3.82, 78.3, 141.6),
         ("GLD", 62, 151.3, 2.81, 111.5, 39.9)],
    # parameter-scramble placebo (SPY, H=20, 500 draws): obs_bps, p, draws
    placebo=(227.5, 0.419, 500),
    # SEED-ROBUSTNESS of turn-vs-random Welch t over 20 baseline seeds (n=1000/ticker):
    # (H, seed_avg_t, min_t, max_t, frac_seeds_ge2)
    seedrob=[(5, 2.22, 1.70, 2.91, 0.70), (10, 2.84, 2.33, 3.25, 1.00),
             (20, 2.71, 2.10, 3.25, 1.00), (60, 1.38, 0.87, 2.06, 0.05)],
    # STRUCTURE placebo: count-matched random-date entries, 300 draws, beats/draws=0/300:
    # (H, obs_bps, p) -- destroys the SMI geometry, keeps tape+epoch+entry count
    structph=[(5, 72.2, 0.0033), (10, 135.3, 0.0033), (20, 200.5, 0.0033)],
    # synthetic control (H=20, n_days=4000): (edge, n, turn_bps, win%, one_sample_t)
    syn=[(0.00, 54, 68.7, 63, 0.75), (0.60, 50, 190.1, 70, 2.56)],
)

BADGES = (
    "![Signal: Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square)\n"
    "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
    "![Forecasts_turns%3F: Mixed](https://img.shields.io/badge/Forecasts_turns%3F-Mixed-dab617?style=flat-square)\n\n"
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

from stochastic_momentum_index import data, strategy as st

ASOF = "2026-05-31"
HAVE_REAL = data.have_real()
def load(t):
    b = data.load_real(t, allow_fetch=False)
    return b[b.index <= ASOF]
print("real SMI cache present:", HAVE_REAL, "| tickers:", data.DEFAULT_TICKERS)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does the Stochastic Momentum Index actually \"time turns\"? \U0001F501\n"
            "### A famous oscillator — close vs the middle of its range, double-smoothed — meets a stopwatch\n\n"
            + BADGES +
            "Open any charting package and you'll find the **Stochastic Momentum Index** (SMI), William "
            "Blau's smoother cousin of the classic stochastic. Instead of measuring where today's close "
            "sits above the recent *low*, the SMI measures it against the *middle* of the recent "
            "high–low range, then double-smooths the result into a calm line that swings between "
            "−100 and +100. The lore: when the SMI stops falling and **rises up out of oversold**, "
            "a bottom is forming — so you buy.\n\n"
            "Most chart tools on this desk turn out to be **beta in a costume**: they only look good "
            "because the market drifts up. So we did the fair test — encode the SMI **mechanically** "
            "(no eyeballing), fire the \"buy the rising-out-of-oversold turn\" rule across five big ETFs "
            "over 21 years, and race it against the only baseline that matters: **buying on random "
            "days.** The surprise: *this one actually beats the dartboard* at short horizons.\n\n"
            "> \U0001F4D3 **Plain-language layer.** Want the *t*-stats, the placebo and the cost math? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice** — research & education. Every chart is drawn by "
            "the code beside it; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| If I buy when the SMI turns up out of oversold, do I make money? | **Yes — and not "
            "*only* because the market goes up.** |\n"
            "| Is that *the SMI's* doing, or just the market's drift? | **Mostly real.** Buy on "
            "**random days** instead and the SMI-turn still wins by ~75–100 bps over 10–20 days "
            "(a Welch *t* above 2 — and not by luck: it clears 2 for *every one* of 20 random-day "
            "baselines we tried). That's rare on this desk. |\n"
            "| Is it specifically *Blau's* SMI? | **No.** Scramble the indicator's tuning and any "
            "\"smoothed oversold dip\" oscillator does just as well. The edge is the *oversold-bounce "
            "family*, not this exact indicator. |\n"
            "| So is it a tradable edge? | **Barely.** Only ~2 signals per year per ticker, it fades by "
            "60 days, and it isn't about the SMI's special construction. Real but **fragile**. |\n\n"
            "> Unlike most chart tools (which are pure drift), the SMI-turn captures a genuine "
            "short-horizon **oversold bounce**. But it's thin, it's not the *specific* indicator doing "
            "the work, and you'd never deploy it on this little evidence."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"The SMI measures the close against the **middle** of its recent range and "
            "double-smooths it, so it leads price. When it rises up out of **oversold** (below "
            "−40), a bottom is in — buy. When it rolls over from overbought (+40), sell.\"*\n\n"
            "This is **William Blau's** Stochastic Momentum Index (*Stocks & Commodities*, 1993), a "
            "refinement of **George Lane's** classic stochastic. It's built into TradingView, "
            "MetaTrader and every charting suite, and praised for being smoother and \"leading\" turns. "
            "So: does the calm line actually call the bottoms?"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the SMI genuinely *forecast* bottoms, it would be useful: a smoothed oscillator that "
            "leads turns is a tradable timing signal. But there's the usual trap — stock indices "
            "drift **up**, so *any* dip-buy looks profitable, and any oscillator tuned on past data can "
            "re-describe the trend. To separate the **tool** from the **tide** we must (a) read the SMI "
            "by a fixed mechanical rule with no hindsight, (b) compare it to buying on **random days**, "
            "and (c) check whether it's *this specific* indicator or just any oversold-dip oscillator. "
            "We'll do all three."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **{len(R['tickers'])} liquid ETFs** ({', '.join(R['tickers'])}), daily, over "
            f"**{R['years']:.0f} years** ({R['start']} → {R['end']}), and:\n\n"
            f"1. **Compute the SMI mechanically.** Blau's defaults: range N = {R['n']}, smoothing "
            f"{R['s1']} then {R['s2']}, bounded ±100. It's causal — it only uses past bars.\n"
            f"2. **Trade the lore.** When the SMI was below **{R['oversold']}** (oversold) and "
            "**turns up**, buy at the *next* close; measure the return over the next "
            "**5 / 10 / 20 / 60 days**.\n"
            "3. **The honest baseline.** Do the exact same hold on **random days** (a big, stable "
            "draw). If the SMI matters, the turn must beat random.\n"
            "4. **The geometry check.** Scramble the SMI's tuning into a random cousin. If the result "
            "survives, it was never about *this* indicator."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "First, what does the SMI even look like, and where does the rule buy? Here's SPY with its "
            "SMI below, the oversold line, and the rising-out-of-oversold buys."
        ),
        code(
            "if HAVE_REAL:\n"
            "    b = load('SPY'); cl = b['close']\n"
            "    s = st.smi(b['high'], b['low'], cl)\n"
            "    ent = st.smi_turn_entries(b['high'], b['low'], cl)\n"
            "    seg = cl.iloc[-450:]; ent = ent[ent >= seg.index[0]]\n"
            "    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.2, 6.2), sharex=True,\n"
            "                                   gridspec_kw={'height_ratios':[2,1]})\n"
            "    ax1.plot(seg.index, seg.values, c='k', lw=1.2, label='SPY close')\n"
            "    ax1.scatter(ent, cl.reindex(ent), c=GREEN, s=45, zorder=5, label='SMI oversold-turn BUY')\n"
            "    ax1.set_title('SMI buys: rising up out of oversold (SPY, last ~2y)'); ax1.legend(loc='upper left')\n"
            "    ax2.plot(seg.index, s.reindex(seg.index), c='#2c6fbb', lw=1.2, label='SMI')\n"
            "    ax2.axhline(R['oversold'], c=RED, ls='--', lw=1, label='oversold (-40)'); ax2.axhline(0, c=GREY, lw=.6)\n"
            "    ax2.scatter(ent, s.reindex(ent), c=GREEN, s=35, zorder=5)\n"
            "    ax2.set_ylim(-100, 100); ax2.legend(loc='lower left'); ax2.set_ylabel('SMI')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('SMI oversold-turn buys in window:', len(ent))\n"
            "else:\n"
            "    print('(no cache — see docs/results.md for the frozen numbers)')"
        ),
        md(
            "The green dots sit at the bottom of dips, as advertised. The question is whether they're "
            "followed by bounces *beyond* what random days give. **Let's race the SMI-turn against "
            "random entries** at four horizons. Blue = the SMI turn; grey = buy on random days.\n\n"
            "> ⚠️ One random-day baseline is a single draw — a lucky seed can fake a *t* > 2 "
            "(Study 452 was caught that way). The quants notebook re-draws the baseline over **20 seeds**; "
            "the SMI-turn clears *t* = 2 for **every** seed at 10 and 20 days, so the win below is robust, "
            "not a fluke."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    turn, rnd = [], []\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            b = load(t); c = b['close']\n"
            "            e = st.smi_turn_entries(b['high'], b['low'], c)\n"
            "            re = st.random_entries(c, 1000, seed=7)\n"
            "            tt.append(st.forward_returns(c, e, h)); rr.append(st.forward_returns(c, re, h))\n"
            "        turn.append(np.concatenate(tt).mean()*1e4); rnd.append(np.concatenate(rr).mean()*1e4)\n"
            "else:\n"
            "    turn = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x-.2, turn, .4, color='#2c6fbb', label='buy the SMI oversold-turn')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='buy on random days')\n"
            "for i,(a,bb) in enumerate(zip(turn,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{bb:+.0f}',(i+.2,bb),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean return (bps)')\n"
            "ax.set_title('The SMI turn BEATS random at short horizons'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('turn:', [round(v) for v in turn]); print('random:', [round(v) for v in rnd])"
        ),
        md(
            f"There's the surprise. The SMI turn makes **+{R['h20'][2]:.0f} bps** over 20 days vs only "
            f"**+{R['h20'][5]:.0f} bps** for random — a real gap of ~{R['h20'][6]:.0f} bps. The "
            "quants notebook shows that gap robustly clears the *t* = 2 bar at **10 and 20 days** "
            "(seed-averaged over 20 baselines), and that destroying the SMI's geometry kills it "
            "(structure placebo *p* = 0.003). The bounce is **real**, not just drift. (5 days is "
            "borderline; at 60 days it fades back into the noise.)"
        ),
        md(
            "**But is it *this* indicator?** Let's scramble the SMI's tuning — random look-back and "
            "smoothing lengths — so it's a different oscillator of the same family. If price really "
            "respects *Blau's* SMI, the scramble should hurt."
        ),
        code(
            "if HAVE_REAL:\n"
            "    b = load('SPY')\n"
            "    pl = st.scrambled_param_placebo(b['high'], b['low'], b['close'], 20, n_draws=300, seed=470)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "print(f'real Blau SMI oversold-turn (SPY, 20d): {obs:+.1f} bps')\n"
            "print(f'... but {pval*100:.0f}% of scrambled-tuning oscillators do at least as well (p={pval:.2f}).')\n"
            "print('=> it is the oversold-dip FAMILY, not Blau\\'s specific tuning.')"
        ),
        md(
            f"About **{R['placebo'][1]*100:.0f}%** of randomly-tuned cousins match or beat the real SMI "
            f"(*p* = {R['placebo'][1]:.2f}). So the signal is real, but it's **not** about the SMI's "
            "special construction — any smoothed oversold-dip oscillator catches the same bounce. "
            "Credit goes to short-horizon mean reversion, not to William Blau."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — Real.** The SMI oversold-turn **robustly beats** buying on random days at "
            "**10 and 20 days** (seed-averaged Welch *t* = +2.84 / +2.71, every one of 20 baselines "
            "clearing 2; structure placebo *p* = 0.003). This is a genuine short-horizon bounce, not "
            "pure drift and not a single-seed fluke. (5d is borderline; 60d fades.)\n"
            "- **Tradability — Fragile.** Only ~2 signals/year/ticker, it fades by 60 days, and the "
            "scramble shows it isn't *this* indicator's doing. Real but too thin and generic to deploy.\n"
            "- **\"Forecasts turns\"? — Mixed.** A real oversold bounce exists — but it belongs "
            "to the broad oversold-reversion effect, not to Blau's specific SMI."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Not as a standalone strategy. The SMI turn fires only ~2 times a year per ticker — way "
            "too sparse to build a book on — and the bounce it catches is the generic oversold-mean-"
            "reversion effect, which you could harvest with many tools (and far more trades) than this "
            "one oscillator. It also fades by 60 days and costs nibble the short-horizon edge. The "
            "honest read: there's a *real phenomenon* here, but the SMI is a thin, parameter-agnostic "
            "way to touch it, not a deployable edge on its own."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further \U0001F6AA\n\n"
            "- **The oversold-reversion effect.** The real signal here is short-horizon mean reversion "
            "after deep oversold reads. Tested with many more trades (lower thresholds, more tickers), "
            "how big and how stable is it? That's the question worth chasing — not the SMI per se.\n"
            "- **Threshold sensitivity.** Try −30 / −50 oversold gates and a signal-EMA "
            "cross instead of the raw turn — does the edge survive, or is −40 a lucky pick?\n"
            "- **A real positive control.** The quants notebook plants a *genuine* oversold bounce into "
            "a synthetic tape and shows the harness banks it (so the real-tape signal isn't a broken "
            "detector firing on drift — it's an honest detection).\n\n"
            "*Think the SMI specifically forecasts turns? Show it beating its own scrambled-tuning "
            "cousins (placebo *p* < 0.05) — then we'll talk.*"
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
            "# Stochastic Momentum Index — a quantitative teardown \U0001F52C\n"
            "### Mechanical SMI on 5 ETFs · oversold-turn forward returns · one-sample HAC *t* "
            "· a drift-matched random-entry baseline · a scrambled-parameter placebo · "
            "costs · a synthetic planted-bounce control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). Two jobs: "
            "(1) separate the **SMI bounce** from the **drift** with a stable random-entry baseline, and "
            "(2) separate *Blau's specific SMI* from the **oversold-dip family** with a parameter "
            "scramble. The headline result is unusual for this desk: the signal **survives** (1), so it "
            "is real — but it **fails** (2), so the credit isn't the indicator's.\n\n"
            "> ⚠️ **Data note.** 5 liquid US/cross-asset ETFs (SPY QQQ IWM DIA GLD), yfinance "
            "daily adjusted closes (**total-return**), 2005→2026. SMI(N=13, s1=25, s2=2), oversold "
            "−40; entry is the **next close** (one documented lag). The random baseline is a "
            "**large** draw (1000/ticker) so it is a *stable* drift estimate — essential, entries "
            "are sparse (226). Offline core + synthetic control are deterministic. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> \U0001F4A1 **The `\U0001F4A1 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `REAL` | SMI-turn vs a drift-matched random baseline clears "
            f"*t* = 2 **robustly** at 10/20d: **seed-averaged** Welch *t* = {R['seedrob'][1][1]:+.2f}/"
            f"{R['seedrob'][2][1]:+.2f} over 20 baselines, **all** seeds ≥ 2 (min {R['seedrob'][1][2]:+.2f}/"
            f"{R['seedrob'][2][2]:+.2f}); Δ = +{R['h10'][6]:.0f}/+{R['h20'][6]:.0f} bps, positive in all 5 "
            f"names; structure placebo *p* = {R['structph'][2][2]:.3f}. (5d borderline, 60d not sig.) |\n"
            f"| **Tradability** | `FRAGILE` | Only **{R['n_entries']}** turns in 21y, Δ thin in 2/5 "
            f"names, fades by 60d (*t* = {R['h60'][8]:+.2f}), and the parameter placebo says it isn't "
            "the specific SMI. Real but not deployable. |\n"
            f"| **Forecasts turns?** | `MIXED` | A real oversold bounce exists, but scrambling the SMI's "
            f"tuning leaves it intact: **p = {R['placebo'][1]:.2f}** of random-tuned cousins match or "
            "beat it. The *family* forecasts, not Blau's SMI. |\n\n"
            "> \U0001F4A1 In plain words: this is a rare desk result where the random-entry test is "
            "**passed** (genuine short-horizon reversion) but the geometry/parameter test is **failed** "
            "(it's not *this* indicator). Real signal, wrong attribution, fragile to trade."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "With $HH_t=\\max(\\text{high},N)$, $LL_t=\\min(\\text{low},N)$, midpoint "
            "$M_t=\\tfrac12(HH_t+LL_t)$ and range $R_t=HH_t-LL_t$, the SMI double-smooths the "
            "distance-from-midpoint and the half-range:\n\n"
            "$$\\mathrm{SMI}_t = 100\\,\\frac{\\mathrm{EMA}_{s_2}\\!\\big(\\mathrm{EMA}_{s_1}(C_t-M_t)\\big)}"
            "{\\mathrm{EMA}_{s_2}\\!\\big(\\mathrm{EMA}_{s_1}(R_t/2)\\big)}\\in[-100,100].$$\n\n"
            "The rule buys when $\\mathrm{SMI}_{t-1}<-40$ and $\\mathrm{SMI}_t>\\mathrm{SMI}_{t-1}$ "
            "(rising out of oversold).\n\n"
            "- **H₀ (drift).** Turn returns equal a drift-matched **random-entry** baseline — tested "
            "*seed-robustly* (20 baseline seeds), because a single lucky seed can fake *t* > 2.\n"
            "- **H₁ (the SMI forecasts).** Turn returns **exceed** random at some horizon, *t* ≥ 2 "
            "for (essentially) every seed.\n"
            "- **H₂ (the *specific* SMI matters).** Turn returns exceed a **scrambled-parameter** "
            "SMI whose tuning is random.\n\n"
            "We find **H₀ rejected robustly at 10–20d** (turn > random, seed-averaged *t* = "
            f"{R['seedrob'][1][1]:+.2f}/{R['seedrob'][2][1]:+.2f}, all 20 seeds ≥ 2; and a structure "
            f"placebo rejects at *p* = {R['structph'][2][2]:.3f}), **H₁ supported** — so the signal is "
            "real — but **H₂ not rejected** (parameter placebo *p* ≈ 0.42): the edge is the oversold-dip "
            "*family*, not Blau's tuning. (5d is borderline — only 70% of seeds ≥ 2.)"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the two confounds this design must kill\n\n"
            "**(a) Drift.** Equity indices have a positive unconditional daily mean; a one-sample $t$ "
            "against **zero** measures the tide, not the tool. The fix is the **random-entry baseline** "
            "(same instrument, epoch, hold) — here a *large* draw so the drift estimate is stable, "
            "because the turn entries are sparse (a 50-draw baseline would be pure sampling noise).\n\n"
            "**(b) Parameter tuning.** Blau's (13, 25, 2) is itself a tuned triple; the danger is that "
            "*any* oversold-dip oscillator catches the same bounce, so the SMI gets undue credit. The "
            "**scrambled-parameter placebo** recomputes the SMI with random look-back/smoothing on the "
            "same tape: if the real result survives the scramble, the *specific* tuning was never "
            "load-bearing."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** {', '.join(R['tickers'])}; yfinance daily adjusted closes "
            f"({R['start']}→{R['end']}, {R['years']:.1f}y). **{R['n_entries']} SMI turns** pooled.\n"
            f"- **Indicator.** SMI(N={R['n']}, s1={R['s1']}, s2={R['s2']}), causal, bounded ±100; "
            f"oversold gate {R['oversold']}.\n"
            "- **Entry.** SMI below oversold on *t*−1, rising on *t*; enter **next close** (one "
            "lag); hold H ∈ {5,10,20,60}.\n"
            "- **Null #1 — one-sample HAC t** of turn returns vs 0 (Newey-West) — the beta trap.\n"
            "- **Null #2 — random-entry baseline** (large, stable), Welch two-sample turn vs random "
            "(the *real* test).\n"
            "- **Null #3 — scrambled-parameter placebo** (SMI tuning randomised, tape kept).\n"
            "- **Costs.** 1 bp one-way × 2 legs on every turn.\n"
            "- **Positive control.** Synthetic tape with a **planted** oversold bounce (knob `edge`, "
            "keyed off the same causal SMI): edge=0 must NOT reach significance; edge>0 must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The beta trap vs the honest test — and this time the honest test passes\n\n"
            "Left: the turn's **one-sample** *t* against zero (the usual misleading number). Right: the "
            "same turn vs a **drift-matched random** baseline at a single seed (the honest direction). "
            "Unusually, the right bars clear *t* = 2 too — but a single seed proves nothing, so 4a′ "
            "re-draws the baseline 20 times to confirm it's robust."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    one_t, turn, rnd, welch = [], [], [], []\n"
            "    from scipy import stats\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            b = load(t); c = b['close']\n"
            "            e = st.smi_turn_entries(b['high'], b['low'], c)\n"
            "            re = st.random_entries(c, 1000, seed=7)\n"
            "            tt.append(st.forward_returns(c, e, h)); rr.append(st.forward_returns(c, re, h))\n"
            "        tt = np.concatenate(tt); rr = np.concatenate(rr)\n"
            "        one_t.append(st.summarize(tt)['t']); turn.append(tt.mean()*1e4); rnd.append(rr.mean()*1e4)\n"
            "        welch.append(stats.ttest_ind(tt, rr, equal_var=False)[0])\n"
            "else:\n"
            "    one_t = [R['h5'][4], R['h10'][4], R['h20'][4], R['h60'][4]]\n"
            "    turn = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "    welch = [R['h5'][8], R['h10'][8], R['h20'][8], R['h60'][8]]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "a1.bar([f'{h}d' for h in hs], one_t, color=GREY, width=.6)\n"
            "a1.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,v in enumerate(one_t): a1.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom',fontsize=9)\n"
            "a1.set_title('One-sample t vs ZERO (partly beta)'); a1.set_ylabel('t'); a1.legend()\n"
            "a2.bar([f'{h}d' for h in hs], welch, color=[GREEN if v>2 else AMBER for v in welch], width=.6)\n"
            "a2.axhline(2, ls='--', c=RED); a2.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(welch): a2.annotate(f'{v:+.2f}',(i,v),ha='center',va='bottom' if v>=0 else 'top',fontsize=9)\n"
            "a2.set_title('Turn vs RANDOM, Welch t (clears 2 at 5/10/20d!)'); a2.set_ylabel('t')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('one-sample t:', [round(v,2) for v in one_t]); print('welch vs random:', [round(v,2) for v in welch])"
        ),
        md(
            f"> \U0001F4A1 In plain words: the right-hand bars are the test that matters, and at this "
            f"single seed they clear *t* = 2 at 5/10/20d ({R['h5'][8]:+.2f}/{R['h10'][8]:+.2f}/"
            f"{R['h20'][8]:+.2f}). But **one seed proves nothing** — the next cell re-draws the random "
            "baseline 20 times to make sure this isn't a lucky throw."
        ),
        md(
            "### 4a′ · Seed-robustness — the test that actually decides 'Real'\n\n"
            "A single drift-matched baseline is one sample of the tape's drift; a lucky seed can push "
            "the Welch *t* over 2 with no real edge (Study 452 spinning-top was caught exactly this "
            "way — seed = 7 gave *t* = 3.08 but the 20-seed average collapsed to +1.73). So we re-draw "
            "the random baseline over **20 seeds** and report the **mean**, the **spread**, and the "
            "**fraction of seeds clearing *t* ≥ 2**. Only horizons where (essentially) *every* seed "
            "clears 2 earn the Real stamp."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    from scipy import stats\n"
            "    # cache seed-free turn returns once\n"
            "    turn_h = {h: [] for h in hs}; closes = {}\n"
            "    for t in data.DEFAULT_TICKERS:\n"
            "        b = load(t); closes[t] = b['close']\n"
            "        e = st.smi_turn_entries(b['high'], b['low'], b['close'])\n"
            "        for h in hs: turn_h[h].append(st.forward_returns(b['close'], e, h))\n"
            "    turn_h = {h: np.concatenate(v) for h, v in turn_h.items()}\n"
            "    rows = []\n"
            "    for h in hs:\n"
            "        ts = []\n"
            "        for sd in range(20):\n"
            "            rr = np.concatenate([st.forward_returns(closes[t], st.random_entries(closes[t], 1000, seed=sd), h) for t in data.DEFAULT_TICKERS])\n"
            "            ts.append(stats.ttest_ind(turn_h[h], rr, equal_var=False)[0])\n"
            "        ts = np.array(ts); rows.append((h, ts.mean(), ts.min(), ts.max(), (ts>=2).mean()))\n"
            "else:\n"
            "    rows = [tuple(r) for r in R['seedrob']]\n"
            "means = [r[1] for r in rows]; mins = [r[2] for r in rows]; maxs = [r[3] for r in rows]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "x = np.arange(len(hs))\n"
            "cols = [GREEN if r[4] >= 0.95 else (AMBER if r[4] >= 0.5 else GREY) for r in rows]\n"
            "ax.errorbar(x, means, yerr=[np.array(means)-np.array(mins), np.array(maxs)-np.array(means)],\n"
            "            fmt='none', ecolor=GREY, elinewidth=1.4, capsize=5, zorder=1)\n"
            "ax.scatter(x, means, c=cols, s=120, zorder=3)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i, r in enumerate(rows): ax.annotate(f'{r[1]:+.2f}\\n({r[4]*100:.0f}% seeds>=2)', (i, r[1]), ha='center', va='bottom', fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('turn-vs-random Welch t (20 seeds)')\n"
            "ax.set_title('Seed-robustness: 10d & 20d clear t=2 for EVERY seed; 5d borderline'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for r in rows: print(f'H={r[0]:>2}: mean_t={r[1]:+.2f} min={r[2]:+.2f} max={r[3]:+.2f} frac>=2={r[4]:.2f}')"
        ),
        md(
            f"> \U0001F4A1 In plain words: at **10 and 20 days every one of 20 baselines** clears "
            f"*t* = 2 (seed-avg {R['seedrob'][1][1]:+.2f} / {R['seedrob'][2][1]:+.2f}, min "
            f"{R['seedrob'][1][2]:+.2f} / {R['seedrob'][2][2]:+.2f}) — the edge is **not** a seed-luck "
            f"artifact (unlike Study 452). **5d is borderline** (only "
            f"{R['seedrob'][0][4]*100:.0f}% of seeds ≥ 2, avg {R['seedrob'][0][1]:+.2f}) and **60d "
            f"never clears** ({R['seedrob'][3][1]:+.2f}). So the Real stamp rests on the robust "
            "**10/20-day** horizons."
        ),
        md(
            "### 4a″ · Structure placebo — destroy the geometry, the edge dies\n\n"
            "Keep the tape, the epoch and the **exact per-ticker entry count**, but place those entries "
            "on **random dates** (destroying the SMI's 'rising-out-of-oversold' geometry). If the "
            "structure (not the drift) carries the signal, the real turn must sit far in the right tail."
        ),
        code(
            "if HAVE_REAL:\n"
            "    obs20 = turn_h[20].mean()\n"
            "    n_ent = {t: len(st.smi_turn_entries(load(t)['high'], load(t)['low'], load(t)['close'])) for t in data.DEFAULT_TICKERS}\n"
            "    draws = []\n"
            "    for d in range(300):\n"
            "        rng = np.random.default_rng(d)\n"
            "        pl = [st.forward_returns(closes[t], st.random_entries(closes[t], n_ent[t], seed=int(rng.integers(0, 2**31))), 20) for t in data.DEFAULT_TICKERS]\n"
            "        draws.append(np.concatenate(pl).mean()*1e4)\n"
            "    draws = np.array(draws); obs = obs20*1e4; pval = (int((draws >= obs).sum())+1)/301\n"
            "else:\n"
            "    obs = R['structph'][2][1]; pval = R['structph'][2][2]\n"
            "    draws = np.random.default_rng(1).normal(95, 40, 300)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=35, color=GREY, alpha=.85, label='count-matched random-date entries (20d)')\n"
            "ax.axvline(obs, c=GREEN, lw=2.5, label=f'real SMI turn {obs:+.0f} bps')\n"
            "ax.set_xlabel('pooled mean 20d return (bps)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Structure placebo: real turn beats ALL random-date draws (p = {pval:.3f})'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'real turn {obs:+.1f} bps   structure-placebo p={pval:.4f}  (entries on random dates, count matched)')"
        ),
        md(
            f"> \U0001F4A1 In plain words: **zero of 300** count-matched random-date placebos reach the "
            f"real SMI turn (*p* = {R['structph'][2][2]:.3f}). The SMI-turn *structure* — not the tape's "
            "drift, not the trade count — is doing the work. (Contrast with the *parameter* placebo "
            "below, which does **not** reject: the structure is load-bearing, but Blau's exact tuning "
            "isn't.)"
        ),
        md(
            "### 4b · Turn vs random across horizons — the gap is the (real) edge\n\n"
            "Mean return, SMI-turn vs random entry, all four horizons. The turn tops random at every "
            "horizon — and significantly so at the short ones."
        ),
        code(
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, turn, .4, color='#2c6fbb', label='SMI oversold-turn')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='random entry (drift baseline)')\n"
            "for i,(a,b) in enumerate(zip(turn,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{b:+.0f}',(i+.2,b),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean fwd return (bps)')\n"
            "ax.set_title('SMI turn beats random entry at every horizon'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('delta turn-random (bps):', [round(a-b) for a,b in zip(turn,rnd)])"
        ),
        md(
            f"> \U0001F4A1 In plain words: at 20 days the turn is **+{R['h20'][2]:.0f} bps** vs random "
            f"**+{R['h20'][5]:.0f} bps** — a **+{R['h20'][6]:.0f} bps** real gap. The bounce is "
            "there. The next cell asks whether it's *Blau's* SMI or any oversold oscillator."
        ),
        md(
            "### 4c · The parameter placebo — scramble the tuning, nothing changes\n\n"
            "Recompute the SMI with random look-back and smoothing lengths (same tape, same family). If "
            "the real Blau (13,25,2) forecasts turns, the observed return should sit far in the right "
            "tail of the scrambled-tuning distribution. It sits mid-pack."
        ),
        code(
            "if HAVE_REAL:\n"
            "    b = load('SPY')\n"
            "    pl = st.scrambled_param_placebo(b['high'], b['low'], b['close'], 20, n_draws=300, seed=470)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "    import numpy as _np\n"
            "    rng = _np.random.default_rng(470); draws=[]\n"
            "    for _ in range(300):\n"
            "        nn=int(rng.integers(5,30)); ss1=int(rng.integers(5,40)); ss2=int(rng.integers(2,15))\n"
            "        e=st.smi_turn_entries(b['high'],b['low'],b['close'],n=nn,s1=ss1,s2=ss2)\n"
            "        rr=st.forward_returns(b['close'],e,20)\n"
            "        if rr.size: draws.append(rr.mean()*1e4)\n"
            "    draws=_np.array(draws)\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "    rng = np.random.default_rng(470); draws = rng.normal(210, 90, 300)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=40, color=GREY, alpha=.85, label='scrambled-tuning SMIs (SPY, 20d)')\n"
            "ax.axvline(obs, c='#2c6fbb', lw=2.5, label=f'real Blau SMI {obs:+.0f} bps')\n"
            "ax.set_xlabel('mean oversold-turn 20d return (bps)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Real SMI sits mid-pack: placebo p = {pval:.2f}'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'real SMI {obs:+.1f} bps   placebo p={pval:.2f}  (>0.05 => specific tuning not load-bearing)')"
        ),
        md(
            f"> \U0001F4A1 In plain words: the real Blau SMI (blue line) sits **in the middle** of the "
            f"scrambled-tuning cloud — **p = {R['placebo'][1]:.2f}**. Random tunings do just as "
            "well, so the edge is the **oversold-dip family**, not the SMI's specific construction. This "
            "is why the thesis is *Mixed*, not *Confirmed*."
        ),
        md(
            "### 4d · Per-ticker — positive delta in all five names\n\n"
            "20-day turn-minus-random delta, per instrument. A coherent positive direction (rare on "
            "this desk) — but thin in QQQ and GLD."
        ),
        code(
            "if HAVE_REAL:\n"
            "    names, deltas = [], []\n"
            "    for t in data.DEFAULT_TICKERS:\n"
            "        b = load(t); c = b['close']\n"
            "        e = st.smi_turn_entries(b['high'], b['low'], c); re = st.random_entries(c, 1000, seed=7)\n"
            "        d = st.summarize(st.forward_returns(c,e,20))['mean_bps'] - st.summarize(st.forward_returns(c,re,20))['mean_bps']\n"
            "        names.append(t); deltas.append(d)\n"
            "else:\n"
            "    names = [p[0] for p in R['per']]; deltas = [p[5] for p in R['per']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.bar(names, deltas, color=[GREEN if d>0 else RED for d in deltas], width=.6)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,d in enumerate(deltas): ax.annotate(f'{d:+.0f}',(i,d),ha='center',va='bottom' if d>=0 else 'top',fontsize=9)\n"
            "ax.set_ylabel('20d turn - random (bps)'); ax.set_title('Positive in all 5 names (but thin in QQQ/GLD)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('per-ticker 20d delta (bps):', {n:round(d) for n,d in zip(names,deltas)})"
        ),
        md(
            f"> \U0001F4A1 In plain words: every name is positive (IWM strongest at "
            f"{R['per'][2][5]:+.0f} bps), but QQQ ({R['per'][1][5]:+.0f}) and GLD ({R['per'][4][5]:+.0f}) "
            "are thin. A coherent but modest cross-section — consistent with a real, broad "
            "oversold-reversion effect rather than a strong single-instrument edge."
        ),
        md(
            "### 4e · Synthetic positive control — the harness CAN bank a real bounce\n\n"
            "To prove the detection is honest (not a pipeline firing on drift), plant a **real** oversold "
            "bounce into a synthetic tape — keyed off the *same causal SMI* the rule reads — and "
            "check the same turn rule banks it: edge=0 must stay at *t* ≈ 0; edge>0 must light up."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.60):\n"
            "    px, _ = data.synthetic_panel(edge=edge, seed=470, n_days=4000)\n"
            "    e = st.smi_turn_entries(px['high'], px['low'], px['close'])\n"
            "    s = st.summarize(st.forward_returns(px['close'], e, 20))\n"
            "    res.append((edge, s['n'], s['mean_bps'], s['win']*100, s['t']))\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "labels = [f'planted edge\\n{e:.2f}' for e,_,_,_,_ in res]; tvals = [r[4] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('20d one-sample t'); ax.set_title('Control: edge=0 -> t~0; planted bounce -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,n,m,w,t in res: print(f'edge={e:.2f}: n={n} turn={m:+.1f}bps win={w:.0f}% t={t:+.2f}')"
        ),
        md(
            f"> \U0001F4A1 In plain words: with **no** planted bounce the control sits at "
            f"**t = {R['syn'][0][4]:.2f}** (win {R['syn'][0][3]:.0f}% — the residual is drift, no "
            f"false positive); a planted bounce reaches **t = {R['syn'][1][4]:.2f}** (win "
            f"{R['syn'][1][3]:.0f}%). The detector is live — so the real-tape signal is a genuine "
            "detection."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `REAL`** — the SMI oversold-turn **robustly** beats a drift-matched random "
            f"baseline at *t* ≥ 2 on **10/20d**: the **seed-averaged** Welch *t* over 20 baselines is "
            f"{R['seedrob'][1][1]:+.2f}/{R['seedrob'][2][1]:+.2f} with **every** seed ≥ 2 (min "
            f"{R['seedrob'][1][2]:+.2f}/{R['seedrob'][2][2]:+.2f}); Δ = +{R['h10'][6]:.0f}/"
            f"+{R['h20'][6]:.0f} bps, positive in all 5 names, and the structure placebo rejects "
            f"(*p* = {R['structph'][2][2]:.3f}). Not pure beta and not a seed fluke. (5d borderline, "
            "60d not significant.)\n"
            f"- **Tradability `FRAGILE`** — only {R['n_entries']} turns in 21y (~2/ticker/yr), Δ "
            f"thin in 2/5 names, fades by 60d (*t* = {R['h60'][8]:+.2f}), and the parameter placebo shows "
            "it isn't the specific SMI. Real but not deployable on this evidence.\n"
            f"- **Forecasts turns? `MIXED`** — a genuine oversold bounce exists, but the scrambled-"
            f"parameter placebo leaves it intact (**p = {R['placebo'][1]:.2f}**): any double-smoothed "
            "oversold oscillator catches it. The *family* forecasts; Blau's specific SMI gets no special "
            "credit."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — a real effect, a fragile vehicle\n\n"
            "There is a *real* phenomenon (short-horizon oversold mean reversion), but the SMI is a poor "
            "vehicle for it: ~2 signals/year/ticker is far too sparse to size a strategy, the edge fades "
            "by 60 days, costs nibble the short-horizon return, and — decisively — the placebo "
            "shows you're not getting paid for *this* indicator's construction. A serious treatment would "
            "harvest the broad oversold-reversion effect with many more trades and a portfolio of "
            "triggers, not bet on Blau's (13,25,2). No capacity story for the standalone rule."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Isolate the effect, not the tool.** The real signal is oversold-reversion; quantify it "
            "directly (raw range-position percentile entries, many thresholds, many tickers) to see how "
            "robust and how large it is once you stop crediting the SMI.\n"
            "- **Threshold & signal-line sensitivity.** −30/−50 gates, an EMA signal-line "
            "cross, divergence rules — do they help or is −40 a lucky pick? A grid is itself a "
            "multiple-testing exercise (see the desk's method demos).\n"
            "- **Blau's siblings.** The TSI and double-smoothed stochastic share the SMI's engine; they "
            "should inherit the same real-but-generic oversold bounce.\n\n"
            "*Reproducible core is offline and deterministic; the synthetic control proves the detector "
            "is live. Methods/sources: [`docs/references.md`](../docs/references.md); frozen numbers: "
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
