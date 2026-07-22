"""Generate the two narrative notebooks for Study 800 (High-Frequency / Weekly Reversal).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached weekly panel
under ../_cache/ (built once from the shared daily S&P 500 panel) and otherwise quote the
frozen headline numbers in ``R`` (mirroring docs/results.md). The synthetic positive
control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (weekly total-return S&P
# 500 panel, 2010-01-08 -> 2026-05-29; 856 weeks x 500 tickers; fingerprint e9d0f3b08c1c).
R = dict(
    start="2010-01-08", end="2026-05-29", asof="2026-05-29", fp="e9d0f3b08c1c",
    n_weeks=856, n_tickers=500, names_per_week=476, per_quintile=95, cs_med_bps=44,
    # headline (skip=0)
    spread_bps=17.5, spread_t=2.47, spread_lags=6, spread_ann=9.1, hit=53.4,
    hit_k=456, hit_n=854, wilson=(50.0, 56.7),
    loser_ann=24.4, loser_t=4.99, winner_ann=15.3, winner_t=3.73,
    loser_turn=78, winner_turn=78,
    # skip=1 killer
    skip1_bps=-1.7, skip1_t=-0.28,
    # beta / null
    beta=0.293, alpha_ann=3.83, random_excess_bps=0.11, random_draws=51240,
    # flat cost sweep (borrow 50bps/yr on short)
    cost_sweep={0: (16.6, 2.34), 5: (0.9, 0.13), 10: (-14.8, -2.10), 20: (-46.1, -6.61)},
    breakeven_bps=5.6,
    # empirical CS bounce haircut
    haircut_half_bps=-28.8, haircut_half_t=-4.29,
    haircut_full_bps=-74.2, haircut_full_t=-11.46,
    # decay sub-periods: (bps, t, n)
    subperiods={"2010-2015": (25.9, 4.00, 310), "2016-2020": (23.4, 1.22, 261),
                "2021-2026": (2.9, 0.26, 283)},
    # synthetic control
    syn_null_t0=-0.64, syn_null_t1=-0.26, syn_null_fire=0, syn_null_seeds=12,
    syn_rev_t0=16.22, syn_rev_t1=20.99, syn_bnc_t0=10.71, syn_bnc_t1=-1.35,
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Bid-ask bounce%3F: Confirmed](https://img.shields.io/badge/Bid--ask%20bounce%3F-Confirmed-8b949e?style=flat-square)\n\n"
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

from hf_reversal import data, strategy as st

try:
    CLOSE, SPREAD = data.load_real()
    HAVE_REAL = True
except Exception as e:
    CLOSE = SPREAD = None
    HAVE_REAL = False
    print("real panel unavailable (offline synthetic-only run):", e)
print("real panel present:", HAVE_REAL,
      "| weeks:", (0 if CLOSE is None else CLOSE.shape[0]),
      "| tickers:", (0 if CLOSE is None else CLOSE.shape[1]))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Do last week's losers bounce back next week? ⚡\n"
            "### Weekly reversal — a real-looking pattern that turns out to be the bid-ask "
            "spread wearing a costume\n\n"
            + BADGES +
            "Here's a tidy little money machine, at least on paper: every Friday, buy the "
            "stocks that fell the most this week and short the ones that rose the most, then "
            "unwind a week later. The losers, the story goes, 'over-fell' and snap back; the "
            "winners 'over-rose' and give some back. It's the faster, punchier cousin of the "
            "month-long reversal — more trades, more signal, right?\n\n"
            "On the raw numbers it looks *real*. Then we make it wait one week — and the whole "
            "thing vanishes.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the skip test and the cost math? "
            "See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** S&P 500 weekly total-return closes, 2010→2026, ~476 names a "
            "week. Current-membership universe (**survivorship-biased** — the delisted losers "
            "are missing, so the raw edge is already an *over*statement). House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Do last week's losers beat last week's winners next week? | **On the raw tape, "
            f"yes.** The loser-minus-winner spread is **+{R['spread_bps']:.0f} bps/week** "
            f"(≈ +{R['spread_ann']:.0f}%/yr), and it's not just market beta. It clears the "
            "statistical bar. |\n"
            f"| Is it a *real* edge, or a measurement artifact? | **Artifact.** Make the trade "
            f"wait just **one week** between spotting the loser and buying it, and the spread "
            f"collapses to **{R['skip1_bps']:+.0f} bps** — statistically zero. The 'bounce' was "
            "the bid-ask spread, not the stock. |\n"
            f"| Does it still work lately? | **No.** Even the raw version has faded to nothing "
            f"since ~2021 (**+{R['subperiods']['2021-2026'][0]:.0f} bps, t = "
            f"{R['subperiods']['2021-2026'][1]:.2f}**). |\n"
            f"| Could you trade it anyway? | **No.** You'd flip ~{R['loser_turn']}% of the book "
            f"every week, and the losers you'd buy are the *illiquid* names with the widest "
            f"spreads — the cost of trading them is bigger than the edge. Break-even is "
            f"~{R['breakeven_bps']:.0f} bps, which nobody hits on names like these. |\n\n"
            "> The pattern is real in the data. It is not real in your brokerage account."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"A stock that just had a bad week over-shot on the way down — sellers "
            "panicked, liquidity dried up — so it tends to snap back next week. Buy the week's "
            "losers, short the week's winners.\"*\n\n"
            "It's short-horizon **mean-reversion**, documented since Lehmann (1990) and "
            "Jegadeesh (1990). The weekly version is sold as *better* than the monthly one "
            "(our [study 329](../../329-one-month-reversal/)) because it turns faster: more "
            "reversals, more chances to profit. This notebook asks whether the extra signal is "
            "real — or just more of the same illusion, faster."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · The trap: bid-ask bounce\n\n"
            "Here's the catch that haunts *every* short-horizon reversal. Stock prices don't "
            "trade at one number — there's a **bid** (where you sell) a hair below the **ask** "
            "(where you buy). The closing print randomly lands on one or the other. A stock "
            "whose Friday close happened to land on the *bid* looks like it fell a little extra "
            "— so it's tagged a 'loser' — and next week's close is likely to land back near the "
            "*ask*, looking like a 'bounce.' **No one made a dime; the price never really "
            "moved.** The narrower and more illiquid the stock, the bigger this fake bounce.\n\n"
            "So we need a way to tell a *real* rebound from this bid-ask ghost. There's a clean "
            "one: **make the trade wait a week.**"
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know\n\n"
            "- **The spread.** Each Friday, rank ~476 S&P 500 names by this week's return; the "
            "bottom fifth are 'losers,' the top fifth 'winners.' Buy losers, short winners, "
            "measure next week's gap.\n"
            "- **The ghost-buster (the skip test).** Do the exact same thing, but leave a "
            "**one-week gap** between the ranking Friday and the day you actually buy. If the "
            "bounce was real mean-reversion, it's still there a week later. If it was bid-ask "
            "bounce, it's gone — because the fake wiggle only connects *adjacent* closes.\n"
            "- **The trading-cost check.** Charge each stock its *own* estimated bid-ask spread "
            "when you trade it. Losers are illiquid; they cost the most."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's look\n\n"
            "**First, the raw spread, and what one week of waiting does to it.**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    r0 = st.detect_spread(CLOSE, skip=0)\n"
            "    r1 = st.detect_spread(CLOSE, skip=1)\n"
            "    a, b = r0['mean_bps'], r1['mean_bps']\n"
            "else:\n"
            "    a, b = R['spread_bps'], R['skip1_bps']\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.4))\n"
            "ax.bar(['buy right away\\n(skip=0)','wait one week\\n(skip=1)'], [a, b],\n"
            "       color=[GREEN, RED], width=.55)\n"
            "for i,v in enumerate([a, b]): ax.annotate(f'{v:+.1f} bps',(i,v),ha='center',\n"
            "    va='bottom' if v>=0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('loser − winner, next-week spread (bps)')\n"
            "ax.set_title('One week of waiting erases the whole \"bounce\"')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'buy now: {a:+.1f} bps  |  wait a week: {b:+.1f} bps')"
        ),
        md(
            f"There it is. Buy the losers immediately and you 'earn' **+{R['spread_bps']:.0f} "
            f"bps a week**. Wait a single week and it's **{R['skip1_bps']:+.0f} bps** — nothing. "
            "A real rebound wouldn't care whether you bought Friday or the following Friday. "
            "This one cares completely, which is the fingerprint of **bid-ask bounce**, not "
            "mean-reversion.\n\n"
            "**Second: could you even trade the raw version, before worrying about whether it's "
            "real?**"
        ),
        code(
            "cs = R['cost_sweep']\n"
            "xs = sorted(cs); ys = [cs[c][0] for c in xs]\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "cols = [GREEN if y>0 else RED for y in ys]\n"
            "ax.bar([f'{c} bps' for c in xs], ys, color=cols, width=.6)\n"
            "for i,y in enumerate(ys): ax.annotate(f'{y:+.0f}',(i,y),ha='center',\n"
            "    va='bottom' if y>=0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.axvline(0.5+R['breakeven_bps']/5.0-0.0, ls='--', c=GREY, lw=1)\n"
            "ax.set_ylabel('net spread per week (bps)')\n"
            "ax.set_xlabel('trading cost charged each side')\n"
            "ax.set_title(f'Break-even is ~{R[\"breakeven_bps\"]:.0f} bps — and the losers cost far more')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('net weekly spread by cost:', {c: cs[c][0] for c in xs})"
        ),
        md(
            f"The raw edge is gone by a trading cost of just **~{R['breakeven_bps']:.0f} bps** a "
            f"side. That would be fine for the most liquid ETFs — but this book *buys the "
            f"week's biggest losers*, which are exactly the jumpy, illiquid names whose real "
            f"round-trip cost runs many times that. When we charge each stock its own estimated "
            f"spread, even at *half* the estimate the book bleeds **{R['haircut_half_bps']:+.0f} "
            "bps a week**. You are paying the spread to harvest the spread.\n\n"
            "**Third: has it survived the last few years?**"
        ),
        code(
            "sp = R['subperiods']\n"
            "labels = list(sp); vals = [sp[k][0] for k in labels]; ts = [sp[k][1] for k in labels]\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "cols = [GREEN if t>2 else (AMBER if t>1 else RED) for t in ts]\n"
            "ax.bar(labels, vals, color=cols, width=.6)\n"
            "for i,(v,t_) in enumerate(zip(vals,ts)): ax.annotate(f'{v:+.0f} bps\\n(t={t_:.2f})',\n"
            "    (i,v),ha='center',va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('raw weekly spread (bps)')\n"
            "ax.set_title('Even the raw pattern has faded to zero since ~2021')\n"
            "plt.tight_layout(); plt.show()\n"
            "print({k: sp[k] for k in labels})"
        ),
        md(
            f"The whole thing was carried by the early 2010s. Since ~2021 the *raw* spread is "
            f"**+{R['subperiods']['2021-2026'][0]:.0f} bps (t = "
            f"{R['subperiods']['2021-2026'][1]:.2f})** — indistinguishable from zero. The "
            "electronic market-makers and stat-arb desks that competed away the monthly "
            "reversal have reached the weekly clock too."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Weak.** Yes, the raw spread clears the bar (+{R['spread_bps']:.0f} "
            f"bps/wk, and it's not just beta). But a real edge doesn't evaporate the moment you "
            f"wait one week ({R['skip1_bps']:+.0f} bps), and this one does. It's bid-ask bounce, "
            "and it's faded on top of that.\n"
            "- **Tradability — Mirage.** ~78% of the book turns over weekly, into the market's "
            "least liquid names, and the break-even cost is below what those names actually "
            "cost to trade. There is no paycheck here.\n"
            "- **Is it bid-ask bounce? — Confirmed.** The one-week skip removes 100% of it."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **The general lesson.** *Any* eye-catching reversal at a very short horizon "
            "should be met with the same question: does it survive a one-period skip? If not, "
            "you've measured the spread, not the stock.\n"
            "- **Sibling studies:** the [one-month reversal](../../329-one-month-reversal/) "
            "(same idea, 4× slower clock — same autopsy, same result), the "
            "[long-term reversal](../../196-long-term-reversal/) (multi-*year*, a genuinely "
            "different animal), and [industry-relative reversal](../../538-industry-relative-reversal/) "
            "(fade the move *relative to peers*). See [docs/references.md](docs/references.md) "
            "for the exact dedup.\n\n"
            "*Think the weekly reversal is tradable? Show a net, certifiable spread after your "
            "real fill costs on the actual names — losers included — and a one-week skip. Then "
            "we'll talk.*"
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
            "# Weekly cross-sectional reversal — a quantitative teardown 🔬\n"
            "### The loser-minus-winner HAC spread · the skip=0→skip=1 bid-ask-bounce killer · "
            "an empirical Corwin-Schultz bounce haircut · a flat cost sweep with borrow · the "
            "beta decomposition · the McLean-Pontiff decay split · a two-knob synthetic control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "The claim — **last week's losers beat last week's winners next week** — is the "
            "Lehmann (1990) weekly reversal, the fast cousin of the monthly Jegadeesh reversal "
            "([study 329](../../329-one-month-reversal/)). The job here is to measure the raw "
            "spread, then run the microstructure autopsy the horizon demands: the *skip-a-week* "
            "test and an *empirical* bid-ask-bounce haircut.\n\n"
            "> ⚠️ **Data note.** Weekly total-return S&P 500 closes (shared daily panel "
            "resampled to `W-FRI`), 2010-01-08 → 2026-05-29, ~476 names/week, ~95 per quintile. "
            "**Survivorship-biased** (current membership projected backwards; delisted losers "
            "absent) — every positive spread is an upper bound, named on the Signal axis. "
            "Numbers in [`docs/results.md`](../docs/results.md) (fingerprint `" + R["fp"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | raw skip=0 spread **{R['spread_bps']:+.1f} bps/wk**, HAC "
            f"**t = {R['spread_t']:.2f}** (clears the bar, beta {R['beta']:.2f} so not disguised "
            f"beta) — but skip=1 **t = {R['skip1_t']:.2f}** and post-2021 **t = "
            f"{R['subperiods']['2021-2026'][1]:.2f}** ⇒ not robust |\n"
            f"| **Tradability** | `MIRAGE` | ~{R['loser_turn']}% weekly one-way turnover; "
            f"break-even ~{R['breakeven_bps']:.1f} bps; empirical CS haircut "
            f"**{R['haircut_half_bps']:+.1f} bps/wk** at half-spread |\n"
            f"| **Bid-ask bounce?** | `CONFIRMED` | one-week skip removes 100% of the spread "
            f"(t {R['spread_t']:.2f} → {R['skip1_t']:.2f}) |\n\n"
            "> 💡 In plain words: there is a genuinely significant raw spread, and it isn't "
            "beta — but it is bid-ask bounce (dies at skip=1), it has decayed to zero since "
            "2021, and it costs more to trade than it pays."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r_{i,t}$ be name $i$'s weekly total return. Rank the cross-section on "
            "$r_{i,t-1-s}$ (the formation return, $s$ = skip weeks); long the bottom quintile "
            "$L$, short the top quintile $W$; the payoff is the week-$t$ spread "
            "$\\text{sp}_t = \\bar r_{L,t} - \\bar r_{W,t}$.\n\n"
            "- **H₁ (reversal).** $E[\\text{sp}_t] > 0$ at $s=0$ — last week's losers out-return "
            "last week's winners.\n"
            "- **H₂ (microstructure).** The bulk of any $s=0$ spread is **bid-ask bounce**: a "
            "close landing on the bid fakes a low $r_{t-1}$ and a high $r_t$. A one-week skip "
            "($s=1$) breaks the shared close and should kill a bounce-driven spread while "
            "sparing a real one.\n"
            "- **H₃ (capture).** After the loser leg's *own* effective spread and ~78% weekly "
            "turnover, the net spread is positive.\n\n"
            f"We find **H₁ supported on the raw tape** (HAC t = {R['spread_t']:.2f}), "
            f"**H₂ confirmed** (skip=1 t = {R['skip1_t']:.2f} — the spread is bounce), and "
            f"**H₃ rejected** (net {R['haircut_half_bps']:+.1f} bps at half the estimated "
            "spread). The raw signal is real but non-robust and untradeable."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · Inference design\n\n"
            "The book's weekly return series is mildly autocorrelated (overlapping formation "
            "windows, persistent liquidity regimes), so the **primary is a Newey-West HAC "
            "*t*** with automatic Bartlett lag length. The hit-rate carries a **Wilson** "
            "interval; a **random-portfolio null** (loser-leg-size subsets drawn each week) "
            "confirms any loser excess is signal, not concentration; the **beta decomposition** "
            "rules out disguised market exposure; and the decay split (2010-15 / 2016-20 / "
            "2021-26) tests McLean-Pontiff post-publication fade. Zero look-ahead: one shift, "
            "applied once in `trailing_return` — the signal for week $t$ is week $t-1-s$'s "
            "return, known at that close."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md("## 3 · The teardown"),
        md(
            "### 3a · The raw spread and the skip-a-week killer\n\n"
            "The decisive plot of the whole study: the spread at skip=0 (Jegadeesh — same close "
            "forms *and* prices) versus skip=1 (a one-week gap that defuses the bounce)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    res0 = st.quintile_returns(st.trailing_return(CLOSE, skip=0), CLOSE, spreads=SPREAD, q=0.20)\n"
            "    res1 = st.quintile_returns(st.trailing_return(CLOSE, skip=1), CLOSE, spreads=SPREAD, q=0.20)\n"
            "    s0 = st.summarize(res0['spread'].dropna()); s1 = st.summarize(res1['spread'].dropna())\n"
            "    a, ta, b, tb = s0['mean']*1e4, s0['tstat'], s1['mean']*1e4, s1['tstat']\n"
            "    print(f\"skip=0: {a:+.1f} bps/wk  HAC t={ta:+.2f}  (n={s0['n']}, hit {s0['hit_rate']*100:.1f}%)\")\n"
            "    print(f\"skip=1: {b:+.1f} bps/wk  HAC t={tb:+.2f}  (n={s1['n']})\")\n"
            "else:\n"
            "    a, ta, b, tb = R['spread_bps'], R['spread_t'], R['skip1_bps'], R['skip1_t']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.4))\n"
            "bars = ax.bar(['skip=0\\n(t=%.2f)'%ta,'skip=1\\n(t=%.2f)'%tb], [a, b],\n"
            "              color=[GREEN, RED], width=.55)\n"
            "for i,v in enumerate([a, b]): ax.annotate(f'{v:+.1f} bps',(i,v),ha='center',\n"
            "    va='bottom' if v>=0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('loser − winner spread (bps/week)')\n"
            "ax.set_title('The entire spread is bid-ask bounce: it dies at skip=1')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: the raw spread **{R['spread_bps']:+.1f} bps/wk** (HAC t = "
            f"{R['spread_t']:.2f}) clears the desk bar — but inserting one week between forming "
            f"and holding drops it to **{R['skip1_bps']:+.1f} bps** (t = {R['skip1_t']:.2f}). A "
            "reversal that only exists when the same close forms the signal *and* prices the "
            "entry is the textbook bid-ask-bounce illusion (Lo-MacKinlay 1990). H₂ confirmed."
        ),
        md(
            "### 3b · It isn't beta, and it isn't concentration\n\n"
            "Before blaming everything on bounce: is the raw spread just disguised market "
            "exposure, or lucky concentration? Neither."
        ),
        code(
            "if HAVE_REAL:\n"
            "    beta, alpha = st.beta_alpha(res0['spread'], res0['market'])\n"
            "    rp = st.random_portfolio_returns(st.trailing_return(CLOSE, skip=0), CLOSE, n_draws=30, seed=800)\n"
            "    rex = rp.mean()*1e4\n"
            "else:\n"
            "    beta, alpha, rex = R['beta'], R['alpha_ann']/100, R['random_excess_bps']\n"
            "print(f'spread beta vs EW market = {beta:+.3f}   annual alpha = {alpha*100:+.2f}%/yr')\n"
            "print(f'random-portfolio null mean excess = {rex:+.2f} bps (centred at zero)')\n"
            "fig, ax = plt.subplots(figsize=(7.6, 3.4))\n"
            "ax.barh(['spread beta'], [beta], color=GREY, height=.4)\n"
            "ax.axvline(0, c='k', lw=.8); ax.axvline(1, ls='--', c=RED, lw=1)\n"
            "ax.set_xlim(-0.2, 1.1); ax.set_title(f'Dollar-neutral spread is near market-neutral (beta {beta:.2f})')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: the dollar-neutral spread's beta is **{R['beta']:.2f}** "
            f"(alpha {R['alpha_ann']:+.2f}%/yr) — not disguised beta — and a random loser-sized "
            f"quintile earns **{R['random_excess_bps']:+.2f} bps** (zero). So the raw effect is "
            "genuine cross-sectional structure. That is *why* it's graded Weak, not None: the "
            "problem is microstructure and decay, not that the raw number is noise."
        ),
        md(
            "### 3c · The empirical bid-ask-bounce haircut\n\n"
            "The flat cost sweep understates the truth because it charges every name the same "
            "bps. Losers are the illiquid names. Charge each leg its **own** Corwin-Schultz "
            "(2012) effective spread on its turnover — the honest microstructure haircut."
        ),
        code(
            "labels = ['gross','flat 5bps','flat 10bps','CS half','CS full']\n"
            "if HAVE_REAL:\n"
            "    g = st.summarize(res0['spread'].dropna())['mean']*1e4\n"
            "    n5 = st.summarize(st.net_spread(res0,5,50).dropna())['mean']*1e4\n"
            "    n10 = st.summarize(st.net_spread(res0,10,50).dropna())['mean']*1e4\n"
            "    hh = st.summarize(st.bounce_haircut_spread(res0,50,0.5).dropna())['mean']*1e4\n"
            "    hf = st.summarize(st.bounce_haircut_spread(res0,50,1.0).dropna())['mean']*1e4\n"
            "else:\n"
            "    g = R['spread_bps']; n5 = R['cost_sweep'][5][0]; n10 = R['cost_sweep'][10][0]\n"
            "    hh = R['haircut_half_bps']; hf = R['haircut_full_bps']\n"
            "vals = [g, n5, n10, hh, hf]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "cols = [GREY]+[GREEN if v>0 else RED for v in vals[1:]]\n"
            "ax.bar(labels, vals, color=cols, width=.62)\n"
            "for i,v in enumerate(vals): ax.annotate(f'{v:+.0f}',(i,v),ha='center',\n"
            "    va='bottom' if v>=0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('net spread (bps/week)')\n"
            "ax.set_title('Charge the losers their own spread and the book bleeds')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(dict(zip(labels, [round(v,1) for v in vals])))"
        ),
        md(
            f"> 💡 In plain words: break-even flat cost is **~{R['breakeven_bps']:.1f} bps** "
            f"(net t = {R['cost_sweep'][5][1]:.2f} already at 5 bps, {R['cost_sweep'][10][1]:.2f} "
            f"at 10 bps). The *empirical* haircut — each leg paying its own estimated spread — is "
            f"**{R['haircut_half_bps']:+.1f} bps/wk even at half-spread** (t = "
            f"{R['haircut_half_t']:.2f}). You are paying the bid-ask spread to harvest the "
            "bid-ask spread. H₃ rejected; **Tradability = MIRAGE**."
        ),
        md(
            "### 3d · Post-publication decay\n\n"
            "Even the *raw* (bounce-inflated) spread has faded — the McLean-Pontiff (2016) "
            "signature at the weekly horizon."
        ),
        code(
            "sp = R['subperiods']\n"
            "if HAVE_REAL:\n"
            "    sp = {}\n"
            "    for lo,hi in [('2010','2015'),('2016','2020'),('2021','2026')]:\n"
            "        s = st.summarize(res0.loc[lo:hi,'spread'].dropna())\n"
            "        sp[f'{lo}-{hi}'] = (s['mean']*1e4, s['tstat'], s['n'])\n"
            "labels = list(sp); vals = [sp[k][0] for k in labels]; ts = [sp[k][1] for k in labels]\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "cols = [GREEN if t>2 else (AMBER if t>1 else RED) for t in ts]\n"
            "ax.bar(labels, vals, color=cols, width=.6)\n"
            "for i,(v,t_) in enumerate(zip(vals,ts)): ax.annotate(f'{v:+.0f} bps\\n(t={t_:.2f})',\n"
            "    (i,v),ha='center',va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('raw skip=0 spread (bps/wk)')\n"
            "ax.set_title('The full-sample t is carried by the early 2010s')\n"
            "plt.tight_layout(); plt.show()\n"
            "print({k: (round(sp[k][0],1), round(sp[k][1],2), sp[k][2]) for k in labels})"
        ),
        md(
            f"> 💡 In plain words: 2010-2015 was **+{R['subperiods']['2010-2015'][0]:.0f} bps "
            f"(t = {R['subperiods']['2010-2015'][1]:.2f})**; 2021-2026 is "
            f"**+{R['subperiods']['2021-2026'][0]:.0f} bps (t = "
            f"{R['subperiods']['2021-2026'][1]:.2f})** — gone. The same competition that killed "
            "the monthly reversal has reached the weekly clock."
        ),
        md(
            "### 3e · Faithful-engine control — real reversal vs pure bounce\n\n"
            "The two-knob synthetic panel proves the skip test does what we claim: it keeps a "
            "**genuine** reversal and destroys a **pure bid-ask bounce**. The null must stay "
            "quiet across seeds."
        ),
        code(
            "null0 = np.array([st.detect_spread(data.synthetic_panel(reversal=0,bounce=0,seed=800+i)[0])['tstat'] for i in range(8)])\n"
            "rev = data.synthetic_panel(reversal=0.5, bounce=0.0, seed=800)[0]\n"
            "bnc = data.synthetic_panel(reversal=0.0, bounce=0.008, seed=800)[0]\n"
            "rv = (st.detect_spread(rev,0)['tstat'], st.detect_spread(rev,1)['tstat'])\n"
            "bc = (st.detect_spread(bnc,0)['tstat'], st.detect_spread(bnc,1)['tstat'])\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "x = np.arange(3); w=.35\n"
            "ax.bar(x-w/2, [null0.mean(), rv[0], bc[0]], w, color=GREY, label='skip=0')\n"
            "ax.bar(x+w/2, [null0.mean(), rv[1], bc[1]], w, color=GREEN, label='skip=1')\n"
            "ax.axhline(2, ls='--', c=RED, lw=1); ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(0,c='k',lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels(['null','planted REAL\\nreversal','planted PURE\\nbounce'])\n"
            "ax.set_ylabel('spread HAC t'); ax.legend()\n"
            "ax.set_title('Skip keeps a real reversal (green stays high), kills a bounce (green collapses)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'null mean t (skip0) = {null0.mean():+.2f} | real {rv[0]:+.1f}->{rv[1]:+.1f} | bounce {bc[0]:+.1f}->{bc[1]:+.1f}')"
        ),
        md(
            f"> 💡 In plain words: the null averages t = {R['syn_null_t0']:+.2f} and never fires "
            f"({R['syn_null_fire']}/{R['syn_null_seeds']} seeds |t|≥2). A planted **real** "
            f"reversal survives the skip (t {R['syn_rev_t0']:.1f} → {R['syn_rev_t1']:.1f}); a "
            f"planted **pure bounce** dies (t {R['syn_bnc_t0']:.1f} → {R['syn_bnc_t1']:.1f}) — "
            "exactly the signature the real tape shows. The engine is a faithful detector; the "
            "real-tape verdict is about the market, not the method. *(Control only — never "
            "cited in support of the real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 4 · The verdict\n\n"
            f"- **Signal `WEAK`** — raw skip=0 spread {R['spread_bps']:+.1f} bps/wk, HAC t = "
            f"{R['spread_t']:.2f}, near-market-neutral (beta {R['beta']:.2f}, alpha "
            f"{R['alpha_ann']:+.2f}%/yr): a real, measurable spread — but **not robust**. The "
            f"one-week skip removes it entirely (t = {R['skip1_t']:.2f}) and it has decayed to "
            f"zero post-2021 (t = {R['subperiods']['2021-2026'][1]:.2f}). Survivorship-biased "
            "upper bound.\n"
            f"- **Tradability `MIRAGE`** — ~{R['loser_turn']}% weekly one-way turnover into "
            f"illiquid names; break-even ~{R['breakeven_bps']:.1f} bps flat; the empirical CS "
            f"haircut is {R['haircut_half_bps']:+.1f} bps/wk at half-spread. Uninvestable.\n"
            "- **Bid-ask bounce? `CONFIRMED`** — the one-week skip removes 100% of the spread, "
            "and charging each leg its own spread turns it deeply negative."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 5 · Going further\n\n"
            "- **The weekly clock is not cleaner than the monthly one.** More turnover buys "
            "more *bounce*, not more signal — the extra 'reversals' are the bid-ask spread "
            "sampled more often.\n"
            "- **Dedup map:** [329-one-month-reversal](../../329-one-month-reversal/) (the "
            "monthly Jegadeesh spec — same autopsy, skip=1-month kills it, decayed post-2002), "
            "[196-long-term-reversal](../../196-long-term-reversal/) (multi-year De Bondt-Thaler "
            "— different mechanism), [538-industry-relative-reversal](../../538-industry-relative-reversal/) "
            "(fade the industry-relative move).\n\n"
            "*The reproducible core is offline and deterministic; frozen numbers live in "
            "[`docs/results.md`](../docs/results.md), sources in "
            "[`docs/references.md`](../docs/references.md).*"
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
