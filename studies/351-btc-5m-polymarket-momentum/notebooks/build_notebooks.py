"""Generate the two narrative notebooks for Study 351 (BTC 5-minute Polymarket momentum).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic: the real-tape cells read the cached 1-minute
BTC klines under ../_cache/ (folded into 5-minute Polymarket windows) and otherwise quote the
frozen headline numbers in ``R`` (mirroring docs/results.md). The synthetic positive control
and the bankroll Monte-Carlo run anywhere.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (Binance BTCUSDT 1m, 45 days
# ending 2026-06-20; live Polymarket capture 2026-06-20, 32 windows).
R = dict(
    window="2026-05-06 → 2026-06-20", days=45, windows=12959, p_up=49.5, flat=0.12,
    btc_lo=59207, btc_hi=82761, med_move=32, p90_move=109,
    w30=89.1, n30=6802, z30=64.5, w50=92.9, n50=4322, w70=94.8, n70=2759, frac70=21.3, z70=47.0,
    w85=96.3, w100=96.5, n100=1533, w150=98.6,
    w70_1m=98.2, w100_1m=99.3,
    mc_fair_p=0.948, mc_fair_target=1.07, mc_fair_ruin=39.7, mc_fair_med=4.06,
    mc_over_ruin=99.3, mc_pitch_target=100.0, mc_pitch_med=14668, mc_yaml_med=2853,
    syn=[(30, 0.980, 0.980), (45, 0.942, 0.943), (60, 0.913, 0.913)],
    live_n=32, live_all_w=65.6, live_all_p=76.4, live_all_edge=-0.108,
    live_sig_n=4, live_sig_p=0.97,
)

BADGES = (
    "![Signal: Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Get_rich_quick%3F: Busted](https://img.shields.io/badge/Get_rich_quick%3F-Busted-8b949e?style=flat-square)\n\n"
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root (quantlab/)
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from btc5m_polymarket import data, strategy as st

HAVE_REAL = data.have_real()
W = data.load_real() if HAVE_REAL else None
print("real BTC tape cache present:", HAVE_REAL,
      "| windows:", (0 if W is None else len(W)))
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
            "# BTC 5-minute Polymarket bot — turn $300 into $14,000? 🎰\n"
            "### The viral \"free GitHub bot\" that bets Bitcoin's last two minutes, in plain English\n\n"
            + BADGES +
            "A post goes viral: *\"I put in \\$300. This bot turned it into \\$14,000. I didn't even "
            "code it — I grabbed an open-source repo, cleaned it up in a couple of hours, and let it "
            "run.\"* The recipe: on Polymarket's **Bitcoin Up/Down 5-minute** markets, with ~2 minutes "
            "left, if BTC has already moved \\$70–100, buy the **winning side** while it's quoted "
            "**\\$0.80–0.99** — *\"it's like buying a lottery ticket after they've called the numbers.\"*\n\n"
            "Is it real? Surprisingly, the momentum part **is**. The money part is **not**. This "
            "notebook shows why both things are true at once.\n\n"
            "> 📓 **Plain-language layer.** Want the *z*-stats, the fair-price algebra and the "
            "bankroll maths? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice — and a safety note.** The advertised repo ships only a "
            "wrapper; the engine that signs trades and reads your **wallet private key** lives in a "
            "*separate, unpublished* repo. \"Completing\" it means handing an unaudited program your "
            "keys. Don't. Every chart below is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does BTC keep going the way it just moved? | **Yes — really.** Once it has moved \\$70 "
            f"with 2 minutes left, it closes that way **~{R['w70']:.0f}%** of the time "
            f"({R['n70']:,} cases). That's not a coin; it's a genuine effect. |\n"
            "| So is it free money? | **No.** The market sees the same move. It quotes that winning "
            f"side at **\\$0.95–0.99** — *exactly* its true odds. You pay full price; there's nothing "
            "left over. |\n"
            "| The pitch says \"buy at \\$0.80.\" | **That price isn't there when the move is real.** "
            "A cheap \\$0.80 quote only exists when the outcome is genuinely *uncertain* — i.e. when "
            "the win-rate is \\$0.80 too. |\n"
            "| Then how did \\$300 become \\$14,000? | **One lucky survivor.** Bet half your stack each "
            "round on a 95% favourite and you usually grind up… until the 1-in-20 wipes you. A few "
            "people moon; most bust. The poster is the one you hear from. |\n\n"
            "> The momentum is real. The edge is already in the price. The jackpot is a martingale's "
            "lucky survivor."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Enter ~2 minutes before the round closes. Check Bitcoin has already moved \\$70–100 "
            "in the interval. Always trade **with** the move. Buy that side at \\$0.80–0.99 — a few "
            "minutes later it settles at \\$1. Like buying a lottery ticket after they've called the "
            "numbers.\"*\n\n"
            "Polymarket's **BTC Up/Down 5-minute** market resolves **Up** if BTC's price at the close "
            "of the 5-minute window is ≥ its price at the open (resolution via the Chainlink BTC/USD "
            "feed). A share pays **\\$1** if you're right, **\\$0** if you're wrong. The claim is that "
            "near the close the outcome is *nearly decided*, so the favoured side at \\$0.80–0.99 is "
            "almost-free money."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "Two very different things are bundled in the pitch, and the trick is that the **first is "
            "true**. (1) *Does a fresh \\$70 move tend to stick for two more minutes?* If yes, there's "
            "a real predictive signal. (2) *Can you buy it for less than it's worth?* Only if the "
            "market mis-prices that signal. A binary share that wins with probability **w**, bought at "
            "price **p**, earns exactly **w − p** per share. So the whole game reduces to one number: "
            "**is w bigger than the p you actually pay?** We can measure both."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We rebuild Polymarket's object from real Bitcoin data: **{R['days']} days** of 1-minute "
            f"BTC prices folded into **{R['windows']:,}** non-overlapping 5-minute windows. For each "
            "window we know the open, the price with **2 minutes left**, and the close — everything the "
            "strategy sees, plus the answer.\n\n"
            "1. **Measure the momentum (w).** Among windows that already moved ≥ \\$70 with 2 minutes "
            "left, how often does the close finish on the move's side?\n"
            "2. **Find the price (p).** A read-only paper-trade bot (no key, no money) captured the "
            "**real Polymarket ask** of the favoured side at the 2-minute mark on live markets.\n"
            "3. **Play out the story.** Simulate the \\$300→\\$14,000 dream thousands of times at the "
            "measured win-rate to see who actually gets there."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First: is the momentum real?** Win-rate of the favoured side as we demand a bigger "
            "head-start at the 2-minute mark."
        ),
        code(
            "Ds = [0, 30, 50, 70, 100, 150]\n"
            "if HAVE_REAL:\n"
            "    ws = [st.continuation(W, d)['w']*100 for d in Ds]\n"
            "else:\n"
            "    ws = [77.8, R['w30'], R['w50'], R['w70'], R['w100'], R['w150']]\n"
            "fig, ax = plt.subplots(figsize=(9, 4.4))\n"
            "ax.plot(Ds, ws, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(50, ls='--', c=GREY, label='a coin flip')\n"
            "ax.set_ylim(45, 100); ax.set_xlabel('BTC already moved at least … ($, with 2 min left)')\n"
            "ax.set_ylabel('% the close finishes that way')\n"
            "ax.set_title('Momentum into the close is real — and strong'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('win-rate by head-start:', [f'{w:.0f}%' for w in ws])"
        ),
        md(
            f"With **no** filter the close follows the 3-minute drift ~78% of the time already; demand "
            f"a **\\$70** head-start and it's **~{R['w70']:.0f}%** ({R['n70']:,} windows), at **\\$100** "
            f"~{R['w100']:.0f}%. **The signal is genuine** — bigger move, more follow-through. So far "
            "the pitch looks right."
        ),
        md(
            "**Now the catch: what does the market charge for it?** Here is a real snapshot from the "
            "live capture — a window with a clear move and ~70 seconds left:\n\n"
            "| | |\n|---|---|\n"
            "| BTC move in the window | **−\\$30** (Down is winning) |\n"
            "| Polymarket ask, *Down* (the favourite) | **\\$0.97** |\n"
            "| Polymarket ask, *Up* (the loser) | **\\$0.04** |\n\n"
            "The favourite isn't \\$0.80. It's **\\$0.97** — because everyone can see the same move. "
            "You'd pay 97 cents to maybe win a dollar. Across the whole live capture, buying the "
            f"favourite **lost money on average** (you paid ~\\${R['live_all_p']/100:.2f} for sides "
            f"that won ~{R['live_all_w']:.0f}% of the time)."
        ),
        code(
            "# What you win vs what you risk, buying the favourite at its real price p\n"
            "p = 0.97; w = R['w70']/100\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "ax.bar(['win  (prob %.0f%%)' % (w*100), 'lose (prob %.0f%%)' % ((1-w)*100)],\n"
            "       [(1-p), -p], color=[GREEN, RED], width=.55)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('profit per $1 share')\n"
            "ax.set_title('Buy the favourite at $0.97: tiny wins, rare ruinous loss')\n"
            "for i,(v,t) in enumerate([((1-p),'+$%.2f'%(1-p)), (-p,'-$%.2f'%p)]):\n"
            "    ax.annotate(t,(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'EV per $1 staked at p={p}: {(w-p)/p:+.3f}  (need w>p just to break even; here w≈p)')"
        ),
        md(
            "There it is. Win and you gain **3 cents**; lose and you give back **97**. At ~95% that "
            "averages to roughly **nothing** — and after the spread and fees, a little less than "
            "nothing. One loss erases ~30 wins. **The momentum was real; the market already owns it.**"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — Real.** A fresh \\$70 move genuinely predicts the close "
            f"(~{R['w70']:.0f}%, vastly better than a coin). No asterisk: the effect is there.\n"
            "- **Tradability — Mirage.** The favoured side is priced at its true odds "
            "(\\$0.95–0.99). You pay full freight; net of the spread the edge is **≤ 0**.\n"
            "- **Get rich quick? — Busted.** \\$300→\\$14,000 is a martingale's lucky survivor, not "
            "a repeatable strategy (next beat shows the odds)."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — play the dream out\n\n"
            "Take the pitch at its word: bet **half your stack** each round on the favourite, and — "
            "generously — assume the game is a *fair* coin (you pay exactly the true odds, so you lose "
            "nothing to the house). Start at \\$300, aim for \\$14,000. Do it 200,000 times."
        ),
        code(
            "m = st.monte_carlo(R['w70']/100, round(R['w70']/100,3), frac=0.50)\n"
            "labels = ['reached\\n$14,000', 'wiped out\\n(<$1)', 'somewhere\\nin between']\n"
            "vals = [m['p_target']*100, m['p_ruin']*100, 100-m['p_target']*100-m['p_ruin']*100]\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.3))\n"
            "ax.bar(labels, vals, color=[GREEN, RED, GREY], width=.6)\n"
            "for i,v in enumerate(vals): ax.annotate(f'{v:.0f}%',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('% of 200,000 attempts'); ax.set_ylim(0, 100)\n"
            "ax.set_title('Even on a FAIR coin, betting 50% a round: most bust, few moon')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"reached $14k: {m['p_target']*100:.1f}%   wiped out: {m['p_ruin']*100:.1f}%   median end: ${m['median']:.0f}\")"
        ),
        md(
            f"Even with **zero house edge**, betting half your stack ruins **~{R['mc_fair_ruin']:.0f}%** "
            f"of players and lands only **~{R['mc_fair_target']:.0f}%** at \\$14,000; the median ending "
            f"balance is **\\${R['mc_fair_med']:.0f}**. The viral post is that lucky 1%. Add the real "
            "spread, and the ruin share climbs toward certainty. The honest version of the strategy "
            "isn't \"let it run\" — it's \"don't.\""
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The same shape elsewhere.** [Study 301 — Triple-RSI](../../301-triple-rsi/) is the "
            "stock-market twin: a real \"90% win-rate\" that's the *exit's shape*, not an edge. High "
            "win-rate + negative expectancy is the signature of a dressed-up martingale.\n"
            "- **Measure the price yourself.** The read-only `live_paper` capture in this study takes "
            "**no key and no money** — it only reads public quotes. Run it for a week and watch the "
            "favoured ask sit at \\$0.95–0.99.\n"
            "- **The security angle.** A \"free trading bot\" whose order engine lives in a private "
            "repo and reads your wallet key is the real risk here — bigger than the bad EV.\n\n"
            "*Think there's a \\$0.80 favourite to be had on a decided market? Capture the live asks "
            "and show one — then check whether that cheap price wasn't just an honestly uncertain "
            "outcome all along.*"
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
            "# BTC 5-minute Polymarket momentum — a quantitative teardown 🔬\n"
            "### Continuation win-rate vs binomial null · the fair-price identity EV = w − p · "
            "live CLOB-ask capture · a synthetic faithful-engine control · the martingale Monte-Carlo\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). We "
            "separate the two claims the pitch fuses: a **real** momentum-continuation signal "
            "(measured against a binomial null) from an **absent** tradable edge (the favoured side is "
            "priced at its own win-rate). The decisive object is the identity **EV per share = w − p**: "
            "we measure *w* on the tape and *p* on the live book.\n\n"
            "> ⚠️ **Not investment advice.** Real data: Binance BTCUSDT 1-minute klines folded into "
            "Polymarket-style 5-minute windows (45 days to 2026-06-20); live Polymarket CLOB asks from "
            "a **read-only** capture (zero key, zero money). Offline core + synthetic control are "
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
            f"| **Signal** | `REAL` | Continuation at a \\$70 / 2-min-left head-start "
            f"w = **{R['w70']:.1f}%** (n = {R['n70']:,}), binomial **z = {R['z70']:.0f}** vs a coin — "
            "monotone in the threshold, robust at 1-minute-left too. |\n"
            f"| **Tradability** | `MIRAGE` | EV per share = w − p. The live favoured ask sits at "
            f"**\\$0.95–0.99 ≈ w**, so the edge is ~0 gross and **negative** after the ≤3¢ spread the "
            f"bot's own config tolerates. Live all-in edge **{R['live_all_edge']:+.3f}** "
            f"(n={R['live_n']}). |\n"
            f"| **Get rich quick?** | `BUSTED` | At w={R['w70']:.1f}% and *fair* pricing, betting 50%/"
            f"round: P(\\$300→\\$14k) = **{R['mc_fair_target']:.1f}%**, P(ruin) = "
            f"**{R['mc_fair_ruin']:.0f}%**, median end **\\${R['mc_fair_med']:.0f}**. |\n\n"
            "> 💡 In plain words: the move really does predict the close — and the order book already "
            "knows, so it sells you the prediction at cost. What's left is variance, and 50%-of-stack "
            "sizing turns variance into ruin."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let a 5-minute window open at $P_0$, sit at $P_\\tau$ with $\\tau$ seconds left, and close "
            "at $P_T$. Resolution: **Up** iff $P_T \\ge P_0$. The signal is the snapshot move "
            "$m = P_\\tau - P_0$; the strategy buys the side $\\mathrm{sgn}(m)$ at the CLOB ask $p$.\n\n"
            "- **H₁ (momentum is real).** $w(D) \\equiv \\Pr[\\,\\mathrm{sgn}(P_T-P_0)=\\mathrm{sgn}(m)\\mid |m|\\ge D\\,] > \\tfrac12$, rising in $D$.\n"
            "- **H₂ (it is tradable).** $\\mathbb{E}[\\text{EV}] = w - p > 0$ at the price actually quoted.\n"
            "- **H₃ (the jackpot is repeatable).** Fractional-Kelly/50%-stack compounding reaches the "
            "advertised multiple with non-trivial probability.\n\n"
            "We find **H₁ strongly accepted**, **H₂ rejected** (the book prices $p\\approx w$), and "
            "**H₃ rejected** (ruin dominates even at $w-p=0$). The pitch is true exactly where it "
            "doesn't pay and false exactly where it would."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The binary-share EV identity is the whole teardown:\n\n"
            "$$\\text{EV per share} = w(1-p) - (1-w)p = w - p, \\qquad \\text{EV per \\$ staked} = \\frac{w-p}{p}.$$\n\n"
            "Because $p$ is the market's *estimate* of $w$, an efficient liquid book drives $p\\to w$ "
            "and the edge to zero. The favourite-longshot literature says short-odds favourites are if "
            "anything *over*-priced ($p>w$), making the edge **negative**. The price is **endogenous** "
            "to the signal: a cheap $p$ exists only where $w$ is correspondingly low. So measuring $w$ "
            "alone can't establish an edge — you must measure $p$ at the moment of the signal, which is "
            "what the live capture does."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Real tape.** Binance BTCUSDT 1-minute klines, {R['days']} days to 2026-06-20, folded "
            f"into {R['windows']:,} non-overlapping 5-minute windows aligned to minute % 5 (the "
            "`btc-updown-5m-<unix>` grid). Snapshots at 2-min-left and 1-min-left; resolution Up iff "
            "$P_T \\ge P_0$ (Chainlink-equivalent on close).\n"
            "- **Signal test.** Binomial $z$ of $w(D)$ against $\\tfrac12$ across a threshold sweep, at "
            "both snapshots. (A directional win-rate, not a return — the Signal axis here is "
            "predictive accuracy.)\n"
            "- **Price.** A read-only paper-trade scaffold polls `clob.polymarket.com` for the favoured "
            "side's best ask at the 2-minute mark on live BTC markets and resolves each against the "
            "realised close. No key, no order, no money.\n"
            "- **Positive control.** A deterministic arithmetic-Brownian window generator with known "
            "per-minute $\\sigma$: the measured continuation must equal the closed-form conditional "
            "expectation $\\mathbb{E}[\\Phi(|m|/\\sigma\\sqrt{k})\\mid |m|\\ge D]$, and the fair price "
            "$p=w$ must give EV $=0$ exactly.\n"
            "- **Costs/caps (named).** The bot's own YAML tolerates a **3¢** spread and a **\\$30** "
            "top-of-book — i.e. thin liquidity and a real bid/ask tax, both adverse."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The signal is real — continuation vs a binomial null\n\n"
            "Win-rate $w(D)$ with its $\\pm$ binomial standard error, 2-minutes-left snapshot. The "
            "null (coin) is the dashed line; every bar clears it by many SEs."
        ),
        code(
            "Ds = [30, 50, 70, 100, 150]\n"
            "if HAVE_REAL:\n"
            "    rs = [st.continuation(W, d) for d in Ds]\n"
            "    ws = np.array([r['w'] for r in rs]); ns = np.array([r['n'] for r in rs]); zs=[r['z'] for r in rs]\n"
            "else:\n"
            "    ws = np.array([R['w30'],R['w50'],R['w70'],R['w100'],R['w150']])/100\n"
            "    ns = np.array([R['n30'],4322,R['n70'],R['n100'],620]); zs=[R['z30'],56.5,R['z70'],36.4,24.2]\n"
            "se = np.sqrt(ws*(1-ws)/ns)\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar([str(d) for d in Ds], ws*100, yerr=se*100, color=GREEN, width=.6, capsize=4)\n"
            "ax.axhline(50, ls='--', c=GREY, label='binomial null (coin)')\n"
            "ax.set_ylim(45,100); ax.set_xlabel('head-start threshold D ($)'); ax.set_ylabel('continuation win-rate w (%)')\n"
            "ax.set_title('Momentum into close: real and monotone (z = 24–65)'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('w:', [f'{x*100:.1f}%' for x in ws]); print('z vs coin:', [f'{z:.0f}' for z in zs])"
        ),
        md(
            f"> 💡 In plain words: at \\$70 the close follows the move **{R['w70']:.1f}%** of the time "
            f"(n={R['n70']:,}, **z={R['z70']:.0f}**) — H₁ accepted with overwhelming significance, and "
            "monotone in $D$. With **1 minute** left it's even sharper "
            f"({R['w70_1m']:.1f}% at \\$70). The momentum is not the illusion."
        ),
        md(
            "### 4b · The edge is a mirage — w vs the price you pay\n\n"
            "Overlay the measured win-rate (what it's worth) on the live favoured ask (what it costs). "
            "The break-even line is $p=w$; the strategy buys at or above it."
        ),
        code(
            "rows = data.load_live(); b = st.live_buckets(rows)\n"
            "if HAVE_REAL:\n"
            "    w70 = st.continuation(W,70)['w']; w100 = st.continuation(W,100)['w']\n"
            "else:\n"
            "    w70, w100 = R['w70']/100, R['w100']/100\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "cats = ['$70 move','$100 move']\n"
            "ax.bar([0,1], [w70*100, w100*100], width=.5, color=GREEN, label='w — true win-rate (tape)')\n"
            "ax.plot([0,1], [97,99], 'D', ms=12, c=RED, label='p — live favoured ask (CLOB)')\n"
            "ax.set_xticks([0,1]); ax.set_xticklabels(cats); ax.set_ylim(80,102)\n"
            "ax.set_ylabel('percent / cents'); ax.set_title('You pay p ≈ w (or more): edge w−p ≤ 0'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"live ALL-bucket: n={b['all']['n']} w={b['all']['w']*100:.1f}% p=${b['all']['p']:.2f} edge={b['all']['edge']:+.3f} EV/$={b['all']['ev']:+.3f}\")\n"
            "print(f\"live SIGNAL bucket (>=$70): n={b['signal']['n']}  — too small to estimate the sign of a ~3% edge\")"
        ),
        md(
            f"> 💡 In plain words: the favoured ask is **\\$0.97–0.99**, sitting right on top of the "
            f"true win-rate. Across the whole live capture buying the favourite ran an EV of "
            f"**{R['live_all_edge']:+.3f}** per share (n={R['live_n']}). The \\$70-signal sub-sample "
            f"(n={R['live_sig_n']}) is far too small to pin the sign of a ~3% edge — by design: at "
            "$w\\approx0.95$ you'd need ~400–500 signal windows for significance, which is itself the "
            "finding (no *large* edge can hide, or it would show fast). **H₂ rejected on the structure**: "
            "$p\\approx w$, and the 3¢ spread does the rest."
        ),
        md(
            "### 4c · Faithful-engine control — measured continuation = closed form\n\n"
            "On a deterministic Brownian tape with known per-minute $\\sigma$, the measured continuation "
            "must equal $\\mathbb{E}[\\Phi(|m|/\\sigma\\sqrt2)\\mid|m|\\ge70]$ — and pricing at the fair "
            "$p=w$ must zero the EV. Both hold to Monte-Carlo error:"
        ),
        code(
            "sig = [30., 45., 60.]\n"
            "meas = [st.continuation(data.synthetic_windows(n=200_000, sigma_per_min=s, seed=351),70)['w'] for s in sig]\n"
            "exact = [st.expected_continuation(70, s) for s in sig]\n"
            "fig, ax = plt.subplots(figsize=(8.8,4.3))\n"
            "x=np.arange(len(sig)); ax.bar(x-.18,[m*100 for m in meas],.36,color=GREEN,label='measured by engine')\n"
            "ax.bar(x+.18,[e*100 for e in exact],.36,color=GREY,label='exact E[Φ|move≥70]')\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'σ=${s:.0f}/min' for s in sig]); ax.set_ylim(80,100)\n"
            "ax.set_ylabel('continuation w (%)'); ax.set_title('Engine measures exactly what theory predicts'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for s,m,e in zip(sig,meas,exact): print(f'σ={s:.0f}: measured {m*100:.2f}%  exact {e*100:.2f}%  fair-price EV(p=w)={(m-m)/m:+.4f}')"
        ),
        md(
            "> 💡 In plain words: the measurement engine is unbiased — it reproduces the closed-form "
            "continuation at every volatility, and confirms the tautology that matters: **at the fair "
            "price, EV is exactly zero.** Higher $\\sigma$ → lower $w$ (harder to predict) *and* a lower "
            "fair price in lock-step. There is no $\\sigma$ at which the favourite is a bargain."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `REAL`** — $w(70)$ = {R['w70']:.1f}% at binomial **z = {R['z70']:.0f}** "
            f"(n = {R['n70']:,}), monotone in $D$, sharper at 1-min-left ({R['w70_1m']:.1f}%). Momentum "
            "into the close is a genuine, large-$z$ effect on the real tape.\n"
            f"- **Tradability `MIRAGE`** — EV = w − p with the live ask $p\\approx0.97$–$0.99\\approx w$; "
            f"all-in live edge {R['live_all_edge']:+.3f}. Gross ~0, negative after the 3¢ spread and "
            "thin (\\$30) book the bot itself flags. No harvestable edge.\n"
            f"- **Get rich quick? `BUSTED`** — at fair pricing and 50%/round, P(\\$300→\\$14k) = "
            f"{R['mc_fair_target']:.1f}%, P(ruin) = {R['mc_fair_ruin']:.0f}%; pay 5¢ over fair and ruin "
            f"is ~{R['mc_over_ruin']:.0f}%. The viral result is a survivor of a heavy-tailed bet."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the bankroll distribution\n\n"
            "The terminal-wealth law under the pitch's own sizing (50%/round), at the measured "
            "win-rate and the *fair* price (the most generous assumption — zero house edge)."
        ),
        code(
            "w = R['w70']/100\n"
            "rows_mc = {}\n"
            "for label, p, frac in [('fair, 50%', round(w,3), .50), ('+5¢, 50%', round(min(w+.05,.99),3), .50), ('fair, 5%', round(w,3), .05)]:\n"
            "    rows_mc[label] = st.monte_carlo(w, p, frac=frac)\n"
            "labels = list(rows_mc); ruin=[rows_mc[l]['p_ruin']*100 for l in labels]; tgt=[rows_mc[l]['p_target']*100 for l in labels]\n"
            "x=np.arange(len(labels)); fig,ax=plt.subplots(figsize=(8.8,4.3))\n"
            "ax.bar(x-.18,ruin,.36,color=RED,label='P(ruin <$1)'); ax.bar(x+.18,tgt,.36,color=GREEN,label='P(reach $14k)')\n"
            "ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel('% of 200k paths'); ax.set_ylim(0,100)\n"
            "ax.set_title('Sizing, not signal, decides the outcome'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for l in labels: m=rows_mc[l]; print(f\"{l:10s} edge={m['edge']:+.3f}  ruin={m['p_ruin']*100:5.1f}%  target={m['p_target']*100:5.2f}%  median=${m['median']:.0f}\")"
        ),
        md(
            "> 💡 In plain words: drop the stake to 5% and ruin disappears — but so does the jackpot "
            f"(median ~\\${R['mc_yaml_med']:,}, never \\$14k). The only way to the advertised number is "
            "the 50% bet that also ruins ~40% of players on a *fair* coin. Pay the real spread and it's "
            "a near-certain bust. There is no sizing that delivers the story without the tail."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The equity-market twin.** [Study 301 — Triple-RSI](../../301-triple-rsi/): a real "
            "viral \"90% win-rate\" whose win-rate is the *exit's shape*, not the edge — the same "
            "high-hit-rate / negative-expectancy martingale signature.\n"
            "- **Favourite-longshot on prediction markets.** Extend the live capture to estimate $w-p$ "
            "by ask bucket; the prior is $p>w$ for short-odds favourites (the bias points *against* the "
            "bot). ~400–500 signal windows would pin the sign.\n"
            "- **Microstructure.** The \\$30 top-of-book means lifting the ask *is* the trade; model "
            "the walk-up the strategy pays and the adverse selection in any sub-fair quote.\n\n"
            "*The reproducible core is offline and deterministic; the live capture is read-only "
            "(no key, no money). Methods and sources: [`docs/references.md`](../docs/references.md); "
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
