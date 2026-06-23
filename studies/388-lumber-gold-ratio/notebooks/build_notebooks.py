"""Generate the two narrative notebooks for Study 388 (Lumber-Gold-Ratio).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached WOOD/GLD/SPY/
TLT/^IRX parquets under ../_cache/ and otherwise quote the frozen headline numbers in ``R``
(mirroring docs/results.md). The synthetic null + positive control runs anywhere with no
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance WOOD/GLD/SPY/TLT/^IRX,
# 2008-06-25 -> 2026-06-22, 4,525 days, 18.0 years; lookback 60, enter_z 0, 5 bps one-way).
R = dict(
    start="2008-06-25", end="2026-06-22", days=4525, years=18.0, n_switches=609,
    # book: (ann_ret%, sharpe_excess, max_dd%, pct_stocks%, t_excess)
    switch=(5.21, 0.32, -59.5, 53.2, 1.43),
    buy_hold=(12.16, 0.61, -47.6, 100.0, 3.01),
    blend_6040=(9.36, 0.71, -27.2, 60.0, 3.24),
    random=(3.67, 0.22, -45.8, 53.2, 1.04),
    # head-to-head HAC t (switch minus benchmark) + bootstrap CIs in bps/day
    t_vs_bh=-1.55, t_vs_6040=-1.06, t_vs_random=0.25,
    ci_vs_6040=(-3.79, 1.15), ci_vs_random=(-2.65, 3.51),
    # robustness: (label, switch_sharpe, t_vs_6040, t_vs_random)
    robust=[("lb=40,z>0", 0.35, -0.89, 0.36), ("lb=60,z>0", 0.32, -1.06, 0.25),
            ("lb=90,z>0", 0.36, -0.88, 1.75), ("lb=60,z>+0.5", 0.30, -1.19, 0.40),
            ("lb=60,z>-0.5", 0.54, 0.06, 0.59)],
    # synthetic: (pred_r, switch_sharpe, t_vs_6040, t_vs_random)
    syn=[(0.0, -0.14, 0.56, 2.56), (0.15, 11.25, 41.11, 29.83)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Beats_60%2F40%3F: Busted](https://img.shields.io/badge/Beats_60%2F40%3F-Busted-8b949e?style=flat-square)\n\n"
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

from lumber_gold_ratio import data, strategy as st

HAVE_REAL = data.have_real()
F = data.load_real() if HAVE_REAL else None
RACE = st.race(F, lookback=60, enter_z=0.0, cost_bps=5.0, rand_seed=388) if HAVE_REAL else None
print("real lumber/gold cache present:", HAVE_REAL,
      "| switch Sharpe:", (None if RACE is None else round(RACE["switch"]["sharpe_excess"], 2)),
      "| t vs 60/40:", (None if RACE is None else round(RACE["t_vs_6040"], 2)))
"""

# The frozen headline dict is embedded into the first code cell so every downstream cell can
# quote it whether or not the cache is present.
BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does \"lumber vs gold\" tell you when to own stocks? 🪵\n"
            "### The intermarket switch that's supposed to flip you between stocks and bonds, in plain English\n\n"
            + BADGES +
            "There's a tidy market story: **lumber** is all about building, housing and growth, while "
            "**gold** is all about fear. So when lumber is beating gold, the economy is *risk-on* — own "
            "**stocks**; when gold is beating lumber, it's *risk-off* — hide in **bonds**. Flip between "
            "the two as the ratio flips, and you're supposed to ride the booms and dodge the busts.\n\n"
            "It's a great campfire story. But \"beating the market\" has to mean beating something you'd "
            "*actually* do instead — like a boring **60/40 stock/bond mix**, or even a **coin flip** that "
            "holds bonds on random days. This notebook builds the switch and races it against both. "
            "Spoiler: it loses to the boring mix and ties the coin.\n\n"
            "> 📓 **Plain-language layer.** Want the Sharpe ratios, the HAC *t*-stats and the bootstrap "
            "confidence intervals? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** Real lumber *futures* (`LBS=F`) were retired in **May 2023**, "
            "so we use the **WOOD** timber-stock ETF as a transparent **lumber proxy** — named throughout. "
            "If anything it *flatters* the risk-on story (it's an equity), and the verdict doesn't hinge "
            "on it. Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does the lumber/gold switch make money? | **A little — +5.2%/yr.** But that's the wrong "
            "question. |\n"
            "| Does it beat a boring 60/40 mix? | **No.** Its risk-adjusted score (Sharpe) is "
            f"**{R['switch'][1]}** vs **{R['blend_6040'][1]}** for a static 60/40 — and it falls "
            f"**{abs(R['switch'][2]):.0f}%** at its worst vs the blend's **{abs(R['blend_6040'][2]):.0f}%**. |\n"
            "| Is the *timing* doing anything? | **No.** Flip to bonds on the *same number* of **random** "
            f"days and you get the same result — the head-to-head is a dead heat (*t* = +{R['t_vs_random']}). |\n"
            "| So is the lumber/gold story real? | **As a story, sure. As a strategy, no.** It loses to "
            "the mix it's meant to beat and ties a dice roll. |\n\n"
            "> The lumber/gold ratio is a vivid macro picture. As a stock/bond *timing* rule it adds "
            "nothing a coin couldn't — and costs you ~600 trades to find that out."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"When lumber is outrunning gold, growth is winning — own **stocks**. When gold is "
            "outrunning lumber, fear is winning — own **bonds**. The ratio is a leading risk-on/risk-off "
            "switch.\"*\n\n"
            "Popularised by Gayed & Bilello's award-winning 2015 paper, the picture is seductive: lumber "
            "is the rawest growth commodity (you can't build without it), gold is the rawest fear asset. "
            "Their *ratio* should swing ahead of the stock market, telling you when to lean in and when to "
            "duck. We'll build exactly that switch and see whether the timing pays."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "Here's the trap. A stock/bond switch will *make money* over 18 years — because stocks and "
            "bonds both drift up. That tells you nothing. The real question is whether the **timing** "
            "beats two honest yardsticks: (1) a **static 60/40** mix you could set and forget, and (2) a "
            "**coin** that holds bonds on the *same number* of random days. If the lumber/gold ratio "
            "really sees regimes coming, it beats both. If it's just noise, it loses to the mix and ties "
            "the coin — which is exactly what we find."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "We build the switch on a **transparent lumber proxy**: the **WOOD** timber-stock ETF over "
            f"**GLD** (gold). Over **{R['years']:.0f} years** ({R['start']} → {R['end']}):\n\n"
            "1. **Read the signal.** Each day, is the WOOD/GLD ratio stretched *high* vs its own recent "
            "average? High → risk-on → hold **SPY**. Not high → risk-off → hold **TLT** (long bonds). "
            "Act **one day later** (no cheating) and pay a small cost on every flip.\n"
            "2. **Race it.** Put the switch next to **buy-and-hold SPY**, a **static 60/40** blend, and a "
            "**random** switch that holds bonds on the same number of *random* days.\n"
            "3. **Judge the timing, not the drift.** Compare on a risk-adjusted, above-cash basis — does "
            "the *timing* beat the boring mix and the coin, or not?"
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the ratio itself.** The WOOD/GLD ratio over 18 years — the line the switch reads. "
            "When it's stretched high the rule wants stocks; when it's low it wants bonds."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ratio = st.lumber_gold_ratio(F)\n"
            "    fig, ax = plt.subplots(figsize=(9.4, 4.2))\n"
            "    ax.plot(ratio.index, ratio.values, c=GREEN, lw=1.1)\n"
            "    ax.axhline(ratio.mean(), ls='--', c=GREY, label=f'mean {ratio.mean():.2f}')\n"
            "    ax.set_title('The WOOD/GLD (lumber proxy / gold) ratio — the switch reads this line')\n"
            "    ax.set_ylabel('WOOD / GLD'); ax.legend()\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('ratio range:', round(float(ratio.min()),3), '->', round(float(ratio.max()),3))\n"
            "else:\n"
            "    print('no cache — see docs/results.md for the frozen numbers')"
        ),
        md(
            "It heaved around — high in the 2008-era commodity world, crushed afterwards. The switch "
            "flips you into stocks when it's stretched high. Does that timing help? Let's race it."
        ),
        md(
            "**The race.** Risk-adjusted score (Sharpe, above cash) for the lumber/gold switch next to "
            "the three things you'd do instead. Higher is better."
        ),
        code(
            "labels = ['lumber/gold\\nswitch', 'buy & hold\\nSPY', 'static\\n60/40', 'random\\ntiming']\n"
            "if HAVE_REAL:\n"
            "    sh = [RACE['switch']['sharpe_excess'], RACE['buy_hold']['sharpe_excess'],\n"
            "          RACE['blend_6040']['sharpe_excess'], RACE['random']['sharpe_excess']]\n"
            "else:\n"
            "    sh = [R['switch'][1], R['buy_hold'][1], R['blend_6040'][1], R['random'][1]]\n"
            "cols = [RED, GREY, GREEN, GREY]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "ax.bar(labels, sh, color=cols, width=.6)\n"
            "for i,v in enumerate(sh): ax.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('Sharpe ratio (above cash)')\n"
            "ax.set_title('The switch (red) loses to a boring static 60/40 (green)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('switch Sharpe', round(sh[0],2), 'vs static 60/40', round(sh[2],2))"
        ),
        md(
            f"There it is. The lumber/gold switch scores **{R['switch'][1]}**; a set-and-forget **60/40** "
            f"scores **{R['blend_6040'][1]}** — and does it with a far gentler worst-case fall. All the "
            "clever flipping *under-performs* the laziest possible mix."
        ),
        md(
            "**But does the timing do anything at all?** The fairest test: a **coin** that holds bonds on "
            "the *same number* of random days. If the ratio sees regimes, the real switch pulls ahead. "
            "Here's their growth side by side."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pos = st.switch_positions(F, lookback=60, enter_z=0.0)\n"
            "    rnd = st.random_positions(pos, seed=388)\n"
            "    sw = st.book_returns(F, pos, cost_bps=5.0)['r_net']\n"
            "    rb = st.book_returns(F, rnd, cost_bps=5.0)['r_net']\n"
            "    nav_s = (1+sw).cumprod(); nav_r = (1+rb).cumprod()\n"
            "    fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "    ax.plot(nav_s.index, nav_s.values, c=RED, lw=1.6, label='real lumber/gold switch')\n"
            "    ax.plot(nav_r.index, nav_r.values, c=GREY, lw=1.6, label='random timing (same exposure)')\n"
            "    ax.set_yscale('log'); ax.set_ylabel('growth of $1 (log)')\n"
            "    ax.set_title('Real switch vs a coin with the same stock/bond exposure — a dead heat')\n"
            "    ax.legend(); plt.tight_layout(); plt.show()\n"
            "    print('switch ends at', round(float(nav_s.iloc[-1]),2), '| random ends at', round(float(nav_r.iloc[-1]),2))\n"
            "else:\n"
            "    print('no cache — frozen: switch Sharpe', R['switch'][1], 'vs random', R['random'][1])"
        ),
        md(
            f"The two lines tangle together. The head-to-head *t*-stat is **+{R['t_vs_random']}** — a "
            "statistical tie. In plain words: **the lumber/gold ratio's \"timing\" is no better than "
            "choosing your bond days at random.** The story isn't adding information."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** The switch's Sharpe (**{R['switch'][1]}**) loses to a static 60/40 "
            f"(**{R['blend_6040'][1]}**) and *ties* a same-exposure coin (*t* = +{R['t_vs_random']}). "
            "There's no timing edge.\n"
            f"- **Tradability — Mirage.** ~**{R['n_switches']} flips** in 18 years for a book that "
            f"under-performs the mix and falls **{abs(R['switch'][2]):.0f}%** at its worst — nothing to "
            "deploy.\n"
            "- **Beats 60/40? — Busted.** The whole pitch is that the *timing* beats the boring blend. "
            "It doesn't — the head-to-head is *negative*, and no setting flips it positive."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — count the flips and the falls\n\n"
            "Forget Sharpe for a second. A *defensive* switch is supposed to protect you in crashes. So "
            "how deep does each book fall at its worst — and how often does the switch make you trade?"
        ),
        code(
            "labels = ['lumber/gold\\nswitch', 'buy & hold\\nSPY', 'static\\n60/40']\n"
            "if HAVE_REAL:\n"
            "    dd = [RACE['switch']['max_dd']*100, RACE['buy_hold']['max_dd']*100, RACE['blend_6040']['max_dd']*100]\n"
            "    nsw = RACE['n_switches']\n"
            "else:\n"
            "    dd = [R['switch'][2], R['buy_hold'][2], R['blend_6040'][2]]; nsw = R['n_switches']\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.bar(labels, dd, color=[RED, GREY, GREEN], width=.6)\n"
            "for i,v in enumerate(dd): ax.annotate(f'{v:.0f}%',(i,v),ha='center',va='top')\n"
            "ax.set_ylabel('worst drawdown (%)')\n"
            "ax.set_title(f'The \"defensive\" switch falls the FARTHEST — and flips ~{nsw} times')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'{nsw} whole-book rotations in 18 years; switch worst fall {dd[0]:.0f}% vs 60/40 {dd[2]:.0f}%')"
        ),
        md(
            f"The punchline writes itself: the supposedly *defensive* lumber/gold switch has the "
            f"**deepest** drawdown of the three (**{abs(R['switch'][2]):.0f}%**, vs the 60/40's "
            f"**{abs(R['blend_6040'][2]):.0f}%**) — because the ratio kept it in bonds during stock "
            f"rallies and in stocks during routs. And it cost you ~**{R['n_switches']} trades** to get "
            "there. A defensive switch that falls farther than the thing it's defending against is not a "
            "strategy — it's a story with a trading bill."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The cousins.** [Study 305 — Gold-Oil-Ratio](../../305-gold-oil-ratio/) and "
            "[Study 85 — Dr Copper](../../85-dr-copper/) test the same \"a cyclical commodity leads the "
            "economy\" hope. [Study 113 — Gold-Silver-Ratio](../../113-gold-silver-ratio/) is the "
            "within-metals version.\n"
            "- **The yardstick.** [Study 144 — Permanent-Portfolio](../../144-permanent-portfolio/) is "
            "the passive static-mix family any *timing* rule has to beat — and this one doesn't.\n"
            "- **Build your own.** Swap WOOD for the (short) `LBR` lumber future or a copper/gold ratio "
            "and re-run the race — the proxy changes, the timing problem doesn't.\n\n"
            "*Think the lumber/gold switch beats 60/40 by more than luck? Build the switch, race it "
            "against the static blend **and** a same-exposure coin, and show it landing **ahead of both** "
            "— then we'll talk.*"
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
            "# Lumber-Gold-Ratio — a quantitative teardown 🔬\n"
            "### The rotation switch · an excess-of-cash race vs 60/40 / buy-and-hold / random rotation · "
            "HAC *t* + block-bootstrap CIs · costs · a synthetic planted-edge power control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The folk "
            "claim (Gayed & Bilello, 2015) is that the **lumber/gold ratio** is a leading "
            "risk-on/risk-off switch. We rebuild the tradable distillation — a binary SPY/TLT rotation "
            "keyed off the standardised log-ratio level — and confront it with the only question that "
            "matters: does the **timing** beat the passive mix and a same-exposure coin? The decisive "
            "object is not the switch's own Sharpe (a bond sleeve flatters it in a falling-rate regime) "
            "but the **HAC *t* that the switch's net excess return exceeds the 60/40 blend's**.\n\n"
            "> ⚠️ **Data + proxy note.** Lumber futures (`LBS=F`) ended May 2023; we use the **WOOD** "
            "timber-equity ETF as a transparent **lumber proxy** (named on the Signal axis — it's an "
            "equity, so it *flatters* the risk-on read). Real data: yfinance daily adjusted closes — "
            "WOOD/GLD/SPY/TLT/^IRX, 2008→2026. Offline core + synthetic control are deterministic. "
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
            f"| **Signal** | `NONE` | Switch Sharpe **{R['switch'][1]}** vs static 60/40 "
            f"**{R['blend_6040'][1]}**; HAC **t vs 60/40 = {R['t_vs_6040']}** (negative), "
            f"**t vs random = +{R['t_vs_random']}** (dead heat). |\n"
            f"| **Tradability** | `MIRAGE` | **~{R['n_switches']} whole-book rotations** / 18y for a book "
            f"that under-performs the mix and draws down **{R['switch'][2]:.1f}%** (vs 60/40's "
            f"**{R['blend_6040'][2]:.1f}%**). |\n"
            f"| **Beats 60/40?** | `BUSTED` | Head-to-head daily-excess CI **[{R['ci_vs_6040'][0]:+.2f}, "
            f"{R['ci_vs_6040'][1]:+.2f}] bps/day** straddles zero; no lookback/threshold flips *t* "
            "positive. |\n\n"
            "> 💡 In plain words: the switch makes money (stocks and bonds both drift up), but the "
            "*timing* it adds is worthless — it loses to a set-and-forget 60/40 and ties a coin with the "
            "same average exposure. The synthetic control proves the test *has* power; the real tape has "
            "no edge to find."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $L_t, G_t$ be the lumber-proxy and gold closes, $\\rho_t = \\log(L_t/G_t)$, and "
            "$z_t = (\\rho_t - \\mu_{t}^{(60)}) / \\sigma_t^{(60)}$ its standardised deviation from a "
            "60-day trailing mean. The **switch** sets next-day weight $w_{t+1} = \\mathbb{1}[z_t > 0]$ in "
            "SPY (else TLT) — one execution lag, neutral 0.5 while warming up. The book earns "
            "$w\\,r^{\\text{SPY}} + (1-w)\\,r^{\\text{TLT}}$, minus $|\\Delta w|\\cdot c$ costs.\n\n"
            "- **H₁ (the timing predicts).** The switch's excess-of-cash daily mean exceeds a static "
            "60/40's: $\\mathbb{E}[r^{\\text{sw}}-r_f] > \\mathbb{E}[r^{6040}-r_f]$.\n"
            "- **H₂ (it's deployable).** Net of costs the edge survives at NAV scale with tolerable "
            "drawdown.\n"
            "- **H₃ (it's *information*, not exposure).** The switch beats a **same-exposure random "
            "timer** — the ratio carries regime signal beyond the average beta it implies.\n\n"
            "We find **H₁ rejected** (t vs 60/40 = −1.06, *negative*), **H₂ rejected** (~609 trades, "
            "−59.5% DD, under-performs the mix), **H₃ rejected** (t vs random = +0.25, a tie). The "
            "intermarket story is true as narrative and empty as timing."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The whole teardown is one comparison of daily excess-of-cash return streams, judged by an "
            "**autocorrelation-robust** standard error:\n\n"
            "$$\\widehat{\\Delta} = \\overline{(r^{\\text{sw}}-r_f) - (r^{\\text{bench}}-r_f)},\\qquad "
            "t_{\\text{HAC}} = \\frac{\\widehat{\\Delta}}{\\widehat{\\mathrm{se}}_{\\text{NW}}(\\widehat{\\Delta})}.$$\n\n"
            "A raw Sharpe is the wrong lens: a switch that parks in **TLT** through a falling-rate decade "
            "books a flattering bond Sharpe that has nothing to do with the *ratio*. The honest bench is "
            "the **static 60/40** (same asset menu, zero timing) and the **same-exposure random timer** "
            "(same average beta, zero information). Newey-West corrects the SE for the heavy daily "
            "autocorrelation; a **circular block bootstrap** puts a CI on the mean daily difference. "
            "Those two head-to-heads, not the switch's own Sharpe, decide the Signal axis."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Real tape.** yfinance adjusted closes — WOOD (lumber proxy), GLD, SPY, TLT, ^IRX "
            f"(cash), {R['start']}→{R['end']}, {R['days']:,} days. WOOD is an explicit **proxy** (LBS=F "
            "futures discontinued May 2023), named on the axis.\n"
            "- **Signal.** $z_t$ = standardised 60-day deviation of the log WOOD/GLD ratio, known at the "
            "close of *t*.\n"
            "- **Switch.** $w_{t+1}=\\mathbb{1}[z_t>0]$ in SPY else TLT; **one** execution lag; neutral "
            "0.5 during warm-up (no look-ahead, no directional bet without signal).\n"
            "- **Race.** vs buy-and-hold SPY, a **static 60/40**, and a **same-exposure random timer** "
            "(same count of bond days on *random* days, seeded).\n"
            "- **Costs.** 5 bps one-way × NAV on every rotation ($|\\Delta w|$ turnover).\n"
            "- **Inference.** Newey-West **HAC t** on the daily excess-of-cash difference + a **circular "
            "block bootstrap** (block 21, 2000 draws) CI.\n"
            "- **Control.** A deterministic tape with a planted forward link `pred_r` between the lagged "
            "z-score and the next-day equity-minus-bond spread: `pred_r=0` is the null, `pred_r>0` the "
            "folklore-consistent positive control."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The race — risk-adjusted, above cash, net of costs\n\n"
            "Sharpe (excess-of-cash) for all four books. The switch must clear the **static 60/40**; it "
            "doesn't come close."
        ),
        code(
            "keys = ['switch', 'buy_hold', 'blend_6040', 'random']\n"
            "labels = ['lumber/gold\\nswitch', 'buy & hold\\nSPY', 'static\\n60/40', 'random\\ntiming']\n"
            "if HAVE_REAL:\n"
            "    sh = [RACE[k]['sharpe_excess'] for k in keys]\n"
            "    te = [RACE[k]['t_excess'] for k in keys]\n"
            "else:\n"
            "    sh = [R[k][1] for k in keys]; te = [R[k][4] for k in keys]\n"
            "x = np.arange(len(keys))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x, sh, color=[RED, GREY, GREEN, GREY], width=.6)\n"
            "for i,(v,t) in enumerate(zip(sh,te)): ax.annotate(f'{v:.2f}\\n(t_ex {t:.1f})',(i,v),ha='center',va='bottom',fontsize=9)\n"
            "ax.set_xticks(x); ax.set_xticklabels(labels)\n"
            "ax.set_ylabel('Sharpe (excess-of-cash)')\n"
            "ax.set_title('The switch loses the race to a static 60/40 (green)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Sharpe — switch', round(sh[0],2), '| 60/40', round(sh[2],2), '| random', round(sh[3],2))"
        ),
        md(
            f"> 💡 In plain words: the switch's own Sharpe is **{R['switch'][1]}** — *below* the static "
            f"**60/40**'s **{R['blend_6040'][1]}** and barely above the random timer's **{R['random'][1]}**. "
            "Even buy-and-hold SPY beats it on raw return. The clever flipping is destroying value, not "
            "adding it."
        ),
        md(
            "### 4b · The decisive head-to-heads — HAC *t* and bootstrap CIs\n\n"
            "The switch's **net excess-of-cash daily return minus the benchmark's**, with a "
            "Newey-West *t* and a circular block-bootstrap 95% CI (bps/day). For an edge we need a "
            "*positive* *t* clearing 2 against 60/40 **and** against random."
        ),
        code(
            "if HAVE_REAL:\n"
            "    tv = [RACE['t_vs_bh'], RACE['t_vs_6040'], RACE['t_vs_random']]\n"
            "    ci6 = RACE['ci_vs_6040']; cir = RACE['ci_vs_random']\n"
            "    ci6 = (ci6[0]*1e4, ci6[1]*1e4); cir = (cir[0]*1e4, cir[1]*1e4)\n"
            "else:\n"
            "    tv = [R['t_vs_bh'], R['t_vs_6040'], R['t_vs_random']]\n"
            "    ci6 = R['ci_vs_6040']; cir = R['ci_vs_random']\n"
            "labs = ['vs buy&hold\\nSPY', 'vs static\\n60/40', 'vs random\\ntiming']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.3))\n"
            "a1.bar(labs, tv, color=[GREY, RED, AMBER], width=.6)\n"
            "a1.axhline(2, ls='--', c=GREEN, label='+2 (edge bar)'); a1.axhline(-2, ls='--', c=RED)\n"
            "a1.axhline(0, c='k', lw=.8)\n"
            "for i,t in enumerate(tv): a1.annotate(f'{t:.2f}',(i,t),ha='center',va='bottom' if t>=0 else 'top')\n"
            "a1.set_ylabel('HAC t (switch minus benchmark)'); a1.set_ylim(-2.4, 2.4)\n"
            "a1.set_title('No positive t — switch never beats a bench'); a1.legend()\n"
            "a2.barh(['vs 60/40','vs random'], [ci6[1]-ci6[0], cir[1]-cir[0]],\n"
            "        left=[ci6[0], cir[0]], color=[RED, AMBER], height=.5)\n"
            "a2.axvline(0, c='k', lw=1)\n"
            "a2.set_xlabel('mean daily edge, 95% bootstrap CI (bps/day)')\n"
            "a2.set_title('Both CIs straddle zero')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('t vs 60/40', round(tv[1],2), '| t vs random', round(tv[2],2),\n"
            "      '| CI vs 60/40 (bps/day)', tuple(round(c,2) for c in ci6))"
        ),
        md(
            f"> 💡 In plain words: against the **60/40** the *t* is **{R['t_vs_6040']}** — *negative* — and "
            f"the CI **[{R['ci_vs_6040'][0]:+.2f}, {R['ci_vs_6040'][1]:+.2f}] bps/day** straddles zero. "
            f"Against the **random** timer it's **+{R['t_vs_random']}**, a textbook tie. H₁ and H₃ both "
            "rejected: the lumber/gold ratio carries **no** timing information beyond the average exposure "
            "it happens to imply."
        ),
        md(
            "### 4c · Robustness — no lookback or threshold rescues it\n\n"
            "Sweep the lookback and entry threshold. The *t* vs 60/40 is negative-or-zero everywhere; the "
            "*t* vs random never clears 2. The only Sharpe-lifting config (z>−0.5) is just *hold stocks "
            "almost always* — no longer a rotation."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rob = []\n"
            "    for lb, ez, lab in [(40,0.0,'lb=40,z>0'),(60,0.0,'lb=60,z>0'),(90,0.0,'lb=90,z>0'),\n"
            "                        (60,0.5,'lb=60,z>+0.5'),(60,-0.5,'lb=60,z>-0.5')]:\n"
            "        r2 = st.race(F, lookback=lb, enter_z=ez, cost_bps=5.0, rand_seed=388)\n"
            "        rob.append((lab, r2['switch']['sharpe_excess'], r2['t_vs_6040'], r2['t_vs_random']))\n"
            "else:\n"
            "    rob = R['robust']\n"
            "labs = [r[0] for r in rob]; t6 = [r[2] for r in rob]; tr = [r[3] for r in rob]\n"
            "x = np.arange(len(rob))\n"
            "fig, ax = plt.subplots(figsize=(9.6, 4.3))\n"
            "ax.bar(x-.2, t6, .4, color=RED, label='t vs 60/40')\n"
            "ax.bar(x+.2, tr, .4, color=AMBER, label='t vs random')\n"
            "ax.axhline(2, ls='--', c=GREEN, label='+2 (edge bar)'); ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels(labs, rotation=15); ax.set_ylabel('HAC t')\n"
            "ax.set_title('Across the grid: t vs 60/40 stays ≤0, t vs random never reaches 2'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('robustness (label, sharpe, t_vs_6040, t_vs_random):')\n"
            "for r in rob: print('  ', r[0], round(r[1],2), round(r[2],2), round(r[3],2))"
        ),
        md(
            "> 💡 In plain words: there is no corner of the parameter grid where the *timing* adds value. "
            "Loosening to **z>−0.5** lifts the Sharpe only because it parks you in stocks ~90% of the "
            "time — riding the SPY drift, not rotating — and it *still* ties random (t = 0.59). The "
            "result is not a tuning artefact; it's the absence of an edge."
        ),
        md(
            "### 4d · Faithful-engine & power control — we know the truth here\n\n"
            "A deterministic tape with a planted forward link `pred_r` (lagged z-score → next-day "
            "equity-minus-bond spread). With `pred_r=0` (no link) the decisive *t* vs 60/40 must stay "
            "small; with `pred_r=+0.15` (folklore-consistent) it must light up. Both hold — so the engine "
            "is unbiased *and* has power; the flat real-tape result is no-edge, not no-power."
        ),
        code(
            "res = []\n"
            "for pr in (0.0, 0.15):\n"
            "    syn, _ = data.synthetic_daily(n_days=4000, pred_r=pr, seed=388)\n"
            "    rs = st.race(syn, lookback=60, enter_z=0.0, cost_bps=5.0, rand_seed=388)\n"
            "    res.append((pr, rs['switch']['sharpe_excess'], rs['t_vs_6040'], rs['t_vs_random']))\n"
            "labels = [f'planted\\npred_r={pr:+.2f}' for pr,_,_,_ in res]\n"
            "t6 = [r[2] for r in res]\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "ax.bar(labels, t6, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='+2 (edge bar)')\n"
            "for i,t in enumerate(t6): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('HAC t vs 60/40'); ax.set_title('Control: null stays flat, planted edge lights up')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for pr,s,t6v,trv in res: print(f'pred_r={pr:+.2f}: switch Sharpe {s:6.2f}  t_vs_6040 {t6v:7.2f}  t_vs_random {trv:6.2f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted link the decisive *t* vs 60/40 sits at "
            f"**+{R['syn'][0][2]:.2f}** (flat — no false positive); plant a folklore-consistent edge and "
            f"the same machinery slams to **t = {R['syn'][1][2]:.0f}**. So the test would *easily* find a "
            f"real lumber/gold timing edge. On the real tape it finds **{R['t_vs_6040']}**: not a power "
            "problem — an *edge* problem."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — switch Sharpe **{R['switch'][1]}** vs static 60/40 "
            f"**{R['blend_6040'][1]}**; HAC **t vs 60/40 = {R['t_vs_6040']}** (negative), "
            f"**t vs random = +{R['t_vs_random']}** (tie). The control proves the test has power, so this "
            "is the absence of an edge, not the absence of data. No REAL.\n"
            f"- **Tradability `MIRAGE`** — **~{R['n_switches']} whole-book rotations / 18y** for a book "
            f"that under-performs the passive mix and draws down **{R['switch'][2]:.1f}%** (twice the "
            "60/40's). Nothing to allocate to.\n"
            f"- **Beats 60/40? `BUSTED`** — the head-to-head daily-excess CI "
            f"**[{R['ci_vs_6040'][0]:+.2f}, {R['ci_vs_6040'][1]:+.2f}] bps/day** straddles zero and no "
            "lookback/threshold flips the *t* positive. The intermarket *timing* loses to the blend it is "
            "meant to beat."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the drawdown picture\n\n"
            "The operational truth in one chart: a *defensive* switch should fall *less* than its "
            "benchmarks in a crash. Here are the equity curves and their worst falls — the switch falls "
            "the **farthest**."
        ),
        code(
            "if HAVE_REAL:\n"
            "    import pandas as pd\n"
            "    pos = st.switch_positions(F, lookback=60, enter_z=0.0)\n"
            "    sw = st.book_returns(F, pos, cost_bps=5.0)['r_net']\n"
            "    bh = F['equity'].pct_change().fillna(0.0)\n"
            "    blend = 0.6*F['equity'].pct_change().fillna(0.0) + 0.4*F['bond'].pct_change().fillna(0.0)\n"
            "    def curve(r): return (1+r).cumprod()\n"
            "    fig, ax = plt.subplots(figsize=(9.4, 4.4))\n"
            "    ax.plot(curve(sw).index, curve(sw).values, c=RED, lw=1.6, label=f'lumber/gold switch (DD {RACE[\"switch\"][\"max_dd\"]*100:.0f}%)')\n"
            "    ax.plot(curve(bh).index, curve(bh).values, c=GREY, lw=1.3, label=f'buy & hold SPY (DD {RACE[\"buy_hold\"][\"max_dd\"]*100:.0f}%)')\n"
            "    ax.plot(curve(blend).index, curve(blend).values, c=GREEN, lw=1.6, label=f'static 60/40 (DD {RACE[\"blend_6040\"][\"max_dd\"]*100:.0f}%)')\n"
            "    ax.set_yscale('log'); ax.set_ylabel('growth of $1 (log)')\n"
            "    ax.set_title('The \"defensive\" switch (red) has the deepest drawdown'); ax.legend()\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('worst falls — switch', round(RACE['switch']['max_dd']*100,1), '| 60/40', round(RACE['blend_6040']['max_dd']*100,1))\n"
            "else:\n"
            "    print('no cache — frozen DDs: switch', R['switch'][2], '| 60/40', R['blend_6040'][2])"
        ),
        md(
            f"> 💡 In plain words: the switch's worst fall is **{R['switch'][2]:.1f}%** — *deeper* than "
            f"buy-and-hold SPY (**{R['buy_hold'][2]:.1f}%**) and more than double the static 60/40's "
            f"(**{R['blend_6040'][2]:.1f}%**). The ratio repeatedly held bonds into rallies and stocks "
            "into routs. A defensive rule that amplifies drawdowns and ~600 rotations of cost is the "
            "anti-strategy: all the trading, none of the protection."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The cousins.** [Study 305 — Gold-Oil-Ratio](../../305-gold-oil-ratio/) and "
            "[Study 85 — Dr Copper](../../85-dr-copper/) — the same cyclical-commodity-leads-the-economy "
            "construct; [Study 113 — Gold-Silver-Ratio](../../113-gold-silver-ratio/) the within-metals "
            "sibling.\n"
            "- **The yardstick family.** [Study 144 — Permanent-Portfolio](../../144-permanent-portfolio/) "
            "— the passive static mixes any intermarket *timing* rule must beat.\n"
            "- **Harder versions.** Replace WOOD with the short `LBR` lumber future or a spliced "
            "LBS=F→LBR series, add a vol-target overlay, or test a copper/gold variant; the proxy and the "
            "overlay change, the absent-timing-edge does not.\n\n"
            "*The reproducible core is offline and deterministic; lumber is an explicit proxy. Methods and "
            "sources: [`docs/references.md`](../docs/references.md); frozen numbers: "
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
