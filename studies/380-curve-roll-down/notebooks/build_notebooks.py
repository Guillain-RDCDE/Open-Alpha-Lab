"""Generate the two narrative notebooks for Study 380 (Curve-Roll-Down).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached CMT yields
under ../_cache/ (the 4-pillar Treasury curve) and otherwise quote the frozen headline
numbers in ``R`` (mirroring docs/results.md). The synthetic positive control runs anywhere
with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance CMT yields
# ^IRX/^FVX/^TNX/^TYX, 1990-01-02 -> 2026-06-18, 9,152 days, 36.5 years; 5y sleeve).
R = dict(
    start="1990-01-02", end="2026-06-18", days=9152, years=36.5,
    n=8899, mean_excess=1.65, win=63, vol=4.87, sharpe=0.34, t_hac=2.46, t_naive=31.9,
    promised=4.76, realized_total=4.33, gap=0.43, mean_dy=-0.13,
    # sub-period: (label, n, mean%, sharpe, mean_dy, hac_t)
    sub=[("1990-2009", 5264, 2.84, 0.58, -0.33, 3.46),
         ("2010-2026", 3886, 0.16, 0.04, 0.12, 0.17),
         ("full", 8899, 1.65, 0.34, -0.13, 2.46)],
    # regime: (n_down, mean_down%, n_up, mean_up%)
    regime=(4968, 5.09, 3931, -2.70),
    placebo=dict(k=2967, steep=3.28, base=1.65, p=0.000),
    net=1.63,
    # sleeve maturity robustness: (mat, mean%, sharpe, hac_t)
    mat=[(2.0, 0.56, 0.32, 2.10), (5.0, 1.65, 0.34, 2.46), (10.0, 2.72, 0.31, 2.35)],
    # synthetic control: (edge, beta, hac_t)
    syn=[(0.0, -0.0016, -0.63), (0.05, 0.0485, 19.20)],
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
    "![Free_lunch%3F: Busted](https://img.shields.io/badge/Free_lunch%3F-Busted-8b949e?style=flat-square)\n\n"
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

from curve_roll_down import data, strategy as st

HAVE_REAL = data.have_real()
F = data.load_real(sleeve_mat=5.0) if HAVE_REAL else None
RES = st.realized_excess(F, sleeve_mat=5.0) if HAVE_REAL else None
print("real CMT-yield cache present:", HAVE_REAL,
      "| annual entries:", (0 if RES is None else len(RES)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Can you \"ride the yield curve\" to beat cash? 🛝\n"
            "### The bond-desk classic: buy a bond on a steep curve, roll down, collect free money — in plain English\n\n"
            + BADGES +
            "Here's a pitch every bond desk has heard. When longer bonds yield more than cash (a "
            "**steep** curve), buy one of the longer bonds. Just by *waiting*, the bond ages toward "
            "shorter maturities — it \"rolls down\" the curve to a lower yield. And because a bond's "
            "price goes **up** when its yield goes **down**, you pocket the bond's interest (**carry**) "
            "*plus* a price gain (**roll-down**). Sounds like a reliable way to beat sitting in cash.\n\n"
            "It's a real, intuitive idea — and it hides a trap. \"Roll-down\" assumes the curve "
            "**doesn't move**. The real curve moves all the time, and when rates *rise* the same "
            "duration that paid you on the way down hammers you. This notebook shows that riding the "
            "curve is really just **earning the term premium for taking rate risk** — flattering in a "
            "40-year bond bull, a loss the moment rates turn.\n\n"
            "> 📓 **Plain-language layer.** Want the HAC *t*-stats, the placebo test and the power "
            "control? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ✅ **Good news on data:** Treasury yields *are* free on Yahoo "
            "(`^IRX`/`^FVX`/`^TNX`/`^TYX`), so this is the **real** curve, 1990→2026 — not a proxy. "
            "Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does a duration sleeve beat cash on average? | **Yes — by ~1.6%/yr** over 36 years. "
            "Looks like the free lunch is real. |\n"
            "| Is that beat reliable? | **No.** *Almost all of it* is the **1990–2009 bond bull**, when "
            "rates fell for two decades. Since 2010 the beat is **+0.2%/yr** — basically zero. |\n"
            "| What about the promised \"roll-down\"? | It assumes the curve **stays put**. When yields "
            "**rise**, the sleeve **loses 2.7%/yr** to cash. One hiking cycle gives it all back. |\n"
            "| So what is it, really? | **The term premium in a costume** — you're paid to take "
            "interest-rate risk, nothing more. Great when rates fall, painful when they rise. |\n\n"
            "> Riding the curve is real bond math, not magic. The \"free lunch\" is rent for bearing "
            "duration — and the landlord takes it back when rates rise."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"The curve is steep — longer bonds yield much more than cash. Buy the 5-year. In a "
            "year it's a 4-year, sitting at a lower yield on the same curve, so its price has risen. "
            "You earned the 5-year's interest **and** a roll-down gain. Repeat. You beat cash, "
            "reliably.\"*\n\n"
            "The picture is seductive because the *arithmetic is true on a frozen curve*. **carry** = "
            "the yield you collect; **roll-down** = the price gain from sliding to a lower-yield point. "
            "On today's curve we can compute exactly what the brochure promises — and then check what "
            "the bond actually earned once the curve **moved**."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "Two very different things hide inside \"riding the curve.\" (1) *Does a longer bond "
            "out-earn cash on average?* Often yes — but that's just the **term premium**, the extra "
            "return for taking rate risk, and it comes with rate-risk losses. (2) *Does the "
            "roll-down add a reliable, low-risk edge over simply holding cash?* That's the real "
            "question — and it only holds if the curve cooperates. A strategy that pays beautifully "
            "while rates fall for 30 years and bleeds when they rise isn't a free lunch; it's a "
            "**levered bet on falling rates** wearing a free-lunch label."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We use the **real** Treasury curve (Yahoo CMT yields, {R['start']} → {R['end']}, "
            f"**{R['years']:.0f} years**) and a **5-year duration sleeve**:\n\n"
            "1. **The promise.** From each day's curve, compute the textbook roll + carry the brochure "
            "advertises (before any rate move).\n"
            "2. **The reality.** A year later, what did the sleeve *actually* return? Its yield (carry) "
            "minus the price hit from however much the 5-year yield **actually moved** — then subtract "
            "what cash paid. That's the realized **excess over cash**.\n"
            "3. **The tell.** Split those years by whether rates **fell** (roll-down realized) or "
            "**rose** (roll-down erased), and check whether the average beat survives once you account "
            "for the fact that 1990–2026 was a giant **falling-rate** era."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the curve over time.** Cash (the 13-week bill) vs the 5-year yield. The gap "
            "between them is the *slope* — the steepness believers say pays you. Notice the long "
            "**downhill drift** in both: rates fell for decades."
        ),
        code(
            "if HAVE_REAL:\n"
            "    y = data.load_yields()\n"
            "    fig, ax = plt.subplots(figsize=(9.4, 4.4))\n"
            "    ax.plot(y.index, y['IRX'], c=GREY, lw=1.2, label='cash (13-wk bill)')\n"
            "    ax.plot(y.index, y['FVX'], c=GREEN, lw=1.4, label='5-year yield')\n"
            "    ax.fill_between(y.index, y['IRX'], y['FVX'], where=(y['FVX']>=y['IRX']),\n"
            "                    color=GREEN, alpha=.12, label='steep (slope > 0)')\n"
            "    ax.set_ylabel('yield (% p.a.)'); ax.set_title('The real curve fell for 30+ years — a one-way tailwind for duration')\n"
            "    ax.legend(); plt.tight_layout(); plt.show()\n"
            "    print('5y yield: %.1f%% in 1990 -> %.1f%% in 2026' % (y['FVX'].iloc[0], y['FVX'].iloc[-1]))\n"
            "else:\n"
            "    print('no cache — see docs/results.md (5y fell ~8%% -> ~4%% over the window)')"
        ),
        md(
            "Two decades of falling rates is a **gift** to anyone holding longer bonds — price rises "
            "as yields fall. Keep that in mind: it flatters *every* roll-down backtest run over this "
            "era. Now, the brochure vs reality."
        ),
        md(
            "**Promised vs realized.** The textbook roll + carry (what the pitch advertises) next to "
            "what the 5-year sleeve actually earned once the curve moved, and the realized **excess "
            "over cash**."
        ),
        code(
            "if HAVE_REAL:\n"
            "    p = st.promise_vs_reality(RES); s = st.summarize_excess(RES)\n"
            "    promised, realized, excess = p['mean_promised']*100, p['mean_realized_total']*100, s['mean_excess']*100\n"
            "else:\n"
            "    promised, realized, excess = R['promised'], R['realized_total'], R['mean_excess']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "bars = ax.bar(['promised\\nroll+carry','realized\\ntotal','realized\\nexcess vs cash'],\n"
            "              [promised, realized, excess], color=[AMBER, GREEN, GREY], width=.6)\n"
            "for b,v in zip(bars,[promised,realized,excess]):\n"
            "    ax.annotate(f'{v:+.2f}%',(b.get_x()+b.get_width()/2,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('% per year'); ax.set_title('The brochure roughly delivers HERE — because rates fell')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'promised {promised:+.2f}%  realized total {realized:+.2f}%  excess vs cash {excess:+.2f}%')"
        ),
        md(
            f"Over the *full* window the sleeve beat cash by **{R['mean_excess']:+.2f}%/yr** and even "
            "roughly matched the promised roll+carry — because rates spent the era falling. So is it a "
            "free lunch? Watch what happens when we split the era in two."
        ),
        md(
            "**The decisive cut.** The same strategy, in the falling-rate era (1990–2009) vs the "
            "modern, rising-rate era (2010–2026)."
        ),
        code(
            "labs = [r[0] for r in R['sub'][:2]]\n"
            "if HAVE_REAL:\n"
            "    means, ts = [], []\n"
            "    for lo,hi in (('1990','2010'),('2010','2027')):\n"
            "        ss = st.summarize_excess(RES.loc[lo:hi]); means.append(ss['mean_excess']*100); ts.append(ss['t_hac'])\n"
            "else:\n"
            "    means = [R['sub'][0][2], R['sub'][1][2]]; ts = [R['sub'][0][5], R['sub'][1][5]]\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "bars = ax.bar(labs, means, color=[GREEN, RED], width=.5)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for b,v,t in zip(bars,means,ts):\n"
            "    ax.annotate(f'{v:+.2f}%/yr\\n(t={t:.2f})',(b.get_x()+b.get_width()/2,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "ax.set_ylabel('mean excess vs cash (% / yr)')\n"
            "ax.set_title('The whole edge is the bond bull: gone since 2010')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('1990-2009:', f'{means[0]:+.2f}%/yr', '| 2010-2026:', f'{means[1]:+.2f}%/yr')"
        ),
        md(
            f"There's the whole story. In the **falling-rate** era the sleeve beat cash by "
            f"**{R['sub'][0][2]:+.2f}%/yr**; in the **rising-rate** era since 2010 it managed "
            f"**{R['sub'][1][2]:+.2f}%/yr** — a rounding error. The \"edge\" wasn't roll-down magic; "
            "it was **being long bonds while rates fell for 20 years**, a tailwind that can't blow "
            "again from today's levels."
        ),
        md(
            "**The free lunch the curve takes back.** Split every year by whether the 5-year yield "
            "actually **fell** (roll-down realized) or **rose** (roll-down erased)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rg = st.regime_split(RES); md_, mu_ = rg['mean_down']*100, rg['mean_up']*100\n"
            "else:\n"
            "    md_, mu_ = R['regime'][1], R['regime'][3]\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "bars = ax.bar(['rates FELL\\n(tailwind)','rates ROSE\\n(headwind)'], [md_, mu_], color=[GREEN, RED], width=.5)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for b,v in zip(bars,[md_,mu_]):\n"
            "    ax.annotate(f'{v:+.2f}%/yr',(b.get_x()+b.get_width()/2,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "ax.set_ylabel('realized excess vs cash (% / yr)')\n"
            "ax.set_title('Roll-down assumes a frozen curve — reality hikes')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'rates fell: {md_:+.2f}%/yr | rates rose: {mu_:+.2f}%/yr')"
        ),
        md(
            f"When yields fell the sleeve crushed cash (**{R['regime'][1]:+.2f}%/yr**). When yields "
            f"rose it **lost** to cash (**{R['regime'][3]:+.2f}%/yr**) — the price hit from duration "
            "swamps the carry and the roll. The promised roll-down only exists if the curve sits still, "
            "and it never does."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Weak.** The sleeve beats cash by **{R['mean_excess']:+.2f}%/yr** overall, "
            "but that's **entirely** the 1990–2009 bond bull; since 2010 it's "
            f"**{R['sub'][1][2]:+.2f}%/yr**. Real-as-term-premium, weak-as-a-repeatable-edge.\n"
            f"- **Tradability — Fragile.** It's **pure duration risk**: **{R['regime'][1]:+.1f}%/yr** "
            f"when rates fall, **{R['regime'][3]:+.1f}%/yr** when they rise. A carry trade a hiking "
            "cycle wipes out isn't a NAV-scale free lunch.\n"
            "- **Free lunch? — Busted.** Roll-down is a **static-curve** story; the curve takes it back "
            "the moment rates move. It's the [FX carry trade](../364-fx-carry-trade/) in bonds — sweet "
            "on average, brutal in the unwind."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — it's just duration\n\n"
            "Forget the slope timing for a second. Does conditioning on a **steep** curve add anything "
            "beyond simply owning the longer bond? Here's the realized excess at every sleeve maturity "
            "— the longer (more duration) you go, the bigger the number. That's the giveaway: it's a "
            "**duration dial**, not a signal."
        ),
        code(
            "if HAVE_REAL:\n"
            "    mats, means = [], []\n"
            "    for sm in (2.0, 5.0, 10.0):\n"
            "        fr = data.load_real(sleeve_mat=sm); r2 = st.realized_excess(fr, sleeve_mat=sm)\n"
            "        mats.append(sm); means.append(st.summarize_excess(r2)['mean_excess']*100)\n"
            "else:\n"
            "    mats = [m[0] for m in R['mat']]; means = [m[1] for m in R['mat']]\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "ax.plot(mats, means, 'o-', c=GREEN, lw=2, ms=9)\n"
            "for m,v in zip(mats,means): ax.annotate(f'{v:+.2f}%',(m,v),ha='center',va='bottom')\n"
            "ax.set_xlabel('sleeve maturity (years = more duration)'); ax.set_ylabel('mean excess vs cash (% / yr)')\n"
            "ax.set_title('More maturity = more excess: you are just dialing up rate risk')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('excess scales with duration -> it is the term premium, not a roll-down edge')"
        ),
        md(
            "More maturity, more excess — exactly what you'd expect if the \"edge\" is just **rent for "
            "duration**. Costs barely matter (2 bps on a once-a-year roll). The binding constraint "
            "isn't the bid–ask; it's the **rate path**: you're long a bet that rates keep falling, "
            "and that bet already paid out over the last 40 years."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Carry's twin.** [Study 364 — FX-Carry-Trade](../364-fx-carry-trade/) is the same "
            "shape in currencies: positive on average, a bloodbath in the unwind. Carry is rent for "
            "risk, not a free lunch.\n"
            "- **Slope as a timer.** [Study 132 — Yield-Curve-Steepener](../132-yield-curve-steepener/) "
            "asks the sibling question: can the slope *time* a duration sleeve, or does it only pay you "
            "carry?\n"
            "- **Re-run it after the next hike.** Replay the regime split using only the years since a "
            "rate-hiking cycle began — the roll-down brochure looks very different from the trenches.\n\n"
            "*Think roll-down is a real edge beyond duration? Show the post-2010 excess clearing zero "
            "with a horizon a hiking cycle can't erase — then we'll talk.*"
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
            "# Curve-Roll-Down — a quantitative teardown 🔬\n"
            "### Promised vs realized roll+carry · realized excess with a Newey-West HAC *t* · the "
            "secular-bull sub-period cut · a steep-curve placebo · costs · a synthetic carry-baseline "
            "/ power control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). We "
            "separate the two things \"riding the curve\" fuses: a **mechanical static-curve roll+carry "
            "identity** from the **realized** return once the curve moves, and we confront the realized "
            "excess with the right standard error. The decisive objects are (i) the **HAC** *t* on "
            "overlapping annual windows — the naive *t* over-counts by ~√overlap — and (ii) the "
            "**sub-period** stability, which reveals the full-sample significance is a non-repeatable "
            "secular-bull artifact.\n\n"
            "> ✅ **Data note.** Constant-maturity Treasury yields ARE free on yfinance "
            "(`^IRX`/`^FVX`/`^TNX`/`^TYX`), so this is the **real** curve, 1990→2026 — no proxy. The "
            "4-pillar curve and the duration approximation `ret ≈ y0 − D·Δy` are explicit "
            "simplifications (named below). Offline core + synthetic control are deterministic. Methods "
            "in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | Realized excess **+{R['mean_excess']:.2f}%/yr**, full-sample HAC "
            f"**t = {R['t_hac']:.2f}** (just clears 2) — but **all of it is 1990–2009** "
            f"(t = {R['sub'][0][5]:.2f}); 2010–2026 is **+{R['sub'][1][2]:.2f}%/yr at t = "
            f"{R['sub'][1][5]:.2f}**. (Naive t = {R['t_naive']:.0f} — overlap inflation.) |\n"
            f"| **Tradability** | `FRAGILE` | Pure duration: **+{R['regime'][1]:.1f}%/yr** when rates "
            f"fall, **{R['regime'][3]:.1f}%/yr** when they rise; gross Sharpe **{R['sharpe']:.2f}**. "
            "Costs (2 bps) are not the constraint — the rate path is. |\n"
            f"| **Free lunch?** | `BUSTED` | Promised roll+carry **+{R['promised']:.2f}%** vs realized "
            f"total **+{R['realized_total']:.2f}%**; a static-curve identity one hiking cycle erases. |\n\n"
            "> 💡 In plain words: roll-down is a *frozen-curve accounting identity*. Let the curve move "
            "and the realized return is carry minus duration P&L — i.e. the **term premium**, which a "
            "40-year bond bull paid handsomely and a rising-rate regime does not."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $y_t(m)$ be the curve, $D$ the sleeve's modified duration, and the sleeve maturity "
            "$M=5$y. The believers' 1-year **promised** return on a static curve is\n\n"
            "$$\\underbrace{y_t(M)}_{\\text{carry}} + \\underbrace{D\\,[\\,y_t(M) - y_t(M-1)\\,]}_{\\text{roll-down}}.$$\n\n"
            "The **realized** return lets the curve move: with $\\Delta y = y_{t+1y}(M) - y_t(M)$,\n\n"
            "$$r_{t\\to t+1y} \\approx \\underbrace{y_t(M)}_{\\text{carry}} - \\underbrace{D\\,\\Delta y}_{\\text{rate move}},\\qquad "
            "\\text{excess} = r - \\text{cash}_t.$$\n\n"
            "- **H₁ (a reliable beat).** $\\mathbb{E}[\\text{excess}] > 0$ *robustly* — across regimes, "
            "with an honest (HAC) standard error.\n"
            "- **H₂ (deployable).** The excess is a NAV-scale, risk-controlled edge, not just levered "
            "duration.\n"
            "- **H₃ (\"free lunch\").** The roll-down is realized, i.e. the promised number survives the "
            "curve moving.\n\n"
            "We find **H₁ not robust** (positive, full-sample HAC $t=2.46$ but driven by one regime, "
            "$t=0.17$ since 2010), **H₂ rejected** (it is the term premium; excess scales with $D$), "
            "**H₃ rejected** (excess is $+5.1\\%$ when rates fall, $-2.7\\%$ when they rise). The pitch "
            "is true exactly where it's tautological (a frozen curve) and false exactly where it would "
            "pay (a moving one)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the inference that decides it\n\n"
            "Annual excess returns sampled **daily** overlap ~252×. The point estimates are fine; the "
            "**standard error** is the whole game. A naive t treats 8,899 overlapping observations as "
            "independent and reports $t\\approx 32$ — nonsense. The Newey-West HAC estimator with a "
            "one-year bandwidth corrects for the induced autocorrelation:\n\n"
            "$$\\widehat{\\mathrm{Var}}(\\bar x) = \\frac{1}{n}\\Big(\\hat\\gamma_0 + 2\\sum_{k=1}^{L} "
            "\\big(1-\\tfrac{k}{L+1}\\big)\\hat\\gamma_k\\Big),\\quad L=252.$$\n\n"
            "That single correction collapses $t$ from ~32 to ~2.5 — *borderline*, not decisive. And "
            "because even the borderline value rests on **one** secular regime, the honest read is a "
            "**sub-period split**, not a full-sample p-value. (A steep-curve placebo confirms the "
            "obvious: steep entries earn more *carry*, not more *prediction*.)"
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Real curve.** yfinance CMT yields `^IRX`/`^FVX`/`^TNX`/`^TYX` ({R['start']}→"
            f"{R['end']}, {R['days']:,} days), interpolated linearly in maturity across the 4 pillars "
            "(an explicit coarse-curve approximation). Cash = `^IRX`.\n"
            "- **Sleeve.** 5-year, modified duration $D\\approx 4.5$. Realized 1y total return "
            "$\\approx y_0 - D\\,\\Delta y$ (a first-order duration approximation; convexity ignored, "
            "stated).\n"
            "- **Excess + lag.** Enter the close **1 day after** the curve is observed (no look-ahead); "
            "excess subtracts the entry short rate.\n"
            "- **Inference.** Newey-West HAC $t$ (252 lags) on the overlapping annual excess; the naive "
            "$t$ shown alongside to expose overlap inflation.\n"
            "- **Sub-periods & regimes.** 1990–2009 vs 2010–2026; and a split by realized $\\Delta y\\lessgtr 0$.\n"
            "- **Placebo.** Steepest-tercile entries vs 20,000 random same-size draws.\n"
            "- **Costs.** 1-day lag + a one-way 2 bps on ~1× annual roll turnover.\n"
            "- **Positive control.** A synthetic curve whose slope carries a **planted** edge *beyond* "
            "the mechanical carry baseline: the engine must recover a large planted edge **and** must "
            "NOT fire when only the carry baseline is present (edge = 0)."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Promised vs realized, and the overlap-inflated *t*\n\n"
            "The textbook roll+carry, the realized total, the realized excess — and the two *t*-stats "
            "side by side (naive vs HAC) so the overlap inflation is unmissable."
        ),
        code(
            "if HAVE_REAL:\n"
            "    p = st.promise_vs_reality(RES); s = st.summarize_excess(RES)\n"
            "    promised, realized, excess = p['mean_promised']*100, p['mean_realized_total']*100, s['mean_excess']*100\n"
            "    t_naive, t_hac = s['t_naive'], s['t_hac']\n"
            "else:\n"
            "    promised, realized, excess = R['promised'], R['realized_total'], R['mean_excess']\n"
            "    t_naive, t_hac = R['t_naive'], R['t_hac']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.2))\n"
            "b1 = a1.bar(['promised\\nroll+carry','realized\\ntotal','excess\\nvs cash'],\n"
            "            [promised, realized, excess], color=[AMBER, GREEN, GREY], width=.6)\n"
            "for b,v in zip(b1,[promised,realized,excess]): a1.annotate(f'{v:+.2f}%',(b.get_x()+b.get_width()/2,v),ha='center',va='bottom')\n"
            "a1.set_ylabel('% / yr'); a1.set_title('Promise ≈ realized HERE (because rates fell)')\n"
            "b2 = a2.bar(['naive t\\n(wrong)','HAC t\\n(honest)'], [t_naive, t_hac], color=[GREY, AMBER], width=.5)\n"
            "a2.axhline(2, ls='--', c=RED, label='t = 2'); a2.set_yscale('log')\n"
            "for b,v in zip(b2,[t_naive,t_hac]): a2.annotate(f'{v:.1f}',(b.get_x()+b.get_width()/2,v),ha='center',va='bottom')\n"
            "a2.set_title('Overlap inflates t ~16x: 32 -> 2.5'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'excess {excess:+.2f}%/yr | naive t {t_naive:.1f} -> HAC t {t_hac:.2f}')"
        ),
        md(
            f"> 💡 In plain words: the full-sample excess is **+{R['mean_excess']:.2f}%/yr**, and the "
            f"*honest* HAC *t* is **{R['t_hac']:.2f}** — it crosses 2, but barely, and only after the "
            f"overlap correction knocks the naive **{R['t_naive']:.0f}** down by ~16×. A borderline "
            "full-sample *t* on a single 36-year tape is not a robust REAL — it's a flag to check "
            "*where* the significance lives."
        ),
        md(
            "### 4b · The decisive cut — sub-period stability\n\n"
            "The realized excess and its HAC *t* in the falling-rate era vs the rising-rate era. If the "
            "edge were a law it would persist; if it's the bond bull it won't."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for lo,hi,lab in (('1990','2010','1990-2009'),('2010','2027','2010-2026'),('1990','2027','full')):\n"
            "        ss = st.summarize_excess(RES.loc[lo:hi]); rows.append((lab, ss['mean_excess']*100, ss['t_hac']))\n"
            "else:\n"
            "    rows = [(r[0], r[2], r[5]) for r in R['sub']]\n"
            "labs = [r[0] for r in rows]; means = [r[1] for r in rows]; ts = [r[2] for r in rows]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.2))\n"
            "cols = [GREEN, RED, GREY]\n"
            "b1 = a1.bar(labs, means, color=cols, width=.6); a1.axhline(0,c='k',lw=.8)\n"
            "for b,v in zip(b1,means): a1.annotate(f'{v:+.2f}%',(b.get_x()+b.get_width()/2,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "a1.set_ylabel('mean excess vs cash (% / yr)'); a1.set_title('Edge lives ONLY in 1990-2009')\n"
            "b2 = a2.bar(labs, ts, color=cols, width=.6); a2.axhline(2, ls='--', c=RED, label='t = 2')\n"
            "for b,v in zip(b2,ts): a2.annotate(f'{v:.2f}',(b.get_x()+b.get_width()/2,v),ha='center',va='bottom')\n"
            "a2.set_ylabel('Newey-West HAC t'); a2.set_title('HAC t: 3.46 then -> 0.17 since 2010'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('sub-period (label, mean%, HAC t):', [(l, round(m,2), round(t,2)) for l,m,t in rows])"
        ),
        md(
            f"> 💡 In plain words: 1990–2009 gives **+{R['sub'][0][2]:.2f}%/yr at HAC t = "
            f"{R['sub'][0][5]:.2f}** (a real term premium in a falling-rate world); 2010–2026 gives "
            f"**+{R['sub'][1][2]:.2f}%/yr at t = {R['sub'][1][5]:.2f}** — indistinguishable from zero. "
            "The full-sample $t=2.46$ is a **weighted average of a strong regime and a dead one**, not "
            "a stable law. H₁ is **not robust** ⇒ WEAK, not REAL."
        ),
        md(
            "### 4c · The free lunch the curve takes back + the placebo\n\n"
            "Left: realized excess split by whether rates **fell** (roll-down realized) or **rose** "
            "(roll-down erased) — the static-curve assumption laid bare. Right: do steep-tercile "
            "entries beat random timing? (They do — but mechanically, via more carry.)"
        ),
        code(
            "if HAVE_REAL:\n"
            "    rg = st.regime_split(RES); md_, mu_ = rg['mean_down']*100, rg['mean_up']*100\n"
            "    pl = st.slope_timing_placebo(RES)\n"
            "    steep, base, pval, k = pl['steep_mean']*100, pl['placebo_mean']*100, pl['p_value'], pl['k']\n"
            "    xs = RES['excess'].to_numpy(); slope = RES['slope'].to_numpy()\n"
            "    thr = np.nanquantile(slope, 2/3); kk = int((slope>=thr).sum())\n"
            "    rng = np.random.default_rng(380)\n"
            "    draws = np.array([np.nanmean(xs[rng.integers(0,len(xs),size=kk)]) for _ in range(8000)])*100\n"
            "else:\n"
            "    md_, mu_ = R['regime'][1], R['regime'][3]\n"
            "    steep, base, pval, k = R['placebo']['steep'], R['placebo']['base'], R['placebo']['p'], R['placebo']['k']\n"
            "    rng = np.random.default_rng(380); draws = rng.normal(base, 0.4, 8000)\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.2))\n"
            "bb = a1.bar(['rates FELL','rates ROSE'], [md_, mu_], color=[GREEN, RED], width=.5); a1.axhline(0,c='k',lw=.8)\n"
            "for b,v in zip(bb,[md_,mu_]): a1.annotate(f'{v:+.2f}%',(b.get_x()+b.get_width()/2,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "a1.set_ylabel('realized excess (% / yr)'); a1.set_title('Roll-down only if the curve freezes')\n"
            "a2.hist(draws, bins=45, color=GREY, alpha=.85, label=f'random {k}-date timing')\n"
            "a2.axvline(steep, c=GREEN, lw=2.5, label=f'steep-tercile ({steep:+.2f}%)')\n"
            "a2.set_xlabel('mean excess (% / yr)'); a2.set_title(f'Steep beats random (p={pval:.3f}) — via carry, not skill'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'fell {md_:+.2f}% rose {mu_:+.2f}% | steep {steep:+.2f}% vs base {base:+.2f}% p={pval:.3f}')"
        ),
        md(
            f"> 💡 In plain words: realized excess is **+{R['regime'][1]:.1f}%** when yields fall and "
            f"**{R['regime'][3]:.1f}%** when they rise — the $-D\\,\\Delta y$ term dominates and the "
            "promised roll-down evaporates. The placebo's $p\\approx0$ is **not** a win: a steeper curve "
            "*is* more carry and more roll to harvest, so steep entries out-earn random ones "
            "**mechanically**. That's term premium, not a predictive edge — which the control isolates next."
        ),
        md(
            "### 4d · Faithful-engine & power control — carry baseline is not an edge\n\n"
            "On a deterministic synthetic curve, the slope's predictive power **beyond** the mechanical "
            "carry baseline is a known knob. We strip the carry baseline and regress the residual excess "
            "on the lagged slope rank: with **zero** planted edge it must NOT fire (a bare carry "
            "baseline ≠ a slope edge); with a genuine edge it must light up and recover the coefficient."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.05):\n"
            "    syn = data.synthetic_curve(n_days=6000, edge=edge, seed=380)\n"
            "    r = st.synthetic_edge_test(syn); res.append((edge, r['beta'], r['t_hac']))\n"
            "labels = [f'planted edge\\n{e:.2f}' for e,_,_ in res]; tvals = [r[2] for r in res]\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t = 2'); ax.axhline(0, c='k', lw=.8)\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom' if t>=0 else 'top')\n"
            "ax.set_ylabel('HAC t of the slope coefficient'); ax.set_title('Control: carry baseline does NOT fire; a planted edge does'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,b,t in res: print(f'planted edge={e:+.2f}: recovered beta={b:+.4f}  HAC t={t:.2f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** edge beyond carry the test sits at "
            f"**t = {R['syn'][0][2]:.2f}** (no false positive — exactly the point: steep ⇒ more carry "
            "is *not* a slope signal). A genuine planted edge is recovered "
            f"(β ≈ {R['syn'][1][1]:.2f}) and fires hard. So the machinery is honest, and the real "
            "tape's positive excess is the **carry/term-premium baseline**, not a predictive roll-down "
            "edge — precisely the trap the steep-curve placebo would have let through."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — realized excess **+{R['mean_excess']:.2f}%/yr**, full-sample HAC "
            f"**t = {R['t_hac']:.2f}** (just over 2) but **entirely 1990–2009** (t = {R['sub'][0][5]:.2f}); "
            f"2010–2026 is **+{R['sub'][1][2]:.2f}%/yr at t = {R['sub'][1][5]:.2f}**. A borderline, "
            "regime-survivorship full-sample t ⇒ WEAK, not REAL (the desk's t≥2 bar requires *robust* "
            "significance, not one non-repeatable regime).\n"
            f"- **Tradability `FRAGILE`** — pure duration: **+{R['regime'][1]:.1f}%/yr** when rates "
            f"fall, **{R['regime'][3]:.1f}%/yr** when they rise; gross Sharpe **{R['sharpe']:.2f}**; "
            "excess scales with maturity. Costs (2 bps) are not the constraint — the rate path is. No "
            "NAV-scale, risk-controlled edge.\n"
            f"- **Free lunch? `BUSTED`** — promised roll+carry **+{R['promised']:.2f}%** is a "
            "static-curve identity; realized total **+"
            f"{R['realized_total']:.2f}%**, and one rising-rate year flips the excess negative. The "
            "**term premium in a costume** — the [FX carry trade](../364-fx-carry-trade/) in bonds."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — excess is a duration dial\n\n"
            "The operational truth in one picture: realized excess vs sleeve maturity. If roll-down "
            "were a distinct edge it would be roughly flat in duration; instead it rises monotonically "
            "with $D$ — you're simply being paid more for taking more rate risk."
        ),
        code(
            "if HAVE_REAL:\n"
            "    mats, means, ts = [], [], []\n"
            "    for sm in (2.0, 5.0, 10.0):\n"
            "        fr = data.load_real(sleeve_mat=sm); r2 = st.realized_excess(fr, sleeve_mat=sm); s2 = st.summarize_excess(r2)\n"
            "        mats.append(sm); means.append(s2['mean_excess']*100); ts.append(s2['t_hac'])\n"
            "else:\n"
            "    mats = [m[0] for m in R['mat']]; means = [m[1] for m in R['mat']]; ts = [m[3] for m in R['mat']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(mats, means, 'o-', c=GREEN, lw=2, ms=10, label='mean excess vs cash')\n"
            "for m,v,t in zip(mats,means,ts): ax.annotate(f'{v:+.2f}%\\n(t={t:.2f})',(m,v),ha='center',va='bottom')\n"
            "ax.set_xlabel('sleeve maturity (years ≈ duration)'); ax.set_ylabel('mean excess (% / yr)')\n"
            "ax.set_title('Excess scales with duration — it is the term premium, not roll-down alpha'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('excess by maturity:', [(m, round(v,2)) for m,v in zip(mats,means)])"
        ),
        md(
            "> 💡 In plain words: a 2y sleeve earns ~0.6%, a 10y sleeve ~2.7% — the excess is "
            "**proportional to the rate risk you take**, the signature of a term premium, not of a "
            "roll-down edge that should exist independent of duration. There is no maturity, threshold "
            "or cost regime that turns this into a deployable, risk-controlled free lunch: the rarity "
            "that made the 1990–2009 number large (a one-way rate decline) is **exactly** what can't "
            "repeat. Ride the curve and you are, simply, **long rates falling**."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Carry's twin.** [Study 364 — FX-Carry-Trade](../364-fx-carry-trade/): identical "
            "carry-as-premium / free-lunch-busted structure across currencies — positive on average, "
            "violent in the unwind. Reading them together is the cleanest statement that *carry is rent "
            "for risk*.\n"
            "- **Slope as a timer.** [Study 132 — Yield-Curve-Steepener](../132-yield-curve-steepener/) "
            "— does the slope *predict* duration returns, or only pay carry? (Our placebo says the "
            "latter.)\n"
            "- **Neighbouring premia.** [Study 115 — Credit-Spreads](../115-credit-spreads/) and "
            "[Study 119 — Real-Rate-Regime](../119-real-rate-regime/) — the same regime-dependence in "
            "adjacent rates/macro carry.\n\n"
            "*The reproducible core is offline and deterministic; the real curve is genuine CMT yields, "
            "the 4-pillar interpolation and `ret ≈ y0 − D·Δy` are stated approximations. Methods and "
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
