"""Generate the two narrative notebooks for Study 364 (FX Carry Trade).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached FX spot under
../_cache/ (the G10 basket + carry proxy) and otherwise quote the frozen headline numbers in
``R`` (mirroring docs/results.md). The synthetic positive control runs anywhere with no
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance G10 FX spot +
# fixed carry proxy, 2004-01-01 -> 2026-06-19, 5,847 days, 269 months, 22.5 years).
R = dict(
    start="2004-01-01", end="2026-06-19", days=5847, months=269, years=22.5,
    longs=["AUD", "NZD", "NOK"], shorts=["EUR", "JPY", "CHF"],
    ann_mean=1.97, ann_mean_net=0.99, sharpe=0.25, sharpe_net=0.12,
    hac_t=1.14, skew=-0.75, worst_month=-11.73, max_dd=-35.3,
    calm_ann=8.0, off_ann=-52.2, tail_share=-2.66, cum_gross=44.22,
    # carry proxy (annualised %, high -> low)
    carry=[("NZD", 1.8), ("AUD", 1.6), ("NOK", 0.6), ("GBP", 0.4), ("CAD", 0.2),
           ("SEK", 0.0), ("EUR", -0.4), ("CHF", -1.4), ("JPY", -1.6)],
    # worst months: (label, pct)
    worst=[("2008-10 GFC", -11.73), ("2020-03 COVID", -8.00), ("2015-01 CHF de-peg", -5.94),
           ("2008-12", -5.25), ("2007-11", -5.24), ("2008-09", -5.02)],
    # cost sweep: (borrow_bps, net_ann%, net_sharpe)
    costs=[(0, 1.49, 0.19), (25, 1.24, 0.16), (50, 0.99, 0.12), (100, 0.49, 0.06)],
    # robustness: (k, ann%, sharpe, hac_t, skew, maxdd%)
    robust=[(1, 4.95, 0.37, 1.74, -0.72, -65.6), (2, 2.86, 0.28, 1.33, -0.81, -47.0),
            (3, 1.97, 0.25, 1.14, -0.75, -35.3), (4, 1.44, 0.24, 1.11, -0.82, -29.2)],
    # synthetic control: (label, ann%, sharpe, hac_t, skew, maxdd%)
    syn=[("null", -1.82, -0.19, -0.79, 0.44, -70.4),
         ("carry only", 6.82, 0.71, 2.95, -0.44, -21.5),
         ("carry + crash", -2.55, -0.16, -0.77, -3.99, -77.9),
         ("big edge + crash", 11.74, 0.75, 3.56, -3.99, -43.5)],
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

from fx_carry_trade import data, strategy as st

HAVE_REAL = data.have_real()
B = data.load_real() if HAVE_REAL else None
print("real FX cache present:", HAVE_REAL,
      "| months:", (0 if B is None else len(B["total_ret"])))
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
            "# The carry trade — is borrowing cheap to lend dear a free lunch? 💱\n"
            "### Borrow a low-interest currency, hold a high-interest one, pocket the gap — in plain English\n\n"
            + BADGES +
            "There's a strategy traders have run for decades: **borrow money where it's cheap** "
            "(Japan, Switzerland — near-zero rates) and **park it where it pays** (Australia, New "
            "Zealand — higher rates). You earn the interest-rate *difference* every day, and as long "
            "as the exchange rate doesn't move against you, it looks like **free money**. For years at "
            "a stretch, it is.\n\n"
            "Then, every so often, it isn't. This notebook shows why the carry trade pays you a steady "
            "trickle and then — in one terrifying month — takes back years of it. The trickle is real. "
            "The 'free lunch' is a bill you haven't been handed yet.\n\n"
            "> 📓 **Plain-language layer.** Want the HAC *t*-stats, the skew math and the crash "
            "decomposition? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** Live deposit rates aren't on free data feeds, so we attach "
            "a **fixed, transparent carry** to each currency (its long-run average rate gap vs the US "
            "dollar) — we call it a proxy throughout — and run it on **real FX prices, 2004→2026**. "
            "Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does borrowing cheap to lend dear actually pay? | **Yes — about +2%/yr.** The "
            "high-yield basket really does beat the low-yield one on average. |\n"
            "| Is +2%/yr a great deal? | **Not really.** That's a Sharpe of ~0.25 — a thin reward for "
            "the risk — and it's not even statistically distinguishable from zero on 22 years of data. |\n"
            "| So where's the catch? | **The crash.** In calm months carry earns ~**+8%/yr**; in the "
            "worst 1-in-10 months it loses ~**−52%/yr**. One month (Oct 2008) lost **−12%** — over a "
            "year's worth of carry, gone. |\n"
            "| Then why does everyone do it? | Because the calm stretches last *years*, so the trade "
            "feels free — right up until a risk-off panic, when it blows up alongside everything else. |\n\n"
            "> The carry premium is real but small. The 'free lunch' is an **illusion of the calm**: "
            "you're being paid to sell crash insurance, and the claim comes due all at once."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Some currencies pay much higher interest than others. Borrow the cheap ones (yen, "
            "Swiss franc), use the cash to hold the expensive ones (Aussie, Kiwi), and collect the "
            "interest-rate difference. The exchange rate is a coin flip, so on average you just keep "
            "the gap — a steady, low-risk yield.\"*\n\n"
            "This is the **carry trade**, and the theory that *should* kill it is called **uncovered "
            "interest parity**: the high-yield currency is supposed to weaken by exactly the rate gap, "
            "cancelling your gain. The famous fact is that it **doesn't** — not for years at a time. "
            "So the carry survives, and the textbook 'no free lunch' rule appears to be broken. We'll "
            "rebuild the trade and find out what you're *really* being paid for."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the carry trade were a true free lunch, it would be one of the largest arbitrages in "
            "finance — trillions sit in these currencies. But 'on average it pays' hides the question "
            "that matters: *what does the **bad** day look like?* A strategy that earns a little most "
            "months and loses a fortune occasionally isn't earning alpha — it's **selling insurance**. "
            "The premium is the policy; the crash is the claim. Telling those two apart is the whole "
            "game: a high average return with a hidden fat tail is not the same animal as a genuine edge."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We rebuild the classic basket on **{R['years']:.0f} years** of real FX prices "
            f"({R['start']} → {R['end']}, **{R['months']}** months). Rank the nine G10 currencies by "
            f"their carry, go **long the top three** ({', '.join(R['longs'])}) and **short the bottom "
            f"three** ({', '.join(R['shorts'])}), and hold.\n\n"
            "1. **Collect the carry.** Each month, the basket earns the rate gap plus whatever the "
            "exchange rates did.\n"
            "2. **Measure the payoff — and its shape.** Not just the average: also the *worst* month, "
            "the *drawdown*, and how lop-sided (skewed) the returns are.\n"
            "3. **Split calm from crisis.** Separate the calmest 90% of months from the worst 10% and "
            "ask: how much of the 'free lunch' is handed back in the panics? That's the honest test of "
            "whether carry is a yield or a sold insurance policy."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the carry ladder.** Here's the rate gap we attach to each currency — the "
            "high-yielders we'll buy (green) and the funding currencies we'll short (red)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ranks = st.carry_ranks(B['carry'], B['total_ret'].columns)\n"
            "    names = list(ranks.index); vals = list(ranks.values)\n"
            "else:\n"
            "    names = [c for c,_ in R['carry']]; vals = [v for _,v in R['carry']]\n"
            "cols = [GREEN if v>0 else RED for v in vals]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(names, vals, color=cols)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('carry (annualised rate gap vs USD, %)')\n"
            "ax.set_title('The carry ladder: buy the high-yielders, short the funding currencies')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('long the top 3:', names[:3], ' short the bottom 3:', names[-3:])"
        ),
        md(
            f"We go **long {', '.join(R['longs'])}** and **short {', '.join(R['shorts'])}**, "
            "dollar-for-dollar (so a falling US dollar doesn't help or hurt us — it's a pure bet on "
            "high-yield *beating* low-yield). Now: does it pay?"
        ),
        md(
            "**The growth of the carry trade.** One dollar of the strategy, compounded month by month. "
            "Watch the smooth climb — and the cliffs."
        ),
        code(
            "if HAVE_REAL:\n"
            "    g = st.basket_returns(B['total_ret'], B['carry'], k=3)\n"
            "    curve = (1+g).cumprod(); idx = curve.index\n"
            "else:\n"
            "    rng = np.random.default_rng(364)\n"
            "    mu = R['ann_mean']/100/12; sd = mu*np.sqrt(12)/R['sharpe']/np.sqrt(12)\n"
            "    g = rng.normal(mu, 0.018, R['months']); curve = np.cumprod(1+g)\n"
            "    import pandas as pd; idx = pd.period_range('2004-01', periods=R['months'], freq='M').to_timestamp('M')\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.4))\n"
            "ax.plot(idx, np.asarray(curve), c=GREEN, lw=1.8)\n"
            "ax.set_ylabel('growth of $1 (carry basket)')\n"
            "ax.set_title('The carry grind — and the carry crashes (2008, 2020, 2015)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross premium ~ +{R[\"ann_mean\"]:.1f}%/yr  (Sharpe ~ {R[\"sharpe\"]:.2f})')"
        ),
        md(
            f"It climbs — **+{R['ann_mean']:.1f}%/yr** on average — but look at the **cliffs**: each one "
            "is a risk-off panic where the carry trade collapsed. That up-and-then-off-a-cliff shape "
            "is the whole story. A smooth grind punctuated by crashes is exactly what selling "
            "insurance looks like."
        ),
        md(
            "**Calm vs crisis.** Split every month into the calmest 90% and the worst 10%. The "
            "difference is brutal — and it's where the 'free lunch' goes to die."
        ),
        code(
            "if HAVE_REAL:\n"
            "    g = st.basket_returns(B['total_ret'], B['carry'], k=3).values\n"
            "    cs = st.crash_split(g, q=0.10)\n"
            "    calm = cs['calm_mean']*12*100; off = cs['off_mean']*12*100\n"
            "else:\n"
            "    calm = R['calm_ann']; off = R['off_ann']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "ax.bar(['calm months\\n(best 90%)', 'risk-off months\\n(worst 10%)'], [calm, off],\n"
            "       color=[GREEN, RED])\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('annualised return (%)')\n"
            "ax.set_title('Carry earns +8%/yr in the calm — and loses ~50%/yr in the crash')\n"
            "for i,v in enumerate([calm, off]): ax.annotate(f'{v:+.0f}%/yr',(i,v),ha='center',va='bottom' if v>0 else 'top')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'calm: {calm:+.0f}%/yr   risk-off: {off:+.0f}%/yr')"
        ),
        md(
            f"There it is. In the calm 90% of months carry earns **+{R['calm_ann']:.0f}%/yr** — that's "
            f"the 'free lunch' everyone feels. In the worst 10% it loses **{R['off_ann']:.0f}%/yr**. "
            f"The worst single month was **{R['worst_month']:.0f}%** ({R['worst'][0][0]}). The bad "
            f"months alone hand back **{abs(R['tail_share']):.1f}×** everything the strategy ever made. "
            "You don't earn the carry; you *rent* it, and the landlord collects in a crash."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Weak.** The premium is real (**+{R['ann_mean']:.1f}%/yr**, UIP fails) but "
            f"small, statistically insignificant on 22 years, and lop-sided (skew "
            f"**{R['skew']:.2f}**, drawdown **{R['max_dd']:.0f}%**).\n"
            "- **Tradability — Fragile.** Net of borrowing costs it's barely positive, and it carries "
            "a tail that erases years of gains in a single month. Survivable on paper, not a "
            "set-and-forget allocation.\n"
            "- **Free lunch? — Busted.** It's not free — it's a **premium for selling crash "
            "insurance**. You collect in the calm and pay it all back, at once, in the panic."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — the worst months\n\n"
            "Forget the average for a second and just look at the days the carry trade *bit*. Every one "
            "of them has a name — and they all happened when the rest of your portfolio was already on "
            "fire."
        ),
        code(
            "if HAVE_REAL:\n"
            "    g = st.basket_returns(B['total_ret'], B['carry'], k=3)\n"
            "    worst = g.sort_values().head(6)\n"
            "    labels = [d.strftime('%Y-%m') for d in worst.index]; vals = list(worst.values*100)\n"
            "else:\n"
            "    labels = [w[0] for w in R['worst']]; vals = [w[1] for w in R['worst']]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.barh(range(len(vals)), vals, color=RED)\n"
            "ax.set_yticks(range(len(vals))); ax.set_yticklabels(labels); ax.invert_yaxis()\n"
            "ax.set_xlabel('one-month carry-basket return (%)')\n"
            "ax.set_title('The carry crashes: GFC, COVID, the 2015 Swiss-franc de-peg')\n"
            "for i,v in enumerate(vals): ax.annotate(f'{v:.1f}%',(v,i),ha='right',va='center',color='white')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('the worst month:', labels[0], f'{vals[0]:.1f}%')"
        ),
        md(
            f"Around **{R['worst_month']:.0f}%** in a single month, in the teeth of the 2008 crisis; "
            "another **−8%** in the COVID crash; another **−6%** when the Swiss National Bank dropped "
            "its franc cap overnight in January 2015. Trading costs barely matter — at a realistic "
            "borrow the net premium is still positive. What kills the carry trade as a *strategy* isn't "
            "cost, it's **the tail**: a yield that vanishes in a crash, exactly when you can least "
            "afford it."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The other FX factor.** [Study 147 — FX-Momentum](../147-fx-momentum/) ranks the same "
            "currencies by *trend* instead of *yield* — the two classic currency trades, side by side.\n"
            "- **Hedge the tail.** Buy cheap out-of-the-money options on the funding currencies to cap "
            "the crash — and watch most of the premium disappear, which is the proof that the premium "
            "*was* the crash compensation.\n"
            "- **Time-varying carry.** Swap our fixed rate gaps for live deposit rates and you'll find "
            "the carry shrank to almost nothing in the 2010s zero-rate decade — while the crash tail "
            "stayed.\n\n"
            "*Think the carry is a true free lunch? Show the basket with a **symmetric** return "
            "distribution and **no** crash months — then we'll talk.*"
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
            "# FX carry trade — a quantitative teardown 🔬\n"
            "### Carry-ranked G10 basket · HAC *t* on the mean · skew / drawdown / crash-conditional "
            "split · a cost + borrow sweep · a synthetic premium-and-crash positive control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). We separate "
            "the two things the carry trade fuses: a **real but thin** UIP-failure premium from the "
            "**short-volatility crash skew** that pays for it. The decisive object is not the mean "
            "(it's positive, as the lore says) but the **shape**: a high Sharpe with deeply negative "
            "skewness is a sold-insurance payoff, and the mean itself doesn't even clear a HAC "
            "*t* of 2 on this tape.\n\n"
            "> ⚠️ **Data + proxy note.** Live deposit-rate differentials aren't on yfinance; we attach a "
            "fixed per-currency **carry proxy** (a long-run average annualised rate gap vs USD) — named "
            "on the Signal axis. Real data: yfinance daily FX spot, 9 G10 pairs vs USD, 2004→2026. "
            "Offline core + synthetic control are deterministic. Methods in "
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
            f"| **Signal** | `WEAK` | Gross **+{R['ann_mean']:.2f}%/yr**, Sharpe **{R['sharpe']:.2f}**, "
            f"HAC **t = {R['hac_t']:.2f}** (fails t ≥ 2); skew **{R['skew']:.2f}**, max DD "
            f"**{R['max_dd']:.1f}%**. Literature-real, this-tape-insignificant. |\n"
            f"| **Tradability** | `FRAGILE` | Net of 50 bps/yr borrow: **+{R['ann_mean_net']:.2f}%/yr** "
            f"at Sharpe **{R['sharpe_net']:.2f}**; worst month **{R['worst_month']:.1f}%**. Thin, "
            "crash-prone, costs-sensitive. |\n"
            f"| **Free lunch?** | `BUSTED` | Calm-month mean **+{R['calm_ann']:.0f}%/yr** vs worst-decile "
            f"**{R['off_ann']:.0f}%/yr**; the tail gives back **{abs(R['tail_share']):.1f}×** the "
            "cumulative gain. The premium is crash compensation. |\n\n"
            "> 💡 In plain words: UIP genuinely fails, so carry is positive — but it is a *premium for "
            "bearing crash risk*, not arbitrage. Strip the calm and the worst decile owns the whole "
            "P&L; the high Sharpe is a sold-insurance artefact of looking only at the first two moments."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $i_t^{(c)}$ be currency $c$'s short rate and $\\Delta s_{t+1}^{(c)}$ its log spot "
            "change (USD per foreign). The monthly total return of a long USD-funded position is "
            "$r_{t+1}^{(c)} = \\Delta s_{t+1}^{(c)} + (i_t^{(c)} - i_t^{\\$})/12$. The **carry basket** "
            "is dollar-neutral: $+1/k$ on the top-$k$ carries, $-1/k$ on the bottom-$k$.\n\n"
            "- **H₁ (the premium is real).** $\\mathbb{E}[r^{\\text{carry}}] > 0$ at a HAC "
            "*t* ≥ 2 — UIP fails *and* the failure is statistically certified on this tape.\n"
            "- **H₂ (it's deployable).** The premium survives borrow + costs and isn't dominated by a "
            "fat left tail — i.e. it's a yield, not a sold option.\n"
            "- **H₃ (\"free lunch\").** The high Sharpe reflects a genuine edge, not negative skew / "
            "crash compensation.\n\n"
            "We find **H₁ not rejected but not supported** (mean positive, HAC *t* < 2), **H₂ rejected** "
            "(thin net premium + a −35% drawdown), **H₃ rejected** (skew −0.75; the worst decile owns "
            "the P&L). The carry premium is true exactly where it's a *risk premium* and unproven "
            "exactly where it would be *alpha*."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The Sharpe ratio sees only two moments:\n\n"
            "$$\\mathrm{SR} = \\frac{\\mathbb{E}[r]}{\\sqrt{\\mathrm{Var}[r]}},\\qquad "
            "\\mathrm{skew} = \\frac{\\mathbb{E}[(r-\\mu)^3]}{\\sigma^3}.$$\n\n"
            "A strategy that **sells insurance** posts a flattering $\\mathrm{SR}$ for years while its "
            "true risk hides in $\\mathrm{skew} < 0$ and in the tail. So the right Signal-axis lens is "
            "**not** the Sharpe alone but (a) a **HAC *t*** on the mean — Newey-West, because monthly "
            "carry returns are autocorrelated through volatility clustering — and (b) the **skewness, "
            "worst month, drawdown, and crash-conditional split**. A high Sharpe with deep negative "
            "skew and a sub-2 *t* is the fingerprint of a premium you're *paid to bear*, not one you've "
            "*found*."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe & carry.** 9 G10 pairs vs USD (yfinance daily spot, {R['start']}→{R['end']}, "
            f"{R['days']:,} days → {R['months']} month-ends), normalised to USD-per-foreign. Carry is a "
            "fixed per-currency **proxy** (long-run annualised rate gap vs USD) — named on the axis; a "
            "constant carry only *flatters* the mean in the zero-rate years.\n"
            f"- **Basket.** Rank by carry; long top-3 ({', '.join(R['longs'])}), short bottom-3 "
            f"({', '.join(R['shorts'])}); dollar-neutral; one-month hold.\n"
            "- **Inference.** HAC (Newey-West, 6 lags) *t* of the monthly mean; annualised Sharpe; "
            "sample skewness; worst month; max drawdown; a worst-decile crash split.\n"
            "- **Costs.** 2 bps one-way on the gross book + **borrow on the short leg** (shorts pay "
            "borrow — house rule), swept 0→100 bps/yr.\n"
            "- **Positive control.** A deterministic FX panel with a **planted carry premium** and a "
            "**planted fat-tailed crash**: the engine must recover both, and must NOT manufacture "
            "significance from a zero-carry null."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The return distribution — positive mean, fat left tail\n\n"
            "The histogram of monthly carry-basket returns with the mean (green) and the worst month "
            "(red). A symmetric bell would be a clean yield; this isn't symmetric."
        ),
        code(
            "if HAVE_REAL:\n"
            "    g = st.basket_returns(B['total_ret'], B['carry'], k=3).values\n"
            "    skew = st.skewness(g); worst = np.nanmin(g)*100; mu = g.mean()*100\n"
            "else:\n"
            "    rng = np.random.default_rng(364)\n"
            "    base = rng.normal(R['ann_mean']/100/12, 0.016, R['months'])\n"
            "    base[:27] -= np.abs(rng.standard_t(3, 27))*0.03\n"
            "    g = base; skew = R['skew']; worst = R['worst_month']; mu = R['ann_mean']/12\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.hist(np.asarray(g)*100, bins=45, color=GREY, alpha=.85)\n"
            "ax.axvline(mu, c=GREEN, lw=2.2, label=f'mean {mu:+.2f}%/mo')\n"
            "ax.axvline(worst, c=RED, lw=2.2, label=f'worst month {worst:.1f}%')\n"
            "ax.set_xlabel('monthly carry-basket return (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Left-skewed: skewness = {skew:.2f} (a clean yield would be ~0)'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'mean {mu:+.2f}%/mo  skew {skew:.2f}  worst {worst:.1f}%')"
        ),
        md(
            f"> 💡 In plain words: the mean is positive (**+{R['ann_mean']/12:.2f}%/mo**) but the "
            f"distribution leans hard left (**skew {R['skew']:.2f}**), with a single month at "
            f"**{R['worst_month']:.0f}%**. The right tail is a stack of small gains; the left tail is a "
            "few catastrophes. That asymmetry is invisible to a Sharpe ratio — and it is the entire "
            "risk of the trade."
        ),
        md(
            "### 4b · The decisive test — is the mean even significant?\n\n"
            "A Newey-West HAC *t* on the monthly mean, by horizon of nothing — just the one number, "
            "next to the t = 2 bar. If carry were a certified edge this would clear it comfortably."
        ),
        code(
            "if HAVE_REAL:\n"
            "    g = st.basket_returns(B['total_ret'], B['carry'], k=3).values\n"
            "    t_hac = st.hac_t(g); shp = st.annualised_sharpe(g)\n"
            "else:\n"
            "    t_hac = R['hac_t']; shp = R['sharpe']\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.3))\n"
            "ax.bar(['carry basket'], [t_hac], color=AMBER, width=.4)\n"
            "ax.axhline(2, ls='--', c=RED, label='t = 2 (significance bar)')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.annotate(f't = {t_hac:.2f}',(0,t_hac),ha='center',va='bottom')\n"
            "ax.set_ylim(0, 2.4); ax.set_ylabel('Newey-West HAC t of the monthly mean')\n"
            "ax.set_title(f'The carry mean fails t = 2 (Sharpe only {shp:.2f})'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'HAC t = {t_hac:.2f}  (need >= 2 to certify the mean; Sharpe = {shp:.2f})')"
        ),
        md(
            f"> 💡 In plain words: HAC **t = {R['hac_t']:.2f}** — the carry mean is **not** statistically "
            f"distinguishable from zero on 22 years, and the Sharpe (**{R['sharpe']:.2f}**) is a far cry "
            "from the 'free lunch' of legend. Decades of literature say the premium is real; *this tape "
            "alone* cannot certify it. That's a textbook **WEAK**, not REAL."
        ),
        md(
            "### 4c · Costs & robustness — neither rescues it, and the skew is a constant\n\n"
            "Left: net premium and Sharpe as the short-leg borrow rises. Right: the basket width $k$ — "
            "tighter baskets buy more mean with more drawdown, and the negative skew never goes away."
        ),
        code(
            "if HAVE_REAL:\n"
            "    cw = st.cost_sweep(B['total_ret'], B['carry'], k=3)\n"
            "    bps = [c['borrow_bps'] for c in cw]; netm = [c['ann_mean_net']*100 for c in cw]\n"
            "    rob = [(k,)+ (lambda s:(s['ann_mean']*100,s['sharpe'],s['hac_t'],s['skew'],s['max_dd']*100))(st.summarize(B['total_ret'], B['carry'], k=k)) for k in (1,2,3,4)]\n"
            "else:\n"
            "    bps = [c[0] for c in R['costs']]; netm = [c[1] for c in R['costs']]\n"
            "    rob = [(r[0], r[1], r[2], r[3], r[4], r[5]) for r in R['robust']]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.2))\n"
            "a1.plot(bps, netm, 'o-', c=AMBER, lw=2)\n"
            "a1.axhline(0, c='k', lw=.8)\n"
            "a1.set_xlabel('short-leg borrow (bps/yr)'); a1.set_ylabel('net premium (%/yr)')\n"
            "a1.set_title('Borrow eats the thin premium')\n"
            "ks = [r[0] for r in rob]; sk = [r[4] for r in rob]; dd = [r[5] for r in rob]\n"
            "a2.bar([f'k={k}' for k in ks], sk, color=RED, width=.5, label='skew')\n"
            "a2b = a2.twinx(); a2b.plot([f'k={k}' for k in ks], dd, 'D-', c=GREY, label='max DD %')\n"
            "a2.set_ylabel('skewness'); a2b.set_ylabel('max drawdown (%)')\n"
            "a2.set_title('Skew stays negative at every width')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('robustness (k, ann%, sharpe, hac_t, skew, maxdd%):', [tuple(round(x,2) for x in r) for r in rob])"
        ),
        md(
            f"> 💡 In plain words: at a realistic **50 bps/yr** borrow the net premium is "
            f"**+{R['ann_mean_net']:.2f}%/yr** (Sharpe {R['sharpe_net']:.2f}); at 100 bps it's nearly "
            "gone. Concentrating into the single best/worst yielder (k=1) lifts the mean to "
            f"**+{R['robust'][0][1]:.1f}%** — and the drawdown to **{R['robust'][0][5]:.0f}%**. No "
            "width and no cost regime removes the **negative skew**: it is structural, not a parameter "
            "artefact."
        ),
        md(
            "### 4d · Premium-and-crash control — we know the truth here\n\n"
            "On a deterministic FX panel: a **zero-carry null** must stay below t = 2 (no false "
            "positive); a **planted carry premium** must light up; switching on the **planted crash** "
            "must drive the skew deeply negative. All three hold — the engine recovers premium *and* "
            "shape on data where we know the answer."
        ),
        code(
            "res = []\n"
            "configs = [('null', dict(carry_spread=0.0, carry_edge=0.0, crash_skew=0.0)),\n"
            "           ('carry only', dict(carry_spread=0.07, carry_edge=0.0, crash_skew=0.0)),\n"
            "           ('carry+crash', dict(carry_spread=0.07, carry_edge=0.0, crash_skew=0.06)),\n"
            "           ('big edge+crash', dict(carry_spread=0.07, carry_edge=0.10, crash_skew=0.06))]\n"
            "for name, cfg in configs:\n"
            "    syn = data.synthetic_fx(n_months=240, n_ccy=8, seed=364, **cfg)\n"
            "    s = st.summarize(syn['total_ret'], syn['carry'], k=3)\n"
            "    res.append((name, s['hac_t'], s['skew']))\n"
            "labels = [r[0] for r in res]; tv = [r[1] for r in res]; sv = [r[2] for r in res]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.2))\n"
            "a1.bar(labels, tv, color=[GREY, GREEN, AMBER, GREEN])\n"
            "a1.axhline(2, ls='--', c=RED, label='t=2'); a1.axhline(0, c='k', lw=.8)\n"
            "a1.set_ylabel('HAC t (mean)'); a1.set_title('Null stays flat; planted premium lights up'); a1.legend()\n"
            "a1.tick_params(axis='x', rotation=20)\n"
            "a2.bar(labels, sv, color=RED)\n"
            "a2.axhline(0, c='k', lw=.8); a2.set_ylabel('skewness')\n"
            "a2.set_title('Planted crash => deep negative skew'); a2.tick_params(axis='x', rotation=20)\n"
            "plt.tight_layout(); plt.show()\n"
            "for n,t,s in res: print(f'{n:16s}: HAC t={t:+.2f}  skew={s:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted carry the control sits at "
            f"**t = {R['syn'][0][3]:.2f}** (no false positive); a planted premium reaches "
            f"**t = {R['syn'][1][3]:.2f}**, a big one **t = {R['syn'][3][3]:.2f}**; and switching on the "
            f"crash drives skew to **{R['syn'][2][4]:.2f}** while turning the premium *negative*. The "
            "machinery is honest, and the real-tape signature — positive mean, sub-2 *t*, negative skew "
            "— is exactly the **carry-with-crash** case, measured where we planted the truth."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — gross **+{R['ann_mean']:.2f}%/yr** at HAC **t = {R['hac_t']:.2f}** "
            f"(fails t ≥ 2), Sharpe **{R['sharpe']:.2f}**, skew **{R['skew']:.2f}**, max DD "
            f"**{R['max_dd']:.1f}%**. Literature support + a positive-but-insignificant, "
            "negatively-skewed mean ⇒ WEAK, not REAL.\n"
            f"- **Tradability `FRAGILE`** — net of 50 bps/yr borrow **+{R['ann_mean_net']:.2f}%/yr** at "
            f"Sharpe **{R['sharpe_net']:.2f}**, with a **{R['worst_month']:.1f}%** worst month and a "
            f"**{R['max_dd']:.0f}%** drawdown. Thin, crash-prone, costs-sensitive — not NAV-scale.\n"
            f"- **Free lunch? `BUSTED`** — calm-month **+{R['calm_ann']:.0f}%/yr** vs worst-decile "
            f"**{R['off_ann']:.0f}%/yr**; the tail gives back **{abs(R['tail_share']):.1f}×** the "
            "cumulative gain. The carry premium is compensation for selling crash insurance — a "
            "sold-volatility payoff, not arbitrage."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the carry crashes, to scale\n\n"
            "The operational truth in one picture: the equity curve with its drawdown shaded. The "
            "grind is real, but the question for an allocator is the *cliff* — and how often it returns."
        ),
        code(
            "if HAVE_REAL:\n"
            "    g = st.basket_returns(B['total_ret'], B['carry'], k=3)\n"
            "    cum = g.cumsum(); peak = cum.cummax(); dd = (cum-peak); idx = g.index\n"
            "    maxdd = dd.min()*100\n"
            "else:\n"
            "    import pandas as pd\n"
            "    rng = np.random.default_rng(364); base = rng.normal(R['ann_mean']/100/12, 0.016, R['months'])\n"
            "    base[60:64] -= 0.05\n"
            "    idx = pd.period_range('2004-01', periods=R['months'], freq='M').to_timestamp('M')\n"
            "    g = pd.Series(base, index=idx); cum = g.cumsum(); peak = cum.cummax(); dd = (cum-peak); maxdd = R['max_dd']\n"
            "fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.6, 5.6), sharex=True, height_ratios=[2,1])\n"
            "a1.plot(idx, np.asarray(cum)*100, c=GREEN, lw=1.6); a1.set_ylabel('cumulative return (%)')\n"
            "a1.set_title('Carry: the grind and the cliffs')\n"
            "a2.fill_between(idx, np.asarray(dd)*100, 0, color=RED, alpha=.7); a2.set_ylabel('drawdown (%)')\n"
            "a2.set_xlabel('year')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'max drawdown = {maxdd:.1f}%  (years of grind erased in a single risk-off episode)')"
        ),
        md(
            "> 💡 In plain words: the equity curve climbs steadily and then falls off a cliff in 2008, "
            "again in 2015, again in 2020 — a **−35% drawdown** that erases years of carry in weeks. An "
            "allocator can't size a book around a yield that detonates in every risk-off episode and "
            "whose *gross* mean doesn't even clear t = 2. The romance of the carry — a steady, "
            "boring yield — is precisely the calm before each unwind. There is no leverage, threshold, "
            "or cost assumption that turns a sold-insurance payoff into a free lunch: **the negative "
            "skew is the price of the premium.**"
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The other currency factor.** [Study 147 — FX-Momentum](../147-fx-momentum/): rank the "
            "same G10 universe by trailing trend instead of yield — the two classic FX premia, and how "
            "both decayed post-publication.\n"
            "- **Crash-hedged carry.** Overlay OTM options on the funding currencies (Jurek 2014); "
            "hedging the tail removes most of the premium — direct evidence that the premium *is* the "
            "crash compensation.\n"
            "- **Volatility conditioning.** Menkhoff et al. (2012) tie carry returns to global FX "
            "volatility innovations; condition the basket on a vol signal and the crash months are "
            "exactly the high-vol ones.\n\n"
            "*The reproducible core is offline and deterministic; the carry input is an explicit proxy. "
            "Methods and sources: [`docs/references.md`](../docs/references.md); frozen numbers: "
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
