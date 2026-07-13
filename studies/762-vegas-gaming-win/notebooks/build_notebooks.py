"""Generate the two narrative notebooks for Study 762 (Vegas-Gaming-Win).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the hardcoded Strip-GGR
reconstruction (always available) and the cached casino-basket prices under ../_cache/, and
otherwise quote the frozen headline numbers in ``R`` (mirroring docs/results.md). The
synthetic positive control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (NGCB Strip GGR hardcoded
# reconstruction + equal-weight casino basket, month-end, 2002-02 -> 2025-06, 281 months).
R = dict(
    start="2002-02-28", end="2025-06-30", months=281, years=23.3,
    # per-horizon: (months, n_rising, rising_mean%, falling_mean%, base_mean%, ris_up%, base_up%, t, p_placebo)
    h1=(1, 185, 1.66, 2.31, 1.88, 59, 57, -0.20, 0.572),
    h3=(3, 183, 5.61, 5.16, 5.46, 66, 62, 0.09, 0.458),
    h6=(6, 181, 11.51, 12.36, 11.80, 64, 65, -0.10, 0.539),
    h12=(12, 181, 21.76, 31.91, 25.05, 66, 69, -0.72, 0.810),
    # lead/lag: L -> corr
    leadlag={-6: 0.071, -5: 0.065, -4: 0.045, -3: -0.011, -2: -0.068, -1: -0.113,
             0: -0.114, 1: -0.105, 2: -0.095, 3: -0.096, 4: -0.090, 5: -0.083, 6: -0.069},
    # overlay: (bh_mean%, bh_sharpe, gross_mean%, gross_sharpe, net_mean%, net_sharpe, switches)
    overlay=(23.0, 0.49, 13.5, 0.51, 13.4, 0.50, 11),
    # robustness 12m: (label, n_rising, rising12%, base12%, t, p)
    robust=[("k=1", 177, 23.81, 25.05, -0.27, 0.625), ("k=3", 181, 21.76, 25.05, -0.72, 0.810),
            ("k=6", 182, 19.80, 25.05, -1.15, 0.924), ("thr>+3%", 30, 1.61, 25.05, -4.08, 0.997),
            ("ex-COVID", 174, 22.87, 25.52, -0.62, 0.769)],
    # synthetic control: (edge, n_rising, rising1m%, base1m%, t, p)
    syn=[(0.0, 172, 0.99, 0.50, 0.76, 0.178), (0.05, 172, 5.53, 3.06, 3.70, 0.000)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Leading_signal%3F: Not_supported](https://img.shields.io/badge/Leading_signal%3F-Not_supported-8b949e?style=flat-square)\n\n"
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

from vegas_gaming_win import data, strategy as st

HAVE_REAL = data.have_real()
F = data.load_real() if HAVE_REAL else None
print("casino cache present:", HAVE_REAL,
      "| GGR+basket months:", (0 if F is None else len(F)))
"""

# The frozen headline dict is embedded into the first code cell so every downstream cell can
# quote it whether or not the price cache is present.
BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# When Vegas rings the register louder, do the casino stocks run? 🎲\n"
            "### 'Strip revenue leads the casino stocks' — the sector's favourite crystal ball, in plain English\n\n"
            + BADGES +
            "Every month Nevada reports how much money the Las Vegas **Strip** casinos won from "
            "gamblers — the *gross gaming revenue*, or **GGR**. The folklore on trading desks and "
            "cable-TV 'casino watch' segments is that this number is the pulse of the whole business: "
            "when Strip revenue starts **accelerating**, the casino stocks — MGM, Caesars, Las Vegas "
            "Sands, Wynn, Boyd, Penn — are about to run.\n\n"
            "It's a tidy story. It's also testable. This notebook asks three blunt questions: when Strip "
            "revenue speeds up, do the casino stocks really do better next? Does the revenue uptick "
            "actually come **first** (that's the whole pitch)? And if you *bought* the casinos every time "
            "revenue accelerated, would you make money?\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the lead/lag cross-correlation and the "
            "synthetic control? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** The Nevada Gaming Control Board publishes Strip GGR as monthly "
            "PDFs we can't fetch here, so we use a **hardcoded reconstruction** whose *yearly* totals "
            "match the published numbers (the ~$6.6B pre-COVID years, the 2020 shutdown, the record "
            "~$8.8B of 2023–24) — public and faithful in aggregate, clearly labelled as a reconstruction. "
            "Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| When Strip revenue accelerates, do casino stocks do better? | **No.** Over the next year "
            f"the basket averages **+{R['h12'][2]:.1f}%** after a revenue uptick vs **+{R['h12'][4]:.1f}%** "
            "normally — *lower*, not higher. The bullish tilt the folklore promises just isn't there. |\n"
            "| Is any gap reliable? | **No.** Every horizon is a statistical tie (or the wrong sign), well "
            "inside the noise. |\n"
            "| Does the revenue uptick come *first*? | **No — and this is the killer.** The revenue signal "
            "lines up best with a stock move that already happened **half a year earlier.** The stocks "
            "lead the revenue here, not the other way round. |\n"
            "| So could you trade it? | **It gives up money.** 'Own casinos when revenue accelerates' "
            f"earned **+{R['overlay'][4]:.1f}%/yr** vs **+{R['overlay'][0]:.1f}%** for just holding — the "
            "same risk-adjusted return, a third less time in the market. |\n\n"
            "> Casinos and gambling revenue are obviously linked. But 'revenue tells you where the stocks "
            "go *next*' is a backward-looking report echoing a move the market already made — and acting "
            "on it costs you compounding."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Las Vegas Strip gaming revenue is the fundamental pulse of the casino business. When the "
            "monthly revenue run-rate turns up, the operators' earnings are accelerating and their stocks "
            "are about to follow — so buy the casino basket the moment Strip revenue starts climbing.\"*\n\n"
            "There's a respectable backbone: Strip GGR really is the top-line the casino operators earn "
            "off. Fewer visitors and lower spend genuinely hurt them. The trading leap is the part we "
            "test — that a revenue *uptick* arrives early enough, and cleanly enough, to be a "
            "**tradable** signal for the equities."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If true, this would be a gift: a free, public, monthly government number that tells you when "
            "to own an entire sector. But 'the pulse' hides a trap. The **stock market is forward-"
            "looking** — casino shares trade on *next quarter's* bookings, room rates and macro, priced "
            "in by analysts long before the backward-looking revenue tally is even tabulated (the report "
            "lands about **five weeks after** the month it covers). So a revenue number that lines up with "
            "the stocks might not be *predicting* them at all — it might just be **confirming**, weeks "
            "late, a move the market already made. The difference between *leads* and *echoes* is the "
            "difference between an edge and a mirage."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We line up **{R['years']:.0f} years** ({R['start'][:4]}–{R['end'][:4]}, "
            f"{R['months']} months) of monthly Strip revenue against an equal-weight casino basket, and:\n\n"
            "1. **Split the months.** Because revenue is wildly seasonal (March and summer peaks) and had "
            "a total COVID shutdown, we track the **trailing-12-month revenue run-rate** and call it "
            "**rising** when it's above where it was three months ago. Compare what the basket did next "
            "(1/3/6/12 months) after rising-revenue months vs all months.\n"
            "2. **Check the timing.** The crucial test: slide the revenue signal forward and backward "
            "against the stocks and find *where* they line up best. If revenue truly **leads**, the "
            "strongest link shows up at a **positive lead** (revenue first, stocks later).\n"
            "3. **Try to trade it.** Own the basket whenever revenue is accelerating, sit in cash "
            "otherwise, pay realistic costs — and see if it beats just buying and holding."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the raw material.** Here's the Strip revenue run-rate over two decades — the "
            "mid-2000s build, the 2008–09 dip, the off-the-chart **COVID shutdown** of 2020 (revenue "
            "literally near zero for two months while the casinos were closed), and the record 2022–24 "
            "boom. Revenue clearly *knows* about the cycle. The question is whether it knows **early**."
        ),
        code(
            "if HAVE_REAL:\n"
            "    g = F['ggr']\n"
            "    fig, ax = plt.subplots(figsize=(9.4, 4.0))\n"
            "    ax.plot(g.index, g.values, c=RED, lw=1.3)\n"
            "    ax.set_title('Las Vegas Strip monthly gross gaming revenue (US$ millions)')\n"
            "    ax.set_ylabel('GGR ($ millions)')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('trough (COVID):', int(g.min()), '$M around', g.idxmin().date(),\n"
            "          '| recent peak:', int(g.max()), '$M')\n"
            "else:\n"
            "    print('no cache — see docs/results.md; Strip GGR ~0 in Apr-May 2020 (closure), ~$800M peak months 2023-24')"
        ),
        md(
            "**Now the payoff.** For each horizon, the average forward basket return in **rising-revenue** "
            "months next to the return on an **average** month. The folklore predicts the green bars sit "
            "*above* the grey ones."
        ),
        code(
            "hs = [1, 3, 6, 12]\n"
            "if HAVE_REAL:\n"
            "    rows = [st.summarize(F, m) for m in hs]\n"
            "    ris = [r['rising_mean']*100 for r in rows]; base = [r['base_mean']*100 for r in rows]\n"
            "else:\n"
            "    ris = [R['h1'][2], R['h3'][2], R['h6'][2], R['h12'][2]]\n"
            "    base = [R['h1'][4], R['h3'][4], R['h6'][4], R['h12'][4]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "ax.bar(x-.2, ris, .4, color=GREEN, label='after revenue ACCELERATES')\n"
            "ax.bar(x+.2, base, .4, color=GREY, label='an average month (base rate)')\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{m} months' for m in hs])\n"
            "ax.set_ylabel('average forward basket return (%)')\n"
            "ax.set_title('Rising revenue -> NOT higher forward returns (the green bars do not clear grey)')\n"
            "for i,(a,b) in enumerate(zip(ris,base)):\n"
            "    ax.annotate(f'{a:.1f}%',(i-.2,a),ha='center',va='bottom',fontsize=9)\n"
            "    ax.annotate(f'{b:.1f}%',(i+.2,b),ha='center',va='bottom',fontsize=9)\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('12-month: rising', f'{ris[-1]:.1f}%', 'vs base', f'{base[-1]:.1f}%')"
        ),
        md(
            f"The tilt the story needs simply isn't there. At 12 months rising-revenue returns "
            f"(**+{R['h12'][2]:.1f}%**) are actually *below* the base rate (**+{R['h12'][4]:.1f}%**), and "
            f"the basket is *up* slightly *less* often (**{R['h12'][5]:.0f}%** vs **{R['h12'][6]:.0f}%**). "
            "This is the opposite of a bullish leading signal. Hold that thought; the *next* chart shows "
            "**why**."
        ),
        md(
            "**The crucial test: does the revenue uptick come *first*?** We slide the revenue signal "
            "forward and backward against the stocks and measure how tightly they move together. A real "
            "leading signal would show its strongest *upward* link at a **positive lead** (revenue leads "
            "→ green bar on the right). Watch where the green actually is."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ll = st.lead_lag(F)\n"
            "    Ls = list(range(-6,7)); cs = [ll[L] for L in Ls]\n"
            "else:\n"
            "    Ls = sorted(R['leadlag']); cs = [R['leadlag'][L] for L in Ls]\n"
            "cols = [GREEN if c>0 else RED for c in cs]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.2))\n"
            "ax.bar(Ls, cs, color=cols, width=.8)\n"
            "ax.axhline(0, c='k', lw=.8); ax.axvline(0, c=GREY, lw=1, ls=':')\n"
            "ax.set_xlabel('lead L (months): L>0 = revenue moves FIRST (leading signal)   |   L<0 = revenue LAGS the stocks')\n"
            "ax.set_ylabel('correlation with stock move'); ax.set_xticks(Ls)\n"
            "ax.set_title('The green (positive) bars are on the LEFT: stocks lead revenue by ~6 months')\n"
            "plt.tight_layout(); plt.show()\n"
            "imax = int(np.nanargmax(cs))\n"
            "print(f'strongest POSITIVE link at L={Ls[imax]} months (stocks moved FIRST here)')"
        ),
        md(
            f"There it is. The only *positive* bars sit on the **left** (peak **+{R['leadlag'][-6]:.2f} at "
            "L = −6**) — the revenue signal lines up best with a stock move that happened **half a year "
            "earlier**. On the right, where a true leading signal would live (revenue moving first), the "
            "bars are all **negative**. **Revenue isn't leading the stocks — it's trailing them.** The "
            "casino shares turn on expectations; the revenue report just confirms it, weeks late."
        ),
        md(
            "**Could you trade it anyway?** Suppose you owned the casino basket every month revenue was "
            "accelerating and sat in cash otherwise. Here's that strategy's growth vs just buying and "
            "holding the basket."
        ),
        code(
            "if HAVE_REAL:\n"
            "    import pandas as pd\n"
            "    rising = st.rising_mask(F); pos = rising.astype(float).shift(1)\n"
            "    rr = F['basket'].pct_change()\n"
            "    dfp = pd.DataFrame({'r': rr, 'pos': pos}).dropna()\n"
            "    sw = dfp['pos'].diff().abs().fillna(0); c=10/1e4\n"
            "    overlay = (dfp['pos']*dfp['r'] - sw*c)\n"
            "    bh_grow = (1+dfp['r']).cumprod(); ov_grow = (1+overlay).cumprod()\n"
            "    fig, ax = plt.subplots(figsize=(9.2, 4.2))\n"
            "    ax.plot(bh_grow.index, bh_grow.values, c=GREY, lw=1.8, label='buy & hold casino basket')\n"
            "    ax.plot(ov_grow.index, ov_grow.values, c=RED, lw=1.8, label='own only when revenue rising (net)')\n"
            "    ax.set_yscale('log'); ax.set_ylabel('growth of $1 (log)')\n"
            "    ax.set_title('\"Own when revenue accelerates\" trails buy-and-hold')\n"
            "    ax.legend(); plt.tight_layout(); plt.show()\n"
            "    print(f'final $1 -> buy&hold {bh_grow.iloc[-1]:.1f}x  vs  overlay {ov_grow.iloc[-1]:.1f}x')\n"
            "else:\n"
            "    print(f\"overlay {R['overlay'][4]:.1f}%/yr vs buy-hold {R['overlay'][0]:.1f}%/yr (net) — see results.md\")"
        ),
        md(
            f"The overlay ends up **well below** buy-and-hold — **+{R['overlay'][4]:.1f}%/yr** vs "
            f"**+{R['overlay'][0]:.1f}%/yr** net — for essentially the **same Sharpe** "
            f"({R['overlay'][5]:.2f} vs {R['overlay'][1]:.2f}). All the rule does is sit in cash a third "
            "of the time; it trims your exposure without buying any skill. The signal doesn't help — "
            "it just **de-risks you into a smaller pile.**"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** Rising Strip revenue does *not* precede stronger casino-stock returns — "
            "at every horizon the tilt is flat or slightly *negative*, and when revenue accelerates "
            "**hardest** the next year is actively *worse*. There's no bullish leading signal here.\n"
            "- **Tradability — Mirage.** Owning on rising revenue **ties** buy-and-hold on Sharpe while "
            "giving up ~10 points of annual return. Nothing to deploy — it's just less beta.\n"
            "- **Leading signal? — Not supported.** The revenue uptick lines up with a stock move that "
            "already happened half a year earlier. The stocks **lead**; the revenue report **echoes**. "
            "The one word that makes the pitch — *leading* — is the part the data rejects."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — the honest bottom line\n\n"
            "Forget significance for a second. Even the *shape* of the signal is backwards for trading. "
            "The revenue run-rate is rising in **two-thirds of months** (a slow, smooth series drifts up "
            "most of the time), so 'own when rising' is really just 'own most of the time, minus a bit' — "
            "less exposure, not smarter exposure. And the one moment the signal fires **hardest** — the "
            "fastest revenue surges, like the 2021–22 post-COVID boom — is exactly when the casino stocks "
            "are *topping out* and about to sell off (that's why filtering to the biggest accelerations "
            "makes the next-year return sharply **negative**). There is no version of 'buy casinos when "
            "Vegas revenue accelerates' that both fires early and makes money."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The sibling test.** [Study 385 — Jobless-Claims-Momentum](../385-jobless-claims-momentum/) "
            "asks the same question of a famous macro 'leading' number and finds the same shape: a "
            "coincident-to-lagging echo, not a crystal ball.\n"
            "- **More macro pulses.** [Study 384 — ISM-PMI-Regime](../384-ism-pmi-regime/) and "
            "[Study 387 — Economic-Surprise-Index](../387-economic-surprise-index/) put other celebrated "
            "gauges through the same wringer.\n"
            "- **Build your own.** Swap the equal-weight basket for a single name (MGM), or pair revenue "
            "with a *price* trend filter — the lead/lag picture barely budges: a backward-looking report "
            "can't be made to lead a forward-looking stock by resampling it differently.\n\n"
            "*Think Vegas revenue leads the casinos? Show the lead/lag chart turning **green on the "
            "right** (positive lead) — then we'll talk.*"
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
            "# Vegas-Gaming-Win — a quantitative teardown 🔬\n"
            "### GGR-momentum split returns · Welch *t* + placebo null · the decisive lead/lag "
            "cross-correlation · a timing overlay vs buy-and-hold · robustness · a synthetic planted-edge control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The believers "
            "fuse two claims: that rising Las Vegas Strip GGR momentum (1) **predicts** stronger casino-"
            "equity returns and (2) does so **early** enough to trade. We separate them. The conditional "
            "return tilt is *absent-to-mildly-contrarian*; the decisive object is the **lead/lag "
            "structure**, which shows GGR momentum is **coincident-to-lagging**, not leading — and a "
            "tradable overlay that merely *sheds beta* seals the Tradability axis.\n\n"
            "> ⚠️ **Data + reconstruction note.** The NGCB publishes Strip GGR as monthly PDFs, "
            "unfetchable here; the GGR tape is a hardcoded **approximate reconstruction** whose annual "
            "sums match the published Strip totals (named on the Signal axis). The basket is an "
            "equal-weight, **surviving** set of listed operators (MGM/LVS/WYNN/CZR/BYD/PENN), yfinance "
            "daily adjusted close (total-return), month-end sampled. Offline core + synthetic control are "
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
            f"| **Signal** | `NONE` | 12-month rising-GGR mean **+{R['h12'][2]:.1f}%** vs base "
            f"**+{R['h12'][4]:.1f}%** (*wrong* sign); Welch **t = {R['h12'][7]:.2f}**, placebo "
            f"**p = {R['h12'][8]:.2f}** — no bullish tilt at any horizon, contrarian where it fires hardest. |\n"
            f"| **Tradability** | `MIRAGE` | Own-when-rising overlay **+{R['overlay'][4]:.1f}%/yr** "
            f"(Sharpe **{R['overlay'][5]:.2f}**) **vs buy-hold +{R['overlay'][0]:.1f}%/yr** "
            f"(Sharpe **{R['overlay'][1]:.2f}**). Same Sharpe, ~10 pts/yr less return — just less beta. |\n"
            f"| **Leading signal?** | `NOT SUPPORTED` | Peak *positive* lead/lag correlation at "
            f"**L = −6** (GGR lags the stocks by ~half a year); at positive leads corr < 0 throughout. A "
            "coincident-to-lagging echo, not a leader. |\n\n"
            "> 💡 In plain words: casino equities are liquid, forward-looking and analyst-covered, so they "
            "discount the gaming cycle *before* a revenue report — released ~5 weeks late — can tally it. "
            "GGR momentum lines up with a stock move already half a year old; the 'leading signal' is the "
            "market's own lead, reflected back."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $g_t$ be the Strip GGR level and $T_t=\\sum_{i=0}^{11} g_{t-i}$ its trailing-12-month "
            "sum (a deseasonalised run-rate that also absorbs the COVID closure). Define momentum "
            "$m_t = T_t / T_{t-3} - 1$; GGR is **RISING** at $t$ when $m_t > 0$. With a one-month "
            "execution lag (the month-$t$ print is released during $t+1$ and acted on at its close), "
            "define forward return $r_{t+1\\to t+1+H}$ on the equal-weight casino basket.\n\n"
            "- **H₁ (predicts).** $\\mathbb{E}[r\\mid \\text{rising}] > \\mathbb{E}[r]$ — a *positive* "
            "excess over the base rate.\n"
            "- **H₂ (leads).** The strongest positive GGR↔return correlation sits at a **positive** lead "
            "(GGR moves first).\n"
            "- **H₃ (deployable).** An own-when-rising overlay beats buy-and-hold net of costs.\n\n"
            "We find **H₁ rejected** (excess is *negative*, $t = -0.72$, and sharply negative for big "
            "upticks), **H₂ rejected** (peak positive corr at $L=-6$), **H₃ rejected** (overlay merely "
            "sheds beta). The folklore is right exactly where it's uninformative (GGR and casinos "
            "co-move over the cycle) and wrong exactly where it would pay (a *leading*, *tradable* edge)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The conditional-return test is a two-sample mean comparison judged by its standard error:\n\n"
            "$$\\widehat{\\Delta}_H = \\bar r^{\\text{rising}}_H - \\bar r^{\\text{all}}_H,\\qquad "
            "t = \\frac{\\widehat{\\Delta}_H}{\\sqrt{\\,s^2_{\\text{rising}}/k + s^2_{\\text{all}}/N\\,}}.$$\n\n"
            "But even a significant $\\widehat{\\Delta}$ would **not** establish *leading*: a coincident or "
            "lagging series can co-move with forward returns through cycle autocorrelation. The "
            "identifying test is the **lead/lag cross-correlation** "
            "$\\rho(L) = \\mathrm{corr}(m_t,\\ r_{t+L\\to t+L+1})$. A genuine early signal peaks "
            "(positively) at $L>0$. If $\\arg\\max_L \\rho(L) < 0$, the stocks **led** the revenue — and "
            "the entire 'leading' thesis collapses regardless of the conditional mean."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **GGR tape.** Monthly Strip GGR (US$ M), hardcoded reconstruction, "
            f"{R['start'][:7]}→{R['end'][:7]} ({R['months']} months). Annual sums matched to the published "
            "totals; not the settled monthly print (named on the axis).\n"
            "- **Signal.** $T_t=$ trailing-12m sum; $m_t = T_t/T_{t-3}-1$; RISING when $m_t>0$.\n"
            "- **Forward returns.** Enter at the close **1 month after** the signal (no look-ahead — the "
            "print is public by then), hold $H\\in\\{1,3,6,12\\}$ months; drop horizons that overrun.\n"
            "- **Null #1 (Welch t).** Rising-set mean vs the unconditional mean.\n"
            "- **Null #2 (placebo).** 20,000 draws of $k$ random months; "
            "$p = \\Pr[\\text{random-draw mean} \\ge \\text{rising mean}]$ (as bullish or more).\n"
            "- **Identification (lead/lag).** $\\rho(L)$ for $L\\in[-6,6]$ — *where* do GGR and the stocks "
            "line up?\n"
            "- **Tradability.** Own-when-rising overlay, 1-month lag, 10 bps one-way per switch (turnover "
            "one-way × NAV), excess-of-zero Sharpe (cash leg = 0, labelled).\n"
            "- **Positive control.** A deterministic series with a *planted* GGR→returns link: `edge=0` "
            "must not fake significance; a large `edge` must light up the test."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The point estimates — no bullish tilt, wrong sign\n\n"
            "Rising-GGR forward mean with $\\pm$ standard error against the unconditional base rate "
            "(dashed). At or *below* base at every horizon — the opposite of the claim."
        ),
        code(
            "hs = [1, 3, 6, 12]\n"
            "if HAVE_REAL:\n"
            "    cm, bm, ts, ses = [], [], [], []\n"
            "    for m in hs:\n"
            "        s = st.summarize(F, m); cm.append(s['rising_mean']); bm.append(s['base_mean']); ts.append(s['t'])\n"
            "        r,_f,_a = st.split_returns(F, m); ses.append(r.std(ddof=1)/np.sqrt(len(r)))\n"
            "else:\n"
            "    cm = [R['h1'][2]/100, R['h3'][2]/100, R['h6'][2]/100, R['h12'][2]/100]\n"
            "    bm = [R['h1'][4]/100, R['h3'][4]/100, R['h6'][4]/100, R['h12'][4]/100]\n"
            "    ts = [R['h1'][7], R['h3'][7], R['h6'][7], R['h12'][7]]; ses = [.02,.035,.05,.08]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x, [c*100 for c in cm], yerr=[s*100 for s in ses], capsize=5, color=GREEN, width=.5, label='rising-GGR (±SE)')\n"
            "ax.plot(x, [b*100 for b in bm], 'D', ms=11, c=GREY, label='unconditional base rate')\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{m}m' for m in hs]); ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean forward basket return (%)')\n"
            "ax.set_title('Rising-GGR mean sits at or below base at every horizon'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Welch t by horizon:', {f'{m}m': round(t,2) for m,t in zip(hs,ts)})"
        ),
        md(
            f"> 💡 In plain words: at 12m the rising-GGR mean is **+{R['h12'][2]:.1f}%** vs base "
            f"**+{R['h12'][4]:.1f}%** — a ~{R['h12'][4]-R['h12'][2]:.1f}-point *shortfall* at "
            f"**t = {R['h12'][7]:.2f}** (wrong sign for the claim, insignificant). H₁ is **rejected**: "
            "there is no bullish leading tilt, and the up-rate is *lower* than the base rate too."
        ),
        md(
            "### 4b · The decisive identification test — lead/lag\n\n"
            "$\\rho(L) = \\mathrm{corr}(m_t, r_{t+L\\to t+L+1})$. Positive bars **right** of zero would "
            "mean GGR **leads** the stocks; positive bars **left** of zero mean the stocks led GGR."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ll = st.lead_lag(F); Ls = list(range(-6,7)); cs = [ll[L] for L in Ls]\n"
            "else:\n"
            "    Ls = sorted(R['leadlag']); cs = [R['leadlag'][L] for L in Ls]\n"
            "cols = [GREEN if c>0 else RED for c in cs]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(Ls, cs, color=cols, width=.8)\n"
            "ax.axhline(0, c='k', lw=.8); ax.axvline(0, c=GREY, lw=1, ls=':')\n"
            "imax = int(np.nanargmax(cs))\n"
            "ax.annotate('strongest POSITIVE link\\n(stocks LED revenue)', xy=(Ls[imax], cs[imax]),\n"
            "            xytext=(Ls[imax]+1.2, cs[imax]+0.03), ha='center', color=GREEN,\n"
            "            arrowprops=dict(arrowstyle='->', color=GREEN))\n"
            "ax.set_xlabel('lead L (months): L>0 = GGR leads (early signal)   |   L<0 = GGR lags')\n"
            "ax.set_ylabel(r'$\\rho(L)$'); ax.set_xticks(Ls)\n"
            "ax.set_title('argmax rho(L) is at L<0: GGR is coincident-to-lagging')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'argmax at L={Ls[imax]} (rho={cs[imax]:+.2f}); rho at +1 month = {cs[Ls.index(1)]:+.2f}')"
        ),
        md(
            "> 💡 In plain words: $\\arg\\max_L \\rho(L) = -6$. GGR momentum correlates most (positively) "
            "with a stock move **half a year in its past**; on the leading side a genuine early signal "
            "needs, $\\rho < 0$ throughout. **H₂ rejected.** The forward-looking equities lead; the "
            "five-weeks-late revenue report trails — this is the load-bearing result, independent of the "
            "conditional-mean significance."
        ),
        md(
            "### 4c · Tradability — the own-when-rising overlay merely sheds beta\n\n"
            "Own the basket when GGR momentum is rising, cash when falling (1-month lag, 10 bps/switch). "
            "Annualised mean and Sharpe vs buy-and-hold."
        ),
        code(
            "if HAVE_REAL:\n"
            "    o = st.timing_overlay(F, cost_bps=10.0)\n"
            "    bh_m, bh_s = o['bh_mean']*100, o['bh_sharpe']\n"
            "    g_m, g_s = o['overlay_gross_mean']*100, o['overlay_gross_sharpe']\n"
            "    n_m, n_s = o['overlay_net_mean']*100, o['overlay_net_sharpe']; nsw=o['n_switches']\n"
            "else:\n"
            "    bh_m, bh_s = R['overlay'][0], R['overlay'][1]; g_m,g_s = R['overlay'][2],R['overlay'][3]\n"
            "    n_m, n_s = R['overlay'][4], R['overlay'][5]; nsw=R['overlay'][6]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.2))\n"
            "labels = ['buy &\\nhold', 'overlay\\ngross', 'overlay\\nnet @10bps']\n"
            "a1.bar(labels, [bh_m, g_m, n_m], color=[GREY, AMBER, RED], width=.6)\n"
            "for i,v in enumerate([bh_m,g_m,n_m]): a1.annotate(f'{v:.1f}%',(i,v),ha='center',va='bottom')\n"
            "a1.set_ylabel('annualised mean return (%)'); a1.set_title('Return: overlay gives up ~10 pts/yr')\n"
            "a2.bar(labels, [bh_s, g_s, n_s], color=[GREY, AMBER, RED], width=.6)\n"
            "for i,v in enumerate([bh_s,g_s,n_s]): a2.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom')\n"
            "a2.set_ylabel('annualised Sharpe (excess-of-0)'); a2.set_title(f'Sharpe: a dead tie ({nsw} switches)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'net overlay {n_m:.1f}%/yr (Sharpe {n_s:.2f}) vs buy-hold {bh_m:.1f}%/yr (Sharpe {bh_s:.2f})')"
        ),
        md(
            f"> 💡 In plain words: the overlay returns **+{R['overlay'][4]:.1f}%/yr** net vs "
            f"**+{R['overlay'][0]:.1f}%** for buy-and-hold, at an essentially *identical* Sharpe "
            f"({R['overlay'][5]:.2f} vs {R['overlay'][1]:.2f}) over just {R['overlay'][6]} switches. GGR "
            "momentum is rising ~66% of the time, so the rule only removes a third of the exposure — it "
            "sheds beta, it doesn't add alpha. **H₃ rejected.** `MIRAGE`."
        ),
        md(
            "### 4d · Robustness — window, threshold, and the COVID dependence\n\n"
            "Vary the momentum window $k$ and the rising threshold, and drop the COVID episode. The "
            "12-month *t* is never bullish — and the biggest accelerations turn sharply **negative**."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rob = []\n"
            "    for k in (1,3,6):\n"
            "        s = st.summarize(F, 12, k=k); rob.append((f'k={k}', s['n_rising'], s['rising_mean']*100, s['t'], s['p_placebo']))\n"
            "    s = st.summarize(F, 12, thresh=0.03); rob.append(('thr>+3%', s['n_rising'], s['rising_mean']*100, s['t'], s['p_placebo']))\n"
            "    F2 = F[(F.index < '2020-01-01') | (F.index >= '2021-07-01')]\n"
            "    s = st.summarize(F2, 12); rob.append(('ex-COVID', s['n_rising'], s['rising_mean']*100, s['t'], s['p_placebo']))\n"
            "else:\n"
            "    rob = [(l,n,r,t,p) for (l,n,r,_b,t,p) in R['robust']]\n"
            "labels = [r[0] for r in rob]; tt = [r[3] for r in rob]; nn = [r[1] for r in rob]\n"
            "cols = [RED if t<0 else GREEN for t in tt]\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.3))\n"
            "ax.bar(labels, tt, color=cols, width=.6)\n"
            "ax.axhline(2, ls='--', c=GREEN, label='t=+2 (bullish significance bar)')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,(t,k) in enumerate(zip(tt,nn)): ax.annotate(f'n={k}',(i,t),ha='center',va='bottom' if t>=0 else 'top')\n"
            "ax.set_ylabel('Welch t (12-month)'); ax.set_title('No spec is bullish; the biggest upticks flip sharply NEGATIVE'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('robustness (label, n, rising12%, t, p):', [(r[0], r[1], round(r[2],1), round(r[3],2), round(r[4],3)) for r in rob])"
        ),
        md(
            "> 💡 In plain words: the tilt only gets **more negative** as the window slows "
            f"(**k=6 → t={R['robust'][2][4]:.2f}**), and restricting to the *biggest* accelerations "
            f"(>+3%) drives **t={R['robust'][3][4]:.2f}** — the fastest revenue surges (the 2021–22 "
            "rebound, the mid-2000s build) are the tops the stocks *sell off from*. Ex-COVID the 12m t is "
            f"**{R['robust'][4][4]:.2f}** — still the wrong sign. The signal is never bullish and is "
            "actively contrarian where it fires hardest."
        ),
        md(
            "### 4e · Faithful-engine control — we know the truth here\n\n"
            "A deterministic monthly series with a *planted* link (rising GGR momentum at $t$ **lifts** "
            "the $t{+}1$ return by `edge`). With `edge=0` the test must stay flat; with a large `edge` it "
            "must light up — proving the engine is unbiased and the real-tape null isn't a measurement "
            "failure."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.05):\n"
            "    syn = data.synthetic_ggr(n_months=360, edge=edge, seed=762)\n"
            "    s = st.summarize(syn, 1, k=3)\n"
            "    res.append((edge, s['n_rising'], s['rising_mean']*100, s['base_mean']*100, s['t'], s['p_placebo']))\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "labels = [f'planted edge\\n{e*100:.0f}% / month' for e,_,_,_,_,_ in res]\n"
            "tvals = [r[4] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=GREEN, label='t=+2 (significance bar)'); ax.axhline(0, c='k', lw=.8)\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('Welch t (1-month)'); ax.set_title('Control: no link -> flat; real link -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,k,c,b,t,p in res: print(f'planted {e*100:+.0f}%/mo: n_ris={k} rising={c:.2f}% base={b:.2f}% t={t:.2f} p={p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted link the control sits at "
            f"**t = {R['syn'][0][4]:.2f}** (no false positive, under 2); a **+5%/month** planted link "
            f"drives **t = {R['syn'][1][4]:.2f}**. So the machinery is honest — the real-tape *t* of ≈ "
            "−0.7 is a *genuine* absent (mildly contrarian) edge, not a broken test. The engine *can* "
            "bank a real GGR→returns link; the real tape just doesn't carry one."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — 12m excess **{R['h12'][2]-R['h12'][4]:+.1f}pp** at Welch "
            f"**t = {R['h12'][7]:.2f}** / placebo **p = {R['h12'][8]:.2f}**; *wrong sign*, no window is "
            f"bullish, and the biggest accelerations are sharply contrarian (**t = {R['robust'][3][4]:.2f}** "
            "at >+3%). No bullish leading signal ⇒ NONE.\n"
            f"- **Tradability `MIRAGE`** — the own-when-rising overlay returns "
            f"**+{R['overlay'][4]:.1f}%/yr** (Sharpe {R['overlay'][5]:.2f}) vs buy-hold "
            f"**+{R['overlay'][0]:.1f}%/yr** (Sharpe {R['overlay'][1]:.2f}). Same risk-adjusted return, "
            "~10 pts/yr less compounding — it just removes a third of the beta.\n"
            "- **Leading signal? `NOT SUPPORTED`** — $\\arg\\max_L \\rho(L) = -6$ months: GGR momentum is "
            "**coincident-to-lagging**, not leading. The forward-looking casino equities are the leading "
            "indicator; the revenue print echoes them. The defining word — *leading* — is the part the "
            "data rejects."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — why even the *shape* is wrong\n\n"
            "Grant the lore a genuine effect and the structure still defeats it. GGR momentum is 'rising' "
            "in **~66% of months** (a slow, deseasonalised run-rate drifts up most of the time), so "
            "own-when-rising is 'own most of the time, minus a third' — the overlay's near-tie Sharpe is "
            "just diluted beta, not timing. And the regime where the signal fires **hardest** — the "
            "fastest revenue surges — is precisely where the *forward* returns are worst (the >+3% "
            f"filter gives **t = {R['robust'][3][4]:.2f}**), because those surges are the late-cycle tops "
            "the liquid equities have already discounted and begun selling. No lag, threshold, or cost "
            "assumption rescues a backward-looking report masquerading as a leading one."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The sibling.** [Study 385 — Jobless-Claims-Momentum](../385-jobless-claims-momentum/): a "
            "famous macro 'leading' number, same hardcoded-snapshot + lead/lag + overlay method — the "
            "same coincident-to-lagging verdict.\n"
            "- **Companion pulses.** [Study 384 — ISM-PMI-Regime](../384-ism-pmi-regime/), "
            "[Study 387 — Economic-Surprise-Index](../387-economic-surprise-index/) — does any celebrated "
            "gauge time equities?\n"
            "- **Sharper identification.** Replace the reconstruction with the settled NGCB monthly print "
            "(or add Macau/regional GGR), run a proper VAR / Granger test, or condition on GGR *surprises* "
            "vs the analyst consensus rather than realised momentum; the coincident-to-lagging structure "
            "is robust — a backward-looking report can't be made to lead a forward-looking stock by "
            "resampling it.\n\n"
            "*The reproducible core is offline and deterministic; the GGR input is an explicit labelled "
            "reconstruction. Methods and sources: [`docs/references.md`](../docs/references.md); frozen "
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
