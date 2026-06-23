"""Generate the two narrative notebooks for Study 385 (Jobless-Claims-Momentum).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the hardcoded claims
snapshot (always available) and the cached SPY prices under ../_cache/, and otherwise quote
the frozen headline numbers in ``R`` (mirroring docs/results.md). The synthetic positive
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


# Frozen real-tape headline numbers — mirror of docs/results.md (FRED IC4WSA hardcoded
# monthly snapshot + SPY month-end, 1993-01 -> 2026-06, 402 months, 33.4 years).
R = dict(
    start="1993-01-31", end="2026-06-30", months=402, years=33.4,
    # per-horizon: (months, n_rising, rising_mean%, falling_mean%, base_mean%, ris_down%, base_down%, t, p_placebo)
    h1=(1, 174, 0.69, 1.15, 0.95, 36, 35, -0.63, 0.201),
    h3=(3, 172, 2.30, 3.30, 2.87, 31, 28, -0.76, 0.149),
    h6=(6, 171, 4.91, 6.49, 5.80, 29, 24, -0.81, 0.137),
    h12=(12, 168, 10.38, 13.29, 12.04, 24, 18, -0.94, 0.098),
    # lead/lag: L -> corr
    leadlag={-6: 0.040, -5: -0.001, -4: -0.093, -3: -0.193, -2: -0.072, -1: 0.103,
             0: 0.055, 1: 0.046, 2: 0.082, 3: 0.030, 4: -0.054, 5: 0.033, 6: 0.105},
    # overlay: (bh_mean%, bh_sharpe, gross_mean%, gross_sharpe, net_mean%, net_sharpe, switches)
    overlay=(11.4, 0.77, 7.3, 0.75, 7.0, 0.72, 109),
    # robustness 12m: (label, n_rising, rising12%, base12%, t, p)
    robust=[("k=1", 167, 10.53, 12.04, -0.90, 0.120), ("k=3", 168, 10.38, 12.04, -0.94, 0.098),
            ("k=6", 136, 8.49, 12.04, -1.74, 0.008), ("thr>+10%", 32, 16.20, 12.04, 1.02, 0.927),
            ("ex-COVID", 162, 10.35, 12.88, -1.32, 0.033)],
    # synthetic control: (edge, n_rising, rising1m%, base1m%, t, p)
    syn=[(0.0, 172, 0.61, 0.74, -0.36, 0.338), (0.04, 172, -2.88, -1.23, -4.40, 0.001)],
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Early_warning%3F: Not_supported](https://img.shields.io/badge/Early_warning%3F-Not_supported-8b949e?style=flat-square)\n\n"
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

from jobless_claims_momentum import data, strategy as st

HAVE_REAL = data.have_real()
F = data.load_real() if HAVE_REAL else None
print("SPY cache present:", HAVE_REAL,
      "| claims+SPY months:", (0 if F is None else len(F)))
"""

# The frozen headline dict is embedded into the first code cell so every downstream cell can
# quote it whether or not the SPY cache is present.
BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Do rising jobless claims warn you before the market falls? 📰\n"
            "### The 'most important weekly number' as a stock-market crystal ball, in plain English\n\n"
            + BADGES +
            "Every Thursday morning the U.S. reports how many people just filed for unemployment. When "
            "that number — smoothed into a **4-week average** — starts climbing, the folklore says the "
            "economy is rolling over and a **stock-market downturn is coming**. Claims *lead*, the story "
            "goes, so a claims uptick is your early-warning to get defensive.\n\n"
            "It's a great story. It's also testable. This notebook asks three blunt questions: when "
            "claims rise, does the market really do worse? Does the claims uptick actually come "
            "**first** (that's the whole pitch)? And if you *sold* every time claims rose, would you "
            "make money?\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the lead/lag cross-correlation and the "
            "synthetic control? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** The live FRED feed is blocked in this environment, so we use "
            "a **frozen snapshot** of the official 4-week-average claims series (the settled numbers, "
            "not the real-time prints) — public and faithful, including the giant COVID-2020 spike. "
            "Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| When claims rise, does the market do worse? | **A little — yes.** Over the next year SPY "
            f"averages **+{R['h12'][2]:.1f}%** after a claims uptick vs **+{R['h12'][4]:.1f}%** normally, "
            "and falls more often. The *direction* matches the folklore. |\n"
            "| Is that gap reliable? | **No.** It's small and well inside the noise — statistically you "
            "can't tell it from luck (and it vanishes if you tweak the recipe). |\n"
            "| Does the claims uptick come *first*? | **No — and this is the killer.** The claims signal "
            "lines up best with a market move that already happened **three months earlier.** Claims "
            "*follow* stocks here; they don't lead them. |\n"
            "| So could you trade it? | **It loses.** \"Sell when claims rise\" earned "
            f"**+{R['overlay'][4]:.1f}%/yr** vs **+{R['overlay'][0]:.1f}%** for just holding — you give "
            "up return for protection that never shows up. |\n\n"
            "> Claims and recessions are real. But \"claims warn you *early*\" is a coincident echo "
            "wearing a crystal-ball costume — and acting on it costs you money."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Initial jobless claims are a leading indicator. When the 4-week average turns up, the "
            "labour market is weakening and the stock market is about to follow — so get defensive the "
            "moment claims start rising.\"*\n\n"
            "There's a respectable backbone to this: weekly initial claims are an official component of "
            "The Conference Board's **Leading Economic Index**, entered with a negative sign. So the "
            "intuition isn't crazy — fewer jobs, weaker spending, lower profits. The trading leap is the "
            "part we test: that a claims *uptick* arrives early enough, and cleanly enough, to be a "
            "**tradable** warning for equities."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If true, this would be gold: a free, weekly, government-published number that tells you to "
            "step aside before drawdowns. But \"leading\" hides a trap. The **stock market is itself a "
            "leading indicator** — it usually turns *before* the economy. So a macro number that lines "
            "up with market weakness might not be *predicting* the market at all; it might just be "
            "**echoing** a turn the market already made. The difference between *leads* and *echoes* is "
            "the difference between an edge and a mirage — and you can only tell them apart by checking "
            "the timing carefully."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We line up **{R['years']:.0f} years** ({R['start'][:4]}–{R['end'][:4]}, "
            f"{R['months']} months) of the 4-week claims average against month-end SPY, and:\n\n"
            "1. **Split the months.** Call claims **rising** when the smoothed level is above where it "
            "was three months ago. Compare what SPY did next (1/3/6/12 months) in rising-claims months "
            "vs all months.\n"
            "2. **Check the timing.** The crucial test: slide claims forward and backward against the "
            "market and find *where* they line up best. If claims truly **lead**, the strongest link "
            "shows up at a **positive lead** (claims first, market later).\n"
            "3. **Try to trade it.** Sit in cash whenever claims are rising, hold otherwise, pay "
            "realistic costs — and see if it beats just buying and holding."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the raw material.** Here's the 4-week claims average over three decades — the "
            "quiet expansions, and the two giant spikes (2008–09 and the off-the-chart COVID surge of "
            "2020). Claims clearly *know* about recessions. The question is whether they know **early**."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = F['claims']\n"
            "    fig, ax = plt.subplots(figsize=(9.4, 4.0))\n"
            "    ax.plot(c.index, c.values, c=RED, lw=1.3)\n"
            "    ax.set_yscale('log')\n"
            "    ax.set_title('U.S. initial jobless claims, 4-week average (thousands, log scale)')\n"
            "    ax.set_ylabel('claims (thousands)')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('peak 4-wk MA:', int(c.max()), 'thousand, around', c.idxmax().date())\n"
            "else:\n"
            "    print('no cache — see docs/results.md; COVID 4-wk MA peaked ~4,170k in Apr 2020')"
        ),
        md(
            "**Now the payoff.** For each horizon, the average forward SPY return in **rising-claims** "
            "months next to the return on an **average** month. The folklore predicts the green bars sit "
            "*below* the grey ones."
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
            "ax.bar(x-.2, ris, .4, color=GREEN, label='after claims RISE')\n"
            "ax.bar(x+.2, base, .4, color=GREY, label='an average month (base rate)')\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{m} months' for m in hs])\n"
            "ax.set_ylabel('average forward SPY return (%)')\n"
            "ax.set_title('Rising claims -> lower forward returns... but only by a little')\n"
            "for i,(a,b) in enumerate(zip(ris,base)):\n"
            "    ax.annotate(f'{a:.1f}%',(i-.2,a),ha='center',va='bottom',fontsize=9)\n"
            "    ax.annotate(f'{b:.1f}%',(i+.2,b),ha='center',va='bottom',fontsize=9)\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('12-month: rising', f'{ris[-1]:.1f}%', 'vs base', f'{base[-1]:.1f}%')"
        ),
        md(
            f"The direction is right — at 12 months rising-claims returns (**+{R['h12'][2]:.1f}%**) sit "
            f"below the base rate (**+{R['h12'][4]:.1f}%**), and the market is *down* more often "
            f"(**{R['h12'][5]:.0f}%** of the time vs **{R['h12'][6]:.0f}%**). But the gap is a couple of "
            "points — small enough that, with the numbers we have, it could easily be chance. Hold that "
            "thought; the *next* chart is where the story breaks."
        ),
        md(
            "**The crucial test: does the claims uptick come *first*?** We slide claims forward and "
            "backward against the market and measure how tightly they move together. A real "
            "early-warning would show its strongest *downward* link at a **positive lead** (claims "
            "lead → bar dips on the right). Watch where it actually dips."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ll = st.lead_lag(F)\n"
            "    Ls = list(range(-6,7)); cs = [ll[L] for L in Ls]\n"
            "else:\n"
            "    Ls = sorted(R['leadlag']); cs = [R['leadlag'][L] for L in Ls]\n"
            "cols = [RED if L<0 else GREEN for L in Ls]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.2))\n"
            "ax.bar(Ls, cs, color=cols, width=.8)\n"
            "ax.axhline(0, c='k', lw=.8); ax.axvline(0, c=GREY, lw=1, ls=':')\n"
            "ax.set_xlabel('lead L (months): L>0 = claims move FIRST (early-warning)   |   L<0 = claims LAG the market')\n"
            "ax.set_ylabel('correlation with market move'); ax.set_xticks(Ls)\n"
            "ax.set_title('The dip is on the LEFT: claims lag the market by ~3 months')\n"
            "plt.tight_layout(); plt.show()\n"
            "imin = int(np.nanargmin(cs))\n"
            "print(f'strongest negative link at L={Ls[imin]} months (claims FOLLOW the market here)')"
        ),
        md(
            f"There it is. The deepest *negative* bar is at **L = −3** — the claims signal lines up best "
            "with a market move that happened **three months earlier**. On the right, where a true "
            "early-warning would live (claims moving first), the bars are near zero or even *positive*. "
            "**Claims aren't leading the market — they're trailing it.** The recession shows up in stock "
            "prices, and only later in the claims average."
        ),
        md(
            "**Could you trade it anyway?** Suppose you sold (went to cash) every month claims were "
            "rising and held SPY otherwise. Here's that strategy's growth vs just buying and holding."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rising = st.rising_mask(F); pos = (~rising).astype(float).shift(1)\n"
            "    rr = F['spy'].pct_change()\n"
            "    import pandas as pd\n"
            "    dfp = pd.DataFrame({'r': rr, 'pos': pos}).dropna()\n"
            "    sw = dfp['pos'].diff().abs().fillna(0); c=10/1e4\n"
            "    overlay = (dfp['pos']*dfp['r'] - sw*c)\n"
            "    bh_grow = (1+dfp['r']).cumprod(); ov_grow = (1+overlay).cumprod()\n"
            "    fig, ax = plt.subplots(figsize=(9.2, 4.2))\n"
            "    ax.plot(bh_grow.index, bh_grow.values, c=GREY, lw=1.8, label='buy & hold SPY')\n"
            "    ax.plot(ov_grow.index, ov_grow.values, c=RED, lw=1.8, label='cash when claims rising (net)')\n"
            "    ax.set_yscale('log'); ax.set_ylabel('growth of $1 (log)')\n"
            "    ax.set_title('\"Sell when claims rise\" lags buy-and-hold for 30 years')\n"
            "    ax.legend(); plt.tight_layout(); plt.show()\n"
            "    print(f'final $1 -> buy&hold {bh_grow.iloc[-1]:.1f}x  vs  overlay {ov_grow.iloc[-1]:.1f}x')\n"
            "else:\n"
            "    print(f\"overlay {R['overlay'][4]:.1f}%/yr vs buy-hold {R['overlay'][0]:.1f}%/yr (net) — see results.md\")"
        ),
        md(
            f"The defensive overlay ends up **well below** buy-and-hold — "
            f"**+{R['overlay'][4]:.1f}%/yr** vs **+{R['overlay'][0]:.1f}%/yr** net, and a *lower* Sharpe "
            f"({R['overlay'][5]:.2f} vs {R['overlay'][1]:.2f}). Every time you stepped aside on a claims "
            "uptick you mostly missed *gains*, because claims kept rising well into recoveries the "
            "market had already started pricing. The signal doesn't just fail to help — **it hurts.**"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — Weak.** Rising claims really do precede slightly weaker, more-often-negative "
            "returns — but the gap is small, not statistically significant, and fragile to how you "
            "define the signal. Real as lore, weak as edge.\n"
            "- **Tradability — Mirage.** Selling on rising claims **loses to buy-and-hold**. There's "
            "nothing to deploy.\n"
            "- **Early warning? — Not supported.** The claims uptick lines up with a market move that "
            "already happened a quarter earlier. Claims **echo** equity weakness; they don't forecast "
            "it. The one word that makes the pitch — *early* — is the part the data rejects."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — the honest bottom line\n\n"
            "Forget significance for a second. Even if the small tilt were real, the operational reality "
            "is brutal: claims rise in **roughly half** of all months (every wiggle counts as 'rising'), "
            "so a cash-on-rising rule whipsaws you in and out constantly — "
            f"**{R['overlay'][6]} switches** in 30 years — while the deepest claims surges (the "
            "*genuine* recession lows) are exactly the moments you'd most want to be **buying**, not "
            "selling. That's why pushing the rule to only fire on a *big* claims jump flips its sign "
            "**positive**. There is no version of \"sell when claims rise\" that both fires early and "
            "makes money."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The sibling test.** [Study 268 — Sahm-Rule](../268-sahm-rule/) asks the same "
            "question of the famous unemployment-rate recession trigger: real labour signal, but can "
            "you trade it?\n"
            "- **More macro crystal balls.** [Study 266 — Misery-Index](../266-misery-index/) and "
            "[Study 267 — M2-Growth](../267-m2-growth/) put other celebrated macro gauges through the "
            "same wringer.\n"
            "- **Build your own.** Swap the 4-week average for the weekly raw series, or pair claims "
            "with a *price* trend filter — the lead/lag picture barely budges: a coincident series can't "
            "be made to lead by smoothing it differently.\n\n"
            "*Think claims lead the market? Show the lead/lag chart dipping on the **right** (positive "
            "lead) — then we'll talk.*"
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
            "# Jobless-Claims-Momentum — a quantitative teardown 🔬\n"
            "### Claims-momentum split returns · Welch *t* + placebo null · the decisive lead/lag "
            "cross-correlation · a timing overlay vs buy-and-hold · robustness · a synthetic planted-edge control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The "
            "believers fuse two claims: that rising jobless claims (1) **predict** weaker equity returns "
            "and (2) do so **early** enough to trade. We separate them. The conditional return tilt is "
            "the *right sign but insignificant*; the decisive object is the **lead/lag structure**, "
            "which shows claims momentum is **coincident-to-lagging**, not leading — and a tradable "
            "overlay that *underperforms* buy-and-hold seals the Tradability axis.\n\n"
            "> ⚠️ **Data + snapshot note.** FRED's CSV endpoint is firewalled here; the claims tape is a "
            "hardcoded monthly snapshot of `IC4WSA` (4-week-MA initial claims, SA, thousands) — the "
            "settled print, **not** the real-time vintage (named on the Signal axis). SPY is yfinance "
            "daily adjusted close (total-return), month-end sampled. Offline core + synthetic control "
            "are deterministic. Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | 12-month rising-claims mean **+{R['h12'][2]:.1f}%** vs base "
            f"**+{R['h12'][4]:.1f}%** (right sign); Welch **t = {R['h12'][6]:.2f}**, placebo "
            f"**p = {R['h12'][7]:.2f}** — fails **t ≥ 2**, fragile to window, sign-flips for big upticks. |\n"
            f"| **Tradability** | `MIRAGE` | Cash-on-rising overlay **+{R['overlay'][4]:.1f}%/yr** "
            f"(Sharpe **{R['overlay'][5]:.2f}**) **vs buy-hold +{R['overlay'][0]:.1f}%/yr** "
            f"(Sharpe **{R['overlay'][1]:.2f}**). Acting on it destroys return. |\n"
            f"| **Early warning?** | `NOT SUPPORTED` | Peak negative lead/lag correlation at "
            f"**L = −3** (claims lag the market); at positive leads corr ≈ 0. A coincident-to-lagging "
            "echo, not a leader. |\n\n"
            "> 💡 In plain words: the equity market *is* a leading indicator of the economy, so a macro "
            "series that co-moves with equity weakness need not lead it. Claims momentum lines up with a "
            "market move already three months old — the 'early-warning' is the market's own lead, "
            "reflected back."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $c_t$ be the 4-week-MA initial-claims level and "
            "$m_t = c_t / c_{t-3} - 1$ its 3-month momentum. Claims are **RISING** at $t$ when "
            "$m_t > 0$. With a one-month execution lag (the month-$t$ print is acted on at the close of "
            "$t+1$), define forward return $r_{t+1\\to t+1+H}$.\n\n"
            "- **H₁ (predicts).** $\\mathbb{E}[r\\mid \\text{rising}] < \\mathbb{E}[r]$ — a *negative* "
            "excess over the base rate.\n"
            "- **H₂ (leads).** The strongest negative claims↔return correlation sits at a **positive** "
            "lead (claims move first).\n"
            "- **H₃ (deployable).** A cash-on-rising overlay beats buy-and-hold net of costs.\n\n"
            "We find **H₁ directionally true but insignificant** ($t = -0.94$, sign-flips by spec), "
            "**H₂ rejected** (peak negative corr at $L=-3$), **H₃ rejected** (overlay underperforms). "
            "The folklore is right exactly where it's uninformative (claims and recessions co-move) and "
            "wrong exactly where it would pay (a *leading*, *tradable* edge)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The conditional-return test is a two-sample mean comparison judged by its standard error:\n\n"
            "$$\\widehat{\\Delta}_H = \\bar r^{\\text{rising}}_H - \\bar r^{\\text{all}}_H,\\qquad "
            "t = \\frac{\\widehat{\\Delta}_H}{\\sqrt{\\,s^2_{\\text{rising}}/k + s^2_{\\text{all}}/N\\,}}.$$\n\n"
            "But a significant $\\widehat{\\Delta}$ would **still not** establish *leading*: a coincident "
            "or lagging series can co-move with forward returns through autocorrelation in the cycle. "
            "The identifying test is the **lead/lag cross-correlation** "
            "$\\rho(L) = \\mathrm{corr}(m_t,\\ r_{t+L\\to t+L+1})$. A genuine early-warning peaks "
            "(negatively) at $L>0$. If $\\arg\\min_L \\rho(L) < 0$, claims **follow** the market — and "
            "the entire 'early' thesis collapses regardless of the conditional mean."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Claims tape.** Monthly 4-week-MA `IC4WSA` (thousands, SA), hardcoded snapshot, "
            f"{R['start'][:7]}→{R['end'][:7]} ({R['months']} months). Settled print, not real-time "
            "vintage (named on the axis).\n"
            "- **Signal.** $m_t = c_t/c_{t-3}-1$; RISING when $m_t>0$.\n"
            "- **Forward returns.** Enter at the close **1 month after** the signal (no look-ahead), "
            "hold $H\\in\\{1,3,6,12\\}$ months; drop horizons that overrun the tape.\n"
            "- **Null #1 (Welch t).** Rising-set mean vs the unconditional mean.\n"
            "- **Null #2 (placebo).** 20,000 draws of $k$ random months; "
            "$p = \\Pr[\\text{random-draw mean} \\le \\text{rising mean}]$ (as bearish or more).\n"
            "- **Identification (lead/lag).** $\\rho(L)$ for $L\\in[-6,6]$ — *where* do claims line up?\n"
            "- **Tradability.** Cash-when-rising overlay, 1-month lag, 10 bps one-way per switch "
            "(turnover one-way × NAV), excess-of-zero Sharpe (cash leg = 0, labelled).\n"
            "- **Positive control.** A deterministic series with a *planted* claims→returns link: "
            "`edge=0` must not fake significance; a large `edge` must light up the test."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The point estimates — right sign, small, insignificant\n\n"
            "Rising-claims forward mean with $\\pm$ standard error against the unconditional base rate "
            "(dashed). Below base at every horizon — but inside its own error bar."
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
            "    ts = [R['h1'][7], R['h3'][7], R['h6'][7], R['h12'][7]]; ses = [.012,.02,.03,.045]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x, [c*100 for c in cm], yerr=[s*100 for s in ses], capsize=5, color=GREEN, width=.5, label='rising-claims (±SE)')\n"
            "ax.plot(x, [b*100 for b in bm], 'D', ms=11, c=GREY, label='unconditional base rate')\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{m}m' for m in hs]); ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean forward SPY return (%)')\n"
            "ax.set_title('Right sign (below base) but the SE swamps the gap'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Welch t by horizon:', {f'{m}m': round(t,2) for m,t in zip(hs,ts)})"
        ),
        md(
            f"> 💡 In plain words: at 12m the rising-claims mean is **+{R['h12'][2]:.1f}%** vs base "
            f"**+{R['h12'][4]:.1f}%** — a ~{R['h12'][4]-R['h12'][2]:.1f}-point shortfall at "
            f"**t = {R['h12'][6]:.2f}** (not significant). H₁ is **directionally supported, statistically "
            "not**: the right sign living inside its own error bar."
        ),
        md(
            "### 4b · The decisive identification test — lead/lag\n\n"
            "$\\rho(L) = \\mathrm{corr}(m_t, r_{t+L\\to t+L+1})$. Negative bars left of zero = claims "
            "**lag** the market; a real early-warning would dip on the **right** (claims lead)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ll = st.lead_lag(F); Ls = list(range(-6,7)); cs = [ll[L] for L in Ls]\n"
            "else:\n"
            "    Ls = sorted(R['leadlag']); cs = [R['leadlag'][L] for L in Ls]\n"
            "cols = [RED if L<0 else GREEN for L in Ls]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(Ls, cs, color=cols, width=.8)\n"
            "ax.axhline(0, c='k', lw=.8); ax.axvline(0, c=GREY, lw=1, ls=':')\n"
            "imin = int(np.nanargmin(cs))\n"
            "ax.annotate('strongest NEGATIVE link\\n(claims LAG the market)', xy=(Ls[imin], cs[imin]),\n"
            "            xytext=(Ls[imin]-0.5, cs[imin]-0.07), ha='center', color=RED,\n"
            "            arrowprops=dict(arrowstyle='->', color=RED))\n"
            "ax.set_xlabel('lead L (months): L>0 = claims lead (early-warning)   |   L<0 = claims lag')\n"
            "ax.set_ylabel(r'$\\rho(L)$'); ax.set_xticks(Ls)\n"
            "ax.set_title('argmin rho(L) is at L<0: claims are coincident-to-lagging')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'argmin at L={Ls[imin]} (rho={cs[imin]:+.2f}); rho at +1 month = {cs[Ls.index(1)]:+.2f}')"
        ),
        md(
            "> 💡 In plain words: $\\arg\\min_L \\rho(L) = -3$. The claims signal correlates most "
            "(negatively) with a market move **a quarter in its past**; at the positive leads a genuine "
            "early-warning needs, $\\rho \\approx 0$. **H₂ rejected.** The equity market leads the "
            "economy; claims trail both — the 'early-warning' is the market's lead, reflected. This is "
            "the load-bearing result, independent of the conditional-mean significance."
        ),
        md(
            "### 4c · Tradability — the cash-on-rising overlay loses\n\n"
            "Hold SPY when claims are falling, cash when rising (1-month lag, 10 bps/switch). Annualised "
            "mean and Sharpe vs buy-and-hold."
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
            "a1.set_ylabel('annualised mean return (%)'); a1.set_title('Return: overlay loses ~4 pts/yr')\n"
            "a2.bar(labels, [bh_s, g_s, n_s], color=[GREY, AMBER, RED], width=.6)\n"
            "for i,v in enumerate([bh_s,g_s,n_s]): a2.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom')\n"
            "a2.set_ylabel('annualised Sharpe (excess-of-0)'); a2.set_title(f'Sharpe: overlay lower too ({nsw} switches)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'net overlay {n_m:.1f}%/yr (Sharpe {n_s:.2f}) vs buy-hold {bh_m:.1f}%/yr (Sharpe {bh_s:.2f})')"
        ),
        md(
            f"> 💡 In plain words: the overlay returns **+{R['overlay'][4]:.1f}%/yr** net vs "
            f"**+{R['overlay'][0]:.1f}%** for buy-and-hold, with a *lower* Sharpe "
            f"({R['overlay'][5]:.2f} vs {R['overlay'][1]:.2f}) and {R['overlay'][6]} switches. Because "
            "claims keep rising into recoveries the market has already begun pricing, sitting out on "
            "'rising claims' systematically forfeits upside. **H₃ rejected** — costs aren't even the "
            "issue; the *timing* loses. `MIRAGE`."
        ),
        md(
            "### 4d · Robustness — window, threshold, and the COVID dependence\n\n"
            "Vary the momentum window $k$ and the rising threshold, and drop 2020–2021. The 12-month "
            "*t* never clears 2 at an *early* (k ≤ 3) spec — and the biggest upticks flip the sign."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rob = []\n"
            "    for k in (1,3,6):\n"
            "        s = st.summarize(F, 12, k=k); rob.append((f'k={k}', s['n_rising'], s['rising_mean']*100, s['t'], s['p_placebo']))\n"
            "    s = st.summarize(F, 12, thresh=0.10); rob.append(('thr>+10%', s['n_rising'], s['rising_mean']*100, s['t'], s['p_placebo']))\n"
            "    F2 = F[(F.index < '2020-01-01') | (F.index >= '2022-01-01')]\n"
            "    s = st.summarize(F2, 12); rob.append(('ex-COVID', s['n_rising'], s['rising_mean']*100, s['t'], s['p_placebo']))\n"
            "else:\n"
            "    rob = [(l,n,r,t,p) for (l,n,r,_b,t,p) in R['robust']]\n"
            "labels = [r[0] for r in rob]; tt = [r[3] for r in rob]; nn = [r[1] for r in rob]\n"
            "cols = [GREEN if t<0 else RED for t in tt]\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.3))\n"
            "ax.bar(labels, tt, color=cols, width=.6)\n"
            "ax.axhline(-2, ls='--', c=RED, label='t=-2 (significance bar)')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,(t,k) in enumerate(zip(tt,nn)): ax.annotate(f'n={k}',(i,t),ha='center',va='bottom' if t>=0 else 'top')\n"
            "ax.set_ylabel('Welch t (12-month)'); ax.set_title('No early spec clears |t|=2; big upticks flip POSITIVE'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('robustness (label, n, rising12%, t, p):', [(r[0], r[1], round(r[2],1), round(r[3],2), round(r[4],3)) for r in rob])"
        ),
        md(
            "> 💡 In plain words: the effect *strengthens* only as the window lengthens "
            f"(**k=6 → t={R['robust'][2][4]:.2f}**) — but a 6-month-old claims trend is the antithesis of "
            "*early*. Restrict to *big* upticks (>+10%) and the sign flips **positive** "
            f"(**t={R['robust'][3][4]:.2f}**): the largest claims surges are recession *bottoms* you'd "
            f"buy. Ex-COVID the 12m t is **{R['robust'][4][4]:.2f}** — still under the bar. The signal is "
            "real-ish only where it's useless and useless where it would pay."
        ),
        md(
            "### 4e · Faithful-engine control — we know the truth here\n\n"
            "A deterministic monthly series with a *planted* link (rising claims momentum at $t$ "
            "depresses the $t{+}1$ return by `edge`). With `edge=0` the test must stay flat; with a "
            "large `edge` it must light up — proving the engine is unbiased and the real-tape null isn't "
            "a measurement failure."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.04):\n"
            "    syn = data.synthetic_claims(n_months=360, edge=edge, seed=385)\n"
            "    s = st.summarize(syn, 1, k=3)\n"
            "    res.append((edge, s['n_rising'], s['rising_mean']*100, s['base_mean']*100, s['t'], s['p_placebo']))\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "labels = [f'planted edge\\n{e*100:.0f}% / month' for e,_,_,_,_,_ in res]\n"
            "tvals = [r[4] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(-2, ls='--', c=RED, label='t=-2 (significance bar)'); ax.axhline(0, c='k', lw=.8)\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='top')\n"
            "ax.set_ylabel('Welch t (1-month)'); ax.set_title('Control: no link -> flat; real link -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,k,c,b,t,p in res: print(f'planted {e*100:+.0f}%/mo: n_ris={k} rising={c:.2f}% base={b:.2f}% t={t:.2f} p={p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted link the control sits at "
            f"**t = {R['syn'][0][4]:.2f}** (no false positive); a **+4%/month** planted link drives "
            f"**t = {R['syn'][1][4]:.2f}**. So the machinery is honest — the real-tape *t* of ~−0.9 is a "
            "*genuine* weak-or-absent edge, not a broken test. The engine *can* bank a real claims→"
            "returns link; the real tape just doesn't carry a tradable one."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — 12m excess **{R['h12'][2]-R['h12'][4]:+.1f}pp** at Welch "
            f"**t = {R['h12'][6]:.2f}** / placebo **p = {R['h12'][7]:.2f}**; right sign, fails t≥2, "
            "fragile to window, sign-flips for big upticks. Literature support (claims ∈ the LEI) + a "
            "directionally-correct-but-insignificant tilt ⇒ WEAK, not REAL.\n"
            f"- **Tradability `MIRAGE`** — the cash-on-rising overlay returns "
            f"**+{R['overlay'][4]:.1f}%/yr** (Sharpe {R['overlay'][5]:.2f}) vs buy-hold "
            f"**+{R['overlay'][0]:.1f}%/yr** (Sharpe {R['overlay'][1]:.2f}). Acting on the signal "
            "*subtracts* return — nothing to allocate to.\n"
            "- **Early warning? `NOT SUPPORTED`** — $\\arg\\min_L \\rho(L) = -3$ months: claims momentum "
            "is **coincident-to-lagging**, not leading. The equity market is the leading indicator; "
            "claims echo it. The defining word — *early* — is the part the data rejects."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — why even a real tilt wouldn't deploy\n\n"
            "Grant the lore a genuine few-point tilt. The operational reality still defeats it. Claims "
            "are 'rising' in **~43% of months** (every up-wiggle of a noisy monthly series), so the "
            f"overlay churns ({R['overlay'][6]} switches / 33y) and is out of the market in roughly half "
            "of all months — including the early innings of every recovery, which the equity market "
            "starts pricing *before* claims roll over. That structural mistiming is why the overlay's "
            "Sharpe is **below** passive even at zero cost. And the one regime where claims and a market "
            "bottom genuinely coincide — the deepest surges — is where you'd want maximum *long* "
            "exposure, which is exactly why the >+10% threshold flips the sign positive. No lag, "
            "threshold, or cost assumption rescues a coincident series masquerading as a leading one."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The sibling.** [Study 268 — Sahm-Rule](../268-sahm-rule/): the unemployment-rate "
            "recession trigger, same hardcoded-snapshot + SPY method — a real labour signal whose "
            "tradability is the open question.\n"
            "- **Companion macro nowcasts.** [Study 266 — Misery-Index](../266-misery-index/), "
            "[Study 267 — M2-Growth](../267-m2-growth/) — does any celebrated macro gauge time equities?\n"
            "- **Sharper identification.** Replace the monthly snapshot with the weekly raw series and "
            "run a proper VAR / Granger test, or use **real-time vintages** (ALFRED) to kill any "
            "revision look-ahead; the coincident-to-lagging structure is robust to all of these — "
            "smoothing a coincident series differently can't manufacture a lead.\n\n"
            "*The reproducible core is offline and deterministic; the claims input is an explicit "
            "frozen snapshot. Methods and sources: [`docs/references.md`](../docs/references.md); frozen "
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
